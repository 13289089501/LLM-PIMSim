"""
LLM-PIMSim v2 — experiment_runner
编排一个实验: 读配置 → 选模型 → 造硬件 → 映射 → 放置 → 运行 → 保存
用户只需要: 写 YAML + 调 run_experiment(config_path)
"""
import json, os
from config_loader import load_experiment
from hardware_factory import build_hardware
from mapping_engine import MappingEngine
from placement_engine import PlacementEngine
from model_lib import get_model
from engine import SimulationEngine
from workload_model import build_model_workload, LayerGroup
from workload_adapter import WorkloadAdapter


def run_experiment(exp_path: str, save: bool = True, verbose: bool = True) -> dict:
    return run_experiment_with_mapping(exp_path, compute_map_override=None,
                                       save=save, verbose=verbose)


def run_experiment_with_mapping(exp_path: str, compute_map_override: dict = None,
                                save: bool = True, verbose: bool = True) -> dict:
    """运行 experiment.yaml，但允许调用方注入算子→硬件的 mapping 覆盖。
    compute_map_override: {op_id: compute_device_id} —— 前端拖拽形成的映射。
    YAML 里的 mapping 只作为兜底（未被覆盖的算子用它）；被覆盖的算子用前端映射。
    """
    ing = load_experiment(exp_path)

    # --- 1. 选模型 ---
    model = get_model(ing.model)
    operators = {op.id: op for op in model["operators"]}
    data_objects = {d.id: d for d in model["data_objects"]}

    # --- 2. 造硬件 + 互连 ---
    devices, interconnect = build_hardware(ing.cfg)

    # --- 3. 映射（算子→设备 + 数据源）：
    #    YAML 解析出基础 mapping，"前端覆盖"能覆写指定算子的目标设备 ---
    mapping_eng = MappingEngine(ing.cfg.mapping,
                                default_device=ing.cfg.mapping_default_device,
                                default_source=ing.cfg.mapping_default_source)
    op_specs = mapping_eng.apply(operators)

    compute_map = {}       # op_id -> compute_device
    input_specs = {}       # op_id -> [InputSpec]
    for oid, spec in op_specs.items():
        compute_map[oid] = spec.compute_device
        input_specs[oid] = list(spec.inputs)

    # 应用前端注入的覆盖映射
    if compute_map_override:
        for op_id, hw_id in compute_map_override.items():
            if op_id in compute_map:
                compute_map[op_id] = hw_id
                # 校验硬件存在 & 精度兼容由 engine/performance 处理

    # --- 4. 放置（数据初始驻留，可冗余多设备）---
    placement_eng = PlacementEngine(ing.cfg.placement,
                                    default_device=ing.cfg.placement_default)
    placement = placement_eng.apply(data_objects)

    # --- 5. 运行 ---
    engine = SimulationEngine()
    engine.build(
        hardware_units=devices,
        interconnect=interconnect,
        operators=operators,
        data_objects=data_objects,
        compute_map=compute_map,
        placement=placement,
        input_specs=input_specs,
    )
    result = engine.run(seed=ing.seed)

    result.metadata["model"] = ing.model
    result.metadata["experiment"] = ing.name
    if compute_map_override:
        result.metadata["mapping_override_applied"] = len(compute_map_override)

    if verbose:
        _report(result)

    if save:
        _save(result, ing.output_dir, ing.name)

    return {"result": result, "result_dict": result.to_dict(), "ingredient": ing}



def _report(result):
    bd = result.breakdown
    print("=" * 60)
    print(f"  实验: {result.metadata.get('experiment','?')}   模型: {result.metadata.get('model','?')}")
    print("=" * 60)
    print(f"  总延迟:     {result.total_latency_ns/1e6:>10.2f} ms")
    print(f"  计算:       {bd.compute_ns/1e6:>10.2f} ms")
    print(f"  搬运:       {bd.transfer_ns/1e6:>10.2f} ms")
    print(f"  同步等待:   {bd.sync_ns/1e6:>10.2f} ms")
    print(f"  瓶颈类型:   {result.bottleneck.name}")
    print(f"  理由:       {result.bottleneck_rationale}")
    if result.data_source_notes:
        print(f"  数据源决策记录: 共 {len(result.data_source_notes)} 条"
              f"（已存 JSON，此处仅显示前 3 条）")
        # 聚合统计：固定源 vs 就近参考 vs 告警
        pinned = sum(1 for n in result.data_source_notes if "[参考]" not in n and "指定" in n)
        auto = sum(1 for n in result.data_source_notes if "[参考]" in n)
        warn = len(result.data_source_notes) - pinned - auto
        print(f"    用户固定源: {pinned} | 就近参考: {auto} | 告警/异常: {warn}")
        for n in result.data_source_notes[:3]:
            print(f"    {n}")


def _save(result, out_dir: str, name: str):
    # 输出目录锚定到 v2 项目根（保证相对路径稳定，不受运行目录影响）
    base = os.path.dirname(os.path.abspath(__file__))
    # 若 out_dir 是绝对路径则直接用，否则视为相对 v2 根
    if not os.path.isabs(out_dir):
        out_path = os.path.join(base, out_dir)
    else:
        out_path = out_dir
    os.makedirs(out_path, exist_ok=True)
    path = os.path.join(out_path, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"  [保存] → {path}")


# ============================================================
# workload 驱动路径：用 kernel 粒度 workload + adapter 跑仿真
# ============================================================
def run_workload_experiment(exp_path: str, compute_map_override: dict = None,
                            save: bool = True, verbose: bool = True,
                            weight_blocks: list = None) -> dict:
    """用 kernel 粒度的 workload 驱动仿真（替代 model_lib 路径）。
    从 experiment.yaml 读数: model 参数 + 硬件/映射/放置。
    current 只展开 Prefill（定值 cost）。
    compute_map_override: {kernel_id: hw_id} —— 前端拖拽形成的映射，覆盖 YAML。
    weight_blocks: GUI 权重块列表 [{weight_id, weight_class, device, partitions:[{partition_id, device, bytes}]}]
        —— 被切分的权重会展开为分片 DataObject 并触发调度器 ALL-GATHER。
    """
    ing = load_experiment(exp_path)

    # --- 1. 从配置取模型维度参数（若 experiment.yaml 有 workload 字段则用之，
    #        否则用内置模型名对应的维度）---
    model_cfg = _resolve_model_dims(ing)
    wl = build_model_workload(**model_cfg)

    # --- 2. 展开成可执行算子图 ---
    adp = WorkloadAdapter(wl)
    ex = adp.build_executable()
    operators = ex["operators_map"]
    data_objects = ex["data_map"]

    # --- 3. 造硬件 + 互连 ---
    devices, interconnect = build_hardware(ing.cfg)

    # --- 4. 映射 ---
    mapping_eng = MappingEngine(ing.cfg.mapping,
                                default_device=ing.cfg.mapping_default_device,
                                default_source=ing.cfg.mapping_default_source)
    op_specs = mapping_eng.apply(operators)
    compute_map = {}
    input_specs = {}
    for oid, spec in op_specs.items():
        compute_map[oid] = spec.compute_device
        input_specs[oid] = list(spec.inputs)

    # 应用前端拖拽形成的映射覆盖
    if compute_map_override:
        for k, hw in compute_map_override.items():
            # 前端 key 可能是 kernel_id（如 L0_ffn_gate），也是 operators 的 id
            if k in compute_map:
                compute_map[k] = hw
            else:
                # 尝试按名称匹配（id 完全一致时通常能命中）
                for oid in compute_map:
                    if oid == k:
                        compute_map[oid] = hw
                        break

    # --- 5. 放置 ---
    placement_eng = PlacementEngine(ing.cfg.placement,
                                    default_device=ing.cfg.placement_default)
    placement = placement_eng.apply(data_objects)

    # --- 5b. 权重分片展开：切分权重 → 分片 DataObject + placement 覆盖 + ALL-GATHER 表 ---
    weight_shards = _expand_weight_blocks(data_objects, placement, weight_blocks)

    # --- 6. 运行 ---
    engine = SimulationEngine()
    engine.build(
        hardware_units=devices,
        interconnect=interconnect,
        operators=operators,
        data_objects=data_objects,
        compute_map=compute_map,
        placement=placement,
        input_specs=input_specs,
        weight_shards=weight_shards,
    )
    result = engine.run(seed=ing.seed)
    result.metadata["model"] = ing.model
    result.metadata["experiment"] = ing.name
    result.metadata["workload_source"] = "kernel_workload"
    result.metadata["num_operators"] = len(operators)
    result.metadata["kv_range_bytes"] = list(wl.kv_range_bytes)
    result.metadata["num_layers"] = wl.num_layers
    result.metadata["weight_shard_count"] = sum(
        len(s) for s in weight_shards.values()) if weight_shards else 0
    if compute_map_override:
        result.metadata["mapping_override_applied"] = len(
            [k for k in compute_map_override if k in compute_map])

    if verbose:
        _report(result)
    if save:
        _save(result, ing.output_dir, ing.name)

    return {"result": result, "result_dict": result.to_dict(),
            "ingredient": ing, "workload": wl}


def _expand_weight_blocks(data_objects: dict, placement: dict,
                          weight_blocks: list) -> dict:
    """把 GUI 传入的 weight_blocks 展开为运行时权重分片。

    - 被切分（partitions 非空）的权重：每个 partition 注册为独立 DataObject，
      放置到其 device；返回 {weight_id: {partition_id: device}} 供调度器 ALL-GATHER。
    - 未切分但指定了 device 的权重：直接覆盖该权重数据的初始驻留设备。
    """
    from contracts import DataObject, DataType
    weight_shards: dict = {}
    for wb in weight_blocks or []:
        wid = wb.get("weight_id") or ""
        if not wid:
            continue
        parts = wb.get("partitions") or []
        if parts:
            shards = {}
            for p in parts:
                pid = p.get("partition_id") or ""
                if not pid:
                    continue
                dev = p.get("device") or ""
                if pid not in data_objects:
                    data_objects[pid] = DataObject(
                        id=pid, name=pid, data_type=DataType.WEIGHT,
                        size_bytes=int(p.get("bytes") or 0))
                if dev:
                    placement[pid] = [dev]
                shards[pid] = dev
            if shards:
                weight_shards[wid] = shards
        else:
            dev = wb.get("device") or ""
            if dev and wid in data_objects:
                placement[wid] = [dev]
    return weight_shards


def _resolve_model_dims(ing) -> dict:
    """解析模型维度: 优先用 experiment.yaml 的 workload 字段，否则查内置模型。"""
    # 从 experiment.yaml 顶层取 workload
    cfg = ing.cfg
    # 尝试从 已加载的 ExperimentConfig 找维度（此处接入 config 的 workload 段）
    # 为兼容当前 config_loader，先回退到内置模型参数表
    _MODEL_DIMS = {
        "llama7b": dict(num_layers=32, hidden=4096, ffn_size=11008, num_heads=32, head_dim=128, vocab=32000, pbytes=2),
    }
    dims = dict(_MODEL_DIMS.get(ing.model, _MODEL_DIMS["llama7b"]))
    dims["input_tokens"] = 2048
    dims["decode_steps"] = 128
    return dims
