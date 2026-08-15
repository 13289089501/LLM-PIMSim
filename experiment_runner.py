"""
LLM-PIMSim v3 — experiment_runner
编排一个实验: 读配置 → 造 workload（kernel 粒度）→ 造硬件 → 映射 → 放置 → 运行 → 保存
用户只需要: 写 YAML + 调 run_experiment(config_path)

v3 解耦：各步骤职责已下沉到本文顶部 import 的 core.* 系统——
  - 算子系统 core.operator_sys：build_model_workload / WorkloadAdapter
  - 核心调度器 core.engine：SimulationEngine
  - 输出系统 core.exporter：report / save_json / result_to_dict
本文件只做"编排"，不含算子建模 / 调度 / 序列化等具体实现。
"""
from config_loader import load_experiment, ConfigError
from hardware_factory import build_hardware
from core.hardware_sys import build_frontend_custom_hardware
from mapping_engine import MappingEngine
from placement_engine import PlacementEngine
from core.engine import SimulationEngine
from core.operator_sys import build_model_workload, WorkloadAdapter
from core import exporter as _exporter
from core.common import DataObject, DataType


def run_experiment(exp_path: str, save: bool = True, verbose: bool = True) -> dict:
    """运行一个实验（v3 统一走 workload/kernel 路径——新标准）。
    该函数仅作兼容别名，内部委托 run_workload_experiment。"""
    return run_experiment_with_mapping(exp_path, compute_map_override=None,
                                       save=save, verbose=verbose)


def run_experiment_with_mapping(exp_path: str, compute_map_override: dict = None,
                                save: bool = True, verbose: bool = True) -> dict:
    """运行 experiment.yaml，但允许调用方注入算子→硬件的 mapping 覆盖。

    v3 整合说明：不再使用旧的 model_lib 内置算子图路径（L0_q_gemm/L0_attn 等命名
    与 workload/kernel 命名冲突，且不支持权重分片/KV 动态）。统一委托给
    run_workload_experiment（kernel 粒度 + WeightBlock ALL-GATHER 新标准）。

    compute_map_override: {kernel_id: compute_device_id} —— 前端拖拽形成的映射。
    YAML 里的 mapping 只作为兜底（未被覆盖的算子用它）；被覆盖的算子用前端映射。
    """
    return run_workload_experiment(
        exp_path, compute_map_override=compute_map_override,
        save=save, verbose=verbose)



def _report(result):
    """打印控制台摘要（委托输出系统 core.exporter.report）。"""
    print(_exporter.report(result))


def _save(result, out_dir: str, name: str):
    """保存 JSON（委托输出系统 core.exporter.save_json）。"""
    path = _exporter.save_json(result, out_dir, name)
    print(f"  [保存] → {path}")


def _construct_plan(exp_path, compute_map_override, weight_blocks, splits_override,
                    workload_override=None, link_table_override=None,
                    custom_hardware=None) -> dict:
    """加载实验并构造完整运行计划（workload/硬件/映射/放置/切片），并做运行前充分校验。

    校验未通过会抛 ConfigError —— 这是"校验通过即可完整运行、校验不过则拒绝运行"的统一判定，
    运行与校验（validate_runnable）共用本函数，保证两边结论完全一致。

    workload_override: 可选 {input_tokens, generate_tokens, ...} —— 前端让用户自定义
    输入 token 数与自回归步数时注入，仅覆盖 experiment.yaml 的 workload 段，不改变
    硬件/映射/放置等任何其它配置。
    link_table_override: 可选 {kind: {kind: GB/s}} —— 前端画布编辑的链路带宽表，
    覆盖/新增到实验的链路表（链路系统）。
    custom_hardware: 可选 list[dict] —— 前端 serializeState 的 hardware 列表；
    其中"自定义设备"（backId==id）会被注入实验硬件集，成为真实后端设备。
    """
    ing = load_experiment(exp_path)
    if workload_override:
        ing.cfg.workload = {**(ing.cfg.workload or {}), **workload_override}

    # 前端自定义硬件 → 注入实验硬件集（跳过与既有后端设备重名的，避免覆盖）
    custom_cfgs = {}
    if custom_hardware:
        for hid, hc in build_frontend_custom_hardware(custom_hardware).items():
            if hid not in ing.cfg.hardware:
                ing.cfg.hardware[hid] = hc
            custom_cfgs[hid] = hc

    model_cfg = _resolve_model_dims(ing)
    wl = build_model_workload(**model_cfg)
    adp = WorkloadAdapter(wl)
    ex = adp.build_executable()
    operators = ex["operators_map"]
    data_objects = ex["data_map"]
    devices, link_table = build_hardware(ing.cfg)
    if link_table_override:
        link_table.update(link_table_override)
    # 自定义设备的 links 也同步进链路表（即使前端未传 link_table_override 也保证其种类可达）
    for hc in custom_cfgs.values():
        if getattr(hc, "links", None):
            link_table.add_type(hc.link_type, hc.links)

    mapping_eng = MappingEngine(ing.cfg.mapping,
                                default_device=ing.cfg.mapping_default_device,
                                default_source=ing.cfg.mapping_default_source)
    op_specs = mapping_eng.apply(operators)
    compute_map = {}
    input_specs = {}
    for oid, spec in op_specs.items():
        compute_map[oid] = spec.compute_device
        input_specs[oid] = list(spec.inputs)

    # 应用前端拖拽形成的映射覆盖（严格：目标设备必须在后端硬件集内；否则视为配置不合法，拒绝运行）
    if compute_map_override:
        for k, hw in compute_map_override.items():
            if hw not in devices:
                raise ConfigError(
                    f"算子映射无效：'{k}' 的目标设备 '{hw}' 不在本实验硬件中"
                    f"（本实验硬件: {sorted(devices.keys())}）。请检查画布设备与实验硬件是否一致。")
            if k in compute_map:
                compute_map[k] = hw
            else:
                for oid in compute_map:
                    if oid == k:
                        compute_map[oid] = hw
                        break

    # 运行前充分校验：所有算子的目标设备必须存在且能执行该算子
    bad = []
    for oid, hw in compute_map.items():
        op = operators.get(oid)
        if op is None:
            continue
        if hw not in devices:
            bad.append(f"算子 {oid} 的目标设备 '{hw}' 不存在于本实验硬件")
        else:
            hw_unit = devices[hw]
            if not hw_unit.can_execute(op):
                bad.append(f"算子 {oid}({op.op_type}, 精度 {getattr(op,'execution_precision',None) or getattr(op,'data_precision',None)}) "
                           f"无法在设备 {hw}({getattr(hw_unit,'device_type',None)}) 上执行")
    if bad:
        raise ConfigError("运行前校验未通过（配置无法完整运行）：\n  " + "\n  ".join(bad[:15]) +
                          ("\n  …" if len(bad) > 15 else ""))

    placement_eng = PlacementEngine(ing.cfg.placement,
                                    default_device=ing.cfg.placement_default)
    placement = placement_eng.apply(data_objects)
    weight_shards = _expand_weight_blocks(data_objects, placement, weight_blocks)
    cfg_splits = list(ing.cfg.workload.get("splits") or [])
    if splits_override:
        cfg_splits = _merge_splits(cfg_splits, splits_override)
    op_splits = _build_op_splits(cfg_splits, wl, set(devices.keys()))

    return {"wl": wl, "operators": operators, "data_objects": data_objects,
            "devices": devices, "link_table": link_table,
            "compute_map": compute_map, "input_specs": input_specs,
            "placement": placement, "weight_shards": weight_shards,
            "op_splits": op_splits, "ing": ing}


def validate_runnable(exp_path: str, compute_map_override: dict = None,
                      weight_blocks: list = None, splits_override: list = None,
                      workload_override: dict = None,
                      link_table_override: dict = None,
                      custom_hardware: list = None) -> dict:
    """运行前充分校验（与 run_workload_experiment 完全同一套判定）。
    返回 {"ok": bool, "errors": [str]}；ok=False 即不允许运行。"""
    try:
        _construct_plan(exp_path, compute_map_override, weight_blocks, splits_override,
                        workload_override, link_table_override, custom_hardware)
        return {"ok": True, "errors": []}
    except ConfigError as e:
        return {"ok": False, "errors": [str(e)]}


# ============================================================
# workload 驱动路径：用 kernel 粒度 workload + adapter 跑仿真
# ============================================================
def run_workload_experiment(exp_path: str, compute_map_override: dict = None,
                            save: bool = True, verbose: bool = True,
                            weight_blocks: list = None,
                            splits_override: list = None,
                            workload_override: dict = None,
                            link_table_override: dict = None,
                            custom_hardware: list = None) -> dict:
    """用 kernel 粒度的 workload 驱动仿真（标准路径）。
    从 experiment.yaml 读数: model 参数 + 硬件/映射/放置 + workload(splits切片)。
    compute_map_override: {kernel_id: hw_id} —— 前端拖拽形成的映射，覆盖 YAML。
    weight_blocks: GUI 权重块列表 [{weight_id, weight_class, device, partitions:[{partition_id, device, bytes}]}]
        —— 被切分的权重会展开为分片 DataObject 并触发调度器 ALL-GATHER。
    splits_override: 前端算子切割产生的切片规则（list[dict{op,dim,parts,devices}]）——
        与 experiment.yaml 的 workload.splits 合并，前端切割真正驱动后端张量并行执行。
    workload_override: 可选 {input_tokens, generate_tokens, ...} —— 前端用户自定义
        输入 token 数 / 自回归步数时注入，覆盖 experiment.yaml 的 workload 段。
    link_table_override: 可选 {kind: {kind: GB/s}} —— 前端画布编辑的链路带宽表。
    custom_hardware: 可选 list[dict] —— 前端 serializeState 的 hardware 列表；
        自定义设备（backId==id）会被注入为真实后端设备。
    """
    plan = _construct_plan(exp_path, compute_map_override, weight_blocks, splits_override,
                           workload_override, link_table_override, custom_hardware)
    ing = plan["ing"]
    wl = plan["wl"]
    operators = plan["operators"]
    data_objects = plan["data_objects"]
    devices = plan["devices"]
    link_table = plan["link_table"]
    compute_map = plan["compute_map"]
    input_specs = plan["input_specs"]
    placement = plan["placement"]
    weight_shards = plan["weight_shards"]
    op_splits = plan["op_splits"]

    # --- 6. 运行 ---
    engine = SimulationEngine()
    engine.build(
        hardware_units=devices,
        link_table=link_table,
        operators=operators,
        data_objects=data_objects,
        compute_map=compute_map,
        placement=placement,
        input_specs=input_specs,
        weight_shards=weight_shards,
        op_splits=op_splits,
    )
    result = engine.run(seed=ing.seed)
    # --- 6b. 运行后完成度校验：未满员则判定方案无效（不能当作有效结果/对比）---
    from core.validator import validate_completion
    comp_vr = validate_completion(result)
    if not comp_vr.valid:
        result.metadata["completion_valid"] = False
        result.metadata["completion_errors"] = [{"code": i.code, "message": i.message}
                                                for i in comp_vr.errors]
        for i in comp_vr.errors:
            result.add_note(f"[完成度校验] {i.message}")
    else:
        result.metadata["completion_valid"] = True

    result.metadata["model"] = ing.model
    result.metadata["experiment"] = ing.name
    result.metadata["workload_source"] = "kernel_workload"
    result.metadata["num_operators"] = len(operators)
    result.metadata["kv_range_bytes"] = list(wl.kv_range_bytes)
    result.metadata["num_layers"] = wl.num_layers
    result.metadata["weight_shard_count"] = sum(
        len(s) for s in weight_shards.values()) if weight_shards else 0
    # 记录被算子切片切分了多少个算子（供诊断）
    result.metadata["op_split_count"] = len(op_splits) if op_splits else 0
    if compute_map_override:
        result.metadata["mapping_override_applied"] = len(
            [k for k in compute_map_override if k in compute_map])

    if verbose:
        _report(result)
    if save:
        _save(result, ing.output_dir, ing.name)

    return {"result": result, "result_dict": _exporter.result_to_dict(result),
            "ingredient": ing, "workload": wl}


def _expand_weight_blocks(data_objects: dict, placement: dict,
                          weight_blocks: list) -> dict:
    """把 GUI 传入的 weight_blocks 展开为运行时权重分片。

    - 被切分（partitions 非空）的权重：每个 partition 注册为独立 DataObject，
      放置到其 device；返回 {weight_id: {partition_id: device}} 供调度器 ALL-GATHER。
    - 未切分但指定了 device 的权重：直接覆盖该权重数据的初始驻留设备。
    """
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


def _merge_splits(cfg_splits: list, override: list) -> list:
    """把前端算子的切片规则覆盖/合并进配置的分片规则（前端切割优先级更高）。

    匹配规则：优先按 (op, dim) 对同算子同维覆盖；其余追加。
    """
    merged = []
    used = set()
    for o in override or []:
        key = (o.get("op"), o.get("dim"))
        merged.append(o)
        used.add(key)
    for c in cfg_splits or []:
        key = (c.get("op"), c.get("dim"))
        if key not in used:
            merged.append(c)
    return merged


def _build_op_splits(splits_cfg, wl, hw_ids) -> dict:
    """把配置里的 splits 转成 engine 的 op_splits（维度切分 + 权重分片式张量并行）。

    配置格式（experiment.yaml 的 workload.splits，list[dict]）:
        - op: "L*_ffn_down"        # 算子 id，支持 fnmatch 通配（如 L*_ffn_down）
          dim: N                   # 沿哪个维度切（M/K/N/seq/kv_len，须在该算子 attributes 里）
          parts: 2                 # 切成几份（等分）
          devices: [gpu0, gpu1]    # 各切片落到这些设备
    仅对存在于算子集中的 id 生效；devices 里不在硬件集的自动作废（至少留 2 个才切）。
    `dim` 缺省时退化为"计算量均分"（旧行为）。
    返回 {op_id: {"devices":[...], "slice_flops":[...], "dim":dim}} —— slice_flops 由切割系统
    split_kernel_dict 按维度比例算得（张量并行）。
    """
    from fnmatch import fnmatch
    from core.splitter import split_kernel_dict
    op_splits = {}
    if not splits_cfg:
        return op_splits
    # 建立算子 id → kernel dict（用于维度切分）
    kernel_by_id = {}
    for layer in (wl.layers if wl else []):
        for k in layer:
            kernel_by_id[k.id] = k.to_dict()
    for k in (wl.kernels if wl else []):
        kernel_by_id[k.id] = k.to_dict()

    for rule in splits_cfg or []:
        pat = (rule.get("op") or "").strip()
        devices = [d for d in (rule.get("devices") or []) if d in hw_ids]
        if not pat or len(devices) < 2:
            continue
        dim = (rule.get("dim") or "").strip()
        try:
            nparts = int(rule.get("parts", len(devices)))
        except (TypeError, ValueError):
            nparts = len(devices)
        nparts = max(2, nparts)
        for op_id in kernel_by_id:
            if not (fnmatch(op_id, pat) or op_id == pat):
                continue
            kdict = kernel_by_id[op_id]
            spec = {"devices": list(devices), "dim": dim}
            try:
                if dim:
                    # 维度切分：用切割系统 split_kernel_dict 等分
                    total = kdict.get("attributes", {}).get(dim)
                    if total is None:
                        continue
                    parts = _equal_parts(total, nparts) if isinstance(total, (int, float)) \
                        else None
                    if parts:
                        slices = split_kernel_dict(kdict, dim, parts)
                        spec["slice_flops"] = [int(s["compute_flops_range"][1]) for s in slices]
                # 无 dim/无 slices → 走计算量均分（旧行为，slice_flops 由 engine 推算）
            except Exception:
                spec.pop("slice_flops", None)
            op_splits[op_id] = spec
    return op_splits


def _equal_parts(total, n):
    """把 total 等分为 n 份（整数，余数分摊到前几份），返回列表。"""
    if not total or n <= 0:
        return None
    base = int(total) // n
    rem = int(total) % n
    return [base + (1 if i < rem else 0) for i in range(n)]


def _resolve_model_dims(ing) -> dict:
    """解析模型维度: 优先用 experiment.yaml 的 workload 字段，否则查内置模型维表。

    返回可直接传给 build_model_workload 的关键字参数 dict。
    支持键: num_layers / hidden / ffn_size / num_heads / head_dim / vocab /
            pbytes / input_tokens / generate_tokens / batch。
    未写或非法的键回退到内置模型维表 + 默认序列规模。
    """
    # 内置模型维表（单一事实来源，见 model_lib.MODEL_DIMS）
    from model_lib import MODEL_DIMS
    dims = dict(MODEL_DIMS.get(ing.model, MODEL_DIMS["llama_gb"]))

    # 用 experiment.yaml 的 workload 段覆盖（用户自定义维度）
    wl = ing.cfg.workload or {}
    for key in ("num_layers", "hidden", "ffn_size", "num_heads",
                "head_dim", "vocab", "pbytes"):
        if wl.get(key) is not None:
            try:
                dims[key] = int(wl[key])
            except (TypeError, ValueError):
                raise ConfigError(f"workload.{key} 必须是整数，收到: {wl[key]!r}")

    # 序列规模：默认 prefill=2048 token，generate=128（自回归/Decode 步数 → KV 终点）
    try:
        dims["input_tokens"] = int(wl.get("input_tokens", 2048))
    except (TypeError, ValueError):
        raise ConfigError(f"workload.input_tokens 必须是整数，收到: {wl.get('input_tokens')!r}")
    # generate_tokens 决定自回归步数/ KV 增长；decode_steps 为旧名（兼容，优先 generate_tokens）
    gen_raw = wl.get("generate_tokens", wl.get("decode_steps", 128))
    try:
        dims["generate_tokens"] = int(gen_raw)
    except (TypeError, ValueError):
        raise ConfigError(f"workload.generate_tokens 必须是整数，收到: {gen_raw!r}")
    try:
        dims["batch"] = int(wl.get("batch", 1))
    except (TypeError, ValueError):
        raise ConfigError(f"workload.batch 必须是整数，收到: {wl.get('batch')!r}")
    return dims
