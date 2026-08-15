"""
LLM-PIMSim v3 GUI — 可视化拓扑编辑器
启动: python gui_app.py
v3 说明：
  - 统一走 workload/kernel 路径（新标准），移除旧的 model_lib 算子图主路径。
  - 结果面板新增"搬运量明细"与"算子完成度诊断"，便于从界面定位问题。
"""
import json, os, sys
import re
import yaml
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_experiment
from mapping_engine import MappingEngine
from experiment_runner import run_experiment, run_workload_experiment
from model_lib import list_models
from core.operator_sys import build_model_workload
from core.validator import validate_config, Issue

app = Flask(__name__)
BASE = Path(__file__).parent

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/files")
def api_files():
    configs = sorted(str(p.relative_to(BASE/"configs")).replace("\\","/")
                     for p in (BASE/"configs").rglob("*.yaml"))
    results = sorted(f.name for f in (BASE/"results").glob("*.json"))
    return jsonify({"configs": configs, "results": results})


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """对比两个实验结果 JSON：{names: ["a.json","b.json"]} → 对比表。"""
    d = request.get_json() or {}
    names = (d.get("names") or [])[:5]
    rows = []
    for n in names:
        p = BASE / "results" / n
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        bd = data.get("breakdown", {})
        rows.append({
            "name": n,
            "latency_ms": round((data.get("total_latency_ms") or 0), 3),
            "compute_ms": round((bd.get("compute_ns") or 0) / 1e6, 3),
            "transfer_ms": round((bd.get("transfer_ns") or 0) / 1e6, 3),
            "sync_ms": round((bd.get("sync_ns") or 0) / 1e6, 3),
            "local_rw_ms": round((bd.get("local_rw_ns") or 0) / 1e6, 3),
            "bottleneck": data.get("bottleneck", "?"),
            "movement_mb": round((data.get("movement_bytes", {}).get("total_bytes") or 0) / 1e6, 3),
            "complete": bool(data.get("diagnostics", {}).get("finished_operators", 0) ==
                             data.get("diagnostics", {}).get("total_operators", 0)),
            "kv_bytes": data.get("metadata", {}).get("kv_range_bytes"),
        })
    if not rows:
        return jsonify({"error": "未找到可对比的结果文件。"})
    return jsonify({"rows": rows})


@app.route("/api/experiments")
def api_experiments():
    """列出 experiments/ 下的所有实验入口 YAML（NN_*.yaml），供下拉框动态填充。"""
    exps_dir = BASE/"configs"/"experiments"
    entries = []
    if exps_dir.is_dir():
        for p in sorted(exps_dir.glob("*.yaml")):
            if p.stem.lower() == "template":
                continue
            # 子配置文件不是实验入口：
            #  1) 旧式：以类型为前缀（如 hardware_gpu、mapping_pim）
            #  2) 新式：包含 _hardware/_interconnect/_mapping/_placement 段（如 03_x_hardware）
            first_tok = p.stem.split("_")[0].lower()
            if first_tok in ("hardware", "interconnect", "mapping", "placement"):
                continue
            if any(k in p.stem for k in ("_hardware", "_interconnect", "_mapping", "_placement")):
                continue
            entries.append({"path": f"experiments/{p.name}", "name": p.stem})
    return jsonify({"experiments": entries})

@app.route("/api/experiment/create", methods=["POST"])
def api_experiment_create():
    """新建实验：{name, model, mode, clone_from} → 生成独立实验。
    mode='blank'：从空模板（一套默认子配置）开始；mode='ref'：从参考实验克隆（默认 04_ic_reference）。"""
    d = request.get_json() or {}
    name = (d.get("name") or "").strip()
    model = (d.get("model") or "llama_gb").strip()
    if not name:
        return jsonify({"ok": False, "error": "实验名不能为空。"})
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        return jsonify({"ok": False, "error": "实验名只能包含字母、数字、下划线和连字符。"})
    mode = (d.get("mode") or "blank").strip().lower()
    if mode not in ("blank", "ref"):
        mode = "blank"

    exps_dir = BASE / "configs" / "experiments"
    exps_dir.mkdir(parents=True, exist_ok=True)

    # 入口统一命名 NN_name.yaml，沿用 CLI 的 ??_*.yaml 约定
    prefix = _next_exp_prefix(exps_dir)
    exp_stem = f"{prefix}_{name}"
    entry_path = exps_dir / f"{exp_stem}.yaml"
    if entry_path.exists():
        return jsonify({"ok": False, "error": f"实验 {exp_stem} 已存在，请换一个名字。"})

    files = {}
    seed = 42
    if mode == "ref":
        # 从参考实验克隆其子配置文件
        clone_from = (d.get("clone_from") or "experiments/04_ic_reference.yaml").strip()
        src_entry = BASE / "configs" / clone_from
        src_base, src_files = exps_dir, {}
        try:
            src_doc = yaml.safe_load(src_entry.read_text(encoding="utf-8"))
            src_files = ((src_doc or {}).get("experiment", {}) or {}).get("files", {}) or {}
            src_base = src_entry.parent
            seed = ((src_doc or {}).get("experiment", {}) or {}).get("seed", 42)
        except Exception:
            pass
        for key, fname in src_files.items():
            if not fname:
                continue
            src_f = src_base / fname
            if not src_f.exists():
                continue
            dst_name = f"{exp_stem}_{key}.yaml"
            (exps_dir / dst_name).write_text(src_f.read_text(encoding="utf-8"), encoding="utf-8")
            files[key] = dst_name
    else:
        # 从头开始：生成一套最小默认子配置（干净模板）
        files = _blank_experiment_files(exps_dir, exp_stem)

    out = {
        "experiment": {
            "name": exp_stem,
            "model": model,
            "seed": seed,
            "files": files,
            "output": {"dir": "results/"},
        }
    }
    entry_path.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return jsonify({"ok": True, "experiment": f"experiments/{entry_path.name}", "name": exp_stem})


def _blank_experiment_files(exps_dir, exp_stem) -> dict:
    """从头开始：生成一份干净的最小实验模板（单 GPU + 通用映射/放置），供用户自由添加。"""
    stem = exp_stem
    hw = """# {stem}_hardware.yaml —— 空白模板（默认单 GPU，可自行加设备）
devices:
  - id: gpu0
    type: GPU
    compute:
      peak_tflops: 312
    efficiency:
      GEMM: 0.85
      LayerNorm: 0.25
      Softmax: 0.30
    memory:
      capacity_gb: 80
      read_bandwidth_gbs: 2039
      write_bandwidth_gbs: 2039
      read_latency_ns: 400
      write_latency_ns: 400
"""
    intc = f"# {stem}_interconnect.yaml —— 空白模板（暂无跨设备连接，可自行添加）\nlinks: []\n"
    mapping = f"""# {stem}_mapping.yaml —— 空白模板（默认所有算子放 GPU，可自行改）
default:
  compute_device: gpu0
  data_source: auto
rules:
  - op_type: GEMM
    device: gpu0
  - op_type: LayerNorm
    device: gpu0
  - op_type: Softmax
    device: gpu0
  - op_type: Residual
    device: gpu0
  - op_type: Activation
    device: gpu0
  - op_type: Embedding
    device: gpu0
  - op_type: LMHead
    device: gpu0
  - op_type: KVCacheUpdate
    device: gpu0
"""
    placement = f"""# {stem}_placement.yaml —— 空白模板（默认所有数据放 GPU）
default_device: gpu0
initial:
  - data_type: WEIGHT
    devices: [gpu0]
  - data_type: KV_CACHE
    devices: [gpu0]
  - data_type: ACTIVATION
    devices: [gpu0]
  - data_type: TEMPORARY
    devices: [gpu0]
"""
    files = {
        "hardware": f"{stem}_hardware.yaml",
        "interconnect": f"{stem}_interconnect.yaml",
        "mapping": f"{stem}_mapping.yaml",
        "placement": f"{stem}_placement.yaml",
    }
    for key, content in (("hardware", hw), ("interconnect", intc),
                         ("mapping", mapping), ("placement", placement)):
        (exps_dir / files[key]).write_text(content.format(stem=stem), encoding="utf-8")
    return files

def _next_exp_prefix(exps_dir):
    """取下一个实验序号（01、02 …）。"""
    nums = []
    for p in exps_dir.glob("*.yaml"):
        m = re.match(r"^(\d{2})_", p.name)
        if m:
            nums.append(int(m.group(1)))
    n = max(nums, default=0) + 1
    return f"{n:02d}"

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """算子运行参考：依据实验的 mapping.yaml（rules + default）为每个算子给出推荐运行设备。
    若规则/默认设备不在当前画布硬件里，则退化为"按效率选最优可用设备"。
    输入 {experiment, operators:[{id, op_type, precision}], hardware:[{id, type}]}
    输出 {recommend:{op_id: device}, note}"""
    from contracts import Operator
    d = request.get_json() or {}
    exp_path = (d.get("experiment") or "experiments/04_ic_reference.yaml").strip()
    ops_in = d.get("operators") or []
    hw_in = d.get("hardware") or []

    # 画布硬件 id 集合（含后端真实 id 和前端展示 id，尽量都算）
    hw_ids = set()
    for h in hw_in:
        hw_ids.add(h.get("id"))
        if h.get("backId"):
            hw_ids.add(h.get("backId"))

    recommendation, note = {}, ""
    try:
        ing = load_experiment(str(BASE/"configs"/exp_path))
        # 组装 Operator 供 MappingEngine.apply 匹配规则
        from contracts import Operator, PrecisionLevel
        ops = {}
        for o in ops_in:
            oid = o.get("id") or ""
            if not oid: continue
            ops[oid] = Operator(
                id=oid, name=oid, op_type=(o.get("op_type") or "GEMM"),
                flops=0,
                required_precision=_prec(o.get("precision") or "FP16", PrecisionLevel),
            )
        eng = MappingEngine(ing.cfg.mapping,
                            default_device=ing.cfg.mapping_default_device,
                            default_source=ing.cfg.mapping_default_source)
        specs = eng.apply(ops)
        for oid, spec in specs.items():
            dev = spec.compute_device
            # 若推荐设备不在画布硬件，找效率最优可用设备兜底
            if dev not in hw_ids:
                alt = _best_device_by_eff(ops[oid], hw_in)
                if alt:
                    recommendation[oid] = alt
                else:
                    recommendation[oid] = dev  # 保留 mapping 建议（可能画布还没加该硬件）
            else:
                recommendation[oid] = dev
        note = f"依据 {exp_path} 的映射规则生成 {len(recommendation)} 条运筹推荐。"
    except Exception as e:
        # mapping 加载失败时退化为按效率推荐
        for o in ops_in:
            oid = o.get("id") or ""
            if not oid: continue
            opt = o.get("op_type") or "GEMM"
            alt = _best_device_by_eff({"op_type": opt, "precision": o.get("precision") or "FP16"}, hw_in)
            if alt: recommendation[oid] = alt
        note = "映射加载失败，按各设备算力效率代荐：" + str(e)
    return jsonify({"recommend": recommendation, "note": note})

@app.route("/api/reference", methods=["POST"])
def api_reference():
    """一键部署参考：为整套画布生成【映射 + 完整数据流连线】并用校验器验证。
    输入（前端画布现状，id 均为画布块 id）：
      { experiment, hardware:[{id,type,backId,compute,...}],
        operators:[{id, kernelId, op_type, precision, inputs, outputs, intermediates, is_kv_dependent}] }
    输出：
      { ok, reference:{op_id: backId}, connections:[{from,to}], links:[{from,to,label}],
        note, valid, errors:[{code,message}] }
    连线策略（已通过校验规则 B1/B2/B3/B4/D1/C1 验证）：
      - 权重/初值：执行硬件_r -> 算子_in（权重驻留执行硬件）
      - 算子间数据：上游 算子_outN -> 下游 算子_inN
      - 中间值：算子_mid -> 执行硬件_w（写回），下游再从 执行硬件_r 读取
      - 每算子输出：算子_outN -> 执行硬件_w
      - 跨设备执行时自动补 GPU<->PIM 之类互连链路"""
    from contracts import Operator, PrecisionLevel
    d = request.get_json() or {}
    exp_path = (d.get("experiment") or "experiments/04_ic_reference.yaml").strip()
    hw_in = d.get("hardware") or []
    ops_in = d.get("operators") or []
    wb_in = d.get("weight_blocks") or []   # 前端权重块（含 device/bytes/num_layers）

    # 1) 每算子推荐设备（IC 参考策略）：backId
    recommendation = _recommend_devices(exp_path, ops_in, hw_in)

    # 2) backId -> 画布硬件块 id（供连线端口用）
    back2id = {}
    for h in hw_in:
        if h.get("backId"):
            back2id[h["backId"]] = h["id"]
    # id -> 硬件块
    hw_by_id = {h["id"]: h for h in hw_in}

    # 3) 构造连线（用画布块 id）。算子 id 即前端画布块 id（如 op0），端口 = opid_in{i}/_out{j}/_mid
    def producer_of(d):
        for o in ops_in:
            if d in (o.get("outputs") or []) or d in (o.get("intermediates") or []):
                return o
        return None

    conns, links = [], []
    # 已使用的执行硬件集合，用于跨设备加链路
    used_dev_ids = set()
    # 权重数据名集合（前端权重块负责连线，后端不重复生成）
    weight_names = {w.get("weight_id") for w in wb_in}
    for o in ops_in:
        oid = o.get("id") or ""
        if not oid:
            continue
        dev_back = recommendation.get(o.get("kernelId") or oid) or recommendation.get(oid)
        dev_id = back2id.get(dev_back)
        if not dev_id or dev_id not in hw_by_id:
            continue  # 该算子推荐设备不在画布，跳过（由前端提示）
        used_dev_ids.add(dev_id)
        for i, data in enumerate(o.get("inputs") or []):
            # 权重输入：由前端权重块端口连线负责，后端跳过（避免双套线）
            if data in weight_names:
                continue
            src = producer_of(data)
            if src and src.get("id") != oid:
                if data in (src.get("intermediates") or []):
                    # 中间值已写回目标执行硬件，从那里读
                    conns.append({"from": dev_id + "_r", "to": oid + "_in%d" % i})
                else:
                    srcd = (src.get("outputs") or []).index(data)
                    conns.append({"from": src["id"] + "_out%d" % srcd, "to": oid + "_in%d" % i})
            else:
                conns.append({"from": dev_id + "_r", "to": oid + "_in%d" % i})
        for j in range(len(o.get("outputs") or [])):
            conns.append({"from": oid + "_out%d" % j, "to": dev_id + "_w"})
        if o.get("intermediates"):
            conns.append({"from": oid + "_mid", "to": dev_id + "_w"})

    # 4) 跨设备执行：给涉及的设备两两补互连链路（GPU<->PIM 等，带宽取默认 120GB/s）
    used = sorted(used_dev_ids)
    for a in range(len(used)):
        for b in range(a + 1, len(used)):
            links.append({"from": used[a] + "_r", "to": used[b] + "_w", "label": "120", "isLink": True})
            links.append({"from": used[b] + "_r", "to": used[a] + "_w", "label": "120", "isLink": True})

    # 5) 用校验器验证这套配置（模拟前端 serializeState 结构）
    # 注意：校验 map 里 compute_map 的键是 kernelId（见 constraints._normalize），
    #       而连线端口用的是前端画布算子 id（op0 等）。
    compute_map = {}
    for o in ops_in:
        kid = o.get("kernelId") or o.get("id") or ""
        dev_back = recommendation.get(kid)
        dev_id = back2id.get(dev_back)
        if kid and dev_id:
            compute_map[kid] = dev_id
    all_conns = conns + links
    # 参考自校验：模拟前端会补上的权重端口线（前端 syncWeightConns），
    # 否则跳过了权重线导致 B1/W1 误报——参考的 valid 应反映"部署后"的真实状态。
    for w in wb_in:
        wid = w.get("weight_id") or ""
        if not wid:
            continue
        for kid, slot in (w.get("input_slots") or {}).items():
            oid = next((o.get("id") for o in ops_in if (o.get("kernelId") or o.get("id")) == kid), None)
            if oid:
                parts = w.get("partitions") or []
                if parts:
                    for p in parts:
                        all_conns.append({"from": p.get("partition_id") + "_r",
                                          "to": "%s_in%d" % (oid, slot)})
                else:
                    all_conns.append({"from": wid + "_r", "to": "%s_in%d" % (oid, slot)})
    try:
        vr = validate_config(hw_in, ops_in, all_conns, compute_map, wb_in)
        valid, errs = vr.valid, [{"code": i.code, "message": i.message} for i in vr.issues if i.level == "error"]
    except Exception as e:
        valid, errs = False, [{"code": "REF", "message": str(e)}]
    return jsonify({
        "ok": True,
        "reference": recommendation,
        "connections": conns,
        "links": links,
        "compute_map": compute_map,
        "valid": valid,
        "errors": errs,
        "note": "一键参考已生成：%d 个算子映射 + %d 条数据线 + %d 条互连链路。校验状态=%s" % (
            len(recommendation), len(conns), len(links), "通过" if valid else "未通过"),
    })

def _recommend_devices(exp_path, ops_in, hw_in):
    """计算每个算子的推荐设备（backId）。IC 参考策略（集成电路视角）：
      - attention 路径 GEMM（q/k/v/o_proj、attn_qk、attn_av）→ GPU（权重小、数据流内聚）
      - FFN GEMM（ffn_gate/up/down）→ DRAM-PIM（权重最大，就地计算）
      - 归约/逐元素（ln、softmax、silu）→ SRAM-PIM（带宽高）**但必须遵守精度能力**：
          FP32 需求的算子（LN/Softmax/RoPE）SRAM-PIM 执行精度不含 FP32 → 只能回退 GPU/CPU
      - resid/kv_update → 就近
    若推荐设备不在画布/不满足精度能力，退化为"画布上支持该算子精度的最优设备"（优先 GPU）。
    """
    from core.precision import HARDWARE_CAPABILITY, PrecisionLevel as _PL

    def _can_execute(hw_type, prec_name, is_data_only=False):
        """画布硬件 type 能否容纳该算子精度。
        is_data_only=True（execution=None，如 KV Cache）→ 查硬件 data 能力；
        否则查 execution 能力。"""
        cap = HARDWARE_CAPABILITY.get((hw_type or "").upper())
        if not cap:
            return True   # 自定义类型：不阻塞
        try:
            prec = _PL.from_name(prec_name or "FP16")
        except ValueError:
            prec = _PL.FP16
        lst = cap["data"] if is_data_only else cap["execution"]
        return prec in lst

    hw_ids = set()
    hw_type = {}
    for h in hw_in:
        hw_ids.add(h.get("id"))
        hw_type[h.get("id")] = h.get("type")
        if h.get("backId"):
            hw_ids.add(h.get("backId"))
            hw_type[h.get("backId")] = h.get("type")
    has = lambda bid: bid in hw_ids

    def _ic_dev(kid, op_type, prec):
        k = (kid or "").lower()
        ot = (op_type or "").upper()
        # 该算子需要 FP32 执行（e.g. LN/Softmax/RoPE）→ 只有 GPU/CPU 能跑，不要推到 SRAM
        need_fp32 = False
        try:
            need_fp32 = (_PL.from_name(prec or "FP16") == _PL.FP32)
        except ValueError:
            need_fp32 = False
        # attention 投影 + QK/AV：GPU
        if ot == "GEMM" and any(s in k for s in ("_proj", "attn_qk", "attn_av")):
            return "gpu0" if has("gpu0") else None
        # FFN 三个 GEMM：DRAM-PIM（大权重就近）；无 PIM 则 GPU
        if ot == "GEMM" and "ffn" in k:
            return "pim0" if (has("pim0") and not need_fp32) \
                else ("gpu0" if has("gpu0") else None)
        # LMHead / Embedding：词表权重放大到 GB 级 → 放 GPU（ReRAM 256MB 放不下）
        if ot in ("LMHEAD", "EMBEDDING"):
            return "gpu0" if has("gpu0") else None
        # 归约/逐元素：SRAM-PIM 高带宽；但 FP32(SRAM 不含)回退 GPU/CPU
        if ot in ("LAYERNORM", "SOFTMAX", "ACTIVATION", "RESIDUAL"):
            if need_fp32:
                return "gpu0" if has("gpu0") else ("cpu0" if has("cpu0") else None)
            return "sram0" if has("sram0") else ("cpu0" if has("cpu0") else ("gpu0" if has("gpu0") else None))
        # KV 写：大容量侧（execution 无要求）
        if ot == "KVCACHE_UPDATE":
            return "pim0" if has("pim0") else ("gpu0" if has("gpu0") else None)
        # 其余 GEMM 兜底
        if ot == "GEMM":
            return "gpu0" if has("gpu0") else None
        return "gpu0" if has("gpu0") else None

    recommendation = {}
    for o in ops_in:
        oid = o.get("kernelId") or o.get("id") or ""
        if not oid:
            continue
        # 精度判定用真实执行精度（execution_precision，None=纯数据算子→用 data_precision）
        exec_prec = o.get("execution_precision")
        is_data_only = exec_prec in (None, "", "None")
        prec = exec_prec or o.get("data_precision") or o.get("precision") or "FP16"
        dev = _ic_dev(oid, o.get("op_type") or "GEMM", prec)
        # 硬性过滤：若推荐的设备不满足该算子精度能力 → 改推 GPU（若也没有 GPU 则按效率）
        if dev and not _can_execute(hw_type.get(dev), prec, is_data_only=is_data_only):
            dev = "gpu0" if has("gpu0") else None
        if dev is None:
            # 推荐的设备类型画布上没有 → 按效率选最优可用设备
            alt = _best_device_by_eff({"op_type": o.get("op_type") or "GEMM",
                                       "precision": prec}, hw_in)
            recommendation[oid] = alt or ("gpu0" if has("gpu0") else "")
        else:
            recommendation[oid] = dev
    return recommendation

def _prec(s, PrecisionLevel):
    try:
        return PrecisionLevel.from_name(s or "FP16")
    except ValueError:
        return PrecisionLevel.FP16

def _best_device_by_eff(op, hw_in):
    """在画布硬件里，为 op 挑效率最高且支持其精度的设备。op:{op_type, precision}"""
    best, best_eff = None, -1
    for h in hw_in:
        # 精度支持
        prec = h.get("precision") or ""
        sup = h.get("supported") or ""
        # efficiency 表（前端序列化的 hardware 可能不带，用默认启发式）
        eff = _eff_guess(h, op.get("op_type") or "GEMM")
        if eff > best_eff:
            best_eff = eff; best = h.get("backId") or h.get("id")
    return best

def _eff_guess(h, op_type):
    """efficiency 启发式：GPU 偏好 GEMM/Attention，PIM 也偏 GEMM；其余给了较低值。"""
    htype = (h.get("type") or "").upper()
    base = {"GEMM":0.8,"ATTENTION":0.6,"SOFTMAX":0.3,"LAYERNORM":0.3,"RESIDUAL":0.5,
            "ACTIVATION":0.5,"LMHEAD":0.8,"EMBEDDING":0.8,"KVCACHE_UPDATE":0.7}.get(op_type.upper(),0.5)
    if "PIM" in htype:
        # PIM 对非 GEMM 类算子效率更低（参考 hardware.yaml 的 efficiency 表）
        return 0.78*base if op_type.upper()=="GEMM" else 0.5*base
    return base

@app.route("/api/validate", methods=["POST"])
def api_validate():
    """配置约束校验：{experiment, compute_map, state, splits} → 校验结果。
    若传了 experiment（前端点"校验配置"时），除前端结构校验外，还跑与运行完全一致的
    后端充分校验（validate_runnable）——保证"校验通过即可完整运行、校验不过则拒绝运行"。"""
    from experiment_runner import validate_runnable
    from config_loader import ConfigError
    d = request.get_json() or {}
    st = d.get("state") or {}
    # 前端结构校验用"画布 id"（st.compute_map 值=画布硬件 id hw0），与前端 serializeState 一致
    vr = validate_config(st.get("hardware") or d.get("hardware"),
                         st.get("operators") or d.get("operators"),
                         st.get("connections") or d.get("connections"),
                         st.get("compute_map") or d.get("compute_map"),
                         st.get("weight_blocks") or d.get("weight_blocks"),
                         link_table=st.get("link_table"))
    # 后端充分校验：与 /api/run 完全一致（真实 workload + 真实硬件 + 覆盖）
    exp_path = d.get("experiment") or ""
    if exp_path:
        run_map = d.get("compute_map") or st.get("run_map") or {}
        wb = _map_weight_devices(d.get("weight_blocks") or st.get("weight_blocks"),
                                 d.get("hardware") or st.get("hardware") or [])
        splits = _map_splits_devices(d.get("splits"), d.get("hardware") or st.get("hardware") or [])
        rv = validate_runnable(str(BASE / "configs" / exp_path),
                               compute_map_override=run_map, weight_blocks=wb,
                               splits_override=splits,
                               workload_override=d.get("workload"),
                               link_table_override=st.get("link_table"),
                               custom_hardware=st.get("hardware"))
        if not rv["ok"]:
            for msg in rv["errors"]:
                vr.issues.append(Issue(code="PRE", level="error", message=msg, targets=[]))
            vr.valid = False
    return jsonify(vr.to_dict())

@app.route("/api/run", methods=["POST"])
def api_run():
    d = request.get_json()
    # d = {experiment, compute_map, state:{hardware,operators,connections,compute_map,weight_blocks}}
    exp_path = d.get("experiment","experiments/04_ic_reference.yaml")
    compute_map = d.get("compute_map")   # {kernel_id: hw_back_id} 前端拖拽映射
    # ── 校验门禁：默认开启；前端可传 run_validation:false 关闭，强制放行 ──
    st = d.get("state") or {}
    if d.get("run_validation", True) and st:
        vr = validate_config(st.get("hardware"), st.get("operators"),
                             st.get("connections"), st.get("compute_map"),
                             st.get("weight_blocks"),
                             link_table=st.get("link_table"))
        if not vr.valid:
            return jsonify({"ok": False, "blocked": True, "validation": vr.to_dict()})
    fp = BASE / "configs" / exp_path
    try:
        # 前端 weight_blocks 里的 device 是画布硬件 id（hw0），调度器需要后端设备 id（gpu0）
        wb = _map_weight_devices(st.get("weight_blocks"), st.get("hardware") or [])
        # 前端算子切割规则（splits）→ 合并进 workloads.splits，真正驱动后端张量并行
        splits = _map_splits_devices(d.get("splits"), st.get("hardware") or [])
        r = run_workload_experiment(str(fp), compute_map_override=compute_map,
                                    save=True, verbose=False,
                                    weight_blocks=wb, splits_override=splits,
                                    workload_override=d.get("workload"),
                                    link_table_override=st.get("link_table"),
                                    custom_hardware=st.get("hardware"))
        result = r["result"]
        # v3: 结果面板需要搬运量明细 + 完成度诊断
        diag = result.diagnostics or {}
        mb = result.movement_bytes or {}
        # v3.2 关键路径（输出系统）
        from core.exporter import build_critical_path
        crit = build_critical_path(result)
        return jsonify({"ok":True, "total_latency_ms": result.total_latency_ns/1e6,
                        "bottleneck": result.bottleneck.name,
                        "breakdown": {"compute_ns": result.breakdown.compute_ns,
                                      "transfer_ns": result.breakdown.transfer_ns,
                                      "sync_ns": result.breakdown.sync_ns,
                                      "local_read_ns": result.breakdown.local_read_ns,
                                      "local_write_ns": result.breakdown.local_write_ns},
                        "op_count": len(result.operator_timings),
                        "override_applied": result.metadata.get("mapping_override_applied",0),
                        "weight_shard_count": result.metadata.get("weight_shard_count",0),
                        "rationale": result.bottleneck_rationale,
                        "movement_total_bytes": mb.get("total_bytes", 0),
                        "movement_per_link": mb.get("per_link", []),
                        "diagnostics": diag,
                        "critical_path": crit,
                        "completion_valid": bool(result.metadata.get("completion_valid", True)),
                        "completion_errors": result.metadata.get("completion_errors", [])})
    except Exception as e:
        from config_loader import ConfigError
        if isinstance(e, ConfigError):
            # 运行前充分校验未通过 → 阻止运行并提示（不改写结果文件）
            return jsonify({"ok": False, "blocked": True,
                            "validation": {"valid": False, "error_count": 1,
                                           "issues": [{"code": "PRE", "level": "error",
                                                       "message": str(e), "targets": []}]}})
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()[-2000:]}), 500


def _map_splits_devices(splits, hardware):
    """把前端算子切割规则里 devices 的画布硬件 id 映射为后端设备 id（未映射保留原值）。"""
    id2back = {h.get("id"): (h.get("backId") or h.get("id")) for h in (hardware or [])}
    out = []
    for s in (splits or []):
        s = dict(s)
        devs = s.get("devices") or []
        s["devices"] = [id2back.get(d, d) for d in devs]
        out.append(s)
    return out


def _map_weight_devices(weight_blocks, hardware):
    """把 weight_blocks 里的画布硬件 id（hw0）映射为后端设备 id（gpu0），
    供调度器做 placement/ALL-GATHER；未知 id 保留原值。"""
    id2back = {h.get("id"): (h.get("backId") or h.get("id")) for h in (hardware or [])}

    def _dev(v):
        return id2back.get(v, v)

    out = []
    for w in (weight_blocks or []):
        w = dict(w)
        w["device"] = _dev(w.get("device") or "")
        parts = []
        for p in (w.get("partitions") or []):
            p = dict(p)
            p["device"] = _dev(p.get("device") or "")
            parts.append(p)
        w["partitions"] = parts
        out.append(w)
    return out

@app.route("/api/workload")
def api_workload():
    model = request.args.get("model","llama_gb")
    MODELS = {
        "llama_gb": dict(hidden=16384, ffn=65536, heads=128, head_dim=128, vocab=32000, layers=16, seq=2048),
    }
    m = MODELS.get(model, MODELS["llama_gb"])
    # 用户可自定义 输入 token 数 与 自回归/生成 token 数（决定 KV 规模与序列规模）
    try:
        input_tokens = int(request.args.get("input_tokens", m["seq"]))
    except (TypeError, ValueError):
        input_tokens = m["seq"]
    try:
        generate_tokens = int(request.args.get("generate_tokens",
                               request.args.get("decode_steps", 128)))
    except (TypeError, ValueError):
        generate_tokens = 128
    from workload_model import build_model_workload
    wl = build_model_workload(
        hidden=m["hidden"], ffn_size=m["ffn"], num_heads=m["heads"],
        head_dim=m["head_dim"], vocab=m["vocab"], num_layers=m["layers"],
        input_tokens=input_tokens, generate_tokens=generate_tokens
    )
    resp = wl.to_dict()
    # 全局算子（Embedding / LMHead）不在 layers 里，单独返回供依赖图展示首尾两端
    resp["global_kernels"] = [k.to_dict() for k in wl.kernels
                              if k.id in ("embedding", "lm_head")]
    return jsonify(resp)

@app.route("/api/hardware_capability")
def api_hardware_capability():
    """返回 5 类硬件的三维能力表（类别 + 数据精度 + 执行精度），与后端精度系统
    core.precision.HARDWARE_CAPABILITY 完全一致（只读，单一事实来源）。
    前端据此在硬件块上展示"线性/非线性支持 + 数据/执行精度"，避免前后端各写一套。"""
    from core.precision import HARDWARE_CAPABILITY
    out = {}
    for htype, cap in HARDWARE_CAPABILITY.items():
        out[htype] = {
            "categories": [c.name for c in cap.get("categories", [])],
            "data": [p.name for p in cap.get("data", [])],
            "execution": [p.name for p in cap.get("execution", [])],
        }
    return jsonify(out)


@app.route("/api/link_defaults")
def api_link_defaults():
    """返回链路系统出厂默认表（7 种默认种类间的对称带宽，GB/s）+ 缺省带宽。
    前端据此初始化 N×N 链路带宽表，避免前后端各写一套（单一事实来源）。"""
    from core.link_sys import DEFAULT_LINK_BW_TABLE, DEFAULT_LINK_BW_GBS
    return jsonify({"table": DEFAULT_LINK_BW_TABLE, "fallback": DEFAULT_LINK_BW_GBS})


@app.route("/api/models")
def api_models():
    return jsonify(list_models())

@app.route("/api/split", methods=["POST"])
def api_split():
    """算子切割：{model, kernel, dim, parts:[...]} → 新的 kernel dict 列表"""
    from core.splitter import split_kernel_dict
    d = request.get_json()
    model = d.get("model", "llama_gb")
    kernel_id = d.get("kernel", "")
    dim = d.get("dim", "")
    parts = d.get("parts", [])
    if not dim or not isinstance(parts, list) or not len(parts):
        return jsonify({"error": "需要 dim 和 parts"}), 400
    # 从 workload 找到该 kernel
    MODELS = {"llama_gb":[16384,65536,128,128,32000,16,2048]}
    m = MODELS.get(model, MODELS["llama_gb"])
    from workload_model import build_model_workload
    wl = build_model_workload(hidden=m[0], ffn_size=m[1], num_heads=m[2],
                              head_dim=m[3], vocab=m[4], num_layers=m[5],
                              input_tokens=m[6], decode_steps=128)
    kernels = []
    for layer in wl.layers:
        kernels.extend(layer)
    target = None
    for k in kernels:
        kid = k.id if hasattr(k, 'id') else k.get('id')
        if kid == kernel_id:
            target = k
            break
    if target is None:
        return jsonify({"error": f"未找到算子 {kernel_id}"}), 404
    try:
        # Kernel 对象转 dict 再切割
        kdict = target.to_dict() if hasattr(target, 'to_dict') else target
        result = split_kernel_dict(kdict, dim, parts)
        return jsonify({"ok": True, "kernels": result})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()[-800:]}), 400

@app.route("/api/weights")
def api_weights():
    """权重块列表：按模型维度生成 WeightBlock（可切分），供前端渲染"权重块"节点。
    输入: ?model=llama_gb&experiment=xxx&split=W_mlp:2,W_attn:4 （split 可选：类名:片数）
    输出: {weight_blocks:[{weight_id, weight_class, layer, rows, cols, bytes, split_dim,
          consumers, input_slots, device(默认建议), partitions:[{partition_id, rows, cols, bytes, device}]}],
          default_device}
    """
    from weights import build_weight_blocks
    model = request.args.get("model", "llama_gb")
    split_arg = request.args.get("split", "")     # "W_mlp:2,W_attn:4"
    exp_path = (request.args.get("experiment") or "experiments/04_ic_reference.yaml").strip()
    MODELS = {
        "llama_gb": dict(hidden=16384, ffn=65536, heads=128, vocab=32000, layers=16),
    }
    m = MODELS.get(model, MODELS["llama_gb"])
    class_split = {}
    for part in split_arg.split(","):
        part = part.strip()
        if ":" in part:
            cls, n = part.split(":", 1)
            if cls in ("W_attn", "W_mlp", "W_ln", "W_head", "W_embed") and n.isdigit():
                class_split[cls] = int(n)
    # 只生成第 1 层权重块（与前端画布只显示 layers[0] 的算子对应），避免画布被 32 层撑爆
    blocks = build_weight_blocks(model, num_layers=1, h=m["hidden"],
                                 f=m["ffn"], nh=m["heads"], v=m["vocab"],
                                 precision_bytes=2, class_split=class_split or None)
    # IC 参考：按权重类别推荐放置设备（与 placement.yaml 一致）：
    #   W_attn→GPU、W_mlp→纯存储DRAM(dram_mem0)、W_ln→SRAM-PIM、W_head/W_embed→GPU
    #   （词表权重放大到 GB 级，ReRAM 256MB 放不下，故词表放 GPU）
    default_dev = _exp_default_device(exp_path) or "gpu0"
    cls_dev = {
        "W_attn": "gpu0", "W_mlp": "dram_mem0", "W_ln": "sram0",
        "W_head": "gpu0", "W_embed": "gpu0",
    }
    # 各权重类的实际数据精度字节数——与算子系统 OPERATOR_PRECISION_RULES 的数据精度一致：
    #   W_attn/W_mlp=INT8(1B)、W_head=FP8(1B)、W_embed=FP16(2B)、W_ln=FP32(4B，未建模兜底)
    CLS_BYTES = {"W_attn": 1, "W_mlp": 1, "W_ln": 4, "W_head": 1, "W_embed": 2}
    out = []
    for wb in blocks.values():
        dev = cls_dev.get(wb.weight_class) or default_dev
        pb = CLS_BYTES.get(wb.weight_class, 2)
        actual_bytes = wb.rows * wb.cols * pb
        # 全局权重（Embedding / LMHead，layer=None）只有一份；逐层权重才 × 全模型层数
        num_layers = 1 if wb.layer is None else m["layers"]
        out.append({
            "weight_id": wb.weight_id, "weight_class": wb.weight_class,
            "layer": wb.layer, "rows": wb.rows, "cols": wb.cols,
            "bytes": actual_bytes, "split_dim": wb.split_dim,
            "consumers": wb.consumers, "input_slots": wb.input_slots,
            "ports": wb.to_port_dict(),   # v3 结构化端口（权重→算子的数据流）
            "device": dev,
            "num_layers": num_layers,
            "partitions": [{"partition_id": p.partition_id, "rows": p.rows,
                             "cols": p.cols, "bytes": p.rows * p.cols * pb,
                             "device": p.device or dev}
                            for p in wb.partitions],
        })
    return jsonify({"weight_blocks": out, "default_device": default_dev})


def _exp_default_device(exp_path: str) -> str:
    """读取实验 mapping 的默认设备，作为权重块的默认放置建议。"""
    try:
        ing = load_experiment(str(BASE / "configs" / exp_path))
        return ing.cfg.mapping_default_device or ""
    except Exception:
        return ""


@app.route("/api/read")
def api_read():
    path = request.args.get("path","")
    fp = BASE / "configs" / path
    return jsonify({"content": fp.read_text(encoding="utf-8")}) if fp.exists() else ("",404)

@app.route("/api/write", methods=["POST"])
def api_write():
    d = request.get_json()
    (BASE/"configs"/d["path"]).write_text(d["content"], encoding="utf-8")
    return jsonify({"ok":True})

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>LLM-PIMSim v3 — 可视化拓扑编辑器</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1d212b;--text:#c8cdd6;--text2:#7a8494;--accent:#61afef;
  --green:#98c379;--orange:#d19a66;--purple:#c678dd;--red:#e06c75;--border:#2a2f3a;--border2:#39404d;
  --gpu:#61afef;--dram:#98c379;--sram:#e5c07b;--reram:#c678dd;--cpu:#abb2bf;
  --radius:8px;--shadow:0 6px 24px rgba(0,0,0,.45);}
*{margin:0;padding:0;box-sizing:border-box}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:#333a47;border-radius:5px}
::-webkit-scrollbar-thumb:hover{background:#444d5e}
::-webkit-scrollbar-track{background:transparent}
body{font:13px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--text);display:flex;height:100vh;overflow:hidden}
/* toolbar */
.toolbar{width:244px;background:linear-gradient(180deg,var(--panel),#13161c);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;overflow-y:auto}
.toolbar h2{font-size:15px;padding:16px 16px 8px;color:#fff;font-weight:700;letter-spacing:.5px}
.toolbar .section{padding:8px 12px;border-bottom:1px solid var(--border)}
.toolbar label{display:block;font-size:11px;color:var(--text2);margin-bottom:3px;text-transform:uppercase;letter-spacing:.5px}
.toolbar select,.toolbar input[type=text],.toolbar input[type=number]{width:100%;padding:6px 8px;border:1px solid var(--border);border-radius:4px;background:#2c313a;color:var(--text);font-size:12px;margin-bottom:8px}
.toolbar button{width:100%;padding:8px 12px;border:none;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;margin:3px 0;transition:all .15s}
.btn-add{background:#20242d;color:var(--text);border:1px dashed var(--border2)}
.btn-add:hover{background:#2a3040;border-color:var(--accent);color:var(--accent)}
.btn-run{background:linear-gradient(135deg,var(--accent),#3f9ff0);color:#0f1115;box-shadow:0 3px 12px rgba(97,175,239,.35)}
.btn-run:hover{background:linear-gradient(135deg,#6ec1ff,#4aa8ef);box-shadow:0 4px 16px rgba(97,175,239,.5)}
.btn-save{background:#20242d;color:var(--green);border:1px solid #3a5040}
.btn-save:hover{background:#2a3a2e}
.btn-validate{background:#20242d;color:var(--orange);border:1px solid #5a4630}
.btn-validate:hover{background:#3a3120}
/* canvas */
.canvas-wrap{flex:1;position:relative;overflow:auto;background:radial-gradient(circle,#2c313a 1px,transparent 1px);background-size:24px 24px}
#canvas{position:relative;width:4000px;height:3000px;min-width:100%;min-height:100%}
#svg-lines{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:1}
#svg-lines line,#svg-lines path{stroke:#5c6370;stroke-width:2;fill:none}
#svg-lines line.active,#svg-lines path.active{stroke:var(--accent);stroke-width:3}
/* blocks */
.block{position:absolute;border-radius:var(--radius);cursor:grab;user-select:none;z-index:2;transition:box-shadow .15s,transform .15s,border-color .15s}
.block:hover{z-index:10;border-color:var(--accent)!important}
.block.dragging{box-shadow:0 10px 40px rgba(0,0,0,.55);z-index:100;cursor:grabbing;opacity:.92;transform:scale(1.02)}
.block.selected{box-shadow:0 0 0 2px var(--accent),0 6px 20px rgba(0,0,0,.45)}
.hw-block{min-width:220px;max-width:300px;border:2px solid #4a5260;color:var(--text);box-shadow:var(--shadow)}
.hw-block .hw-header{padding:9px 12px;border-radius:var(--radius) var(--radius) 0 0;font-weight:700;font-size:12px;color:#0f1115;letter-spacing:.3px}
.hw-block .hw-body{padding:8px 12px;background:#1d212b;border-radius:0 0 var(--radius) var(--radius);font-size:11px}
.hw-block .hw-body .param{display:flex;justify-content:space-between;padding:2px 0;color:var(--text2)}
.hw-block .hw-body .param .val{color:var(--text)}
.op-block{min-width:280px;background:#1d212b;border:1px solid #39404d;padding:10px 28px;font-size:11px;border-radius:var(--radius);box-shadow:var(--shadow)}
.op-block .op-row{display:flex;justify-content:space-between;gap:8px;padding:2.5px 0;border-bottom:1px solid #232833;align-items:baseline}
.op-block .op-row:last-of-type{border-bottom:none}
.op-block .op-row>span:first-child{color:var(--text2);flex-shrink:0}
.op-block .op-row .val{color:#e8ecf2;text-align:right;word-break:break-all}
.op-block .op-row .c{color:var(--accent)}
.op-block .op-row .p{color:#c678dd;font-weight:600}
.op-block .op-row .m{color:var(--green)}
.op-block .op-row .mid{color:var(--purple);font-weight:600}
.op-block .op-row .o{color:var(--orange)}
.op-block .op-row .rec-val{color:#98c379;font-weight:600}
.op-block .op-row b{color:var(--accent)}
.op-block .split-btn{cursor:pointer;color:var(--accent);font-size:11px;font-weight:600}
.op-block .split-btn:hover{color:#93c5fd;text-decoration:underline}
.op-block .split-badge{display:inline-block}
/* 可拖入设备的标签（算子运行设备 / 权重存储设备） */
.op-tag,.w-tag{display:inline-block;padding:1px 9px;border-radius:11px;border:1px dashed var(--border2);color:var(--text2);font-weight:600;cursor:grab;user-select:none;background:#20242d;transition:all .12s}
.op-tag:hover,.w-tag:hover{border-color:var(--green);color:#98c379}
.op-tag.placed,.w-tag.placed{border-style:solid;border-color:var(--green);color:var(--green)}
.op-tag.dragging,.w-tag.dragging{opacity:.55}
/* weight blocks */
.w-block{min-width:210px;background:linear-gradient(180deg,#2a2438,#221d2e);border:1px solid #6a5acd;padding:8px 12px;font-size:11px;z-index:2;border-radius:var(--radius);box-shadow:var(--shadow)}
.w-block .w-head{display:flex;justify-content:space-between;align-items:center;gap:8px;border-bottom:1px solid #3d3552;padding-bottom:5px;margin-bottom:4px}
.w-block .w-head b{color:#c792ea;font-size:12px}
.w-block .w-class{display:inline-block;font-size:9px;font-weight:700;padding:1px 6px;border-radius:3px;background:#6a5acd;color:#fff;letter-spacing:.4px}
.w-block .w-row{display:flex;justify-content:space-between;gap:8px;padding:2px 0;color:var(--text2)}
.w-block .w-row .val{color:#e5e5e5;text-align:right;word-break:break-all}
.w-block .w-row .v{color:#c792ea;font-weight:600}
.w-block .w-dev{color:var(--green);font-weight:600}
.w-block .w-shard{border-top:1px dashed #4a3f66;margin-top:4px;padding-top:4px;font-size:10px;color:#b8a6e0}
.w-block .w-shard .sp{color:var(--text2)}
.w-block .w-tools{display:flex;gap:8px;margin-top:5px;border-top:1px dashed #3d3552;padding-top:4px}
.w-block .w-tools span{cursor:pointer;font-size:10px;font-weight:600}
.w-block .w-tools .sp-btn{color:#c792ea}
.w-block .w-tools .sp-btn:hover{text-decoration:underline}
.w-block .w-tools .del-btn{color:var(--red);margin-left:auto}
.w-block .w-tools .del-btn:hover{color:#f87171}
/* ports */
.port{position:absolute;width:14px;height:14px;border-radius:50%;border:2px solid #555;background:#2c313a;z-index:3;cursor:crosshair}
.port:hover{transform:scale(1.3);z-index:20}
.port .plabel{position:absolute;top:-16px;left:50%;transform:translateX(-50%);font-size:9px;color:var(--text2);white-space:nowrap;pointer-events:none;background:rgba(26,29,35,.8);padding:0 4px;border-radius:3px}
.port.read{right:-6px;top:50%;transform:translateY(-50%);border-color:var(--accent)}
.port.write{left:-6px;top:50%;transform:translateY(-50%);border-color:var(--green)}
.port.input{left:-6px;top:50%;transform:translateY(-50%);border-color:#999}
.port.output{right:-6px;top:50%;transform:translateY(-50%);border-color:var(--orange)}
.port.mid{left:50%;top:-12px;transform:translateX(-50%);border-color:var(--purple)}
.port.connected{background:var(--accent)}
/* status */
.status{position:fixed;bottom:12px;left:252px;font-size:11px;color:var(--text2);z-index:200;background:var(--panel);padding:6px 12px;border-radius:4px}
.result-overlay{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--panel);border:1px solid var(--border);border-radius:8px;z-index:300;display:none;min-width:380px;box-shadow:0 16px 64px rgba(0,0,0,.6)}
.result-overlay.show{display:block}
.result-overlay .ro-head{display:flex;align-items:center;justify-content:space-between;padding:12px 18px;border-bottom:1px solid var(--border);cursor:move;user-select:none}
.result-overlay .ro-head .ro-title{font-size:13px;font-weight:600;color:#fff}
.result-overlay .ro-close{margin:0;padding:2px 8px;background:none;border:none;color:var(--text2);font-size:18px;line-height:1;cursor:pointer;border-radius:4px}
.result-overlay .ro-close:hover{color:#fff;background:#3a4050}
.result-overlay .ro-body{padding:16px 18px;max-height:70vh;overflow:auto}
.result-overlay h3{color:#fff;margin:0 0 12px}
.result-overlay .metric{display:flex;justify-content:space-between;padding:4px 0;font-size:13px}
.result-overlay .metric .v{font-weight:600;color:var(--accent)}
.result-overlay .diag{border:1px solid var(--border);border-radius:6px;padding:8px 10px;background:#2c313a}
.result-overlay .diag.warn{border-color:#5a4630}
.result-overlay .diag.ok{border-color:#3a5040}
.result-overlay button{margin-top:12px;padding:8px 16px;background:var(--accent);color:#1a1d23;border:none;border-radius:4px;cursor:pointer;font-weight:600}
/* modal */
.modal-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.6);z-index:500;justify-content:center;align-items:center}
.modal-overlay.show{display:flex}
.modal{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:22px 26px;width:400px;max-height:88vh;overflow:auto;box-shadow:0 16px 48px rgba(0,0,0,.5)}
.drag-modal{position:relative;user-select:none}
.modal-head{display:flex;align-items:center;justify-content:space-between;margin:-8px 0 12px;cursor:move;user-select:none}
.modal-head>span{font-size:15px;font-weight:700;color:#fff}
.modal-x{margin:0;padding:0 6px;border:none;background:none;font-size:22px;line-height:1;color:var(--text2);cursor:pointer;border-radius:4px}
.modal-x:hover{color:#fff;background:#3a4050}
.modal h3{color:#fff;margin-bottom:16px;font-size:15px}
.modal label{display:block;font-size:11px;color:var(--text2);margin:8px 0 3px;text-transform:uppercase;letter-spacing:.4px}
.modal input,.modal select{width:100%;padding:7px 10px;border:1px solid var(--border);border-radius:5px;background:#2c313a;color:var(--text);font-size:12px;margin-bottom:4px}
.modal .btn-row{display:flex;gap:8px;margin-top:16px;justify-content:flex-end}
.modal .btn-row button{padding:8px 18px;border-radius:5px;font-size:12px;font-weight:600;cursor:pointer}
/* grouped operator */
.op-in-hw{margin:4px 6px;border:1px solid #444;border-radius:4px;padding:4px 8px;background:#2c313a;font-size:10px;display:flex;align-items:center;gap:6px;position:relative}
.op-in-hw .detach-btn{cursor:pointer;color:var(--red);font-weight:700;font-size:13px;line-height:1;margin-left:auto}
.op-in-hw .detach-btn:hover{color:#f87171}
/* connections list */
.conn-item{display:flex;align-items:center;gap:8px;padding:4px 0;font-size:11px;border-bottom:1px solid var(--border)}
.conn-item span{flex:1;color:var(--text2)}
.conn-item .del{color:var(--red);cursor:pointer;font-weight:700}
/* dependency drawer */
.dep-btn{position:fixed;left:244px;top:12px;z-index:400;background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:7px 14px;font-size:12px;font-weight:600;cursor:pointer;color:var(--text);box-shadow:0 2px 8px rgba(0,0,0,.3);display:flex;align-items:center;gap:6px}
.dep-btn:hover{border-color:var(--accent);color:var(--accent)}
.dep-drawer{position:fixed;right:0;top:0;bottom:0;width:340px;background:var(--panel);border-left:1px solid var(--border);z-index:450;transform:translateX(100%);transition:transform .2s;display:flex;flex-direction:column;box-shadow:-8px 0 24px rgba(0,0,0,.4)}
.dep-drawer.show{transform:translateX(0)}
.dep-header{display:flex;justify-content:space-between;align-items:center;padding:14px 16px;border-bottom:1px solid var(--border);font-weight:700;color:#fff;font-size:14px}
.dep-header .close{cursor:pointer;color:var(--red);font-size:18px;line-height:1}
.dep-body{flex:1;overflow-y:auto;padding:14px 16px}
.dep-layer{font-size:12px;color:var(--text2);margin:6px 0 8px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.dep-op{border:1px solid #3a4050;border-radius:5px;padding:7px 9px;margin-bottom:6px;background:#2c313a}
.dep-op .dep-opname{font-weight:600;color:#e5e5e5;font-size:11px}
.dep-op .dep-data{font-size:10px;color:var(--text2);margin-top:2px;line-height:1.5}
.dep-op .dep-data .tag{display:inline-block;padding:0 5px;border-radius:3px;margin-right:4px;font-size:9px}
.dep-op .in{background:#334;color:#9aa}
.dep-op .mid{background:#4a3355;color:#d9b8e6}
.dep-op .out{background:#554033;color:#e6c9a8}
</style></head>
<body>

<div class="toolbar" id="toolbar">
  <h2>LLM-PIMSim</h2>

  <div class="section">
    <label>模型</label>
    <select id="sel-model" onchange="loadModel()">
      <option value="llama_gb">llama（GB）</option>
    </select>
  </div>

  <div class="section">
    <label>序列规模（KV 随生成增长）</label>
    <div style="display:flex;gap:6px;align-items:flex-end">
      <div style="flex:1">
        <label style="margin-bottom:1px">输入 Token 数</label>
        <input type="number" id="input-tokens" value="2048" min="1" step="1" style="margin-bottom:0" onchange="onWorkloadChanged()">
      </div>
      <div style="flex:1">
        <label style="margin-bottom:1px">生成 Token 数</label>
        <input type="number" id="generate-tokens" value="128" min="0" step="1" style="margin-bottom:0" onchange="onWorkloadChanged()">
      </div>
    </div>
    <button class="btn-add" style="margin-top:6px" onclick="applyWorkload()">↻ 按规模重载模型</button>
    <div style="font-size:10px;color:var(--text2);margin-top:4px;line-height:1.5">输入=Prefill 序列长度；生成=自回归步数（KV 终点=输入+生成）。改后点重载，算子计算/存储量随 KV 规模变化。</div>
  </div>

  <div class="section">
    <label>依赖关系图</label>
    <button class="btn-add" onclick="toggleDepDrawer()">&#128196; 查看算子依赖关系</button>
  </div>

  <div class="section">
    <label>硬件</label>
    <button class="btn-add" onclick="addPresetHW('GPU')">GPU（预设）</button>
    <button class="btn-add" onclick="addPresetHW('DRAM_PIM')">DRAM-PIM（预设）</button>
    <button class="btn-add" onclick="addPresetHW('SRAM_PIM')">SRAM-PIM（预设）</button>
    <button class="btn-add" onclick="addPresetHW('RERAM_PIM')">ReRAM-PIM（预设）</button>
    <button class="btn-add" onclick="addPresetHW('CPU')">CPU（预设）</button>
    <button class="btn-add" onclick="addPresetHW('SRAM')" style="border-color:var(--border2);color:#56b6c2">SRAM（纯存储）</button>
    <button class="btn-add" onclick="addPresetHW('DRAM')" style="border-color:var(--border2);color:#d19a66">DRAM（纯存储）</button>
    <button class="btn-add" onclick="showCustomHWModal()" style="border-color:var(--accent);color:var(--accent)">+ 自定义硬件</button>
  </div>

  <div class="section">
    <label>实验</label>
    <div style="display:flex;gap:4px;align-items:center;flex-wrap:wrap">
      <select id="sel-experiment" style="flex:1;min-width:120px" onchange="applyRecommendation()">
        <option value="">加载中...</option>
      </select>
      <button class="btn-validate" title="新建实验（克隆现有实验模板）" onclick="openNewExperiment()">＋ 新建</button>
      <button class="btn-add" title="按所选实验的映射规则，把算子重新部署到推荐设备" onclick="applyRecommendation()">↻ 部署参考</button>
    </div>
    <div style="display:flex;gap:12px;align-items:center;margin-top:6px;font-size:11px;color:var(--text1)">
      <label style="display:flex;align-items:center;gap:4px;cursor:pointer">
        <input type="checkbox" id="check-validate" checked> 运行前校验
      </label>
      <span style="color:var(--text2)">算子方块=参考已放入硬件；点硬件内算子右侧 × 即可解出修改</span>
    </div>
    <div style="display:flex;gap:4px;margin-top:8px;flex-wrap:wrap">
      <button class="btn-validate" onclick="validateSim()">&#128269; 校验配置</button>
      <button class="btn-run" onclick="runSim()">&#9654; 运行仿真</button>
      <button class="btn-save" onclick="saveConfig()">保存配置</button>
      <button class="btn-validate" title="对比两个已保存的实验结果" onclick="openCompare()">⚖ 对比结果</button>
    </div>
  </div>

  <div class="section">
    <label>画布缩放（Ctrl+滚轮）</label>
    <div style="display:flex;gap:4px;align-items:center">
      <button class="btn-add" style="flex:1" onclick="zoomCanvas(1)">＋</button>
      <button class="btn-add" style="flex:1" onclick="zoomCanvas(-1)">－</button>
      <button class="btn-add" style="flex:1" onclick="resetCanvasZoom()">1:1</button>
      <button class="btn-add" style="flex:1" onclick="fitCanvas()">适应</button>
      <span id="zoom-label" style="font-size:11px;color:var(--text2);width:44px;text-align:right">100%</span>
    </div>
  </div>

  <!-- 结果对比弹层 -->
  <div id="cmp-overlay" class="modal-overlay">
    <div class="modal drag-modal" id="cmp-modal" style="width:640px">
      <div class="modal-head" data-drag="cmp-modal">
        <span>结果对比</span>
        <button class="modal-x" onclick="closeCompare()" title="关闭">&times;</button>
      </div>
      <label style="display:block;font-size:11px;color:var(--text2);margin-bottom:4px">选择要对比的结果（可多选，随后等后端对比）</label>
      <div id="cmp-list" style="max-height:150px;overflow:auto;border:1px solid var(--border);border-radius:5px;padding:6px"></div>
      <div id="cmp-msg" style="font-size:11px;color:var(--text2);margin-top:8px;min-height:14px"></div>
      <div style="display:flex;gap:8px;margin-top:12px;justify-content:flex-end">
        <button class="btn-validate" onclick="closeCompare()">取消</button>
        <button class="btn-run" onclick="doCompare()">对比</button>
      </div>
    </div>
  </div>

  <!-- 新建实验弹层 -->
  <div id="new-exp-overlay" class="modal-overlay">
    <div class="modal drag-modal" id="new-exp-modal">
      <div class="modal-head" data-drag="new-exp-modal">
        <span>新建实验</span>
        <button class="modal-x" onclick="closeNewExperiment()" title="关闭">&times;</button>
      </div>
      <label style="display:block;font-size:11px;color:var(--text2);margin-bottom:3px">实验名（字母/数字/下划线/连字符）</label>
      <input type="text" id="new-exp-name" placeholder="例如 my_pim_study" style="width:100%;box-sizing:border-box;padding:6px">
      <label style="display:block;font-size:11px;color:var(--text2);margin:10px 0 3px">模型</label>
      <select id="new-exp-model" style="width:100%;padding:6px;box-sizing:border-box"></select>
      <label style="display:block;font-size:11px;color:var(--text2);margin:12px 0 4px">起始方式</label>
      <div style="display:flex;gap:6px;margin-bottom:8px">
        <label style="flex:1;display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;padding:6px 8px;border:1px solid var(--border);border-radius:5px;background:#2c313a">
          <input type="radio" name="new-start" value="blank" checked onchange="onStartMode()"> 从头开始（空模板）
        </label>
        <label style="flex:1;display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;padding:6px 8px;border:1px solid var(--border);border-radius:5px;background:#2c313a">
          <input type="radio" name="new-start" value="ref" onchange="onStartMode()"> 从参考实验开始
        </label>
      </div>
      <div id="new-ref-box" style="display:none">
        <label style="display:block;font-size:11px;color:var(--text2);margin-bottom:3px">参考实验</label>
        <select id="new-exp-clone" style="width:100%;padding:6px;box-sizing:border-box"></select>
      </div>
      <div id="new-exp-msg" style="font-size:11px;margin-top:10px;min-height:16px"></div>
      <div style="display:flex;gap:8px;margin-top:10px;justify-content:flex-end">
        <button class="btn-validate" onclick="closeNewExperiment()">取消</button>
        <button class="btn-run" onclick="createExperiment()">创建</button>
      </div>
    </div>
  </div>

  <div class="section" style="font-size:10px;color:var(--text2);line-height:1.6">
    <b>操作提示：</b><br>
    拖拽算子/权重的「标签」到硬件方块 = 放置（运行/存储）<br>
    从读端口拖线到算子输入 = 数据源<br>
    从算子输出拖线到写端口 = 输出目标
  </div>
</div>

<div class="canvas-wrap" id="canvas-wrap">
  <div id="canvas">
    <svg id="svg-lines"></svg>
  </div>
</div>

<div class="status" id="status">就绪。拖拽硬件方块和算子来配置。</div>

<button id="conn-del" title="删除选中连线" onclick="delConn(selectedConn)"
  style="display:none;position:fixed;z-index:420;background:var(--red);color:#fff;border:none;border-radius:50%;width:22px;height:22px;line-height:20px;text-align:center;cursor:pointer;font-size:15px;font-weight:700;box-shadow:0 2px 8px rgba(0,0,0,.5)">×</button>

<div class="result-overlay" id="result-overlay">
  <div class="ro-head" id="ro-head">
    <span class="ro-title">LLM-PIMSim</span>
    <button class="ro-close" title="关闭" onclick="closeResult()">&times;</button>
  </div>
  <div class="ro-body" id="ro-body"></div>
</div>

<div class="dep-drawer" id="dep-drawer">
  <div class="dep-header">
    <span>算子依赖关系图</span>
    <span class="close" onclick="toggleDepDrawer()">&times;</span>
  </div>
  <div class="dep-body" id="dep-body"></div>
</div>

<div class="modal-overlay" id="split-modal">
  <div class="modal">
    <h3>算子切割（张量并行）</h3>
    <label>算子</label><input id="s-kernel" readonly>
    <label>切割维度</label><select id="s-dim"></select>
    <label>切成几片（≥2，等分）</label><input id="s-parts" value="2" type="number" min="2" step="1">
    <div id="s-preview" style="font-size:10px;color:var(--text2);margin-top:6px;line-height:1.5">切割后将生成多个算子图标，名称加 #1/#2…，计算量与存储量按片数等分，精度不变。</div>
    <div class="btn-row">
      <button onclick="closeSplitModal()" style="background:#444;color:#ccc">取消</button>
      <button onclick="doSplit()" style="background:var(--accent);color:#1a1d23">确定切割</button>
    </div>
  </div>
</div>

<div class="modal-overlay" id="hw-modal">
  <div class="modal">
    <h3>自定义硬件（1/2）</h3>
    <label>名称</label><input id="c-name" value="my-device">
    <label>类型</label><select id="c-type"><option>GPU</option><option>DRAM_PIM</option><option>SRAM_PIM</option><option>RERAM_PIM</option><option>CPU</option></select>
    <label>算力</label><input id="c-compute" value="100 TFLOPS">
    <label>容量</label><input id="c-mem" value="64 GB">
    <label>读带宽</label><input id="c-rbw" value="1000 GB/s">
    <label>写带宽</label><input id="c-wbw" value="800 GB/s">
    <label>精度（逗号分隔：FP32,FP16,INT8,INT4）</label><input id="c-precision" value="FP32,FP16,INT8,INT4">
    <div class="btn-row">
      <button onclick="closeCustomHWModal()" style="background:#444;color:#ccc">取消</button>
      <button onclick="nextCustomHW()" style="background:var(--accent);color:#1a1d23">下一步</button>
    </div>
  </div>
</div>

<div class="modal-overlay" id="hw-link-modal">
  <div class="modal">
    <h3>自定义硬件（2/2）· 链路带宽</h3>
    <div id="c-link-title" style="font-size:12px;color:var(--text2);margin-bottom:10px"></div>
    <label>到已有设备种类的链路带宽（GB/s）</label>
    <div id="c-links" style="max-height:260px;overflow:auto;border:1px solid var(--border);border-radius:5px;padding:6px;background:#1d212b"></div>
    <div class="btn-row">
      <button onclick="cancelCustomHWAll()" style="background:#444;color:#ccc">取消</button>
      <button onclick="backCustomHW()" style="background:#444;color:#ccc">上一步</button>
      <button onclick="confirmCustomHW()" style="background:var(--accent);color:#1a1d23">确定添加</button>
    </div>
  </div>
</div>

<script>
// ─── State ───
let blocks={}, hwCounter=0, opCounter=0, connections=[], linkId=0;
let dragState=null, connectState=null, selectedBlock=null, tagDragState=null;
let currentWorkload=null;   // 当前加载的 workload（供依赖图展示）
let pendingSplits=[];       // 前端算子切割规则，运行时会传给后端真实执行（张量并行）
let canvasScale=1;          // v3.1：画布缩放系数（Ctrl+滚轮 / 工具栏按钮）
let selectedConn=-1;        // 当前选中的连线在 connections 数组中的下标（-1=未选中）
// ─── 链路系统：N×N 对称带宽表（{kind: {kind: gbs}}），从后端 /api/link_defaults 初始化 ───
let linkBw={};
let LINK_FALLBACK=100;
// 前端预设硬件参数——与后端 core.hardware_sys.DEFAULT_DEVICE_PARAMS 对齐（容量保持 GB 级原值，不调大）
// v3.1：模型放大到 GB 级（单层 FFN 权重≈1GB）→ 设备容量保持小值，让内存/搬运成为约束。
//       CPU 为纯执行单元（容量≈0 + 4MB 缓存）；新增纯存储单元 SRAM / DRAM。
const HW_TYPES={
  GPU:{color:'#61afef',label:'GPU',compute:'312 TFLOPS',mem:'80 GB',rBW:'2039 GB/s',wBW:'2039 GB/s'},
  DRAM_PIM:{color:'#98c379',label:'DRAM-PIM',compute:'1.2 TFLOPS',mem:'8 GB',rBW:'307.2 GB/s',wBW:'307.2 GB/s'},
  SRAM_PIM:{color:'#e5c07b',label:'SRAM-PIM',compute:'500 TFLOPS',mem:'512 MB',rBW:'1500000 GB/s',wBW:'1500000 GB/s'},
  RERAM_PIM:{color:'#c678dd',label:'ReRAM-PIM',compute:'20 TFLOPS',mem:'256 MB',rBW:'128 GB/s',wBW:'32 GB/s'},
  CPU:{color:'#abb2bf',label:'CPU',compute:'45.9 TFLOPS',mem:'4 MB',rBW:'307.2 GB/s',wBW:'307.2 GB/s'},
  SRAM:{color:'#56b6c2',label:'SRAM（纯存储）',compute:'0 TFLOPS',mem:'256 MB',rBW:'1000 GB/s',wBW:'1000 GB/s'},
  DRAM:{color:'#d19a66',label:'DRAM（纯存储）',compute:'0 TFLOPS',mem:'64 GB',rBW:'200 GB/s',wBW:'200 GB/s'}
};
// 前端硬件类型 → 后端 experiment.yaml 里的真实设备 id（04_ic_reference 用 gpu0/pim0/sram0/rram0）
const PRESET_BACKID={GPU:'gpu0',DRAM_PIM:'pim0',SRAM_PIM:'sram0',RERAM_PIM:'reram0',CPU:'cpu0',SRAM:'sram_mem0',DRAM:'dram_mem0'};
// 硬件三维能力表（类别 + 数据精度 + 执行精度）——从后端 /api/hardware_capability 拉取，
// 保证与 core.precision.HARDWARE_CAPABILITY 完全一致（单一事实来源，不再前端另写一套）。
let HW_CAP={};
function _capSummary(type){
  let cap=HW_CAP[type];
  if(!cap) return `<div class="param"><span>能力</span><span style="color:var(--text2);font-size:9px">载入中…</span></div>`;
  // 纯存储单元：不执行任何算子，只展示可存精度
  if(!(cap.categories||[]).length){
    let data=(cap.data||[]).join('/');
    return `<div class="param"><span>类型</span><span class="val" style="color:#56b6c2">纯存储（不执行算子）</span></div>
      <div class="param"><span>可存精度</span><span class="val" style="color:#c678dd;font-size:10px">${data}</span></div>`;
  }
  let cats=(cap.categories||[]).map(c=>c==='NONLINEAR'?'非线性':'线性').join('+');
  let catColor=(cap.categories||[]).length>1?'#98c379':'#61afef';
  let data=(cap.data||[]).join('/');
  let exec=(cap.execution||[]).join('/');
  return `<div class="param"><span>类别</span><span class="val" style="color:${catColor}">${cats}</span></div>
    <div class="param"><span>数据精度</span><span class="val" style="color:#c678dd;font-size:10px">${data}</span></div>
    <div class="param"><span>执行精度</span><span class="val" style="color:#c678dd;font-size:10px">${exec}</span></div>`;
}

// ─── Hardware: preset (non-editable) + custom (editable via modal) ───
function addPresetHW(type){
  let t=HW_TYPES[type];if(!t)return;
  _makeHWBlock(type, t.label, t.compute, t.mem, t.rBW, t.wBW, false);
}
function _linkKinds(){
  // 当前链路表里已有的设备种类（大写、排序）；尚未加载则给 7 种默认
  let ks=Object.keys(linkBw).map(k=>String(k).toUpperCase());
  if(!ks.length) ks=['CPU','GPU','DRAM_PIM','SRAM_PIM','RERAM_PIM','SRAM','DRAM'];
  return Array.from(new Set(ks)).sort();
}
function _linkBwDefaultFor(kind){
  kind=String(kind).toUpperCase();
  let r=linkBw[kind]||{};
  return (r[kind]!=null && r[kind]>0) ? r[kind] : LINK_FALLBACK;
}
let pendingCustom={};   // 第一步填写的临时值，第二步填链路带宽后一起创建
function showCustomHWModal(){
  document.getElementById('hw-modal').classList.add('show');
}
function closeCustomHWModal(){document.getElementById('hw-modal').classList.remove('show')}
function closeCustomHWLinkModal(){document.getElementById('hw-link-modal').classList.remove('show')}
function cancelCustomHWAll(){closeCustomHWModal();closeCustomHWLinkModal();}
function nextCustomHW(){
  let name=document.getElementById('c-name').value||'custom';
  let id=name.toLowerCase().replace(/\s+/g,'-');
  pendingCustom={
    name:name, id:id, kind:id.toUpperCase(),
    type:document.getElementById('c-type').value,
    compute:document.getElementById('c-compute').value,
    mem:document.getElementById('c-mem').value,
    rBW:document.getElementById('c-rbw').value,
    wBW:document.getElementById('c-wbw').value,
    precision:document.getElementById('c-precision').value,
  };
  // 填充第二个弹窗的链路带宽输入
  let box=document.getElementById('c-links');
  let rows=_linkKinds().map(k=>
    `<div style="display:flex;align-items:center;gap:6px;margin:3px 0">
       <span style="flex:1;font-size:11px;color:var(--text2)">↔ ${k}</span>
       <input data-lk="${k}" type="number" step="1" min="0" value="${_linkBwDefaultFor(k)}" style="width:90px;padding:3px 6px;background:#2c313a;border:1px solid var(--border);color:var(--text);border-radius:3px">
     </div>`).join('');
  rows+=`<div style="display:flex;align-items:center;gap:6px;margin:3px 0">
     <span style="flex:1;font-size:11px;color:var(--text2)">↔ 同种类互连</span>
     <input data-lk="__self__" type="number" step="1" min="0" value="${LINK_FALLBACK}" style="width:90px;padding:3px 6px;background:#2c313a;border:1px solid var(--border);color:var(--text);border-radius:3px">
   </div>`;
  box.innerHTML=rows;
  document.getElementById('c-link-title').textContent='设备 '+pendingCustom.name+'（种类 '+pendingCustom.kind+'）';
  document.getElementById('hw-modal').classList.remove('show');
  document.getElementById('hw-link-modal').classList.add('show');
}
function backCustomHW(){
  document.getElementById('hw-link-modal').classList.remove('show');
  document.getElementById('hw-modal').classList.add('show');
}
function confirmCustomHW(){
  let p=pendingCustom;
  let color=HW_TYPES[p.type]?HW_TYPES[p.type].color:'#abb2bf';
  // 收集链路带宽（对称写入全局表 linkBw）
  let links={};
  document.querySelectorAll('#c-links input[data-lk]').forEach(inp=>{
    let k=inp.getAttribute('data-lk');
    let v=parseFloat(inp.value);
    if(isNaN(v)||v<=0) v=LINK_FALLBACK;
    if(k==='__self__'){ links[p.kind]=v; }
    else { links[String(k).toUpperCase()]=v; }
  });
  linkBw[p.kind]=linkBw[p.kind]||{};
  Object.entries(links).forEach(([k,v])=>{
    if(k===p.kind){ linkBw[p.kind][p.kind]=v; }
    else { linkBw[p.kind][k]=v; linkBw[k]=linkBw[k]||{}; linkBw[k][p.kind]=v; }
  });
  _makeHWBlock(p.type, p.name+' ('+p.type+')',
    p.compute, p.mem, p.rBW, p.wBW, true, p.id, color,
    p.id, p.precision, p.kind, links);
  closeCustomHWLinkModal();
}

function _makeHWBlock(type, label, compute, mem, rBW, wBW, editable, forceId, forceColor, backId, precisionStr, linkType, links){
  let id=forceId||('hw'+(hwCounter++));
  if(!backId) backId=PRESET_BACKID[type]||id;   // 预设类型→后端真实id
  let color=forceColor||HW_TYPES[type].color;
  // 硬件块纵向排列（避免重叠遮挡参数）：按已有硬件数量向下排
  let hwIdx = Object.keys(blocks).filter(k=>blocks[k] && blocks[k].type && blocks[k].type!=='operator' && blocks[k].type!=='weight').length;
  let x=20, y=40+hwIdx*220;
  if(!precisionStr) precisionStr=(HW_TYPES[type]&&HW_TYPES[type].precision)||'FP32/FP16/INT8/INT4';
  if(blocks[id]){blocks[id].el.remove();delete blocks[id]}
  let el=document.createElement('div');
  el.className='block hw-block';el.id=id;el.style.left=x+'px';el.style.top=y+'px';
  el.style.borderColor=color;
  let paramsHTML=editable
    ? `<div class="param"><span>算力</span><input value="${compute}" onchange="uHP('${id}','compute',this.value)" style="width:90px;background:#1e2229;border:1px solid #444;color:#ccc;font-size:10px;padding:2px 4px;border-radius:3px;text-align:right"></div>
       <div class="param"><span>容量</span><input value="${mem}" onchange="uHP('${id}','mem',this.value)" style="width:90px;background:#1e2229;border:1px solid #444;color:#ccc;font-size:10px;padding:2px 4px;border-radius:3px;text-align:right"></div>
       <div class="param"><span>读带宽</span><input value="${rBW}" onchange="uHP('${id}','rBW',this.value)" style="width:90px;background:#1e2229;border:1px solid #444;color:#ccc;font-size:10px;padding:2px 4px;border-radius:3px;text-align:right"></div>
       <div class="param"><span>写带宽</span><input value="${wBW}" onchange="uHP('${id}','wBW',this.value)" style="width:90px;background:#1e2229;border:1px solid #444;color:#ccc;font-size:10px;padding:2px 4px;border-radius:3px;text-align:right"></div>
       ${_capSummary(type)}`
    : `<div class="param"><span>算力</span><span class="val">${compute}</span></div>
       <div class="param"><span>容量</span><span class="val">${mem}</span></div>
       <div class="param"><span>读带宽</span><span class="val">${rBW}</span></div>
       <div class="param"><span>写带宽</span><span class="val">${wBW}</span></div>
       ${_capSummary(type)}`;
  el.innerHTML=`
    <div class="hw-header" style="background:${color};display:flex;justify-content:space-between;align-items:center">
      <span>${label} <span style="font-weight:400;font-size:10px">${id}</span></span>
      <span style="cursor:pointer;font-size:16px;line-height:1" onclick="deleteBlock('${id}')" title="删除">&times;</span>
    </div>
    <div class="hw-body" id="${id}_body">${paramsHTML}</div>
    <div class="port read" data-port="${id}_r" data-port-type="read" title="读端口"><span class="plabel">读</span></div>
    <div class="port write" data-port="${id}_w" data-port-type="write" title="写端口"><span class="plabel">写</span></div>
  `;
  document.getElementById('canvas').appendChild(el);
  blocks[id]={type:type,el:el,x:x,y:y,opGroup:[],params:{compute,mem,rBW,wBW},editable,backId,precision:precisionStr,
              linkType:String(linkType||type).toUpperCase(), links:links||{}};
  el.querySelectorAll('.port').forEach(p=>p.addEventListener('mousedown',portMouseDown));
  makeDraggable(el);
  updateLinkSelects();
  updateStatus('已添加 '+label);
}
function uHP(id,k,v){if(blocks[id]){blocks[id].params=blocks[id].params||{};blocks[id].params[k]=v}}
function deleteBlock(id){
  if(!blocks[id])return;
  // detach grouped operators
  (blocks[id].opGroup||[]).forEach(od=>{detachOp(od)});
  blocks[id].el.remove();
  connections=connections.filter(c=>!c.from.startsWith(id+'_')&&!c.to.startsWith(id+'_'));
  // 若删除的是硬件：该硬件上的权重块恢复为未放置
  Object.entries(blocks).forEach(([wid,b])=>{ if(b.type==='weight'&&b.parentHW===id){detachWeight(wid);} });
  delete blocks[id];
  drawConnections();updateLinkSelects();refreshConnList();
  updateStatus('已删除 '+id);
}

// ─── Load operators from model ───
function getWorkloadParams(){
  let it=parseInt(document.getElementById('input-tokens').value);
  let gt=parseInt(document.getElementById('generate-tokens').value);
  if(!it||it<1) it=2048;
  if(isNaN(gt)||gt<0) gt=128;
  return {input_tokens:it, generate_tokens:gt};
}
function applyWorkload(){ loadModel(); }
function onWorkloadChanged(){
  // 用户改规模后仅提示，点击"重载"才真正刷新（避免每次键入都重拉）
  updateStatus('序列规模已修改：输入='+(document.getElementById('input-tokens').value||'?')+
    ' 生成='+(document.getElementById('generate-tokens').value||'?')+' —— 点「按规模重载模型」生效');
}
// 算子类别（线性/非线性）：与后端校验器 core.validator._op_category 一致
function _opCategory(opType){
  let o=(opType||'').toUpperCase();
  return (['LAYERNORM','SOFTMAX','ACTIVATION','RESIDUAL','ROPE'].includes(o))?'NONLINEAR':'LINEAR';
}
function loadModel(){
  let m=document.getElementById('sel-model').value;
  let wp=getWorkloadParams();
  fetch('/api/workload?model='+m+'&input_tokens='+wp.input_tokens+'&generate_tokens='+wp.generate_tokens)
  .then(r=>r.json()).then(wl=>{
    // 清除旧算子
    Object.keys(blocks).filter(k=>k.startsWith('op')).forEach(k=>{blocks[k].el.remove();delete blocks[k]});
    opCounter=0;
    // 存下当前 workload（供依赖图展示）
    currentWorkload=wl;
    let kernels=wl.layers?.[0]||[];
    if(!kernels.length)return;
    let x0=600,y0=40;
    kernels.forEach((k,i)=>{
      let id='op'+(opCounter++);
      let x=x0+Math.floor(i/4)*400,y=y0+(i%4)*330;
      let kvHint=k.is_kv_dependent?'KV动态':'';
      let cat=_opCategory(k.op_type||'');
      let catLabel=cat==='NONLINEAR'?'非线性':'线性';
      let catColor=cat==='NONLINEAR'?'#d19a66':'#61afef';
      let dataPrec=k.data_precision||k.precision||'FP16';
      let execPrec=k.execution_precision||'—(纯数据)';
      // 输入/输出/中间数据
      let inData=(k.inputs||[]).join(', ')||'—';
      let outData=(k.outputs||[]).join(', ')||'—';
      let midData=(k.intermediates||[]).join(', ')||'';
      // attributes 序列化（M/K/N 或 seq 等）
      let attrs=Object.entries(k.attributes||{}).map(([a,b])=>a+'='+b).join(', ')||'—';
      let el=document.createElement('div');
      el.className='block op-block';el.id=id;el.style.left=x+'px';el.style.top=y+'px';
      // 输入端口在左、输出端口在右（多端口纵向排布）
      let nIn=(k.inputs||[]).length, nOut=(k.outputs||[]).length;
      let inTop=(nIn>1?(di)=>20+di*60/(nIn-1):()=>50);
      let outTop=(nOut>1?(do_)=>20+do_*60/(nOut-1):()=>50);
      let inPorts=(k.inputs||[]).map((d,di)=>`<div class="port input" data-port="${id}_in${di}" data-port-type="input" style="top:${inTop(di)}%" title="输入${di+1}: ${d}"><span class="plabel">in${di+1}</span></div>`).join('');
      let outPorts=(k.outputs||[]).map((d,do_)=>`<div class="port output" data-port="${id}_out${do_}" data-port-type="output" style="top:${outTop(do_)}%" title="输出${do_+1}: ${d}"><span class="plabel">out${do_+1}</span></div>`).join('');
      let midPort=(k.intermediates||[]).length?`<div class="port mid" data-port="${id}_mid" data-port-type="mid" title="中间值: ${(k.intermediates||[]).join(', ')}"><span class="plabel">中间值</span></div>`:'';
      el.innerHTML=`
        <div class="op-row"><span>算子</span><b>${k.name||k.op_type||''}</b><span style="margin-left:auto;font-size:9px;font-weight:700;color:${catColor};border:1px solid ${catColor};border-radius:3px;padding:0 4px">${catLabel}</span>${kvHint?`<span style="font-size:9px;color:#c792ea;font-weight:600">${kvHint}</span>`:''}</div>
        <div class="op-row"><span>类型</span><span class="val">${k.op_type||''}</span></div>
        <div class="op-row"><span>ID</span><span class="val">${k.id||''}</span></div>
        <div class="op-row"><span>数据精度</span><span class="val p">${dataPrec}</span></div>
        <div class="op-row"><span>执行精度</span><span class="val p">${execPrec}</span></div>
        <div class="op-row"><span>计算量</span><span class="val c">${k.compute_gflops||'—'}</span></div>
        <div class="op-row"><span>存储量</span><span class="val m">${k.memory||'—'}</span></div>
        ${midData?`<div class="op-row"><span>中间值</span><span class="val mid">${midData}</span></div>`:''}
        <div class="op-row"><span>形状</span><span class="val">${attrs}</span></div>
        <div class="op-row"><span>输入</span><span class="val" style="color:#7aa2c4">${inData}</span></div>
        <div class="op-row"><span>输出</span><span class="val" style="color:#d19a66">${outData}</span></div>
        <div class="op-row rec-row"><span>推荐设备</span><span class="val rec-val" data-rec="1">…(载入中)</span></div>
        <div class="op-row"><span>运行设备</span><span class="op-tag" data-tag="1" title="拖动此标签到硬件方块 = 算子在该设备运行">—(未放置)</span></div>
        <div style="margin-top:6px;border-top:1px dashed #3a4050;padding-top:4px"><span class="split-btn" onclick="openSplitModal('${id}')">✂ 切割</span></div>
        ${inPorts}${outPorts}${midPort}
      `;
      document.getElementById('canvas').appendChild(el);
      blocks[id]={type:'operator',el:el,x:x,y:y,data:k,kernelId:k.id||id,name:(k.name||k.op_type||''),
        inputs:(k.inputs||[]),outputs:(k.outputs||[]),intermediates:(k.intermediates||[]),
        isSlice:false,sliceOf:null,recEl:el.querySelector('.rec-val'),tagEl:el.querySelector('.op-tag')};
      el.querySelectorAll('.port').forEach(p=>p.addEventListener('mousedown',portMouseDown));
      makeTagDraggable(blocks[id].tagEl, id);
      makeDraggable(el);
    });
    updateStatus('已加载模型: '+m+', 每层'+kernels.length+'个算子 (输入'+wp.input_tokens+' token, 生成'+wp.generate_tokens+' token)');
    renderLayerBar();
    loadWeights().then(()=>applyRecommendation());
  }).catch(e=>updateStatus('加载模型失败: '+e));
}

// 需求1 层折叠：显示"L0 模板 + 其余层折叠"，提示摆放会按 mapping 规则应用到全部层
function renderLayerBar(){
  let total = (currentWorkload && currentWorkload.num_layers) || 32;
  let bar=document.getElementById('layer-bar');
  if(!bar){
    bar=document.createElement('div');
    bar.id='layer-bar';
    bar.style.cssText='position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:150;background:rgba(23,26,33,.92);border:1px solid var(--border2);border-radius:20px;padding:6px 16px;font-size:12px;color:var(--text);display:flex;gap:10px;align-items:center;box-shadow:var(--shadow);backdrop-filter:blur(6px)';
    document.getElementById('canvas-wrap').appendChild(bar);
  }
  bar.innerHTML=`<span style="color:var(--accent);font-weight:600">📚 图层</span>
    <span style="color:var(--text)">显示 <b>L0 模板</b></span>
    <span style="color:var(--text2)">＋ ${total-1} 层折叠</span>
    <span title="算子/权重的标签拖入硬件后会按 mapping 规则应用到全部层" style="color:var(--green);cursor:help">✅ 摆放→全部层</span>`;
}

// ─── 权重块：画布一级节点（用户先决定权重放哪个硬件）───
let currentWeights=[];   // 后端返回的 WeightBlock 列表（供依赖图/切割用）
let wCounter=0;
function loadWeights(){
  let m=document.getElementById('sel-model').value;
  let exp=document.getElementById('sel-experiment')?document.getElementById('sel-experiment').value:'';
  // 清除旧权重块（含硬件体内的紧凑条目）
  Object.keys(blocks).filter(k=>blocks[k].type==='weight').forEach(k=>{
    let grp=document.getElementById(k+'_wgrp'); if(grp) grp.remove();
    blocks[k].el.remove(); delete blocks[k];
  });
  Object.keys(blocks).forEach(hid=>{ if(blocks[hid].weightGroup) blocks[hid].weightGroup=[]; });
  wCounter=0;
  return fetch('/api/weights?model='+m+'&experiment='+encodeURIComponent(exp||'experiments/04_ic_reference.yaml'))
  .then(r=>r.json()).then(d=>{
    currentWeights=d.weight_blocks||[];
    // backId -> 画布硬件块 id（用于自动放置默认设备）
    let backToId={};
    Object.entries(blocks).forEach(([id,b])=>{
      if(b.type!=='operator' && b.backId) backToId[b.backId]=id;
    });
    // 权重块放在算子区下方，按类别分组横向排布（避开算子网格）
    let x0=600,y0=1400;
    let col=0,row=0,perRow=6;
    currentWeights.forEach((wb,i)=>{
      let id='w'+(wCounter++);
      let x=x0+(i%perRow)*300, y=y0+Math.floor(i/perRow)*150;
      let parts=wb.partitions||[];
      let shardHTML=parts.length
        ? `<div class="w-shard">已切 <b style="color:#c792ea">${parts.length}</b> 片：<span class="sp">${parts.map(p=>p.partition_id).join(', ')}</span></div>`
        : '';
      let el=document.createElement('div');
      el.className='block w-block';el.id=id;el.style.left=x+'px';el.style.top=y+'px';
      // 权重端口：未切割 1 个读端口 {weightId}_r；切割后每个分片一个端口 {partition_id}_r（垂直排布）
      let portHTML = parts.length
        ? parts.map((p,pi)=>`<div class="port read" data-port="${p.partition_id}_r" data-port-type="read" style="top:${15+pi*28}%" title="权重分片: ${p.partition_id}"><span class="plabel">片${pi+1}</span></div>`).join('')
        : `<div class="port read" data-port="${wb.weight_id}_r" data-port-type="read" style="top:50%" title="权重: ${wb.weight_id}"><span class="plabel">权重</span></div>`;
      el.innerHTML=`
        <div class="w-head"><b title="${wb.weight_id}">${wb.weight_id}</b><span class="w-class">${wb.weight_class||''}</span></div>
        <div class="w-row"><span>形状</span><span class="val v">${wb.rows}×${wb.cols}</span></div>
        <div class="w-row"><span>大小</span><span class="val">${fmtBytes(wb.bytes)}</span></div>
        <div class="w-row"><span>消费者</span><span class="val">${(wb.consumers||[]).length} 个算子</span></div>
        <div class="w-row"><span>存储设备</span><span class="w-tag" data-tag="1" title="拖动此标签到硬件方块 = 权重存储在该设备">—(未放置)</span></div>
        ${shardHTML}
        <div class="w-tools">
          <span class="sp-btn" onclick="splitWeight('${id}')">✂ 切割</span>
          <span class="del-btn" onclick="deleteWeight('${id}')">×</span>
        </div>
        ${portHTML}
      `;
      document.getElementById('canvas').appendChild(el);
      blocks[id]={type:'weight',el:el,x:x,y:y,weightId:wb.weight_id,weightClass:wb.weight_class,
        consumers:wb.consumers||[],inputSlots:wb.input_slots||{},parts:parts,
        bytes:wb.bytes||0,numLayers:wb.num_layers||1,
        device:'',parentHW:null,tagEl:el.querySelector('.w-tag'),devEl:el.querySelector('.w-tag')};
      makeTagDraggable(blocks[id].tagEl, id);
      makeDraggable(el);
      el.querySelectorAll('.port').forEach(p=>p.addEventListener('mousedown',portMouseDown));
      // 默认设备建议（backId）→ 若画布有对应硬件，自动放置
      let hwId=backToId[wb.device];
      if(hwId){ groupWeightToHW(id, hwId); }
    });
    updateStatus('已加载 '+currentWeights.length+' 个权重块——拖动其标签到硬件方块即可决定权重放置');
  }).catch(e=>{updateStatus('加载权重块失败: '+e); currentWeights=[];});
}
function fmtBytes(b){
  if(!b&&b!==0)return '—';
  if(b>=1e9)return (b/1e9).toFixed(1)+' GB';
  if(b>=1e6)return (b/1e6).toFixed(1)+' MB';
  if(b>=1e3)return (b/1e3).toFixed(1)+' KB';
  return b+' B';
}
// 权重标签拖入硬件 → 记录设备 + 在硬件体内生成紧凑条目；权重主体仍留在画布上供连线
function groupWeightToHW(wid, hwId){
  let wb=blocks[wid], hw=blocks[hwId]; if(!wb||!hw)return;
  // 若已放其它硬件，先解除旧放置
  if(wb.parentHW && wb.parentHW!==hwId) detachWeight(wid);
  wb.parentHW=hwId;
  wb.device=hw.backId||hwId;
  // 在硬件体内生成紧凑权重条目（权重主体仍保留在画布上）
  let item=document.getElementById(wid+'_wgrp');
  if(!item){
    item=document.createElement('div');
    item.className='op-in-hw'; item.id=wid+'_wgrp';
    item.style.borderColor='#6a5acd';
    let wbadge = wb.isPartition
      ? `<span style="color:#e5c07b">(片${wb.partIndex+1}/${wb.partTotal})</span>`
      : ((wb.parts&&wb.parts.length)?`<span style="color:#b8a6e0">(${wb.parts.length}片)</span>`:'');
    item.innerHTML=`<span style="color:#c792ea;font-weight:600" title="${wb.weightId}">${wb.weightId}</span>
      <span style="color:var(--text2)">${wb.weightClass||''} · ${fmtBytes(wb.bytes)}</span>
      ${wbadge}
      <span class="detach-btn" onclick="detachWeight('${wid}')" title="解除放置">&times;</span>`;
    hw.el.querySelector('.hw-body').appendChild(item);
  }
  hw.weightGroup=hw.weightGroup||[]; if(!hw.weightGroup.includes(wid)) hw.weightGroup.push(wid);
  _updateWTag(wid);
  syncWeightConns(wid);
  updateStatus('权重 '+wb.weightId+' 已放置到 '+hwId+'（该硬件存储此权重）');
}
function detachWeight(wid){
  let wb=blocks[wid]; if(!wb)return;
  let oldHW=wb.parentHW;
  // 移除硬件内的紧凑条目
  let item=document.getElementById(wid+'_wgrp'); if(item) item.remove();
  if(oldHW&&blocks[oldHW]) blocks[oldHW].weightGroup=(blocks[oldHW].weightGroup||[]).filter(x=>x!==wid);
  wb.parentHW=null; wb.device='';
  _updateWTag(wid);
  syncWeightConns(wid);
}
function deleteWeight(wid){
  let wb=blocks[wid]; if(!wb)return;
  // 移除该权重块自动生成的连线（按 _autoW 标记）与硬件内条目
  connections=connections.filter(c=>c._autoW!==wid);
  let grp=document.getElementById(wid+'_wgrp'); if(grp) grp.remove();
  if(wb.parentHW && blocks[wb.parentHW]){
    blocks[wb.parentHW].weightGroup=(blocks[wb.parentHW].weightGroup||[]).filter(x=>x!==wid);
  }
  wb.el.remove(); delete blocks[wid];
  drawConnections(); refreshConnList();
  updateStatus('已删除权重块 '+wb.weightId);
}
// 为该权重块自动生成/刷新连线：从权重所在硬件的"读端口"（{hwId}_r）出发，连到消费
// 该权重的算子输入端口。权重块放入硬件后会隐藏，改用硬件读端口表示"该硬件提供此权重"，
// 与校验器 W1（权重完整性/ALL-GATHER）的"设备读端口可达"判定一致，且连线始终可见。
function syncWeightConns(wid){
  let wb=blocks[wid]; if(!wb)return;
  // 移除该权重块现有的自动连线
  connections=connections.filter(c=>c._autoW!==wid);
  if(!wb.parentHW) return;
  let hwId=wb.parentHW;
  // 分片块：算子输入里引用的是"原权重名"（partitionOf）；整块：引用 weightId 自身
  let wname = wb.isPartition ? (wb.partitionOf||wb.weightId) : wb.weightId;
  // 找消费该权重的算子块：inputs 里包含该权重名
  Object.entries(blocks).forEach(([oid,b])=>{
    if(b.type!=='operator')return;
    let idx=(b.inputs||[]).indexOf(wname);
    if(idx<0) return;
    connections.push({from:hwId+'_r', to:oid+'_in'+idx, label:'W:'+wb.weightId, lat:0, _autoW:wid});
  });
  drawConnections(); refreshConnList();
}
// 切割权重：输入片数 → 拉取分片 → 生成 N 个独立权重分片图标（可分别拖入不同硬件，ALL-GATHER）
function splitWeight(wid){
  let wb=blocks[wid]; if(!wb)return;
  if(wb.isPartition){updateStatus('该块已是权重分片，无需再切');return}
  let cls=wb.weightClass||'';
  let n=prompt('将权重类 '+cls+' 切成几片？（如 2）','2');
  if(!n)return;
  n=parseInt(n);
  if(!n||n<2){updateStatus('片数必须≥2');return}
  let m=document.getElementById('sel-model').value;
  let exp=document.getElementById('sel-experiment')?document.getElementById('sel-experiment').value:'';
  fetch('/api/weights?model='+m+'&experiment='+encodeURIComponent(exp||'experiments/04_ic_reference.yaml')+'&split='+cls+':'+n)
  .then(r=>r.json()).then(d=>{
    let nb=(d.weight_blocks||[]).find(x=>x.weight_id===wb.weightId);
    if(!nb){updateStatus('切割失败：未找到权重 '+wb.weightId);return}
    let parts=nb.partitions||[];
    if(!parts.length){updateStatus('切割失败：后端未返回分片');return}
    _replaceWithWeightParts(wid, wb, parts);
    updateStatus('权重 '+wb.weightId+' 已切成 '+parts.length+' 片（生成 '+parts.length+' 个权重分片图标，可分别拖入不同硬件，ALL-GATHER）');
  }).catch(e=>updateStatus('切割失败: '+e));
}
// 把整块权重替换为 N 个独立分片图标（每个分片可单独拖入不同硬件，存储量已等分）
function _replaceWithWeightParts(wid, wb, parts){
  let oldX=wb.x, oldY=wb.y;
  // 移除原块及其硬件内条目、自动连线
  let grp=document.getElementById(wid+'_wgrp'); if(grp) grp.remove();
  if(wb.parentHW && blocks[wb.parentHW]){
    blocks[wb.parentHW].weightGroup=(blocks[wb.parentHW].weightGroup||[]).filter(x=>x!==wid);
  }
  connections=connections.filter(c=>c._autoW!==wid);
  wb.el.remove(); delete blocks[wid];
  parts.forEach((p,pi)=>{
    let pid=wid+'_p'+pi;
    let x=oldX+pi*280, y=oldY;
    let el=document.createElement('div');
    el.className='block w-block'; el.id=pid; el.style.left=x+'px'; el.style.top=y+'px';
    el.innerHTML=`
      <div class="w-head"><b title="${p.partition_id}">${p.partition_id}</b><span class="w-class">${wb.weightClass||''}</span><span style="margin-left:auto;font-size:9px;font-weight:700;color:#e5c07b;border:1px solid #e5c07b;border-radius:3px;padding:0 4px">片${pi+1}/${parts.length}</span></div>
      <div class="w-row"><span>形状</span><span class="val v">${p.rows}×${p.cols}</span></div>
      <div class="w-row"><span>大小</span><span class="val">${fmtBytes(p.bytes||0)}</span></div>
      <div class="w-row"><span>消费者</span><span class="val">${(wb.consumers||[]).length} 个算子</span></div>
      <div class="w-row"><span>存储设备</span><span class="w-tag" data-tag="1" title="拖动此标签到硬件方块 = 该权重分片存储在该设备">—(未放置)</span></div>
      <div class="w-tools"><span class="del-btn" onclick="deleteWeight('${pid}')">×</span></div>
    `;
    document.getElementById('canvas').appendChild(el);
    blocks[pid]={type:'weight',el:el,x:x,y:y,
      weightId:p.partition_id, weightClass:wb.weightClass,
      partitionOf:wb.weightId, isPartition:true, partIndex:pi, partTotal:parts.length,
      consumers:wb.consumers||[], inputSlots:wb.inputSlots||{},
      parts:[], bytes:p.bytes||0, numLayers:wb.numLayers||1,
      device:'', parentHW:null, tagEl:el.querySelector('.w-tag'), devEl:el.querySelector('.w-tag')};
    makeTagDraggable(blocks[pid].tagEl, pid);
    makeDraggable(el);
  });
  drawConnections(); refreshConnList(); updateLinkSelects();
}
// 收集画布权重块 → 校验/运行用的 weight_blocks
// 分片图标（isPartition）按 partitionOf 归组回原权重，每个分片可放不同设备（ALL-GATHER）。
// num_layers 用每个权重自己的值：全局权重（Embedding/LMHead）=1，逐层权重=全模型层数。
function collectWeightBlocks(){
  let wEntries=[];
  Object.entries(blocks).forEach(([id,b])=>{
    if(b.type!=='weight')return;
    wEntries.push({id,b});
  });
  // 按"父权重"分组：未分片 = weightId 自身；分片 = partitionOf
  let groups={};
  wEntries.forEach(({id,b})=>{
    let parent=b.partitionOf||b.weightId;
    (groups[parent]=groups[parent]||[]).push({id,b});
  });
  let out=[];
  Object.keys(groups).forEach(parent=>{
    let items=groups[parent];
    let main=items[0].b;
    if(main.isPartition){
      // 分片权重：每个分片一个 device
      let partitions=items.map(({id,b})=>({
        partition_id:b.weightId, bytes:b.bytes||0, device:b.parentHW||''
      }));
      out.push({
        weight_id:parent, weight_class:main.weightClass,
        consumers:main.consumers||[], input_slots:main.inputSlots||{},
        device:'', bytes:0, num_layers:main.numLayers||1, partitions:partitions
      });
    } else {
      // 未分片整块
      let hwId=main.parentHW||'';
      let parts=(main.parts||[]).map(p=>({partition_id:p.partition_id, bytes:p.bytes||0, device:hwId}));
      out.push({
        weight_id:main.weightId, weight_class:main.weightClass,
        consumers:main.consumers||[], input_slots:main.inputSlots||{},
        device:hwId, bytes:main.bytes||0, num_layers:main.numLayers||1, partitions:parts
      });
    }
  });
  return out;
}

// ─── 算子运行参考（推荐设备）───
function applyRecommendation(){
  // 先重置当前参考：把所有已放进硬件的算子叉出来，清空连线，再按新参考重新部署
  Object.keys(blocks).forEach(oid=>{
    let b=blocks[oid];
    // 切片（用户显式张量并行）不被"部署参考"重置，保留其独立放置
    if(b && b.type==='operator' && b.parentHW && !b.isSlice){ try{detachOp(oid);}catch(e){} }
  });
  connections=[]; drawConnections(); refreshConnList(); updateLinkSelects();
  loadWeights().then(()=>_applyRef());   // 等权重就绪后再部署参考，避免权重块未就绪的竞态
}

function _applyRef(){
  // 收集当前画布硬件与算子（连同 inputs/outputs/intermediates 供后端构造数据流连线）
  let hardware=[], operators=[];
  Object.entries(blocks).forEach(([id,b])=>{
    if(b.type==='operator'){
      if(b.isSlice) return;   // 切片块为视觉展示，参考部署按逻辑算子处理（由 pendingSplits 驱动）
      let k=b.data||{};
      operators.push({id:id, kernelId:b.kernelId||id, op_type:k.op_type||'', precision:k.precision||'FP16',
        inputs:b.inputs||[], outputs:b.outputs||[], intermediates:b.intermediates||[],
        is_kv_dependent:!!k.is_kv_dependent,
        data_precision:k.data_precision||null, execution_precision:k.execution_precision||null});
    } else {
      let p=b.params||{};
      hardware.push({id:id,type:b.type,backId:b.backId,precision:b.precision,
        compute:p.compute, mem:p.mem});
    }
  });
  let exp=document.getElementById('sel-experiment')?document.getElementById('sel-experiment').value:'';
  fetch('/api/reference',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({experiment:exp||'experiments/04_ic_reference.yaml', operators:operators, hardware:hardware,
      weight_blocks:collectWeightBlocks()})})
  .then(r=>r.json()).then(d=>{
    let rec=d.reference||{};
    // 把后端返回的设备后端 id 映射回画布硬件块 id
    let backToId={};
    Object.entries(blocks).forEach(([id,b])=>{
      if(b.type!=='operator' && b.backId) backToId[b.backId]=id;
    });
    // 1) 把算子放入推荐硬件
    let deployed=0;
    Object.entries(blocks).forEach(([id,b])=>{
      if(b.type!=='operator'||b.isSlice)return;   // 切片块保持浮动，不入硬件
      let kid=b.kernelId||id;
      let dev=rec[kid];
      let hwBlockId=backToId[dev]||null;
      if(dev && hwBlockId){
        groupOpToHW(id, hwBlockId);
        if(b.recEl) b.recEl.textContent=hwBlockId;
        b.recommendedDevice=hwBlockId;
        b.recommendedBackId=dev;
        deployed++;
      } else {
        if(b.recEl) b.recEl.textContent=(dev?backToId[dev]||dev:(hwBlockId||'—'))+(dev&&!hwBlockId?'(未加硬件)':'');
        b.recommendedDevice=hwBlockId;
        b.recommendedBackId=dev;
      }
    });
    // 2) 应用参考的数据流连线 + 互连链路；跳过"权重输入"连线（权重连线统一由
    //    权重块端口生成，避免同一条权重输入出现多套线 → 画布上"很多同名权重"）
    let refConns=(d.connections||[]).concat(d.links||[]);
    refConns.forEach(c=>{
      // 若目标端口是某算子的输入端口，且该输入数据名是权重名 → 跳过，由权重端口线负责
      let m=/^(op\d+)_in(\d+)$/.exec(c.to||'');
      if(m){
        let b=blocks[m[1]];
        let dname=(b&&b.inputs)?b.inputs[parseInt(m[2])]:null;
        if(dname && currentWeights.some(w=>w.weight_id===dname)) return;
      }
      connections.push({from:c.from, to:c.to, label:c.label||'', lat:c.lat||0, isLink:!!c.isLink});
    });
    drawConnections(); refreshConnList(); updateLinkSelects();
    let status=(d.note||'')+(d.valid?'  ✓ 校验通过':'  ✗ 校验未通过，请查看提示');
    if(d.valid===false && d.errors&&d.errors.length){
      status+=' | '+d.errors.map(e=>e.code+':'+e.message).join(' ; ').slice(0,120);
    }
    updateStatus(status);
    // 重建权重自动连线（applyRecommendation 开头清空了 connections）
    Object.keys(blocks).forEach(wid=>{ if(blocks[wid].type==='weight') syncWeightConns(wid); });
  }).catch(e=>{
    Object.entries(blocks).forEach(([id,b])=>{ if(b.type==='operator'&&b.recEl) b.recEl.textContent='—'; });
    updateStatus('参考部署失败: '+e.message);
  });
}

// ─── Drag & drop ───
function makeDraggable(el){
  el.addEventListener('mousedown',e=>{
    if(e.target.classList.contains('port'))return;
    // 标签是独立拖拽目标（拖入设备=放置），不触发主体拖拽
    if(e.target.closest && (e.target.closest('.op-tag')||e.target.closest('.w-tag')))return;
    if(e.button!==0)return;
    e.preventDefault();
    document.querySelectorAll('.block.selected').forEach(b=>b.classList.remove('selected'));
    el.classList.add('selected');
    selectedBlock=el;
    let rect=el.getBoundingClientRect();
    dragState={el:el,ox:e.clientX-rect.left,oy:e.clientY-rect.top,startX:parseInt(el.style.left),startY:parseInt(el.style.top)};
    el.classList.add('dragging');
  });
}
document.addEventListener('mousemove',e=>{
  if(dragState){
    let wrap=document.getElementById('canvas-wrap');
    let dx=(e.clientX-dragState.ox)/canvasScale, dy=(e.clientY-dragState.oy)/canvasScale;
    let wx=wrap.scrollLeft,wy=wrap.scrollTop;
    dragState.el.style.left=(dx+wx-252)+'px';
    dragState.el.style.top=(dy+wy-0)+'px';
    drawConnections();
  }
});
document.addEventListener('mouseup',e=>{
  // 标签拖放：鼠标落到某个硬件上方 → 算子运行于此/权重存储于此
  if(tagDragState){
    let td=tagDragState;
    tagDragState=null;
    document.body.style.cursor='';
    if(td.el) td.el.classList.remove('dragging');
    let hwId=_hwAtPoint(e.clientX,e.clientY);
    if(hwId){
      if(td.type==='weight') groupWeightToHW(td.id, hwId);
      else groupOpToHW(td.id, hwId);
    }
    return;
  }
  // 主体拖拽：仅移动位置，不再"整块放入设备"
  if(dragState){
    dragState.el.classList.remove('dragging');
    let id=dragState.el.id;
    if(blocks[id]){
      blocks[id].x=parseInt(dragState.el.style.left);
      blocks[id].y=parseInt(dragState.el.style.top);
    }
    dragState=null;
  }
});
// 标签：独立拖拽目标（拖入设备=放置）
function makeTagDraggable(tagEl, id){
  if(!tagEl)return;
  tagEl.addEventListener('mousedown',e=>{
    e.stopPropagation(); e.preventDefault();
    if(e.button!==0)return;
    let b=blocks[id]; if(!b)return;
    tagDragState={id:id, type:b.type, el:tagEl};
    tagEl.classList.add('dragging');
    document.body.style.cursor='grabbing';
  });
}
function _hwAtPoint(cx,cy){
  let found=null;
  Object.entries(blocks).forEach(([hid,b])=>{
    if(!b.type||b.type==='operator'||b.type==='weight')return;
    let r=b.el.getBoundingClientRect();
    if(cx>=r.left&&cx<=r.right&&cy>=r.top&&cy<=r.bottom) found=hid;
  });
  return found;
}

// ─── 标签文案（放置状态）───
function _tagText(hwId){ return hwId ? ('→ '+hwId) : '—(未放置)'; }
function _updateOpTag(id){
  let b=blocks[id]; if(!b||b.type!=='operator'||!b.tagEl)return;
  b.tagEl.textContent=_tagText(b.parentHW);
  b.tagEl.classList.toggle('placed', !!b.parentHW);
}
function _updateWTag(id){
  let b=blocks[id]; if(!b||b.type!=='weight'||!b.tagEl)return;
  b.tagEl.textContent=_tagText(b.parentHW);
  b.tagEl.classList.toggle('placed', !!b.parentHW);
}

// ─── Operator-Hardware Grouping ───
// 第一层实时校验（需求3）：拖入时快速提示（容量/精度），不强制阻止，标橙警告
function quickCheckOp(hw, k){
  if(!hw||!k)return null;
  let p=hw.params||{};
  let mem=_parseUnit(p.mem); let peak=_parseUnit(p.compute);
  let opMem=_parseUnit(k.memory_bytes_range?k.memory_bytes_range[1]:0);
  let warn=[];
  if(peak<=0) warn.push('该硬件算力为0');
  if(opMem>0&&mem>0&&opMem>mem) warn.push('超出容量 '+(opMem/1e6).toFixed(0)+'MB');
  return warn.length?warn.join('；'):null;
}
function _parseUnit(str){
  if(str==null)return 0; if(typeof str==='number')return str;
  let m=String(str).match(/^([\d.]+)\s*(B|KB|MB|GB|TB|FLOPS|TOPS|TFLOPS|GB\/s)?$/i);
  if(!m)return parseFloat(str)||0;
  let v=parseFloat(m[1]); let u=(m[2]||'').toUpperCase();
  if(u==='GB'||u==='TFLOPS'||u==='GB/S')return v*1e9;
  if(u==='MB')return v*1e6; if(u==='KB')return v*1e3; if(u==='TB')return v*1e12;
  return v;
}
function groupOpToHW(opId, hwId){
  let op=blocks[opId], hw=blocks[hwId]; if(!op||!hw)return;
  let w=quickCheckOp(hw, op.data||{});
  if(w) updateStatus('⚠ 实时校验：算子 '+opId+' 放入 '+'('+w+')');
  if(op.parentHW) detachOp(opId);
  // 在硬件体内生成紧凑条目（表示"算子运行于此设备"）
  let k=op.data||{};
  let item=document.createElement('div');
  item.className='op-in-hw';item.id=opId+'_grp';
  item.innerHTML=`<span style="color:#e5e5e5">${k.id||opId}</span><span style="color:var(--text2)">${k.type||k.kernel_type||''}</span><span class="detach-btn" onclick="detachOp('${opId}')" title="解除映射">&times;</span>`;
  hw.el.querySelector('.hw-body').appendChild(item);
  // 算子块保持可见：其 in/out 端口保留，作为数据流连线的锚点；
  // 仅用 parentHW 记录"算子运行在此设备"。
  op.parentHW=hwId;
  hw.opGroup=hw.opGroup||[];hw.opGroup.push(opId);
  hw.el.style.minHeight='auto';
  // 切片：标签放置 = 真正决定该片运行设备（更新张量并行规则里对应槽位）
  if(op.splitRuleIdx!=null && op.sliceIdxInRule!=null){
    let rule=pendingSplits[op.splitRuleIdx];
    if(rule && Array.isArray(rule.devices)){
      rule.devices[op.sliceIdxInRule]=hw.backId||hwId;
    }
  }
  _updateOpTag(opId);
  drawConnections();   // 重绘：数据与算子同设备的线隐藏，跨设备线保留
  updateStatus('算子 '+opId+' 已映射到 '+hwId);
}
function detachOp(opId){
  let op=blocks[opId]; if(!op||!op.parentHW)return;
  let hw=blocks[op.parentHW];
  // Remove grouped item
  let grp=document.getElementById(opId+'_grp');if(grp)grp.remove();
  // Clean up（算子主体仍留在画布原位，仅清除映射记录）
  if(hw) hw.opGroup=(hw.opGroup||[]).filter(x=>x!=opId);
  op.parentHW=null;
  // 切片解除放置：清空其张量并行规则里对应槽位的设备
  if(op.splitRuleIdx!=null && op.sliceIdxInRule!=null){
    let rule=pendingSplits[op.splitRuleIdx];
    if(rule && Array.isArray(rule.devices)) rule.devices[op.sliceIdxInRule]='';
  }
  _updateOpTag(opId);
  drawConnections();   // 解除映射后重绘：跨设备数据线重新显示
  updateStatus('算子 '+opId+' 已解除映射');
}

// ─── Connections list ───
function refreshConnList(){
  let el=document.getElementById('conn-list');if(!el)return;
  // 用真实索引（connections 全数组下标），避免过滤后下标错位导致删错/删不掉
  el.innerHTML=connections.map((c,idx)=>({c,idx})).filter(x=>x.c.isLink)
    .map(({c,idx})=>`<div class="conn-item"><span>${c.from.replace('_r','')} ↔ ${c.to.replace('_w','')} @ ${c.label||'?'}</span><span class="del" onclick="delConn(${idx})" title="删除">&times;</span></div>`).join('');
}
function delConn(i){
  if(i==null || i<0 || i>=connections.length)return;
  connections.splice(i,1);
  selectedConn=-1;
  drawConnections();refreshConnList();updateLinkSelects();
  updateStatus('连接已删除');
}

// ─── Port connections ───
function portMouseDown(e){
  e.stopPropagation();e.preventDefault();
  let portId=e.target.dataset.port;
  connectState={from:portId,el:e.target};
  e.target.classList.add('connected');
}
document.addEventListener('mousemove',e=>{
  if(!connectState)return;
  let svg=document.getElementById('svg-lines');
  svg.querySelectorAll('.temp-line').forEach(l=>l.remove());
  // 端点相对 canvas 原点的坐标（#canvas 与 #svg-lines 同一坐标系；视觉坐标 ÷ scale）
  let fromRect=connectState.el.getBoundingClientRect();
  let canvasRect=document.getElementById('canvas').getBoundingClientRect();
  let x1=(fromRect.left+fromRect.width/2-canvasRect.left)/canvasScale;
  let y1=(fromRect.top+fromRect.height/2-canvasRect.top)/canvasScale;
  // 鼠标相对 canvas 原点的坐标
  let x2=(e.clientX-canvasRect.left)/canvasScale;
  let y2=(e.clientY-canvasRect.top)/canvasScale;
  let line=document.createElementNS('http://www.w3.org/2000/svg','line');
  line.setAttribute('x1',x1);line.setAttribute('y1',y1);
  line.setAttribute('x2',x2);line.setAttribute('y2',y2);
  line.setAttribute('stroke','#61afef');line.setAttribute('stroke-width','2');
  line.setAttribute('stroke-dasharray','6 3');
  line.classList.add('temp-line');
  svg.appendChild(line);
});
document.addEventListener('mouseup',e=>{
  if(!connectState)return;
  document.getElementById('svg-lines').querySelectorAll('.temp-line').forEach(l=>l.remove());
  connectState.el.classList.remove('connected');
  // Find target port
  let target=document.elementFromPoint(e.clientX,e.clientY);
  if(target&&target.dataset.port&&target.dataset.port!==connectState.from){
    connections.push({from:connectState.from,to:target.dataset.port});
    updateStatus('已连接: '+connectState.from+' �� '+target.dataset.port);
    drawConnections();
  }
  connectState=null;
});

// ─── Draw connections ───
function drawConnections(){
  let svg=document.getElementById('svg-lines');
  svg.innerHTML='';
  let canvasRect=document.getElementById('canvas').getBoundingClientRect();
  let delBtn=document.getElementById('conn-del');
  let selMid=null;
  connections.forEach((c,i)=>{
    let fromEl=document.querySelector('[data-port="'+c.from+'"]');
    let toEl=document.querySelector('[data-port="'+c.to+'"]');
    if(!fromEl||!toEl)return;
    // 端口或其祖先不可见（如权重块折叠进硬件）→ 不绘制，避免线残留在外面
    if(!_portVisible(fromEl)||!_portVisible(toEl))return;
    // 数据与算子同设备（无需跨设备搬运）→ 隐藏该条数据流线
    if(_isCoLocated(c))return;
    let fRect=fromEl.getBoundingClientRect(),tRect=toEl.getBoundingClientRect();
    let x1=(fRect.left+fRect.width/2-canvasRect.left)/canvasScale;
    let y1=(fRect.top+fRect.height/2-canvasRect.top)/canvasScale;
    let x2=(tRect.left+tRect.width/2-canvasRect.left)/canvasScale;
    let y2=(tRect.top+tRect.height/2-canvasRect.top)/canvasScale;
    let path=document.createElementNS('http://www.w3.org/2000/svg','path');
    let mx=(x1+x2)/2, my=(y1+y2)/2;
    path.setAttribute('d',`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`);
    path.setAttribute('stroke','#5c6370');path.setAttribute('stroke-width','2');
    path.setAttribute('fill','none');
    path.setAttribute('pointer-events','stroke');   // 只让线本身可点击，不遮挡下方方块
    path.style.cursor='pointer';
    path.setAttribute('data-idx',i);
    path.addEventListener('click',ev=>{ev.stopPropagation();selectConn(i);});
    if(i===selectedConn){
      path.setAttribute('class','active');
      selMid={x:canvasRect.left+mx*canvasScale, y:canvasRect.top+my*canvasScale};
    }
    svg.appendChild(path);
  });
  // 定位选中线中点的"× 删除"按钮
  if(delBtn){
    if(selMid){
      delBtn.style.left=(selMid.x-11)+'px';
      delBtn.style.top=(selMid.y-11)+'px';
      delBtn.style.display='block';
    }else{
      delBtn.style.display='none';
    }
  }
}
function selectConn(i){
  if(selectedConn===i){ selectedConn=-1; }
  else { selectedConn=i; }
  drawConnections();
  if(selectedConn>=0 && connections[selectedConn]){
    let c=connections[selectedConn];
    updateStatus('已选中连线：'+c.from+' → '+c.to+'（点 × 或按 Delete 删除）');
  }else{
    updateStatus('就绪。拖拽硬件方块和算子来配置。');
  }
}
function _portVisible(el){
  let n=el;
  while(n&&n!==document.body){
    if(n.style&&n.style.display==='none')return false;
    n=n.parentElement;
  }
  return true;
}
// 判定一条数据流线是否"数据存储设备 == 算子运行设备"（共处同设备 → 无需搬运 → 隐藏）。
// 输入线：{hw}_r → {op}_in{i}；输出/中间值线：{op}_out{j}/{op}_mid → {hw}_w。
function _isCoLocated(c){
  // 输入线：来源是某设备的读端口 → 目标算子输入端口
  let mIn=/^(.+)_in(\d+)$/.exec(c.to||'');
  if(mIn){
    let opId=mIn[1];
    let hwM=/^(.+)_r$/.exec(c.from||'');
    if(hwM && blocks[opId] && blocks[opId].parentHW===hwM[1]) return true;
    return false;
  }
  // 输出/中间值线：目标端口是某设备的写端口，来源是某算子的 out/mid 端口
  let hwW=/^(.+)_w$/.exec(c.to||'');
  if(hwW){
    let hwId=hwW[1];
    let mOut=/^(.+)_out(\d+)$/.exec(c.from||'');
    let mMid=/^(.+)_mid$/.exec(c.from||'');
    let opId=(mOut&&mOut[1])||(mMid&&mMid[1]);
    if(opId && blocks[opId] && blocks[opId].parentHW===hwId) return true;
  }
  return false;
}
// Redraw on scroll
document.getElementById('canvas-wrap').addEventListener('scroll',drawConnections);

// 选中连线后：Delete/Backspace 删除，Esc 取消选中
document.addEventListener('keydown',e=>{
  if(selectedConn<0)return;
  if(e.key==='Delete'||e.key==='Backspace'){
    e.preventDefault();
    delConn(selectedConn);
  }else if(e.key==='Escape'){
    selectedConn=-1; drawConnections();
  }
});
// 点击画布空白处（非连线、非方块端口）取消选中
document.getElementById('canvas-wrap').addEventListener('click',e=>{
  if(selectedConn<0)return;
  let t=e.target;
  if(t && t.tagName && t.tagName.toLowerCase()==='path')return;   // 点击的是连线本身
  if(t && t.closest && t.closest('.block'))return;               // 点击的是方块/端口
  selectedConn=-1; drawConnections();
});

// ─── 画布缩放（v3.1）───
function setCanvasScale(s){
  s=Math.min(3.0, Math.max(0.2, s));
  canvasScale=s;
  let c=document.getElementById('canvas');
  c.style.transform='scale('+s+')';
  c.style.transformOrigin='0 0';
  drawConnections();
  let z=document.getElementById('zoom-label');
  if(z) z.textContent=Math.round(s*100)+'%';
  updateStatus('缩放：'+Math.round(s*100)+'%');
}
function zoomCanvas(delta){ setCanvasScale(canvasScale*(delta>0?1.15:1/1.15)); }
function resetCanvasZoom(){ setCanvasScale(1); }
function fitCanvas(){
  let wrap=document.getElementById('canvas-wrap');
  let vw=wrap.clientWidth-30, vh=wrap.clientHeight-30;
  let iw=parseInt(document.getElementById('canvas').style.width)||4000;
  let ih=parseInt(document.getElementById('canvas').style.height)||3000;
  setCanvasScale(Math.min(vw/iw, vh/ih, 1));
}
document.getElementById('canvas-wrap').addEventListener('wheel',e=>{
  if(!e.ctrlKey)return;
  e.preventDefault();
  zoomCanvas(e.deltaY<0?1:-1);
},{passive:false});

// ─── 链路带宽表已迁入"自定义硬件"弹窗（链路系统）───
// 旧的"互连"工具条与手动 addLink 已移除；updateLinkSelects 保留为兼容空函数。
function updateLinkSelects(){/* no-op：互连工具条已移除，链路带宽改由自定义硬件弹窗填写 */}
// ─── 完整序列化画布状态（硬件/算子/连线/映射）——供校验与运行 ───
function serializeState(){
  let hardware=[], operators=[], compute_map={}, run_map={}, mappedCount=0;
  Object.entries(blocks).forEach(([id,b])=>{
    if(b.type==='operator'){
      let k=b.data||{};
      operators.push({
        id:id, kernelId:b.kernelId||id, op_type:k.op_type||'',
        precision:k.precision, inputs:b.inputs||[], outputs:b.outputs||[],
        intermediates:b.intermediates||[],
        memory_bytes_range:k.memory_bytes_range||null,
        is_kv_dependent:!!k.is_kv_dependent,
        attributes:k.attributes||{},
        data_precision:k.data_precision||null,
        execution_precision:k.execution_precision||null
      });
      if(b.parentHW){
        compute_map[b.kernelId||id]=b.parentHW;
        let hw=blocks[b.parentHW];
        if(hw){ run_map[b.kernelId||id]=hw.backId||b.parentHW; mappedCount++; }
      } else if(b.recommendedDevice){
        // 未拖拽映射的算子，用推荐设备作为默认参考（用户可拖拽覆盖）
        compute_map[b.kernelId||id]=b.recommendedDevice;
        run_map[b.kernelId||id]=b.recommendedBackId||b.recommendedDevice;
      }
    } else {
      let p=b.params||{};
      hardware.push({id:id,type:b.type,backId:b.backId,precision:b.precision,
        compute:p.compute, mem:p.mem, rBW:p.rBW, wBW:p.wBW,
        linkType:b.linkType||b.type, links:b.links||{}});
    }
  });
  let conns=connections.map(c=>({from:c.from,to:c.to,label:c.label,lat:c.lat,isLink:!!c.isLink}));
  return {hardware:hardware,operators:operators,connections:conns,
          compute_map:compute_map,run_map:run_map,mappedCount:mappedCount,
          weight_blocks:collectWeightBlocks(),
          workload:getWorkloadParams(),
          link_table:linkBw};
}
function renderValidation(vr){
  let issues=vr.issues||[]; let errs=issues.filter(i=>i.level==='error');
  let warns=issues.filter(i=>i.level==='warning');
  let color=vr.valid?'var(--green)':'var(--red)';
  showResult('配置校验', '<h3 style="color:'+color+'">'+(vr.valid?'✔ 配置有效，可以运行':'✘ 配置无效，无法运行')+'</h3>'+
    '<div style="font-size:11px;color:var(--text2);margin-bottom:8px">'+errs.length+' 个错误 · '+warns.length+' 个警告</div>'+
    (issues.length?issues.map(i=>
      '<div style="padding:3px 0;font-size:11px;'+(i.level==='error'?'color:var(--red)':'color:var(--orange)')+'">'+
      '<b>['+i.code+']</b> '+i.message+'</div>').join(''):
      '<div style="color:var(--green);font-size:12px;padding:6px 0">所有约束检查通过。</div>'));
}
/* 显示结果弹窗（含标题 + 正文，正文写入 ro-body） */
function showResult(title, html){
  let overlay=document.getElementById('result-overlay');
  document.querySelector('#result-overlay .ro-title').textContent=title||'LLM-PIMSim';
  document.getElementById('ro-body').innerHTML=html||'';
  overlay.classList.add('show');
}
function closeResult(){
  document.getElementById('result-overlay').classList.remove('show');
}
/* 结果弹窗可拖动（拖头部） */
(function makeResultDraggable(){
  let head=document.getElementById('ro-head');
  if(!head) return;
  head.addEventListener('mousedown',function(e){
    let overlay=document.getElementById('result-overlay');
    if(e.target.classList && e.target.classList.contains('ro-close')) return;
    e.preventDefault();
    let sx=e.clientX, sy=e.clientY;
    let oleft=overlay.offsetLeft, otop=overlay.offsetTop;
    // 去掉居中的 transform，转成显式 left/top 以便拖动
    overlay.style.left=oleft+'px'; overlay.style.top=otop+'px';
    overlay.style.transform='none';
    function move(ev){
      overlay.style.left=(oleft+ev.clientX-sx)+'px';
      overlay.style.top=(otop+ev.clientY-sy)+'px';
    }
    function up(){
      document.removeEventListener('mousemove',move);
      document.removeEventListener('mouseup',up);
    }
    document.addEventListener('mousemove',move);
    document.addEventListener('mouseup',up);
  });
})();
/* 通用弹层可拖动：.modal-head[data-drag=弹层id] 拖头部移动整个 .drag-modal */
(function makeModalsDraggable(){
  document.querySelectorAll('.modal-head[data-drag]').forEach(function(head){
    head.addEventListener('mousedown',function(e){
      if(e.target.classList && e.target.classList.contains('modal-x')) return;
      e.preventDefault();
      let modal=document.getElementById(head.getAttribute('data-drag'));
      if(!modal) return;
      let sx=e.clientX, sy=e.clientY;
      let otx=0, oty=0;
      if(modal.style.transform){
        const m=/translate\((-?[\d.]+)px,\s*(-?[\d.]+)px\)/.exec(modal.style.transform);
        if(m){ otx=parseFloat(m[1]); oty=parseFloat(m[2]); }
      }
      function move(ev){
        modal.style.transform='translate('+(otx+ev.clientX-sx)+'px,'+(oty+ev.clientY-sy)+'px)';
      }
      function up(){
        document.removeEventListener('mousemove',move);
        document.removeEventListener('mouseup',up);
      }
      document.addEventListener('mousemove',move);
      document.addEventListener('mouseup',up);
    });
  });
})();
function validateSim(){
  let st=serializeState();
  showResult('配置校验', '<div style="color:var(--text2)">正在校验配置...</div>');
  let exp=document.getElementById('sel-experiment')?document.getElementById('sel-experiment').value:'';
  fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({experiment:exp||'experiments/04_ic_reference.yaml',
      compute_map:st.run_map, state:st, splits:pendingSplits, workload:st.workload})})
  .then(r=>r.json()).then(d=>renderValidation(d))
  .catch(e=>showResult('配置校验', '<h3 style="color:var(--red)">校验失败</h3><div>'+e.message+'</div>'));
}
function runSim(){
  let exp=document.getElementById('sel-experiment').value;
  let runValidation=document.getElementById('check-validate').checked;
  if(!exp){ updateStatus('请先选择一个实验。'); return; }
  showResult('运行仿真', '<div style="color:var(--text2)">正在运行仿真... 请稍候...</div>');

  // 完整序列化画布状态
  let st=serializeState();
  let mapNote = st.mappedCount>0 ? ('已映射'+st.mappedCount+'个算子') : '（当前未映射任何算子，使用配置默认）';
  updateStatus('正在运行 '+exp+(runValidation?'':'（已跳过校验）')+' ... '+mapNote);

  fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({experiment:exp, compute_map:st.run_map, state:st,
      run_validation:runValidation, splits: pendingSplits, workload: st.workload})})
  .then(r=>r.json()).then(d=>{
    if(d.blocked){ renderValidation(d.validation); updateStatus('配置不合法，已阻止运行'); return; }
    if(d.error){showResult('错误', '<h3 style="color:var(--red)">错误</h3><pre style="font-size:10px;color:var(--text2)">'+d.error+'\\n'+((d.trace||'').slice(0,500))+'</pre>');return}
    let bd=d.breakdown||{};
    // ── v3 搬运量明细 ──
    let mv=d.movement_total_bytes||0;
    let links=(d.movement_per_link||[]).slice(0,8);
    let mvHtml=`<div class="metric"><span>📦 搬运字节</span><span class="v">${(mv/1e6).toFixed(1)} MB</span></div>`;
    if(links.length){
      mvHtml+=`<div style="margin-top:4px;font-size:10px">`+links.map(l=>
        `<div style="display:flex;justify-content:space-between;padding:1px 0;color:var(--text2)">
          <span>${l.src} → ${l.dst}</span><span style="color:var(--text)">${(l.bytes/1e6).toFixed(1)} MB</span></div>`
      ).join('')+`</div>`;
    }
    // 注：不再显示"完成度/完整性校验"（校验充分：能运行即完整；未过校验在运行前已拦截）
    showResult('仿真结果', `
      <div class="metric"><span>⏱ 总延迟</span><span class="v">${(d.total_latency_ms||0).toFixed(2)} ms</span></div>
      <div class="metric"><span>⚙ 计算</span><span class="v">${((bd.compute_ns||0)/1e6).toFixed(2)} ms</span></div>
      <div class="metric"><span>🔗 搬运</span><span class="v">${((bd.transfer_ns||0)/1e6).toFixed(2)} ms</span></div>
      <div class="metric"><span>🔁 同步等待</span><span class="v">${((bd.sync_ns||0)/1e6).toFixed(2)} ms</span></div>
      <div class="metric"><span>💾 本地读写</span><span class="v">${(((bd.local_read_ns||0)+(bd.local_write_ns||0))/1e6).toFixed(2)} ms</span></div>
      <div class="metric"><span>🔎 瓶颈</span><span class="v" style="color:${d.bottleneck==='COMPUTE'?'#61afef':'#e06c75'}">${d.bottleneck||'?'}</span></div>
      <div class="metric"><span>✂ 权重切片数</span><span class="v">${d.weight_shard_count||0}</span></div>
      ${mvHtml}
      ${_criticalPathHtml(d.critical_path)}
      <div style="color:var(--text2);font-size:10px;margin-top:8px;border-top:1px solid var(--border);padding-top:6px">${d.rationale||''}</div>
      ${_friendlyTip(d)}
      <div style="color:var(--green);font-size:10px;margin-top:6px">${mapNote}${d.override_applied>=0?' · 后端覆盖'+d.override_applied+'个算子':''}${pendingSplits.length?' · 张量并行切片'+pendingSplits.length+'条':''}</div>
    `);
    updateStatus('完成: '+d.total_latency_ms.toFixed(2)+'ms, 瓶颈='+d.bottleneck+' ('+mapNote+')');
  }).catch(e=>{
    showResult('错误', '<h3 style="color:var(--red)">错误</h3><div>'+e.message+'</div>');
  });
}

// 需求6 初学者友好：一句话人话提示
function _friendlyTip(d){
  let b=d.bottleneck||'';
  let tip=b==='COMPUTE'?'💡 瓶颈在计算：可换更高算力设备 / 提高算子效率 / 增大张量并行切片。'
    : (b==='COMMUNICATION'?'💡 瓶颈在搬运：可把数据/权重就近算子放置，或压缩跨设备搬运。'
      : (b==='SYNCHRONIZATION'?'💡 瓶颈在同步等待：检查依赖链，减少不必要等待。'
        : '💡 本地读写占主导：可优化激活/中间值的驻留。'));
  return `<div style="font-size:11px;color:var(--green);margin-top:6px">${tip}</div>`;
}

// 需求5 关键路径视图：展示后端算出的 critical_path（决定总延时的主链）
function _criticalPathHtml(cp){
  if(!cp || !cp.ops || !cp.ops.length){
    return '<div class="diag" style="margin-top:10px;margin-bottom:4px"><b style="color:var(--accent)">📊 关键路径</b><div style="font-size:10px;color:var(--text2);margin-top:3px">暂无关键路径数据。</div></div>';
  }
  let top = cp.ops.slice(-10).reverse();   // 尾部是最晚结束的关键算子，取末尾 10 个倒序
  let chips = top.map(o=>
    `<span title="${o.op_id} · 计算${o.compute_ms}ms · 搬${o.transfer_ms}ms · 本地${o.local_rw_ms}ms" style="display:inline-block;background:#233346;color:#9cc8f2;border:1px solid #2f4a63;border-radius:4px;padding:2px 6px;margin:2px;font-size:10px">${o.op_id}</span>`).join('');
  return `<div class="diag" style="margin-top:10px">
    <b style="color:var(--accent)">📊 关键路径（${cp.ops.length} 个节点，tail）</b>
    <div style="font-size:10px;color:var(--text2);margin-top:3px">${cp.explanation||''}</div>
    <div style="margin-top:5px">${chips}</div>
  </div>`;
}

// ─── 结果对比 ───
let cmpSel = [];
function openCompare(){
  cmpSel = [];
  fetch('/api/files').then(r=>r.json()).then(d=>{
    const list = d.results || [];
    let box = document.getElementById('cmp-list');
    box.innerHTML = list.length
      ? list.map(n=>`<label style="display:flex;align-items:center;gap:6px;padding:3px 4px;font-size:12px;cursor:pointer">
          <input type="checkbox" value="${n}" onchange="cmpToggle(this)"> <span>${n}</span></label>`).join('')
      : '<div style="color:var(--text2);font-size:11px">暂无已保存的结果。请先运行一个实验。</div>';
    document.getElementById('cmp-msg').textContent = '';
  });
  document.getElementById('cmp-overlay').classList.add('show');
}
function cmpToggle(cb){
  if(cb.checked) cmpSel.push(cb.value);
  else cmpSel = cmpSel.filter(n=>n!==cb.value);
}
function closeCompare(){ document.getElementById('cmp-overlay').classList.remove('show'); }
function doCompare(){
  if(cmpSel.length < 2){ document.getElementById('cmp-msg').textContent='请至少勾选两个结果。'; return; }
  document.getElementById('cmp-msg').textContent='对比中...';
  fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({names:cmpSel})})
  .then(r=>r.json()).then(d=>{
    if(d.error){ document.getElementById('cmp-msg').textContent=d.error; return; }
    const rows = d.rows||[];
    let html=`<table style="width:100%;border-collapse:collapse;font-size:11px;margin-top:10px">
      <tr style="color:var(--text2);text-align:left">
        <th>结果</th><th>总延迟ms</th><th>计算</th><th>搬运</th><th>本地读写</th><th>瓶颈</th><th>搬运MB</th><th>完成</th></tr>`;
    rows.forEach(r=>{
      html+=`<tr style="border-top:1px solid var(--border)">
        <td>${r.name}</td><td><b style="color:var(--accent)">${r.latency_ms}</b></td>
        <td>${r.compute_ms}</td><td>${r.transfer_ms}</td><td>${r.local_rw_ms}</td>
        <td>${r.bottleneck}</td><td>${r.movement_mb}</td>
        <td style="color:${r.complete?'var(--green)':'var(--red)'}">${r.complete?'✔':'✘'}</td></tr>`;
    });
    html+=`</table>`;
    showResult('结果对比', html);
    closeCompare();
  }).catch(e=>{ document.getElementById('cmp-msg').textContent='对比失败: '+e.message; });
}

// ─── 实验清单 & 新建实验 ───
function loadExperiments(){
  fetch('/api/experiments').then(r=>r.json()).then(d=>{
    let sel=document.getElementById('sel-experiment');
    let list=d.experiments||[];
    sel.innerHTML='';
    if(!list.length){ sel.innerHTML='<option value="">无实验</option>'; return; }
    list.forEach(e=>{
      let o=document.createElement('option');
      o.value=e.path; o.textContent=e.name; sel.appendChild(o);
    });
  }).catch(()=>{
    let sel=document.getElementById('sel-experiment');
    sel.innerHTML='<option value="">加载失败</option>';
  });
}
function openNewExperiment(){
  // 填充模型下拉
  let msel=document.getElementById('new-exp-model');
  msel.innerHTML='';
  ['llama_gb'].forEach(id=>{
    let o=document.createElement('option'); o.value=id; o.textContent=(id==='llama_gb')?'llama（GB）':id; msel.appendChild(o);
  });
  // 填充"参考实验"下拉（默认选中 04_ic_reference，或用当前实验作为兜底）
  let csel=document.getElementById('new-exp-clone');
  let curVal=document.getElementById('sel-experiment').value;
  fetch('/api/experiments').then(r=>r.json()).then(d=>{
    let list=d.experiments||[];
    csel.innerHTML='';
    list.forEach(e=>{
      let o=document.createElement('option'); o.value=e.path; o.textContent=e.name; csel.appendChild(o);
    });
    // 优先默认选参考样例 04_ic_reference，其次当前实验
    if(list.some(e=>e.name==='04_ic_reference')) csel.value='experiments/04_ic_reference.yaml';
    else if(curVal && list.some(e=>e.path===curVal)) csel.value=curVal;
  });
  // 默认"从头开始"
  document.querySelector('input[name="new-start"][value="blank"]').checked=true;
  onStartMode();
  document.getElementById('new-exp-name').value='';
  document.getElementById('new-exp-msg').textContent='';
  document.getElementById('new-exp-overlay').classList.add('show');
}
function onStartMode(){
  let v=(document.querySelector('input[name="new-start"]:checked')||{}).value||'blank';
  document.getElementById('new-ref-box').style.display = (v==='ref')?'block':'none';
}
function closeNewExperiment(){
  document.getElementById('new-exp-overlay').classList.remove('show');
}
// 重置为空白实验：显示所有算子与权重（未放置），无硬件、无连线（供"从头开始"使用）
function resetToBlank(){
  // 1) 重置所有算子：解除映射、恢复浮动、干净网格布局
  let ops=Object.keys(blocks).filter(id=>blocks[id].type==='operator');
  let x0=600, y0=40;
  ops.forEach((id,i)=>{
    let b=blocks[id];
    let x=x0+Math.floor(i/4)*400, y=y0+(i%4)*330;
    b.el.style.left=x+'px'; b.el.style.top=y+'px';
    b.el.style.display='block';
    b.x=x; b.y=y;
    b.parentHW=null;
    b.recommendedDevice=null; b.recommendedBackId=null;
    if(b.recEl) b.recEl.textContent='—';
    _updateOpTag(id);
  });
  // 2) 移除所有硬件（保留权重块，随后由 loadWeights 重新加载为"未放置"）
  Object.keys(blocks).filter(id=>blocks[id].type!=='operator' && blocks[id].type!=='weight').forEach(id=>{
    blocks[id].el.remove(); delete blocks[id];
  });
  // 3) 清空连线与切割规则
  connections=[]; pendingSplits=[];
  drawConnections(); refreshConnList(); updateLinkSelects();
  // 4) 重新加载权重块（画布无硬件 → 权重以"未放置"浮动显示），并排到算子下方避免重叠
  loadWeights().then(()=>{
    let wids=Object.keys(blocks).filter(id=>blocks[id].type==='weight');
    let wy=1400;
    wids.forEach((id,i)=>{
      let b=blocks[id];
      let x=600+(i%6)*300, y=wy+Math.floor(i/6)*150;
      b.el.style.left=x+'px'; b.el.style.top=y+'px';
      b.x=x; b.y=y;
    });
    updateStatus('空白实验：已显示所有算子与权重（未放置），无硬件、无连线');
  });
}
function createExperiment(){
  let name=document.getElementById('new-exp-name').value.trim();
  let model=document.getElementById('new-exp-model').value;
  let mode=(document.querySelector('input[name="new-start"]:checked')||{}).value||'blank';
  let clone = mode==='ref'
    ? (document.getElementById('new-exp-clone').value||'experiments/04_ic_reference.yaml')
    : '';
  let msg=document.getElementById('new-exp-msg');
  msg.style.color='var(--orange)';
  msg.textContent='创建中...';
  fetch('/api/experiment/create',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:name, model:model, mode:mode, clone_from:clone})})
  .then(r=>r.json()).then(d=>{
    if(!d.ok){ msg.style.color='var(--red)'; msg.textContent=d.error||'创建失败。'; return; }
    msg.style.color='var(--green)';
    msg.textContent='已创建 '+d.name;
    // 刷新实验下拉并选中新实验
    loadExperiments();
    setTimeout(()=>{
      let sel2=document.getElementById('sel-experiment');
      sel2.value=d.experiment;
      closeNewExperiment();
      // 从头开始 → 空白画布（只显示算子）；从参考开始 → 部署 IC 参考映射
      if(mode==='blank'){ resetToBlank(); }
      else { applyRecommendation(); }
    },300);
  }).catch(e=>{ msg.style.color='var(--red)'; msg.textContent='请求失败: '+e.message; });
}

// ─── Save ───
function saveConfig(){
  let hwYaml='devices:\n';
  Object.entries(blocks).forEach(([id,b])=>{
    if(b.type==='operator')return;
    let lt=String(b.linkType||b.type).toUpperCase();
    let linkLines='';
    let links=b.links||{};
    if(lt!==String(b.type).toUpperCase() || Object.keys(links).length){
      linkLines+='    links:\n';
      Object.entries(links).forEach(([k,v])=>{ linkLines+=`      ${k}: ${v}\n`; });
    }
    hwYaml+=`  - id: ${id}\n    type: ${b.type}\n` +
      (lt!==String(b.type).toUpperCase() ? `    link_type: ${lt}\n` : '') +
      `    compute:\n      peak_tflops: 300\n    memory:\n      capacity_gb: 80\n` + linkLines;
  });
  // 链路系统：N×N 对称带宽表（link_bw_gbs 格式，GB/s）
  let icYaml='# 链路系统：N×N 对称链路带宽表（GB/s）\nlink_bw_gbs:\n';
  Object.keys(linkBw).sort().forEach(a=>{
    let row=linkBw[a]||{};
    let cols=Object.keys(row).sort().map(b=>`      ${b}: ${row[b]}`).join('\n');
    icYaml+=`  ${a}:\n${cols}\n`;
  });
  fetch('/api/write',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:'hardware_gui.yaml',content:hwYaml})});
  fetch('/api/write',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:'interconnect_gui.yaml',content:icYaml})});
  updateStatus('已保存: hardware_gui.yaml, interconnect_gui.yaml（含链路带宽表）');
}

function updateStatus(msg){document.getElementById('status').textContent=msg||'就绪。'}

// ─── 依赖关系图抽屉 ───
function toggleDepDrawer(){
  let d=document.getElementById('dep-drawer');
  d.classList.toggle('show');
  if(d.classList.contains('show')) renderDepDrawer();
}
function renderDepDrawer(){
  let body=document.getElementById('dep-body');
  if(!currentWorkload||!currentWorkload.layers||!currentWorkload.layers.length){
    body.innerHTML='<div style="color:var(--text2);font-size:11px">请先加载模型以查看依赖关系。</div>';
    return;
  }
  let kernels=currentWorkload.layers[0]||[];
  if(!kernels.length){body.innerHTML='<div style="color:var(--text2)">无算子</div>';return}
  let globals=currentWorkload.global_kernels||[];
  let emb=globals.find(k=>k.id==='embedding');
  let lm=globals.find(k=>k.id==='lm_head');

  // 链条：Embedding → 16 个 L0 算子 → LMHead（全局算子补齐首尾）
  let chain=[];
  if(emb) chain.push(emb);
  kernels.forEach(k=>chain.push(k));
  if(lm) chain.push(lm);
  let wList=(currentWeights||[]).slice();
  let wMap={}; wList.forEach(w=>{wMap[w.weight_id]=w;});

  function _nd(d){return String(d).replace(/^L\d+_/,'L0_');}   // L31_layer_output → L0_layer_output
  function _esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
  function _trunc(s,n){s=String(s||'');return s.length>n?s.slice(0,n-1)+'…':s;}

  // 端口配色：同一数据名（端口）的所有线同色；按首次出现顺序循环调色板，保证相邻端口不同色
  const PALETTE=['#61afef','#d19a66','#98c379','#c678dd','#e06c75','#56b6c2','#e5c07b','#f08d49'];
  let dataColor={}, colorIdx=0;
  function colorOf(d){let k=_nd(d);if(!(k in dataColor)){let i=colorIdx%PALETTE.length;dataColor[k]={color:PALETTE[i],marker:'arr'+i};colorIdx++;}return dataColor[k];}

  // 数据名 → 生产算子
  let producer={};
  chain.forEach(k=>{(k.outputs||[]).forEach(d=>producer[_nd(d)]=k.id);(k.intermediates||[]).forEach(d=>producer[_nd(d)]=k.id);});

  // 上游依赖 + 最长路径分层（列号）
  let upstream={};
  chain.forEach(k=>{let ups=[];(k.inputs||[]).forEach(d=>{let nd=_nd(d);if(producer[nd]&&producer[nd]!==k.id&&!ups.includes(producer[nd]))ups.push(producer[nd]);});upstream[k.id]=ups;});
  let level={};
  function calcLevel(id){if(level[id]!==undefined)return level[id];let m=0;(upstream[id]||[]).forEach(u=>m=Math.max(m,calcLevel(u)+1));level[id]=m;return m;}
  chain.forEach(k=>calcLevel(k.id));

  // ── 布局（算子画大点）──
  const NODE_W=184, COL_GAP=64, ROW_GAP=22, WX=8, WW=152, TOP=16;
  function nodeH(k){let n=Math.max((k.inputs||[]).length,(k.outputs||[]).length+(k.intermediates||[]).length);return Math.max(66,n*18+46);}
  let byLevel={}; chain.forEach(k=>{(byLevel[level[k.id]]=byLevel[level[k.id]]||[]).push(k);});
  let levels=Object.keys(byLevel).map(Number).sort((a,b)=>a-b);
  let maxLevel=levels[levels.length-1];
  let colX={}; let firstX=WX+WW+120;   // 权重列与首算子列之间留宽，用于竖直走线通道
  levels.forEach((lv,i)=>{colX[lv]=firstX+i*(NODE_W+COL_GAP);});

  let pos={}, hh={};
  levels.forEach(lv=>{let x=colX[lv],y=TOP;byLevel[lv].forEach(k=>{hh[k.id]=nodeH(k);pos[k.id]={x,y};y+=hh[k.id]+ROW_GAP;});});

  // 外部输入/输出终端（保证每个端口都有连线）
  let inTerm={x:WX,y:TOP,w:WW,h:54,label:'输入',data:'input_ids'};
  let outTermX=colX[levels[levels.length-1]]+NODE_W+COL_GAP;
  let outTerm={x:outTermX,y:TOP,w:WW,h:54,label:'输出',data:'logits'};

  let wPos={}, wH={}; let wy=inTerm.y+inTerm.h+ROW_GAP+4;
  wList.forEach(w=>{let h=Math.max(54,(w.consumers||[]).length*14+24);wH[w.weight_id]=h;wPos[w.weight_id]={x:WX,y:wy};wy+=h+ROW_GAP;});

  let maxY=TOP;
  chain.forEach(k=>maxY=Math.max(maxY,pos[k.id].y+hh[k.id]));
  wList.forEach(w=>maxY=Math.max(maxY,wPos[w.weight_id].y+wH[w.weight_id]));
  maxY=Math.max(maxY, inTerm.y+inTerm.h, outTerm.y+outTerm.h);

  function inY(k,i){let n=(k.inputs||[]).length,top=48,usable=hh[k.id]-top-10;return pos[k.id].y+top+(n>1?i*usable/(n-1):usable/2);}
  function outY(k,j){let outs=(k.outputs||[]).concat(k.intermediates||[]);let n=outs.length,top=48,usable=hh[k.id]-top-10;return pos[k.id].y+top+(n>1?j*usable/(n-1):usable/2);}

  // ── 收集边（端口级，颜色=数据名；srcL/tgtL 记录所在层，供走线通道分配）──
  let edges=[];
  chain.forEach(k=>{(k.inputs||[]).forEach((d,di)=>{
    if(d==='input_ids'){edges.push({x1:inTerm.x+WW,y1:inTerm.y+inTerm.h/2,x2:pos[k.id].x,y2:inY(k,di),c:colorOf(d),span:1,srcL:-1,tgtL:level[k.id]});}
    else if(wMap[d]){let wp=wPos[d],p=pos[k.id];edges.push({x1:wp.x+WW,y1:wp.y+wH[d]/2,x2:p.x,y2:inY(k,di),c:colorOf(d),span:level[k.id]+1,srcL:-1,tgtL:level[k.id]});}
  });});
  chain.forEach(kA=>{let outs=(kA.outputs||[]).concat(kA.intermediates||[]);outs.forEach((d,oj)=>{let nd=_nd(d);chain.forEach(kB=>{if(kB.id===kA.id)return;let di=(kB.inputs||[]).findIndex(dd=>_nd(dd)===nd);if(di<0)return;let pa=pos[kA.id],pb=pos[kB.id];let span=level[kB.id]-level[kA.id];edges.push({x1:pa.x+NODE_W,y1:outY(kA,oj),x2:pb.x,y2:inY(kB,di),c:colorOf(nd),span,srcL:level[kA.id],tgtL:level[kB.id]});});});});
  chain.forEach(k=>{(k.outputs||[]).forEach((d,oj)=>{if(d==='logits'){edges.push({x1:pos[k.id].x+NODE_W,y1:outY(k,oj),x2:outTerm.x,y2:outTerm.y+outTerm.h/2,c:colorOf(d),span:1,srcL:level[k.id],tgtL:level[k.id]+1});}});});

  // ── 正交走线通道分配：每个竖直段分配唯一 x，保证线不重叠、间距≥两条线宽 ──
  // 通道区间：-1=权重列与首算子列之间；L=第 L 列与第 L+1 列之间；maxLevel=末列与输出终端之间
  function gapRange(L){
    if(L<0) return {start:WX+WW, end:firstX};
    let s=colX[L]+NODE_W;
    let e=(L>=maxLevel)?outTerm.x:colX[L+1];
    return {start:s, end:e};
  }
  let gapCount={}, gapIdx={};
  edges.forEach(e=>{
    if(e.span<=1){ gapCount[e.srcL]=(gapCount[e.srcL]||0)+1; }
    else { gapCount[e.srcL]=(gapCount[e.srcL]||0)+1; gapCount[e.tgtL-1]=(gapCount[e.tgtL-1]||0)+1; }
  });
  function channelX(L){
    let r=gapRange(L), n=gapCount[L]||1, idx=gapIdx[L]||0;
    gapIdx[L]=idx+1;
    return r.start + (idx+1)*(r.end-r.start)/(n+1);
  }
  edges.forEach(e=>{
    if(e.span<=1){ e.vx=channelX(e.srcL); }
    else { e.dropX=channelX(e.srcL); e.entryX=channelX(e.tgtL-1); }
  });

  // 跨列长边 → 底部横向车道（每条唯一 y，绝不穿过算子卡片）
  let longEdges=edges.filter(e=>e.span>1);
  let laneStart=maxY+26;
  longEdges.forEach((e,i)=>{e.laneY=laneStart+i*17;});
  let ch=(longEdges.length?laneStart+longEdges.length*17+18:maxY+20);
  let cw=outTerm.x+outTerm.w+16;

  // 全部折线：短边 Z 形（横-竖-横），长边（横-竖-横-竖-横），只含水平/竖直段
  function edgePath(e){
    if(e.span<=1){return `M ${e.x1},${e.y1} L ${e.vx},${e.y1} L ${e.vx},${e.y2} L ${e.x2},${e.y2}`;}
    return `M ${e.x1},${e.y1} L ${e.dropX},${e.y1} L ${e.dropX},${e.laneY} L ${e.entryX},${e.laneY} L ${e.entryX},${e.y2} L ${e.x2},${e.y2}`;
  }

  // ── 渲染：先画线（直线、加粗、无名字），再画节点（节点盖住线端点，线只走空隙）──
  let svg='';
  svg+=`<defs>${PALETTE.map((c,i)=>`<marker id="arr${i}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="${c}"/></marker>`).join('')}</defs>`;
  edges.forEach(e=>{svg+=`<path d="${edgePath(e)}" fill="none" stroke="${e.c.color}" stroke-width="2.6" stroke-linecap="round" marker-end="url(#${e.c.marker})" opacity="0.92"/>`;});

  // 输入/输出终端
  svg+=`<rect x="${inTerm.x}" y="${inTerm.y}" width="${inTerm.w}" height="${inTerm.h}" rx="10" fill="#1d2a1f" stroke="#98c379" stroke-width="1.7"/>
    <text x="${inTerm.x+14}" y="${inTerm.y+32}" font-size="14" font-weight="700" fill="#98c379">${_esc(inTerm.label)}</text>
    <circle cx="${inTerm.x+inTerm.w}" cy="${inTerm.y+inTerm.h/2}" r="5" fill="#98c379" stroke="#0f1115" stroke-width="1.6"/>`;
  svg+=`<rect x="${outTerm.x}" y="${outTerm.y}" width="${outTerm.w}" height="${outTerm.h}" rx="10" fill="#2a1c1e" stroke="#e06c75" stroke-width="1.7"/>
    <text x="${outTerm.x+14}" y="${outTerm.y+32}" font-size="14" font-weight="700" fill="#e06c75">${_esc(outTerm.label)}</text>
    <circle cx="${outTerm.x}" cy="${outTerm.y+outTerm.h/2}" r="5" fill="#e06c75" stroke="#0f1115" stroke-width="1.6"/>`;

  // 权重节点
  wList.forEach(w=>{
    let p=wPos[w.weight_id], h=wH[w.weight_id];
    svg+=`<rect x="${p.x}" y="${p.y}" width="${WW}" height="${h}" rx="10" fill="#241f33" stroke="#6a5acd" stroke-width="1.6"/>
      <text x="${p.x+12}" y="${p.y+20}" font-size="12" font-weight="700" fill="#c792ea">${_esc(_trunc(w.weight_id,17))}</text>
      <text x="${p.x+12}" y="${p.y+37}" font-size="9.5" fill="#b8a6e0">${_esc(w.weight_class||'')} · ${fmtBytes(w.bytes||0)}</text>
      <circle cx="${p.x+WW}" cy="${p.y+h/2}" r="5" fill="#c792ea" stroke="#0f1115" stroke-width="1.6"/>`;
  });

  // 算子节点
  chain.forEach(k=>{
    let p=pos[k.id], h=hh[k.id];
    let cat=_opCategory(k.op_type||'');
    let catColor=cat==='NONLINEAR'?'#d19a66':'#61afef';
    let isGlobal=(k.id==='embedding'||k.id==='lm_head');
    let stroke=isGlobal?'#e5c07b':catColor;
    let fill=isGlobal?'#282516':'#1c2129';
    let ports='';
    (k.inputs||[]).forEach((d,i)=>{ports+=`<circle cx="${p.x}" cy="${inY(k,i)}" r="5" fill="#61afef" stroke="#0f1115" stroke-width="1.6"/>`;});
    let outs=(k.outputs||[]).concat(k.intermediates||[]);
    outs.forEach((d,j)=>{ports+=`<circle cx="${p.x+NODE_W}" cy="${outY(k,j)}" r="5" fill="#d19a66" stroke="#0f1115" stroke-width="1.6"/>`;});
    svg+=`<rect x="${p.x}" y="${p.y}" width="${NODE_W}" height="${h}" rx="10" fill="${fill}" stroke="${stroke}" stroke-width="1.6"/>
      <rect x="${p.x}" y="${p.y}" width="${NODE_W}" height="5" rx="2.5" fill="${stroke}" opacity="0.5"/>
      <text x="${p.x+13}" y="${p.y+23}" font-size="13" font-weight="700" fill="#e8ecf2">${_esc(_trunc(k.name||k.id,15))}</text>
      <text x="${p.x+13}" y="${p.y+39}" font-size="9.5" fill="#8a93a5">${_esc((k.op_type||'')+(k.is_kv_dependent?' · KV':''))}</text>
      ${ports}`;
  });

  let totalLayers=currentWorkload.num_layers||32;
  let legend=`<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;font-size:10.5px;color:var(--text2);background:#14171d;border:1px solid var(--border);border-radius:6px;padding:9px 13px;margin-bottom:10px">
    <span><b style="color:#61afef">●</b> 输入端口（左）</span>
    <span><b style="color:#d19a66">●</b> 输出端口（右）</span>
    <span><b style="color:#e5c07b">■</b> 全局算子（Embedding/LMHead）</span>
    <span><b style="color:#98c379">■</b> 输入 / <b style="color:#e06c75">■</b> 输出</span>
    <span>同色线 = 同一数据端口</span>
    <span style="margin-left:auto;color:var(--text2)">📚 显示 L0 模板，其余 ${totalLayers-1} 层折叠</span>
  </div>
  <div style="font-size:10px;color:var(--text2);background:#14171d;border:1px solid var(--border);border-radius:6px;padding:7px 12px;margin-bottom:10px">
    💡 本图 ${chain.length} 个节点 = <b style="color:#e5c07b">Embedding</b> + <b>16 个 L0 算子</b> + <b style="color:#e5c07b">LMHead</b>；
    画布默认只显示 L0 的 16 个算子（Embedding/LMHead 为折叠的全局算子，运行时会展开到完整 ${totalLayers} 层）。
  </div>`;

  body.innerHTML=`${legend}<div style="overflow:auto;max-height:calc(100vh - 120px);background:radial-gradient(circle,#1d2129 1px,transparent 1px);background-size:20px 20px;border:1px solid var(--border);border-radius:8px;padding:12px">
    <svg width="${cw}" height="${ch}" viewBox="0 0 ${cw} ${ch}" style="display:block">${svg}</svg>
  </div>`;
  document.querySelector('.dep-drawer').style.width='960px';
  document.querySelector('.dep-body').style.overflow='hidden';
}

// ─── 算子切割（张量并行）───
let currentSplitBlock=null;
function _equalParts(total, n){
  let base=Math.floor(total/n), rem=total%n;
  let out=[]; for(let i=0;i<n;i++) out.push(base+(i<rem?1:0));
  return out;
}
function _wildcardOp(id){
  return String(id).replace(/^L\d+_/, 'L*_');   // L0_ffn_down → L*_ffn_down（让全部层生效）
}
function openSplitModal(opBlockId){
  let b=blocks[opBlockId]; if(!b)return;
  let k=b.data; if(!k)return;
  currentSplitBlock=opBlockId;
  document.getElementById('s-kernel').value=k.id||opBlockId;
  document.getElementById('s-parts').value='2';
  // 可切割维度：attributes 里的数值字段（M/K/N/seq 等，或 'kv_len' 动态维度）
  let dimSel=document.getElementById('s-dim');
  dimSel.innerHTML='';
  let attrs=k.attributes||{};
  Object.entries(attrs).forEach(([d,v])=>{
    if(typeof v==='number' || String(v).toLowerCase()==='kv_len'){
      let o=document.createElement('option');o.value=d;o.textContent=d+' = '+v;dimSel.appendChild(o);
    }
  });
  if(!dimSel.options.length){dimSel.innerHTML='<option value="">无可切割维度</option>'}
  document.getElementById('s-preview').textContent='切割后将生成多个算子图标，名称加 #1/#2…，计算量与存储量按片数等分，精度不变。';
  document.getElementById('split-modal').classList.add('show');
}
function closeSplitModal(){document.getElementById('split-modal').classList.remove('show');currentSplitBlock=null}
function doSplit(){
  if(!currentSplitBlock)return;
  let b=blocks[currentSplitBlock]; let k=b.data;
  let dim=document.getElementById('s-dim').value;
  let n=parseInt(document.getElementById('s-parts').value);
  if(!dim){updateStatus('切割失败：请选择切割维度');return}
  if(!n||n<2){updateStatus('切割失败：片数必须≥2');return}

  // 目标设备：用画布上所有硬件（后端张量并行需 ≥2 设备）。取硬件块的 backId。
  let hwDevs=[];
  Object.entries(blocks).forEach(([id,b2])=>{
    if(b2.type!=='operator' && b2.type!=='weight' && b2.backId) hwDevs.push(b2.backId);
  });
  hwDevs=[...new Set(hwDevs)];
  if(hwDevs.length<2){
    closeSplitModal();
    updateStatus('张量并行需至少 2 个硬件设备在画布上。请先添加硬件。');
    return;
  }
  n=Math.min(n, hwDevs.length);
  let devices=hwDevs.slice(0,n);

  // 等分该维度
  let v=k.attributes[dim];
  let parts=(typeof v==='number')?_equalParts(v, n):Array.from({length:n},()=>1);

  let model=document.getElementById('sel-model').value;
  fetch('/api/split',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model:model, kernel:k.id, dim:dim, parts:parts})})
  .then(r=>r.json()).then(d=>{
    if(d.error){updateStatus('切割失败: '+d.error);return}
    let oldBlockId=currentSplitBlock;   // 先保存：closeSplitModal 会把它置 null
    closeSplitModal();
    let slices=d.kernels||[];
    if(!slices.length){updateStatus('切割失败：后端未返回切片');return}
    // 记录张量并行切片规则（通配 L*_ 让全部层生效）→ 运行时后端真实执行
    pendingSplits = pendingSplits.filter(s=>!(s.op===k.id && s.dim===dim));
    pendingSplits.push({op: _wildcardOp(k.id), dim: dim, parts: slices.length, devices: devices});
    let ruleIdx = pendingSplits.length - 1;
    // 用切片替换原算子块：生成 N 个独立算子图标，每个切片自带标签、可独立放设备
    _replaceWithSlices(oldBlockId, slices, dim, devices, ruleIdx);
    updateStatus('算子 '+k.id+' 已沿 '+dim+' 切成 '+slices.length+' 片（生成 '+slices.length+' 个算子图标，计算/存储等分，全部层生效）');
  }).catch(e=>updateStatus('切割失败: '+e));
}
// 用后端返回的切片 kernel 列表，替换原算子块为 N 个"独立"切片算子图标（各自带标签、可独立放设备）
function _replaceWithSlices(oldId, slices, dim, devices, ruleIdx){
  let b=blocks[oldId]; if(!b)return;
  let oldX=b.x, oldY=b.y, logicalId=b.kernelId||((b.data&&b.data.id)||oldId);
  let recDev=b.recommendedDevice||b.parentHW||'';
  let recBack=b.recommendedBackId||(blocks[recDev]?blocks[recDev].backId:recDev);
  // backId -> 画布硬件块 id（用于切片初始/标签放置）
  let backToId={};
  Object.entries(blocks).forEach(([hid,b2])=>{
    if(b2.type!=='operator' && b2.type!=='weight' && b2.backId) backToId[b2.backId]=hid;
  });
  let newIds=[];
  // 移除旧块（含其在硬件内的紧凑条目与 opGroup 记录）
  if(b.parentHW && blocks[b.parentHW]){
    blocks[b.parentHW].opGroup=(blocks[b.parentHW].opGroup||[]).filter(x=>x!==oldId);
    let grp=document.getElementById(oldId+'_grp'); if(grp) grp.remove();
  }
  b.el.remove(); delete blocks[oldId];
  // 逐片创建新算子块
  slices.forEach((sk,i)=>{
    let id=oldId+'_s'+(i+1);
    newIds.push(id);
    let x=oldX+i*410, y=oldY;
    let cat=_opCategory(sk.op_type||'');
    let catLabel=cat==='NONLINEAR'?'非线性':'线性';
    let catColor=cat==='NONLINEAR'?'#d19a66':'#61afef';
    let dataPrec=sk.data_precision||sk.precision||'FP16';
    let execPrec=sk.execution_precision||'—(纯数据)';
    let inData=(sk.inputs||[]).join(', ')||'—';
    let outData=(sk.outputs||[]).join(', ')||'—';
    let midData=(sk.intermediates||[]).join(', ')||'';
    let attrs=Object.entries(sk.attributes||{}).map(([a,v2])=>a+'='+v2).join(', ')||'—';
    let nIn=(sk.inputs||[]).length, nOut=(sk.outputs||[]).length;
    let inTop=(nIn>1?(di)=>20+di*60/(nIn-1):()=>50);
    let outTop=(nOut>1?(do_)=>20+do_*60/(nOut-1):()=>50);
    let inPorts=(sk.inputs||[]).map((dd,di)=>`<div class="port input" data-port="${id}_in${di}" data-port-type="input" style="top:${inTop(di)}%" title="输入${di+1}: ${dd}"><span class="plabel">in${di+1}</span></div>`).join('');
    let outPorts=(sk.outputs||[]).map((dd,do_)=>`<div class="port output" data-port="${id}_out${do_}" data-port-type="output" style="top:${outTop(do_)}%" title="输出${do_+1}: ${dd}"><span class="plabel">out${do_+1}</span></div>`).join('');
    let midPort=(sk.intermediates||[]).length?`<div class="port mid" data-port="${id}_mid" data-port-type="mid" title="中间值: ${(sk.intermediates||[]).join(', ')}"><span class="plabel">中间值</span></div>`:'';
    let el=document.createElement('div');
    el.className='block op-block'; el.id=id; el.style.left=x+'px'; el.style.top=y+'px';
    el.innerHTML=`
      <div class="op-row"><span>算子</span><b>${sk.name||sk.id||''}</b><span style="margin-left:auto;font-size:9px;font-weight:700;color:#e5c07b;border:1px solid #e5c07b;border-radius:3px;padding:0 4px" title="沿 ${dim} 切分的第 ${i+1}/${slices.length} 片">切片${i+1}/${slices.length}</span></div>
      <div class="op-row"><span>类型</span><span class="val">${sk.op_type||''} <span style="color:${catColor};font-size:9px">${catLabel}</span></span></div>
      <div class="op-row"><span>ID</span><span class="val">${sk.id||''}</span></div>
      <div class="op-row"><span>数据精度</span><span class="val p">${dataPrec}</span></div>
      <div class="op-row"><span>执行精度</span><span class="val p">${execPrec}</span></div>
      <div class="op-row"><span>计算量</span><span class="val c">${sk.compute_gflops||'—'}</span></div>
      <div class="op-row"><span>存储量</span><span class="val m">${sk.memory||'—'}</span></div>
      ${midData?`<div class="op-row"><span>中间值</span><span class="val mid">${midData}</span></div>`:''}
      <div class="op-row"><span>形状</span><span class="val">${attrs}</span></div>
      <div class="op-row"><span>输入</span><span class="val" style="color:#7aa2c4">${inData}</span></div>
      <div class="op-row"><span>输出</span><span class="val" style="color:#d19a66">${outData}</span></div>
      <div class="op-row"><span>运行设备</span><span class="op-tag" data-tag="1" title="拖动此标签到硬件方块 = 该切片算子在该设备运行">—(未放置)</span></div>
      ${inPorts}${outPorts}${midPort}
    `;
    document.getElementById('canvas').appendChild(el);
    blocks[id]={type:'operator',el:el,x:x,y:y,data:sk,
      kernelId:logicalId,           // 映射仍用逻辑算子 id（后端按它做张量并行）
      name:(sk.name||sk.id||''),
      inputs:(sk.inputs||[]),outputs:(sk.outputs||[]),intermediates:(sk.intermediates||[]),
      isSlice:true,sliceOf:logicalId,sliceIndex:i+1,sliceDim:dim,
      splitRuleIdx:ruleIdx, sliceIdxInRule:i,
      recommendedDevice:recDev,recommendedBackId:recBack,
      recEl:null,tagEl:el.querySelector('.op-tag')};
    el.querySelectorAll('.port').forEach(p=>p.addEventListener('mousedown',portMouseDown));
    makeTagDraggable(blocks[id].tagEl, id);
    makeDraggable(el);
    // 初始放置：切片 i 对应张量并行设备列表第 i 个（backId → 画布 id）
    let hwId=backToId[devices[i]];
    if(hwId) groupOpToHW(id, hwId);
  });
  // 转移连线：输入 fan-in 到每片、输出/中间值 fan-out 自每片；权重线由 syncWeightConns 重建
  _remapSplitConnections(oldId, newIds);
  Object.keys(blocks).forEach(wid=>{ if(blocks[wid].type==='weight') syncWeightConns(wid); });
  drawConnections(); refreshConnList(); updateLinkSelects();
}
function _remapSplitConnections(oldId, newIds){
  let keep=[];
  connections.forEach(c=>{
    if(c._autoW) return;   // 权重自动连线稍后由 syncWeightConns 重建（避免引用已删端口）
    let mIn=/^(.*)_in(\d+)$/.exec(c.to);
    if(mIn && mIn[1]===oldId){
      newIds.forEach(nid=>keep.push({from:c.from, to:nid+'_in'+mIn[2], label:c.label, lat:c.lat, isLink:c.isLink}));
      return;
    }
    let mOut=/^(.*)_out(\d+)$/.exec(c.from);
    let mMid=/^(.*)_mid$/.exec(c.from);
    if((mOut && mOut[1]===oldId) || (mMid && mMid[1]===oldId)){
      newIds.forEach(nid=>{
        let from=(mOut)?(nid+'_out'+mOut[2]):(nid+'_mid');
        keep.push({from:from, to:c.to, label:c.label, lat:c.lat, isLink:c.isLink});
      });
      return;
    }
    keep.push(c);
  });
  connections=keep;
}


// ─── Init ───
// 链路系统：先拉后端默认链路带宽表（单一事实来源），供自定义硬件"加一栏"填写与已有种类间的带宽
fetch('/api/link_defaults').then(r=>r.json()).then(d=>{
  linkBw = JSON.parse(JSON.stringify(d.table||{}));
  LINK_FALLBACK = d.fallback||100;
}).catch(()=>{ linkBw = {}; });
// 先拉后端硬件能力表（单一事实来源），再添加 5 类参考硬件
// （IC 参考需要 GPU/DRAM-PIM/SRAM-PIM/ReRAM-PIM + 纯存储 DRAM 放 FFN 权重）
fetch('/api/hardware_capability').then(r=>r.json()).then(cap=>{
  HW_CAP=cap||{};
  addPresetHW('GPU');
  setTimeout(()=>{addPresetHW('DRAM_PIM')},120);
  setTimeout(()=>{addPresetHW('SRAM_PIM')},240);   // IC 参考：LN/Softmax/激活 → SRAM-PIM
  setTimeout(()=>{addPresetHW('RERAM_PIM')},360);  // IC 参考：词表 LMHead/Embedding → ReRAM-PIM
  setTimeout(()=>{addPresetHW('DRAM')},480);       // 纯存储 DRAM：放 W_mlp FFN 权重
  setTimeout(()=>loadModel(),520);
  setTimeout(()=>loadExperiments(),140);
}).catch(()=>{
  addPresetHW('GPU');
  setTimeout(()=>{addPresetHW('DRAM_PIM')},120);
  setTimeout(()=>{addPresetHW('SRAM_PIM')},240);
  setTimeout(()=>{addPresetHW('RERAM_PIM')},360);
  setTimeout(()=>{addPresetHW('DRAM')},480);
  setTimeout(()=>loadModel(),520);
  setTimeout(()=>loadExperiments(),140);
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("\n  LLM-PIMSim GUI — Visual Topology Editor")
    print("  Open: http://127.0.0.1:5000\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
