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
    model = (d.get("model") or "llama7b").strip()
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
        # LMHead / Embedding：词表大权重 → RERAM-PIM
        if ot in ("LMHEAD", "EMBEDDING"):
            return "reram0" if has("reram0") else ("gpu0" if has("gpu0") else None)
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
                         st.get("weight_blocks") or d.get("weight_blocks"))
    # 后端充分校验：与 /api/run 完全一致（真实 workload + 真实硬件 + 覆盖）
    exp_path = d.get("experiment") or ""
    if exp_path:
        run_map = d.get("compute_map") or st.get("run_map") or {}
        wb = _map_weight_devices(d.get("weight_blocks") or st.get("weight_blocks"),
                                 d.get("hardware") or st.get("hardware") or [])
        splits = _map_splits_devices(d.get("splits"), d.get("hardware") or st.get("hardware") or [])
        rv = validate_runnable(str(BASE / "configs" / exp_path),
                               compute_map_override=run_map, weight_blocks=wb,
                               splits_override=splits)
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
                             st.get("weight_blocks"))
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
                                    weight_blocks=wb, splits_override=splits)
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
    model = request.args.get("model","llama7b")
    MODELS = {
        "llama7b": dict(hidden=4096, ffn=11008, heads=32, head_dim=128, vocab=32000, layers=32, seq=2048),
    }
    m = MODELS.get(model, MODELS["llama7b"])
    from workload_model import build_model_workload
    wl = build_model_workload(
        hidden=m["hidden"], ffn_size=m["ffn"], num_heads=m["heads"],
        head_dim=m["head_dim"], vocab=m["vocab"], num_layers=m["layers"],
        input_tokens=m["seq"], decode_steps=128
    )
    return jsonify(wl.to_dict())

@app.route("/api/models")
def api_models():
    return jsonify(list_models())

@app.route("/api/split", methods=["POST"])
def api_split():
    """算子切割：{model, kernel, dim, parts:[...]} → 新的 kernel dict 列表"""
    from core.splitter import split_kernel_dict
    d = request.get_json()
    model = d.get("model", "llama7b")
    kernel_id = d.get("kernel", "")
    dim = d.get("dim", "")
    parts = d.get("parts", [])
    if not dim or not isinstance(parts, list) or not len(parts):
        return jsonify({"error": "需要 dim 和 parts"}), 400
    # 从 workload 找到该 kernel
    MODELS = {"llama7b":[4096,11008,32,128,32000,32,2048]}
    m = MODELS.get(model, MODELS["llama7b"])
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
    输入: ?model=llama7b&experiment=xxx&split=W_mlp:2,W_attn:4 （split 可选：类名:片数）
    输出: {weight_blocks:[{weight_id, weight_class, layer, rows, cols, bytes, split_dim,
          consumers, input_slots, device(默认建议), partitions:[{partition_id, rows, cols, bytes, device}]}],
          default_device}
    """
    from weights import build_weight_blocks
    model = request.args.get("model", "llama7b")
    split_arg = request.args.get("split", "")     # "W_mlp:2,W_attn:4"
    exp_path = (request.args.get("experiment") or "experiments/04_ic_reference.yaml").strip()
    MODELS = {
        "llama7b": dict(hidden=4096, ffn=11008, heads=32, vocab=32000, layers=32),
    }
    m = MODELS.get(model, MODELS["llama7b"])
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
    # IC 参考：按权重类别推荐放置设备（W_attn→GPU、W_mlp→DRAM-PIM、W_ln→SRAM-PIM、词表→RERAM）
    default_dev = _exp_default_device(exp_path) or "gpu0"
    cls_dev = {
        "W_attn": "gpu0", "W_mlp": "pim0", "W_ln": "sram0",
        "W_head": "reram0", "W_embed": "reram0",
    }
    out = []
    for wb in blocks.values():
        dev = cls_dev.get(wb.weight_class) or default_dev
        out.append({
            "weight_id": wb.weight_id, "weight_class": wb.weight_class,
            "layer": wb.layer, "rows": wb.rows, "cols": wb.cols,
            "bytes": wb.bytes, "split_dim": wb.split_dim,
            "consumers": wb.consumers, "input_slots": wb.input_slots,
            "ports": wb.to_port_dict(),   # v3 结构化端口（权重→算子的数据流）
            "device": dev,
            "num_layers": m["layers"],
            "partitions": [{"partition_id": p.partition_id, "rows": p.rows,
                             "cols": p.cols, "bytes": p.bytes,
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
.op-block{min-width:280px;background:#1d212b;border:1px solid #39404d;padding:10px 12px;font-size:11px;border-radius:var(--radius);box-shadow:var(--shadow)}
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
.modal{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:22px 26px;width:380px;box-shadow:0 16px 48px rgba(0,0,0,.5)}
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
      <option value="llama7b">LLaMA-7B</option>
    </select>
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
    <button class="btn-add" onclick="showCustomHWModal()" style="border-color:var(--accent);color:var(--accent)">+ 自定义硬件</button>
  </div>

  <div class="section">
    <label>互连</label>
    <div style="display:flex;gap:4px">
      <select id="link-src" style="width:48%"><option>--</option></select>
      <select id="link-dst" style="width:48%"><option>--</option></select>
    </div>
    <input type="number" id="link-bw" placeholder="带宽 (GB/s)" step="1" min="1">
    <input type="number" id="link-lat" placeholder="延迟 (ns)" step="1" min="0">
    <button class="btn-add" onclick="addLink()">+ 连接</button>
    <div id="conn-list" style="margin-top:6px"></div>
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
    拖拽算子方块到硬件方块内 = Mapping<br>
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
    <h3>算子切割</h3>
    <label>算子</label><input id="s-kernel" readonly>
    <label>切割维度</label><select id="s-dim"></select>
    <label>分段值（逗号分隔，如 5,15）</label><input id="s-parts" value="5,15">
    <div style="font-size:10px;color:var(--text2);margin-top:6px">沿所选维度将算子切成多段，计算量/存储按比例变化，精度不变</div>
    <div class="btn-row">
      <button onclick="closeSplitModal()" style="background:#444;color:#ccc">取消</button>
      <button onclick="doSplit()" style="background:var(--accent);color:#1a1d23">确定切割</button>
    </div>
  </div>
</div>

<div class="modal-overlay" id="hw-modal">
  <div class="modal">
    <h3>自定义硬件</h3>
    <label>名称</label><input id="c-name" value="my-device">
    <label>类型</label><select id="c-type"><option>GPU</option><option>DRAM_PIM</option><option>SRAM_PIM</option><option>RERAM_PIM</option><option>CPU</option></select>
    <label>算力</label><input id="c-compute" value="100 TFLOPS">
    <label>容量</label><input id="c-mem" value="64 GB">
    <label>读带宽</label><input id="c-rbw" value="1000 GB/s">
    <label>写带宽</label><input id="c-wbw" value="800 GB/s">
    <label>精度（逗号分隔：FP32,FP16,INT8,INT4）</label><input id="c-precision" value="FP32,FP16,INT8,INT4">
    <div class="btn-row">
      <button onclick="closeCustomHWModal()" style="background:#444;color:#ccc">取消</button>
      <button onclick="addCustomHW()" style="background:var(--accent);color:#1a1d23">确定</button>
    </div>
  </div>
</div>

<script>
// ─── State ───
let blocks={}, hwCounter=0, opCounter=0, connections=[], linkId=0;
let dragState=null, connectState=null, selectedBlock=null;
let currentWorkload=null;   // 当前加载的 workload（供依赖图展示）
let pendingSplits=[];       // 前端算子切割规则，运行时会传给后端真实执行（张量并行）
const HW_TYPES={
  GPU:{color:'#61afef',label:'GPU',compute:'300 TFLOPS',mem:'80 GB',rBW:'2000 GB/s',wBW:'1800 GB/s'},
  DRAM_PIM:{color:'#98c379',label:'DRAM-PIM',compute:'50 TOPS',mem:'512 GB',rBW:'800 GB/s',wBW:'600 GB/s'},
  SRAM_PIM:{color:'#e5c07b',label:'SRAM-PIM',compute:'200 TOPS',mem:'128 MB',rBW:'4000 GB/s',wBW:'3000 GB/s'},
  RERAM_PIM:{color:'#c678dd',label:'ReRAM-PIM',compute:'100 TOPS',mem:'1000 GB',rBW:'300 GB/s',wBW:'100 GB/s'},
  CPU:{color:'#abb2bf',label:'CPU',compute:'1 TFLOPS',mem:'64 GB',rBW:'50 GB/s',wBW:'50 GB/s'}
};
// 前端硬件类型 → 后端 experiment.yaml 里的真实设备 id（02_gpu_pim 用 gpu0/pim0）
const PRESET_BACKID={GPU:'gpu0',DRAM_PIM:'pim0',SRAM_PIM:'sram0',RERAM_PIM:'reram0',CPU:'cpu0'};

// ─── Hardware: preset (non-editable) + custom (editable via modal) ───
function addPresetHW(type){
  let t=HW_TYPES[type];if(!t)return;
  _makeHWBlock(type, t.label, t.compute, t.mem, t.rBW, t.wBW, false);
}
function showCustomHWModal(){document.getElementById('hw-modal').classList.add('show')}
function closeCustomHWModal(){document.getElementById('hw-modal').classList.remove('show')}
function addCustomHW(){
  let name=document.getElementById('c-name').value||'custom';
  let type=document.getElementById('c-type').value;
  let id=name.toLowerCase().replace(/\s+/g,'-');
  let color=HW_TYPES[type]?HW_TYPES[type].color:'#abb2bf';
  _makeHWBlock(type, name+' ('+type+')',
    document.getElementById('c-compute').value,
    document.getElementById('c-mem').value,
    document.getElementById('c-rbw').value,
    document.getElementById('c-wbw').value, true, id, color,
    null, document.getElementById('c-precision').value);
  closeCustomHWModal();
}

function _makeHWBlock(type, label, compute, mem, rBW, wBW, editable, forceId, forceColor, backId, precisionStr){
  let id=forceId||('hw'+(hwCounter++));
  if(!backId) backId=PRESET_BACKID[type]||id;   // 预设类型→后端真实id
  let color=forceColor||HW_TYPES[type].color;
  let x=40+hwCounter*30,y=40+hwCounter*40;
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
       <div class="param"><span>精度</span><span style="color:#c678dd;font-size:10px;text-align:right">${precisionStr}</span></div>`
    : `<div class="param"><span>算力</span><span class="val">${compute}</span></div>
       <div class="param"><span>容量</span><span class="val">${mem}</span></div>
       <div class="param"><span>读带宽</span><span class="val">${rBW}</span></div>
       <div class="param"><span>写带宽</span><span class="val">${wBW}</span></div>
       <div class="param"><span>精度</span><span class="val" style="color:#c678dd">${precisionStr}</span></div>`;
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
  blocks[id]={type:type,el:el,x:x,y:y,opGroup:[],params:{compute,mem,rBW,wBW},editable,backId,precision:precisionStr};
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
function loadModel(){
  let m=document.getElementById('sel-model').value;
  fetch('/api/workload?model='+m).then(r=>r.json()).then(wl=>{
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
      let x=x0+Math.floor(i/4)*310,y=y0+(i%4)*150;
      let kvHint=k.is_kv_dependent?'[KV动态]':'';
      // 输入/输出/中间数据
      let inData=(k.inputs||[]).join(', ')||'—';
      let outData=(k.outputs||[]).join(', ')||'—';
      let midData=(k.intermediates||[]).join(', ')||'—';
      // attributes 序列化（M/K/N 或 seq 等）
      let attrs=Object.entries(k.attributes||{}).map(([a,b])=>a+'='+b).join(', ')||'—';
      let el=document.createElement('div');
      el.className='block op-block';el.id=id;el.style.left=x+'px';el.style.top=y+'px';
      // 多个输入 → 多个读端口（纵向分布），多个输出 → 多个写端口
      let nIn=(k.inputs||[]).length, nOut=(k.outputs||[]).length;
      let inTop=(nIn>1?(di)=>25+di*50/(nIn-1):()=>50);  // 多输入纵向排布
      let outTop=(nOut>1?(do_)=>25+do_*50/(nOut-1):()=>50);
      let inPorts=(k.inputs||[]).map((d,di)=>`<div class="port read" data-port="${id}_in${di}" data-port-type="read" style="top:${inTop(di)}%" title="读(${di+1}): ${d}"><span class="plabel">读${di+1}</span></div>`).join('');
      let outPorts=(k.outputs||[]).map((d,do_)=>`<div class="port write" data-port="${id}_out${do_}" data-port-type="write" style="top:${outTop(do_)}%" title="写(${do_+1}): ${d}"><span class="plabel">写${do_+1}</span></div>`).join('');
      let midPort=(k.intermediates||[]).length?`<div class="port mid" data-port="${id}_mid" data-port-type="mid" title="中间值: ${(k.intermediates||[]).join(', ')}"><span class="plabel">中间值</span></div>`:'';
      el.innerHTML=`
        <div class="op-row"><span>类型</span><b>${k.op_type||''}</b>${kvHint?`<span class="tag-kv" style="margin-left:auto;font-size:9px;color:#c792ea;font-weight:600">${kvHint}</span>`:''}</div>
        <div class="op-row"><span>算子ID</span><span class="val">${k.id||''}</span></div>
        <div class="op-row"><span>算子精度</span><span class="val p">${k.precision||'FP16'}</span></div>
        <div class="op-row"><span>计算量</span><span class="val c">${k.compute_gflops||'—'}</span></div>
        <div class="op-row"><span>内存需求</span><span class="val m">${k.memory||'—'}</span></div>
        <div class="op-row"><span>形状/参数</span><span class="val">${attrs}</span></div>
        <div class="op-row rec-row"><span>推荐设备</span><span class="val rec-val" data-rec="1">…(载入中)</span></div>
        <div style="margin-top:6px;border-top:1px dashed #3a4050;padding-top:4px"><span class="split-btn" onclick="openSplitModal('${id}')">✂ 切割</span></div>
        ${inPorts}${outPorts}${midPort}
      `;
      document.getElementById('canvas').appendChild(el);
      blocks[id]={type:'operator',el:el,x:x,y:y,data:k,kernelId:k.id||id,inputs:(k.inputs||[]),outputs:(k.outputs||[]),intermediates:(k.intermediates||[]),recEl:el.querySelector('.rec-val')};
      el.querySelectorAll('.port').forEach(p=>p.addEventListener('mousedown',portMouseDown));
      makeDraggable(el);
    });
    updateStatus('已加载模型: '+m+', 每层'+kernels.length+'个算子');
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
    <span title="算子/权重拖入硬件后会按 mapping 规则应用到全部层" style="color:var(--green);cursor:help">✅ 摆放→全部层</span>`;
}

// ─── 权重块：画布一级节点（用户先决定权重放哪个硬件）───
let currentWeights=[];   // 后端返回的 WeightBlock 列表（供依赖图/切割用）
let wCounter=0;
function loadWeights(){
  let m=document.getElementById('sel-model').value;
  let exp=document.getElementById('sel-experiment')?document.getElementById('sel-experiment').value:'';
  // 清除旧权重块
  Object.keys(blocks).filter(k=>blocks[k].type==='weight').forEach(k=>{blocks[k].el.remove();delete blocks[k]});
  wCounter=0;
  return fetch('/api/weights?model='+m+'&experiment='+encodeURIComponent(exp||'experiments/04_ic_reference.yaml'))
  .then(r=>r.json()).then(d=>{
    currentWeights=d.weight_blocks||[];
    // backId -> 画布硬件块 id（用于自动放置默认设备）
    let backToId={};
    Object.entries(blocks).forEach(([id,b])=>{
      if(b.type!=='operator' && b.backId) backToId[b.backId]=id;
    });
    // 权重块放在算子区下方，按类别分组横向排布
    let x0=600,y0=560;
    let col=0,row=0,perRow=6;
    currentWeights.forEach((wb,i)=>{
      let id='w'+(wCounter++);
      let x=x0+(i%perRow)*260, y=y0+Math.floor(i/perRow)*150;
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
        <div class="w-row"><span>所在设备</span><span class="val w-dev" data-dev="1">—(未放置)</span></div>
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
        device:'',parentHW:null,devEl:el.querySelector('[data-dev]')};
      makeDraggable(el);
      el.querySelectorAll('.port').forEach(p=>p.addEventListener('mousedown',portMouseDown));
      // 默认设备建议（backId）→ 若画布有对应硬件，自动放置
      let hwId=backToId[wb.device];
      if(hwId){ groupWeightToHW(id, hwId); }
    });
    updateStatus('已加载 '+currentWeights.length+' 个权重块——拖入硬件方块即可决定权重放置');
  }).catch(e=>{updateStatus('加载权重块失败: '+e); currentWeights=[];});
}
function fmtBytes(b){
  if(!b&&b!==0)return '—';
  if(b>=1e9)return (b/1e9).toFixed(1)+' GB';
  if(b>=1e6)return (b/1e6).toFixed(1)+' MB';
  if(b>=1e3)return (b/1e3).toFixed(1)+' KB';
  return b+' B';
}
// 权重块拖入硬件 → 记录设备 + 自动生成权重→算子的连线
function groupWeightToHW(wid, hwId){
  let wb=blocks[wid], hw=blocks[hwId]; if(!wb||!hw)return;
  wb.parentHW=hwId;
  wb.device=hw.backId||hwId;
  if(wb.devEl) wb.devEl.textContent=hwId;
  syncWeightConns(wid);
  updateStatus('权重 '+wb.weightId+' 已放置到 '+hwId);
}
function detachWeight(wid){
  let wb=blocks[wid]; if(!wb)return;
  wb.parentHW=null; wb.device='';
  if(wb.devEl) wb.devEl.textContent='—(未放置)';
  syncWeightConns(wid);
}
function deleteWeight(wid){
  let wb=blocks[wid]; if(!wb)return;
  // 移除该权重块自动生成的连线（按 _autoW 标记）
  connections=connections.filter(c=>c._autoW!==wid);
  wb.el.remove(); delete blocks[wid];
  drawConnections(); refreshConnList();
  updateStatus('已删除权重块 '+wb.weightId);
}
// 为该权重块自动生成/刷新连线：从权重块自身的端口（{weightId}_r / {partition_id}_r）
// 出发，连到消费该权重的算子输入端口 —— 权重是否有端口连接一目了然。
function syncWeightConns(wid){
  let wb=blocks[wid]; if(!wb)return;
  // 移除该权重块现有的自动连线
  connections=connections.filter(c=>c._autoW!==wid);
  if(!wb.parentHW) return;
  // 权重端口集合：未切割 → [{weightId}_r]；切割 → 每片一个端口 {partition_id}_r
  let ports = (wb.parts&&wb.parts.length)
    ? wb.parts.map(p=>p.partition_id+'_r')
    : [wb.weightId+'_r'];
  // 找消费该权重的算子块：inputs 里包含该权重名
  Object.entries(blocks).forEach(([oid,b])=>{
    if(b.type!=='operator')return;
    let idx=(b.inputs||[]).indexOf(wb.weightId);
    if(idx<0) return;
    ports.forEach(pt=>{
      connections.push({from:pt, to:oid+'_in'+idx, label:'W:'+pt, lat:0, _autoW:wid});
    });
  });
  drawConnections(); refreshConnList();
}
// 切割权重：输入片数 → 重新拉取带分片的权重块
function splitWeight(wid){
  let wb=blocks[wid]; if(!wb)return;
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
    wb.parts=nb.partitions||[];
    wb.device=''; wb.parentHW=null;
    if(wb.devEl) wb.devEl.textContent='—(未放置)';
    // 更新块内分片显示
    let shardEl=wb.el.querySelector('.w-shard');
    let shardHTML=wb.parts.length?`<div class="w-shard">已切 <b style="color:#c792ea">${wb.parts.length}</b> 片：<span class="sp">${wb.parts.map(p=>p.partition_id).join(', ')}</span></div>`:'';
    if(shardEl){shardEl.outerHTML=shardHTML;} else if(shardHTML){
      let tools=wb.el.querySelector('.w-tools');
      tools.insertAdjacentHTML('beforebegin',shardHTML);
    }
    syncWeightConns(wid);
    updateStatus('权重 '+wb.weightId+' 已切成 '+wb.parts.length+' 片（ALL-GATHER）');
  }).catch(e=>updateStatus('切割失败: '+e));
}
// 收集画布权重块 → 校验/运行用的 weight_blocks
function collectWeightBlocks(){
  let out=[];
  let numLayers=0;
  Object.entries(blocks).forEach(([id,b])=>{
    if(b.type!=='weight')return;
    if(b.numLayers) numLayers=Math.max(numLayers,b.numLayers);
  });
  Object.entries(blocks).forEach(([id,b])=>{
    if(b.type!=='weight')return;
    let parts=b.parts||[];
    let hwId=b.parentHW||'';   // 画布硬件 id（校验器按它查硬件）
    let partitions=parts.map(p=>({
      partition_id:p.partition_id,
      bytes:p.bytes||0,
      device: hwId   // 分片随整块放置在同一画布硬件
    }));
    out.push({
      weight_id:b.weightId, weight_class:b.weightClass,
      consumers:b.consumers||[], input_slots:b.inputSlots||{},
      device:hwId,
      bytes:b.bytes||0,
      num_layers:numLayers||1,
      partitions:partitions
    });
  });
  return out;
}

// ─── 算子运行参考（推荐设备）───
function applyRecommendation(){
  // 先重置当前参考：把所有已放进硬件的算子叉出来，清空连线，再按新参考重新部署
  Object.keys(blocks).forEach(oid=>{
    let b=blocks[oid];
    if(b && b.type==='operator' && b.parentHW){ try{detachOp(oid);}catch(e){} }
  });
  connections=[]; drawConnections(); refreshConnList(); updateLinkSelects();
  loadWeights();   // 权重块跟随当前实验重载（默认设备来自该实验 mapping）
  // 收集当前画布硬件与算子（连同 inputs/outputs/intermediates 供后端构造数据流连线）
  let hardware=[], operators=[];
  Object.entries(blocks).forEach(([id,b])=>{
    if(b.type==='operator'){
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
      if(b.type!=='operator')return;
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
  if(!dragState)return;
  let wrap=document.getElementById('canvas-wrap');
  let dx=e.clientX-dragState.ox,dy=e.clientY-dragState.oy;
  let wx=wrap.scrollLeft,wy=wrap.scrollTop;
  dragState.el.style.left=(dx+wx-252)+'px';
  dragState.el.style.top=(dy+wy-0)+'px';
  drawConnections();
});
document.addEventListener('mouseup',e=>{
  if(dragState){
    dragState.el.classList.remove('dragging');
    let id=dragState.el.id;
    if(blocks[id]){
      blocks[id].x=parseInt(dragState.el.style.left);
      blocks[id].y=parseInt(dragState.el.style.top);
    }
    // Check if operator/weight dropped inside hardware → group them
    if(blocks[id]&&!blocks[id].parentHW&&(blocks[id].type==='operator'||blocks[id].type==='weight')){
      let opRect=dragState.el.getBoundingClientRect();
      let matchedHW=null;
      Object.entries(blocks).forEach(([hid,b])=>{
        if(!b.type||b.type==='operator'||b.type==='weight')return;
        let hwRect=b.el.getBoundingClientRect();
        let cx=opRect.left+opRect.width/2,cy=opRect.top+opRect.height/2;
        if(cx>=hwRect.left&&cx<=hwRect.right&&cy>=hwRect.top&&cy<=hwRect.bottom){
          matchedHW=hid;
        }
      });
      if(matchedHW){
        if(blocks[id].type==='weight') groupWeightToHW(id, matchedHW);
        else groupOpToHW(id, matchedHW);
      }
    }
    dragState=null;
  }
});

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
  // Create compact in-group element
  let k=op.data||{};
  let item=document.createElement('div');
  item.className='op-in-hw';item.id=opId+'_grp';
  item.innerHTML=`<span style="color:#e5e5e5">${k.id||opId}</span><span style="color:var(--text2)">${k.type||k.kernel_type||''}</span><span class="detach-btn" onclick="detachOp('${opId}')" title="解除映射">&times;</span>`;
  hw.el.querySelector('.hw-body').appendChild(item);
  // Hide original block and track
  op.el.style.display='none';
  op.parentHW=hwId;
  hw.opGroup=hw.opGroup||[];hw.opGroup.push(opId);
  // Expand HW to show grouped operators
  hw.el.style.minHeight='auto';
  drawConnections();   // 折叠后重绘：指向隐藏端口的线不再绘制
  updateStatus('算子 '+opId+' 已映射到 '+hwId);
}
function detachOp(opId){
  let op=blocks[opId]; if(!op||!op.parentHW)return;
  let hw=blocks[op.parentHW];
  // Remove grouped item
  let grp=document.getElementById(opId+'_grp');if(grp)grp.remove();
  // Restore original block next to HW
  op.el.style.display='block';
  let hwX=parseInt(hw.el.style.left),hwY=parseInt(hw.el.style.top);
  op.el.style.left=(hwX+280)+'px'; op.el.style.top=hwY+'px';
  op.x=hwX+280;op.y=hwY;
  // Clean up
  if(hw) hw.opGroup=(hw.opGroup||[]).filter(x=>x!=opId);
  op.parentHW=null;
  drawConnections();   // 展开后重绘：端口恢复可见，线恢复显示
  updateStatus('算子 '+opId+' 已解除映射');
}

// ─── Connections list ───
function refreshConnList(){
  let el=document.getElementById('conn-list');if(!el)return;
  el.innerHTML=connections.filter(c=>c.isLink).map((c,i)=>`<div class="conn-item"><span>${c.from.replace('_r','')} ↔ ${c.to.replace('_w','')} @ ${c.label||'?'}</span><span class="del" onclick="delConn(${i})">&times;</span></div>`).join('');
}
function delConn(i){
  connections.splice(i,1);drawConnections();refreshConnList();updateLinkSelects();
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
  // 端点相对 canvas 原点的坐标（#canvas 与 #svg-lines 同一坐标系）
  let fromRect=connectState.el.getBoundingClientRect();
  let canvasRect=document.getElementById('canvas').getBoundingClientRect();
  let x1=fromRect.left+fromRect.width/2-canvasRect.left;
  let y1=fromRect.top+fromRect.height/2-canvasRect.top;
  // 鼠标相对 canvas 原点的坐标
  let x2=e.clientX-canvasRect.left;
  let y2=e.clientY-canvasRect.top;
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
  connections.forEach((c,i)=>{
    let fromEl=document.querySelector('[data-port="'+c.from+'"]');
    let toEl=document.querySelector('[data-port="'+c.to+'"]');
    if(!fromEl||!toEl)return;
    // 端口或其祖先不可见（算子已被折叠进硬件 display:none）→ 不绘制，避免线残留在外面
    if(!_portVisible(fromEl)||!_portVisible(toEl))return;
    let fRect=fromEl.getBoundingClientRect(),tRect=toEl.getBoundingClientRect();
    let x1=fRect.left+fRect.width/2-canvasRect.left;
    let y1=fRect.top+fRect.height/2-canvasRect.top;
    let x2=tRect.left+tRect.width/2-canvasRect.left;
    let y2=tRect.top+tRect.height/2-canvasRect.top;
    let path=document.createElementNS('http://www.w3.org/2000/svg','path');
    let mx=(x1+x2)/2;
    path.setAttribute('d',`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`);
    path.setAttribute('stroke','#5c6370');path.setAttribute('stroke-width','2');
    path.setAttribute('fill','none');
    svg.appendChild(path);
  });
}
function _portVisible(el){
  let n=el;
  while(n&&n!==document.body){
    if(n.style&&n.style.display==='none')return false;
    n=n.parentElement;
  }
  return true;
}
// Redraw on scroll
document.getElementById('canvas-wrap').addEventListener('scroll',drawConnections);

// ─── Interconnect links ───
function updateLinkSelects(){
  let srcSel=document.getElementById('link-src'),dstSel=document.getElementById('link-dst');
  let hwIds=Object.keys(blocks).filter(k=>blocks[k].type!=='operator');
  srcSel.innerHTML=dstSel.innerHTML='<option>--</option>';
  hwIds.forEach(id=>{let o=document.createElement('option');o.value=id;o.textContent=id;srcSel.appendChild(o);dstSel.appendChild(o.cloneNode(true))});
}
function addLink(){
  let src=document.getElementById('link-src').value,dst=document.getElementById('link-dst').value;
  let bw=parseFloat(document.getElementById('link-bw').value)||100;
  let lat=parseInt(document.getElementById('link-lat').value)||500;
  if(!src||!dst||src===dst||src==='--'||dst==='--')return;
  connections.push({from:src+'_r',to:dst+'_w',label:bw+'GB/s',lat:lat,isLink:true});
  updateStatus('链路已添加: '+src+' -> '+dst+' @ '+bw+' GB/s, '+lat+'ns');
  document.getElementById('link-bw').value='';document.getElementById('link-lat').value='';
  drawConnections();refreshConnList();
}
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
        compute:p.compute, mem:p.mem, rBW:p.rBW, wBW:p.wBW});
    }
  });
  let conns=connections.map(c=>({from:c.from,to:c.to,label:c.label,lat:c.lat,isLink:!!c.isLink}));
  return {hardware:hardware,operators:operators,connections:conns,
          compute_map:compute_map,run_map:run_map,mappedCount:mappedCount,
          weight_blocks:collectWeightBlocks()};
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
      compute_map:st.run_map, state:st, splits:pendingSplits})})
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
      run_validation:runValidation, splits: pendingSplits})})
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
  ['llama7b'].forEach(id=>{
    let o=document.createElement('option'); o.value=id; o.textContent=id; msel.appendChild(o);
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
    // 刷新实验下拉并选中新实验，随后按新实验的映射规则部署参考
    loadExperiments();
    setTimeout(()=>{
      let sel2=document.getElementById('sel-experiment');
      sel2.value=d.experiment;
      closeNewExperiment();
      applyRecommendation();   // 部署新实验的参考映射（算子放入推荐硬件）
    },300);
  }).catch(e=>{ msg.style.color='var(--red)'; msg.textContent='请求失败: '+e.message; });
}

// ─── Save ───
function saveConfig(){
  let hwYaml='devices:\n';
  Object.entries(blocks).forEach(([id,b])=>{
    if(b.type==='operator')return;
    let t=HW_TYPES[b.type];
    hwYaml+=`  - id: ${id}\n    type: ${b.type}\n    compute:\n      peak_tflops: 300\n    memory:\n      capacity_gb: 80\n`;
  });
  let icYaml='links:\n';
  connections.filter(c=>c.isLink).forEach(c=>{
    icYaml+=`  - src: ${c.from.replace('_r','')}\n    dst: ${c.to.replace('_w','')}\n    bandwidth_gbs: ${c.label||100}\n    latency_ns: ${c.lat||500}\n`;
  });
  fetch('/api/write',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:'hardware_gui.yaml',content:hwYaml})});
  fetch('/api/write',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:'interconnect_gui.yaml',content:icYaml})});
  updateStatus('已保存: hardware_gui.yaml, interconnect_gui.yaml');
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
  let kernels=currentWorkload.layers[0];
  if(!kernels||!kernels.length){body.innerHTML='<div style="color:var(--text2)">无算子</div>';return}

  // 权重源：currentWeights 里的权重块（画布权重节点），作为额外"源节点"
  let weightNodes = (currentWeights||[]).map((wb,i)=>({
    id: wb.weight_id, label: wb.weight_id, cls: wb.weight_class||'',
    consumers: wb.consumers||[]
  }));

  // 构建 数据名→生产算子 与 数据名→消费算子列表，用于分层与连线
  let producer={}, consumerMap={};
  kernels.forEach(k=>{
    (k.outputs||[]).forEach(d=>producer[d]=k.id);
    (k.intermediates||[]).forEach(d=>producer[d]=k.id);
  });
  kernels.forEach(k=>{
    (k.inputs||[]).forEach(d=>{consumerMap[d]=(consumerMap[d]||[]);consumerMap[d].push(k.id)});
  });
  // 权重 → 消费算子：权重块名（L0_q_w 等）直接作为数据名
  weightNodes.forEach(wn=>{
    wn.consumers.forEach(kid=>{ consumerMap[wn.id]=(consumerMap[wn.id]||[]); if(!consumerMap[wn.id].includes(kid)) consumerMap[wn.id].push(kid); });
  });

  // 确定每算子的依赖层级：level = 其所有输入来源算子的 max level + 1
  let upstream={};  // opId -> [producerOpIds]
  kernels.forEach(k=>{
    (k.inputs||[]).forEach(d=>{ if(producer[d]&&producer[d]!==k.id){upstream[k.id]=(upstream[k.id]||[]);upstream[k.id].push(producer[d]);} });
  });
  let level={};
  function calcLevel(opId){
    if(level[opId]!==undefined)return level[opId];
    let ups=upstream[opId]||[];
    if(!ups.length){level[opId]=0;return 0;}
    let maxL=0; ups.forEach(u=>{ maxL=Math.max(maxL,calcLevel(u)+1); });
    level[opId]=maxL; return maxL;
  }
  kernels.forEach(k=>calcLevel(k.id));

  // 分层布局：相同 level 的算子放同一"列"，列=层级序；每列内垂直均匀分布
  let nodeW=170;
  let layersByLevel={};
  kernels.forEach(k=>{ layersByLevel[level[k.id]]=(layersByLevel[level[k.id]]||[]).push&&0; });
  // 重建为 map level->[opId]
  layersByLevel={};
  kernels.forEach(k=>{ (layersByLevel[level[k.id]]=layersByLevel[level[k.id]]||[]).push(k.id); });
  let levels=Object.keys(layersByLevel).map(Number).sort((a,b)=>a-b);
  let colX={};
  levels.forEach((lv,i)=>{ colX[lv]=250+i*230; });   // 最左 230px 留给权重源列

  // 节点高度随端口数增长；同一列内垂直依次排列（间距 >= NODE_GAP），避免多端口节点重叠
  const nodeH=70, NODE_GAP=20;
  const nodeHeight=function(k){let n=Math.max((k.inputs||[]).length,(k.outputs||[]).length);return Math.max(nodeH,(n>0?(25+n*20):nodeH)+8);};
  let positions={};
  let maxY=20;
  // 权重源列（最左）
  let wPositions={};
  let wy=20;
  weightNodes.forEach(wn=>{ wPositions[wn.id]={x:20,y:wy,h:nodeHeight({inputs:wn.consumers,outputs:[]})}; wy+=wPositions[wn.id].h+NODE_GAP; });
  maxY=Math.max(maxY,wy);
  levels.forEach(lv=>{
    let ids=layersByLevel[lv];
    let x=colX[lv];
    let y=20;
    let idH={};
    ids.forEach(oid=>{let k=kernels.find(kk=>kk.id===oid);idH[oid]=nodeHeight(k||{});});
    ids.forEach((oid,vi)=>{ positions[oid]={x:x,y:y,h:idH[oid]}; y+=idH[oid]+NODE_GAP; });
    maxY=Math.max(maxY,y);
  });
  let cw=250+levels.length*230;
  let ch=maxY+20;

  // SVG 连线：每条数据依赖画一条独立线，终点精确指向目标的对应输入端口 y，
  // 起点连到生产者的对应输出端口 y。多输入算子会得到指向不同端口的多条线。
  // 端口中心基准：第 di 个输入/输出端口中心 y（相对节点顶）= PORT_TOP + di*PORT_STEP
  const PORT_TOP=30, PORT_STEP=20;
  let lines='';
  let lineSet=new Set();
  // 权重 → 消费算子边（紫色）
  weightNodes.forEach(wn=>{
    let pw=wPositions[wn.id]; if(!pw)return;
    (wn.consumers||[]).forEach(opB=>{
      if(!positions[opB])return;
      let key=wn.id+'->'+opB; if(lineSet.has(key))return; lineSet.add(key);
      let pb=positions[opB];
      let inIdx=opBInputIdx(opB,wn.id);
      let y1=pw.y+nodeHeight({inputs:wn.consumers,outputs:[]})/2;
      let y2=pb.y+PORT_TOP+inIdx*PORT_STEP+5;
      let x1=pw.x+170, x2=pb.x;
      let my=(y1+y2)/2;
      lines+=`<path d="M${x1},${y1} C${x1+40},${y1} ${x2-40},${y2} ${x2},${y2}" fill="none" stroke="#c792ea" stroke-width="1.5" marker-end="url(#arrow)" opacity="0.9"/>`;
      lines+=`<text x="${(x1+x2)/2+10}" y="${my-4}" font-size="8" fill="#c792ea" text-anchor="middle">权重:${wn.cls}</text>`;
    });
  });
  kernels.forEach(kA=>{
    let outDatas=((kA.outputs||[]).concat(kA.intermediates||[]));
    outDatas.forEach(d=>{
      let consumers=consumerMap[d]||[];
      consumers.forEach(opB=>{
        if(opB===kA.id)return;
        // 按（生产者,数据,目标算子）去重：不同数据 → 同一算子 各画一条；同一数据只画一条。
        let key=kA.id+'->'+d+'->'+opB;
        if(lineSet.has(key))return;lineSet.add(key);
        let pa=positions[kA.id],pb=positions[opB];
        if(!pa||!pb)return;
        // 源端口 y：d 在 kA.outputs 里则用对应输出端口，否则中间量从节点中心出
        let outIdx=(kA.outputs||[]).indexOf(d);
        let y1=outIdx>=0? pa.y+PORT_TOP+outIdx*PORT_STEP+5 : pa.y+nodeH/2;
        // 目标端口 y：d 在 opB.inputs 里的序号 → 对应输入端口
        let inIdx=opBInputIdx(opB,d);
        let y2=pb.y+PORT_TOP+inIdx*PORT_STEP+5;
        let x1=pa.x+nodeW, x2=pb.x;
        let my=(y1+y2)/2;
        lines+=`<path d="M${x1},${y1} C${x1+50},${y1} ${x2-50},${y2} ${x2},${y2}" fill="none" stroke="#61afef" stroke-width="1.5" marker-end="url(#arrow)" opacity="0.85"/>`;
        lines+=`<rect x="${(x1+x2)/2-8}" y="${my-9}" width="66" height="10" fill="#1a1d23" rx="3" opacity="0.92"/>`;
        lines+=`<text x="${(x1+x2)/2+25}" y="${my-1}" font-size="8" fill="#7aa2c4" text-anchor="middle">${d}</text>`;
      });
    });
  });
  function opBInputIdx(opB,d){let idx=(kernels||[]).find(k=>k.id===opB);let arr=(idx&&idx.inputs)||[];let i=arr.indexOf(d);return i>=0?i:0;}

  // 节点 HTML（端口位置与上方 PORT_TOP/PORT_STEP 一致，第 di 个端口 top = 30+di*20）
  let nodes='';
  // 权重源节点（最左列，紫色描边）
  weightNodes.forEach(wn=>{
    let p=wPositions[wn.id]; if(!p)return;
    let h=p.h;
    nodes+=`<div style="position:absolute;left:${p.x}px;top:${p.y}px;width:150px;min-height:${h}px;background:#2a2438;border:1px solid #6a5acd;border-radius:6px;padding:6px 8px;font-size:10px;color:#e5e5e5">
      <div style="font-weight:700;color:#c792ea">${wn.label}</div>
      <div style="color:#b8a6e0;font-size:9px">权重 · ${wn.cls||''}</div>
      <div style="color:#7a8299;font-size:8px;margin-top:2px">供 ${(wn.consumers||[]).length} 算子</div>
    </div>`;
  });
  kernels.forEach(k=>{
    let p=positions[k.id];
    let nIn=(k.inputs||[]).length, nOut=(k.outputs||[]).length;
    let nPorts=Math.max(nIn,nOut);
    let h=Math.max(nodeH, (nPorts>0?(25+nPorts*20):nodeH)+8);
    let inPorts=(k.inputs||[]).map((d,di)=>`<div style="position:absolute;left:-6px;top:${30+di*20}px;width:12px;height:12px;border-radius:50%;background:#61afef;border:1px solid #fff" title="输入${di+1}: ${d}"><span style="position:absolute;left:14px;top:-2px;font-size:7px;color:#7aa2c4;white-space:nowrap">${d}</span></div>`).join('');
    let outPorts=(k.outputs||[]).map((d,di)=>`<div style="position:absolute;right:-6px;top:${30+di*20}px;width:12px;height:12px;border-radius:50%;background:#d19a66;border:1px solid #fff" title="输出${di+1}: ${d}"><span style="position:absolute;right:14px;top:-2px;font-size:7px;color:#d19a66;white-space:nowrap">${d}</span></div>`).join('');
    let midT=(k.intermediates||[]).join(', ');
    nodes+=`<div style="position:absolute;left:${p.x}px;top:${p.y}px;width:${nodeW}px;min-height:${h}px;background:#252931;border:1px solid #3a4050;border-radius:6px;padding:6px 8px;font-size:10px;color:#e5e5e5">
      <div style="font-weight:700;color:${k.is_kv_dependent?'#c678dd':'#e5e5e5'}">${k.id}</div>
      <div style="color:#98c379;font-size:9px">${k.op_type} · ${k.precision||'FP16'}</div>
      ${midT?`<div style="color:#c678dd;font-size:8px;margin-top:2px">中: ${midT}</div>`:''}
      <div style="color:#7a8299;font-size:8px;margin-top:2px">${(k.compute_gflops||'')}</div>
      ${inPorts}${outPorts}
    </div>`;
  });

  // 图例
  let legend=`<div style="position:absolute;right:8px;top:4px;background:#1a1d23;padding:6px 10px;border-radius:5px;font-size:9px;color:var(--text2);border:1px solid var(--border)">
    <span style="color:#61afef">●输入</span> <span style="color:#d19a66">●输出</span> <span style="color:#c678dd">●KV动态算子</span>
    <div style="margin-top:3px;color:#7aa2c4">→ 依赖连线（标数据名）</div>
    <div style="margin-top:3px;color:#c792ea">→ 权重输入（紫色，左侧为权重块）</div>
  </div>`;

  body.innerHTML=`<div style="position:relative;width:${cw}px;height:${ch+10}px">
    <svg width="${cw}" height="${ch+10}" style="position:absolute;top:0;left:0">
      <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#61afef"/></marker></defs>
      ${lines}
    </svg>
    ${nodes}${legend}
  </div>`;
  // 面板宽度要能容纳大图
  document.querySelector('.dep-drawer').style.width='860px';
  document.querySelector('.dep-body').style.overflow='auto';
}

// ─── 算子切割 ───
let currentSplitBlock=null;
function openSplitModal(opBlockId){
  let b=blocks[opBlockId]; if(!b)return;
  let k=b.data; if(!k)return;
  currentSplitBlock=opBlockId;
  document.getElementById('s-kernel').value=k.id||opBlockId;
  document.getElementById('s-parts').value='';
  // 可切割维度：attributes 里的数值字段（M/K/N/seq 等，排除字符串 'kv_len' 也可选）
  let dimSel=document.getElementById('s-dim');
  dimSel.innerHTML='';
  let attrs=k.attributes||{};
  Object.entries(attrs).forEach(([d,v])=>{
    // 允许数值或 'kv_len' 动态维度
    if(typeof v==='number' || String(v).toLowerCase()==='kv_len'){
      let o=document.createElement('option');o.value=d;o.textContent=d+' = '+v;dimSel.appendChild(o);
    }
  });
  if(!dimSel.options.length){dimSel.innerHTML='<option value="">无可切割维度</option>'}
  document.getElementById('s-parts').value='';
  // 预填提示：把当前值拆两半
  let dim=dimSel.value;let v=attrs[dim];
  if(typeof v==='number'){document.getElementById('s-parts').value=Math.round(v/2)+','+(v-Math.round(v/2));}
  document.getElementById('split-modal').classList.add('show');
}
function closeSplitModal(){document.getElementById('split-modal').classList.remove('show');currentSplitBlock=null}
function doSplit(){
  if(!currentSplitBlock)return;
  let b=blocks[currentSplitBlock]; let k=b.data;
  let dim=document.getElementById('s-dim').value;
  let partsStr=document.getElementById('s-parts').value.trim();
  let parts=partsStr.split(',').map(s=>parseFloat(s.trim())).filter(x=>!isNaN(x));
  if(!dim||!parts.length){updateStatus('切割失败：请选维度并填分段值');return}

  // 目标设备：用画布上所有硬件（后端张量并行需 ≥2 设备）。取硬件块的 backId。
  let hwDevs=[];
  Object.entries(blocks).forEach(([id,b2])=>{
    if(b2.type!=='operator' && b2.backId) hwDevs.push(b2.backId);
  });
  hwDevs=[...new Set(hwDevs)];
  if(hwDevs.length<2){
    closeSplitModal();
    updateStatus('张量并行需至少 2 个硬件设备在画布上。请先添加硬件。');
    return;
  }

  let model=document.getElementById('sel-model').value;
  fetch('/api/split',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model:model, kernel:k.id, dim:dim, parts:parts})})
  .then(r=>r.json()).then(d=>{
    if(d.error){updateStatus('切割失败: '+d.error);return}
    closeSplitModal();
    // 记录张量并行切片规则 → 运行时后端真实执行
    pendingSplits = pendingSplits.filter(s=>!(s.op===k.id && s.dim===dim));
    pendingSplits.push({op: k.id, dim: dim, parts: parts.length, devices: hwDevs});
    // 在原算子块上打"已切"标记，不生成假的 [1] 子算子块
    if(!b.splitBadge){
      let badge=document.createElement('span');
      badge.className='split-badge';
      badge.style.cssText='display:inline-block;margin-left:6px;padding:0 6px;border-radius:3px;background:#5a4630;color:#e6c9a8;font-size:10px';
      badge.textContent='';
      badge.setAttribute('data-dims', dim);
      b.el.querySelector('.op-row b')?.after(badge);
      b.splitBadge=badge;
    }
    b.splitBadge.textContent='⚡已张量并行('+dim+'×'+parts.length+'片)';
    b.splitBadge.style.background='#3a5040';
    b.splitBadge.style.color='#98c379';
    // 若该算子已拖入某硬件，仍保留（子切片由后端分派到各设备）
    updateStatus('算子 '+k.id+' 已配置沿 '+dim+' 切成 '+parts.length+' 片张量并行（运行时生效）');
  }).catch(e=>updateStatus('切割失败: '+e));
}


// ─── Init ───
addPresetHW('GPU');
setTimeout(()=>{addPresetHW('DRAM_PIM')},200);
setTimeout(()=>{addPresetHW('SRAM_PIM')},300);   // IC 参考：LN/Softmax/激活 → SRAM-PIM
setTimeout(()=>loadModel(),500);
setTimeout(()=>loadExperiments(),100);
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("\n  LLM-PIMSim GUI — Visual Topology Editor")
    print("  Open: http://127.0.0.1:5000\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
