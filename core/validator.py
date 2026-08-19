"""
LLM-PIMSim v3 core.validator — 【校验系统】
=================================
以 LLM 推理 + 集成电路体系结构专家视角，对用户在 GUI 中搭建的
"硬件拓扑 + 算子映射 + 数据连线" 配置做合法性校验。

任何 ERROR 级别的问题都会阻止仿真运行；WARNING 只提示、不阻止。

约束分类（与设计文档对应）：
  A. 算子-硬件兼容   —— 算子能否放在该硬件上执行
  B. 数据流完整性    —— 输入/输出/中间值是否都正确连接
  C. 存储容量        —— 数据是否放得下
  D. 数据可达性      —— 数据能否从源硬件搬运到执行硬件
  E. 全局有效性      —— 配置整体是否自洽
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import re

# 精度等级统一口径：与 core.precision.PrecisionLevel 的枚举值一致。
from core.precision import PrecisionLevel as _PL, HARDWARE_CAPABILITY
PRECISION_RANK = {
    "FP32": _PL.FP32.value, "BF16": _PL.BF16.value,
    "FP16": _PL.FP16.value, "FP8": _PL.FP8.value,
    "INT8": _PL.INT8.value, "INT4": _PL.INT4.value,
}
RANK_NAME = {p.value: p.name for p in _PL}
# 明确等级范围，供校验提示文案使用
_RANK_LABEL = {v: name for name, v in PRECISION_RANK.items()}

# op_type → 算子类别（用于 A3 与精度系统的类别能力匹配）。未知 op_type 默认 LINEAR。
_NONLINEAR_OP_TYPES = {"LAYERNORM", "SOFTMAX", "ACTIVATION", "RESIDUAL", "ROPE"}


def _op_category(op_type: str):
    """按 op_type 字符串返回 LINEAR/NONLINEAR（与 core.precision.OperatorCategory 对应值）。"""
    from core.common import OperatorCategory
    return (OperatorCategory.NONLINEAR
            if (op_type or "").strip().upper() in _NONLINEAR_OP_TYPES
            else OperatorCategory.LINEAR)


# ---------------------------------------------------------------- 工具函数
def _parse_unit(s: Any) -> float:
    """把 '512 GB' / '300 TFLOPS' / '2000 GB/s' / 数字 解析成基础单位数值。"""
    if s is None:
        return 0.0
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip().upper().replace(" ", "")
    num = ""
    for ch in t:
        if ch.isdigit() or ch in ".+-eE":
            num += ch
        else:
            break
    try:
        v = float(num)
    except ValueError:
        return 0.0
    for unit, m in [("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3),
                    ("TFLOPS", 1e12), ("TOPS", 1e12), ("GFLOPS", 1e9),
                    ("FLOPS", 1.0)]:
        if unit in t:
            return v * m
    return v


def _precision_rank(precision: Any) -> int:
    """算子要求的精度等级 → 数值等级（FP32=6 … INT4=1，与 contracts.PrecisionLevel 对齐），未知返回 0。"""
    if precision is None:
        return int(_PL.FP16)  # 默认 FP16
    if isinstance(precision, (int, float)):
        return int(precision)
    return PRECISION_RANK.get(str(precision).strip().upper(), 0)


def _parse_port(port_id: str):
    """解析端口 id：'gpu_r'→('hw_read','gpu',None)；'op3_in0'→('op_in','op3',0)；'op3_mid'→('op_mid','op3',None)。"""
    p = str(port_id)
    if p.endswith("_mid"):
        return ("op_mid", p[:-4], None)
    if p.endswith("_r"):
        return ("hw_read", p[:-2], None)
    if p.endswith("_w"):
        return ("hw_write", p[:-2], None)
    if "_in" in p:
        base, idx = p.rsplit("_in", 1)
        try:
            return ("op_in", base, int(idx))
        except ValueError:
            pass
    if "_out" in p:
        base, idx = p.rsplit("_out", 1)
        try:
            return ("op_out", base, int(idx))
        except ValueError:
            pass
    return ("unknown", p, None)


# ---------------------------------------------------------------- 结果结构
@dataclass
class Issue:
    code: str
    level: str  # "error" | "warning"
    message: str
    targets: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    valid: bool
    issues: List[Issue] = field(default_factory=list)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                {"code": i.code, "level": i.level, "message": i.message, "targets": i.targets}
                for i in self.issues
            ],
        }


# ---------------------------------------------------------------- 校验器
class ConstraintChecker:
    """对一整套 GUI 配置做静态合法性校验。输入全部为 dict（与前端 JSON 对齐）。"""

    def __init__(self, hardware: list = None, operators: list = None,
                 connections: list = None, compute_map: dict = None,
                 weight_blocks: list = None, link_table: dict = None):
        self.hw_raw = hardware or []
        self.ops_raw = operators or []
        self.conns = connections or []
        self.compute_map = compute_map or {}   # {kernelId: 硬件块 id}
        self.weight_raw = weight_blocks or []  # [{weight_id, weight_class, partitions:[{partition_id}], consumers, input_slots}]
        self.link_table = link_table or None   # 前端链路系统 N×N 带宽表 {kind: {kind: gbs}}
        self.issues: List[Issue] = []

        # 规范化后的内部结构
        self.hw: Dict[str, dict] = {}
        self.ops: Dict[str, dict] = {}
        self.op_mapping: Dict[str, str] = {}   # op 块 id -> 硬件块 id
        self.conn_by_port: Dict[str, list] = {}  # port -> [conn]
        self.link_pairs: List[tuple] = []        # (hwA, hwB) 互连
        self.weights: Dict[str, dict] = {}       # weight_id -> normalized
        self.weight_by_consumer: Dict[str, list] = {}  # kernelId -> [weight_id]

    # ------------------------------------------------------------ 预处理
    def _normalize(self):
        # 硬件
        for h in self.hw_raw:
            hw_id = h.get("id", "")
            if not hw_id:
                continue
            self.hw[hw_id] = {
                "id": hw_id,
                "type": h.get("type", ""),
                "linkType": h.get("linkType") or h.get("type", ""),
                "backId": h.get("backId", hw_id),
                "capacity_bytes": _parse_unit(h.get("mem") or h.get("capacity")),
                "peak_ops": _parse_unit(h.get("compute") or h.get("peak")),
                "precision": h.get("precision", "FP32/FP16/INT8/INT4"),
                "read_bw": _parse_unit(h.get("rBW")),
                "write_bw": _parse_unit(h.get("wBW")),
            }
        # 算子
        for o in self.ops_raw:
            op_id = o.get("id", "")
            if not op_id:
                continue
            ins = o.get("inputs") or []
            outs = o.get("outputs") or []
            mids = o.get("intermediates") or []
            self.ops[op_id] = {
                "id": op_id,
                "kernelId": o.get("kernelId", op_id),
                "op_type": o.get("op_type", ""),
                "precision": o.get("precision", "FP16"),
                "inputs": ins,
                "outputs": outs,
                "intermediates": mids,
                "n_items": max(1, len(ins) + len(outs) + len(mids)),
                "mem_max": float(o.get("memory_bytes_max") or 0.0),
                "is_kv": bool(o.get("is_kv_dependent")),
            }
            # memory_bytes_range → 取 max
            rng = o.get("memory_bytes_range")
            if rng and isinstance(rng, (list, tuple)) and len(rng) == 2:
                try:
                    self.ops[op_id]["mem_max"] = max(float(rng[0]), float(rng[1]))
                except (TypeError, ValueError):
                    pass
        # 映射：compute_map 键是 kernelId，映射到算子
        for o in self.ops.values():
            target = self.compute_map.get(o["kernelId"])
            if target:
                self.op_mapping[o["id"]] = str(target)
        # 连线索引
        for c in self.conns:
            frm, to = str(c.get("from", "")), str(c.get("to", ""))
            self.conn_by_port.setdefault(frm, []).append(c)
            self.conn_by_port.setdefault(to, []).append(c)
            if c.get("isLink"):
                kindA, baseA, _ = _parse_port(frm)
                kindB, baseB, _ = _parse_port(to)
                if kindA == "hw_read" and kindB == "hw_write":
                    self.link_pairs.append((baseA, baseB))
        # 权重块：把每个 WeightBlock 规范化；同时记录 算子(kernelId) -> 所需权重 及 该权重在算子里的输入序
        self.weight_ids = set()   # 所有权重块 id + 分片 id（供端口存在性校验）
        for w in self.weight_raw:
            wid = w.get("weight_id") or ""
            if not wid:
                continue
            parts_raw = w.get("partitions") or []
            parts = [p.get("partition_id") or p.get("id") or f"{wid}" for p in parts_raw]
            # 记录每片所在设备（供 W1 判定覆盖）
            parts_device = {}
            for p in parts_raw:
                pid = p.get("partition_id") or p.get("id") or wid
                pdev = p.get("device") or ""
                if pdev:
                    parts_device[pid] = pdev
            if not parts:
                parts = [wid]  # 未切割：整块视为一个"逻辑分片"=自身
            self.weight_ids.add(wid)
            for pid in parts:
                self.weight_ids.add(pid)
            consumers = w.get("consumers") or []
            input_slots = w.get("input_slots") or {}
            # 权重字节（容量校验用）：整块 bytes + 每片 bytes；num_layers 用于放大到全模型
            wbytes = int(w.get("bytes") or 0)
            part_bytes = {}
            for p in parts_raw:
                pid = p.get("partition_id") or p.get("id") or wid
                try:
                    part_bytes[pid] = int(p.get("bytes") or 0)
                except (TypeError, ValueError):
                    part_bytes[pid] = 0
            num_layers = int(w.get("num_layers") or 1) or 1
            self.weights[wid] = {
                "weight_id": wid, "weight_class": w.get("weight_class", ""),
                "partitions": parts, "partition_ids": set(parts),
                "consumers": consumers, "input_slots": input_slots,
                "device": w.get("device") or "", "parts_device": parts_device,
                "bytes": wbytes, "part_bytes": part_bytes, "num_layers": num_layers,
            }
            for kid in consumers:
                self.weight_by_consumer.setdefault(kid, []).append(wid)


    def _has_link(self, a: str, b: str) -> bool:
        if a == b:
            return True
        # v3.2 链路系统：前端传了 N×N 带宽表（含缺省带宽）→ 任意两设备种类间可达。
        if self.link_table:
            return True
        return (a, b) in self.link_pairs or (b, a) in self.link_pairs

    def _add(self, code, level, message, targets=None):
        self.issues.append(Issue(code=code, level=level, message=message, targets=targets or []))

    # ------------------------------------------------------------ 规则实现
    def _rule_E_global(self):
        if not self.hw:
            self._add("E1", "error", "画布上没有硬件：至少需要添加一个硬件设备。")
        if not self.ops:
            self._add("E2", "error", "画布上没有算子：请先加载模型。")
        seen = {}
        for hid in self.hw:
            seen[hid] = seen.get(hid, 0) + 1
        for hid, n in seen.items():
            if n > 1:
                self._add("E3", "error", f"硬件 id '{hid}' 重复，必须唯一。", [hid])

    def _rule_A_mapping(self):
        # A1 每个算子必须映射到硬件
        for o in self.ops.values():
            if o["id"] not in self.op_mapping:
                self._add("A1", "error",
                          f"算子 {o['kernelId']} 没有映射到任何硬件（请把算子拖入硬件方块）。",
                          [o["id"]])
            else:
                hw_id = self.op_mapping[o["id"]]
                if hw_id not in self.hw:
                    self._add("A2", "error",
                              f"算子 {o['kernelId']} 映射到不存在的硬件 '{hw_id}'。",
                              [o["id"], hw_id])
                else:
                    self._rule_A_compat(o, self.hw[hw_id])

    def _rule_A_compat(self, o: dict, hw: dict):
        # A4 硬件必须有计算能力 —— 放在最前面，避免被下面的精度/类别分支提前 return
        # 吞掉（此前对 GPU/CPU/PIM 等已知类型是死代码，算力为 0 的硬件也能通过校验）。
        if hw["peak_ops"] <= 0:
            self._add("A4", "error",
                      f"硬件 {hw['id']} 计算能力为 0，不能执行任何算子（包括 {o['kernelId']}）。",
                      [o["id"], hw["id"]])
            return
        # A3 精度/类别兼容。v3.2：与精度系统(core.precision.HARDWARE_CAPABILITY)严格对齐，
        # 不再用"等级 ≥"近似，而是按硬件类型查能力表，做"类别 + 精度列表包含"的精确判定；
        # 仅当硬件类型不在能力表(自定义类型)时才回退到等级判断。
        # v3.3：区分 data/execution —— 若算子 execution_precision 显式为 None（如 KV_Cache
        # 纯数据访问），则只查硬件的"数据精度"能力；否则查"执行精度"能力。
        cap = HARDWARE_CAPABILITY.get(hw["type"])
        exec_prec = o.get("execution_precision")
        if exec_prec in (None, "", "None"):
            # 前端未传 execution_precision 或显式 None → 需要知道它是"纯数据算子"还是"未知"
            # 用 data_precision（若有）否则用 precision 字段
            data_prec = o.get("data_precision") or o["precision"]
            op_prec = data_prec or "FP16"
            prec_names = [p.strip().upper() for p in str(op_prec).split("/") if p.strip()]
            if not prec_names:
                self._add("E4", "error", f"算子 {o['kernelId']} 精度 '{op_prec}' 非法。", [o["id"]])
                return
        else:
            op_prec = exec_prec or o["precision"] or "FP16"
            prec_names = [p.strip().upper() for p in str(op_prec).split("/") if p.strip()]
            if not prec_names:
                self._add("E4", "error", f"算子 {o['kernelId']} 精度 '{op_prec}' 非法。", [o["id"]])
                return

        if cap is not None:
            cat = _op_category(o["op_type"])
            # 1) 类别能力
            if cat not in cap.get("categories", []):
                self._add("A3", "error",
                          f"算子 {o['kernelId']}({o['op_type']}) 属 {cat.name} 类，"
                          f"硬件 {hw['id']}({hw['type']}) 不执行该类别算子——类别不兼容，无法放置。",
                          [o["id"], hw["id"]])
                return
            # 2) 精度能力：execution 为 None → 查 data；否则查 execution
            cap_precs = cap.get("data", []) if exec_prec in (None, "", "None") else cap.get("execution", [])
            ok_prec = False
            for pn in prec_names:
                try:
                    prec = _PL.from_name(pn)
                except ValueError:
                    continue
                if prec in cap_precs:
                    ok_prec = True
                    break
            if not ok_prec and prec_names:
                kind = "数据" if exec_prec in (None, "", "None") else "执行"
                self._add("A3", "error",
                          f"算子 {o['kernelId']} 要求{kind}精度 {op_prec}，"
                          f"但硬件 {hw['id']}({hw['type']}) 的{kind}精度仅支持 "
                          f"{[p.name for p in cap_precs]}——精度不兼容，无法放置。",
                          [o["id"], hw["id"]])
            return

        # 回退：自定义硬件类型（不在能力表）→ 用画布 precision 字段做等级判断
        hw_max, _ = self._hw_precision(hw)
        op_rank = max(_precision_rank(p) for p in prec_names) if prec_names else 0
        if op_rank == 0:
            self._add("E4", "error", f"算子 {o['kernelId']} 精度 '{op_prec}' 非法。", [o["id"]])
            return
        if hw_max < op_rank:
            self._add("A3", "error",
                      f"算子 {o['kernelId']} 要求 {op_prec}(等级{op_rank})，"
                      f"但硬件 {hw['id']} 最高只支持等级{_rank_name(hw_max)}——精度不兼容，无法执行。",
                      [o["id"], hw["id"]])

    def _hw_precision(self, hw: dict):
        names = []
        for part in str(hw["precision"]).replace("/", ",").split(","):
            p = part.strip().upper()
            if p:
                names.append(p)
        ranks = [PRECISION_RANK.get(n, 0) for n in names]
        return (max(ranks) if ranks else int(_PL.FP32)), names

    def _rule_B_connections(self):
        # 端口存在性检查（C2）：每条连线的两个端口必须真实存在
        for c in self.conns:
            frm, to = str(c.get("from", "")), str(c.get("to", ""))
            if not self._port_exists(frm):
                self._add("C2", "error", f"连线起点 '{frm}' 不存在（可能引用了已删除的端口）。")
            if not self._port_exists(to):
                self._add("C2", "error", f"连线终点 '{to}' 不存在（可能引用了已删除的端口）。")

        # B4 连线方向合法性
        for c in self.conns:
            frm, to = str(c.get("from", "")), str(c.get("to", ""))
            if c.get("isLink"):
                continue  # 链路单独校验（D2/D3）
            kindA, baseA, idxA = _parse_port(frm)
            kindB, baseB, idxB = _parse_port(to)
            valid = (
                (kindA == "hw_read" and kindB == "op_in")
                or (kindA == "op_out" and kindB == "hw_write")
                or (kindA == "op_mid" and kindB == "hw_write")
                or (kindA == "op_out" and kindB == "op_in")
            )
            if not valid:
                self._add("B4", "error",
                          f"连线 {frm} → {to} 方向非法：数据必须从硬件读端口流向算子输入、"
                          f"从算子输出流向硬件写端口（或另一个算子的输入）。",
                          [frm, to])

        # B1 每个算子的每个输入端口必须有连线
        for o in self.ops.values():
            for i in range(len(o["inputs"])):
                port = f"{o['id']}_in{i}"
                if not self.conn_by_port.get(port):
                    self._add("B1", "error",
                              f"算子 {o['kernelId']} 的输入 {i+1}（{o['inputs'][i]}）没有连接任何数据源。",
                              [o["id"]])
            # B2 每个输出端口必须有连线
            for j in range(len(o["outputs"])):
                port = f"{o['id']}_out{j}"
                if not self.conn_by_port.get(port):
                    self._add("B2", "error",
                              f"算子 {o['kernelId']} 的输出 {j+1}（{o['outputs'][j]}）没有连接到任何硬件存储。",
                              [o["id"]])
            # B3 中间值端口必须有连线
            if o["intermediates"]:
                port = f"{o['id']}_mid"
                if not self.conn_by_port.get(port):
                    self._add("B3", "error",
                              f"算子 {o['kernelId']} 产生中间值（{o['intermediates'][0]} 等），"
                              f"中间值端口未连接任何硬件存储。",
                              [o["id"]])

    def _port_exists(self, port: str) -> bool:
        kind, base, idx = _parse_port(port)
        if kind in ("hw_read", "hw_write"):
            # 硬件端口或权重块端口（权重 id / 分片 id 都是合法端口主体）
            return base in self.hw or base in self.weight_ids
        if kind == "op_mid":
            o = self.ops.get(base)
            return bool(o and o["intermediates"])
        if kind == "op_in":
            o = self.ops.get(base)
            return bool(o and 0 <= idx < len(o["inputs"]))
        if kind == "op_out":
            o = self.ops.get(base)
            return bool(o and 0 <= idx < len(o["outputs"]))
        return False

    def _rule_D_reachability(self):
        # D2 链路两端存在
        for c in self.conns:
            if not c.get("isLink"):
                continue
            frm, to = str(c.get("from", "")), str(c.get("to", ""))
            kindA, baseA, _ = _parse_port(frm)
            kindB, baseB, _ = _parse_port(to)
            if kindA != "hw_read" or kindB != "hw_write":
                self._add("B4", "error", f"互连链路 {frm} → {to} 必须连接两个硬件的读写端口。", [frm, to])
                continue
            if baseA not in self.hw:
                self._add("D2", "error", f"链路起点硬件 '{baseA}' 不存在。", [baseA])
            if baseB not in self.hw:
                self._add("D2", "error", f"链路终点硬件 '{baseB}' 不存在。", [baseB])
            # D3 带宽有效
            bw = _parse_unit(c.get("label") or c.get("bandwidth"))
            if bw <= 0:
                self._add("D3", "error", f"链路 {baseA} → {baseB} 带宽必须大于 0。", [baseA, baseB])

        # D1 输入数据可达：输入源硬件与执行硬件之间必须有链路
        for o in self.ops.values():
            hw_exec = self.op_mapping.get(o["id"])
            if not hw_exec or hw_exec not in self.hw:
                continue
            for i in range(len(o["inputs"])):
                port = f"{o['id']}_in{i}"
                for c in self.conn_by_port.get(port, []):
                    frm = str(c.get("from", ""))
                    kind, base, _ = _parse_port(frm)
                    # 权重端口：可达性由 W1 权重完整性规则负责（必须连接全部切分片），
                    # 不参与硬件级链路检查（权重块本身不是硬件设备）
                    if kind == "hw_read" and base in self.weight_ids:
                        continue
                    src_hw = None
                    if kind == "hw_read":
                        src_hw = base
                    elif kind == "op_out":
                        prod = self.op_mapping.get(base)
                        src_hw = prod
                    if src_hw and src_hw != hw_exec and not self._has_link(src_hw, hw_exec):
                        self._add("D1", "error",
                                  f"算子 {o['kernelId']} 在硬件 {hw_exec} 上执行，但输入数据来自 "
                                  f"硬件 {src_hw}，两者之间没有互连链路，数据无法搬运。",
                                  [o["id"], src_hw, hw_exec])

    def _rule_C_capacity(self):
        # C1 写入每个硬件的数据总量 <= 该硬件容量
        charged: Dict[str, float] = {}
        for c in self.conns:
            if c.get("isLink"):
                continue
            frm, to = str(c.get("from", "")), str(c.get("to", ""))
            kindA, baseA, _ = _parse_port(frm)
            kindB, baseB, _ = _parse_port(to)
            # 数据写入硬件的情形：算子输出/中间值 → 硬件写端口
            if kindA in ("op_out", "op_mid") and kindB == "hw_write":
                o = self.ops.get(baseA)
                if o:
                    item_size = o["mem_max"] / o["n_items"]
                    charged[baseB] = charged.get(baseB, 0.0) + item_size
            # 数据存储在硬件的情形：硬件读端口 → 算子输入（该数据驻留在硬件里）
            if kindA == "hw_read" and kindB == "op_in":
                # 权重端口不计入硬件容量（权重字节单独统计，见下方 C1-权重）
                if baseA in self.weight_ids:
                    continue
                o = self.ops.get(baseB)
                if o:
                    # 若该输入数据由画布内某算子产出，其驻留已在“输出写回”时计过费，
                    # 这里不再重复计入源硬件容量（避免同一中间张量沿依赖链被重复计数）。
                    mm = re.match(r"^(.+)_in(\d+)$", str(to))
                    produced = False
                    if mm:
                        didx = int(mm.group(2))
                        dname = o["inputs"][didx] if didx < len(o["inputs"]) else None
                        if dname is not None:
                            produced = any(
                                dname in op2["outputs"] or dname in op2["intermediates"]
                                for op2 in self.ops.values())
                    if not produced:
                        item_size = o["mem_max"] / o["n_items"]
                        charged[baseA] = charged.get(baseA, 0.0) + item_size

        # ── C1 权重容量：每个权重块的字节（× 全模型层数）计入其所在硬件 ──
        layers = 1
        for w in self.weights.values():
            wbytes = w.get("bytes") or 0
            layers = w.get("num_layers") or 1
            parts = w.get("partitions") or []
            # 未切割时 partitions 会被规范化为 [wid]（逻辑分片），此时整块字节计入 device；
            # 只有"真切割"（分片数 >1 且分片有独立字节）才逐片统计。
            real_split = len(parts) > 1 and any(
                (w.get("part_bytes", {}).get(pid) or 0) > 0 for pid in parts)
            if real_split:
                for pid in parts:
                    dev = w.get("parts_device", {}).get(pid) or w.get("device") or ""
                    if not dev:
                        continue
                    pbytes = w.get("part_bytes", {}).get(pid) or 0
                    charged[dev] = charged.get(dev, 0.0) + pbytes * layers
            else:
                dev = w.get("device") or ""
                if dev:
                    charged[dev] = charged.get(dev, 0.0) + wbytes * layers

        for hw_id, total in charged.items():
            cap = self.hw.get(hw_id, {}).get("capacity_bytes", 0.0)
            if cap <= 0:
                self._add("C1", "error", f"硬件 {hw_id} 容量为 0，无法存储任何数据。", [hw_id])
            elif total > cap:
                self._add("C1", "error",
                          f"硬件 {hw_id} 需要存储约 {total/1e6:.1f} MB 数据（含权重×全模型层数），"
                          f"但容量只有 {cap/1e6:.1f} MB——存储空间不足。", [hw_id])

    def _rule_B_cycle(self):
        # B5 算子间依赖图（输出→输入连线）必须无环
        import collections
        adj = collections.defaultdict(list)
        for c in self.conns:
            if c.get("isLink"):
                continue
            kindA, baseA, _ = _parse_port(str(c.get("from", "")))
            kindB, baseB, _ = _parse_port(str(c.get("to", "")))
            if kindA == "op_out" and kindB == "op_in" and baseA in self.ops and baseB in self.ops:
                adj[baseA].append(baseB)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {oid: WHITE for oid in self.ops}
        stack = []
        cycle_found = []

        def dfs(u):
            color[u] = GRAY
            stack.append(u)
            for v in adj.get(u, []):
                if color.get(v, WHITE) == GRAY:
                    cycle_found.append(list(stack[stack.index(v):]) + [v])
                    return
                if color.get(v, WHITE) == WHITE:
                    dfs(v)
                    if cycle_found:
                        return
            stack.pop()
            color[u] = BLACK

        for oid in list(self.ops):
            if color.get(oid, WHITE) == WHITE:
                dfs(oid)
                if cycle_found:
                    break
        if cycle_found:
            path = " → ".join(cycle_found[0])
            self._add("B5", "error", f"算子依赖图存在环：{path}——这些算子互相等待，永远无法调度。")

    # ------------------------------------------------------------ 权重完整性
    def _rule_W_weight_integrity(self):
        """W1 权重完整性（ALL-GATHER 语义）：需要某类权重的算子，必须能读取该类权重的全部切片。
        每个 WeightBlock 都可能被切成 N 片（分片可放不同设备）。算子若声明依赖它，则它对应
        的输入端口必须覆盖该类权重的所有分片位置，否则视为拿不到完整权重、不可运行。

        覆盖判定：算子的权重输入端口上，凡来源能解析到某分片所在设备的读端口（或直接是该分片
        的端口），即视为读到该分片。若权重未切分（单个逻辑分片），则只需连到它所在设备的读端口。
        """
        if not self.weights:
            return
        for o in self.ops.values():
            kid = o.get("kernelId") or o.get("id")
            if not kid:
                continue
            for wid in self.weight_by_consumer.get(kid, []):
                w = self.weights.get(wid)
                if not w:
                    continue
                in_idx = w["input_slots"].get(kid)
                if in_idx is None:
                    in_idx = 0  # 定位不到输入序，按第 0 个输入处理
                port = f"{o['id']}_in{in_idx}"
                # 该算子该输入端口所有连线的来源
                sources = [str(c.get("from", "")) for c in self.conn_by_port.get(port, [])]
                # 分片 -> 所在设备（未切分时 partition_ids=={wid}，device 取 weights 自身 device）
                all_ids = list(w["partition_ids"])
                covered = set()
                for pid in all_ids:
                    # 直接连到该分片端口
                    direct = any(s == pid or s == f"{pid}_r" or s == f"{pid}_w" for s in sources)
                    if direct:
                        covered.add(pid)
                        continue
                    # 通过设备读端口：找到该分片所在设备，看是否有一条来源端口落在这个设备上
                    dev = self._partition_device(w, pid)
                    if dev and dev in self.hw:
                        if any(s == f"{dev}_r" or s == f"{dev}_w" for s in sources):
                            covered.add(pid)
                missing = set(all_ids) - covered
                if missing:
                    label = f"{wid}（{w['weight_class']}）"
                    if len(all_ids) <= 1:
                        self._add("W1", "error",
                                  f"算子 {kid} 需要权重 {label}，但其权重输入端口未连接到该权重的存放设备。",
                                  [o["id"], wid])
                    else:
                        miss = ", ".join(sorted(missing))
                        self._add("W1", "error",
                                  f"算子 {kid} 需要权重 {label} 的完整内容，却只连接了部分切片——"
                                  f"缺少 {miss}。必须连接全部 {len(all_ids)} 个切片（ALL-GATHER）"
                                  f"才能获得完整权重，否则不得运行。",
                                  [o["id"], wid])

    def _partition_device(self, w: dict, partition_id: str) -> str:
        """返回某个权重切片所在的设备。若分片指定了 device 则返回之；
        否则回退到整块权重的 device 字段。"""
        # weight 规范化时把分片 device 存到 w['partitions'] 里的映射；此处从原 weight_raw 派生的
        # simpler：在规范化时已把每片的分片device 存到 w['parts_device']？这里直接尝试取。
        dev = w.get("parts_device", {}).get(partition_id)
        if dev:
            return dev
        return w.get("device") or ""

    # ------------------------------------------------------------ 主入口
    def validate(self) -> ValidationResult:
        self.issues = []
        self._normalize()
        self._rule_E_global()
        self._rule_A_mapping()
        self._rule_B_connections()
        self._rule_D_reachability()
        self._rule_C_capacity()
        self._rule_B_cycle()
        self._rule_W_weight_integrity()
        return ValidationResult(valid=not any(i.level == "error" for i in self.issues),
                                issues=self.issues)


# ---------------------------------------------------------------- 便捷入口
def validate_config(hardware=None, operators=None, connections=None,
                    compute_map=None, weight_blocks=None,
                    link_table=None) -> ValidationResult:
    return ConstraintChecker(hardware, operators, connections,
                             compute_map, weight_blocks, link_table).validate()


# ---------------------------------------------------------------- 运行后完成度校验
def validate_completion(result_or_diagnostics, total_ops=None, finished=None,
                        unfinished=None) -> ValidationResult:
    """校验"完成度是否满员"。若存在未跑完的算子 → error 并列出。

    入参可传：
      - SimulationResult（读 .diagnostics），
      - 或纯 dict（含 total_operators/finished_operators/unfinished_operators），
      - 或直接传 total_ops/finished/unfinished。
    返回 ValidationResult：完成度不足 → valid=False，并给出可读原因。
    """
    from dataclasses import dataclass
    diag = {}
    if isinstance(result_or_diagnostics, dict):
        diag = result_or_diagnostics
    else:
        d = getattr(result_or_diagnostics, "diagnostics", None) or {}
        diag = d if isinstance(d, dict) else {}

    total = total_ops if total_ops is not None else diag.get("total_operators", 0)
    fin = finished if finished is not None else diag.get("finished_operators", 0)
    unfin = unfinished if unfinished is not None else diag.get("unfinished_operators", [])

    issues = []
    if total and fin < total:
        n_miss = total - fin
        message = (f"完成度不足：{fin}/{total} 个算子运行结束，{n_miss} 个算子未完成。"
                   f"所得总延迟不包含这些算子，不能作为有效方案。首个未完成："
                   f"{unfin[0]['op_id'] if unfin else '?'}"
                   f"(type={unfin[0].get('op_type') if unfin else '?'}, "
                   f"device={unfin[0].get('compute_device') if unfin else '?'})。")
        issues.append(Issue(code="F1", level="error", message=message,
                            targets=[u.get("op_id") for u in unfin[:10]]))
    return ValidationResult(valid=not any(i.level == "error" for i in issues),
                            issues=issues)


# 完成度校验的错误码
COMPLETION_CODE = "F1"


def _rank_name(r: int) -> str:
    return RANK_NAME.get(r, f"等级{r}")
