"""
LLM-PIMSim v3 core.weight_sys — 【权重系统】

职责：
  1. WeightBlock / WeightPartition —— 权重的数据模型（按张量形状归类为一级节点）
  2. 权重归类表（WEIGHT_MEMBERSHIP / CLASS_SPLIT_DIM）
  3. build_weight_blocks —— 为给定模型维度生成完整权重块集合
  4. 按形状计算字节（bytes），以及"某类是否可切 / 可切维度建议"

=> 切割算法**不在此系统**。本系统只负责"建块、归类、按形状算体积、给出可切维建议"；
  实际把一块切成 N 片的几何算法位于【切割系统 core.splitter】。
  为避免权重系统反向依赖切割系统，切割由调用方注入的 `make_partitions` 回调完成
  （默认不切）。依赖方向：调用方 → 切割系统 → 权重系统。

依赖：仅 stdlib + core.common（可选）。不依赖算子系统/调度器。
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------- 常量
# 权重后缀 → 所属类别（与新算子 DAG 匹配）
WEIGHT_MEMBERSHIP = {
    "qkv_weight":    "W_attn",
    "o_proj_weight": "W_attn",
    "gate_weight":   "W_mlp",
    "up_weight":     "W_mlp",
    "down_weight":   "W_mlp",
    "q_w":     "W_attn",
    "k_w":     "W_attn",
    "v_w":     "W_attn",
    "o_w":     "W_attn",
    "ffn_gw":  "W_mlp",
    "ffn_uw":  "W_mlp",
    "ffn_down_w": "W_mlp",
    "ffn1_w":  "W_mlp",
    "ffn2_w":  "W_mlp",
}

# 每个类的"可切维"建议：rows(行) / cols(列)；None = 不建议切
CLASS_SPLIT_DIM = {
    "W_attn": "cols",   # 沿 head 列
    "W_mlp":  "cols",   # 沿中间维
    "W_ln":   None,     # 小向量，不建议切
    "W_head": "rows",
    "W_embed": "rows",
}


# ---------------------------------------------------------------- 数据模型
# 权重端口方向：权重是静态数据源，总是"输出(提供数据)"给消费它的算子。
WEIGHT_PORT_DIRECTION = "out"


@dataclass
class WeightPort:
    """权重的一个结构化端口 —— 表达"该权重把自身数据提供给某算子的某个输入"。

    用于统一"端口"概念（与算子系统后续统一），替代松散的 input_slots 字典。
    """
    weight_id: str             # 所属权重块 id
    direction: str = WEIGHT_PORT_DIRECTION   # 固定 "out"（权重作为数据源输出）
    target_op: str = ""        # 消费该权重的算子 kernel_id，如 L0_qkv_proj
    input_slot: int = 0        # 目标算子的第几个输入（0 起）
    shape: tuple = ()          # (rows, cols)，与权重块形状一致
    data_type: str = "WEIGHT"


@dataclass
class WeightPartition:
    """权重类的一个切分片。"""
    partition_id: str          # 如 W_mlp.L0.p0
    weight_class: str          # 所属类
    index: int                 # 第几片
    rows: int
    cols: int
    bytes: int = 0
    device: str = ""           # 该分片所在设备（由 placement 决定）；未放置时留空


@dataclass
class WeightBlock:
    """一类权重的具体实例（某层的某类权重）。"""
    weight_id: str             # 如 W_mlp.L0
    weight_class: str          # W_mlp / W_attn / W_ln / W_head
    layer: Optional[int]       # None=全局
    rows: int
    cols: int
    bytes: int = 0
    consumers: List[str] = field(default_factory=list)     # 兼容字段（由 ports 派生）
    input_slots: Dict[str, int] = field(default_factory=dict)  # 兼容字段（由 ports 派生）
    split_dim: Optional[str] = None      # rows/cols/None
    partitions: List[WeightPartition] = field(default_factory=list)
    device: str = ""                     # 未切时所在设备
    ports: List[WeightPort] = field(default_factory=list)  # 结构化端口（v3.1 新增）

    # ---- 端口辅助 ----
    def port_for(self, op_kernel_id: str) -> Optional["WeightPort"]:
        """返回指向某算子的端口；未找到返回 None。"""
        for p in self.ports:
            if p.target_op == op_kernel_id:
                return p
        return None

    def to_port_dict(self) -> list:
        """把端口转成可序列化 dict 列表，供输出/展示使用。"""
        return [{"weight_id": p.weight_id, "direction": p.direction,
                 "target_op": p.target_op, "input_slot": p.input_slot,
                 "shape": list(p.shape), "data_type": p.data_type}
                for p in self.ports]


# ---------------------------------------------------------------- 端口构建
def build_weight_ports(weight_id: str, weight_class: str, rows: int, cols: int,
                       consumers: List[str], input_slots: Dict[str, int]) -> List[WeightPort]:
    """按"消费算子列表 + 输入槽位"生成一个权重块的全部结构化端口。

    每个消费该权重块的算子对应一个端口（方向=out，目标是该算子，槽位取自 input_slots）。
    input_slots 缺失某项时按 0 兜底（旧数据兼容）。
    """
    ports = []
    for op in consumers:
        slot = input_slots.get(op, 0) if isinstance(input_slots, dict) else 0
        ports.append(WeightPort(
            weight_id=weight_id, direction=WEIGHT_PORT_DIRECTION,
            target_op=op, input_slot=slot, shape=(rows, cols), data_type="WEIGHT"))
    return ports


# ---------------------------------------------------------------- 归类
def detect_weight_class(weight_name: str) -> Optional[str]:
    """根据权重命名识别其类别；非权重返回 None。"""
    if weight_name is None:
        return None
    if weight_name in ("lm_head_w", "lm_head_weight"):
        return "W_head"
    if weight_name in ("embed_weight", "embed_w", "input_embed"):
        return "W_embed"
    tail = weight_name.split("_", 1)[-1]
    for key, cls in WEIGHT_MEMBERSHIP.items():
        if weight_name.endswith(key) or tail == key:
            return cls
    return None


# ---------------------------------------------------------------- 构建
def build_weight_blocks(model_name: str, num_layers: int,
                        h: int, f: int, nh: int, v: int,
                        precision_bytes: int = 2,
                        class_split: Optional[Dict[str, int]] = None,
                        make_partitions: Optional[Callable] = None) -> Dict[str, WeightBlock]:
    """为给定模型生成完整的 WeightBlock 字典 {weight_id: WeightBlock}。

    class_split: {类名: 切分数}。例如 {'W_mlp': 2} 表示把每层 W_mlp 切 2 片。
    make_partitions: 切割算法回调（(WeightBlock, n) -> 就地填充 partitions）。
       由【切割系统】提供，经调用方注入；None 表示不切分。
    """
    class_split = class_split or {}
    blocks: Dict[str, WeightBlock] = {}

    def add(weight_id, weight_class, rows, cols, layer, consumers, input_slots):
        wb = WeightBlock(
            weight_id=weight_id, weight_class=weight_class, layer=layer,
            rows=rows, cols=cols, bytes=rows * cols * precision_bytes,
            consumers=consumers, input_slots=input_slots,
            split_dim=CLASS_SPLIT_DIM.get(weight_class),
            ports=build_weight_ports(weight_id, weight_class, rows, cols,
                                     consumers, input_slots),
        )
        parts = class_split.get(weight_class)
        if parts and parts > 1 and make_partitions is not None:
            make_partitions(wb, parts)
        blocks[weight_id] = wb
        return wb

    add("embed_weight", "W_embed", v, h, None,
        consumers=["Embedding"], input_slots={"Embedding": 1})
    add("lm_head_weight", "W_head", v, h, None,
        consumers=["LMHead"], input_slots={"LMHead": 1})

    for L in range(num_layers):
        pre = f"L{L}_"
        add(f"{pre}qkv_weight", "W_attn", 3 * h, h, L, [pre + "qkv_proj"], {pre + "qkv_proj": 1})
        add(f"{pre}o_proj_weight", "W_attn", h, h, L, [pre + "o_proj"], {pre + "o_proj": 1})
        add(f"{pre}gate_weight", "W_mlp", f, h, L, [pre + "ffn_gate"], {pre + "ffn_gate": 1})
        add(f"{pre}up_weight", "W_mlp", f, h, L, [pre + "ffn_up"], {pre + "ffn_up": 1})
        add(f"{pre}down_weight", "W_mlp", h, f, L, [pre + "ffn_down"], {pre + "ffn_down": 1})
    return blocks


# 便捷：返回某算子需要的所有权重类
def consumers_weight_classes(blocks: Dict[str, WeightBlock],
                             kernel_id: str) -> List[str]:
    return [b.weight_class for b in blocks.values() if kernel_id in b.consumers]
