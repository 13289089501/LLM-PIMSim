"""
LLM-PIMSim v2 — WeightBlock 权重数据模型
=================================================================
背景：当前实现把权重当作算子的"输入字符串"，没有独立的权重节点。
      本模块把权重提升为一级数据节点 WeightBlock：

        - 权重按【张量形状】归类（不是按算子），供用户决定放哪个硬件；
        - 每类权重是"可切原子单元"，可切成分片放不同设备；
        - 切割后，需要该类的算子必须连接其全部切分片才允许运行（ALL-GATHER 语义）。

归类主基调（按形状，见模块头注释）：
    W_attn  q/k/v/o 四投影        各 [H,H]    · 沿 head 列可切
    W_mlp   ffn_gate/up/down      各 [H,·]    · 沿中间维可切
    W_ln    ln1/ln2 权重          各 [H]      · 小向量，通常不切
    W_head  lm_head（全局）        [V,H]      · 沿行(词表)可切
    W_embed embed_weight（全局）    [V,H]      · 视同嵌入，可归 W_head 或独立
=================================================================
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------- 常量
# 权重后缀 → 所属类别
WEIGHT_MEMBERSHIP = {
    "ln1_w":   "W_ln",
    "ln2_w":   "W_ln",
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

# 每个类的"可切维"建议：rows(行) / cols(列)
CLASS_SPLIT_DIM = {
    "W_attn": "cols",   # 沿 head 列
    "W_mlp":  "cols",   # 沿中间维
    "W_ln":   None,     # 小向量，不建议切
    "W_head": "rows",
    "W_embed": "rows",
}


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
    consumers: List[str] = field(default_factory=list)   # kernelId 列表（需要它的算子）
    input_slots: Dict[str, int] = field(default_factory=dict)  # {kernelId: inputIdx}
    split_dim: Optional[str] = None      # rows/cols/None
    partitions: List[WeightPartition] = field(default_factory=list)  # 切分片
    device: str = ""                     # 未切时所在设备


def detect_weight_class(weight_name: str) -> Optional[str]:
    """根据权重命名识别其类别；非权重返回 None。"""
    if weight_name is None:
        return None
    name = weight_name
    if name in ("lm_head_w",):
        return "W_head"
    if name in ("embed_weight", "embed_w", "input_embed"):
        return "W_embed"
    # 去掉层前缀得到权重尾名，例: L0_q_w -> q_w
    tail = name.split("_", 1)[-1]
    # 处理 L{n}_ 前缀已剥；直接匹配尾名
    for key, cls in WEIGHT_MEMBERSHIP.items():
        if name.endswith(key) or tail == key:
            return cls
    return None


def model_config_to_shapes(model_cfg: dict) -> dict:
    """把 model_lib 返回的 config 转成本模块需要的形状字典。
    model_cfg 需含 hidden_size/ffn_size/num_heads/head_dim/vocab_size/num_layers/precision。"""
    h = model_cfg.get("hidden_size", 4096)
    f = model_cfg.get("ffn_size", 11008)
    nh = model_cfg.get("num_heads", 32)
    v = model_cfg.get("vocab_size", 32000)
    return {"hidden": h, "ffn": f, "heads": nh, "vocab": v}


def build_weight_blocks(model_name: str, num_layers: int,
                        h: int, f: int, nh: int, v: int,
                        precision_bytes: int = 2,
                        class_split: Optional[Dict[str, int]] = None) -> Dict[str, WeightBlock]:
    """为给定模型生成完整的 WeightBlock 字典 {weight_id: WeightBlock}。
    class_split: {类名: 切分数}。例如 {'W_mlp': 2} 表示把每层 W_mlp 切 2 片。"""
    class_split = class_split or {}
    blocks: Dict[str, WeightBlock] = {}

    def add(weight_id, weight_class, rows, cols, layer, consumers, input_slots):
        wb = WeightBlock(
            weight_id=weight_id, weight_class=weight_class, layer=layer,
            rows=rows, cols=cols, bytes=rows * cols * precision_bytes,
            consumers=consumers, input_slots=input_slots,
            split_dim=CLASS_SPLIT_DIM.get(weight_class),
        )
        parts = class_split.get(weight_class)
        if parts and parts > 1:
            _make_partitions(wb, parts)
        blocks[weight_id] = wb
        return wb

    # 全局 embeddings / lm_head（每模型一份）
    if "W_embed" in (class_split or {}):
        add("embed_weight", "W_embed", v, h, None,
            consumers=["embedding"], input_slots={"embedding": 0})
    if "W_head" in (class_split or {}):
        add("lm_head_w", "W_head", v, h, None,
            consumers=["lm_head"], input_slots={"lm_head": 0})

    # 每层
    for L in range(num_layers):
        pre = f"L{L}_"

        add(f"{pre}q_w", "W_attn", h, h, L, [pre + "q_proj"], {pre + "q_proj": 1})
        add(f"{pre}k_w", "W_attn", h, h, L, [pre + "k_proj"], {pre + "k_proj": 1})
        add(f"{pre}v_w", "W_attn", h, h, L, [pre + "v_proj"], {pre + "v_proj": 1})
        add(f"{pre}o_w", "W_attn", h, h, L, [pre + "o_proj"], {pre + "o_proj": 1})
        add(f"{pre}ffn_gw", "W_mlp", f, h, L, [pre + "ffn_gate"], {pre + "ffn_gate": 1})
        add(f"{pre}ffn_uw", "W_mlp", f, h, L, [pre + "ffn_up"], {pre + "ffn_up": 1})
        # 注意：workload 里 ffn_down 的输入是 [ffn_up, ffn_silu]，不消费独立权重矩阵，
        # 因此这里不生成 ffn_down_w（避免悬空权重块 / W1 误报）。
        add(f"{pre}ln1_w", "W_ln", h, 1, L, [pre + "ln1"], {pre + "ln1": 1})
        add(f"{pre}ln2_w", "W_ln", h, 1, L, [pre + "ln2"], {pre + "ln2": 1})
    return blocks


def _make_partitions(wb: WeightBlock, n: int, devices: Optional[List[str]] = None):
    """按 wb.split_dim 把权重切成 n 片，生成分量放到 partitions。
    devices: 可选，长度应为 n，给每片指定所在设备（未给则留空）。"""
    if n < 2:
        return
    devices = devices or []
    dim = wb.split_dim
    if dim == "rows":
        elem_bytes = wb.bytes // max(1, wb.rows * wb.cols)
        total = wb.rows
        base = total // n
        rem = total % n
        for i in range(n):
            r = base + (1 if i < rem else 0)
            wb.partitions.append(WeightPartition(
                partition_id=f"{wb.weight_id}.p{i}", weight_class=wb.weight_class,
                index=i, rows=r, cols=wb.cols,
                bytes=r * wb.cols * elem_bytes,
                device=devices[i] if i < len(devices) else ""))
    elif dim == "cols":
        elem_bytes = wb.bytes // max(1, wb.rows * wb.cols)
        total = wb.cols
        base = total // n
        rem = total % n
        for i in range(n):
            c = base + (1 if i < rem else 0)
            wb.partitions.append(WeightPartition(
                partition_id=f"{wb.weight_id}.p{i}", weight_class=wb.weight_class,
                index=i, rows=wb.rows, cols=c,
                bytes=wb.rows * c * elem_bytes,
                device=devices[i] if i < len(devices) else ""))
    # dim==None 或其它：不分片保持为空


def consumers_weight_classes(blocks: Dict[str, WeightBlock],
                             kernel_id: str) -> List[str]:
    """返回某算子需要到的所有权重类。"""
    return [b.weight_class for b in blocks.values() if kernel_id in b.consumers]
