"""
LLM-PIMSim v3 — weights（兼容转发壳）

v3 解耦：权重数据模型（WeightBlock/WeightPartition/归类）已迁至「权重系统」
core.weight_sys；切分算法迁至「切割系统」core.splitter。
本文件转发并保持旧版 `build_weight_blocks(..., class_split=...)` 直接产生分片的行为：
通过在 core.weight_sys.build_weight_blocks 注入 splitter.make_weight_partitions 实现。
"""
from core.weight_sys import (
    WEIGHT_MEMBERSHIP, CLASS_SPLIT_DIM, WEIGHT_PORT_DIRECTION,
    WeightPort, WeightPartition, WeightBlock, build_weight_ports,
    detect_weight_class, consumers_weight_classes,
)
from core.splitter import make_weight_partitions as _make_partitions

# 旧别名：保持 `from weights import _make_partitions` 可用
make_partitions = _make_partitions


def build_weight_blocks(model_name: str, num_layers: int,
                        h: int, f: int, nh: int, v: int,
                        precision_bytes: int = 2,
                        class_split: dict = None) -> dict:
    """同 core.weight_sys.build_weight_blocks，但默认注入切割回调以保持旧行为。"""
    # 延迟 import 避免环
    from core.weight_sys import build_weight_blocks as _bwb
    return _bwb(model_name, num_layers, h, f, nh, v,
                precision_bytes=precision_bytes,
                class_split=class_split,
                make_partitions=_make_partitions)
