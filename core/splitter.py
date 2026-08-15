"""
LLM-PIMSim v3 core.splitter — 【切割系统】

职责：把"计算负载"沿某一维度切开，产出多段子负载。统一处理两类对象：
  1. 算子切割（split_kernel）—— 沿 attributes 里的 M/K/N/seq 等维度把一个 kernel
     切成多段，计算量/存储按该维度比例缩放，精度/类型/输入输出结构保持不变。
  2. 权重切割（split_weight_block）—— 沿 WeightBlock 的 rows/cols 切分为 N 片
     WeightPartition，供 ALL-GATHER（权重完整性）使用。

依赖：core.common（数据结构）+ core.operator_sys（Kernel）+ core.weight_sys（WeightBlock）。
     （这里的系统间依赖只发生在"读数据模型"层，不反向依赖，方向一致。）
"""

import re as _re
from typing import Optional, Mapping

from core.operator_sys import Kernel
from core.weight_sys import WeightBlock, WeightPartition


# =================================================================
# 0. 统一形状 Shape（v3.1 新增，供切割/校验/显示共用同一份"形状真相"）
# =================================================================
# 从算子 id 提取层号：L{n}_xxx ；无前缀视为全局（None）
_LAYER_RE = _re.compile(r"^L(\d+)_")


class Shape:
    """算子的统一形状描述。

    语义：
      - dims        : {维度名: 尺寸}，如 {"M":1, "K":4096, "N":11008} 或 {"seq":2048}
      - split_dims  : 允许沿哪些维度切割（数值型 M/K/N 与字符串 seq/kv_len 均可切）
      - layer       : 层号（从 id "L{n}_" 提取）；None = 全局算子（embedding/lm_head）
    """

    def __init__(self, dims: dict = None, split_dims: list = None, layer: Optional[int] = None):
        self.dims = dict(dims or {})
        self.split_dims = list(split_dims or [])
        self.layer = layer

    def has(self, dim: str) -> bool:
        return dim in self.dims or dim in self.split_dims

    def get(self, dim: str):
        """取某维尺寸；返回 None 表示该维是占位/不可数为值（如 kv_len）。"""
        return self.dims.get(dim)

    def to_dict(self) -> dict:
        return {"dims": dict(self.dims), "split_dims": list(self.split_dims), "layer": self.layer}

    def __repr__(self):
        return f"Shape({self.to_dict()})"


def extract_layer(op_id: str) -> Optional[int]:
    """从算子 id 提取层号；无 L{n}_ 前缀返回 None（全局算子）。"""
    m = _LAYER_RE.match(str(op_id))
    return int(m.group(1)) if m else None


def kernel_shape(kernel) -> Shape:
    """把"kernel dict / 带 attributes 的对象 / 纯 attributes dict"解析成 Shape。

    - 从 id 提层号；
    - 将 attributes 里的数值型维度收进 dims（跳过 "kv_len" 等字符串占位）；
    - "kv_len"/"seq" 这类运行期/动态维也登记为可切维（split_dims）。
    """
    # 归一化取 attributes 与 id
    attrs, op_id = {}, ""
    if isinstance(kernel, Mapping):
        attrs = dict(kernel.get("attributes") or {})
        op_id = str(kernel.get("id", ""))
    elif hasattr(kernel, "attributes"):
        attrs = dict(getattr(kernel, "attributes") or {})
        op_id = str(getattr(kernel, "id", ""))
    else:
        attrs = dict(kernel or {})

    dims, split_dims = {}, []
    for k, v in attrs.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            key = str(k)
            dims[key] = float(v)
            split_dims.append(key)
        elif isinstance(v, str):
            # 字符串值（如 "kv_len"）表示"运行期/动态占位"，登记为可切维但无数值
            split_dims.append(str(k))
    # 幂等去重保序
    seen = set()
    split_dims = [d for d in split_dims if not (d in seen or seen.add(d))]
    return Shape(dims=dims, split_dims=split_dims, layer=extract_layer(op_id))


def split_dim_ratio(shape: Shape, dim: str, parts: list) -> list:
    """计算沿某维切成 parts 时，各段的缩放比例。返回与 parts 等长的比例列表。

    - 数值维度：ratio_i = parts[i] / sum(parts)
    - 动态/占位维度（如 kv_len，无数值）：按 parts 本身视为绝对参考，ratio_i = parts[i]/sum(parts)
    """
    total = sum(float(p) for p in parts)
    return [float(p) / total if total else 1.0 for p in parts]


# =================================================================
# 1. 算子切割
# =================================================================
def split_kernel_dict(kernel: dict, dim: str, parts: list) -> list:
    """把单个序列化 kernel dict 沿 dim 切成 parts 指定的若干段。

    dim 必须是算子的可切维度（attributes 里的数值型 M/K/N/seq，或运行时占位 kv_len）；
    parts 为每段的新值列表，如 [5,15] 表示切成 N=5 与 N=15 两段。
    返回: 新的 kernel dict 列表（每段一个），id 加段序号后缀。
    """
    # 用统一形状校验 dim 是否可切（补全可切维列表，报错更友好）
    shape = kernel_shape(kernel)
    if dim not in shape.split_dims:
        raise ValueError(f"维度 '{dim}' 不可切割，可切割维度: {shape.split_dims}")
    attrs = dict(kernel.get("attributes", {}))
    original = attrs.get(dim)
    if isinstance(original, str):
        if original.lower() == "kv_len":
            ref_total = sum(float(p) for p in parts)
            original_numeric = ref_total
            ratio_base = ref_total
        else:
            raise ValueError(f"维度 '{dim}' 的原始值 '{original}' 无法切割")
    else:
        original_numeric = float(original)
        ratio_base = original_numeric

    total_parts = sum(float(p) for p in parts)
    result = []
    base_flops = kernel.get("compute_flops_range", [0, 0])
    base_mem = kernel.get("memory_bytes_range", [0, 0])
    base_flops_lo, base_flops_hi = float(base_flops[0]), float(base_flops[1])
    base_mem_lo, base_mem_hi = float(base_mem[0]), float(base_mem[1])

    for i, p in enumerate(parts):
        ratio = float(p) / ratio_base if ratio_base and total_parts else 1.0
        new_attrs = dict(attrs)
        new_attrs[dim] = original if isinstance(original, str) else float(p)
        new_flops = base_flops_lo * ratio, base_flops_hi * ratio
        new_mem = base_mem_lo * ratio, base_mem_hi * ratio
        new_kernel = dict(kernel)
        new_kernel["id"] = f"{kernel['id']}[{i+1}]"
        new_kernel["name"] = f"{kernel.get('name', kernel['id'])}#{i+1}"
        new_kernel["attributes"] = new_attrs
        new_kernel["compute_flops_range"] = [new_flops[0], new_flops[1]]
        new_kernel["memory_bytes_range"] = [new_mem[0], new_mem[1]]

        def _gstr(a, b):
            if a == b:
                return f"{a/1e9:.3f}G"
            return f"[{a/1e9:.3f}, {b/1e9:.3f}]G"

        def _mstr(a, b):
            if a == b:
                return f"{a/1e6:.3f}MB"
            return f"[{a/1e6:.3f}, {b/1e6:.3f}]MB"

        new_kernel["compute_gflops"] = _gstr(new_flops[0], new_flops[1])
        new_kernel["memory"] = _mstr(new_mem[0], new_mem[1])
        result.append(new_kernel)
    return result


# =================================================================
# 2. 权重切割
# =================================================================
def make_weight_partitions(wb: WeightBlock, n: int, devices: Optional[list] = None):
    """按 wb.split_dim 把权重切成 n 片，生成分量放到 wb.partitions。

    devices: 可选，长度应为 n，给每片指定所在设备（未给则留空）。
    split_dim 为 None 或其它值时不分片（保持为空）。
    """
    if n < 2:
        return
    devices = devices or []
    dim = wb.split_dim
    elem_bytes = wb.bytes // max(1, wb.rows * wb.cols)
    if dim == "rows":
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
    # dim==None 或其它: 不分片保持为空


# 便捷：给定切分数构造可传给 build_weight_blocks 的回调
def weight_partitioner():
    """返回一个 make_partitions 回调，供 core.weight_sys.build_weight_blocks 使用。"""
    return make_weight_partitions
