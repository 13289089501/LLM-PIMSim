"""
LLM-PIMSim v3 — workload_adapter（兼容转发壳）

v3 解耦：WorkloadAdapter（kernel 粒度 → 可执行算子图）已迁至「算子系统」
core.operator_sys。本文件仅转发，请勿在此添加新逻辑。
"""
from core.operator_sys import WorkloadAdapter

# 兼容旧名的 op_type 映射（若外部使用）
from core.operator_sys import _KERNEL_OP_TYPE as KERNEL_OP_TYPE
