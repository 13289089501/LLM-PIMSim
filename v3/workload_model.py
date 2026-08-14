"""
LLM-PIMSim v3 — workload_model（兼容转发壳）

v3 解耦：kernel 粒度 workload 建模已迁至「算子系统」 core.operator_sys。
本文件仅为保持旧入口兼容而转发，请勿在此添加新逻辑。
新代码请直接 import core.operator_sys。
"""
from core.operator_sys import (
    KernelType, Cost, KVCache, Kernel, PrefillConfig, DecodeConfig,
    LayerGroup, Workload, WorkloadBuilder, build_model_workload,
)
