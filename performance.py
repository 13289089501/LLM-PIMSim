"""
LLM-PIMSim v3 — performance（兼容转发壳）

v3 解耦：性能估算模型已并入「核心调度器」 core.engine.PerformanceModel。
本文件仅转发，请勿在此添加新逻辑。
"""
from core.engine import PerformanceModel
