"""
LLM-PIMSim v3 — engine（兼容转发壳）

v3 解耦：模拟引擎已并入「核心调度器」 core.engine.SimulationEngine。
本文件仅转发，请勿在此添加新逻辑。
"""
from core.engine import SimulationEngine
