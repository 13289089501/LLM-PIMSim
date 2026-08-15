"""
LLM-PIMSim v3 — scheduler（兼容转发壳）

v3 解耦：离散事件调度核心已并入「核心调度器」 core.engine。
本文件仅转发，请勿在此添加新逻辑。
"""
from core.engine import (
    EventQueue, DataStateTable, ResourceStateTable, Scheduler,
)
