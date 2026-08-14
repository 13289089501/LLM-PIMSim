"""
LLM-PIMSim v3 — contracts（兼容转发壳）

v3 解耦：
  - 纯枚举 / 数据结构 迁至「公共底层」 core.common
  - PrecisionLevel 迁至「精度系统」 core.precision
  - 结果序列化迁至「输出系统」 core.exporter（不再挂 SimulseResult.to_dict）

本文件为保持旧入口兼容而聚合转发，请勿在此添加新逻辑。
如需结果序列化，请用 `from core.exporter import result_to_dict`。
"""
from core.common import (
    OpState, DataType, EventType, DeviceType, BottleneckType,
    DataSourceMode, OperatorCategory,
    DataObject, InputSpec, OperatorSpec, Operator, DurationBreakdown,
    Event, OperatorTiming, SimulationResult,
)
from core.precision import PrecisionLevel
