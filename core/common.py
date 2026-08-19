"""
LLM-PIMSim v3 core.common — 公共底层（非"七大系统"之一）

存放被所有系统共享的**纯枚举与数据结构**。本模块不依赖任何业务系统，
是依赖方向的根（最底层）。任何系统都可 import 本模块，但本模块不得反向
import 任何系统。

约定：时间统一 ns | 数据量统一 Byte | 带宽统一 B/s | 算力统一 FLOPS。
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ---------------------------------------------------------------- 枚举
class OpState(IntEnum):
    WAITING = 0; READY = 1; RUNNING = 2; FINISHED = 3
    BLOCKED = 4   # 无法满足用户固定(PINNED)数据源等硬性条件而永久阻塞（计入未完成）


class DataType(IntEnum):
    WEIGHT = 0; ACTIVATION = 1; KV_CACHE = 2; TEMPORARY = 3; INPUT = 4; OUTPUT = 5


class EventType(IntEnum):
    COMPUTE = 0; TRANSFER = 1; SYNC = 2; MEMORY = 3; FINISH = 4


class DeviceType(IntEnum):
    GPU = 0; DRAM_PIM = 1; SRAM_PIM = 2; RERAM_PIM = 3; CPU = 4
    SRAM = 5; DRAM = 6   # v3.1：纯存储单元（只存不整，算力 0）


class BottleneckType(IntEnum):
    COMPUTE = 0; MEMORY = 1; COMMUNICATION = 2; SYNCHRONIZATION = 3; NONE = 99


# 数据源决策模式
class DataSourceMode(IntEnum):
    AUTO = 0     # 调度器就近参考（给出建议）
    PINNED = 1   # 用户固定从某设备取


# 算子类别：固定只有两类（LINEAR / NONLINEAR）。属于"精度系统"的派生枚举，
# 但为减少公共层与系统的耦合，将其也置于公共底层（见 core.precision）。
class OperatorCategory(IntEnum):
    LINEAR = 0
    NONLINEAR = 1

    @classmethod
    def from_name(cls, name: str) -> "OperatorCategory":
        up = str(name).strip().upper()
        if up == "LINEAR":
            return cls.LINEAR
        if up == "NONLINEAR":
            return cls.NONLINEAR
        raise ValueError(f"未知算子类别: '{name}'。可选: LINEAR/NONLINEAR")


# 精度枚举属于"精度系统"，但其等级数值被校验、调度广泛共享引用。
# 为让精度系统保持单一事实来源，PrecisionLevel 定义在 core.precision（见包索引）。
# 说明：本文件不含 PrecisionLevel，避免两层各定义一份造成分叉。


# ---------------------------------------------------------------- 数据结构

@dataclass
class DataObject:
    """一份数据（权重 / 激活 / KV / 输入 / 输出），描述体积、驻留、生产消费关系。"""
    id: str
    name: str
    data_type: DataType
    size_bytes: int
    location: str = ""                    # 主要位置（兼容 v1；初始驻留主设备）
    replica_locations: list = field(default_factory=list)  # v2: 冗余驻留的多设备列表
    ready_time_ns: int = 0
    producer_op: Optional[str] = None
    consumers: list = field(default_factory=list)
    num_shards: int = 1                   # 预留分片字段
    shard_of: str = ""

    def residing_devices(self) -> list:
        """当前驻留的所有设备（含主位置 + 副本）"""
        devs = []
        if self.location:
            devs.append(self.location)
        for d in self.replica_locations:
            if d and d not in devs:
                devs.append(d)
        return devs


@dataclass
class InputSpec:
    """算子一个输入的数据源说明（用户从算子出发决定：这个输入从哪来）"""
    data_id: str
    source_device: str = ""            # 用户指定读源；空 = 用调度器参考(auto)
    pinned: bool = False               # 用户是否明确固定来源（非 auto）
    note: str = ""                     # 参考/告警说明

    @property
    def mode(self) -> DataSourceMode:
        return DataSourceMode.PINNED if self.pinned else DataSourceMode.AUTO


@dataclass
class OperatorSpec:
    """算子级完整配置说明（由 mapping_engine 生成）"""
    op_id: str
    compute_device: str = ""           # 算子放哪算（用户决定）
    inputs: list = field(default_factory=list)   # list[InputSpec]
    devices: list = field(default_factory=list)   # 分片并行设备组（预留）
    split: dict = field(default_factory=dict)     # {dim, num_parts}（预留）


@dataclass
class Operator:
    """一个可执行算子。精度/类别字段的类型定义见 core.precision。"""
    id: str
    name: str
    op_type: str
    flops: int
    required_precision: object = None            # PrecisionLevel（惰性绑定）
    category: OperatorCategory = OperatorCategory.LINEAR
    data_precision: object = None                # PrecisionLevel
    execution_precision: Optional[object] = None # PrecisionLevel or None
    input_ids: list = field(default_factory=list)
    output_ids: list = field(default_factory=list)
    shape_desc: str = ""


@dataclass
class DurationBreakdown:
    compute_ns: int = 0
    local_read_ns: int = 0
    local_write_ns: int = 0
    transfer_ns: int = 0
    sync_ns: int = 0

    @property
    def total_ns(self) -> int:
        return (self.compute_ns + self.local_read_ns + self.local_write_ns
                + self.transfer_ns + self.sync_ns)


@dataclass
class Event:
    id: int
    event_type: EventType
    start_time_ns: int
    end_time_ns: int
    operator_id: str = ""
    resource_id: str = ""
    component: str = ""
    payload: dict = field(default_factory=dict)


@dataclass
class OperatorTiming:
    op_id: str = ""
    op_type: str = ""
    hardware: str = ""
    start_ns: int = 0
    end_ns: int = 0
    duration_ns: int = 0
    compute_ns: int = 0
    local_read_ns: int = 0
    local_write_ns: int = 0
    transfer_ns: int = 0
    sync_ns: int = 0


@dataclass
class SimulationResult:
    """仿真最终结果。序列化为 JSON 见 core.exporter（输出系统）"""
    metadata: dict = field(default_factory=dict)
    total_latency_ns: int = 0
    breakdown: DurationBreakdown = field(default_factory=DurationBreakdown)
    bottleneck: BottleneckType = BottleneckType.NONE
    bottleneck_rationale: str = ""
    operator_timings: list = field(default_factory=list)
    event_trace: list = field(default_factory=list)
    movement_bytes: dict = field(default_factory=dict)
    data_source_notes: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def add_note(self, note: str):
        self.data_source_notes.append(note)


# 便于其它模块统一读取公共数据结构的约定命名
OpStateType = OpState
DataTypeEnum = DataType
BottleneckEnum = BottleneckType


# ---------------------------------------------------------------- 基础设施
class ConfigError(Exception):
    """用户配置错误 —— 信息要友好。各配置/硬件解析系统统一复用。"""
    pass


def load_yaml(path: str) -> dict:
    """读取一个 YAML 文件；缺失/解析失败抛 ConfigError。"""
    import os
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        import yaml
    except ImportError:
        raise ConfigError("缺少 PyYAML，请执行: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
