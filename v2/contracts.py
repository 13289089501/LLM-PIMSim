"""
LLM-PIMSim v2 — 共享数据结构（配置驱动版）
时间统一 ns | 数据量统一 Byte | 带宽统一 B/s
在 v1 基础上:
  - DataObject 支持多副本驻留 replica_locations（冗余驻留）
  - 新增 DataSpec / InputSpec：算子级"数据从哪来"说明（用户决定数据源）
  - 预留分片字段（num_shards / shard_of），第一版不实现切割算法
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class OpState(IntEnum):
    WAITING = 0; READY = 1; RUNNING = 2; FINISHED = 3


class DataType(IntEnum):
    WEIGHT = 0; ACTIVATION = 1; KV_CACHE = 2; TEMPORARY = 3; INPUT = 4; OUTPUT = 5


class EventType(IntEnum):
    COMPUTE = 0; TRANSFER = 1; SYNC = 2; MEMORY = 3; FINISH = 4


class DeviceType(IntEnum):
    GPU = 0; DRAM_PIM = 1; SRAM_PIM = 2; RERAM_PIM = 3; CPU = 4


class PrecisionLevel(IntEnum):
    """六种计算精度，数值=精度等级（越大精度越高，成本越高）。
    注意: 这里按"精度顺序"而非 bit width 排序，以便 IEEE 浮点(FP8/FP16/BF16/FP32)
    与定点(INT4/INT8)在同一规模下可比。保留 HIGH/LOW 语义别名。
    - INT4 = 1 (最低)
    - INT8 = 2
    - FP8  = 3
    - FP16 = 4
    - BF16 = 5  (BF16 独立成档；与 FP16 同属 16-bit 但尾数精度不同)
    - FP32 = 6 (最高)
    """
    INT4 = 1
    INT8 = 2
    FP8 = 3
    FP16 = 4
    BF16 = 5
    FP32 = 6

    # 语义别名（继承自设计文档的 HIGH/LOW 概念）
    @property
    def is_high(self) -> bool:
        return self >= PrecisionLevel.FP16

    @property
    def is_low(self) -> bool:
        return self < PrecisionLevel.FP16

    @classmethod
    def from_name(cls, name: str) -> "PrecisionLevel":
        """从字符串构造，兼容 'FP32'/'fp32'/'HIGH'/'LOW'"""
        up = str(name).strip().upper()
        mapping = {
            "FP32": cls.FP32, "FP16": cls.FP16, "BF16": cls.BF16,
            "FP8": cls.FP8, "INT8": cls.INT8, "INT4": cls.INT4,
            "HIGH": cls.FP16, "LOW": cls.INT8,
        }
        if up in mapping:
            return mapping[up]
        raise ValueError(
            f"未知精度: '{name}'。可选: FP32/BF16/FP16/FP8/INT8/INT4 (或 HIGH/LOW)")


class OperatorCategory(IntEnum):
    """算子类别：当前固定只有两类。
    LINEAR    线性计算（矩阵乘/投影/残差/查表等）
    NONLINEAR 非线性计算（LayerNorm/Softmax/激活等）
    """
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


class BottleneckType(IntEnum):
    COMPUTE = 0; MEMORY = 1; COMMUNICATION = 2; SYNCHRONIZATION = 3; NONE = 99


# 数据源决策模式
class DataSourceMode(IntEnum):
    AUTO = 0     # 调度器就近参考（给出建议）
    PINNED = 1   # 用户固定从某设备取


@dataclass
class DataObject:
    id: str
    name: str
    data_type: DataType
    size_bytes: int
    location: str = ""                    # 主要位置（兼容 v1；初始驻留主设备）
    replica_locations: list = field(default_factory=list)  # v2: 冗余驻留的多设备列表
    ready_time_ns: int = 0
    producer_op: Optional[str] = None
    consumers: list = field(default_factory=list)
    # ---- 分片预留（第一版不实现）----
    num_shards: int = 1
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
    # ---- 分片预留（第一版不实现）----
    devices: list = field(default_factory=list)   # 分片并行设备组
    split: dict = field(default_factory=dict)     # {dim, num_parts}


@dataclass
class Operator:
    id: str
    name: str
    op_type: str
    flops: int
    required_precision: PrecisionLevel = PrecisionLevel.FP16
    # ---- 精度与算子类别系统（v2.3）----
    category: OperatorCategory = OperatorCategory.LINEAR
    data_precision: PrecisionLevel = PrecisionLevel.FP16
    execution_precision: Optional[PrecisionLevel] = None
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
    transfer_ns: int = 0
    sync_ns: int = 0


@dataclass
class SimulationResult:
    metadata: dict = field(default_factory=dict)
    total_latency_ns: int = 0
    breakdown: DurationBreakdown = field(default_factory=DurationBreakdown)
    bottleneck: BottleneckType = BottleneckType.NONE
    bottleneck_rationale: str = ""
    operator_timings: list = field(default_factory=list)
    event_trace: list = field(default_factory=list)
    movement_bytes: dict = field(default_factory=dict)
    # ---- v2: 数据源决策记录（告警/参考）----
    data_source_notes: list = field(default_factory=list)

    def add_note(self, note: str):
        self.data_source_notes.append(note)

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "total_latency_ns": self.total_latency_ns,
            "total_latency_ms": round(self.total_latency_ns / 1_000_000, 3),
            "breakdown": {
                "compute_ns": self.breakdown.compute_ns,
                "transfer_ns": self.breakdown.transfer_ns,
                "sync_ns": self.breakdown.sync_ns,
                "local_rw_ns": self.breakdown.local_read_ns + self.breakdown.local_write_ns,
            },
            "bottleneck": self.bottleneck.name,
            "bottleneck_rationale": self.bottleneck_rationale,
            "operator_timings": [
                {"op_id": t.op_id, "op_type": t.op_type,
                 "hardware": t.hardware, "start_ns": t.start_ns,
                 "end_ns": t.end_ns, "duration_ns": t.duration_ns,
                 "compute_ns": t.compute_ns, "transfer_ns": t.transfer_ns}
                for t in self.operator_timings
            ],
            "event_trace": [
                {"id": e.id, "type": e.event_type.name,
                 "start_ns": e.start_time_ns, "end_ns": e.end_time_ns,
                 "operator": e.operator_id, "resource": e.resource_id,
                 "component": e.component}
                for e in self.event_trace
            ],
            "movement_bytes": self.movement_bytes,
            "data_source_notes": self.data_source_notes,
        }
