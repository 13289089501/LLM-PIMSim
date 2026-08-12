"""
LLM-PIMSim v2 — 硬件模型（配置驱动版）
HardwareUnit: 通用硬件零件（能力+状态），由 hardware_factory 从 YAML 构造
Interconnect v2: 支持非对称读写带宽（读/写方向各自独立）
"""
from dataclasses import dataclass, field
from typing import Optional
from contracts import DeviceType, PrecisionLevel


@dataclass
class HardwareUnit:
    """一个硬件零件 —— GPU / DRAM-PIM / SRAM-PIM / ReRAM-PIM / HBM...
    所有硬件共用同一结构，仅参数不同（配置驱动，不需要为每类建子类）。"""
    id: str
    name: str
    device_type: DeviceType

    # --- 计算 ---
    peak_compute_flops: float = 0.0       # 峰值 FLOPS
    # 默认支持全部四种精度（FP32/FP16/INT8/INT4）
    supported_precision: list = field(default_factory=lambda: [
        PrecisionLevel.FP32, PrecisionLevel.FP16,
        PrecisionLevel.INT8, PrecisionLevel.INT4,
    ])
    parallelism: int = 1
    efficiency_table: dict = field(default_factory=dict)   # {op_type: efficiency}

    # --- 存储 ---
    memory_capacity_bytes: int = 0
    read_bandwidth_Bps: float = 0.0       # 本地读带宽
    write_bandwidth_Bps: float = 0.0      # 本地写带宽
    read_latency_ns: int = 0
    write_latency_ns: int = 0

    # --- 分片预留（第一版不实现算法，字段先存在）---
    num_banks: int = 1
    compute_units: int = 1

    # --- 动态状态 ---
    available_time_ns: int = 0
    current_task: str = ""

    def supports_precision(self, required: PrecisionLevel) -> bool:
        """硬件是否能执行该精度要求的算子。
        规则: 硬件支持的最高精度等级 >= 算子要求的精度等级，才允许计算。
        (与设计文档的 高精度需求不能被低精度硬件执行 一致)。
        """
        if not self.supported_precision:
            return False
        # 硬件支持的最高精度等级（PrecisionLevel 是 IntEnum，数值即等级）
        hw_max_level = max(p.value for p in self.supported_precision)
        return hw_max_level >= required.value

    def efficiency_for(self, op_type: str) -> float:
        return self.efficiency_table.get(op_type, 1.0)

    def can_fit(self, size_bytes: int) -> bool:
        return size_bytes <= self.memory_capacity_bytes

    def reset_state(self):
        self.available_time_ns = 0
        self.current_task = ""

    # 预留：功耗模型占位（第一版返回 0）
    def estimate_energy(self, duration_ns: int) -> float:
        return 0.0


class Link:
    """一条有向链路，读写带宽各自独立（非对称）"""
    def __init__(self, src: str, dst: str, read_bw: float, write_bw: float, latency_ns: int):
        self.src = src
        self.dst = dst
        self.read_bw_Bps = read_bw     # src 从 dst 读数据的带宽
        self.write_bw_Bps = write_bw   # src 往 dst 写数据的带宽
        self.latency_ns = latency_ns


@dataclass
class Interconnect:
    """设备间互连（v2: 每条连接读/写方向带宽独立）"""
    links: list = field(default_factory=list)   # list[Link]

    def add_link(self, src, dst, read_bandwidth_Bps, write_bandwidth_Bps, latency_ns):
        self.links.append(Link(src, dst, read_bandwidth_Bps, write_bandwidth_Bps, latency_ns))

    def add_undirected(self, a, b, read_bandwidth_Bps, write_bandwidth_Bps, latency_ns,
                       bidirectional: bool = True):
        """加一条（可双向）连接。双向时反向链路的读写与正向定义成对。
        约定: 单向写 a→b 时 read_bw(a→b)=用户给的 read, write_bw(a→b)=用户给的 write。
        反向 b→a 默认对称采用相同带宽（也可用 default_connect 覆盖）。"""
        self.add_link(a, b, read_bandwidth_Bps, write_bandwidth_Bps, latency_ns)
        if bidirectional:
            self.add_link(b, a, write_bandwidth_Bps, read_bandwidth_Bps, latency_ns)

    def find_link(self, src: str, dst: str) -> Optional[Link]:
        """查 src→dst 的链路；无直接连接返回 None"""
        for lnk in self.links:
            if lnk.src == src and lnk.dst == dst:
                return lnk
        return None
