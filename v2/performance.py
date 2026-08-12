"""
LLM-PIMSim v2 — 性能模型（纯函数，无状态）
公式:
  T_compute   = FLOPs / (PeakFLOPS × efficiency)
  T_local_rw  = latency + data_size / bandwidth
  T_transfer  = T_read(src) + T_link(源读方向) + T_write(dst)
  —— v2: 链路读写带宽独立（非对称），从 src 读用 read_bw，写到 dst 用 write_bw
"""
from contracts import Operator, DurationBreakdown, PrecisionLevel, DataSourceMode


class PerformanceModel:
    """性能估算器 —— 纯函数，只读硬件参数，返回 DurationBreakdown"""

    def __init__(self, hardware_units: dict, interconnect):
        self.hw = hardware_units          # {id: HardwareUnit}
        self.interconnect = interconnect  # Interconnect

    def can_execute(self, op: Operator, hw_id: str) -> bool:
        hw = self.hw.get(hw_id)
        if hw is None:
            return False
        return hw.supports_precision(op.required_precision)

    def compute_time_ns(self, op: Operator, hw_id: str) -> int:
        hw = self.hw.get(hw_id)
        if hw is None:
            return 0
        eff = hw.efficiency_for(op.op_type)
        if op.flops == 0 or hw.peak_compute_flops == 0 or eff == 0:
            return 0
        return int(op.flops / (hw.peak_compute_flops * eff) * 1e9)

    def local_read_time_ns(self, data_size_bytes: int, hw_id: str) -> int:
        hw = self.hw.get(hw_id)
        if hw is None:
            return 0
        if data_size_bytes == 0 or hw.read_bandwidth_Bps == 0:
            return hw.read_latency_ns
        return int(hw.read_latency_ns + data_size_bytes / hw.read_bandwidth_Bps * 1e9)

    def local_write_time_ns(self, data_size_bytes: int, hw_id: str) -> int:
        hw = self.hw.get(hw_id)
        if hw is None:
            return 0
        if data_size_bytes == 0 or hw.write_bandwidth_Bps == 0:
            return hw.write_latency_ns
        return int(hw.write_latency_ns + data_size_bytes / hw.write_bandwidth_Bps * 1e9)

    def transfer_time_ns(self, data_size_bytes: int, src_hw_id: str, dst_hw_id: str) -> int:
        """
        跨设备搬运 src→dst:
          T = T_read(src) + T_link(源读方向) + T_write(dst)
        链路用 Link 对象的 read_bw（src 从 dst 读）——注意方向语义:
          数据从 src 搬到 dst，路径是 src 读(src 本地) → src→dst 链路 → dst 写。
        链路是 src→dst 方向，数据在链路上是 src 发出、dst 接收，
        因此占用的带宽通道 = src→dst 的 write 方向（即 Link.write_bw）。
        """
        link = self.interconnect.find_link(src_hw_id, dst_hw_id) if self.interconnect else None
        if link is None:
            # 无直连 → 不可搬运，返回 0 由调用方告警
            return 0

        t_read = self.local_read_time_ns(data_size_bytes, src_hw_id)   # 源本地读
        # 链路传输：数据流 src→dst，占 src→dst 链路写方向带宽 write_bw
        link_bw = link.write_bw_Bps
        if link_bw > 0:
            t_link = int(link.latency_ns + data_size_bytes / link_bw * 1e9)
        else:
            t_link = link.latency_ns
        t_write = self.local_write_time_ns(data_size_bytes, dst_hw_id) # 目的本地写
        return t_read + t_link + t_write

    def estimate(self, op: Operator, hw_id: str, data_sizes: dict = None) -> DurationBreakdown:
        bd = DurationBreakdown()
        if not self.can_execute(op, hw_id):
            return bd
        bd.compute_ns = self.compute_time_ns(op, hw_id)
        if data_sizes:
            for did, size in data_sizes.items():
                bd.local_read_ns += self.local_read_time_ns(size, hw_id)
        return bd
