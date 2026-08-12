"""
LLM-PIMSim v2 — HardwareFactory
把 config_loader 解析出的 HardwareConfig / LinkConfig
变成真正的 HardwareUnit 对象和 Interconnect 连接。
新增硬件 = YAML 加一段 + 预设表加一行，工厂代码不用改。
"""
from contracts import DeviceType, PrecisionLevel
from hardware import HardwareUnit, Interconnect
from config_loader import HardwareConfig, LinkConfig, ExperimentConfig

# 字符串类型 → 枚举
_DEVICE_TYPE_MAP = {
    "GPU": DeviceType.GPU,
    "DRAM_PIM": DeviceType.DRAM_PIM,
    "SRAM_PIM": DeviceType.SRAM_PIM,
    "RERAM_PIM": DeviceType.RERAM_PIM,
    "CPU": DeviceType.CPU,
    "HBM": DeviceType.GPU,   # HBM 作为存储视为可用设备，简化为 GPU 运算能力 0
}


class HardwareFactory:
    """从配置创建硬件零件与互连网络"""

    @staticmethod
    def create_devices(hardware_cfg: dict) -> dict:
        """{id: HardwareConfig} -> {id: HardwareUnit}"""
        from contracts import PrecisionLevel
        units = {}
        for hid, hc in hardware_cfg.items():
            # 精度支持：配置指定则用配置，否则默认支持全部四种
            if hc.precision:
                supported_precision = list(hc.precision)
            else:
                supported_precision = [
                    PrecisionLevel.FP32, PrecisionLevel.FP16,
                    PrecisionLevel.INT8, PrecisionLevel.INT4,
                ]
            units[hid] = HardwareUnit(
                id=hid,
                name=hid,
                device_type=_DEVICE_TYPE_MAP.get(hc.type, DeviceType.GPU),
                peak_compute_flops=hc.peak_f,
                memory_capacity_bytes=hc.mem_bytes,
                read_bandwidth_Bps=hc.read_bw,
                write_bandwidth_Bps=hc.write_bw,
                read_latency_ns=hc.read_lat_ns,
                write_latency_ns=hc.write_lat_ns,
                parallelism=hc.parallelism,
                efficiency_table=dict(hc.efficiency),
                supported_precision=supported_precision,
            )
        return units

    @staticmethod
    def create_interconnect(link_cfgs: list) -> Interconnect:
        """list[LinkConfig] -> Interconnect（非对称读写）"""
        inter = Interconnect()
        for lk in link_cfgs:
            inter.add_link(lk.src, lk.dst, lk.read_bw, lk.write_bw, lk.lat_ns)
        return inter


def build_hardware(cfg: ExperimentConfig):
    """完整装配：硬件 + 互连"""
    devices = HardwareFactory.create_devices(cfg.hardware)
    interconnect = HardwareFactory.create_interconnect(cfg.links)
    return devices, interconnect
