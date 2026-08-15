"""
LLM-PIMSim v3 — hardware（兼容转发壳）

v3 解耦：硬件数据模型已迁至「硬件系统」 core.hardware_sys；
链路带宽表已迁至「链路系统」 core.link_sys。
本文件仅转发，请勿在此添加新逻辑。
"""
from core.hardware_sys import HardwareUnit
from core.link_sys import LinkBandwidthTable
