"""
LLM-PIMSim v3 — hardware_factory（兼容转发壳）

v3 解耦：硬件构建已迁至「硬件系统」 core.hardware_sys。
本文件仅转发，请勿在此添加新逻辑。
"""
from core.hardware_sys import HardwareFactory, build_hardware
