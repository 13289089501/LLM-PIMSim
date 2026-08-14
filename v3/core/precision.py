"""
LLM-PIMSim v3 core.precision — 【精度系统】

职责：
  1. PrecisionLevel —— 六种计算精度的唯一枚举（等级越大精度越高、成本越高）
  2. 精度等级换算 / 字节换算等纯工具
  3. HARDWARE_CAPABILITY —— 5 种硬件"按精度 + 算子类别"的三维能力表
     （category / data / execution）。硬件哪类算符可做、哪种精度可读写运算，
     由本表集中规定，避免散落硬编码。

依赖：仅 core.common（OperatorCategory 枚举）。
不依赖（不反向 import）：算子系统 / 调度器 / 校验 / 输出等业务系统。

注意：
  - OP 的 18 类固定规则（category/data/execution）属【算子系统】，不在此处。
  - 硬件的"算力/容量/带宽"等性能参数属硬件建模，不在此处；此处只关心"能做哪
    类算法、支持哪种精度"的能力语义。
"""

from enum import IntEnum
from core.common import OperatorCategory


class PrecisionLevel(IntEnum):
    """六种计算精度，数值 = 精度等级（越大精度越高，成本越高）。
    按"精度顺序"而非 bit width 排序，以便浮点(FP8/FP16/BF16/FP32)与定点(INT4/INT8)
    在同一规模下可比。
      INT4=1 < INT8=2 < FP8=3 < FP16=4 < BF16=5 < FP32=6
    """
    INT4 = 1
    INT8 = 2
    FP8 = 3
    FP16 = 4
    BF16 = 5
    FP32 = 6

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


# ---------------------------------------------------------------- 工具

def precision_to_bytes(precision: PrecisionLevel) -> int:
    """按精度等级决定数据字节数: FP32=4, FP16/BF16=2, FP8=1, INT8=1, INT4≈1(取整)。"""
    if precision == PrecisionLevel.FP32:
        return 4
    if precision in (PrecisionLevel.FP16, PrecisionLevel.BF16):
        return 2
    return 1  # FP8 / INT8 / INT4 近似 1 字节（真实 INT4 是 0.5，取整简化）


# ---------------------------------------------------------------- 硬件能力表

_L = OperatorCategory.LINEAR
_N = OperatorCategory.NONLINEAR

HARDWARE_CAPABILITY = {
    "CPU": dict(
        categories=[_L, _N],
        data=[PrecisionLevel.FP32, PrecisionLevel.BF16, PrecisionLevel.FP16,
              PrecisionLevel.INT8, PrecisionLevel.INT4],
        execution=[PrecisionLevel.FP32, PrecisionLevel.BF16, PrecisionLevel.FP16,
                   PrecisionLevel.INT8],
    ),
    "GPU": dict(
        # v3 修正：GPU 作为通用处理器可执行 LINEAR 与 NONLINEAR（真实 GPU 可跑
        # LayerNorm/Softmax/激活等）。早期版本写成 [LINEAR] 导致任何 GPU-only 部署
        # 都无法完成（每层都含非线性算子）——这是设计矛盾，已于 v3 修正。
        categories=[_L, _N],
        data=[PrecisionLevel.FP32, PrecisionLevel.BF16, PrecisionLevel.FP16,
              PrecisionLevel.FP8, PrecisionLevel.INT8, PrecisionLevel.INT4],
        execution=[PrecisionLevel.FP32, PrecisionLevel.BF16, PrecisionLevel.FP16,
                   PrecisionLevel.FP8, PrecisionLevel.INT8],
    ),
    "SRAM_PIM": dict(
        categories=[_L, _N],
        data=[PrecisionLevel.BF16, PrecisionLevel.FP16, PrecisionLevel.INT8,
              PrecisionLevel.INT4],
        execution=[PrecisionLevel.BF16, PrecisionLevel.FP16, PrecisionLevel.INT8],
    ),
    "DRAM_PIM": dict(
        categories=[_L],
        data=[PrecisionLevel.BF16, PrecisionLevel.FP16, PrecisionLevel.INT8,
              PrecisionLevel.INT4],
        execution=[PrecisionLevel.BF16, PrecisionLevel.FP16, PrecisionLevel.INT8],
    ),
    "RERAM_PIM": dict(
        categories=[_L, _N],
        data=[PrecisionLevel.BF16, PrecisionLevel.FP16, PrecisionLevel.FP8,
              PrecisionLevel.INT8, PrecisionLevel.INT4],
        execution=[PrecisionLevel.BF16, PrecisionLevel.FP16, PrecisionLevel.FP8,
                   PrecisionLevel.INT8],
    ),
}


def hardware_category_supported(device_type_name: str, category: OperatorCategory) -> bool:
    """查询某类硬件是否支持某算子类别（LINEAR/NONLINEAR）。"""
    cap = HARDWARE_CAPABILITY.get(device_type_name)
    return bool(cap and category in cap.get("categories", []))


def hardware_data_supported(device_type_name: str, precision: PrecisionLevel) -> bool:
    """查询某类硬件是否支持以该精度存储/读写数据。"""
    cap = HARDWARE_CAPABILITY.get(device_type_name)
    return bool(cap and precision in cap.get("data", []))


def hardware_execution_supported(device_type_name: str, precision: PrecisionLevel) -> bool:
    """查询某类硬件是否支持以该精度执行计算。"""
    cap = HARDWARE_CAPABILITY.get(device_type_name)
    return bool(cap and precision in cap.get("execution", []))
