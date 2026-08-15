"""
LLM-PIMSim v3 — precisions（兼容转发壳）

v3 解耦：
  - PrecisionLevel / 精度工具 / HARDWARE_CAPABILITY 属【精度系统】 core.precision
  - OperatorCategory / Operator 18 类固定规则 / 兼容矩阵 属【算子系统】 core.operator_sys
本文件为保持旧入口兼容而聚合转发，请勿在此添加新逻辑。
"""
from core.precision import (
    PrecisionLevel, precision_to_bytes, HARDWARE_CAPABILITY,
    hardware_category_supported, hardware_data_supported, hardware_execution_supported,
)
from core.common import OperatorCategory
from core.operator_sys import (
    OPERATOR_PRECISION_RULES, COMPATIBILITY_MATRIX,
    operator_rule_by_name, apply_operator_rule,
)


def categorize_fallback(name: str, op_type: str, precision):
    """回退归类：按 op_type 判断类别、双精度取同一 precision。保留旧签名。"""
    from core.common import OperatorCategory as _OC
    _NONLINEAR_OP_TYPES = {"LayerNorm", "Softmax", "Activation"}
    cat = (_OC.NONLINEAR if op_type in _NONLINEAR_OP_TYPES else _OC.LINEAR)
    return cat, precision, precision
