"""LLM-PIMSim v2 — 精度与算子类别系统（集中规则，v2.4）

本模块是 Precision / Operator Category / 18 类算子固定配置 /
5 种硬件能力 Capability 的**单一事实来源**。
- 18 类算子的 category / data_precision / execution_precision 已固定，不要改动。
- 5 种硬件的三维 Capability 已固定，不要改动。
- 所有构建方（workload_adapter / model_lib）都从本表取值，避免散落硬编码。
"""
from contracts import OperatorCategory, PrecisionLevel


# ============================================================
# 18 类算子的固定配置（key = 算子名，与 workload_model 中的 Kernel.name / id 对应）
# 每个条目: dict(category, data, execution)。
#   execution = None 仅允许 Embedding / KV_Cache_Write / KV_Cache_Read（纯数据访问/更新型算子）。
# ============================================================
OPERATOR_PRECISION_RULES = {
    "Embedding":     dict(category=OperatorCategory.LINEAR,    data=PrecisionLevel.FP16, execution=None),
    "LN1":           dict(category=OperatorCategory.NONLINEAR, data=PrecisionLevel.FP32, execution=PrecisionLevel.FP32),
    "QKV_proj":      dict(category=OperatorCategory.LINEAR,    data=PrecisionLevel.INT8, execution=PrecisionLevel.INT8),
    "RoPE":          dict(category=OperatorCategory.NONLINEAR, data=PrecisionLevel.FP32, execution=PrecisionLevel.FP32),
    "KV_Cache_Write":dict(category=OperatorCategory.LINEAR,    data=PrecisionLevel.INT4, execution=None),
    "KV_Cache_Read": dict(category=OperatorCategory.LINEAR,    data=PrecisionLevel.INT4, execution=None),
    "Attn_Score":    dict(category=OperatorCategory.LINEAR,    data=PrecisionLevel.INT8, execution=PrecisionLevel.INT8),
    "Softmax":       dict(category=OperatorCategory.NONLINEAR, data=PrecisionLevel.FP32, execution=PrecisionLevel.FP32),
    "Attn_Context":  dict(category=OperatorCategory.LINEAR,    data=PrecisionLevel.INT8, execution=PrecisionLevel.INT8),
    "O_proj":        dict(category=OperatorCategory.LINEAR,    data=PrecisionLevel.INT8, execution=PrecisionLevel.INT8),
    "Residual1":     dict(category=OperatorCategory.NONLINEAR, data=PrecisionLevel.FP16, execution=PrecisionLevel.FP16),
    "LN2":           dict(category=OperatorCategory.NONLINEAR, data=PrecisionLevel.FP32, execution=PrecisionLevel.FP32),
    "FFN_gate":      dict(category=OperatorCategory.LINEAR,    data=PrecisionLevel.INT8, execution=PrecisionLevel.INT8),
    "FFN_up":        dict(category=OperatorCategory.LINEAR,    data=PrecisionLevel.INT8, execution=PrecisionLevel.INT8),
    "SiLU":          dict(category=OperatorCategory.NONLINEAR, data=PrecisionLevel.FP16, execution=PrecisionLevel.FP16),
    "FFN_down":      dict(category=OperatorCategory.LINEAR,    data=PrecisionLevel.INT8, execution=PrecisionLevel.INT8),
    "Residual2":     dict(category=OperatorCategory.NONLINEAR, data=PrecisionLevel.FP16, execution=PrecisionLevel.FP16),
    "LMHead":        dict(category=OperatorCategory.LINEAR,    data=PrecisionLevel.FP8,  execution=PrecisionLevel.FP8),
}


def operator_rule_by_name(name: str) -> dict:
    """按算子名查固定规则；未知名返回 None（不抛错，便于未匹配算子走旧默认）。"""
    return OPERATOR_PRECISION_RULES.get(name)


# 便捷构造辅助（供 OperatorBuilder / workload_adapter / model_lib 使用）
def apply_operator_rule(op, name: str):
    """把固定规则应用到任意对象，需该对象有 category/data_precision/execution_precision 三属性。"""
    rule = OPERATOR_PRECISION_RULES.get(name)
    if rule is None:
        return op
    op.category = rule["category"]
    op.data_precision = rule["data"]
    op.execution_precision = rule["execution"]
    return op


# op_type(op_type 字符串) 与 name 子串 → 类别归类（供无法精确匹配名字的旧路径使用）
_NONLINEAR_OP_TYPES = {"LayerNorm", "Softmax", "Activation"}
def categorize_fallback(name: str, op_type: str, precision: PrecisionLevel):
    """回退归类：按 op_type 判断类别、双精度取同一 precision。
    返回 (category, data_precision, execution_precision)。"""
    cat = (OperatorCategory.NONLINEAR
           if op_type in _NONLINEAR_OP_TYPES else OperatorCategory.LINEAR)
    return cat, precision, precision


# ============================================================
# 5 种硬件的固定三维 Capability
# key = DeviceType 名（CPU / GPU / SRAM_PIM / DRAM_PIM / RERAM_PIM）
# 每项: dict(categories=[..], data=[..], execution=[..])
# ============================================================
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
        categories=[_L],
        data=[PrecisionLevel.FP32, PrecisionLevel.BF16, PrecisionLevel.FP16,
              PrecisionLevel.INT8],
        execution=[PrecisionLevel.FP32, PrecisionLevel.BF16, PrecisionLevel.FP16,
                   PrecisionLevel.INT8],
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


# 5 种硬件的 18×5 兼容性矩阵（仅用于验证脚本，不做运行时判断）
# 行 = 算子名，列 = 设备名。True=可执行，False=不可。
COMPATIBILITY_MATRIX = {
    "Embedding":     {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "LN1":           {"CPU": True, "GPU": False, "SRAM_PIM": False, "DRAM_PIM": False, "RERAM_PIM": False},
    "QKV_proj":      {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "RoPE":          {"CPU": True, "GPU": False, "SRAM_PIM": False, "DRAM_PIM": False, "RERAM_PIM": False},
    "KV_Cache_Write":{"CPU": True, "GPU": False, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "KV_Cache_Read": {"CPU": True, "GPU": False, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "Attn_Score":    {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "Softmax":       {"CPU": True, "GPU": False, "SRAM_PIM": False, "DRAM_PIM": False, "RERAM_PIM": False},
    "Attn_Context":  {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "O_proj":        {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "Residual1":     {"CPU": True, "GPU": False, "SRAM_PIM": True, "DRAM_PIM": False, "RERAM_PIM": True},
    "LN2":           {"CPU": True, "GPU": False, "SRAM_PIM": False, "DRAM_PIM": False, "RERAM_PIM": False},
    "FFN_gate":      {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "FFN_up":        {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "SiLU":          {"CPU": True, "GPU": False, "SRAM_PIM": True, "DRAM_PIM": False, "RERAM_PIM": True},
    "FFN_down":      {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "Residual2":     {"CPU": True, "GPU": False, "SRAM_PIM": True, "DRAM_PIM": False, "RERAM_PIM": True},
    "LMHead":        {"CPU": False, "GPU": False, "SRAM_PIM": False, "DRAM_PIM": False, "RERAM_PIM": True},
}
