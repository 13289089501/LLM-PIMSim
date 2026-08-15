"""
LLM-PIMSim v3 — constraints（兼容转发壳）

v3 解耦：配置约束校验已迁移至「校验系统」 core.validator。
本文件仅转发，请勿在此添加新逻辑。
"""
from core.validator import (
    Issue, ValidationResult, ConstraintChecker, validate_config,
    PRECISION_RANK, RANK_NAME,
)
