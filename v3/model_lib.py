"""
LLM-PIMSim v3 — model_lib（模型维度登记表，非算子图生成）

v3 解耦：旧的 model_lib 内置算子图生成（L0_q_gemm 等）已废弃——仿真统一走
「算子系统」core.operator_sys 的 workload 路径。本模块只保留：
  - 模型维度登记表（MODEL_DIMS，单一事实来源，供 experiment_runner._resolve_model_dims 与 /api/models 使用）
  - list_models()（供 GUI 下拉展示）

如需生成 workload，请用 core.operator_sys.build_model_workload 并配合 experiment.yaml
的 workload 段或 MODEL_DIMS。
"""

# 内置模型维度表（single source of truth）。新模型在此注册即可。
MODEL_DIMS = {
    "llama7b": dict(num_layers=32, hidden=4096, ffn_size=11008,
                    num_heads=32, head_dim=128, vocab=32000, pbytes=2),
}


def list_models() -> list:
    return list(MODEL_DIMS.keys())
