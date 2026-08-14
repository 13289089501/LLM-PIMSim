"""
LLM-PIMSim v1 — 内置模型库
用户只需选名字，所有算子自动生成。
当前支持: llama7b

每个模型输出:
  - operators: list[Operator]  (算子 DAG)
  - data_objects: list[DataObject]  (数据目录)
  - config dict (供报告使用)
"""
from contracts import DataObject, DataType, Operator, PrecisionLevel, OperatorCategory
from precisions import categorize_fallback


def _builder_precision_to_bytes(precision) -> int:
    """按精度等级决定数据字节数: FP32=4, FP16=2, INT8=1, INT4=0.5(取整1)"""
    from contracts import PrecisionLevel
    try:
        if isinstance(precision, PrecisionLevel):
            level = precision
        else:
            level = PrecisionLevel.from_name(precision)
    except Exception:
        level = PrecisionLevel.FP16
    if level == PrecisionLevel.FP32:
        return 4
    if level == PrecisionLevel.FP16:
        return 2
    if level == PrecisionLevel.INT8:
        return 1
    return 1  # INT4 用 1 字节近似（真实是 0.5，取整简化）


def _build_llama_like(name: str, num_layers: int, hidden_size: int, ffn_size: int,
                      num_heads: int, head_dim: int, precision=PrecisionLevel.FP16,
                      vocab_size: int = 32000, max_seq_len: int = 2048) -> dict:
    """通用 LLaMA-like 模型构建器"""
    from contracts import PrecisionLevel
    ops = []
    datas = {}

    # 精度字节数（按精度等级）
    pbytes = _builder_precision_to_bytes(precision)

    # 规范化算子的 required_precision（字符串 H""IGH/LOW 也接受）
    req_precision = precision if isinstance(precision, PrecisionLevel) else PrecisionLevel.from_name(precision)

    def make_data(data_id, dname, dtype, size):
        d = DataObject(id=data_id, name=dname, data_type=dtype, size_bytes=size)
        datas[data_id] = d
        return d

    def make_op(op_id, oname, otype, flops_val, inputs, outputs, shape=""):
        # 精度与算子类别（v2.3）：按 op_type 归类类别，双精度取模型精度
        cat, data_prec, exec_prec = categorize_fallback(oname, otype, req_precision)
        o = Operator(id=op_id, name=oname, op_type=otype, flops=flops_val,
                     required_precision=req_precision,
                     category=cat,
                     data_precision=data_prec,
                     execution_precision=exec_prec,
                     input_ids=list(inputs), output_ids=list(outputs),
                     shape_desc=shape)
        ops.append(o)
        return o

    # --- 全局数据 ---
    # Embedding 权重
    embed_w_size = vocab_size * hidden_size * pbytes
    make_data("embed_weight", "Embedding Weight", DataType.WEIGHT, embed_w_size)

    # 每层数据 + 算子
    for L in range(num_layers):
        pre = f"L{L}"

        # 权重
        q_w = make_data(f"{pre}_q_weight", f"{pre} Q Weight", DataType.WEIGHT,
                        hidden_size * hidden_size * pbytes)
        k_w = make_data(f"{pre}_k_weight", f"{pre} K Weight", DataType.WEIGHT,
                        hidden_size * hidden_size * pbytes)
        v_w = make_data(f"{pre}_v_weight", f"{pre} V Weight", DataType.WEIGHT,
                        hidden_size * hidden_size * pbytes)
        o_w = make_data(f"{pre}_o_weight", f"{pre} Out Weight", DataType.WEIGHT,
                        hidden_size * hidden_size * pbytes)
        ffn1_w = make_data(f"{pre}_ffn1_weight", f"{pre} FFN1 Weight", DataType.WEIGHT,
                           hidden_size * ffn_size * pbytes)
        ffn2_w = make_data(f"{pre}_ffn2_weight", f"{pre} FFN2 Weight", DataType.WEIGHT,
                           ffn_size * hidden_size * pbytes)
        ln1_w = make_data(f"{pre}_ln1_weight", f"{pre} LayerNorm1 Weight", DataType.WEIGHT,
                          hidden_size * pbytes)
        ln2_w = make_data(f"{pre}_ln2_weight", f"{pre} LayerNorm2 Weight", DataType.WEIGHT,
                          hidden_size * pbytes)

        # 激活（临时）
        inp_act = make_data(f"{pre}_input_act", f"{pre} Input Activation", DataType.ACTIVATION,
                            max_seq_len * hidden_size * pbytes)
        ln1_out = make_data(f"{pre}_ln1_out", f"{pre} LN1 Output", DataType.TEMPORARY,
                            max_seq_len * hidden_size * pbytes)
        q_out = make_data(f"{pre}_q_out", f"{pre} Q Output", DataType.TEMPORARY,
                          max_seq_len * hidden_size * pbytes)
        k_out = make_data(f"{pre}_k_out", f"{pre} K Output", DataType.TEMPORARY,
                          max_seq_len * hidden_size * pbytes)
        v_out = make_data(f"{pre}_v_out", f"{pre} V Output", DataType.TEMPORARY,
                          max_seq_len * hidden_size * pbytes)
        attn_out = make_data(f"{pre}_attn_out", f"{pre} Attention Output", DataType.TEMPORARY,
                             max_seq_len * hidden_size * pbytes)
        o_proj_out = make_data(f"{pre}_o_proj_out", f"{pre} O-Proj Output", DataType.TEMPORARY,
                               max_seq_len * hidden_size * pbytes)
        resid1 = make_data(f"{pre}_resid1", f"{pre} Residual1 Out", DataType.TEMPORARY,
                           max_seq_len * hidden_size * pbytes)
        ln2_out = make_data(f"{pre}_ln2_out", f"{pre} LN2 Output", DataType.TEMPORARY,
                            max_seq_len * hidden_size * pbytes)
        ffn1_out = make_data(f"{pre}_ffn1_out", f"{pre} FFN1 Output", DataType.TEMPORARY,
                             max_seq_len * ffn_size * pbytes)
        ffn2_out = make_data(f"{pre}_ffn2_out", f"{pre} FFN2 Output", DataType.TEMPORARY,
                             max_seq_len * hidden_size * pbytes)
        layer_out = make_data(f"{pre}_output_act", f"{pre} Output Activation", DataType.ACTIVATION,
                              max_seq_len * hidden_size * pbytes)

        # KV Cache（decode 阶段会增长）
        kv_size = 2 * num_layers * num_heads * head_dim * max_seq_len * pbytes
        kvc = make_data(f"{pre}_kv_cache", f"{pre} KV Cache", DataType.KV_CACHE, kv_size)

        # --- 算子（每层）---
        # LayerNorm 1: input → ln1_out
        flops_ln = hidden_size * 5  # 约 5× hidden
        make_op(f"{pre}_ln1", f"{pre} LayerNorm1", "LayerNorm", flops_ln,
                [inp_act.id, ln1_w.id], [ln1_out.id], f"[{max_seq_len},{hidden_size}]")

        # QKV GEMM
        flops_qkv = 3 * max_seq_len * hidden_size * hidden_size * 2
        make_op(f"{pre}_q_gemm", f"{pre} Q GEMM", "GEMM", flops_qkv // 3,
                [ln1_out.id, q_w.id], [q_out.id], f"[{max_seq_len},{hidden_size}]x[{hidden_size},{hidden_size}]")
        make_op(f"{pre}_k_gemm", f"{pre} K GEMM", "GEMM", flops_qkv // 3,
                [ln1_out.id, k_w.id], [k_out.id], f"[{max_seq_len},{hidden_size}]x[{hidden_size},{hidden_size}]")
        make_op(f"{pre}_v_gemm", f"{pre} V GEMM", "GEMM", flops_qkv // 3,
                [ln1_out.id, v_w.id], [v_out.id], f"[{max_seq_len},{hidden_size}]x[{hidden_size},{hidden_size}]")

        # Attention (Softmax + 加权求和)
        flops_attn = max_seq_len * max_seq_len * hidden_size * 2
        make_op(f"{pre}_attn", f"{pre} Attention", "Attention", flops_attn,
                [q_out.id, k_out.id, v_out.id, kvc.id], [attn_out.id],
                f"QKV softmax-attn [{max_seq_len},{hidden_size}]")

        # Output Projection
        flops_o = max_seq_len * hidden_size * hidden_size * 2
        make_op(f"{pre}_o_proj", f"{pre} O-Projection", "GEMM", flops_o,
                [attn_out.id, o_w.id], [o_proj_out.id],
                f"[{max_seq_len},{hidden_size}]x[{hidden_size},{hidden_size}]")

        # Residual Add 1 (当做 0 FLOPs 的同步点)
        make_op(f"{pre}_resid1", f"{pre} Residual1 Add", "Residual", 0,
                [inp_act.id, o_proj_out.id], [resid1.id], "element-wise add")

        # LayerNorm 2
        make_op(f"{pre}_ln2", f"{pre} LayerNorm2", "LayerNorm", flops_ln,
                [resid1.id, ln2_w.id], [ln2_out.id], f"[{max_seq_len},{hidden_size}]")

        # FFN GEMM1 (up-projection)
        flops_ffn1 = max_seq_len * hidden_size * ffn_size * 2
        make_op(f"{pre}_ffn1", f"{pre} FFN1 GEMM", "GEMM", flops_ffn1,
                [ln2_out.id, ffn1_w.id], [ffn1_out.id],
                f"[{max_seq_len},{hidden_size}]x[{hidden_size},{ffn_size}]")

        # FFN GEMM2 (down-projection)
        flops_ffn2 = max_seq_len * ffn_size * hidden_size * 2
        make_op(f"{pre}_ffn2", f"{pre} FFN2 GEMM", "GEMM", flops_ffn2,
                [ffn1_out.id, ffn2_w.id], [ffn2_out.id],
                f"[{max_seq_len},{ffn_size}]x[{ffn_size},{hidden_size}]")

        # Residual Add 2
        make_op(f"{pre}_resid2", f"{pre} Residual2 Add", "Residual", 0,
                [resid1.id, ffn2_out.id], [layer_out.id], "element-wise add")

    # LM Head (最后一层输出投影到词表)
    lm_head_size = hidden_size * vocab_size * pbytes
    make_data("lm_head_weight", "LM Head Weight", DataType.WEIGHT, lm_head_size)
    final_in = datas[f"L{num_layers-1}_output_act"]
    final_out = make_data("logits", "Logits Output", DataType.OUTPUT,
                          max_seq_len * vocab_size * pbytes)
    flops_head = max_seq_len * hidden_size * vocab_size * 2
    make_op("lm_head", "LM Head", "GEMM", flops_head,
            [final_in.id, "lm_head_weight"], [final_out.id],
            f"[{max_seq_len},{hidden_size}]x[{hidden_size},{vocab_size}]")

    # 建立数据消费者关系
    for op in ops:
        for iid in op.input_ids:
            if iid in datas and op.id not in datas[iid].consumers:
                datas[iid].consumers.append(op.id)
        for oid in op.output_ids:
            if oid in datas:
                datas[oid].producer_op = op.id

    total_weight_bytes = sum(d.size_bytes for d in datas.values() if d.data_type == DataType.WEIGHT)
    total_flops = sum(o.flops for o in ops)

    return {
        "config": {"name": name, "num_layers": num_layers, "hidden_size": hidden_size,
                   "ffn_size": ffn_size, "num_heads": num_heads, "head_dim": head_dim,
                   "vocab_size": vocab_size, "max_seq_len": max_seq_len,
                   "precision": precision.name, "total_weight_gb": round(total_weight_bytes / 1e9, 2),
                   "total_flops": total_flops},
        "operators": ops,
        "data_objects": list(datas.values()),
    }


# ============================================================
# 预置模型
# ============================================================

QUERYABLE_MODELS = {}


def _register(name: str, builder_func, **kwargs):
    QUERYABLE_MODELS[name] = lambda: builder_func(name=name, **kwargs)


_register("llama7b", _build_llama_like,
          num_layers=32, hidden_size=4096, ffn_size=11008, num_heads=32, head_dim=128,
          vocab_size=32000, max_seq_len=2048)


def get_model(model_name: str) -> dict:
    """用户只需传名字: 'llama7b'"""
    if model_name not in QUERYABLE_MODELS:
        raise KeyError(f"Unknown model '{model_name}'. Available: {list(QUERYABLE_MODELS.keys())}")
    return QUERYABLE_MODELS[model_name]()


def list_models() -> list:
    return list(QUERYABLE_MODELS.keys())
