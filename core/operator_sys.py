"""
LLM-PIMSim v3 core.operator_sys — 【算子系统】

职责（"算子"这一概念的全部业务逻辑集中于此）：
  1. 18 类算子的固定规则表（category / data / execution）——单一事实来源
  2. 算子数据建模：Operator 由 common 提供，本系统负责"生成算子图"
     （kernel 粒度 → 可执行算子图）
  3. Workload 生成：内置模型维度 + kernel 粒度 workload（含 KV 动态 cost_fn）
  4. WorkloadAdapter：把 kernel 粒度 Workload 展开成可执行算子图

依赖：core.common（数据结构/枚举）+ core.precision（PrecisionLevel/OperatorCategory）。
不反向依赖：调度器 / 校验 / 输出 / 权重 / 切割 等系统。
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional

from core.common import (DataObject, DataType, Operator, OperatorCategory)
from core.precision import PrecisionLevel


# =================================================================
# 1. 18 类算子的固定配置（key = 算子名）
# =================================================================
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


def operator_rule_by_name(name: str) -> Optional[dict]:
    """按算子名查固定规则；未知名返回 None（不抛错，便于未匹配算子走默认）。"""
    return OPERATOR_PRECISION_RULES.get(name)


def apply_operator_rule(op, name: str):
    """把固定规则应用到任意对象，需该对象有 category/data_precision/execution_precision 三属性。"""
    rule = OPERATOR_PRECISION_RULES.get(name)
    if rule is None:
        return op
    op.category = rule["category"]
    op.data_precision = rule["data"]
    op.execution_precision = rule["execution"]
    return op


# 兼容矩阵（仅用于校验/展示，不作为运行时判定来源）
COMPATIBILITY_MATRIX = {
    "Embedding":     {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "LN1":           {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": False, "RERAM_PIM": False},
    "QKV_proj":      {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "RoPE":          {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": False, "RERAM_PIM": False},
    "KV_Cache_Write":{"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "KV_Cache_Read": {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "Attn_Score":    {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "Softmax":       {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": False, "RERAM_PIM": False},
    "Attn_Context":  {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "O_proj":        {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "Residual1":     {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": False, "RERAM_PIM": True},
    "LN2":           {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": False, "RERAM_PIM": False},
    "FFN_gate":      {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "FFN_up":        {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "SiLU":          {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": False, "RERAM_PIM": True},
    "FFN_down":      {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
    "Residual2":     {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": False, "RERAM_PIM": True},
    "LMHead":        {"CPU": True, "GPU": True, "SRAM_PIM": True, "DRAM_PIM": True, "RERAM_PIM": True},
}


# =================================================================
# 2. Kernel 类型（kernel 粒度算子）
# =================================================================
class KernelType(IntEnum):
    GEMM = 0            # Q/K/V/O proj, FFN gate/up/down, attn QK/AV
    LAYERNORM = 1
    SOFTMAX = 2
    ACTIVATION = 3      # SiLU/GELU/RoPE
    RESIDUAL = 4
    LMHEAD = 5
    EMBEDDING = 6
    KVCACHE_UPDATE = 7


@dataclass
class Cost:
    """成本统一 [min,max] 区间（定值即 min==max）。"""
    compute_flops_min: float = 0.0
    compute_flops_max: float = 0.0
    memory_bytes_min: float = 0.0
    memory_bytes_max: float = 0.0

    @staticmethod
    def fixed(compute: float, memory: float) -> "Cost":
        return Cost(compute, compute, memory, memory)

    @staticmethod
    def range(c_min, c_max, m_min, m_max) -> "Cost":
        return Cost(c_min, c_max, m_min, m_max)

    @property
    def is_fixed(self) -> bool:
        return (self.compute_flops_min == self.compute_flops_max
                and self.memory_bytes_min == self.memory_bytes_max)

    def glops_str(self) -> str:
        if self.is_fixed:
            return f"{self.compute_flops_min/1e9:.3f}G"
        return f"[{self.compute_flops_min/1e9:.3f}, {self.compute_flops_max/1e9:.3f}]G"

    def mb_str(self) -> str:
        if self.is_fixed:
            return f"{self.memory_bytes_min/1e6:.3f}MB"
        return f"[{self.memory_bytes_min/1e6:.3f}, {self.memory_bytes_max/1e6:.3f}]MB"


@dataclass
class KVCache:
    layer_id: int
    batch_size: int
    num_heads: int
    head_dim: int
    current_seq_len: int
    dtype_bytes: int = 2

    @property
    def size_bytes(self) -> int:
        return (2 * self.batch_size * self.num_heads * self.current_seq_len
                * self.head_dim * self.dtype_bytes)

    def at_seq(self, s: int) -> "KVCache":
        return KVCache(self.layer_id, self.batch_size, self.num_heads,
                       self.head_dim, s, self.dtype_bytes)

    def size_at(self, s: int) -> int:
        return self.at_seq(s).size_bytes


@dataclass
class Kernel:
    id: str
    name: str
    op_type: KernelType
    inputs: list
    intermediates: list
    outputs: list
    cost: Cost = None
    attributes: dict = field(default_factory=dict)
    required_precision: int = 4          # 默认 FP16（与 PrecisionLevel.FP16 对齐）
    cost_fn: Optional[Callable] = None   # 预留: cost_fn(kv_len, batch) -> (flops, mem)

    def to_dict(self) -> dict:
        _PRE_NAME = {1: "INT4", 2: "INT8", 3: "FP8", 4: "FP16", 5: "BF16", 6: "FP32"}
        # 精度取"真实执行精度"（与后端 Operator 一致，来自 OPERATOR_PRECISION_RULES）：
        # 否则前端显示 FP16 会误导推荐/校验（如 LN 实为 FP32，前端却显示 FP16 → 被推荐到 SRAM）。
        rule = operator_rule_by_name(self.name)
        data_prec = None
        exec_prec = None
        if rule is not None:
            data_prec = rule["data"]
            exec_prec = rule["execution"]   # 可能 None（KV_Cache 纯数据访问）
            shown = exec_prec if exec_prec is not None else data_prec
            prec_name = _PRE_NAME.get(shown.value, shown.name)
        else:
            prec_name = _PRE_NAME.get(self.required_precision, f"P{self.required_precision}")
        return {
            "id": self.id, "name": self.name, "op_type": self.op_type.name,
            "inputs": list(self.inputs), "intermediates": list(self.intermediates),
            "outputs": list(self.outputs),
            "attributes": self.attributes,
            "compute_flops_range": [self.cost.compute_flops_min, self.cost.compute_flops_max],
            "compute_gflops": self.cost.glops_str(),
            "memory_bytes_range": [self.cost.memory_bytes_min, self.cost.memory_bytes_max],
            "memory": self.cost.mb_str(),
            "is_kv_dependent": self.cost_fn is not None,
            "precision": prec_name,
            "data_precision": data_prec.name if data_prec is not None else None,
            "execution_precision": exec_prec.name if exec_prec is not None else None,
        }


# =================================================================
# 3. Prefill / Decode / LayerGroup / Workload
# =================================================================
@dataclass
class PrefillConfig:
    input_tokens: int = 1024
    batch_size: int = 1


@dataclass
class DecodeConfig:
    decode_steps: int = 128
    batch_size: int = 1


@dataclass
class LayerGroup:
    layer_start: int
    layer_end: int
    mapping_hint: str = ""

    @property
    def layer_range(self) -> List[int]:
        return list(range(self.layer_start, self.layer_end + 1))


@dataclass
class Workload:
    config: dict = field(default_factory=dict)
    kernels: list = field(default_factory=list)
    layers: list = field(default_factory=list)
    kv_range_bytes: tuple = (0, 0)
    layer_groups: list = field(default_factory=list)
    prefill: PrefillConfig = None
    decode: DecodeConfig = None
    decode_kv_start: int = 0
    decode_kv_end: int = 0

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def total_compute(self) -> tuple:
        cmin = sum(k.cost.compute_flops_min for k in self.kernels)
        cmax = sum(k.cost.compute_flops_max for k in self.kernels)
        return cmin, cmax

    def total_memory(self) -> tuple:
        mmin = sum(k.cost.memory_bytes_min for k in self.kernels)
        mmax = sum(k.cost.memory_bytes_max for k in self.kernels)
        return mmin, mmax

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "num_layers": self.num_layers,
            "layer_groups": [{"start": g.layer_start, "end": g.layer_end,
                              "hint": g.mapping_hint} for g in self.layer_groups],
            "kv_range_bytes": list(self.kv_range_bytes),
            "total_compute_flops_range": list(self.total_compute()),
            "total_memory_bytes_range": list(self.total_memory()),
            "decode_kv_len_range": [self.decode_kv_start, self.decode_kv_end],
            "layers": [
                [k.to_dict() for k in layer_kernels] for layer_kernels in self.layers
            ],
        }


# =================================================================
# 4. WorkloadBuilder —— kernel 粒度 workload 生成（含 KV 动态）
# =================================================================
class WorkloadBuilder:
    def __init__(self, hidden: int, ffn_size: int, num_heads: int, head_dim: int,
                 vocab: int, pbytes: int = 2):
        self.h = hidden
        self.f = ffn_size
        self.nh = num_heads
        self.hd = head_dim
        self.vocab = vocab
        self.b = pbytes

    # ---------- cost helpers ----------
    def _layernorm_compute(self, seq, batch):
        return seq * self.h * 6 * batch

    def _layernorm_mem(self, seq, batch):
        return 2 * seq * self.h * batch * self.b

    def _gemm_flops(self, m, n, k):
        return 2 * m * n * k

    def _make_layer_kernels(self, layer, seq, batch):
        """第 layer 层的 16 个算子模板（Embedding/LMHead 是全局算子，不在层内）。"""
        pre = f"L{layer}_"
        h, f, nh, hd, B = self.h, self.f, self.nh, self.hd, self.b
        m = batch * seq
        kernels = []

        kernels.append(Kernel(
            id=f"{pre}ln1", name="LN1", op_type=KernelType.LAYERNORM,
            inputs=[f"{pre}hidden_states"], intermediates=[], outputs=[f"{pre}ln1_out"],
            attributes={"seq": seq}))

        kernels.append(Kernel(
            id=f"{pre}qkv_proj", name="QKV_proj", op_type=KernelType.GEMM,
            inputs=[f"{pre}ln1_out", f"{pre}qkv_weight"], intermediates=[],
            outputs=[f"{pre}q", f"{pre}k", f"{pre}v"],
            attributes={"M": m, "K": h, "N": 3 * h, "triple": True}))

        kernels.append(Kernel(
            id=f"{pre}rope", name="RoPE", op_type=KernelType.ACTIVATION,
            inputs=[f"{pre}q", f"{pre}k"], intermediates=[],
            outputs=[f"{pre}q_rot", f"{pre}k_rot"], attributes={"kind": "rope", "seq": seq}))

        kernels.append(Kernel(
            id=f"{pre}kv_cache_write", name="KV_Cache_Write", op_type=KernelType.KVCACHE_UPDATE,
            inputs=[f"{pre}k_rot", f"{pre}v"], intermediates=[], outputs=[f"{pre}kv_cache"],
            attributes={"kv_len": "kv_len"}))

        kernels.append(Kernel(
            id=f"{pre}kv_cache_read", name="KV_Cache_Read", op_type=KernelType.KVCACHE_UPDATE,
            inputs=[f"{pre}kv_cache"], intermediates=[],
            outputs=[f"{pre}full_k", f"{pre}full_v"], attributes={"kv_len": "kv_len"}))

        kernels.append(Kernel(
            id=f"{pre}attn_score", name="Attn_Score", op_type=KernelType.GEMM,
            inputs=[f"{pre}q_rot", f"{pre}full_k"], intermediates=[f"{pre}attn_scores"],
            outputs=[], attributes={"M": m, "K": h, "N": "kv_len"}))

        kernels.append(Kernel(
            id=f"{pre}softmax", name="Softmax", op_type=KernelType.SOFTMAX,
            inputs=[f"{pre}attn_scores"], intermediates=[],
            outputs=[f"{pre}attn_probs"], attributes={"kv_len": "kv_len"}))

        kernels.append(Kernel(
            id=f"{pre}attn_context", name="Attn_Context", op_type=KernelType.GEMM,
            inputs=[f"{pre}attn_probs", f"{pre}full_v"], intermediates=[],
            outputs=[f"{pre}attn_context"], attributes={"M": m, "K": "kv_len", "N": h}))

        kernels.append(Kernel(
            id=f"{pre}o_proj", name="O_proj", op_type=KernelType.GEMM,
            inputs=[f"{pre}attn_context", f"{pre}o_proj_weight"], intermediates=[],
            outputs=[f"{pre}attn_output"], attributes={"M": m, "K": h, "N": h}))

        kernels.append(Kernel(
            id=f"{pre}resid1", name="Residual1", op_type=KernelType.RESIDUAL,
            inputs=[f"{pre}attn_output", f"{pre}hidden_states"], intermediates=[],
            outputs=[f"{pre}h1"], attributes={}))

        kernels.append(Kernel(
            id=f"{pre}ln2", name="LN2", op_type=KernelType.LAYERNORM,
            inputs=[f"{pre}h1"], intermediates=[], outputs=[f"{pre}ln2_out"],
            attributes={"seq": seq}))

        kernels.append(Kernel(
            id=f"{pre}ffn_gate", name="FFN_gate", op_type=KernelType.GEMM,
            inputs=[f"{pre}ln2_out", f"{pre}gate_weight"], intermediates=[],
            outputs=[f"{pre}gate_out"], attributes={"M": m, "K": h, "N": f}))

        kernels.append(Kernel(
            id=f"{pre}ffn_up", name="FFN_up", op_type=KernelType.GEMM,
            inputs=[f"{pre}ln2_out", f"{pre}up_weight"], intermediates=[],
            outputs=[f"{pre}up_out"], attributes={"M": m, "K": h, "N": f}))

        kernels.append(Kernel(
            id=f"{pre}ffn_silu", name="SiLU", op_type=KernelType.ACTIVATION,
            inputs=[f"{pre}gate_out", f"{pre}up_out"], intermediates=[],
            outputs=[f"{pre}silu_out"], attributes={"kind": "silu"}))

        kernels.append(Kernel(
            id=f"{pre}ffn_down", name="FFN_down", op_type=KernelType.GEMM,
            inputs=[f"{pre}silu_out", f"{pre}down_weight"], intermediates=[],
            outputs=[f"{pre}mlp_output"], attributes={"M": m, "K": f, "N": h}))

        kernels.append(Kernel(
            id=f"{pre}resid2", name="Residual2", op_type=KernelType.RESIDUAL,
            inputs=[f"{pre}mlp_output", f"{pre}h1"], intermediates=[],
            outputs=[f"{pre}layer_output"], attributes={}))
        return kernels

    def _bind_cost(self, k: Kernel, layer, seq, batch):
        h, f, nh, hd = self.h, self.f, self.nh, self.hd
        V = self.vocab
        m = batch * seq
        H4 = h * 4
        W8 = 1

        if k.op_type == KernelType.LAYERNORM:
            k.cost = Cost.fixed(5 * h, 2 * H4); k.cost_fn = None
        elif k.op_type == KernelType.RESIDUAL:
            k.cost = Cost.fixed(0, 2 * H4); k.cost_fn = None
        elif k.op_type == KernelType.ACTIVATION:
            if k.attributes.get("kind") == "rope":
                k.cost = Cost.fixed(6 * h, 2 * H4)
            else:
                k.cost = Cost.fixed(3 * f, 2 * f * 4)
            k.cost_fn = None
        elif "kv_cache_write" in k.id:
            k.cost = Cost.fixed(0, h); k.cost_fn = None
        elif "kv_cache_read" in k.id:
            def _kvr(kv):
                return (0, kv * nh * hd)
            k.cost_fn = _kvr; k.cost = None
        elif "attn_score" in k.id:
            def _qk(kv):
                return (2 * nh * hd * kv, (nh * hd * 4) + (kv * nh * hd * 0.5) + (kv * nh * 4))
            k.cost_fn = _qk; k.cost = None
        elif "softmax" in k.id:
            def _sm(kv):
                return (3 * nh * kv, 2 * (kv * nh * 4))
            k.cost_fn = _sm; k.cost = None
        elif "attn_context" in k.id:
            def _ac(kv):
                return (2 * nh * hd * kv, (kv * nh * 4) + (kv * nh * hd * 0.5) + (nh * hd * 4))
            k.cost_fn = _ac; k.cost = None
        elif "qkv_proj" in k.id:
            k.cost = Cost.fixed(3 * 2 * h * h, (m * h + 3 * h) * 4 + 3 * h * h * W8)
            k.cost_fn = None
        elif "ffn_down" in k.id:
            k.cost = Cost.fixed(2 * f * h, (m * f + h) * 4 + f * h * W8)
            k.cost_fn = None
        else:
            M = k.attributes.get("M", m); K = k.attributes.get("K", h); N = k.attributes.get("N", h)
            c = 2 * M * K * N
            mc = (M + N) * 4 + K * N * W8
            k.cost = Cost.fixed(c, mc); k.cost_fn = None
        return k

    def build(self, num_layers, input_tokens, generate_tokens, batch=1,
              layer_groups: Optional[List[LayerGroup]] = None) -> Workload:
        prefill_seq = input_tokens
        kv_start = input_tokens
        kv_end = input_tokens + generate_tokens

        template_kernels = self._make_layer_kernels(0, prefill_seq, batch)
        layers = []
        all_kernels = []
        prev_layer_output = "hidden_states"

        for L in range(num_layers):
            layer_kernels = []
            for tk in template_kernels:
                new_id = tk.id.replace("L0_", f"L{L}_")
                inputs = [i.replace("L0_", f"L{L}_") for i in tk.inputs]
                outputs = [o.replace("L0_", f"L{L}_") for o in tk.outputs]
                hidden_id = f"{prev_layer_output}" if L == 0 else prev_layer_output
                inputs = [hidden_id if (i == f"L{L}_hidden_states") else i for i in inputs]
                k = Kernel(
                    id=new_id, name=tk.name, op_type=tk.op_type,
                    inputs=inputs,
                    intermediates=[i.replace("L0_", f"L{L}_") for i in tk.intermediates],
                    outputs=outputs,
                    attributes=dict(tk.attributes),
                )
                k = self._bind_cost(k, L, prefill_seq, batch)
                if k.cost_fn is not None:
                    c0, m0 = k.cost_fn(kv_start)
                    c1, m1 = k.cost_fn(kv_end)
                    k.cost = Cost.range(min(c0, c1), max(c0, c1), min(m0, m1), max(m0, m1))
                layer_kernels.append(k)
            layers.append(layer_kernels)
            all_kernels.extend(layer_kernels)
            prev_layer_output = f"L{L}_layer_output"

        self._add_global_ops(all_kernels, num_layers, prefill_seq, batch, kv_start, kv_end)

        groups = layer_groups or [LayerGroup(1, num_layers)]
        kv_min = 2 * self.nh * self.hd * kv_start * self.b * batch * num_layers
        kv_max = 2 * self.nh * self.hd * kv_end * self.b * batch * num_layers

        wl = Workload(
            config={"hidden": self.h, "ffn_size": self.f, "num_heads": self.nh,
                    "head_dim": self.hd, "vocab": self.vocab,
                    "num_layers": num_layers, "pbytes": self.b,
                    "input_tokens": input_tokens, "generate_tokens": generate_tokens},
            kernels=all_kernels, layers=layers,
            kv_range_bytes=(kv_min, kv_max),
            layer_groups=groups,
            prefill=PrefillConfig(input_tokens=input_tokens, batch_size=batch),
            decode=DecodeConfig(decode_steps=generate_tokens, batch_size=batch),
            decode_kv_start=kv_start, decode_kv_end=kv_end,
        )
        return wl

    def _add_global_ops(self, kernels, num_layers, seq, batch, kv_start, kv_end):
        h, V, B = self.h, self.vocab, self.b
        m = batch * seq
        kernels.append(Kernel(
            id="embedding", name="Embedding", op_type=KernelType.EMBEDDING,
            inputs=["input_ids", "embed_weight"], intermediates=[], outputs=["hidden_states"],
            cost=Cost.fixed(0, m * h * 4), attributes={}))
        lm_flops = 2 * h * V
        lm_mem = (m * h + V) * 4 + h * V * 1
        kernels.append(Kernel(
            id="lm_head", name="LMHead", op_type=KernelType.LMHEAD,
            inputs=[f"L{num_layers-1}_layer_output", "lm_head_weight"], intermediates=[],
            outputs=["logits"],
            cost=Cost.fixed(lm_flops, lm_mem),
            attributes={"M": m, "K": h, "N": V}))


def build_model_workload(*, hidden, ffn_size, num_heads, head_dim, vocab,
                         num_layers, input_tokens, generate_tokens=None,
                         decode_steps=None, batch=1, pbytes=2, layer_groups=None):
    """一行构建完整 workload。

    generate_tokens: 自回归/Decode 步数（决定 KV 终点 = input_tokens + generate_tokens）。
    decode_steps: 兼容旧名；若给 generate_tokens 则优先用之，否则 fallback decode_steps。
    """
    if generate_tokens is None:
        generate_tokens = decode_steps if decode_steps is not None else 0
    builder = WorkloadBuilder(hidden=hidden, ffn_size=ffn_size, num_heads=num_heads,
                              head_dim=head_dim, vocab=vocab, pbytes=pbytes)
    return builder.build(num_layers, input_tokens, generate_tokens, batch, layer_groups)


# =================================================================
# 5. WorkloadAdapter —— kernel 粒度 → 可执行算子图
# =================================================================
_KERNEL_OP_TYPE = {
    KernelType.GEMM: "GEMM", KernelType.LAYERNORM: "LayerNorm",
    KernelType.SOFTMAX: "Softmax", KernelType.ACTIVATION: "Activation",
    KernelType.RESIDUAL: "Residual", KernelType.LMHEAD: "LMHead",
    KernelType.EMBEDDING: "Embedding", KernelType.KVCACHE_UPDATE: "KVCacheUpdate",
}


def _data_dtype_lookup(name: str) -> DataType:
    n = name.lower()
    if "_w" in n or "weight" in n:
        return DataType.WEIGHT
    if "kv" in n:
        return DataType.KV_CACHE
    if "logits" in n or "input_ids" in n:
        return DataType.OUTPUT
    return DataType.TEMPORARY


class WorkloadAdapter:
    """把 kernel 粒度 Workload 展开成可执行算子图（Prefill 单次 forward）。"""

    def __init__(self, workload: Workload):
        self.workload = workload

    def _data_sizes(self, kernel, per_data_mem_hint: dict = None) -> dict:
        ids = list(kernel.inputs) + list(kernel.intermediates) + list(kernel.outputs)
        # v3.2 KV 建模：KV 相关算子用 max（KV 终点即 input_tokens+generate_tokens 的规模），
        # 使 KV 增长真正反映到算子存储量；非 KV 算子 min==max。
        total = kernel.cost.memory_bytes_max if kernel.cost and kernel.cost_fn is not None \
            else (kernel.cost.memory_bytes_min if kernel.cost else 0)
        if not ids or total <= 0:
            return {d: 0 for d in ids}
        base = total / len(ids)
        return {d: base for d in ids}

    def build_executable(self) -> dict:
        wl = self.workload
        all_ops = []
        all_data = {}
        data_producer = {}
        data_consumers = {}

        def _register_data(did: str):
            if did not in all_data:
                all_data[did] = DataObject(
                    id=did, name=did, data_type=_data_dtype_lookup(did), size_bytes=0)

        def _register_op(op: Operator, k):
            all_ops.append(op)
            sizes = self._data_sizes(k)
            for d in op.input_ids:
                _register_data(d)
                all_data[d].size_bytes = max(all_data[d].size_bytes, int(sizes.get(d, 0)))
                data_consumers.setdefault(d, []).append(op.id)
            for d in k.intermediates:
                _register_data(d)
                all_data[d].size_bytes = max(all_data[d].size_bytes, int(sizes.get(d, 0)))
            for d in op.output_ids:
                _register_data(d)
                all_data[d].size_bytes = max(all_data[d].size_bytes, int(sizes.get(d, 0)))
                data_producer[d] = op.id

        def _do_kernel(k):
            op_type_str = _KERNEL_OP_TYPE.get(k.op_type, k.op_type.name)
            rule = operator_rule_by_name(k.name)
            # v3.2 KV 建模：KV 相关算子用 max（KV 终点 input_tokens+generate_tokens 的计算量），
            # 使 generate_tokens 直接影响算子 FLOPs；非 KV 算子 min==max。
            kflops = int(k.cost.compute_flops_max) if (k.cost_fn is not None) \
                else int(k.cost.compute_flops_min)
            if rule is not None:
                op = Operator(
                    id=k.id, name=k.name, op_type=op_type_str,
                    flops=kflops,
                    required_precision=(rule["execution"] if rule["execution"] is not None
                                        else rule["data"]),
                    category=rule["category"],
                    data_precision=rule["data"],
                    execution_precision=rule["execution"],
                    input_ids=list(k.inputs),
                    output_ids=list(k.outputs),
                    shape_desc=str(k.attributes),
                )
            else:
                cat = (OperatorCategory.LINEAR
                       if op_type_str in ("GEMM", "LMHead", "Embedding",
                                          "Residual", "KVCacheUpdate")
                       else OperatorCategory.NONLINEAR)
                req = PrecisionLevel.from_name(
                    {1: "INT4", 2: "INT8", 3: "FP8", 4: "FP16", 5: "BF16", 6: "FP32"}
                    .get(k.required_precision, 4))
                op = Operator(
                    id=k.id, name=k.name, op_type=op_type_str,
                    flops=kflops,
                    required_precision=req,
                    category=cat,
                    data_precision=req,
                    execution_precision=req,
                    input_ids=list(k.inputs),
                    output_ids=list(k.outputs),
                    shape_desc=str(k.attributes),
                )
            _register_op(op, k)

        for layer_kernels in wl.layers:
            for k in layer_kernels:
                _do_kernel(k)
        for k in wl.kernels:
            if k.id in ("embedding", "lm_head"):
                _do_kernel(k)

        return {
            "operators": all_ops,
            "operators_map": {op.id: op for op in all_ops},
            "data_objects": list(all_data.values()),
            "data_map": all_data,
        }
