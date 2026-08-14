"""
LLM-PIMSim v2 — kernel 粒度 workload model（算子定义层）
========================================================
纯 workload，不含硬件位置/执行策略。Placement(数据放哪) 与 Mapping
(算子在哪算) 由外部独立配置决定。

关键设计（满足需求）:
  1. 算子粒度细到 hardware kernel:
     Attention 拆 QK^T / Softmax / QK·V
     FFN 拆 gate / up / act(SiLU) / down（SwiGLU）
     LMHead / Embedding / KVCacheUpdate 独立
  2. 统一 Kernel 结构: inputs/intermediates/outputs + cost + attributes
  3. Cost 统一 [min,max] 区间（定值即 min==max）
  4. KVCache 动态建模
  5. Decode 不展开逐 token:
     - 每个 kernel 模板保存 cost_fn(kv_len) 关系式（若随 KV 变）
     - workload range = 用 KV 起点/终点长度代入 cost_fn 求 [min,max]
     - 底层预留 cost_fn 扩展位（未来逐 token 精确动态）
  6. Prefill(定值) / Decode(范围) 两阶段
  7. LayerGroup: 层模板一次定义，可分层段绑不同 mapping
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional


# ============================================================
# 算子类型（kernel 粒度）
# ============================================================
class KernelType(IntEnum):
    GEMM = 0            # 矩阵乘: Q/K/V/O proj, FFN gate/up/down, attn QK/AV
    LAYERNORM = 1
    SOFTMAX = 2
    ACTIVATION = 3      # SiLU/GELU
    RESIDUAL = 4        # elementwise add（有 memory cost）
    LMHEAD = 5          # 独立 LMHead
    EMBEDDING = 6
    KVCACHE_UPDATE = 7  # 新 K/V 追加到 KV Cache


# ============================================================
# Cost：统一 [min,max] 区间，定值即 min==max
# ============================================================
@dataclass
class Cost:
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


# ============================================================
# KVCache
# ============================================================
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
        """一层 KV: 2(K,V) × batch × heads × seq_len × head_dim × dtype"""
        return (2 * self.batch_size * self.num_heads * self.current_seq_len
                * self.head_dim * self.dtype_bytes)

    def at_seq(self, s: int) -> "KVCache":
        return KVCache(self.layer_id, self.batch_size, self.num_heads,
                       self.head_dim, s, self.dtype_bytes)

    def size_at(self, s: int) -> int:
        return self.at_seq(s).size_bytes


# ============================================================
# Kernel 统一结构
# ============================================================
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
    required_precision: int = 3          # FP16
    cost_fn: Optional[Callable] = None   # 预留: cost_fn(kv_len, batch)->(flops,mem) 逐 token 动态求值

    def to_dict(self) -> dict:
        # 精度名称映射：1=INT4, 2=INT8, 3=FP16, 4=FP32
        _PRE_NAME = {1: "INT4", 2: "INT8", 3: "FP16", 4: "FP32"}
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
        }


# ============================================================
# 阶段配置
# ============================================================
@dataclass
class PrefillConfig:
    input_tokens: int = 1024
    batch_size: int = 1


@dataclass
class DecodeConfig:
    decode_steps: int = 128
    batch_size: int = 1


# ============================================================
# LayerGroup: 层段分组（1-based）
# ============================================================
@dataclass
class LayerGroup:
    layer_start: int
    layer_end: int
    mapping_hint: str = ""

    @property
    def layer_range(self) -> List[int]:
        return list(range(self.layer_start, self.layer_end + 1))


# ============================================================
# Workload
# ============================================================
@dataclass
class Workload:
    config: dict = field(default_factory=dict)
    kernels: list = field(default_factory=list)
    layers: list = field(default_factory=list)
    kv_range_bytes: tuple = (0, 0)       # (min,max) 全模型 KV 总量
    layer_groups: list = field(default_factory=list)
    prefill: PrefillConfig = None
    decode: DecodeConfig = None
    # Decode 相关的 KV 起点/终点序列（供 GUI/分析）
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


# ============================================================
# WorkloadBuilder — 生成 kernel 粒度 workload，解 Decode 动态 range
# ============================================================
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

    # ---------- 构建单层模板 ----------
    def _make_layer_kernels(self, layer, seq, batch):
        """构建第 layer 层模板，包含 16 个算子（Embedding/LMHead 是全局算子，不在层内）。
        建立层内算子间的数据依赖：中间张量作为前一算子 output 与后一算子 input 串联。
        对 KV 相关算子（Attn_Score/Softmax/Attn_Context/KV_Cache_Read）绑定 cost_fn(kv_len)。
        """
        pre = f"L{layer}_"
        h, f, nh, hd, B = self.h, self.f, self.nh, self.hd, self.b
        m = batch * seq          # 行数（Prefill: seq; Decode Q 侧=1，见下）
        kernels = []

        # 1) LN1: hidden_states -> ln1_out
        kernels.append(Kernel(
            id=f"{pre}ln1", name="LN1", op_type=KernelType.LAYERNORM,
            inputs=[f"{pre}hidden_states"], intermediates=[],
            outputs=[f"{pre}ln1_out"], attributes={"seq": seq},
        ))

        # 2) QKV_proj：合并 Q/K/V 三次投影（读 qkv_weight）-> q, k, v
        #    N=3*hidden（与方案一致，供算子切割沿 N 切出 Q/K/V）；cost 由 _bind_cost 的 qkv 分支固定 3·(2·h·h)
        kernels.append(Kernel(
            id=f"{pre}qkv_proj", name="QKV_proj", op_type=KernelType.GEMM,
            inputs=[f"{pre}ln1_out", f"{pre}qkv_weight"], intermediates=[],
            outputs=[f"{pre}q", f"{pre}k", f"{pre}v"],
            attributes={"M": m, "K": h, "N": 3 * h, "triple": True},
        ))

        # 3) RoPE：对 q/k 施加旋转位置编码 -> q_rot, k_rot
        #    提供 seq 供算子切割沿序列维分片（cost 由 _bind_cost 按 kind=rope 固定 6*h）
        kernels.append(Kernel(
            id=f"{pre}rope", name="RoPE", op_type=KernelType.ACTIVATION,
            inputs=[f"{pre}q", f"{pre}k"], intermediates=[],
            outputs=[f"{pre}q_rot", f"{pre}k_rot"], attributes={"kind": "rope", "seq": seq},
        ))

        # 4) KV_Cache_Write：把当前步 K/V 追加进 kv_cache（固定增量）
        kernels.append(Kernel(
            id=f"{pre}kv_cache_write", name="KV_Cache_Write", op_type=KernelType.KVCACHE_UPDATE,
            inputs=[f"{pre}k_rot", f"{pre}v"], intermediates=[], outputs=[f"{pre}kv_cache"],
            attributes={"kv_len": "kv_len"},
        ))

        # 5) KV_Cache_Read：从 kv_cache 读出全部历史 K/V（含当前步）-> full_k, full_v
        kernels.append(Kernel(
            id=f"{pre}kv_cache_read", name="KV_Cache_Read", op_type=KernelType.KVCACHE_UPDATE,
            inputs=[f"{pre}kv_cache"], intermediates=[],
            outputs=[f"{pre}full_k", f"{pre}full_v"], attributes={"kv_len": "kv_len"},
        ))

        # 6) Attn_Score = Q·K^T（依赖 kv_len）
        kernels.append(Kernel(
            id=f"{pre}attn_score", name="Attn_Score", op_type=KernelType.GEMM,
            inputs=[f"{pre}q_rot", f"{pre}full_k"], intermediates=[f"{pre}attn_scores"],
            outputs=[], attributes={"M": m, "K": h, "N": "kv_len"},
        ))

        # 7) Softmax（依赖 kv_len）
        kernels.append(Kernel(
            id=f"{pre}softmax", name="Softmax", op_type=KernelType.SOFTMAX,
            inputs=[f"{pre}attn_scores"], intermediates=[], outputs=[f"{pre}attn_probs"],
            attributes={"kv_len": "kv_len"},
        ))

        # 8) Attn_Context = P·V（依赖 kv_len）
        kernels.append(Kernel(
            id=f"{pre}attn_context", name="Attn_Context", op_type=KernelType.GEMM,
            inputs=[f"{pre}attn_probs", f"{pre}full_v"], intermediates=[],
            outputs=[f"{pre}attn_context"], attributes={"M": m, "K": "kv_len", "N": h},
        ))

        # 9) O_proj
        kernels.append(Kernel(
            id=f"{pre}o_proj", name="O_proj", op_type=KernelType.GEMM,
            inputs=[f"{pre}attn_context", f"{pre}o_proj_weight"], intermediates=[],
            outputs=[f"{pre}attn_output"], attributes={"M": m, "K": h, "N": h},
        ))

        # 10) Residual1: attn_output + hidden_states -> h1
        kernels.append(Kernel(
            id=f"{pre}resid1", name="Residual1", op_type=KernelType.RESIDUAL,
            inputs=[f"{pre}attn_output", f"{pre}hidden_states"], intermediates=[],
            outputs=[f"{pre}h1"], attributes={},
        ))

        # 11) LN2
        kernels.append(Kernel(
            id=f"{pre}ln2", name="LN2", op_type=KernelType.LAYERNORM,
            inputs=[f"{pre}h1"], intermediates=[], outputs=[f"{pre}ln2_out"],
            attributes={"seq": seq},
        ))

        # 12-13) FFN gate / up（并行）
        kernels.append(Kernel(
            id=f"{pre}ffn_gate", name="FFN_gate", op_type=KernelType.GEMM,
            inputs=[f"{pre}ln2_out", f"{pre}gate_weight"], intermediates=[],
            outputs=[f"{pre}gate_out"], attributes={"M": m, "K": h, "N": f},
        ))
        kernels.append(Kernel(
            id=f"{pre}ffn_up", name="FFN_up", op_type=KernelType.GEMM,
            inputs=[f"{pre}ln2_out", f"{pre}up_weight"], intermediates=[],
            outputs=[f"{pre}up_out"], attributes={"M": m, "K": h, "N": f},
        ))

        # 14) SiLU: gate_out * silu(up_out) -> silu_out
        kernels.append(Kernel(
            id=f"{pre}ffn_silu", name="SiLU", op_type=KernelType.ACTIVATION,
            inputs=[f"{pre}gate_out", f"{pre}up_out"], intermediates=[],
            outputs=[f"{pre}silu_out"], attributes={"kind": "silu"},
        ))

        # 15) FFN_down
        kernels.append(Kernel(
            id=f"{pre}ffn_down", name="FFN_down", op_type=KernelType.GEMM,
            inputs=[f"{pre}silu_out", f"{pre}down_weight"], intermediates=[],
            outputs=[f"{pre}mlp_output"], attributes={"M": m, "K": f, "N": h},
        ))

        # 16) Residual2: mlp_output + h1 -> layer_output（跨层连接用）
        kernels.append(Kernel(
            id=f"{pre}resid2", name="Residual2", op_type=KernelType.RESIDUAL,
            inputs=[f"{pre}mlp_output", f"{pre}h1"], intermediates=[],
            outputs=[f"{pre}layer_output"], attributes={},
        ))

        # 层内算子到此为止；Embedding / LMHead 放全局
        return kernels

    # ---------- 绑定 cost 到每个 kernel ----------
    # 成本遵循 LLaMA-7B decode(batch=1,seq=1) 常数表 + 4 个 KV 动态 cost_fn。
    # 变量: h, f, V(vocab), nh, hd；字节约定: 激活 FP32=4B、KV INT4=0.5B、INT8 权重=1B。
    def _bind_cost(self, k: Kernel, layer, seq, batch):
        h, f, nh, hd = self.h, self.f, self.nh, self.hd
        V = self.vocab
        m = batch * seq                 # decode(seq=1) 时 m=1
        H4 = h * 4
        W8 = 1                          # INT8 权重字节

        if k.op_type == KernelType.LAYERNORM:
            # LN: FLOPs=5*h；mem=(in+out) 两个 hidden ×4
            k.cost = Cost.fixed(5 * h, 2 * H4)
            k.cost_fn = None
        elif k.op_type == KernelType.RESIDUAL:
            k.cost = Cost.fixed(0, 2 * H4)   # 两个输入各一个 hidden
            k.cost_fn = None
        elif k.op_type == KernelType.ACTIVATION:
            # RoPE: FLOPs=6*h(处理 q,k two)，mem=2*h×4；SiLU: FLOPs=3*f，mem=(f+f)×4
            if k.attributes.get("kind") == "rope":
                k.cost = Cost.fixed(6 * h, 2 * H4)
            else:
                k.cost = Cost.fixed(3 * f, 2 * f * 4)
            k.cost_fn = None
        elif "kv_cache_write" in k.id:
            # 固定增量：当前步 K/V 追加，不随 kv_len 增长。FLOPs=0，mem = 2*h*0.5 = h
            k.cost = Cost.fixed(0, h)
            k.cost_fn = None
        elif "kv_cache_read" in k.id:
            # FLOPs=0；memory = kv_len*nh*hd*2*0.5 = kv_len*nh*hd（K/V 各半，INT4）
            def _kvr(kv):
                return (0, kv * nh * hd)
            k.cost_fn = _kvr; k.cost = None
        elif "attn_score" in k.id:
            # FLOPs = 2*nh*hd*kv；mem = Q(nh*hd*4)+K(kv*nh*hd*0.5)+out(kv*nh*4)
            def _qk(kv):
                return (2 * nh * hd * kv, (nh * hd * 4) + (kv * nh * hd * 0.5) + (kv * nh * 4))
            k.cost_fn = _qk; k.cost = None
        elif "softmax" in k.id:
            # FLOPs = 3*nh*kv；mem = in(kv*nh*4)+out(kv*nh*4)
            def _sm(kv):
                return (3 * nh * kv, 2 * (kv * nh * 4))
            k.cost_fn = _sm; k.cost = None
        elif "attn_context" in k.id:
            # FLOPs = 2*nh*hd*kv；mem = probs(kv*nh*4)+V(kv*nh*hd*0.5)+out(nh*hd*4)
            def _ac(kv):
                return (2 * nh * hd * kv, (kv * nh * 4) + (kv * nh * hd * 0.5) + (nh * hd * 4))
            k.cost_fn = _ac; k.cost = None
        elif "qkv_proj" in k.id:
            # QKV_proj: FLOPs=3*(2*h*h)；mem=(m*h 输入 + 3*h 输出)*4 + 权重(3*h*h)*1
            k.cost = Cost.fixed(3 * 2 * h * h, (m * h + 3 * h) * 4 + 3 * h * h * W8)
            k.cost_fn = None
        elif "ffn_down" in k.id:
            # FFN_down: FLOPs=2*f*h；mem=(m*f 输入 + h 输出)*4 + 权重(f*h)*1
            k.cost = Cost.fixed(2 * f * h, (m * f + h) * 4 + f * h * W8)
            k.cost_fn = None
        else:  # 其余 GEMM：O_proj / FFN_gate / FFN_up，按 attributes M/K/N
            M = k.attributes.get("M", m)
            K = k.attributes.get("K", h)
            N = k.attributes.get("N", h)
            c = 2 * M * K * N
            mc = (M + N) * 4 + K * N * W8
            k.cost = Cost.fixed(c, mc)
            k.cost_fn = None
        return k

    # ---------- 构建整个 workload ----------
    def build(self, num_layers, input_tokens, decode_steps, batch=1,
              layer_groups: Optional[List[LayerGroup]] = None) -> Workload:
        prefill_seq = input_tokens
        kv_start = input_tokens            # Decode 起点 KV 长
        kv_end = input_tokens + decode_steps   # Decode 终点 KV 长（decode_steps 次增长）

        # 层模板 kernel 结构（用 prefill seq 建形）
        template_kernels = self._make_layer_kernels(0, prefill_seq, batch)

        layers = []
        all_kernels = []
        # 跨层隐藏状态 id：L0 = Embedding 输出的全局 hidden_states；
        # L>0 = 上一层 Res2 的 layer_output（同一 DataObject，形成跨层数据依赖链）。
        prev_layer_output = "hidden_states"

        for L in range(num_layers):
            layer_kernels = []
            for tk in template_kernels:
                # 复制到当前层
                new_id = tk.id.replace("L0_", f"L{L}_")
                inputs = [i.replace("L0_", f"L{L}_") for i in tk.inputs]
                outputs = [o.replace("L0_", f"L{L}_") for o in tk.outputs]
                # 跨层输入替换：LN1(input[0])、Residual1(input[1]) 消费的 hidden_states
                # 改为真正的隐藏状态来源（L0=Embedding 输出，L>0=上一层 layer_output）。
                hidden_id = f"{prev_layer_output}" if L == 0 else prev_layer_output
                inputs = [hidden_id if (i == f"L{L}_hidden_states") else i for i in inputs]
                k = Kernel(
                    id=new_id, name=tk.name, op_type=tk.op_type,
                    inputs=inputs,
                    intermediates=[i.replace("L0_", f"L{L}_") for i in tk.intermediates],
                    outputs=outputs,
                    attributes=dict(tk.attributes),
                )
                # 绑定 cost + 对 KV 算子求 kv_start/kv_end 范围
                k = self._bind_cost(k, L, prefill_seq, batch)
                if k.cost_fn is not None:
                    c0, m0 = k.cost_fn(kv_start)
                    c1, m1 = k.cost_fn(kv_end)
                    k.cost = Cost.range(
                        min(c0, c1), max(c0, c1),
                        min(m0, m1), max(m0, m1))
                layer_kernels.append(k)
            layers.append(layer_kernels)
            all_kernels.extend(layer_kernels)
            # 本层输出作为下一层输入 hidden_states
            prev_layer_output = f"L{L}_layer_output"

        # LMHead / Embedding 全局算子（随 vocab，不随层）
        self._add_global_ops(all_kernels, num_layers, prefill_seq, batch, kv_start, kv_end)

        groups = layer_groups or [LayerGroup(1, num_layers)]

        # KV 范围（全模型，跨层累计）
        kv_min = 2 * self.nh * self.hd * kv_start * self.b * batch * num_layers
        kv_max = 2 * self.nh * self.hd * kv_end * self.b * batch * num_layers

        wl = Workload(
            config={"hidden": self.h, "ffn_size": self.f, "num_heads": self.nh,
                    "head_dim": self.hd, "vocab": self.vocab,
                    "num_layers": num_layers, "pbytes": self.b,
                    "input_tokens": input_tokens, "decode_steps": decode_steps},
            kernels=all_kernels, layers=layers,
            kv_range_bytes=(kv_min, kv_max),
            layer_groups=groups,
            prefill=PrefillConfig(input_tokens=input_tokens, batch_size=batch),
            decode=DecodeConfig(decode_steps=decode_steps, batch_size=batch),
            decode_kv_start=kv_start, decode_kv_end=kv_end,
        )
        return wl

    def _add_global_ops(self, kernels, num_layers, seq, batch, kv_start, kv_end):
        """追加全局算子: Embedding + LMHead（随 vocab，不随 KV 变）。"""
        h, V, B = self.h, self.vocab, self.b
        m = batch * seq
        # Embedding: FLOPs=0，mem≈输入 token 的 hidden 激活读（FP32）
        kernels.append(Kernel(
            id="embedding", name="Embedding", op_type=KernelType.EMBEDDING,
            inputs=["input_ids", "embed_weight"], intermediates=[], outputs=["hidden_states"],
            cost=Cost.fixed(0, m * h * 4), attributes={},
        ))
        # LMHead: FLOPs=2*h*V；mem=(m*h 输入 + V 输出)*4 + h*V 权重
        lm_flops = 2 * h * V
        lm_mem = (m * h + V) * 4 + h * V * 1
        kernels.append(Kernel(
            id="lm_head", name="LMHead", op_type=KernelType.LMHEAD,
            inputs=[f"L{num_layers-1}_layer_output", "lm_head_weight"], intermediates=[],
            outputs=["logits"],
            cost=Cost.fixed(lm_flops, lm_mem),
            attributes={"M": m, "K": h, "N": V},
        ))


# 便捷入口
def build_model_workload(*, hidden, ffn_size, num_heads, head_dim, vocab,
                         num_layers, input_tokens, decode_steps, batch=1,
                         pbytes=2, layer_groups=None):
    """一行构建完整 workload。"""
    builder = WorkloadBuilder(hidden=hidden, ffn_size=ffn_size, num_heads=num_heads,
                              head_dim=head_dim, vocab=vocab, pbytes=pbytes)
    return builder.build(num_layers, input_tokens, decode_steps, batch, layer_groups)


# ============================================================
# 算子切割（kernel splitting）
# 用户可以对算子的可切割维度参数（M/K/N/seq 等）进行切分，
# 例如把 N=20 切成 N=5 和 N=15。切割后：
#   - 计算量 compute_flops 按该维度比例缩放（compute = 2*M*N*K，线性于每维）
#   - 存储 memory 近似按比例缩放（读/写数据量随维度线性）
#   - 精度 precision、算子类型 op_type、输入/输出结构保持不变
# ============================================================
def split_kernel_dict(kernel: dict, dim: str, parts: list) -> list:
    """把单个序列化 kernel dict 沿 dim 切成 parts 指定的若干段。

    参数:
      kernel: 来自 /api/workload 的 kernel dict（含 attributes 和成本字段）
      dim:   切割维度名，必须是 attributes 里的键（如 'N','K','M','seq'）
      parts: 每段的新值列表，如 [5,15] 表示切成 N=5 与 N=15 两段
             所有段之和应等于原值（不做强制校验，允许用户自由）。
    返回:
      新的 kernel dict 列表（每段一个），id 加段序号后缀。
    """
    attrs = dict(kernel.get("attributes", {}))
    if dim not in attrs:
        raise ValueError(f"维度 '{dim}' 不在算子参数中，可切割维度: {list(attrs.keys())}")
    original = attrs[dim]
    # 允许 numeric 或 'kv_len' 字符串（KV 相关维度通常写 'kv_len'）
    if isinstance(original, str):
        if original.lower() == "kv_len":
            # KV 相关的动态维度：用 decode 终值近似作为参考（前端会传具体 parts 值）
            ref_total = sum(float(p) for p in parts)
            original_numeric = ref_total
            ratio_base = ref_total
        else:
            raise ValueError(f"维度 '{dim}' 的原始值 '{original}' 无法切割")
    else:
        original_numeric = float(original)
        ratio_base = original_numeric

    total_parts = sum(float(p) for p in parts)
    result = []
    base_flops = kernel.get("compute_flops_range", [0, 0])
    base_mem = kernel.get("memory_bytes_range", [0, 0])
    base_flops_lo, base_flops_hi = base_flops
    base_mem_lo, base_mem_hi = base_mem
    # 若成本是字符串（compute_gflops 等）则从原数值提取
    base_flops_lo = float(base_flops_lo); base_flops_hi = float(base_flops_hi)
    base_mem_lo = float(base_mem_lo); base_mem_hi = float(base_mem_hi)

    for i, p in enumerate(parts):
        ratio = float(p) / ratio_base if ratio_base and total_parts else 1.0
        new_attrs = dict(attrs)
        new_attrs[dim] = original if isinstance(original, str) else float(p)
        new_flops = base_flops_lo * ratio, base_flops_hi * ratio
        new_mem = base_mem_lo * ratio, base_mem_hi * ratio
        # 复算子精度、类型、输入输出结构
        new_kernel = dict(kernel)
        new_kernel["id"] = f"{kernel['id']}[{i+1}]"
        new_kernel["name"] = f"{kernel.get('name', kernel['id'])}#{i+1}"
        new_kernel["attributes"] = new_attrs
        new_kernel["compute_flops_range"] = [new_flops[0], new_flops[1]]
        new_kernel["memory_bytes_range"] = [new_mem[0], new_mem[1]]
        # 重新格式化成本字符串
        def _gstr(a, b):
            if a == b:
                return f"{a/1e9:.3f}G"
            return f"[{a/1e9:.3f}, {b/1e9:.3f}]G"
        def _mstr(a, b):
            if a == b:
                return f"{a/1e6:.3f}MB"
            return f"[{a/1e6:.3f}, {b/1e6:.3f}]MB"
        new_kernel["compute_gflops"] = _gstr(new_flops[0], new_flops[1])
        new_kernel["memory"] = _mstr(new_mem[0], new_mem[1])
        # 精度、op_type、inputs、intermediates、outputs、is_kv_dependent 保持原样
        result.append(new_kernel)
    return result

