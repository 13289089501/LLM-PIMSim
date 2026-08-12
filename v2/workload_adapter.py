"""
LLM-PIMSim v2 — workload_adapter
把 kernel 粒度的 Workload 展开成【可执行算子图】，供现有 Scheduler/Performance 消费。

职责:
  1. 把 workload 中每层的 kernel 转成可执行 Operator（id=原 kernel id，FLOPs 用 Prefill 代表值）
  2. 把 kernel 的 inputs/intermediates/outputs 注册成 DataObject（带 size）
  3. 输出 (operators, data_objects)，可喂给 SimulationEngine.build

阶段说明:
  当前展开 Prefill（一次完整 forward，定值 cost，用 cost.compute_flops_min）。
  Decode 的 KV 动态 range 保留在 Workload.to_dict()，执行层面暂用 Prefill 代表值。
"""
from contracts import DataObject, DataType, Operator, PrecisionLevel
from workload_model import Workload, KernelType


# kernel 类型 → 可执行算子 op_type 字符串（供 efficiency 查表 / mapping 匹配）
KERNEL_OP_TYPE = {
    KernelType.GEMM: "GEMM",
    KernelType.LAYERNORM: "LayerNorm",
    KernelType.SOFTMAX: "Softmax",
    KernelType.ACTIVATION: "Activation",
    KernelType.RESIDUAL: "Residual",
    KernelType.LMHEAD: "LMHead",
    KernelType.EMBEDDING: "Embedding",
    KernelType.KVCACHE_UPDATE: "KVCacheUpdate",
}

_PREC_VALID = {PrecisionLevel.INT4, PrecisionLevel.INT8,
               PrecisionLevel.FP16, PrecisionLevel.FP32}


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
    """把 Workload 转成可执行算子图（Prefill 单次 forward）。"""

    def __init__(self, workload: Workload):
        self.workload = workload

    def _data_sizes(self, kernel, per_data_mem_hint: dict = None) -> dict:
        """把 kernel 的 memory 总量分摊到各数据，作为 size 近似。"""
        ids = list(kernel.inputs) + list(kernel.intermediates) + list(kernel.outputs)
        total = kernel.cost.memory_bytes_min if kernel.cost else 0
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
                    id=did, name=did, data_type=_data_dtype_lookup(did), size_bytes=0,
                )

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
            op_type_str = KERNEL_OP_TYPE.get(k.op_type, k.op_type.name)
            rp = k.required_precision
            # Kernel.required_precision 是整数等级(1-4)，映射到 PrecisionLevel 枚举
            try:
                req_prec = PrecisionLevel(rp) if isinstance(rp, int) else PrecisionLevel.from_name(str(rp))
            except Exception:
                req_prec = PrecisionLevel.FP16
            op = Operator(
                id=k.id,
                name=k.name,
                op_type=op_type_str,
                flops=int(k.cost.compute_flops_min),   # Prefill 代表值
                required_precision=req_prec,
                input_ids=list(k.inputs),
                output_ids=list(k.outputs),
                shape_desc=str(k.attributes),
            )
            _register_op(op, k)

        # 遍历所有层 + 全局 kernel（embedding / lm_head 已在 wl.kernels）
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
