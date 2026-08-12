"""
LLM-PIMSim v2 — 模拟引擎（配置驱动薄壳）
引擎只接收"硬件图、算子图、映射、放置、数据源说明"这些数据，
不认识 YAML / GPU / FFN 等概念。build -> run -> SimulationResult。
"""
from contracts import SimulationResult
from scheduler import Scheduler
from performance import PerformanceModel


class SimulationEngine:
    def __init__(self):
        self.hw = {}
        self.interconnect = None
        self.ops = {}
        self.datas = {}
        self.compute_map = {}       # {op_id: compute_device}
        self.placement = {}         # {data_id: [devices]}
        self.input_specs = {}       # {op_id: [InputSpec]}
        self.perf = None

    def build(self, *, hardware_units, interconnect, operators,
              data_objects, compute_map, placement, input_specs=None,
              weight_shards=None):
        """全部 keyword-only。operators/data_objects 可以是 list 或 dict。"""
        self.hw = hardware_units
        self.interconnect = interconnect
        self.ops = {op.id: op for op in operators} if not isinstance(operators, dict) else dict(operators)
        self.datas = {d.id: d for d in data_objects} if not isinstance(data_objects, dict) else dict(data_objects)
        self.compute_map = dict(compute_map)
        self.placement = dict(placement)
        self.input_specs = dict(input_specs or {})
        self.weight_shards = dict(weight_shards or {})
        return self

    def run(self, seed=42) -> SimulationResult:
        self.perf = PerformanceModel(self.hw, self.interconnect)
        notes = []

        sched = Scheduler(
            operators=self.ops,
            data_objects=self.datas,
            hardware_units=self.hw,
            interconnect=self.interconnect,
            perf_model=self.perf,
            compute_map=self.compute_map,
            input_specs=self.input_specs,
            placement=self.placement,
            note_collector=notes,
            weight_shards=self.weight_shards,
        )
        sched.run()
        raw = sched.collect_result()

        result = SimulationResult(
            metadata={
                "hardware": [h.name for h in self.hw.values()],
                "seed": seed,
            },
            total_latency_ns=raw["total_latency_ns"],
            breakdown=raw["breakdown"],
            bottleneck=raw["bottleneck"],
            bottleneck_rationale=raw["rationale"],
            operator_timings=raw["op_timings"],
            event_trace=raw["event_trace"],
        )
        for n in notes:
            result.add_note(n)
        return result
