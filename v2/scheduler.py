"""
LLM-PIMSim v2 — 离散事件调度器（配置驱动版）
相对 v1 的关键升级:
  1. 数据冗余多设备驻留: 一份数据可在多个设备各有副本，各自独立就绪
  2. 用户决定数据源: op 的每个输入, 用户 pinned(from=xxx) 严格生效;
     用户没写(auto) 用"就近参考"(min 就绪+搬运耗时)
  3. 告警: 用户指定的源不可用/未就绪 → 记录 note, 不静默更改
  4. 预留分片: schedule 感知 num_shards/多设备, 但第一版不实现切割算法
其余(事件队列/主循环/状态表)保持 v1 骨架。
"""
import heapq
from contracts import (
    Event, EventType, Operator, DataObject, OperatorTiming,
    OpState, DataType, DurationBreakdown, InputSpec, OperatorSpec,
    DataSourceMode,
)


class EventQueue:
    """优先队列 —— 按 start_time 升序，同时间按拓扑序"""

    def __init__(self):
        self._heap = []
        self._counter = 0

    def push(self, event: Event):
        heapq.heappush(self._heap, (event.start_time_ns, self._counter, event))
        self._counter += 1

    def pop(self) -> Event:
        return heapq.heappop(self._heap)[2]

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self):
        return len(self._heap)


class DataStateTable:
    """
    数据状态表 v2 —— 支持冗余多设备驻留
    内部: self._by_device: {data_id: {device_id: ready_time_ns}}
    """
    def __init__(self):
        self._by_device: dict[str, dict[str, int]] = {}

    def place_multi(self, obj_id: str, devices: list, ready_time_ns: int = 0):
        """一份数据在多个设备驻留，各副本初始就绪时间相同"""
        entry = self._by_device.setdefault(obj_id, {})
        for dev in devices:
            if dev:
                entry[dev] = ready_time_ns

    def set_ready(self, obj_id: str, device_id: str, ready_time_ns: int):
        """某设备上的副本就绪"""
        self._by_device.setdefault(obj_id, {})[device_id] = ready_time_ns

    def copies(self, obj_id: str) -> dict:
        """返回 {device_id: ready_time_ns} 当前所有驻留副本"""
        return self._by_device.get(obj_id, {})

    def has_copy(self, obj_id: str, device_id: str) -> bool:
        return device_id in self._by_device.get(obj_id, {})

    def ready_time_on(self, obj_id: str, device_id: str) -> int:
        return self._by_device.get(obj_id, {}).get(device_id, -1)

    def is_any_ready(self, obj_id: str) -> bool:
        """是否至少有一份副本就绪过（就绪时间>=0）"""
        copies = self._by_device.get(obj_id, {})
        return any(t >= 0 for t in copies.values())

    def earliest_ready(self, obj_id: str) -> int:
        """所有副本中最早就绪时间（用于依赖判定）"""
        copies = self._by_device.get(obj_id, {})
        good = [t for t in copies.values() if t >= 0]
        return min(good) if good else -1


class ResourceStateTable:
    def __init__(self):
        self.busy_until: dict[str, int] = {}
        self.current_task: dict[str, str] = {}

    def mark_busy(self, hw_id: str, until_ns: int, task: str):
        self.busy_until[hw_id] = until_ns
        self.current_task[hw_id] = task

    def free_at(self, hw_id: str) -> int:
        return self.busy_until.get(hw_id, 0)


class Scheduler:
    """
    配置驱动调度核心。
    不做"该把算子放哪"的决策 —— 计算设备完全来自 compute_map(用户决定)。
    数据源: pinned(from=xxx) 严格按用户指定; auto 用就近参考并记录。
    """

    def __init__(self, operators: dict, data_objects: dict,
                 hardware_units: dict, interconnect,
                 perf_model, compute_map: dict, 
                 input_specs: dict = None, placement: dict = None,
                 note_collector: list = None,
                 weight_shards: dict = None):
        self.operators = operators          # {id: Operator}
        self.data_objects = data_objects    # {id: DataObject}
        self.hw = hardware_units            # {id: HardwareUnit}
        self.interconnect = interconnect
        self.perf = perf_model
        self.compute_map = compute_map      # {op_id: compute_device}（用户决定）
        self.input_specs = input_specs or {}  # {op_id: list[InputSpec]}
        self.placement = placement or {}    # {data_id: [devices]} 初始驻留
        self.notes = note_collector if note_collector is not None else []
        # 权重分片表: {weight_data_id: {partition_id: device_id}}
        # 需要该权重的算子必须"读齐全部切分片"（ALL-GATHER）才允许运行。
        self.weight_shards = weight_shards or {}

        # 运行态
        self.data_state = DataStateTable()
        self.resource_state = ResourceStateTable()
        self.queue = EventQueue()
        self.clock_ns = 0

        self.op_states: dict[str, OpState] = {}
        self.event_trace: list = []
        self.op_timings: list = []
        self._next_event_id = 0

    def _new_event(self, etype, start, end, op_id="", hw_id="", component="") -> Event:
        ev = Event(id=self._next_event_id, event_type=etype,
                   start_time_ns=start, end_time_ns=end,
                   operator_id=op_id, resource_id=hw_id, component=component)
        self._next_event_id += 1
        return ev

    def _note(self, msg: str):
        self.notes.append(f"[t={self.clock_ns}ns] {msg}")

    # ─── 初始化 ───
    def initialize(self):
        for oid, op in self.operators.items():
            self.op_states[oid] = OpState.WAITING
        # 按 placement 冗余多设备驻留
        for did, dobj in self.data_objects.items():
            devs = self.placement.get(did, [])
            if devs:
                self.data_state.place_multi(did, devs, 0)
                dobj.replica_locations = list(devs)
                dobj.location = devs[0] if devs else ""

    # ─── 主循环 ───
    def run(self):
        self.initialize()
        self._check_ready_ops()
        while not self.queue.is_empty():
            event = self.queue.pop()
            self.clock_ns = event.end_time_ns
            self.event_trace.append(event)
            self._process_event(event)
            self._check_ready_ops()

    def _process_event(self, event: Event):
        op_id = event.operator_id
        hw_id = event.resource_id

        if event.event_type == EventType.COMPUTE:
            self.op_states[op_id] = OpState.FINISHED
            self.resource_state.mark_busy(hw_id, event.end_time_ns, "")
            op = self.operators[op_id]
            for out_id in op.output_ids:
                self.data_state.set_ready(out_id, hw_id, event.end_time_ns)
                dobj = self.data_objects.get(out_id)
                if dobj:
                    dobj.ready_time_ns = event.end_time_ns
                    dobj.location = hw_id

        elif event.event_type == EventType.TRANSFER:
            payload = event.payload
            data_id = payload.get("data_id", "")
            dst = payload.get("dst", "")
            self.data_state.set_ready(data_id, dst, event.end_time_ns)
            dobj = self.data_objects.get(data_id)
            if dobj:
                dobj.ready_time_ns = event.end_time_ns
                dobj.location = dst

        elif event.event_type == EventType.SYNC:
            pass

    # ─── 就绪检查 ───
    def _check_ready_ops(self):
        for oid, op in self.operators.items():
            if self.op_states[oid] != OpState.WAITING:
                continue
            # 依赖检查（ALL-GATHER 感知）：普通输入任一就绪即可；
            # 权重分片输入必须所有切分片都在其所在设备就绪（缺一片 = 拿不到完整权重）。
            if op.input_ids:
                ready = True
                for i in op.input_ids:
                    shards = self.weight_shards.get(i)
                    if shards:
                        for pid, dev in shards.items():
                            if not dev:
                                if not self.data_state.copies(pid):
                                    ready = False
                                    break
                            elif not self.data_state.has_copy(pid, dev):
                                ready = False
                                break
                        if not ready:
                            break
                    else:
                        if not self.data_state.copies(i):
                            ready = False
                            break
                if not ready:
                    continue
            target_hw = self.compute_map.get(oid, "")
            if not target_hw:
                continue
            if not self.perf.can_execute(op, target_hw):
                continue
            self.op_states[oid] = OpState.READY
            self._schedule_op(oid, op, target_hw)

    # ─── 核心：调度单个算子（含数据源决策） ───
    def _schedule_op(self, oid: str, op: Operator, target_hw: str):
        specs = {s.data_id: s for s in self.input_specs.get(oid, [])}
        max_arrival_ns = [self.clock_ns]     # 用单元素 list 以便就地累加
        transfer_ns_total = [0]
        transfer_events = []

        for in_id in op.input_ids:
            spec = specs.get(in_id)

            shards = self.weight_shards.get(in_id)
            if shards:
                # === 权重分片 ALL-GATHER：从每个切分片所在设备 gather 到目标硬件 ===
                for pid, dev in shards.items():
                    if not dev:
                        src = self._nearest_source(pid, target_hw)
                        if src is None:
                            self._note(f"算子 {oid} 权重分片 {pid}: 无任何就绪副本，等待。")
                            self.op_states[oid] = OpState.WAITING
                            return
                    else:
                        if not self.data_state.has_copy(pid, dev):
                            self._note(f"算子 {oid} 权重分片 {pid}: 设备 {dev} 上无副本，等待。")
                            self.op_states[oid] = OpState.WAITING
                            return
                        src = dev
                    src_ready = self.data_state.ready_time_on(pid, src)
                    self.notes.append(f"[ALL-GATHER] 算子 {oid} 权重分片 {pid} 从 {src} 汇集"
                                      f" (就绪 {src_ready}ns)")
                    self._ensure_data_at(pid, src, target_hw, src_ready,
                                         max_arrival_ns, transfer_ns_total, transfer_events,
                                         oid, op, None)
                continue

            if spec is not None and spec.mode == DataSourceMode.PINNED and spec.source_device:
                # === 用户固定读源，严格生效 ===
                src = spec.source_device
                copies = self.data_state.copies(in_id)
                if src not in copies:
                    # 用户指定源上根本没有这份数据 → 告警 + 回退到就近参考
                    self._note(
                        f"算子 {oid} 输入 {in_id}: 用户指定从 {src} 读, 但该设备无此数据副本"
                        f"(现有副本: {list(copies.keys())})。无法按用户指令执行，该输入标记为未就绪。")
                    # 不能读取到，视为该输入永远不就绪 → 标记异常，返回
                    self.op_states[oid] = OpState.FINISHED  # 避免死循环
                    return
                src_ready = copies.get(src, -1)
                if src_ready < 0:
                    self._note(f"算子 {oid} 输入 {in_id}: {src} 上的副本尚未就绪，等待。")
                    self.op_states[oid] = OpState.WAITING
                    return
                # 从 src 取数据，搬运到 target_hw（若不同）
                self._ensure_data_at(in_id, src, target_hw, src_ready,
                                     max_arrival_ns, transfer_ns_total, transfer_events,
                                     oid, op, spec)
            else:
                # === auto: 就近参考 ===
                src = self._nearest_source(in_id, target_hw)
                if src is None:
                    # 没有任何就绪副本
                    self._note(f"算子 {oid} 输入 {in_id}: 无任何就绪副本，等待。")
                    self.op_states[oid] = OpState.WAITING
                    return
                src_ready = self.data_state.ready_time_on(in_id, src)
                # 参考来源记录（供用户审阅）
                self.notes.append(f"[参考] 算子 {oid} 输入 {in_id} 数据源就近选择 {src}"
                                  f" (就绪 {src_ready}ns)")
                self._ensure_data_at(in_id, src, target_hw, src_ready,
                                     max_arrival_ns, transfer_ns_total, transfer_events,
                                     oid, op, spec)

        # 计算事件
        compute_dur = self.perf.compute_time_ns(op, target_hw)
        start_ns = max(self.clock_ns, self.resource_state.free_at(target_hw), max_arrival_ns[0])
        end_ns = start_ns + compute_dur
        sync_ns = max(0, max_arrival_ns[0] - max(self.clock_ns, self.resource_state.free_at(target_hw)))

        compute_ev = self._new_event(EventType.COMPUTE, start_ns, end_ns,
                                     op_id=oid, hw_id=target_hw, component="COMPUTE")
        self.queue.push(compute_ev)
        self.op_states[oid] = OpState.RUNNING
        self.resource_state.mark_busy(target_hw, end_ns, oid)

        self.op_timings.append(OperatorTiming(
            op_id=oid, op_type=op.op_type, hardware=target_hw,
            start_ns=start_ns, end_ns=end_ns, duration_ns=compute_dur,
            compute_ns=compute_dur,
            transfer_ns=transfer_ns_total[0],
            sync_ns=sync_ns,
        ))

    def _ensure_data_at(self, in_id, src, target_hw, src_ready,
                        max_arrival_ns, transfer_ns_total, transfer_events,
                        oid, op, spec, _via_pinned=False):
        """确保数据从 src 到达 target_hw。已在则直接取就绪时间，否则插搬运。"""
        dobj = self.data_objects.get(in_id)
        size = dobj.size_bytes if dobj else 0

        if src == target_hw or self.data_state.has_copy(in_id, target_hw):
            # 驻留命中或无搬运
            t = self.data_state.ready_time_on(in_id, target_hw)
            if t < 0:
                t = src_ready
            max_arrival_ns[0] = max(max_arrival_ns[0], t)
            return

        t_transfer = self.perf.transfer_time_ns(size, src, target_hw)
        if t_transfer <= 0:
            max_arrival_ns[0] = max(max_arrival_ns[0], src_ready)
            return
        transfer_start = max(self.clock_ns, src_ready,
                             self.resource_state.free_at(src))
        transfer_end = transfer_start + t_transfer
        ev = self._new_event(EventType.TRANSFER, transfer_start, transfer_end,
                             op_id=oid, hw_id=src, component="TRANSFER")
        ev.payload = {"data_id": in_id, "src": src, "dst": target_hw,
                      "size_bytes": size}
        self.queue.push(ev)
        transfer_events.append(ev)
        transfer_ns_total[0] += t_transfer
        max_arrival_ns[0] = max(max_arrival_ns[0], transfer_end)

    def _nearest_source(self, in_id, target_hw) -> str:
        """就近参考：min(就绪时间 + 搬运耗时)；若已在 target 则命中。"""
        copies = self.data_state.copies(in_id)
        if not copies:
            return None
        if target_hw in copies:
            return target_hw  # 命中，零搬运
        best = None
        best_cost = None
        dobj = self.data_objects.get(in_id)
        size = dobj.size_bytes if dobj else 0
        for dev, ready in copies.items():
            if ready < 0:
                continue
            if dev == target_hw:
                return target_hw
            t = self.perf.transfer_time_ns(size, dev, target_hw)
            cost = ready + t
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = dev
        return best if best else (target_hw if target_hw in copies else None)

    # ─── 结果收集 ───
    def collect_result(self) -> dict:
        total_compute = sum(t.compute_ns for t in self.op_timings)
        total_transfer = sum(t.transfer_ns for t in self.op_timings)
        total_sync = sum(t.sync_ns for t in self.op_timings)
        max_end = max((t.end_ns for t in self.op_timings), default=0)

        breakdown = DurationBreakdown(
            compute_ns=total_compute, transfer_ns=total_transfer, sync_ns=total_sync)

        from contracts import BottleneckType
        parts = {
            BottleneckType.COMPUTE: total_compute,
            BottleneckType.COMMUNICATION: total_transfer,
            BottleneckType.SYNCHRONIZATION: total_sync,
        }
        dominant = max(parts, key=parts.get)
        rationale = (f"Compute={total_compute/1e6:.2f}ms, "
                     f"Transfer={total_transfer/1e6:.2f}ms, "
                     f"Sync={total_sync/1e6:.2f}ms")

        return {
            "total_latency_ns": max_end,
            "breakdown": breakdown,
            "bottleneck": dominant,
            "rationale": rationale,
            "op_timings": self.op_timings,
            "event_trace": self.event_trace,
        }
