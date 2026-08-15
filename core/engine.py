"""
LLM-PIMSim v3 core.engine — 【核心调度器】

职责：
  1. PerformanceModel —— 纯函数性能估算（计算/搬运/读写耗时公式）
  2. 离散事件内核    —— EventQueue / DataStateTable(冗余驻留) / ResourceStateTable
  3. Scheduler        —— 配置驱动调度核心（数据源决策、权重分片 ALL-GATHER）
  4. SimulationEngine —— 组装硬件图 / 算子图 / 映射 / 放置 并运行

依赖：core.common（数据结构/枚举）+ core.precision（精度语义，交给硬件/算子）。
本系统不做硬件建模、不做校验、不做序列化，只负责"调度出时间/事件/结果"。

顶层兼容：由 `engine.py` 转发 SimulationEngine；`scheduler.py`、`performance.py`
转发对应类（保持旧入口可用）。
"""

import heapq
from typing import Optional

from core.common import (
    Event, EventType, Operator, DataObject, OperatorTiming,
    OpState, DurationBreakdown, InputSpec, BottleneckType, DataSourceMode,
)
from core.precision import PrecisionLevel


# =================================================================
# PerformanceModel —— 性能估算（纯函数，只读硬件参数）
# =================================================================
class PerformanceModel:
    def __init__(self, hardware_units: dict, interconnect=None, link_table=None):
        self.hw = hardware_units
        self.interconnect = interconnect          # 旧 Interconnect 兼容（未给 link_table 时用）
        self.link_table = link_table              # 新链路系统：LinkBandwidthTable

    def _type_name(self, hw) -> str:
        """返回该硬件在链路表里的"设备种类"字符串。"""
        tn = getattr(hw, "type_name", None)
        if tn:
            return tn
        dt = getattr(hw, "device_type", None)
        if dt is not None:
            return getattr(dt, "name", str(dt))
        return str(getattr(hw, "id", "?"))

    def can_execute(self, op: Operator, hw_id: str) -> bool:
        hw = self.hw.get(hw_id)
        if hw is None:
            return False
        return hw.can_execute(op)

    def compute_time_ns(self, op: Operator, hw_id: str) -> int:
        hw = self.hw.get(hw_id)
        if hw is None:
            return 0
        eff = hw.efficiency_for(op.op_type)
        if op.flops == 0 or eff == 0:
            return 0
        # v3.1：按精度算力 —— 优先用算子"执行精度"对应的峰值；无执行精度则用数据精度；
        #       两者都不可得再回退到硬件单值峰值。实现"不同精度算力不同"。
        peak = getattr(hw, "get_peak_compute", None)
        if peak is not None:
            prec = op.execution_precision if op.execution_precision is not None else op.data_precision
            p = peak(prec) if prec is not None else peak(None)
            if p and p > 0:
                return int(op.flops / (p * eff) * 1e9)
        if hw.peak_compute_flops == 0:
            return 0
        return int(op.flops / (hw.peak_compute_flops * eff) * 1e9)

    def local_read_time_ns(self, data_size_bytes: int, hw_id: str) -> int:
        hw = self.hw.get(hw_id)
        if hw is None:
            return 0
        if data_size_bytes == 0 or hw.read_bandwidth_Bps == 0:
            return hw.read_latency_ns
        return int(hw.read_latency_ns + data_size_bytes / hw.read_bandwidth_Bps * 1e9)

    def local_write_time_ns(self, data_size_bytes: int, hw_id: str) -> int:
        hw = self.hw.get(hw_id)
        if hw is None:
            return 0
        if data_size_bytes == 0 or hw.write_bandwidth_Bps == 0:
            return hw.write_latency_ns
        return int(hw.write_latency_ns + data_size_bytes / hw.write_bandwidth_Bps * 1e9)

    def transfer_time_ns(self, data_size_bytes: int, src_hw_id: str, dst_hw_id: str) -> int:
        """跨设备搬运 src→dst（v3.2 链路系统口径，链路只有带宽、无独立延迟）：
            T = T_read(src) + S/链路带宽(src种类, dst种类) + T_write(dst)
        读写延迟/带宽由硬件系统提供；链路带宽由【链路系统】N×N 对称表提供。"""
        src = self.hw.get(src_hw_id)
        dst = self.hw.get(dst_hw_id)
        if src is None or dst is None:
            return 0

        # 解析链路带宽（B/s）：优先新链路表，其次旧 Interconnect 兼容。
        if self.link_table is not None:
            link_bw = self.link_table.get_bw_bps(self._type_name(src), self._type_name(dst))
        elif self.interconnect is not None:
            link = self.interconnect.find_link(src_hw_id, dst_hw_id)
            link_bw = link.write_bw_Bps if link else 0.0
        else:
            # 无任何链路信息 → 不建模跨设备搬运（沿用旧语义，返回 0）
            return 0
        if link_bw <= 0:
            return 0

        t_read = self.local_read_time_ns(data_size_bytes, src_hw_id)
        t_write = self.local_write_time_ns(data_size_bytes, dst_hw_id)
        t_link = int(data_size_bytes / link_bw * 1e9)
        return t_read + t_link + t_write

    def estimate(self, op: Operator, hw_id: str, data_sizes: dict = None) -> DurationBreakdown:
        bd = DurationBreakdown()
        if not self.can_execute(op, hw_id):
            return bd
        bd.compute_ns = self.compute_time_ns(op, hw_id)
        if data_sizes:
            for did, size in data_sizes.items():
                bd.local_read_ns += self.local_read_time_ns(size, hw_id)
        return bd


# =================================================================
# 离散事件内核
# =================================================================
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
    """数据状态表 v2 —— 支持冗余多设备驻留 {data_id: {device_id: ready_time_ns}}"""

    def __init__(self):
        self._by_device: dict[str, dict[str, int]] = {}

    def place_multi(self, obj_id: str, devices: list, ready_time_ns: int = 0):
        entry = self._by_device.setdefault(obj_id, {})
        for dev in devices:
            if dev:
                entry[dev] = ready_time_ns

    def set_ready(self, obj_id: str, device_id: str, ready_time_ns: int):
        self._by_device.setdefault(obj_id, {})[device_id] = ready_time_ns

    def copies(self, obj_id: str) -> dict:
        return self._by_device.get(obj_id, {})

    def has_copy(self, obj_id: str, device_id: str) -> bool:
        return device_id in self._by_device.get(obj_id, {})

    def ready_time_on(self, obj_id: str, device_id: str) -> int:
        return self._by_device.get(obj_id, {}).get(device_id, -1)

    def is_any_ready(self, obj_id: str) -> bool:
        copies = self._by_device.get(obj_id, {})
        return any(t >= 0 for t in copies.values())

    def earliest_ready(self, obj_id: str) -> int:
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


# =================================================================
# Scheduler —— 配置驱动调度核心
# =================================================================
class Scheduler:
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
        self.compute_map = compute_map      # {op_id: compute_device}
        self.input_specs = input_specs or {}
        self.placement = placement or {}    # {data_id: [devices]}
        self.notes = note_collector if note_collector is not None else []
        self.weight_shards = weight_shards or {}   # {weight_data_id: {partition_id: device_id}}

        self.data_state = DataStateTable()
        self.resource_state = ResourceStateTable()
        self.queue = EventQueue()
        self.clock_ns = 0

        self.op_states: dict[str, OpState] = {}
        self.event_trace: list = []
        self.op_timings: list = []
        self._next_event_id = 0
        self.movement: dict = {}            # {(src,dst): bytes}
        # v3.2 链路并发规则：一条链路同一时刻只能搬一份数据（串行排队）
        self.link_state: dict = {}          # {(src,dst): busy_until_ns}
        self.not_runnable: list = []

    def _new_event(self, etype, start, end, op_id="", hw_id="", component="") -> Event:
        ev = Event(id=self._next_event_id, event_type=etype,
                   start_time_ns=start, end_time_ns=end,
                   operator_id=op_id, resource_id=hw_id, component=component)
        self._next_event_id += 1
        return ev

    def _note(self, msg: str):
        self.notes.append(f"[t={self.clock_ns}ns] {msg}")

    def initialize(self):
        for oid, op in self.operators.items():
            self.op_states[oid] = OpState.WAITING
        for did, dobj in self.data_objects.items():
            devs = self.placement.get(did, [])
            if devs:
                self.data_state.place_multi(did, devs, 0)
                dobj.replica_locations = list(devs)
                dobj.location = devs[0] if devs else ""

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

    def _check_ready_ops(self):
        for oid, op in self.operators.items():
            if self.op_states[oid] != OpState.WAITING:
                continue
            if op.input_ids:
                ready = True
                for i in op.input_ids:
                    shards = self.weight_shards.get(i)
                    if shards:
                        for pid, dev in shards.items():
                            if not dev:
                                if not self.data_state.copies(pid):
                                    ready = False; break
                            elif not self.data_state.has_copy(pid, dev):
                                ready = False; break
                        if not ready:
                            break
                    else:
                        if not self.data_state.copies(i):
                            ready = False; break
                if not ready:
                    continue
            target_hw = self.compute_map.get(oid, "")
            if not target_hw:
                continue
            if not self.perf.can_execute(op, target_hw):
                continue
            self.op_states[oid] = OpState.READY
            self._schedule_op(oid, op, target_hw)

    def _schedule_op(self, oid: str, op: Operator, target_hw: str):
        specs = {s.data_id: s for s in self.input_specs.get(oid, [])}
        max_arrival_ns = [self.clock_ns]
        transfer_ns_total = [0]
        local_read_ns_total = [0]
        transfer_events = []

        for in_id in op.input_ids:
            spec = specs.get(in_id)
            shards = self.weight_shards.get(in_id)
            if shards:
                for pid, dev in shards.items():
                    if not dev:
                        src = self._nearest_source(pid, target_hw)
                        if src is None:
                            self._note(f"算子 {oid} 权重分片 {pid}: 无任何就绪副本，等待。")
                            self.op_states[oid] = OpState.WAITING; return
                    else:
                        if not self.data_state.has_copy(pid, dev):
                            self._note(f"算子 {oid} 权重分片 {pid}: 设备 {dev} 上无副本，等待。")
                            self.op_states[oid] = OpState.WAITING; return
                        src = dev
                    src_ready = self.data_state.ready_time_on(pid, src)
                    self.notes.append(f"[ALL-GATHER] 算子 {oid} 权重分片 {pid} 从 {src} 汇集"
                                      f" (就绪 {src_ready}ns)")
                    self._ensure_data_at(pid, src, target_hw, src_ready,
                                         max_arrival_ns, transfer_ns_total, transfer_events,
                                         oid, op, None, local_read_ns_total)
                continue

            if spec is not None and spec.mode == DataSourceMode.PINNED and spec.source_device:   # PINNED
                src = spec.source_device
                copies = self.data_state.copies(in_id)
                if src not in copies:
                    self._note(
                        f"算子 {oid} 输入 {in_id}: 用户指定从 {src} 读, 但该设备无此数据副本"
                        f"(现有副本: {list(copies.keys())})。无法按用户指令执行，该输入标记为未就绪。")
                    self.op_states[oid] = OpState.FINISHED  # 避免死循环（见 v3 通道）
                    return
                src_ready = copies.get(src, -1)
                if src_ready < 0:
                    self._note(f"算子 {oid} 输入 {in_id}: {src} 上的副本尚未就绪，等待。")
                    self.op_states[oid] = OpState.WAITING; return
                self._ensure_data_at(in_id, src, target_hw, src_ready,
                                     max_arrival_ns, transfer_ns_total, transfer_events,
                                     oid, op, spec, local_read_ns_total)
            else:
                src = self._nearest_source(in_id, target_hw)
                if src is None:
                    self._note(f"算子 {oid} 输入 {in_id}: 无任何就绪副本，等待。")
                    self.op_states[oid] = OpState.WAITING; return
                src_ready = self.data_state.ready_time_on(in_id, src)
                self.notes.append(f"[参考] 算子 {oid} 输入 {in_id} 数据源就近选择 {src}"
                                  f" (就绪 {src_ready}ns)")
                self._ensure_data_at(in_id, src, target_hw, src_ready,
                                     max_arrival_ns, transfer_ns_total, transfer_events,
                                     oid, op, spec, local_read_ns_total)

        compute_dur = self.perf.compute_time_ns(op, target_hw)
        start_ns = max(self.clock_ns, self.resource_state.free_at(target_hw), max_arrival_ns[0])
        end_ns = start_ns + compute_dur
        sync_ns = max(0, max_arrival_ns[0] - max(self.clock_ns, self.resource_state.free_at(target_hw)))

        # v3.2 本地读写：写自己的各输出到目标设备本地的时长
        local_write_ns = 0
        for out_id in op.output_ids:
            od = self.data_objects.get(out_id)
            lw = self.perf.local_write_time_ns(od.size_bytes if od else 0, target_hw)
            local_write_ns += lw

        compute_ev = self._new_event(EventType.COMPUTE, start_ns, end_ns,
                                     op_id=oid, hw_id=target_hw, component="COMPUTE")
        self.queue.push(compute_ev)
        self.op_states[oid] = OpState.RUNNING
        self.resource_state.mark_busy(target_hw, end_ns, oid)

        self.op_timings.append(OperatorTiming(
            op_id=oid, op_type=op.op_type, hardware=target_hw,
            start_ns=start_ns, end_ns=end_ns, duration_ns=compute_dur,
            compute_ns=compute_dur,
            local_read_ns=local_read_ns_total[0], local_write_ns=local_write_ns,
            transfer_ns=transfer_ns_total[0],
            sync_ns=sync_ns))

    def _ensure_data_at(self, in_id, src, target_hw, src_ready,
                        max_arrival_ns, transfer_ns_total, transfer_events,
                        oid, op, spec, local_read_ns_total=None, _via_pinned=False):
        dobj = self.data_objects.get(in_id)
        size = dobj.size_bytes if dobj else 0
        if src == target_hw or self.data_state.has_copy(in_id, target_hw):
            t = self.data_state.ready_time_on(in_id, target_hw)
            if t < 0:
                t = src_ready
            max_arrival_ns[0] = max(max_arrival_ns[0], t)
            # v3.2 本地读写：输入已在目标设备本地 → 记一次本地读时长
            if local_read_ns_total is not None:
                local_read_ns_total[0] += self.perf.local_read_time_ns(size, target_hw)
            return
        t_transfer = self.perf.transfer_time_ns(size, src, target_hw)
        if t_transfer <= 0:
            max_arrival_ns[0] = max(max_arrival_ns[0], src_ready)
            return
        link_key = (src, target_hw)
        # v3.2 链路并发规则：同一条链路上一时刻只能搬一份数据 → 排队等到该链路空闲
        link_free = self.link_state.get(link_key, 0)
        transfer_start = max(self.clock_ns, src_ready, self.resource_state.free_at(src), link_free)
        transfer_end = transfer_start + t_transfer
        self.link_state[link_key] = transfer_end   # 预留链路占用
        ev = self._new_event(EventType.TRANSFER, transfer_start, transfer_end,
                             op_id=oid, hw_id=src, component="TRANSFER")
        ev.payload = {"data_id": in_id, "src": src, "dst": target_hw, "size_bytes": size}
        self.queue.push(ev)
        transfer_events.append(ev)
        transfer_ns_total[0] += t_transfer
        max_arrival_ns[0] = max(max_arrival_ns[0], transfer_end)   # 数据到达目标设备的时刻
        self.movement[(src, target_hw)] = self.movement.get((src, target_hw), 0) + size

    def _nearest_source(self, in_id, target_hw) -> Optional[str]:
        copies = self.data_state.copies(in_id)
        if not copies:
            return None
        if target_hw in copies:
            return target_hw
        best = None; best_cost = None
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
                best_cost = cost; best = dev
        return best if best else (target_hw if target_hw in copies else None)

    # ─── 结果收集 ───
    def collect_result(self) -> dict:
        total_compute = sum(t.compute_ns for t in self.op_timings)
        total_transfer = sum(t.transfer_ns for t in self.op_timings)
        total_sync = sum(t.sync_ns for t in self.op_timings)
        # v3.2 本地读写也纳入 breakdown
        total_local_read = sum(t.local_read_ns for t in self.op_timings)
        total_local_write = sum(t.local_write_ns for t in self.op_timings)
        max_end = max((t.end_ns for t in self.op_timings), default=0)
        breakdown = DurationBreakdown(compute_ns=total_compute, transfer_ns=total_transfer,
                                      sync_ns=total_sync,
                                      local_read_ns=total_local_read,
                                      local_write_ns=total_local_write)
        parts = {BottleneckType.COMPUTE: total_compute,
                 BottleneckType.COMMUNICATION: total_transfer,
                 BottleneckType.SYNCHRONIZATION: total_sync,
                 BottleneckType.MEMORY: total_local_read + total_local_write}
        dominant = max(parts, key=parts.get)
        rationale = (f"Compute={total_compute/1e6:.2f}ms, "
                     f"Transfer={total_transfer/1e6:.2f}ms, "
                     f"Sync={total_sync/1e6:.2f}ms, "
                     f"LocalRW={ (total_local_read+total_local_write)/1e6:.2f}ms")

        unfinished = []
        for oid, st in self.op_states.items():
            if st != OpState.FINISHED:
                op = self.operators.get(oid)
                unfinished.append({"op_id": oid,
                                   "op_type": op.op_type if op else "",
                                   "compute_device": self.compute_map.get(oid, ""),
                                   "state": st.name})
        total_ops = len(self.op_states)
        movement_total = sum(self.movement.values()) if self.movement else 0
        movement_bytes = {
            "total_bytes": movement_total,
            "per_link": [{"src": s, "dst": d, "bytes": b}
                         for (s, d), b in sorted(self.movement.items(), key=lambda x: -x[1])],
        }
        return {
            "total_latency_ns": max_end,
            "breakdown": breakdown,
            "bottleneck": dominant,
            "rationale": rationale,
            "op_timings": self.op_timings,
            "event_trace": self.event_trace,
            "movement_bytes": movement_bytes,
            "diagnostics": {"total_operators": total_ops,
                            "finished_operators": total_ops - len(unfinished),
                            "unfinished_operators": unfinished},
        }


# =================================================================
# SimulationEngine —— 组装并运行
# =================================================================
class SimulationEngine:
    def __init__(self):
        self.hw = {}
        self.interconnect = None
        self.link_table = None
        self.ops = {}
        self.datas = {}
        self.compute_map = {}
        self.placement = {}
        self.input_specs = {}
        self.weight_shards = {}
        self.perf = None

    def build(self, *, hardware_units, interconnect=None, link_table=None, operators,
              data_objects, compute_map, placement, input_specs=None,
              weight_shards=None, op_splits=None):
        self.hw = hardware_units
        self.interconnect = interconnect
        self.link_table = link_table
        self.ops = {op.id: op for op in operators} if not isinstance(operators, dict) else dict(operators)
        self.datas = {d.id: d for d in data_objects} if not isinstance(data_objects, dict) else dict(data_objects)
        self.compute_map = dict(compute_map)
        self.placement = dict(placement)
        self.input_specs = dict(input_specs or {})
        self.weight_shards = dict(weight_shards or {})
        # v3.2 算子切片 → 多设备并行：预展开为多个子算子，各自映射到切片设备
        # op_splits: {op_id: {"devices": [hw_id, ...]}} —— 把该算子的计算量均分到 devices 上并行执行
        self._agg = {}   # 逻辑算子 -> 其 sub-op id 列表（用于结果聚合）
        if op_splits:
            self._expand_op_splits(op_splits)
        return self

    def _expand_op_splits(self, op_splits: dict):
        """把一个算子在逻辑上切成 N 片、映射到 N 个设备并行（维度切分 + 权重分片式张量并行）。

        op_splits[op_id] = {"devices": [...], "slice_flops": [f1..fn], "dim": dim}
          - slice_flops 由切割系统 split_kernel_dict 按维度比例算得（张量并行）；缺省则均分。
          - 该算子的权重输入会被分片到各切片设备（weight_shards → ALL-GATHER 语义）。
        每个切片=原算子的一份复制（flops=对应片段），input/output 不变；放到各自设备。
        逻辑算子整体结束 = 最后完成的那片；其 timing 聚合回逻辑算子（见 _aggregate_slices）。
        """
        import copy
        new_ops = dict(self.ops)
        for op_id, spec in op_splits.items():
            op = new_ops.get(op_id)
            if op is None or not spec:
                continue
            devices = [d for d in (spec.get("devices") or []) if d]
            if not devices:
                continue
            n = len(devices)
            slice_flops = spec.get("slice_flops")
            if not slice_flops or len(slice_flops) != n:
                slice_flops = [op.flops / n] * n   # 缺省：计算量均分
            sub_ids = []
            for i, dev in enumerate(devices):
                sub_id = f"{op_id}#slice{i+1}"
                sub = copy.copy(op)
                sub.id = sub_id
                sub.flops = int(slice_flops[i])
                new_ops[sub_id] = sub
                self.compute_map[sub_id] = dev
                sub_ids.append(sub_id)
            # 权重分片式张量并行：把算子的权重输入分片到各切片设备，纳入 ALL-GATHER
            self._shard_op_weights(op, devices)
            # 原算子的映射移除（逻辑算子不再作为单个执行单位）
            self.compute_map.pop(op_id, None)
            self._agg[op_id] = sub_ids
        self.ops = new_ops

    def _shard_op_weights(self, op, devices: list):
        """把该算子的权重输入按设备数分片，注册 weight_shards → ALL-GATHER。

        对被切分的 GEMM 等算子：它的权重（input_ids 里 *_weight 的数据）沿可切维等分为 N 片，
        各放一个设备，调度器要求"读齐全部切片"（ALL-GATHER）才能真正执行。
        """
        n = len(devices)
        if n < 2:
            return
        for wid in op.input_ids:
            dobj = self.datas.get(wid)
            if dobj is None or dobj.data_type.name != "WEIGHT":
                continue
            if wid in self.weight_shards:
                continue   # 已分片（可能来自独立权重切割）
            shards = {}
            # 每片按设备放置（字节均分近似；真实维度信息由切割系统提供，此处按片数分摊）
            base = dobj.size_bytes // n
            rem = dobj.size_bytes % n
            for i, dev in enumerate(devices):
                pid = f"{wid}.tp{i+1}"
                part_bytes = base + (1 if i < rem else 0)
                if pid not in self.datas:
                    self.datas[pid] = DataObject(id=pid, name=pid,
                                                 data_type=dobj.data_type,
                                                 size_bytes=part_bytes)
                self.placement[pid] = [dev]
                shards[pid] = dev
            self.weight_shards[wid] = shards

    def _aggregate_slices(self, raw: dict) -> dict:
        """把切片子算子的 timing 聚合成逻辑算子（并行 → 墙钟=最后结束那片）。"""
        from core.common import OperatorTiming
        by_logical = {}   # logical_id -> [OperatorTiming...]（已完成的子切片）
        kept = []
        for t in raw["op_timings"]:
            base, sep, n = t.op_id.rpartition("#slice")
            if sep and base in self._agg:
                by_logical.setdefault(base, []).append(t)
            else:
                kept.append(t)

        aggregated = list(kept)
        for log_id, subs in by_logical.items():
            if not subs:
                continue
            start = min(s.start_ns for s in subs)
            end = max(s.end_ns for s in subs)          # 并行 → 最后结束的切片
            compute = sum(s.compute_ns for s in subs)
            transfer = sum(s.transfer_ns for s in subs)
            local_read = sum(s.local_read_ns for s in subs)
            local_write = sum(s.local_write_ns for s in subs)
            hws = "/".join(sorted({s.hardware for s in subs}))
            aggregated.append(OperatorTiming(
                op_id=log_id, op_type=subs[0].op_type, hardware=hws,
                start_ns=start, end_ns=end, duration_ns=end - start,
                compute_ns=compute,
                local_read_ns=local_read, local_write_ns=local_write,
                transfer_ns=transfer, sync_ns=max(s.sync_ns for s in subs)))
        raw["op_timings"] = aggregated

        # 修正 total_latency（逻辑切片可能整体比任何单算子更晚结束）
        if aggregated:
            raw["total_latency_ns"] = max(raw.get("total_latency_ns", 0),
                                          max(t.end_ns for t in aggregated))
        # 修正 diagnostics：把子切片视为完成，且"有聚合结果的逻辑算子"也算完成
        sub_ids = {sid for ids in self._agg.values() for sid in ids}
        aggregated_done = set(by_logical.keys())   # 这些逻辑算子已有聚合 timing → 视为完成
        diag = raw.get("diagnostics", {})
        unfin = [u for u in diag.get("unfinished_operators", [])
                 if u["op_id"] not in sub_ids and u["op_id"] not in aggregated_done]
        total = diag.get("total_operators", 0) - len(sub_ids)
        diag["total_operators"] = total
        diag["unfinished_operators"] = unfin
        diag["finished_operators"] = total - len(unfin)
        raw["diagnostics"] = diag
        return raw

    def run(self, seed=42):
        from core.common import SimulationResult
        self.perf = PerformanceModel(self.hw, self.interconnect, self.link_table)
        notes = []
        sched = Scheduler(
            operators=self.ops, data_objects=self.datas,
            hardware_units=self.hw, interconnect=self.interconnect,
            perf_model=self.perf, compute_map=self.compute_map,
            input_specs=self.input_specs, placement=self.placement,
            note_collector=notes, weight_shards=self.weight_shards,
        )
        sched.run()
        raw = sched.collect_result()
        if self._agg:
            raw = self._aggregate_slices(raw)

        result = SimulationResult(
            metadata={"hardware": [h.name for h in self.hw.values()], "seed": seed},
            total_latency_ns=raw["total_latency_ns"],
            breakdown=raw["breakdown"],
            bottleneck=raw["bottleneck"],
            bottleneck_rationale=raw["rationale"],
            operator_timings=raw["op_timings"],
            event_trace=raw["event_trace"],
            movement_bytes=raw["movement_bytes"],
            diagnostics=raw["diagnostics"],
        )
        for n in notes:
            result.add_note(n)
        return result
