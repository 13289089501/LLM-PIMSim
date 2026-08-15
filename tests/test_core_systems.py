# -*- coding: utf-8 -*-
"""
v3 core 七大系统解耦功能测试。
验证各系统独立可用、依赖方向正确（无循环），并覆盖关键行为：
  精度系统 / 算子系统 / 权重系统 / 切割系统 / 校验系统 / 输出系统 / 调度器
"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPrecisionSystem(unittest.TestCase):
    def test_level_ordering(self):
        from core.precision import PrecisionLevel as P
        # 等级单一日径：INT4=1 < FP8=3 < FP16=4 < FP32=6
        self.assertEqual([P.INT4.value, P.INT8.value, P.FP8.value,
                          P.FP16.value, P.BF16.value, P.FP32.value],
                         [1, 2, 3, 4, 5, 6])
        self.assertTrue(P.FP16.is_high)
        self.assertTrue(P.INT8.is_low)

    def test_hw_capability_decoupled(self):
        from core.precision import HARDWARE_CAPABILITY, hardware_execution_supported
        # GPU 可执行 NONLINEAR（v3 修正后的能力表）
        from core.common import OperatorCategory
        self.assertIn(OperatorCategory.NONLINEAR,
                      HARDWARE_CAPABILITY["GPU"]["categories"])
        self.assertTrue(hardware_execution_supported("GPU", __import__(
            "core.precision", fromlist=["PrecisionLevel"]).PrecisionLevel.FP16))


class TestOperatorSystem(unittest.TestCase):
    def test_18_rules(self):
        from core.operator_sys import OPERATOR_PRECISION_RULES
        self.assertEqual(len(OPERATOR_PRECISION_RULES), 18)
        self.assertIn("QKV_proj", OPERATOR_PRECISION_RULES)

    def test_build_workload(self):
        from core.operator_sys import build_model_workload
        wl = build_model_workload(hidden=4096, ffn_size=11008, num_heads=32,
                                  head_dim=128, vocab=32000, num_layers=2,
                                  input_tokens=32, decode_steps=8)
        self.assertEqual(wl.num_layers, 2)
        self.assertEqual(len(wl.layers[0]), 16)   # 每层 16 个算子
        self.assertEqual(len(wl.kernels), 2 * 16 + 2)  # + Embedding/LMHead


class TestWeightAndSplitSystems(unittest.TestCase):
    def test_weight_split_decoupled(self):
        # 权重系统建块，切割系统切分（两系统解耦，通过回调注入）
        from core.weight_sys import build_weight_blocks
        from core.splitter import make_weight_partitions
        blocks = build_weight_blocks("llama7b", num_layers=1, h=4096, f=11008,
                                     nh=32, v=32000, precision_bytes=2,
                                     class_split={"W_mlp": 4},
                                     make_partitions=make_weight_partitions)
        mlp = next(b for b in blocks.values() if b.weight_class == "W_mlp")
        self.assertEqual(len(mlp.partitions), 4)  # 切成 4 片
        total = sum(p.bytes for p in mlp.partitions)
        self.assertEqual(total, mlp.bytes)        # 各片字节和 == 整块字节

    def test_kernel_split(self):
        from core.splitter import split_kernel_dict
        kernel = {"id": "L0_ffn_gate", "attributes": {"M": 1, "K": 4096, "N": 11008},
                  "compute_flops_range": [2e11, 2e11], "memory_bytes_range": [1e6, 1e6]}
        parts = split_kernel_dict(kernel, "N", [5504, 5504])
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["attributes"]["N"], 5504)

    def test_kernel_shape(self):
        # v3.1 统一形状：dims + 可切维 + 层号（从 id 提取）
        from core.splitter import kernel_shape, split_kernel_dict
        s = kernel_shape({"id": "L0_qkv_proj", "attributes": {"M": 1, "K": 4096, "N": 12288}})
        self.assertEqual(s.layer, 0)
        self.assertEqual(s.dims["K"], 4096.0)
        self.assertIn("N", s.split_dims)
        # 全局算子层号为 None
        self.assertIsNone(kernel_shape({"id": "lm_head", "attributes": {"N": 32000}}).layer)
        # 动态 kv_len 也被登记为可切维、但无数值
        sdyn = kernel_shape({"id": "L0_attn_score",
                             "attributes": {"M": 1, "K": 128, "N": "kv_len"}})
        self.assertIn("N", sdyn.split_dims)
        self.assertNotIn("N", sdyn.dims)     # kv_len 占位不进 dims（无数值）
        # 非法维报错应给出可切维清单
        with self.assertRaises(ValueError):
            split_kernel_dict({"id": "L0_ffn_gate", "attributes": {"M": 1, "K": 4096, "N": 11008}},
                              "XYZ", [1])

    def test_weight_ports(self):
        # 结构化端口：每个消费算子的权重都生成一个 out 端口（方向/目标/槽位/形状）
        from core.weight_sys import build_weight_blocks, WEIGHT_PORT_DIRECTION
        blocks = build_weight_blocks("llama7b", num_layers=1, h=4096, f=11008,
                                     nh=32, v=32000, precision_bytes=2)
        qkv = blocks["L0_qkv_weight"]
        self.assertEqual(len(qkv.ports), 1)
        port = qkv.port_for("L0_qkv_proj")
        self.assertIsNotNone(port)
        self.assertEqual(port.direction, WEIGHT_PORT_DIRECTION)
        self.assertEqual(port.target_op, "L0_qkv_proj")
        self.assertEqual(port.input_slot, 1)          # LN1_out 是第0输入，qkv_weight 是第1
        self.assertEqual(tuple(port.shape), (3 * 4096, 4096))
        self.assertEqual(port.data_type, "WEIGHT")
        # 兼容字段仍可用
        self.assertEqual(qkv.input_slots.get("L0_qkv_proj"), 1)
        self.assertIn("L0_qkv_proj", qkv.consumers)
        # 序列化视图
        pd = qkv.to_port_dict()
        self.assertEqual(pd[0]["target_op"], "L0_qkv_proj")



class TestValidationSystem(unittest.TestCase):
    def test_validate_ok_empty_errors(self):
        from core.validator import ConstraintChecker
        # 空配置应报 E1/E2（无硬件/无算子），但接口可调用
        vr = ConstraintChecker(hardware=[], operators=[], connections=[],
                               compute_map={}, weight_blocks=[]).validate()
        self.assertTrue(hasattr(vr, "valid"))

    def test_a3_matches_precision_system(self):
        # v3.2：A3 应与精度系统能力表对齐（类别 + 执行精度"列表包含"，而非等级近似）
        from core.validator import validate_config

        def a3_of(hw, op):
            vr = validate_config(hardware=[hw], operators=[op],
                connections=[{'from': hw['id'] + '_r', 'to': 'op0_in0'},
                             {'from': 'op0_out0', 'to': hw['id'] + '_w'}],
                compute_map={op['kernelId']: hw['id']}, weight_blocks=[])
            return [i for i in vr.issues if i.code in ('A3', 'E4')]

        gpu = {'id': 'gpu0', 'type': 'GPU', 'precision': 'FP32/FP16/INT8/INT4'}
        dram = {'id': 'pim0', 'type': 'DRAM_PIM', 'precision': 'FP32/FP16/INT8/INT4'}
        base = {'id': 'op0', 'kernelId': 'L0_ln1', 'inputs': ['x'], 'outputs': ['y']}
        ln = dict(base, op_type='LayerNorm', precision='FP32')
        gemm_i8 = dict(base, kernelId='L0_ffn_gate', op_type='GEMM', precision='INT8')
        kv_i4 = dict(base, kernelId='kv', op_type='KVCacheUpdate', precision='INT4')

        # GPU 支持 NONLINEAR + FP32 执行 → 通过
        self.assertEqual(a3_of(gpu, ln), [])
        # DRAM_PIM 只算 LINEAR，LayerNorm(NONLINEAR) → 类别不兼容
        self.assertTrue(any(i.code == 'A3' for i in a3_of(dram, ln)))
        # GPU 支持 INT8 执行 → 通过
        self.assertEqual(a3_of(gpu, gemm_i8), [])
        # KV_Cache 是纯数据算子（execution=None）→ A3 查 data 能力，GPU data 含 INT4 → 通过
        # （v3.2 口径：execution=None 的算子不再误按"执行精度 INT4"拒绝，见 CHANGELOG §15）
        self.assertEqual(a3_of(gpu, kv_i4), [])

    def test_completion_validation(self):
        # v3.2：完成度校验 —— 未满员 → 判无效（F1）
        from core.validator import validate_completion
        ok = {"total_operators": 514, "finished_operators": 514,
              "unfinished_operators": []}
        self.assertTrue(validate_completion(ok).valid)
        bad = {"total_operators": 514, "finished_operators": 225,
               "unfinished_operators": [{"op_id": "L0_ln1", "op_type": "LayerNorm",
                                          "compute_device": "gpu0", "state": "WAITING"}]}
        vr = validate_completion(bad)
        self.assertFalse(vr.valid)
        self.assertTrue(any(i.code == "F1" for i in vr.errors))


class TestOutputSystem(unittest.TestCase):
    def test_result_to_dict(self):
        from core.common import SimulationResult, DurationBreakdown
        from core.exporter import result_to_dict
        r = SimulationResult(total_latency_ns=1000,
                             breakdown=DurationBreakdown(compute_ns=700, transfer_ns=300),
                             diagnostics={"total_operators": 2, "finished_operators": 2})
        d = result_to_dict(r)
        self.assertEqual(d["total_latency_ms"], 0.001)
        self.assertEqual(d["diagnostics"]["finished_operators"], 2)
        self.assertIn("movement_bytes", d)

    def test_critical_path_and_local_rw(self):
        # v3.2：输出系统有关键路径归因 + breakdown/算子级本地读写
        from core.common import SimulationResult, DurationBreakdown, OperatorTiming
        from core.exporter import result_to_dict, build_critical_path
        ts = [OperatorTiming(op_id="a", op_type="GEMM", hardware="gpu0",
                             start_ns=0, end_ns=3000, duration_ns=3000,
                             compute_ns=2000, local_read_ns=500, local_write_ns=500),
              OperatorTiming(op_id="b", op_type="GEMM", hardware="gpu1",
                             start_ns=3000, end_ns=6000, duration_ns=3000,
                             compute_ns=2500, transfer_ns=500)]
        r = SimulationResult(total_latency_ns=6000, operator_timings=ts,
                             breakdown=DurationBreakdown(compute_ns=4500,
                                                         local_read_ns=500,
                                                         local_write_ns=500,
                                                         transfer_ns=500))
        cp = build_critical_path(r)
        self.assertGreaterEqual(len(cp["ops"]), 2)
        # 回溯链应含 a 与 b，且 b 在最末（最晚结束）
        self.assertEqual(cp["ops"][-1]["op_id"], "b")
        # 本地读写进入 breakdown 与算子级
        d = result_to_dict(r)
        self.assertEqual(d["breakdown"]["local_read_ns"], 500)
        self.assertEqual(d["breakdown"]["local_rw_ns"], 1000)
        self.assertEqual(d["operator_timings"][0]["local_read_ns"], 500)
        self.assertIn("analysis", d)
        self.assertIn("critical_path", d["analysis"])


class TestHardwareSystem(unittest.TestCase):
    def test_parse_and_factory(self):
        # 硬件系统：从 YAML 解析出 HardwareConfig 并经工厂装配成 HardwareUnit
        import os
        from core.common import load_yaml
        from core.hardware_sys import parse_hardware, HardwareFactory
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        doc = load_yaml(os.path.join(base, "configs", "hardware.yaml"))
        cfg = parse_hardware(doc)
        units = HardwareFactory.create_devices(cfg)
        self.assertIn("gpu0", units)
        gpu = units["gpu0"]
        self.assertEqual(gpu.name, "gpu0")
        self.assertGreater(gpu.peak_compute_flops, 0)
        self.assertTrue(gpu.memory_capacity_bytes > 0)

    def test_interconnect_parse(self):
        import os
        from core.common import load_yaml
        from core.hardware_sys import parse_hardware, parse_interconnect
        from core.link_sys import LinkBandwidthTable
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hw_doc = load_yaml(os.path.join(base, "configs", "hardware.yaml"))
        int_doc = load_yaml(os.path.join(base, "configs", "interconnect.yaml"))
        cfg = parse_hardware(hw_doc)
        table = parse_interconnect(int_doc, cfg)
        self.assertIsInstance(table, LinkBandwidthTable)
        # 对称：GPU↔DRAM_PIM 写一个值，双向同带宽
        self.assertGreater(table.get_bw_gbs("GPU", "DRAM_PIM"), 0)
        self.assertEqual(table.get_bw_gbs("GPU", "DRAM_PIM"),
                         table.get_bw_gbs("DRAM_PIM", "GPU"))

    def test_precision_specific_peak(self):
        # v3.1：不同精度算力应不同（INT8/INT4 通常快于 FP16/FP32）
        import os
        from core.common import load_yaml
        from core.hardware_sys import parse_hardware, HardwareFactory
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg = parse_hardware(load_yaml(os.path.join(base, "configs", "hardware.yaml")))
        gpu = HardwareFactory.create_devices(cfg)["gpu0"]
        fp16, int8, int4 = (gpu.get_peak_compute(p) for p in ("FP16", "INT8", "INT4"))
        self.assertGreater(fp16, 0)
        self.assertGreater(int8, fp16)        # INT8 > FP16
        self.assertGreater(int4, int8)        # INT4 > INT8
        # 不可执行精度：RERAM FP32 算力为 0（能力表 execution 不含 FP32）
        rram = HardwareFactory.create_devices(cfg)["rram0"]
        self.assertEqual(rram.get_peak_compute("FP32"), 0.0)

    def test_all_op_efficiency(self):
        # v3.1：8 类 op_type 在五类硬件上应有效率（未配置项不得回退 1.0）
        import os
        from core.common import load_yaml
        from core.hardware_sys import parse_hardware, HardwareFactory
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg = parse_hardware(load_yaml(os.path.join(base, "configs", "hardware.yaml")))
        units = HardwareFactory.create_devices(cfg)
        ops8 = ["GEMM", "LayerNorm", "Softmax", "Activation",
                "Residual", "LMHead", "Embedding", "KVCacheUpdate"]
        for name, hw in units.items():
            for op in ops8:
                self.assertNotEqual(hw.efficiency_for(op), 1.0,
                                    f"{name} 缺 {op} 效率（回落 1.0）")



class TestSchedulerSystem(unittest.TestCase):
    def test_engine_runs_small(self):
        # 用最小硬件+算子跑通离散事件调度
        from core.engine import SimulationEngine, PerformanceModel
        from core.common import Operator, DataObject, DataType, OperatorCategory
        from core.precision import PrecisionLevel

        class FakeHW:
            id = "gpu0"; name = "gpu0"
            peak_compute_flops = 1e12
            read_bandwidth_Bps = 1e9; write_bandwidth_Bps = 1e9
            read_latency_ns = 0; write_latency_ns = 0

            def can_execute(self, op):
                return True
            def efficiency_for(self, op_type):
                return 1.0

        hw_units = {"gpu0": FakeHW()}
        op = Operator(id="op1", name="GEMM1", op_type="GEMM", flops=1e9,
                      category=OperatorCategory.LINEAR,
                      data_precision=PrecisionLevel.FP16, input_ids=["x"], output_ids=["y"])
        data = DataObject(id="x", name="x", data_type=DataType.ACTIVATION, size_bytes=100)
        eng = SimulationEngine()
        eng.build(hardware_units=hw_units, interconnect=None,
                  operators=[op], data_objects=[data],
                  compute_map={"op1": "gpu0"}, placement={"x": ["gpu0"]})
        res = eng.run(seed=42)
        self.assertGreater(res.total_latency_ns, 0)
        self.assertEqual(res.diagnostics["finished_operators"], 1)

    def _mk_hw(self, name, peak=1e12):
        class FakeHW:
            peak_compute_flops = peak
            read_bandwidth_Bps = 1e9; write_bandwidth_Bps = 1e9
            read_latency_ns = 0; write_latency_ns = 0
            def can_execute(self, op): return True
            def efficiency_for(self, ot): return 1.0
            def get_peak_compute(self, prec=None): return self.peak_compute_flops
        h = FakeHW(); h.id = name; h.name = name
        return h

    def test_op_split_parallel(self):
        # v3.2 算子切片 → 多设备并行：整块 3e12 FLOP 跑 ~3000ns(ns)，切 3 片并行 → ~1000ns
        from core.engine import SimulationEngine
        from core.common import Operator, DataObject, DataType, OperatorCategory
        from core.precision import PrecisionLevel
        hw = {"gpu0": self._mk_hw("gpu0"), "gpu1": self._mk_hw("gpu1"),
              "gpu2": self._mk_hw("gpu2")}
        op = Operator(id="big", name="B", op_type="GEMM", flops=3e12,
                      category=OperatorCategory.LINEAR,
                      data_precision=PrecisionLevel.FP16,
                      execution_precision=PrecisionLevel.FP16,
                      input_ids=["x"], output_ids=["y"])
        data = DataObject(id="x", name="x", data_type=DataType.ACTIVATION, size_bytes=100)
        # 不切片基线
        eng = SimulationEngine()
        eng.build(hardware_units=hw, interconnect=None, operators=[op], data_objects=[data],
                  compute_map={"big": "gpu0"}, placement={"x": ["gpu0"]})
        t_single = eng.run(seed=1).total_latency_ns
        # 切 3 片 → 并行
        eng2 = SimulationEngine()
        eng2.build(hardware_units=hw, interconnect=None, operators=[op], data_objects=[data],
                   compute_map={"big": "gpu0"}, placement={"x": ["gpu0"]},
                   op_splits={"big": {"devices": ["gpu0", "gpu1", "gpu2"]}})
        res = eng2.run(seed=1)
        self.assertLess(res.total_latency_ns, t_single)   # 并行应更快
        self.assertAlmostEqual(res.total_latency_ns, t_single / 3, delta=1)
        self.assertEqual(res.diagnostics["finished_operators"], 1)
        # 聚合后仍是逻辑算子一条 timing，compute 总和不变（compute_ns：flops/peak*1e9 ns）
        agg = [t for t in res.operator_timings if t.op_id == "big"]
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0].compute_ns, int(3e9))   # 3e12/1e12*1e9 ns = 3e9 ns

    def test_op_split_dim_weight_shard(self):
        # v3.3 维度切分 + 权重分片式张量并行：
        #   - slice_flops 按维度比例（非均分）传入；
        #   - 算子权重被分片到各切片设备 → weight_shards/ALL-GATHER 生效。
        from core.engine import SimulationEngine
        from core.common import Operator, DataObject, DataType, OperatorCategory
        from core.precision import PrecisionLevel
        hw = {"gpu0": self._mk_hw("gpu0"), "gpu1": self._mk_hw("gpu1")}
        # flops=2e12（两片各 1e12）→ 单片 compute=1e12/(1e12)*1e9=1e9 ns
        op = Operator(id="ffn_down", name="B", op_type="GEMM", flops=2e12,
                      category=OperatorCategory.LINEAR,
                      data_precision=PrecisionLevel.FP16,
                      execution_precision=PrecisionLevel.FP16,
                      input_ids=["act", "w"], output_ids=["y"])
        act = DataObject(id="act", name="act", data_type=DataType.ACTIVATION, size_bytes=100)
        w = DataObject(id="w", name="w", data_type=DataType.WEIGHT, size_bytes=20)
        eng = SimulationEngine()
        # dim=N 切 2 片：每片 flops=1e12(维度比例)，权重 w 分片到 gpu0/gpu1
        eng.build(hardware_units=hw, interconnect=None, operators=[op],
                  data_objects=[act, w],
                  compute_map={"ffn_down": "gpu0"},
                  placement={"act": ["gpu0"], "w": ["gpu0"]},
                  op_splits={"ffn_down": {"devices": ["gpu0", "gpu1"],
                                          "slice_flops": [1e12, 1e12], "dim": "N"}})
        res = eng.run(seed=1)
        self.assertEqual(res.diagnostics["finished_operators"], 1)
        # 权重被分片且放入 ALL-GATHER（weight_shards 有 w）
        self.assertIn("w", eng.weight_shards)
        self.assertEqual(len(eng.weight_shards["w"]), 2)   # 2 片 → gpu0/gpu1
        # 聚合逻辑算子：compute 总和 = 2e9 ns（2×1e12 flops / 1e12 峰值）
        agg = [t for t in res.operator_timings if t.op_id == "ffn_down"]
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0].compute_ns, int(2e9))

    def test_link_concurrency(self):
        # v3.2 链路并发规则：同一链路同时只搬一份 → 两次搬运同源同目的则串行排队
        from core.engine import SimulationEngine
        from core.link_sys import LinkBandwidthTable
        from core.common import Operator, DataObject, DataType, OperatorCategory
        from core.precision import PrecisionLevel

        src = self._mk_hw("src"); src.type_name = "SRC"
        tgt = self._mk_hw("tgt"); tgt.type_name = "TGT"
        hw = {"src": src, "tgt": tgt}
        table = LinkBandwidthTable(); table.set_bw("SRC", "TGT", 1.0)   # 1 GB/s 对称

        data = DataObject(id="x", name="x", data_type=DataType.ACTIVATION, size_bytes=int(1e9))
        # 两个算子都在 tgt 上，都要从 src 读 x（走同一条链路 src->tgt）
        opA = Operator(id="A", name="A", op_type="GEMM", flops=1,
                       category=OperatorCategory.LINEAR, data_precision=PrecisionLevel.FP16,
                       input_ids=["x"], output_ids=["a"])
        opB = Operator(id="B", name="B", op_type="GEMM", flops=1,
                       category=OperatorCategory.LINEAR, data_precision=PrecisionLevel.FP16,
                       input_ids=["x"], output_ids=["b"])
        eng = SimulationEngine()
        eng.build(hardware_units=hw, link_table=table, operators=[opA, opB],
                  data_objects=[data],
                  compute_map={"A": "tgt", "B": "tgt"},
                  placement={"x": ["src"]})     # x 初始在 src，不冗余到 tgt
        res = eng.run(seed=1)
        # 每次搬运 1e9B/1e9Bps = 1s（链路段）；两次搬运同一链路串行 → 至少 2s
        self.assertGreaterEqual(res.total_latency_ns, 2_000_000_000)
        self.assertEqual(res.diagnostics["finished_operators"], 2)

    def test_link_concurrency_helpers(self):
        # 直接验证 Scheduler.link_state 对同链路排队的设定
        from core.common import Operator, DataObject, DataType, OperatorCategory
        from core.engine import Scheduler, PerformanceModel
        from core.link_sys import LinkBandwidthTable
        from core.precision import PrecisionLevel

        src, tgt = self._mk_hw("src"), self._mk_hw("tgt")
        src.type_name = "SRC"; tgt.type_name = "TGT"
        hw = {"src": src, "tgt": tgt}
        table = LinkBandwidthTable(); table.set_bw("SRC", "TGT", 1.0)
        data = DataObject(id="x", name="x", data_type=DataType.ACTIVATION, size_bytes=int(1e9))
        op = Operator(id="A", name="A", op_type="GEMM", flops=1,
                      category=OperatorCategory.LINEAR, data_precision=PrecisionLevel.FP16,
                      input_ids=["x"], output_ids=["a"])
        perf = PerformanceModel(hw, None, table)
        sched = Scheduler(operators={"A": op}, data_objects={"x": data},
                          hardware_units=hw, interconnect=None, perf_model=perf,
                          compute_map={"A": "tgt"}, placement={"x": ["src"]})
        sched.initialize()
        # 第一次 transfer
        sched._ensure_data_at("x", "src", "tgt", 0, [0], [0], [], "A", op, None)
        t1 = sched.link_state.get(("src", "tgt"), 0)
        # 第二次同链路 transfer（源就绪 0）应排队到 t1 之后
        sched._ensure_data_at("x", "src", "tgt", 0, [0], [0], [], "A", op, None)
        t2 = sched.link_state.get(("src", "tgt"), 0)
        self.assertGreater(t2, t1)   # 两次同链路 → 第二次起点被推到第一次结束之后


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    unittest.main(verbosity=2)
