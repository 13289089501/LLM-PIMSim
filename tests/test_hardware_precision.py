"""Hardware Model：多精度 Peak Compute 基础测试（纯 stdlib，无需 pytest）。

覆盖范围（仅 Hardware 数据层查询，不触碰 Performance/Operator 逻辑）：
  T1 同一 Hardware 可保存多精度 Peak Compute
  T2 FP16 与 INT8 返回不同 Peak Compute
  T3 正确判断“已配置”精度为支持
  T4 正确判断“未配置”精度为不支持
  T5 不同 Hardware 可配置不同数量/类型的精度

附加：
  单元换算独立性（TFLOPS/TOPS/GMAC/s 各精度独立存值）
  字符串精度查询、旧格式向后兼容，以及从 configs/hardware.yaml 实际加载验证。
"""
import os
import sys
import unittest

# 让测试能导入 v2 根目录下的模块（独立运行时可发现 tests/../ = v2）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts import PrecisionLevel as PL, OperatorCategory as OC, Operator
from hardware import HardwareUnit
from hardware_factory import HardwareFactory
import config_loader


def _mk_op(category, data, execution):
    return Operator(id="t", name="t", op_type="X", flops=1,
                    category=category, data_precision=data,
                    execution_precision=execution)


def _gpu_peak_map():
    return {
        PL.FP32: 100e12,
        PL.FP16: 300e12,
        PL.INT8: 600e12,
    }


def _make_hw(peak_map, name="hw0"):
    return HardwareUnit(
        id=name, name=name, device_type=DeviceType_GPU(),
        peak_compute_flops=peak_map.get(PL.FP16, 0.0),
        peak_throughput_by_precision=dict(peak_map),
        supported_precision=list(peak_map.keys()),
    )


def DeviceType_GPU():
    from contracts import DeviceType
    return DeviceType.GPU


class TestMultiPrecisionStorage(unittest.TestCase):
    """T1: 同一 Hardware 可保存多个精度的 Peak Compute。"""

    def test_same_hw_holds_multiple_precisions(self):
        hw = _make_hw(_gpu_peak_map())
        self.assertEqual(len(hw.peak_throughput_by_precision), 3)
        self.assertIn(PL.FP32, hw.peak_throughput_by_precision)
        self.assertIn(PL.FP16, hw.peak_throughput_by_precision)
        self.assertIn(PL.INT8, hw.peak_throughput_by_precision)

    def test_units_are_independent_per_precision(self):
        # 同一硬件上用不同单位(TOPS/TFLOPS/GMAC/s)独立存值
        hw = _make_hw({
            PL.FP16: 300e12,        # 300 TFLOPS
            PL.INT8: 600e12,        # 600 TOPS
            PL.INT4: 200e12,        # 200 TMAC/s (=2*100e12)
        })
        self.assertNotEqual(
            hw.get_peak_compute(PL.FP16),
            hw.get_peak_compute(PL.INT8),
        )
        # INT4 也是独立的一个值且数值正确（200 TMAC/s => 2e14 FLOPS）
        self.assertEqual(hw.get_peak_compute(PL.INT4), 200e12)


class TestDistinctValues(unittest.TestCase):
    """T2: 不同精度可返回不同 Peak Compute。"""

    def test_fp16_vs_int8(self):
        hw = _make_hw(_gpu_peak_map())
        self.assertEqual(hw.get_peak_compute(PL.FP16), 300e12)
        self.assertEqual(hw.get_peak_compute(PL.INT8), 600e12)
        self.assertNotEqual(hw.get_peak_compute(PL.FP16), hw.get_peak_compute(PL.INT8))

    def test_string_query(self):
        hw = _make_hw(_gpu_peak_map())
        self.assertEqual(hw.get_peak_compute("FP16"), 300e12)
        self.assertEqual(hw.get_peak_compute("int8"), 600e12)  # 大小写/格式不敏感


class TestSupportedKnownPrecision(unittest.TestCase):
    """T3: 已配置的精度应判定为支持。"""

    def test_configured_is_supported(self):
        hw = _make_hw(_gpu_peak_map())
        self.assertTrue(hw.supports_compute_precision(PL.FP16))
        self.assertTrue(hw.supports_compute_precision(PL.INT8))
        self.assertTrue(hw.supports_compute_precision("FP32"))


class TestUnsupportedMissingPrecision(unittest.TestCase):
    """T4: 未配置的精度应判定为不支持（返回 0 算力）。"""

    def test_missing_is_unsupported(self):
        hw = _make_hw(_gpu_peak_map())  # 只配 FP32/FP16/INT8
        self.assertFalse(hw.supports_compute_precision(PL.INT4))
        # 未配置精度返回 0.0 算力
        self.assertEqual(hw.get_peak_compute(PL.INT4), 0.0)


class TestDifferentHwDifferentPrecisions(unittest.TestCase):
    """T5: 不同 Hardware 可配置不同数量/类型的精度。"""

    def test_gpu_pim_sram_vary(self):
        gpu = _make_hw(_gpu_peak_map(), "gpu0")
        pim = _make_hw({PL.INT8: 50e12}, "pim0")
        sram = _make_hw({PL.INT8: 200e12, PL.INT4: 400e12}, "sram0")

        # GPU: FP32+FP16+INT8
        self.assertEqual(set(gpu.supported_precision), {PL.FP32, PL.FP16, PL.INT8})
        self.assertTrue(gpu.supports_compute_precision("FP16"))
        self.assertFalse(gpu.supports_compute_precision("INT4"))

        # PIM: 仅 INT8
        self.assertEqual(set(pim.supported_precision), {PL.INT8})
        self.assertTrue(pim.supports_compute_precision("INT8"))
        self.assertFalse(pim.supports_compute_precision("FP16"))  # 未配置即不支持
        self.assertEqual(pim.get_peak_compute("INT8"), 50e12)

        # SRAM: INT8 + INT4
        self.assertEqual(set(sram.supported_precision), {PL.INT8, PL.INT4})
        self.assertFalse(sram.supports_compute_precision("FP16"))


class TestYamlAndFactory(unittest.TestCase):
    """从真实 configs/hardware.yaml 加载 + 工厂装配，验证配置端派生与旧格式兼容。"""

    @classmethod
    def setUpClass(cls):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hw_path = os.path.join(base, "configs", "hardware.yaml")
        from core.common import load_yaml
        from core.hardware_sys import parse_hardware
        doc = load_yaml(hw_path)
        cls.hw_cfg = parse_hardware(doc)
        cls.units = HardwareFactory.create_devices(cls.hw_cfg)

    def test_yaml_multi_precision_derived(self):
        gpu_cfg = self.hw_cfg["gpu0"]
        # v2.3+ hardware.yaml 用单值主峰值(peak_tflops)，precision 由 peak_by_precision 派生
        #（旧格式单值会回填到默认全部精度），且配置层 precision 与 peak_by_precision 保持一致。
        self.assertEqual(set(gpu_cfg.precision), set(gpu_cfg.peak_by_precision.keys()))
        self.assertIn(PL.FP16, gpu_cfg.peak_by_precision)
        self.assertIn(PL.INT8, gpu_cfg.peak_by_precision)
        # 单值主峰值填满默认精度：INT4 在配置层也声明（A100 支持 INT4）
        gpu = self.units["gpu0"]
        self.assertTrue(gpu.supports_compute_precision("FP16"))
        self.assertTrue(gpu.supports_compute_precision("INT8"))
        # 运行时真正"能否执行某算子"由 precisions.py 固定三维 Capability 判定。
        # v3 修正：GPU 现可执行 LINEAR + NONLINEAR（真实 GPU 可跑 LayerNorm/Softmax），
        # 早期版本 GPU 能力写成 [LINEAR]，导致任何 GPU-only 部署都无法完成——该设计矛盾已修正。
        self.assertTrue(gpu.can_execute(_mk_op(
            OC.NONLINEAR, PL.FP16, PL.FP16)))     # GPU 现支持 NONLINEAR 类别 + FP16

    def test_yaml_minimal_precision_hw(self):
        pim = self.units["pim0"]
        # DRAM-PIM 单值主峰(1.2 TFLOPS FP16)；配置层派生 fill 默认精度
        self.assertIn(PL.FP16, pim.supported_precision)
        self.assertEqual(pim.get_peak_compute("FP16"), 1.2e12)
        # DRAM-PIM 仅 LINEAR：NONLINEAR 不可执行
        self.assertFalse(pim.can_execute(_mk_op(OC.NONLINEAR, PL.FP16, PL.FP32)))

        sram = self.units["sram0"]
        self.assertEqual(sram.get_peak_compute("FP16"), 500e12)
        # SRAM-PIM 支持 NONLINEAR 类别（且支持 FP16 execution）
        self.assertTrue(sram.can_execute(_mk_op(OC.NONLINEAR, PL.FP16, PL.FP16)))
        # 但 SRAM execution 不含 FP32：NONLINEAR/FP32 不可执行
        self.assertFalse(sram.can_execute(_mk_op(OC.NONLINEAR, PL.FP16, PL.FP32)))

    def test_compat_legacy_single_value(self):
        # 旧格式单值（peak_tops）应回填到全部默认精度，且旧接口 peak_compute_flops 仍在
        cfg = config_loader.HardwareConfig(
            id="legacy", type="GPU",
            peak_f=100e12,
            peak_by_precision={p: 100e12 for p in [PL.FP32, PL.FP16, PL.INT8, PL.INT4]},
            mem_bytes=1, read_bw=1, write_bw=1, read_lat_ns=1, write_lat_ns=1,
            parallelism=1, efficiency={}, precision=[PL.FP32, PL.FP16, PL.INT8, PL.INT4],
        )
        hw = HardwareFactory.create_devices({"legacy": cfg})["legacy"]
        self.assertEqual(hw.peak_compute_flops, 100e12)   # 兼容字段仍有效
        self.assertEqual(hw.get_peak_compute("FP16"), 100e12)
        self.assertEqual(hw.get_peak_compute(None), 100e12)  # 空 => 用兼容字段


if __name__ == "__main__":
    unittest.main(verbosity=2)
