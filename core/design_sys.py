"""
LLM-PIMSim v4 core.design_sys — 【硬件设计系统】

在原有"用户直接输入硬件参数"的自定义硬件路径之外，新增一条"基于真实硬件结构
设计并自动生成参数"的路径。用户从硬件架构出发（存内计算 / 近存计算），选择
存储介质、计算资源、互联方式与资源组织方式；系统据此建立硬件结构模型，再自动
推导出与原有自定义硬件模块完全一致的硬件参数（HardwareConfig 同格式），从而
零侵入地复用后续的精度检查、算子映射、性能计算与仿真流程。

三阶段流水线（严格按设计规格）：
  ① 用户设计规格（UserDesignSpec）：记录总体架构、介质、计算资源、容量与规模、
     资源对应关系、互联方式、部署层级。
  ② 硬件结构模型（HardwareStructureModel）：由规格自动生成通道数、存储阵列/宏块
     数量、计算资源数量、资源在层级上的分布与连接关系；每条"存储↔计算"连接
     记录互联方式并计算有效数据带宽 = min(存储带宽, 互联带宽, 计算接口带宽)。
  ③ 参数推导：由结构模型生成原有仿真器需要的硬件参数（峰值算力、效率表、
     精度能力、容量、读写带宽/延迟…），输出为 core.hardware_sys.HardwareConfig
     同格式对象 + 链路表条目（近存计算的有效内部带宽写入该设计种类的对角线）。

设计决策（v1，已与用户确认）：
  - 存内计算（CIM）：存储介质决定基本计算机制（SRAM→位线/字线数字计算、
    DRAM→存储库/子阵列数字计算、ReRAM→交叉阵列电流累加模拟计算），用户不选
    计算单元，只设总容量 + 计算资源密度（低/中/高/自定义），介质模型把密度转成
    实际阵列数与计算并行度。一个设计 → 一个设备（存储与计算天然耦合）。
  - 近存计算（NMC）：存储资源与计算资源分开建模，一个设计 → 单个设备
    （存储参数来自存储介质、算力参数来自计算资源），每条连接的有效内部带宽
    写入该设备种类在链路表中的"对角线"（同种类自互连）条目。
  - 算子效率：本系统内置一套独立效率规则（按计算机制类别 × 算子类型 × 精度 ×
    密度），与原有出厂效率表无关，但输出仍是 {op_type: 利用率} 字典，不修改
    原有效率数据结构和计算逻辑。
  - 资源数量不做硬编码上限（list 可扩展）；第一版 UI 层限制 ≤2 种存储/计算资源。

依赖：core.common（ConfigError）+ core.precision（PrecisionLevel）+
     core.hardware_sys（HardwareConfig，仅借其字段格式）。
不依赖：调度 / 校验 / 输出 / 权重 / 切割系统。
"""

from dataclasses import dataclass, field
from enum import IntEnum
from math import ceil
from typing import Optional

from core.common import ConfigError
from core.precision import PrecisionLevel
from core.hardware_sys import HardwareConfig


# =================================================================
# 0. 枚举与常量
# =================================================================
class ArchitectureType(IntEnum):
    """总体架构：存内计算 / 近存计算（第一版不提供"混合"顶层选项——
    用户通过同时选择多种介质/计算资源与组织方式即可构建混合硬件）。"""
    IN_MEMORY = 0    # 存内计算 CIM
    NEAR_MEMORY = 1  # 近存计算 NMC


class DeploymentMode(IntEnum):
    """资源部署层级（第一版实现通道级两种；系统内部用"部署层级"概念，
    未来可扩展 BANK / STACK / CHIP 等层级）。"""
    CHANNEL_INTERNAL = 0   # 通道内混合：每个通道内同时存在多种资源
    CHANNEL_CROSS = 1      # 通道间混合：不同资源分别部署在不同通道


class DensityLevel(IntEnum):
    """计算资源密度 / 计算并行度等级。"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CUSTOM = 3


DENSITY_LEVELS = {
    "LOW":    {"label": "低", "factor": 0.3},
    "MEDIUM": {"label": "中", "factor": 1.0},
    "HIGH":   {"label": "高", "factor": 2.0},
    "CUSTOM": {"label": "自定义", "factor": None},
}

# 通道间混合的跨通道访问开销（ns/通道跳）与通道共享带宽折减，v1 近似模型
CROSS_CHANNEL_LAT_NS = 20.0

# 效率字典固定使用的 8 类算子 key（与 core.operator_sys 的 op_type 一一对应）
EFF_OP_KEYS = ["GEMM", "LayerNorm", "Softmax", "Activation",
               "Residual", "LMHead", "Embedding", "KVCacheUpdate"]

_PRECISION_NAMES = ["FP32", "BF16", "FP16", "FP8", "INT8", "INT4"]


# =================================================================
# 1. 预设库
# =================================================================
# 存储介质预设（CIM 与 NMC 存储资源共用；NMC 只用其存储侧参数）。
# 容量用字节；阵列规模（array_bytes）为"默认宏块/子阵列/交叉阵列"尺寸；
# flops_per_array / read_bw_gbs_per_array / write_bw_gbs_per_array 为"单个阵列"
# 的标定值——以"推荐容量 + 中等密度"为基准复现现有出厂设备的量级：
#   SRAM-CIM 512MB→500 TFLOPS / 1.5PB/s；DRAM-CIM 8GB→1.2 TFLOPS / 307.2GB/s；
#   ReRAM-CIM 256MB→20 TFLOPS / 读128 写32 GB/s。
MEDIA_PRESETS = {
    "SRAM": {
        "label": "SRAM（静态随机存取存储器）",
        "mechanism": "数字位线/字线计算（bitline/wordline digital compute）",
        "array_kind": "存储宏块 macro",
        "array_bytes": 256 * 2 ** 10,            # 256 KB / 宏块
        "capacity_bytes_min": 8 * 2 ** 20,       # 8 MB
        "capacity_bytes_recommend": 512 * 2 ** 20,  # 512 MB
        "capacity_bytes_max": 4 * 2 ** 30,       # 4 GB
        "capacity_granularity_bytes": 2 ** 20,   # 1 MB
        "flops_per_array": 244.1e9,              # ≈122 MAC/cycle @1GHz
        "parallelism_per_array": 1,
        "read_bw_gbs_per_array": 732.4,          # 1.5 PB/s ÷ 2048 宏块
        "write_bw_gbs_per_array": 732.4,
        "read_lat_ns": 2,
        "write_lat_ns": 2,
        "device_type": "SRAM_PIM",               # 能力语义（见 core.precision）
        "precision_scale": {"FP32": 0.25, "BF16": 1.0, "FP16": 1.0,
                            "FP8": 2.0, "INT8": 2.0, "INT4": 4.0},
        "efficiency_class": "SRAM_CIM",
    },
    "DRAM": {
        "label": "DRAM（动态随机存取存储器）",
        "mechanism": "存储库/子阵列数字计算（bank/subarray digital compute）",
        "array_kind": "存储库 bank",
        "array_bytes": 64 * 2 ** 10,             # 64 KB / 子阵列
        "capacity_bytes_min": 1 * 2 ** 30,       # 1 GB
        "capacity_bytes_recommend": 8 * 2 ** 30, # 8 GB
        "capacity_bytes_max": 64 * 2 ** 30,      # 64 GB
        "capacity_granularity_bytes": 2 ** 30,   # 1 GB
        "flops_per_array": 9.16e6,               # 8GB ÷ 64KB = 131072 → 1.2 TFLOPS
        "parallelism_per_array": 1,
        "read_bw_gbs_per_array": 2.34,           # 307.2 GB/s ÷ 131072
        "write_bw_gbs_per_array": 2.34,
        "read_lat_ns": 50,
        "write_lat_ns": 50,
        "device_type": "DRAM_PIM",
        "precision_scale": {"FP32": 0.5, "BF16": 1.0, "FP16": 1.0,
                            "FP8": 2.0, "INT8": 2.0, "INT4": 4.0},
        "efficiency_class": "DRAM_CIM",
    },
    "RRAM": {
        "label": "ReRAM（阻变存储器）",
        "mechanism": "交叉阵列电流累加模拟计算（crossbar analog current accumulation）",
        "array_kind": "交叉阵列 crossbar",
        "array_bytes": 1 * 2 ** 20,              # 1 MB / 交叉阵列
        "capacity_bytes_min": 8 * 2 ** 20,       # 8 MB
        "capacity_bytes_recommend": 256 * 2 ** 20,  # 256 MB
        "capacity_bytes_max": 4 * 2 ** 30,       # 4 GB
        "capacity_granularity_bytes": 2 ** 20,   # 1 MB
        "flops_per_array": 78.1e9,               # 256MB ÷ 1MB = 256 → 20 TFLOPS
        "parallelism_per_array": 1,
        "read_bw_gbs_per_array": 0.5,            # 128 GB/s ÷ 256
        "write_bw_gbs_per_array": 0.125,         # 32 GB/s ÷ 256
        "read_lat_ns": 10,
        "write_lat_ns": 100,
        "device_type": "RERAM_PIM",
        "precision_scale": {"FP32": 0.0, "BF16": 0.5, "FP16": 1.0,
                            "FP8": 1.0, "INT8": 1.0, "INT4": 2.0},
        "efficiency_class": "RRAM_CIM",
    },
}

# 近存计算资源预设（NMC 的计算侧）。
# flops_per_unit：单个计算单元的峰值算力；interface_bw_gbs：计算单元接口带宽
# （有效内部带宽 = min(存储带宽, 互联带宽, 计算接口带宽) 的组成项之一）；
# device_type：该计算资源在 core.precision.HARDWARE_CAPABILITY 中采用的能力语义。
COMPUTE_PRESETS = {
    "MAC_ARRAY": {
        "label": "数字矩阵乘加阵列（digital MAC array）",
        "flops_per_unit": 32.8e12,               # 128×128 MAC @1GHz ≈ 32.8 TFLOPS
        "parallelism_per_unit": 1,
        "interface_bw_gbs": 256.0,
        "buffer_bytes": 2 * 2 ** 20,             # 2 MB 片上暂存
        "device_type": "DRAM_PIM",               # 纯线性（矩阵乘）能力语义
        "precision_scale": {"FP32": 0.5, "BF16": 1.0, "FP16": 1.0,
                            "FP8": 2.0, "INT8": 2.0, "INT4": 4.0},
        "efficiency_class": "MAC_ARRAY",
    },
    "CROSSBAR_ARRAY": {
        "label": "模拟交叉阵列（analog crossbar array）",
        "flops_per_unit": 20.0e12,
        "parallelism_per_unit": 1,
        "interface_bw_gbs": 64.0,
        "buffer_bytes": 1 * 2 ** 20,
        "device_type": "RERAM_PIM",              # 线性 + 低精度模拟能力语义
        "precision_scale": {"FP32": 0.0, "BF16": 0.5, "FP16": 1.0,
                            "FP8": 1.0, "INT8": 1.0, "INT4": 2.0},
        "efficiency_class": "CROSSBAR_ARRAY",
    },
    "SIMD_CLUSTER": {
        "label": "SIMD 核簇（SIMD core cluster）",
        "flops_per_unit": 15.0e12,               # 32 lane × 512 FLOP/cycle @1GHz
        "parallelism_per_unit": 8,
        "interface_bw_gbs": 512.0,
        "buffer_bytes": 8 * 2 ** 20,             # 8 MB 片上暂存
        "device_type": "GPU",                    # 通用计算（线 + 非线性）能力语义
        "precision_scale": {"FP32": 0.5, "BF16": 1.0, "FP16": 1.0,
                            "FP8": 1.5, "INT8": 2.0, "INT4": 2.0},
        "efficiency_class": "SIMD_CLUSTER",
    },
}

# 互联方式预设（NMC 存储↔计算 连接使用）
INTERCONNECT_PRESETS = {
    "BUS":  {"label": "内部总线（bus）", "bw_gbs": 64.0, "lat_ns": 20.0},
    "NOC":  {"label": "片上网络 NoC（mesh）", "bw_gbs": 256.0, "lat_ns": 10.0},
    "POINT_TO_POINT": {"label": "点对点直连（P2P）", "bw_gbs": 512.0, "lat_ns": 5.0},
}

# 独立算子效率规则（新规则集，与 core.hardware_sys 的出厂效率表无关）。
# key = 计算机制类别；value = 8 类算子的基础利用率。
# 最终效率 = 基础值 × 精度修正 × 密度修正（见 _efficiency_table）。
DESIGN_EFFICIENCY_RULES = {
    "SRAM_CIM": {
        # 数字位线计算：GEMM 高、elementwise 带宽受限中等
        "GEMM": 0.80, "LayerNorm": 0.25, "Softmax": 0.30,
        "Activation": 0.30, "Residual": 0.35, "LMHead": 0.75,
        "Embedding": 0.50, "KVCacheUpdate": 0.40,
    },
    "DRAM_CIM": {
        # 库级数字计算：GEMM 可用，带宽受限的 elementwise 低
        "GEMM": 0.70, "LayerNorm": 0.15, "Softmax": 0.20,
        "Activation": 0.15, "Residual": 0.20, "LMHead": 0.60,
        "Embedding": 0.30, "KVCacheUpdate": 0.25,
    },
    "RRAM_CIM": {
        # 模拟交叉阵列：GEMM 尚可，非线性/高精度类弱
        "GEMM": 0.60, "LayerNorm": 0.10, "Softmax": 0.10,
        "Activation": 0.10, "Residual": 0.15, "LMHead": 0.50,
        "Embedding": 0.25, "KVCacheUpdate": 0.20,
    },
    "MAC_ARRAY": {
        # 专用矩阵乘加阵列：GEMM 利用高，非 GEMM 算子需旁路支持、利用率低
        "GEMM": 0.85, "LayerNorm": 0.15, "Softmax": 0.15,
        "Activation": 0.20, "Residual": 0.25, "LMHead": 0.80,
        "Embedding": 0.40, "KVCacheUpdate": 0.30,
    },
    "CROSSBAR_ARRAY": {
        "GEMM": 0.60, "LayerNorm": 0.08, "Softmax": 0.08,
        "Activation": 0.08, "Residual": 0.12, "LMHead": 0.50,
        "Embedding": 0.20, "KVCacheUpdate": 0.15,
    },
    "SIMD_CLUSTER": {
        # 通用核簇：各类算子均有中等利用率
        "GEMM": 0.55, "LayerNorm": 0.30, "Softmax": 0.30,
        "Activation": 0.35, "Residual": 0.35, "LMHead": 0.60,
        "Embedding": 0.40, "KVCacheUpdate": 0.30,
    },
}

# 效率的精度修正（按设备"主执行精度"取一档；FP32 模拟/通用成本高、低精度高）
_EFF_PRECISION_MODIFIER = {
    PrecisionLevel.FP32: 0.60, PrecisionLevel.BF16: 1.00,
    PrecisionLevel.FP16: 1.00, PrecisionLevel.FP8: 1.10,
    PrecisionLevel.INT8: 1.10, PrecisionLevel.INT4: 1.20,
}


def _eff_density_modifier(density_factor: float) -> float:
    """密度对效率的温和修正：密度越高（计算资源越密集），利用率略升。"""
    return 0.9 + 0.1 * density_factor


# =================================================================
# 2. 用户设计规格（阶段①）
# =================================================================
@dataclass
class StorageSpec:
    """一条存储资源（NMC 用；CIM 由 spec.media 单介质表示）"""
    media: str                 # MEDIA_PRESETS 的 key
    capacity_bytes: int
    array_bytes: int = 0       # 0 = 用介质默认阵列规模


@dataclass
class ComputeSpec:
    """一条计算资源（NMC 用）"""
    resource: str              # COMPUTE_PRESETS 的 key
    count: int = 1             # 计算单元数量


@dataclass
class ConnectionSpec:
    """一条"存储资源 ↔ 计算资源"连接（记录互联方式）"""
    storage_idx: int           # 指向 UserDesignSpec.storages 的下标
    compute_idx: int           # 指向 UserDesignSpec.computes 的下标
    interconnect: str          # INTERCONNECT_PRESETS 的 key


@dataclass
class DeploymentSpec:
    """资源部署层级（通道级）"""
    mode: DeploymentMode = DeploymentMode.CHANNEL_INTERNAL
    channels: int = 1


@dataclass
class UserDesignSpec:
    """用户设计规格（阶段① 产物，直接由前端/调用方填写）"""
    name: str = "design"
    architecture: ArchitectureType = ArchitectureType.IN_MEMORY

    # ---- CIM 字段 ----
    media: str = ""            # 存储介质（CIM 必填）
    capacity_bytes: int = 0
    density: DensityLevel = DensityLevel.MEDIUM
    custom_density_factor: float = 1.0   # density=CUSTOM 时使用
    array_bytes: int = 0       # 0 = 介质默认阵列规模（"默认或用户指定的阵列规模"）

    # ---- NMC 字段（底层不写死 2 个上限）----
    storages: list = field(default_factory=list)     # list[StorageSpec]
    computes: list = field(default_factory=list)     # list[ComputeSpec]
    connections: list = field(default_factory=list)  # list[ConnectionSpec]
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)


# =================================================================
# 3. 硬件结构模型（阶段②）
# =================================================================
@dataclass
class ArrayGroup:
    """一类存储阵列的组织信息"""
    media: str
    array_kind: str
    array_bytes: int
    count: int                      # 阵列总数
    per_channel: list = field(default_factory=list)   # 各通道的阵列数


@dataclass
class UnitGroup:
    """一类计算资源单元的组织信息"""
    resource: str
    count: int
    per_channel: list = field(default_factory=list)


@dataclass
class ConnectionModel:
    """一条已解析的连接：互联方式 + 三方带宽 + 有效内部带宽"""
    storage_idx: int
    compute_idx: int
    interconnect: str
    storage_bw_gbs: float
    compute_if_bw_gbs: float
    link_bw_gbs: float
    effective_bw_gbs: float


@dataclass
class HardwareStructureModel:
    """具体硬件结构模型（阶段② 产物）"""
    architecture: ArchitectureType
    channels: int
    deployment: DeploymentMode
    media: str = ""
    arrays: list = field(default_factory=list)       # list[ArrayGroup]
    units: list = field(default_factory=list)        # list[UnitGroup]
    connections: list = field(default_factory=list)  # list[ConnectionModel]
    total_capacity_bytes: int = 0
    total_parallelism: int = 0


# =================================================================
# 4. 阶段②：结构模型构建
# =================================================================
def _media_preset(media: str) -> dict:
    up = str(media).strip().upper()
    pre = MEDIA_PRESETS.get(up)
    if pre is None:
        raise ConfigError(f"未知存储介质: '{media}'。可选: {sorted(MEDIA_PRESETS.keys())}")
    return pre


def _snap_capacity(capacity_bytes, pre: dict) -> int:
    """容量对齐到粒度，并校验在 [min, max] 内。"""
    g = int(pre["capacity_granularity_bytes"])
    cap = int(capacity_bytes)
    cap = int(ceil(cap / g)) * g
    if cap < pre["capacity_bytes_min"]:
        raise ConfigError(
            f"容量 {_fmt_bytes(cap)} 低于介质 {pre['label']} 允许下限 "
            f"{_fmt_bytes(pre['capacity_bytes_min'])}")
    if cap > pre["capacity_bytes_max"]:
        raise ConfigError(
            f"容量 {_fmt_bytes(cap)} 超过介质 {pre['label']} 允许上限 "
            f"{_fmt_bytes(pre['capacity_bytes_max'])}")
    return cap


def _fmt_bytes(b) -> str:
    """字节 → 可读字符串（十进制单位，与前端解析器 core.hardware_sys._frontend_* 口径一致）。"""
    b = float(b or 0)
    for unit, base in (("PB", 1e15), ("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= base:
            v = b / base
            return f"{v:.2f} {unit}" if v != int(v) else f"{int(v)} {unit}"
    return f"{int(b)} B"


def _fmt_gbs(gbs) -> str:
    g = float(gbs or 0)
    if g >= 1e6:
        return f"{g/1e6:.2f} PB/s"
    if g >= 1e3:
        return f"{g/1e3:.2f} TB/s"
    return f"{g:.2f} GB/s"


def _density_factor(spec: UserDesignSpec) -> float:
    if spec.density == DensityLevel.CUSTOM:
        f = float(spec.custom_density_factor or 0)
        if f <= 0:
            raise ConfigError("自定义计算密度必须 > 0")
        return f
    return float(DENSITY_LEVELS[spec.density.name]["factor"])


def build_structure_model(spec: UserDesignSpec) -> HardwareStructureModel:
    """阶段②：把用户设计规格转成具体硬件结构模型。"""
    if spec.architecture == ArchitectureType.IN_MEMORY:
        return _build_cim_structure(spec)
    return _build_nmc_structure(spec)


def _build_cim_structure(spec: UserDesignSpec) -> HardwareStructureModel:
    pre = _media_preset(spec.media)
    cap = _snap_capacity(spec.capacity_bytes or pre["capacity_bytes_recommend"], pre)
    array_bytes = int(spec.array_bytes) or int(pre["array_bytes"])
    num_arrays = max(1, int(ceil(cap / array_bytes)))
    df = _density_factor(spec)
    parallelism = max(1, int(round(num_arrays * df)))
    channels = max(1, int(spec.deployment.channels))
    per_channel = _distribute(num_arrays, channels)

    return HardwareStructureModel(
        architecture=spec.architecture,
        channels=channels,
        deployment=spec.deployment.mode,
        media=pre["label"],
        arrays=[ArrayGroup(media=spec.media, array_kind=pre["array_kind"],
                           array_bytes=array_bytes, count=num_arrays,
                           per_channel=per_channel)],
        total_capacity_bytes=cap,
        total_parallelism=parallelism,
    )


def _build_nmc_structure(spec: UserDesignSpec) -> HardwareStructureModel:
    if not spec.storages:
        raise ConfigError("近存计算至少需要 1 条存储资源")
    if not spec.computes:
        raise ConfigError("近存计算至少需要 1 种计算资源")
    if not spec.connections:
        raise ConfigError("近存计算需要定义至少 1 条 存储↔计算 连接")
    n_s, n_c = len(spec.storages), len(spec.computes)
    for c in spec.connections:
        if not (0 <= c.storage_idx < n_s and 0 <= c.compute_idx < n_c):
            raise ConfigError(
                f"连接指向不存在的资源: storage_idx={c.storage_idx}, compute_idx={c.compute_idx}"
                f"（当前 {n_s} 存储 × {n_c} 计算）")

    channels = max(1, int(spec.deployment.channels))
    arrays, units = [], []
    total_cap = 0
    total_par = 0
    storage_bw = []   # 各存储资源的读带宽（GB/s）
    compute_if = []   # 各计算资源的接口带宽（GB/s）
    for s in spec.storages:
        pre = _media_preset(s.media)
        cap = _snap_capacity(s.capacity_bytes or pre["capacity_bytes_recommend"], pre)
        array_bytes = int(s.array_bytes) or int(pre["array_bytes"])
        n_arr = max(1, int(ceil(cap / array_bytes)))
        total_cap += cap
        arrays.append(ArrayGroup(media=s.media, array_kind=pre["array_kind"],
                                 array_bytes=array_bytes, count=n_arr,
                                 per_channel=_distribute(n_arr, channels)))
        storage_bw.append(n_arr * pre["read_bw_gbs_per_array"])
    for c in spec.computes:
        cpre = _compute_preset(c.resource)
        n = max(1, int(c.count or 1))
        total_par += n * int(cpre["parallelism_per_unit"])
        units.append(UnitGroup(resource=c.resource, count=n,
                               per_channel=_distribute(n, channels)))
        compute_if.append(n * float(cpre["interface_bw_gbs"]))

    conns = []
    for c in spec.connections:
        ipre = _interconnect_preset(c.interconnect)
        s_bw = storage_bw[c.storage_idx]
        c_bw = compute_if[c.compute_idx]
        eff = min(s_bw, float(ipre["bw_gbs"]), c_bw)
        conns.append(ConnectionModel(
            storage_idx=c.storage_idx, compute_idx=c.compute_idx,
            interconnect=c.interconnect,
            storage_bw_gbs=round(s_bw, 3),
            compute_if_bw_gbs=round(c_bw, 3),
            link_bw_gbs=float(ipre["bw_gbs"]),
            effective_bw_gbs=round(eff, 3)))

    return HardwareStructureModel(
        architecture=spec.architecture,
        channels=channels,
        deployment=spec.deployment.mode,
        arrays=arrays, units=units, connections=conns,
        total_capacity_bytes=total_cap,
        total_parallelism=total_par,
    )


def _compute_preset(resource: str) -> dict:
    up = str(resource).strip().upper()
    pre = COMPUTE_PRESETS.get(up)
    if pre is None:
        raise ConfigError(f"未知计算资源: '{resource}'. 可选: {sorted(COMPUTE_PRESETS.keys())}")
    return pre


def _interconnect_preset(name: str) -> dict:
    up = str(name).strip().upper()
    pre = INTERCONNECT_PRESETS.get(up)
    if pre is None:
        raise ConfigError(f"未知互联方式: '{name}'. 可选: {sorted(INTERCONNECT_PRESETS.keys())}")
    return pre


def _distribute(n: int, channels: int) -> list:
    """把 n 个资源均分到 channels 个通道（余数摊给前几个通道）。"""
    if channels <= 1:
        return [n]
    base, rem = divmod(n, channels)
    return [base + (1 if i < rem else 0) for i in range(channels)]


# =================================================================
# 5. 阶段③：参数推导
# =================================================================
def _primary_exec_precision(scale: dict) -> PrecisionLevel:
    """设备的"主执行精度"：优先 FP16；模拟计算（FP32 为 0）取 INT8。"""
    for name in ("FP16", "BF16", "INT8"):
        if scale.get(name, 0) and scale.get(name, 0) > 0:
            return PrecisionLevel.from_name(name)
    return PrecisionLevel.FP16


def _efficiency_table(eff_class: str, scale: dict, density_factor: float) -> dict:
    """独立效率规则：基础表 × 精度修正(主执行精度) × 密度修正。"""
    base = DESIGN_EFFICIENCY_RULES.get(eff_class)
    if base is None:
        base = DESIGN_EFFICIENCY_RULES["MAC_ARRAY"]
    prec = _primary_exec_precision(scale)
    pm = _EFF_PRECISION_MODIFIER.get(prec, 1.0)
    dm = _eff_density_modifier(density_factor)
    out = {}
    for k in EFF_OP_KEYS:
        v = base.get(k, 0.5) * pm * dm
        out[k] = round(max(0.05, min(0.95, v)), 3)
    return out


def _merge_efficiency_tables(tables: list, scale_union: dict, density_factor: float) -> dict:
    """多计算资源/多机制时：按算子取各表最大值（由能力最强的单元承担该算子）。"""
    pm = _EFF_PRECISION_MODIFIER.get(_primary_exec_precision(scale_union), 1.0)
    dm = _eff_density_modifier(density_factor)
    out = {}
    for k in EFF_OP_KEYS:
        v = max((t.get(k, 0.5) for t in tables), default=0.5) * pm * dm
        out[k] = round(max(0.05, min(0.95, v)), 3)
    return out


def _peak_by_precision(peak_f: float, scale: dict) -> dict:
    out = {}
    for p in _PRECISION_NAMES:
        mult = float(scale.get(p, 0) or 0)
        if mult > 0:
            out[PrecisionLevel.from_name(p)] = peak_f * mult
    return out


def _supported_precision(scale: dict) -> list:
    return [PrecisionLevel.from_name(p) for p in _PRECISION_NAMES
            if float(scale.get(p, 0) or 0) > 0]


def derive_cim(spec: UserDesignSpec) -> tuple:
    """存内计算：一个设计 → 一个 HardwareConfig（存储与计算天然耦合）。"""
    struct = build_structure_model(spec)   # 阶段②
    pre = _media_preset(spec.media)
    df = _density_factor(spec)
    arr = struct.arrays[0]
    num_arrays = arr.count

    peak_f = num_arrays * float(pre["flops_per_array"]) * df
    read_bw = num_arrays * float(pre["read_bw_gbs_per_array"]) * 1e9   # B/s
    write_bw = num_arrays * float(pre["write_bw_gbs_per_array"]) * 1e9
    scale = dict(pre["precision_scale"])
    eff = _efficiency_table(pre["efficiency_class"], scale, df)
    kind = (spec.name or "cim").strip().upper().replace(" ", "-")

    hc = HardwareConfig(
        id=spec.name, type=pre["device_type"], peak_f=peak_f,
        peak_by_precision=_peak_by_precision(peak_f, scale),
        mem_bytes=struct.total_capacity_bytes,
        read_bw=read_bw, write_bw=write_bw,
        read_lat_ns=int(pre["read_lat_ns"]), write_lat_ns=int(pre["write_lat_ns"]),
        parallelism=struct.total_parallelism, efficiency=eff,
        precision=_supported_precision(scale),
        link_type=kind, links={kind: round(read_bw / 1e9, 3)},
    )
    return hc, struct, kind


def derive_nmc(spec: UserDesignSpec) -> tuple:
    """近存计算：一个设计 → 一个 HardwareConfig（存储参数来自介质，
    算力参数来自计算资源；有效内部带宽写入链路表对角线）。"""
    struct = build_structure_model(spec)   # 阶段②
    read_bw_total = write_bw_total = 0.0
    read_lat = write_lat = 0
    capacity = struct.total_capacity_bytes
    scale_union = {}
    eff_tables = []
    peak_f = 0.0
    parallelism = 0
    device_type = "DRAM_PIM"

    for s_idx, s in enumerate(spec.storages):
        pre = _media_preset(s.media)
        # struct.arrays 与 storages 按下标一一对应（可能含相同介质的多条存储）
        arr = struct.arrays[s_idx]
        read_bw_total += arr.count * float(pre["read_bw_gbs_per_array"])
        write_bw_total += arr.count * float(pre["write_bw_gbs_per_array"])
        read_lat = max(read_lat, int(pre["read_lat_ns"]))
        write_lat = max(write_lat, int(pre["write_lat_ns"]))

    for c in spec.computes:
        cpre = _compute_preset(c.resource)
        n = max(1, int(c.count or 1))
        peak_f += n * float(cpre["flops_per_unit"])
        parallelism += n * int(cpre["parallelism_per_unit"])
        # 计算资源侧的片上暂存计入容量（相对存储容量通常可忽略）
        capacity += n * int(cpre["buffer_bytes"])
        scale_union = _union_scale(scale_union, cpre["precision_scale"])
        eff_tables.append(DESIGN_EFFICIENCY_RULES[cpre["efficiency_class"]])
        # 能力语义：只要有一个通用计算资源（可跑非线性）→ 取 GPU 语义（能力超集）
        if cpre["device_type"] == "GPU":
            device_type = "GPU"
        elif device_type != "GPU" and cpre["device_type"] == "RERAM_PIM":
            device_type = "RERAM_PIM"

    # 部署层级修正（v1 近似）：
    #   通道间混合：跨资源域访问需穿越通道边界 → 延迟增加；通道共享互联 → 有效带宽折减
    df = 1.0
    ch = max(1, struct.channels)
    if struct.deployment == DeploymentMode.CHANNEL_CROSS and ch > 1:
        read_lat += int(CROSS_CHANNEL_LAT_NS * (ch - 1))
        write_lat += int(CROSS_CHANNEL_LAT_NS * (ch - 1))

    eff = _merge_efficiency_tables(eff_tables, scale_union, df)
    kind = (spec.name or "nmc").strip().upper().replace(" ", "-")

    # 有效内部带宽 = min(所有连接的有效带宽)；通道间混合按通道数折减（互联竞争）
    eff_bws = [c.effective_bw_gbs for c in struct.connections]
    diag = min(eff_bws) if eff_bws else read_bw_total
    if struct.deployment == DeploymentMode.CHANNEL_CROSS:
        diag = diag / ch

    hc = HardwareConfig(
        id=spec.name, type=device_type, peak_f=peak_f,
        peak_by_precision=_peak_by_precision(peak_f, scale_union),
        mem_bytes=capacity,
        read_bw=read_bw_total * 1e9, write_bw=write_bw_total * 1e9,
        read_lat_ns=read_lat, write_lat_ns=write_lat,
        parallelism=parallelism, efficiency=eff,
        precision=_supported_precision(scale_union),
        link_type=kind, links={kind: round(diag, 3)},
    )
    return hc, struct, kind


def _union_scale(a: dict, b: dict) -> dict:
    """两个精度缩放表取并集（按精度逐项取较大值：设备可把算子派给能力最强的单元）。"""
    out = dict(a)
    for k, v in b.items():
        out[k] = max(float(out.get(k, 0) or 0), float(v or 0))
    return out


def design_to_hardware_config(spec: UserDesignSpec) -> tuple:
    """阶段③ 入口：用户设计规格 → (HardwareConfig, HardwareStructureModel, kind)。

    返回的 HardwareConfig 与自定义硬件解析产物同格式，可直接交给
    HardwareFactory.create_devices 走原有构建流程。
    """
    if spec.architecture == ArchitectureType.IN_MEMORY:
        return derive_cim(spec)
    return derive_nmc(spec)


# =================================================================
# 6. 序列化（API / 前端预览用）
# =================================================================
def structure_to_dict(struct: HardwareStructureModel) -> dict:
    out = {
        "architecture": struct.architecture.name,
        "channels": struct.channels,
        "deployment": struct.deployment.name,
        "total_capacity_bytes": struct.total_capacity_bytes,
        "total_capacity": _fmt_bytes(struct.total_capacity_bytes),
        "total_parallelism": struct.total_parallelism,
        "arrays": [{
            "media": a.media, "array_kind": a.array_kind,
            "array_bytes": a.array_bytes,
            "array_size": _fmt_bytes(a.array_bytes),
            "count": a.count, "per_channel": a.per_channel,
        } for a in struct.arrays],
        "units": [{
            "resource": u.resource, "count": u.count, "per_channel": u.per_channel,
        } for u in struct.units],
        "connections": [{
            "storage_idx": c.storage_idx, "compute_idx": c.compute_idx,
            "interconnect": c.interconnect,
            "storage_bw_gbs": c.storage_bw_gbs,
            "compute_if_bw_gbs": c.compute_if_bw_gbs,
            "link_bw_gbs": c.link_bw_gbs,
            "effective_bw_gbs": c.effective_bw_gbs,
            "effective_bw": _fmt_gbs(c.effective_bw_gbs),
        } for c in struct.connections],
    }
    if struct.media:
        out["media"] = struct.media
    return out


def config_to_dict(hc: HardwareConfig, struct: HardwareStructureModel,
                   kind: str, architecture: ArchitectureType) -> dict:
    """把推导出的硬件对象转成前端友好的参数字典（与自定义硬件块字段一致）。"""
    peak_tf = hc.peak_f / 1e12
    prec_str = "/".join(p.name for p in hc.precision)
    return {
        "id": hc.id,
        "name": hc.id,
        "type": hc.type,
        "kind": kind,
        "architecture": architecture.name,
        "compute": f"{peak_tf:.3f} TFLOPS",
        "compute_tflops": round(peak_tf, 3),
        "mem": _fmt_bytes(hc.mem_bytes),
        "mem_bytes": hc.mem_bytes,
        "rBW": _fmt_gbs(hc.read_bw / 1e9),
        "rBW_gbs": round(hc.read_bw / 1e9, 3),
        "wBW": _fmt_gbs(hc.write_bw / 1e9),
        "wBW_gbs": round(hc.write_bw / 1e9, 3),
        "read_lat_ns": hc.read_lat_ns,
        "write_lat_ns": hc.write_lat_ns,
        "precision": prec_str,
        "precision_list": [p.name for p in hc.precision],
        "efficiency": dict(hc.efficiency),
        "parallelism": hc.parallelism,
        "links": dict(hc.links),
        "effective_internal_bw_gbs": hc.links.get(kind, 0.0),
        "structure": structure_to_dict(struct),
    }


# =================================================================
# 7. 顶层入口（API 使用）
# =================================================================
def derive_design(spec_dict: dict) -> dict:
    """把前端提交的 JSON 设计规格 → 推导结果 dict。

    返回 {"ok": True, "device": {...}, "hardware_config": {...}, "structure": {...},
          "links": {kind: {kind: gbs}}}
    或抛 ConfigError（由调用方转成 {ok: False, error}）。
    """
    spec = _parse_spec_dict(spec_dict)
    hc, struct, kind = design_to_hardware_config(spec)
    return {
        "ok": True,
        "device": config_to_dict(hc, struct, kind, spec.architecture),
        "hardware_config": {
            "id": hc.id, "type": hc.type,
            "peak_tflops": round(hc.peak_f / 1e12, 4),
            "mem_bytes": hc.mem_bytes, "read_bw": hc.read_bw,
            "write_bw": hc.write_bw,
            "read_lat_ns": hc.read_lat_ns, "write_lat_ns": hc.write_lat_ns,
            "parallelism": hc.parallelism, "efficiency": dict(hc.efficiency),
            "precision": [p.name for p in hc.precision],
            "peak_by_precision": {p.name: v for p, v in hc.peak_by_precision.items()},
            "link_type": kind, "links": dict(hc.links),
        },
        "structure": structure_to_dict(struct),
        "links": {kind: dict(hc.links)},
        "architecture": spec.architecture.name,
    }


def _parse_spec_dict(d: dict) -> UserDesignSpec:
    """把前端 JSON 设计规格解析为 UserDesignSpec（宽松解析 + 明确报错）。"""
    arch = str(d.get("architecture") or "IN_MEMORY").strip().upper()
    if arch not in ("IN_MEMORY", "NEAR_MEMORY"):
        raise ConfigError(f"未知总体架构: '{arch}'. 可选: IN_MEMORY / NEAR_MEMORY")

    spec = UserDesignSpec(
        name=str(d.get("name") or "design").strip() or "design",
        architecture=ArchitectureType.IN_MEMORY if arch == "IN_MEMORY"
        else ArchitectureType.NEAR_MEMORY,
    )
    dep = d.get("deployment") or {}
    mode = str(dep.get("mode") or "CHANNEL_INTERNAL").strip().upper()
    try:
        spec.deployment = DeploymentSpec(
            mode=DeploymentMode[mode],
            channels=int(dep.get("channels", 1) or 1),
        )
    except KeyError:
        raise ConfigError(f"未知部署模式: '{mode}'. 可选: CHANNEL_INTERNAL / CHANNEL_CROSS")
    spec.deployment.channels = max(1, spec.deployment.channels)

    if spec.architecture == ArchitectureType.IN_MEMORY:
        spec.media = str(d.get("media") or "").strip().upper()
        _media_preset(spec.media)   # 校验存在
        spec.capacity_bytes = int(d.get("capacity_bytes") or 0)
        den = str(d.get("density") or "MEDIUM").strip().upper()
        try:
            spec.density = DensityLevel[den]
        except KeyError:
            raise ConfigError(f"未知计算密度: '{den}'. 可选: LOW/MEDIUM/HIGH/CUSTOM")
        if spec.density == DensityLevel.CUSTOM:
            spec.custom_density_factor = float(d.get("custom_density_factor") or 1.0)
        spec.array_bytes = int(d.get("array_bytes") or 0)
    else:
        for s in (d.get("storages") or []):
            spec.storages.append(StorageSpec(
                media=str(s.get("media") or "").strip().upper(),
                capacity_bytes=int(s.get("capacity_bytes") or 0),
                array_bytes=int(s.get("array_bytes") or 0)))
        for c in (d.get("computes") or []):
            spec.computes.append(ComputeSpec(
                resource=str(c.get("resource") or "").strip().upper(),
                count=int(c.get("count") or 1)))
        for c in (d.get("connections") or []):
            spec.connections.append(ConnectionSpec(
                storage_idx=int(c.get("storage_idx") or 0),
                compute_idx=int(c.get("compute_idx") or 0),
                interconnect=str(c.get("interconnect") or "").strip().upper()))
    return spec


def presets() -> dict:
    """预设库清单（供 /api/design/presets 与前端向导渲染）。"""
    def _media_out(key, pre):
        return {
            "key": key, "label": pre["label"], "mechanism": pre["mechanism"],
            "array_kind": pre["array_kind"],
            "array_bytes": pre["array_bytes"],
            "array_size": _fmt_bytes(pre["array_bytes"]),
            "capacity_bytes_min": pre["capacity_bytes_min"],
            "capacity_min": _fmt_bytes(pre["capacity_bytes_min"]),
            "capacity_bytes_recommend": pre["capacity_bytes_recommend"],
            "capacity_recommend": _fmt_bytes(pre["capacity_bytes_recommend"]),
            "capacity_bytes_max": pre["capacity_bytes_max"],
            "capacity_max": _fmt_bytes(pre["capacity_bytes_max"]),
            "capacity_granularity_bytes": pre["capacity_granularity_bytes"],
            "capacity_granularity": _fmt_bytes(pre["capacity_granularity_bytes"]),
            "device_type": pre["device_type"],
            "read_lat_ns": pre["read_lat_ns"], "write_lat_ns": pre["write_lat_ns"],
        }

    def _compute_out(key, pre):
        return {
            "key": key, "label": pre["label"],
            "flops_per_unit_tflops": round(float(pre["flops_per_unit"]) / 1e12, 3),
            "interface_bw_gbs": pre["interface_bw_gbs"],
            "buffer_bytes": pre["buffer_bytes"],
            "buffer": _fmt_bytes(pre["buffer_bytes"]),
            "device_type": pre["device_type"],
            "precision": [p for p, v in pre["precision_scale"].items() if v > 0],
        }

    def _interconnect_out(key, pre):
        return {"key": key, "label": pre["label"],
                "bw_gbs": pre["bw_gbs"], "lat_ns": pre["lat_ns"]}

    return {
        "architectures": [
            {"key": "IN_MEMORY", "label": "存内计算（CIM）",
             "desc": "计算直接发生在存储阵列内部，存储与计算天然紧密结合"},
            {"key": "NEAR_MEMORY", "label": "近存计算（NMC）",
             "desc": "存储单元与计算单元相互独立、物理邻近，通过内部互联传输数据"},
        ],
        "media": {k: _media_out(k, v) for k, v in MEDIA_PRESETS.items()},
        "computes": {k: _compute_out(k, v) for k, v in COMPUTE_PRESETS.items()},
        "interconnects": {k: _interconnect_out(k, v)
                          for k, v in INTERCONNECT_PRESETS.items()},
        "density_levels": {k: dict(v) for k, v in DENSITY_LEVELS.items()},
        "deployment_modes": [
            {"key": "CHANNEL_INTERNAL", "label": "通道内混合",
             "desc": "每个通道内部同时存在多种资源"},
            {"key": "CHANNEL_CROSS", "label": "通道间混合",
             "desc": "不同类型资源分别部署在不同通道（跨通道访问有延迟/带宽代价）"},
        ],
        "eff_op_keys": list(EFF_OP_KEYS),
        "max_resources_hint": "第一版界面建议 ≤2 种存储与 ≤2 种计算资源（底层不限）",
    }
