"""
LLM-PIMSim v3 core.hardware_sys — 【硬件系统】

职责（"硬件"这一概念的完整闭环逻辑集中于此）：
  1. 硬件数据结构：HardwareUnit（能力+状态，含链路种类 type_name）
  2. 硬件 YAML 解析：HardwareConfig + 出厂预设表 + 单位换算（链路带宽委托【链路系统】）
  3. 硬件构建工厂：HardwareFactory（config → HardwareUnit）+ build_hardware 装配
     （返回 (devices, LinkBandwidthTable)，链路带宽表由 core.link_sys 提供）

依赖：core.common（DeviceType 枚举 / ConfigError / load_yaml）+ core.precision
     （PrecisionLevel 枚举 / HARDWARE_CAPABILITY 硬件精度能力表 / precision_to_bytes）
     + core.link_sys（链路系统：N×N 对称带宽查找表）。

不依赖：算子系统 / 调度器 / 校验 / 输出 / 权重 / 切割（若有需要只传数据模型）。

顶层兼容：`hardware.py`、`hardware_factory.py` 转发本系统；`config_loader.py` 的
硬件解析部分也委托本系统。
"""

from dataclasses import dataclass, field

from core.common import DeviceType, OperatorCategory, ConfigError, load_yaml
from core.precision import PrecisionLevel, HARDWARE_CAPABILITY
from core.link_sys import LinkBandwidthTable


# =================================================================
# 1. 出厂预设（默认设备参数；链路带宽表见【链路系统】core.link_sys）
# =================================================================

# 设备出厂预设（YAML 里写了 value 就覆盖它）
# 单位约定: peak_tflops 存 TFLOPS(解析乘 1e12→FLOPS)、带宽存 GB/s(解析乘 1e9→Bps)、
#           容量 mem_gb/mem_mb 存 GB/MB(解析乘 1e9/1e6→Byte)、延迟统一 ns。
# 数据来源（硬件校准参考，v3.1 细化）见 修改文档/硬件系统.txt 的"数据来源"节：
#   CPU       : Intel Xeon Platinum 8480+ (Sapphire Rapids)，DDR5-4800，BF16 AMX 峰值
#   GPU       : NVIDIA A100 80GB SXM，Tensor Core dense（按精度见峰值表）
#   SRAM-PIM  : SRAM digital-CIM（ISSCC 类，INT8/FP16 为主）
#   DRAM-PIM  : Samsung HBM2-PIM Aquabolt-XL（单 stack，FP16 主精度）
#   ReRAM-PIM : RRAM/NeuRRAM-CIM，模拟实现，INT8/低精度为主（读写不对称）
#
# peak_by_precision_tflops : {精度名: TFLOPS/TOPS 数值}，出厂"按精度峰值算力"表，
#   供 parse_peak_throughput 的默认分支使用；YAML 里显式写 peak_throughput 可整体覆盖。
# efficiency_default       : {op_type: 利用率}，覆盖 GEMM/LayerNorm/Softmax/Activation/
#   Residual/LMHead/Embedding/KVCacheUpdate 共 8 类（对应全部 18 个具体算子）。
DEFAULT_DEVICE_PARAMS = {
    "CPU": {
        "peak_tflops_default": 45.9,   # BF16 AMX 峰值
        # v3.1：CPU 为纯执行单元，容量≈0，仅保留极小运行缓存（4MB）
        "mem_mb_default": 4,
        "read_bw_gbs_default": 307.2, "write_bw_gbs_default": 307.2,
        "read_lat_ns_default": 90, "write_lat_ns_default": 90,
        "parallelism_default": 1,
        "peak_by_precision_tflops": {
            # Sapphire Rapids AMX：BF16 45.9；FP16 相近；INT8 更高；FP32 靠 AVX-512 较低
            "FP32": 3.2, "BF16": 45.9, "FP16": 45.9,
            "FP8": 0.0, "INT8": 91.8, "INT4": 0.0,
        },
        "efficiency_default": {
            # CPU 通用核，无专用矩阵引擎，各类算子利用率都偏低
            "GEMM": 0.40, "LayerNorm": 0.25, "Softmax": 0.25,
            "Activation": 0.30, "Residual": 0.30, "LMHead": 0.35,
            "Embedding": 0.30, "KVCacheUpdate": 0.20,
        },
    },
    "GPU": {
        "peak_tflops_default": 312.0,
        "mem_gb_default": 80,
        "read_bw_gbs_default": 2039.0, "write_bw_gbs_default": 2039.0,
        "read_lat_ns_default": 400, "write_lat_ns_default": 400,
        "parallelism_default": 100,
        "peak_by_precision_tflops": {
            # A100 80GB SXM4 Tensor Core dense：TF32(≈FP32 计算路径)156，
            # FP16/BF16 312，INT8 624，INT4 1248；无原生 FP8（以 FP8≈INT8 近折算）
            "FP32": 156.0, "BF16": 312.0, "FP16": 312.0,
            "FP8": 624.0, "INT8": 624.0, "INT4": 1248.0,
        },
        "efficiency_default": {
            # Ampere 实测/roofline：Tensor-Core 类 GEMM 利用最高；elementwise 带宽受限利用低
            "GEMM": 0.85, "LayerNorm": 0.25, "Softmax": 0.30,
            "Activation": 0.30, "Residual": 0.35, "LMHead": 0.80,
            "Embedding": 0.50, "KVCacheUpdate": 0.40,
        },
    },
    "SRAM_PIM": {
        "peak_tflops_default": 500.0,
        "mem_mb_default": 512,
        "read_bw_gbs_default": 1500000.0, "write_bw_gbs_default": 1500000.0,
        "read_lat_ns_default": 2, "write_lat_ns_default": 2,
        "parallelism_default": 32,
        "peak_by_precision_tflops": {
            # SRAM digital-CIM：FP16 500 基准；INT8 2×，INT4 4×；FP32 因模拟受限于 1/4
            "FP32": 125.0, "BF16": 500.0, "FP16": 500.0,
            "FP8": 1000.0, "INT8": 1000.0, "INT4": 2000.0,
        },
        "efficiency_default": {
            "GEMM": 0.80, "LayerNorm": 0.25, "Softmax": 0.30,
            "Activation": 0.30, "Residual": 0.35, "LMHead": 0.75,
            "Embedding": 0.50, "KVCacheUpdate": 0.40,
        },
    },
    "DRAM_PIM": {
        "peak_tflops_default": 1.2,
        "mem_gb_default": 8,
        "read_bw_gbs_default": 307.2, "write_bw_gbs_default": 307.2,
        "read_lat_ns_default": 50, "write_lat_ns_default": 50,
        "parallelism_default": 16,
        "peak_by_precision_tflops": {
            # HBM2-PIM Aquabolt-XL 单 stack，FP16 主精度 1.2；FP32 折半；低精度 2×/4×
            "FP32": 0.6, "BF16": 1.2, "FP16": 1.2,
            "FP8": 2.4, "INT8": 2.4, "INT4": 4.8,
        },
        "efficiency_default": {
            "GEMM": 0.70, "LayerNorm": 0.15, "Softmax": 0.20,
            "Activation": 0.15, "Residual": 0.20, "LMHead": 0.60,
            "Embedding": 0.30, "KVCacheUpdate": 0.25,
        },
    },
    "RERAM_PIM": {
        "peak_tflops_default": 20.0,    # INT8
        "mem_mb_default": 64,
        "read_bw_gbs_default": 128.0, "write_bw_gbs_default": 32.0,
        "read_lat_ns_default": 10, "write_lat_ns_default": 100,
        "parallelism_default": 64,
        "peak_by_precision_tflops": {
            # RRAM 模拟 CIM：主精度 FP16/INT8 均为 20 档；INT4 40、FP8 20、BF16 10；
            # FP32 模拟难以实现且能力表 execution 不含 FP32 → 0（不可执行）。
            # （以 FP16 为主精度基准，便于 YAML 单值 peak_tflops 语义一致）
            "FP32": 0.0, "BF16": 10.0, "FP16": 20.0,
            "FP8": 20.0, "INT8": 20.0, "INT4": 40.0,
        },
        "efficiency_default": {
            "GEMM": 0.60, "LayerNorm": 0.10, "Softmax": 0.10,
            "Activation": 0.10, "Residual": 0.15, "LMHead": 0.50,
            "Embedding": 0.25, "KVCacheUpdate": 0.20,
        },
    },
    # v3.1 新增：纯存储单元（只存不整，算力 0，不执行任何算子）
    "SRAM": {
        "peak_tflops_default": 0.0,
        "mem_mb_default": 256,             # 256 MB 片上 SRAM
        "read_bw_gbs_default": 1000.0, "write_bw_gbs_default": 1000.0,
        "read_lat_ns_default": 2, "write_lat_ns_default": 2,
        "parallelism_default": 1,
        "peak_by_precision_tflops": {"FP32": 0.0, "BF16": 0.0, "FP16": 0.0,
                                     "FP8": 0.0, "INT8": 0.0, "INT4": 0.0},
        "efficiency_default": {},
    },
    "DRAM": {
        "peak_tflops_default": 0.0,
        "mem_gb_default": 64,              # 64 GB 主存
        "read_bw_gbs_default": 200.0, "write_bw_gbs_default": 200.0,
        "read_lat_ns_default": 60, "write_lat_ns_default": 60,
        "parallelism_default": 1,
        "peak_by_precision_tflops": {"FP32": 0.0, "BF16": 0.0, "FP16": 0.0,
                                     "FP8": 0.0, "INT8": 0.0, "INT4": 0.0},
        "efficiency_default": {},
    },
}


# =================================================================
# 2. 单位换算
# =================================================================
def throughput_to_flops(value, unit: str) -> float:
    """把"数值+单位"(TFLOPS/TOPS/GMAC/s/TMAC/s/FLOPS) 统一换算成 FLOPS。
    TFLOPS 与 TOPS 都按 1e12；GMAC/s、TMAC/s 按 1 MAC=2 FLOP。"""
    v = float(value)
    u = (unit or "").strip().upper()
    if not u:
        return v
    mults = {
        "TFLOPS": 1e12, "TFLOP/S": 1e12,
        "TOPS": 1e12, "TOP/S": 1e12,
        "GFLOPS": 1e9, "GFLOP/S": 1e9,
        "GMAC/S": 2e9, "TMAC/S": 2e12,
        "FLOPS": 1.0, "FLOP/S": 1.0,
    }
    return v * mults.get(u, 1e12)


def _default_precision_list():
    from core.precision import PrecisionLevel as _PL
    return [_PL.FP32, _PL.BF16, _PL.FP16, _PL.FP8, _PL.INT8, _PL.INT4]


# 7 种默认设备种类 + 1 个兼容别名（HBM→GPU）
KNOWN_DEVICE_TYPES = {"GPU", "DRAM_PIM", "SRAM_PIM", "RERAM_PIM", "CPU", "HBM",
                      "SRAM", "DRAM"}


def _norm_device_type(t: str) -> str:
    """规范化设备 type 字符串。v3.2：允许用户自定义种类（不再强制属于 7 种默认）。
    自定义种类用 GPU 出厂预设兜底参数，能力（capability）按 GPU 处理；
    其在链路表中的"种类"以 link_type（缺省=type）为准。"""
    t = str(t).strip().upper()
    if not t:
        raise ConfigError("硬件设备缺少 type 字段。")
    return t


# =================================================================
# 3. 解析出的中间结构（dataclass）
# =================================================================
@dataclass
class HardwareConfig:
    """单个硬件的解析后配置（标准单位：FLOPS / Byte / Bps / ns）"""
    id: str
    type: str                          # 基础种类 GPU / DRAM_PIM / ... 或用户自定义种类
    peak_f: float
    mem_bytes: int
    read_bw: float
    write_bw: float
    read_lat_ns: int
    write_lat_ns: int
    parallelism: int
    efficiency: dict
    precision: list = field(default_factory=list)
    peak_by_precision: dict = field(default_factory=dict)   # {PrecisionLevel: FLOPS}
    link_type: str = ""                # 链路表里的"设备种类"（缺省=type；自定义硬件常=其唯一名）
    links: dict = field(default_factory=dict)   # {其它种类: 链路带宽 GB/s}（自定义硬件"加一栏"）


# =================================================================
# 4. 硬件 YAML 解析
# =================================================================
def parse_peak_throughput(comp: dict, pre: dict):
    """解析 compute 下的峰值算力，返回 (兼容 peak_f, peak_by_precision)。"""
    _PL = PrecisionLevel
    throughput = comp.get("peak_throughput", {})
    if throughput and isinstance(throughput, dict):
        peak_by_precision = {}
        for pname, entry in throughput.items():
            try:
                prec = _PL.from_name(str(pname))
            except ValueError:
                raise ConfigError(f"peak_throughput 含未知精度: '{pname}'。可选: FP32/FP16/BF16/INT8/INT4")
            if isinstance(entry, dict):
                peak_by_precision[prec] = throughput_to_flops(
                    entry.get("value", 0), entry.get("unit", ""))
            else:
                peak_by_precision[prec] = throughput_to_flops(entry, "TFLOPS")
        peak_f = peak_by_precision.get(_PL.FP16) or next(iter(peak_by_precision.values()), 0.0)
        return peak_f, peak_by_precision

    pre_peak = pre.get("peak_tflops_default", 300.0)
    if "peak_tflops" in comp:
        peak_f = float(comp["peak_tflops"]) * 1e12
    elif "peak_tops" in comp:
        peak_f = float(comp["peak_tops"]) * 1e12
    elif "peak_tops_default" in pre:
        peak_f = float(pre.get("peak_tops_default")) * 1e12
    else:
        peak_f = float(pre_peak) * 1e12

    # v3.1：有出厂"按精度算力表"时，按相对比例把 YAML 单值(peak_f)折算到各精度
    #       （peak_f 视为"主精度参考算力"，其它精度 = peak_f × 出厂表[精度]/出厂表[主精度]）
    pre_map = pre.get("peak_by_precision_tflops") or {}
    if pre_map:
        def _val(pname):
            return float(pre_map.get(pname, 0.0)) if isinstance(pre_map.get(pname), (int, float)) else 0.0
        main_tf = _val("FP16")
        if main_tf <= 0:   # 主精度非 FP16 时，取表里首个非 0 精度作为基准
            nonzero = {k: float(v) for k, v in pre_map.items()
                       if isinstance(v, (int, float)) and float(v) > 0}
            main_tf = max(nonzero.values()) if nonzero else 1.0
            if main_tf == 0:
                main_tf = 1.0
        peak_by_precision = {}
        for p in _default_precision_list():
            v = _val(p.name)
            peak_by_precision[p] = peak_f * (v / main_tf) if v > 0 else 0.0
        return peak_f, peak_by_precision

    # 旧逻辑：无按精度表 → 单值铺满所有精度（向后兼容）
    peak_by_precision = {p: peak_f for p in _default_precision_list()}
    return peak_f, peak_by_precision


def parse_hardware(doc: dict) -> dict:
    """hardware.yaml -> {id: HardwareConfig}"""
    out = {}
    devices = doc.get("devices", [])
    if not devices:
        raise ConfigError("hardware.yaml 的 'devices' 不能为空")
    for d in devices:
        dev_id = d.get("id")
        dtype = _norm_device_type(d.get("type", ""))
        if not dev_id:
            raise ConfigError(f"hardware 设备缺少 id: {d}")
        if dev_id in out:
            raise ConfigError(f"重复的设备 id: {dev_id}")
        # 自定义种类（不在 7 种默认里）用 GPU 出厂预设兜底参数
        pre = DEFAULT_DEVICE_PARAMS.get(dtype) or DEFAULT_DEVICE_PARAMS.get("GPU", {})
        comp = d.get("compute", {})
        mem = d.get("memory", {})
        # 效率表：YAML 显式写的覆盖同 key；未写的用出厂默认补齐（保证 8 类算子都有效率，而非缺失项回退 1.0）
        eff_over = d.get("efficiency") or {}
        eff_default = pre.get("efficiency_default", {})
        eff = {**eff_default, **eff_over}
        peak_f, peak_by_precision = parse_peak_throughput(comp, pre)

        mem_bytes = pre.get("mem_gb_default", 80) * 1e9
        if "capacity_gb" in mem:
            mem_bytes = float(mem["capacity_gb"]) * 1e9
        elif "capacity_mb" in mem:
            mem_bytes = float(mem["capacity_mb"]) * 1e6
        elif pre.get("mem_mb_default") is not None:
            mem_bytes = float(pre.get("mem_mb_default")) * 1e6   # v3.1：MB 级默认容量（CPU/SRAM）

        def _bw(gbs_key, gbps_key, default):
            if gbs_key in mem:
                return float(mem[gbs_key]) * 1e9
            if gbps_key in mem:
                return float(mem[gbps_key]) * 1e9
            return default

        read_bw = _bw("read_bandwidth_gbs", "read_bw_gbs",
                      pre.get("read_bw_gbs_default", 2000) * 1e9)
        write_bw = _bw("write_bandwidth_gbs", "write_bw_gbs",
                       pre.get("write_bw_gbs_default", 1800) * 1e9)
        read_lat = int(mem.get("read_latency_ns", pre.get("read_lat_ns_default", 100)))
        write_lat = int(mem.get("write_latency_ns", pre.get("write_lat_ns_default", 100)))
        parallel = int(comp.get("parallelism", pre.get("parallelism_default", 1)))

        precision = []
        if peak_by_precision:
            precision = list(peak_by_precision.keys())
        else:
            precision_raw = d.get("supported_precision", [])
            if precision_raw:
                for pname in precision_raw:
                    try:
                        precision.append(PrecisionLevel.from_name(str(pname)))
                    except ValueError:
                        raise ConfigError(f"设备 {dev_id} 的 supported_precision 含未知精度: '{pname}'."
                                          f" 可选 FP32/FP16/INT8/INT4")
            else:
                precision = _default_precision_list()

        # v3.2 链路系统：设备的"链路种类"（link_type，缺省=type）+ 自定义硬件"加一栏"的
        # 到其它种类间的链路带宽（links）。数值统一存 GB/s，解析时不乘 1e9，交给链路系统。
        link_type = (str(d.get("link_type") or "").strip().upper()) or dtype
        links = {}
        for k, v in (d.get("links") or {}).items():
            try:
                links[str(k).strip().upper()] = float(v)
            except (TypeError, ValueError):
                raise ConfigError(
                    f"设备 {dev_id} 的 links[{k!r}] 必须是数值带宽(GB/s)，收到: {v!r}")

        out[dev_id] = HardwareConfig(
            id=dev_id, type=dtype, peak_f=peak_f,
            peak_by_precision=peak_by_precision, mem_bytes=int(mem_bytes),
            read_bw=read_bw, write_bw=write_bw, read_lat_ns=read_lat,
            write_lat_ns=write_lat, parallelism=parallel, efficiency=dict(eff),
            precision=precision, link_type=link_type, links=links,
        )
    return out


def parse_interconnect(doc: dict, hardware: dict) -> "LinkBandwidthTable":
    """interconnect.yaml -> LinkBandwidthTable（N×N 对称带宽表，GB/s）。

    构造顺序（后者覆盖前者）：
      1) 出厂默认表（core.link_sys.DEFAULT_LINK_BW_TABLE，7 种默认种类）；
      2) hardware.yaml 每设备的 links 段（自定义硬件"加一栏"：到其它种类间的带宽）；
      3) interconnect.yaml 的 link_bw_gbs 全局表（按"种类"覆盖/新增）。

    同时兼容旧式 interconnect.yaml 的 `links:` 列表（src/dst 为设备 id + read/write 带宽），
    旧式"read/write 不对称"在对称化时取两者较大者作为该种类对的带宽。
    """
    table = LinkBandwidthTable.default()

    # 1) 合并 hardware.yaml 里每设备的 links（自定义硬件"加一栏"）
    for hc in hardware.values():
        kind = getattr(hc, "link_type", "") or hc.type
        if getattr(hc, "links", None):
            table.add_type(kind, hc.links)

    # 2) 新格式：link_bw_gbs（种类 → 种类 → GB/s），支持 {kind: {kind: gbs}} 或 {kind: gbs}
    link_bw = doc.get("link_bw_gbs") or {}
    if isinstance(link_bw, dict):
        for a, row in link_bw.items():
            if isinstance(row, dict):
                for b, v in row.items():
                    table.set_bw(a, b, v)
            else:
                table.set_bw(a, a, row)

    # 3) 旧格式兼容：links 列表（设备 id → 种类），read/write 对称化取较大者
    id_kind = {hc.id: (getattr(hc, "link_type", "") or hc.type) for hc in hardware.values()}
    for lk in (doc.get("links") or []):
        src = lk.get("src")
        dst = lk.get("dst")
        if not src or not dst:
            raise ConfigError(f"连接缺少 src/dst: {lk}")
        ka = id_kind.get(src, str(src).upper())
        kb = id_kind.get(dst, str(dst).upper())
        read_gbs = float(lk.get("read_bandwidth_gbs", lk.get("read_bw_gbs", 0)) or 0)
        write_gbs = float(lk.get("write_bandwidth_gbs", lk.get("write_bw_gbs", 0)) or 0)
        single = float(lk.get("bandwidth_gbs", 0) or 0)
        gbs = max(read_gbs, write_gbs, single)
        if gbs > 0:
            table.set_bw(ka, kb, gbs)
    return table


# =================================================================
# 5. 硬件数据模型（build 后的运行时对象）
# =================================================================
def _cap_key(device_type) -> str:
    return device_type.name


@dataclass
class HardwareUnit:
    """一个硬件零件 —— GPU / DRAM-PIM / SRAM-PIM / ReRAM-PIM / HBM...
    所有硬件共用同一结构，仅参数不同（配置驱动，不需要为每类建子类）。"""
    id: str
    name: str
    device_type: DeviceType
    type_name: str = ""              # 链路系统索引用的"设备种类"（link_type 或 type，大写）

    peak_compute_flops: float = 0.0
    peak_throughput_by_precision: dict = field(default_factory=dict)
    supported_precision: list = field(default_factory=lambda: [  # 默认支持全部精度
        PrecisionLevel.FP32, PrecisionLevel.BF16, PrecisionLevel.FP16,
        PrecisionLevel.FP8, PrecisionLevel.INT8, PrecisionLevel.INT4])
    supported_operator_categories: list = field(default_factory=list)
    supported_data_precisions: list = field(default_factory=list)
    supported_execution_precisions: list = field(default_factory=list)
    parallelism: int = 1
    efficiency_table: dict = field(default_factory=dict)

    memory_capacity_bytes: int = 0
    read_bandwidth_Bps: float = 0.0
    write_bandwidth_Bps: float = 0.0
    read_latency_ns: int = 0
    write_latency_ns: int = 0

    num_banks: int = 1
    compute_units: int = 1

    available_time_ns: int = 0
    current_task: str = ""

    def __post_init__(self):
        cap = HARDWARE_CAPABILITY.get(_cap_key(self.device_type))
        if cap:
            if not self.supported_operator_categories:
                self.supported_operator_categories = list(cap["categories"])
            if not self.supported_data_precisions:
                self.supported_data_precisions = list(cap["data"])
            if not self.supported_execution_precisions:
                self.supported_execution_precisions = list(cap["execution"])

    def can_execute(self, operator) -> bool:
        ok, _ = self.check_execution_compatibility(operator)
        return ok

    def check_execution_compatibility(self, operator):
        reasons = []
        if operator.category not in self.supported_operator_categories:
            reasons.append(
                f"Category incompatible: 算子 {operator.id}({operator.name}) 类别"
                f"{operator.category.name} 不在 {[c.name for c in self.supported_operator_categories]}")
        if operator.data_precision not in self.supported_data_precisions:
            reasons.append(f"Data precision {operator.data_precision.name} is not supported "
                           f"(算子 {operator.id}({operator.name}))")
        if operator.execution_precision is not None:
            if operator.execution_precision not in self.supported_execution_precisions:
                reasons.append(f"Execution precision {operator.execution_precision.name} is not supported "
                               f"(算子 {operator.id}({operator.name}))")
        return (len(reasons) == 0), reasons

    def supports_precision(self, required: PrecisionLevel) -> bool:
        if not self.supported_precision:
            return False
        hw_max_level = max(p.value for p in self.supported_precision)
        return hw_max_level >= required.value

    def supports_compute_precision(self, precision) -> bool:
        prec = precision if isinstance(precision, PrecisionLevel) else PrecisionLevel.from_name(str(precision))
        if self.peak_throughput_by_precision:
            return prec in self.peak_throughput_by_precision
        return prec in self.supported_precision

    def get_peak_compute(self, precision=None) -> float:
        if precision is None:
            return self.peak_compute_flops
        prec = precision if isinstance(precision, PrecisionLevel) else PrecisionLevel.from_name(str(precision))
        if self.peak_throughput_by_precision:
            return self.peak_throughput_by_precision.get(prec, 0.0)
        return self.peak_compute_flops if prec in self.supported_precision else 0.0

    def efficiency_for(self, op_type: str) -> float:
        return self.efficiency_table.get(op_type, 1.0)

    def can_fit(self, size_bytes: int) -> bool:
        return size_bytes <= self.memory_capacity_bytes

    def reset_state(self):
        self.available_time_ns = 0
        self.current_task = ""

    def estimate_energy(self, duration_ns: int) -> float:
        return 0.0   # 预留：功耗模型占位


# =================================================================
# 6. 构建工厂
# =================================================================
_DEVICE_TYPE_MAP = {
    "GPU": DeviceType.GPU,
    "DRAM_PIM": DeviceType.DRAM_PIM,
    "SRAM_PIM": DeviceType.SRAM_PIM,
    "RERAM_PIM": DeviceType.RERAM_PIM,
    "CPU": DeviceType.CPU,
    "HBM": DeviceType.GPU,   # HBM 作为存储视为可用设备，简化为 GPU 运算能力 0
    "SRAM": DeviceType.SRAM,   # v3.1：纯存储
    "DRAM": DeviceType.DRAM,   # v3.1：纯存储
}


class HardwareFactory:
    """从配置创建硬件零件（链路带宽表由【链路系统】core.link_sys 提供）。"""

    @staticmethod
    def create_devices(hardware_cfg: dict) -> dict:
        units = {}
        for hid, hc in hardware_cfg.items():
            supported_precision = list(hc.precision) if hc.precision else [
                PrecisionLevel.FP32, PrecisionLevel.FP16,
                PrecisionLevel.INT8, PrecisionLevel.INT4]
            units[hid] = HardwareUnit(
                id=hid, name=hid,
                device_type=_DEVICE_TYPE_MAP.get(hc.type, DeviceType.GPU),
                type_name=(hc.link_type or hc.type).upper(),
                peak_compute_flops=hc.peak_f,
                peak_throughput_by_precision=dict(hc.peak_by_precision),
                memory_capacity_bytes=hc.mem_bytes,
                read_bandwidth_Bps=hc.read_bw,
                write_bandwidth_Bps=hc.write_bw,
                read_latency_ns=hc.read_lat_ns,
                write_latency_ns=hc.write_lat_ns,
                parallelism=hc.parallelism,
                efficiency_table=dict(hc.efficiency),
                supported_precision=supported_precision,
            )
        return units


def build_hardware(cfg):
    """完整装配：硬件 + 链路带宽表。cfg 需含 .hardware(.dict)、.links(LinkBandwidthTable)。

    返回 (devices: dict, link_table: LinkBandwidthTable)。"""
    devices = HardwareFactory.create_devices(cfg.hardware)
    link_table = cfg.links
    if not isinstance(link_table, LinkBandwidthTable):
        # 兜底：cfg.links 不是链路表时（旧调用/手搭 cfg），用默认表。
        link_table = LinkBandwidthTable.default()
    return devices, link_table


# =================================================================
# 7. 前端自定义硬件 → HardwareConfig（前端加自定义硬件 = 自动新增后端设备）
# =================================================================
def _frontend_number(s, default=0.0) -> float:
    """取前端字符串（如 '100 TFLOPS' / '64 GB' / '1000 GB/s'）开头的数值部分。"""
    if s is None:
        return default
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip()
    num = ""
    for ch in t:
        if ch.isdigit() or ch in ".+-eE":
            num += ch
        else:
            break
    try:
        return float(num)
    except ValueError:
        return default


def _frontend_mem_bytes(s) -> float:
    """'64 GB' / '512 MB' → Byte。缺省按 GB。"""
    t = str(s or "").strip().upper().replace(" ", "")
    n = _frontend_number(s, 0.0)
    if "TB" in t:
        return n * 1e12
    if "GB" in t:
        return n * 1e9
    if "MB" in t:
        return n * 1e6
    if "KB" in t:
        return n * 1e3
    return n * 1e9


def _frontend_bw_bps(s) -> float:
    """'1000 GB/s' → B/s。缺省按 GB/s。"""
    t = str(s or "").strip().upper().replace(" ", "")
    n = _frontend_number(s, 0.0)
    if "TB" in t:
        return n * 1e12
    if "GB" in t:
        return n * 1e9
    if "MB" in t:
        return n * 1e6
    return n * 1e9


def build_frontend_custom_hardware(hw_list) -> dict:
    """把前端 serializeState 的 hardware 列表里的"自定义设备"（backId == id）转成
    {id: HardwareConfig}，供注入实验硬件集——前端加自定义硬件即在仿真里新增一个后端设备。

    预设设备（backId != id，映射到实验已有后端设备，如 gpu0）会被跳过。
    参数缺失时用其 base type 的出厂预设兜底。
    """
    out = {}
    for h in hw_list or []:
        hid = str(h.get("id") or "").strip()
        if not hid:
            continue
        back = str(h.get("backId") or hid).strip()
        if back != hid:
            continue   # 预设设备：不新增后端设备
        dtype = str(h.get("type") or "GPU").strip().upper()
        pre = DEFAULT_DEVICE_PARAMS.get(dtype) or DEFAULT_DEVICE_PARAMS.get("GPU", {})

        comp = {"peak_tflops": _frontend_number(
            h.get("compute"), pre.get("peak_tflops_default", 1.0))}
        peak_f, peak_by_precision = parse_peak_throughput(comp, pre)

        mem_bytes = int(_frontend_mem_bytes(h.get("mem"))
                        or pre.get("mem_gb_default", 80) * 1e9)
        read_bw = _frontend_bw_bps(h.get("rBW")) or pre.get("read_bw_gbs_default", 2000) * 1e9
        write_bw = _frontend_bw_bps(h.get("wBW")) or pre.get("write_bw_gbs_default", 1800) * 1e9
        read_lat = int(pre.get("read_lat_ns_default", 100))
        write_lat = int(pre.get("write_lat_ns_default", 100))
        parallel = int(pre.get("parallelism_default", 1))
        eff = dict(pre.get("efficiency_default", {}))

        precision = []
        for pname in str(h.get("precision") or "FP32,FP16,INT8,INT4").replace("/", ",").split(","):
            pname = pname.strip().upper()
            if not pname:
                continue
            try:
                precision.append(PrecisionLevel.from_name(pname))
            except ValueError:
                pass
        if not precision:
            precision = list(peak_by_precision.keys()) or _default_precision_list()

        link_type = str(h.get("linkType") or hid).strip().upper()
        links = {}
        for k, v in (h.get("links") or {}).items():
            try:
                links[str(k).strip().upper()] = float(v)
            except (TypeError, ValueError):
                pass

        out[hid] = HardwareConfig(
            id=hid, type=dtype, peak_f=peak_f, mem_bytes=mem_bytes,
            read_bw=read_bw, write_bw=write_bw, read_lat_ns=read_lat,
            write_lat_ns=write_lat, parallelism=parallel, efficiency=eff,
            precision=precision, peak_by_precision=peak_by_precision,
            link_type=link_type, links=links,
        )
    return out
