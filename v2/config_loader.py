"""
LLM-PIMSim v2 — 配置加载器
职责:
  1. 读 YAML 文件（hardware/interconnect/mapping/placement/experiment）
  2. 校验必填字段，报错信息清晰
  3. 单位换算：把"人话单位"(GB/GB/s/TFLOPS/TOPS) 统一成内部标准 (Byte/Bps/FLOPS)
  4. 填充默认值（默认连接表、默认设备参数）
"""
import os
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 出厂预设表（默认连接的参考值）
# ============================================================
# 默认连接表：key = (DeviceType名, DeviceType名) -> {read_gbs, write_gbs, lat_ns}
DEFAULT_CONNECT_TABLE = {
    ("GPU", "GPU"):             {"read_gbs": 600,  "write_gbs": 600,  "lat_ns": 100},
    ("GPU", "DRAM_PIM"):        {"read_gbs": 120,  "write_gbs": 80,   "lat_ns": 500},
    ("DRAM_PIM", "GPU"):        {"read_gbs": 120,  "write_gbs": 80,   "lat_ns": 500},
    ("GPU", "SRAM_PIM"):        {"read_gbs": 200,  "write_gbs": 200,  "lat_ns": 100},
    ("SRAM_PIM", "GPU"):        {"read_gbs": 200,  "write_gbs": 200,  "lat_ns": 100},
    ("DRAM_PIM", "DRAM_PIM"):   {"read_gbs": 600,  "write_gbs": 300,  "lat_ns": 150},
    ("DRAM_PIM", "SRAM_PIM"):   {"read_gbs": 150,  "write_gbs": 100,  "lat_ns": 200},
    ("SRAM_PIM", "DRAM_PIM"):   {"read_gbs": 150,  "write_gbs": 100,  "lat_ns": 200},
}

# 设备出厂预设（默认参数，YAML 里写了 value 就覆盖它）
DEFAULT_DEVICE_PARAMS = {
    "GPU": {
        "peak_tflops_default": 300.0, "mem_gb_default": 80,
        "read_bw_gbs_default": 2000.0, "write_bw_gbs_default": 1800.0,
        "read_lat_ns_default": 100, "write_lat_ns_default": 100,
        "parallelism_default": 100,
        "efficiency_default": {"GEMM": 0.85, "Attention": 0.60, "Softmax": 0.30, "LayerNorm": 0.25},
    },
    "DRAM_PIM": {
        "peak_tops_default": 50.0, "mem_gb_default": 512,
        "read_bw_gbs_default": 800.0, "write_bw_gbs_default": 600.0,
        "read_lat_ns_default": 50, "write_lat_ns_default": 80,
        "parallelism_default": 16,
        "efficiency_default": {"GEMM": 0.70, "Attention": 0.40, "Softmax": 0.20, "LayerNorm": 0.15},
    },
    "SRAM_PIM": {
        "peak_tops_default": 200.0, "mem_mb_default": 128,
        "read_bw_gbs_default": 4000.0, "write_bw_gbs_default": 3000.0,
        "read_lat_ns_default": 5, "write_lat_ns_default": 5,
        "parallelism_default": 32,
        "efficiency_default": {"GEMM": 0.80, "Attention": 0.50, "Softmax": 0.30, "LayerNorm": 0.25},
    },
    "RERAM_PIM": {
        "peak_tops_default": 100.0, "mem_gb_default": 1000,
        "read_bw_gbs_default": 300.0, "write_bw_gbs_default": 100.0,
        "read_lat_ns_default": 200, "write_lat_ns_default": 2000,
        "parallelism_default": 64,
        "efficiency_default": {"GEMM": 0.60, "Attention": 0.30, "Softmax": 0.10, "LayerNorm": 0.10},
    },
}


# ============================================================
# 解析出的中间结构
# ============================================================
@dataclass
class HardwareConfig:
    """单个硬件的解析后配置"""
    id: str
    type: str                          # GPU / DRAM_PIM / SRAM_PIM / RERAM_PIM
    peak_f: float                      # 标准单位 FLOPS
    mem_bytes: int                     # 标准单位 Byte
    read_bw: float                     # Bps
    write_bw: float                    # Bps
    read_lat_ns: int
    write_lat_ns: int
    parallelism: int
    efficiency: dict
    precision: list = field(default_factory=list)   # 支持的精度，空=默认支持全部


@dataclass
class LinkConfig:
    """解析后的连接配置（带宽已归一为 Bps）"""
    src: str
    dst: str
    read_bw: float
    write_bw: float
    lat_ns: int


@dataclass
class MappingRule:
    """算子级配置规则"""
    op: str                            # 算子 id 或 op_type（由 op_key 区分）
    op_key: str                        # "op_id" 或 "op_type"
    device: str = ""                   # 计算设备
    inputs: list = field(default_factory=list)   # list[dict] {data, from}
    devices: list = field(default_factory=list)  # 分片设备组（预留）
    split: dict = field(default_factory=dict)    # {dim, num_parts}（预留）


@dataclass
class PlacementRule:
    """数据初始驻留规则"""
    target: str                        # data_id 或 data_type（由 key 区分）
    key: str                           # "data_id" | "data_type"
    devices: list = field(default_factory=list)
    default_gbs_gbs: str = ""          # 未用


@dataclass
class ExperimentConfig:
    name: str
    model: str
    seed: int
    hardware: dict = field(default_factory=dict)          # {id: HardwareConfig}
    interconnect: Optional[object] = None                 # built later
    links: list = field(default_factory=list)             # list[LinkConfig]
    mapping: list = field(default_factory=list)           # list[MappingRule]
    mapping_default_device: str = ""
    mapping_default_source: str = "auto"
    placement: list = field(default_factory=list)         # list[PlacementRule]
    placement_default: str = ""
    output_dir: str = "results"


class ConfigError(Exception):
    """配置错误 —— 用户配置文件问题，信息要友好"""
    pass


def _load_yaml(path: str) -> dict:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        import yaml
    except ImportError:
        raise ConfigError("缺少 PyYAML，请执行: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_device_type(t: str) -> str:
    t = str(t).strip().upper()
    allowed = {"GPU", "DRAM_PIM", "SRAM_PIM", "RERAM_PIM", "CPU", "HBM"}
    if t not in allowed:
        raise ConfigError(
            f"未知硬件类型: '{t}'。可选: {sorted(allowed)}。"
            "提示: 用 type 决定出厂预设，再用 compute/memory 覆盖参数。")
    return t


# ============================================================
# 主加载函数
# ============================================================
def load_experiment(exp_path: str) -> "ExperimentIngredient":
    """读取一个 experiment.yaml，返回装配所需的原料"""
    exp = _load_yaml(exp_path)
    base_dir = os.path.dirname(os.path.abspath(exp_path))

    cfg = ExperimentConfig(
        name=exp.get("experiment", {}).get("name", "unnamed"),
        model=exp.get("experiment", {}).get("model", "tiny"),
        seed=exp.get("experiment", {}).get("seed", 42),
    )
    # 输出目录
    out = exp.get("experiment", {}).get("output", {})
    cfg.output_dir = out.get("dir", "results")

    # 定位各子配置文件（相对 experiment.yaml 所在目录）
    files = exp.get("experiment", {}).get("files", {})
    hw_path = _resolve(base_dir, files.get("hardware", "hardware.yaml"))
    int_path = _resolve(base_dir, files.get("interconnect", "interconnect.yaml"))
    map_path = _resolve(base_dir, files.get("mapping", "mapping.yaml"))
    plc_path = _resolve(base_dir, files.get("placement", "placement.yaml"))

    # 解析 hardware
    cfg.hardware = _parse_hardware(_load_yaml(hw_path))
    # 解析 interconnect
    cfg.links = _parse_interconnect(_load_yaml(int_path), cfg.hardware)
    # 解析 mapping
    map_doc = _load_yaml(map_path)
    cfg.mapping = _parse_mapping(map_doc)
    cfg.mapping_default_device = map_doc.get("default", {}).get("compute_device", "")
    cfg.mapping_default_source = map_doc.get("default", {}).get("data_source", "auto")
    # 解析 placement
    plc_doc = _load_yaml(plc_path)
    cfg.placement = _parse_placement(plc_doc)
    cfg.placement_default = plc_doc.get("default_device", "")

    return ExperimentIngredient(cfg)


def _resolve(base_dir: str, rel: str) -> str:
    """相对 experiment.yaml 解析路径；绝对路径直接用"""
    if os.path.isabs(rel):
        return rel
    return os.path.join(base_dir, rel)


def _parse_hardware(doc: dict) -> dict:
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

        pre = DEFAULT_DEVICE_PARAMS.get(dtype, {})
        comp = d.get("compute", {})
        mem = d.get("memory", {})
        eff = d.get("efficiency", pre.get("efficiency_default", {}))

        # 峰值计算
        peak_f = pre.get("peak_tflops_default", 300.0) * 1e12  # FLOPS 基准
        if "peak_tflops" in comp:
            peak_f = float(comp["peak_tflops"]) * 1e12
        elif "peak_tops" in comp:
            peak_f = float(comp["peak_tops"]) * 1e12

        # 容量
        mem_bytes = pre.get("mem_gb_default", 80) * 1e9
        if "capacity_gb" in mem:
            mem_bytes = float(mem["capacity_gb"]) * 1e9
        elif "capacity_mb" in mem:
            mem_bytes = float(mem["capacity_mb"]) * 1e6

        # 带宽
        def _bw(field_name, gbs_key, gbps_key, default):
            if gbs_key in mem:
                return float(mem[gbs_key]) * 1e9
            if gbps_key in mem:
                return float(mem[gbps_key]) * 1e9
            return default

        read_bw = _bw("read", "read_bandwidth_gbs", "read_bw_gbs",
                       pre.get("read_bw_gbs_default", 2000) * 1e9)
        write_bw = _bw("write", "write_bandwidth_gbs", "write_bw_gbs",
                       pre.get("write_bw_gbs_default", 1800) * 1e9)
        read_lat = int(mem.get("read_latency_ns", pre.get("read_lat_ns_default", 100)))
        write_lat = int(mem.get("write_latency_ns", pre.get("write_lat_ns_default", 100)))
        parallel = int(comp.get("parallelism", pre.get("parallelism_default", 1)))

        # 精度支持：hardware.yaml 里写 supported_precision 则覆盖（支持 4 档之一或多个），
        # 不写或为空则默认支持全部四种。
        precision_raw = d.get("supported_precision", [])
        precision = []
        if precision_raw:
            from contracts import PrecisionLevel as _PL
            for pname in precision_raw:
                try:
                    precision.append(_PL.from_name(str(pname)))
                except ValueError as e:
                    raise ConfigError(f"设备 {dev_id} 的 supported_precision 含未知精度: '{pname}'. "
                                      f"可选 FP32/FP16/INT8/INT4")

        out[dev_id] = HardwareConfig(
            id=dev_id, type=dtype, peak_f=peak_f, mem_bytes=int(mem_bytes),
            read_bw=read_bw, write_bw=write_bw, read_lat_ns=read_lat,
            write_lat_ns=write_lat, parallelism=parallel, efficiency=dict(eff),
            precision=precision,
        )
    return out


def _parse_interconnect(doc: dict, hardware: dict) -> list:
    """interconnect.yaml -> list[LinkConfig]
    未写带宽的，用默认连接表（按源/目的设备类型查）填参考值。"""
    out = []
    links = doc.get("links", [])
    # 先把 id->type 映射建好，用于查默认连接表
    id_type = {hid: hc.type for hid, hc in hardware.items()}

    for lk in links:
        src = lk.get("src")
        dst = lk.get("dst")
        if not src or not dst:
            raise ConfigError(f"连接缺少 src/dst: {lk}")
        if src not in id_type:
            raise ConfigError(f"连接引用未知设备 src='{src}'，设备列表: {list(id_type.keys())}")
        if dst not in id_type:
            raise ConfigError(f"连接引用未知设备 dst='{dst}'，设备列表: {list(id_type.keys())}")

        # 默认值：查 (src_type, dst_type)
        pair = (id_type[src], id_type[dst])
        defaults = DEFAULT_CONNECT_TABLE.get(pair, {"read_gbs": 100, "write_gbs": 50, "lat_ns": 500})

        read_bw = float(lk.get("read_bandwidth_gbs",
                        lk.get("read_bw_gbs", defaults["read_gbs"]))) * 1e9
        write_bw = float(lk.get("write_bandwidth_gbs",
                        lk.get("write_bw_gbs", defaults["write_gbs"]))) * 1e9
        lat = int(lk.get("latency_ns", defaults["lat_ns"]))

        bidirectional = lk.get("bidirectional", True)

        # 展开双向
        out.append(LinkConfig(src=src, dst=dst, read_bw=read_bw, write_bw=write_bw, lat_ns=lat))
        if bidirectional and (dst, src) != (src, dst):
            # 反向：对称取（读换写）
            back_pair = (id_type[dst], id_type[src])
            back_def = DEFAULT_CONNECT_TABLE.get(back_pair, defaults)
            back_read = float(lk.get("read_bandwidth_gbs", back_def["read_gbs"])) * 1e9
            back_write = float(lk.get("write_bandwidth_gbs", back_def["write_gbs"])) * 1e9
            out.append(LinkConfig(src=dst, dst=src, read_bw=back_read, write_bw=back_write,
                                  lat_ns=lat))
    return out


def _parse_mapping(doc: dict) -> list:
    out = []
    _KNOWN_TYPES = {"GEMM", "Attention", "Softmax", "LayerNorm", "Residual",
                    "Activation", "LMHead", "Embedding", "KVCacheUpdate"}
    for rule in doc.get("rules", []):
        # 兼容两种写法: op: xxx 或 op_type: GEMM
        if "op" in rule:
            op = str(rule["op"])
            # 判断是算子 id 还是类型:
            #  - 含通配符 * 的 → 视为 op_id 通配（如 L*_ffn_gate）
            #  - 纯大写且匹配已知类型名 → op_type
            if "*" in op:
                op_key = "op_id"
            elif op.isupper() and op in _KNOWN_TYPES:
                op_key = "op_type"
            elif op in _KNOWN_TYPES:
                op_key = "op_type"
            else:
                op_key = "op_id"
        elif "op_type" in rule:
            op = str(rule["op_type"])
            op_key = "op_type"
        else:
            raise ConfigError(f"mapping 规则缺少 'op' 或 'op_type': {rule}")
        inputs = []
        for i in rule.get("inputs", []):
            if not isinstance(i, dict) or "data" not in i:
                raise ConfigError(f"mapping 规则 inputs 格式错误: {i}")
            src = i.get("from", "")
            inputs.append({
                "data_id": str(i["data"]),
                "from": src,
                "pinned": bool(src),   # 写了 from 即用户固定
            })
        out.append(MappingRule(
            op=op, op_key=op_key,
            device=rule.get("device", ""),
            inputs=inputs,
            devices=rule.get("devices", []),
            split=rule.get("split", {}),
        ))
    return out


def _parse_placement(doc: dict) -> list:
    out = []
    for rule in doc.get("initial", []):
        if "data_type" in rule:
            target = str(rule["data_type"]).upper()
            key = "data_type"
        elif "data_id" in rule:
            target = str(rule["data_id"])
            key = "data_id"
        else:
            raise ConfigError(f"placement 规则缺少 data_type/data_id: {rule}")
        devices = rule.get("devices", [])
        if not devices:
            raise ConfigError(f"placement 规则 {target} 缺少 devices")
        out.append(PlacementRule(target=target, key=key, devices=[str(d) for d in devices]))
    return out


# ============================================================
# 装配产物
# ============================================================
class ExperimentIngredient:
    """加载完成后供各引擎使用的所有原料"""
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def model(self) -> str:
        return self.cfg.model

    @property
    def seed(self) -> int:
        return self.cfg.seed

    @property
    def output_dir(self) -> str:
        return self.cfg.output_dir
