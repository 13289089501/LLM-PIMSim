"""
LLM-PIMSim v3 — 实验配置加载器（装配层）

职责：
  1. 读 experiment.yaml，定位其引用的 hardware/interconnect/mapping/placement 子配置
  2. 解析映射(MappingRule) / 放置(PlacementRule) / workload 段
  3. 装配为 ExperimentIngredient 供各引擎使用

v3 解耦：
  - 硬件/YAML 加载、单位换算、硬件与互联解析已下沉到「硬件系统」core.hardware_sys。
    本文件的 DEFAULT_* / _parse_hardware / _parse_interconnect 等已移出。
  - ConfigError / load_yaml 来自 core.common（公共底层），由本文件重导出以保持旧接口。
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from core.common import ConfigError, load_yaml
from core.hardware_sys import (
    HardwareConfig, parse_hardware, parse_interconnect,
)
from core.link_sys import LinkBandwidthTable


# ============================================================
# 解析出的中间结构（映射 / 放置 / 实验）
# ============================================================
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


@dataclass
class ExperimentConfig:
    name: str
    model: str
    seed: int
    hardware: dict = field(default_factory=dict)          # {id: HardwareConfig}
    interconnect: Optional[object] = None                 # built later（兼容字段）
    links: object = None                                  # LinkBandwidthTable（链路系统）
    mapping: list = field(default_factory=list)           # list[MappingRule]
    mapping_default_device: str = ""
    mapping_default_source: str = "auto"
    placement: list = field(default_factory=list)         # list[PlacementRule]
    placement_default: str = ""
    output_dir: str = "results"
    workload: dict = field(default_factory=dict)          # 模型维度/输入输出规模


def _resolve(base_dir: str, rel: str) -> str:
    """相对 experiment.yaml 解析路径；绝对路径直接用"""
    if os.path.isabs(rel):
        return rel
    return os.path.join(base_dir, rel)


# ============================================================
# 主加载函数
# ============================================================
def load_experiment(exp_path: str) -> "ExperimentIngredient":
    """读取一个 experiment.yaml，返回装配所需的原料"""
    exp = load_yaml(exp_path)
    base_dir = os.path.dirname(os.path.abspath(exp_path))

    cfg = ExperimentConfig(
        name=exp.get("experiment", {}).get("name", "unnamed"),
        model=exp.get("experiment", {}).get("model", "llama7b"),
        seed=exp.get("experiment", {}).get("seed", 42),
    )
    out = exp.get("experiment", {}).get("output", {})
    cfg.output_dir = out.get("dir", "results")

    # 定位各子配置文件（相对 experiment.yaml 所在目录）
    files = exp.get("experiment", {}).get("files", {})
    hw_path = _resolve(base_dir, files.get("hardware", "hardware.yaml"))
    int_path = _resolve(base_dir, files.get("interconnect", "interconnect.yaml"))
    map_path = _resolve(base_dir, files.get("mapping", "mapping.yaml"))
    plc_path = _resolve(base_dir, files.get("placement", "placement.yaml"))

    # 解析 hardware / interconnect —— 委托硬件系统
    cfg.hardware = parse_hardware(load_yaml(hw_path))
    cfg.links = parse_interconnect(load_yaml(int_path), cfg.hardware)

    # 解析 mapping
    map_doc = load_yaml(map_path)
    cfg.mapping = _parse_mapping(map_doc)
    cfg.mapping_default_device = map_doc.get("default", {}).get("compute_device", "")
    cfg.mapping_default_source = map_doc.get("default", {}).get("data_source", "auto")

    # 解析 placement
    plc_doc = load_yaml(plc_path)
    cfg.placement = _parse_placement(plc_doc)
    cfg.placement_default = plc_doc.get("default_device", "")

    # workload 段（自定义模型维度与输入规模）
    wl_doc = exp.get("experiment", {}).get("workload", {}) or {}
    if isinstance(wl_doc, dict):
        cfg.workload = dict(wl_doc)

    return ExperimentIngredient(cfg)


def _parse_mapping(doc: dict) -> list:
    out = []
    _KNOWN_TYPES = {"GEMM", "Attention", "Softmax", "LayerNorm", "Residual",
                    "Activation", "LMHead", "Embedding", "KVCacheUpdate"}
    for rule in doc.get("rules", []):
        if "op" in rule:
            op = str(rule["op"])
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
                "pinned": bool(src),
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


# 兼容别名：旧代码 `from config_loader import _load_yaml` 可用
_load_yaml = load_yaml
