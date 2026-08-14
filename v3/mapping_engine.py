"""
LLM-PIMSim v3 — MappingEngine
根据 mapping.yaml 的规则 + 模型算子清单，生成:
  - compute_map:  {op_id: compute_device}       算子放哪算（用户决定）
  - input_specs:  {op_id: list[InputSpec]}      算子每份输入从哪来（用户决定）
调度器负责：用户固定(pinned)的来源严格生效；auto 的来源给"就近参考"。

v3 解耦：从 core.common 取数据结构，不再经由 contracts 转发层。
"""
from fnmatch import fnmatch
from core.common import OperatorSpec, InputSpec
from config_loader import MappingRule


class MappingEngine:
    def __init__(self, rules: list, default_device: str, default_source: str):
        self.rules = rules               # list[MappingRule]
        self.default_device = default_device
        self.default_source = default_source

    def apply(self, operators: dict) -> dict:
        """
        operators: {op_id: Operator}
        returns dict of OperatorSpec, keyed by op_id
        """
        specs = {}
        # 1. 先按规则填充
        for op_id, op in operators.items():
            rule = self._match_rule(op_id, op.op_type)
            spec = OperatorSpec(op_id=op_id)
            if rule is not None:
                spec.compute_device = rule.device
                spec.devices = list(rule.devices) if rule.devices else [rule.device]
                spec.split = dict(rule.split)
                # 数据源：规则 inputs -> InputSpec
                for i in rule.inputs:
                    spec.inputs.append(InputSpec(
                        data_id=i["data_id"],
                        source_device=i["from"],
                        pinned=i["pinned"],
                    ))
            else:
                spec.compute_device = self.default_device
                spec.devices = [self.default_device] if self.default_device else []
            specs[op_id] = spec

        return specs

    def _match_rule(self, op_id: str, op_type: str) -> MappingRule:
        # 精确 op_id 优先
        for r in self.rules:
            if r.op_key == "op_id" and (r.op == op_id or fnmatch(op_id, r.op)):
                return r
        # 再按 op_type
        for r in self.rules:
            if r.op_key == "op_type" and (r.op == op_type or fnmatch(r.op, op_type)):
                return r
        return None
