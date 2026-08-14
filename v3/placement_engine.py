"""
LLM-PIMSim v3 — PlacementEngine
根据 placement.yaml 的规则 + 模型数据对象清单，生成:
  - initial_placement: {data_id: [devices]}   数据初始驻留（可冗余多设备）
用户先决定数据初始放哪（权重/激活/中间数据），冗余驻留也在此表达。

v3 解耦：从 core.common 取数据结构。
"""
from fnmatch import fnmatch
from core.common import DataObject, DataType
from config_loader import PlacementRule


class PlacementEngine:
    def __init__(self, rules: list, default_device: str):
        self.rules = rules
        self.default_device = default_device

    def apply(self, data_objects: dict) -> dict:
        """
        data_objects: {data_id: DataObject}
        returns {data_id: [devices]} —— 初始驻留设备列表（可能多设备=冗余）
        """
        result = {}
        for did, dobj in data_objects.items():
            devs = self._match(did, dobj.data_type)
            result[did] = devs
        return result

    def _match(self, data_id: str, data_type: DataType) -> list:
        # data_id 精确/通配优先
        for r in self.rules:
            if r.key == "data_id" and fnmatch(data_id, r.target):
                return list(r.devices)
        # data_type
        type_name = data_type.name  # e.g. WEIGHT
        for r in self.rules:
            if r.key == "data_type" and fnmatch(type_name, r.target):
                return list(r.devices)
        # 默认
        if self.default_device:
            return [self.default_device]
        return []
