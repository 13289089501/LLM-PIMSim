"""
LLM-PIMSim v3 core.link_sys — 【链路系统】

职责（"链路带宽"这一概念的完整闭环逻辑集中于此）：
  1. 数据结构：LinkBandwidthTable —— N×N 对称带宽查找表（按"设备种类 kind"索引）
  2. 出厂预设：7 种默认设备种类间的链路带宽表（对称，单位 GB/s）
  3. 解析/装配：从 hardware.yaml（每设备 links 段，自定义硬件"加一栏"）与
     interconnect.yaml（link_bw_gbs 全局表）合并成单一带宽表，供核心调度器
     计算跨设备搬运延时。

约定（与硬件系统/调度系统的分工）：
  - 表只存"链路带宽"（GB/s），不含链路延迟；
  - 读/写延迟与读/写带宽由【硬件系统】core.hardware_sys 提供；
  - 跨设备搬运延时（见 core.engine.PerformanceModel.transfer_time_ns）：
        T(A→B) = A读延迟 + S/A读带宽 + S/链路带宽(A,B) + B写延迟 + S/B写带宽
  - 表按"设备种类"（type / link_type 字符串）索引，且对称：bandwidth(A,B)==bandwidth(B,A)；
    对角线表示"同种类不同设备之间"的互连带宽（如 GPU↔GPU 的 NVLink）。

依赖：无（纯数据结构 + 单位换算）。不依赖硬件/算子/调度，是可与硬件系统并列的独立系统。

顶层兼容：`hardware.py` 转发 `LinkBandwidthTable`；`hardware_factory.build_hardware`
返回 (devices, link_table)。
"""


# =================================================================
# 1. 出厂预设：7 种默认设备种类间的对称链路带宽（GB/s）
# =================================================================
# 表只写"上三角 + 对角线"；读取时对称镜像，无需写满。
# 数值口径（延续 v3.1 硬件校准，见 修改文档/链路系统.txt 的"数据来源"节）：
#   GPU↔GPU(NVLink) 600；GPU↔PIM/PCIe 类 100~200；PIM↔PIM 更高；
#   SRAM-PIM↔SRAM-PIM 与 SRAM 片上互连取值较高；ReRAM 写受限但链路取读写上限。
DEFAULT_LINK_BW_TABLE = {
    "CPU":       {"CPU": 300.0, "GPU": 100.0, "DRAM_PIM": 120.0, "SRAM_PIM": 200.0,
                  "RERAM_PIM": 100.0, "SRAM": 200.0, "DRAM": 200.0},
    "GPU":       {"GPU": 600.0, "DRAM_PIM": 120.0, "SRAM_PIM": 200.0,
                  "RERAM_PIM": 100.0, "SRAM": 200.0, "DRAM": 150.0},
    "DRAM_PIM":  {"DRAM_PIM": 600.0, "SRAM_PIM": 150.0, "RERAM_PIM": 100.0,
                  "SRAM": 200.0, "DRAM": 150.0},
    "SRAM_PIM":  {"SRAM_PIM": 1500.0, "RERAM_PIM": 100.0, "SRAM": 500.0, "DRAM": 200.0},
    "RERAM_PIM": {"RERAM_PIM": 128.0, "SRAM": 100.0, "DRAM": 100.0},
    "SRAM":      {"SRAM": 1000.0, "DRAM": 200.0},
    "DRAM":      {"DRAM": 200.0},
}

# 未显式配置的"种类对"回退到此缺省带宽（GB/s），保证 N×N 表永远闭合、任意两设备可达。
DEFAULT_LINK_BW_GBS = 100.0


# =================================================================
# 2. LinkBandwidthTable —— N×N 对称带宽查找表
# =================================================================
class LinkBandwidthTable:
    """N×N 对称链路带宽查找表。

    key = 设备种类字符串（type / link_type，统一大写）；value = GB/s。
    对称不变量：set_bw/get_bw 自动镜像 A↔B，保证 bandwidth(A,B)==bandwidth(B,A)。
    """

    def __init__(self, table: dict = None, fallback_gbs: float = DEFAULT_LINK_BW_GBS):
        self._bw: dict = {}                       # {kind: {kind: gbs}}
        self.fallback_gbs = float(fallback_gbs)
        if table:
            self.update(table)

    # ---- 基础工具 ----
    @staticmethod
    def _norm(k) -> str:
        return str(k).strip().upper()

    # ---- 写入 ----
    def update(self, table: dict) -> "LinkBandwidthTable":
        """合并一张表，支持两种形态：
          - {kind: {kind: gbs}}：逐对写入（对称）；
          - {kind: gbs}：视为"该种类与自身"（同种类互连）的带宽。"""
        for a, row in (table or {}).items():
            a = self._norm(a)
            if isinstance(row, dict):
                for b, v in row.items():
                    self.set_bw(a, self._norm(b), v)
            else:
                self.set_bw(a, a, row)
        return self

    def set_bw(self, a, b, gbs) -> "LinkBandwidthTable":
        """设置种类 a↔b 的链路带宽（GB/s），对称写入。"""
        a, b = self._norm(a), self._norm(b)
        gbs = float(gbs)
        if a == b:
            self._bw.setdefault(a, {})[a] = gbs
        else:
            self._bw.setdefault(a, {})[b] = gbs
            self._bw.setdefault(b, {})[a] = gbs
        return self

    def add_type(self, kind, bw_map: dict = None) -> "LinkBandwidthTable":
        """新增一个"设备种类"（自定义硬件）：登记它到已有种类间的带宽。
        bw_map: {已有种类: GB/s}；未给的配对回退缺省带宽；同种类互连可取 bw_map[kind]。"""
        kind = self._norm(kind)
        self._bw.setdefault(kind, {})
        for other, gbs in (bw_map or {}).items():
            self.set_bw(kind, self._norm(other), gbs)
        return self

    # ---- 读取 ----
    def get_bw_gbs(self, a, b) -> float:
        a, b = self._norm(a), self._norm(b)
        if a == b:
            return self._bw.get(a, {}).get(a, self.fallback_gbs)
        row = self._bw.get(a, {})
        if b in row:
            return row[b]
        row = self._bw.get(b, {})
        if a in row:
            return row[a]
        return self.fallback_gbs

    def get_bw_bps(self, a, b) -> float:
        """返回 a↔b 的链路带宽（B/s，供搬运耗时公式用）。"""
        return self.get_bw_gbs(a, b) * 1e9

    # ---- 其它 ----
    def types(self) -> list:
        """当前表里登记的所有设备种类（大写、排序）。"""
        return sorted(self._bw.keys())

    def to_dict(self) -> dict:
        """完整对称表（供序列化/落盘/前端展示）。"""
        return {a: dict(row) for a, row in sorted(self._bw.items())}

    @classmethod
    def default(cls) -> "LinkBandwidthTable":
        """以 7 种默认设备种类的出厂表构造。"""
        return cls(DEFAULT_LINK_BW_TABLE)
