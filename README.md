# LLM-PIMSim

> An architecture-level discrete-event simulator for analyzing LLM inference acceleration with Processing-In-Memory (PIM) and heterogeneous computing architectures.

---

## 版本与快速开始

本仓库包含两代实现：

| 版本 | 目录 | 说明 |
|---|---|---|
| **v3（最新·推荐）** | `v3/` | 八大系统解耦（精度/硬件/算子/权重/切割/校验/输出/调度）+ 前后端打通 + GUI + IC 参考实验，可直接运行 |
| v2 | `v2/` | 早期配置驱动实现（供参考历史） |
| v1 | `/v1` | 最早原型（已不进仓库） |

### 快速开始（v3）

```bash
cd v3
pip install -r requirements.txt     # flask + pyyaml

# 方式 A：命令行（跑内置 IC 参考实验）
python run.py configs/experiments/04_ic_reference.yaml

# 方式 B：图形界面（拖拽拓扑编辑器 + 实时校验 + 依赖图 + 关键路径）
python gui_app.py                    # 打开 http://127.0.0.1:5000
```

v3 详细文档见 [`v3/README.md`](v3/README.md) 与 [`v3/CHANGELOG_v3.md`](v3/CHANGELOG_v3.md)。

### v3 核心特性

- **八大系统解耦**：`core/` 下按 精度 / 硬件 / 算子 / 权重 / 切割 / 校验 / 输出 / 调度 单向依赖。
- **配置驱动**：硬件/互连/映射/放置 全由 YAML 描述，另提供一键 `04_ic_reference` 参考部署。
- **张量并行切片**：`workload.splits` 支持按维度切分 + 权重分片式多设备并行（ALL-GATHER）。
- **校验充分性**：校验通过即可完整运行，校验不过则拒绝运行（前后端同一套判定）。
- **GUI**：拖拽部署、层折叠、依赖图 DAG、关键路径视图、结果对比、初学者友好提示。

---

[English](#english) | [中文](#中文)

---

# 中文

## 1. 项目简介

**LLM-PIMSim** 是一个面向 **大语言模型（LLM）推理与存内计算（PIM）架构研究**的架构级性能模拟器。

本项目的目标并不是进行 RTL 级、电路级或器件级精确仿真，而是建立一个**可解释、可扩展的离散事件模拟框架**，用于研究和理解 LLM 推理过程中的：

* 计算开销
* 本地存储访问开销
* 跨设备数据搬运开销
* 数据依赖与同步等待
* 权重驻留（Weight Residency）
* KV Cache 访问
* GPU 与不同 PIM 架构之间的性能差异

项目希望回答的核心问题是：

> **LLM 推理到底受计算能力、存储带宽还是数据通信限制？PIM 架构为什么能够加速某些 LLM 工作负载，而在另一些情况下收益有限？**

---

## 2. 设计目标

LLM-PIMSim 旨在提供一个用于研究 LLM-PIM 系统架构的统一分析平台。

### 主要目标

* 模拟 Transformer/LLM 推理过程
* 支持算子级计算图和数据依赖
* 支持 GPU、DRAM-PIM、SRAM-PIM 等异构设备抽象
* 模拟计算、本地存储访问和跨设备数据传输
* 支持权重预先放置和 Weight Residency 分析
* 支持 KV Cache 建模
* 支持多设备并行执行和同步等待
* 基于离散事件模拟生成完整执行时间线
* 分析 Compute / Memory / Communication / Synchronization Bottleneck
* 为复现和理解 LLM-PIM 加速架构论文提供基础平台

---

## 3. 项目定位

LLM-PIMSim 是一个：

```text
Architecture-level Performance Simulator
```

而不是：

```text
RTL Simulator
Circuit Simulator
Device-level Simulator
```

因此，本项目重点关注：

```text
LLM Workload
        +
Hardware Architecture
        +
Data Placement
        +
Operator Mapping
        ↓
Discrete Event Simulation
        ↓
Performance Analysis
```

目前不以精确模拟以下内容为目标：

* RTL 时序
* 门级逻辑
* DRAM Bank/Row Buffer 详细行为
* SRAM Bit-cell
* ReRAM 器件物理特性
* ADC/DAC 电路细节
* Cache Coherence
* 热分析

这些内容可以作为未来扩展方向。

---

## 4. 核心设计思想

### 4.1 离散事件模拟

项目采用 **Discrete Event Simulation（DES，离散事件模拟）**。

系统时间不会按照固定步长推进，而是直接跳转到下一个事件发生的时间点。

例如：

```text
Time = 0
│
├── PIM0: FFN Compute Started
│
├── GPU0: Attention Compute Started
│
├── Link: Weight Transfer Started
│
▼
Time = 5 ms
│
└── Weight Transfer Finished
        │
        ▼
     Data Ready
        │
        ▼
     Next Operator Can Start
```

核心事件包括：

* Compute Event
* Transfer Event
* Memory Access Event
* Synchronization / Dependency Event

---

### 4.2 计算与数据移动分离

一个算子的性能不能简单表示为：

[
T=\frac{FLOPs}{Peak\ Compute}
]

LLM-PIMSim 将性能拆分为多个来源：

[
T_{compute}
]

[
T_{local\ memory}
]

[
T_{transfer}
]

以及由数据依赖产生的：

[
T_{synchronization}
]

第一版以清晰、可解释的模型为优先。

---

### 4.3 数据依赖驱动执行

Operator 只有在：

1. 所有输入数据已经准备完成；
2. 必要数据已经位于可访问位置；
3. 所需硬件资源可用；

时才能开始执行。

例如：

```text
PIM0 ── Result A ─┐
PIM1 ── Result B ─┤
PIM2 ── Result C ─┼──> Softmax
PIM3 ── Result D ─┘
```

Softmax 必须等待：

[
ReadyTime=
\max(
Ready_A,
Ready_B,
Ready_C,
Ready_D
)
]

这使模拟器能够分析：

> 多设备并行计算后产生的同步等待和负载不均衡。

---

## 5. 性能模型

### 5.1 计算时间

基本计算模型：

[
T_{compute}
===========

\frac{FLOPs}
{PeakCompute \times Efficiency}
]

其中：

* `FLOPs`：算子计算量
* `PeakCompute`：目标设备的峰值计算能力
* `Efficiency`：算子在该设备上的有效计算效率

第一版中，Efficiency 可以根据：

```text
Hardware Type + Operator Type
```

进行配置。

---

### 5.2 本地存储访问

读取：

[
T_{read}
========

L_{read}
+
\frac{D}{BW_{read}}
]

写入：

[
T_{write}
=========

L_{write}
+
\frac{D}{BW_{write}}
]

其中：

* (L)：固定访问延迟
* (D)：数据量
* (BW)：对应读写带宽

读写参数独立配置。

---

### 5.3 跨设备数据传输

对于：

```text
Device A
    │
    │ Data
    ▼
Device B
```

第一版采用：

```text
Source Read
     +
Interconnect Transfer
     +
Destination Write
```

因此：

[
T_{transfer}
============

T_{read}(A)
+
T_{link}(A,B)
+
T_{write}(B)
]

其中：

[
T_{link}
========

L_{link}
+
\frac{D}{BW_{link}}
]

---

## 6. Weight Residency

Weight Residency 用于描述：

> 算子执行时，所需权重是否已经位于执行设备中。

例如：

```text
FFN Weight Location = DRAM-PIM0
FFN Compute Device  = DRAM-PIM0
```

则：

```text
Weight Residency = True
```

无需进行跨设备权重搬运。

而：

```text
FFN Weight Location = DRAM-PIM0
FFN Compute Device  = GPU0
```

则需要：

```text
DRAM-PIM0
    │
    │ Weight Transfer
    ▼
GPU0
```

产生相应的数据传输事件。

权重初始放置由用户配置，模拟器负责根据数据位置和算子映射自动处理执行过程中需要的数据移动。

---

## 7. KV Cache

LLM-PIMSim 将 KV Cache 作为特殊数据对象进行建模。

在 Decode 阶段，Attention 会持续访问历史 Token 对应的 Key 和 Value。

随着序列长度增加：

[
KV\ Cache\ Size
\propto
Sequence\ Length
]

因此模拟器可以研究：

* KV Cache 的存储位置
* KV Cache 的访问设备
* 跨设备 KV 数据搬运
* Sequence Length 对 Attention 延迟的影响
* Decode 阶段瓶颈变化

---

## 8. 瓶颈分析

LLM-PIMSim 将主要性能瓶颈划分为：

### Compute Bound

计算时间占主导。

```text
Compute
████████████████████
Memory
████
Transfer
██
```

优化方向通常包括：

* 增加计算能力
* 提高算子执行效率
* 提高并行度

---

### Memory Bound

本地存储访问成为主要限制。

优化方向通常包括：

* 提高 Memory Bandwidth
* 改进数据复用
* 降低 Memory Access

---

### Communication Bound

跨设备数据移动成为主要限制。

例如：

```text
GPU ↔ PIM
PIM ↔ PIM
Memory ↔ Compute
```

优化方向包括：

* 提高互连带宽
* 降低数据搬运量
* 提高数据局部性
* 使用 Weight Residency

---

### Synchronization Bound

多个任务或数据分支完成时间不一致，导致后续任务必须等待最慢的依赖。

例如：

```text
PIM0: 2 ms
PIM1: 3 ms
PIM2: 3 ms
PIM3: 10 ms
```

后续全局操作最早只能在：

```text
10 ms
```

开始。

---

## 9. 系统架构

```text
                         User Configuration
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Config Parser  │
                        └────────┬────────┘
                                 │
                 ┌───────────────┼────────────────┐
                 ▼               ▼                ▼
          ┌────────────┐  ┌────────────┐  ┌────────────┐
          │ LLM Model  │  │  Hardware  │  │ Data Model │
          └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
                │               │                │
                └───────────────┼────────────────┘
                                ▼
                       ┌──────────────────┐
                       │  Operator Graph  │
                       └────────┬─────────┘
                                ▼
                       ┌──────────────────┐
                       │    Scheduler     │
                       │ Discrete Events  │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Performance Model│
                       └────────┬─────────┘
                                ▼
                       ┌──────────────────┐
                       │   Event Trace    │
                       └────────┬─────────┘
                                ▼
                       ┌──────────────────┐
                       │ Result Analyzer  │
                       └────────┬─────────┘
                                ▼
                    Structured Results / API / JSON
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             CLI / Experiment         Future GUI
```

---

## 10. 项目结构

项目计划采用以下结构：

```text
LLM-PIMSim/
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── models/
│   ├── hardware/
│   ├── placement/
│   └── experiments/
│
├── src/
│   └── llm_pimsim/
│       ├── core/
│       ├── models/
│       ├── hardware/
│       ├── performance/
│       ├── scheduler/
│       ├── analyzer/
│       ├── io/
│       └── api/
│
├── tests/
│
├── examples/
│
├── results/
│
└── docs/
```

具体代码组织将在项目开发过程中根据架构设计进一步确定。

---

## 11. 主要模块

项目核心模块包括：

```text
LLM Model
Operator Model
Data Model
Hardware Model
Performance Model
Scheduler
Experiment Analyzer
```

模块设计的详细说明见：

```text
docs/
```

中的项目设计文档。

核心原则是：

> **模型描述、性能估算、事件调度、结果分析和用户界面彼此解耦。**

---

## 12. GUI 规划

本项目未来计划提供图形化界面。

预计支持：

### 硬件拓扑

例如：

```text
       ┌────────┐
       │ GPU0   │
       └────┬───┘
            │
         Link / NoC
       ┌────┴────┐
       ▼         ▼
   DRAM-PIM0  DRAM-PIM1
```

### 数据移动可视化

显示：

```text
Source
   │
   │ Data Transfer
   ▼
Destination
```

### 执行时间线

例如：

```text
GPU0      |----Attention----|      |--Softmax--|

PIM0           |------FFN------|

Link      |--Weight Transfer--|
```

### 性能分析

显示：

* Total Latency
* Compute Time
* Memory Time
* Transfer Time
* Synchronization Time
* Bottleneck
* Operator Statistics

因此，从项目开始就要求：

> **Simulation Core 不依赖 GUI。**

核心仿真结果将通过结构化数据输出，为未来的 CLI、Web GUI 或桌面 GUI 提供统一接口。

---

## 13. 配置驱动

LLM-PIMSim 采用配置驱动设计。

用户未来可以通过配置描述：

### LLM

```yaml
model:
  layers: 32
  hidden_size: 4096
  attention_heads: 32
  precision: FP16
```

### Hardware

```yaml
devices:
  GPU0:
    type: GPU

  DRAM_PIM0:
    type: DRAM_PIM
```

### Data Placement

```yaml
placement:
  weights:
    FFN: DRAM_PIM0
    Attention: GPU0
```

### Operator Mapping

```yaml
mapping:
  Attention: GPU0
  FFN: DRAM_PIM0
```

配置格式可能随着项目发展调整。

---

## 14. 当前开发状态

> **项目目前处于架构设计与 MVP 开发准备阶段。**

当前重点：

* [x] 明确项目研究目标
* [x] 完成总体模拟器设计
* [x] 完成硬件抽象设计
* [x] 完成数据模型设计
* [x] 完成 LLM 算子模型设计
* [x] 完成性能模型设计
* [x] 完成离散事件调度设计
* [x] 完成软件架构初步设计
* [x] 完成实验分析设计
* [ ] 完成 MVP 软件架构评审
* [ ] 实现最小可运行版本
* [ ] 实现基础 Transformer 推理模拟
* [ ] 实现 GPU + PIM 异构模拟
* [ ] 实现实验与结果分析
* [ ] 实现图形化界面

---

## 15. 开发路线

### Phase 1 — Architecture

完成：

* 软件架构评审
* 数据接口定义
* 核心数据结构
* 模块边界设计

目标：

> 建立稳定的软件骨架。

---

### Phase 2 — MVP

实现最小可运行模拟器：

* 简单 Operator
* 简单 Hardware
* 基础 Performance Model
* Compute Event
* Transfer Event
* Event Queue

目标：

> 完成一个端到端的离散事件仿真流程。

---

### Phase 3 — LLM & PIM

加入：

* Transformer Operator Graph
* Attention
* FFN
* KV Cache
* Weight Residency
* GPU + PIM
* 数据依赖

目标：

> 支持基础 LLM-PIM 架构分析。

---

### Phase 4 — Experiment & Analysis

加入：

* 批量实验
* 参数扫描
* Operator 统计
* Data Movement 统计
* Bottleneck Analysis
* Speedup Analysis

---

### Phase 5 — Visualization

开发：

* 硬件拓扑可视化
* 数据流可视化
* Event Timeline
* Performance Dashboard

---

## 16. 贡献

欢迎对以下方向做出贡献：

* LLM Operator 建模
* Transformer 性能模型
* GPU/PIM 硬件模型
* DRAM-PIM
* SRAM-PIM
* HBM-PIM
* 新型 PIM 架构
* 调度算法
* 性能分析
* 可视化
* 测试与验证

如果你希望新增硬件或算子模型，请尽量遵循项目的模块化接口设计，避免直接修改核心调度逻辑。

---

## 17. 项目原则

本项目遵循以下原则：

### 可解释性优先

相比复杂但难以理解的黑盒模型，我们更重视：

> 能够解释一个性能结果为什么产生。

---

### 架构可扩展性优先

不将具体 GPU、PIM 或 LLM 参数写死在核心代码中。

---

### 模型与调度分离

```text
What to execute
      ≠
Where to execute
      ≠
When to execute
      ≠
How long it takes
```

分别由不同模块处理。

---

### Simulation Core 与 GUI 分离

```text
Simulation Core
        ↓
Structured Results
        ↓
CLI / API / GUI
```

GUI 不参与核心仿真逻辑。

---

## 18. 限制与免责声明

LLM-PIMSim 是一个架构级研究工具。

模拟结果依赖于：

* 硬件参数
* 性能模型
* 算子效率参数
* 数据放置策略
* 调度策略

因此：

> 模拟结果不应被直接视为真实硬件测量结果。

项目更适合用于：

* 架构设计空间探索
* 性能趋势分析
* PIM 架构比较
* 数据移动分析
* 瓶颈定位
* 辅助理解研究论文中的设计选择

如果需要获得精确的真实硬件性能，应结合：

* 论文实验数据
* 硬件测量
* 更细粒度的硬件模拟

进行验证。

---

## 19. Citation

如果本项目未来用于论文或研究工作，Citation 信息将在正式发布后补充。

```bibtex
@software{llm_pimsim,
  title  = {LLM-PIMSim: An Architecture-Level Simulator for LLM Inference with Processing-In-Memory},
  author = {Project Contributors},
  year   = {2026}
}
```

---

## 20. License

本项目采用 **MIT License** 发布，详见 `v3/LICENSE`。可自由使用、修改、分发（含商用），需保留版权与许可证声明。

---

## 21. Contact

欢迎通过 GitHub Issues 和 Pull Requests 参与项目讨论与贡献。

---

# English

## LLM-PIMSim

**LLM-PIMSim** is an architecture-level, discrete-event performance simulator for studying LLM inference acceleration with Processing-In-Memory (PIM) and heterogeneous computing architectures.

The project focuses on understanding and analyzing:

* Compute cost
* Local memory access cost
* Inter-device data movement
* Data dependencies and synchronization
* Weight residency
* KV cache access
* GPU/PIM heterogeneous execution
* Compute, memory, communication, and synchronization bottlenecks

The central question is:

> **What actually limits LLM inference performance, and when does PIM provide meaningful acceleration?**

### Key Features

* Architecture-level LLM inference modeling
* Operator graph and data dependency modeling
* Discrete-event simulation
* Heterogeneous GPU/PIM hardware abstraction
* Compute, memory, and communication latency modeling
* Weight residency analysis
* KV cache modeling
* Multi-device synchronization modeling
* Event timeline generation
* Bottleneck analysis
* Extensible architecture for future PIM devices and scheduling policies
* Planned GUI and visualization support

### Project Status

✅ **v3 is released** — a full implementation with eight decoupled core systems, GUI, and an IC reference experiment. See the Chinese quick-start above and `v3/README.md`.

### Architecture

```text
Configuration
      ↓
LLM / Hardware / Data Models
      ↓
Operator Graph
      ↓
Discrete Event Scheduler
      ↓
Performance Models
      ↓
Event Trace
      ↓
Result Analysis
      ↓
Structured Results
      ↓
CLI / API / Future GUI
```

See the Chinese documentation above for the current detailed design.

---

## License

本项目采用 **MIT License** 发布。

## Citation

Citation 信息将在正式发布后补充。
