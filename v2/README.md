# LLM-PIMSim v2 — 配置驱动的 LLM×PIM 异构推理仿真器

面向大语言模型推理架构探索的轻量级离散事件模拟器。
在**纯软件环境**下回答一个体系结构问题：

> 把 LLM 的算子、权重、中间结果放到 GPU / DRAM-PIM / SRAM-PIM / ReRAM-PIM
> 这些异构设备上，哪种部署方案延迟最低？瓶颈在哪？

v2 的核心设计原则：**算子固定，数据与硬件可变；方案即配置，结果即 JSON**。

---

## 1. 项目解决什么问题

真实 LLM 推理（特别是 PIM 异构场景）的部署决策很难靠直觉做：

- GPU 的 HBM 容量有限，**不可能放下全模型权重**——哪些权重必须放 GPU、哪些可以放 PIM？
- 同一层里有 GEMM（高算术强度）、LayerNorm / Softmax（带宽受限、精度敏感），**分别放哪个设备**最优？
- 权重被切分到多个 PIM 通道后，算子必须**读齐全部切片**才能算（ALL-GATHER），调度器如何表达这个依赖？
- 中间结果应该**就地存放**还是搬回大容量设备？每次跨设备搬运要付多少带宽代价？

本模拟器用离散事件调度（DES）+ 可配置性能模型，把这些问题变成可复现的数字：
给出**总延迟、计算/搬运/同步分解、瓶颈类型、逐算子时间线、事件轨迹**，
并内置一套**配置校验器**，在运行前拦截不合理方案（容量超限、精度不兼容、数据不可达、权重不完整等）。

### 典型研究问题（可直接回答）

- GPU-only vs GPU+PIM vs GPU+PIM+SRAM：哪个快？瓶颈在哪？
- FFN 权重放 PIM 相比放 GPU，能省多少权重搬运？
- 权重切 2 片 / 4 片跨 PIM 通道，ALL-GATHER 的搬运开销多大？
- LN/Softmax 放到 SRAM-PIM 对总延迟的影响？

---

## 2. 主要方法

### 2.1 配置驱动（方案 = YAML）

用户不写 Python，通过 4 类配置描述整套方案：

| 配置 | 作用 | 对应文件 |
|---|---|---|
| **hardware** | 设备清单（类型/算力/容量/带宽/算子效率） | `hardware.yaml` |
| **interconnect** | 设备间连接（非对称读写带宽/延迟） | `interconnect.yaml` |
| **mapping** | 算子 → 计算设备 + 每份数据从哪读 | `mapping.yaml` |
| **placement** | 数据（权重/激活/KV）初始驻留在哪 | `placement.yaml` |
| **experiment** | 一次实验（模型/种子/引用上述文件/输出目录） | `configs/experiments/NN_*.yaml` |

### 2.2 两级驱动路径

- **model_lib 路径**（`run.py` 默认）：内置模型库直接生成算子图。
- **workload 路径**（`run.py --workload` / GUI）：从 kernel 粒度 workload 展开可执行算子图
  （16 个算子/层 × 32 层 = 514 个算子），支持 KV 动态维度。**GUI 与推荐实验使用此路径**。

### 2.3 权重一级节点 + ALL-GATHER（v2 核心新特性）

权重不再只是算子的"输入字符串"，而是独立数据节点 `WeightBlock`：

- **按张量形状归类**（不按算子）：

  | 类别 | 成员（每层） | 可切维 |
  |---|---|---|
  | `W_attn` | q/k/v/o 四投影 `[H,H]` | 沿 head 列 |
  | `W_mlp` | ffn_gate/up/down `[H,·]` | 沿中间维 |
  | `W_ln` | ln1/ln2 `[H]` | 通常不切 |
  | `W_head` / `W_embed` | 词表级（全局） | 沿词表行 |

- **可切割**：一类权重可切 N 片放不同设备；需要它的算子必须**连接全部切片**才允许运行
  （**W1 权重完整性 / ALL-GATHER 语义**，校验器 + 调度器双实现）。
- **容量真实化**：每个权重块的字节 × 全模型层数计入所在硬件容量（C1 规则），
  "GPU 8GB 塞下全部权重"这类方案会被校验器直接拦截。

### 2.4 IC 参考方案（一键部署）

内置一套以集成电路视角设计的参考部署，加载模型即可一键生成：

- **算子**：attention 路径 GEMM → GPU；FFN 三个 GEMM + KV 更新 → DRAM-PIM；
  LN / Softmax / 激活 → SRAM-PIM（高带宽、FP32 精度）。
- **权重**：`W_attn` → GPU；`W_mlp` → DRAM-PIM；`W_ln` → SRAM-PIM；词表 → ReRAM-PIM。
- **中间结果就地存放**，只有跨设备算子边界才搬运小尺寸激活。
- 生成完整数据流连线 + 互连链路，并用校验器验证后返回 valid/errors。

### 2.5 离散事件调度内核

- 事件队列 + 时钟跳变推进；算子状态机 WAITING → READY → RUNNING → FINISHED。
- 数据源决策：用户固定源（pinned）严格生效；未指定则"就近参考"（min 就绪+搬运耗时），
  所有决策记录到 `data_source_notes` 供审阅。
- 权重分片 ALL-GATHER：等全部切片就绪才 READY，从各自设备分别 gather。

### 2.6 配置校验器（运行前把关）

A 算子-硬件兼容 · B 数据流完整性 · C 存储容量（含权重×层数）·
D 数据可达性 · E 全局有效性 · **W1 权重完整性（ALL-GATHER）**。
任何 ERROR 阻止运行；WARNING 仅提示。

---

## 3. 安装方法

### 环境要求

- Python **3.9+**（Windows / Linux / macOS 均可）
- 无 GPU 需求——纯软件仿真，普通笔记本即可运行

### 安装步骤

```bash
# 1. 进入项目 v2 目录
cd v2

# 2.（推荐）创建虚拟环境
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

> 依赖仅 `flask` 与 `pyyaml`（无 requirements.txt 时执行
> `pip install flask pyyaml` 即可）。

### 验证安装

```bash
python -c "from config_loader import load_experiment; print('OK')"
```

---

## 4. 使用方法

### 方式 A：命令行（CLI）

```bash
cd v2

# 跑单个实验（默认 model_lib 路径）
python run.py configs/experiments/01_gpu_only.yaml

# 用 workload 路径（kernel 粒度，推荐）
python run.py --workload configs/experiments/02_gpu_pim.yaml

# 跑全部实验并对比
python run.py --all
python run.py --workload --all

# 不指定实验 = 跑第一个
python run.py
```

结果保存到 `v2/results/<实验名>.json`，控制台打印摘要：

```
  实验: 02_gpu_pim   模型: llama7b
  总延迟:     508.43 ms
  计算:      551.53 ms
  搬运:       76.37 ms
  同步等待:    1.87 ms
  瓶颈类型:   COMPUTE
  数据源决策记录: 共 963 条（已存 JSON）
```

### 方式 B：图形界面（GUI，推荐用于探索）

```bash
cd v2
python gui_app.py
```

浏览器打开 **http://127.0.0.1:5000**：

1. 选实验（或「＋ 新建」克隆模板建新实验）→ 自动加载模型 + IC 参考部署；
2. 画布上：**算子块** 拖入 **硬件块** = 决定算子放哪；
   **权重块**（紫色）拖入硬件 = 决定权重放哪，点 `✂ 切割` 可切分（ALL-GATHER）；
3. 点「🔍 校验配置」查看是否通过；点「▶ 运行仿真」出延迟/瓶颈结果；
4. 「查看算子依赖关系」弹出 DAG，含权重源节点（紫色）与数据流连线。

### 新增一个实验

```bash
# 方法1：GUI 里点「＋ 新建」（自动克隆模板）
# 方法2：手写一个入口 YAML
cp configs/experiments/01_gpu_only.yaml configs/experiments/03_my_study.yaml
# 改 name/model/files 指向自己的 hardware/mapping/placement
```

### 自定义方案速查

| 目标 | 改哪里 |
|---|---|
| 加/删硬件、改算力/容量/带宽 | `hardware.yaml` 的 `devices` |
| 改设备连接/读写带宽 | `interconnect.yaml` 的 `links` |
| 某类算子放哪个设备 | `mapping.yaml` 的 `rules`（`op` / `op_type` + `device`） |
| 权重/激活初始驻留哪 | `placement.yaml` 的 `initial` |
| 权重切分几片（GUI） | 权重块 `✂ 切割`；或 API `split=W_mlp:2` |

---

## 5. 输出说明与示例

### 5.1 输出文件

`v2/results/<实验名>.json`，顶层字段：

| 字段 | 含义 |
|---|---|
| `total_latency_ms` | 总延迟（端到端，含搬运/同步等待） |
| `bottleneck` / `bottleneck_rationale` | 瓶颈类型与理由（COMPUTE / COMMUNICATION / SYNCHRONIZATION） |
| `breakdown` | compute / transfer / sync / local_rw 分解 |
| `operator_timings` | 逐算子开始/结束/硬件/各分量耗时（514 条） |
| `event_trace` | 离散事件轨迹（COMPUTE / TRANSFER，644 条） |
| `data_source_notes` | 数据源决策记录（固定源 / [参考] / 告警） |
| `movement_bytes` | 数据搬运量统计 |
| `metadata` | 硬件清单 / 模型 / 层数 / 权重切片数 / 覆盖数 |

### 5.2 真实输出示例（02_gpu_pim，llama7b）

```json
{
  "metadata": {
    "hardware": ["gpu0", "pim0", "sram0"],
    "model": "llama7b",
    "experiment": "02_gpu_pim",
    "workload_source": "kernel_workload",
    "num_operators": 514,
    "num_layers": 32,
    "weight_shard_count": 0
  },
  "total_latency_ms": 508.432,
  "breakdown": {
    "compute_ns": 551525804,
    "transfer_ns": 76368562,
    "sync_ns": 1871008,
    "local_rw_ns": 0
  },
  "bottleneck": "COMPUTE",
  "bottleneck_rationale": "Compute=551.53ms, Transfer=76.37ms, Sync=1.87ms",
  "operator_timings": [
    {
      "op_id": "L0_ln1", "op_type": "LayerNorm", "hardware": "sram0",
      "start_ns": 0, "end_ns": 457, "duration_ns": 457,
      "compute_ns": 457, "transfer_ns": 0
    }
  ],
  "event_trace": [
    {"id": 0, "type": "COMPUTE", "start_ns": 0, "end_ns": 457,
     "operator": "L0_ln1", "resource": "sram0", "component": "COMPUTE"}
  ]
}
```

### 5.3 GUI 结果示例

运行后弹窗显示：
- 总延迟 508.43 ms · 计算 551.53 ms · 搬运 76.37 ms · 同步等待 1.87 ms · 瓶颈 COMPUTE
- 校验不通过时列出错误码（A1/A3/B1/C1/D1/W1 …）与可读信息，阻止运行。

### 5.4 校验输出示例（阻止运行的典型情况）

| 错误码 | 含义 | 示例 |
|---|---|---|
| `C1` | 硬件容量不足 | "硬件 gpu0 需要存储约 10480 MB 权重，容量只有 8192 MB" |
| `W1` | 权重切片未读齐（ALL-GATHER） | "算子 L0_ffn_gate 需要权重 L0_ffn_gw，却只连接了部分切片——缺少 L0_ffn_gw.p1" |
| `A3` | 精度不兼容 | "算子要求 FP16(等级3)，硬件最高只支持等级2" |
| `D1` | 数据不可达 | "输入数据来自硬件 pim0，与执行硬件 gpu0 之间没有互连链路" |

---

## 目录结构

```
v2/
├── run.py                    # CLI 入口（--all / --workload）
├── gui_app.py                # GUI 入口（Flask，http://127.0.0.1:5000，内嵌前端）
├── config_loader.py          # 读 YAML + 单位换算 + 默认值
├── hardware_factory.py       # 按配置造硬件 + 默认连接表
├── mapping_engine.py         # 算子→设备 + 数据源 from
├── placement_engine.py       # 数据初始驻留（可冗余多设备）
├── workload_model.py         # kernel 粒度 LLM workload 生成（含 KV 动态）
├── workload_adapter.py       # workload → 可执行算子图
├── weights.py                # WeightBlock 权重模型（归类/切分/ALL-GATHER）
├── experiment_runner.py      # 编排：配置→模型→硬件→映射→放置→运行→保存
├── constraints.py            # 配置校验器（A/B/C/D/E/W1 规则）
├── scheduler.py / engine.py / performance.py   # DES 内核 + 性能模型
├── contracts.py / hardware.py / model_lib.py   # 数据结构与内置模型
├── configs/
│   ├── hardware.yaml / interconnect.yaml       # 参考硬件与互连
│   ├── mapping_pim.yaml / placement_pim.yaml   # 参考映射与放置
│   └── experiments/NN_*.yaml                   # 实验入口（01_gpu_only, 02_gpu_pim…）
└── results/                  # 输出 JSON
```

## 与其他文档的关系

- 架构与设计：见项目根目录 `01~08` 设计文档；
- 操作手册：`09-用户使用手册.txt`；
- 版本更新：`10-版本更新记录.txt`（本版新增权重模型 / W1 / C1 / ALL-GATHER / IC 参考）。

## 许可证

本软件以 **MIT License** 发布，详见 [`LICENSE`](./LICENSE)。
可自由使用、修改、分发（含商用），需保留版权与许可证声明。
