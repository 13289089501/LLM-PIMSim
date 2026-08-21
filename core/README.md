# LLM-PIMSim v4 core —— 十大系统解耦架构

本目录是 v3 解耦重构后的**核心层**。按功能划分为 **10 大系统**（v4 新增
「硬件设计系统」），外加 1 个公共底层（`common.py`，不属于"系统"，是所有系统的
依赖根基）。

## 依赖方向（单向，禁止反向/循环）

```
core.common（公共底层：纯数据结构/枚举 + ConfigError/load_yaml，零业务依赖）
   │
   ▼
core.precision（精度系统）
   │
   ▼
core.hardware_sys（硬件系统）    core.operator_sys（算子系统）    core.weight_sys（权重系统）
   │    │                          │                ▲ 切割依赖算子 Kernel / 权重 WeightBlock
   │    └─► core.link_sys（链路系统，N×N 对称带宽表）│
   │                                │                │
   │                                └──────► core.splitter（切割系统，统一算子+权重切分）
   │                                            │
   ▼                                            ▼
core.design_sys（硬件设计系统，v4 新增：读 hardware_sys/link_sys 的
   │             HardwareConfig/LinkBandwidthTable 格式，输出同格式对象）
   ▼
core.validator（校验系统）  core.exporter（输出系统）
            │                     │
            ▼                     ▼
core.engine（核心调度器）← 用硬件 + 链路系统的带宽表算搬运延时
```

## 系统职责一览

| 系统 | 文件 | 职责 |
|---|---|---|
| 公共底层（非系统） | `common.py` | 纯枚举（OpState/DataType/DeviceType/...）+ 数据结构（DataObject/Operator/Event/SimulationResult）+ ConfigError/load_yaml |
| 精度系统 | `precision.py` | `PrecisionLevel` 等级、`HARDWARE_CAPABILITY` 硬件能力表、精度↔字节换算 |
| 硬件系统 | `hardware_sys.py` | 硬件 YAML 解析（HardwareConfig）、出厂预设表、单位换算、HardwareUnit、HardwareFactory/build_hardware |
| 链路系统 | `link_sys.py` | `LinkBandwidthTable`：N×N 对称链路带宽查找表（只存带宽，不含延迟）+ 7 种默认种类出厂表 |
| 算子系统 | `operator_sys.py` | 18 类算子固定规则、Kernel/Workload 建模、WorkloadBuilder、WorkloadAdapter |
| 权重系统 | `weight_sys.py` | WeightBlock/WeightPartition、权重归类、build_weight_blocks |
| 切割系统 | `splitter.py` | `split_kernel_dict`（算子沿 M/K/N 切）+ `make_weight_partitions`（权重 rows/cols 切） |
| **硬件设计系统（v4 新增）** | `design_sys.py` | 用户设计规格 → 硬件结构模型 → 参数推导（输出与自定义硬件同格式的 HardwareConfig + 链路对角线条目）；含介质/计算资源/互联/密度/部署层级预设库与独立算子效率规则 |
| 校验系统 | `validator.py` | ConstraintChecker（A/B/C/D/E/W1 规则） |
| 输出系统 | `exporter.py` | 结果 dict 序列化、JSON 落盘、控制台报告 |
| 核心调度器 | `engine.py` | PerformanceModel、离散事件内核、Scheduler、SimulationEngine |

## 顶层旧模块 = 兼容转发壳

为不破坏既有测试 / GUI / 脚本的 import 路径，以下顶层模块改成了转发薄壳，
**请勿在新代码中使用它们**，应直接 import `core.*`：

- `contracts.py`        → core.common + core.precision
- `precisions.py`       → core.precision + core.operator_sys
- `hardware.py`         → core.hardware_sys + core.link_sys
- `hardware_factory.py` → core.hardware_sys + core.link_sys
- `config_loader.py`    → 保留装配层（MappingRule/PlacementRule/Experiment/load_experiment），
                          硬件解析委托 core.hardware_sys，ConfigError/load_yaml 来自 core.common
- `constraints.py`      → core.validator
- `scheduler.py`        → core.engine
- `performance.py`      → core.engine
- `engine.py`           → core.engine
- `workload_model.py`   → core.operator_sys
- `workload_adapter.py` → core.operator_sys
- `weights.py`          → core.weight_sys（并把 core.splitter 的切割注入 build_weight_blocks）
- `model_lib.py`        → 精简为模型维度登记表（MODEL_DIMS / list_models）

## 关键约定

- 时间统一 **ns**；数据量统一 **Byte**；带宽统一 **B/s**；算力统一 **FLOPS**。
- 精度枚举 `PrecisionLevel` 的等级值即唯一比较口径（INT4=1 … FP32=6），全项目统一
  使用 core.precision.PrecisionLevel，不允许出现第二套精度等级表。
- 硬件解析 / 出厂预设 / 单位换算只存在于 core.hardware_sys（单一事实来源）。
- 链路带宽只存在于 core.link_sys（N×N 对称表，单一事实来源）；读写延迟/带宽由硬件系统提供。
  跨设备搬运延时 = 源读延迟 + S/源读带宽 + S/链路带宽 + 目标写延迟 + S/目标写带宽。
- 结果序列化只走 core.exporter（不要调用 `SimulationResult.to_dict()`，该方法已在 refactor 中移除）。
