# LLM-PIMSim v3.1 总结

LLM-PIMSim 是一个面向 **LLM 推理 + 存内计算(PIM) 异构系统** 的架构级仿真器：以集成电路视角模拟
算子/权重在 GPU、DRAM-PIM、SRAM-PIM、ReRAM-PIM 等设备上的放置与数据搬运，输出延迟、瓶颈与搬运量。

本文档记录 v3.1 相对 v3.0 的主要改进。

---

## 1. 架构：九大系统解耦

代码重构为 `core/` 下的 **9 大系统 + 1 个公共底层**，依赖方向单向、无循环：

| 系统 | 文件 | 职责 |
|---|---|---|
| （公共底层） | `common.py` | 纯枚举与数据结构 + ConfigError/load_yaml |
| 精度系统 | `precision.py` | PrecisionLevel、硬件能力表、精度↔字节换算 |
| 硬件系统 | `hardware_sys.py` | 硬件 YAML 解析、出厂预设、单位换算、HardwareUnit |
| 链路系统 | `link_sys.py` | **N×N 对称链路带宽查找表**（本次新增） |
| 算子系统 | `operator_sys.py` | 18 类算子规则、Kernel/Workload 建模 |
| 权重系统 | `weight_sys.py` | 权重块建模 |
| 切割系统 | `splitter.py` | 算子/权重沿维度切分 |
| 校验系统 | `validator.py` | 运行前合法性校验（A/B/C/D/E/W1） |
| 输出系统 | `exporter.py` | 结果序列化、JSON 落盘、关键路径归因 |
| 核心调度器 | `engine.py` | 离散事件调度、性能估算、结果收集 |

顶层旧模块（contracts/precisions/...）保留为兼容转发薄壳。

## 2. 新增「链路系统」core/link_sys

- **N×N 对称带宽查找表**：按"设备种类（type / link_type）"索引，7 种默认种类（CPU/GPU/SRAM_PIM/
  DRAM_PIM/RERAM_PIM/SRAM/DRAM）+ n 种用户自定义；带宽对称（A→B == B→A）。
- **只存带宽、无链路延迟**：跨设备搬运延时统一为
  `T(A→B) = A读延迟 + S/A读带宽 + S/链路带宽(A,B) + B写延迟 + S/B写带宽`
  （读写延迟/带宽由硬件系统提供）。
- 出厂默认表 `DEFAULT_LINK_BW_TABLE` + 缺省带宽 100 GB/s，保证任意设备种类间可达；
  `interconnect.yaml` 用 `link_bw_gbs` 声明覆盖/新增。

## 3. 硬件系统增强

- 8 类算子（GEMM/LayerNorm/Softmax/Activation/Residual/LMHead/Embedding/KVCacheUpdate）
  在各类硬件上的效率查找表补全。
- **按精度峰值算力**：`peak_by_precision`（FP32/BF16/FP16/FP8/INT8/INT4 各自算力不同）。
- 新增纯存储单元 **SRAM / DRAM**（只存不整，算力 0）。
- 允许自定义设备种类（`link_type` + `links`），参数缺失用 GPU 预设兜底。

## 4. 前端交互重构

- **标签式放置**：算子/权重主体始终留在画布外用于连线；每个算子/权重自带「运行/存储设备」标签，
  把标签拖入硬件方块 = 算子在该设备运行 / 权重在该设备存储，设备体内显示紧凑条目。
- **连线显隐规则**：数据流线仅在"数据存储设备 == 算子运行设备"时隐藏（无需搬运）；
  输入数据、输出数据、算子三者同设备时两条线都隐藏；跨设备线始终可见。
- **连线可选中/删除**：点击线高亮，中点出现 × 删除按钮，支持 Delete/Backspace/Esc。
- **自定义硬件两步弹窗**：第一步填基本参数，第二步填到已有设备种类的链路带宽；
  自定义硬件 = 自动在仿真中新增一个后端设备（`build_frontend_custom_hardware` 注入）。
- **切片真正独立**：算子切割后生成 N 个独立算子（各带标签），可分别放到不同设备；
  标签放置/解除会实时改写该切片在张量并行规则 `pendingSplits[].devices[i]` 的槽位。

## 5. 领域专家参考实验 04_ic_reference

按集成电路领域常识（非最短延时）重新编写：

- **GPU**（A100 80GB）：FP32 非线性（LayerNorm/Softmax/RoPE，唯一支持 FP32 执行的加速器）、
  注意力 GEMM（qkv/score/context/o_proj）、KV 缓存、词表 Embedding/LMHead。
- **DRAM-PIM**：FFN 三个 GEMM（gate/up/down，权重最大 → 存内计算收益来源）。
- **SRAM-PIM**（1.5PB/s）：逐元素/激活（SiLU/Residual）。
- **ReRAM-PIM**：容量小（256MB），GB 级词表/FFN 权重放不下 → 不承载算子。
- **大容量 DRAM**（64GB 纯存储）：FFN 权重（W_mlp≈51GB），PIM 单 stack 放不下 → 流式读取。
- 互连使用新链路系统 `link_bw_gbs` 对称表；校验通过、258/258 算子完整运行。

## 6. 快速开始

```bash
pip install -r requirements.txt

# CLI 运行参考实验
python run.py configs/experiments/04_ic_reference.yaml

# GUI 可视化拓扑编辑器
python gui_app.py            # http://127.0.0.1:5000

# 运行测试
python -m unittest discover -s tests
```

关键概念：`configs/experiments/*.yaml` 为实验入口（引用 hardware/interconnect/mapping/placement
四个子配置）；`configs/hardware.yaml`、`configs/interconnect.yaml` 为默认单机配置示例。
