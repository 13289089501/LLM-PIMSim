# LLM-PIMSim v3 整合说明（相对 v2 的改动）

本文件记录从 v2 → v3 的整合内容，方便你核对"以新版本为标准的统一"都改了什么。

## 1. 仿真主路径统一（移除旧 model_lib 算子图主路径）

- `experiment_runner.py`：`run_experiment` / `run_experiment_with_mapping`
  现在**委托** `run_workload_experiment`。旧的 model_lib 直接生成算子图并调度
  （`L0_q_gemm` / `L0_attn` / `L0_ffn1` 等命名）已停止作为仿真入口。
- `run.py`：不再区分 `--workload`，统一走 workload/kernel 路径。
- `model_lib.py` 仍保留，但**仅作模型维度参数表**（`/api/models`、`list_models` 查询来源），
  不参与算子图生成。

## 2. 算子 / 权重命名统一（workload 标准）

- 权重 id 统一为：`embed_weight` / `L0_qkv_weight` / `L0_o_proj_weight` /
  `L0_gate_weight` / `L0_up_weight` / `L0_down_weight` / `lm_head_weight`。
- `configs/placement.yaml` 里的样例从旧的 `L0_ffn1_weight` 改为 `L0_down_weight`，
  使其在新命名下真正生效（旧命名在新路径下静默失效）。

## 3. 精度判定统一口径

- `constraints.py` 的 `PRECISION_RANK` / `RANK_NAME` 改为与 `contracts.PrecisionLevel`
  枚举值一致：`INT4=1 < INT8=2 < FP8=3 < FP16=4 < BF16=5 < FP32=6`。
- 消除了"校验器（旧等级 FP16=3）"与"调度器（FP16=4）"两套口径打架的问题。
- `_precision_rank` 默认值、`_hw_precision` 默认等级同步修正。

## 4. GPU 能力修正（重要设计矛盾修复）

- `precisions.py`：GPU 的 `HARDWARE_CAPABILITY` 从 `categories=[LINEAR]`
  修正为 `categories=[LINEAR, NONLINEAR]`，并补充 `FP8 / INT4` 数据精度。
- 原因：GPU 只允许 LINEAR 时，任何含 LayerNorm/Softmax/激活的部署（尤其
  GPU-only 基线）都无法完成——每层都有非线性算子。旧版三个实验因此大量算子
  "永远等待"但不报错，总延迟失真。
- 修正后验证：三个实验均 **514/514 算子完成**。
- `COMPATIBILITY_MATRIX` 的 GPU 行同步更新。

## 5. 结果新增诊断字段（此前为空/缺失）

- `scheduler.collect_result` 新增：
  - `diagnostics`：`total_operators` / `finished_operators` / `unfinished_operators`
    （列出尚未完成的算子及其 type/device/state）。
  - `movement_bytes`：`total_bytes` + `per_link`（按 src→dst 汇总的搬运字节）。
- `contracts.SimulationResult` 新增 `diagnostics` 字段并纳入 `to_dict`。
- CLI `_report` 打印完成度 / 搬运字节 / 未完成算子前几项。

## 6. 实验可自定义模型与规模

- `config_loader.py`：`ExperimentConfig` 新增 `workload` 字段，读取
  `experiment.workload: {...}`。
- `_resolve_model_dims` 从 workload 段覆盖
  `num_layers/hidden/ffn_size/num_heads/head_dim/vocab/pbytes`，并读
  `input_tokens/decode_steps/batch`（不再写死 2048/128/1）。非法值报清晰 `ConfigError`。

## 7. GUI 前端升级（结果可诊断性）

- `gui_app.py` `/api/run` 返回 `movement_total_bytes` / `movement_per_link` / `diagnostics`。
- 结果弹窗新增：
  - **搬运字节 + 分设备明细**（前 8 条链路）。
  - **算子完成度**：全部完成时显示绿字 `514/514`；
    未全部完成时高亮橙字警告，列出未完成算子（含算子名/类型/目标设备），
    并提示"总延迟未包含这些算子，请检查映射"。
  - 权重切片数。
- 标题与版本号更新为 v3。

## 8. 持续发现的问题（建议下一步）

- **Decode 动态成本仍未真正进入总延迟**：`workload_adapter` 执行层使用 Prefill 固定
  `compute_flops_min`，KV 动态的 `cost_fn(kv_len)` 只在 `Workload.to_dict()` 展示，
  未参与调度。`movement_bytes` 已补齐，但 Decode 逐 token 的 KV 膨胀放大效应
  尚未计入总延迟。
- **现有结果文件在 v3 下已重新生成**，避免与旧命名/旧口径数据混淆。

## 9. 七大系统解耦重构（core/）

把原本混在一体的代码按"系统"重新组织到 `core/` 下，强调单向依赖、消除循环：

- 新增 **`core/common.py`**（公共底层）：纯枚举 + 数据结构，零业务依赖，是所有系统的根。
- **精度系统 `core/precision.py`**：PrecisionLevel / 硬件能力表 / 精度↔字节换算。
- **算子系统 `core/operator_sys.py`**：18 类算子规则 + Kernel/Workload/Builder/Adapter。
- **权重系统 `core/weight_sys.py`**：权重块建模/归类。
- **切割系统 `core/splitter.py`**：统一算子（split_kernel_dict）+ 权重（make_weight_partitions）切分。
- **校验系统 `core/validator.py`**：ConstraintChecker（原 constraints.py）。
- **输出系统 `core/exporter.py`**：结果序列化（result_to_dict）/ 落盘（save_json）/ 报告（report）；
  移除了挂在 `SimulationResult` 上的 `to_dict`（避免 common → exporter 反向依赖）。
- **核心调度器 `core/engine.py`**：PerformanceModel + 离散事件内核 + Scheduler + SimulationEngine。

顶层旧文件（contracts/precisions/constraints/scheduler/performance/engine/workload_model/
workload_adapter/weights/model_lib）改写为**兼容转发薄壳**：旧 import 仍可用，新代码用 `core.*`。

验证：`tests/test_core_systems.py` 覆盖各系统独立功能；全部三个测试文件通过；
三个实验结果与重构前一致（55.00 / 14074.57 / 55.00 ms），无回归，无循环依赖。

## 10. 新增「硬件系统」core/hardware_sys

把原本散落在 `hardware.py` / `hardware_factory.py` / `config_loader.py` 的硬件逻辑
收拢为独立系统 `core/hardware_sys.py`：

- **硬件数据结构**：`HardwareUnit`（含链路种类 `type_name`）。
- **硬件解析**：`HardwareConfig` / `parse_hardware` / `parse_interconnect` +
  出厂预设表（`DEFAULT_DEVICE_PARAMS`）+ 单位换算
  `throughput_to_flops`（自 config_loader 移入）。
- **构建工厂**：`HardwareFactory` / `build_hardware`（原 hardware_factory.py，返回
  `(devices, LinkBandwidthTable)`）。

> v3.2 起：原先的 `Link` / `Interconnect` / `DEFAULT_CONNECT_TABLE` / `LinkConfig` 已移除，
> 链路带宽收敛到新增的「链路系统」`core.link_sys`（见下方 §12）。

配套调整：
- `config_loader.py` 瘦身为装配层：`MappingRule/PlacementRule/ExperimentConfig/
  load_experiment/ExperimentIngredient` 保留；硬件/互连解析委托 `core.hardware_sys`。
- `ConfigError` 与 `load_yaml` 上移到 **core.common**（公共底层，供配置/硬件系统复用），
  由 `config_loader` 重导出保持旧接口。
- 顶层 `hardware.py` / `hardware_factory.py` 改为转发薄壳。

依赖方向：硬件系统 → core.common + core.precision（无反向依赖）。
测试：`tests/test_core_systems.py` 新增 `TestHardwareSystem`（YAML 解析 + 工厂装配 +
互连双向展开）；原硬件精度测试 `test_hardware_precision.py` 改用 core.hardware_sys 后仍全部通过。

## 11. 输出系统 & 基础功能增强（本轮）

### 输出系统
- **本地读写(local_rw)真正纳入**：`scheduler` 在算子计时累积 local_read/local_write；breakdown
  与算子级 op_timings 均带；瓶颈 MEMORY 也计入本地读写总量。
- **关键路径归因**：`core/exporter.build_critical_path(result)` 输出"时间主链"关键算子序列
  （每节点 compute/local_rw/transfer/sync 构成 + 人话解释），随 result['analysis'] 与 GUI 返回。
- **完成度校验接入**：`core/validator.validate_completion(result)` —— 未满员判无效（F1）；
  `experiment_runner` 运行后调用并把 completion_valid 写入结果 metadata；GUI 置顶红条提示
  "方案未通过完成度校验（结果无效，不可用于对比）"。

### IC 参考样例（任务 1）
- 新增 `configs/experiments/04_ic_reference*.yaml`：以集成电路专家视角做的完整参考部署
  （GPU 承担 attention 链 + FP32 非线性 + 词表；DRAM-PIM 承担 FFN GEMM；SRAM-PIM 承担
  SiLU/Residual 逐元素；ReRAM-PIM 承担 FP8 词表 LMHead；权重就近算子驻留）。
- 能过校验、514/514 全完成；搬运仅 ~56MB（vs 02 的 ~78MB，体现就近驻留收益）。
- 给用户"开箱即跑"参考，在此上改少数位置即可做对比。

### 基础功能（任务 2）
- 新建实验：`/api/experiment/create` 已可克隆模板新建独立实验（含子配置文件）。
- 结果对比：
  - 后端 `/api/compare`（读结果 JSON 返回对比行）。
  - 前端工具栏新增"⚖ 对比结果"弹层，勾选多个结果后展示对比表。
  - CLI：`python run.py --compare 01_gpu_only 02_gpu_pim 04_ic_reference`
    （读取已保存结果对比）；`--all` 对比行扩展（延迟/计算/搬运/本地读写/完成度）。
- 新建实验两种起始方式：`/api/experiment/create` 支持 `mode`——
  - `mode='blank'`：从头开始，生成一套干净的单 GPU 空模板（硬件/互连/映射/放置齐全，可跑通）。
  - `mode='ref'`：从参考实验克隆（默认 04_ic_reference），在其上修改即可做对比。
  - 前端「＋ 新建」弹层改为单选按钮"从头开始 / 从参考实验开始"。

## 12. 全系统审计后的打通/清理（本轮）

- **算子切片统一到切割系统 + 张量并行升级**：
  - 以往算子切片用"整算子均分"（engine 内），与切割系统 split_kernel_dict 脱节；现已统一：
    `experiment.yaml` 的 `workload.splits` 用 `{op, dim, parts, devices}`，由
    `_build_op_splits` 调 `core/splitter.split_kernel_dict` 按维度比例算每片 flops（张量并行）。
  - 调度 `engine._expand_op_splits` 消费 slice_flops，并把该算子的权重分片到各切片设备
    （`weight_shards` → ALL-GATHER 语义），实现"维度切分 + 权重分片式张量并行"。
  - 缺省（无 dim）退化为计算量均分，向后兼容。
- **打通 WeightPort**：`/api/weights` 返回结构化端口 `ports`（权重→算子的数据流），
  前端可消费；WeightBlock.to_port_dict 作为统一视图。
- **清理残留**：
  - 删除 `configs/experiments/03_my_pim_study*.yaml`（名不副实、与 01 GPU-only 重复的迷路配置）。
  - 删除 `model_lib.get_model` / `resolve_model_dims` 死代码。
  - 修正过时 docstring（run_workload_experiment"只展开 Prefill"、_resolve_model_dims 的 decode_steps）。
- **保留为文档化预留**（未强行打通以避免引入错误模型）：`HardwareUnit.parallelism / num_banks /
  compute_units / estimate_energy` 当前未被调度消费，作为能力预留。如需启用并行度需明确建模语义。

## 13. 前后端打通 + 前端待办（本轮）

- **6 项前端待办全部实现并通过核验**：
  1) 层折叠展示：renderLayerBar（只渲染 L0 模板，其余层折叠提示），前后端解耦不冲突。
  2) 参数范围：KV 动态算子显示 [min,max] 范围；并修复 kvHint（原死代码）→ 现在真实显示 KV 动态标记。
  3) 拖动第一层校验：quickCheckOp（容量/算力实时提示）+ 完整校验走 /api/validate。
  4) 算子依赖图 DAG：renderDepDrawer（producer→consumer 边 + 权重源列），数据来自 /api/workload。
  5) 关键路径视图：修复"后端算、前端未显示" —— 结果面板渲染 _criticalPathHtml(d.critical_path)。
  6) 初学者友好图标 + 人话提示：结果面板带 ⏱⚙🔁💾🔎✂📊 图标 + _friendlyTip。
- **算子切割真打通（消灭"界面变后端没动"假完成）**：
  - 前端切割记录 pendingSplits → /api/run 带 splits → 后端 splits_override 合并 workload.splits
    → 张量并行（维度切分+权重分片式）真执行。不再生成假 `[1]` 子算子块。
- **前端样式美化**：深色主题升级（渐变卡片/圆角/阴影/现代滚动条/渐变压按钮/hover 高亮/图层条）。
- **跨系统核验**：通读全部 txt 前端相关条目，逐条对应前端实现与后端接口；关键路径假完成点已修复复验。
- 回归：全部测试通过，实验无回归。

## 14. 校验充分性 + 只留一个参考实验（本轮）

- **校验充分性（核心修复）**：
  - 之前"校验通过却运行不完整"根因：GUI 部署参考可能把算子映射到两类非法设备——
    a) 本实验硬件里不存在的设备（`01` 只有 gpu0，画布却有 sram0/pim0）；
    b) 精度能力不支持该算子的设备（SRAM 不含 FP32，却放 LN/Softmax）。
  - 修复：
    - `_recommend_devices` 改为精度感知（FP32 算子不再推荐到 SRAM，改 GPU/CPU），并对推荐结果按硬件能力过滤。
    - `run_workload_experiment` 增加**运行前充分校验**：合并最终 compute_map 后，凡目标设备不存在于本实验硬件、或该设备无法执行该算子 → 抛 ConfigError 拒绝运行。API 层捕获后返回 blocked。
  - 效果：**校验通过即可完整运行；校验不过则不允许运行**（后端硬门禁，不受前端跳过 checkbox 影响）。
- **移除完成度/完整性检查显示**：CLI `report` 与 GUI 结果面板不再显示"算子完成度/完成度校验"警告
  （因为校验已充分，能运行即完整；未过会在运行前被拦）。
- **只留一个参考实验**：删除 `01_gpu_only` / `02_gpu_pim` / `05_1` 及其子配置、历史结果；
  仅保留 `04_ic_reference.yaml` 作为最新参考。多处默认实验指针改为 04。
- 验证：合法配置 514/514 完整；FP32→sram、不存在设备均被 PRE 拒绝；全部测试通过。

## 15. "校验过但运行拒"彻底修复（本轮）

- **用户复现**：GUI 点"校验"显示通过，点"运行"却被拒（4 个 FP32 算子放 SRAM 卡住）。
- **二次根因**（上轮只修了部分）：
  1. `/api/workload` 的 `precision` 用默认 required_precision（FP16），与后端真实 execution（LN=FP32）不一致 → 前端推荐/校验基于错误精度 → 推荐到 SRAM。
  2. 校验与运行判定**两套逻辑**（validate_config vs run 内联 preflight），可能不一致。
  3. KV_Cache 类算子 execution=None，前端结构校验误按"执行精度 INT4"判，GPU 不含 INT4 被误拦。
- **修复**：
  - `Kernel.to_dict` 返回**真实执行精度**（execution 优先，None 用 data）+ 新增 `data_precision`/`execution_precision` 字段；前端 serializeState/applyRecommendation 携带；校验器 A3 按 execution/data 区分（execution=None 查 data 能力）。
  - `_recommend_devices` 精度感知 + 能力硬过滤（含 execution=None 的数据算子用 data 能力判断）。
  - **统一判定**：新增 `experiment_runner.validate_runnable()` 与运行共用 `_construct_plan()`（同一套 workload+硬件+覆盖+充分校验）；`/api/validate` 在传 experiment 时同样跑 validate_runnable → **校验与运行结论完全一致**。
- **验证（真实 GUI 一致 payload）**：
  - 04 部署参考：校验 valid=True，运行 ok=True（514/514，7037.85ms）。
  - 非法（FP32→sram0）：校验 valid=False（A3+PRE），运行 blocked（同样被拒）。
  - 空画布：校验与运行都拦（E2），一致。
- 全量测试通过；GUI 已重启（http://127.0.0.1:5000）。

## 16. 新增「链路系统」core/link_sys（v3.2）

把链路带宽从"interconnect.yaml 按设备 id 配 read/write 带宽 + 类型对默认表"重构为
**N×N 对称带宽查找表**（按"设备种类"索引：7 种默认 + n 种用户自定义）。

- **新模块 `core/link_sys.py`**：`DEFAULT_LINK_BW_TABLE`（7×7 对称出厂表）+ `LinkBandwidthTable`
  （对称 set/get/add_type/update/to_dict + 缺省 100 GB/s）。
- **搬运延时口径**：`T(A→B) = A读延迟 + S/A读带宽 + S/链路带宽(A,B) + B写延迟 + S/B写带宽`，
  链路不再单独设延迟；读写延迟/带宽由硬件系统提供。
- **硬件系统**：`HardwareConfig` 增加 `link_type`/`links`；`HardwareUnit` 增加 `type_name`；
  `_norm_device_type` 允许自定义种类；`parse_interconnect` 返回 `LinkBandwidthTable`；
  移除 `Link`/`Interconnect`/`DEFAULT_CONNECT_TABLE`/`LinkConfig`。
- **核心调度器**：`PerformanceModel.transfer_time_ns` 改走带宽表；`SimulationEngine` 透传 `link_table`。
- **打通**：`config_loader` / `experiment_runner`（`link_table_override`）/ `core.validator`
  （D1 可达性按全连通表）/ 前端（自定义硬件弹窗"链路带宽"填写区、`/api/link_defaults`、
  `serializeState` 输出 `link_table`、`/api/run`+`/api/validate` 透传）。
- **配置**：`configs/interconnect.yaml` 改为 `link_bw_gbs` 对称表格式。
- **前端自定义硬件 = 自动新增后端设备**：自定义设备 `backId` 改为其自身 id；新增
  `core.hardware_sys.build_frontend_custom_hardware()` 把前端自定义设备解析成 `HardwareConfig`，
  `experiment_runner` 的 `custom_hardware` 参数在加载实验后将其注入硬件集（重名跳过）并把 links
  同步进链路表；`/api/run`、`/api/validate` 透传 `state.hardware`。
- 文档：`修改文档/链路系统.txt`、`core/README.md`、`README.md`、`core/__init__.py` 同步更新。

## 17. 前端连线按"数据↔算子共处"显隐（v3.2）

- **端口语义澄清**（与代码一致）：算子 `in` ← 设备读端口 `_r` = 算子从该设备读取数据；
  算子 `out` → 设备写端口 `_w` = 算子输出写入该设备（见 `core/validator.py` B4 方向校验）。
- **修复**：原先算子放进设备时整块 `display:none`，导致其 in/out 端口消失、所有数据线一律隐藏。
  现改为：算子块保持可见（端口保留作锚点），`drawConnections` 按"数据存储设备 == 算子运行设备
  （parentHW）"判定——同设备则该条数据流线隐藏，跨设备则保留；输入数据/输出数据/算子三者同设备时
  两条线都隐藏。
- 权重线（`{hw}_r → {op}_in`）同样适用该规则：权重与算子同设备则隐藏、跨设备则显示。
- 视觉显隐不影响 `connections` 数组本身（校验/运行仍收到全部连线）。
- **连线可选中/删除**：点击线选中（高亮 + 加粗），中点出现红色 × 按钮点按删除；也可按
  Delete/Backspace 删除、Esc 或点击空白处取消选中。

## 18. 前端放置改为"标签拖入设备"（v3.2）

- 取消"算子/权重整块拖入设备"的放置方式；算子/权重主体始终留在画布外用于连线。
- 每个算子块新增「运行设备」标签、每个权重块新增「存储设备」标签（可拖拽的小芯片）；
  把标签拖到硬件方块上 = 算子在此运行 / 权重在此存储，设备体内仍显示紧凑条目。
- 算子/权重切割后各生成 N 个主体，每个主体自带标签。
- 主体拖拽仅移动位置；标签拖拽负责放置（`makeTagDraggable` + 落点检测 `_hwAtPoint`）。
- **切割切片真正独立**：算子切片是 N 个独立块，各自标签可独立放到不同设备；标签放置/解除会
  实时改写该切片在张量并行规则 `pendingSplits[].devices[i]` 里对应槽位的设备（`groupOpToHW`/
  `detachOp` 按 `splitRuleIdx`/`sliceIdxInRule` 更新），运行即按每个切片的实际放置设备执行。


