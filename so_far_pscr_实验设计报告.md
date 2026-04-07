# SoFar-PSCR 实验设计报告

## 1. 项目目标

本项目基于 SoFar 现有可运行链路，设计一个轻量可插拔的增强方案 **PSCR（Part-aware Segmentation + Confidence Re-Verification）**，在尽量控制训练成本的前提下，提升语义朝向推理（semantic orientation reasoning）的准确性与可靠性。

本次实验的核心目标不是重训 SoFar 全模型，而是在其现有流程中加入少量关键模块，使系统能够：

1. 显式识别决定朝向的功能部件；
2. 在朝向预测中同时利用对象级与部件级信息；
3. 对低可靠样本进行二次确认，减少错误输出；
4. 最终在 Open6DOR V2 与 6-DoF SpatialBench 上取得稳定提升。

---

## 2. 当前 SoFar 基线流程

当前 SoFar 的主流程可以概括为：

**自然语言指令 → 目标物体定位与分割 → 目标点云构建 → PointSO 朝向预测 → Scene Graph 构建与后续动作推理**

具体如下：

### 2.1 指令理解
系统接收用户输入的自然语言指令，例如：
- 把杯子的把手朝右放
- 把刀刃朝向面包

然后通过现有的视觉语言模型对指令进行解析，抽取：
- 目标物体（target object）
- 朝向关系（orientation relation）
- 可能的参考物体（reference object）

这一阶段的特点是：目前更偏向物体级语义理解，还没有显式建模“功能部件（functional part）”。

### 2.2 目标物体定位与分割
SoFar 利用 Florence-2 与 SAM 完成基于文本提示的目标定位与分割：
- Florence-2：根据文本提示找到目标区域
- SAM：输出像素级 mask（掩码，即物体对应区域）

输出包括：
- object bbox（目标框）
- object mask（目标掩码）
- 局部图像区域

### 2.3 点云构建
基于 RGB-D（彩色图+深度图）和目标 mask，将二维图像中的目标区域反投影为三维点云（point cloud，三维点集合）。

输出包括：
- 目标物体点云
- 目标局部几何结构

### 2.4 PointSO 朝向预测
原始 SoFar 使用 PointSO，根据：
- 目标物体点云
- 文本朝向描述

预测物体应该满足的语义朝向。

当前这一部分本质上是：

**object-level orientation prediction（物体级朝向预测）**

即模型主要依赖整个物体的三维形状和文本语义来推断方向，而不是显式关注某个关键部件。

### 2.5 场景推理与动作生成
得到朝向后，系统继续进行：
- scene graph（场景图）构建
- 6-DoF（六自由度）推理
- 后续动作规划与执行

---

## 3. 核心问题分析

当前基线存在的关键问题是：

很多朝向指令真正依赖的不是“整个物体的外形”，而是“某个功能部件的方向”。

例如：
- 杯子的把手方向决定 handle direction
- 刀的刀刃方向决定 blade direction
- USB 插头端决定 plug-in direction
- 手电筒的发光端决定 emitting direction

如果系统只看整个物体，会出现以下问题：
1. 部件信息被整体外形淹没；
2. 小部件、遮挡部件难以被准确建模；
3. 一次性预测结果缺乏可靠性判断；
4. 当 segmentation 不稳定时，orientation prediction 很容易跟着出错。

因此，本项目的目标是：

**将 SoFar 的物体级语义朝向推理，升级为“部件感知 + 可靠性控制”的语义朝向推理流程。**

---

## 4. 新设计模块概述

本项目新增四个模块：

1. Part-aware Instruction Decomposer（部件感知指令拆解器）
2. Dual-stage Grounding（双阶段定位模块）
3. Part-conditioned Orientation Head（部件条件朝向预测头）
4. Confidence Estimation + Re-Verification（置信度评估与重验证模块）

其中：
- 模块 1 和模块 2 主要是推理流程改造，不训练；
- 模块 3 是主训练模块；
- 模块 4 中的置信度头需要训练，重验证逻辑本身不训练。

---

## 5. 新模块详细设计

### 5.1 Part-aware Instruction Decomposer

#### 功能
将原始自然语言指令拆解为结构化语义字段：
- target_object（目标物体）
- functional_part（关键功能部件）
- relation（朝向关系）
- reference_frame（参考坐标系，即方向相对谁定义）

#### 示例
输入：
- 把杯子的把手朝右放

输出：
- object = mug
- part = handle
- relation = right
- frame = observer

#### 作用
该模块的目的不是重新训练一个语言模型，而是通过现有 Qwen/VLM prompt 机制，把指令从“物体级语义”进一步细化到“部件级语义”。

#### 第一阶段实现方式
不训练，采用：
- Prompt 约束
- JSON 输出格式
- 必要时辅以少量规则修正

#### 预期产出
每个样本生成一个结构化 json，供后续模块使用。

---

### 5.2 Dual-stage Grounding

#### 功能
分两步完成视觉定位：

**Stage 1：Object Grounding（目标物体定位）**
- 在整张图中找目标物体
- 输出 object bbox、object mask、object score

**Stage 2：Part Grounding inside ROI（目标区域内的部件定位）**
- 在目标物体 ROI（局部区域）内寻找关键功能部件
- 输出 part bbox、part mask、part score

#### 为什么这样设计
如果直接在整图中找“handle”“blade”“plug”这类部件，误检会很严重。
先定位目标物体，再在局部区域中找部件，稳定性会明显更高。

#### 第一阶段实现方式
不训练，沿用 SoFar 当前的 Florence-2 + SAM 流程，只调整调用方式和 prompt。

#### 缓存要求
这一阶段必须做离线缓存，避免后续重复跑分割造成时间浪费。
建议缓存内容：
- object bbox
- object mask
- object score
- part bbox
- part mask
- part score
- ROI 坐标
- 图像尺寸

---

### 5.3 Part-conditioned Orientation Head

#### 功能
在朝向预测时，不仅使用对象级点云，还要显式使用部件级点云和几何先验，使模型从 object-level orientation 升级为 part-conditioned orientation（部件条件朝向预测）。

#### 输入
1. 对象级点云 `P_obj`
2. 部件级点云 `P_part`
3. 文本特征 `f_txt`
4. 几何先验 `g`

#### 几何先验建议包括
- object centroid（物体中心）
- part centroid（部件中心）
- `v_cp = c_part - c_obj`（部件相对物体中心方向向量）
- part PCA 主轴（主方向）
- part/object 面积比
- part 是否贴边/被截断

#### 主要思想
原始 SoFar：
- object point cloud + text → orientation

新方案：
- object point cloud + part point cloud + text + geometry prior → orientation

#### 第一阶段训练策略
- 冻结 PointSO backbone
- 冻结文本编码器
- 只训练一个轻量 fusion head（融合头）与输出头

#### 训练目标
第一版只使用两个损失：
1. orientation loss（朝向主损失）
2. geometry consistency loss（几何一致性损失）

#### 目标
先回答一个问题：

**只引入 part 分支，是否就能提升 orientation 相关指标。**

---

### 5.4 Confidence Estimation + Re-Verification

该模块分为两部分。

#### 5.4.1 Confidence Estimation（置信度评估）

##### 功能
判断当前这一次朝向预测结果是否可靠。

##### 输入特征
建议第一版使用：
- object grounding score
- part grounding score
- part/object 面积比
- part 是否触边
- orientation head top1-top2 gap（第一候选与第二候选差值）
- 预测方向与 `v_cp` 的夹角
- 预测方向与 part 主轴的夹角

##### 输出
输出一个标量 `r ∈ [0,1]`，表示当前结果的可靠程度。

##### 第一版训练方式
将其作为小型分类器或轻量 MLP 来训练：
- orientation 预测正确 → high confidence
- orientation 预测错误 → low confidence

##### 目标
回答第二个问题：

**模型能否识别自己什么时候不可靠。**

---

#### 5.4.2 Re-Verification（重验证）

##### 功能
当 `r < τ`（置信度低于阈值）时，不直接接受当前预测，而是做一次轻量级复查。

##### 第一版只做三种复查方式
1. Prompt Rephrase（提示词改写）
   - handle → mug handle
   - blade → knife cutting edge
   - plug → plug-in end

2. ROI Rescale（局部区域重缩放）
   - 对目标区域重新裁剪
   - 放大 1.2x 或缩小 0.9x

3. Candidate Rerank（候选重排序）
   - 若部件分割存在多个候选，则根据文本匹配、几何合理性和朝向一致性重新排序

##### 目标
回答第三个问题：

**低置信度样本在重验证之后，是否能得到纠正。**

---

## 6. 修改后的完整 pipeline

新的完整流程为：

**自然语言指令**  
→ **Part-aware Instruction Decomposer（指令拆解）**  
→ **Object Grounding（目标物体定位）**  
→ **Part Grounding inside ROI（部件定位）**  
→ **Object/Part Point Construction（对象级与部件级点云构建）**  
→ **Part-conditioned Orientation Head（部件条件朝向预测）**  
→ **Confidence Estimation（置信度评估）**  
→ **若低置信，则触发 Re-Verification（重验证）**  
→ **最终 orientation prediction**  
→ **接回 SoFar 原有 scene graph 与后续执行链路**

与原始 SoFar 的区别可以总结为：

原始 SoFar：
- 指令 → 物体级定位 → 物体级点云 → PointSO → 朝向

改进后：
- 指令 → 部件感知拆解 → 物体定位 → 部件定位 → 双点云构建 → 部件条件朝向预测 → 置信度评估 → 重验证 → 朝向

---

## 7. 实验总体执行顺序

整个实验建议按 8 个阶段推进，严格逐步进行，不同时改多个模块。

---

## 8. 阶段 1：Baseline 固化

### 任务
1. 跑通原版 SoFar 在 Open6DOR V2 上的推理；
2. 跑通原版 SoFar 在 6-DoF SpatialBench 上的推理；
3. 保存 baseline 结果；
4. 抽取 100 个失败样本做人工分析。

### 目标
明确基线的错误来源，至少分成：
- part 缺失类
- part 被遮挡类
- 多实例混淆类
- segmentation 错误类
- PointSO 误判类
- 语义模糊类

### 产出
- baseline_results.json
- baseline_errors.xlsx
- hard_cases 图像/样本列表

---

## 9. 阶段 2：Part-aware Parser 接入（不训练）

### 任务
1. 给现有文本理解链增加结构化字段；
2. 输出 object/part/relation/frame；
3. 在日志中记录解析结果。

### 验证方式
随机抽取 100 条指令进行人工检查。

### 阶段目标
确保后续所有模块都能拿到 `functional_part` 字段。

---

## 10. 阶段 3：Dual-stage Grounding 接入（不训练）

### 任务
1. 完成 object grounding；
2. 完成 ROI 内 part grounding；
3. 对 Open6DOR V2 做 object/part mask 离线缓存。

### 验证方式
人工检查 100 个样本的 object/part mask。

### 阶段目标
证明“先找物体，再在局部找部件”在视觉上是合理的，并为后续训练准备缓存数据。

---

## 11. 阶段 4：构建 object/part point 数据集

### 任务
1. 将 object mask + depth 反投影为 object point cloud；
2. 将 part mask + depth 反投影为 part point cloud；
3. 统一采样点数；
4. 计算几何先验；
5. 存成可训练数据集。

### 每个样本至少保存
- object points
- part points
- text prompt
- orientation label
- geometry priors
- object score
- part score

### 阶段目标
建立自己的训练数据格式，为后续训练 Part Head 做准备。

---

## 12. 阶段 5：训练 Part-conditioned Orientation Head

### 训练范围
仅训练：
- fusion head
- output head

冻结：
- PointSO backbone
- 文本编码器

### 第一版训练配置建议
- epoch：20~30
- batch size：16~32（视显存而定）
- optimizer：AdamW
- learning rate：1e-4 左右起步
- 保存最佳 checkpoint

### 阶段目标
回答最核心的问题：

**只加 part-aware orientation head，是否就能比原版 baseline 更好。**

---

## 13. 阶段 6：训练 Confidence Head

### 任务
1. 用 Part Head 跑一遍验证集/训练集；
2. 收集 orientation 预测正确与错误样本；
3. 构建 confidence 数据集；
4. 训练一个轻量 confidence head。

### 训练目标
先做最简单的二分类：
- reliable（可靠）
- unreliable（不可靠）

### 阶段目标
验证 confidence 是否有区分度，低 confidence 样本是否显著更容易出错。

---

## 14. 阶段 7：接入 Re-Verification（不训练）

### 任务
1. 对低置信样本触发复查；
2. 第一版只做：
   - prompt 改写
   - ROI rescale
   - candidate rerank

### 阶段目标
验证低置信样本在重验证后能否被纠正。

---

## 15. 阶段 8：全链评测与消融实验

### 必须保存的实验版本
1. Baseline
2. + Part Parser
3. + Dual Grounding
4. + Part Head
5. + Part Head + Confidence
6. + Part Head + Confidence + Re-Verification
7. Full（完整版本）

### 目标
形成明确的消融链条，搞清楚到底是哪一个模块在贡献性能提升。

---

## 15.1 Stage 2-8 的本地 / 服务器分工建议

以下分工基于当前可用环境：
- 本地：Windows + GTX 1660 Ti
- 服务器：用于大模型推理、批量缓存、正式训练、全量评测

原则上：
- 只要涉及 Qwen 大批量推理、Grounding/SAM 批量缓存、正式训练、全量 benchmark，优先放服务器。
- 只要是规则开发、结果清洗、数据整理、人工检查、轻量脚本开发，优先放本地。

### Stage 2：Part-aware Parser 接入

本地可做：
- 设计 parser 输出 schema（object / part / relation / frame）
- 写 prompt 模板、JSON 清洗、规则修正、日志格式
- 对已跑出的 parser 结果做人查与错误分桶

服务器执行：
- 用 Qwen / 现有 VLM 跑批量 parser 结果
- 生成正式结构化 json 与统计日志

一句话：
- 本地负责“把 parser 规则和格式写好”
- 服务器负责“真正批量跑 parser”

### Stage 3：Dual-stage Grounding 接入

本地可做：
- 写 object / part cache 的目录结构与命名规则
- 写可视化检查脚本、抽样检查脚本、失败样本整理脚本
- 人工检查 100 个 object / part mask

服务器执行：
- 批量跑 Florence / SAM / Grounding 流程
- 批量生成 object bbox、object mask、part bbox、part mask、score、ROI cache

一句话：
- 本地负责“看结果、管缓存、改脚本”
- 服务器负责“真正把全部 mask 跑出来”

### Stage 4：构建 object / part point 数据集

本地可做：
- 写数据格式定义
- 写点云采样、几何先验计算、标注字段整理脚本
- 小规模样本验证和可视化
- 对已导出的结果做清洗与统计

服务器执行：
- 批量从 mask + depth 构建 object point cloud / part point cloud
- 批量生成训练集文件

一句话：
- 本地负责“把数据集格式和处理逻辑写清楚”
- 服务器负责“把整套训练数据批量导出来”

### Stage 5：训练 Part-conditioned Orientation Head

本地可做：
- 写模型结构、loss、训练配置、日志与评估代码
- 小规模 dry-run（仅在显存允许时）
- 看训练曲线、整理结果、调参记录

服务器执行：
- 正式训练
- 保存 checkpoint
- 跑验证集与正式对比实验

一句话：
- 本地负责“写训练代码、看训练结果、改配置”
- 服务器负责“真正训练模型”

### Stage 6：训练 Confidence Head

本地可做：
- 写 confidence 特征抽取逻辑
- 整理正确 / 错误样本表
- 先做手工特征统计、小型 MLP 原型

服务器执行：
- 用完整数据集批量抽特征
- 正式训练 confidence head
- 跑验证集分桶统计

一句话：
- 本地负责“把置信度特征和小模型逻辑写好”
- 服务器负责“正式跑全量训练与评估”

### Stage 7：接入 Re-Verification

本地可做：
- 写 prompt 改写规则
- 写 ROI rescale 逻辑
- 写 candidate rerank 策略
- 设计低置信触发条件
- 分析哪些 hard case 适合重验证

服务器执行：
- 对低置信样本批量触发 re-verification
- 跑大模型复查并统计纠错率

一句话：
- 本地负责“定义复查策略”
- 服务器负责“让模型真正跑第二遍”

### Stage 8：全链评测与消融实验

本地可做：
- 汇总不同实验版本结果
- 画表、画图、整理结论
- 写消融分析、写阶段总结
- 做错误案例复盘

服务器执行：
- 跑各实验组的正式 benchmark
- 跑 Open6DOR / SpatialBench 全量评测
- 生成所有版本的原始输出

一句话：
- 本地负责“分析结果、写结论”
- 服务器负责“把所有实验版本真正跑完”

### 当前最现实的执行分工

Windows + GTX 1660 Ti 更适合：
- 规则开发
- 数据清洗
- 人工检查
- 结果分析
- 可视化
- 训练代码开发
- 小规模验证

服务器更适合：
- Qwen / Grounding / SAM 批量推理
- mask / 点云 / cache 批量生成
- Part Head / Confidence Head 正式训练
- Open6DOR / SpatialBench 全量 benchmark
- Re-Verification 全量复查

### 简化结论

- Stage 2：本地写 parser，服务器批量跑 parser
- Stage 3：本地看 mask，服务器批量生 mask
- Stage 4：本地定数据格式，服务器批量导数据
- Stage 5：本地写训练代码，服务器正式训练
- Stage 6：本地做特征整理，服务器正式训练 confidence
- Stage 7：本地写复查策略，服务器批量触发复查
- Stage 8：本地写分析总结，服务器跑正式 benchmark

---

## 16. 实验分组设计

### Group 0：Baseline
原版 SoFar

### Group 1：Part Parser Only
只加部件解析

### Group 2：Dual Grounding
加入 object+part 分割，但不改 PointSO 主体

### Group 3：Part Head
加入部件条件朝向预测头

### Group 4：Part Head + Confidence
加入置信度评估

### Group 5：Part Head + Confidence + Re-Verification
加入完整可靠性控制流程

### Group 6：Full + optional LoRA
在前面结果成立后，再尝试对 PointSO 末端做小规模 LoRA/微调

---

## 17. 每个阶段重点关注的日志

### Parser 阶段
- object 解析错误率
- part 漏抽率
- relation 错误率

### Grounding 阶段
- object 成功率
- part 成功率
- 小部件失败率
- 多实例误检率

### Part Head 阶段
- 训练 loss 是否收敛
- 验证集 orientation 指标是否上升
- 哪些类别提升最大

### Confidence 阶段
- 低 confidence 桶的错误率是否更高
- confidence 分桶后错误率分布是否合理

### Re-Verification 阶段
- 低置信样本中有多少被纠正
- 有多少被“改坏”
- 触发比例是否过高

---

## 18. 数据子集策略

第一轮实验不要直接跑全类。

建议优先选取最能体现“part 决定 orientation”的物体类别：
- mug
- knife
- bottle
- USB-like object
- flashlight
- pan
- spoon / ladle

第一轮任务优先选择 rotation 明显依赖功能部件的样本。

这样可以更快验证：

**part-aware 信息是否真的有效。**

---

## 19. GPU 资源与时间预算

以下为当前路线下的工程预算估计，不是官方声明时间，而是面向你当前“冻结 backbone、只训小头”的执行成本估计。

### 19.1 Baseline 推理
- 4090：4~8 GPU 小时
- 5090：3~6 GPU 小时

### 19.2 Dual-stage Grounding 缓存
- 4090：6~12 GPU 小时
- 5090：4~8 GPU 小时

### 19.3 Part Head 训练
- 4090：8~15 GPU 小时
- 5090：5~10 GPU 小时

### 19.4 Confidence Head 训练
- 4090：2~5 GPU 小时
- 5090：1~3 GPU 小时

### 19.5 全量消融与 Re-Verification 推理
- 4090：8~15 GPU 小时
- 5090：5~10 GPU 小时

### 第一版总预算
- 4090：约 28~55 GPU 小时
- 5090：约 18~37 GPU 小时

如果后续加入 PointSO 末端 LoRA 或更多增强模块，则预算还会继续上升。

---

## 20. 风险点与止损条件

### 风险 1：part grounding 不稳定
**止损方案：** 先只做最相关类别，不做全类。

### 风险 2：Part Head 没提升
**止损方案：** 先检查 part mask 质量，而不是立刻推翻 head 设计。

### 风险 3：confidence 没有区分能力
**止损方案：** 先用手工特征 + 小 MLP 做最简 baseline。

### 风险 4：re-check 计算开销太大
**止损方案：** 只对低 confidence 前 20% 样本触发。

---

## 21. 第一月执行计划

### 第 1 周
完成：
1. 原版 SoFar 跑通 Open6DOR/SpatialBench
2. 保存 baseline 结果
3. 抽取 100 个错误样本
4. 完成 Part-aware parser
5. 输出结构化 json
6. 做 100 条人工检查

### 第 2 周
完成：
1. 完成 Dual-stage Grounding
2. 生成 object/part mask 缓存
3. 人工检查 100 个 mask 样本
4. 完成 point builder

### 第 3 周
完成：
1. 构建 part-head 数据集
2. 跑第一版 Part Head 训练
3. 生成第一版结果

### 第 4 周
完成：
1. Confidence Head
2. Re-Verification 逻辑
3. 全量评测与消融

---

## 22. 执行原则总结

后续实验推进时，必须坚持以下原则：

1. 每次只回答一个问题，不同时改多个模块；
2. 先做最轻量、最容易验证的版本；
3. 先保证有完整消融链，再考虑更强模型；
4. 优先使用可缓存、可复现、可回放的中间结果；
5. 先把工程链路跑顺，再考虑更复杂的训练策略。

本项目的核心不是“换更强的分割器”，而是：

**把 SoFar 的语义朝向预测，从“物体级一次性推理”，升级为“部件感知 + 置信度驱动重验证”的可靠推理流程。**

---

## 23. 后续可继续扩展的内容

在本报告对应的第一版实验完成后，可进一步扩展：

1. 对 PointSO 末端做 LoRA；
2. 引入多视角一致性；
3. 引入更强的部件文本库；
4. 对 confidence 做更严格的 calibration（校准）；
5. 做更系统的 hard-case benchmark。

但这些都不应作为第一阶段主线，必须在基础版本验证有效后再考虑。
