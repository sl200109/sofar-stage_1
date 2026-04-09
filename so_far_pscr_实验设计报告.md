# SoFar-PSCR 实验设计报告

## 1. 项目目标

本项目基于 SoFar 现有可运行链路，设计一个轻量可插拔的增强方案 **PSCR（Part-aware Segmentation + Confidence Re-Verification）**，在尽量控制训练成本的前提下，提升语义朝向推理（semantic orientation reasoning）的准确性与可靠性。

> **本文主创新为 PSCR；Fast Open6DOR 仅作为面向 Open6DOR 基准的辅助系统优化分支，用于降低实验成本、支持高频迭代，不作为通用 6-DoF 推理主结论的核心依据。**

报告中的主次关系固定为：
- `PSCR`：约 `65%–75%` 权重，负责主要科学问题、主要训练模块、主要性能主张；
- `Fast Open6DOR`：约 `25%–35%` 权重，负责工程/系统优化贡献，强调高迭代效率，但明确是 `benchmark-aware auxiliary contribution`。

本次实验的核心目标不是重训 SoFar 全模型，而是在其现有流程中加入少量关键模块，使系统能够：

1. 显式识别决定朝向的功能部件；
2. 在朝向预测中同时利用对象级与部件级信息；
3. 对低可靠样本进行二次确认，减少错误输出；
4. 在不牺牲研究严谨性的前提下，显著降低 Open6DOR 上的实验迭代成本；
5. 最终在 Open6DOR V2 与 6-DoF SpatialBench 上形成一条可对照、可消融、可写论文的实验链。

**表 1：PSCR 主线 vs Fast Open6DOR 辅线对照表**

| 对比维度 | PSCR 主线 | Fast Open6DOR 辅线 |
| --- | --- | --- |
| 定位 | 主创新 | 辅助创新 |
| 核心目标 | 提升通用语义朝向推理准确率与可靠性 | 降低 Open6DOR 实验成本与时延 |
| 主要模块 | parser、part head、confidence、re-verification | task-aware routing、minimal object set、asymmetric perception、lightweight scene state、fast/heavy dual path |
| 是否训练 | 是，Stage 5/6 为主训练阶段 | 否，主要是系统优化 |
| 主风险 | part grounding / confidence 无收益 | 容易滑成 benchmark-specific shortcut |

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

当前基线存在两类核心问题。

### 3.1 主科学问题：部件决定朝向，但基线仍以物体级推理为主

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

因此，**PSCR 主线要解决的问题**是：

**如何将 SoFar 的物体级语义朝向推理，升级为“部件感知 + 可靠性控制”的语义朝向推理流程。**

### 3.2 辅助系统问题：Open6DOR 原链路实验成本过高

现有日志已经证明，原始 SoFar 链路在 Open6DOR 上具有明显的工程成本问题：
- 单例推理常接近分钟级；
- 全量 Open6DOR case 数量大，无法支持高频迭代；
- 在没有系统优化的情况下，后续 parser / confidence / re-check 的实验成本会被放大。

因此，**Fast Open6DOR 辅线要解决的问题**是：

**如何在不替代 PSCR 主结论的前提下，把 Open6DOR 上的实验验证从高成本重链路，改造成可迭代的 benchmark-aware fast path。**

---

## 4. 新设计模块概述

本项目总体上包含两层设计：

### 4.1 PSCR 主线模块

PSCR 主线新增四个模块：

1. Part-aware Instruction Decomposer（部件感知指令拆解器）
2. Dual-stage Grounding（双阶段定位模块）
3. Part-conditioned Orientation Head（部件条件朝向预测头）
4. Confidence Estimation + Re-Verification（置信度评估与重验证模块）

其中：
- 模块 1 和模块 2 主要是推理流程改造，不训练；
- 模块 3 是主训练模块；
- 模块 4 中的置信度头需要训练，重验证逻辑本身不训练。

### 4.2 Fast Open6DOR 辅助分支

Fast Open6DOR 不是第二主方法，而是服务于 Open6DOR 高效验证的一条辅助系统分支。它的核心目标不是证明“更强的通用 6-DoF 推理能力”，而是：

1. 减少 Open6DOR 上的高成本重链路调用；
2. 支持 pilot 10 case / 100 case 的高频实验；
3. 为 PSCR 主线提供更可承受的系统验证基础设施。

它计划包含的优化点包括：
- task-aware routing
- minimal object set
- asymmetric perception
- lightweight scene state
- fast/heavy dual path

> 它的价值在于提高实验效率；它的风险在于容易被误解为 benchmark-specific shortcut，因此必须在报告中明确写出允许使用的信息边界。

### 4.3 Fast Open6DOR 信息边界

**表 2：Fast Open6DOR 信息边界表**

| 信息来源 | 是否允许 | 使用方式 | 风险说明 |
| --- | --- | --- | --- |
| instruction | 允许 | 主输入，负责关系、目标物体、方向模式解析 | 无额外 benchmark 偏置 |
| RGB-D / image | 允许 | 主输入，负责感知与几何状态构建 | 无额外 benchmark 偏置 |
| task_config | 允许，但受限 | 仅用于 fast path 的 task-aware routing / schema extraction | 会削弱通用性主张，必须单独标注 |
| directory / task id | 默认不允许 | 不作为默认主输入 | 容易滑成 benchmark-specific shortcut |

Fast Open6DOR 允许使用 Open6DOR 提供的任务配置元信息，以提升推理效率与链路稳定性；但这会削弱其作为通用 6-DoF 推理方法的主张。因此本文将其严格定位为 `benchmark-aware auxiliary contribution`，而非 PSCR 主线结论的主要依据。

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

### 5.5 Fast Open6DOR 辅助分支

#### 定位
Fast Open6DOR 只服务于 Open6DOR 的高频迭代与系统实验，不替代 PSCR 主线，也不承担“通用 6-DoF 推理提升”的主结论。

#### 核心组成
1. task-aware routing  
2. minimal object set  
3. asymmetric perception（目标物体重处理，参考物体轻处理）  
4. lightweight scene state  
5. fast/heavy dual path  

#### 设计原则
- 优先减少大模型调用和重感知链长度；
- 优先服务 `pilot 10 case / 100 case` 的快速验证；
- 当 fast path 不稳定时，必须允许 fallback 到原重链路；
- 不依赖 directory / task id 作为默认主输入。

#### 研究价值
这条分支的研究价值不在于“更通用”，而在于：
- 非对称感知是否能显著降低系统成本；
- 轻量 scene state 是否能替代重 scene graph 完成 Open6DOR 基本任务；
- confidence/fallback 是否能将快链路控制在可接受误差内。

#### 风险
如果规则化过头，它会从“辅助系统优化”滑向“benchmark-specific shortcut”。因此必须持续用边界表、风险表和 heavy-path 对照来约束它。

---

## 6. 修改后的完整 pipeline

新的完整流程分成主路径与辅助路径：

### 6.1 PSCR 主路径

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

### 6.2 Fast Open6DOR 辅助路径

**instruction / RGB-D / 受限 task_config**  
→ **task-aware routing**  
→ **minimal object set**  
→ **asymmetric perception**  
→ **lightweight scene state**  
→ **fast reasoning**  
→ **若低置信或失败，则 fallback 到 heavy path**

与原始 SoFar 的区别可以总结为：

原始 SoFar：
- 指令 → 物体级定位 → 物体级点云 → PointSO → 朝向

PSCR 主线：
- 指令 → 部件感知拆解 → 物体定位 → 部件定位 → 双点云构建 → 部件条件朝向预测 → 置信度评估 → 重验证 → 朝向

Fast Open6DOR 辅线：
- instruction / task-aware routing → 最小对象集合 → 非对称感知 → 轻量状态 → 快链路推理 → 失败时回退重链路

---

## 7. 实验总体执行顺序

整个实验仍按 8 个阶段推进，但每个阶段只回答一个主问题，不同时改多个主问题。

**表 3：修订后的 Stage 1–8 路线图**

| 阶段 | 唯一主问题 | 主产出 | Fast 是否挂靠 |
| --- | --- | --- | --- |
| Stage 1 | 当前基线的真实起点是什么？ | baseline 结果、错误样本、运行日志 | 是，提供 Open6DOR pilot 基线 |
| Stage 2 | 部件级指令拆解是否稳定可用？ | parser json、人工检查记录 | 是，fast parser 可挂靠 |
| Stage 3 | 先找物体、再找部件是否有效？ | object/part mask cache | 是，minimal object set / asymmetric perception 可挂靠 |
| Stage 4 | 能否构建稳定可训练的数据表示？ | object/part point dataset | 否，Fast 只复用缓存思路 |
| Stage 5 | part-aware orientation 是否真的有效？ | part head checkpoint、验证结果 | 否，保持主训练问题单纯 |
| Stage 6 | confidence 是否真的能识别不可靠样本？ | confidence dataset、confidence head | 是，仅作为 fast fallback trigger 的附带用途 |
| Stage 7 | 低置信样本通过复查/回退是否能被纠正？ | re-verification / fallback 策略与结果 | 是，heavy-path fallback 可挂靠 |
| Stage 8 | 各模块到底贡献了什么？ | 消融结果、主表、结论 | 是，fast 系统实验单列汇总 |

---

## 8. 阶段 1：Baseline 固化

### 唯一主问题
当前基线在本地/服务器环境中的真实起点是什么？

### 任务
1. 跑通原版 SoFar 在 6-DoF SpatialBench 上的全量推理；
2. 跑通原版 SoFar 在 Open6DOR V2 上的 pilot 推理；
3. 保存 baseline 结果、运行日志与错误样本；
4. 抽取 100 个失败样本做人工分析。

### 目标
明确基线的真实错误来源，至少分成：
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
- SpatialBench 全量日志
- Open6DOR pilot 日志

---

## 9. 阶段 2：Part-aware Parser 接入（不训练）

### 唯一主问题
部件级指令拆解是否稳定可用？

### 任务
1. 给现有文本理解链增加结构化字段；
2. 输出 object / part / relation / frame；
3. 在日志中记录解析结果；
4. 为 Fast Open6DOR 提供受限的 task-aware routing / schema extraction。

### 验证方式
随机抽取 100 条指令进行人工检查。

### 阶段目标
确保后续所有主线模块都能拿到 `functional_part` 字段，并确保 fast parser 只是辅助解析分支，不取代主线 parser。

---

## 10. 阶段 3：Dual-stage Grounding 接入（不训练）

### 唯一主问题
先找物体、再找部件是否有效？

### 任务
1. 完成 object grounding；
2. 完成 ROI 内 part grounding；
3. 对 Open6DOR V2 做 object/part mask 离线缓存；
4. 在 Fast Open6DOR 中尝试 minimal object set 与 asymmetric perception。

### 验证方式
人工检查 100 个样本的 object/part mask，并对 fast path 的 reference-object 轻处理策略做小样本核查。

### 阶段目标
证明“先找物体，再在局部找部件”在视觉上是合理的，并验证目标物体重处理、参考物体轻处理是否能节省成本而不明显伤害稳定性。

---

## 11. 阶段 4：构建 object/part point 数据集

### 唯一主问题
能否构建稳定可训练的数据表示？

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

### 唯一主问题
part-aware orientation 是否真的有效？

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

### 唯一主问题
confidence 是否真的能识别不可靠样本？

### 任务
1. 用 Part Head 跑一遍验证集/训练集；
2. 收集 orientation 预测正确与错误样本；
3. 构建 confidence 数据集；
4. 训练一个轻量 confidence head。

### 训练目标
先做最简单的二分类：
- reliable（可靠）
- unreliable（不可靠）

### 附带作用
如果 confidence 确实有效，它可以进一步作为 Fast Open6DOR 中 fast/heavy dual path 的 fallback trigger，但这不是本阶段的主问题。

### 阶段目标
验证 confidence 是否有区分度，低 confidence 样本是否显著更容易出错。

---

## 14. 阶段 7：接入 Re-Verification（不训练）

### 唯一主问题
低置信样本通过复查/回退是否能被纠正？

### 任务
1. 对低置信样本触发复查；
2. 第一版只做：
   - prompt 改写
   - ROI rescale
   - candidate rerank
3. 对 Fast Open6DOR 引入 heavy-path fallback。

### 阶段目标
验证低置信样本在重验证后能否被纠正，并观察 fast path 在 fallback 之后的稳定性是否改善。

---

## 15. 阶段 8：全链评测与消融实验

### 唯一主问题
各模块到底贡献了什么？

### 目标
形成明确的消融链条，搞清楚：
1. PSCR 各模块对主精度指标的贡献；
2. Fast Open6DOR 各优化点对时延与稳定性的贡献；
3. fallback 是否能有效控制快链路误差。

---

## 15.1 Stage 2-8 的本地 / 服务器分工建议

以下分工基于当前可用环境：
- 本地：Windows + GTX 1660 Ti
- 服务器：用于大模型推理、批量缓存、正式训练、正式评测

原则上：
- 只要涉及 Qwen 大批量推理、Grounding/SAM 批量缓存、正式训练、正式 benchmark，优先放服务器；
- 只要是规则开发、结果清洗、数据整理、人工检查、轻量脚本开发，优先放本地。

### Stage 2：Part-aware Parser 接入

本地可做：
- 设计 parser 输出 schema（object / part / relation / frame）
- 写 prompt 模板、JSON 清洗、规则修正、日志格式
- 对已跑出的 parser 结果做人查与错误分桶

服务器执行：
- 用 Qwen / 现有 VLM 跑批量 parser 结果
- 生成正式结构化 json 与统计日志

### Stage 3：Dual-stage Grounding 接入

本地可做：
- 写 object / part cache 的目录结构与命名规则
- 写可视化检查脚本、抽样检查脚本、失败样本整理脚本
- 人工检查 100 个 object / part mask

服务器执行：
- 批量跑 Florence / SAM / Grounding 流程
- 批量生成 object bbox、object mask、part bbox、part mask、score、ROI cache

### Stage 4：构建 object / part point 数据集

本地可做：
- 写数据格式定义
- 写点云采样、几何先验计算、标注字段整理脚本
- 小规模样本验证和可视化

服务器执行：
- 批量从 mask + depth 构建 object point cloud / part point cloud
- 批量生成训练集文件

### Stage 5–7：训练与复查

本地可做：
- 写 Part Head / Confidence Head / Re-Verification 的逻辑
- 做小规模 dry-run
- 看训练曲线、整理结果、调参记录

服务器执行：
- 正式训练
- 保存 checkpoint
- 跑验证集与复查策略

### Stage 8：全链评测与消融

本地可做：
- 汇总不同实验版本结果
- 画表、画图、整理结论
- 写消融分析、写阶段总结

服务器执行：
- 跑各实验组的正式 benchmark
- 跑 SpatialBench 全量与 Open6DOR 受控验证
- 生成所有版本的原始输出

---

## 16. 实验分组设计

### 16.1 PSCR 主线实验分组

**表 4：PSCR 主线实验分组表**

| Group | 新增模块 | 目的 | 对照对象 |
| --- | --- | --- | --- |
| Group 0 | Baseline | 建立真实起点 | 无 |
| Group 1 | + Part Parser | 验证部件级指令拆解是否有帮助 | Group 0 |
| Group 2 | + Dual Grounding | 验证 object+part 定位是否有帮助 | Group 1 |
| Group 3 | + Part Head | 验证 part-conditioned orientation 是否有帮助 | Group 2 |
| Group 4 | + Confidence | 验证 confidence 是否有区分力 | Group 3 |
| Group 5 | + Re-Verification | 验证低置信样本复查是否有用 | Group 4 |
| Group 6 | Full + optional LoRA | 作为主线最终增强版本 | Group 5 |

### 16.2 Fast Open6DOR 系统实验分组

**表 5：Fast Open6DOR 系统实验分组表**

| 版本 | 加入优化 | 目标 | 主要指标 |
| --- | --- | --- | --- |
| Fast-0 | Heavy baseline | 建立 Open6DOR 原链路时延基线 | success / avg latency |
| Fast-1 | + task-aware routing | 减少解析成本 | success / avg latency |
| Fast-2 | + minimal object set | 减少感知对象数量 | success / median latency |
| Fast-3 | + asymmetric perception | 降低 reference object 感知成本 | success / p90 latency |
| Fast-4 | + lightweight scene state | 缩短最终推理链 | success / avg latency |
| Fast-5 | + fallback | 在快链路下控制错误 | success / error / fallback trigger rate |

---

## 17. 评测指标与论文可用结果

### 17.1 评测指标表

**表 6：评测指标表**

| Benchmark / 分支 | 主指标 | 辅指标 | 系统指标 |
| --- | --- | --- | --- |
| SpatialBench | total accuracy | position rel/abs, orientation rel/abs | runtime errors |
| Open6DOR PSCR 主线 | position / rotation / 6-DoF 指标 | hard-case breakdown | elapsed time |
| Fast Open6DOR | success rate | error count | avg latency, median latency, p90 latency, fallback trigger rate |

### 17.2 最终论文可用结果表规划

**表 7：最终论文可用结果表规划**

| 表名 | 用于哪一节 | 预计填什么结果 | 归属 |
| --- | --- | --- | --- |
| Baseline vs PSCR 总表 | 主实验 | SpatialBench / Open6DOR 主指标提升 | PSCR |
| PSCR 模块消融表 | 消融实验 | Parser / Grounding / Part Head / Confidence / Re-Verification | PSCR |
| Hard-case 分桶表 | 误差分析 | 错误类型、遮挡、多实例、部件缺失 | PSCR |
| Fast Open6DOR 时延对比表 | 系统实验 | Heavy vs Fast-0~5 时延与成功率 | Fast |
| Fast Open6DOR fallback 表 | 系统实验 | fallback 前后 success / error 变化 | Fast |
| 资源成本表 | 实验设置 | 服务器成本、本地成本、是否必须全量 | 共享 |

---

## 18. 数据子集与验证策略

本项目默认采用**双层验证**：

- **主验证集**：`SpatialBench` 全量  
  用于验证 PSCR 主线的准确率、错误模式与稳定性
- **辅助验证集**：`Open6DOR 10-case / 100-case pilot`  
  用于验证 Fast Open6DOR 和系统优化效果
- 只有在必要阶段，才追加更大规模 Open6DOR 评测

第一轮实验不要直接跑全类。建议优先选取最能体现“part 决定 orientation”的物体类别：
- mug
- knife
- bottle
- USB-like object
- flashlight
- pan
- spoon / ladle

这样可以更快验证：

**part-aware 信息是否真的有效。**

---

## 19. GPU 资源与时间预算

以下预算强调“实验内容充足，但总时间尽可能少”，因此默认将 `SpatialBench 全量` 作为主回归，将 `Open6DOR pilot` 作为常规验证入口。

**表 8：资源与时间预算表**

| 模块 / 环节 | 服务器成本 | 本地成本 | 是否必须全量 |
| --- | --- | --- | --- |
| SpatialBench baseline / 回归 | 中 | 低 | 是 |
| Open6DOR 10-case / 100-case pilot | 低到中 | 低 | 否 |
| Open6DOR 全量 heavy baseline | 高 | 无 | 否，只有最终必要时追加 |
| Dual-stage Grounding 缓存 | 中到高 | 低 | 否，优先做子集 |
| Part Head 训练 | 中到高 | 低 | 是 |
| Confidence Head 训练 | 低到中 | 低 | 是 |
| Re-Verification / fallback 实验 | 中 | 低 | 否，优先 pilot |

简化判断：
- PSCR 主线：值得投入训练与全量主回归成本；
- Fast Open6DOR：优先做低成本 pilot，不作为每轮全量必跑项。

---

## 20. 风险、边界与止损条件

**表 9：风险-边界-止损表**

| 风险 | 触发条件 | 止损方案 | 是否影响主结论 |
| --- | --- | --- | --- |
| part grounding 不稳定 | 小部件定位失败率高 | 先只做最相关类别，不做全类 | 是 |
| Part Head 没提升 | 主指标无提升或下降 | 先检查 part mask 质量，不立刻推翻 head 设计 | 是 |
| confidence 没有区分能力 | 低 confidence 桶不比高 confidence 桶更差 | 先用手工特征 + 小 MLP 做最简 baseline | 是 |
| re-check 计算开销太大 | 复查触发比例过高 | 只对低 confidence 前 20% 样本触发 | 否 |
| Fast Open6DOR 越界成 shortcut | 过度依赖 task metadata 或目录信息 | 用边界表约束输入，并保留 heavy baseline 对照 | 否，但影响辅助结论解释 |
| Fast path 速度快但精度不稳 | success / error 波动大 | 强制引入 heavy-path fallback | 否，但影响辅助结论 |

---

## 21. 第一月执行计划

### 第 1 周
完成：
1. 原版 SoFar 跑通 SpatialBench 全量
2. 跑通 Open6DOR pilot
3. 保存 baseline 结果与运行日志
4. 抽取 100 个错误样本
5. 完成 Part-aware parser
6. 做 100 条人工检查

### 第 2 周
完成：
1. 完成 Dual-stage Grounding
2. 生成 object / part mask 缓存
3. 人工检查 100 个 mask 样本
4. 完成 point builder
5. 启动 Fast Open6DOR 最小对象集合与轻量感知实验

### 第 3 周
完成：
1. 构建 part-head 数据集
2. 跑第一版 Part Head 训练
3. 生成第一版主线结果
4. 跑 Fast Open6DOR pilot 时延对照

### 第 4 周
完成：
1. Confidence Head
2. Re-Verification 逻辑
3. fast/heavy fallback 验证
4. 主线与辅线实验结果汇总

---

## 22. 执行原则总结

后续实验推进时，必须坚持以下原则：

1. 每个阶段只回答一个主问题，不同时改多个主问题；
2. 先做最轻量、最容易验证的版本；
3. PSCR 是主创新，Fast Open6DOR 是辅助创新；
4. Fast Open6DOR 可以提高效率，但不能抢主结论；
5. 优先使用可缓存、可复现、可回放的中间结果；
6. Open6DOR 默认使用 pilot 验证，避免每轮全量高成本；
7. 先把工程链路跑顺，再考虑更复杂的训练策略。

本项目的核心不是“换更强的分割器”，而是：

**把 SoFar 的语义朝向预测，从“物体级一次性推理”，升级为“部件感知 + 置信度驱动重验证”的可靠推理流程；同时以 Fast Open6DOR 作为受控的辅助系统优化分支，降低 Open6DOR 上的实验成本。**

---

## 23. 后续可继续扩展的内容

在本报告对应的第一版实验完成后，可进一步扩展：

1. 对 PointSO 末端做 LoRA；
2. 引入多视角一致性；
3. 引入更强的部件文本库；
4. 对 confidence 做更严格的 calibration（校准）；
5. 做更系统的 hard-case benchmark；
6. 对 Fast Open6DOR 做更严格的泛化边界检验。

但这些都不应作为当前阶段主线，必须在基础版本验证有效后再考虑。
