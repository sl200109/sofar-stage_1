# SoFar-PSCR 实验设计报告

## 1. 项目目标

本项目以现有 `SoFar` 可运行链路为基础，构建一个**轻量、可插拔、以语义朝向推理为中心**的增强方案：

**PSCR（Part-aware Segmentation + Confidence Re-Verification）**

本文保持 `PSCR` 作为主方法名，不把项目整体改写为“通用 Agent 系统”。后续新增的 agent 元素，被定义为：

- 面向当前 `SoFar/Stage 5` 的**轻量决策层**
- 服务于语义朝向推理的**任务受限闭环**
- 不追求开放式对话或任意 tool use

本文的方法结构分为两层：

### 1.1 PSCR Backbone
- Part-aware Instruction Decomposer
- Dual-stage Grounding
- Part-conditioned Orientation Head

### 1.2 Lightweight Agent Layer
- Re-Verification Agent
- Module-Selection Agent

这里的“agent”不是开放式大模型 agent，而是：

> 面向语义朝向推理的受限决策闭环。

它利用当前系统已经具备的观测信号、置信度、gating 结果和 fallback 机制，对现有模块进行**自检、选择、切换与回退**。

### 1.3 主次创新排序
1. `PSCR backbone`
2. `Re-Verification Agent`
3. `Module-Selection Agent`
4. `Fast Open6DOR` 作为 benchmark-aware auxiliary branch

### 1.4 当前结论导向
- `Open6DOR`：适合做**单目标操控朝向**的 agent 闭环验证
- `SpatialBench`：适合做**题型筛选 + 模块调用策略**验证
- `Fast Open6DOR`：继续作为系统加速与高频验证支撑，不承担主科学结论

---

## 2. 当前 SoFar 基线流程

当前 `SoFar` 的主链路可以概括为：

**自然语言指令 -> 相关对象提取 -> object grounding / segmentation -> object point cloud -> orientation prediction -> scene graph -> 最终 reasoning**

### 2.1 指令理解
系统从自然语言中抽取：
- target object
- related objects
- orientation-related description

### 2.2 目标物体定位与分割
基于 `Florence-2 + SAM` 生成：
- object bbox
- object mask
- 局部图像区域

### 2.3 点云构建
基于 `RGB-D + mask` 反投影为三维点云：
- object points
- geometry priors

### 2.4 PointSO / Orientation 分支
基线主要根据：
- object-level point cloud
- text orientation description

输出语义朝向预测。

### 2.5 Scene Graph 与最终推理
将 object-level 几何和朝向信息写入 scene graph，再交由 VLM 做最终推理或问答。

---

## 3. 核心问题分析

当前系统的问题不应只概括为“part 不够”或“Open6DOR 太慢”，而应拆成三类。

### 3.1 表示问题：baseline 仍是 object-level orientation
许多指令真正依赖的是**功能部件的朝向**，而不是整个物体的外形主轴。例如：
- mug handle
- knife blade
- USB plug end
- bottle cap

因此，仅依赖 object-level point cloud 容易出现：
- 关键部件被整体几何淹没
- 小部件 / 遮挡部件难以稳定建模
- orientation 输出缺乏可解释性

### 3.2 可靠性问题：同一个 Stage 5 head 并非对所有样本都有效
当前本地实验已经证明：
- `Open6DOR-only` 有一定效果，但不是所有 orientation mode 都适合注入
- `SpatialBench-only` 能学到单域信号，但并不等于适合所有 SpatialBench 题型
- `Open6DOR` 中某些模式（如 `plug_right`、`sideways_textual`）接入后并不稳定
- `SpatialBench` 中大量 orientation 题并不是“单对象方向向量”问题

因此，系统需要自检：
- 当前证据是否可靠？
- 当前任务是否适合当前模块？
- 当前结果是否应保留、切换还是回退？

### 3.3 调度问题：不同任务不适合一条固定 pipeline
当前实验已经观察到：
- `Open6DOR` 更像单目标操控朝向生成
- `SpatialBench` 更像混合型空间 VQA
- unified shared head 在 mixed training 下阶段性失败

这说明：

> “统一模型 + 固定流程” 不是当前可行路线。

后续重点不应继续押注更大规模 mixed training，而应转向：
- task-aware routing
- module gating
- confidence-guided fallback
- prompt switching

---

## 4. 新设计模块概述

## 4.1 Layer A: PSCR Backbone

### 4.1.1 Part-aware Instruction Decomposer
将指令拆解为：
- target object
- functional part
- relation / orientation mode
- reference frame

### 4.1.2 Dual-stage Grounding
先定位 object，再在 ROI 中定位 functional part：
- object grounding
- part grounding inside ROI

### 4.1.3 Part-conditioned Orientation Head
输入：
- object points
- part points
- text prompt
- geometry priors

输出：
- 单对象 3D direction vector

## 4.2 Layer B: Agentic Decision Layer

### 4.2.1 Re-Verification Agent
围绕“当前结果是否可靠”构建自检式闭环：
- observe
- assess
- act
- verify

### 4.2.2 Module-Selection Agent
围绕“当前任务适合调用哪条模块链”构建受限策略层：
- 是否启用 Stage 5
- object-first 还是 part-first
- 是否重建 scene graph
- 是否切换 prompt
- 是否退回 baseline

## 4.3 Fast Open6DOR 辅助分支
`Fast Open6DOR` 保持 benchmark-aware auxiliary contribution 定位：
- 服务高频实验
- 降低 Open6DOR 系统验证成本
- 不承担主科学结论

---

## 5. 新模块详细设计

### 5.1 Part-aware Instruction Decomposer

#### 功能
将自然语言从 object-level 表述转成 part-aware 结构化语义：
- `target_object`
- `functional_part`
- `relation`
- `reference_frame`
- `orientation_mode`

#### 实现方式
第一版不训练，主要通过：
- prompt constraint
- structured JSON output
- 少量规则后处理

#### 输出
为后续 grounding、Stage 5、agent routing 提供统一结构化入口。

---

### 5.2 Dual-stage Grounding

#### 功能
分两步完成视觉定位：
1. `object grounding`
2. `part grounding inside ROI`

#### 设计原因
直接在全图找 `handle / blade / plug` 容易高误检；先找到目标 object，再在局部找 part，稳定性更高。

#### 第一版实现
继续复用 `Florence-2 + SAM`，以流程改造为主，不重新训练分割模型。

#### 缓存要求
离线缓存以下中间结果：
- object bbox / mask / score
- part bbox / mask / score
- ROI 信息
- point-data cache

---

### 5.3 Part-conditioned Orientation Head

#### 功能
从 object-level orientation 升级为 part-conditioned orientation prediction。

#### 输入
- `P_obj`: object points
- `P_part`: part points
- `f_txt`: text feature
- `g`: geometry priors

#### 建议几何先验
- object centroid
- part centroid
- `part_to_object_vector`
- part 主轴 / object 主轴
- part ratio
- grounding score

#### 训练目标
第一版聚焦回答：

> 仅引入 part-conditioned 分支，能否提升单域语义朝向预测？

#### 当前实验结论
- `Open6DOR-only`：初步有效
- `SpatialBench-only`：单域效果更强
- `Combined / Balanced / Filtered / Curriculum`：阶段性失败

这意味着 Stage 5 更适合作为**单域增强模块**，而不是统一跨域 head。

---

### 5.4 Confidence Estimation + Re-Verification Agent

原先的 “Confidence Estimation + Re-Verification” 在本文中被提升为一个**轻量、自检式、受限动作空间的 agent 闭环**。

#### 5.4.1 输入信号（Observe）
第一版明确绑定当前本地代码中已经存在的信号：
- `parser_confidence`
- `stage5_applied`
- `stage5_skip_reason`
- `stage5_fallback_scene_graph_used`
- `object_score`
- `part_score`
- `part_ratio`
- `orientation_mode`
- `task_type`
- `parser object count`
- `fallback_required`
- scene-graph mismatch / missing target object

#### 5.4.2 可靠性判断（Assess）
Agent 判断两个问题：
1. 当前结果是否可靠？
2. 当前任务是否适合当前模块？

#### 5.4.3 有限动作集合（Act）
第一版的动作集合限定为：
- `reuse current result`
- `switch prompt`
- `rebuild ROI / ROI rescale`
- `part-first grounding`
- `skip Stage 5 injection`
- `fallback to baseline reasoning`

这里不做开放式 tool-using，不允许任意自由搜索和任意模块组合。

#### 5.4.4 结果校验（Verify）
动作执行后，检查：
- 是否提升了输出一致性
- 是否减少错误注入
- 若无改善，则回退 baseline 路径

#### 5.4.5 第一版实现定位
第一版将其定义为**policy layer**：
- 可先用规则 / 阈值实现
- 不要求新增大规模训练
- 后续可再轻量学习化

#### 研究目标
回答的问题是：

> 系统能否识别“什么时候当前证据不可靠”，并通过小闭环避免错误输出？

---

### 5.5 Module-Selection Agent

这是一个与 Re-Verification Agent 并列的独立小节，而不是其附属规则。

#### 定义
Module-Selection Agent 是一个**task-aware module routing 的受限 agent**。

#### 可调模块集合
第一版只允许在当前本地已有模块中做选择：
- object-level grounding
- part-level grounding
- stage5 orientation head
- scene-graph injection
- orientation-aware prompt
- baseline reasoning fallback

#### 决策问题
它需要回答：
- 这题要不要启用 Stage 5？
- 该先 object-first 还是 part-first？
- 是否重建 scene graph？
- 是否切到 orientation-aware prompt？
- 是否直接退回 baseline？

#### 强调边界
这不是“大而全工具代理”，而是：

> 一个模块选择策略层。

#### 与当前代码的对应关系
- `SpatialBench gating`：它在 VQA 上的雏形
- `Open6DOR mode gating`：它在 manipulation 上的雏形

#### 研究目标
回答的问题是：

> 正确题型能否被正确路由到合适模块，错误题型能否被正确跳过或回退？

---

### 5.6 Fast Open6DOR 辅助分支

#### 定位
继续保持辅助系统分支身份，不并入 agent 主创新。

#### 作用
- task-aware routing
- minimal object set
- lightweight scene state
- dual path inference
- perception cache reuse

#### 与主线关系
它提升的是**实验效率与系统可迭代性**，不是通用方法主张。

---

## 6. 当前工程支点与 Agent 对齐

当前本地代码已经具备 agent 化所需的观测信号和部分动作，但尚未被统一包装为独立 `agent controller`。

| existing signal / action | current code location | future agent role |
| --- | --- | --- |
| `stage5_applied` / `stage5_skip_reason` | [open6dor_perception.py](D:\桌面\sofar实验同步\sofar\open6dor\open6dor_perception.py) | 观测当前 Stage 5 是否被采用 / 为什么被跳过 |
| scene graph fallback | [open6dor_perception.py](D:\桌面\sofar实验同步\sofar\open6dor\open6dor_perception.py) | 低可靠 case 的回退动作 |
| Stage 5 prediction + injection | [stage5_inference.py](D:\桌面\sofar实验同步\sofar\serve\stage5_inference.py) | 受限工具之一 |
| SpatialBench applicability classifier | [spatialbench_stage5.py](D:\桌面\sofar实验同步\sofar\serve\spatialbench_stage5.py) | Module-Selection Agent 的早期 routing policy |
| SpatialBench prompt switching | [eval_spatialbench.py](D:\桌面\sofar实验同步\sofar\spatialbench\eval_spatialbench.py), [qwen_inference.py](D:\桌面\sofar实验同步\sofar\serve\qwen_inference.py) | orientation-aware prompt 的切换动作 |
| Open6DOR mode gating | [open6dor_perception.py](D:\桌面\sofar实验同步\sofar\open6dor\open6dor_perception.py) | manipulation 场景下的模块选择雏形 |
| parser / routing hints | [qwen_inference.py](D:\桌面\sofar实验同步\sofar\serve\qwen_inference.py) | 可靠性评估与任务匹配判断 |

这意味着论文叙事必须写清楚：

> 当前代码不是“agent 尚未开始”，而是“已经具备 agent 化所需信号与动作，下一步是把它们整合成统一的轻量决策层”。

---

## 7. 修改后的完整 Pipeline

### 7.1 Open6DOR 主路径
1. parser 生成 object / part / orientation_mode
2. dual-stage grounding 构建 object / part 证据
3. stage5 head 产生单对象方向向量
4. Re-Verification Agent 判断是否可信
5. Module-Selection Agent 判断是否保留注入、切换 prompt、重建 ROI、或回退 baseline
6. 最终输出 result / target orientation

### 7.2 SpatialBench 主路径
1. parser 识别题型、对象数、orientation 语义
2. Module-Selection Agent 先判断题型是否适合当前 Stage 5
3. 适合时：注入 stage5 orientation evidence + stronger prompt
4. 不适合时：直接跳过 Stage 5，保留 baseline reasoning
5. Re-Verification Agent 只在适用子题和低可靠样本上触发更细粒度检查

---

## 8. 实验总体执行顺序

### Stage 1
Baseline 固化

### Stage 2
Part-aware Parser 接入

### Stage 3
Dual-stage Grounding 接入

### Stage 4
构建 object / part point 数据集

### Stage 5
Part-conditioned Orientation Head 训练与单域回接

### Stage 6
Confidence Estimation + Re-Verification Agent

### Stage 7
Module-Selection Agent（routing / gating / prompt switching / part-first fallback）

### Stage 8
Agent ablation 与 benchmark evaluation

---

## 9. 阶段 1：Baseline 固化

### 唯一主问题
明确基线错误来源。

### 任务
1. 跑通原版 SoFar 在 Open6DOR V2 上的推理
2. 跑通原版 SoFar 在 6-DoF SpatialBench 上的推理
3. 固化 baseline 结果
4. 抽取失败样本做人审

### 产出
- baseline results
- baseline errors
- hard cases list

---

## 10. 阶段 2：Part-aware Parser 接入（不训练）

### 唯一主问题
让系统显式知道“功能部件是什么”。

### 任务
1. 增加结构化字段：object / part / relation / frame / orientation_mode
2. 记录 parser confidence
3. 在日志中保留结构化解析结果

### 阶段目标
后续所有模块都能拿到 `functional_part` 与 `orientation_mode`。

---

## 11. 阶段 3：Dual-stage Grounding 接入（不训练）

### 唯一主问题
证明“先找 object，再找 part”是合理的。

### 任务
1. 完成 object grounding
2. 完成 ROI 内 part grounding
3. 建立 stage3 cache

### 阶段目标
为 part-conditioned head 和 agent 验证准备更稳的视觉证据。

---

## 12. 阶段 4：构建 object / part point 数据集

### 唯一主问题
把 Stage 3 视觉证据转为可训练数据。

### 任务
1. object mask -> object points
2. part mask -> part points
3. 统一采样点数
4. 计算 geometry priors
5. 存为 Stage 5 数据格式

### 样本字段
- object points
- part points
- text prompt
- orientation label
- geometry priors
- object / part score
- label confidence

### 阶段目标
形成 Stage 5 训练原料，而非一次性临时缓存。

---

## 13. 阶段 5：训练 Part-conditioned Orientation Head

### 唯一主问题
仅靠 part-conditioned head，能否在单域上学到有效 orientation signal？

### 当前结论
- `Open6DOR-only`：初步有效
- `SpatialBench-only`：单域效果更强
- `Combined / Balanced / Filtered / Curriculum`：阶段性失败

### 阶段目标
把 Stage 5 定位为：
- 单域增强模块
- 供后续 agent 调度使用的 orientation evidence provider

---

## 14. 阶段 6：Confidence Estimation + Re-Verification Agent

### 唯一主问题
系统能否识别不可靠样本，并通过受限闭环减少错误输出？

### 任务
1. 定义 observe 信号集合
2. 设计 assess 规则 / 阈值
3. 实现有限动作集合
4. 实现 verify 与 fallback

### 最少实验分组
1. baseline SoFar
2. PSCR backbone only
3. PSCR + direct Stage 5 injection
4. PSCR + Re-Verification Agent

### 关键指标
- final task accuracy
- fallback rate
- re-verification trigger rate
- corrected-error count
- over-trigger rate
- extra latency

### Open6DOR 重点
- agent 是否避免低可靠 case 的错误注入
- 是否比“无脑全注入”更稳

### SpatialBench 重点
- 是否减少无关题型误注入
- 适用子题上是否比 baseline / direct injection 更好

---

## 15. 阶段 7：Module-Selection Agent

### 唯一主问题
系统能否根据任务类型、对象状态和证据质量，选择合适的模块链路？

### 任务
1. 设计 routing policy
2. 把启用 / 跳过 / 回退决定统一化
3. 将 prompt switching、scene graph rebuild、part-first grounding 纳入受限动作集

### 最少实验分组
1. fixed pipeline
2. heuristic routing
3. routing + re-verification

### 关键指标
- module invocation distribution
- stage5 usage precision
- fallback precision
- task-type specific accuracy

### SpatialBench 必做分桶
- single-object direction
- angle / relation
- count / quantity
- route / navigation
- unsupported semantics

### 阶段目标
证明“正确题型正确调用，错误题型正确跳过”。

---

## 16. 阶段 8：全链评测与消融实验

### 唯一主问题
在 agent 化决策层加入后，最终系统是否比 backbone-only 更稳、更可控？

### 必要消融
- backbone only
- + direct Stage 5 injection
- + Re-Verification Agent
- + Module-Selection Agent

### 重点结论
文档最终必须能明确写出：
1. `Open6DOR` 与 `SpatialBench` 不是同一 orientation problem
2. unified shared head 当前阶段性失败，后续重点转向 agentic control
3. agent 元素是“受限决策闭环”，不是通用大模型 agent

---

## 17. 实验分组设计

### 17.1 PSCR 主线实验
- baseline SoFar
- PSCR backbone
- PSCR backbone + Stage 5 direct injection
- PSCR backbone + Re-Verification Agent
- PSCR backbone + Module-Selection Agent

### 17.2 Fast Open6DOR 辅助实验
- heavy path
- fast path
- fast path + cache
- fast path + Stage 5 single-domain injection
- fast path + mode gating

---

## 18. 评测指标与论文可用结果

### 18.1 Open6DOR
- final task success / orientation-related success
- stage5_applied rate
- stage5_skip_reason distribution
- fallback scene graph usage
- latency

### 18.2 SpatialBench
- overall accuracy
- position accuracy
- orientation accuracy
- task-type bucket accuracy
- Stage 5 apply / skip distribution

### 18.3 Agent-specific 指标
- corrected-error count
- over-trigger rate
- module usage precision
- fallback precision

---

## 19. 数据子集与验证策略

### Open6DOR
- 不默认全模式启用 Stage 5
- 优先在更适合的 orientation mode 子集上验证
- 当前已有信号表明：`upright` 类比 `plug_right / sideways_textual` 更稳定

### SpatialBench
- 不再把整个 orientation 集合视为统一任务
- 先按题型筛选
- 优先验证 `single_object_direction` 子题

---

## 20. GPU 资源与时间预算

后续资源优先级调整为：
1. Open6DOR 单域 agent 闭环验证
2. SpatialBench 子题 routing / gating 验证
3. 最后才考虑更复杂的统一模型或 dataset-specific conditioning

这意味着：
- 不再优先投入到新的 mixed training 变体
- 优先投入到 agent policy 的轻量验证与小规模 ablation

---

## 21. 风险、边界与止损条件

### 风险 1
Stage 5 学到的是几何主轴，不是任务目标朝向

### 风险 2
SpatialBench orientation 题型异质性过强，不能共享同一解释空间

### 风险 3
agent 叙事如果写得过大，会被误读为“通用 agent 框架”

### 止损条件
- 若某模块在适用子集上都无正向信号，则只保留为负结果
- 若某 routing / re-verification 只能增加延迟不能减少错误，则停止扩大实验规模

---

## 22. 第一月执行计划

### 第 1 周
- 完成报告改写
- 固化当前单域训练与负结果结论
- 明确 agent 主线叙事

### 第 2 周
- 实现 Re-Verification Agent 的规则版
- 对 Open6DOR 做低可靠样本验证

### 第 3 周
- 实现 Module-Selection Agent 的 heuristic 版
- 对 SpatialBench 做题型路由验证

### 第 4 周
- 做 agent ablation
- 输出论文可用表格与图示

---

## 23. 执行原则总结

1. `PSCR` 仍是主方法名，agent 是后续创新层，不整体 rename。
2. 不再把 mixed training 当作主线。
3. agent 元素只做轻量、受限、为当前 SoFar/Stage 5 服务的闭环。
4. `Open6DOR` 与 `SpatialBench` 分开对待，不再硬套统一 orientation 假设。
5. 报告叙事必须严格对齐当前本地代码，避免“方法先写了、代码没支撑”的印象。

---

## 24. 后续可继续扩展的内容

如果主线成立，后续可再扩展：
- learned re-verification policy
- dataset-specific conditioning
- shared trunk + dataset-specific small heads
- pairwise angle head / hinge head / relation-specific heads for SpatialBench

但这些都不应抢在当前轻量 agent 主线之前。

---

## 25. 2026-04-22 工程落地状态

当前工程已经不再停留在“agent 叙事”，而是进入了 Stage 6/7/8 的本地实现完成、服务器 smoke 已落地的阶段。

### 已完成
- `semantic_orientation_agent.py`
  - 已统一 `AgentDecision` 风格 schema
  - 已补齐 `SpatialBench` 决策 / verify
  - 已补齐 `Open6DOR` 决策 / verify
  - 已补齐 `AutoModeSelection Agent` 的 routing 入口
- `SpatialBench`
  - 已支持 `agent-mode off|dataset|auto`
  - 已支持 centralized agent trace 输出
  - 已完成 `rule_v2`：`axis_direction / reference_alignment / camera_alignment`
  - 已完成 `stage5_applicable14` targeted pilot
- `Open6DOR`
  - 已支持 `direct_allow / conditional_verify / baseline_only` 三档执行带
  - 已支持 `shadow_stage5_for_debug`
  - 已支持 verify 后拒绝 Stage 5 并回退 parser/baseline orientation
  - 已完成 `dataset / auto` 两组 `20-case smoke`
- `Stage 8`
  - 已新增独立评测 runner：`sofar/analysis/stage8_agent_eval.py`
  - 已新增独立消融 runner：`sofar/analysis/stage8_agent_ablation.py`
  - 已完成 `baseline / direct_stage5 / agent` 三组小规模 smoke

### 当前代码与本文设计的一致性
- `SpatialBench` 已落地为“题型路由优先”的 VQA agent
- `Open6DOR` 已落地为“单目标操控朝向闭环”的 manipulation agent
- `AutoModeSelection Agent` 已实现为上层路由接口，而不是额外重写一套新 pipeline

### 当前阶段性结论
- `SpatialBench`：
  - 顺序前 `20` 条 `Stage 8` 结果主要说明“前 20 条并非适合 Stage 5 的题型”，不支持把 SpatialBench 直接推进到全数据正式训练
  - `stage5_applicable14` targeted pilot 已证明：在适用子题上，agent 路由、prompt 分流与 verification 是可工作的
- `Open6DOR`：
  - 当前 controller 已经稳定具备 `direct / reject / fallback / shadow` 行为
  - 因而下一步重点转为扩大 Open6DOR 有效训练数据并启动下一轮正式训练

### 仍需谨慎的点
- 当前 `Stage 8` 对 `Open6DOR` 的 `correct` 仍按 `status == success` 统计，属于控制器稳定性指标，而不是最终的 orientation ground-truth 指标
- 因此现阶段可以据此决定“开始正式训练”，但还不能据此直接宣称最终 orientation 性能提升

### 当前工程文档分工
为避免实验设计报告继续膨胀为交接手册，当前工程文档分工固定为：
- `当前代码地图与交接总表.md`：当前阶段唯一总入口，负责主线代码地图、创新点映射、上传/回传规则和当前服务器操作步骤
- `baseline_stage5_tracker.md`：保留时间线、阶段性结论与风险判断
- `baseline_stage5_todolist.json`：只保留任务状态机，不写长篇解释
- `交接操作.txt`：只保留“当前该做什么”的服务器执行手册

因此，本文档后续只继续承担两类内容：
- 方法叙事与实验设计主线
- 工程落地状态的阶段性补记

本文档不再重复记录命令、上传/回传文件夹细节与完整代码地图。
