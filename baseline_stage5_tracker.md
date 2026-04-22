# Baseline Stage 5 Tracker

## 阶段目标
- 完成 `Stage 5` 训练、回接、效果诊断，并把后续路线从“继续混训”转向“agent 控制层验证”。
- 在不改动 `PSCR backbone` 主线定义的前提下，推进 `Stage 6/7/8`：
  - `Re-Verification Agent`
  - `Module-Selection Agent`
  - `AutoModeSelection Agent`
  - 独立评测/消融框架

## 当前结论
- 更新时间：`2026-04-22`
- `Stage 1-4`：主链已完成，Stage 4 cache 可作为 Stage 5 训练与推理输入。
- `Stage 5`：训练结论已经收口。
  - `Open6DOR-only`：当前作为 Open6DOR 单域最优 head。
  - `SpatialBench-only`：当前作为 SpatialBench 单域最优 head。
  - `Combined / Balanced / Filtered / Curriculum`：当前阶段性判负。
- `Stage 6/7/8`：本地代码已经完成，且关键入口通过 `py_compile` 静态检查。
- 当前还不能写成最终实验结论的部分：
  - `SpatialBench` dataset/auto agent smoke 还没做服务器验证
  - `Open6DOR` dataset/auto agent smoke 还没做服务器验证
  - `Stage 8` baseline / direct_stage5 / agent 三组对比表还没跑出来

## 当前主线代码状态快照

| 文件 | 当前状态 | 当前作用 |
| --- | --- | --- |
| `sofar/serve/semantic_orientation_agent.py` | done | 统一的规则版 agent controller，覆盖 SpatialBench、Open6DOR、auto route |
| `sofar/serve/agent_debug.py` | done | 统一 agent trace / eval bundle 的集中写盘 |
| `sofar/spatialbench/eval_spatialbench.py` | done | SpatialBench dataset/auto agent 接入点 |
| `sofar/open6dor/open6dor_perception.py` | done | Open6DOR direct / conditional / baseline / shadow 执行带接入点 |
| `sofar/analysis/stage8_agent_eval.py` | done | baseline / direct_stage5 / agent 三组独立评测 runner |
| `sofar/analysis/stage8_agent_ablation.py` | done | 三组模式消融汇总 runner |

完整代码地图与交接入口见：`当前代码地图与交接总表.md`

## 时间线

### 2026-04-16：Stage 5 训练结论收口
- `Open6DOR-only`：
  - `best_val_loss = 0.376927`
  - `test mean_cosine = 0.289814`
  - `test mean_angle_deg = 67.6826`
- `SpatialBench-only`：
  - `best_val_loss = 0.223075`
  - `test mean_cosine = 0.554985`
  - `test mean_angle_deg = 49.1498`
- `Combined / Balanced / Filtered / Curriculum`：全部未形成可接受的共享 head 路线。

### 2026-04-16：Stage 5 回接与效果诊断完成
- `Open6DOR`：单域 head 已稳定回接到完整 pipeline，20-case 与后续样本扩展证明工程链路稳定。
- `SpatialBench`：直接注入 Stage 5 没有带来 orientation 提升，进一步分析确认问题不在“链路没通”，而在“题型异质性太强、很多题并不适合单对象方向向量 head”。
- 由此得到中间结论：
  - `Open6DOR` 更适合做单目标操控朝向验证。
  - `SpatialBench` 更适合做题型筛选、routing、fallback 策略验证。

### 2026-04-16：Open6DOR mode gating 落地
- 已在 `open6dor_perception.py` 中加入 mode-level gating。
- 默认只对 `upright`、`upright_lens_forth` 放行直接 Stage 5。
- `plug_right`、`lying_flat` 后续被纳入 `conditional verify` 主线。
- `upside_down`、`sideways`、`cap_*` 等模式不再走“无脑全注入”。

### 2026-04-17：文档叙事改成 PSCR backbone + lightweight agent layer
- `so_far_pscr_实验设计报告.md` 已重写为：
  - `PSCR backbone`
  - `Re-Verification Agent`
  - `Module-Selection Agent`
- 主判断明确为：
  - `Open6DOR` 与 `SpatialBench` 不是同一个 orientation problem
  - shared mixed head 当前阶段性失败
  - 后续重点改成 agentic control，而不是继续混训

### 2026-04-17：Stage 6 初版 agent 化完成
- 新增 `semantic_orientation_agent.py` 的统一 schema 初版。
- SpatialBench 从纯 gating 升级为规则版 routing。
- Open6DOR 初步接入 decision schema 与汇总字段。

### 2026-04-21：Stage 6/7/8 全量开发完成
- 已完成 `semantic_orientation_agent.py` 升级：
  - 统一 `controller / policy_version / selected_execution_mode / selected_actions`
  - SpatialBench `decide + verify`
  - Open6DOR `decide + verify`
  - `auto route`
- 已完成 `agent_debug.py`：
  - `output/agent_debug/<dataset>/<run_id>/`
  - `output/agent_eval/<run_id>/<dataset>/<mode>/`
- 已完成 `eval_spatialbench.py` 升级：
  - 支持 `--agent-mode off|dataset|auto`
  - 支持 `--agent-save-trace`
  - 支持 summary 中的 agent 分布字段
- 已完成 `open6dor_perception.py` 升级：
  - `direct_allow`
  - `conditional_verify`
  - `baseline_only`
  - `shadow_stage5_for_debug`
- 已完成 `Stage 8` 独立 runner：
  - `sofar/analysis/stage8_agent_eval.py`
  - `sofar/analysis/stage8_agent_ablation.py`
- 已通过静态检查：
  - `semantic_orientation_agent.py`
  - `agent_debug.py`
  - `eval_spatialbench.py`
  - `open6dor_perception.py`
  - `stage8_agent_eval.py`
  - `stage8_agent_ablation.py`

### 2026-04-22：SpatialBench 策略调整为“双轨验证”
- 已确认顺序 `20-case dataset agent smoke` 可能出现“全量 skip”，原因不是规则失效，而是前 `20` 条样本未命中 `single_object_direction` 子题。
- 已确认当前 `SpatialBench` 全集按现有规则只有 `14` 条 `single_object_direction` 样本，且本地现成 Stage 4 cache 只覆盖其中 `1` 条。
- 因此当前策略调整为两条并行验证线：
  - 顺序 smoke：继续用于观察 `decision distribution / skip distribution`
  - `stage5_applicable14` pilot：专门用于补 Stage 3/4 cache，并验证 `Stage 5` 在真正适用子题上能否介入
- 已在 `eval_spatialbench.py` 中加入：
  - `--pilot stage5_applicable14`
  - `--sample-id-file`
  - `--sample-ids`
  - `--stage5-category-filter`
- 已新增 `sofar/spatialbench/pilots/stage5_applicable14.json`，作为当前 SpatialBench targeted smoke 的固定入口。

### 2026-04-22：SpatialBench rule_v2 本地完成
- 已把 `SpatialBench` 的 Stage 5 适用子题从单一 `single_object_direction` 拆成三类：
  - `axis_direction`
  - `reference_alignment`
  - `camera_alignment`
- 已修正旧版 `ambiguous_target` 误杀：
  - `parser_object_count = 2` 的“目标 + 参照物”题，不再被一票否决
  - `200 / 216 / 218` 这类题现在在规则层允许进入 Stage 5 路由
- 已在以下代码中接入 `rule_v2`：
  - `sofar/serve/spatialbench_stage5.py`
  - `sofar/serve/semantic_orientation_agent.py`
  - `sofar/serve/qwen_inference.py`
  - `sofar/serve/system_prompts.py`
  - `sofar/spatialbench/eval_spatialbench.py`
- 新版日志会额外记录 `prompt_variant`，后续服务器 smoke 重点检查：
  - 是否出现 `use_stage5_with_orientation_prompt` 的新增命中
  - 是否能看到 `axis_direction / reference_alignment / camera_alignment` 的分流
  - 这些命中是否比旧版“全量 skip”更符合题型语义

## 当前风险判断
- `SpatialBench` 上大量样本被 skip 并不等于 agent 失败，前提是 skip 的理由和题型语义一致。
- `Open6DOR` 上 direct Stage 5 不是对所有 orientation mode 都有效，所以 verify / fallback / shadow 是必要设计，不是附加复杂度。
- 在 `Stage 8` 的三组对比表没有跑出来之前，不能把“agent 有效”写成论文结论。

## 当前下一步
- 当前最优先是按 `交接操作.txt` 完成：
  - SpatialBench dataset agent smoke
  - SpatialBench `stage5_applicable14` targeted smoke
  - Open6DOR dataset agent smoke
  - SpatialBench auto agent smoke
  - Open6DOR auto agent smoke
  - Stage 8 eval smoke
  - Stage 8 ablation smoke
- 当前不再优先推进：
  - 新的 mixed training 变体
  - 更大规模 shared head 训练
  - 在没有 agent 验证的前提下继续扩大 direct injection 实验
