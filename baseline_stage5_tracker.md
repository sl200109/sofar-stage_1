# Baseline Stage 5 Tracker

## 阶段目标
- 完成 `Stage 5` 训练、回接、效果诊断，并把后续路线从“继续混训”转向“agent 控制层验证”。
- 在不改动 `PSCR backbone` 主线定义的前提下，推进 `Stage 6/7/8`：
  - `Re-Verification Agent`
  - `Module-Selection Agent`
  - `AutoModeSelection Agent`
  - 独立评测/消融框架
- 在 `Stage 8 smoke` 跑通后，决定是否进入下一轮正式训练，以及优先在哪个数据集上扩大训练数据。

## 当前结论
- 更新时间：`2026-04-24`
- `Stage 1-4`：主链已完成，`Stage 4 cache` 可作为 `Stage 5` 训练与推理输入。
- `Stage 5`：训练结论已经收口。
  - `Open6DOR-only`：当前作为 Open6DOR 单域最优 head。
  - `SpatialBench-only`：当前作为 SpatialBench 单域最优 head。
  - `Combined / Balanced / Filtered / Curriculum`：当前阶段性判负。
- `Stage 6/7/8`：本地代码已经完成，且关键入口通过 `py_compile` 静态检查；服务器侧 smoke/eval/ablation 也已经跑完。
- 当前主判断已经明确：
  - `Open6DOR`：可以进入下一轮正式训练，先走“增量补 Stage 4 cache -> Open6DOR-only 正式训练 -> 未见样本评测 -> 回接 20-case smoke”主线。
  - `SpatialBench`：当前不进入全数据正式训练，继续作为“适用子题 + agent routing / prompt / fallback”验证基准。

## 当前主线代码状态快照

| 文件 | 当前状态 | 当前作用 |
| --- | --- | --- |
| `sofar/serve/semantic_orientation_agent.py` | done | 统一的规则版 agent controller，覆盖 SpatialBench、Open6DOR、auto route |
| `sofar/serve/agent_debug.py` | done | 统一 agent trace / eval bundle 的集中写盘 |
| `sofar/spatialbench/eval_spatialbench.py` | done | SpatialBench dataset/auto agent 接入点 |
| `sofar/open6dor/open6dor_perception.py` | done | Open6DOR direct / conditional / baseline / shadow 执行带接入点 |
| `sofar/analysis/stage8_agent_eval.py` | done | baseline / direct_stage5 / agent 三组独立评测 runner，已支持 `--reuse-source` |
| `sofar/analysis/stage8_agent_ablation.py` | done | 三组模式消融汇总 runner |
| `sofar/orientation/stage5_train_pilot.py` | done | Stage 5 单域训练入口，下一轮正式训练直接复用 |
| `sofar/orientation/stage5_eval_unseen.py` | done | Stage 5 未见样本评测入口，下一轮正式训练直接复用 |

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
- 中间结论：
  - `Open6DOR` 更适合做单目标操控朝向验证。
  - `SpatialBench` 更适合做题型筛选、routing、fallback 策略验证。

### 2026-04-17：文档叙事改成 PSCR backbone + lightweight agent layer
- `so_far_pscr_实验设计报告.md` 已重写为：
  - `PSCR backbone`
  - `Re-Verification Agent`
  - `Module-Selection Agent`
- 主判断明确为：
  - `Open6DOR` 与 `SpatialBench` 不是同一个 orientation problem
  - shared mixed head 当前阶段性失败
  - 后续重点改成 agentic control，而不是继续混训

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

### 2026-04-22：四个 20-case smoke 已全部完成
- `SpatialBench dataset agent smoke`：已完成。
- `SpatialBench auto agent smoke`：已完成。
- `Open6DOR dataset agent smoke`：已完成。
- `Open6DOR auto agent smoke`：已完成。
- 关键结果：
  - `SpatialBench auto` 已稳定路由到 `spatialbench_semantic_orientation_agent`。
  - `Open6DOR auto` 已稳定路由到 `open6dor_semantic_orientation_agent`。
  - `Open6DOR auto` 中已经真实出现：
    - `use_stage5_direct`
    - `reject_stage5_keep_parser_orientation`
    - `shadow_stage5_for_debug`
  - 这说明当前 controller 不只是“接口接上了”，而是已经在真实样本上触发了 direct / fallback / shadow 三种行为。

### 2026-04-22：SpatialBench `rule_v2` targeted pilot 已完成
- `stage5_applicable14` 定向子集已经跑完。
- 本地结果文件：`sofar/output/eval_spatialbench_20260422_152854.json`
- 关键结果：
  - `processed_samples = 14`
  - `total_accuracy = 0.4286`
  - `agent_decision_distribution = {use_stage5_with_orientation_prompt: 11, skip_stage5_due_to_task_mismatch: 2, skip_stage5_due_to_ambiguous_target: 1}`
  - `agent_category_distribution = {single_object_axis_direction: 3, single_object_reference_alignment: 7, single_object_camera_alignment: 2, target_selection_or_relation: 2}`
  - `agent_verification_distribution = {accepted: 11, not_run: 3}`
  - `stage5_usage_precision = 1.0`
  - `fallback_precision = 1.0`
- 这说明 `SpatialBench` 这条线当前更适合作为“适用子题上的 agent/routing 验证”，而不是直接拿去做全数据正式训练。

### 2026-04-22：Stage 8 eval / ablation smoke 已完成
- `SpatialBench` 顺序前 `20` 条 `Stage 8` 结果：
  - `baseline total_accuracy = 0.25`
  - `direct_stage5 total_accuracy = 0.30`
  - `agent total_accuracy = 0.25`
- 这组结果的意义不是“Stage 5 无效”，而是：顺序前 `20` 条主要是 `non_orientation / angle_or_relation / count_or_quantity`，本来就不是适合统一 Stage 5 注入的样本。
- `Open6DOR` 顺序前 `20` 条 `Stage 8` 结果：
  - `baseline valid_result_rate = 1.0`
  - `direct_stage5 valid_result_rate = 1.0`
  - `agent valid_result_rate = 1.0`
  - `agent stage5_acceptance_rate = 0.3`
  - `agent fallback_rate = 0.7`
  - `agent corrected_error_count = 13`
- 这说明 `Open6DOR` 的 agent controller 已经具备稳定的“接受 / 拒绝 / 回退 / shadow”行为。
- 重要限制：
  - 当前 `stage8_agent_eval.py` 对 `Open6DOR` 的 `correct` 仍然是按 `status == success` 统计，属于“控制器稳定性指标”，还不是最终的 orientation ground-truth 指标。
  - 因此现在可以据此决定“开始正式训练”，但还不能把它写成“orientation 性能已经提升”的最终论文结论。
- 额外记录：
  - 当多次对不同数据集复用同一个 `sofar/output/agent_eval_smoke` 根目录时，根目录下的 `summary.json / ablation_summary.json` 会被最后一次运行覆盖。
  - 当前应以子目录结果为准：
    - `sofar/output/agent_eval_smoke/spatialbench/...`
    - `sofar/output/agent_eval_smoke/open6dor/...`

## 当前风险判断
- `SpatialBench` 上大量样本被 skip 不等于 agent 失败，关键在于 skip 理由是否与题型语义一致。
- `SpatialBench` 的正式训练风险仍然很高，因为题型异质性太强；当前更合理的做法是继续保留“适用子题验证”而不是推进全数据训练。
- `Open6DOR` 上 direct Stage 5 不是对所有 orientation mode 都有效，所以 verify / fallback / shadow 是必要设计，不是附加复杂度。
- `Open6DOR` 当前已经满足“进入下一轮正式训练”的工程条件，但还缺少更严格的 ground-truth orientation 评测指标，后续需要补上。

## 当前下一步
- 当前最优先：开始 `Open6DOR` 下一轮正式训练。
- 具体顺序已经收口为：
  1. 先用 `100 + speed-profile off` 增量补 `Open6DOR Stage 3/4 cache`
  2. 跑一次 `Stage 5 dry-run`
  3. 跑 `Open6DOR-only` 正式训练 Round 1
  4. 跑未见样本评测
  5. 用新 checkpoint 做一次 `Open6DOR 20-case` 回接 smoke
- 当前不再优先推进：
  - 新的 mixed training 变体
  - `SpatialBench` 全数据正式训练
  - 在没有 GT orientation 指标的前提下继续扩大 `Stage 8` 数量并直接宣称性能提升
