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
  - `Open6DOR-only`：`Round 1 formal` 已完成，是当前最新正式训练结果。
    - `best_val_loss = 0.201873`
    - `final_test_metrics.mean_cosine = 0.546283`
    - `unseen eval mean_cosine = 0.50615`
    - `unseen eval mean_angle_deg = 50.4271`
  - `SpatialBench-only`：当前作为 SpatialBench 单域最优 head。
  - `Combined / Balanced / Filtered / Curriculum`：当前阶段性判负。
- `Stage 6/7/8`：本地代码已经完成，且关键入口通过 `py_compile` 静态检查；服务器侧 smoke/eval/ablation 也已经跑完。
- 当前主判断已经明确：
  - `Open6DOR`：正式训练、未见样本评测、20-case 回接 smoke 都已完成；下一步应转向 `mode-aware` 接回诊断，而不是继续把它当成全 mode 统一最优头。
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

### 2026-04-24：Open6DOR formal Round 1 已完成
- 增量 cache 扩容已完成：
  - `Stage 3` 新增补跑 `100` 条 pending
  - `Stage 4` 已对新增与历史 pending 做增量补齐
- `Stage 5 dry-run` 已跑通：
  - `dataset_size = 16`
  - `loss = 1.571849`
- `Open6DOR formal Round 1` 训练完成：
  - `train_size = 320`
  - `val_size = 50`
  - `test_size = 29`
  - `best_val_loss = 0.201873`
  - `final_test_metrics.mean_cosine = 0.546283`
- 未见样本评测完成：
  - `dataset_size = 30`
  - `weighted_loss = 0.34554`
  - `mean_cosine = 0.50615`
  - `mean_angle_deg = 50.4271`
  - `median_angle_deg = 30.8102`
- 中间判断：
  - 相比早期小样本 `Open6DOR-only`，这轮 head 本身明显更强。
  - 训练与未见样本结果都足够支持“继续接回验证”，不是伪提升。

### 2026-04-24：Open6DOR 新 checkpoint 回接 smoke 已完成
- 本轮回接文件：
  - `sofar/output/open6dor_perception_summary_20260424_170143.json`
  - `sofar/output/stage5_open6dor_pipeline_records_20260424_170143.json`
- 工程稳定性结论：
  - `20/20 success`
  - `0 error`
  - `avg_success_sec = 16.35`
  - `agent_debug/open6dor/20260424_170143/` 已完整生成
- 与上一版 `20260422_212450` 对比后的 mode 结论：
  - `plug_right`：
    - `oldApplied 0 -> newApplied 2`
    - 有 2 条 USB-lighter case 从 reject 变成 accepted
  - `lying_flat`：
    - `oldApplied 0 -> newApplied 4`
    - binder 系列 case 明显改善
  - `upright`：
    - `oldApplied 6 -> newApplied 0`
    - 这批 20-case 里出现了明确退化
  - `clip_sideways`：
    - 继续维持 `shadow only`
- 当前最准确的结论：
  - 新 checkpoint 不是“全模式统一最优接回版本”
  - 它更像是一个对 `lying_flat` 和部分 `plug_right` 更强、但对 `upright` 更弱的 mode-specific head
  - 因此下一步优先做 `mode-aware checkpoint / threshold / routing`，而不是直接继续追加全局训练轮次

## 当前风险判断
- `SpatialBench` 上大量样本被 skip 不等于 agent 失败，关键在于 skip 理由是否与题型语义一致。
- `SpatialBench` 的正式训练风险仍然很高，因为题型异质性太强；当前更合理的做法是继续保留“适用子题验证”而不是推进全数据训练。
- `Open6DOR` 上 direct Stage 5 不是对所有 orientation mode 都有效，所以 verify / fallback / shadow 是必要设计，不是附加复杂度。
- `Open6DOR` 新 checkpoint 已证明 head 更强，但接回收益是 mode-specific；如果不做 mode-aware 处理，容易把 `upright` 的退化掩盖掉。
- `Open6DOR` 还缺少更严格的 ground-truth orientation 评测指标，后续需要补上。

## 当前下一步
- 当前最优先：做 `Open6DOR mode-level` 接回分析和策略修正。
- 具体顺序已经收口为：
  1. 用当前 `Round 1 formal checkpoint` 做更大一点的 `Open6DOR agent eval`
  2. 按 `upright / plug_right / lying_flat / others` 做分桶
  3. 决定是走 `mode-specific checkpoint` 还是先修 `upright verify / head`
- 当前不再优先推进：
  - 新的 mixed training 变体
  - `SpatialBench` 全数据正式训练
  - 在没有 GT orientation 指标与 mode 分桶结论前继续扩大 `Stage 8` 数量并直接宣称性能提升
