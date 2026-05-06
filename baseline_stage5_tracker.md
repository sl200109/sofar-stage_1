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
- 2026-04-28 最新日志补充：`stage5_pilot_best.pth` 目前应冻结为 `upright expert`，不再作为全模式统一 ckpt 继续混训。
- 0428 最新 `Open6DOR` 输出显示：`Stage 5` 仅 `7/100` 样本被直接启用，`93/100` 样本回退或 shadow；直接收益主要集中在 `upright`，其他方向没有形成稳定提升。
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
- `Open6DOR` 当前更进一步的结论是：不要按每个细粒度 mode 单训 ckpt，优先拆成少量任务族专家，再由 agent 选择 ckpt 执行。
- `20260428_222616` 的 family routing smoke 已证明：三桶 routing 元数据写入正确，但当前 shared `stage5_pilot_best.pth` 在 `upright` 任务上仍输出负 `z` 或近水平 `z`，因此被 verifier 全部拒绝。

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

### 2026-04-28：Open6DOR ckpt expert split 决策已确定
- 最新日志复核：
  - `total_records = 100`
  - `stage5_applied_count = 7`
  - `fallback_count = 93`
  - `stage5_mode_distribution` 仍然明显偏向 `upright / lying_flat / plug_right / upside_down` 这类大桶，而非细粒度小桶。
- 结论：
  - `stage5_pilot_best.pth` 冻结为 `upright expert`
  - 不再按 `cap_forth / upside_down_textual / clip_sideways` 这类细粒度 mode 单训 ckpt
  - 下一轮改成 3 个左右的任务族专家：
    - `upright / vertical`
    - `flat / upside_down / lying_flat`
    - `plug / cap / sideways`
  - `agent` 层负责按任务族和置信信号选择 ckpt，而不是让一个 ckpt 硬吃全部任务
  - 每个专家只在自己的 bucket 上做 held-out 验证，再看跨 bucket 退化

### 2026-04-28：Open6DOR family expert routing 入口已在本地实现
- 本轮本地改动文件：
  - `sofar/open6dor/open6dor_perception.py`
  - `sofar/serve/runtime_paths.py`
- 已实现内容：
  - 在 `open6dor_perception.py` 新增 `task_family` 级 Stage 5 ckpt routing 入口
  - 保持原有 `agent` 判断“是否运行 Stage 5”，新增本地路由判断“运行哪个 ckpt”
  - 新增三桶任务族：
    - `upright_vertical`
    - `flat_upside_down_lying_flat`
    - `plug_cap_sideways`
  - 新增 CLI / env override：
    - `--stage5-expert-routing`
    - `--stage5-upright-expert-checkpoint`
    - `--stage5-flat-expert-checkpoint`
    - `--stage5-plug-expert-checkpoint`
    - `SOFAR_STAGE5_OPEN6DOR_UPRIGHT_CKPT`
    - `SOFAR_STAGE5_OPEN6DOR_FLAT_CKPT`
    - `SOFAR_STAGE5_OPEN6DOR_PLUG_CKPT`
  - 当开启 `task_family` routing 且某任务族没有对应 ckpt 时，不再默认复用当前 `upright` ckpt。
  - 已把 `checkpoint_family / checkpoint_source / checkpoint_route_reason / checkpoint_route_signals` 写入 Stage 5 输出记录与 summary。
- 本轮验证状态：
  - 仅完成本地静态验证：`python -m py_compile sofar/open6dor/open6dor_perception.py sofar/serve/runtime_paths.py`
  - 尚未完成服务器 smoke，因此不能记为服务器通过

### 2026-04-28：Open6DOR family routing 服务器 smoke 与 upright 语义问题已确认
- 服务器 smoke 结果文件：
  - `sofar/output/open6dor_perception_summary_20260428_222616.json`
  - `sofar/output/stage5_open6dor_pipeline_records_20260428_222616.json`
- 已确认通过的点：
  - `task_family` 路由分桶正确写入：
    - `upright_vertical = 8`
    - `flat_upside_down_lying_flat = 5`
    - `plug_cap_sideways = 7`
  - 缺少 `flat / plug` 专家 ckpt 时，strict routing 会稳定跳过 Stage 5，不再错误复用 shared upright ckpt。
- 新确认的 blocker：
  - 当前 shared `stage5_pilot_best.pth` 虽然会被路由到 `upright_vertical`，但这批 `upright` 样本全部被 verifier 拒绝。
  - 关键证据是 `stage5_direction_vector.z` 全部为负或接近 `0`，例如：
    - `z=-0.7539`
    - `z=-0.6048`
    - `z=-0.0622`
  - 这说明当前更优先的问题不是继续扩专家数，而是先查清 `top / bottom / z-up` 的语义是否反了，或训练标签与 verifier 的坐标定义是否不一致。

### 2026-04-28：upright 语义诊断代码与 family manifest splitter 已在本地补齐
- 本轮本地改动文件：
  - `sofar/serve/stage5_inference.py`
  - `sofar/open6dor/open6dor_perception.py`
  - `sofar/orientation/stage5_split_open6dor_manifest.py`
- 已实现内容：
  - `stage5_inference.py`
    - 新增 `orientation_diagnostics`
    - 对每次预测记录 `vector x/y/z`、`top_up_score`、`bottom_down_score`
    - 对 `top / bottom` 请求语义打标签：如 `top_up_consistent`、`top_points_down`、`top_axis_ambiguous`
  - `open6dor_perception.py`
    - 日志中新增 `stage5 diagnostics status=... attrs=... xyz=(...)`
    - records / csv / summary 新增诊断字段与 `semantic_status` 分布
  - `stage5_split_open6dor_manifest.py`
    - 可把 Open6DOR manifest 直接拆成：
      - `upright_vertical`
      - `flat_upside_down_lying_flat`
      - `plug_cap_sideways`
    - 供 `stage5_train_pilot.py --manifest-path` 直接复用
- 本轮验证状态：
  - 仅完成本地静态验证：`python -m py_compile sofar/serve/stage5_inference.py sofar/open6dor/open6dor_perception.py sofar/orientation/stage5_split_open6dor_manifest.py`
  - 新一轮服务器 smoke 与 family manifest split 尚未执行

### 2026-04-28：family manifest split 已在服务器完成，flat / plug 决定从零训练
- 服务器已完成：
  - `stage5_split_open6dor_manifest.py --min-label-confidence 0.65`
- 当前三桶样本量：
  - `upright_vertical = 130`
  - `flat_upside_down_lying_flat = 140`
  - `plug_cap_sideways = 94`
- 当前 split 统计：
  - `flat`: `train=121, val=11, test=8`
  - `plug`: `train=76, val=10, test=8`
  - `upright`: `train=98, val=24, test=8`
- 已确认下一步策略：
  - `flat expert` 从零训练
  - `plug expert` 从零训练
  - 不复用当前 `stage5_pilot_best.pth` 作为这两桶的 init checkpoint
- 原因：
  - 当前 shared ckpt 在 `upright` 上已暴露 `top / bottom / z` 语义冲突
  - 在这个问题没查清前，不应把其参数偏置继续带入 `flat` 和 `plug` 专家

### 2026-04-28：flat / plug scratch Round 1 已完成，结果显示仍需更强的数据与语义约束
- 训练结果文件：
  - `sofar/output/stage5_open6dor_flat_expert_round1_scratch/stage5_tiny_train_summary.json`
  - `sofar/output/stage5_open6dor_plug_expert_round1_scratch/stage5_tiny_train_summary.json`
- `flat / upside_down / lying_flat` scratch Round 1：
  - `best_val_loss = 0.310211`
  - `final_test_metrics.mean_cosine = -0.276815`
  - 结论：当前 scratch 版本明显不稳定，不能直接作为可用专家
- `plug / cap / sideways` scratch Round 1：
  - `best_val_loss = 0.567769`
  - `final_test_metrics.mean_cosine = 0.132154`
  - 结论：略好于 flat，但仍不足以作为稳定专家接回
- 新判断：
  - 任务族拆分方向是对的，但仅凭当前 scratch + 现有 label 质量，还不足以得到可用的 flat / plug expert
  - 后续若继续训练，需要先增强语义标签清洗或考虑更合适的 warm-start / curriculum，而不是直接扩大训练轮次

### 2026-04-29：upright 语义冲突已定位到 Stage 5 manifest 标签源
- 本轮本地改动文件：
  - `sofar/serve/stage5_manifest.py`
  - `sofar/serve/stage5_inference.py`
  - `sofar/orientation/stage5_refresh_manifests.py`
- 本地静态验证：
  - `python -m py_compile sofar/serve/stage5_manifest.py sofar/serve/stage5_inference.py sofar/orientation/stage5_refresh_manifests.py`
- 本地针对旧 manifest 的轻量复核结果：
  - `upright_total = 130`
  - `upright_positive_z = 48`
  - `upright_negative_z = 47`
  - `upright_zeroish_z = 35`
  - 负例的 `train_target_direction_source` 主要来自 `geometry_priors.part_to_object_vector`
- 已实现修复：
  - `stage5_manifest.py`
    - 对 `open6dor` 的 `upright / vertical / upside_down` 任务优先写入语义轴标签，而不是优先继承 `geometry_priors.part_to_object_vector`
    - 目标是阻断 `upright` 被错误写成负 `z` 或近水平 `z` 的训练标签
  - `stage5_inference.py`
    - 对 `upright / upside_down` 的显式属性做单一 canonical attr 归一化
    - 避免同一请求同时保留 `top` 和 `bottom` 造成 verifier 语义冲突
  - `stage5_refresh_manifests.py`
    - 提供独立入口，直接从当前 `Stage 4` records/cache 重建 `stage5_manifest_*_smoke.jsonl`
- 当前判断：
  - `upright` 的首要问题已经明确是上游训练标签噪声，不是 verifier 阈值单点问题
  - 因此下一步应先刷新 manifest，再决定是否重训 `upright expert`
  - `flat / plug` 当前不建议直接切到 warm-start 或 curriculum；优先保留“先做更干净 label 清洗”的路线

### 2026-04-29：upright semanticfix Round 2 已完成且结果显著正确
- 服务器结果文件：
  - `sofar/output/stage5_open6dor_upright_expert_round2_semanticfix/stage5_manifest_summary.json`
  - `sofar/output/stage5_open6dor_upright_expert_round2_semanticfix/stage5_tiny_train_summary.json`
- manifest 复核结果：
  - `open6dor.available_entries = 407`
  - `upright_vertical` 子桶训练集全部写入 `orientation_mode.upright_semantic_axis`
  - `train/val/test` 的 `label_source_counts` 均为 `orientation_mode.upright_semantic_axis`
- 训练结果：
  - `train_size = 98`
  - `val_size = 24`
  - `test_size = 8`
  - `best_val_loss = 1e-06`
  - `final_test_metrics.mean_cosine = 0.999998`
- 结论：
  - 这轮 upright semanticfix 已经把旧的 `z` 符号污染修掉了
  - 当前结果可以作为 `upright expert` 的新基线候选
  - 下一步应优先把这版 `upright` ckpt 接回 `task_family routing` 做 smoke，而不是继续修 upright 标签
  - `flat / plug` 仍保持“先 label 清洗、暂不 warm-start / curriculum”的判断不变

### 2026-04-29：upright 新 ckpt task_family smoke 已通过
- 服务器结果文件：
  - `sofar/output/open6dor_perception_summary_20260429_144250.json`
  - `sofar/output/stage5_open6dor_pipeline_records_20260429_144250.json`
- 结果概览：
  - `success_count = 20`
  - `error_count = 0`
  - `used_stage5_count = 8`
  - `fallback_count = 12`
  - `stage5_mode_distribution = upright: 8, lying_flat: 5, plug_right: 6, clip_sideways: 1`
- 关键结论：
  - `upright` 的 8 条全部命中新 `stage5_open6dor_upright_expert_round2_semanticfix/stage5_pilot_best.pth`
  - `stage5_checkpoint_source = family_default_shared`
  - `stage5_semantic_status_distribution = top_up_consistent: 8`
  - `stage5_direction_vector` 已恢复为接近正 `z`
  - `decision_distribution = use_stage5_direct: 8, reject_stage5_keep_parser_orientation: 12`
- 解释：
  - 这说明新 `upright expert` 已经可作为 `task_family` 路由下的有效 expert 基线
  - 其余 12 条主要落在 `plug_cap_sideways` 和 `flat_upside_down_lying_flat`，但这两个任务族仍缺对应 ckpt，因此被严格路由跳过
  - 目前没有证据表明 `upright` 回接后对其它桶造成新的负面影响，因为未配置专家的桶本来就走 fallback

### 2026-04-29：flat / plug label 清洗规则已补齐到语义轴层
- 本轮本地改动文件：
  - `sofar/serve/stage5_manifest.py`
- 清洗规则更新：
  - `flat_upside_down_lying_flat`
    - `upside_down* -> orientation_mode.upside_down_semantic_axis`
    - `lying_flat -> orientation_mode.lying_flat_semantic_axis`
    - `lying_sideways -> orientation_mode.lying_sideways_semantic_axis`
  - `plug_cap_sideways`
    - `plug_right / plug_left -> orientation_mode.plug_*_semantic_axis`
    - `handle_right / handle_left / handle_right_jaw_left -> orientation_mode.handle_*_semantic_axis`
    - `cap_right_bottom_left / cap_left_bottom_right -> orientation_mode.cap_*_semantic_axis`
    - `clip_sideways / sideways -> orientation_mode.*_sideways_semantic_axis`
- 本地不落盘 replay 结果：
  - `flat_upside_down_lying_flat`
    - 所有样本都能被重算到语义轴源
    - `z` 结果变成 `107 neg / 33 zero / 0 pos`
  - `plug_cap_sideways`
    - 所有样本都能被重算到语义轴源
    - `z` 结果全部为 `0`
- 当前判断：
  - 这轮修复已经把 `flat / plug` 从几何 priors 依赖里拉出来了
  - 下一步应该在服务器上刷新 round3 manifest，再决定 `flat / plug` 是继续从零训还是引入 warm-start
  - 这一步之前不建议直接开新训练

### 2026-04-29：本地 label-cleaning smoke test 已建立
- 新增测试文件：
  - `tests/test_stage5_open6dor_label_cleaning.py`
  - `tests/__init__.py`
- 覆盖范围：
  - `upright` 语义轴优先
  - `flat` family 的 `upside_down / lying_flat / lying_sideways / clip_sideways`
  - `plug` family 的 `plug / handle / cap / sideways`
- 本地验证结果：
  - `python -m py_compile tests/test_stage5_open6dor_label_cleaning.py tests/__init__.py`
  - `python -m unittest tests.test_stage5_open6dor_label_cleaning`
  - 结果通过，3 个测试全部 OK
- 意义：
  - 以后 `flat / plug` 的 label cleaning 回归可以先走这个本地 smoke test
  - 不需要为了这类规则回归去跑大模型或做服务器 smoke

### 2026-04-29：round3 manifest 已刷新，flat / plug 清洗生效
- 服务器结果文件：
  - `sofar/output/stage5_manifest_summary.json`
  - `sofar/output/stage5_open6dor_family_manifests_round3`
- 结果概览：
  - `open6dor.available_entries = 407`
  - `upright_vertical = 130`
  - `flat_upside_down_lying_flat = 140`
  - `plug_cap_sideways = 99`
- 关键确认：
  - `upright_vertical` 仍是 `orientation_mode.upright_semantic_axis` 全量
  - `flat_upside_down_lying_flat` 已切到 `upside_down / lying_flat / lying_sideways` 的语义轴层
  - `plug_cap_sideways` 已切到 `plug / handle / cap / sideways` 的语义轴层
- 当前判断：
  - round3 manifest 刷新成功，清洗规则已生效
  - 这一步之后已决定 `flat / plug` 继续按 from-scratch 训练
  - 当前不引入 warm-start，避免把 `upright` 的 `z-up` 偏置带入横向语义任务族

### 2026-04-29：flat / plug Round 2 训练策略已收口为 from-scratch
- 原因：
  - `flat_upside_down_lying_flat` 已切成 `107 neg / 33 zero / 0 pos`
  - `plug_cap_sideways` 已切成全量 `z = 0` 的水平语义分布
  - 这两桶与 `upright` 的 `z-up` 表达空间明显不同
- 训练输入：
  - `flat_upside_down_lying_flat`: `train=121, val=11, test=8`
  - `plug_cap_sideways`: `train=80, val=10, test=9`
- 当前决策：
  - `flat expert Round 2`: from-scratch
  - `plug expert Round 2`: from-scratch
  - 当前不做 warm-start / curriculum 对照

### 2026-04-29：三专家联合接回 smoke 已进入当前验证主线
- 当前三专家 checkpoint：
  - `upright`: `stage5_open6dor_upright_expert_round2_semanticfix/stage5_pilot_best.pth`
  - `flat`: `stage5_open6dor_flat_expert_round2_scratch/stage5_pilot_best.pth`
  - `plug`: `stage5_open6dor_plug_expert_round2_scratch/stage5_pilot_best.pth`
- 当前执行方式：
  - `open6dor_perception.py --stage5-expert-routing task_family`
  - 显式挂载 `--stage5-upright-expert-checkpoint`
  - 显式挂载 `--stage5-flat-expert-checkpoint`
  - 显式挂载 `--stage5-plug-expert-checkpoint`
- 当前目标：
  - 验证三桶 `upright / flat / plug` 是否能各自命中对应 expert
  - 对比单专家版本的 `accepted / rejected / fallback` 分布

### 2026-04-29：三专家联合接回 smoke 已完成
- 服务器结果文件：
  - `sofar/output/open6dor_perception_summary_20260429_152132.json`
  - `sofar/output/stage5_open6dor_pipeline_records_20260429_152132.json`
- 结果概览：
  - `success_count = 40`
  - `error_count = 0`
  - `used_stage5_count = 14`
  - `fallback_count = 26`
  - `decision_distribution = {use_stage5_direct: 8, use_stage5_conditional_verify: 6, reject_stage5_keep_parser_orientation: 15, shadow_stage5_for_debug: 11}`
- family 级结论：
  - `upright_vertical`
    - `8/8` direct verified
    - `semantic_status = top_up_consistent`
    - 新 `upright expert` 继续稳定
  - `plug_cap_sideways`
    - `6` 条 `plug_right` 已进入 `stage5_conditional_verified`
    - `11` 条 `cap / clip` 仍然走 `shadow_stage5_for_debug`
    - 说明 `plug expert` 已开始提供有效增益，但当前 policy 仍偏保守
  - `flat_upside_down_lying_flat`
    - 本轮 `15` 条 `lying_flat` 全部 fallback
    - 不是训练崩了，而是 verifier 直接按 `|z|` 阈值拒绝：如 `|z|=0.9996 threshold=0.45`
    - 这说明 `flat expert` 已学到稳定向量，但当前 `lying_flat` 接回规则与新语义标签定义不一致
- 当前判断：
  - 三专家方案整体是成立的
  - `upright` 已完全可用
  - `plug` 已从训练转正推进到接回转正
  - `flat` 的主要 blocker 已从“训练质量”转成“agent verifier / mode rule 仍按旧逻辑拒绝”

## 当前风险判断
- `SpatialBench` 上大量样本被 skip 不等于 agent 失败，关键在于 skip 理由是否与题型语义一致。
- `SpatialBench` 的正式训练风险仍然很高，因为题型异质性太强；当前更合理的做法是继续保留“适用子题验证”而不是推进全数据训练。
- `Open6DOR` 上 direct Stage 5 不是对所有 orientation mode 都有效，所以 verify / fallback / shadow 是必要设计，不是附加复杂度。
- `Open6DOR` 新 checkpoint 已证明 head 更强，但接回收益是 mode-specific；如果不做 mode-aware 处理，容易把 `upright` 的退化掩盖掉。
- 当前 `upright` blocker 更具体地表现为：`top` 语义请求与 `direction_vector.z` 的符号冲突；如果不先解决这个语义/坐标问题，继续训练更多专家只会复制同类错误。
- `Open6DOR` 还缺少更严格的 ground-truth orientation 评测指标，后续需要补上。
- 细粒度 mode 继续拆 ckpt 的边际收益预计很低，当前优先级低于任务族专家拆分与 routing。
- `flat / plug` 现在虽已决定从零训练，但训练完成后仍必须先做各自桶内 held-out 验证，不能直接写成全模式收益。
- `flat / plug` scratch Round 1 已验证：训练链路通了，但模型质量还不够，不能直接接回主链。
- 2026-04-29 本地已完成 `lying_flat` verifier semantic-axis 修正，并补齐 `old_rule_pass / new_rule_pass / target_axis / cosine_to_target_axis / verifier_rule_version / verifier_decision_reason` 调试字段；服务器 smoke 仍待复跑确认。

## 当前下一步
- 当前最优先：做 `Open6DOR mode-level` 接回分析和策略修正。
- 具体顺序已经收口为：
  1. 冻结 `stage5_pilot_best.pth`，明确标成 `upright expert`
  2. 按任务族拆分下一轮专家 ckpt 方案
  3. 先做 `upright / vertical`、`flat / upside_down / lying_flat`、`plug / cap / sideways` 三桶规划
  4. 先刷新 `stage5_manifest_open6dor_smoke.jsonl`，消除 `upright` 的负 `z` 标签污染
  5. 用新的 `stage5_split_open6dor_manifest.py` 切出 round2 三桶 manifest
  6. `upright semanticfix Round 2` 已完成并确认语义修复有效
  7. `upright` 新 ckpt 的 `task_family smoke` 已通过
  8. `flat` 与 `plug` scratch Round 1 已完成，当前记录为负结果
  9. `flat / plug` label 清洗规则已经补齐到语义轴层
  10. 本地 label-cleaning smoke test 已建立
  11. round3 manifest 已刷新，flat / plug 清洗生效
  12. `flat / plug Round 2` 已决定继续 from-scratch 训练
  13. 三专家联合接回 smoke 已完成
  14. 已在本地修正 `lying_flat` 的 verifier / routing 规则，下一步同步到服务器后复跑同一批三专家 smoke
  15. 复跑时重点确认 `upright` 仍 `8/8 direct`、`plug_right` 仍 conditional verified、`lying_flat` 不再 `15/15 fallback`
- 当前不再优先推进：
  - 新的 mixed training 变体
  - `SpatialBench` 全数据正式训练
  - 在没有 GT orientation 指标与任务族分桶结论前继续扩大 `Stage 8` 数量并直接宣称性能提升

## 20. 2026-04-29 lying_flat verifier semantic-axis 修正已服务器复跑通过
- 服务器结果：
  - `open6dor_perception_summary_20260429_154727.json`
  - `stage5_open6dor_pipeline_records_20260429_154727.json`
- 关键结论：
  - `success_count = 40`
  - `error_count = 0`
  - `used_stage5_count = 29`
  - `fallback_count = 11`
  - `upright = 8/8 accepted`
  - `plug_right = 6/6 conditional verified`
  - `lying_flat = 15/15 accepted`
- verifier 侧确认：
  - `old_rule_pass = 25`
  - `new_rule_pass = 40`
  - `verifier_rule_version` 已分成 `rule_v2_legacy` 与 `rule_v3_semantic_axis`
  - `lying_flat` 不再被旧 `|z|` 规则整桶拒绝
- 当前判断：
  - 三专家 task_family smoke 已完成闭环
  - `lying_flat` 的接回逻辑已和 manifest semantic label 对齐
  - `upright` 与 `plug` 行为保持不变

## 21. 2026-04-29 论文级短耗时实验准备完成
- 本地已完成：
  - 40-case before/after verifier ablation
  - latency / overhead 表
  - cap / clip / shadow-only 分析
  - 400-case mode-balanced subset manifest
  - 400-case final method command 文本
- 产物目录：
  - `paper_results/open6dor_short_experiments_20260429/`
- 当前判断：
  - 40-case smoke 已闭环
  - 下一步进入 400-case mode-balanced formal evaluation
  - 当前不继续训练、不继续改 verifier、不跑全量 4389、不先跑完整 ablation

## 22. 2026-04-29 400-case subset source corrected to server-side from4389
- 旧的 407-source local Stage5-ready manifest 不再作为 formal 400-case 抽样源。
- 新增 server-side 采样 helper 与脚本，正式抽样改为：
  - 枚举 `/data/coding/SoFar/datasets/open6dor_v2/open6dor_v2/task_refine_6dof/`
  - 从全量 task tree 生成 `open6dor_eval_subset_400_from4389_seed42.*`
- 当前状态：
  - 只完成脚本与交接更新
  - 未启动 400-case evaluation
  - 仍然不训练、不改 verifier

## 23. 2026-04-29 400-case mode parser 与 validation gate 加固
- `parse_orientation_mode_from_task_dir` 已改为自底向上扫描所有 path parts。
- 现在能正确解析类似：
  - `.../Place_xxx.__plug_right/20240824-xxxx_no_interaction -> plug_right`
  - `.../Place_xxx.__lying_flat/20240824-xxxx_no_interaction -> lying_flat`
- `from4389` summary 现在强制包含并检查：
  - `available_total`
  - `empty_mode_count`
  - `other_family_count`
  - `available_family_distribution`
  - `selected_family_distribution`
  - `selected_orientation_mode_distribution`
- validation gate 规则：
  - `empty_mode_count > 0` 或 `selected_total != 400` 或 `selected_family_distribution` 不符合预期时，不生成 final method command
  - 仍保留 manifest / summary 输出，方便服务器复核

## 24. 2026-04-30 from4389 subset 首轮服务器生成失败点已定位
- 服务器已成功生成：
  - `sofar/paper_results/open6dor_short_experiments_20260429/open6dor_eval_subset_400_from4389_seed42.json`
  - `sofar/paper_results/open6dor_short_experiments_20260429/open6dor_eval_subset_400_from4389_seed42_task_list.json`
  - `sofar/paper_results/open6dor_short_experiments_20260429/open6dor_eval_subset_400_from4389_seed42_summary.json`
- 但该轮 `summary.validation.passed = false`，关键字段为：
  - `available_total = 1744`
  - `empty_mode_count = 0`
  - `other_family_count = 735`
  - `selected_total = 400`
  - `selected_family_distribution = {cap_clip_sideways: 104, flat_upside_down_lying_flat: 112, other: 41, plug_right: 34, upright_vertical: 109}`
- 失败原因已明确：
  - `parse_orientation_mode_from_task_dir` 已经正确，`run-level path` 不是当前 blocker
  - 真正问题是 `mode -> task_family` taxonomy 过窄，导致 `handle_right / blade_right / spout_right / remote_control_forth / watch_upright` 等大量有效 mode 被归入 `other`
  - 当前严格白名单下 `plug_right` 仅 `34` 条，因此不可能满足 `100` 条目标配额

## 25. 2026-04-30 已在本地扩充 task-family taxonomy，并加固 command gate
- 本轮本地改动文件：
  - `sofar/open6dor/eval_subset_sampling.py`
  - `sofar/tools/generate_open6dor_eval_subset_400_from4389.py`
  - `sofar/tools/analyze_open6dor_stage5_records.py`
  - `tests/test_open6dor_eval_subset_sampling.py`
- 已实现内容：
  - `upright_vertical` 扩到 `watch_upright / tape_measure_upright` 等 upright 派生 mode
  - `flat_upside_down_lying_flat` 扩到 `lower_rim` 等 flat-like mode
  - `plug_right` 家族扩到 `handle_* / blade_* / prong_right / spout_right / ballpoint_right / clasp_right / bulb_right_handle_left`
  - `cap_clip_sideways` 扩到 `cap_* / clip_* / sideways* / *_forth / *_far`
  - 增加关键词兜底规则，尽量把当前已知 mode 全部并入四个目标 family，避免再落入 `other`
  - `generate_open6dor_eval_subset_400_from4389.py` 与 `analyze_open6dor_stage5_records.py` 现在都会在 summary 中写入 `final_method_command_generated`
  - 如果 validation 不通过，会明确删除旧的 `open6dor_400_final_method_command.txt`
- 本地验证状态：
  - `python -m py_compile sofar/open6dor/eval_subset_sampling.py sofar/tools/generate_open6dor_eval_subset_400_from4389.py sofar/tools/analyze_open6dor_stage5_records.py tests/test_open6dor_eval_subset_sampling.py`
  - `python -m unittest tests.test_open6dor_eval_subset_sampling`
  - 两项均已通过
- 当前结论：
  - 这轮仍然不跑 400-case evaluation
  - 下一步只需要把更新后的 sampling 代码同步到服务器，然后重跑 `from4389` manifest 生成，重新检查 `other_family_count` 和 `selected_family_distribution`

## 26. 2026-05-06 subset400 正式运行结果已回传，当前 blocker 已定位
- 服务器 `subset400` 结果文件：
  - `sofar/output/open6dor_perception_summary_open6dor_eval_subset_400_from4389_seed42_task_list_20260430_151507.json`
  - `sofar/output/stage5_open6dor_pipeline_records_open6dor_eval_subset_400_from4389_seed42_task_list_20260430_151507.json`
  - `sofar/output/open6dor_perception_open6dor_eval_subset_400_from4389_seed42_task_list_20260430_151507.log`
- 核心结果：
  - `total_tasks = 400`
  - `success_count = 350`
  - `error_count = 50`
  - `used_stage5_count = 13`
  - `fallback_count = 387`
- 最关键的两个错误源：
  - `50` 个错误中有 `47` 个发生在 `reasoning`，其中 `46` 个是 `No valid Open6DOR reasoning JSON found in model output`
  - 运行时 `task_family / execution_band` 仍落后于抽样 taxonomy，`blade_right / ballpoint_right / tape_measure_upright / watch_upright` 等 mode 在 runtime 里仍大量被视为 `unknown` 或被过早 `fallback_required`

## 27. 2026-05-06 已在本地针对 subset400 错误做定向修正
- 本轮本地改动文件：
  - `sofar/open6dor/open6dor_perception.py`
  - `sofar/serve/semantic_orientation_agent.py`
  - `sofar/serve/qwen_inference.py`
  - `tests/test_open6dor_runtime_agent_modes.py`
  - `tests/test_qwen_open6dor_reasoning_json.py`
- 已实现内容：
  - `open6dor_perception.py`
    - 运行时 `infer_open6dor_stage5_task_family` 改为复用 `eval_subset_sampling.py` 的 taxonomy 结果，再映射到 `upright / flat / plug_cap_sideways`
    - `build_joint_object_context(...)` 不再只依赖对象模板；当对象模板缺失时，会回退到 mode-aware generic hints
    - 新覆盖的高频对象/模式包括：
      - `watch_upright`
      - `tape_measure_upright`
      - `handle_right / handle_left / handle_right_jaw_left`
      - `blade_right / blades_right`
      - `ballpoint_right / clasp_right / prong_right / spout_right`
      - `remote_control_forth / earpiece_far / multimeter_forth`
  - `semantic_orientation_agent.py`
    - `direct_allow` 扩到 `upright_textual / watch_upright / tape_measure_upright`
    - `conditional_verify` 扩到 `handle_* / blade_* / ballpoint_right / prong_right / spout_right / lower_rim / upside_down*`
  - `qwen_inference.py`
    - `Open6DOR reasoning JSON` 提取增强，新增对以下输出格式的恢复：
      - `x=0.53, y=0.11, z=0.30`
      - `x: 0.53, y: 0.11, z: 0.30`
      - `(0.37, 0.21, 0.34)`
- 本地验证状态：
  - `python -m py_compile sofar/open6dor/open6dor_perception.py sofar/serve/semantic_orientation_agent.py sofar/serve/qwen_inference.py tests/test_open6dor_runtime_agent_modes.py tests/test_qwen_open6dor_reasoning_json.py`
  - `python -m unittest tests.test_open6dor_runtime_agent_modes tests.test_qwen_open6dor_reasoning_json`
  - 已通过；`qwen_inference` 的 2 个解析测试在当前本地环境下按依赖情况跳过，不影响语法与代码集成检查
- 当前结论：
  - 这轮改动是直接针对 `subset400` 的主错误点，不涉及新训练，也不重改 verifier 主逻辑
  - 下一步应在服务器上复跑同一份 `subset400 task list`，重点看：
    - `error_count` 是否下降
    - `used_stage5_count` 是否上升
    - `watch_upright / tape_measure_upright / blade_right / handle_* / ballpoint_right` 是否不再大面积 `fallback_required`

## 28. 2026-05-06 已转为低成本 follow-up 子集，不再优先复跑 subset400
- 变更原因：
  - 用户明确不希望继续为 `400-case` 全量复跑付费
  - `subset400` 首轮结果已经足够定位 blocker，继续整批复跑性价比低
- 本轮新增文件：
  - `sofar/tools/generate_open6dor_followup_subsets.py`
- 本地已生成的新产物：
  - `sofar/paper_results/open6dor_short_experiments_20260429/open6dor_error_replay_50_from_subset400.json`
  - `sofar/paper_results/open6dor_short_experiments_20260429/open6dor_error_replay_50_from_subset400_task_list.json`
  - `sofar/paper_results/open6dor_short_experiments_20260429/open6dor_error_replay_50_from_subset400_summary.json`
  - `sofar/paper_results/open6dor_short_experiments_20260429/open6dor_error_replay_50_from_subset400_command.txt`
  - `sofar/paper_results/open6dor_short_experiments_20260429/open6dor_paper_core_120_seed42.json`
  - `sofar/paper_results/open6dor_short_experiments_20260429/open6dor_paper_core_120_seed42_task_list.json`
  - `sofar/paper_results/open6dor_short_experiments_20260429/open6dor_paper_core_120_seed42_summary.json`
  - `sofar/paper_results/open6dor_short_experiments_20260429/open6dor_paper_core_120_seed42_final_method_command.txt`
- 子集设计：
  - `error_replay_50`：直接复用 subset400 首轮的全部 `50` 个 error case，用于低成本验证 runtime taxonomy 与 reasoning JSON 修正
  - `paper_core_120`：从已完成的 `subset400` 中抽取 `120` 条低成本论文主结果子集，仅保留 `upright_vertical / flat_upside_down_lying_flat / plug_right` 三个 family，各 `40` 条
  - `paper_core_120` 显式排除了：
    - `cap/clip/sideways`
    - subset400 首轮已经报错的样本
- 当前结论：
  - 下一步不再优先复跑 `400-case`
  - 服务器侧优先顺序改为：
    - 先跑 `error_replay_50`
    - 再跑 `paper_core_120`
  - 只有这两步结果足够稳定时，才考虑是否回到更大规模 formal evaluation

## 29. 2026-05-06 已撤销本地 OOM 特化修改，改回评测前代码
- 背景：
  - 用户已切换到 `V100` 服务器
  - 不再保留此前为 `3090 paper_core_120` 首轮 OOM 临时加的低显存 / OOM retry 特化逻辑
- 本轮回退文件：
  - `sofar/serve/qwen_inference.py`
  - `sofar/open6dor/open6dor_perception.py`
  - `tests/test_qwen_open6dor_reasoning_json.py`
- 已撤销内容：
  - `qwen_inference.py` 中的 `use_cache=False` 默认改动
  - `qwen_inference.py` 中的 OOM 后短 token 重试
  - `open6dor_perception.py` 中的 OOM 专用 CUDA cache 清理逻辑
  - 对应的本地 OOM 回归测试
- 保留不变的内容：
  - runtime taxonomy 扩充
  - reasoning JSON 提取增强
  - 三专家 routing
  - verifier 与 semantic-axis 接回逻辑
- 当前结论：
  - 下一步仍按低成本顺序执行：
    - 先跑 `error_replay_50`
    - 再跑 `paper_core_120`

## 30. 2026-05-06 same-subset evaluator ablation runner 已在本地补齐
- 本轮本地新增文件：
  - `sofar/analysis/run_open6dor_subset_ablation.py`
  - `tests/test_open6dor_ablation_runner.py`
- 已实现内容：
  - 在同一个 `task_list` 上按 method 生成独立输出目录：
    - `baseline_only`
    - `pscr_rule_v2_safe`
    - `pscr_rule_v3_verified`（缺实现时仅预留并跳过）
    - `pscr_shadow`
    - `pscr_direct_no_verify`（缺安全开关时仅预留并跳过）
  - 支持：
    - `--dry-run`
    - `--eval-only`
    - `--skip-existing`
    - `--max-tasks`
    - `task_list first-N` 切片并写入独立临时清单
  - 不改 `eval_open6dor.py` 公式；通过 subset mirror dataset root 把每个 method 的 `result.json` 接回官方 evaluator 口径
  - 汇总输出：
    - `ablation_summary.json`
    - `ablation_summary.csv`
    - `error_breakdown.csv`
    - `commands_dry_run.txt / commands_executed.txt`
- 本地验证状态：
  - `python -m py_compile sofar/analysis/run_open6dor_subset_ablation.py tests/test_open6dor_ablation_runner.py`
  - `python -m unittest tests.test_open6dor_ablation_runner`
  - 两项均已通过
- 当前结论：
  - 这一步只补 evaluator 闭环与 same-subset ablation runner
  - 不改 Stage5 policy / agent policy / evaluator 公式 / 训练代码
  - 默认交接与 smoke 命令均限制为 `--max-tasks 20` 或 `50`，不在本地阶段直接跑完整 `400-case`
