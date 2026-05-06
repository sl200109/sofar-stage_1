# Open6DOR Short Experiments

## 40-Case Verifier Ablation

| Metric | Before | After |
| --- | --- | --- |
| success_count | 40 | 40 |
| error_count | 0 | 0 |
| used_stage5_count | 14 | 29 |
| fallback_count | 26 | 11 |
| rejected_count | 15 | 0 |
| upright_direct_count | 8 | 8 |
| plug_right_conditional_count | 6 | 6 |
| lying_flat_accepted_count | 0 | 15 |
| lying_flat_fallback_count | 15 | 0 |
| old_rule_pass | 25 | 25 |
| new_rule_pass | 25 | 40 |
| verifier_rule_version_distribution | {"rule_v2_legacy": 25, "none": 15} | {"rule_v2_legacy": 25, "rule_v3_semantic_axis": 15} |

## Latency / Overhead

| Metric | Mean | Median | Max |
| --- | --- | --- | --- |
| total_sec | 17.0000 | 16.7600 | 19.2600 |
| qwen_joint_sec | 16.7300 | 16.4600 | 18.8600 |
| pcd_sec | 0.1400 | 0.1200 | 0.3500 |
| stage5_head_sec | 0.0000 | 0.0000 | 0.0000 |

结论: Stage5/agent overhead 相对 Qwen joint reasoning 很小, `stage5_head_sec` 远低于 `qwen_joint_sec`.

## Cap / Clip / Shadow-only

| Mode | Count | Selected Exec | Shadow Used | Shadow Accepted | Fallback |
| --- | --- | --- | --- | --- | --- |
| cap_left_bottom_right | 5 | {"stage5_shadow_only": 5} | 5 | 5 | 5 |
| cap_right_bottom_left | 4 | {"stage5_shadow_only": 4} | 4 | 4 | 4 |
| clip_sideways | 2 | {"stage5_shadow_only": 2} | 2 | 2 | 2 |

保守结论: cap/clip/sideways 当前保持 shadow-only 或 fallback, 不纳入强主结果 claim.

## 400-Case Subset

子集 manifest 已生成, 见同目录 JSON 文件.
Formal 400-case sampling is delegated to the server-side from4389 sampler.

## Low-Cost Follow-Up

- `open6dor_error_replay_50_from_subset400`
  - replay all 50 previous subset400 error cases
  - purpose: validate runtime taxonomy and reasoning JSON fixes cheaply
- `open6dor_paper_core_120_seed42`
  - 40 `upright_vertical`
  - 40 `flat_upside_down_lying_flat`
  - 40 `plug_right`
  - excludes `cap/clip/sideways`
  - excludes previous subset400 error cases

Current execution order:
1. run `error_replay_50`
2. if stable, run `paper_core_120`
