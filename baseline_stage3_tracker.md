# Baseline Stage 3 Tracker

## 阶段目标
- Stage 3 的唯一主问题：**先找物体、再找部件是否有效？**
- 本阶段先验证 `Dual-stage Grounding` 的结构链、缓存格式和失败模式，不直接宣称精度提升。
- `PSCR` 仍是主创新；`Fast Open6DOR` 只复用 Stage 3 的视觉基础设施。

## 当前状态
- 更新时间：2026-04-15
- 结论：**Stage 3 可以按当前工程 gate 正式结束。**

## 本轮最新结果

### Open6DOR Stage 3（10-case）
- 命令：`python open6dor/open6dor_perception.py --stage3-grounding-only --limit 10 --speed-profile conservative`
- 结果：
  - `success = 10`
  - `partial = 0`
  - `error = 0`
- 最新同步文件：
  - `stage3_open6dor_grounding_records_20260415_150410.json`
  - `stage3_open6dor_perception_20260415_150410.log`
- 变化：
  - 旧的 `USB + plug_right` 三条 `object_grounding` 失败已全部消失
  - `part_query` 现在统一收敛到 `silver plug end`
  - `plug_right` 的细长小物体场景已经被打通

### SpatialBench Stage 3（10-case）
- 命令：`python spatialbench/eval_spatialbench.py --stage3-grounding-only --limit 10 --speed-profile conservative`
- 结果：
  - `success = 10`
  - `partial = 0`
  - `error = 0`
- 这说明：
  - 之前 `50-case` 里的大量 OOM，更像是并发占卡造成的干扰
  - 在单独占卡条件下，SpatialBench 的 Stage 3 结构链已经可以稳定跑通

## 当前判断
- `SpatialBench Stage 3`：
  - 可以视为**结构 smoke 通过**
  - 当前无需再为了工程推进补 `50-case`
- `Open6DOR Stage 3`：
  - 当前 `10-case` 已经通过
  - 旧的 `plug_right object_grounding` 阻塞已经解除
- `Stage 3` 整体：
  - 可以按当前工程 gate **结束**
  - 已经足够支撑 Stage 4 和后续 Stage 5 的推进

## 当前最需要修的点
1. `Open6DOR Stage 3` 对 `plug_right` 类任务的 object grounding 不稳
2. `Stage 2 Open6DOR` 的 `orientation_mode` 仍未收口；这会继续污染 `Stage 3` 的 part query / routing

## 下一步
- 当前不再需要继续重复 Stage 3 的 `50-case`。
- 后续如果进入论文正式主表阶段，再考虑是否做更大规模复核。
- 现在可以把精力切换到 Stage 5。
