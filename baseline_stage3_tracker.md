# Baseline Stage 3 Tracker

## 阶段目标
- Stage 3 的唯一主问题：**先找物体、再找部件是否有效？**
- 本阶段先验证 `Dual-stage Grounding` 的结构链、缓存格式和失败模式，不直接宣称精度提升。
- `PSCR` 仍是主创新；`Fast Open6DOR` 只复用 Stage 3 的视觉基础设施。

## 当前状态
- 更新时间：2026-04-15
- 结论：**Stage 3 还不能结束，但这轮 10-case 已经足够说明现在不该继续跑 50-case。**

## 本轮最新结果

### Open6DOR Stage 3（10-case）
- 命令：`python open6dor/open6dor_perception.py --stage3-grounding-only --limit 10 --speed-profile conservative`
- 结果：
  - `success = 7`
  - `partial = 0`
  - `error = 3`
- 3 条失败全部发生在 `object_grounding`：
  - `Place_the_USB_behind_the_cup_on_the_table.__plug_right`
  - `Place_the_USB_behind_the_lighter_on_the_table.__plug_right`（20240824-164221）
  - `Place_the_USB_behind_the_pen_on_the_table.__plug_right`
- 共同错误：
  - `Object grounding returned no usable bbox`
- 共同特征：
  - 都是 `USB`
  - 都是 `orientation_mode = plug_right`
  - 都是细长小物体 + 细长参考场景

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
  - 暂时不需要为了“证明能跑”再补 `50-case`
- `Open6DOR Stage 3`：
  - 还**不能收口**
  - 当前 10-case 已经暴露出系统性 object grounding 问题，所以继续跑 `50-case` 只会扩大同类失败，信息增量不高
- `Stage 3` 整体：
  - 还**不能结束**
  - 主要不是样本数不够，而是 `Open6DOR` 的失败模式已经足够明确，应该先修再测

## 当前最需要修的点
1. `Open6DOR Stage 3` 对 `plug_right` 类任务的 object grounding 不稳
2. `Stage 2 Open6DOR` 的 `orientation_mode` 仍未收口；这会继续污染 `Stage 3` 的 part query / routing

## 下一步
- **先不要跑 Stage 3 的 50-case。**
- 先做这两件事：
  1. 修 `Stage 2 orientation_mode`
  2. 修 `Open6DOR Stage 3` 的 `plug_right` object grounding
- 修完后优先重跑：
  - `python open6dor/open6dor_perception.py --stage3-grounding-only --limit 10 --speed-profile conservative`
- 只有当这轮 `10-case` 重新达到：
  - `Open6DOR success = 10/10`
  - 且失败模式不再集中出现在 `plug_right`
  才值得重新评估要不要补 `50-case`
