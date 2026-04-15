# Baseline Stage 4 Tracker

## 阶段目标
- Stage 4 的唯一主问题：**能否构建稳定可训练的 object/part point data 表示？**
- 本阶段重点是验证点云缓存、采样逻辑和点数据质量，不直接宣称训练收益。
- `PSCR` 主线会把 Stage 4 产物作为后续 Stage 5 的训练输入。

## 当前状态
- 更新时间：2026-04-15
- 结论：**Stage 4 逻辑能跑，但点数据质量和缓存新鲜度都还没到可收口状态。**

## 本轮最新结果

### Open6DOR Stage 4（10-case）
- 命令：`python open6dor/open6dor_perception.py --stage4-pointdata-only --limit 10 --speed-profile conservative`
- 表面结果：
  - `success = 10`
  - `error = 0`
- 但这批结果**不能直接信**，因为：
  - 同一批 10 个任务里，`Stage 3 Open6DOR` 刚刚有 `3/10` 明确失败
  - 可 `Stage 4` 却在这 3 个失败任务上仍然给出了 `success`
  - 这 3 条还出现了非常整齐的：
    - `object_point_count = 4096`
    - `part_point_count = 4096`
    - `part_ratio = 1.0`
- 高概率解释：
  - `Stage 4` 吃到了**旧的 Stage 3 cache**
  - 也就是说，这轮 `Stage 4 Open6DOR 10-case` 混入了历史缓存，不能代表当前代码状态

### SpatialBench Stage 4（10-case）
- 命令：`python spatialbench/eval_spatialbench.py --stage4-pointdata-only --limit 10 --speed-profile conservative`
- 结果：
  - `success = 10`
  - `error = 0`
- 但当前也暴露出一个明确问题：
  - sample `id = 4`
  - `object_point_count = 715`
  - `part_point_count = 1067`
  - `part_ratio = 1.4923`
- 这说明：
  - `part_points` 没有被严格限制在 `object_points` 内
  - 当前 `part_mask ⊆ object_mask` 的一致性还不可靠

## 当前判断
- `Open6DOR Stage 4`：
  - 当前 10-case **不可信**
  - 问题不是“有没有跑通”，而是“是不是新鲜结果”
- `SpatialBench Stage 4`：
  - 结构上已经能跑
  - 但点数据质量还没过关，至少存在 `part_ratio > 1` 的一致性问题
- `Stage 4` 整体：
  - 现在**不该继续补 50-case**
  - 因为当前 10-case 已经暴露出“缓存污染”和“mask 一致性”两个核心问题

## 当前最需要修的点
1. `Open6DOR Stage 4` 需要在 fresh cache 条件下重跑
2. `SpatialBench Stage 4` 需要确保 `part_mask` 被限制在 `object_mask` 内

## 下一步
- **先不要跑 Stage 4 的 50-case。**
- 先做这两件事：
  1. 清理或隔离 `Open6DOR` 当前 10-task slice 的旧 `stage3/stage4` cache，再重跑 `Stage 3 -> Stage 4`
  2. 修 `part_mask ⊆ object_mask` 的一致性约束后，重跑 `SpatialBench Stage 4 10-case`
- 推荐的下一轮验证顺序：
  1. `Open6DOR Stage 3 10-case`（修完后）
  2. `Open6DOR Stage 4 10-case`（fresh cache）
  3. `SpatialBench Stage 4 10-case`（修完 mask 一致性后）
- 只有当这些 `10-case` 都变干净后，才值得讨论要不要补 `50-case`
