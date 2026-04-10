# Baseline Stage 4 Tracker

## 阶段目标
- Stage 4 的唯一主问题：**能否构建稳定可训练的 object/part point data 表示？**
- 本阶段先完成本地 point-data 骨架与缓存格式开发，不做训练与精度结论。
- `PSCR` 主线使用 Stage 4 产物构建训练数据；`Fast Open6DOR` 只复用其中的 lightweight 字段。

## 当前状态
- 更新时间：2026-04-10
- 状态：Stage 4 逻辑本身已经打通，但现有 smoke 结果仍受 Stage 3 上游状态影响。
- 当前已知事实：
  - SpatialBench `50 case` 首轮结果：`success = 7`、`error = 43`、平均耗时约 `0.14s`。
  - 这 43 条失败都来自缺少 Stage 3 cache，因此不是 Stage 4 自己坏了。
  - Open6DOR `10 case` 首轮结果：`success = 10`、`error = 0`、平均耗时约 `0.44s`。
  - 但这批 Open6DOR Stage 4 point cache 也是建立在旧的 Stage 2/3 parser 结果上，因此当前只能视为**结构通路验证**，不能直接当正式数据基线。
- 当前判断：
  - Stage 4 的点数据构建与缓存格式可用。
  - SpatialBench 需要等待 Stage 3 单卡复核后再看是否能稳定扩大规模。
  - Open6DOR 需要等 Stage 2 对齐修复、Stage 3 重跑后，再生成可信的 Stage 4 缓存。

## 当前缓存格式
- 每个样本/任务目录下至少包含：
  - `point_data_cache.json`
  - `object_points.npz`
  - `part_points.npz`
- `point_data_cache.json` 当前包含：
  - Stage 3 cache 引用路径
  - parser 输出摘要
  - grounding 摘要
  - `geometry_priors`
  - 采样点数配置

## 下一步
- 先不要把旧 Open6DOR Stage 4 point cache 当成正式数据。
- 待 Stage 2 对齐修复后，重跑 Open6DOR Stage 3，再重跑：
  - `python open6dor/open6dor_perception.py --stage4-pointdata-only --limit 10 --speed-profile conservative`
  - 稳定后再跑 `--limit 50`
- SpatialBench Stage 4 继续依赖新的 Stage 3 单卡结果，不单独背锅。
