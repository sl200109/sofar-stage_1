# Baseline Stage 4 Tracker

## 阶段目标
- Stage 4 的唯一主问题：**能否构建稳定可训练的 object/part point data 表示？**
- 本阶段先完成本地 point-data 骨架与缓存格式开发，不做训练与精度结论。
- `PSCR` 主线使用 Stage 4 产物构建训练数据；`Fast Open6DOR` 只复用其中的 lightweight 字段。

## 当前状态
- 更新时间：2026-04-09
- 状态：本地骨架开发已完成，等待 Stage 3 服务器 smoke 后手动执行 point-data 构建。
- 本轮已完成的本地工作：
  - 在现有入口中补入 `stage4-pointdata-only` 模式，不新增独立脚本。
  - 定义 Stage 4 点数据缓存格式：`point_data_cache.json`、`object_points.npz`、`part_points.npz`。
  - 接入 object/part 采样与几何先验计算。
  - 保证 Stage 4 默认读取 Stage 3 cache，不需要重复做 grounding。

## 本轮代码修改
- 代码文件：
  - `sofar/serve/stage4_point_data.py`
  - `sofar/spatialbench/eval_spatialbench.py`
  - `sofar/open6dor/open6dor_perception.py`
- 修改目的：
  - 把 Stage 4 需要的点数据构建骨架先在本地搭好。
  - 统一 object/part 采样与 geometry priors 输出格式。
  - 让 Stage 4 后续可以直接喂给 Stage 5 训练或 Fast Open6DOR 的 lightweight 分支。

## Stage 4 缓存格式
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

## 当前几何先验
- 当前先导出：
  - `object_point_count`
  - `part_point_count`
  - `part_ratio`
  - `object_centroid`
  - `part_centroid`
  - `part_to_object_vector`
- 这轮只先钉死字段与缓存层，不做更复杂几何特征工程。

## 服务器上传后手动执行
- SpatialBench Stage 4 smoke（依赖 Stage 3 cache）：
  - `python spatialbench/eval_spatialbench.py --stage4-pointdata-only --limit 20 --speed-profile conservative`
- SpatialBench 扩大到 100 条：
  - `python spatialbench/eval_spatialbench.py --stage4-pointdata-only --limit 100 --speed-profile conservative`
- Open6DOR Stage 4 smoke（依赖 open6dor10 的 Stage 3 cache）：
  - `python open6dor/open6dor_perception.py --stage4-pointdata-only --pilot open6dor10 --speed-profile conservative`

## 预期输出
- SpatialBench：
  - `output/stage4_spatialbench_point_records.json`
  - `output/stage4_spatialbench_point_records.csv`
  - `output/stage4_spatialbench_point_cache/<id>/...`
- Open6DOR：
  - `output/stage4_open6dor_point_records_open6dor10.json`
  - `output/stage4_open6dor_point_records_open6dor10.csv`
  - `<task_dir>/output/stage4/...`

## 验收标准
- Stage 4 不重复依赖 detection/SAM，可直接从 Stage 3 cache 构建点数据。
- object / part point 数据字段稳定，后续 Stage 5 无需再猜缓存格式。
- geometry priors 字段可读、可追溯、能支持后续训练或辅助系统分支。

## 下一步
- 先完成 Stage 3 服务器 smoke 与 cache 抽检。
- 再执行 Stage 4 point-data smoke。
- 确认缓存字段稳定后，进入 Stage 5 数据读入与训练入口设计。
