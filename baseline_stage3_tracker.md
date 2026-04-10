# Baseline Stage 3 Tracker

## 阶段目标
- Stage 3 的唯一主问题：**先找物体、再找部件是否有效？**
- 本阶段先完成本地骨架与缓存层开发，不做服务器效果收口。
- `PSCR` 仍是主创新；`Fast Open6DOR` 只挂靠 Stage 3 的视觉基础设施，不抢主问题。

## 当前状态
- 更新时间：2026-04-09
- 状态：本地骨架开发已完成，等待上传服务器做 Stage 2/3 smoke。
- 本轮已完成的本地工作：
  - 固化 Stage 2 -> Stage 3 的输入接口，不改 Stage 2 主字段语义。
  - 在现有入口中补入 `stage3-grounding-only` 模式，不新增独立脚本。
  - 为 SpatialBench 与 Open6DOR 接入 object grounding + part grounding 骨架。
  - 定义统一的 Stage 3 缓存格式：`object_part_cache.json`、`object_mask.npz`、`part_mask.npz`、`roi_meta.json`。
  - 为 Fast Open6DOR 接出最小对象集合与非对称感知挂点：目标物体重处理，参考物体轻处理。

## 本轮代码修改
- 代码文件：
  - `sofar/serve/stage3_grounding.py`
  - `sofar/spatialbench/eval_spatialbench.py`
  - `sofar/open6dor/open6dor_perception.py`
- 修改目的：
  - 把 Stage 3 从“想法”变成可上传即测的缓存骨架。
  - 保证后续 Stage 4 可直接读取 Stage 3 缓存，不必重做 grounding。
  - 保证服务器侧仍靠手动命令执行，不需要额外工具脚本。

## Stage 3 缓存格式
- 每个样本/任务目录下至少包含：
  - `object_part_cache.json`
  - `object_mask.npz`
  - `part_mask.npz`
  - `roi_meta.json`
- `object_part_cache.json` 当前包含：
  - 样本级元信息
  - parser 输出摘要
  - `object_bbox_xyxy` / `part_bbox_xyxy`
  - `object_score` / `part_score`
  - `image_size`
  - `status` / `failed_stage`
  - 分阶段耗时

## 关键行为约定
- `functional_part` 为空时，允许跳过 part grounding。
- object grounding 失败时，样本记录为 `object-stage failure`。
- object grounding 成功但 part grounding 失败时，样本记录为 `partial`，并保留 object 结果。
- Fast Open6DOR 分支当前只对 `picked object` 做重点 grounding，`reference object` 默认轻处理。

## 服务器上传后手动执行
- Stage 2 parser 先做 smoke：
  - `python spatialbench/eval_spatialbench.py --stage2-parser-only --limit 20 --speed-profile conservative`
  - `python open6dor/open6dor_perception.py --stage2-parser-only --pilot open6dor10 --limit 10 --speed-profile conservative`
- SpatialBench Stage 3 grounding smoke：
  - `python spatialbench/eval_spatialbench.py --stage3-grounding-only --limit 20 --speed-profile conservative`
- SpatialBench 扩大到 100 条：
  - `python spatialbench/eval_spatialbench.py --stage3-grounding-only --limit 100 --speed-profile conservative`
- Open6DOR Stage 3 grounding smoke：
  - `python open6dor/open6dor_perception.py --stage3-grounding-only --pilot open6dor10 --limit 5 --speed-profile conservative`
- Open6DOR open6dor10 全 pilot：
  - `python open6dor/open6dor_perception.py --stage3-grounding-only --pilot open6dor10 --speed-profile conservative`

## 预期输出
- SpatialBench：
  - `output/stage3_spatialbench_grounding_records.json`
  - `output/stage3_spatialbench_grounding_records.csv`
  - `output/stage3_spatialbench_cache/<id>/...`
- Open6DOR：
  - `output/stage3_open6dor_grounding_records_open6dor10.json`
  - `output/stage3_open6dor_grounding_records_open6dor10.csv`
  - `<task_dir>/output/stage3/...`

## 验收标准
- Stage 2 对外 schema 没被改坏，Stage 3 能直接消费。
- object grounding / part grounding / cache 导出链路稳定。
- `partial` 与 `error` 能被明确区分，不会把失败样本吞掉。
- Stage 4 不需要再猜 Stage 3 的字段含义。

## 下一步
- 上传 Stage 2/3 修改到服务器。
- 先做 parser smoke，再做 grounding smoke。
- 确认 cache 字段与失败标签稳定后，再进入 Stage 4 point-data 构建。
