# Baseline Stage 2 Tracker

## 阶段目标
- Stage 2 的唯一主问题：**部件级指令拆解是否稳定可用。**
- 本阶段只完成 parser 层开发与结构化记录，不做训练。
- `PSCR` 是主创新，`Fast Open6DOR` 只是辅助解析分支，不能抢主线结论。

## 当前状态
- 更新时间：2026-04-09
- 状态：本地开发已完成，等待上传服务器做 smoke 与人工抽检。
- 本轮已完成的本地工作：
  - 明确 Stage 2 统一 schema。
  - 在现有代码中补入 Part-aware parser prompt。
  - 在现有代码中补入 Fast Open6DOR parser prompt。
  - 补入 parser JSON 清洗、字段补全、reference frame / relation / confidence 归一化逻辑。
  - 在现有入口中补入 `stage2 parser-only` 手动执行模式，不新增独立脚本。
  - 规划好服务器侧 smoke 命令与输出文件。

## 本轮代码修改
- 代码文件：
  - `sofar/serve/system_prompts.py`
  - `sofar/serve/qwen_inference.py`
  - `sofar/spatialbench/eval_spatialbench.py`
  - `sofar/open6dor/open6dor_perception.py`
- 修改目的：
  - 用统一 schema 支撑 `PSCR` 主线 parser。
  - 为 `Fast Open6DOR` 提供受限的 task-aware parser。
  - 保证后续服务器侧只需手动执行命令，不再临时写新脚本。

## Stage 2 统一 schema
- 主线 parser 最少输出：
  - `target_object`
  - `functional_part`
  - `relation`
  - `reference_frame`
- 主线 parser 可选补充：
  - `reference_object`
  - `direction_attributes`
  - `parser_confidence`
  - `raw_text`
- Fast Open6DOR 辅助字段：
  - `picked_object`
  - `related_objects`
  - `orientation_mode`
  - `relation`
  - `reference_frame`
  - `routing_hints`
  - `parser_confidence`

## 信息边界
- 允许：
  - `instruction`
  - `RGB-D / image`
  - `task_config`，但仅用于 fast path 的 `task-aware routing / schema extraction`
- 默认不允许：
  - `directory / task id` 作为主输入

## 服务器上传后手动执行
- SpatialBench parser smoke：
  - `python spatialbench/eval_spatialbench.py --stage2-parser-only --limit 20 --speed-profile conservative`
- SpatialBench 扩大到 100 条：
  - `python spatialbench/eval_spatialbench.py --stage2-parser-only --limit 100 --speed-profile conservative`
- Open6DOR fast parser smoke：
  - `python open6dor/open6dor_perception.py --stage2-parser-only --pilot open6dor10 --limit 10 --speed-profile conservative`
- Open6DOR open6dor10 全 pilot：
  - `python open6dor/open6dor_perception.py --stage2-parser-only --pilot open6dor10 --speed-profile conservative`

## 预期输出
- SpatialBench：
  - `output/stage2_spatialbench_parser_records.json`
  - `output/stage2_spatialbench_parser_records.csv`
- Open6DOR：
  - `output/stage2_open6dor_parser_records_open6dor10.json`
  - `output/stage2_open6dor_parser_records_open6dor10.csv`

## 验收标准
- 主线 parser 输出字段稳定，不再频繁缺 key 或 schema 漂移。
- Fast Open6DOR parser 的 `routing_hints` 可直接支持后续 fast path 判断。
- SpatialBench 100 条与 Open6DOR open6dor10 的 parser 输出可直接人工抽检。
- Stage 3 不需要再改 Stage 2 schema。

## 下一步
- 上传本轮修改到服务器。
- 先跑 20 条 SpatialBench smoke 和 10 条 Open6DOR smoke。
- 确认输出字段稳定后，再扩到 100 条人工抽检集。
