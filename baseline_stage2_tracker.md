# Baseline Stage 2 Tracker

## 阶段目标
- Stage 2 的唯一主问题：**部件级指令拆解是否稳定可用。**
- 本阶段只完成 parser 层开发与结构化记录，不做训练。
- `PSCR` 是主创新，`Fast Open6DOR` 只是辅助解析分支，不能抢主线结论。

## 当前状态
- 更新时间：2026-04-10
- 结论：**Stage 2 还不能正式完结，但已经只差一次短重跑确认。**
- 已确认的结果：
  - SpatialBench `50 case`：
    - `success = 49`
    - `error = 1`
    - 平均 `parser_confidence ≈ 0.9263`
    - `functional_part` 非空：`38/50`
    - `reference_object` 非空：`40/50`
    - `reference_frame` 分布：`object-centric = 45`，`scene-centric = 4`，空值 `1`
    - 单条失败为 sample `22`：`probability tensor contains either inf, nan or element < 0`
  - Open6DOR `50 case`：
    - `success = 50`
    - `error = 0`
    - 平均 `parser_confidence = 0.95`
    - `relation = behind` 全部稳定输出
    - `orientation_mode`、`routing_hints` 均已正常落盘
- 当前阻塞点：
  - Open6DOR Stage 2 的语义检查发现 `picked_object` 与参考物体发生角色反转。
  - 以当前同步回本地的 50 条记录粗查，约 `26/50` 条存在“应为被操作物体，却被解析成参考物体”的问题。
  - 典型例子：`Place the USB behind the bottle...` 被解析成 `picked_object = bottle`、`related_objects = ["USB"]`。
- 因此当前判断：
  - Stage 2 的**运行稳定性**已经成立。
  - Stage 2 的 **Open6DOR fast parser 语义对齐** 还需要一次修正后重跑确认。
  - 在这次确认完成前，`Stage 3/4` 里基于旧 Open6DOR Stage 2 parser 产出的结果都只能视为**临时 smoke 结果**。

## 本轮代码修改
- 代码文件：
  - `sofar/serve/system_prompts.py`
  - `sofar/serve/qwen_inference.py`
  - `sofar/spatialbench/eval_spatialbench.py`
  - `sofar/open6dor/open6dor_perception.py`
- 最新补丁：
  - 已修复 Stage 2 CSV 导出只写表头的问题。
  - 已在 `qwen_inference.py` 中加入 Open6DOR `picked_object` 对 `task_config.target_obj_name` 的对齐修正。
  - 已在 `system_prompts.py` 中补强约束：若提供 `task_config.target_obj_name`，则 `picked_object` 必须与之严格一致。

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

## 预期输出
- SpatialBench：
  - `output/stage2_spatialbench_parser_records.json`
  - `output/stage2_spatialbench_parser_records.csv`
- Open6DOR：
  - `output/stage2_open6dor_parser_records.json`
  - `output/stage2_open6dor_parser_records.csv`

## 验收标准
- 主线 parser 输出字段稳定，不再频繁缺 key 或 schema 漂移。
- Fast Open6DOR parser 的 `routing_hints` 可直接支持后续 fast path 判断。
- Open6DOR `picked_object` 必须与 `task_config.target_obj_name` 对齐，不再出现大规模 target/reference 反转。
- Stage 3 不需要再改 Stage 2 schema。

## 下一步
- 先将最新的 `system_prompts.py` 与 `qwen_inference.py` 重新上传服务器。
- 手动重跑：
  - `python open6dor/open6dor_perception.py --stage2-parser-only --limit 50 --speed-profile conservative`
- 重跑后优先检查：
  - `picked_object` 是否与 `task_config.target_obj_name` 对齐
  - `related_objects` 是否保留真正参考物体
  - `stage2_open6dor_parser_records.csv` 是否完整生成
- 若这一步通过，再正式将 Stage 2 标为完成，并重新执行 Open6DOR 的 Stage 3/4 smoke。
