# Baseline Stage 2 Tracker

## 阶段目标
- Stage 2 的唯一主问题：**部件级指令拆解是否稳定可用。**
- 本阶段只完成 parser 层开发与结构化记录，不做训练。
- `PSCR` 是主创新，`Fast Open6DOR` 只是辅助解析分支，不能抢主线结论。

## 当前状态
- 更新时间：2026-04-15
- 结论：**Stage 2 可以正式结束。**
- 已确认的结果：
  - SpatialBench `50 case`：
    - `success = 49`
    - `error = 1`
    - 平均 `parser_confidence ≈ 0.9263`
    - `functional_part` 非空：`38/50`
    - `reference_object` 非空：`40/50`
    - `reference_frame` 分布：`object-centric = 45`，`scene-centric = 4`，空值 `1`
    - 单条失败为 sample `22`：`probability tensor contains either inf, nan or element < 0`
  - Open6DOR `50 case`（2026-04-15 15:03 最新重跑）：
    - `success = 50`
    - `error = 0`
    - 平均 `parser_confidence = 0.95`
    - `picked_object` 与目标物体 `50/50` 对齐
    - `relation = behind` 全部稳定输出
    - `orientation_mode` 与 `task_rotation_label` 对齐：`50/50`
- 当前阻塞点：
  - 旧的 target/reference 反转问题已经解决。
  - 旧的 `orientation_mode` 偏差也已经解决。
- 因此当前判断：
  - Stage 2 的**运行稳定性**成立。
  - Stage 2 的 **Open6DOR target object 对齐** 成立。
  - Stage 2 的 **Open6DOR orientation_mode 语义对齐** 也成立。
  - Stage 2 已满足进入 Stage 3/4 后续判断的前提条件。

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
  - **2026-04-15 晚间新增本地补丁**：
    - `stage2_fast_open6dor_parser` 现在会把 `rot_tag_detail / rotation_instruction` 一起送入 fast parser
    - `orientation_mode` 归一化层会优先对齐 `task_config.rot_tag_detail`
    - 已加入常见标签的规范化映射，如 `lying flat -> lying_flat`、`sideways -> lying_sideways`

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
- Open6DOR `orientation_mode` 必须与任务后缀语义基本对齐，尤其不能把大批 `lying_flat` 错写成 `upright`。
- Stage 3 不需要再改 Stage 2 schema。

## 下一步
- Stage 2 不再需要继续重复重跑。
- 重点切换到：
  - `Open6DOR Stage 3/4` 是否真的吃到了这轮新代码
  - `SpatialBench Stage 4` 的 `part_ratio > 1` 是否被修掉
- 只有在拿到新的 `Stage 3/4` 同步产物后，才能判断是否进入 Stage 5。
