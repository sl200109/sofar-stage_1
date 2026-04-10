# Baseline Stage 3 Tracker

## 阶段目标
- Stage 3 的唯一主问题：**先找物体、再找部件是否有效？**
- 本阶段先完成本地骨架与缓存层开发，不做服务器效果收口。
- `PSCR` 仍是主创新；`Fast Open6DOR` 只挂靠 Stage 3 的视觉基础设施，不抢主问题。

## 当前状态
- 更新时间：2026-04-10
- 状态：Stage 3 骨架已打通，但当前同步回来的结果需要按真实前提重新解读。
- 当前已知事实：
  - SpatialBench `50 case` 首轮结果：`success = 7`、`error = 43`。
  - 这批失败里大部分是 `CUDA OOM`，但用户已明确说明当时为了省钱让两个 test 同时跑，因此该结论**不能直接当作 Stage 3 自身显存不稳的最终判断**。
  - Open6DOR `50 case` 首轮结果：`success = 48`、`partial = 2`、`error = 0`。
  - 但这批 Open6DOR Stage 3 结果建立在旧的 Stage 2 fast parser 输出之上；现已确认旧 parser 在不少样本里把 `picked_object` 与参考物体反了，因此这批结果只能视为**临时 smoke**，不能作为正式基线。
- 当前判断：
  - Stage 3 的代码骨架、缓存格式和手动入口都已经就位。
  - SpatialBench 需要在**单独占卡**条件下重跑，才能判断显存与 part grounding 的真实稳定性。
  - Open6DOR 需要在 Stage 2 target/reference 对齐修复后重新跑 Stage 3，旧结果不作为正式结论。

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

## 当前约定
- `functional_part` 为空时，允许跳过 part grounding。
- object grounding 失败时，样本记录为 `object-stage failure`。
- object grounding 成功但 part grounding 失败时，样本记录为 `partial`，并保留 object 结果。
- Fast Open6DOR 分支当前只对 `picked object` 做重点 grounding，`reference object` 默认轻处理。

## 下一步
- 先完成 Stage 2 Open6DOR 对齐修复后的短重跑。
- 然后重跑：
  - `python open6dor/open6dor_perception.py --stage3-grounding-only --limit 10 --speed-profile conservative`
  - 稳定后再跑 `--limit 50`
- SpatialBench Stage 3 需要在**不并发其他 test** 的条件下单独重跑，再判断是否真有显存问题。
