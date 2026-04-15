# Baseline Stage 5 Tracker

## 阶段目标
- Stage 5 的唯一主问题：**part-aware orientation 的训练链能否建立，并在有足够 Stage 4 cache 后进入正式小规模训练？**
- 当前必须明确：`Stage 5 dry-run` 已经通过，但**正式训练还缺训练数据量**。

## 当前状态
- 更新时间：2026-04-15
- 结论：**Stage 5 训练入口已建立，但当前优先级已经切换到“补充训练数据”，而不是直接拿 10 个 cache 开始训练。**

## 已完成部分
### 1. Stage 5 训练入口已成立
- 已有：
  - [stage5_manifest.py](D:\桌面\sofar实验同步\sofar\serve\stage5_manifest.py)
  - [Stage4PointCache.py](D:\桌面\sofar实验同步\sofar\orientation\datasets\Stage4PointCache.py)
  - [PartConditionedOrientationHead.py](D:\桌面\sofar实验同步\sofar\orientation\models\PartConditionedOrientationHead.py)
  - [stage5_dry_run.py](D:\桌面\sofar实验同步\sofar\orientation\stage5_dry_run.py)
  - [stage5_train_pilot.py](D:\桌面\sofar实验同步\sofar\orientation\stage5_train_pilot.py)
- 已验证：
  - 本地 CPU dry-run 成功
  - 云服务器 GPU dry-run 成功

### 2. 当前训练标签口径已初步定义
- primary label：
  - `normalized geometry_priors.part_to_object_vector`
- fallback label：
  - Open6DOR 用 `orientation_mode template axis`
  - 兜底用 `[0, 0, 1]`
- 同时引入：
  - `train_label_confidence`

## 当前关键问题
- 服务器上当前 `Stage 4` cache 规模仍然只有 `10-case` 级别
- 这个规模只够：
  - 验证训练链能不能跑
  - 验证 manifest / dataset / checkpoint / log 是否工作
- **不够做有意义的小规模正式训练**

## 当前判断
- 现在不应该继续重复 `Stage 5 dry-run`
- 现在也不应该直接拿当前 `10-case` cache 开始正式训练
- 当前最合理的下一步是：
  - **先扩 Stage 4 cache 覆盖范围**
  - 再进入 Stage 5 小规模正式训练

## 建议的数据扩展顺序
### 第一优先级：Open6DOR
- 原因：
  - orientation_mode 标签更明确
  - 与 PSCR 主线更贴近
- 建议先扩到：
  - `100-case`
- 执行口径：
  - `Stage 3/4` 现已改为**默认增量补跑**
  - 会优先跳过已有有效 cache 的任务
  - 因此后续 `--limit 100` 表示“补 100 个新的缺失任务”，而不是重算前面已完成的 10 个

### 第二优先级：SpatialBench
- 原因：
  - 可作为补充训练/验证来源
  - 总规模本来也不大
- 建议先扩到：
  - `100-case`
  - 后续再视情况扩到全量 `223`

## 下一步
1. 在云服务器继续执行 `Stage 3/4` 扩量，而不是训练
2. 把新增 `Stage 4` cache 同步回本地
3. 确认 manifest 中可用样本数达到更合理量级
4. 再启动 `stage5_train_pilot.py`
