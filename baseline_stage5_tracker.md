# Baseline Stage 5 Tracker

## 阶段目标
- Stage 5 的唯一主问题：**在已有 Stage 4 point cache 的基础上，part-conditioned orientation head 能否完成第一轮小规模训练，并在未训练样本上形成可解释的测试结果？**

## 当前状态
- 更新时间：2026-04-15
- 结论：
  - `Stage 5 dry-run` 已完成并通过
  - 当前已不再是“只有 10-case cache”
  - 现在已经有足够的 `100-case` 级别样本支持第一轮 tiny-train pilot

## 已完成部分
### 1. Stage 5 训练与评测入口已具备
- 已有：
  - [stage5_manifest.py](D:\桌面\sofar实验同步\sofar\serve\stage5_manifest.py)
  - [Stage4PointCache.py](D:\桌面\sofar实验同步\sofar\orientation\datasets\Stage4PointCache.py)
  - [PartConditionedOrientationHead.py](D:\桌面\sofar实验同步\sofar\orientation\models\PartConditionedOrientationHead.py)
  - [stage5_dry_run.py](D:\桌面\sofar实验同步\sofar\orientation\stage5_dry_run.py)
  - [stage5_train_pilot.py](D:\桌面\sofar实验同步\sofar\orientation\stage5_train_pilot.py)
  - [stage5_eval_unseen.py](D:\桌面\sofar实验同步\sofar\orientation\stage5_eval_unseen.py)
- 已验证：
  - 本地 CPU dry-run 成功
  - 云服务器 GPU dry-run 成功

### 2. 当前训练标签与 loss 口径已固定
- 训练标签：
  - primary：`normalized geometry_priors.part_to_object_vector`
  - fallback：`orientation_mode template axis`
  - confidence：`train_label_confidence`
- 当前训练 loss：
  - `0.7 * cosine_direction_loss + 0.3 * smooth_l1_direction_loss`
  - 再乘 `train_label_confidence`

## 最新数据分析
### Open6DOR
- `Stage 3`: `100/100 success`
- `Stage 4`: `100/100 success`
- 点云质量：
  - `object_point_count` 平均 `4096.0`
  - `part_point_count` 平均 `2399.45`
  - `part_ratio` 平均 `0.5858`
  - `part_ratio > 1`：`0`
- 结论：
  - `Open6DOR` 当前可以直接作为第一轮训练数据源

### SpatialBench
- `Stage 3`: `97/100 success`, `3/100 error`
- `Stage 4`: `97/100 success`, `3/100 error`
- 3 条失败均来自上游缺失 `Stage 3 cache`
  - `id = 22, 92, 96`
- 点云质量：
  - `object_point_count` 平均 `3938.33`
  - `part_point_count` 平均 `1573.37`
  - `part_ratio` 平均 `0.3998`
  - `part_ratio > 1`：`0`
- 结论：
  - `SpatialBench` 当前也可以用于训练
  - 第一轮训练时可直接使用成功的 `97` 条，不必为了这 `3` 条卡住主线

## 当前判断
- `Open6DOR`：可用
- `SpatialBench`：可用
- `Combined`：可用
- 因此当前已经可以开始第一轮 `100-case` 级别的小规模训练

## 建议的训练顺序
### 第一优先级：Open6DOR-only
- 目标：
  - 验证 orientation-mode 较明确时，模型是否更容易学到方向
- 建议：
  - 先做 `Open6DOR-only` tiny-train pilot

### 第二优先级：SpatialBench-only
- 目标：
  - 看更加多样的 functional_part / relation 指令是否仍能收敛
- 建议：
  - 在 `Open6DOR-only` 跑通后单独做一次

### 第三优先级：Combined
- 目标：
  - 看两类数据混合后，模型是否能得到更稳的泛化

## 下一步
1. 上传最新版：
   - `sofar/serve`
   - `sofar/orientation`
   - `sofar/open6dor`
2. 在云服务器上开始首轮 tiny-train pilot
3. 训练完成后：
   - 查看 `stage5_tiny_train_summary.json`
   - 查看 `stage5_pilot_best.pth`
4. 再在 `test` split 上执行未训练样本评测
5. 同步：
   - `stage5_tiny_train_summary.json`
   - `stage5_test_eval_summary.json`
   回本地分析
