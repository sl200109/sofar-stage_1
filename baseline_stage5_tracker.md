# Baseline Stage 5 Tracker

## 阶段目标
- Stage 5 的唯一主问题：**part-aware orientation 输入能否真实进入训练链，并完成最小 dry-run？**
- 当前阶段目标不是证明最终精度提升，而是把 `Stage 4 cache -> manifest -> Dataset/DataLoader -> Part-conditioned Orientation Head -> loss/backward` 这条链路接通。

## 当前状态
- 更新时间：2026-04-15
- 结论：**Stage 5 本地骨架已搭起，并完成一次最小 dry-run。**

## 本轮新增实现
### 1. Stage 4 smoke manifest
- 新增工具：
  - [stage5_manifest.py](D:\桌面\sofar实验同步\sofar\serve\stage5_manifest.py)
- 已生成：
  - [stage5_manifest_spatialbench_smoke.jsonl](D:\桌面\sofar实验同步\sofar\output\stage5_manifest_spatialbench_smoke.jsonl)
  - [stage5_manifest_open6dor_smoke.jsonl](D:\桌面\sofar实验同步\sofar\output\stage5_manifest_open6dor_smoke.jsonl)
  - [stage5_manifest_summary.json](D:\桌面\sofar实验同步\sofar\output\stage5_manifest_summary.json)

### 2. Stage 5 Dataset / DataLoader
- 新增：
  - [Stage4PointCache.py](D:\桌面\sofar实验同步\sofar\orientation\datasets\Stage4PointCache.py)
- 当前能力：
  - 读取 `jsonl manifest`
  - 加载 `object_points.npz` / `part_points.npz`
  - 组合为固定点数输入
  - 输出：
    - `points`
    - `target_direction`
    - `prior_vector`
    - `instruction`
    - `meta`

### 3. Part-conditioned Orientation Head 最小入口
- 新增：
  - [PartConditionedOrientationHead.py](D:\桌面\sofar实验同步\sofar\orientation\models\PartConditionedOrientationHead.py)
  - [stage5_dry_run.py](D:\桌面\sofar实验同步\sofar\orientation\stage5_dry_run.py)
- 说明：
  - 当前是**最小 dry-run 版**，目标是验证训练链路，不是最终定稿架构
  - 不依赖外部 CLIP 权重下载，便于本地和服务器快速 smoke

## 本地 dry-run 结果
- 命令口径：
  - `python orientation/stage5_dry_run.py --repo-root ... --prefer-dataset spatialbench --batch-size 2 --num-points 1024 --max-samples 4 --device cpu`
- 结果文件：
  - [stage5_dry_run_summary.json](D:\桌面\sofar实验同步\sofar\output\stage5_dry_run_summary.json)
- 当前结果：
  - `manifest_path = stage5_manifest_spatialbench_smoke.jsonl`
  - `dataset_size = 4`
  - `batch_size = 2`
  - `num_points = 1024`
  - `device = cpu`
  - `loss = 0.470613`
- 这说明：
  - manifest 读取正常
  - Dataset/DataLoader 正常
  - 模型 forward 正常
  - loss/backward/optimizer.step 正常

## 当前边界
- 现在已经能证明：
  - Stage 5 训练入口是可实现、可执行的
  - Stage 4 cache 已足够支持最小训练 smoke
- 现在还不能证明：
  - 最终 benchmark 性能已经提升
  - 当前 dry-run 的监督目标仍是 smoke 用伪标签，不是最终正式训练标签

## 当前判断
- Stage 5 已经可以正式进入“服务器最小训练 dry-run”阶段
- 当前最值得做的不是再回头补 Stage 2/3/4，而是：
  - 把这套 Stage 5 代码传到云服务器
  - 直接在云服务器利用已有 Open6DOR `stage4` cache 重新生成 manifest
  - 跑一次 GPU 版最小 dry-run

## 下一步
1. 上传 Stage 5 新文件到云服务器
2. 在云服务器执行 `orientation/stage5_dry_run.py`
3. 观察：
   - manifest 是否能同时读到 Open6DOR / SpatialBench
   - batch 是否正常
   - loss 是否稳定
   - checkpoint / summary 是否正常落盘
