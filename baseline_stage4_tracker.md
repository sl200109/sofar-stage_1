# Baseline Stage 4 Tracker

## 阶段目标
- Stage 4 的唯一主问题：**能否构建稳定可训练的 object/part point data 表示？**
- 当前口径：Stage 4 已经按工程 gate 收口，重点从“能不能生成 point cache”切换到“这些 cache 如何交给 Stage 5 使用”。

## 当前状态
- 更新时间：2026-04-15
- 结论：**Stage 4 已完成工程收口，并已完成向 Stage 5 的 smoke manifest 交接。**

## 已验证结果
### SpatialBench Stage 4
- `10-case`：`success = 10`，`error = 0`
- 关键修复已生效：
  - sample `id = 4` 的 `part_ratio` 已从 `1.4923` 降到 `0.8965`
  - 说明 `part_mask ⊆ object_mask` 的 containment 修补已经落地

### Open6DOR Stage 4
- `10-case`：`success = 10`，`error = 0`
- stale cache 假成功问题已解除
- 但本地当前**没有同步逐任务 `output/stage4` cache 目录**，只有 records 文件

## 新增交接状态
### Stage 4 -> Stage 5 manifest 交接
- 已新增 Stage 5 manifest builder：
  - [stage5_manifest.py](D:\桌面\sofar实验同步\sofar\serve\stage5_manifest.py)
- 本地已生成：
  - [stage5_manifest_spatialbench_smoke.jsonl](D:\桌面\sofar实验同步\sofar\output\stage5_manifest_spatialbench_smoke.jsonl)
  - [stage5_manifest_open6dor_smoke.jsonl](D:\桌面\sofar实验同步\sofar\output\stage5_manifest_open6dor_smoke.jsonl)
  - [stage5_manifest_summary.json](D:\桌面\sofar实验同步\sofar\output\stage5_manifest_summary.json)

### 当前 manifest 可用性
- SpatialBench：
  - `10/10` 本地可用
- Open6DOR：
  - `0/10` 本地可用
  - 原因不是 Stage 4 失败，而是本地未同步逐任务 `output/stage4/point_data_cache.json`、`object_points.npz`、`part_points.npz`

## 当前判断
- Stage 4 已经完成了“训练原料缓存层”
- 但还没有单独形成“大规模正式训练集”
- 当前最合理的用法是：
  - 先用 SpatialBench smoke cache 驱动 Stage 5 本地 dry-run
  - 再在云服务器上用已有 Open6DOR cache 继续扩展

## 下一步
- Stage 4 本地不再继续重复补跑
- 后续如需补充训练样本，优先在云服务器上直接使用已有 Open6DOR `stage4` 缓存构建 manifest
