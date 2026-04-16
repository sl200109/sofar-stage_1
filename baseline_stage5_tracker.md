# Baseline Stage 5 Tracker

## 阶段目标
- 训练并验证 `Part-conditioned Orientation Head`
- 判断单域训练、混合训练、迁移训练哪条线成立
- 将各自最优 checkpoint 回接到各自 pipeline，并验证完整推理链
- 在工程接入通过后，开始做小规模效果测试

## 当前结论
- 更新时间：2026-04-16
- `Open6DOR-only`：成立，可作为 `Open6DOR` 单域最优 head
- `SpatialBench-only`：成立，而且当前是最好的训练结果，可作为 `SpatialBench` 单域最优 head
- `Combined / Balanced / Filtered / Curriculum`：都不成立，统一共享 head 的跨域训练当前阶段性判负

## 已完成
### 1. Stage 5 训练链打通
- `Stage 4 cache -> manifest -> Dataset/DataLoader -> head -> train/eval`
- 本地 CPU dry-run：通过
- 云服务器 CUDA dry-run：通过

### 2. 单域训练结果
#### Open6DOR-only
- `best_val_loss = 0.376927`
- `test mean_cosine = 0.289814`
- `test mean_angle_deg = 67.6826`
- 结论：初步有效，能学到方向信号

#### SpatialBench-only
- `best_val_loss = 0.223075`
- `test mean_cosine = 0.554985`
- `test mean_angle_deg = 49.1498`
- 结论：当前最好，泛化明显强于 `Open6DOR-only`

### 3. 统一训练负结果链
#### Combined
- `test mean_cosine = 0.121857`
- `test mean_angle_deg = 81.2177`

#### Balanced Combined
- `test mean_cosine = -0.021954`
- `test mean_angle_deg = 92.5007`

#### Filtered Combined
- `test mean_cosine = -0.040750`
- `test mean_angle_deg = 93.0818`

#### Curriculum: SpatialBench -> Open6DOR
- `Open6DOR test mean_cosine = 0.019723`
- `Open6DOR test mean_angle_deg = 89.2241`

### 4. 单域最优 checkpoint 回接已完成
- 新增：
  - `sofar/serve/stage5_inference.py`
- 新增 runtime path：
  - `SOFAR_STAGE5_OPEN6DOR_CKPT`
  - `SOFAR_STAGE5_SPATIALBENCH_CKPT`
- `Open6DOR` 新增 full pipeline 开关：
  - `--use-stage5-head`
  - `--stage5-checkpoint`
  - `--stage5-device`
  - `--stage5-num-points`
  - `--rerun-existing`
- `SpatialBench` 新增 full pipeline 开关：
  - `--use-stage5-head`
  - `--stage5-checkpoint`
  - `--stage5-device`
  - `--stage5-num-points`
- `SpatialBench` full eval 现在也支持 `--limit`

### 5. Open6DOR 单域回接 smoke 已通过
- 汇总文件：`sofar/output/stage5_open6dor_pipeline_records.json`
- `20/20 success`
- `stage5_applied = 20/20`
- `error_count = 0`
- `pipeline_mode = joint` 覆盖 `20/20`
- 当前平均单样本耗时约 `16.33s`
- 结论：`Stage 5` 单域最优 head 已稳定接入 `Open6DOR` 完整推理链

### 6. SpatialBench 单域回接 smoke 已跑通，但效果未起色
- 汇总文件：`sofar/output/eval_spatialbench_progress.json`
- `processed_samples = 20`
- `failed_samples = 0`
- `position_relative_accuracy = 0.75`
- `position_absolute_accuracy = 0.5`
- `orientation_relative_accuracy = 0.0`
- `orientation_absolute_accuracy = 0.0`
- `total_accuracy = 0.3`
- 日志中已确认 `Stage 5` 预测被注入：
  - 出现 `[spatialbench] sample <id> stage5 direction=[...]`
- 结论：工程接入是通的，但当前这版 head 接入后对 `SpatialBench orientation` 还没有带来正向效果

### 7. SpatialBench 当前无提升的诊断
- 当前 `Stage 5` head 的训练目标是：预测单个目标对象的 `3D direction vector`
- 但 `SpatialBench` 当前这批 `orientation` 题里，很多并不是“单对象方向预测”任务，而是：
  - 角度估计：如 camera 与 ground 的夹角
  - 两对象角差：如两辆车、叉子与刀、佛像与大门
  - 开合角度：如 door opened degree
  - 平行判断：如两扇门是否平行
  - 计数/转弯类：如 road 的 turns
- 这意味着：
  - `Stage 5` 当前 head 的输出空间和 `SpatialBench orientation` 的题目需求并不一致
  - 即使 scene graph 里已经注入了 `stage5 direction`，Qwen 也未必能把“一个对象的方向向量”转化成“两个对象的角差 / 开合角 / 平行关系 / 转弯数量”
- 额外还有两个结构性问题：
  - 多实例同名目标时，当前注入逻辑默认命中第一个同名节点，例如“两辆车”问题很容易注错对象
  - `vqa_reasoning_prompt` 明确写了 scene graph 只是参考，模型仍可能主要依赖图像本身
- 当前最稳判断：
  - `SpatialBench` 不适合直接把这版 `Stage 5` head 全量接入整套 orientation 题
  - 如果继续做 `SpatialBench`，更合理的是：
    - 先做“适用题型筛选 / gating”
    - 只在“单目标、单方向语义、可由 object/part 点云支持”的题型里启用 Stage 5
    - 对角度比较 / 多对象关系 / 开合角 / 平行类题，先保留原 pipeline

### 8. SpatialBench 题型筛选与 Stage 5 专用推理已接入
- 已新增题型筛选 helper：
  - `sofar/serve/spatialbench_stage5.py`
- 已在 `eval_spatialbench.py` 中接入：
  - 先按题型和 parser 输出判断是否适合 `Stage 5`
  - 只有适合子集才执行 `predict_from_stage4_dir(...)`
  - 不再对所有题盲目注入 `Stage 5`
- 已新增更强的 Stage 5 reasoning prompt：
  - `vqa_reasoning_stage5_prompt`
  - 明确告诉模型：
    - `stage5_head` 是 orientation evidence
    - 对适用题型优先结合 scene graph orientation 和 `stage5_evidence`
- 已将方向向量语言化后写入 scene graph：
  - `stage5_summary`
  - `stage5_evidence.readable_summary`
  - `stage5_evidence.axis_hint`
- 当前按“只看问题文本”的粗筛统计：
  - `orientation total = 96`
  - `single_object_direction`：`14`
  - `angle_or_relation`：`40`
  - `count_or_quantity`：`21`
  - `route_or_navigation`：`10`
  - `unsupported_orientation_semantics`：`11`
- 这说明：
  - 当前 `Stage 5` 天然只适合 `SpatialBench orientation` 里的一个小子集
  - 后续正确方向不是全量硬接，而是“子集验证 -> 再决定是否细化模块”

## 当前主线判断
### 成立的主线
- `Open6DOR`：`Stage5 Open6DOR-only best checkpoint`
- `SpatialBench`：`Stage5 SpatialBench-only best checkpoint`

### 当前不再优先推进
- 新的 mixed training 变体
- 新的 curriculum 变体

## 下一步
### 最高优先级
- `Open6DOR`：开始做小规模效果测试
  - 先扩到 `100-case`
  - 继续保留集中汇总文件，便于对比 `stage5_applied / target_orientation / error`
- `SpatialBench`：先做效果诊断，不盲目扩量
  - 当前先看 `20-case` 为什么 orientation 全错
  - 若需要，再做和 baseline 的同样本对照

### 次优先级
- 统一 `Open6DOR stage5_target_orientation` 的键名归一化
  - 例如 `upright / top_up / bottom_down / plug_right / silver plug end`
- 若 `Open6DOR 100-case` 结果稳定，再决定是否扩更大规模
- `SpatialBench` 若继续推进，先做：
  - `orientation` 题型细分
  - `Stage 5` 适用子集 smoke
  - 同名多实例目标注入的 disambiguation

### 更后续
- 如仍想研究统一模型，再转向：
  - `dataset-specific conditioning`
  - `shared trunk + dataset-specific small head`
