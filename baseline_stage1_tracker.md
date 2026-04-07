# Baseline Stage 1 Tracker

## 阶段目标
- 以服务器 `/data/coding/SoFar` 为最终运行环境，跑通 baseline 单样例链路。
- 固化路径配置入口，避免 `scripts/` 和 `serve/` 中继续散落服务器绝对路径。
- 为后续 Open6DOR / SpatialBench 基线评测准备可直接执行的服务器清单。

## 当前状态
- 状态：进行中
- 更新时间：2026-04-02
- 本地策略：只改代码与文档，不安装依赖，不做真实模型运行。
- 服务器策略：上传本地修改到 `/data/coding/SoFar` 后执行 smoke test 和后续评测。
- 当前执行顺序：
  - 先让 `spatialbench/eval_spatialbench.py` 跑完
  - 再跑 `open6dor/open6dor_perception.py`
  - 再跑 `open6dor/eval_open6dor.py`
  - 最后跑 `python scripts/stage1_collect_baseline.py --hard-case-limit 100`
- 当前默认后端：默认走本地 Qwen，后续服务器命令不再需要执行 `unset SOFAR_LLM_BACKEND`

## 已完成
- 已确认参考仓库为 `qizekun/SoFar`。
- 已确认最终运行目录为服务器 `/data/coding/SoFar`。
- 已确认服务器是完整上游仓库，本地仅做修改后上传。
- 已确认本地不安装依赖。
- 已确认阶段 1 主目标是 baseline 单样例跑通并固化执行清单。
- 已完成当前本地仓库与上游结构差异排查。
- 已梳理当前仓库中的服务器相关路径与资源依赖。
- 已新增统一路径模块 `serve/runtime_paths.py`。
- 已将 `scripts/open6dor_demo.py`、`scripts/manipulation_demo.py`、`scripts/vqa_demo.py` 接入统一路径逻辑。
- 已将 `scripts/navigation_demo.py`、`scripts/get_mask.py`、`scripts/video_predict_orientation.py` 接入统一路径逻辑。
- 已清理 `scripts/qwen_demo.py` 的重复残留版本，并改为统一路径逻辑。
- 已将 `serve/pointso.py` 改为通过集中配置读取 PointSO 配置和 checkpoint。
- 已将 `open6dor/open6dor_perception.py`、`open6dor/eval_open6dor.py`、`spatialbench/eval_spatialbench.py` 接入统一路径逻辑。
- 已修复 `open6dor/eval_open6dor.py` 中 6-DoF 位置准确率重复计数问题（`pos.append(success)` 重复）。
- 已新增 `scripts/stage1_collect_baseline.py`，用于生成：
  - `output/baseline_results.json`
  - `output/hard_cases.json`
  - `output/baseline_errors.csv`（环境支持时额外导出 `.xlsx`）
- 已新增动态任务清单：
  - `baseline_stage1_todolist.json`
  - `scripts/stage1_todo.py`
- 已完成修改文件的静态语法检查：
  - `serve/runtime_paths.py`
  - `serve/pointso.py`
  - `scripts/open6dor_demo.py`
  - `scripts/manipulation_demo.py`
  - `scripts/vqa_demo.py`
  - `scripts/navigation_demo.py`
  - `scripts/get_mask.py`
  - `scripts/video_predict_orientation.py`
  - `scripts/qwen_demo.py`
  - `open6dor/open6dor_perception.py`
  - `open6dor/eval_open6dor.py`
  - `spatialbench/eval_spatialbench.py`
  - `scripts/stage1_collect_baseline.py`
  - `scripts/stage1_todo.py`
- 已读取 `So Far PSCR 实验设计报告`，确认当前主线仍是“先固化 baseline 与错误分桶”，不与后续 part-aware / confidence 模块并行改动。
- 已根据 2026-04-02 的服务器日志定位次级 smoke 首轮问题：
  - `scripts/manipulation_demo.py`：Qwen manipulation parsing 返回 `list[dict]` 时被旧归一化逻辑吞成空对象列表，后续 scene graph 为空导致 `interact_object_id` 越界。
  - `scripts/navigation_demo.py`：`segmentation/grounding_dino.py` 未自动注入仓库内 `GroundingDINO` 路径，触发 `ModuleNotFoundError: groundingdino`。
  - `scripts/vqa_demo.py`：当前停在深度模型初始化阶段，日志显示为手动中断，尚未确认存在代码级错误。
- 已在本地补丁修复：
  - `serve/qwen_inference.py`：兼容 manipulation / VQA 解析结果为 `list[dict]` 的 schema。
  - `scripts/manipulation_demo.py`：为空对象列表、空 scene graph、越界 object id 增加显式报错。
  - `segmentation/grounding_dino.py`：自动接入仓库内置 `GroundingDINO` 包路径。
- 已完成上述新增补丁的静态语法检查：
  - `serve/qwen_inference.py`
  - `scripts/manipulation_demo.py`
  - `segmentation/grounding_dino.py`
- 已根据 2026-04-02 的服务器日志确认次级 smoke 二轮结果：
  - `scripts/manipulation_demo.py`：已跑通并输出 `Result`，当前可视为 smoke 通过。
  - `scripts/vqa_demo.py`：失败原因为缺少 `checkpoints/metric_depth_vit_large_800k.pth`，属于资源缺失，不是当前脚本逻辑错误。
  - `scripts/navigation_demo.py`：通过导包阶段后，GroundingDINO 继续尝试加载 `bert-base-uncased`；若本地无对应目录则会回退联网下载，在受限环境下卡住/失败。
- 已根据 2026-04-02 的服务器日志确认次级 smoke 最新结果：
  - `scripts/manipulation_demo.py`：已跑通并输出 `Result`。
  - `scripts/vqa_demo.py`：在补齐 `metric_depth_vit_large_800k.pth` 后已跑通，完成 object parsing / SAM / depth estimation / scene graph / spatial reasoning。
  - `scripts/navigation_demo.py`：最新日志已确认直接阻塞于缺少 `checkpoints/groundingdino_swinb_cogcoor.pth`；`bert-base-uncased` 仅在该 checkpoint 补齐后、且本地无文本编码器缓存时才会成为后续潜在阻塞。
- 已继续在本地补丁固化资源路径：
  - `serve/runtime_paths.py`：新增 `SOFAR_METRIC3D_CKPT`、`SOFAR_GROUNDINGDINO_TEXT_ENCODER`。
  - `depth/metric3dv2.py`：改为通过统一路径定位 Metric3D checkpoint，并在缺失时给出明确报错。
  - `segmentation/GroundingDINO/groundingdino/util/get_tokenlizer.py`：优先使用本地 `bert-base-uncased` 目录，缺失时给出明确报错。
  - `scripts/navigation_demo.py`：接入统一路径逻辑与仓库根目录导入逻辑。
- 已记录服务器网络访问备选方案，供后续补 checkpoint / 模型目录时复用：
  - `export HF_ENDPOINT=https://hf-mirror.com` 后再执行 `snapshot_download(...)`
  - `wget -c https://hf-mirror.com/...`
  - `git clone https://ghfast.top/https://github.com/...`
- 已根据 2026-04-02 的最新服务器日志继续定位 `navigation_demo.py`：
  - 补齐 `groundingdino_swinb_cogcoor.pth` 后，错误从缺 checkpoint 推进为 `GroundingDINO` 自定义扩展 `_C` 未加载。
  - 原仓库 `ms_deform_attn.py` 已包含纯 PyTorch fallback，但分支判断只检查 `torch.cuda.is_available()` 与 `value.is_cuda`，未检查 `_C` 是否成功导入，导致在 GPU 张量场景下仍误走坏分支并触发 `NameError: _C is not defined`。
- 已在本地补丁修复：
  - `segmentation/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py`：当 `_C` 未成功导入时，强制回退到纯 PyTorch `multi_scale_deformable_attn_pytorch(...)` 路径。
- 已完成上述新增补丁的静态语法检查：
  - `segmentation/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py`
- 已根据 2026-04-02 的最新服务器日志确认 `scripts/navigation_demo.py` 已跑通：
  - 成功完成 GroundingDINO + SAM 检测流程。
  - 成功输出目标中心、朝向向量、与坐标轴夹角、转向角度与目标位置。
  - `manipulation_demo.py`、`vqa_demo.py`、`navigation_demo.py` 三个次级 smoke 已全部完成。
- 已为批量入口补充统一落盘行为：
  - `serve/batch_logging.py`：新增批量运行时间戳日志与 JSON 双写工具。
  - `open6dor/open6dor_perception.py`：运行时自动在根 `output/` 下保存时间戳日志，并输出批处理 summary（固定文件名 + 时间戳文件名）；单样本结果仍保存在各任务目录下的 `output/result.json`。
  - `open6dor/eval_open6dor.py`：运行时自动在根 `output/` 下保存时间戳日志，并将 `eval_pos.json`、`eval_rot.json`、`eval_6dof.json` 同时写为固定文件名和时间戳文件名。
  - `spatialbench/eval_spatialbench.py`：运行时自动在根 `output/` 下保存时间戳日志，并将 `eval_spatialbench.json` 同时写为固定文件名和时间戳文件名。
  - 同时修复 `open6dor/eval_open6dor.py` 的 6-DoF 位置准确率重复计数问题，避免 batch 指标偏高。
- 已完成上述新增补丁的静态语法检查：
  - `serve/batch_logging.py`
  - `open6dor/open6dor_perception.py`
  - `open6dor/eval_open6dor.py`
  - `spatialbench/eval_spatialbench.py`
- 已根据 2026-04-02 的服务器日志确认 `spatialbench/eval_spatialbench.py` 当前首个阻塞点：
  - 时间戳日志已正常写入 `output/eval_spatialbench_20260402_153701.log`，说明新的批量日志机制生效。
  - 实际失败原因为 `FileNotFoundError: /data/coding/SoFar/datasets/6dof_spatialbench/images/0.png`。
  - 结合本地同步目录可见 `datasets/6dof_spatialbench` 当前仅包含 `spatial_data.json`，说明 SpatialBench 数据集图片未就位，当前是数据集资源缺失而非评测主逻辑错误。
- 已根据 2026-04-02 的后续服务器日志确认批量评测最新状态：
  - `open6dor/open6dor_perception.py` 已成功写出时间戳日志与 summary，但 `tqdm` 显示 `0it`，说明当前 `open6dor_v2` 数据目录结构未匹配到任何任务目录，需继续核对数据是否完整解压到脚本预期层级。
  - `open6dor/eval_open6dor.py` 已成功运行并写出 `eval_pos/eval_rot/eval_6dof` 结果文件，但当前全为 0.0；结合 `open6dor_perception.py` 的 `0it`，高度怀疑是前置感知结果未生成而非评测逻辑本身出错。
  - `spatialbench/eval_spatialbench.py` 在图片数据补齐后继续推进，当前首个模型级阻塞变为 Qwen 视觉编码阶段 `torch.cuda.OutOfMemoryError`。
- 已在本地补丁修复 / 增强：
  - `spatialbench/eval_spatialbench.py`：为 Qwen 解析与推理增加按最大边缩放的降显存策略，并在 OOM 时自动降一档重试。
  - `spatialbench/eval_spatialbench.py`：单样本失败不再直接终止整个 batch，而是记录失败后继续后续样本，最终 summary 增加 `failed_samples` 字段。
- 已完成上述新增补丁的静态语法检查：
  - `spatialbench/eval_spatialbench.py`
- 已根据 2026-04-02 的最新服务器日志继续推进 batch 状态：
  - `spatialbench/eval_spatialbench.py` 目前已能正常逐样本执行，历史日志 `output/eval_spatialbench_20260402_155626.log` 显示已跑到约 `48/223` 后手动中断。
  - 已为 `spatialbench/eval_spatialbench.py` 新增两种续跑能力：
    - 正常断点续跑：读取 `output/eval_spatialbench_progress.json`
    - 一次性从旧日志恢复：`--recover-log /data/coding/SoFar/output/eval_spatialbench_20260402_155626.log`
  - `Open6DOR` 数据集现已下载到服务器；后续需在 SpatialBench 跑完后重新执行 `open6dor/open6dor_perception.py`，再判断目录层级与感知产物是否正常。
- 已完成上述新增补丁的静态语法检查：
  - `serve/runtime_paths.py`
  - `depth/metric3dv2.py`
  - `segmentation/GroundingDINO/groundingdino/util/get_tokenlizer.py`
  - `scripts/navigation_demo.py`

## Todo
- 上传当前修改到服务器 `/data/coding/SoFar`。
- 在服务器检查以下资源是否存在：
  - `/data/coding/SoFar/checkpoints/Qwen2.5-VL-3B`
  - `/data/coding/SoFar/checkpoints/Florence-2-base`
  - `/data/coding/SoFar/checkpoints/sam_vit_h_4b8939.pth`
  - `/data/coding/SoFar/checkpoints/small.pth` 或可替代的 `small_finetune.pth`
  - `/data/coding/SoFar/datasets/open6dor_v2`
  - `/data/coding/SoFar/datasets/6dof_spatialbench`
  - 当前补充：`6dof_spatialbench/images/` 已补齐；`open6dor_v2` 已下载，待通过 `open6dor/open6dor_perception.py` 重新验证目录层级。
- 在服务器执行 `scripts/open6dor_demo.py` 单样例 smoke test。
- 记录 `output/result.json`、`output/scene.npy`、`output/picked_obj.npy`、mask 可视化产物是否正常生成。
- 若 Open6DOR 需要 finetuned PointSO 权重，设置 `SOFAR_POINTSO_CKPT=small_finetune.pth` 后重新验证。
- 在服务器补做次级验证：
  - `scripts/manipulation_demo.py`
  - `scripts/vqa_demo.py`
  - `scripts/navigation_demo.py`
  - 当前状态：上述三个次级 smoke 已全部通过。
  - 说明：`manipulation_demo.py` 已通过；后续复跑前需先补传本地最新
    `serve/qwen_inference.py`、`scripts/manipulation_demo.py`、`segmentation/grounding_dino.py`、
    `serve/runtime_paths.py`、`depth/metric3dv2.py`、`segmentation/GroundingDINO/groundingdino/util/get_tokenlizer.py`、`scripts/navigation_demo.py`。
  - 说明：`vqa_demo.py` 已通过；相关资源要求为 `metric_depth_vit_large_800k.pth` 或 `SOFAR_METRIC3D_CKPT`。
  - 说明：`navigation_demo.py` 复跑前优先补齐：
    文件夹：`checkpoints`
    文件：`groundingdino_swinb_cogcoor.pth`
    服务器目标路径：`/data/coding/SoFar/checkpoints/groundingdino_swinb_cogcoor.pth`
  - 说明：若出现 `NameError: _C is not defined`，需补传本地补丁：
    文件夹：`segmentation/GroundingDINO/groundingdino/models/GroundingDINO`
    文件：`ms_deform_attn.py`
    服务器目标路径：`/data/coding/SoFar/segmentation/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py`
  - 说明：若补齐上述 GroundingDINO checkpoint 后仍出现文本编码器下载问题，再补齐：
    文件夹：`checkpoints/bert-base-uncased`
    文件：`bert-base-uncased` 模型目录
    服务器目标路径：`/data/coding/SoFar/checkpoints/bert-base-uncased/`
    或设置 `SOFAR_GROUNDINGDINO_TEXT_ENCODER` 指向其本地路径。
- 在服务器完整仓库中核对并固化批量评测入口的实际参数和结果目录：
  - `python open6dor/open6dor_perception.py`
  - `python open6dor/eval_open6dor.py`
  - `python spatialbench/eval_spatialbench.py`
  - 当前推荐执行顺序：
    - `python spatialbench/eval_spatialbench.py`
    - `python open6dor/open6dor_perception.py`
    - `python open6dor/eval_open6dor.py`
- 在服务器执行阶段一结果汇总：
  - `python scripts/stage1_collect_baseline.py --hard-case-limit 100`
  - 期望生成 `output/baseline_results.json`、`output/hard_cases.json`、`output/baseline_errors.csv`
- 用动态清单维护进度：
  - `python scripts/stage1_todo.py show`
  - `python scripts/stage1_todo.py set <task_id> --status done --notes "<备注>"`
- 后续每完成一项服务器操作，及时把对应事项从 `Todo` 挪到 `已完成`。

## 阻塞项
- 本地仓库不包含完整上游依赖模块，无法在本地做真实运行验证。
- 服务器上 PointSO 实际使用 `small.pth` 还是 `small_finetune.pth` 仍需你运行时确认。
- Open6DOR / SpatialBench 批量评测脚本的具体 CLI 参数需要在服务器完整仓库内按实际上游文件再核对一次。
- `spatialbench/eval_spatialbench.py` 当前仍未最终跑完，需先从已保存进度继续执行至完成。
- `open6dor/open6dor_perception.py` 需要在数据补齐后重新运行一次，确认不再出现 `0it`。

## 服务器交接清单
- 工作目录：
  - `cd /data/coding/SoFar`
- 可选环境变量：
  - `export SOFAR_ROOT=/data/coding/SoFar`
  - `export SOFAR_QWEN_PATH=/data/coding/SoFar/checkpoints/Qwen2.5-VL-3B`
  - `export SOFAR_OUTPUT_DIR=/data/coding/SoFar/output`
  - `export SOFAR_POINTSO_CKPT=small.pth`
- 若 Open6DOR 更适合使用 finetuned PointSO：
  - `export SOFAR_POINTSO_CKPT=small_finetune.pth`
- 服务器前检查：
  - `ls /data/coding/SoFar/checkpoints`
  - `ls /data/coding/SoFar/datasets/open6dor_v2`
  - `ls /data/coding/SoFar/datasets/6dof_spatialbench`
- 服务器网络下载备选：
  - `export HF_ENDPOINT=https://hf-mirror.com`
  - `wget -c https://hf-mirror.com/...`
  - `git clone https://ghfast.top/https://github.com/...`
- 单样例 smoke test：
  - `python scripts/open6dor_demo.py`
- 产物检查：
  - `ls /data/coding/SoFar/output/result.json`
  - `ls /data/coding/SoFar/output/scene.npy`
  - `ls /data/coding/SoFar/output/picked_obj.npy`
- 后续评测入口：
  - `python spatialbench/eval_spatialbench.py`
  - `python open6dor/open6dor_perception.py`
  - `python open6dor/eval_open6dor.py`
  - `python scripts/stage1_collect_baseline.py --hard-case-limit 100`
- SpatialBench 断点续跑：
  - 旧日志恢复一次：`python spatialbench/eval_spatialbench.py --recover-log /data/coding/SoFar/output/eval_spatialbench_20260402_155626.log`
  - 正常续跑：`python spatialbench/eval_spatialbench.py`

## 运行结果与产物
- 当前状态：
  - `scripts/open6dor_demo.py` 已跑通。
  - `scripts/manipulation_demo.py` 已跑通。
  - `scripts/vqa_demo.py` 已跑通。
  - `scripts/navigation_demo.py` 已跑通。
  - `spatialbench/eval_spatialbench.py` 已进入正式 batch，历史日志显示已跑到约 `48/223`，当前可从 progress 文件或旧日志恢复后继续。
  - `open6dor/open6dor_perception.py` 与 `open6dor/eval_open6dor.py` 需在数据补齐后重新运行，上一轮 `0it / 0.0` 不作为有效最终结果。
- 预期首批关键产物：
  - `/data/coding/SoFar/output/result.json`
  - `/data/coding/SoFar/output/scene.npy`
  - `/data/coding/SoFar/output/picked_obj.npy`
  - `/data/coding/SoFar/output/picked_obj_mask.npy`
  - `/data/coding/SoFar/output/sam_annotated_image.jpg`
