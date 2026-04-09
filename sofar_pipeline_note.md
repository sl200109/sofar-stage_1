# on open6dor

## 6step pipeline

### Step 1：输入 RGB-D 图像和语言指令
一张RGB观测图和一个语言指令，语言指令是类似于比如“把杯子的把手朝右”“把刀刃朝向某个物体”这类 6-DoF 操作要求

### Step 2：让 VLM 提取“任务相关对象短语”
给定对应的自然语言query后，让VLM从自然语言指令里提取出一组与任务相关的对象短语，也就是识别下这条任务里到底有哪些相关的物体

### Step 3：对这些相关物体做分割，并构建对应的点云
根据上一步提取出的对象短语，结合 Florence-2 和 SAM，在图像中找到这些相关物体的位置并做分割，得到每个物体对应的 mask
然后再利用 RGB-D 信息，把这些物体依据depth数据从二维图像区域反投影成三维点云表示，点云数据是在线生成的 
这一步的结果是：系统已经知道“相关物体是谁、在哪、长什么样”。

note：florence2是一个通用VLM，负责看图后按照prompt做视觉任务，在本文中与SAM合作完成分割任务。
具体分工：
Florence-2：先根据文字找到候选区域/框
SAM：再把这个区域分得更细，输出 mask
why？ SAM 很强在“分割”，但它本身不擅长直接理解“文字里说的是哪个物体”。
Florence-2 可以做 文本驱动的视觉定位

### Step 4：为每个物体生成语义朝向描述，并用 PointSO 做朝向预测
这里只知道物体的位置还不够，因为 Open6DOR 任务关心的不只是“物体在哪”，还关心“物体应该朝哪边”。  
所以 SoFar 会继续让 VLM 生成和当前任务相关的语义朝向描述，比如 handle direction、opening direction 这种。  
然后再用 PointSO 根据“物体点云 + 朝向描述”预测该物体的语义朝向。

### Step 5：构建 6-DoF scene graph
把前面得到的信息整合起来，构建一个 6-DoF scene graph。  
这里面会包含：
- 相关物体的类别和实例
- 每个物体的三维位置
- 每个物体的尺寸信息
- 每个物体的语义朝向信息
- 物体和物体之间的相对关系

这一步相当于把原始图像，转成一个更结构化、更适合推理的场景表示。

### Step 6：把 scene graph 和图像再交给 VLM 做最终推理
最后，SoFar 会把 6-DoF scene graph、原始图像和语言指令一起输入给 VLM，做最终的空间推理。  
这里 VLM 要回答的问题是：
- 目标物体最后应该放到哪里
- 目标物体最后应该朝向哪里
- 最终的 6-DoF 变换结果是什么

也就是说，前面1-5步是在做“场景理解和结构化表示”，最后一步是把原始图像、自然语言指令、6dof场景图一起输入给 VLM，做最终的空间推理，然后系统再把这个输出解析成可执行的 6-DoF 表示。才是在做“真正的 6-DoF 推理”。

---

## 为什么它在 open6dor 上会慢

因为它不是一个“输入图像后一次前向直接输出结果”的模型，而是一条串行的多阶段 pipeline。

它慢主要慢在这几件事：

1. **VLM 不只调用一次**
   - 先提取相关对象短语
   - 再生成语义朝向描述
   - 最后还要做最终的空间推理

2. **不是只检测一次就结束**
   - 它要对相关物体做分割
   - 还要把这些物体构造成点云

3. **每个物体还要额外做朝向理解**
   - 不只是知道它是什么
   - 还要知道它的 handle / opening / blade 这类语义朝向

4. **最后还要构图再推理**
   - 先构建 6-DoF scene graph
   - 再把 graph 重新喂给 VLM 做高层推理


# 6-DoF Spatialbench数据集
spatial visual-question-answering benchmark，用来评估 orientation-aware reasoning。而且它不是操作任务，而是VQA 问答任务
它包含：
223 个人工标注样本
每个样本是一张 RGB 图 + 一个四选一问题
有两个 track：
position
orientation
覆盖 object counting、spatial relations、object-facing direction 等任务。
## 一句话记忆：6-DoF SpatialBench 更像“看图回答：这个物体朝哪边、谁在谁左边、朝向关系对不对”。

# open6dor数据集
是一个 6-DoF object rearrangement benchmark
## 输入：场景图像 / RGB-D + 语言指令
## 输出：目标物体的 最终 6-DoF 结果
## 关注点：
   目标物体位置对不对
   目标物体朝向对不对
   最后能不能完成物体重排/操作任务

---

## 当前代码中的新 pipeline

上面写的是更接近原始 SoFar baseline 的流程；当前仓库里的 `Open6DOR` 主链已经做了一些轻量优化，但核心组件没有变，主要变化如下。

### 1. 先裁剪最小对象集合，再进入感知
当前代码不会一开始就让大模型提取一长串相关对象，而是会先根据任务指令和 `task_info` 抽出最小必要对象集合，通常只保留：
- picked object
- 1 到 2 个 related objects

这样做的目的是减少 detection、SAM、scene graph 的负担。

### 2. 优先走一次 joint Qwen
原始流程更像是两次 VLM：
- 第一次抽取 picked object、related objects、direction attributes、target orientation
- 第二次根据 scene graph 做最终 target position reasoning

当前代码优先尝试一次 joint Qwen，尽量直接输出：
- picked_object
- related_objects
- direction_attributes
- target_orientation
- target_position

如果 joint 路径失败，再自动回退到旧的双阶段路径。

### 3. orientation 变成半模板化
当前增加了 orientation template 先验，不再完全依赖 Qwen 临场生成方向属性。  
例如常见物体会优先补充：
- mug -> handle / opening / bottom
- USB -> plug end / silver plug end / larger end
- apple -> top / bottom

所以现在更像是“模板先验 + PointSO + joint/fallback 推理”的混合方式。

### 4. 最终推理使用 lightweight scene graph
当前仍然保留 scene graph 这一步，但给最终推理喂入的是 lightweight scene graph，而不是一整套冗长的重文本图结构。  
它主要只保留最关键的：
- picked object 的 center / bbox / init_orientation
- related objects 的 center / bbox

这样做是为了缩短 prompt、提高推理稳定性，并降低实验迭代成本。

### 5. 新增了单案例 perception cache
当前代码为单个 Open6DOR case 增加了中间缓存，例如：
- `perception_cache.json`
- `mask_cache.npz`

这样后续如果只改 parser、reasoning、confidence 之类的上层逻辑，就可以优先复用 detection / SAM / scene graph 结果，不必每次从感知层全部重跑。

### 一句话总结
原始 SoFar 更像“重 scene graph + 双阶段 VLM 推理”，当前代码版更像“最小对象集合 + lightweight scene graph + 单次 joint Qwen + fallback”的轻量优化版。
