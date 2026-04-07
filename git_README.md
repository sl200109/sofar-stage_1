# Git 同步说明

这个文件记录 `D:\桌面\sofar实验同步` 目录最常用的 Git 版本管理命令。

## 仓库位置

本地仓库根目录：

```powershell
D:\桌面\sofar实验同步
```

GitHub 远程仓库：

```text
https://github.com/sl200109/sofar-stage_1.git
```

## 每次改完代码后的标准流程

### 1. 进入仓库目录

```powershell
cd D:\桌面\sofar实验同步
```

### 2. 查看当前改动

```powershell
git status
```

### 3. 查看具体改了什么

```powershell
git diff
```

### 4. 把改动加入暂存区

如果这次改动都要提交：

```powershell
git add .
```

如果只想提交部分文件：

```powershell
git add sofar\serve\utils.py
git add sofar\spatialbench\eval_spatialbench.py
git add baseline_stage1_tracker.md
```

### 5. 提交一个版本记录

```powershell
git commit -m "fix: reduce DBSCAN memory spikes in spatialbench"
```

### 6. 推送到 GitHub

```powershell
git push
```

## 最短常用版本

如果你已经确认当前改动都需要保存：

```powershell
cd D:\桌面\sofar实验同步
git add .
git commit -m "说明这次改了什么"
git push
```

## 推荐的提交信息写法

常见前缀：

- `fix:` 修 bug
- `feat:` 加功能
- `docs:` 改文档
- `chore:` 维护类修改

示例：

```powershell
git commit -m "fix: stabilize spatialbench resume after OOM kill"
git commit -m "feat: add progress checkpoint for open6dor perception"
git commit -m "docs: update stage1 tracker and handoff notes"
```

## 如果远程仓库有新改动

先拉取，再推送：

```powershell
git pull --rebase
git push
```

## 查看历史版本

```powershell
git log --oneline
```

## 查看某个文件的历史

```powershell
git log --oneline -- sofar\spatialbench\eval_spatialbench.py
```

## 当前仓库默认忽略的内容

这些目录默认不会上传到 GitHub：

- `sofar/checkpoints/`
- `sofar/datasets/`
- `sofar/output/`

这是故意保留的设置，避免大模型权重、数据集和运行输出把仓库撑爆。

## 第一次推送或远程异常时可检查

查看远程仓库地址：

```powershell
git remote -v
```

如果远程地址不对：

```powershell
git remote set-url origin https://github.com/sl200109/sofar-stage_1.git
```

## 安全目录问题

如果 Git 提示 `detected dubious ownership`，执行：

```powershell
git config --global --add safe.directory "D:/桌面/sofar实验同步"
```

## 一句话记忆版

```powershell
git status
git add .
git commit -m "说明这次改了什么"
git push
```
