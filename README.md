# VLA 琶洲项目结合 · LingBot-VLA2 场景增强复现与优化

> 2026-11-02 · 琶洲算法二开大赛 A 赛道（二次开发技术创新）相关研究仓库

本仓库做三件事：
1. 复现 **LingBot-VLA 2**（`jason-xzs/lingbot-vla-v2`，分支 `feat/robotwin-scene-augmentation`）的 **场景增强模块**；
2. 把它从 torch 依赖中剥离并**优化**成可在普通 CPU 环境运行的确定性模块；
3. 在 **DGX（NVIDIA GB10）** 上实跑，并与本机做**像素级对齐验证**。

---

## 1. 先给结论

| 事项 | 结论 |
|---|---|
| 完整训练复现 | ❌ 不可行。仓库是 **6B 参数 VLA**，官方 **6×A800 80GB**，两阶段 5 万+1 万步；单台 GB10 做不了 |
| 算法模块复现 | ✅ 可行。`scene_augmentation.py` 只依赖 numpy/torch/scipy，只改像素、不动 state/action/mask |
| DGX 实跑 | ✅ 已跑通，比本机快约 2×，输出像素级一致（diff=0.0000） |
| 与赛题关系 | ⚠️ A 赛道是「二次开发实操」，现场离线、禁云端 AI，**不跑 VLA 大模型**；本仓库的价值是赛前**感知鲁棒性数据增强**与仿真泛化研究 |

---

## 2. 琶洲 A 赛道要点

- **任务**：轮式人形平台（轮式底盘 + 人形半身 + 双指夹爪），赛前夹稳空托盘 → 备餐台抓两份瓶装水装盘 → 端盘导航到客人桌 → 逐份上饮并语音通知。
- **硬约束**：现场离线、禁止联网/云端 AI；每队总赛时 90 分钟，正式运行 5 分钟/轮，最多两轮；平台只提供导航/机械臂/夹爪/视觉/语音基础接口。
- **评分**：基础任务 100 分（场景操作 75 + 任务五 15 + 任务六 10）。
- **判断**：这是传统机器人编程 + 视觉识别 + 任务调度，不是 VLA 推理赛道。VLA/场景增强只能作为赛前研究与数据工具。

---

## 3. 上游仓库分析

| 项目 | 配置 |
|---|---|
| 基座模型 | `lingbot-vla-v2-6b`（6B） |
| 训练硬件 | 6 × A800 80GB |
| 阶段一 | 0 → 50,000 步，全量微调，无图像增强 |
| 阶段二 | 50,000 → 60,000 步，场景增强 + seen 指令采样 |
| 权重合并 | θ = 0.25×θ_50k + 0.75×θ_60k |
| 增强权重 | original 0.40 / clutter 0.35 / texture 0.10 / scene 0.15 |
| 环境 | Python 3.12、PyTorch 2.8、CUDA 12.8、FlashAttention 2.8.3 |

场景增强分支的核心文件：`lingbotvla/data/vla_data/scene_augmentation.py`。

---

## 4. 本仓库做了什么

### 4.1 复现（`repro/scene_aug_numpy.py`）

把上游 `scene_augmentation.py` 移植为**纯 numpy 版**，保留全部语义：

- `SceneLibrary`：加载 clutter 库（tasks/meta/offsets/rgb/body/shade/episode_foreground/median）
- `generate_texture`：程序化纹理
- `head_background` / `wrist_background`：背景分割
- `_smooth_components`：连通域平滑
- `ScenePlan.apply`：纹理替换 + clutter 粘贴，**只改像素**

### 4.2 优化

| 编号 | 优化 | 效果 |
|---|---|---|
| O1 | 去 torch：`_upsample` 用 numpy 双线性替代 `F.interpolate` | 系统 Python + numpy/scipy/PIL 即可跑，无需 6B 环境与 CUDA |
| O2 | clutter 边缘羽化（`feather=2px`） | 消除 1px 硬拼接缝；`feather=0` 可完全复现原版 |
| O3 | 程序纹理 5 → 7 类（新增木纹、大理石） | 背景多样性提升 |
| O4 | 高斯斑点纹理向量化 | 去掉 python 循环，texture 分支稳定在 10ms 级 |

所有优化都保持「只改像素，state/action/mask 不变」的不变式，并保持同 seed 确定性。

### 4.3 效果（`repro/out/`）

- `synth_compare.png`：original / clutter / texture / scene 四分支
- `synth_opt.png`：scene 复现版 vs 优化版
- `duck_compare.png`：真实素材（microduck 鸭子渲染）增强前后对比
- `scene_aug_demo.mp4`：original | scene 增强，12 个随机 seed 扫描（15fps）
- `branch_cycle.mp4`：original → clutter → texture → scene 四分支循环

视频生成：`cd repro && python make_video.py`（依赖系统 ffmpeg；本机需设 `OPENBLAS_NUM_THREADS=1` 规避内存分配失败）。

---

## 5. 快速开始

### 本机（Windows）

```bash
cd repro
python demo.py
```

### DGX（Linux，纯 CPU）

```bash
ssh nvidia@<dgx-host>
cd /path/to/repro
DUCK_IMG=./duck_src.png /home/nvidia/anaconda3/bin/python demo.py
```

依赖：`numpy` `scipy` `pillow`（无需 torch / CUDA）。

---

## 6. DGX 实测

| 分支 | 本机 (Win, Py3.13) | DGX (GB10, anaconda Py3.12) | 加速 |
|---|---|---|---|
| original | 7.3 ms | 4.7 ms | ≈1.6× |
| clutter | 8.1 ms | 3.1 ms | ≈2.6× |
| texture | 21.7 ms | 10.5 ms | ≈2.1× |
| scene | 20.2 ms | 9.6 ms | ≈2.1× |

一致性：三张输出图在本机与 DGX 上**像素级完全一致**（identical=True，mean_abs_diff=0.0000）。

---

## 7. 与琶洲 A 赛道的结合建议

**能直接用的**
- 赛前数据增强：把现场备餐台/客人桌图像做 clutter/texture 增强，扩充视觉识别训练集。
- 仿真泛化验证：检验视觉模块在「多干扰物、换桌面纹理」下的稳定性。
- 离线可部署：纯 numpy 版无重型依赖，可直接进赛前离线环境。

**不能指望的**
- 现场不跑 VLA 大模型（赛题禁止联网/云端 AI）。
- 这套 6B 训练流程不是比赛程序——它解决「感知数据多样性」，不解决「端盘任务调度」。

**务实路线**：赛题主线用传统导航 + 视觉 + 状态机；VLA/场景增强放在赛前数据与鲁棒性研究，形成「算法研究 + 工程落地」双线叙事。

---

## 8. 目录结构

```
vla-pazhou/
├─ README.md
├─ repro/
│  ├─ scene_aug_numpy.py      # 纯 numpy 复现 + 优化版
│  ├─ demo.py                 # 四分支演示 + 鸭子素材演示
│  ├─ synthetic_library.npz   # 合成场景库
│  ├─ duck_src.png            # microduck 渲染素材
│  ├─ out/                    # 本机输出
│  └─ out_dgx/                # DGX 输出（像素级一致）
└─ site/
   ├─ index.html              # 展示页
   └─ img/                    # 效果图
```

---

## 9. 上游与许可

- 上游：https://github.com/jason-xzs/lingbot-vla-v2 （分支 `feat/robotwin-scene-augmentation`）
- 上游基础：Robbyant/LingBot-VLA 2 · RoboTwin 2.0
- 本仓库复现代码为独立实现，仅用于研究与赛前准备。
