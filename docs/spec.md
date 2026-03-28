# VH-01 MVP Spec

## 1. 目标

本 MVP 的目标不是一次性完整实现 [temp-02.md](./temp-02.md) 与 [temp-03.md](./temp-03.md) 中的全部理论对象，而是尽可能小成本地验证如下主线是否能够在工程上跑通：

$$
X_{1:T}
\to
Z_{1:T}
\to
R_{i,j,l}
\to
\text{条件相关局部动态更新}
\to
\text{更稳定的条件生成结果}.
$$

更具体地说，MVP 需要验证四件事：

1. 能学到稳定的潜表示 `Z_{1:T}`；
2. 能以多起点、多跨度的残差 `R_{i,j,l}` 训练出动态一致性；
3. 条件 `c` 能通过局部动态修正项真正影响生成；
4. 在简单任务上，条件更新与动态一致性能够带来可观测的效果提升。

## 2. 非目标

当前阶段不作为 MVP 目标的内容包括：

1. 显式完整恢复 `\mathcal M_T`；
2. 显式求解全局测度 `\mu_c` 或 `\nu_c`；
3. 显式完整参数化无穷小生成元 `\mathcal L_c`；
4. 直接接入大规模真实视频数据集；
5. 完整实现纤维丛、图册或全局几何恢复。

这些对象在 MVP 中保留为理论解释，而不是第一版工程交付内容。

## 3. 当前选择的 MVP 任务

为了尽快验证主线，MVP 采用合成视频任务，而不直接上真实视频数据。

### 3.1 数据形式

每个样本是一段短视频：

$$
X_{1:T}\in \mathbb R^{T\times C\times H\times W},
$$

默认设定为：

1. `T = 8`
2. `H = W = 32`
3. `C = 3`

### 3.2 条件形式

条件 `c` 采用离散组合式属性条件，由若干子属性组成，例如：

1. 形状：`square` / `circle`
2. 颜色：`red` / `green` / `blue` / `yellow`
3. 水平运动方向：`left` / `right`
4. 垂直运动方向：`up` / `down`
5. 尺寸：`small` / `large`
6. 速度：`slow` / `medium` / `fast`
7. 运动类型：`linear` / `sin-x` / `sin-y`
8. 背景类型：`plain` / `gradient` / `grid`

这样做的理由是：

1. 条件可控；
2. 正负条件容易构造；
3. 能直接测试“条件是否真正参与动态更新”。

### 3.3 合成视频分布

每段视频包含单个前景目标在简单背景上的运动。目标的位置、速度、尺寸、颜色、形状与局部轨迹形态由条件与少量随机变量共同决定。这样可以让：

1. 条件 `c` 对视频结构有真实影响；
2. 动态结构足够简单，便于快速训练；
3. GPU 测试时能在很短时间内得到可观测结果。
4. 条件不会完全决定轨迹，从而保留“部分信息条件”的理论设定。

## 4. MVP 模型结构

当前 MVP 采用“半显式局部结构路线”，不直接完整参数化 `\mathcal L_c`，而是只参数化当前最关键的局部更新入口。

### 4.1 核心模块

MVP 至少包含以下模块：

1. `ConditionEncoder`
   把离散条件 `c` 编码成条件嵌入 `h_c`。

2. `FrameEncoder`
   把单帧 `x_t` 编码为潜状态 `z_t`。

3. `FrameDecoder`
   把潜状态 `z_t` 解码回单帧。

4. `BaseDynamics`
   表示无条件或弱条件下的基准动力 `\phi_0`。

5. `ConditionDeltaDynamics`
   表示条件相关局部修正项 `\Delta\phi_\theta(z,c)`。

6. `VideoDynamicsModel`
   组合以上模块，形成
   $$
   \phi_c(z,c)=\phi_0(z)+\Delta\phi_\theta(z,c).
   $$

7. `ConditionEnergyHead`（扩展版）
   在保留 `\Delta\phi_\theta(z,c)` 主线的同时，用一个显式条件能量头近似承载
   `\Phi_{\mathrm{cond}}(Z;c)`，
   使条件支持强度不再只由固定距离定义。

### 4.2 当前最推荐的条件入口

MVP 第一版不直接实现

$$
\mathcal L_c f
=
b_c\cdot\nabla f + \frac12 \operatorname{tr}(A_c\nabla^2 f),
$$

而是优先实现

$$
\phi_c(z,c)=\phi_0(z)+\Delta\phi_\theta(z,c).
$$

这是当前最合理的折中，因为：

1. 它仍然保留“条件改变局部生成结构”的理论核心；
2. 它与 `R_{i,j,l}` 的训练目标直接相连；
3. 它比显式生成元实现明显更轻量。

## 5. MVP 训练目标

MVP 第一版采用简化但与理论一致的训练目标：

$$
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{base}}
+ \alpha \mathcal L_{\mathrm{rep}}
+ \beta \mathcal L_{\mathrm{dyn}}
+ \gamma \mathcal L_{\mathrm{cond}}
+ \eta \mathcal L_{\mathrm{reg}}.
$$

### 5.1 基础目标

在当前 MVP 中，`\mathcal L_{\mathrm{base}}` 直接采用视频重建损失或帧重建损失，用于保证模型有基本的视频表示与生成能力。

### 5.2 表示目标

MVP 中的 `\mathcal L_{\mathrm{rep}}` 采用轻量近似，而不强行实现完整的局部 Jacobian 一致性。第一版优先采用：

1. 潜表示平滑项；
2. 邻时刻潜增量稳定项；
3. 或轻量局部一致性项。

在扩展版中，可进一步加入局部线性一致性近似项 `\mathcal L_{\mathrm{loc}}`，用有限差分方式逼近

$$
J_D(z_t,c)(z_{t+1}-z_t),
$$

从而把 [temp-02.md](./temp-02.md) 与 [temp-03.md](./temp-03.md) 中的局部线性化直觉真正写入训练。

### 5.3 动力一致性目标

MVP 核心结构损失为：

$$
\mathcal L_{\mathrm{dyn}}^{\mathrm{norm}}
:=
\frac{
\sum_l\sum_{1\le i<j\le T}\omega_{ij}\|R_{i,j}^{(l)}\|^2
}{
\sum_l\sum_{1\le i<j\le T}\omega_{ij}
}.
$$

并采用课程式权重：

1. 前期偏重短跨度；
2. 中后期逐步提高长跨度权重。

### 5.4 条件更新目标

当前 MVP 推荐采用组合式条件目标：

$$
\mathcal L_{\mathrm{cond}}
=
\mathcal L_{\mathrm{nce}}
+ \lambda_{\mathrm{gap}}\mathcal L_{\mathrm{gap}}.
$$

其中：

1. `\mathcal L_{\mathrm{nce}}` 提供稳定的条件相对排序；
2. `\mathcal L_{\mathrm{gap}}` 在困难负样本上增强分离；
3. 负条件采用课程式难度策略，从随机负样本逐步增加困难负样本比例。

在扩展版中，条件相对排序可不再依赖固定的欧氏距离，而改由可学习的条件能量头给出，以更贴近

$$
\Phi_{\mathrm{cond}}(Z;c) + \lambda E_{\mathrm{dyn}}(Z;c)
$$

这一路线的工程化表达。

### 5.5 正则项

MVP 中 `\mathcal L_{\mathrm{reg}}` 采用轻量版本，优先约束：

1. 条件修正项 `\Delta\phi_\theta` 的幅度；
2. 条件修正项在相邻时间或相邻样本上的平滑性；
3. 潜空间动力更新的稳定性。

## 6. 训练阶段

MVP 采用四阶段训练组织：

### Stage 1: 表示预热

优化：

$$
\mathcal L_{\mathrm{base}} + \alpha \mathcal L_{\mathrm{rep}}.
$$

目标：先学到稳定的编码、解码与基本潜表示。

### Stage 2: 动力结构学习

在 Stage 1 基础上，引入：

$$
\beta \mathcal L_{\mathrm{dyn}}.
$$

目标：学到多起点、多跨度动态一致性。

### Stage 3: 条件更新学习

在 Stage 2 基础上，引入：

$$
\gamma \mathcal L_{\mathrm{cond}}
+ \eta \mathcal L_{\mathrm{reg}}.
$$

目标：让条件真正影响局部动态更新。

### Stage 4: 联合微调

联合优化全部项，以获得更平衡的最终模型。

## 7. 工程结构

MVP 第一版计划采用如下工程结构：

```text
.
├── .venv/
├── docs/
│   ├── spec.md
│   ├── temp-02.md
│   └── temp-03.md
├── pyproject.toml
├── README.md
├── configs/
│   └── mvp.yaml
├── scripts/
│   └── train_mvp.py
└── src/
    └── vh_mvp/
        ├── __init__.py
        ├── config.py
        ├── utils/
        ├── data/
        ├── models/
        ├── losses/
        └── train/
```

## 8. 依赖与运行环境

MVP 第一版采用 Python + PyTorch 路线。

### 8.1 运行环境要求

1. 使用项目根目录下的 `.venv`；
2. 在 `.venv` 内安装依赖；
3. 默认支持 CUDA GPU；
4. 训练脚本需要自动检测 `cuda` / `cpu`。

### 8.2 首批依赖

第一版建议安装：

1. `torch`
2. `torchvision`
3. `numpy`
4. `pydantic` 或 `pyyaml`
5. `tqdm`
6. `einops`
7. `matplotlib`

如果后续需要结果可视化或日志系统，再增量引入更重的依赖。

## 9. 真实数据接入约定

当前代码已经预留了 `folder` 数据模式，对应配置示例见 [realdata.example.yaml](/home/void0312/AIGC/VH-01/configs/realdata.example.yaml)。

### 9.1 目录形式

推荐的第一版真实数据组织方式为：

```text
data/example_dataset/
├── train.jsonl
├── val.jsonl
└── samples/
    ├── sample_0001/
    │   ├── 000.png
    │   ├── 001.png
    │   └── ...
    └── sample_0002/
        ├── 000.png
        ├── 001.png
        └── ...
```

### 9.2 manifest 格式

`train.jsonl` 或 `val.jsonl` 中每一行是一个样本记录，最小格式如下：

```json
{"id":"sample_0001","frames_dir":"samples/sample_0001","condition":{"shape":0,"color":2,"dir_x":1,"dir_y":0,"size":1,"speed":2,"motion":0,"background":1}}
```

当前版本中，`condition` 仍然要求与现有条件编码维度一致，也就是需要提供这 8 个离散字段。这是为了让真实数据入口先与 MVP 训练栈兼容。

后续若接入真正的文本条件或更复杂标注，再把这一步替换为更通用的条件编码器即可。

### 9.3 当前接入策略

真实数据接入的第一步，不是立刻改模型，而是先完成：

1. 把真实样本整理成 `frames_dir + condition` 的 manifest；
2. 确保每个样本至少有 `seq_len` 帧；
3. 先以当前离散条件格式跑通训练入口；
4. 再决定是否把条件分支扩展成文本或其他模态。

## 10. 成功标准

MVP 完成的最低标准不是“理论全部实现”，而是满足以下检查项：

1. 训练脚本可在 `.venv` 内启动；
2. 模型可在 GPU 上完成至少一个短训练过程；
3. 重建损失下降；
4. 动态一致性损失 `\mathcal L_{\mathrm{dyn}}` 下降；
5. 条件准确率 `cond_acc` 高于随机猜测基线；
6. 正条件与负条件的能量间隔 `energy_gap` 出现可观测分离；
7. 训练过程能够输出历史日志、CSV 曲线与样本图；
8. 在简单条件下生成的视频结构明显优于无条件或错误条件版本。

## 11. 当前最推荐的工程起点

结合理论成熟度与实现难度，当前最推荐的工程起点是：

> 以合成视频数据为载体，使用轻量视频自编码器 + 基准动力模块 + 条件动态修正模块，先把 `R_{i,j,l}` 与条件更新主线跑通，再把测度、候选集与生成元解释作为训练后的分析层。

因此，下一步工程实践的优先级应当是：

1. 建立 `.venv`；
2. 安装首批依赖；
3. 搭建合成数据集；
4. 搭建最小模型与训练循环；
5. 完成第一轮 smoke test。
