# VH-01 理论到工程映射

## 1. 当前主链

当前工程主线应当按以下顺序理解：

$$
R_{i,j,l}
\rightsquigarrow
\text{局部响应几何}
\rightsquigarrow
(\,x,\;T_x\mathcal M_T,\;b_c,\;\Sigma_x,\;A_x,\;\rho_c\,)
\rightsquigarrow
\mathcal L_c
\rightsquigarrow
\mu_c.
$$

这意味着工程优先级也应当是：

1. 先让局部基点与局部切向对象处在同一语义下；
2. 再让漂移、扩散与弱形式测度恢复稳定；
3. 最后再讨论离散读取。

## 2. 理论对象与当前工程对象

| 理论对象 | 当前工程对象 | 当前状态 |
| --- | --- | --- |
| `R_{i,j,l}` | [dynamics_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) 与 [response_signature](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) | 已实现 |
| 局部基点 `x` | [chart_latents](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 与 [trajectory_state](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) | 已实现，但仍偏弱 |
| 局部切向子空间 `T_x\mathcal M_T` | [trajectory_tangent_frame](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 与 [trajectory_tangent_projector](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) | 已接入第一版 |
| 漂移 `b_c(x)` | [trajectory_drift](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) | 已实现 |
| 切空间内部结构 `\Sigma_x` | [local_tangent_covariance](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) | 已接入第一版 |
| 二阶结构 `A_x = U_x\Sigma_x U_x^\top` | [local_diffusion_matrix](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) | 已实现切空间参数化 |
| 密度 `\rho_c` | [measure_log_density](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) | 已实现离散 batch 近似 |
| 弱形式 `\mathcal L_c^* \mu_c = 0` | [local_measure_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) | 已实现第一版近似 |

## 3. 为什么当前优先修局部几何对象

最近几轮实验已经表明：

1. 单纯给 `A_c` 加 trace 校准损失，没有解决问题；
2. 降低 `\varepsilon I` 的 floor，也没有解决问题；
3. 把 `A_c` 改写成“形状 + 尺度”的结构化参数化，仍没有解决问题；
4. 但一旦把状态、漂移与协方差目标统一到同一个 chart-space，连续测度指标就明显改善。

因此，当前更接近理论的判断是：

> 主矛盾不在 `A_c` 头的形式，而在局部几何对象是否真的由同一个局部基点与局部切向结构来组织。

## 4. 当前下一步

基于这个映射，当前最合理的工程顺序是：

1. 保持局部切向标架与切向兼容量这条主线，把 `A_x` 明确限制为 `U_x\Sigma_x U_x^\top`；
2. 用谱、迹、各向异性等基不变量来约束 `\Sigma_x`，而不是只盯环境坐标里的非对角项；
3. 重新验证 `measure_stationarity`、`trace`、`tangent_spectrum_alignment` 与切向兼容指标；
4. 在切空间内部结构更可信之后，再继续讨论如何把响应不变量稳定地映射为非平凡的 `\Sigma_x`。
