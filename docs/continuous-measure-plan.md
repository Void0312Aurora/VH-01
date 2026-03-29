# VH-01 连续测度工程方案

## 1. 目标

本方案用于把当前理论主线重新落回连续对象上，而不是继续停留在离散候选打分层。

理论主线按 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md) 与 [temp-03.md](/home/void0312/AIGC/VH-01/docs/temp-03.md) 应写成：

$$
R_{i,j,l}
\rightsquigarrow
\text{局部响应几何}
\rightsquigarrow
\mathcal L_c
\rightsquigarrow
\mu_0
\xrightarrow{\mathfrak T_c}
\mu_c
\xrightarrow{\mathrm{Read}}
\Gamma.
$$

因此，工程上的正确顺序应当是：

1. 先在连续背景 `\mathcal M_T` 上近似局部结构；
2. 再由局部结构构造生成元候选；
3. 再由生成元的弱形式恢复全局测度；
4. 最后才允许离散化和读取。

## 2. 第一版工程假设

由于 `\mathcal M_T` 还没有显式图册，这里采用一个温和近似：

1. 用整条潜轨迹 `Z_{1:T}` 的时序矩图坐标 `x=\chi_\theta(Z_{1:T})` 作为 `\mathcal M_T` 上局部基点的近似；
2. 用 `R_{i,j,l}` 诱导的响应签名 `\Psi_c(Z^{(l)})` 近似局部邻域关系；
3. 用条件相关的局部漂移 `b_c(x)` 与二阶结构 `A_c(x)` 近似 `\mathcal L_c`；
4. 用一个连续密度头 `\rho_c(x)` 或 `\log \rho_c(x)` 近似 `\mu_c` 的密度表示。

这不是完整证明，只是一个第一版可训练近似。

## 2.1 本轮理论修正对应的工程约束

结合 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md) 与 [temp-03.md](/home/void0312/AIGC/VH-01/docs/temp-03.md) 的新修正，当前工程上应额外遵守三条约束：

1. `\mathcal M_T` 不再被近似成单一唯一 chart，而应允许被若干局部图册或分层支撑近似；
2. 条件 `c` 的首要作用不再优先解释为“深度改写整个 `\mathcal L_c`”，而是优先解释为对基准测度 `\mu_0` 的局部倾斜更新；
3. `A_c` 的条件依赖应当被视为比“密度倾斜”更强的假设，只有在响应证据足够明确时才值得引入。

在切空间主线下，上述近似还应进一步理解为：

1. `trajectory_point` 首先承担的是“整条轨迹作为 `\mathcal Z^T` 中一点”的局部基点近似角色；
2. `trajectory_summary_context` 只承担辅助上下文角色，不再直接充当局部基点；
3. 真正需要补上的对象是该基点附近的局部切向子空间；
4. 全局化的关键不是再发明更复杂的 chart，而是让局部切向对象在邻域中具有兼容性。

因此，当前最贴近理论的新工程顺序应写成：

$$
R_{i,j,l}
\rightsquigarrow
\text{局部图册/局部状态}
\rightsquigarrow
(\,b_0,A_0,\mu_0\,)
\xrightarrow{\;\tau_c\;}
\mu_c,
$$

其中 `\tau_c` 表示条件测度倾斜，而不是一个先验固定候选集。

## 3. 局部对象

### 3.1 局部状态与局部切向对象

第一版中，用轨迹的低阶时序矩

$$
x = \chi_\theta(Z_{1:T})
$$

表示 `\mathcal M_T` 上一点。旧实现中，`x` 由以下量拼接后再投影得到：

1. 轨迹 summary；
2. 一阶时序差分均值；
3. 二阶时序差分均值；
4. 首末端点差；
5. 时间维标准差。

这使它比单纯的均值 summary 更接近“局部基点”而不是静态类别摘要，但仍然把局部基点与摘要上下文混在了一起。

当前修正后，更合适的对象拆分是：

1. `trajectory_point`：由整条 chart 后轨迹经时序感知聚合得到的轨迹点坐标；
2. `trajectory_summary_context`：由低阶时序矩得到的摘要上下文；
3. `trajectory_state`：只保留为兼容入口，对齐到 `trajectory_point`。

但在当前修正后，这还不够。更贴近理论的下一步是：

1. 在每个局部基点附近，再显式近似一个低维切向子空间；
2. 用该切向子空间去承载局部漂移和局部二阶结构；
3. 再讨论这些局部切向对象在邻域中的兼容性，从而为全局化做准备。

### 3.2 局部响应签名

对每个样本保留

$$
\Psi_c(Z^{(l)}) = \bigl(R_{i,j}^{(l)}\bigr)_{1\le i<j\le T}
$$

的压缩签名，用于构造批内局部邻域 `\mathfrak N_c`。第一版里采用“按跨度聚合后的残差均值/方差”作为签名。

### 3.3 局部生成元入口

按 [temp-03.md](/home/void0312/AIGC/VH-01/docs/temp-03.md) 的半显式路线，第一版只显式参数化：

1. 漂移 `b_c(x)`  
2. 二阶结构 `A_c(x)`  
3. 密度头 `\log \rho_c(x)`

其中：

- `b_c(x)` 由现有 `\phi_c` 的平均局部推进量近似；
- `A_c(x)` 由 PSD 因子头给出，即
  $$
  A_c(x)\approx B_c(x)B_c(x)^\top + \varepsilon I;
  $$
- `\log \rho_c(x)` 由单独的密度头给出。

## 4. 第一版损失

### 4.1 漂移一致性

用有限差分近似一阶项：

$$
\mathcal L_{\mathrm{drift}}
=
\|b_c(x)-\widehat{\Delta z}\|^2.
$$

### 4.2 扩散一致性

用局部增量的中心二阶矩近似二阶结构对角：

$$
\mathcal L_{\mathrm{diff}}
=
\|\operatorname{diag}(A_c)-\widehat{\operatorname{Var}}(\Delta z)\|^2.
$$

### 4.3 弱形式测度约束

不直接求解 `\mathcal L_c^*\mu_c=0`，而是使用弱形式：

$$
\int \mathcal L_c f \, d\mu_c \approx 0.
$$

当前版本对三类测试函数施加约束，形成 `\mathcal L_{\mathrm{stationary}}`：

1. 线性测试函数；
2. 二次测试函数；
3. 沿若干固定方向的三角测试函数。

第三类的作用是避免弱形式只约束极低阶矩，从而让“连续测度一致性”不至于退化成只压线性/二次统计量。

### 4.4 响应邻域平滑

在由 `\Psi_c` 诱导的局部邻域上，对

- 状态坐标
- 漂移
- 扩散对角
- 对数密度

施加平滑约束，得到 `\mathcal L_{\mathrm{smooth}}`。

## 5. 当前工程改动

本轮实现新增了四类正式对象：

1. [mvp.py](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 中的 `trajectory_state`
2. [mvp.py](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 中的 `trajectory_drift`
3. [mvp.py](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 中的 `local_diffusion_diag`
4. [mvp.py](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 中的 `measure_log_density`
5. [mvp.py](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 中的 `trajectory_tangent_frame`
6. [mvp.py](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 中的 `trajectory_tangent_projector`

同时新增了弱形式损失：

1. [objectives.py](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) 中的 `response_signature`
2. [objectives.py](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) 中的 `local_neighbor_smoothness_loss`
3. [objectives.py](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) 中的 `local_measure_loss`
4. `tangent_projection / tangent_bundle_compatibility` 相关项，用于显式约束漂移、扩散与局部观测增量的切向性

训练器也新增了对应指标：

- `local_drift`
- `local_diffusion`
- `measure_stationarity`
- `measure_trig_stationarity`
- `response_smoothness`
- `tangent_projection`
- `tangent_observation_residual`
- `tangent_drift_residual`
- `tangent_diffusion_residual`
- `tangent_bundle_compatibility`
- `tangent_frame_orthogonality`
- `tangent_projector_trace`

## 5.1 本轮推进结果

在第一版连续测度骨架之上，本轮又向前推进了一步：

1. 把 `A_c` 从“只在损失中使用对角近似”升级成了真正参与损失与弱形式约束的低秩 PSD 二阶结构；
2. 新增了针对连续测度几何的验证脚本，用来单独检查：
   - 对角项误差；
   - 非对角项误差；
   - 预测扩散迹与经验迹的尺度关系；
   - PSD 最小特征值。

对应实验表明：

1. full-PSD 版本在 synthetic 与 real-data 上都保持了稳定训练，`val_meas` 没有因为二阶结构升级而发散；
2. 但当前数据上的经验协方差非对角能量非常小，因此 full-PSD 与旧版在现有短训指标上不会自然拉开明显差距；
3. 当前更需要关注的新问题不是“PSD 是否可训练”，而是 `A_c` 的尺度校准：预测扩散迹明显大于经验迹，说明弱形式约束已经开始工作，但还不足以把二阶结构的绝对尺度压回到更合理的范围。
4. 后续的 synthetic 校准试验也给出了一个明确的负结果：单纯加入 trace 对齐项、降低 `\varepsilon I` 的下界，甚至把二阶结构改写成“形状 + 独立尺度”参数化，都没有在当前局部图坐标下有效收缩 trace gap，反而会明显恶化 `measure_stationarity`。
5. 相比之下，把局部状态、漂移与协方差目标统一到同一个 chart-space 后，synthetic 与 real-data 的 `measure_stationarity` 都出现了明显改善。这说明当前更关键的问题不是“再怎么修 `A_c` 的尺度”，而是“局部几何对象是否处在同一个坐标语义下”。
6. 在此基础上继续把 chart 升级成显式利用时间邻域的 temporal chart 后，synthetic 指标还能继续改善，但 real-data 上尚未超过 pointwise `chart_target`。这说明 temporal 邻域信息是有效方向，但当前数据规模和骨干下，最稳的主线仍是 pointwise `chart_target`。
7. 新增的耦合 synthetic 诊断数据已经显式抬高了目标非对角能量（`target_offdiag_energy ≈ 1.66e-3`），说明“缺少非对角信号”不再是主要问题；但模型预测非对角能量仍停留在 `~1e-6`，并且 full/diag 监督对照几乎不拉开，这表明当前瓶颈已转向 `A_c` 的结构表达与优化耦合，而不是监督目标是否包含非对角项。
8. 第一轮“状态协方差投影特征”结构升级（`state_cov_proj_dim=8`）给出负结果：`measure_stationarity` 从 `0.000357` 恶化到 `0.000758`。这说明直接在状态头拼接低维协方差特征并不能自动提升非对角响应，后续升级应更贴近 `R_{i,j,l}` 到生成元的结构映射，而不是继续堆叠状态特征。
9. 第一轮“响应签名上下文”升级也给出了偏负结果：虽然 `diffusion_offdiag_mse` 从 `0.001672` 降到 `0.000900`，但对应目标非对角能量也同步下降，预测非对角能量本身没有被抬高，且 `measure_stationarity` 仍高于基线。这说明“把 `response_signature` 作为附加上下文拼到二阶头输入”还不足以让 `A_c` 真正响应 `R_{i,j,l}` 的结构信息。
10. 第一轮“局部图册 + 测度倾斜”修正已经在工程上成功激活了两个新对象：`measure_tilt_abs_mean ≈ 0.0267` 表明条件倾斜头确实在工作，`chart_expert_entropy ≈ 1.355`（接近 `\log 4`）表明多 chart 没有塌到单专家；但 `measure_stationarity = 0.001464` 仍明显高于 `coupled_full` 基线，`pred_offdiag_energy` 也仍远低于目标量级。这说明“本体拆分更对”并不自动带来“当前 `A_c` 更强”，真正堵塞点仍在 `R_{i,j,l}` 到二阶结构的识别与表达。
11. 保留更完整的上三角 `R_{i,j}` 输入后，`response_ctx` 路线的训练行为确实发生了明显变化：`local_diffusion` 与 `diffusion_offdiag_mse` 大幅下降，说明旧的 `span_stats` 压缩确实会影响后续识别；但新的结果并没有直接提升目标耦合识别，反而使 `raw_target_offdiag_energy` 与 `target_offdiag_energy` 几乎塌到 `1e-7`。这说明完整输入在当前链路中更容易被吸收到“近对角潜表示”里，而不是自动转化成更强的 `A_c` 结构学习。
12. 第一轮“显式局部切空间”实现已经工程化落地：模型现在会直接输出局部切向标架，并在训练里显式惩罚漂移、扩散和局部观测增量偏离该切空间，同时对相邻样本的切向投影子施加兼容性约束。短训几何审计显示，`tangent_frame_orthogonality ≈ 5.18e-08`、`tangent_projector_trace = 8.0`、`tangent_bundle_compatibility ≈ 1.35e-05`，说明“局部切空间对象”本身已经被稳定学出；并且 `measure_stationarity = 0.000309`，略优于 `coupled_full` 的 `0.000357`。但与此同时，`target_offdiag_energy` 也塌到 `6.26e-07`，远低于 `coupled_full` 的 `1.66e-03`。这说明切空间化当前更像是在规范化和吸收局部几何自由度，而不是已经解决了 `R_{i,j,l}` 到强非对角二阶结构的识别。
13. 在此基础上进一步加入“切空间内部谱不变量”约束后，连续测度主指标继续下降到 `measure_stationarity = 9.35e-05`，`local_drift` 也显著减小，说明把二阶结构改写为 `A_x = U_x\Sigma_x U_x^\top` 并对 `\Sigma_x` 的谱进行约束，确实能进一步稳定局部识别链路；但这一轮的 `target_offdiag_energy` 只提升到 `1.33e-06`，仍远低于 `coupled_full` 的强耦合目标，而 `target_tangent_anisotropy ≈ 5.66` 明显高于 `pred_tangent_anisotropy ≈ 2.09`。这说明“谱约束”已经开始暴露切空间内部结构的识别缺口，但它仍主要在做稳定化，还没有真正填平目标与预测之间的内部结构差距。

## 6. 当前版本的边界

这一版仍然只是“连续测度路线”的第一步，而不是完整实现。

还没有做到的包括：

1. 显式恢复完整图册意义上的 `\mathcal M_T`
2. 显式拼接出严格定义的连续 `\mathcal L_c`
3. 真正数值求解 Fokker-Planck 或稳态 PDE
4. 在连续测度完成前彻底移除旧的离散读取层

因此，更准确的定位是：

> 当前版本已开始把工程主线从“离散打分”拉回到“局部响应 -> 生成元入口 -> 弱形式测度恢复”，但仍处于连续测度工程化的第一阶段。

## 7. 下一步

如果继续沿这条路线推进，优先级应为：

1. 继续沿“局部基点 + 局部切向子空间”的组合对象推进，把当前已接入的 `trajectory_tangent_frame` 从正则化对象进一步提升为结构主线；
2. 把基准局部结构与条件测度更新拆开，优先实现 `\log \rho_c = \log \rho_0 + \tau_c` 这一路线；
3. `A_c` 的条件化不再作为默认主线，而是回退为“有证据时再打开”的增强项；
4. 为弱形式测度约束引入更丰富的测试函数族；
5. 在图册坐标、基准扩散与测度倾斜都更可信之后，再回头审查是否需要更强的条件生成元修正；
6. 当前更具体的优先问题，不是继续微调 `atlas_tilt` 参数，而是补足“响应签名如何唯一诱导局部二阶结构”这一识别约束；
7. 在工程上，下一步最直接的对象不是完整全局生成元，而是把已经接入的“局部切向标架 + 邻域兼容量 + 切空间内部谱不变量”进一步变成 `A_c` 的主参数化舞台，而不是只作为附加正则；
8. 最后再决定是否需要把读取层重新接回连续测度之后。
