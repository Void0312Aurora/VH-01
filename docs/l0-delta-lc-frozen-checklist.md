# `\mathcal L_0 + \delta \mathcal L_c` 冻结清单

## 范围

本轮只处理一件事：

> 把当前已经一级对象化的 `\mathcal L_c`，进一步显式拆成“基准生成元 `\mathcal L_0`”与“条件增量 `\delta \mathcal L_c`”。

对应理论位置见 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L1455) 到 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L1458)。

## 理论依据

理论文档已经明确给出：

1. 训练阶段先形成基准支持结构 `\mu_0`。
2. 条件 `c` 先通过 `\tau_c` 倾斜 `\mu_0`。
3. 在需要时，再通过 `\delta \mathcal L_c` 把这种条件更新连续实现为局部生成机制变化。

因此，当前工程里更合理的对象关系不是“只存在一个 joint 的 `\mathcal L_c`”，而应是：

1. `\mathcal L_0`：无条件的局部漂移、局部二阶结构与基准测度。
2. `\delta \mathcal L_c`：相对 `\mathcal L_0` 的条件增量。
3. `\mathcal L_c = \mathcal L_0 + \delta \mathcal L_c`。

## 当前缺口

当前代码虽然已经有：

1. `BaseMeasure / ConditionalTilt / ConditionalMeasure`
2. `LocalGenerator`
3. `base_dynamics + cond_delta`

但还缺：

1. 显式的 base generator 对象；
2. 显式的 conditional generator delta 对象；
3. 在 `joint` 测度模式下，也能写成“基准密度 + 条件倾斜”的稳定分解；
4. 一个最小但直接的 `\delta \mathcal L_c` 预算约束，避免条件增量无节制膨胀。

## 不在范围内

以下不在本轮：

1. atlas / connection / global transport
2. query/readout 主链重写
3. `I_x^\star` 的新参数化
4. `graph_tau` 公式重写

## 实施项

### A1. 显式对象化 `\mathcal L_0`

新增 `BaseLocalGenerator`，至少包含：

1. `base_measure`
2. `drift`
3. `diffusion_matrix`
4. `tangent_core_cov`

### A2. 显式对象化 `\delta \mathcal L_c`

新增 `ConditionalGeneratorDelta`，至少包含：

1. `conditional_tilt`
2. `drift`
3. `diffusion_matrix`
4. `tangent_core_cov`

并满足：

1. `drift_full = drift_base + drift_delta`
2. `diffusion_full = diffusion_base + diffusion_delta`
3. `log_mu_c = log_mu_0 + log_tilt`

### A3. `joint` 测度模式下的基准/增量分解

旧版 `measure_density_mode=joint` 下，`base_measure` 仍然依赖真实条件输入，不是真正的 `\mu_0`。

本轮改成：

1. `\mu_0` 用 zero-condition baseline 近似；
2. `\tau_c` 用 `joint(cond) - joint(zero-cond)` 近似；
3. 从而在 `joint` 和 `tilted` 两种模式下都能得到统一的 `\mu_0 + \tau_c` 分解。

### A4. 最小 `\delta \mathcal L_c` 预算约束

新增 `generator_delta_budget`。

只惩罚：

1. `\delta b_c`
2. `\delta A_c`

默认权重为 `0.0`，不改变旧配置；需要时可小权重启用，表达“条件更新优先走 `\tau_c`，`δ\mathcal L_c` 只在必要时介入”。

### A5. 测试

至少覆盖：

1. `LocalGenerator` 的 full/base/delta 分解一致性；
2. `joint` 模式下 `log_mu_c = log_mu_0 + log_tilt`；
3. `local_measure_loss(...)` 在启用 `generator_delta_budget` 时可正常反传；
4. smoke 训练链通过。

## 关闭标准

满足以下条件即可关闭：

1. 代码里存在 `BaseLocalGenerator` 与 `ConditionalGeneratorDelta`。
2. `LocalGenerator` 显式包含 base/delta 分解。
3. `joint` 与 `tilted` 两种测度模式都能写成 `\mu_0 + \tau_c` 分解。
4. `generator_delta_budget` 已接入 config/trainer/loss。
5. pytest 通过。
6. smoke 训练通过。
