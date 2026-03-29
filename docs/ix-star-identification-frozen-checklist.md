# `I_x^\star \to (P_x,b_x,\Sigma_x,\tau_c)` 识别约束冻结清单

## 范围

本轮只处理两项直接影响训练目标的识别约束：

1. `\Sigma_x` 的非平凡性约束。
2. `\tau_c` 对几何残差的过度吸收约束。

这两项都直接来自 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L1071) 到 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L1252)。

## 理论依据

理论文档已经给出三类识别增强原则：

1. 规范固定保证模规范唯一性。
2. 谱不变量与邻域一致性保证可识别性。
3. 各向异性与结构变化下界保证非平凡性。

对应到当前代码，`P_x`、`b_x`、`\Sigma_x`、`\tau_c` 都已经有 target 路径，但还缺两个直接的约束：

1. 当 `I_x^\star` 显示出明显各向异性、低有效秩、较大谱间隙时，预测的 `\Sigma_x` 不能退化回近各向同性结构。
2. 当局部几何残差仍然较大时，`|\tau_c|` 不能继续无限增大去“吸收”本应由 `P_x,b_x,\Sigma_x` 表达的结构差异。

## 不在范围内

以下不在本轮：

1. 重写 `P_x` 的参数化方式。
2. 重写 `graph_tau` 的构造公式。
3. 处理 atlas / connection / global transport。
4. 处理 `\mathcal L_0 + \delta \mathcal L_c` 分解。

## 实施项

### A1. `\Sigma_x` 非平凡性约束

新增 `tangent_nontriviality`，只使用已经存在的预测谱与 invariant target 谱。

它由三部分组成：

1. 各向异性下界：若 target anisotropy 更大，则惩罚预测各向异性过低。
2. 有效秩上界：若 target 更低秩，则惩罚预测有效秩过高。
3. 谱间隙下界：若 target 主谱间隙更强，则惩罚预测谱间隙过弱。

### A2. `\tau_c` overreach 约束

新增 `measure_tilt_overreach`。

做法是：

1. 先从 `I_x^\star` 读取几何信号强度，主要用 `anisotropy / spectral_gap / asymmetry`。
2. 再读取当前几何残差，主要用 `local_drift / local_diffusion / tangent_projection / tangent_spectrum_alignment / tangent_shape_alignment`。
3. 若几何信号强且几何残差仍大，则对 `|log_tilt|` 施加额外惩罚。

这不是否定 `\tau_c`，而是阻止它在几何尚未学好的时候继续吸收局部结构差异。

### A3. 训练接线

新增两个权重：

1. `tangent_nontriviality_weight`
2. `measure_tilt_overreach_weight`

默认值均为 `0.0`，保持旧配置不变。

### A4. 测试

至少覆盖：

1. 非平凡性约束在“预测过于各向同性”时为正。
2. `tilt_overreach` 在“几何残差高且 tilt 很大”时为正。
3. `local_measure_loss(...)` 在 identification-active 配置下可正常反传。

## 关闭标准

满足以下条件即可关闭：

1. 代码里存在 `tangent_nontriviality` 与 `measure_tilt_overreach`。
2. trainer/config 已能显式启用这两个约束。
3. pytest 通过。
4. identification-active 训练烟测通过。
