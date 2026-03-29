# `\Psi_x \to I_x^\star` 冻结清单

## 范围

本轮只处理两件事：

1. 把 `I_x^\star` 从“在线临时拼装的若干中间张量”提升为显式 target bundle。
2. 把 `local_measure_targets(...)` 中 response-based 路径的主监督切到这个 bundle 上，使 `chart_delta/bootstrap` 退回辅助尺度或缺省回退角色。

## 不在范围内

以下事项不在本轮内：

1. 把 `\Pi_x(\Psi_x^c) \in E_x` 完整实现为独立纤维对象。
2. 重写 atlas / connection / global transport。
3. 处理 `\mathcal L_0 + \delta \mathcal L_c` 分解。
4. 调整 query / readout 主链。

## 实施项

### A1. 显式 `I_x^\star` target

新增 `ResponseInvariantTarget`，至少包含：

1. 从 raw response bundle 导出的 `descriptor_triangle / signed_triangle / magnitude_triangle / mask`。
2. 响应算子不变量：`eigvals / trace / effective_rank / anisotropy / asymmetry / spectral_gap / scale_profile`。
3. 由响应不变量诱导的局部 target：`tangent_frame / tangent_projector / tangent_drift / tangent_cov / identifiable_tangent_cov / support_tilt / graph_tau`。

### A2. frozen / semi-frozen 语义

1. 若传入 `target_model`，则 `I_x^\star` 由 teacher/EMA 路径在 `no_grad` 下构造，视为 frozen target。
2. 若未传入 `target_model`，则仍在 `no_grad` 下由当前模型构造，视为 semi-frozen target。

### A3. 主监督切换

1. `measure_target_mode=response_invariant_bootstrap` 时，主 shape target 由 `I_x^\star` 的 identifiable tangent covariance 提供。
2. bootstrap 只负责提供尺度或在 invariant target 缺失时回退。
3. `measure_target_mode=response_jet`、`drift_target_mode=response_jet`、`tilt_target_mode=response_support/graph_tau` 应显式优先消费 `I_x^\star` bundle，而不是再次各自在线拼装。

### B1. 测试

至少覆盖：

1. `ResponseInvariantTarget` 能从 raw response bundle 稳定构造。
2. `local_measure_targets(...)` 返回显式 `invariant_target`。
3. `response_invariant_bootstrap` 下，target shape 由 invariant target 决定，bootstrap 只参与尺度。
4. teacher 路径下的 invariant target 为 detached/frozen。
5. `local_measure_loss(...)` 在 invariant-first 路径上可正常反传。

### B2. 训练烟测

补一份最小 measure-active smoke 配置，验证训练链可跑通。

## 关闭标准

满足以下条件即可关闭本轮：

1. 代码里存在显式 `ResponseInvariantTarget`。
2. `local_measure_targets(...)` 返回 `invariant_target` 并以其生成主监督。
3. bootstrap 不再承担 response-based 路径下的主结构 target。
4. pytest 通过。
5. measure-active 训练烟测通过。
