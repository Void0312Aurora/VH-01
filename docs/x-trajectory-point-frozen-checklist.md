# 局部基点 `x` 修复冻结清单

## 1. 冻结说明

本文只服务于一个问题：

1. 理论里的局部基点 `x` 是 `\mathcal Z^T` / `\mathcal M_T` 上“整条轨迹作为一个状态点”的局部锚点；
2. 当前实现却把 [trajectory_state](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 同时当作轨迹点坐标、摘要上下文、局部几何锚点来用，语义发生混用。

自本文起，这个问题的修复范围冻结。
若没有明确改 scope，后续只按本文列出的任务推进，不再临时追加新的独立任务。

冻结时间：

1. 2026-03-29

## 2. 本问题的关闭标准

只有当下面 6 条全部满足时，才能把“局部基点 `x` 与 `trajectory_state` 混用”判定为关闭：

1. 代码里存在显式的轨迹点接口，用来表示“整条轨迹作为 `\mathcal Z^T` 中一个点”的坐标。
2. 代码里存在显式的摘要上下文接口；它不再承担局部基点语义。
3. active 的局部测度、局部扩散、局部几何 target 构造，默认都围绕轨迹点接口而不是摘要接口。
4. `trajectory_tangent_frame` 不再只从摘要特征直接回归，而是至少同时围绕轨迹点与摘要上下文构造。
5. 兼容入口仍然可用，训练链和生成链不会因为对象拆分而断掉。
6. 有固定测试覆盖对象拆分、默认锚点切换、局部测度前向/反向、训练 smoke、生成 smoke。

## 3. 明确不在本次范围内的事项

下面这些问题虽然重要，但不属于“关闭局部基点 `x` 与 `trajectory_state` 混用”的必要条件：

1. `\mu_0 / \mathfrak T_c / \mu_c / Read` 的对象化。
2. `query / support / condition update` 的理论统一。
3. 条件化编码器 `E_t(X_{1:T}, c)` 的重写。
4. continuous-measure 链路是否升级为默认主线。
5. 联络、平行传输与全局图册兼容性的完整实现。
6. 把 `\Psi_x \mapsto I_x^\star` 的不变量提取做成最终形式。

如果后面要做这些，应该另开任务，不算作这里的派生工作。

## 4. 执行任务

### A1. 文档冻结对象分工

目标：

1. 明确 `x / \Psi_x / I_x^\star / trajectory_summary_context` 四层对象的角色。
2. 撤回“继续把 `trajectory_state` 提升为局部图坐标”的旧表述。

状态：

1. 已完成

结果：

1. 新增本文，冻结 `x / \Psi_x / I_x^\star / trajectory_summary_context` 的对象分工。
2. [next-phase-plan.md](/home/void0312/AIGC/VH-01/docs/next-phase-plan.md) 已撤回“继续把 `trajectory_state` 提升为局部图坐标”的旧表述。
3. [continuous-measure-plan.md](/home/void0312/AIGC/VH-01/docs/continuous-measure-plan.md) 已改写为 `trajectory_point` 与 `trajectory_summary_context` 分工叙述。
4. [theory-mismatch-audit.md](/home/void0312/AIGC/VH-01/docs/theory-mismatch-audit.md) 已补充当前修补状态。

### A2. 显式拆分轨迹点与摘要上下文

目标：

1. 新增显式 `trajectory_point(...)` 接口。
2. 新增显式 `trajectory_summary_context(...)` 接口。
3. `trajectory_state(...)` 保留为兼容入口，但语义对齐到轨迹点而不是摘要。

状态：

1. 已完成

结果：

1. [trajectory_point](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 已作为显式接口加入模型。
2. [trajectory_summary_context](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 已作为显式摘要上下文接口加入模型。
3. [trajectory_state](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 保留为兼容入口，但已对齐到 `trajectory_point`。

### A3. 重接局部测度与 target 构造入口

目标：

1. `local_measure_context(...)`、`local_diffusion_context(...)`、`measure_log_density_components(...)` 默认围绕轨迹点工作。
2. [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) 中的邻域、response-jet、graph-tau 入口改为轨迹点。

状态：

1. 已完成

结果：

1. [local_measure_context](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 和 [measure_log_density_components](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 默认已切到 `trajectory_point`。
2. [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) 已显式返回 `source_point / source_summary_context`，并用 `source_point` 构造 response-jet 与 graph-tau。
3. [local_measure_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) 已把默认 state anchor 改为 `trajectory_point`。

### A4. 修复切空间入口

目标：

1. `trajectory_tangent_frame(...)` 不再只从摘要特征回归。
2. 它至少要同时消费轨迹点与摘要上下文，避免切空间语义继续停留在“摘要的切空间”上。

状态：

1. 已完成

结果：

1. [trajectory_tangent_frame](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 已改为同时消费 `trajectory_point` 与 `trajectory_summary_context`。
2. [local_tangent_structure](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 已围绕新的 point anchor 构造切空间入口。

### B1. 固定测试覆盖对象拆分

目标：

1. 测试 `trajectory_point / trajectory_summary_context / trajectory_state` 的角色关系。
2. 测试默认 local measure anchor 已切到轨迹点。

状态：

1. 已完成

结果：

1. 新增 [test_x_trajectory_point_frozen_checklist.py](/home/void0312/AIGC/VH-01/tests/test_x_trajectory_point_frozen_checklist.py#L1)。
2. 测试覆盖了 `trajectory_point / trajectory_summary_context / trajectory_state` 的角色关系、默认 local-measure anchor、tangent frame 显式输入一致性，以及 `local_measure_targets` 的 point anchor。

### B2. 固定测试覆盖局部测度链

目标：

1. 保证 descriptor-aware 模式下 `local_measure_loss` 仍可前向/反向。
2. 保证对象拆分后 response context 与局部几何头仍能正常工作。

状态：

1. 已完成

结果：

1. `descriptor-aware` 模式下，新的 point anchor 与原有 response context 已共同通过 [local_measure_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) 前向/反向验证。
2. 与 [test_r_ijl_frozen_checklist.py](/home/void0312/AIGC/VH-01/tests/test_r_ijl_frozen_checklist.py#L1) 联跑时，没有打断已关闭的 `R_{i,j,l}` 冻结问题。

### B3. 训练 smoke

目标：

1. 运行最小训练配置，确认训练链不因对象拆分而中断。

状态：

1. 已完成

结果：

1. `pytest -q tests/test_x_trajectory_point_frozen_checklist.py tests/test_r_ijl_frozen_checklist.py` 已通过，结果为 `9 passed`。
2. `PYTHONPATH=src python scripts/train_mvp.py --config configs/smoke.yaml` 已正常跑完 2 个 epoch。

### B4. 生成 smoke 与文档收口

目标：

1. 跑最小生成闭环，确认 rollout / decode / 候选读取仍有限。
2. 把本文状态更新为冻结完成态，并回写审计文档。

状态：

1. 已完成

结果：

1. 最小生成闭环 smoke 已由固定测试覆盖，并随 `pytest` 一并通过。
2. 本文已更新为冻结完成态，不再把这一项写成“继续追加任务”的开放问题。

## 5. 允许的唯一派生工作

后续允许出现的派生工作只有两种：

1. 为完成 A1-A4、B1-B4 直接产生的 shape / dtype / compile / numerical stability 修复。
2. 为让新增测试通过而做的最小兼容修补。

除此之外，不新增新的独立任务。

## 6. 当前结论

当前这项冻结问题已经关闭。

关闭依据如下：

1. 轨迹点与摘要上下文已经显式拆分。
2. active 的局部测度与 target builder 已经默认围绕轨迹点工作。
3. 切空间入口已不再是“摘要单输入”。
4. 兼容入口、固定测试、训练 smoke、生成 smoke 都已经通过。

这只代表“局部基点 `x` 与 `trajectory_state` 混用”这一项已关闭，不代表所有理论错配都已关闭。
