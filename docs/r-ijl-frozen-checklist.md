# `R_{i,j,l}` 修复冻结清单

## 1. 冻结说明

本文只服务于一个问题：

1. `R_{i,j,l}` 在实现里被过早标量化，导致理论里的“局部响应对象”过早退化成 scalar MSE。

自本文起，这个问题的修复范围冻结。
后续若没有明确改 scope，我只按本文列出的剩余任务推进，不再临时追加新的独立任务。

冻结时间：

1. 2026-03-29

## 2. 本问题的关闭标准

只有当下面 5 条全部满足时，才能把“`R_{i,j,l}` 被过早标量化”判定为关闭：

1. 代码里存在唯一、明确、可复用的原始 residual 构造路径，`R_{i,j,l}` 可被显式访问。
2. `E_dyn` 在工程语义上是从原始 residual 对象导出的能量，而不是另一条独立的 scalar-first 主路径。
3. 所有 active 的 response geometry 路径都从 structured residual / descriptor 出发，而不是从 scalar energy triangle 出发。
4. `response_context` 进入局部几何头时，可以直接消费 descriptor-aware signature，而不再被限制为 signed scalar summary。
5. 有固定测试覆盖对象恢复、signature 维度、训练 smoke、生成 smoke；不是只靠临时手工命令验证。

## 3. 明确不在本次范围内的事项

下面这些问题虽然重要，但不属于“关闭 `R_{i,j,l}` 标量化问题”的必要条件：

1. `\mu_0 / \mathfrak T_c / \mu_c / Read` 的对象化。
2. `query / support / condition update` 的理论统一。
3. `x`、局部邻域、切空间、漂移、扩散围绕同一个局部对象的完全统一。
4. 条件化编码器 `E_t(X_{1:T}, c)` 的重写。
5. continuous-measure 链路是否升级为默认主线。

如果后面要做这些，应该另开任务，不算作这里的派生工作。

## 4. 已完成项

### A1. 恢复原始 residual 对象

状态：

1. 已完成

结果：

1. [ResponseTriangleBundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L204) 已保存 `residual_triangle / energy_triangle / mask`。
2. [response_triangle_bundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L273) 已把 `pred_delta - true_delta` 显式保留下来。
3. 旧接口 [response_triangle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L297) 仅作兼容包装。

### A2. response geometry 改为消费 structured descriptor

状态：

1. 已完成

结果：

1. [response_descriptor_triangle_from_bundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L315) 已从 residual bundle 构造 descriptor。
2. [response_operator](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L1006) 相关链路已改为基于 descriptor，而不是 scalar energy triangle。
3. response-jet 已从 descriptor 通道出发构造局部几何目标。

### A3. `response_context` 已支持 descriptor-aware signature

状态：

1. 已完成

结果：

1. [response_signature_dim](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L228) 新增 `descriptor_span_stats` 与 `descriptor_full_triangle`。
2. [response_signature](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L1147) 已能直接从 descriptor triangle 构造 signature。
3. [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L778) 已能直接走 descriptor-aware signature。
4. 训练入口和评估脚本已按 `channels` 计算新的 signature 维度。

## 5. 剩余任务

下面 4 项原本是冻结剩余清单；截至 2026-03-29，它们已经全部完成。

### B1. 把 `dynamics_loss` 改成“由 residual 对象导出的能量”

状态：

1. 已完成

结果：

1. [dynamics_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L45) 现在通过共享 helper [\_compute_response_triangle_components](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L76) 取到 `energy_triangle`。
2. [response_triangle_bundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L273) 与 `dynamics_loss` 已共享同一条 residual 构造逻辑。
3. 默认配置重新执行过 `configs/smoke.yaml` 两个 epoch，训练链正常结束，`dyn_loss` 数值保持原有量级。

完成标准：

1. 已达成。
2. 已达成。
3. 已达成。

### B2. 清理或降级旧的 scalar-first response 辅助路径

状态：

1. 已完成

结果：

1. 旧的 `_response_signature_from_triangle`、`_response_operator_from_triangle`、`_flatten_response_channels` 已从 [objectives.py](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) 移除。
2. `rg` 级别检查已经确认 `src/`、`scripts/`、`tests/` 中不再存在这些旧 helper 的 active 引用。

完成标准：

1. 已达成。
2. 已达成。
3. 已达成。

### B3. 把当前手工 smoke 验证固化成正式测试

状态：

1. 已完成

结果：

1. 新增 [conftest.py](/home/void0312/AIGC/VH-01/tests/conftest.py#L1) 和 [test_r_ijl_frozen_checklist.py](/home/void0312/AIGC/VH-01/tests/test_r_ijl_frozen_checklist.py#L1)。
2. 固定测试已覆盖：
   - residual bundle 的 shape / mask / energy 对齐；
   - `dynamics_loss` 与 bundle energy 聚合一致；
   - legacy 与 descriptor-aware 两种 signature 模式的维度一致；
   - `descriptor_span_stats` 下 `local_measure_loss` 可前向/反向；
   - 一条最小 rollout / decode / query-responsive 生成 smoke。
3. `PYTHONPATH=src pytest -q tests/test_r_ijl_frozen_checklist.py` 已通过，结果为 `5 passed`。

完成标准：

1. 已达成。
2. 已达成。

### B4. 关闭文档与代码语义差异

状态：

1. 已完成

结果：

1. [theory-mismatch-audit.md](/home/void0312/AIGC/VH-01/docs/theory-mismatch-audit.md#L369) 已补充冻结问题的收口状态与验证结果。
2. 本文已经从“待办清单”更新为“冻结完成清单”，不会再让 `R_{i,j,l}` 这一个问题在文档层面重复扩 scope。

完成标准：

1. 已达成。
2. 已达成。

## 6. 允许的唯一派生工作

为了避免后面“做完一个又冒出新的任务”，这里再约束一次。

后续允许出现的派生工作只有两种：

1. 为了完成 B1-B4 而直接产生的 shape / dtype / compile / numerical stability 修复。
2. 为了让新增测试通过而做的最小兼容修补。

除此之外，不新增新的独立任务。

## 7. 当前结论

当前这项冻结问题已经关闭。

关闭依据如下：

1. 上游 residual 对象已恢复。
2. `dynamics_loss` 已回到共享 residual 构造路径。
3. active response geometry 已经脱离 scalar-first helper。
4. descriptor-aware `response_context` 已可用。
5. 固定测试、`pytest`、默认 smoke 训练都已经通过。

这只代表“`R_{i,j,l}` 被过早标量化”这一项已关闭，不代表所有理论错配都已关闭。
