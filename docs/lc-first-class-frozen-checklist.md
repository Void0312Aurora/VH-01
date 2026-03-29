# `\mathcal L_c` 一级对象化冻结清单

## 1. 冻结说明

本文只服务于一个问题：

1. 理论里 `\mathcal L_c` 位于 `局部响应几何 -> \mathcal L_c -> \mu_c` 的中间层；
2. 当前实现虽然已有漂移、扩散、切向结构、条件测度与弱形式损失，但 `\mathcal L_c` 本身还不是清晰、独立、可调用的工程对象。

自本文起，这个问题的修复范围冻结。
若没有明确改 scope，后续只按本文列出的任务推进，不再临时追加新的独立任务。

冻结时间：

1. 2026-03-29

## 2. 本问题的关闭标准

只有当下面 6 条全部满足时，才能把“`\mathcal L_c` 不是一级工程对象”判定为关闭：

1. 模型侧存在显式 `LocalGenerator` 或等价对象，用来统一表达当前局部生成元。
2. target 侧存在显式 `LocalGeneratorTarget` 或等价对象，而不是只返回散字段字典。
3. `local_measure_loss(...)` 的主要 `\mathcal L_c f` 数值作用，不再由 loss 直接手写，而是调用 generator 对象接口。
4. 现有漂移、扩散、切向结构、条件测度这些 active 数值路径，不会因为对象化而断链。
5. 有固定测试覆盖 generator 对象构造、generator 作用接口、target 对象构造、训练 smoke。
6. 默认 smoke 训练能正常结束，相关冻结测试不会因为这一步被打坏。

## 3. 明确不在本次范围内的事项

下面这些问题虽然重要，但不属于“关闭 `\mathcal L_c` 不是一级工程对象”的必要条件：

1. 重写 `response_jet / graph_tau` 的具体数值公式。
2. 把 batch-kNN 近邻彻底升级成数据集级或图册级邻域。
3. 完整实现 `\mathcal L_0 + \delta \mathcal L_c` 的扰动分解。
4. 重新设计 query/readout 规则。
5. `\mu_0 \to \mathfrak T_c \to \mu_c \to Read` 的进一步对象化。
6. 新的训练指标或大规模基线切换。

如果后面要做这些，应该另开任务，不算作这里的派生工作。

## 4. 执行任务

### A1. 冻结设计边界

目标：

1. 明确本轮只做 `\mathcal L_c` 的一级对象化。
2. 明确第一刀不重写已有数值定义，只做等价重构。

状态：

1. 已完成

结果：

1. 已新增 [lc-first-class-design.md](/home/void0312/AIGC/VH-01/docs/lc-first-class-design.md#L1)。
2. 本文已冻结“先对象化，后再考虑深层数值重写”的实现边界。

### A2. 模型侧新增 `LocalGenerator`

目标：

1. 新增统一的 generator 上下文与 generator 对象。
2. 将 drift / diffusion / tangent / conditional measure 收口到同一局部生成元对象。

状态：

1. 已完成

结果：

1. 已新增 `GeneratorContext` 与 `LocalGenerator`。
2. 模型侧已提供 `model.local_generator(...)`，统一返回 drift / diffusion / tangent / conditional measure。
3. active 路径不再需要分别手工拼接这些局部生成元部件。

### A3. target 侧新增 `LocalGeneratorTarget`

目标：

1. 把 `local_measure_targets(...)` 返回的大字典升级成目标生成元对象。
2. 在过渡期保持旧字段兼容。

状态：

1. 已完成

结果：

1. `local_measure_targets(...)` 已返回 `LocalGeneratorTarget`。
2. 过渡期兼容访问已保留，`targets["field"]` 与 `targets.get(...)` 仍可用。
3. `source_point / source_measure / tangent target / response-jet target` 已收口进目标生成元对象。

### A4. 重接 `local_measure_loss(...)`

目标：

1. 把 `\mathcal L_c f` 的主要数值作用迁移到 generator 对象方法。
2. 让 loss 负责聚合，而不是负责“定义生成元”。

状态：

1. 已完成

结果：

1. `local_measure_loss(...)` 已通过 `LocalGenerator` 消费 drift / diffusion / measure。
2. `\mathcal L_c f` 的主要数值作用已迁移为 `generator.apply_linear / apply_quadratic / apply_trig / apply_radial`。
3. loss 现在负责聚合指标，不再负责“临时定义生成元对象”。

### B1. 固定测试覆盖 generator 对象

目标：

1. 测试 `LocalGenerator` 构造与数值有限性。
2. 测试 generator 作用接口与当前旧实现等价或近等价。
3. 测试 `LocalGeneratorTarget` 构造与兼容字段。

状态：

1. 已完成

结果：

1. 已新增 [test_lc_first_class_frozen_checklist.py](/home/void0312/AIGC/VH-01/tests/test_lc_first_class_frozen_checklist.py#L1)。
2. 已覆盖 `LocalGenerator` 构造、`apply_*` 数值接口、`LocalGeneratorTarget` 兼容访问、`local_measure_loss(...)` 反传。

### B2. 回归测试与训练 smoke

目标：

1. 跑相关冻结测试，确认现有已关闭问题不被打坏。
2. 跑默认 smoke 训练，确认训练链不断。

状态：

1. 已完成

结果：

1. 已执行基线 `pytest -q tests/test_mu0_tc_muc_read_design.py tests/test_x_trajectory_point_frozen_checklist.py tests/test_conditional_encoder_frozen_checklist.py tests/test_r_ijl_frozen_checklist.py`。
2. 基线结果为 `16 passed`。
3. 已执行 `PYTHONPATH=src python scripts/train_mvp.py --config configs/smoke.yaml`。
4. 基线 smoke 训练已正常跑完 2 个 epoch。
5. 修复后已执行 `pytest -q tests/test_lc_first_class_frozen_checklist.py tests/test_mu0_tc_muc_read_design.py tests/test_x_trajectory_point_frozen_checklist.py tests/test_conditional_encoder_frozen_checklist.py tests/test_r_ijl_frozen_checklist.py`。
6. 修复后结果为 `20 passed`。
7. 修复后已再次执行 `PYTHONPATH=src python scripts/train_mvp.py --config configs/smoke.yaml`。
8. 修复后 smoke 训练仍正常跑完 2 个 epoch。

### B3. 文档收口

目标：

1. 修复完成后回写审计文档与理论映射文档。
2. 把本文从“待执行”改成“冻结完成态”。

状态：

1. 已完成

结果：

1. 已回写 [theory-mismatch-audit.md](/home/void0312/AIGC/VH-01/docs/theory-mismatch-audit.md#L99)。
2. 已回写 [theory-engineering-map.md](/home/void0312/AIGC/VH-01/docs/theory-engineering-map.md#L51)。
3. 本文已收口为冻结完成态。

## 5. 允许的唯一派生工作

后续允许出现的派生工作只有两种：

1. 为完成 A2-A4、B1-B3 直接产生的 shape / dtype / compile / numerical stability 修复。
2. 为让新增 generator 对象测试通过而做的最小兼容修补。

除此之外，不新增新的独立任务。

## 6. 当前结论

当前这项冻结问题已经按本文范围关闭。

当前已经完成的是：

1. `LocalGenerator` 已成为模型侧显式对象；
2. `LocalGeneratorTarget` 已成为 target 侧显式对象；
3. `local_measure_loss(...)` 已主要经由 generator 接口实现 `\mathcal L_c f`；
4. 冻结测试已覆盖对象构造、数值接口与训练反传；
5. 冻结测试与 smoke 训练都已重新通过。

因此，“`\mathcal L_c` 不是一级工程对象”这一项在当前冻结边界内已修补完成。更深层的 `\mathcal L_0 + \delta \mathcal L_c` 分解与数值重写，属于后续独立任务，不再计入本文。
