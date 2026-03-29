# `\mathcal L_c` 一级工程对象化设计

## 1. 目的

本文只处理一个问题：

> 当前仓库里已经有漂移、扩散、切向结构、密度与弱形式损失，但 `\mathcal L_c` 本身还不是一个清晰、独立、可调用的工程对象。

这里不处理：

1. `R_{i,j,l}` 的对象恢复；
2. 局部基点 `x` 与 `trajectory_point` 的对应；
3. `\mu_0 \to \mathfrak T_c \to \mu_c \to Read` 的对象化；
4. query/readout 规则本身。

本文只聚焦：

1. 理论里的 `\mathcal L_c` 在代码中究竟缺什么；
2. 应该怎样把它升成一级工程对象；
3. 修复顺序与关闭标准。

## 2. 理论里 `\mathcal L_c` 的位置

按 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L780) 与 [theory-engineering-map.md](/home/void0312/AIGC/VH-01/docs/theory-engineering-map.md#L7)，当前主线应当写成：

$$
R_{i,j,l}
\rightsquigarrow
\text{局部响应几何}
\rightsquigarrow
\mathcal L_c
\rightsquigarrow
\mu_c.
$$

其中 `\mathcal L_c` 不是单个损失，也不是单个 head，而是由以下局部对象组成的无穷小生成结构：

1. 局部基点 `x`；
2. 切向子空间 `T_x\mathcal M_T`；
3. 漂移 `b_c(x)`；
4. 二阶结构 `A_c(x)`，或切空间内部结构 `\Sigma_x`；
5. 与测度 `\mu_c` 的弱形式相容关系。

因此，理论要求的不是“代码里分别有 drift 和 diffusion”，而是：

> 这些量必须被统一地组织成可作用于测试函数的局部生成元对象。

## 3. 当前实现里已经有什么

当前仓库里，` \mathcal L_c ` 相关零件大部分已经存在：

1. [trajectory_drift](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L743) 提供漂移近似；
2. [local_tangent_structure](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L652) 提供切向标架与切空间内部协方差；
3. [local_diffusion_matrix](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L817) 提供二阶结构 `A_c`；
4. [conditional_measure](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L919) 提供与 `\mu_c` 相连的密度对象；
5. [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L777) 提供 response-jet / graph-tau / tangent 相关 target；
6. [local_measure_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L1240) 把这些量组合成弱形式损失。

因此，问题并不是“功能没有”，而是“对象没有”。

## 4. 当前真正缺失的东西

### 4.1 缺少 `LocalGenerator`

现在模型无法直接返回：

```python
generator = model.local_generator(latents, cond_embed, ...)
```

也就意味着：

1. 外部拿不到一个统一命名的 `\mathcal L_c`；
2. drift / diffusion / tangent / measure 只能分散读取；
3. 生成元的数值行为只能在 loss 内部临时拼出来。

### 4.2 缺少“生成元作用接口”

当前 [local_measure_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L1395) 直接手写了：

1. 线性测试函数上的作用；
2. 二次测试函数上的作用；
3. 三角测试函数上的作用；
4. 径向项；
5. 与弱形式测度约束相关的 moment 组合。

这意味着：

1. `\mathcal L_c f` 只在 loss 里存在；
2. 模型对象自己并不知道如何作用到测试函数上；
3. 其它分析脚本或评估逻辑无法复用这套生成元行为。

### 4.3 缺少 target 侧的生成元对象

当前 [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L777) 返回的是一个很大的字典，而不是“目标生成元对象”。

这会带来两个问题：

1. target 逻辑和 generator 逻辑没有同构；
2. response-jet / graph-tau / tangent target 只能作为散字段流动，无法以“目标生成元”身份被使用。

### 4.4 组件仍然不是围绕同一上下文产生

当前数值上虽然能跑，但语义上还有裂缝：

1. 漂移来自 rollout 平均；
2. 二阶结构来自局部扩散 head；
3. 测度来自 density head；
4. target 则来自 bootstrap + response-jet + graph-tau 的混合。

如果不先做对象化，这些量很容易继续各自演化，各自解释。

## 5. 修复目标

这项修复的目标不是“改进一个 loss 写法”，而是让代码里真正存在：

1. 一个统一的 `LocalGenerator`；
2. 一个统一的 `LocalGeneratorTarget`；
3. 一套显式的 `generator.apply_*` 行为接口；
4. 一条从 `LocalGenerator` 推出弱形式测度残差的稳定路径。

换句话说，修复完成后，` \mathcal L_c ` 应当成为模型与损失共享的一级对象，而不再只是 loss 内部的隐含拼装结果。

## 6. 推荐的对象设计

### 6.1 `GeneratorContext`

最小上下文对象，负责统一局部状态。

建议字段：

1. `state`
2. `cond_embed`
3. `response_context`
4. `tangent_structure`
5. `conditional_measure`

它解决的是：

> 所有局部生成元部件是否围绕同一个局部上下文构造？

### 6.2 `LocalGenerator`

最核心的新对象。

建议字段：

1. `context`
2. `drift`
3. `diffusion_matrix`
4. `tangent_core_cov`
5. `conditional_measure`

建议方法：

1. `apply_linear(directions)`
2. `apply_quadratic(directions)`
3. `apply_trig(directions, trig_scale)`
4. `apply_radial()`
5. `stationarity_moments(...)`

这样以后：

1. loss 用它；
2. 评估脚本用它；
3. 诊断脚本也可以直接用它。

### 6.3 `LocalGeneratorTarget`

目标生成元对象，用来替代大字典 bundle。

建议字段：

1. `source_point`
2. `source_summary_context`
3. `target_drift`
4. `target_diffusion`
5. `target_tangent_frame`
6. `target_tangent_cov`
7. `target_tangent_projector`
8. `target_neighbor_idx`
9. `target_neighbor_weights`
10. `target_transport`
11. `tilt_target`
12. `diagnostics`

重点不是字段名本身，而是：

> 以后 target 也应被看成“目标生成元或其局部系数”，而不是“散着传的一包张量”。

## 7. 最小迁移顺序

### Stage A. 只做对象封装，不改数值定义

第一步不动数值公式，只做“等价重构”：

1. 新增 `GeneratorContext`
2. 新增 `LocalGenerator`
3. 新增 `LocalGeneratorTarget`
4. 让现有公式迁移进去

这一阶段的目标是：

1. 训练结果不应明显变化；
2. 只是把原来分散的实现重新收口成对象。

### Stage B. 让 `local_measure_loss` 完全消费 `LocalGenerator`

把当前 loss 中手写的：

1. `projected_drift`
2. `projected_diffusion`
3. `linear_moment`
4. `quadratic_moment`
5. `trig_moment`

全部改为调用 `LocalGenerator.apply_*()`。

做到这一步后，才能说：

> `\mathcal L_c` 已经不只是 loss 里的隐式拼装物。

### Stage C. 让 target builder 输出 `LocalGeneratorTarget`

这一步把 [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L777) 的大字典改成正式对象，并保持旧字段兼容一段时间。

### Stage D. 统一 drift / diffusion / measure 的构造上下文

这是更深一层的修补，不一定第一轮就做完，但方向必须明确：

1. 漂移、扩散、测度应围绕同一 `GeneratorContext` 计算；
2. 而不是继续让三条分支各自拿各自的上下文。

## 8. 不建议的修法

下面这些做法不建议采用：

1. 只改文档，不改对象边界；
2. 只在 loss 里包一层 helper，就声称 `\mathcal L_c` 已对象化；
3. 直接重写 response-jet / graph-tau 数值公式；
4. 先做新的训练目标，而不先把生成元对象立起来。

这些做法要么只是表述变好看，要么会让风险过大。

## 9. 关闭标准

只有当下面 5 条全部满足时，才能认为“`\mathcal L_c` 不是一级工程对象”这一项被关闭：

1. 模型侧存在显式 `LocalGenerator` 或等价对象入口；
2. target 侧存在显式 `LocalGeneratorTarget` 或等价对象入口；
3. `local_measure_loss` 不再手写主要的 `\mathcal L_c f` 公式，而是调用 generator 对象接口；
4. 训练 smoke 与现有冻结测试全部通过；
5. 文档中不再需要把 `\mathcal L_c` 描述为“只有 surrogate，没有对象”。

## 10. 当前结论

当前仓库已经有了：

1. 生成元的多数数值零件；
2. 连到 `\mu_c` 的连续测度头；
3. 弱形式损失与 target builder。

但它还没有：

1. `LocalGenerator`
2. `LocalGeneratorTarget`
3. 显式的 `\mathcal L_c f` 作用接口

因此，当前问题不是“完全没实现”，也不是“只差写法好看”，而是：

> 数值零件已有，但生成元本身还没有成为一级工程对象。
