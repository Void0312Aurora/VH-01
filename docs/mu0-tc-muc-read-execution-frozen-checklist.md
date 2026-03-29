# `\mu_0 -> \mathfrak T_c -> \mu_c -> Read` 执行闭环冻结清单

## 范围

本轮只处理一件事：

> 把 query 执行链从“条件 posterior + 计划 posterior + 规则拼接”，切到“条件识别只负责选 `c`，真正执行由 `ConditionalMeasure -> MeasureReadout` 完成”。

这对应 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L1455) 到 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L1459) 的主线。

## 理论依据

理论要求的顺序是：

1. 训练阶段给出 `\mu_0`；
2. 条件 `c` 诱导 `\mathfrak T_c : \mu_0 \mapsto \mu_c`；
3. 读取发生在 `\mu_c` 上，而不是发生在另一个无关 posterior 上。

因此在执行层：

1. `p(c \mid Z_obs)` 仍然可以存在，但它只属于条件识别分支；
2. 一旦要执行或读取 rollout 候选，就应当进入 `ConditionalMeasure`；
3. `top1 / top-k / alpha-mass set` 都应从 `MeasureReadout` 来，而不是从计划 posterior 规则来。

## 当前缺口

当前对象层已经有：

1. `BaseMeasure`
2. `ConditionalTilt`
3. `ConditionalMeasure`
4. `MeasureReadout`
5. `ConditionInferencePosterior`

但 trainer 里的 query 执行仍是：

1. `obs_posterior = p(c \mid Z_obs)`
2. `plan_posterior = softmax(plan_logits)`
3. `query_responsive_selection(obs_posterior, plan_posterior)`

这还不是 `\mu_0 -> \mathfrak T_c -> \mu_c -> Read` 的执行闭环。

## 不在范围内

以下不在本轮：

1. 重写 `ConditionInferencePosterior` 本身
2. 重写 `ConditionalMeasure` 的参数化
3. 引入 `\nu_c`
4. query checkpoint 评分公式重写

## 实施项

### A1. 新增 measure-execution helper

新增一个显式 helper，把：

1. 观测条件 posterior
2. 每个条件下的 rollout `log \mu_c`

组合成真正的执行读取结果。

最小形式为：

1. `obs_posterior -> condition_set`
2. `condition_set` 内对各 `\mu_c` 做 posterior 加权 mixture
3. 对 mixture 后的 rollout 权重做 `MeasureReadout`

### A2. trainer query eval 接线

`evaluate_query_responsive_execution(...)` 改成：

1. 先算 `obs_posterior`
2. 再对 rollout 候选族显式构造 `ConditionalMeasure`
3. 最后调用新的 measure-execution helper

旧 `query_responsive_selection(...)` 保留，但不再是 active query 执行主路径。

### A3. 测试

至少覆盖：

1. posterior + measure mixture 的数学正确性
2. `evaluate_query_responsive_execution(...)` 在小 synthetic catalog 上可正常运行
3. pytest 通过

## 关闭标准

满足以下条件即可关闭：

1. query 执行主路径不再依赖 `plan_posterior`
2. rollout 执行集合来自 `MeasureReadout`
3. 条件识别 posterior 与 measure readout 明确分层
4. pytest 通过
5. 执行闭环烟测通过
