# VH-01 `\mu_0 \to \mathfrak T_c \to \mu_c \to Read` 对象化设计

## 1. 文档目的

本文只回答一个问题：

> 在当前理论路线下，如何把 `\mu_0 \to \mathfrak T_c \to \mu_c \to Read` 真正落成工程对象，而不是继续停留在“若干离散分数 + posterior + 读取规则”的拼接层？

这里不处理：

1. `R_{i,j,l}` 的结构恢复；
2. 局部基点 `x` 与 `trajectory_point` 的关系；
3. 条件化编码器；
4. `\mathcal L_c` 的完全对象化。

本文只聚焦一件事：把“基准测度、条件更新、条件测度、读取”这四层正式分开。

## 2. 理论上应当区分的对象链

按 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L317) 与 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L838) 的表述，当前更准确的链条是：

$$
\mu_0
\xrightarrow{\;\mathfrak T_c\;}
\mu_c
\xrightarrow{\;\mathrm{Read}\;}
\Gamma.
$$

其中：

1. `\mu_0` 是训练后在背景轨迹空间上诱导出的基准测度；
2. `\mathfrak T_c` 是条件 `c` 对基准测度施加的更新或尖锐化；
3. `\mu_c` 是条件更新后的测度；
4. `Read` 才是 `top-1`、`top-k`、`alpha`-质量区域等读取步骤。

若采用相对于 `\mu_0` 的密度比语言，则 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L846) 给出

$$
d\mu_c(Z)
=
\frac{1}{\Xi_c}
e^{\tau_c(Z)}\,d\mu_0(Z),
$$

因此在工程上最自然的分层其实是：

$$
\log d\mu_0
\;+\;
\tau_c
\;\Longrightarrow\;
\log d\mu_c
\;\Longrightarrow\;
\mathrm{Read}[\mu_c].
$$

这里尤其要注意：

- `\tau_c` 更接近“条件倾斜”或“对数密度修正”；
- `Read[\mu_c]` 是更新完成后的读取；
- “条件识别 posterior”并不等于 `\mu_c`。

## 3. 当前代码里已经有什么

### 3.1 已有一半：密度三分量雏形

[measure_log_density_components(...)](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L833) 在 `tilted` 模式下已经返回：

1. `base`
2. `tilt`
3. `total = base + tilt`

这三项天然接近：

1. `base` ↔ `\log d\mu_0`
2. `tilt` ↔ `\tau_c`
3. `total` ↔ `\log d\mu_c`

而 [local_measure_loss(...)](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L1360) 又已经把 `total_log_density` 转成 batch 内权重 `density_weights`，用于弱形式近似。

因此，仓库里并不是完全没有 `\mu_0 / \tau_c / \mu_c` 的底座，而是这套底座还没有被正式对象化。

### 3.2 已有另一半：读取对象

[support.py](/home/void0312/AIGC/VH-01/src/vh_mvp/support.py#L20) 到 [support.py](/home/void0312/AIGC/VH-01/src/vh_mvp/support.py#L153) 已经正式实现了：

1. `CandidatePosterior`
2. `CandidateSet`
3. `QueryResponsiveSelection`

这套对象非常适合承担读取层职责，但它们当前操作的输入仍然主要是 `softmax(logits)` 形式的离散 posterior，而不是明确命名的 `\mu_c` 切片。

### 3.3 另有一条独立分支：条件识别 posterior

[condition_candidate_logits(...)](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L878) 给的是“给定轨迹，对条件目录打分”；
[build_candidate_posterior(...)](/home/void0312/AIGC/VH-01/src/vh_mvp/support.py#L69) 则把它转成离散 posterior。

这条链回答的问题更像是：

$$
p(c \mid Z_{\mathrm{obs}}).
$$

它是条件识别或条件观测分布，不是固定条件 `c` 后的轨迹测度 `\mu_c`。

### 3.4 query 执行里混了两种不同对象

在 [evaluate_query_responsive_execution(...)](/home/void0312/AIGC/VH-01/src/vh_mvp/train/trainer.py#L395) 里，当前同时存在：

1. `obs_posterior`：来自观测轨迹对条件目录的 posterior；
2. `plan_posterior`：来自 rollout 轨迹族相对于查询条件的对齐打分。

随后它们一起进入 [query_responsive_selection(...)](/home/void0312/AIGC/VH-01/src/vh_mvp/support.py#L134)。

这在工程上可运行，但理论上它并不等于“对同一个 `\mu_c` 做读取”。

## 4. 为什么现在还不能算对象化完成

核心不是“没有分数”，而是“对象边界没有立住”。

### 4.1 `joint` 模式无法分离 `\mu_0` 与 `\mathfrak T_c`

在 [measure_log_density_components(...)](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L843) 的 `joint` 模式下，函数直接返回：

1. `base = total`
2. `tilt = 0`
3. `total`

这意味着：

- 当前总密度可以被预测；
- 但基准测度与条件倾斜没有被分开。

因此，只要还停留在 `joint` 模式，就无法严格对象化

$$
\mu_0 \to \mathfrak T_c \to \mu_c.
$$

### 4.2 `\mathfrak T_c` 还不是显式算子

现在的 `tilt_head` 只是输出一个条件相关修正项，但没有一个正式对象表达：

$$
\mathfrak T_c(\mu_0)=\mu_c.
$$

更具体地说，代码里缺少：

1. 输入基准测度表示；
2. 输出条件更新后测度表示；
3. 保存归一化常数或有限近似归一化权重；
4. 显式说明这是“更新”而不是“读取”。

### 4.3 读取层还没有明确接在 `\mu_c` 上

当前 `CandidatePosterior` 与 `CandidateSet` 很成熟，但它们主要消费的是：

1. `condition_candidate_logits`
2. `condition_alignment_energy`
3. 由这些 logits 经过 `softmax` 得到的 posterior

而不是一个明确命名的 `ConditionalMeasure`。

因此，当前更准确的状态是：

- 读取对象已经有了；
- 但被读取的理论对象还没有正式起名并封装。

### 4.4 条件后验与测度读取被混成了一层

当前至少有两种“posterior”：

1. 条件 posterior：`p(c \mid Z_obs)`；
2. 轨迹测度读取：`Read[\mu_c]` 的有限近似。

它们不是一个对象。

如果继续沿用同一套 `posterior / candidate set / selection` 词汇而不拆层，就会持续出现：

- 条件识别结果被误写成 `\mu_c`；
- 读取补丁被误写成 `\mathfrak T_c`；
- query 规则被误写成理论本体。

## 5. 推荐的对象分层

下一阶段应当把对象明确拆成 5 层。

### 5.1 `BaseMeasure`

表示固定轨迹点 `x` 或轨迹族切片上的基准测度代理。

最小字段建议为：

1. `state`
2. `log_base_density`
3. `normalizer` 或有限近似归一化信息

它回答的是：

> 在未施加当前条件 `c` 时，这些轨迹点的基准支持强度是什么？

### 5.2 `ConditionalTilt`

表示条件 `c` 对基准测度施加的局部倾斜。

最小字段建议为：

1. `cond_embed`
2. `log_tilt`
3. 可选的 `transport_metadata`

它回答的是：

> 条件 `c` 让哪些区域相对更重、哪些区域相对更轻？

### 5.3 `ConditionalMeasure`

表示条件更新后的测度代理。

最小字段建议为：

1. `state`
2. `log_base_density`
3. `log_tilt`
4. `log_total_density`
5. `normalized_weights`

它是第一层真正对应 `\mu_c` 的工程对象。

### 5.4 `MeasureReadout`

表示对 `ConditionalMeasure` 的读取。

最小能力应包括：

1. `top1`
2. `topk`
3. `alpha_mass_set`
4. `masked argmax / restricted argmax`

这一层应只处理：

$$
\mathrm{Read}[\mu_c].
$$

它不应再负责“条件更新”或“条件识别”。

### 5.5 `ConditionInferencePosterior`

这是额外保留的一条独立分支，不属于 `\mu_0 \to \mathfrak T_c \to \mu_c \to Read` 主链。

它单独表示：

$$
p(c \mid Z_{\mathrm{obs}}).
$$

保留它的原因不是理论需要，而是工程上 query / catalog / 条件识别仍然需要它。

## 6. 最小接口草案

下面给出推荐的最小接口草案。这里是对象边界草图，不是最终代码签名。

```python
@dataclass
class BaseMeasure:
    state: torch.Tensor
    log_base_density: torch.Tensor


@dataclass
class ConditionalTilt:
    cond_embed: torch.Tensor
    log_tilt: torch.Tensor


@dataclass
class ConditionalMeasure:
    state: torch.Tensor
    log_base_density: torch.Tensor
    log_tilt: torch.Tensor
    log_total_density: torch.Tensor
    normalized_weights: torch.Tensor


@dataclass
class MeasureReadout:
    top1_idx: torch.Tensor
    topk_idx: torch.Tensor | None
    alpha_mask: torch.Tensor | None
    alpha_mass: torch.Tensor | None


@dataclass
class ConditionInferencePosterior:
    logits: torch.Tensor
    probs: torch.Tensor
    top1_idx: torch.Tensor
```

最关键的不是 dataclass 形式本身，而是：

1. `ConditionalMeasure` 必须先于 `MeasureReadout`；
2. `ConditionInferencePosterior` 必须与 `ConditionalMeasure` 分离；
3. query 链如果要组合两者，也必须在更高一层显式组合，而不是在底层对象里混写。

## 7. 与当前代码的最小映射

按当前仓库结构，最自然的最小映射是：

1. `BaseMeasure.log_base_density`
   - 来自 [measure_log_density_components(...)](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L833) 的 `base`
   - 前提是 `measure_density_mode = tilted`

2. `ConditionalTilt.log_tilt`
   - 来自同一接口返回的 `tilt`

3. `ConditionalMeasure.log_total_density`
   - 来自同一接口返回的 `total`

4. `ConditionalMeasure.normalized_weights`
   - 可先沿用 [local_measure_loss(...)](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L1385) 当前的 batch 内 softmax 近似

5. `ConditionInferencePosterior`
   - 来自 [condition_candidate_logits(...)](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L878) + [build_candidate_posterior(...)](/home/void0312/AIGC/VH-01/src/vh_mvp/support.py#L69)

6. `MeasureReadout`
   - 初期可复用 [CandidateSet](/home/void0312/AIGC/VH-01/src/vh_mvp/support.py#L30) 的 `alpha`-质量截取逻辑
   - 但输入应从“logits posterior”切到“conditional measure weights”

## 8. 当前最值得避免的误写

后续文档和实现中，以下说法都应避免：

1. “`condition_candidate_logits` 就是 `\mu_c`。”
2. “`CandidatePosterior` 已经等价于条件测度。”
3. “`query_responsive_selection` 实现了 `\mathfrak T_c`。”
4. “条件读取 posterior 与轨迹测度读取是同一回事。”

更准确的说法应当是：

1. `condition_candidate_logits` 是条件识别分支上的离散打分；
2. `CandidatePosterior` 是读取工具，不是理论本体；
3. `measure_log_density_components` 提供了 `\mu_0 / \tau_c / \mu_c` 的半成品底座；
4. 当前真正缺的是对象封装与链路接线，而不是继续增加一个新的 posterior 规则。

## 9. 后续实施顺序

若按风险最小的方式推进，顺序应当是：

1. 先把 `tilted` 路径上的 `base / tilt / total` 封装成正式对象；
2. 再实现 `ConditionalMeasure -> MeasureReadout` 的读取接口；
3. 然后把 `ConditionInferencePosterior` 与 `MeasureReadout` 在 query 层显式分离；
4. 最后再决定是否要把现有 `query_responsive_selection` 重写成更理论化的高层组合器。

这里不建议的顺序是：

1. 先继续雕刻 posterior 阈值；
2. 先重写 query 规则；
3. 在 `joint` 模式下强行声称已实现 `\mu_0 \to \mathfrak T_c \to \mu_c`。

## 10. 本文结论

这条链当前的真实状态不是“完全没有实现”，也不是“已经实现完毕”，而是：

> 我们已经有了 `\mu_0 / \tau_c / \mu_c` 的密度三分量雏形，也已经有了成熟的读取层工具；真正缺失的是把它们正式连接成
> `BaseMeasure -> ConditionalTilt -> ConditionalMeasure -> MeasureReadout`
> 的对象链，并把条件识别 posterior 从这条链里剥离出去。

因此，下一步的核心任务不是再写一个 posterior 补丁，而是先把对象边界立住。
