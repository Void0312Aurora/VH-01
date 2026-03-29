# VH-01 理论错配与修补清单

## 1. 文档目的

本文用于把 [temp-02.md](./temp-02.md) 中较新的理论叙述，与当前 VH-01 的真实工程状态并列起来看清楚两件事：

1. 哪些对象已经有了比较稳的工程对应；
2. 哪些对象仍然明显错配，或者因为理论对象没有立住而只能依赖工程补丁。

本文不替代 [mapping-audit.md](./mapping-audit.md) 或 [theory-engineering-map.md](./theory-engineering-map.md)。
前两者更偏“已有映射”的整理，而本文更偏“错配与修补顺序”的审计。

## 2. 当前最明显的理论-实现错配

### 2.1 `R_{i,j,l}` 被过早标量化

按 [temp-02.md](./temp-02.md) 的写法，`R_{i,j,l}` 首先应该是一个保留局部响应信息的残差对象；`E_dyn` 只是对该对象进一步取平方范数并聚合后的能量泛函。

但当前实现里，`R_{i,j,l}` 很早就被压成了标量 MSE：

1. [dynamics_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L43) 里直接把 `pred_delta - true_delta` 做平方并对观测维求均值。
2. [response_triangle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L166) 保留下来的上三角对象，也已经是“每个 `(i,j)` 对应一个标量残差能量”的三角阵，而不是原始响应场。
3. 后续 [response_signature](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L924)、[response_operator](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L234)、[local_response_jet_bundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L340) 都建立在这个标量三角上。

这会带来几个直接后果：

1. 响应对象只保留了“强弱”，丢掉了方向、符号、通道差异和局部空间结构。
2. 后续所谓的“响应几何”，实际更接近“残差能量统计的几何”。
3. 由 response-jet 推出来的 `tangent_drift`、`tangent_cov`、`support_tilt`，语义上不再是从局部响应场诱导出来，而是从二次摘要诱导出来。
4. 一些本应可区分的情形会被混在一起，例如：
   - 不同通道上的抵消；
   - 同一能量但方向相反的残差；
   - 同一总能量但局部空间分布不同的残差。

因此，这一项不是“轻微近似”，而是当前理论链条里最明显的语义塌缩点。

### 2.2 局部基点 `x` 与旧版 `trajectory_state` 并不等价

理论里的局部基点 `x` 更接近 `\mathcal M_T` 上某个局部状态点，后续的邻域、切空间、漂移与扩散都应围绕这个对象定义。

旧版代码里的 `trajectory_state` 是整条轨迹经 chart 后再做摘要得到的全局特征；它更像“轨迹摘要坐标”，而不是一个明确的局部状态点。

这会导致：

1. “在 `x` 邻域里做局部响应几何”的语义被削弱；
2. 后面的 kNN 图其实是“轨迹摘要之间的近邻”，不一定等于理论中的局部流形邻域；
3. `x`、`T_x\mathcal M_T`、`b_c(x)`、`\Sigma_x` 并没有严格围绕同一个局部对象定义。

这一项现已通过 [x-trajectory-point-frozen-checklist.md](/home/void0312/AIGC/VH-01/docs/x-trajectory-point-frozen-checklist.md#L1) 进入冻结并完成收口。当前 active 路径里：

1. [trajectory_point](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L468) 被显式立为轨迹点接口；
2. [trajectory_summary_context](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L479) 被保留为摘要上下文；
3. [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L780) 与 [local_measure_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L1252) 默认都已围绕 `trajectory_point` 工作；
4. [trajectory_tangent_frame](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L499) 已改为同时消费 point 与 summary context。

### 2.3 理论里的条件化编码没有在旧版编码器里体现

在 [temp-02.md](./temp-02.md) 的表述里，编码器被允许写成

$$
z_t \approx E_t(X_{1:T}, c).
$$

也就是说，编码端可以依赖整段视频与条件。

旧版 [encode_video](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L315) 实际上是逐帧、无条件编码；条件只进入了解码器和动力学分支：

1. [decode_video](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L299)
2. [step_dynamics](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L550)

因此，旧实现对 `c` 的使用更接近“条件解码 + 条件演化”，而不是“条件化表示学习”。

这一项现已通过 [conditional-encoder-frozen-checklist.md](/home/void0312/AIGC/VH-01/docs/conditional-encoder-frozen-checklist.md#L1) 进入冻结并完成收口。当前 active 路径里：

1. [encode_video](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L337) 已扩展为 `encode_video(video, cond_embed=None)`；
2. [forward](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L358) 已默认走条件化编码；
3. [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L844) 的 teacher/target 重编码路径，在条件已知时也已走条件化编码；
4. query/观测侧仍保留无条件回退编码，不会把现有读取与生成链打断。

### 2.4 `\mu_0 \to \mathfrak T_c \to \mu_c \to Read` 仍未对象化

按 [temp-02.md](./temp-02.md#L302) 的叙述，当前理论要求严格区分：

1. 基准测度 `\mu_0`；
2. 条件更新算子 `\mathfrak T_c`；
3. 更新后的测度 `\mu_c`；
4. 读取算子 `Read`；
5. 读取结果 `\Gamma_k(c), \Gamma_\alpha(c)`。

当前工程里并不存在这些连续对象的显式实现。真实存在的是：

1. [condition_alignment_energy](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L731)
2. [condition_candidate_logits](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L743)
3. [CandidatePosterior](/home/void0312/AIGC/VH-01/src/vh_mvp/support.py#L20)
4. [CandidateSet](/home/void0312/AIGC/VH-01/src/vh_mvp/support.py#L30)

它们是离散代理和读取对象，而不是显式的测度更新链。

### 2.5 `\mathcal L_c` 不是一级工程对象（已按冻结范围修补）

理论里更本体的顺序是

$$
R_{i,j,l}
\rightsquigarrow
\text{局部响应几何}
\rightsquigarrow
\mathcal L_c
\rightsquigarrow
\mu_c.
$$

旧版实现里，`local_measure_targets(...)` 和 `local_measure_loss(...)` 虽然已经在尝试用 response-jet、弱形式平稳性、局部漂移与扩散来逼近这条链，但 `\mathcal L_c` 本身没有成为清晰、独立、可调用的工程对象。

这一项现已按冻结范围完成修补，见：

1. [lc-first-class-design.md](/home/void0312/AIGC/VH-01/docs/lc-first-class-design.md#L1)
2. [lc-first-class-frozen-checklist.md](/home/void0312/AIGC/VH-01/docs/lc-first-class-frozen-checklist.md#L1)

修补后的当前状态是：

1. 模型侧已显式提供 `GeneratorContext` 与 `LocalGenerator`；
2. target 侧已显式提供 `LocalGeneratorTarget`；
3. `local_measure_loss(...)` 的主要 `\mathcal L_c f` 作用已通过 `generator.apply_*()` 接口表达；
4. 冻结测试与 smoke 训练已重新通过。

当前仍未包含在这一项里的，是更深层的 `\mathcal L_0 + \delta \mathcal L_c` 分解与数值重写；那是后续独立问题，不再算作这里的未关闭内容。

### 2.6 理论里的局部邻域 `U_x`，当前已按离散样本层口径修正

这一项现在需要更准确地区分三层对象：

1. 连续层的 `U_x \subset \mathcal M_T`
2. 离散样本层的经验样本云 `\mathcal N_x^{(k)} \subset \mathcal Z^T \cap U_x`
3. 实现层里用于近似经验样本云的 `kNN`

因此，问题不应再被写成“实现用了 `kNN`，所以偏离理论”。更准确的说法是：

1. 在 `\mathcal Z^T` 上用 `kNN` 近似局部邻域，本来就是理论允许的离散化桥梁；
2. 旧版实现真正的问题，是不少局部几何对象只用了“当前 batch 内的临时近邻图”。

旧版主要集中在：

1. [_build_knn_graph](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L285)
2. [_local_response_jet_bundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L340)
3. [local_neighbor_smoothness_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L952)

这会让局部几何强烈依赖 batch 组成，使 `\widehat{\mathcal N}_{x,\mathrm{batch}}^{(k)}` 过度替代了经验样本云本体。

这一项现已按“经验邻域池”口径修补：

1. 新增 [ux-empirical-neighborhood-repair.md](/home/void0312/AIGC/VH-01/docs/ux-empirical-neighborhood-repair.md#L1)
2. `response_jet` 与 `response_smoothness` 都已支持跨 batch 的 detached 邻域引用池
3. 当前实现更接近“`kNN` 近似 `\mathcal Z^T \cap U_x`”，而不再只是纯 batch 图

在此基础上，全局化第一步也已补上：

1. 新增 [globalization-transport-frozen-checklist.md](/home/void0312/AIGC/VH-01/docs/globalization-transport-frozen-checklist.md#L1)
2. reference 邻域现在会缓存局部切向标架
3. `response_jet` 与 `tangent_bundle_compatibility` 已能在 reference 邻居参与时构造离散 transport

当前仍未完成的，是数据集级或图册级全局邻域结构；那属于后续更深层任务。

### 2.7 文档后半段比默认主线实现走得更远

从默认配置看，当前系统的正式主线仍主要是 condition/support/query-aware 读取链，而不是 continuous-measure 主线。

默认主线可参考：

1. [query_balanced_mainline](/home/void0312/AIGC/VH-01/configs/realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_mainline.yaml)

更接近 [temp-02.md](./temp-02.md#L302) 后半段叙述的，是另一条实验配置：

1. [continuous_measure_response_jet_graph_tau](/home/void0312/AIGC/VH-01/configs/realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_continuous_measure_response_jet_graph_tau.yaml)

因此，当前理论主叙述与默认工程主线之间，存在明显的“叙述超前于正式系统状态”的差距。

## 3. 理论缺位导致的工程补丁或语义漂移

### 3.1 `query_responsive_selection` 是读取层补丁，不是理论本体对象

[query_responsive_selection](/home/void0312/AIGC/VH-01/src/vh_mvp/support.py#L134) 的规则是：

1. 先取观测后验候选集；
2. 再取 query-induced planning core；
3. 两者求交；
4. 若交为空，则回退到 planning core。

这条规则在当前工程上有明确价值，但它本质上是“当前没有统一 `\mu_c` 时，为执行链手工拼出的读取补丁”。

### 3.2 `support_refinement_loss` 反映的是后验雕刻，不是自然推出的理论对象

[support_refinement_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L86) 中的 `floor / ceiling / gate / temperature` 一整套阈值结构，说明当前系统是在显式地雕刻离散后验形状。

它对工程有效，但不应被误写成已经实现了 `\mathfrak T_c` 的本体形式。

### 3.3 条件 `c` 在代码里承担了三套不同语义

当前 `c` 同时被用于：

1. 解码条件；
2. 潜动态演化条件；
3. 条件打分或查询条件。

由于缺少统一的条件更新算子，这三条链路语义上并不天然一致：

1. [decode_video](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L299)
2. [step_dynamics](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L550)
3. [condition_alignment_energy](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L731)

### 3.4 当前并存两套切空间语义

一套切空间来自模型头预测：

1. [trajectory_tangent_frame](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L446)

另一套切空间来自 response-jet 目标构造：

1. [_local_response_jet_bundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L340)

训练时常常是前者去拟合后者，但两者并没有围绕同一个局部对象被统一定义。

### 3.5 当前 `measure_log_density` 更像 batch 内相对权重，而非显式全局密度

[measure_log_density_components](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L698) 给出的是一个得分头。

当它在 [local_measure_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L996) 里真正被使用时，又会先在 batch 内 softmax 成权重：

1. [local_measure_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L1143)

这意味着当前“密度”仍然带有很强的 batch-relative 语义。

### 3.6 `graph_tau` 目前更像配置联动下的隐式 target builder

`graph_tau` 的确比单纯 support score 更接近“从基准支持到条件增量”的叙述，但它现在仍埋在 target builder 的一串配置开关里：

1. [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L750)

这说明它尚未成为一级清晰对象，而更像一段实验性构造逻辑。

## 4. 第一优先级修补：恢复结构化 `R_{i,j,l}`

### 4.1 为什么这项必须优先修

如果 `R_{i,j,l}` 一开始就被压成标量，那么后面很多看起来很高级的对象都会跟着偏：

1. `response_signature` 只能描述能量统计；
2. `response_operator` 只能描述能量三角的相关结构；
3. response-jet 推出来的 `drift / covariance / support_tilt` 只能建立在能量几何上；
4. 切空间、扩散、密度目标就会在一开始丢掉最重要的局部响应信息。

因此，这一项应当被视为“上游对象错了，后面全链都跟着歪”的问题。

### 4.2 修补目标

修补后的结构应当满足：

1. `R_{i,j,l}` 先保留为一个有方向、有符号、有局部结构的残差对象；
2. `E_dyn` 只是从该对象导出的标量能量，不再反过来替代该对象；
3. response geometry 使用的是“结构化响应描述”，而不是“每个 `(i,j)` 一个标量能量”；
4. 当前已有的 `dyn_loss` 训练路径可以先保持兼容，不必一次性推翻。

### 4.3 推荐的工程修法

推荐采用“先分离对象，再逐步替换下游”的方式，而不是直接大改整个训练链。

第一步：把“原始残差对象”和“标量能量”分离。

建议新增一个响应构造函数，例如：

1. `response_residual_triangle(...)`
2. 或 `response_triangle_bundle(...)`

它至少返回三类对象：

1. `residual_tensor`
   - 形状上保留 `(batch, span, start, channels, height, width)` 或其等价展平形式；
   - 对应真正的 `pred_delta - true_delta`。
2. `energy_triangle`
   - 即当前实现中的标量残差能量三角；
   - 只作为 `E_dyn`、日志与兼容指标使用。
3. `mask`
   - 用于描述上三角中哪些位置有效。

这样做的关键是：

1. 训练仍可继续使用标量 `dyn_loss`；
2. 但下游几何分析不再被迫直接消费标量能量。

第二步：在结构化残差上定义“响应描述子”，而不是直接用标量能量当通道。

这里不建议一上来就引入复杂的可学习 response encoder。
更稳的第一版是用确定性、可解释的通道化方式，例如：

1. 保留 signed residual 的低频空间池化；
2. 同时保留 `abs residual` 或 `squared residual` 的池化；
3. 必要时再拼接通道均值、方差、极值等简单统计。

一个可行的第一版是：

$$
\Psi_{i,j}^{(n)}
=
\mathrm{Concat}\Bigl(
\mathrm{AvgPool}(R_{i,j}^{(n)}),
\mathrm{AvgPool}(|R_{i,j}^{(n)}|),
\mathrm{AvgPool}\bigl((R_{i,j}^{(n)})^2\bigr)
\Bigr).
$$

这样得到的 `\Psi_{i,j}^{(n)}` 仍是低维对象，但已经比单个标量更接近“局部响应描述”。

第三步：把当前 response 几何链路改为消费结构化描述子。

也就是说：

1. `response_signature` 不再从 `energy_triangle` 构造；
2. `response_channels` 不再等于“每个 `(i,j)` 一个标量能量”；
3. response-jet 拟合的对象改为“结构化响应描述子在局部邻域里的变化”。

换言之，当前链条

$$
\text{scalar energy triangle}
\to
\text{response signature}
\to
\text{response jet}
$$

应改成

$$
\text{structured residual tensor}
\to
\text{response descriptor}
\to
\text{response jet}.
$$

第四步：保留当前 `dyn_loss`，但把它降回“能量泛函”而不是“响应对象本体”。

这一步很重要，因为它可以让改造先落在对象层，而不必一开始就打乱当前训练稳定性。

### 4.4 一条更稳的分阶段实施顺序

建议按下面顺序推进：

1. 新增结构化 residual bundle，先不动现有 loss。
2. 让 `response_signature` 支持从结构化 residual bundle 构造描述子。
3. 让 response-jet 和 response target builder 消费新的描述子。
4. 验证 `response_operator`、`support_tilt`、`tangent_drift`、`tangent_cov` 是否变得更稳定、更有区分力。
5. 最后再考虑是否用新的结构化响应去改写部分动态项或条件项。

### 4.5 当前不建议的做法

当前不建议直接做以下事情：

1. 直接把全分辨率残差场不加压缩地塞进下游几何模块。
   - 这样维度过高，容易造成新的数值不稳定。
2. 一上来就完全废弃当前 `dyn_loss`。
   - 当前 `dyn_loss` 作为训练能量仍然有工程价值。
3. 直接上可学习 response encoder 而不先建立确定性基线。
   - 否则会把“理论对象恢复”和“新模型学习失败”混在一起。

### 4.6 下一步最小代码切入点

如果按“最小侵入、先恢复对象、再替换下游”的原则推进，那么第一刀建议落在下面几个位置：

1. [objectives.py](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py)
   - 新增 `response_triangle_bundle(...)`；
   - 让它同时产出 `residual_tensor / energy_triangle / mask`；
   - 保留当前 [response_triangle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L166) 作为兼容包装器，只返回旧式 `energy_triangle + mask`。
2. [objectives.py](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py)
   - 新增 `response_descriptor_from_bundle(...)`；
   - 用确定性池化把 `residual_tensor` 变成低维描述子；
   - 先不改训练损失，只改 response 几何相关链路的输入来源。
3. [objectives.py](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py)
   - 让 [response_signature](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L924) 支持从新的 descriptor 构造；
   - 让 `local_measure_targets(...)` 优先消费 descriptor，而不是直接消费标量能量三角。
4. [config.py](/home/void0312/AIGC/VH-01/src/vh_mvp/config.py)
   - 之后可考虑增加 `response_descriptor_mode`、`response_pool_kernel`、`response_include_abs` 之类的显式配置；
   - 但第一阶段甚至可以先写死一个确定性默认值，减少变量。
5. [mvp.py](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py)
   - 第一阶段尽量不动模型主体；
   - 只有当 descriptor 维度需要喂给 `response_context_head` 时，再补最小限度的维度适配。

也就是说，第一阶段的目标不是“重写整个模型”，而是：

1. 先恢复结构化响应对象；
2. 再把 response geometry 的输入从“标量能量”换成“结构化响应描述子”；
3. 当前 `dyn_loss`、`query`、`support` 主线先保持不动。

### 4.7 当前已完成的第一阶段修补

这一项现在已经完成了第一阶段的对象恢复和下游接线：

1. [response_triangle_bundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L273) 已经把 `residual_triangle / energy_triangle / mask` 分开保存；
2. 兼容接口 [response_triangle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L297) 仍然保留，只返回旧式的 `energy_triangle + mask`；
3. [response_descriptor_triangle_from_bundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L315) 和 [response_descriptor_from_bundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L353) 已经把结构化 residual 变成确定性的响应描述子；
4. [response_signature](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L1147) 已经不再直接从标量 `energy_triangle` 构造；
5. [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L778) 中的 response-jet / response-operator 链，已经改为优先消费 descriptor，而不是旧的标量能量通道。

这一阶段完成后，最关键的变化是：

1. 代码里终于有了可显式访问的“原始响应对象”；
2. response geometry 不再只能建立在 MSE 三角上；
3. 旧的 `dyn_loss` 仍保持兼容，没有强行打断当前主线训练。

但这一项还没有完全结束，剩下的缺口主要有两类：

1. 当前 `response_signature` 仍是面向固定维度 `response_context` 的压缩摘要，还不是“完整 descriptor 直接入模”；
2. `query / support / condition update` 这一层目前还没有直接消费新的结构化响应对象。

### 4.9 当前已完成的第二阶段修补

在第一阶段之后，`response_signature` 仍然主要依赖 signed scalar summary，这一层语义还偏弱。

这一轮已经继续往前推进了一步：

1. [response_signature_dim](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L228) 现在支持新的 `descriptor_span_stats` 与 `descriptor_full_triangle` 两种模式；
2. [response_signature](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L1147) 和 [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L778) 已经可以直接从 `descriptor_triangle` 构造 signature，而不必先退回 signed 标量三角；
3. 训练入口和评估脚本中的模型构建，也已经按 `channels` 正确计算新的 `response_signature_dim`，避免 descriptor-aware 模式在建模入口处失配。

这一步的意义在于：

1. `response_context_head` 终于有机会直接看到结构化响应描述子的统计，而不是只能看到每个 `(i,j)` 的 signed scalar；
2. 局部几何头消费的 response context，和上游 residual bundle 之间的语义链条更短了；
3. 同时旧的 `span_stats / full_triangle` 模式仍然保留，兼容已有配置和旧 checkpoint 行为。

### 4.10 当前验证结果

这一阶段修补已经做过五类验证：

1. 语法级验证：
   - `python -m py_compile src/vh_mvp/losses/objectives.py src/vh_mvp/models/mvp.py src/vh_mvp/train/trainer.py src/vh_mvp/support.py`
2. 主线训练烟测：
   - `PYTHONPATH=src python scripts/train_mvp.py --config configs/smoke.yaml`
   - 2 个 epoch 正常结束，默认训练链没有因为 residual bundle 改动被打断。
3. response-jet 分支训练与生成烟测：
   - 以 [shorttrain_continuous_measure_chart_target_coupled_tangent_response_jet_graph_tau_balanced.yaml](/home/void0312/AIGC/VH-01/configs/shorttrain_continuous_measure_chart_target_coupled_tangent_response_jet_graph_tau_balanced.yaml) 为蓝本，做了小批量 `train_one_epoch + evaluate` 烟测；
   - `local_drift / local_diffusion / measure_stationarity / response_operator_*` 都正常产出有限值；
   - 另外还基于 `runs/smoke/last.pt` 跑过一次 synthetic 生成闭环烟测，`encode -> rollout -> decode -> posterior -> query_responsive_selection` 全流程输出有限，且候选执行链正常返回。
4. descriptor-aware signature 烟测：
   - 将 `response_signature_mode` 临时切到 `descriptor_span_stats` 后，`signature_shape` 与 `expected_signature_dim` 对齐；
   - 小批量 `train_one_epoch + evaluate` 正常运行；
   - 同一模型下的 rollout / decode 生成链也保持有限输出。
5. 固定测试与冻结收口验证：
   - 新增 [test_r_ijl_frozen_checklist.py](/home/void0312/AIGC/VH-01/tests/test_r_ijl_frozen_checklist.py#L1)；
   - `PYTHONPATH=src pytest -q tests/test_r_ijl_frozen_checklist.py` 通过，结果为 `5 passed`；
   - 测试覆盖了 residual bundle、`dynamics_loss` 聚合一致性、descriptor-aware signature 维度、`local_measure_loss` 反向传播、最小生成 smoke。

### 4.11 冻结问题关闭状态

按 [r-ijl-frozen-checklist.md](/home/void0312/AIGC/VH-01/docs/r-ijl-frozen-checklist.md#L1) 的关闭标准，`R_{i,j,l}` 被过早标量化这一项现在已经关闭。

关闭的具体原因是：

1. 原始 residual 对象已经通过 [response_triangle_bundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L273) 显式存在；
2. [dynamics_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L45) 已和 residual bundle 共享底层 helper，不再维持独立的 scalar-first 主路径；
3. 旧的 scalar-first response helper 已经移除，active response geometry 不再依赖它们；
4. 固定测试、`pytest` 与默认 smoke 训练都已经完成并通过。

这里的“关闭”只针对 `R_{i,j,l}` 标量化这一项。
其它理论偏差，例如 `\mu_0 / \mathfrak T_c / \mu_c / Read` 的对象化，并没有因为这一步自动关闭。

## 5. 建议的修补优先级

在当前阶段，更合理的修补顺序是：

1. `R_{i,j,l}` 的对象塌缩问题已经关闭；
2. 局部基点 `x` 与 `trajectory_state` 混用问题已经关闭；
3. 下一步更值得推进的是 `\Psi_x \mapsto I_x^\star \mapsto (P_x,b_x,\Sigma_x,\tau_c)` 这条 target 构造链的离散化；
4. 然后再明确 `\mu_0 / \mathfrak T_c / \mu_c / Read` 的工程边界；
5. 最后再决定 continuous-measure 链路是否进入默认主线。

## 6. 当前结论

当前 VH-01 的问题不在于“什么都没有实现”，而在于：

1. 前半段动态主线实现较强；
2. 后半段理论对象实现较弱；
3. 两个最明显的上游错配点 `R_{i,j,l}` 标量化、`x` 与 `trajectory_state` 混用，已经完成冻结修补；
4. 现在更关键的缺口转向“如何从局部响应稳定构造 `I_x^\star` 与生成元 target”。

因此，下一步最值得做的，不是继续堆新的读取规则或新的几何目标，而是把 `\Psi_x \mapsto I_x^\star` 这条中介层真正做稳。
