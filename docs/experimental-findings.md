# VH-01 实验总结与结论

## 1. 写作目的

本文用于统一整理当前阶段已经完成的实验结果，并给出与理论主线对应的结论。

完整实验清单仍以 [runs/TEST_INDEX.md](/home/void0312/AIGC/VH-01/runs/TEST_INDEX.md) 为准；本文不逐条重复所有日志，而是把关键阶段、核心结果与最终发现组织成一条清晰主线。

## 2. 研究对象与实验边界

当前全部实验都发生在同一条 MVP 路线内部：

1. 数据：`UCF101 subset semantic` 小规模真实视频子集；
2. 模型底座：轻量视频自编码器 + 基准动力模块 + 条件动态修正模块；
3. 理论对象：
   - 潜轨迹 `Z_{1:T}`
   - 残差张量 `R_{i,j,l}`
   - 动态一致性能量 `E_dyn`
   - 训练后隐含的先验测度 `\mu_0`
   - 条件诱导的更新或尖锐化算子 `\mathfrak T_c`
   - 更新后的测度 `\mu_c`
   - `top-k`、高支持区域等读取结果

需要明确的是：

- 当前实验并未显式实现 `\mathcal M_T`、`\mu_0`、`\mathfrak T_c`、`\mu_c`、`\nu_c` 或显式生成元；
- 当前阶段更多验证的是：这些理论对象是否已经在工程上出现了最小的离散代理与读取机制。

## 3. 实验演进主线

### 3.1 从普通 `top-1` 路线到语义结构路线

前期实验主要围绕：

- 数据链路能否跑通；
- 语义条件能否替代简单哈希条件；
- `semantic_summary` 是否能承载类别信息。

这一阶段的代表性结果是：

- `semantic prototype` 版本把语义线性 `top-1` 提升到 `0.5000`
- MLP probe 同步达到 `0.5000`

这说明：

1. `semantic_summary` 的类别几何确实被显著强化；
2. 但这仍然主要是“下游可读性”层面的结论，而不是理论主线本身的验证。

### 3.2 从 `top-1` 转向支持分布指标

后续实验确认：

- `top-1` 不是当前理论最直接对应的目标；
- 更关键的对象应当是：
  - 真实条件质量 `avg_p_true`
  - 支持宽度 `effective_support_ratio`
  - 候选集大小 `k_alpha`
  - 覆盖指标 `true_in_set_alpha`

在这一阶段，旧版 `semantic_prototype` 被发现存在一个重要问题：

- 训练链路与支持评测对象并不完全对齐；
- 因此此前观测到的一部分“锐化”并不可靠。

### 3.3 对齐版 `aligned`

`aligned` 版本修复了：

1. 条件训练与评测候选集不一致的问题；
2. 负样本与真实有效条件集合脱节的问题；
3. checkpoint 选择与支持目标不一致的问题。

其结果是：

- `cond_top1 = 0.2375`
- `avg_p_true = 0.1221`
- `effective_support_ratio = 0.9554`
- `k_0.90_mean = 8.8375`
- `true_in_set_0.90 = 1.0000`

结论：

- `aligned` 成为了“高覆盖、低锐化”基线；
- 它证明了此前一部分锐化现象确实来自目标错配。

### 3.4 支持链路改造版 `support_refined`

在 `aligned` 基础上，进一步把“真实条件质量 + 支持收缩”写入训练损失，形成 `support_refined`。

其核心结果是：

- `cond_top1 = 0.2375`
- `avg_p_true = 0.1727`
- `effective_support_ratio = 0.6218`
- `k_0.90_mean = 5.6125`
- `true_in_set_0.90 = 0.8250`
- `semantic top-1 = 0.5125`

与 `aligned` 相比：

- `avg_p_true` 明显提高；
- 支持宽度明显收缩；
- 候选集显著变小；
- 但覆盖回落。

结论：

- `support_refined` 第一次把系统推到了“更强收缩、更高真实质量”的新工作区间；
- 当前主要矛盾也由此变得清楚：
  - 不是“能不能收缩”
  - 而是“如何在收缩与覆盖之间选取工作点”

## 4. 条件更新代理与读取层实验

### 4.1 离散读取对象已成为正式模块

当前代码中，被正式对象化的首先不是理论中的 `\mathfrak T_c`，而是更新后离散代理上的读取对象：

- `CandidatePosterior`
- `CandidateSet`
- `top-k` / `alpha`-质量区域对应的成员提取接口

这意味着：

- 读取结果已经具备了被调用、被记录、被比较的工程形式；
- 但这并不等于理论本体中的测度更新算子已经被显式实现。

### 4.2 最小读取约束 rollout 已成立

在 `support_refined` 上，`alpha=0.90` 的候选集约束 rollout 结果为：

- `avg_set_size = 5.6125 / 10`
- `true_in_set_rate = 0.8250`
- `oracle_best_in_set_rate = 0.7500`
- `avg_set_best_future_mse = 0.044206`
- `avg_full_oracle_best_future_mse = 0.043778`
- 差距仅为 `0.000429`

结论：

- 读取得到的高质量区域已经能作为一个有实际作用的受限搜索空间；
- 它明显缩小了搜索空间，同时没有严重损坏 rollout 质量。

对照 `aligned`：

- `avg_set_size = 8.8375 / 10`
- `oracle_best_in_set_rate = 1.0000`
- 但几乎没有形成有效筛选

因此：

- `aligned` 更像保守宽松基线；
- `support_refined` 更像真正有筛选力的读取层底座。

### 4.3 条件扰动实验已经给出读取层结构性结果

在 `support_refined` 上，`alpha=0.90` 的条件扰动实验结果是：

近邻扰动：

- `avg_set_jaccard = 0.9775`
- `top1_switch_rate = 0.1625`
- `true_exit_rate = 0.0375`

远扰动：

- `avg_set_jaccard = 0.8725`
- `top1_switch_rate = 0.6500`
- `true_exit_rate = 0.1250`

结论：

- 近邻条件变化下，读取结果基本保持稳定；
- 远条件变化下，读取结果与选择会发生明显迁移；
- 这说明当前离散读取层已经呈现出“局部稳定、远距可迁移”的结构。

这一步很重要，因为它意味着：

- 读取结果并非统计噪声；
- 条件变化确实在读取层面诱导出了有结构的响应。

### 4.4 读取约束生成对照的结论

在 `support_refined` 上，比较四种模式：

1. `direct query`
2. `support top1`
3. `set-best`
4. `full-oracle`

在 `alpha=0.90` 下，`true query` 的代表结果为：

- `direct_query_mse = 0.046843`
- `support_top1_mse = 0.048260`
- `set_best_mse = 0.044006`
- `full_oracle_mse = 0.043778`

这一结果的含义非常明确：

1. `support_top1` 不能直接替代查询条件；
2. 离散读取结果真正的价值不在于“直接取支持最高候选”；
3. 读取结果的价值在于：它为后续搜索提供了有效且较小的工作空间。

换句话说，当前实验支持的不是：

> 支持打分可以直接给出最终生成结果

而是：

> 更新后离散代理的高质量读取结果，可以把最终生成/选择问题约束到一个更好的局部空间中。

### 4.5 读取规则的第一轮研究

在前述结论基础上，我们继续追问：

> 如果更新后离散代理的读取结果本身已经有价值，那么应当如何在这些读取结果内部选择最终执行项？

为此，我们比较了以下几类非 oracle 规则：

1. 纯观测后验：`obs_top1`
2. 纯规划后验：`plan_top1`
3. 先由规划候选集约束，再由观测后验重排：`obs_on_plan_set_top1`
4. 先由规划候选集约束，再由联合分数重排：`joint_plan_set_top1`
5. 全局联合分数：`joint_union_top1`
6. 作为下界参考的 `plan_set_best`

在 `support_refined`、`alpha=0.90`、`true query` 下，结果为：

- `direct_query_mse = 0.046843`
- `obs_top1_mse = 0.046727`
- `plan_top1_mse = 0.048260`
- `obs_on_plan_set_top1_mse = 0.046938`
- `joint_plan_set_top1_mse = 0.046846`
- `joint_union_top1_mse = 0.046638`
- `plan_set_best_mse = 0.044006`

这一组结果说明：

1. 单独使用 `plan_top1` 仍然最差，说明“谁的支持最高”还不能直接当作最终选择规则。
2. 联合规则确实比 `plan_top1` 稳，也能略微优于 `direct_query`。
3. 但所有非 oracle 规则与 `plan_set_best` 之间仍存在明显差距。

进一步看结构统计，在 `support_refined`、`true query` 下：

- `avg_obs_set_size = 5.6125`
- `avg_plan_set_size = 9.0000`
- `avg_intersection_size = 5.2250`
- `obs_plan_jaccard = 0.5621`

这意味着：

- 观测后验候选集与规划候选集并没有塌缩为同一个集合；
- 但它们的交集又足够大，使得简单联合规则很容易偏向观测端。

这一点可以从匹配率上直接看到。对 `support_refined` 而言：

- `joint_union_top1` 在 `true query` 下有 `87.5%` 的样本与 `obs_top1` 相同；
- 在 `far query` 下仍有 `86.25%` 的样本与 `obs_top1` 相同；
- 在 `far query` 下，`joint_union_top1` 对查询条件的直接匹配率为 `0.0%`。

因此，第一轮选择规则研究给出的真正结论不是“已经找到最终规则”，而是：

> 简单的观测-规划联合打分虽然能改善 `plan_top1`，但仍然过于依赖观测端，还不足以成为查询响应意义下的最终选择规则。

对照 `aligned` 版本，上述现象更明显：

- `avg_obs_set_size = 8.8375`
- `avg_plan_set_size = 9.0000`
- `avg_intersection_size = 8.3250`
- `obs_plan_jaccard = 0.8794`

这说明当观测后验候选集过宽时，选择规则几乎会退化成观测端主导，规划后验难以发挥实质作用。

### 4.6 查询响应读取规则的正式化与执行接入

在第一轮选择规则研究之后，我们把一个更严格的读取补丁正式写入代码。它不等于理论中的 `\mathfrak T_c`，而是当前缺乏统一 `\mu_c` 时的工程近似：

1. 先由观测后验得到 `T_obs^\alpha(x)`
2. 再由查询条件得到规划核心 `C_plan^\beta(c)`
3. 构造统一执行集合

$$
\mathcal T_{\mathrm{exec}}^{\alpha,\beta}(x,c)
=
T_{obs}^{\alpha}(x)\cap C_{plan}^{\beta}(c)
$$

若交集为空，则回退到 `C_plan^\beta(c)`。

最终选择规则为：

$$
\hat i(x,c)=\arg\max_{i\in \mathcal T_{\mathrm{exec}}^{\alpha,\beta}(x,c)} p_{obs}(i\mid x)
$$

在当前实现中，我们采用：

- `obs_alpha = 0.90`
- `plan_core_alpha = 0.50`

它的含义很明确：

- 查询条件决定“允许在哪个局部核心里选”；
- 观测后验只负责在该核心内部做可行性重排；
- 这样就避免了简单联合规则在全局范围内被观测端主导。

在 `support_refined` 上，`true query` 的正式执行结果为：

- `direct_query_mse = 0.046843`
- `support_top1_mse = 0.048260`
- `query_responsive_mse = 0.047052`
- `set_best_mse = 0.044006`
- `full_oracle_mse = 0.043778`

这说明：

1. 正式查询响应规则明显优于 `support_top1`；
2. 它比 `joint_union` 稍保守，MSE 上不再追求最小；
3. 但它换来了更清楚的查询响应结构。

这一点在匹配率上更加明显。对 `support_refined` 而言：

- `true query` 下，`query_responsive_top1` 对查询条件的匹配率为 `27.5%`
- `near query` 下为 `26.25%`
- `far query` 下仍为 `20.0%`

同时，它对 `obs_top1` 的依赖显著下降：

- `true query` 下与 `obs_top1` 相同的比例为 `61.25%`
- `far query` 下仅为 `32.5%`

而此前 `joint_union_top1` 在 `far query` 下与 `obs_top1` 相同的比例仍高达 `86.25%`，且对查询条件的匹配率为 `0.0%`。

因此，新的正式读取规则并不意味着“在 MSE 上已经最优”，但它首次满足了一个更关键的要求：

> 最终执行规则开始真正受到查询条件支配，而不再几乎退化成观测端的变体。

从执行规模上看，在 `support_refined` 上：

- `true query` 的 `avg_exec_set_size = 3.575`
- `near query` 的 `avg_exec_set_size = 3.650`
- `far query` 的 `avg_exec_set_size = 3.425`

回退率分别为：

- `true = 11.25%`
- `near = 11.25%`
- `far = 25.0%`

这说明统一执行集合已经比原始规划候选集更小，且绝大多数情况下可以直接工作。

对照 `aligned`：

- `true query_responsive_mse = 0.054925`
- `far query_responsive_mse = 0.055261`
- `avg_exec_set_size` 仍在 `4.2 - 4.7` 左右

说明：

- `aligned` 可以接入同样的执行规则；
- 但其观测后验过宽，仍然不如 `support_refined` 适合作为正式执行底座。

### 4.7 查询响应读取规则已接回训练主线

在完成上述执行规则正式化之后，我们又把 query-aware 规则接回了训练主线本身，而不再只在分析脚本里使用它。

这一轮 `query_mainline` 的关键变化不是改训练梯度，而是：

1. 在每个 epoch 记录 query-aware 执行指标；
2. 允许 `best.pt` 直接按 query-aware 规则选择；
3. 从而把“支持收缩”和“查询响应执行”放到同一条训练轨迹中观察。

这一轮最重要的结果不是单一最优 checkpoint，而是暴露出了三类明确的工作点：

1. `epoch=7`，即 `best.pt` / `best_query.pt`
   - `val_cond_support_ratio = 0.9397`
   - `val_query_exec_mse = 0.054651`
   - `val_query_match_true = 0.2500`
   - `val_query_fallback_rate = 0.0000`
2. `epoch=8`，即 `best_recon.pt`
   - `val_cond_support_ratio = 0.7676`
   - `val_query_exec_mse = 0.046689`
   - `val_query_match_true = 0.2250`
   - `val_query_fallback_rate = 0.0000`
3. `epoch=9`，即 `best_support.pt`
   - `val_cond_support_ratio = 0.6218`
   - `val_query_exec_mse = 0.047052`
   - `val_query_match_true = 0.2750`
   - `val_query_fallback_rate = 0.1125`

这三点的意义很清楚：

1. `epoch=7` 是“宽集合、零回退”的保守点；
2. `epoch=8` 是当前保存 checkpoint 中执行 MSE 最好的折中点；
3. `epoch=9` 是“更尖、更查询驱动”的点，但开始出现明显回退。

因此，query-aware 规则接回训练主线之后，当前暴露出的新问题不再是“能不能接回去”，而是：

> 当前的 query-aware checkpoint 目标把“零回退”看得太重，以至于 `best.pt` 会被推向更宽、更保守的工作点。

这一步非常关键，因为它说明：

1. query-aware 规则已经成为正式训练观测量；
2. 但 query-aware 的 checkpoint / 训练目标还没有达到最合理的平衡；
3. 下一阶段真正需要重写的，是这套平衡准则，而不是回到无目的试验。

### 4.8 `query_balanced_mainline` 已修复默认 checkpoint 错位

在上一轮 `query_mainline` 暴露出问题之后，我们进一步把 query-aware checkpoint 准则重写为“回退预算 + 复合目标”的形式：

1. 先要求 fallback 不超过预算；
2. 再在预算内综合考虑执行 MSE、查询匹配和支持质量；
3. 仅将 direct-query 的额外劣化作为轻度惩罚，而不再让“零回退”一票否决。

在当前实现中，我们采用：

- `fallback_budget = 0.05`
- `exec_weight = 1.0`
- `match_weight = 0.20`
- `support_weight = 0.05`
- `gap_weight = 0.25`

其结果很清楚：

- `best.pt` 从原先 `query_mainline` 的 `epoch=7` 移到了 `epoch=8`
- `best_query.pt` 也稳定落在 `epoch=8`
- `best_support.pt` 仍为 `epoch=9`

新的默认 checkpoint，即 `epoch=8`，在 `true query` 下的正式执行结果为：

- `direct_query_mse = 0.046434`
- `support_top1_mse = 0.046993`
- `query_responsive_mse = 0.046689`
- `set_best_mse = 0.044174`
- `full_oracle_mse = 0.044130`

同时：

- `true avg_exec_set_size = 4.0375`
- `true fallback_rate = 0.0`
- `near fallback_rate = 1.25%`
- `far fallback_rate = 5.0%`

这一步的含义非常明确：

1. 旧的 query-aware checkpoint 偏差已经被正式修正；
2. 当前默认 `best.pt` 终于不再是“宽而稳”的保守点；
3. 当前系统已经具备了一个更合理的“执行优先默认点”和一个更尖的“查询响应备用点”。

但也要同时看到：

- `epoch=8` 仍不是最查询驱动的点；
- `epoch=9` 在查询匹配上更强；
- 这说明下一步如果还要继续推进，就应当把训练期 query-aware 观测从单一 `true query` 扩展到更丰富的查询扰动，而不是再回到单纯修 checkpoint 的阶段。

## 5. `alpha`、读取结果大小与执行质量的关系

为了避免再次回到无目的调参，这里只做一个最小的关系验证：`support_refined` 在 `alpha=0.80/0.90/0.95` 下的读取约束生成对照。

### 5.1 观测

对 `true query` 而言：

- `alpha=0.80`
  - `avg_set_size = 8.0`
  - `set_best_mse = 0.044099`
  - `gap_to_oracle = 0.000322`
- `alpha=0.90`
  - `avg_set_size = 9.0`
  - `set_best_mse = 0.044006`
  - `gap_to_oracle = 0.000229`
- `alpha=0.95`
  - `avg_set_size = 10.0`
  - `set_best_mse = 0.043778`
  - `gap_to_oracle = 0.000000`

### 5.2 解释

- `alpha` 增大，读取结果变大；
- 读取结果越大，越容易完全覆盖 oracle；
- 但一旦 `alpha=0.95`，候选集已经回到全空间，几乎失去筛选意义。

因此，当前最合理的工作性结论不是“越大越好”，而是：

1. `0.95` 太保守，几乎等于不筛选；
2. `0.80` 更紧，但会带来更大 oracle 差距；
3. `0.90` 是当前最自然的中间工作点。

需要额外强调的是：

- 这里的读取结果大小，指的是“生成规划集合”上的离散读取大小；
- 它和观测后验上的读取大小并不完全相同；
- 这恰好说明我们仍然存在一个未解决问题：
  - 观测侧离散代理
  - 规划侧离散代理
  二者目前还没有被统一成一个显式的 `\mu_c`。

## 6. 当前最稳的结论

截至目前，最稳的结论可以归纳为以下几条。

### 6.1 已经成立的结论

1. `top-1` 不是当前理论主线的核心指标。
2. 训练/评测对象对齐是必要前提；不对齐时，锐化现象可能是伪象。
3. `support_refined` 的确把系统推进到了“更强收缩、更高真实质量”的新工作点。
4. 当前被正式对象化的是离散读取结果，而不是理论本体中的 `\mathfrak T_c`。
5. 读取约束执行链在最小意义上已经成立。
6. 条件扰动会诱导出有结构的读取结果迁移。
7. 读取结果的主要价值在当前阶段表现为“受限搜索空间”，而不是“单点决策器”。
8. 第一轮读取规则研究已经完成，并明确排除了“直接使用 `plan_top1`”这一方案。
9. 简单联合规则虽有帮助，但目前仍明显偏向观测端。
10. 正式的查询响应读取规则已经实现，并将执行集合统一为“观测读取集合 ∩ 规划核心”的形式。
11. 查询响应读取规则已经接回训练主线，开始成为 checkpoint 选择时的正式对象。
12. 当前训练轨迹已经清楚暴露出三类不同工作点：宽而稳、执行优先、查询响应优先。
13. `query_balanced_mainline` 已经把默认 `best.pt` 从“宽而稳”正式修正为“执行优先”的平衡点。

### 6.2 尚未成立的结论

1. 还不能说 `support_top1` 就是最终生成规则。
2. 还不能说离散读取内部的最终选择规则已经解决。
3. 还不能说当前轻量底座已经充分验证了全部理论。
4. 还不能把当前结果直接外推到更大规模、更强生成器。
5. 还不能说观测侧离散代理与规划侧离散代理在理论上已经被统一成显式 `\mu_c`，只能说当前已有一个可操作的工程读取规则。
6. 还不能说当前查询响应读取规则已经在生成质量上达到最优，只能说它在结构上更合理。
7. 还不能说当前训练期 query-aware 观测已经足够完整；它目前仍主要基于 `true query`。

## 7. 当前阶段的最终发现

如果把所有实验结果压缩成一句话，那么当前最准确的表述是：

> VH-01 当前已经完成了从“局部动态一致性 + 条件打分代理学习”到“离散读取结果 + 查询响应读取规则 + query-aware 训练观测”的最小闭环验证；默认 checkpoint 的 query-aware 错位也已被修正，但理论核心对象 `\mu_0`、`\mathfrak T_c`、`\mu_c` 仍未被显式实现。

## 8. 下一步应当做什么

当前这些问题已经基本明确，因此下一步不再是“先把闭环补出来”，而是进入闭环之后的下一阶段：

1. 将训练期 query-aware 观测从单一 `true query` 扩展到更丰富的扰动查询；
2. 在新的默认平衡 checkpoint 基础上进入更大数据范围或更强生成器上的扩展验证；
3. 仅在上述问题驱动下，决定是否需要把 query-aware 目标进一步写入训练损失。
