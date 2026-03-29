# VH-01 理论-工程映射审计

## 1. 审计目的

本文只回答一个问题：

> 当前 VH-01 里，理论对象与工程对象到底是如何对应的？哪些对应是准确的，哪些只是近似或此前表述过重？

## 2. 理论链条

按当前理论，更准确的主线应当写成：

$$
\mu_0 \xrightarrow{\mathfrak T_c} \mu_c \xrightarrow{\mathrm{Read}} \Gamma
$$

其中：

- `\mu_0`：训练阶段隐含赋予全体可能轨迹的先验测度；
- `\mathfrak T_c`：条件 `c` 诱导的测度更新或尖锐化算子；
- `\mu_c`：更新后的测度；
- `Read`：对 `\mu_c` 的读取规则，例如 `top-1`、`top-k`、高支持区域读取。

这里必须分清三层：

1. 测度更新算子；
2. 更新后的测度；
3. 对更新后测度的读取结果。

此前容易混淆的地方就在于把第 3 层误写成了第 1 层。

## 3. 当前工程对象

当前代码里真正明确存在的对象主要是以下几类。

### 3.1 潜动态结构

- `Z_{1:T}` 的潜表示；
- 多起点、多跨度 rollout；
- `R_{i,j,l}` 风格的残差结构；
- `E_dyn` 风格的动态一致性损失。

这一部分对应于 [mvp.py](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 的潜动态链路与 [objectives.py](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) 的训练目标。

### 3.2 条件打分代理

- [mvp.py](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L227) 的 `condition_candidate_logits`
- [mvp.py](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L217) 的 `condition_alignment_energy`

它们提供的是有限目录或 rollout 族上的离散打分，不是显式的连续测度对象。

### 3.3 读取层对象

- [support.py](/home/void0312/AIGC/VH-01/src/vh_mvp/support.py#L19) 的 `CandidatePosterior`
- [support.py](/home/void0312/AIGC/VH-01/src/vh_mvp/support.py#L28) 的 `CandidateSet`
- [support.py](/home/void0312/AIGC/VH-01/src/vh_mvp/support.py#L123) 的 `query_responsive_selection`

这些对象负责：

- 后验排序；
- `top-k` / `alpha`-质量区域截取；
- 在读取结果上执行最终选择。

## 4. 当前准确的映射

当前可以相对稳妥地成立的映射是：

1. `Z_{1:T}` ↔ 编码器得到的潜序列  
2. `R_{i,j,l}` ↔ rollout 与真实轨迹之间的多尺度残差  
3. `E_dyn` ↔ 动态一致性与预测误差项  
4. 条件更新后的有限离散切片 ↔ `condition_candidate_logits` / `condition_alignment_energy` 诱导出的离散打分  
5. `Read[\mu_c]` 的有限近似 ↔ `CandidatePosterior`、`CandidateSet`、`top-k` / `alpha`-质量读取与查询响应读取规则  

更简短地说：

- 潜动态层：已有较直接实现；
- 条件更新层：只有离散代理；
- 读取层：已有明确实现。

## 5. 当前不准确或不能过度声称的映射

以下说法当前都不能算严格准确：

1. “理论中的 `\mathfrak T_c` 已被完整实现。”  
   不准确。当前只有离散打分代理，没有显式的测度更新算子对象。

2. “理论中的 `\mu_c` 已被完整实现。”  
   不准确。当前只有若干离散后验切片与工程拼接规则。

3. “候选集 `T` 就是理论核心对象。”  
   不准确。按当前理论，`T_c` 更应理解为尖锐化或更新操作；固定集合、`top-k`、`alpha`-质量区域都只是读取结果。

4. “查询响应规则就是理论主线的一部分。”  
   不准确。它是当前缺乏统一显式 `\mu_c` 时的工程读取补丁。

5. “当前已经完整实现理论闭环。”  
   不准确。当前更准确的说法是：读取层闭环已成立，但理论本体对象尚未完全显式化。

## 6. 当前最合理的重述

因此，当前 VH-01 的状态更准确地应当表述为：

> 我们已经在潜动态层实现了最小底座，在条件层实现了离散打分代理，在读取层实现了 `top-k` / `alpha`-质量读取与查询响应读取规则；但理论中的 `\mu_0`、`\mathfrak T_c`、`\mu_c` 仍主要停留在解释层，尚未成为明确的工程对象。

## 7. 接下来最该做的事

如果按照这次审计后的逻辑继续推进，那么优先级应当是：

1. 先明确训练链路里哪些部分能被解释为 `\mu_0` 的代理；
2. 再明确条件 `c` 通过什么结构近似 `\mathfrak T_c`；
3. 最后再判断当前离散后验中哪些部分可以被解释为 `\mu_c` 的有限读取。

在这一步完成之前，读取层实验仍然有工程价值，但不应再被表述为理论本体的直接实现。
