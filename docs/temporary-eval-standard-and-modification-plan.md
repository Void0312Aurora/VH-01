# 临时评估标准与修改计划

## 1. 目的

当前项目同时包含两类不同任务：

1. 给定观测前缀与辅助条件，预测后续视频；
2. 给定观测与条件目录，完成条件读取、执行与选择。

这两类任务不能再混用同一组 baseline 和同一组指标。  
本文件给出一个**临时标准**，用于后续实验统一口径，并给出对应修改计划。

## 2. 临时标准

### 2.1 协议 A：前缀条件视频预测

定义：

1. 总条件写作 `c_total = (X_{1:k}^{obs}, c_aux)`；
2. 模型目标是预测 `X_{k+1:T}`；
3. 当前仓库真实数据默认取 `k = 1`，也就是首帧前缀条件预测。

允许比较的模型：

1. 当前 `mainline`
2. 当前 `continuous_measure`
3. 专门的视频预测 baseline，例如当前已实现的 `Conditional ConvLSTM`

只允许使用的指标：

1. `future_mse`
2. `future_psnr`
3. `future_ssim`
4. `recon_mse`
5. `recon_psnr`
6. `recon_ssim`
7. 可选：`LPIPS`
8. 更后面才考虑：`FVD`

禁止得出的结论：

1. 不能用协议 A 的结果直接说明 query/readout 主任务谁更强；
2. 不能用协议 A 的 baseline 代替 query/execution baseline；
3. 不能把协议 A 的失败直接解释成理论主线失败。

### 2.2 协议 B：条件查询/执行主任务

定义：

1. 输入为观测视频与条件目录；
2. 模型目标是进行条件 posterior/readout/execution；
3. 输出包括条件选择、执行集合与执行 rollout。

允许比较的模型：

1. 当前 `mainline`
2. 当前 `continuous_measure`
3. 后续要补的 query/readout baseline

只允许使用的指标：

1. `val_cond_acc`
2. `val_cond_true_prob`
3. `val_cond_true_in90`
4. `val_query_exec_mse`
5. `val_query_direct_mse`
6. `val_query_support_top1_mse`
7. `val_query_oracle_mse`
8. `val_query_exec_set_size`
9. `val_query_match_true`
10. `val_query_fallback_rate`

禁止得出的结论：

1. 不能拿协议 B 的结果评价纯视频质量；
2. 不能用协议 B 的指标倒推 `PSNR/SSIM`；
3. 不能用协议 B 的 baseline 代替视频预测 baseline。

## 3. 当前已有结果如何归类

### 3.1 已有 `ConvLSTM` baseline 结果

已有文档 [same-protocol-baseline-comparison.md](/home/void0312/AIGC/VH-01/docs/same-protocol-baseline-comparison.md#L1) 的结果应当被重新解释为：

1. 这是**协议 A**的结果；
2. 它只说明当前主模型在“首帧前缀条件视频预测”口径下弱于轻量 `ConvLSTM` baseline；
3. 它**不**说明当前主模型在 query/readout 主任务上也弱于该 baseline。

### 3.2 当前项目主线结果

1. `mainline` 与 `continuous_measure` 的 `query_* / cond_* / support_*` 指标属于**协议 B**；
2. 当前 `PSNR/SSIM` 结果属于**协议 A**；
3. 以后所有结果汇报必须明确标注属于哪一类协议。

## 4. 修改计划

### 4.1 第一阶段：先把协议边界固定住

本阶段不改模型，只改评估口径和实验组织。

要做的事：

1. 在现有视频质量对比文档中明确标注“协议 A”；
2. 后续所有实验记录都要在标题或摘要里标明 `协议 A` 或 `协议 B`；
3. 新增一个 query/readout baseline 之前，不再拿 `ConvLSTM` 去评价主任务表现。

阶段关闭标准：

1. 以后不会再出现“用视频预测 baseline 评价查询主任务”的比较；
2. 每份对比结果都能一眼看出属于哪类任务。

### 4.2 第二阶段：补协议 B 的同口径 baseline

本阶段的目标是补一个最小但有效的查询执行 baseline。

建议 baseline：

1. `encoder + condition classifier`：直接做条件 posterior；
2. `posterior -> top-k/readout`：给出执行集合；
3. 可选地配一个简单 rollout 模块，仅用于 execution MSE 比较。

最低要求：

1. 能输出 `cond_acc / cond_true_prob / cond_true_in90`；
2. 能输出 `query_exec_mse / support_top1_mse / exec_set_size`；
3. 评估脚本和当前主线用同一 catalog、同一 split、同一观测协议。

阶段关闭标准：

1. `mainline`、`continuous_measure`、query baseline 三者能在协议 B 下直接对照；
2. 不再需要拿视频预测 baseline 替代 query baseline。

### 4.3 第三阶段：再决定是否修改主模型训练

只有在协议边界和 baseline 都补齐之后，才进入训练侧修改。

决策规则：

1. 如果协议 A 明显弱、协议 B 仍强：
   说明问题主要在视频预测路径；
   这时再考虑加入 `future rollout pixel loss` 或空间 latent dynamics。
2. 如果协议 B 也弱：
   说明主线 query/readout 本身需要修；
   这时优先改 query/posterior/readout，而不是先改视频生成 loss。

## 5. 当前默认行动

下一步默认只做一件事：

1. 按协议 B 补一个最小 query/readout baseline。

在这之前：

1. 不再继续拿 `ConvLSTM` 结果外推到总任务；
2. 不直接因为协议 A 的落后就重写主模型训练目标。
