# 协议 B 基线对比

本文件对应 [temporary-eval-standard-and-modification-plan.md](/home/void0312/AIGC/VH-01/docs/temporary-eval-standard-and-modification-plan.md#L1) 中定义的**协议 B：条件查询/执行主任务**。

## 评估对象

- `mainline`
  - 模型结果：[summary.json](/home/void0312/AIGC/VH-01/runs/eval_query_protocol_mainline_best/metrics/summary.json)
- `mainline_semantic_catalog`
  - 模型结果：[summary.json](/home/void0312/AIGC/VH-01/runs/eval_query_protocol_mainline_semantic_best/metrics/summary.json)
- `continuous_measure`
  - 模型结果：[summary.json](/home/void0312/AIGC/VH-01/runs/eval_query_protocol_continuous_best/metrics/summary.json)
- `continuous_measure_semantic_catalog`
  - 模型结果：[summary.json](/home/void0312/AIGC/VH-01/runs/eval_query_protocol_continuous_semantic_best/metrics/summary.json)
- `conditional_convlstm_baseline`
  - 模型结果：[summary.json](/home/void0312/AIGC/VH-01/runs/eval_query_protocol_baseline_best/metrics/summary.json)
  - 说明：该 baseline 先用冻结的 `Conditional ConvLSTM` 编码特征训练一个最小 condition probe，再用 posterior top-1 做 execution 选择。

## 当前结果

| model | cond_acc | cond_true_prob | cond_true_in90 | cond_support_ratio | query_exec_mse | query_oracle_mse | query_match_true | query_exec_set_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mainline | 0.175000 | 0.158515 | 0.975000 | 0.767700 | 0.047960 | 0.044130 | 0.100000 | 9.000000 |
| mainline_semantic_catalog | 1.000000 | 0.918593 | 1.000000 | 0.157764 | 0.073746 | 0.057643 | 0.112500 | 9.000000 |
| continuous_measure | 0.100000 | 0.100845 | 0.900000 | 0.998395 | 0.050732 | 0.046925 | 0.087500 | 9.000000 |
| continuous_measure_semantic_catalog | 0.525000 | 0.235899 | 0.950000 | 0.707552 | 0.058312 | 0.053770 | 0.087500 | 9.000000 |
| conditional_convlstm_baseline | 0.400000 | 0.233680 | 0.850000 | 0.597145 | 0.027057 | 0.025929 | 0.400000 | 5.675000 |

## 结论

- `semantic_catalog` 改动显著增强了 posterior 识别能力（`mainline` 的 `cond_acc` 从 `0.175` 提升到 `1.0`，`continuous_measure` 从 `0.100` 提升到 `0.525`）。
- 但 execution 没有同步改善，`query_exec_mse` 反而上升（`mainline`: `0.047960 -> 0.073746`，`continuous_measure`: `0.050732 -> 0.058312`）。
- 在当前协议 B 口径下，`conditional_convlstm_baseline` 仍是 execution 最优（`query_exec_mse=0.027057`，`query_match_true=0.400000`）。

## 当前解释

- 这轮修复证明“posterior 读出弱”确实是问题之一，但不是唯一瓶颈。
- 当前主要矛盾已经转移到“识别正确以后，如何把条件 posterior 变成更好的执行选择”，即 `mu_0 -> T_c -> mu_c -> Read` 的闭环耦合强度。

## 复现实验

运行 `mainline`：

```bash
PYTHONPATH=src python scripts/eval_query_protocol_metrics.py \
  --model-type mvp \
  --config configs/realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_mainline.yaml \
  --checkpoint runs/real_ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_mainline/best.pt \
  --output-dir runs/eval_query_protocol_mainline_best \
  --split val \
  --max-samples 80 \
  --query-alpha 0.90 \
  --posterior-temperature 1.0
```

运行 `mainline_semantic_catalog`：

```bash
PYTHONPATH=src python scripts/eval_query_protocol_metrics.py \
  --model-type mvp \
  --config configs/realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_mainline_semantic_catalog.yaml \
  --checkpoint runs/real_ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_mainline_semantic_catalog/best.pt \
  --output-dir runs/eval_query_protocol_mainline_semantic_best \
  --split val \
  --max-samples 80 \
  --query-alpha 0.90 \
  --posterior-temperature 1.0
```

运行 `continuous_measure`：

```bash
PYTHONPATH=src python scripts/eval_query_protocol_metrics.py \
  --model-type mvp \
  --config configs/realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_continuous_measure_response_jet_graph_tau.yaml \
  --checkpoint runs/real_ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_continuous_measure_response_jet_graph_tau/best.pt \
  --output-dir runs/eval_query_protocol_continuous_best \
  --split val \
  --max-samples 80 \
  --query-alpha 0.90 \
  --posterior-temperature 2.0
```

运行 `continuous_measure_semantic_catalog`：

```bash
PYTHONPATH=src python scripts/eval_query_protocol_metrics.py \
  --model-type mvp \
  --config configs/realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_continuous_measure_response_jet_graph_tau_semantic_catalog.yaml \
  --checkpoint runs/real_ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_continuous_measure_response_jet_graph_tau_semantic_catalog/best.pt \
  --output-dir runs/eval_query_protocol_continuous_semantic_best \
  --split val \
  --max-samples 80 \
  --query-alpha 0.90 \
  --posterior-temperature 2.0
```

运行 `Conditional ConvLSTM baseline`：

```bash
PYTHONPATH=src python scripts/eval_query_protocol_metrics.py \
  --model-type conditional_convlstm \
  --config configs/baseline_conditional_convlstm_real_ucf101_subset.yaml \
  --checkpoint runs/baseline_conditional_convlstm_real_ucf101_subset/best.pt \
  --output-dir runs/eval_query_protocol_baseline_best \
  --split val \
  --max-samples 80 \
  --query-alpha 0.90 \
  --posterior-temperature 1.0 \
  --probe-epochs 100 \
  --probe-batch-size 64 \
  --probe-lr 0.01 \
  --probe-weight-decay 0.0001 \
  --probe-type mlp \
  --probe-hidden-dim 128
```
