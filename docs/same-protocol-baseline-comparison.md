# 同口径基线对比

注意：本文件只对应 [temporary-eval-standard-and-modification-plan.md](/home/void0312/AIGC/VH-01/docs/temporary-eval-standard-and-modification-plan.md#L1) 中定义的**协议 A：前缀条件视频预测**。这里的结果不能直接外推到 query/readout 主任务。

本轮对比采用同一真实数据口径：

- 数据：`UCF101 subset semantic`
- split：训练 `325`，验证 `80`
- 帧数：`8`
- 分辨率：`32x32`
- 评估脚本：[eval_standard_video_metrics.py](/home/void0312/AIGC/VH-01/scripts/eval_standard_video_metrics.py) 与 [eval_standard_video_metrics_baseline.py](/home/void0312/AIGC/VH-01/scripts/eval_standard_video_metrics_baseline.py)
- 指标：`recon_mse / recon_psnr / recon_ssim / future_mse / future_psnr / future_ssim`

对比对象：

- `mainline`
  - 配置：[realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_mainline.yaml](/home/void0312/AIGC/VH-01/configs/realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_mainline.yaml)
  - 结果：[summary.json](/home/void0312/AIGC/VH-01/runs/eval_standard_metrics_query_balanced_mainline_best_recon/metrics/summary.json)
- `mainline_semantic_catalog`
  - 配置：[realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_mainline_semantic_catalog.yaml](/home/void0312/AIGC/VH-01/configs/realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_mainline_semantic_catalog.yaml)
  - 结果：[summary.json](/home/void0312/AIGC/VH-01/runs/eval_standard_metrics_query_balanced_mainline_semantic_best_recon/metrics/summary.json)
- `continuous_measure`
  - 配置：[realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_continuous_measure_response_jet_graph_tau.yaml](/home/void0312/AIGC/VH-01/configs/realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_continuous_measure_response_jet_graph_tau.yaml)
  - 结果：[summary.json](/home/void0312/AIGC/VH-01/runs/eval_standard_metrics_continuous_measure_best_recon/metrics/summary.json)
- `continuous_measure_semantic_catalog`
  - 配置：[realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_continuous_measure_response_jet_graph_tau_semantic_catalog.yaml](/home/void0312/AIGC/VH-01/configs/realdata.ucf101_subset_semantic_identity_residual_semantic_prototype_query_balanced_continuous_measure_response_jet_graph_tau_semantic_catalog.yaml)
  - 结果：[summary.json](/home/void0312/AIGC/VH-01/runs/eval_standard_metrics_continuous_semantic_best_recon/metrics/summary.json)
- `conditional_convlstm_baseline`
  - 配置：[baseline_conditional_convlstm_real_ucf101_subset.yaml](/home/void0312/AIGC/VH-01/configs/baseline_conditional_convlstm_real_ucf101_subset.yaml)
  - 结果：[summary.json](/home/void0312/AIGC/VH-01/runs/eval_standard_metrics_baseline_real_best_recon/metrics/summary.json)

## 当前结果

| model | recon_mse | recon_psnr | recon_ssim | future_mse | future_psnr | future_ssim |
| --- | --- | --- | --- | --- | --- | --- |
| mainline | 0.043020 | 14.042479 | 0.336242 | 0.046390 | 13.768651 | 0.320183 |
| mainline_semantic_catalog | 0.063610 | 12.406304 | 0.259008 | 0.064048 | 12.374521 | 0.256633 |
| continuous_measure | 0.044588 | 13.835967 | 0.322225 | 0.048007 | 13.559096 | 0.309191 |
| continuous_measure_semantic_catalog | 0.049473 | 13.367198 | 0.296942 | 0.052089 | 13.165881 | 0.287040 |
| conditional_convlstm_baseline | 0.014905 | 18.684543 | 0.682965 | 0.026565 | 16.296296 | 0.514761 |

## 结论

- 在协议 A（纯视频指标）上，`conditional_convlstm_baseline` 仍显著领先。
- 开启 `semantic_catalog` 后，`mainline` 与 `continuous_measure` 的视频指标都下降，说明这轮修复主要改善的是条件识别能力，不是视频重建质量。
- 因此，协议 A 与协议 B 的目标在当前训练范式下仍存在张力，需要后续做执行闭环层面的联合优化。

## 复现实验

训练 baseline：

```bash
PYTHONPATH=src python scripts/train_conditional_convlstm_baseline.py \
  --config configs/baseline_conditional_convlstm_real_ucf101_subset.yaml
```

评估 baseline：

```bash
PYTHONPATH=src python scripts/eval_standard_video_metrics_baseline.py \
  --config configs/baseline_conditional_convlstm_real_ucf101_subset.yaml \
  --checkpoint runs/baseline_conditional_convlstm_real_ucf101_subset/best_recon.pt \
  --output-dir runs/eval_standard_metrics_baseline_real_best_recon \
  --split val \
  --batch-size 8
```

汇总对比：

```bash
python scripts/compare_standard_video_metrics.py \
  --entry mainline=runs/eval_standard_metrics_query_balanced_mainline_best_recon/metrics/summary.json \
  --entry mainline_semantic_catalog=runs/eval_standard_metrics_query_balanced_mainline_semantic_best_recon/metrics/summary.json \
  --entry continuous_measure=runs/eval_standard_metrics_continuous_measure_best_recon/metrics/summary.json \
  --entry continuous_measure_semantic_catalog=runs/eval_standard_metrics_continuous_semantic_best_recon/metrics/summary.json \
  --entry conditional_convlstm_baseline=runs/eval_standard_metrics_baseline_real_best_recon/metrics/summary.json
```
