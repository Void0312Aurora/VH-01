from __future__ import annotations

import torch

from vh_mvp.baselines.query_protocol import protocol_b_selection_metrics, summarize_encoded_video


def test_summarize_encoded_video_returns_expected_feature_dim() -> None:
    encoded = torch.arange(2 * 3 * 4 * 2 * 2, dtype=torch.float32).view(2, 3, 4, 2, 2)
    summary = summarize_encoded_video(encoded)
    assert summary.shape == (2, 20)
    pooled = encoded.mean(dim=(-1, -2))
    expected_first = pooled[:, 0]
    expected_last = pooled[:, -1]
    expected_delta = expected_last - expected_first
    assert torch.allclose(summary[:, :4], expected_first)
    assert torch.allclose(summary[:, 8:12], expected_last)
    assert torch.allclose(summary[:, 12:16], expected_delta)


def test_protocol_b_selection_metrics_tracks_top1_and_set_best() -> None:
    future_mse = torch.tensor([0.40, 0.10, 0.20, 0.05], dtype=torch.float32)
    posterior_logits = torch.tensor([0.1, 2.5, 1.8, -0.3], dtype=torch.float32)
    metrics = protocol_b_selection_metrics(
        future_mse=future_mse,
        true_idx=2,
        posterior_logits=posterior_logits,
        alpha=0.90,
        temperature=1.0,
    )
    assert metrics["query_support_top1_mse"] == metrics["query_exec_mse"]
    assert metrics["query_support_top1_mse"] == float(future_mse[1].item())
    assert metrics["query_direct_mse"] == float(future_mse[2].item())
    assert metrics["query_oracle_mse"] == float(future_mse[3].item())
    assert metrics["query_exec_gap_to_oracle"] >= 0.0
    assert metrics["query_exec_set_size"] >= 1.0
