from __future__ import annotations

import torch

from vh_mvp.data import ConditionCatalog, SyntheticVideoDataset, condition_tuple_from_tensor
from vh_mvp.losses import response_signature_dim
from vh_mvp.models import VideoDynamicsMVP
from vh_mvp.support import build_condition_inference_posterior, query_measure_execution
from vh_mvp.train.trainer import evaluate_query_responsive_execution


def _build_model(
    *,
    seq_len: int,
    signature_mode: str = "descriptor_span_stats",
    response_context_dim: int = 12,
    tangent_dim: int = 4,
) -> VideoDynamicsMVP:
    return VideoDynamicsMVP(
        channels=3,
        base_channels=8,
        latent_dim=16,
        cond_dim=16,
        hidden_dim=32,
        response_signature_dim=response_signature_dim(seq_len, signature_mode, channels=3),
        response_context_dim=response_context_dim,
        tangent_dim=tangent_dim,
        local_measure_hidden_dim=32,
        local_measure_rank=4,
        local_measure_eps=1e-4,
        local_diffusion_mode="legacy",
        local_diffusion_geometry_mode="tangent",
        local_diffusion_condition_mode="joint",
        measure_density_mode="joint",
        encoder_condition_mode="residual_temporal",
        encoder_condition_hidden_dim=32,
        encoder_condition_scale=0.1,
    )


def _build_catalog(dataset: SyntheticVideoDataset, limit: int = 6) -> ConditionCatalog:
    conditions: list[torch.Tensor] = []
    for idx in range(len(dataset)):
        condition = dataset[idx]["condition"]
        if not any(torch.equal(condition, existing) for existing in conditions):
            conditions.append(condition)
        if len(conditions) >= limit:
            break
    keys = [condition_tuple_from_tensor(condition) for condition in conditions]
    index_by_key = {key: idx for idx, key in enumerate(keys)}
    neighbors: list[list[tuple[int, int]]] = []
    for src_idx, src_key in enumerate(keys):
        entries: list[tuple[int, int]] = []
        for dst_idx, dst_key in enumerate(keys):
            if src_idx == dst_idx:
                continue
            distance = sum(int(a != b) for a, b in zip(src_key, dst_key))
            entries.append((dst_idx, distance))
        entries.sort(key=lambda item: (item[1], item[0]))
        neighbors.append(entries)
    return ConditionCatalog(
        tensor=torch.stack(conditions, dim=0),
        keys=keys,
        texts=[",".join(str(v) for v in key) for key in keys],
        index_by_key=index_by_key,
        neighbors=neighbors,
        label_indices=[-1 for _ in keys],
    )


def test_query_measure_execution_uses_condition_set_to_mix_measures() -> None:
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float32))
    obs_posterior = build_condition_inference_posterior(logits, temperature=1.0)
    rollout_log_weights = torch.log(
        torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.2, 0.3, 0.5],
            ],
            dtype=torch.float32,
        )
    )

    selection = query_measure_execution(
        obs_posterior,
        rollout_log_weights.unsqueeze(0),
        obs_alpha=0.85,
        readout_alpha=0.80,
        temperature=1.0,
    )

    expected_condition_weights = torch.tensor([[2.0 / 3.0, 1.0 / 3.0, 0.0]], dtype=torch.float32)
    expected_rollout_weights = expected_condition_weights[:, :1] * torch.tensor(
        [[0.8, 0.1, 0.1]],
        dtype=torch.float32,
    ) + expected_condition_weights[:, 1:2] * torch.tensor(
        [[0.1, 0.7, 0.2]],
        dtype=torch.float32,
    )

    assert torch.allclose(selection.condition_weights, expected_condition_weights, atol=1e-6, rtol=1e-5)
    assert torch.allclose(selection.rollout_readout.weights, expected_rollout_weights, atol=1e-6, rtol=1e-5)
    assert int(selection.selected_idx.item()) == 0
    assert float(selection.mass().item()) >= 0.80 - 1e-6


def test_evaluate_query_responsive_execution_runs_measure_readout_closure_on_synthetic_dataset() -> None:
    seq_len = 5
    dataset = SyntheticVideoDataset(
        size=24,
        seq_len=seq_len,
        image_size=32,
        seed=137,
        synthetic_mode="coupled",
    )
    catalog = _build_catalog(dataset)
    model = _build_model(seq_len=seq_len)
    model.eval()

    metrics = evaluate_query_responsive_execution(
        model=model,
        dataset=dataset,  # type: ignore[arg-type]
        condition_catalog=catalog,
        device=torch.device("cpu"),
        alpha=0.90,
        obs_alpha=0.90,
        plan_core_alpha=0.50,
        posterior_temperature=1.0,
        max_samples=4,
        catalog_readout_mode="model",
    )

    assert metrics["query_samples"] == 4.0
    assert metrics["query_fallback_rate"] == 0.0
    assert metrics["query_exec_set_size"] >= 1.0
    assert metrics["query_exec_mse"] >= 0.0
    assert metrics["query_support_top1_mse"] >= 0.0
    assert metrics["query_set_best_mse"] >= 0.0
