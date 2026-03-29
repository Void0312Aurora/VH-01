from __future__ import annotations

import torch

from vh_mvp.data import SyntheticVideoDataset
from vh_mvp.losses import response_signature_dim
from vh_mvp.models import BaseMeasure, ConditionalMeasure, ConditionalTilt, VideoDynamicsMVP
from vh_mvp.support import (
    ConditionInferencePosterior,
    MeasureReadout,
    build_candidate_posterior,
    build_condition_inference_posterior,
    candidate_sets_from_posterior,
    measure_readout_from_log_weights,
    query_responsive_selection,
)


def _build_batch(
    *,
    batch_size: int,
    seq_len: int,
    seed: int = 31,
    synthetic_mode: str = "coupled",
) -> tuple[torch.Tensor, torch.Tensor, SyntheticVideoDataset]:
    dataset = SyntheticVideoDataset(
        size=max(batch_size, 16),
        seq_len=seq_len,
        image_size=32,
        seed=seed,
        synthetic_mode=synthetic_mode,
    )
    video = torch.stack([dataset[idx]["video"] for idx in range(batch_size)], dim=0)
    condition = torch.stack([dataset[idx]["condition"] for idx in range(batch_size)], dim=0)
    return video, condition, dataset


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
        measure_density_mode="tilted",
        encoder_condition_mode="residual_temporal",
        encoder_condition_hidden_dim=32,
        encoder_condition_scale=0.1,
    )


def _candidate_conditions(dataset: SyntheticVideoDataset, limit: int = 6) -> torch.Tensor:
    candidates: list[torch.Tensor] = []
    for idx in range(len(dataset)):
        condition = dataset[idx]["condition"]
        if not any(torch.equal(condition, existing) for existing in candidates):
            candidates.append(condition)
        if len(candidates) >= limit:
            break
    return torch.stack(candidates, dim=0)


def test_conditional_measure_matches_legacy_density_components() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition, _ = _build_batch(batch_size=4, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        measure = model.conditional_measure(out.latents, out.cond_embed)
        base, tilt, total = model.measure_log_density_components(out.latents, out.cond_embed)

    assert isinstance(measure, ConditionalMeasure)
    assert isinstance(measure.base_measure, BaseMeasure)
    assert isinstance(measure.conditional_tilt, ConditionalTilt)
    assert torch.allclose(measure.log_base_density, base, atol=1e-6, rtol=1e-5)
    assert torch.allclose(measure.log_tilt, tilt, atol=1e-6, rtol=1e-5)
    assert torch.allclose(measure.log_total_density, total, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        measure.log_total_density,
        measure.log_base_density + measure.log_tilt,
        atol=1e-6,
        rtol=1e-5,
    )
    weights = measure.normalized_weights(temperature=1.0)
    assert weights.shape == (video.size(0),)
    assert torch.isfinite(weights).all()
    assert abs(float(weights.sum().item()) - 1.0) < 1e-6


def test_measure_readout_from_conditional_measure_has_valid_alpha_mass() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition, _ = _build_batch(batch_size=5, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        measure = model.conditional_measure(out.latents, out.cond_embed)
        readout = measure_readout_from_log_weights(measure.log_total_density.squeeze(-1), alpha=0.80, temperature=1.0)

    assert isinstance(readout, MeasureReadout)
    expected_top1 = measure.log_total_density.squeeze(-1).argmax().unsqueeze(0)
    assert torch.equal(readout.top1_idx, expected_top1)
    assert float(readout.mass().item()) >= 0.80 - 1e-6
    scores = torch.arange(readout.weights.size(1), dtype=readout.weights.dtype).unsqueeze(0)
    restricted_idx = readout.restricted_argmax(scores)
    assert bool(readout.mask[0, int(restricted_idx.item())].item())


def test_condition_inference_posterior_is_separate_and_query_compatible() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition, dataset = _build_batch(batch_size=1, seq_len=seq_len)
    candidate_conditions = _candidate_conditions(dataset)

    with torch.no_grad():
        latents = model.encode_video(video)
        obs_logits = model.condition_candidate_logits(latents, candidate_conditions)
        obs_posterior = build_condition_inference_posterior(obs_logits, temperature=1.0)
        legacy_obs_posterior = build_candidate_posterior(obs_logits, temperature=1.0)
        obs_set = candidate_sets_from_posterior(obs_posterior, [0.90])[0.90]

        cond_embed_all = model.condition_encoder(candidate_conditions)
        rollout_latents, _ = model.rollout_from(
            latents[:, 0].expand(candidate_conditions.size(0), -1),
            cond_embed_all,
            steps=seq_len - 1,
        )
        query_embed = cond_embed_all[:1].expand(candidate_conditions.size(0), -1)
        plan_logits = -model.condition_alignment_energy(rollout_latents, query_embed).unsqueeze(0)
        plan_posterior = build_candidate_posterior(plan_logits, temperature=1.0)
        selection = query_responsive_selection(
            obs_posterior=obs_posterior,
            plan_posterior=plan_posterior,
            obs_alpha=0.90,
            plan_core_alpha=0.50,
        )

    assert isinstance(obs_posterior, ConditionInferencePosterior)
    assert torch.allclose(obs_posterior.probs, legacy_obs_posterior.probs, atol=1e-6, rtol=1e-5)
    assert float(obs_set.mass().item()) >= 0.90 - 1e-6
    assert len(selection.member_indices()[0]) >= 1
