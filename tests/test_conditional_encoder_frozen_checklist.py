from __future__ import annotations

import torch

from vh_mvp.data import SyntheticVideoDataset
from vh_mvp.losses import local_measure_targets, response_signature_dim
from vh_mvp.models import VideoDynamicsMVP
from vh_mvp.support import (
    build_candidate_posterior,
    candidate_sets_from_posterior,
    query_responsive_selection,
)


def _build_batch(
    *,
    batch_size: int,
    seq_len: int,
    seed: int = 23,
    synthetic_mode: str = "coupled",
) -> tuple[torch.Tensor, torch.Tensor, SyntheticVideoDataset]:
    dataset = SyntheticVideoDataset(
        size=max(batch_size, 12),
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


def _find_different_condition(dataset: SyntheticVideoDataset, reference: torch.Tensor) -> torch.Tensor:
    for idx in range(len(dataset)):
        candidate = dataset[idx]["condition"]
        if not torch.equal(candidate, reference):
            return candidate
    raise RuntimeError("Failed to find a different condition in the synthetic dataset.")


def test_conditional_encode_video_differs_from_base_and_varies_by_condition() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition, dataset = _build_batch(batch_size=1, seq_len=seq_len)
    alternate_condition = _find_different_condition(dataset, condition[0]).unsqueeze(0)

    with torch.no_grad():
        cond_embed = model.condition_encoder(condition)
        alternate_embed = model.condition_encoder(alternate_condition)
        base_latents = model.encode_video(video)
        conditioned_latents = model.encode_video(video, cond_embed=cond_embed)
        alternate_latents = model.encode_video(video, cond_embed=alternate_embed)

    assert base_latents.shape == conditioned_latents.shape == alternate_latents.shape == (1, seq_len, model.latent_dim)
    assert torch.isfinite(conditioned_latents).all()
    assert (conditioned_latents - base_latents).abs().max().item() > 1e-6
    assert (conditioned_latents - alternate_latents).abs().max().item() > 1e-6


def test_forward_latents_match_conditioned_encode_path() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition, _ = _build_batch(batch_size=2, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        explicit_latents = model.encode_video(video, cond_embed=out.cond_embed)
        base_latents = model.encode_video(video)

    assert torch.allclose(out.latents, explicit_latents, atol=1e-6, rtol=1e-5)
    assert (out.latents - base_latents).abs().max().item() > 1e-6


def test_local_measure_targets_target_model_uses_conditioned_reencode() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    target_model = _build_model(seq_len=seq_len)
    model.eval()
    target_model.eval()
    video, condition, _ = _build_batch(batch_size=3, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        target_cond_embed = target_model.condition_encoder(condition)
        targets = local_measure_targets(
            model=model,
            latents=out.latents,
            video=video,
            cond_embed=out.cond_embed,
            diffusion_target_mode="full",
            measure_target_mode="response_jet",
            measure_target_blend=1.0,
            drift_target_mode="response_jet",
            drift_target_blend=1.0,
            tilt_target_mode="response_support",
            tilt_target_blend=1.0,
            geometry_knn=2,
            geometry_temperature=0.5,
            jet_ridge=1e-3,
            jet_center_weight=1.0,
            tau_ridge=1e-3,
            tau_mean_penalty=1.0,
            tau_drift_scale=0.0,
            signature_mode="descriptor_span_stats",
            decoded=out.recon,
            target_model=target_model,
            target_cond_embed=target_cond_embed,
        )
        expected_latents = target_model.encode_video(video, cond_embed=target_cond_embed)
        expected_point = target_model.trajectory_point(expected_latents)
        unconditioned_point = target_model.trajectory_point(target_model.encode_video(video))

    assert torch.allclose(targets["source_point"], expected_point, atol=1e-6, rtol=1e-5)
    assert (expected_point - unconditioned_point).abs().max().item() > 1e-6


def test_unconditioned_query_generation_smoke_remains_finite() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition, dataset = _build_batch(batch_size=1, seq_len=seq_len)

    candidate_list = [condition.squeeze(0)]
    for idx in range(len(dataset)):
        candidate = dataset[idx]["condition"]
        if not any(torch.equal(candidate, existing) for existing in candidate_list):
            candidate_list.append(candidate)
        if len(candidate_list) >= 6:
            break
    candidate_conditions = torch.stack(candidate_list, dim=0)

    with torch.no_grad():
        latents = model.encode_video(video)
        cond_embed_all = model.condition_encoder(candidate_conditions)
        obs_logits = model.condition_candidate_logits(latents, candidate_conditions)
        obs_posterior = build_candidate_posterior(obs_logits, temperature=1.0)
        rollout_latents, _ = model.rollout_from(
            latents[:, 0].expand(candidate_conditions.size(0), -1),
            cond_embed_all,
            steps=seq_len - 1,
        )
        rollout_video = model.decode_video(rollout_latents, cond_embed_all)
        query_embed = cond_embed_all[:1].expand(candidate_conditions.size(0), -1)
        plan_logits = -model.condition_alignment_energy(rollout_latents, query_embed).unsqueeze(0)
        plan_posterior = build_candidate_posterior(plan_logits, temperature=1.0)
        candidate_set = candidate_sets_from_posterior(plan_posterior, [0.90])[0.90]
        selection = query_responsive_selection(
            obs_posterior=obs_posterior,
            plan_posterior=plan_posterior,
            obs_alpha=0.90,
            plan_core_alpha=0.50,
        )

    assert rollout_video.shape == (candidate_conditions.size(0), seq_len - 1, 3, 32, 32)
    assert int(candidate_set.k_alpha[0].item()) >= 1
    assert len(selection.member_indices()[0]) >= 1
    assert torch.isfinite(latents).all()
    assert torch.isfinite(rollout_video).all()
