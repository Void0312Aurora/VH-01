from __future__ import annotations

import torch

from vh_mvp.data import SyntheticVideoDataset
from vh_mvp.losses import (
    dynamics_loss,
    local_measure_loss,
    response_signature,
    response_signature_dim,
    response_triangle_bundle,
)
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
    seed: int = 7,
    synthetic_mode: str = "coupled",
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = SyntheticVideoDataset(
        size=batch_size,
        seq_len=seq_len,
        image_size=32,
        seed=seed,
        synthetic_mode=synthetic_mode,
    )
    video = torch.stack([dataset[idx]["video"] for idx in range(batch_size)], dim=0)
    condition = torch.stack([dataset[idx]["condition"] for idx in range(batch_size)], dim=0)
    return video, condition


def _build_model(
    *,
    seq_len: int,
    signature_mode: str,
    response_context_dim: int = 12,
    tangent_dim: int = 4,
    measure_density_mode: str = "tilted",
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
        measure_density_mode=measure_density_mode,
    )


def test_response_triangle_bundle_shapes_mask_and_energy() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len, signature_mode="span_stats")
    model.eval()
    video, condition = _build_batch(batch_size=2, seq_len=seq_len)

    with torch.no_grad():
        latents = model.encode_video(video)
        cond_embed = model.condition_encoder(condition)
        bundle = response_triangle_bundle(model, latents, video, cond_embed)

    assert bundle.residual_triangle.shape == (2, 4, 4, 3, 32, 32)
    assert bundle.energy_triangle.shape == (2, 4, 4)
    assert bundle.mask.shape == (4, 4)
    assert bundle.mask.sum(dim=1).tolist() == [4, 3, 2, 1]

    residual_energy = bundle.residual_triangle.square().flatten(3).mean(dim=3)
    mask_f = bundle.mask.unsqueeze(0).expand_as(bundle.energy_triangle)
    assert torch.allclose(bundle.energy_triangle[mask_f], residual_energy[mask_f], atol=1e-6, rtol=1e-5)


def test_dynamics_loss_is_weighted_bundle_energy() -> None:
    seq_len = 5
    short_span_bias = 0.5
    model = _build_model(seq_len=seq_len, signature_mode="span_stats")
    model.eval()
    video, condition = _build_batch(batch_size=3, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        total, per_sample, _ = dynamics_loss(
            model=model,
            latents=out.latents,
            video=video,
            cond_embed=out.cond_embed,
            short_span_bias=short_span_bias,
        )
        bundle = response_triangle_bundle(
            model=model,
            latents=out.latents,
            video=video,
            cond_embed=out.cond_embed,
            decoded=out.recon,
        )

    span_weights = video.new_tensor([1.0 / ((span_idx + 1) ** short_span_bias) for span_idx in range(seq_len - 1)])
    weight_triangle = bundle.mask.to(dtype=video.dtype) * span_weights.view(-1, 1)
    weight_sum = weight_triangle.sum().clamp_min(1e-6)
    expected_per_sample = (bundle.energy_triangle * weight_triangle.unsqueeze(0)).sum(dim=(1, 2)) / weight_sum
    expected_total = expected_per_sample.mean()

    assert torch.allclose(per_sample, expected_per_sample, atol=1e-6, rtol=1e-5)
    assert torch.allclose(total, expected_total, atol=1e-6, rtol=1e-5)


def test_response_signature_dim_matches_legacy_and_descriptor_modes() -> None:
    seq_len = 5
    video, condition = _build_batch(batch_size=2, seq_len=seq_len)

    for mode in ("span_stats", "descriptor_span_stats"):
        model = _build_model(seq_len=seq_len, signature_mode=mode)
        model.eval()
        with torch.no_grad():
            out = model(video, condition)
            signature = response_signature(
                model,
                out.latents,
                video,
                out.cond_embed,
                decoded=out.recon,
                mode=mode,
            )
        expected_dim = response_signature_dim(seq_len, mode, channels=video.size(2))
        assert signature.shape == (video.size(0), expected_dim)
        assert torch.isfinite(signature).all()


def test_local_measure_loss_descriptor_span_stats_backward() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len, signature_mode="descriptor_span_stats")
    model.train()
    video, condition = _build_batch(batch_size=4, seq_len=seq_len)
    out = model(video, condition)
    model.zero_grad(set_to_none=True)

    terms = local_measure_loss(
        model=model,
        latents=out.latents,
        video=video,
        cond_embed=out.cond_embed,
        signature_knn=1,
        signature_temperature=0.5,
        geometry_knn=2,
        geometry_temperature=0.5,
        jet_ridge=1e-3,
        jet_center_weight=1.0,
        tau_ridge=1e-3,
        tau_mean_penalty=1.0,
        tau_drift_scale=0.0,
        density_temperature=1.0,
        test_num_directions=4,
        trig_scale=1.0,
        diffusion_target_mode="full",
        measure_target_mode="response_jet",
        measure_target_blend=1.0,
        drift_target_mode="response_jet",
        drift_target_blend=1.0,
        tilt_target_mode="response_support",
        tilt_target_blend=1.0,
        signature_mode="descriptor_span_stats",
        decoded=out.recon,
    )
    loss = (
        terms["local_drift"]
        + terms["local_diffusion"]
        + terms["measure_stationarity"]
        + terms["response_smoothness"]
        + terms["measure_tilt_alignment"]
    )
    loss.backward()

    assert model.response_context_head is not None
    response_grad = model.response_context_head[0].weight.grad
    diffusion_grad = model.local_diffusion_factor_head[0].weight.grad

    assert response_grad is not None
    assert diffusion_grad is not None
    assert torch.isfinite(response_grad).all()
    assert torch.isfinite(diffusion_grad).all()


def test_generation_smoke_remains_finite() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len, signature_mode="descriptor_span_stats")
    model.eval()
    dataset = SyntheticVideoDataset(
        size=12,
        seq_len=seq_len,
        image_size=32,
        seed=17,
        synthetic_mode="coupled",
    )
    sample = dataset[0]
    video = sample["video"].unsqueeze(0)
    true_condition = sample["condition"].unsqueeze(0)

    candidate_list = [true_condition.squeeze(0)]
    for idx in range(1, len(dataset)):
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
        target_future = video[:, 1:].expand(candidate_conditions.size(0), -1, -1, -1, -1)
        future_mse = ((rollout_video - target_future) ** 2).mean(dim=(1, 2, 3, 4))
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
    assert torch.isfinite(rollout_video).all()
    assert torch.isfinite(future_mse).all()
