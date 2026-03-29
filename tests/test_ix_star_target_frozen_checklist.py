from __future__ import annotations

import torch

from vh_mvp.data import SyntheticVideoDataset
from vh_mvp.losses import (
    ResponseInvariantTarget,
    build_response_invariant_target,
    local_measure_loss,
    local_measure_targets,
    response_signature_dim,
    response_triangle_bundle,
)
from vh_mvp.models import VideoDynamicsMVP


def _build_batch(
    *,
    batch_size: int,
    seq_len: int,
    seed: int = 67,
    synthetic_mode: str = "coupled",
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = SyntheticVideoDataset(
        size=max(batch_size, 16),
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


def test_response_invariant_target_is_built_from_raw_response_bundle() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition = _build_batch(batch_size=4, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        point = model.trajectory_point(out.latents)
        bundle = response_triangle_bundle(
            model=model,
            latents=out.latents,
            video=video,
            cond_embed=out.cond_embed,
            decoded=out.recon,
        )
        invariant_target = build_response_invariant_target(
            point,
            bundle,
            tangent_dim=4,
            geometry_knn=2,
            geometry_temperature=0.5,
            jet_ridge=1e-3,
            jet_center_weight=1.0,
        )

    assert isinstance(invariant_target, ResponseInvariantTarget)
    assert invariant_target.descriptor_triangle.requires_grad is False
    assert invariant_target.response_channels.requires_grad is False
    assert torch.isfinite(invariant_target.eigvals).all()
    assert torch.isfinite(invariant_target.spectral_gap).all()
    assert torch.isfinite(invariant_target.scale_profile).all()
    assert torch.allclose(
        invariant_target.scale_profile.sum(dim=1),
        torch.ones_like(invariant_target.scale_profile.sum(dim=1)),
        atol=1e-5,
        rtol=1e-5,
    )
    assert invariant_target.tangent_projector is not None
    assert invariant_target.support_tilt is not None


def test_local_measure_targets_expose_invariant_target_and_bootstrap_only_sets_scale() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition = _build_batch(batch_size=4, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        targets = local_measure_targets(
            model=model,
            latents=out.latents,
            video=video,
            cond_embed=out.cond_embed,
            diffusion_target_mode="full",
            measure_target_mode="response_invariant_bootstrap",
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
        )

    assert targets.invariant_target is not None
    assert targets.target_tangent_cov is not None
    assert targets.invariant_target.identifiable_tangent_cov is not None
    target_cov = targets.target_tangent_cov
    invariant_cov = targets.invariant_target.identifiable_tangent_cov
    target_cov_norm = target_cov / target_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    invariant_cov_norm = invariant_cov / invariant_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    assert torch.allclose(target_cov_norm, invariant_cov_norm, atol=1e-5, rtol=1e-4)
    bootstrap_trace = targets.bootstrap_diffusion_target.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    target_trace = target_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    assert torch.allclose(target_trace, bootstrap_trace, atol=1e-5, rtol=1e-4)


def test_teacher_path_builds_detached_invariant_target() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    target_model = _build_model(seq_len=seq_len)
    target_model.load_state_dict(model.state_dict())
    model.eval()
    target_model.eval()
    video, condition = _build_batch(batch_size=4, seq_len=seq_len)

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

    assert targets.invariant_target is not None
    assert targets.invariant_target.response_channels.requires_grad is False
    assert torch.allclose(targets.source_point, expected_point, atol=1e-6, rtol=1e-5)


def test_local_measure_loss_invariant_first_path_backward() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.train()
    video, condition = _build_batch(batch_size=4, seq_len=seq_len)
    out = model(video, condition)
    model.zero_grad(set_to_none=True)

    terms = local_measure_loss(
        model=model,
        latents=out.latents,
        video=video,
        cond_embed=out.cond_embed,
        signature_knn=2,
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
        measure_target_mode="response_invariant_bootstrap",
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
        + 0.1 * terms["tangent_projection"]
    )
    loss.backward()

    assert all(torch.isfinite(value).all() for value in terms.values())
    assert model.measure_tilt_head is not None
    assert model.local_diffusion_factor_head is not None
    assert model.measure_tilt_head[-1].weight.grad is not None
    assert model.local_diffusion_factor_head[-1].weight.grad is not None
