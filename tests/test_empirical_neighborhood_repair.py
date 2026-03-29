from __future__ import annotations

import torch

from vh_mvp.data import SyntheticVideoDataset
from vh_mvp.losses import (
    append_geometry_neighborhood_reference,
    append_smoothness_neighborhood_reference,
    build_geometry_neighborhood_reference,
    build_smoothness_neighborhood_reference,
    local_measure_loss,
    local_measure_targets,
    response_signature_dim,
)
from vh_mvp.models import VideoDynamicsMVP


def _build_batch(
    *,
    start: int,
    batch_size: int,
    seq_len: int,
    seed: int = 53,
    synthetic_mode: str = "coupled",
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = SyntheticVideoDataset(
        size=max(start + batch_size, 16),
        seq_len=seq_len,
        image_size=32,
        seed=seed,
        synthetic_mode=synthetic_mode,
    )
    video = torch.stack([dataset[idx]["video"] for idx in range(start, start + batch_size)], dim=0)
    condition = torch.stack([dataset[idx]["condition"] for idx in range(start, start + batch_size)], dim=0)
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


def test_neighborhood_reference_append_truncates_to_fixed_budget() -> None:
    geometry_a = build_geometry_neighborhood_reference(
        model=_build_model(seq_len=5),
        latents=torch.randn(3, 5, 16),
        video=torch.randn(3, 5, 3, 32, 32),
        cond_embed=torch.randn(3, 16),
    )
    geometry_b = build_geometry_neighborhood_reference(
        model=_build_model(seq_len=5),
        latents=torch.randn(4, 5, 16),
        video=torch.randn(4, 5, 3, 32, 32),
        cond_embed=torch.randn(4, 16),
    )
    merged_geometry = append_geometry_neighborhood_reference(None, geometry_a, max_size=5)
    merged_geometry = append_geometry_neighborhood_reference(merged_geometry, geometry_b, max_size=5)
    assert merged_geometry is not None
    assert merged_geometry.size() == 5
    assert merged_geometry.tangent_frames is not None
    assert merged_geometry.tangent_frame_valid is not None
    assert merged_geometry.tangent_frames.size(0) == 5
    assert merged_geometry.tangent_frame_valid.size(0) == 5

    smooth_a = build_smoothness_neighborhood_reference(
        model=_build_model(seq_len=5),
        latents=torch.randn(3, 5, 16),
        video=torch.randn(3, 5, 3, 32, 32),
        cond_embed=torch.randn(3, 16),
        signature_mode="descriptor_span_stats",
    )
    smooth_b = build_smoothness_neighborhood_reference(
        model=_build_model(seq_len=5),
        latents=torch.randn(4, 5, 16),
        video=torch.randn(4, 5, 3, 32, 32),
        cond_embed=torch.randn(4, 16),
        signature_mode="descriptor_span_stats",
    )
    merged_smooth = append_smoothness_neighborhood_reference(None, smooth_a, max_size=5)
    merged_smooth = append_smoothness_neighborhood_reference(merged_smooth, smooth_b, max_size=5)
    assert merged_smooth is not None
    assert merged_smooth.size() == 5


def test_local_measure_targets_upgrade_batch_knn_to_empirical_pool() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video_ref, condition_ref = _build_batch(start=0, batch_size=4, seq_len=seq_len)
    video_cur, condition_cur = _build_batch(start=4, batch_size=2, seq_len=seq_len)

    with torch.no_grad():
        out_ref = model(video_ref, condition_ref)
        out_cur = model(video_cur, condition_cur)
        geometry_reference = build_geometry_neighborhood_reference(
            model=model,
            latents=out_ref.latents,
            video=video_ref,
            cond_embed=out_ref.cond_embed,
            decoded=out_ref.recon,
        )
        targets = local_measure_targets(
            model=model,
            latents=out_cur.latents,
            video=video_cur,
            cond_embed=out_cur.cond_embed,
            diffusion_target_mode="full",
            measure_target_mode="response_jet",
            measure_target_blend=1.0,
            drift_target_mode="response_jet",
            drift_target_blend=1.0,
            tilt_target_mode="response_support",
            tilt_target_blend=1.0,
            geometry_knn=3,
            geometry_temperature=0.5,
            jet_ridge=1e-3,
            jet_center_weight=1.0,
            tau_ridge=1e-3,
            tau_mean_penalty=1.0,
            tau_drift_scale=0.0,
            signature_mode="descriptor_span_stats",
            decoded=out_cur.recon,
            geometry_reference=geometry_reference,
        )

    assert float(targets.geometry_neighbor_pool_size.item()) == 6.0
    assert float(targets.geometry_reference_pool_size.item()) == 4.0
    assert float(targets.geometry_reference_neighbor_ratio.item()) > 0.0
    assert targets.target_neighbor_idx is not None
    assert targets.target_neighbor_weights is not None
    assert targets.target_transport is not None
    assert int(targets.target_neighbor_idx.max().item()) >= out_cur.latents.size(0)
    assert torch.isfinite(targets.target_transport).all()
    assert targets.target_tangent_frame is not None
    assert torch.isfinite(targets.target_tangent_frame).all()


def test_local_measure_loss_uses_empirical_pool_and_remains_finite() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.train()
    video_ref, condition_ref = _build_batch(start=0, batch_size=4, seq_len=seq_len)
    video_cur, condition_cur = _build_batch(start=4, batch_size=3, seq_len=seq_len)

    with torch.no_grad():
        out_ref = model(video_ref, condition_ref)
        geometry_reference = build_geometry_neighborhood_reference(
            model=model,
            latents=out_ref.latents,
            video=video_ref,
            cond_embed=out_ref.cond_embed,
            decoded=out_ref.recon,
        )
        smoothness_reference = build_smoothness_neighborhood_reference(
            model=model,
            latents=out_ref.latents,
            video=video_ref,
            cond_embed=out_ref.cond_embed,
            signature_mode="descriptor_span_stats",
            decoded=out_ref.recon,
        )

    out_cur = model(video_cur, condition_cur)
    model.zero_grad(set_to_none=True)
    terms = local_measure_loss(
        model=model,
        latents=out_cur.latents,
        video=video_cur,
        cond_embed=out_cur.cond_embed,
        signature_knn=3,
        signature_temperature=0.5,
        geometry_knn=3,
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
        decoded=out_cur.recon,
        geometry_reference=geometry_reference,
        smoothness_reference=smoothness_reference,
    )
    loss = (
        terms["local_drift"]
        + terms["local_diffusion"]
        + terms["measure_stationarity"]
        + terms["response_smoothness"]
        + terms["measure_tilt_alignment"]
    )
    loss.backward()

    assert float(terms["response_geometry_pool_size"].item()) == 7.0
    assert float(terms["response_geometry_reference_pool_size"].item()) == 4.0
    assert float(terms["response_geometry_reference_neighbor_ratio"].item()) > 0.0
    assert torch.isfinite(terms["tangent_bundle_compatibility"]).all()
    assert float(terms["response_smoothness_pool_size"].item()) == 7.0
    assert all(torch.isfinite(value).all() for value in terms.values())
    assert model.local_diffusion_factor_head is not None
    assert model.measure_tilt_head is not None
    diffusion_grad = model.local_diffusion_factor_head[-1].weight.grad
    tilt_grad = model.measure_tilt_head[-1].weight.grad
    assert diffusion_grad is not None
    assert tilt_grad is not None
    assert torch.isfinite(diffusion_grad).all()
    assert torch.isfinite(tilt_grad).all()
