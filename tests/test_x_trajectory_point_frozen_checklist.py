from __future__ import annotations

import torch

from vh_mvp.data import SyntheticVideoDataset
from vh_mvp.losses import local_measure_targets, response_signature_dim
from vh_mvp.models import VideoDynamicsMVP


def _build_batch(
    *,
    batch_size: int,
    seq_len: int,
    seed: int = 11,
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
    )


def test_trajectory_point_summary_and_state_roles_are_separated() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition = _build_batch(batch_size=3, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        point = model.trajectory_point(out.latents)
        summary_context = model.trajectory_summary_context(out.latents)
        state = model.trajectory_state(out.latents)
        diagnostics = model.trajectory_state_diagnostics(out.latents)

    assert point.shape == summary_context.shape == state.shape == (3, model.latent_dim)
    assert torch.allclose(state, point, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(point, summary_context)
    assert "chart_expert_entropy" in diagnostics
    assert "chart_expert_max_weight" in diagnostics
    assert "trajectory_point_norm" in diagnostics
    assert torch.isfinite(point).all()
    assert torch.isfinite(summary_context).all()


def test_local_measure_context_defaults_to_trajectory_point() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition = _build_batch(batch_size=2, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        point = model.trajectory_point(out.latents)
        default_context = model.local_measure_context(out.latents, out.cond_embed)
        point_only_context = model.local_measure_context(out.latents, out.cond_embed, include_condition=False)
        explicit_context = model.local_measure_context(out.latents, out.cond_embed, state=point)

    expected = torch.cat([point, out.cond_embed], dim=-1)
    assert torch.allclose(default_context, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(explicit_context, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(point_only_context, point, atol=1e-6, rtol=1e-5)


def test_tangent_frame_uses_point_and_summary_context_consistently() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition = _build_batch(batch_size=2, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        point = model.trajectory_point(out.latents)
        summary_context = model.trajectory_summary_context(out.latents)
        default_frame = model.trajectory_tangent_frame(out.latents)
        explicit_frame = model.trajectory_tangent_frame(
            out.latents,
            point=point,
            summary_context=summary_context,
        )

    assert default_frame is not None
    assert explicit_frame is not None
    assert torch.allclose(default_frame, explicit_frame, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(default_frame).all()


def test_local_measure_targets_use_trajectory_point_anchor() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition = _build_batch(batch_size=4, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        point = model.trajectory_point(out.latents)
        summary_context = model.trajectory_summary_context(out.latents)
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
        )

    assert torch.allclose(targets["source_point"], point, atol=1e-6, rtol=1e-5)
    assert torch.allclose(targets["source_summary_context"], summary_context, atol=1e-6, rtol=1e-5)
    assert targets["target_neighbor_idx"] is not None
    assert targets["target_neighbor_weights"] is not None
    assert targets["target_tangent_frame"] is not None
