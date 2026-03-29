from __future__ import annotations

import torch
import torch.nn.functional as F

from vh_mvp.data import SyntheticVideoDataset
from vh_mvp.losses import (
    LocalGeneratorTarget,
    local_measure_loss,
    local_measure_targets,
    response_signature,
    response_signature_dim,
)
from vh_mvp.models import LocalGenerator, VideoDynamicsMVP


def _build_batch(
    *,
    batch_size: int,
    seq_len: int,
    seed: int = 41,
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


def test_local_generator_matches_existing_component_paths() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition = _build_batch(batch_size=4, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        state = model.trajectory_point(out.latents)
        signatures = response_signature(
            model,
            out.latents,
            video,
            out.cond_embed,
            decoded=out.recon,
            mode="descriptor_span_stats",
        )
        generator = model.local_generator(
            out.latents,
            out.cond_embed,
            response_context=signatures,
            state=state,
        )
        measure = model.conditional_measure(out.latents, out.cond_embed, state=state)
        tangent_structure = model.local_tangent_structure(
            out.latents,
            out.cond_embed,
            response_context=signatures,
            state=state,
        )
        diffusion_matrix = model.local_diffusion_matrix(
            out.latents,
            out.cond_embed,
            response_context=signatures,
            state=state,
            tangent_structure=tangent_structure,
        )
        tangent_core_cov = model.local_tangent_covariance(
            out.latents,
            out.cond_embed,
            response_context=signatures,
            state=state,
            tangent_structure=tangent_structure,
        )
        drift = model.trajectory_drift(out.latents, out.cond_embed)

    assert isinstance(generator, LocalGenerator)
    assert torch.allclose(generator.state, state, atol=1e-6, rtol=1e-5)
    assert torch.allclose(generator.drift, drift, atol=1e-6, rtol=1e-5)
    assert torch.allclose(generator.diffusion_matrix, diffusion_matrix, atol=1e-6, rtol=1e-5)
    assert generator.tangent_structure is not None
    assert tangent_structure is not None
    assert torch.allclose(
        generator.tangent_structure["projector"],
        tangent_structure["projector"],
        atol=1e-6,
        rtol=1e-5,
    )
    assert generator.tangent_core_cov is not None
    assert tangent_core_cov is not None
    assert torch.allclose(generator.tangent_core_cov, tangent_core_cov, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        generator.conditional_measure.log_total_density,
        measure.log_total_density,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.isfinite(generator.density_weights()).all()


def test_local_generator_apply_interfaces_match_reference_formula() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len)
    model.eval()
    video, condition = _build_batch(batch_size=3, seq_len=seq_len)

    with torch.no_grad():
        out = model(video, condition)
        signatures = response_signature(
            model,
            out.latents,
            video,
            out.cond_embed,
            decoded=out.recon,
            mode="descriptor_span_stats",
        )
        generator = model.local_generator(
            out.latents,
            out.cond_embed,
            response_context=signatures,
            state=model.trajectory_point(out.latents),
        )
        directions = F.normalize(
            torch.randn(5, generator.state.size(1), dtype=generator.state.dtype),
            dim=-1,
            eps=1e-6,
        )
        trig_scale = 0.75
        manual_linear = generator.drift @ directions.T
        projected_state = generator.state @ directions.T
        projected_diffusion = torch.einsum("bde,kd,ke->bk", generator.diffusion_matrix, directions, directions)
        manual_quadratic = 2.0 * projected_state * manual_linear + projected_diffusion
        manual_trig = (
            trig_scale * torch.cos(trig_scale * projected_state) * manual_linear
            - 0.5 * (trig_scale**2) * torch.sin(trig_scale * projected_state) * projected_diffusion
        )
        manual_radial = (
            2.0 * (generator.state * generator.drift).sum(dim=1)
            + generator.diffusion_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        )

    assert torch.allclose(generator.apply_linear(directions), manual_linear, atol=1e-6, rtol=1e-5)
    assert torch.allclose(generator.apply_quadratic(directions), manual_quadratic, atol=1e-6, rtol=1e-5)
    assert torch.allclose(generator.apply_trig(directions, trig_scale), manual_trig, atol=1e-6, rtol=1e-5)
    assert torch.allclose(generator.apply_radial(), manual_radial, atol=1e-6, rtol=1e-5)
    assert torch.allclose(generator.trace(), generator.diffusion_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1))


def test_local_measure_targets_return_local_generator_target_with_compat_access() -> None:
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

    assert isinstance(targets, LocalGeneratorTarget)
    assert torch.allclose(targets["source_point"], targets.source_point, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        targets["source_summary_context"],
        targets.source_summary_context,
        atol=1e-6,
        rtol=1e-5,
    )
    assert targets.get("source_measure") is targets.source_measure
    assert targets.source_measure is not None
    assert torch.isfinite(targets.source_measure.log_total_density).all()
    as_dict = targets.as_dict()
    assert as_dict["target_neighbor_idx"] is targets.target_neighbor_idx
    assert as_dict["target_transport"] is targets.target_transport
    assert targets.target_tangent_frame is not None
    assert targets.target_neighbor_idx is not None
    assert targets.target_neighbor_weights is not None


def test_local_measure_loss_first_class_generator_path_backward() -> None:
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
        + 0.1 * terms["tangent_projection"]
    )
    loss.backward()

    assert all(torch.isfinite(value).all() for value in terms.values())
    assert model.measure_tilt_head is not None
    assert model.local_diffusion_factor_head is not None
    tilt_grad = model.measure_tilt_head[-1].weight.grad
    diffusion_grad = model.local_diffusion_factor_head[-1].weight.grad
    assert tilt_grad is not None
    assert diffusion_grad is not None
    assert torch.isfinite(tilt_grad).all()
    assert torch.isfinite(diffusion_grad).all()
