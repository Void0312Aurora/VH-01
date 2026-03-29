from __future__ import annotations

import torch

from vh_mvp.data import SyntheticVideoDataset
from vh_mvp.losses import local_measure_loss, response_signature, response_signature_dim
from vh_mvp.models import BaseLocalGenerator, ConditionalGeneratorDelta, LocalGenerator, VideoDynamicsMVP


def _build_batch(
    *,
    batch_size: int,
    seq_len: int,
    seed: int = 97,
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
    measure_density_mode: str = "joint",
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
        measure_density_mode=measure_density_mode,
        encoder_condition_mode="residual_temporal",
        encoder_condition_hidden_dim=32,
        encoder_condition_scale=0.1,
    )


def test_local_generator_explicitly_splits_base_and_delta_in_joint_mode() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len, measure_density_mode="joint")
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
        base_measure = model.base_measure(out.latents, state=state)

    assert isinstance(generator, LocalGenerator)
    assert isinstance(generator.base_generator, BaseLocalGenerator)
    assert isinstance(generator.conditional_delta, ConditionalGeneratorDelta)
    assert torch.allclose(
        generator.drift,
        generator.base_generator.drift + generator.conditional_delta.drift,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        generator.diffusion_matrix,
        generator.base_generator.diffusion_matrix + generator.conditional_delta.diffusion_matrix,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        generator.conditional_measure.log_total_density,
        generator.base_generator.base_measure.log_base_density + generator.conditional_delta.log_tilt,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        generator.base_measure.log_base_density,
        base_measure.log_base_density,
        atol=1e-6,
        rtol=1e-5,
    )
    if generator.tangent_core_cov is not None:
        assert generator.base_generator.tangent_core_cov is not None
        assert generator.conditional_delta.tangent_core_cov is not None
        assert torch.allclose(
            generator.tangent_core_cov,
            generator.base_generator.tangent_core_cov + generator.conditional_delta.tangent_core_cov,
            atol=1e-6,
            rtol=1e-5,
        )


def test_joint_measure_components_follow_base_plus_tilt_decomposition() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len, measure_density_mode="joint")
    model.eval()
    video, condition = _build_batch(batch_size=3, seq_len=seq_len, seed=99)

    with torch.no_grad():
        out = model(video, condition)
        state = model.trajectory_point(out.latents)
        base, tilt, total = model.measure_log_density_components(out.latents, out.cond_embed, state=state)

    assert torch.isfinite(base).all()
    assert torch.isfinite(tilt).all()
    assert torch.isfinite(total).all()
    assert torch.allclose(total, base + tilt, atol=1e-6, rtol=1e-5)


def test_local_measure_loss_with_delta_budget_backpropagates() -> None:
    seq_len = 5
    model = _build_model(seq_len=seq_len, measure_density_mode="joint")
    model.train()
    video, condition = _build_batch(batch_size=4, seq_len=seq_len, seed=123)
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
        + 0.1 * terms["generator_delta_budget"]
    )
    loss.backward()

    assert all(torch.isfinite(value).all() for value in terms.values())
    assert torch.isfinite(terms["generator_delta_budget"]).all()
    assert torch.isfinite(terms["generator_delta_trace"]).all()
    assert model.base_dynamics.net[-1].weight.grad is not None
    assert model.cond_delta.net[-1].weight.grad is not None
    assert torch.isfinite(model.base_dynamics.net[-1].weight.grad).all()
    assert torch.isfinite(model.cond_delta.net[-1].weight.grad).all()
