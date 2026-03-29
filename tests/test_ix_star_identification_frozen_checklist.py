from __future__ import annotations

import torch

from vh_mvp.data import SyntheticVideoDataset
from vh_mvp.losses import local_measure_loss, response_signature_dim
from vh_mvp.losses.objectives import identification_nontriviality_loss, tilt_overreach_loss
from vh_mvp.models import VideoDynamicsMVP


def _build_batch(
    *,
    batch_size: int,
    seq_len: int,
    seed: int = 71,
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


def test_identification_nontriviality_penalizes_isotropic_prediction() -> None:
    pred_eigs = torch.tensor([[0.34, 0.33, 0.33]], dtype=torch.float32)
    target_eigs = torch.tensor([[0.82, 0.13, 0.05]], dtype=torch.float32)

    loss, stats = identification_nontriviality_loss(pred_eigs, target_eigs)

    assert float(loss.item()) > 0.0
    assert float(stats["target_anisotropy"].item()) > float(stats["pred_anisotropy"].item())
    assert float(stats["pred_effective_rank"].item()) > float(stats["target_effective_rank"].item())
    assert float(stats["target_spectral_gap"].item()) > float(stats["pred_spectral_gap"].item())


def test_tilt_overreach_grows_with_geometry_signal_and_residual() -> None:
    tilt = torch.tensor([[0.7], [-0.5]], dtype=torch.float32)
    low = tilt_overreach_loss(
        tilt,
        geometry_signal=torch.tensor(0.1),
        geometry_residual=torch.tensor(0.1),
    )
    high = tilt_overreach_loss(
        tilt,
        geometry_signal=torch.tensor(3.0),
        geometry_residual=torch.tensor(2.5),
    )

    assert float(high.item()) > float(low.item())
    assert float(high.item()) > 0.0


def test_local_measure_loss_identification_constraints_backward() -> None:
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
        + terms["tangent_nontriviality"]
        + terms["measure_tilt_overreach"]
        + 0.1 * terms["tangent_projection"]
    )
    loss.backward()

    assert all(torch.isfinite(value).all() for value in terms.values())
    assert torch.isfinite(terms["tangent_nontriviality"]).all()
    assert torch.isfinite(terms["measure_tilt_overreach"]).all()
    assert model.local_diffusion_factor_head is not None
    assert model.measure_tilt_head is not None
    diffusion_grad = model.local_diffusion_factor_head[-1].weight.grad
    tilt_grad = model.measure_tilt_head[-1].weight.grad
    assert diffusion_grad is not None
    assert tilt_grad is not None
    assert torch.isfinite(diffusion_grad).all()
    assert torch.isfinite(tilt_grad).all()
