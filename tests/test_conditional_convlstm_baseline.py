from __future__ import annotations

import torch

from vh_mvp.baselines import ConditionalConvLSTMBaseline


def test_baseline_forward_shapes_match_video_protocol() -> None:
    model = ConditionalConvLSTMBaseline(
        channels=3,
        image_size=32,
        cond_dim=32,
        base_channels=8,
        latent_channels=16,
        hidden_channels=16,
    )
    video = torch.rand(2, 8, 3, 32, 32)
    condition = torch.randint(0, 2, (2, 8))
    condition[:, 1] = torch.randint(0, 4, (2,))
    condition[:, 5] = torch.randint(0, 3, (2,))
    condition[:, 6] = torch.randint(0, 3, (2,))
    condition[:, 7] = torch.randint(0, 3, (2,))

    out = model(video, condition)
    assert out.recon.shape == video.shape
    assert out.future.shape == video[:, 1:].shape
    assert torch.isfinite(out.recon).all()
    assert torch.isfinite(out.future).all()


def test_rollout_from_first_produces_finite_latents() -> None:
    model = ConditionalConvLSTMBaseline(
        channels=3,
        image_size=32,
        cond_dim=32,
        base_channels=8,
        latent_channels=16,
        hidden_channels=16,
    )
    first_latent = torch.randn(2, 16, 8, 8)
    condition = torch.randint(0, 2, (2, 8))
    condition[:, 1] = torch.randint(0, 4, (2,))
    condition[:, 5] = torch.randint(0, 3, (2,))
    condition[:, 6] = torch.randint(0, 3, (2,))
    condition[:, 7] = torch.randint(0, 3, (2,))
    cond_embed = model.condition_encoder(condition)
    future_latents = model.rollout_from_first(first_latent, cond_embed, steps=7)
    assert future_latents.shape == (2, 7, 16, 8, 8)
    assert torch.isfinite(future_latents).all()
