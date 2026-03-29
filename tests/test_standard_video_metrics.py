from __future__ import annotations

import torch

from vh_mvp.utils.video_metrics import (
    scalar_psnr_from_mse,
    video_mse_per_sample,
    video_psnr_per_sample,
    video_ssim_per_sample,
)


def test_identical_videos_have_zero_mse_and_high_psnr_ssim() -> None:
    video = torch.rand(2, 4, 3, 16, 16)
    mse = video_mse_per_sample(video, video)
    psnr = video_psnr_per_sample(video, video)
    ssim = video_ssim_per_sample(video, video)

    assert torch.allclose(mse, torch.zeros_like(mse))
    assert torch.all(psnr > 100.0)
    assert torch.allclose(ssim, torch.ones_like(ssim), atol=1e-4)


def test_different_videos_have_lower_quality_scores() -> None:
    video_a = torch.zeros(2, 4, 3, 16, 16)
    video_b = torch.ones(2, 4, 3, 16, 16)
    mse = video_mse_per_sample(video_a, video_b)
    psnr = video_psnr_per_sample(video_a, video_b)
    ssim = video_ssim_per_sample(video_a, video_b)

    assert torch.allclose(mse, torch.ones_like(mse))
    assert torch.allclose(psnr, torch.zeros_like(psnr), atol=1e-6)
    assert torch.all(ssim < 0.2)


def test_scalar_psnr_matches_unit_mse_boundary() -> None:
    assert scalar_psnr_from_mse(1.0) == 0.0
