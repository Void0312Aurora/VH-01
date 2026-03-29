from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def rgb_to_grayscale_video(video: torch.Tensor) -> torch.Tensor:
    if video.size(-3) == 1:
        return video
    if video.size(-3) != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {video.size(-3)}.")
    weights = video.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 1, 3, 1, 1)
    return (video * weights).sum(dim=-3, keepdim=True)


def video_mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Mismatched shapes: {pred.shape} vs {target.shape}")
    return (pred - target).square().mean(dim=(1, 2, 3, 4))


def video_psnr_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
) -> torch.Tensor:
    mse = video_mse_per_sample(pred, target).clamp_min(1e-12)
    return 10.0 * torch.log10((data_range**2) / mse)


def _gaussian_kernel(
    window_size: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, window_size, window_size)


def _ssim_per_frame(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Mismatched shapes: {pred.shape} vs {target.shape}")
    if pred.dim() != 4 or pred.size(1) != 1:
        raise ValueError(f"Expected shape [N, 1, H, W], got {pred.shape}")

    kernel = _gaussian_kernel(
        window_size,
        sigma,
        device=pred.device,
        dtype=pred.dtype,
    )
    padding = window_size // 2

    mu_x = F.conv2d(pred, kernel, padding=padding)
    mu_y = F.conv2d(target, kernel, padding=padding)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred.square(), kernel, padding=padding) - mu_x_sq
    sigma_y_sq = F.conv2d(target.square(), kernel, padding=padding) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, kernel, padding=padding) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1e-12)
    return ssim_map.mean(dim=(1, 2, 3))


def video_ssim_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    pred_gray = rgb_to_grayscale_video(pred)
    target_gray = rgb_to_grayscale_video(target)
    batch, steps, _, height, width = pred_gray.shape
    pred_frames = pred_gray.reshape(batch * steps, 1, height, width)
    target_frames = target_gray.reshape(batch * steps, 1, height, width)
    scores = _ssim_per_frame(
        pred_frames,
        target_frames,
        data_range=data_range,
        window_size=window_size,
        sigma=sigma,
    )
    return scores.view(batch, steps).mean(dim=1)


def try_build_lpips(
    *,
    device: torch.device,
    net: str = "alex",
):
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    try:
        model = lpips.LPIPS(net=net)
    except Exception:
        return None
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def video_lpips_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_model,
) -> torch.Tensor:
    if lpips_model is None:
        raise RuntimeError("LPIPS model is not available.")
    if pred.shape != target.shape:
        raise ValueError(f"Mismatched shapes: {pred.shape} vs {target.shape}")
    batch, steps, channels, height, width = pred.shape
    if channels == 1:
        pred = pred.repeat(1, 1, 3, 1, 1)
        target = target.repeat(1, 1, 3, 1, 1)
    elif channels != 3:
        raise ValueError(f"Expected 1 or 3 channels for LPIPS, got {channels}.")

    pred_flat = pred.reshape(batch * steps, 3, height, width) * 2.0 - 1.0
    target_flat = target.reshape(batch * steps, 3, height, width) * 2.0 - 1.0
    scores = lpips_model(pred_flat, target_flat).view(batch, steps)
    return scores.mean(dim=1)


def scalar_psnr_from_mse(mse: float, *, data_range: float = 1.0) -> float:
    if mse <= 0.0:
        return math.inf
    return 10.0 * math.log10((data_range**2) / mse)
