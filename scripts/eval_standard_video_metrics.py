from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vh_mvp.config import load_config
from vh_mvp.data import FolderVideoDataset
from vh_mvp.models import VideoDynamicsMVP
from vh_mvp.train.trainer import build_model
from vh_mvp.utils.video_metrics import (
    scalar_psnr_from_mse,
    try_build_lpips,
    video_lpips_per_sample,
    video_mse_per_sample,
    video_psnr_per_sample,
    video_ssim_per_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standard video metrics (PSNR/SSIM/optional LPIPS).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--lpips-net", type=str, default="alex")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model_from_config(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[VideoDynamicsMVP, object]:
    cfg = load_config(config_path)
    model = build_model(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    current_state = model.state_dict()
    loaded_state = checkpoint["model"]
    compatible_state = {}
    skipped_keys: list[str] = []
    for key, value in loaded_state.items():
        target = current_state.get(key)
        if target is None or target.shape != value.shape:
            skipped_keys.append(key)
            continue
        compatible_state[key] = value
    model.load_state_dict(compatible_state, strict=False)
    setattr(model, "_standard_eval_skipped_keys", skipped_keys)
    model.eval()
    return model, cfg


def build_dataset(cfg, split: str) -> FolderVideoDataset:
    if cfg.data.kind != "folder":
        raise ValueError("Standard video metric eval currently supports only folder datasets.")
    manifest = cfg.data.manifest_path if split == "train" else (cfg.data.val_manifest_path or cfg.data.manifest_path)
    return FolderVideoDataset(
        root=cfg.data.root,
        manifest_path=manifest,
        seq_len=cfg.data.seq_len,
        image_size=cfg.data.image_size,
    )


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(
    *,
    model: VideoDynamicsMVP,
    dataset: FolderVideoDataset,
    batch_size: int,
    device: torch.device,
    max_samples: int,
    lpips_model,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    recon_mse_sum = 0.0
    recon_psnr_sum = 0.0
    recon_ssim_sum = 0.0
    recon_lpips_sum = 0.0

    future_mse_sum = 0.0
    future_psnr_sum = 0.0
    future_ssim_sum = 0.0
    future_lpips_sum = 0.0

    processed = 0
    rows: list[dict[str, object]] = []

    for batch in loader:
        if max_samples > 0 and processed >= max_samples:
            break

        video = batch["video"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)
        sample_ids = list(batch["sample_id"])

        if max_samples > 0 and processed + video.size(0) > max_samples:
            keep = max_samples - processed
            video = video[:keep]
            condition = condition[:keep]
            sample_ids = sample_ids[:keep]

        out = model(video, condition)
        recon = out.recon
        recon_mse = video_mse_per_sample(recon, video)
        recon_psnr = video_psnr_per_sample(recon, video)
        recon_ssim = video_ssim_per_sample(recon, video)
        recon_lpips = video_lpips_per_sample(recon, video, lpips_model) if lpips_model is not None else None

        rollout_latents, _ = model.rollout_from(out.latents[:, 0], out.cond_embed, steps=video.size(1) - 1)
        rollout_video = model.decode_video(rollout_latents, out.cond_embed)
        target_future = video[:, 1:]
        future_mse = video_mse_per_sample(rollout_video, target_future)
        future_psnr = video_psnr_per_sample(rollout_video, target_future)
        future_ssim = video_ssim_per_sample(rollout_video, target_future)
        future_lpips = (
            video_lpips_per_sample(rollout_video, target_future, lpips_model) if lpips_model is not None else None
        )

        for idx, sample_id in enumerate(sample_ids):
            row = {
                "sample_id": sample_id,
                "recon_mse": float(recon_mse[idx].item()),
                "recon_psnr": float(recon_psnr[idx].item()),
                "recon_ssim": float(recon_ssim[idx].item()),
                "future_mse": float(future_mse[idx].item()),
                "future_psnr": float(future_psnr[idx].item()),
                "future_ssim": float(future_ssim[idx].item()),
            }
            if recon_lpips is not None and future_lpips is not None:
                row["recon_lpips"] = float(recon_lpips[idx].item())
                row["future_lpips"] = float(future_lpips[idx].item())
            rows.append(row)

        recon_mse_sum += float(recon_mse.sum().item())
        recon_psnr_sum += float(recon_psnr.sum().item())
        recon_ssim_sum += float(recon_ssim.sum().item())
        future_mse_sum += float(future_mse.sum().item())
        future_psnr_sum += float(future_psnr.sum().item())
        future_ssim_sum += float(future_ssim.sum().item())
        if recon_lpips is not None and future_lpips is not None:
            recon_lpips_sum += float(recon_lpips.sum().item())
            future_lpips_sum += float(future_lpips.sum().item())

        processed += video.size(0)

    denom = max(processed, 1)
    summary: dict[str, object] = {
        "num_samples": processed,
        "recon_mse": recon_mse_sum / denom,
        "recon_psnr": recon_psnr_sum / denom,
        "recon_ssim": recon_ssim_sum / denom,
        "future_mse": future_mse_sum / denom,
        "future_psnr": future_psnr_sum / denom,
        "future_ssim": future_ssim_sum / denom,
        "future_psnr_from_mse": scalar_psnr_from_mse(future_mse_sum / denom),
    }
    if lpips_model is not None:
        summary["recon_lpips"] = recon_lpips_sum / denom
        summary["future_lpips"] = future_lpips_sum / denom
    return summary, rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = build_model_from_config(args.config, args.checkpoint, device)
    dataset = build_dataset(cfg, args.split)
    lpips_model = try_build_lpips(device=device, net=args.lpips_net)
    summary, rows = evaluate(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        max_samples=args.max_samples,
        lpips_model=lpips_model,
    )

    save_csv(metrics_dir / f"{args.split}_standard_video_metrics.csv", rows)
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "lpips_enabled": lpips_model is not None,
        "skipped_state_keys": getattr(model, "_standard_eval_skipped_keys", []),
        "summary": summary,
    }
    with (metrics_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"device={device}")
    print(f"split={args.split}")
    print(f"lpips_enabled={lpips_model is not None}")
    skipped_keys = getattr(model, "_standard_eval_skipped_keys", [])
    print(f"skipped_state_keys={len(skipped_keys)}")
    print(f"recon_mse={summary['recon_mse']:.6f}")
    print(f"recon_psnr={summary['recon_psnr']:.4f}")
    print(f"recon_ssim={summary['recon_ssim']:.4f}")
    print(f"future_mse={summary['future_mse']:.6f}")
    print(f"future_psnr={summary['future_psnr']:.4f}")
    print(f"future_ssim={summary['future_ssim']:.4f}")
    if "recon_lpips" in summary:
        print(f"recon_lpips={summary['recon_lpips']:.4f}")
    if "future_lpips" in summary:
        print(f"future_lpips={summary['future_lpips']:.4f}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
