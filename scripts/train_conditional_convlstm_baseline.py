from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from vh_mvp.baselines import ConditionalConvLSTMBaseline
from vh_mvp.config import AppConfig, load_config
from vh_mvp.train.trainer import build_dataloaders, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a same-protocol Conditional ConvLSTM-lite baseline.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-train-steps", type=int, default=0)
    return parser.parse_args()


def load_baseline_section(path: str) -> dict[str, float | int]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    section = raw.get("baseline") or {}
    defaults: dict[str, float | int] = {
        "cond_dim": 64,
        "base_channels": 32,
        "latent_channels": 64,
        "hidden_channels": 64,
        "recon_weight": 1.0,
        "future_weight": 1.0,
    }
    defaults.update(section)
    return defaults


def build_baseline(
    *,
    cfg: AppConfig,
    baseline_cfg: dict[str, float | int],
    device: torch.device,
) -> ConditionalConvLSTMBaseline:
    model = ConditionalConvLSTMBaseline(
        channels=cfg.data.channels,
        image_size=cfg.data.image_size,
        cond_dim=int(baseline_cfg["cond_dim"]),
        base_channels=int(baseline_cfg["base_channels"]),
        latent_channels=int(baseline_cfg["latent_channels"]),
        hidden_channels=int(baseline_cfg["hidden_channels"]),
    )
    return model.to(device)


def compute_losses(
    model: ConditionalConvLSTMBaseline,
    video: torch.Tensor,
    condition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(video, condition)
    recon_loss = F.mse_loss(out.recon, video)
    if video.size(1) < 2:
        future_loss = recon_loss.new_tensor(0.0)
    else:
        future_loss = F.mse_loss(out.future, video[:, 1:])
    return recon_loss + future_loss, recon_loss, future_loss


def write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def evaluate(
    *,
    model: ConditionalConvLSTMBaseline,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_future = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            condition = batch["condition"].to(device, non_blocking=True)
            loss, recon_loss, future_loss = compute_losses(model, video, condition)
            total_loss += float(loss.item())
            total_recon += float(recon_loss.item())
            total_future += float(future_loss.item())
            steps += 1
    denom = max(steps, 1)
    return {
        "loss": total_loss / denom,
        "recon": total_recon / denom,
        "future": total_future / denom,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    baseline_cfg = load_baseline_section(args.config)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)
    model = build_baseline(cfg=cfg, baseline_cfg=baseline_cfg, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    amp_enabled = bool(cfg.train.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_loader, val_loader, _, _ = build_dataloaders(cfg)
    history: list[dict[str, float]] = []
    best_future = float("inf")
    best_recon = float("inf")

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_future = 0.0
        steps = 0

        progress = tqdm(train_loader, desc=f"baseline train {epoch:03d}", leave=False)
        for batch in progress:
            if args.max_train_steps and steps >= args.max_train_steps:
                break

            video = batch["video"].to(device, non_blocking=True)
            condition = batch["condition"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with autocast_ctx:
                loss, recon_loss, future_loss = compute_losses(model, video, condition)
                total = (
                    float(baseline_cfg["recon_weight"]) * recon_loss
                    + float(baseline_cfg["future_weight"]) * future_loss
                )

            if amp_enabled:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_norm)
                optimizer.step()

            total_loss += float(total.item())
            total_recon += float(recon_loss.item())
            total_future += float(future_loss.item())
            steps += 1
            progress.set_postfix(
                loss=f"{total_loss / steps:.4f}",
                recon=f"{total_recon / steps:.4f}",
                future=f"{total_future / steps:.4f}",
            )

        train_metrics = {
            "loss": total_loss / max(steps, 1),
            "recon": total_recon / max(steps, 1),
            "future": total_future / max(steps, 1),
        }
        val_metrics = evaluate(model=model, loader=val_loader, device=device)
        record = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_recon": train_metrics["recon"],
            "train_future": train_metrics["future"],
            "val_loss": val_metrics["loss"],
            "val_recon": val_metrics["recon"],
            "val_future": val_metrics["future"],
        }
        history.append(record)
        with (metrics_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_history_csv(metrics_dir / "history.csv", history)

        checkpoint = {
            "model": model.state_dict(),
            "config_path": args.config,
            "baseline": baseline_cfg,
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["future"] < best_future:
            best_future = val_metrics["future"]
            torch.save(checkpoint, output_dir / "best.pt")
        if val_metrics["recon"] < best_recon:
            best_recon = val_metrics["recon"]
            torch.save(checkpoint, output_dir / "best_recon.pt")

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_recon={val_metrics['recon']:.4f} val_future={val_metrics['future']:.4f}"
        )


if __name__ == "__main__":
    main()
