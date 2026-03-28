from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vh_mvp.config import load_config
from vh_mvp.data import FolderVideoDataset
from vh_mvp.models import VideoDynamicsMVP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen video latents with configurable top-1 probes.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-type", type=str, default="linear", choices=("linear", "mlp"))
    parser.add_argument("--probe-hidden-dim", type=int, default=128)
    parser.add_argument("--feature-mode", type=str, default="full", choices=("full", "semantic", "identity"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_model_from_config(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[VideoDynamicsMVP, object]:
    cfg = load_config(config_path)
    model = VideoDynamicsMVP(
        channels=cfg.data.channels,
        base_channels=cfg.model.base_channels,
        latent_dim=cfg.model.latent_dim,
        cond_dim=cfg.model.cond_dim,
        hidden_dim=cfg.model.hidden_dim,
        condition_score_mode=cfg.model.condition_score_mode,
        energy_hidden_dim=cfg.model.energy_hidden_dim,
        identity_num_classes=cfg.model.identity_num_classes,
        identity_hidden_dim=cfg.model.identity_hidden_dim,
        semantic_num_classes=cfg.model.semantic_num_classes,
        semantic_temperature=cfg.model.semantic_temperature,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, cfg


def build_real_datasets(cfg) -> tuple[FolderVideoDataset, FolderVideoDataset]:
    if cfg.data.kind != "folder":
        raise ValueError("Top-1 probe currently supports only folder-based real datasets.")
    train_ds = FolderVideoDataset(
        root=cfg.data.root,
        manifest_path=cfg.data.manifest_path,
        seq_len=cfg.data.seq_len,
        image_size=cfg.data.image_size,
    )
    val_ds = FolderVideoDataset(
        root=cfg.data.root,
        manifest_path=cfg.data.val_manifest_path or cfg.data.manifest_path,
        seq_len=cfg.data.seq_len,
        image_size=cfg.data.image_size,
    )
    return train_ds, val_ds


@torch.no_grad()
def extract_features(
    model: VideoDynamicsMVP,
    dataset: FolderVideoDataset,
    label_to_idx: dict[str, int],
    batch_size: int,
    device: torch.device,
    feature_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    features = []
    targets = []
    sample_ids: list[str] = []
    label_names: list[str] = []

    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        latents = model.encode_video(video)
        semantic_summary, identity_residual, full_summary = model.decompose_summary(latents)
        if feature_mode == "full":
            feature = full_summary
        elif feature_mode == "semantic":
            feature = semantic_summary
        elif feature_mode == "identity":
            if model.identity_classifier is None:
                raise RuntimeError("Identity feature mode requires a model with an identity residual branch.")
            feature = identity_residual
        else:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        features.append(feature.cpu())
        labels = list(batch["label"])
        sample_ids.extend(list(batch["sample_id"]))
        label_names.extend(labels)
        targets.extend(label_to_idx[label] for label in labels)

    return (
        torch.cat(features, dim=0),
        torch.tensor(targets, dtype=torch.long),
        sample_ids,
        label_names,
    )


def normalize_features(
    train_features: torch.Tensor,
    val_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_features - mean) / std, (val_features - mean) / std, mean, std


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def build_probe(
    input_dim: int,
    num_classes: int,
    probe_type: str,
    probe_hidden_dim: int,
) -> nn.Module:
    if probe_type == "linear":
        return nn.Linear(input_dim, num_classes)
    if probe_type == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, probe_hidden_dim),
            nn.ReLU(),
            nn.Linear(probe_hidden_dim, num_classes),
        )
    raise ValueError(f"Unsupported probe_type: {probe_type}")


def train_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    probe_type: str,
    probe_hidden_dim: int,
) -> tuple[nn.Module, list[dict[str, float]], float]:
    probe = build_probe(
        input_dim=train_features.size(1),
        num_classes=num_classes,
        probe_type=probe_type,
        probe_hidden_dim=probe_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(train_features, train_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    history: list[dict[str, float]] = []
    best_acc = -1.0
    best_state = None

    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    val_features = val_features.to(device)
    val_targets = val_targets.to(device)

    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        total_acc = 0.0
        total_steps = 0

        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(batch_features)
            loss = F.cross_entropy(logits, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits.detach(), batch_targets)
            total_steps += 1

        probe.eval()
        with torch.no_grad():
            train_logits = probe(train_features)
            val_logits = probe(val_features)
            train_acc = accuracy_from_logits(train_logits, train_targets)
            val_acc = accuracy_from_logits(val_logits, val_targets)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(total_steps, 1),
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Probe did not produce a valid checkpoint.")
    probe.load_state_dict(best_state)
    return probe, history, best_acc


@torch.no_grad()
def evaluate_probe(
    probe: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    idx_to_label: list[str],
    sample_ids: list[str],
    label_names: list[str],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, str | int]], np.ndarray]:
    features = features.to(device)
    targets = targets.to(device)
    logits = probe(features)
    preds = logits.argmax(dim=1)
    acc = (preds == targets).float().mean().item()

    confusion = np.zeros((len(idx_to_label), len(idx_to_label)), dtype=np.int64)
    records: list[dict[str, str | int]] = []
    for sample_id, label_name, pred_idx, target_idx in zip(
        sample_ids,
        label_names,
        preds.cpu().tolist(),
        targets.cpu().tolist(),
    ):
        pred_label = idx_to_label[pred_idx]
        confusion[target_idx, pred_idx] += 1
        records.append(
            {
                "sample_id": sample_id,
                "label": label_name,
                "pred_label": pred_label,
                "correct": int(pred_idx == target_idx),
            }
        )

    per_class = {}
    for idx, label in enumerate(idx_to_label):
        total = int(confusion[idx].sum())
        per_class[label] = float(confusion[idx, idx] / total) if total else 0.0
    metrics = {
        "top1_acc": acc,
        "num_samples": len(sample_ids),
        "num_classes": len(idx_to_label),
        "macro_acc": float(np.mean(list(per_class.values()))) if per_class else 0.0,
    }
    return metrics | {"per_class_acc": per_class}, records, confusion


def save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_predictions_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "pred_label", "correct"])
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_csv(path: Path, confusion: np.ndarray, idx_to_label: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + idx_to_label)
        for idx, label in enumerate(idx_to_label):
            writer.writerow([label] + confusion[idx].tolist())


def save_history_plot(path: Path, history: list[dict[str, float]]) -> None:
    epochs = [int(row["epoch"]) for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]
    train_loss = [row["train_loss"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].set_title("Probe Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc, label="train_acc")
    axes[1].plot(epochs, val_acc, label="val_acc")
    axes[1].set_title("Probe Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = build_model_from_config(args.config, args.checkpoint, device)
    train_ds, val_ds = build_real_datasets(cfg)

    labels = sorted({sample.label for sample in train_ds.samples + val_ds.samples if sample.label})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = labels

    train_features, train_targets, train_ids, train_label_names = extract_features(
        model=model,
        dataset=train_ds,
        label_to_idx=label_to_idx,
        batch_size=args.probe_batch_size,
        device=device,
        feature_mode=args.feature_mode,
    )
    val_features, val_targets, val_ids, val_label_names = extract_features(
        model=model,
        dataset=val_ds,
        label_to_idx=label_to_idx,
        batch_size=args.probe_batch_size,
        device=device,
        feature_mode=args.feature_mode,
    )
    train_features, val_features, mean, std = normalize_features(train_features, val_features)

    probe, history, best_val_acc = train_probe(
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        num_classes=len(idx_to_label),
        device=device,
        epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        probe_type=args.probe_type,
        probe_hidden_dim=args.probe_hidden_dim,
    )

    train_metrics, train_predictions, train_confusion = evaluate_probe(
        probe, train_features, train_targets, idx_to_label, train_ids, train_label_names, device
    )
    val_metrics, val_predictions, val_confusion = evaluate_probe(
        probe, val_features, val_targets, idx_to_label, val_ids, val_label_names, device
    )

    history_dir = output_dir / "metrics"
    history_dir.mkdir(parents=True, exist_ok=True)
    save_history_csv(history_dir / "probe_history.csv", history)
    save_predictions_csv(history_dir / "train_predictions.csv", train_predictions)
    save_predictions_csv(history_dir / "val_predictions.csv", val_predictions)
    save_confusion_csv(history_dir / "train_confusion.csv", train_confusion, idx_to_label)
    save_confusion_csv(history_dir / "val_confusion.csv", val_confusion, idx_to_label)
    save_history_plot(history_dir / "probe_history.png", history)

    torch.save(
        {
            "probe": probe.state_dict(),
            "label_to_idx": label_to_idx,
            "feature_mean": mean,
            "feature_std": std,
            "history": history,
            "config": {
                "config_path": args.config,
                "checkpoint": args.checkpoint,
                "probe_epochs": args.probe_epochs,
                "probe_batch_size": args.probe_batch_size,
                "probe_lr": args.probe_lr,
                "probe_weight_decay": args.probe_weight_decay,
                "probe_type": args.probe_type,
                "probe_hidden_dim": args.probe_hidden_dim,
                "feature_mode": args.feature_mode,
            },
        },
        output_dir / "probe.pt",
    )

    summary = {
        "checkpoint": args.checkpoint,
        "feature_mode": args.feature_mode,
        "probe_type": args.probe_type,
        "probe_hidden_dim": args.probe_hidden_dim,
        "train": train_metrics,
        "val": val_metrics,
        "best_val_acc": best_val_acc,
    }
    with (history_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"device={device}")
    print(f"classes={len(idx_to_label)}")
    print(f"probe_type={args.probe_type}")
    print(f"train_top1={train_metrics['top1_acc']:.4f} train_macro={train_metrics['macro_acc']:.4f}")
    print(f"val_top1={val_metrics['top1_acc']:.4f} val_macro={val_metrics['macro_acc']:.4f}")
    print(f"best_val_top1={best_val_acc:.4f}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
