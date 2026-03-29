from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from vh_mvp.baselines import ConditionalConvLSTMBaseline, protocol_b_selection_metrics, summarize_encoded_video, train_condition_probe
from vh_mvp.config import load_config
from vh_mvp.data import FolderVideoDataset, build_condition_catalog, condition_tuple_from_tensor
from vh_mvp.support import summarize_condition_distribution
from vh_mvp.train.trainer import (
    build_dataloaders,
    build_model,
    build_condition_targets,
    compute_condition_logits,
    compute_condition_catalog_logits,
    evaluate_query_responsive_execution,
    resolve_device,
)


PROTOCOL_B_KEYS = (
    "cond_acc",
    "cond_true_prob",
    "cond_true_in90",
    "cond_support_ratio",
    "query_direct_mse",
    "query_support_top1_mse",
    "query_exec_mse",
    "query_set_best_mse",
    "query_oracle_mse",
    "query_exec_gap_to_oracle",
    "query_exec_gain_vs_top1",
    "query_match_true",
    "query_exec_set_size",
    "query_fallback_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate protocol-B query/readout metrics for MVP or baseline models.")
    parser.add_argument("--model-type", type=str, required=True, choices=("mvp", "conditional_convlstm"))
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--query-alpha", type=float, default=0.90)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-type", type=str, default="mlp", choices=("linear", "mlp"))
    parser.add_argument("--probe-hidden-dim", type=int, default=128)
    return parser.parse_args()


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def load_mvp_checkpoint(*, config_path: str, checkpoint_path: str, device: torch.device):
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
    model.eval()
    return model, cfg, skipped_keys


def load_baseline_checkpoint(*, config_path: str, checkpoint_path: str, device: torch.device):
    cfg = load_config(config_path)
    baseline_cfg = load_baseline_section(config_path)
    model = ConditionalConvLSTMBaseline(
        channels=cfg.data.channels,
        image_size=cfg.data.image_size,
        cond_dim=int(baseline_cfg["cond_dim"]),
        base_channels=int(baseline_cfg["base_channels"]),
        latent_channels=int(baseline_cfg["latent_channels"]),
        hidden_channels=int(baseline_cfg["hidden_channels"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, cfg, baseline_cfg


@torch.no_grad()
def extract_baseline_features(
    *,
    model: ConditionalConvLSTMBaseline,
    dataset: FolderVideoDataset,
    condition_catalog,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[torch.Tensor]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    features = []
    targets = []
    sample_ids: list[str] = []
    conditions: list[torch.Tensor] = []
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        encoded = model.encode_frames(video)
        features.append(summarize_encoded_video(encoded).cpu())
        condition = batch["condition"]
        target = build_condition_targets(condition, condition_catalog)
        if target is None:
            raise RuntimeError("Condition catalog targets are unexpectedly missing.")
        targets.append(target.cpu())
        sample_ids.extend(list(batch["sample_id"]))
        conditions.extend([row.clone() for row in condition])
    return torch.cat(features, dim=0), torch.cat(targets, dim=0), sample_ids, conditions


def evaluate_baseline_protocol_b(
    *,
    model: ConditionalConvLSTMBaseline,
    train_ds: FolderVideoDataset,
    eval_ds: FolderVideoDataset,
    condition_catalog,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, float]]]:
    train_features, train_targets, _, _ = extract_baseline_features(
        model=model,
        dataset=train_ds,
        condition_catalog=condition_catalog,
        device=device,
        batch_size=args.probe_batch_size,
    )
    eval_features, eval_targets, sample_ids, conditions = extract_baseline_features(
        model=model,
        dataset=eval_ds,
        condition_catalog=condition_catalog,
        device=device,
        batch_size=args.probe_batch_size,
    )

    probe_result = train_condition_probe(
        train_features=train_features,
        train_targets=train_targets,
        val_features=eval_features,
        val_targets=eval_targets,
        probe_type=args.probe_type,
        hidden_dim=args.probe_hidden_dim,
        epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        device=device,
    )
    probe = probe_result.probe.to(device)
    eval_features_device = eval_features.to(device)
    eval_targets_device = eval_targets.to(device)
    with torch.no_grad():
        logits = probe(eval_features_device)
    cond_acc = float((logits.argmax(dim=1) == eval_targets_device).float().mean().item())
    cond_dist = summarize_condition_distribution(
        logits,
        eval_targets_device,
        alpha=args.query_alpha,
        temperature=args.posterior_temperature,
    )

    candidate_conditions = condition_catalog.tensor.to(device)
    cond_embed_all = model.condition_encoder(candidate_conditions)
    limit = min(len(eval_ds), args.max_samples) if args.max_samples > 0 else len(eval_ds)
    rows: list[dict[str, object]] = []
    total = {key: 0.0 for key in PROTOCOL_B_KEYS if key.startswith("query_")}
    for index in range(limit):
        sample = eval_ds[index]
        video = sample["video"].unsqueeze(0).to(device)
        future_target = video[:, 1:]
        encoded = model.encode_frames(video)
        z_start = encoded[:, 0].expand(candidate_conditions.size(0), -1, -1, -1)
        rollout_latents = model.rollout_from_first(z_start, cond_embed_all, steps=video.size(1) - 1)
        rollout_video = model.decode_latents(rollout_latents, cond_embed_all)
        future_mse = ((rollout_video - future_target.expand_as(rollout_video)) ** 2).mean(dim=(1, 2, 3, 4))
        metrics = protocol_b_selection_metrics(
            future_mse=future_mse,
            true_idx=condition_catalog.index_by_key[condition_tuple_from_tensor(sample["condition"])],
            posterior_logits=logits[index].detach().cpu(),
            alpha=args.query_alpha,
            temperature=args.posterior_temperature,
        )
        for key, value in metrics.items():
            total[key] += float(value)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                **{key: float(value) for key, value in metrics.items()},
            }
        )

    denom = max(limit, 1)
    summary: dict[str, object] = {
        "num_samples": limit,
        "probe_best_val_acc": probe_result.best_val_acc,
        "cond_acc": cond_acc,
        "cond_true_prob": float(cond_dist["cond_true_prob"].item()),
        "cond_true_in90": float(cond_dist["cond_true_in90"].item()),
        "cond_support_ratio": float(cond_dist["cond_support_ratio"].item()),
    }
    for key in total:
        summary[key] = total[key] / denom
    return summary, rows, probe_result.history


def filter_protocol_b_metrics(metrics: dict[str, float]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for key in PROTOCOL_B_KEYS:
        if key in metrics:
            summary[key] = metrics[key]
    summary["num_samples"] = metrics.get("query_samples", 0.0)
    return summary


@torch.no_grad()
def evaluate_mvp_protocol_b(
    *,
    model,
    loader,
    dataset: FolderVideoDataset,
    cfg,
    device: torch.device,
    condition_catalog,
    condition_catalog_tensor: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    total_cond_acc = 0.0
    total_cond_true_prob = 0.0
    total_cond_true_in90 = 0.0
    total_cond_support_ratio = 0.0
    steps = 0
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)
        out = model(video, condition)
        logits, targets = compute_condition_logits(
            model=model,
            latents=out.latents,
            condition=condition,
            condition_catalog=condition_catalog,
            condition_catalog_tensor=condition_catalog_tensor,
            catalog_readout_mode=cfg.train.condition_catalog_readout_mode,
        )
        total_cond_acc += float((logits.argmax(dim=1) == targets).float().mean().item())
        cond_dist = summarize_condition_distribution(
            logits,
            targets,
            alpha=args.query_alpha,
            temperature=args.posterior_temperature,
        )
        total_cond_true_prob += float(cond_dist["cond_true_prob"].item())
        total_cond_true_in90 += float(cond_dist["cond_true_in90"].item())
        total_cond_support_ratio += float(cond_dist["cond_support_ratio"].item())
        steps += 1

    summary: dict[str, object] = {
        "cond_acc": total_cond_acc / max(steps, 1),
        "cond_true_prob": total_cond_true_prob / max(steps, 1),
        "cond_true_in90": total_cond_true_in90 / max(steps, 1),
        "cond_support_ratio": total_cond_support_ratio / max(steps, 1),
    }
    summary.update(
        evaluate_query_responsive_execution(
            model=model,
            dataset=dataset,
            condition_catalog=condition_catalog,
            device=device,
            alpha=args.query_alpha,
            obs_alpha=args.query_alpha,
            plan_core_alpha=cfg.train.query_eval_plan_core_alpha,
            posterior_temperature=args.posterior_temperature,
            max_samples=args.max_samples,
            catalog_readout_mode=cfg.train.condition_catalog_readout_mode,
        )
    )
    summary["num_samples"] = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)
    return summary


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type == "mvp":
        model, cfg, skipped_keys = load_mvp_checkpoint(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=device,
        )
        train_loader, val_loader, train_ds, val_ds = build_dataloaders(cfg)
        if not isinstance(train_ds, FolderVideoDataset) or not isinstance(val_ds, FolderVideoDataset):
            raise ValueError("Protocol-B eval currently supports only folder datasets.")
        condition_catalog = build_condition_catalog(train_ds, val_ds)
        condition_catalog_tensor = condition_catalog.tensor.to(device)
        loader = train_loader if args.split == "train" else val_loader
        dataset = train_ds if args.split == "train" else val_ds
        loader = train_loader if args.split == "train" else val_loader
        summary = evaluate_mvp_protocol_b(
            model=model,
            loader=loader,
            dataset=dataset,
            cfg=cfg,
            device=device,
            condition_catalog=condition_catalog,
            condition_catalog_tensor=condition_catalog_tensor,
            args=args,
        )
        payload = {
            "model_type": args.model_type,
            "config": args.config,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "query_alpha": args.query_alpha,
            "posterior_temperature": args.posterior_temperature,
            "skipped_state_keys": skipped_keys,
            "summary": summary,
        }
    else:
        model, cfg, baseline_cfg = load_baseline_checkpoint(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=device,
        )
        train_loader, val_loader, train_ds, val_ds = build_dataloaders(cfg)
        if not isinstance(train_ds, FolderVideoDataset) or not isinstance(val_ds, FolderVideoDataset):
            raise ValueError("Protocol-B eval currently supports only folder datasets.")
        condition_catalog = build_condition_catalog(train_ds, val_ds)
        eval_ds = train_ds if args.split == "train" else val_ds
        summary, rows, probe_history = evaluate_baseline_protocol_b(
            model=model,
            train_ds=train_ds,
            eval_ds=eval_ds,
            condition_catalog=condition_catalog,
            device=device,
            args=args,
        )
        save_csv(metrics_dir / f"{args.split}_protocol_b_rows.csv", rows)
        with (metrics_dir / "probe_history.jsonl").open("w", encoding="utf-8") as handle:
            for row in probe_history:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        payload = {
            "model_type": args.model_type,
            "config": args.config,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "query_alpha": args.query_alpha,
            "posterior_temperature": args.posterior_temperature,
            "probe_type": args.probe_type,
            "probe_epochs": args.probe_epochs,
            "probe_hidden_dim": args.probe_hidden_dim,
            "baseline": baseline_cfg,
            "summary": summary,
        }

    with (metrics_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    summary = payload["summary"]
    print(f"device={device}")
    print(f"model_type={args.model_type}")
    print(f"split={args.split}")
    for key in (
        "cond_acc",
        "cond_true_prob",
        "cond_true_in90",
        "cond_support_ratio",
        "query_exec_mse",
        "query_support_top1_mse",
        "query_oracle_mse",
        "query_match_true",
        "query_exec_set_size",
    ):
        if key in summary:
            print(f"{key}={float(summary[key]):.6f}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
