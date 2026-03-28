from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vh_mvp.config import load_config
from vh_mvp.data import FolderVideoDataset, build_condition_catalog, condition_tuple_from_tensor
from vh_mvp.models import VideoDynamicsMVP
from vh_mvp.support import build_candidate_posterior, candidate_sets_from_posterior, condition_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate condition-support metrics (entropy/effective support/candidate sharpening)."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-source", type=str, default="both", choices=("train", "val", "both"))
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--alphas", type=str, default="0.5,0.8,0.9,0.95")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def parse_alphas(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        alpha = float(part)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        values.append(alpha)
    if not values:
        raise ValueError("No valid alphas parsed from --alphas")
    return sorted(set(values))


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
        raise ValueError("Support metric eval currently supports only folder datasets.")
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


def gather_condition_catalog(
    train_ds: FolderVideoDataset,
    val_ds: FolderVideoDataset,
    source: str,
) -> tuple[torch.Tensor, list[tuple[int, ...]], list[str]]:
    if source == "train":
        catalog = build_condition_catalog(train_ds)
    elif source == "val":
        catalog = build_condition_catalog(val_ds)
    else:
        catalog = build_condition_catalog(train_ds, val_ds)
    return catalog.tensor, catalog.keys, catalog.texts


def _concat(parts: list[torch.Tensor]) -> torch.Tensor:
    if not parts:
        raise RuntimeError("No samples were processed.")
    return torch.cat(parts, dim=0)


@torch.no_grad()
def evaluate_split(
    *,
    model: VideoDynamicsMVP,
    dataset: FolderVideoDataset,
    candidate_conditions: torch.Tensor,
    candidate_to_idx: dict[tuple[int, ...], int],
    candidate_keys: list[tuple[int, ...]],
    batch_size: int,
    device: torch.device,
    posterior_temperature: float,
    alphas: list[float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    candidate_conditions = candidate_conditions.to(device)
    num_candidates = candidate_conditions.size(0)
    entropy_scale = float(np.log(max(num_candidates, 2)))

    p_true_parts: list[torch.Tensor] = []
    nll_true_parts: list[torch.Tensor] = []
    margin_parts: list[torch.Tensor] = []
    top1_correct_parts: list[torch.Tensor] = []
    top1_prob_parts: list[torch.Tensor] = []
    entropy_parts: list[torch.Tensor] = []
    ess_parts: list[torch.Tensor] = []
    top1_idx_parts: list[torch.Tensor] = []
    true_rank_parts: list[torch.Tensor] = []

    alpha_k_parts: dict[float, list[torch.Tensor]] = {alpha: [] for alpha in alphas}
    alpha_k_ratio_parts: dict[float, list[torch.Tensor]] = {alpha: [] for alpha in alphas}
    alpha_true_in_parts: dict[float, list[torch.Tensor]] = {alpha: [] for alpha in alphas}

    records: list[dict[str, object]] = []

    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        condition = batch["condition"]
        sample_ids = list(batch["sample_id"])
        labels = list(batch["label"])

        latents = model.encode_video(video)
        logits = model.condition_candidate_logits(latents, candidate_conditions)
        posterior = build_candidate_posterior(logits, temperature=posterior_temperature)
        probs = posterior.probs

        true_indices = torch.tensor(
            [candidate_to_idx[condition_tuple_from_tensor(row)] for row in condition],
            dtype=torch.long,
            device=device,
        )
        p_true = probs.gather(1, true_indices.unsqueeze(1)).squeeze(1)

        probs_wo_true = probs.clone()
        probs_wo_true.scatter_(1, true_indices.unsqueeze(1), -1.0)
        p_second = probs_wo_true.max(dim=1).values
        margin_true_second = p_true - p_second

        top1_prob, top1_idx = probs.max(dim=1)
        top1_correct = (top1_idx == true_indices).float()
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        eff_support = torch.exp(entropy)
        nll_true = -p_true.clamp_min(1e-12).log()

        rank_by_class = torch.argsort(posterior.sorted_idx, dim=1)
        true_rank = rank_by_class.gather(1, true_indices.unsqueeze(1)).squeeze(1) + 1

        p_true_parts.append(p_true.cpu())
        nll_true_parts.append(nll_true.cpu())
        margin_parts.append(margin_true_second.cpu())
        top1_correct_parts.append(top1_correct.cpu())
        top1_prob_parts.append(top1_prob.cpu())
        entropy_parts.append(entropy.cpu())
        ess_parts.append(eff_support.cpu())
        top1_idx_parts.append(top1_idx.cpu())
        true_rank_parts.append(true_rank.cpu())

        alpha_sets = candidate_sets_from_posterior(posterior, alphas)
        per_alpha_k: dict[float, torch.Tensor] = {}
        per_alpha_true_in: dict[float, torch.Tensor] = {}
        for alpha in alphas:
            k_alpha = alpha_sets[alpha].k_alpha
            true_in_set = (true_rank <= k_alpha).float()
            alpha_k_parts[alpha].append(k_alpha.cpu())
            alpha_k_ratio_parts[alpha].append((k_alpha.float() / float(num_candidates)).cpu())
            alpha_true_in_parts[alpha].append(true_in_set.cpu())
            per_alpha_k[alpha] = k_alpha
            per_alpha_true_in[alpha] = true_in_set

        for row_idx, sample_id in enumerate(sample_ids):
            true_idx = int(true_indices[row_idx].item())
            pred_idx = int(top1_idx[row_idx].item())
            row = {
                "sample_id": sample_id,
                "label": labels[row_idx],
                "true_condition": condition_key(candidate_keys[true_idx]),
                "pred_condition": condition_key(candidate_keys[pred_idx]),
                "pred_condition_prob": float(top1_prob[row_idx].item()),
                "top1_correct": int(top1_correct[row_idx].item()),
                "p_true": float(p_true[row_idx].item()),
                "nll_true": float(nll_true[row_idx].item()),
                "margin_true_second": float(margin_true_second[row_idx].item()),
                "entropy": float(entropy[row_idx].item()),
                "effective_support_size": float(eff_support[row_idx].item()),
                "true_rank": int(true_rank[row_idx].item()),
            }
            for alpha in alphas:
                alpha_key = f"{alpha:.2f}"
                row[f"k_alpha_{alpha_key}"] = int(per_alpha_k[alpha][row_idx].item())
                row[f"true_in_set_{alpha_key}"] = int(per_alpha_true_in[alpha][row_idx].item())
            records.append(row)

    p_true_all = _concat(p_true_parts)
    nll_true_all = _concat(nll_true_parts)
    margin_all = _concat(margin_parts)
    top1_all = _concat(top1_correct_parts)
    top1_prob_all = _concat(top1_prob_parts)
    entropy_all = _concat(entropy_parts)
    ess_all = _concat(ess_parts)
    true_rank_all = _concat(true_rank_parts)

    summary: dict[str, object] = {
        "num_samples": int(p_true_all.numel()),
        "num_candidates": int(num_candidates),
        "cond_top1": float(top1_all.mean().item()),
        "avg_top1_prob": float(top1_prob_all.mean().item()),
        "avg_p_true": float(p_true_all.mean().item()),
        "avg_nll_true": float(nll_true_all.mean().item()),
        "avg_margin_true_second": float(margin_all.mean().item()),
        "support_entropy": float(entropy_all.mean().item()),
        "support_entropy_norm": float((entropy_all / entropy_scale).mean().item()),
        "effective_support_size": float(ess_all.mean().item()),
        "effective_support_ratio": float((ess_all / float(num_candidates)).mean().item()),
        "avg_true_rank": float(true_rank_all.float().mean().item()),
    }

    alpha_summary: dict[str, dict[str, float]] = {}
    for alpha in alphas:
        alpha_name = f"{alpha:.2f}"
        k_all = _concat(alpha_k_parts[alpha]).float()
        k_ratio_all = _concat(alpha_k_ratio_parts[alpha])
        true_in_all = _concat(alpha_true_in_parts[alpha])
        alpha_summary[alpha_name] = {
            "k_alpha_mean": float(k_all.mean().item()),
            "k_alpha_p50": float(k_all.median().item()),
            "k_alpha_p90": float(torch.quantile(k_all, 0.90).item()),
            "k_alpha_ratio_mean": float(k_ratio_all.mean().item()),
            "sharpness_alpha_mean": float((1.0 - k_ratio_all).mean().item()),
            "true_in_set_alpha": float(true_in_all.mean().item()),
        }
    summary["alphas"] = alpha_summary

    return summary, records


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_condition_catalog(path: Path, keys: list[tuple[int, ...]], texts: list[str]) -> None:
    rows = []
    for idx, (key, text) in enumerate(zip(keys, texts)):
        rows.append({"candidate_idx": idx, "condition": condition_key(key), "condition_text": text})
    save_csv(path, rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    alphas = parse_alphas(args.alphas)

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = build_model_from_config(args.config, args.checkpoint, device)
    train_ds, val_ds = build_real_datasets(cfg)
    candidate_conditions, candidate_keys, candidate_texts = gather_condition_catalog(
        train_ds=train_ds,
        val_ds=val_ds,
        source=args.candidate_source,
    )
    candidate_to_idx = {key: idx for idx, key in enumerate(candidate_keys)}

    train_summary, train_rows = evaluate_split(
        model=model,
        dataset=train_ds,
        candidate_conditions=candidate_conditions,
        candidate_to_idx=candidate_to_idx,
        candidate_keys=candidate_keys,
        batch_size=args.batch_size,
        device=device,
        posterior_temperature=args.posterior_temperature,
        alphas=alphas,
    )
    val_summary, val_rows = evaluate_split(
        model=model,
        dataset=val_ds,
        candidate_conditions=candidate_conditions,
        candidate_to_idx=candidate_to_idx,
        candidate_keys=candidate_keys,
        batch_size=args.batch_size,
        device=device,
        posterior_temperature=args.posterior_temperature,
        alphas=alphas,
    )

    save_csv(metrics_dir / "train_support_predictions.csv", train_rows)
    save_csv(metrics_dir / "val_support_predictions.csv", val_rows)
    save_condition_catalog(metrics_dir / "condition_candidates.csv", candidate_keys, candidate_texts)

    summary = {
        "checkpoint": args.checkpoint,
        "candidate_source": args.candidate_source,
        "posterior_temperature": args.posterior_temperature,
        "alphas": alphas,
        "train": train_summary,
        "val": val_summary,
    }
    with (metrics_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"device={device}")
    print(f"candidates={len(candidate_keys)}")
    print(f"train_cond_top1={train_summary['cond_top1']:.4f}")
    print(f"train_eff_support_ratio={train_summary['effective_support_ratio']:.4f}")
    print(f"train_true_in_set_0.90={train_summary['alphas'].get('0.90', {}).get('true_in_set_alpha', float('nan')):.4f}")
    print(f"val_cond_top1={val_summary['cond_top1']:.4f}")
    print(f"val_eff_support_ratio={val_summary['effective_support_ratio']:.4f}")
    print(f"val_true_in_set_0.90={val_summary['alphas'].get('0.90', {}).get('true_in_set_alpha', float('nan')):.4f}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
