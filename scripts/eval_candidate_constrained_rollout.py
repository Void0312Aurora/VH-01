from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from vh_mvp.config import load_config
from vh_mvp.data import FolderVideoDataset, build_condition_catalog, condition_tuple_from_tensor
from vh_mvp.losses import response_signature_dim
from vh_mvp.models import VideoDynamicsMVP
from vh_mvp.support import build_candidate_posterior, candidate_sets_from_posterior, condition_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the minimal candidate-constrained rollout pipeline built on T_alpha(c)."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--candidate-source", type=str, default="both", choices=("train", "val", "both"))
    parser.add_argument("--alpha", type=float, default=0.90)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=80)
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
        chart_hidden_dim=cfg.model.chart_hidden_dim,
        chart_num_experts=cfg.model.chart_num_experts,
        chart_mode=cfg.model.chart_mode,
        chart_residual_scale=cfg.model.chart_residual_scale,
        chart_temporal_hidden_dim=cfg.model.chart_temporal_hidden_dim,
        chart_temporal_kernel_size=cfg.model.chart_temporal_kernel_size,
        state_cov_proj_dim=cfg.model.state_cov_proj_dim,
        response_signature_dim=response_signature_dim(
            cfg.data.seq_len,
            cfg.model.response_signature_mode,
            channels=cfg.data.channels,
        ),
        response_context_dim=cfg.model.response_context_dim,
        local_measure_hidden_dim=cfg.model.local_measure_hidden_dim,
        local_measure_rank=cfg.model.local_measure_rank,
        local_measure_eps=cfg.model.local_measure_eps,
        local_diffusion_mode=cfg.model.local_diffusion_mode,
        local_diffusion_condition_mode=cfg.model.local_diffusion_condition_mode,
        measure_density_mode=cfg.model.measure_density_mode,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, cfg


def build_real_datasets(cfg) -> tuple[FolderVideoDataset, FolderVideoDataset]:
    if cfg.data.kind != "folder":
        raise ValueError("Candidate rollout eval currently supports only folder datasets.")
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


def gather_catalog(train_ds: FolderVideoDataset, val_ds: FolderVideoDataset, source: str):
    if source == "train":
        return build_condition_catalog(train_ds)
    if source == "val":
        return build_condition_catalog(val_ds)
    return build_condition_catalog(train_ds, val_ds)


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_candidate_rollouts(
    *,
    model: VideoDynamicsMVP,
    dataset: FolderVideoDataset,
    candidate_conditions: torch.Tensor,
    candidate_keys: list[tuple[int, ...]],
    candidate_to_idx: dict[tuple[int, ...], int],
    alpha: float,
    posterior_temperature: float,
    max_samples: int,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    num_candidates = candidate_conditions.size(0)
    candidate_conditions_device = candidate_conditions.to(device)
    sample_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    true_in_set_count = 0
    oracle_best_in_set_count = 0
    total_set_size = 0.0
    total_set_mass = 0.0
    total_true_mse = 0.0
    total_set_best_mse = 0.0
    total_full_best_mse = 0.0
    total_posterior_top1_mse = 0.0
    processed = 0

    for index in range(min(len(dataset), max_samples)):
        sample = dataset[index]
        video = sample["video"].unsqueeze(0).to(device)
        condition = sample["condition"].unsqueeze(0)
        sample_id = str(sample["sample_id"])
        label = str(sample["label"])

        latents = model.encode_video(video)
        logits = model.condition_candidate_logits(latents, candidate_conditions_device)
        posterior = build_candidate_posterior(logits, temperature=posterior_temperature)
        candidate_set = candidate_sets_from_posterior(posterior, [alpha])[alpha]

        member_indices = candidate_set.member_indices()[0]
        set_mass = float(candidate_set.mass()[0].item())
        set_size = int(candidate_set.k_alpha[0].item())

        true_idx = candidate_to_idx[condition_tuple_from_tensor(condition[0])]
        true_in_set = int(true_idx in member_indices)

        z_start = latents[:, 0].expand(num_candidates, -1)
        cond_embed_all = model.condition_encoder(candidate_conditions_device)
        rollout_latents, _ = model.rollout_from(z_start, cond_embed_all, steps=video.size(1) - 1)
        rollout_video = model.decode_video(rollout_latents, cond_embed_all)
        target_future = video[:, 1:].expand(num_candidates, -1, -1, -1, -1)
        future_mse = ((rollout_video - target_future) ** 2).mean(dim=(1, 2, 3, 4))

        full_best_idx = int(future_mse.argmin().item())
        set_future_mse = future_mse[member_indices]
        set_best_local = int(set_future_mse.argmin().item())
        set_best_idx = int(member_indices[set_best_local])
        posterior_top1_idx = int(posterior.top1_idx[0].item())

        true_candidate_mse = float(future_mse[true_idx].item())
        full_best_mse = float(future_mse[full_best_idx].item())
        set_best_mse = float(future_mse[set_best_idx].item())
        posterior_top1_mse = float(future_mse[posterior_top1_idx].item())
        oracle_best_in_set = int(full_best_idx in member_indices)

        total_set_size += set_size
        total_set_mass += set_mass
        total_true_mse += true_candidate_mse
        total_set_best_mse += set_best_mse
        total_full_best_mse += full_best_mse
        total_posterior_top1_mse += posterior_top1_mse
        true_in_set_count += true_in_set
        oracle_best_in_set_count += oracle_best_in_set
        processed += 1

        sample_rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "true_condition": condition_key(candidate_keys[true_idx]),
                "alpha": alpha,
                "set_size": set_size,
                "set_ratio": set_size / float(num_candidates),
                "set_mass": set_mass,
                "true_in_set": true_in_set,
                "oracle_best_in_set": oracle_best_in_set,
                "posterior_top1_condition": condition_key(candidate_keys[posterior_top1_idx]),
                "oracle_best_condition": condition_key(candidate_keys[full_best_idx]),
                "set_best_condition": condition_key(candidate_keys[set_best_idx]),
                "true_candidate_future_mse": true_candidate_mse,
                "posterior_top1_future_mse": posterior_top1_mse,
                "oracle_best_future_mse": full_best_mse,
                "set_best_future_mse": set_best_mse,
                "mse_gap_set_vs_full": set_best_mse - full_best_mse,
            }
        )

        for candidate_idx in range(num_candidates):
            candidate_rows.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "alpha": alpha,
                    "candidate_idx": candidate_idx,
                    "candidate_condition": condition_key(candidate_keys[candidate_idx]),
                    "posterior_prob": float(posterior.probs[0, candidate_idx].item()),
                    "in_set": int(candidate_idx in member_indices),
                    "is_true": int(candidate_idx == true_idx),
                    "is_posterior_top1": int(candidate_idx == posterior_top1_idx),
                    "is_oracle_best": int(candidate_idx == full_best_idx),
                    "is_set_best": int(candidate_idx == set_best_idx),
                    "future_mse": float(future_mse[candidate_idx].item()),
                }
            )

    summary = {
        "num_samples": processed,
        "num_candidates": num_candidates,
        "alpha": alpha,
        "avg_set_size": total_set_size / max(processed, 1),
        "avg_set_ratio": (total_set_size / max(processed, 1)) / float(num_candidates),
        "avg_set_mass": total_set_mass / max(processed, 1),
        "true_in_set_rate": true_in_set_count / max(processed, 1),
        "oracle_best_in_set_rate": oracle_best_in_set_count / max(processed, 1),
        "avg_true_candidate_future_mse": total_true_mse / max(processed, 1),
        "avg_posterior_top1_future_mse": total_posterior_top1_mse / max(processed, 1),
        "avg_set_best_future_mse": total_set_best_mse / max(processed, 1),
        "avg_full_oracle_best_future_mse": total_full_best_mse / max(processed, 1),
        "avg_mse_gap_set_vs_full": (total_set_best_mse - total_full_best_mse) / max(processed, 1),
    }
    return summary, sample_rows, candidate_rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, cfg = build_model_from_config(args.config, args.checkpoint, device)
    train_ds, val_ds = build_real_datasets(cfg)
    dataset = train_ds if args.split == "train" else val_ds
    catalog = gather_catalog(train_ds, val_ds, args.candidate_source)
    candidate_conditions = catalog.tensor
    candidate_keys = catalog.keys
    candidate_to_idx = catalog.index_by_key

    summary, sample_rows, candidate_rows = evaluate_candidate_rollouts(
        model=model,
        dataset=dataset,
        candidate_conditions=candidate_conditions,
        candidate_keys=candidate_keys,
        candidate_to_idx=candidate_to_idx,
        alpha=args.alpha,
        posterior_temperature=args.posterior_temperature,
        max_samples=args.max_samples,
        device=device,
    )

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_csv(metrics_dir / f"{args.split}_candidate_rollout_summary.csv", sample_rows)
    save_csv(metrics_dir / f"{args.split}_candidate_rollout_candidates.csv", candidate_rows)
    (metrics_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "split": args.split,
                "candidate_source": args.candidate_source,
                "posterior_temperature": args.posterior_temperature,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"device={device}")
    print(f"split={args.split}")
    print(f"alpha={args.alpha:.2f}")
    print(f"samples={summary['num_samples']}")
    print(f"avg_set_size={summary['avg_set_size']:.4f}")
    print(f"true_in_set_rate={summary['true_in_set_rate']:.4f}")
    print(f"oracle_best_in_set_rate={summary['oracle_best_in_set_rate']:.4f}")
    print(f"avg_set_best_future_mse={summary['avg_set_best_future_mse']:.6f}")
    print(f"avg_full_oracle_best_future_mse={summary['avg_full_oracle_best_future_mse']:.6f}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
