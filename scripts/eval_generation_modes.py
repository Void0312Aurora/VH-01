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
from vh_mvp.support import (
    build_candidate_posterior,
    candidate_sets_from_posterior,
    condition_key,
    query_responsive_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct-query, support-top1, candidate-set-constrained, and full-oracle generation modes."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--candidate-source", type=str, default="both", choices=("train", "val", "both"))
    parser.add_argument("--alpha", type=float, default=0.90)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--obs-alpha", type=float, default=0.90)
    parser.add_argument("--plan-core-alpha", type=float, default=0.50)
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
        response_signature_dim=response_signature_dim(cfg.data.seq_len, cfg.model.response_signature_mode),
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
        raise ValueError("Generation mode eval currently supports only folder datasets.")
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


def pick_perturbation_indices(neighbors: list[tuple[int, int]]) -> tuple[int | None, int | None]:
    near_idx = None
    far_idx = None
    if neighbors:
        near_idx = neighbors[0][0]
        far_idx = neighbors[-1][0]
    return near_idx, far_idx


def update_stats(bucket: dict[str, float], values: dict[str, float]) -> None:
    bucket["count"] += 1.0
    for key, value in values.items():
        bucket[key] += float(value)


def finalize_stats(bucket: dict[str, float]) -> dict[str, float]:
    count = max(bucket.get("count", 0.0), 1.0)
    return {
        key: (value / count if key != "count" else value)
        for key, value in bucket.items()
    }


@torch.no_grad()
def evaluate_generation_modes(
    *,
    model: VideoDynamicsMVP,
    dataset: FolderVideoDataset,
    catalog,
    alpha: float,
    posterior_temperature: float,
    obs_alpha: float,
    plan_core_alpha: float,
    max_samples: int,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidate_conditions = catalog.tensor.to(device)
    cond_embed_all = model.condition_encoder(candidate_conditions)
    num_candidates = candidate_conditions.size(0)

    rows: list[dict[str, object]] = []
    summary_buckets = {
        "true": {"count": 0.0, "avg_set_size": 0.0, "avg_exec_set_size": 0.0, "query_responsive_fallback_rate": 0.0, "avg_direct_query_mse": 0.0, "avg_support_top1_mse": 0.0, "avg_query_responsive_mse": 0.0, "avg_set_best_mse": 0.0, "avg_full_oracle_mse": 0.0, "avg_gain_support_vs_direct": 0.0, "avg_gain_query_responsive_vs_direct": 0.0, "avg_gain_set_vs_direct": 0.0, "avg_gap_query_responsive_to_full": 0.0, "avg_gap_set_to_full": 0.0},
        "near": {"count": 0.0, "avg_set_size": 0.0, "avg_exec_set_size": 0.0, "query_responsive_fallback_rate": 0.0, "avg_direct_query_mse": 0.0, "avg_support_top1_mse": 0.0, "avg_query_responsive_mse": 0.0, "avg_set_best_mse": 0.0, "avg_full_oracle_mse": 0.0, "avg_gain_support_vs_direct": 0.0, "avg_gain_query_responsive_vs_direct": 0.0, "avg_gain_set_vs_direct": 0.0, "avg_gap_query_responsive_to_full": 0.0, "avg_gap_set_to_full": 0.0},
        "far": {"count": 0.0, "avg_set_size": 0.0, "avg_exec_set_size": 0.0, "query_responsive_fallback_rate": 0.0, "avg_direct_query_mse": 0.0, "avg_support_top1_mse": 0.0, "avg_query_responsive_mse": 0.0, "avg_set_best_mse": 0.0, "avg_full_oracle_mse": 0.0, "avg_gain_support_vs_direct": 0.0, "avg_gain_query_responsive_vs_direct": 0.0, "avg_gain_set_vs_direct": 0.0, "avg_gap_query_responsive_to_full": 0.0, "avg_gap_set_to_full": 0.0},
    }

    for index in range(min(len(dataset), max_samples)):
        sample = dataset[index]
        video = sample["video"].unsqueeze(0).to(device)
        true_condition = sample["condition"].unsqueeze(0)
        sample_id = str(sample["sample_id"])
        label = str(sample["label"])

        latents = model.encode_video(video)
        obs_logits = model.condition_candidate_logits(latents, candidate_conditions)
        obs_posterior = build_candidate_posterior(obs_logits, temperature=posterior_temperature)
        z_start = latents[:, 0].expand(num_candidates, -1)
        rollout_latents, _ = model.rollout_from(z_start, cond_embed_all, steps=video.size(1) - 1)
        rollout_video = model.decode_video(rollout_latents, cond_embed_all)
        target_future = video[:, 1:].expand(num_candidates, -1, -1, -1, -1)
        future_mse = ((rollout_video - target_future) ** 2).mean(dim=(1, 2, 3, 4))

        true_idx = catalog.index_by_key[condition_tuple_from_tensor(true_condition[0])]
        near_idx, far_idx = pick_perturbation_indices(catalog.neighbors[true_idx])

        query_specs = [("true", true_idx)]
        if near_idx is not None:
            query_specs.append(("near", near_idx))
        if far_idx is not None and far_idx != near_idx:
            query_specs.append(("far", far_idx))

        full_oracle_idx = int(future_mse.argmin().item())
        full_oracle_mse = float(future_mse[full_oracle_idx].item())

        for query_name, query_idx in query_specs:
            query_embed = cond_embed_all[query_idx : query_idx + 1].expand(num_candidates, -1)
            support_logits = -model.condition_alignment_energy(rollout_latents, query_embed).unsqueeze(0)
            posterior = build_candidate_posterior(support_logits, temperature=posterior_temperature)
            candidate_set = candidate_sets_from_posterior(posterior, [alpha])[alpha]
            qr_selection = query_responsive_selection(
                obs_posterior=obs_posterior,
                plan_posterior=posterior,
                obs_alpha=obs_alpha,
                plan_core_alpha=plan_core_alpha,
            )
            member_indices = candidate_set.member_indices()[0]

            direct_query_idx = int(query_idx)
            support_top1_idx = int(posterior.top1_idx[0].item())
            query_responsive_idx = int(qr_selection.selected_idx[0].item())
            set_best_idx = min(member_indices, key=lambda idx: float(future_mse[idx].item()))

            direct_query_mse = float(future_mse[direct_query_idx].item())
            support_top1_mse = float(future_mse[support_top1_idx].item())
            query_responsive_mse = float(future_mse[query_responsive_idx].item())
            set_best_mse = float(future_mse[set_best_idx].item())
            set_size = int(candidate_set.k_alpha[0].item())
            exec_set_size = len(qr_selection.member_indices()[0])
            qr_fallback = int(qr_selection.used_plan_core_fallback[0].item())

            rows.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "query_type": query_name,
                    "alpha": alpha,
                    "query_condition": condition_key(catalog.keys[query_idx]),
                    "direct_query_condition": condition_key(catalog.keys[direct_query_idx]),
                    "support_top1_condition": condition_key(catalog.keys[support_top1_idx]),
                    "query_responsive_condition": condition_key(catalog.keys[query_responsive_idx]),
                    "set_best_condition": condition_key(catalog.keys[set_best_idx]),
                    "full_oracle_condition": condition_key(catalog.keys[full_oracle_idx]),
                    "set_size": set_size,
                    "exec_set_size": exec_set_size,
                    "query_responsive_fallback": qr_fallback,
                    "direct_query_mse": direct_query_mse,
                    "support_top1_mse": support_top1_mse,
                    "query_responsive_mse": query_responsive_mse,
                    "set_best_mse": set_best_mse,
                    "full_oracle_mse": full_oracle_mse,
                    "gain_support_vs_direct": direct_query_mse - support_top1_mse,
                    "gain_query_responsive_vs_direct": direct_query_mse - query_responsive_mse,
                    "gain_set_vs_direct": direct_query_mse - set_best_mse,
                    "gap_query_responsive_to_full": query_responsive_mse - full_oracle_mse,
                    "gap_set_to_full": set_best_mse - full_oracle_mse,
                }
            )

            update_stats(
                summary_buckets[query_name],
                {
                    "avg_set_size": set_size,
                    "avg_exec_set_size": exec_set_size,
                    "query_responsive_fallback_rate": qr_fallback,
                    "avg_direct_query_mse": direct_query_mse,
                    "avg_support_top1_mse": support_top1_mse,
                    "avg_query_responsive_mse": query_responsive_mse,
                    "avg_set_best_mse": set_best_mse,
                    "avg_full_oracle_mse": full_oracle_mse,
                    "avg_gain_support_vs_direct": direct_query_mse - support_top1_mse,
                    "avg_gain_query_responsive_vs_direct": direct_query_mse - query_responsive_mse,
                    "avg_gain_set_vs_direct": direct_query_mse - set_best_mse,
                    "avg_gap_query_responsive_to_full": query_responsive_mse - full_oracle_mse,
                    "avg_gap_set_to_full": set_best_mse - full_oracle_mse,
                },
            )

    summary = {
        "num_samples": min(len(dataset), max_samples),
        "alpha": alpha,
        "query_modes": {name: finalize_stats(bucket) for name, bucket in summary_buckets.items()},
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, cfg = build_model_from_config(args.config, args.checkpoint, device)
    train_ds, val_ds = build_real_datasets(cfg)
    dataset = train_ds if args.split == "train" else val_ds
    catalog = gather_catalog(train_ds, val_ds, args.candidate_source)

    summary, rows = evaluate_generation_modes(
        model=model,
        dataset=dataset,
        catalog=catalog,
        alpha=args.alpha,
        posterior_temperature=args.posterior_temperature,
        obs_alpha=args.obs_alpha,
        plan_core_alpha=args.plan_core_alpha,
        max_samples=args.max_samples,
        device=device,
    )

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_csv(metrics_dir / f"{args.split}_generation_modes.csv", rows)
    (metrics_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "split": args.split,
                "candidate_source": args.candidate_source,
                "posterior_temperature": args.posterior_temperature,
                "obs_alpha": args.obs_alpha,
                "plan_core_alpha": args.plan_core_alpha,
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
    for mode in ("true", "near", "far"):
        mode_summary = summary["query_modes"][mode]
        print(
            f"{mode}_direct={mode_summary['avg_direct_query_mse']:.6f} "
            f"{mode}_top1={mode_summary['avg_support_top1_mse']:.6f} "
            f"{mode}_query_responsive={mode_summary['avg_query_responsive_mse']:.6f} "
            f"{mode}_set={mode_summary['avg_set_best_mse']:.6f} "
            f"{mode}_oracle={mode_summary['avg_full_oracle_mse']:.6f}"
        )
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
