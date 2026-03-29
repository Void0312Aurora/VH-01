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
        description="Study non-oracle candidate selection rules that combine observation and planning posteriors."
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
        raise ValueError("Candidate selection study currently supports only folder datasets.")
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


def masked_argmax(scores: torch.Tensor, mask: torch.Tensor) -> int | None:
    if not bool(mask.any().item()):
        return None
    return int(scores.masked_fill(~mask, float("-inf")).argmax().item())


def mean_bucket(bucket: dict[str, float]) -> dict[str, float]:
    count = max(bucket.get("count", 0.0), 1.0)
    return {
        key: (value / count if key != "count" else value)
        for key, value in bucket.items()
    }


@torch.no_grad()
def evaluate_selection_rules(
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
    query_buckets: dict[str, dict[str, float]] = {}
    rule_names = (
        "direct_query",
        "obs_top1",
        "plan_top1",
        "query_responsive_top1",
        "obs_on_plan_set_top1",
        "joint_plan_set_top1",
        "joint_all_top1",
        "joint_union_top1",
        "joint_intersection_top1",
        "plan_set_best",
        "full_oracle",
    )

    for mode in ("true", "near", "far"):
        query_buckets[mode] = {
            "count": 0.0,
            "avg_obs_set_size": 0.0,
            "avg_plan_set_size": 0.0,
            "avg_intersection_size": 0.0,
            "avg_union_size": 0.0,
            "intersection_empty_rate": 0.0,
            "obs_plan_jaccard": 0.0,
        }
        for rule_name in rule_names:
            query_buckets[mode][f"{rule_name}_mse"] = 0.0
            query_buckets[mode][f"{rule_name}_gap_to_oracle"] = 0.0

    for index in range(min(len(dataset), max_samples)):
        sample = dataset[index]
        video = sample["video"].unsqueeze(0).to(device)
        true_condition = sample["condition"].unsqueeze(0)
        sample_id = str(sample["sample_id"])
        label = str(sample["label"])

        latents = model.encode_video(video)
        obs_logits = model.condition_candidate_logits(latents, candidate_conditions)
        obs_posterior = build_candidate_posterior(obs_logits, temperature=posterior_temperature)
        obs_set = candidate_sets_from_posterior(obs_posterior, [alpha])[alpha]
        obs_mask = obs_set.mask[0]
        obs_set_members = obs_set.member_indices()[0]
        obs_log_probs = obs_posterior.probs[0].clamp_min(1e-12).log()
        obs_top1_idx = int(obs_posterior.top1_idx[0].item())

        z_start = latents[:, 0].expand(num_candidates, -1)
        rollout_latents, _ = model.rollout_from(z_start, cond_embed_all, steps=video.size(1) - 1)
        rollout_video = model.decode_video(rollout_latents, cond_embed_all)
        target_future = video[:, 1:].expand(num_candidates, -1, -1, -1, -1)
        future_mse = ((rollout_video - target_future) ** 2).mean(dim=(1, 2, 3, 4))
        full_oracle_idx = int(future_mse.argmin().item())
        full_oracle_mse = float(future_mse[full_oracle_idx].item())

        true_idx = catalog.index_by_key[condition_tuple_from_tensor(true_condition[0])]
        near_idx, far_idx = pick_perturbation_indices(catalog.neighbors[true_idx])

        query_specs = [("true", true_idx)]
        if near_idx is not None:
            query_specs.append(("near", near_idx))
        if far_idx is not None and far_idx != near_idx:
            query_specs.append(("far", far_idx))

        for query_name, query_idx in query_specs:
            query_embed = cond_embed_all[query_idx : query_idx + 1].expand(num_candidates, -1)
            plan_logits = -model.condition_alignment_energy(rollout_latents, query_embed).unsqueeze(0)
            plan_posterior = build_candidate_posterior(plan_logits, temperature=posterior_temperature)
            plan_set = candidate_sets_from_posterior(plan_posterior, [alpha])[alpha]
            plan_mask = plan_set.mask[0]
            plan_members = plan_set.member_indices()[0]
            plan_log_probs = plan_posterior.probs[0].clamp_min(1e-12).log()
            plan_top1_idx = int(plan_posterior.top1_idx[0].item())
            qr_selection = query_responsive_selection(
                obs_posterior=obs_posterior,
                plan_posterior=plan_posterior,
                obs_alpha=obs_alpha,
                plan_core_alpha=plan_core_alpha,
            )
            query_responsive_top1_idx = int(qr_selection.selected_idx[0].item())

            union_mask = obs_mask | plan_mask
            intersection_mask = obs_mask & plan_mask
            union_members = torch.nonzero(union_mask, as_tuple=False).squeeze(1).tolist()
            intersection_members = torch.nonzero(intersection_mask, as_tuple=False).squeeze(1).tolist()
            intersection_empty = int(len(intersection_members) == 0)
            intersection_size = len(intersection_members)
            union_size = len(union_members)
            jaccard = 1.0 if union_size == 0 else intersection_size / float(union_size)

            joint_scores = obs_log_probs + plan_log_probs
            obs_on_plan_set_top1_idx = masked_argmax(obs_log_probs, plan_mask)
            joint_plan_set_top1_idx = masked_argmax(joint_scores, plan_mask)
            joint_all_top1_idx = int(joint_scores.argmax().item())
            joint_union_top1_idx = masked_argmax(joint_scores, union_mask)
            joint_intersection_top1_idx = masked_argmax(joint_scores, intersection_mask)
            if joint_union_top1_idx is None:
                raise RuntimeError("Union candidate set should never be empty.")
            if obs_on_plan_set_top1_idx is None or joint_plan_set_top1_idx is None:
                raise RuntimeError("Plan candidate set should never be empty.")
            if joint_intersection_top1_idx is None:
                joint_intersection_top1_idx = joint_union_top1_idx

            plan_set_best_idx = min(plan_members, key=lambda idx: float(future_mse[idx].item()))

            rule_to_idx = {
                "direct_query": int(query_idx),
                "obs_top1": obs_top1_idx,
                "plan_top1": plan_top1_idx,
                "query_responsive_top1": query_responsive_top1_idx,
                "obs_on_plan_set_top1": obs_on_plan_set_top1_idx,
                "joint_plan_set_top1": joint_plan_set_top1_idx,
                "joint_all_top1": joint_all_top1_idx,
                "joint_union_top1": joint_union_top1_idx,
                "joint_intersection_top1": joint_intersection_top1_idx,
                "plan_set_best": int(plan_set_best_idx),
                "full_oracle": full_oracle_idx,
            }

            row = {
                "sample_id": sample_id,
                "label": label,
                "query_type": query_name,
                "alpha": alpha,
                "true_condition": condition_key(catalog.keys[true_idx]),
                "query_condition": condition_key(catalog.keys[query_idx]),
                "obs_set_size": len(obs_set_members),
                "plan_set_size": len(plan_members),
                "intersection_size": intersection_size,
                "union_size": union_size,
                "intersection_empty": intersection_empty,
                "obs_plan_jaccard": jaccard,
            }
            for rule_name, selected_idx in rule_to_idx.items():
                mse = float(future_mse[selected_idx].item())
                row[f"{rule_name}_condition"] = condition_key(catalog.keys[selected_idx])
                row[f"{rule_name}_mse"] = mse
                row[f"{rule_name}_gap_to_oracle"] = mse - full_oracle_mse
                query_buckets[query_name][f"{rule_name}_mse"] += mse
                query_buckets[query_name][f"{rule_name}_gap_to_oracle"] += mse - full_oracle_mse

            query_buckets[query_name]["count"] += 1.0
            query_buckets[query_name]["avg_obs_set_size"] += len(obs_set_members)
            query_buckets[query_name]["avg_plan_set_size"] += len(plan_members)
            query_buckets[query_name]["avg_intersection_size"] += intersection_size
            query_buckets[query_name]["avg_union_size"] += union_size
            query_buckets[query_name]["intersection_empty_rate"] += intersection_empty
            query_buckets[query_name]["obs_plan_jaccard"] += jaccard
            rows.append(row)

    summary = {
        "num_samples": min(len(dataset), max_samples),
        "alpha": alpha,
        "rules": rule_names,
        "query_modes": {mode: mean_bucket(bucket) for mode, bucket in query_buckets.items()},
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

    summary, rows = evaluate_selection_rules(
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
    save_csv(metrics_dir / f"{args.split}_candidate_selection_rules.csv", rows)
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
            f"{mode}_direct={mode_summary['direct_query_mse']:.6f} "
            f"{mode}_obs={mode_summary['obs_top1_mse']:.6f} "
            f"{mode}_plan={mode_summary['plan_top1_mse']:.6f} "
            f"{mode}_query_responsive={mode_summary['query_responsive_top1_mse']:.6f} "
            f"{mode}_obs_plan_set={mode_summary['obs_on_plan_set_top1_mse']:.6f} "
            f"{mode}_joint_plan_set={mode_summary['joint_plan_set_top1_mse']:.6f} "
            f"{mode}_joint_union={mode_summary['joint_union_top1_mse']:.6f} "
            f"{mode}_joint_inter={mode_summary['joint_intersection_top1_mse']:.6f} "
            f"{mode}_set_best={mode_summary['plan_set_best_mse']:.6f}"
        )
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
