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
        description="Evaluate how candidate sets and selected rollouts migrate under condition perturbations."
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
        raise ValueError("Condition perturbation eval currently supports only folder datasets.")
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


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / float(len(union))


def pick_perturbation_indices(neighbors: list[tuple[int, int]]) -> tuple[int | None, int | None]:
    near_idx = None
    far_idx = None
    if neighbors:
        near_idx = neighbors[0][0]
        far_idx = neighbors[-1][0]
    return near_idx, far_idx


@torch.no_grad()
def evaluate_condition_perturbations(
    *,
    model: VideoDynamicsMVP,
    dataset: FolderVideoDataset,
    catalog,
    alpha: float,
    posterior_temperature: float,
    max_samples: int,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidate_conditions = catalog.tensor.to(device)
    cond_embed_all = model.condition_encoder(candidate_conditions)
    num_candidates = candidate_conditions.size(0)

    rows: list[dict[str, object]] = []

    near_jaccard_sum = 0.0
    near_top1_switch = 0
    near_true_exit = 0
    near_selected_mse_shift = 0.0
    near_set_best_mse_shift = 0.0
    near_count = 0

    far_jaccard_sum = 0.0
    far_top1_switch = 0
    far_true_exit = 0
    far_selected_mse_shift = 0.0
    far_set_best_mse_shift = 0.0
    far_count = 0

    processed = 0

    for index in range(min(len(dataset), max_samples)):
        sample = dataset[index]
        video = sample["video"].unsqueeze(0).to(device)
        true_condition = sample["condition"].unsqueeze(0)
        sample_id = str(sample["sample_id"])
        label = str(sample["label"])

        latents = model.encode_video(video)
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

        per_query: dict[str, dict[str, object]] = {}
        for query_name, query_idx in query_specs:
            query_embed = cond_embed_all[query_idx : query_idx + 1].expand(num_candidates, -1)
            support_logits = -model.condition_alignment_energy(rollout_latents, query_embed).unsqueeze(0)
            posterior = build_candidate_posterior(support_logits, temperature=posterior_temperature)
            candidate_set = candidate_sets_from_posterior(posterior, [alpha])[alpha]
            member_indices = candidate_set.member_indices()[0]
            member_set = set(member_indices)
            selected_idx = int(posterior.top1_idx[0].item())
            set_best_idx = min(member_indices, key=lambda idx: float(future_mse[idx].item()))

            per_query[query_name] = {
                "query_idx": query_idx,
                "member_set": member_set,
                "set_size": int(candidate_set.k_alpha[0].item()),
                "selected_idx": selected_idx,
                "selected_mse": float(future_mse[selected_idx].item()),
                "set_best_idx": int(set_best_idx),
                "set_best_mse": float(future_mse[set_best_idx].item()),
                "true_in_set": int(true_idx in member_set),
            }

        true_query = per_query["true"]
        for perturb_name in ("near", "far"):
            if perturb_name not in per_query:
                continue
            pert_query = per_query[perturb_name]
            overlap = jaccard(true_query["member_set"], pert_query["member_set"])
            top1_switch = int(true_query["selected_idx"] != pert_query["selected_idx"])
            true_exit = int(pert_query["true_in_set"] == 0)
            selected_mse_shift = pert_query["selected_mse"] - true_query["selected_mse"]
            set_best_mse_shift = pert_query["set_best_mse"] - true_query["set_best_mse"]

            rows.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "alpha": alpha,
                    "perturbation": perturb_name,
                    "true_condition": condition_key(catalog.keys[true_idx]),
                    "perturbed_condition": condition_key(catalog.keys[int(pert_query['query_idx'])]),
                    "true_set_size": int(true_query["set_size"]),
                    "perturbed_set_size": int(pert_query["set_size"]),
                    "set_jaccard": overlap,
                    "top1_switch": top1_switch,
                    "true_exit": true_exit,
                    "true_selected_condition": condition_key(catalog.keys[int(true_query["selected_idx"])]),
                    "perturbed_selected_condition": condition_key(catalog.keys[int(pert_query["selected_idx"])]),
                    "true_selected_mse": float(true_query["selected_mse"]),
                    "perturbed_selected_mse": float(pert_query["selected_mse"]),
                    "selected_mse_shift": float(selected_mse_shift),
                    "true_set_best_mse": float(true_query["set_best_mse"]),
                    "perturbed_set_best_mse": float(pert_query["set_best_mse"]),
                    "set_best_mse_shift": float(set_best_mse_shift),
                }
            )

            if perturb_name == "near":
                near_jaccard_sum += overlap
                near_top1_switch += top1_switch
                near_true_exit += true_exit
                near_selected_mse_shift += selected_mse_shift
                near_set_best_mse_shift += set_best_mse_shift
                near_count += 1
            else:
                far_jaccard_sum += overlap
                far_top1_switch += top1_switch
                far_true_exit += true_exit
                far_selected_mse_shift += selected_mse_shift
                far_set_best_mse_shift += set_best_mse_shift
                far_count += 1

        processed += 1

    summary = {
        "num_samples": processed,
        "alpha": alpha,
        "near": {
            "count": near_count,
            "avg_set_jaccard": near_jaccard_sum / max(near_count, 1),
            "top1_switch_rate": near_top1_switch / max(near_count, 1),
            "true_exit_rate": near_true_exit / max(near_count, 1),
            "avg_selected_mse_shift": near_selected_mse_shift / max(near_count, 1),
            "avg_set_best_mse_shift": near_set_best_mse_shift / max(near_count, 1),
        },
        "far": {
            "count": far_count,
            "avg_set_jaccard": far_jaccard_sum / max(far_count, 1),
            "top1_switch_rate": far_top1_switch / max(far_count, 1),
            "true_exit_rate": far_true_exit / max(far_count, 1),
            "avg_selected_mse_shift": far_selected_mse_shift / max(far_count, 1),
            "avg_set_best_mse_shift": far_set_best_mse_shift / max(far_count, 1),
        },
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

    summary, rows = evaluate_condition_perturbations(
        model=model,
        dataset=dataset,
        catalog=catalog,
        alpha=args.alpha,
        posterior_temperature=args.posterior_temperature,
        max_samples=args.max_samples,
        device=device,
    )

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_csv(metrics_dir / f"{args.split}_condition_perturbations.csv", rows)
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
    print(f"near_avg_set_jaccard={summary['near']['avg_set_jaccard']:.4f}")
    print(f"near_top1_switch_rate={summary['near']['top1_switch_rate']:.4f}")
    print(f"near_true_exit_rate={summary['near']['true_exit_rate']:.4f}")
    print(f"far_avg_set_jaccard={summary['far']['avg_set_jaccard']:.4f}")
    print(f"far_top1_switch_rate={summary['far']['top1_switch_rate']:.4f}")
    print(f"far_true_exit_rate={summary['far']['true_exit_rate']:.4f}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
