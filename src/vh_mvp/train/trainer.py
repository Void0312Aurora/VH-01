from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vh_mvp.config import AppConfig, load_config
from vh_mvp.data import (
    ConditionCatalog,
    FolderVideoDataset,
    SyntheticVideoDataset,
    build_condition_catalog,
    condition_tuple_from_tensor,
    format_condition_tensor,
    sample_negative_conditions,
)
from vh_mvp.losses import (
    append_geometry_neighborhood_reference,
    append_smoothness_neighborhood_reference,
    build_geometry_neighborhood_reference,
    build_smoothness_neighborhood_reference,
    classification_accuracy,
    compute_stage_weights,
    dynamics_loss,
    GeometryNeighborhoodReference,
    gap_loss,
    latent_representation_loss,
    local_measure_loss,
    local_linearity_loss,
    nce_condition_loss,
    prototype_alignment_loss,
    prototype_separation_loss,
    reconstruction_loss,
    regularization_loss,
    response_signature_dim,
    SmoothnessNeighborhoodReference,
    support_refinement_loss,
)
from vh_mvp.models import VideoDynamicsMVP
from vh_mvp.support import (
    build_condition_inference_posterior,
    query_measure_execution,
    summarize_condition_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mvp.yaml")
    parser.add_argument("--max-train-steps", type=int, default=0)
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


def build_measure_target_teacher(
    model: VideoDynamicsMVP,
    decay: float,
) -> VideoDynamicsMVP | None:
    if decay <= 0.0:
        return None
    teacher = deepcopy(model)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_measure_target_teacher(
    teacher: VideoDynamicsMVP | None,
    student: VideoDynamicsMVP,
    decay: float,
) -> None:
    if teacher is None or decay <= 0.0:
        return
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters(), strict=True):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers(), strict=True):
        teacher_buffer.data.copy_(student_buffer.data)
    teacher.eval()


def scheduled_scale(epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch - start_epoch + 1) / float(warmup_epochs))


def build_dataloaders(cfg: AppConfig) -> tuple[DataLoader, DataLoader, object, object]:
    if cfg.data.kind == "synthetic":
        train_ds = SyntheticVideoDataset(
            size=cfg.data.train_size,
            seq_len=cfg.data.seq_len,
            image_size=cfg.data.image_size,
            seed=cfg.seed,
            synthetic_mode=cfg.data.synthetic_mode,
        )
        val_ds = SyntheticVideoDataset(
            size=cfg.data.val_size,
            seq_len=cfg.data.seq_len,
            image_size=cfg.data.image_size,
            seed=cfg.seed + 100_000,
            synthetic_mode=cfg.data.synthetic_mode,
        )
    elif cfg.data.kind == "folder":
        train_ds = FolderVideoDataset(
            root=cfg.data.root,
            manifest_path=cfg.data.manifest_path,
            seq_len=cfg.data.seq_len,
            image_size=cfg.data.image_size,
        )
        val_manifest = cfg.data.val_manifest_path or cfg.data.manifest_path
        val_ds = FolderVideoDataset(
            root=cfg.data.root,
            manifest_path=val_manifest,
            seq_len=cfg.data.seq_len,
            image_size=cfg.data.image_size,
        )
        shared_labels = sorted(
            {
                sample.label
                for dataset in (train_ds, val_ds)
                for sample in dataset.samples
                if sample.label
            }
        )
        if shared_labels:
            shared_label_to_idx = {label: idx for idx, label in enumerate(shared_labels)}
            train_ds.label_to_idx = shared_label_to_idx
            val_ds.label_to_idx = shared_label_to_idx
    else:
        raise ValueError(f"Unsupported data.kind: {cfg.data.kind}")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds, val_ds


def build_model(cfg: AppConfig, device: torch.device) -> VideoDynamicsMVP:
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
        encoder_condition_mode=cfg.model.encoder_condition_mode,
        encoder_condition_hidden_dim=cfg.model.encoder_condition_hidden_dim,
        encoder_condition_scale=cfg.model.encoder_condition_scale,
        state_cov_proj_dim=cfg.model.state_cov_proj_dim,
        response_signature_dim=response_signature_dim(
            cfg.data.seq_len,
            cfg.model.response_signature_mode,
            channels=cfg.data.channels,
        ),
        response_context_dim=cfg.model.response_context_dim,
        tangent_dim=cfg.model.tangent_dim,
        local_measure_hidden_dim=cfg.model.local_measure_hidden_dim,
        local_measure_rank=cfg.model.local_measure_rank,
        local_measure_eps=cfg.model.local_measure_eps,
        local_diffusion_mode=cfg.model.local_diffusion_mode,
        local_diffusion_geometry_mode=cfg.model.local_diffusion_geometry_mode,
        local_diffusion_condition_mode=cfg.model.local_diffusion_condition_mode,
        measure_density_mode=cfg.model.measure_density_mode,
    )
    return model.to(device)


def build_condition_targets(
    condition: torch.Tensor,
    condition_catalog: ConditionCatalog | None,
) -> torch.Tensor | None:
    if condition_catalog is None:
        return None
    indices = [condition_catalog.index_by_key[condition_tuple_from_tensor(row)] for row in condition]
    return torch.tensor(indices, dtype=torch.long, device=condition.device)


def build_condition_label_targets(
    condition: torch.Tensor,
    condition_catalog: ConditionCatalog | None,
) -> torch.Tensor | None:
    if condition_catalog is None:
        return None
    label_indices: list[int] = []
    for row in condition:
        catalog_idx = condition_catalog.index_by_key[condition_tuple_from_tensor(row)]
        label_idx = condition_catalog.label_indices[catalog_idx]
        if label_idx < 0:
            return None
        label_indices.append(label_idx)
    return torch.tensor(label_indices, dtype=torch.long, device=condition.device)


def build_condition_catalog_label_tensor(
    condition_catalog: ConditionCatalog | None,
    device: torch.device,
) -> torch.Tensor | None:
    if condition_catalog is None or not condition_catalog.label_indices:
        return None
    if any(label_idx < 0 for label_idx in condition_catalog.label_indices):
        return None
    return torch.tensor(condition_catalog.label_indices, dtype=torch.long, device=device)


def compute_condition_catalog_logits(
    *,
    model: VideoDynamicsMVP,
    latents: torch.Tensor,
    condition_catalog: ConditionCatalog,
    condition_catalog_tensor: torch.Tensor | None,
    catalog_readout_mode: str,
) -> torch.Tensor:
    if catalog_readout_mode == "semantic_prototype" and model.semantic_prototypes is not None:
        label_tensor = build_condition_catalog_label_tensor(condition_catalog, latents.device)
        if label_tensor is not None:
            return model.semantic_logits(latents).index_select(1, label_tensor)
    if condition_catalog_tensor is None:
        raise RuntimeError("Condition catalog tensor is required for model-based condition readout.")
    return model.condition_candidate_logits(latents, condition_catalog_tensor)


def compute_condition_alignment_scores(
    *,
    model: VideoDynamicsMVP,
    latents: torch.Tensor,
    condition: torch.Tensor,
    condition_catalog: ConditionCatalog | None,
    catalog_readout_mode: str,
) -> torch.Tensor:
    if catalog_readout_mode == "semantic_prototype" and model.semantic_prototypes is not None:
        label_targets = build_condition_label_targets(condition, condition_catalog)
        if label_targets is not None:
            semantic_logits = model.semantic_logits(latents)
            return -semantic_logits.gather(1, label_targets.unsqueeze(1)).squeeze(1)
    cond_embed = model.condition_encoder(condition)
    return model.condition_alignment_energy(latents, cond_embed)


def sample_valid_negative_conditions(
    condition: torch.Tensor,
    condition_catalog: ConditionCatalog | None,
    condition_catalog_tensor: torch.Tensor | None,
    hard_negative_prob: float,
    max_hamming: int,
) -> torch.Tensor:
    if condition_catalog is None or condition_catalog_tensor is None:
        return sample_negative_conditions(
            condition,
            hard_negative_prob=hard_negative_prob,
            max_edits=max_hamming,
        )

    selected_indices: list[int] = []
    num_candidates = len(condition_catalog.keys)
    for row in condition:
        source_idx = condition_catalog.index_by_key[condition_tuple_from_tensor(row)]
        if random.random() < hard_negative_prob:
            nearby = [idx for idx, dist in condition_catalog.neighbors[source_idx] if dist <= max_hamming]
            if not nearby and condition_catalog.neighbors[source_idx]:
                min_dist = condition_catalog.neighbors[source_idx][0][1]
                nearby = [idx for idx, dist in condition_catalog.neighbors[source_idx] if dist == min_dist]
            pool = nearby or [idx for idx in range(num_candidates) if idx != source_idx]
        else:
            pool = [idx for idx in range(num_candidates) if idx != source_idx]
        selected_indices.append(random.choice(pool))

    index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=condition.device)
    return condition_catalog_tensor.index_select(0, index_tensor)


def compute_condition_logits(
    model: VideoDynamicsMVP,
    latents: torch.Tensor,
    condition: torch.Tensor,
    condition_catalog: ConditionCatalog | None,
    condition_catalog_tensor: torch.Tensor | None,
    catalog_readout_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition_catalog is None or condition_catalog_tensor is None:
        return model.condition_logits_and_targets(latents, condition)
    logits = compute_condition_catalog_logits(
        model=model,
        latents=latents,
        condition_catalog=condition_catalog,
        condition_catalog_tensor=condition_catalog_tensor,
        catalog_readout_mode=catalog_readout_mode,
    )
    targets = build_condition_targets(condition, condition_catalog)
    if targets is None:
        raise RuntimeError("Condition catalog targets are unexpectedly missing.")
    return logits, targets


def compute_one_step_latent_mse(
    model: VideoDynamicsMVP,
    latents: torch.Tensor,
    cond_embed: torch.Tensor,
) -> torch.Tensor:
    if latents.size(1) < 2:
        return latents.new_tensor(0.0)
    current = latents[:, :-1].reshape(-1, latents.size(-1))
    cond_seq = cond_embed.unsqueeze(1).expand(-1, latents.size(1) - 1, -1).reshape(-1, cond_embed.size(-1))
    pred_next, _ = model.step_dynamics(current, cond_seq)
    target = latents[:, 1:].reshape(-1, latents.size(-1))
    return ((pred_next - target) ** 2).mean()


@torch.no_grad()
def evaluate_query_responsive_execution(
    *,
    model: VideoDynamicsMVP,
    dataset: FolderVideoDataset,
    condition_catalog: ConditionCatalog,
    device: torch.device,
    alpha: float,
    obs_alpha: float,
    plan_core_alpha: float,
    posterior_temperature: float,
    max_samples: int,
    catalog_readout_mode: str,
) -> dict[str, float]:
    if max_samples <= 0:
        return {
            "query_direct_mse": 0.0,
            "query_support_top1_mse": 0.0,
            "query_exec_mse": 0.0,
            "query_set_best_mse": 0.0,
            "query_oracle_mse": 0.0,
            "query_exec_gap_to_oracle": 0.0,
            "query_exec_gain_vs_top1": 0.0,
            "query_match_true": 0.0,
            "query_exec_set_size": 0.0,
            "query_fallback_rate": 0.0,
            "query_samples": 0.0,
        }

    candidate_conditions = condition_catalog.tensor.to(device)
    cond_embed_all = model.condition_encoder(candidate_conditions)
    num_candidates = candidate_conditions.size(0)

    total_direct_mse = 0.0
    total_support_top1_mse = 0.0
    total_exec_mse = 0.0
    total_set_best_mse = 0.0
    total_oracle_mse = 0.0
    total_exec_set_size = 0.0
    total_fallback = 0.0
    total_match_true = 0.0
    processed = 0

    for index in range(min(len(dataset), max_samples)):
        sample = dataset[index]
        video = sample["video"].unsqueeze(0).to(device)
        condition = sample["condition"].unsqueeze(0)

        latents = model.encode_video(video)
        obs_logits = compute_condition_catalog_logits(
            model=model,
            latents=latents,
            condition_catalog=condition_catalog,
            condition_catalog_tensor=candidate_conditions,
            catalog_readout_mode=catalog_readout_mode,
        )
        obs_posterior = build_condition_inference_posterior(obs_logits, temperature=posterior_temperature)

        z_start = latents[:, 0].expand(num_candidates, -1)
        rollout_latents, _ = model.rollout_from(z_start, cond_embed_all, steps=video.size(1) - 1)
        rollout_video = model.decode_video(rollout_latents, cond_embed_all)
        target_future = video[:, 1:].expand(num_candidates, -1, -1, -1, -1)
        future_mse = ((rollout_video - target_future) ** 2).mean(dim=(1, 2, 3, 4))

        true_idx = condition_catalog.index_by_key[condition_tuple_from_tensor(condition[0])]
        rollout_state = model.trajectory_point(rollout_latents)
        rollout_log_weights_by_condition: list[torch.Tensor] = []
        for cond_idx in range(num_candidates):
            query_embed = cond_embed_all[cond_idx : cond_idx + 1].expand(num_candidates, -1)
            conditional_measure = model.conditional_measure(
                rollout_latents,
                query_embed,
                state=rollout_state,
            )
            rollout_log_weights_by_condition.append(conditional_measure.log_total_density.squeeze(-1))
        measure_selection = query_measure_execution(
            obs_posterior=obs_posterior,
            rollout_log_weights_by_condition=torch.stack(rollout_log_weights_by_condition, dim=0).unsqueeze(0),
            obs_alpha=obs_alpha,
            readout_alpha=alpha,
            temperature=posterior_temperature,
        )
        plan_members = measure_selection.member_indices()[0]

        direct_mse = float(future_mse[true_idx].item())
        support_top1_mse = float(future_mse[int(measure_selection.rollout_readout.top1_idx[0].item())].item())
        exec_idx = int(measure_selection.selected_idx[0].item())
        exec_mse = float(future_mse[exec_idx].item())
        set_best_idx = min(plan_members, key=lambda idx: float(future_mse[idx].item()))
        set_best_mse = float(future_mse[set_best_idx].item())
        oracle_mse = float(future_mse.min().item())

        total_direct_mse += direct_mse
        total_support_top1_mse += support_top1_mse
        total_exec_mse += exec_mse
        total_set_best_mse += set_best_mse
        total_oracle_mse += oracle_mse
        total_exec_set_size += len(measure_selection.member_indices()[0])
        total_fallback += 0.0
        total_match_true += float(exec_idx == true_idx)
        processed += 1

    denom = max(processed, 1)
    avg_exec_mse = total_exec_mse / denom
    avg_oracle_mse = total_oracle_mse / denom
    return {
        "query_direct_mse": total_direct_mse / denom,
        "query_support_top1_mse": total_support_top1_mse / denom,
        "query_exec_mse": avg_exec_mse,
        "query_set_best_mse": total_set_best_mse / denom,
        "query_oracle_mse": avg_oracle_mse,
        "query_exec_gap_to_oracle": avg_exec_mse - avg_oracle_mse,
        "query_exec_gain_vs_top1": (total_support_top1_mse - total_exec_mse) / denom,
        "query_match_true": total_match_true / denom,
        "query_exec_set_size": total_exec_set_size / denom,
        "query_fallback_rate": total_fallback / denom,
        "query_samples": float(processed),
    }


def append_jsonl(path: Path, record: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_query_checkpoint_score(
    *,
    val_metrics: dict[str, float],
    support_score: float,
    cfg: AppConfig,
) -> tuple[tuple[float, float, float], dict[str, float]]:
    fallback_rate = val_metrics["query_fallback_rate"]
    fallback_budget = cfg.train.query_checkpoint_fallback_budget
    overflow = max(fallback_rate - fallback_budget, 0.0)
    within_budget = 1.0 if overflow <= 0.0 else 0.0
    exec_mse = val_metrics["query_exec_mse"]
    direct_gap = max(exec_mse - val_metrics["query_direct_mse"], 0.0)
    composite = (
        -cfg.train.query_checkpoint_exec_weight * exec_mse
        + cfg.train.query_checkpoint_match_weight * val_metrics["query_match_true"]
        + cfg.train.query_checkpoint_support_weight * support_score
        - cfg.train.query_checkpoint_gap_weight * direct_gap
    )
    tuple_score = (within_budget, -overflow, composite)
    components = {
        "within_fallback_budget": within_budget,
        "fallback_budget": fallback_budget,
        "fallback_rate": fallback_rate,
        "fallback_overflow": overflow,
        "composite": composite,
        "exec_term": -cfg.train.query_checkpoint_exec_weight * exec_mse,
        "match_term": cfg.train.query_checkpoint_match_weight * val_metrics["query_match_true"],
        "support_term": cfg.train.query_checkpoint_support_weight * support_score,
        "gap_penalty": cfg.train.query_checkpoint_gap_weight * direct_gap,
    }
    return tuple_score, components


def compute_measure_checkpoint_score(
    *,
    val_metrics: dict[str, float],
    cfg: AppConfig,
) -> tuple[float, dict[str, float]]:
    trace_penalty = 0.25 * val_metrics["measure_trace_alignment"] if cfg.loss.measure_trace_weight > 0.0 else 0.0
    tilt_penalty = 0.25 * val_metrics["measure_tilt_alignment"] if cfg.loss.measure_tilt_target_weight > 0.0 else 0.0
    tangent_projection_penalty = 0.25 * val_metrics["tangent_projection"] if cfg.loss.tangent_projection_weight > 0.0 else 0.0
    tangent_bundle_penalty = 0.25 * val_metrics["tangent_bundle_compatibility"] if cfg.loss.tangent_compatibility_weight > 0.0 else 0.0
    tangent_spectrum_penalty = 0.02 * val_metrics["tangent_spectrum_alignment"] if cfg.loss.tangent_spectrum_weight > 0.0 else 0.0
    tangent_shape_penalty = 0.10 * val_metrics["tangent_shape_alignment"] if cfg.loss.tangent_shape_weight > 0.0 else 0.0
    measure_score = -(
        val_metrics["measure_stationarity"]
        + 0.5 * val_metrics["response_smoothness"]
        + 0.25 * val_metrics["local_drift"]
        + 0.25 * val_metrics["local_diffusion"]
        + trace_penalty
        + tilt_penalty
        + tangent_projection_penalty
        + tangent_bundle_penalty
        + tangent_spectrum_penalty
        + tangent_shape_penalty
    )
    components = {
        "mode": "measure",
        "measure_score": measure_score,
        "measure_stationarity": val_metrics["measure_stationarity"],
        "measure_trig_stationarity": val_metrics["measure_trig_stationarity"],
        "measure_trace_alignment": val_metrics["measure_trace_alignment"],
        "measure_tilt_alignment": val_metrics["measure_tilt_alignment"],
        "response_smoothness": val_metrics["response_smoothness"],
        "local_drift": val_metrics["local_drift"],
        "local_diffusion": val_metrics["local_diffusion"],
        "tangent_projection": val_metrics["tangent_projection"],
        "tangent_bundle_compatibility": val_metrics["tangent_bundle_compatibility"],
        "tangent_spectrum_alignment": val_metrics["tangent_spectrum_alignment"],
        "tangent_shape_alignment": val_metrics["tangent_shape_alignment"],
    }
    return measure_score, components


def write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_history_plot(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [
        ("recon", "Reconstruction"),
        ("dyn", "Dynamics"),
        ("cond_acc", "Condition Accuracy"),
        ("energy_gap", "Energy Gap"),
    ]
    for ax, (key, title) in zip(axes.flatten(), plots):
        train_key = f"train_{key}"
        val_key = f"val_{key}"
        if train_key in history[0]:
            ax.plot(epochs, [row[train_key] for row in history], label="train")
        if val_key in history[0]:
            ax.plot(epochs, [row[val_key] for row in history], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_samples(
    output_dir: Path,
    model: VideoDynamicsMVP,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    epoch: int,
) -> None:
    model.eval()
    with torch.no_grad():
        video = batch["video"][:4].to(device)
        condition = batch["condition"][:4].to(device)
        condition_texts = batch.get("condition_text", [])
        out = model(video, condition)
        recon = out.recon.cpu()
        original = video.cpu()

    rows = original.size(0)
    cols = original.size(1)
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 1.5, rows * 3.0))
    for r in range(rows):
        for c in range(cols):
            axes[2 * r, c].imshow(original[r, c].permute(1, 2, 0).numpy())
            axes[2 * r, c].axis("off")
            axes[2 * r + 1, c].imshow(recon[r, c].permute(1, 2, 0).numpy())
            axes[2 * r + 1, c].axis("off")
            if c == 0:
                title = condition_texts[r] if r < len(condition_texts) else format_condition_tensor(condition[r].cpu())
                axes[2 * r, c].set_title(title, fontsize=7)
    plt.tight_layout()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(samples_dir / f"epoch_{epoch:03d}.png")
    plt.close(fig)


def train_one_epoch(
    *,
    model: VideoDynamicsMVP,
    target_model: VideoDynamicsMVP | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    cfg: AppConfig,
    device: torch.device,
    epoch: int,
    condition_catalog: ConditionCatalog | None = None,
    condition_catalog_tensor: torch.Tensor | None = None,
    max_train_steps: int = 0,
) -> dict[str, float]:
    model.train()
    if target_model is not None:
        target_model.eval()
    weights = compute_stage_weights(epoch, cfg)
    total_metrics = {
        "loss": 0.0,
        "recon": 0.0,
        "rep": 0.0,
        "loc": 0.0,
        "dyn": 0.0,
        "cond": 0.0,
        "identity": 0.0,
        "semantic_proto": 0.0,
        "semantic_center": 0.0,
        "semantic_proto_sep": 0.0,
        "reg": 0.0,
        "cond_acc": 0.0,
        "cond_true_prob": 0.0,
        "cond_entropy_norm": 0.0,
        "cond_support_ratio": 0.0,
        "cond_true_rank": 0.0,
        "cond_k90": 0.0,
        "cond_true_in90": 0.0,
        "support_refine": 0.0,
        "support_p_true_hinge": 0.0,
        "support_margin_hinge": 0.0,
        "support_ratio_hinge": 0.0,
        "support_gate_mean": 0.0,
        "local_drift": 0.0,
        "local_diffusion": 0.0,
        "measure_stationarity": 0.0,
        "measure_linear_stationarity": 0.0,
        "measure_quadratic_stationarity": 0.0,
        "measure_trig_stationarity": 0.0,
        "measure_trace_alignment": 0.0,
        "measure_pred_trace": 0.0,
        "measure_target_trace": 0.0,
        "response_smoothness": 0.0,
        "response_signature_norm": 0.0,
        "measure_density_entropy": 0.0,
        "measure_tilt_alignment": 0.0,
        "response_operator_trace": 0.0,
        "response_operator_effective_rank": 0.0,
        "response_operator_anisotropy": 0.0,
        "response_operator_asymmetry": 0.0,
        "response_drift_alignment": 0.0,
        "tangent_projection": 0.0,
        "tangent_observation_residual": 0.0,
        "tangent_drift_residual": 0.0,
        "tangent_diffusion_residual": 0.0,
        "tangent_bundle_compatibility": 0.0,
        "tangent_frame_orthogonality": 0.0,
        "tangent_projector_trace": 0.0,
        "tangent_spectrum_alignment": 0.0,
        "tangent_shape_alignment": 0.0,
        "tangent_nontriviality": 0.0,
        "tangent_anisotropy_gap": 0.0,
        "pred_tangent_effective_rank": 0.0,
        "target_tangent_effective_rank": 0.0,
        "pred_tangent_anisotropy": 0.0,
        "target_tangent_anisotropy": 0.0,
        "pred_tangent_trace": 0.0,
        "target_tangent_trace": 0.0,
        "pred_tangent_spectral_gap": 0.0,
        "target_tangent_spectral_gap": 0.0,
        "measure_tilt_overreach": 0.0,
        "generator_base_trace": 0.0,
        "generator_delta_trace": 0.0,
        "generator_delta_drift_norm": 0.0,
        "generator_delta_diffusion_norm": 0.0,
        "generator_delta_tilt_abs_mean": 0.0,
        "generator_delta_budget": 0.0,
        "identity_acc": 0.0,
        "semantic_acc": 0.0,
        "identity_ortho": 0.0,
        "identity_scale": 0.0,
        "energy_gap": 0.0,
        "one_step_latent": 0.0,
    }
    steps = 0
    identity_scale = scheduled_scale(
        epoch=epoch,
        start_epoch=cfg.train.identity_start_epoch,
        warmup_epochs=cfg.train.identity_warmup_epochs,
    )
    measure_active = (
        cfg.loss.local_drift_weight > 0.0
        or cfg.loss.local_diffusion_weight > 0.0
        or cfg.loss.measure_stationarity_weight > 0.0
        or cfg.loss.measure_trace_weight > 0.0
        or cfg.loss.measure_tilt_target_weight > 0.0
        or cfg.loss.response_smoothness_weight > 0.0
        or cfg.loss.tangent_projection_weight > 0.0
        or cfg.loss.tangent_compatibility_weight > 0.0
        or cfg.loss.tangent_spectrum_weight > 0.0
        or cfg.loss.tangent_shape_weight > 0.0
        or cfg.loss.tangent_nontriviality_weight > 0.0
        or cfg.loss.measure_tilt_overreach_weight > 0.0
        or cfg.loss.generator_delta_weight > 0.0
    )
    neighborhood_bank_size = max(int(cfg.loss.response_neighborhood_bank_size), 0)
    geometry_reference: GeometryNeighborhoodReference | None = None
    smoothness_reference: SmoothnessNeighborhoodReference | None = None

    progress = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
    for batch in progress:
        if max_train_steps and steps >= max_train_steps:
            break

        video = batch["video"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)
        label_index = batch["label_index"].to(device, non_blocking=True)
        negative_condition = sample_valid_negative_conditions(
            condition,
            condition_catalog=condition_catalog,
            condition_catalog_tensor=condition_catalog_tensor,
            hard_negative_prob=cfg.loss.hard_negative_prob,
            max_hamming=cfg.loss.max_negative_edits,
        )

        optimizer.zero_grad(set_to_none=True)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (cfg.train.amp and device.type == "cuda")
            else nullcontext()
        )
        with amp_context:
            out = model(video, condition)
            target_cond_embed = None
            if target_model is not None:
                with torch.no_grad():
                    target_cond_embed = target_model.condition_encoder(condition)
            recon_loss = reconstruction_loss(out.recon, video)
            rep_smooth = latent_representation_loss(out.latents)
            loc_loss = (
                local_linearity_loss(model, out.latents, video, out.cond_embed, eps=cfg.loss.loc_eps)
                if cfg.loss.loc_weight > 0.0
                else out.latents.new_tensor(0.0)
            )
            rep_loss = rep_smooth + cfg.loss.loc_weight * loc_loss

            dyn_short_bias = 1.0 if epoch < cfg.train.stage2_epochs else 0.5
            dyn_loss_value, dyn_per_sample, delta_reg = dynamics_loss(
                model=model,
                latents=out.latents,
                video=video,
                cond_embed=out.cond_embed,
                short_span_bias=dyn_short_bias,
            )

            logits, cond_targets = compute_condition_logits(
                model=model,
                latents=out.latents,
                condition=condition,
                condition_catalog=condition_catalog,
                condition_catalog_tensor=condition_catalog_tensor,
                catalog_readout_mode=cfg.train.condition_catalog_readout_mode,
            )
            cond_nce = nce_condition_loss(logits, cond_targets)
            cond_acc = classification_accuracy(logits, cond_targets)
            cond_dist = summarize_condition_distribution(
                logits,
                cond_targets,
                temperature=cfg.train.query_eval_posterior_temperature,
            )
            support_terms = support_refinement_loss(
                logits,
                cond_targets,
                posterior_temperature=cfg.train.query_eval_posterior_temperature,
                p_true_floor=cfg.loss.support_p_true_floor,
                p_true_ceiling=cfg.loss.support_p_true_ceiling,
                margin_floor=cfg.loss.support_margin_floor,
                margin_ceiling=cfg.loss.support_margin_ceiling,
                support_ratio_floor=cfg.loss.support_ratio_floor,
                support_ratio_ceiling=cfg.loss.support_ratio_ceiling,
                gate_p_true=cfg.loss.support_gate_p_true,
                gate_margin=cfg.loss.support_gate_margin,
                gate_temperature=cfg.loss.support_gate_temperature,
            )
            support_refine = (
                cfg.loss.support_p_true_weight * support_terms["support_p_true_hinge"]
                + cfg.loss.support_margin_weight * support_terms["support_margin_hinge"]
                + cfg.loss.support_ratio_weight * support_terms["support_ratio_hinge"]
            )
            measure_terms = {
                "local_drift": out.latents.new_tensor(0.0),
                "local_diffusion": out.latents.new_tensor(0.0),
                "measure_stationarity": out.latents.new_tensor(0.0),
                "measure_linear_stationarity": out.latents.new_tensor(0.0),
                "measure_quadratic_stationarity": out.latents.new_tensor(0.0),
                "measure_trig_stationarity": out.latents.new_tensor(0.0),
                "measure_trace_alignment": out.latents.new_tensor(0.0),
                "measure_pred_trace": out.latents.new_tensor(0.0),
                "measure_target_trace": out.latents.new_tensor(0.0),
                "response_smoothness": out.latents.new_tensor(0.0),
                "response_signature_norm": out.latents.new_tensor(0.0),
                "measure_density_entropy": out.latents.new_tensor(0.0),
                "measure_tilt_alignment": out.latents.new_tensor(0.0),
                "response_operator_trace": out.latents.new_tensor(0.0),
                "response_operator_effective_rank": out.latents.new_tensor(0.0),
                "response_operator_anisotropy": out.latents.new_tensor(0.0),
                "response_operator_asymmetry": out.latents.new_tensor(0.0),
                "response_drift_alignment": out.latents.new_tensor(0.0),
                "tangent_projection": out.latents.new_tensor(0.0),
                "tangent_observation_residual": out.latents.new_tensor(0.0),
                "tangent_drift_residual": out.latents.new_tensor(0.0),
                "tangent_diffusion_residual": out.latents.new_tensor(0.0),
                "tangent_bundle_compatibility": out.latents.new_tensor(0.0),
                "tangent_frame_orthogonality": out.latents.new_tensor(0.0),
                "tangent_projector_trace": out.latents.new_tensor(0.0),
                "tangent_spectrum_alignment": out.latents.new_tensor(0.0),
                "tangent_shape_alignment": out.latents.new_tensor(0.0),
                "tangent_nontriviality": out.latents.new_tensor(0.0),
                "tangent_anisotropy_gap": out.latents.new_tensor(0.0),
                "pred_tangent_effective_rank": out.latents.new_tensor(0.0),
                "target_tangent_effective_rank": out.latents.new_tensor(0.0),
                "pred_tangent_anisotropy": out.latents.new_tensor(0.0),
                "target_tangent_anisotropy": out.latents.new_tensor(0.0),
                "pred_tangent_trace": out.latents.new_tensor(0.0),
                "target_tangent_trace": out.latents.new_tensor(0.0),
                "pred_tangent_spectral_gap": out.latents.new_tensor(0.0),
                "target_tangent_spectral_gap": out.latents.new_tensor(0.0),
                "measure_tilt_overreach": out.latents.new_tensor(0.0),
                "generator_base_trace": out.latents.new_tensor(0.0),
                "generator_delta_trace": out.latents.new_tensor(0.0),
                "generator_delta_drift_norm": out.latents.new_tensor(0.0),
                "generator_delta_diffusion_norm": out.latents.new_tensor(0.0),
                "generator_delta_tilt_abs_mean": out.latents.new_tensor(0.0),
                "generator_delta_budget": out.latents.new_tensor(0.0),
            }
            local_measure_total = out.latents.new_tensor(0.0)
            if measure_active:
                measure_terms = local_measure_loss(
                    model=model,
                    latents=out.latents,
                    video=video,
                    cond_embed=out.cond_embed,
                    signature_knn=cfg.loss.response_signature_knn,
                    signature_temperature=cfg.loss.response_signature_temperature,
                    geometry_knn=cfg.loss.response_geometry_knn,
                    geometry_temperature=cfg.loss.response_geometry_temperature,
                    jet_ridge=cfg.loss.response_jet_ridge,
                    jet_center_weight=cfg.loss.response_jet_center_weight,
                    tau_ridge=cfg.loss.response_tau_ridge,
                    tau_mean_penalty=cfg.loss.response_tau_mean_penalty,
                    tau_drift_scale=cfg.loss.response_tau_drift_scale,
                    density_temperature=cfg.loss.measure_density_temperature,
                    test_num_directions=cfg.loss.measure_test_num_directions,
                    trig_scale=cfg.loss.measure_trig_scale,
                    diffusion_target_mode=cfg.loss.diffusion_target_mode,
                    measure_target_mode=cfg.loss.measure_target_mode,
                    measure_target_blend=cfg.loss.measure_target_blend,
                    drift_target_mode=cfg.loss.drift_target_mode,
                    drift_target_blend=cfg.loss.drift_target_blend,
                    tilt_target_mode=cfg.loss.tilt_target_mode,
                    tilt_target_blend=cfg.loss.tilt_target_blend,
                    signature_mode=cfg.model.response_signature_mode,
                    decoded=out.recon,
                    target_model=target_model,
                    target_cond_embed=target_cond_embed,
                    geometry_reference=geometry_reference,
                    smoothness_reference=smoothness_reference,
                )
                local_measure_total = (
                    cfg.loss.local_drift_weight * measure_terms["local_drift"]
                    + cfg.loss.local_diffusion_weight * measure_terms["local_diffusion"]
                    + cfg.loss.measure_stationarity_weight * measure_terms["measure_stationarity"]
                    + cfg.loss.measure_trace_weight * measure_terms["measure_trace_alignment"]
                    + cfg.loss.response_smoothness_weight * measure_terms["response_smoothness"]
                    + cfg.loss.measure_tilt_target_weight * measure_terms["measure_tilt_alignment"]
                    + cfg.loss.tangent_projection_weight * measure_terms["tangent_projection"]
                    + cfg.loss.tangent_compatibility_weight * measure_terms["tangent_bundle_compatibility"]
                    + cfg.loss.tangent_spectrum_weight * measure_terms["tangent_spectrum_alignment"]
                    + cfg.loss.tangent_shape_weight * measure_terms["tangent_shape_alignment"]
                    + cfg.loss.tangent_nontriviality_weight * measure_terms["tangent_nontriviality"]
                    + cfg.loss.measure_tilt_overreach_weight * measure_terms["measure_tilt_overreach"]
                    + cfg.loss.generator_delta_weight * measure_terms["generator_delta_budget"]
                )

            identity_loss_value = out.latents.new_tensor(0.0)
            identity_acc = out.latents.new_tensor(0.0)
            identity_ortho = out.latents.new_tensor(0.0)
            semantic_proto_loss_value = out.latents.new_tensor(0.0)
            semantic_center = out.latents.new_tensor(0.0)
            semantic_proto_sep = out.latents.new_tensor(0.0)
            semantic_acc = out.latents.new_tensor(0.0)
            if (
                cfg.loss.semantic_proto_weight > 0.0
                and model.semantic_prototypes is not None
                and bool((label_index >= 0).any())
            ):
                semantic_summary, _, _ = model.decompose_summary(out.latents)
                semantic_logits = model.semantic_logits(out.latents)
                semantic_proto_loss_value = F.cross_entropy(semantic_logits, label_index)
                semantic_acc = classification_accuracy(semantic_logits, label_index)
                semantic_center = prototype_alignment_loss(
                    features=semantic_summary,
                    prototypes=model.semantic_prototypes,
                    labels=label_index,
                )
                semantic_proto_sep = prototype_separation_loss(model.semantic_prototypes)
            if (
                cfg.loss.identity_weight > 0.0
                and identity_scale > 0.0
                and model.identity_classifier is not None
                and bool((label_index >= 0).any())
            ):
                identity_logits = model.identity_logits(out.latents)
                identity_loss_value = F.cross_entropy(identity_logits, label_index)
                identity_acc = classification_accuracy(identity_logits, label_index)
                semantic_summary, identity_residual, _ = model.decompose_summary(out.latents)
                semantic_unit = F.normalize(semantic_summary, dim=-1, eps=1e-6)
                residual_unit = F.normalize(identity_residual, dim=-1, eps=1e-6)
                identity_ortho = ((semantic_unit * residual_unit).sum(dim=-1) ** 2).mean()

            negative_embed = model.condition_encoder(negative_condition)
            neg_align = compute_condition_alignment_scores(
                model=model,
                latents=out.latents,
                condition=negative_condition,
                condition_catalog=condition_catalog,
                catalog_readout_mode=cfg.train.condition_catalog_readout_mode,
            )
            pos_align = compute_condition_alignment_scores(
                model=model,
                latents=out.latents,
                condition=condition,
                condition_catalog=condition_catalog,
                catalog_readout_mode=cfg.train.condition_catalog_readout_mode,
            )
            _, dyn_neg_per_sample, _ = dynamics_loss(
                model=model,
                latents=out.latents.detach() if weights["cond"] == 0 else out.latents,
                video=video,
                cond_embed=negative_embed,
                short_span_bias=dyn_short_bias,
            )
            energy_pos = pos_align + cfg.loss.condition_dyn_weight * dyn_per_sample
            energy_neg = neg_align + cfg.loss.condition_dyn_weight * dyn_neg_per_sample
            cond_gap = gap_loss(energy_pos, energy_neg)
            cond_loss_value = cond_nce + weights["gap"] * cond_gap
            energy_gap_value = (energy_neg - energy_pos).mean()
            one_step_latent = compute_one_step_latent_mse(model, out.latents, out.cond_embed)

            cond_delta_norm = pos_align.pow(2).mean()
            temporal_delta_smoothness = (
                ((out.latents[:, 1:] - out.latents[:, :-1]) ** 2).mean()
            )
            reg_loss_value = regularization_loss(
                cond_delta_norm=cond_delta_norm + delta_reg,
                temporal_delta_smoothness=temporal_delta_smoothness,
                delta_reg_weight=cfg.loss.delta_reg_weight,
                delta_temporal_weight=cfg.loss.delta_temporal_weight,
            )

            loss = (
                weights["base"] * recon_loss
                + weights["rep"] * rep_loss
                + weights["dyn"] * dyn_loss_value
                + weights["cond"] * cond_loss_value
                + weights["cond"] * support_refine
                + local_measure_total
                + identity_scale * cfg.loss.identity_weight * identity_loss_value
                + identity_scale * cfg.loss.identity_ortho_weight * identity_ortho
                + cfg.loss.semantic_proto_weight * semantic_proto_loss_value
                + cfg.loss.semantic_center_weight * semantic_center
                + cfg.loss.semantic_proto_sep_weight * semantic_proto_sep
                + weights["reg"] * reg_loss_value
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        update_measure_target_teacher(target_model, model, cfg.train.measure_target_ema_decay)
        if measure_active and neighborhood_bank_size > 0 and out.latents.size(1) > 1:
            with torch.no_grad():
                smoothness_snapshot = build_smoothness_neighborhood_reference(
                    model=model,
                    latents=out.latents.detach(),
                    video=video,
                    cond_embed=out.cond_embed.detach(),
                    signature_mode=cfg.model.response_signature_mode,
                    decoded=out.recon.detach(),
                )
                smoothness_reference = append_smoothness_neighborhood_reference(
                    smoothness_reference,
                    smoothness_snapshot,
                    max_size=neighborhood_bank_size,
                )
                if target_model is not None:
                    target_cond_embed_snapshot = target_model.condition_encoder(condition)
                    target_latents_snapshot = target_model.encode_video(video, cond_embed=target_cond_embed_snapshot)
                    target_decoded_snapshot = target_model.decode_video(target_latents_snapshot, target_cond_embed_snapshot)
                    geometry_snapshot = build_geometry_neighborhood_reference(
                        model=target_model,
                        latents=target_latents_snapshot,
                        video=video,
                        cond_embed=target_cond_embed_snapshot,
                        decoded=target_decoded_snapshot,
                        geometry_knn=cfg.loss.response_geometry_knn,
                        geometry_temperature=cfg.loss.response_geometry_temperature,
                        jet_ridge=cfg.loss.response_jet_ridge,
                        jet_center_weight=cfg.loss.response_jet_center_weight,
                    )
                else:
                    geometry_snapshot = build_geometry_neighborhood_reference(
                        model=model,
                        latents=out.latents.detach(),
                        video=video,
                        cond_embed=out.cond_embed.detach(),
                        decoded=out.recon.detach(),
                        geometry_knn=cfg.loss.response_geometry_knn,
                        geometry_temperature=cfg.loss.response_geometry_temperature,
                        jet_ridge=cfg.loss.response_jet_ridge,
                        jet_center_weight=cfg.loss.response_jet_center_weight,
                    )
                geometry_reference = append_geometry_neighborhood_reference(
                    geometry_reference,
                    geometry_snapshot,
                    max_size=neighborhood_bank_size,
                )

        total_metrics["loss"] += loss.item()
        total_metrics["recon"] += recon_loss.item()
        total_metrics["rep"] += rep_loss.item()
        total_metrics["loc"] += loc_loss.item()
        total_metrics["dyn"] += dyn_loss_value.item()
        total_metrics["cond"] += cond_loss_value.item()
        total_metrics["identity"] += identity_loss_value.item()
        total_metrics["semantic_proto"] += semantic_proto_loss_value.item()
        total_metrics["semantic_center"] += semantic_center.item()
        total_metrics["semantic_proto_sep"] += semantic_proto_sep.item()
        total_metrics["reg"] += reg_loss_value.item()
        total_metrics["cond_acc"] += cond_acc.item()
        total_metrics["cond_true_prob"] += cond_dist["cond_true_prob"].item()
        total_metrics["cond_entropy_norm"] += cond_dist["cond_entropy_norm"].item()
        total_metrics["cond_support_ratio"] += cond_dist["cond_support_ratio"].item()
        total_metrics["cond_true_rank"] += cond_dist["cond_true_rank"].item()
        total_metrics["cond_k90"] += cond_dist["cond_k90"].item()
        total_metrics["cond_true_in90"] += cond_dist["cond_true_in90"].item()
        total_metrics["support_refine"] += support_refine.item()
        total_metrics["support_p_true_hinge"] += support_terms["support_p_true_hinge"].item()
        total_metrics["support_margin_hinge"] += support_terms["support_margin_hinge"].item()
        total_metrics["support_ratio_hinge"] += support_terms["support_ratio_hinge"].item()
        total_metrics["support_gate_mean"] += support_terms["support_gate_mean"].item()
        total_metrics["local_drift"] += measure_terms["local_drift"].item()
        total_metrics["local_diffusion"] += measure_terms["local_diffusion"].item()
        total_metrics["measure_stationarity"] += measure_terms["measure_stationarity"].item()
        total_metrics["measure_linear_stationarity"] += measure_terms["measure_linear_stationarity"].item()
        total_metrics["measure_quadratic_stationarity"] += measure_terms["measure_quadratic_stationarity"].item()
        total_metrics["measure_trig_stationarity"] += measure_terms["measure_trig_stationarity"].item()
        total_metrics["measure_trace_alignment"] += measure_terms["measure_trace_alignment"].item()
        total_metrics["measure_pred_trace"] += measure_terms["measure_pred_trace"].item()
        total_metrics["measure_target_trace"] += measure_terms["measure_target_trace"].item()
        total_metrics["response_smoothness"] += measure_terms["response_smoothness"].item()
        total_metrics["response_signature_norm"] += measure_terms["response_signature_norm"].item()
        total_metrics["measure_density_entropy"] += measure_terms["measure_density_entropy"].item()
        total_metrics["measure_tilt_alignment"] += measure_terms["measure_tilt_alignment"].item()
        total_metrics["response_operator_trace"] += measure_terms["response_operator_trace"].item()
        total_metrics["response_operator_effective_rank"] += measure_terms["response_operator_effective_rank"].item()
        total_metrics["response_operator_anisotropy"] += measure_terms["response_operator_anisotropy"].item()
        total_metrics["response_operator_asymmetry"] += measure_terms["response_operator_asymmetry"].item()
        total_metrics["response_drift_alignment"] += measure_terms["response_drift_alignment"].item()
        total_metrics["tangent_projection"] += measure_terms["tangent_projection"].item()
        total_metrics["tangent_observation_residual"] += measure_terms["tangent_observation_residual"].item()
        total_metrics["tangent_drift_residual"] += measure_terms["tangent_drift_residual"].item()
        total_metrics["tangent_diffusion_residual"] += measure_terms["tangent_diffusion_residual"].item()
        total_metrics["tangent_bundle_compatibility"] += measure_terms["tangent_bundle_compatibility"].item()
        total_metrics["tangent_frame_orthogonality"] += measure_terms["tangent_frame_orthogonality"].item()
        total_metrics["tangent_projector_trace"] += measure_terms["tangent_projector_trace"].item()
        total_metrics["tangent_spectrum_alignment"] += measure_terms["tangent_spectrum_alignment"].item()
        total_metrics["tangent_shape_alignment"] += measure_terms["tangent_shape_alignment"].item()
        total_metrics["tangent_nontriviality"] += measure_terms["tangent_nontriviality"].item()
        total_metrics["tangent_anisotropy_gap"] += measure_terms["tangent_anisotropy_gap"].item()
        total_metrics["pred_tangent_effective_rank"] += measure_terms["pred_tangent_effective_rank"].item()
        total_metrics["target_tangent_effective_rank"] += measure_terms["target_tangent_effective_rank"].item()
        total_metrics["pred_tangent_anisotropy"] += measure_terms["pred_tangent_anisotropy"].item()
        total_metrics["target_tangent_anisotropy"] += measure_terms["target_tangent_anisotropy"].item()
        total_metrics["pred_tangent_trace"] += measure_terms["pred_tangent_trace"].item()
        total_metrics["target_tangent_trace"] += measure_terms["target_tangent_trace"].item()
        total_metrics["pred_tangent_spectral_gap"] += measure_terms["pred_tangent_spectral_gap"].item()
        total_metrics["target_tangent_spectral_gap"] += measure_terms["target_tangent_spectral_gap"].item()
        total_metrics["measure_tilt_overreach"] += measure_terms["measure_tilt_overreach"].item()
        total_metrics["generator_base_trace"] += measure_terms["generator_base_trace"].item()
        total_metrics["generator_delta_trace"] += measure_terms["generator_delta_trace"].item()
        total_metrics["generator_delta_drift_norm"] += measure_terms["generator_delta_drift_norm"].item()
        total_metrics["generator_delta_diffusion_norm"] += measure_terms["generator_delta_diffusion_norm"].item()
        total_metrics["generator_delta_tilt_abs_mean"] += measure_terms["generator_delta_tilt_abs_mean"].item()
        total_metrics["generator_delta_budget"] += measure_terms["generator_delta_budget"].item()
        total_metrics["identity_acc"] += identity_acc.item()
        total_metrics["semantic_acc"] += semantic_acc.item()
        total_metrics["identity_ortho"] += identity_ortho.item()
        total_metrics["identity_scale"] += identity_scale
        total_metrics["energy_gap"] += energy_gap_value.item()
        total_metrics["one_step_latent"] += one_step_latent.item()
        steps += 1

        progress.set_postfix(
            loss=f"{total_metrics['loss'] / steps:.4f}",
            loc=f"{total_metrics['loc'] / steps:.4f}",
            dyn=f"{total_metrics['dyn'] / steps:.4f}",
            cond=f"{total_metrics['cond'] / steps:.4f}",
            acc=f"{total_metrics['cond_acc'] / steps:.3f}",
            ptrue=f"{total_metrics['cond_true_prob'] / steps:.3f}",
            in90=f"{total_metrics['cond_true_in90'] / steps:.3f}",
            srh=f"{total_metrics['support_ratio_hinge'] / steps:.3f}",
            meas=f"{total_metrics['measure_stationarity'] / steps:.3f}",
            sem=f"{total_metrics['semantic_acc'] / steps:.3f}",
            id=f"{total_metrics['identity_acc'] / steps:.3f}",
            idw=f"{total_metrics['identity_scale'] / steps:.2f}",
        )

    return {key: value / max(steps, 1) for key, value in total_metrics.items()}


@torch.no_grad()
def evaluate(
    model: VideoDynamicsMVP,
    target_model: VideoDynamicsMVP | None,
    loader: DataLoader,
    cfg: AppConfig,
    device: torch.device,
    condition_catalog: ConditionCatalog | None = None,
    condition_catalog_tensor: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    if target_model is not None:
        target_model.eval()
    total_recon = 0.0
    total_loc = 0.0
    total_dyn = 0.0
    total_cond = 0.0
    total_identity = 0.0
    total_semantic_proto = 0.0
    total_semantic_center = 0.0
    total_semantic_proto_sep = 0.0
    total_cond_acc = 0.0
    total_cond_true_prob = 0.0
    total_cond_entropy_norm = 0.0
    total_cond_support_ratio = 0.0
    total_cond_true_rank = 0.0
    total_cond_k90 = 0.0
    total_cond_true_in90 = 0.0
    total_support_refine = 0.0
    total_support_p_true_hinge = 0.0
    total_support_margin_hinge = 0.0
    total_support_ratio_hinge = 0.0
    total_support_gate_mean = 0.0
    total_local_drift = 0.0
    total_local_diffusion = 0.0
    total_measure_stationarity = 0.0
    total_measure_linear_stationarity = 0.0
    total_measure_quadratic_stationarity = 0.0
    total_measure_trig_stationarity = 0.0
    total_measure_trace_alignment = 0.0
    total_measure_pred_trace = 0.0
    total_measure_target_trace = 0.0
    total_response_smoothness = 0.0
    total_response_signature_norm = 0.0
    total_measure_density_entropy = 0.0
    total_measure_tilt_alignment = 0.0
    total_response_operator_trace = 0.0
    total_response_operator_effective_rank = 0.0
    total_response_operator_anisotropy = 0.0
    total_response_operator_asymmetry = 0.0
    total_response_drift_alignment = 0.0
    total_tangent_projection = 0.0
    total_tangent_observation_residual = 0.0
    total_tangent_drift_residual = 0.0
    total_tangent_diffusion_residual = 0.0
    total_tangent_bundle_compatibility = 0.0
    total_tangent_frame_orthogonality = 0.0
    total_tangent_projector_trace = 0.0
    total_tangent_spectrum_alignment = 0.0
    total_tangent_shape_alignment = 0.0
    total_tangent_nontriviality = 0.0
    total_tangent_anisotropy_gap = 0.0
    total_pred_tangent_effective_rank = 0.0
    total_target_tangent_effective_rank = 0.0
    total_pred_tangent_anisotropy = 0.0
    total_target_tangent_anisotropy = 0.0
    total_pred_tangent_trace = 0.0
    total_target_tangent_trace = 0.0
    total_pred_tangent_spectral_gap = 0.0
    total_target_tangent_spectral_gap = 0.0
    total_measure_tilt_overreach = 0.0
    total_generator_base_trace = 0.0
    total_generator_delta_trace = 0.0
    total_generator_delta_drift_norm = 0.0
    total_generator_delta_diffusion_norm = 0.0
    total_generator_delta_tilt_abs_mean = 0.0
    total_generator_delta_budget = 0.0
    total_identity_acc = 0.0
    total_semantic_acc = 0.0
    total_identity_ortho = 0.0
    total_energy_gap = 0.0
    total_one_step = 0.0
    steps = 0
    measure_active = (
        cfg.loss.local_drift_weight > 0.0
        or cfg.loss.local_diffusion_weight > 0.0
        or cfg.loss.measure_stationarity_weight > 0.0
        or cfg.loss.measure_trace_weight > 0.0
        or cfg.loss.measure_tilt_target_weight > 0.0
        or cfg.loss.response_smoothness_weight > 0.0
        or cfg.loss.tangent_projection_weight > 0.0
        or cfg.loss.tangent_compatibility_weight > 0.0
        or cfg.loss.tangent_spectrum_weight > 0.0
        or cfg.loss.tangent_shape_weight > 0.0
        or cfg.loss.tangent_nontriviality_weight > 0.0
        or cfg.loss.measure_tilt_overreach_weight > 0.0
        or cfg.loss.generator_delta_weight > 0.0
    )
    neighborhood_bank_size = max(int(cfg.loss.response_neighborhood_bank_size), 0)
    geometry_reference: GeometryNeighborhoodReference | None = None
    smoothness_reference: SmoothnessNeighborhoodReference | None = None

    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)
        label_index = batch["label_index"].to(device, non_blocking=True)
        negative_condition = sample_valid_negative_conditions(
            condition,
            condition_catalog=condition_catalog,
            condition_catalog_tensor=condition_catalog_tensor,
            hard_negative_prob=cfg.loss.hard_negative_prob,
            max_hamming=cfg.loss.max_negative_edits,
        )
        out = model(video, condition)
        target_cond_embed = None
        if target_model is not None:
            target_cond_embed = target_model.condition_encoder(condition)
        recon_loss = reconstruction_loss(out.recon, video)
        loc_loss = (
            local_linearity_loss(model, out.latents, video, out.cond_embed, eps=cfg.loss.loc_eps)
            if cfg.loss.loc_weight > 0.0
            else out.latents.new_tensor(0.0)
        )
        dyn_loss_value, _, _ = dynamics_loss(model, out.latents, video, out.cond_embed, short_span_bias=0.5)
        logits, cond_targets = compute_condition_logits(
            model=model,
            latents=out.latents,
            condition=condition,
            condition_catalog=condition_catalog,
            condition_catalog_tensor=condition_catalog_tensor,
            catalog_readout_mode=cfg.train.condition_catalog_readout_mode,
        )
        cond_loss_value = nce_condition_loss(logits, cond_targets)
        cond_acc = classification_accuracy(logits, cond_targets)
        cond_dist = summarize_condition_distribution(
            logits,
            cond_targets,
            temperature=cfg.train.query_eval_posterior_temperature,
        )
        support_terms = support_refinement_loss(
            logits,
            cond_targets,
            posterior_temperature=cfg.train.query_eval_posterior_temperature,
            p_true_floor=cfg.loss.support_p_true_floor,
            p_true_ceiling=cfg.loss.support_p_true_ceiling,
            margin_floor=cfg.loss.support_margin_floor,
            margin_ceiling=cfg.loss.support_margin_ceiling,
            support_ratio_floor=cfg.loss.support_ratio_floor,
            support_ratio_ceiling=cfg.loss.support_ratio_ceiling,
            gate_p_true=cfg.loss.support_gate_p_true,
            gate_margin=cfg.loss.support_gate_margin,
            gate_temperature=cfg.loss.support_gate_temperature,
        )
        support_refine = (
            cfg.loss.support_p_true_weight * support_terms["support_p_true_hinge"]
            + cfg.loss.support_margin_weight * support_terms["support_margin_hinge"]
            + cfg.loss.support_ratio_weight * support_terms["support_ratio_hinge"]
        )
        measure_terms = {
            "local_drift": out.latents.new_tensor(0.0),
            "local_diffusion": out.latents.new_tensor(0.0),
            "measure_stationarity": out.latents.new_tensor(0.0),
            "measure_linear_stationarity": out.latents.new_tensor(0.0),
            "measure_quadratic_stationarity": out.latents.new_tensor(0.0),
            "measure_trig_stationarity": out.latents.new_tensor(0.0),
            "measure_trace_alignment": out.latents.new_tensor(0.0),
            "measure_pred_trace": out.latents.new_tensor(0.0),
            "measure_target_trace": out.latents.new_tensor(0.0),
            "response_smoothness": out.latents.new_tensor(0.0),
            "response_signature_norm": out.latents.new_tensor(0.0),
            "measure_density_entropy": out.latents.new_tensor(0.0),
            "measure_tilt_alignment": out.latents.new_tensor(0.0),
            "response_operator_trace": out.latents.new_tensor(0.0),
            "response_operator_effective_rank": out.latents.new_tensor(0.0),
            "response_operator_anisotropy": out.latents.new_tensor(0.0),
            "response_operator_asymmetry": out.latents.new_tensor(0.0),
            "response_drift_alignment": out.latents.new_tensor(0.0),
            "tangent_projection": out.latents.new_tensor(0.0),
            "tangent_observation_residual": out.latents.new_tensor(0.0),
            "tangent_drift_residual": out.latents.new_tensor(0.0),
            "tangent_diffusion_residual": out.latents.new_tensor(0.0),
            "tangent_bundle_compatibility": out.latents.new_tensor(0.0),
            "tangent_frame_orthogonality": out.latents.new_tensor(0.0),
            "tangent_projector_trace": out.latents.new_tensor(0.0),
            "tangent_spectrum_alignment": out.latents.new_tensor(0.0),
            "tangent_shape_alignment": out.latents.new_tensor(0.0),
            "tangent_nontriviality": out.latents.new_tensor(0.0),
            "tangent_anisotropy_gap": out.latents.new_tensor(0.0),
            "pred_tangent_effective_rank": out.latents.new_tensor(0.0),
            "target_tangent_effective_rank": out.latents.new_tensor(0.0),
            "pred_tangent_anisotropy": out.latents.new_tensor(0.0),
            "target_tangent_anisotropy": out.latents.new_tensor(0.0),
            "pred_tangent_trace": out.latents.new_tensor(0.0),
            "target_tangent_trace": out.latents.new_tensor(0.0),
            "pred_tangent_spectral_gap": out.latents.new_tensor(0.0),
            "target_tangent_spectral_gap": out.latents.new_tensor(0.0),
            "measure_tilt_overreach": out.latents.new_tensor(0.0),
            "generator_base_trace": out.latents.new_tensor(0.0),
            "generator_delta_trace": out.latents.new_tensor(0.0),
            "generator_delta_drift_norm": out.latents.new_tensor(0.0),
            "generator_delta_diffusion_norm": out.latents.new_tensor(0.0),
            "generator_delta_tilt_abs_mean": out.latents.new_tensor(0.0),
            "generator_delta_budget": out.latents.new_tensor(0.0),
        }
        if measure_active:
            measure_terms = local_measure_loss(
                model=model,
                latents=out.latents,
                video=video,
                cond_embed=out.cond_embed,
                signature_knn=cfg.loss.response_signature_knn,
                signature_temperature=cfg.loss.response_signature_temperature,
                geometry_knn=cfg.loss.response_geometry_knn,
                geometry_temperature=cfg.loss.response_geometry_temperature,
                jet_ridge=cfg.loss.response_jet_ridge,
                jet_center_weight=cfg.loss.response_jet_center_weight,
                tau_ridge=cfg.loss.response_tau_ridge,
                tau_mean_penalty=cfg.loss.response_tau_mean_penalty,
                tau_drift_scale=cfg.loss.response_tau_drift_scale,
                density_temperature=cfg.loss.measure_density_temperature,
                test_num_directions=cfg.loss.measure_test_num_directions,
                trig_scale=cfg.loss.measure_trig_scale,
                diffusion_target_mode=cfg.loss.diffusion_target_mode,
                measure_target_mode=cfg.loss.measure_target_mode,
                measure_target_blend=cfg.loss.measure_target_blend,
                drift_target_mode=cfg.loss.drift_target_mode,
                drift_target_blend=cfg.loss.drift_target_blend,
                tilt_target_mode=cfg.loss.tilt_target_mode,
                tilt_target_blend=cfg.loss.tilt_target_blend,
                signature_mode=cfg.model.response_signature_mode,
                decoded=out.recon,
                target_model=target_model,
                target_cond_embed=target_cond_embed,
                geometry_reference=geometry_reference,
                smoothness_reference=smoothness_reference,
            )
        identity_loss_value = out.latents.new_tensor(0.0)
        identity_acc = out.latents.new_tensor(0.0)
        identity_ortho = out.latents.new_tensor(0.0)
        semantic_proto_loss_value = out.latents.new_tensor(0.0)
        semantic_center = out.latents.new_tensor(0.0)
        semantic_proto_sep = out.latents.new_tensor(0.0)
        semantic_acc = out.latents.new_tensor(0.0)
        if cfg.loss.semantic_proto_weight > 0.0 and model.semantic_prototypes is not None and bool((label_index >= 0).any()):
            semantic_summary, _, _ = model.decompose_summary(out.latents)
            semantic_logits = model.semantic_logits(out.latents)
            semantic_proto_loss_value = F.cross_entropy(semantic_logits, label_index)
            semantic_acc = classification_accuracy(semantic_logits, label_index)
            semantic_center = prototype_alignment_loss(
                features=semantic_summary,
                prototypes=model.semantic_prototypes,
                labels=label_index,
            )
            semantic_proto_sep = prototype_separation_loss(model.semantic_prototypes)
        if cfg.loss.identity_weight > 0.0 and model.identity_classifier is not None and bool((label_index >= 0).any()):
            identity_logits = model.identity_logits(out.latents)
            identity_loss_value = F.cross_entropy(identity_logits, label_index)
            identity_acc = classification_accuracy(identity_logits, label_index)
            semantic_summary, identity_residual, _ = model.decompose_summary(out.latents)
            semantic_unit = F.normalize(semantic_summary, dim=-1, eps=1e-6)
            residual_unit = F.normalize(identity_residual, dim=-1, eps=1e-6)
            identity_ortho = ((semantic_unit * residual_unit).sum(dim=-1) ** 2).mean()
        negative_embed = model.condition_encoder(negative_condition)
        pos_align = compute_condition_alignment_scores(
            model=model,
            latents=out.latents,
            condition=condition,
            condition_catalog=condition_catalog,
            catalog_readout_mode=cfg.train.condition_catalog_readout_mode,
        )
        neg_align = compute_condition_alignment_scores(
            model=model,
            latents=out.latents,
            condition=negative_condition,
            condition_catalog=condition_catalog,
            catalog_readout_mode=cfg.train.condition_catalog_readout_mode,
        )
        _, dyn_neg_per_sample, _ = dynamics_loss(model, out.latents, video, negative_embed, short_span_bias=0.5)
        _, dyn_pos_per_sample, _ = dynamics_loss(model, out.latents, video, out.cond_embed, short_span_bias=0.5)
        energy_pos = pos_align + cfg.loss.condition_dyn_weight * dyn_pos_per_sample
        energy_neg = neg_align + cfg.loss.condition_dyn_weight * dyn_neg_per_sample
        energy_gap_value = (energy_neg - energy_pos).mean()
        one_step_latent = compute_one_step_latent_mse(model, out.latents, out.cond_embed)
        total_recon += recon_loss.item()
        total_loc += loc_loss.item()
        total_dyn += dyn_loss_value.item()
        total_cond += cond_loss_value.item()
        total_identity += identity_loss_value.item()
        total_semantic_proto += semantic_proto_loss_value.item()
        total_semantic_center += semantic_center.item()
        total_semantic_proto_sep += semantic_proto_sep.item()
        total_cond_acc += cond_acc.item()
        total_cond_true_prob += cond_dist["cond_true_prob"].item()
        total_cond_entropy_norm += cond_dist["cond_entropy_norm"].item()
        total_cond_support_ratio += cond_dist["cond_support_ratio"].item()
        total_cond_true_rank += cond_dist["cond_true_rank"].item()
        total_cond_k90 += cond_dist["cond_k90"].item()
        total_cond_true_in90 += cond_dist["cond_true_in90"].item()
        total_support_refine += support_refine.item()
        total_support_p_true_hinge += support_terms["support_p_true_hinge"].item()
        total_support_margin_hinge += support_terms["support_margin_hinge"].item()
        total_support_ratio_hinge += support_terms["support_ratio_hinge"].item()
        total_support_gate_mean += support_terms["support_gate_mean"].item()
        total_local_drift += measure_terms["local_drift"].item()
        total_local_diffusion += measure_terms["local_diffusion"].item()
        total_measure_stationarity += measure_terms["measure_stationarity"].item()
        total_measure_linear_stationarity += measure_terms["measure_linear_stationarity"].item()
        total_measure_quadratic_stationarity += measure_terms["measure_quadratic_stationarity"].item()
        total_measure_trig_stationarity += measure_terms["measure_trig_stationarity"].item()
        total_measure_trace_alignment += measure_terms["measure_trace_alignment"].item()
        total_measure_pred_trace += measure_terms["measure_pred_trace"].item()
        total_measure_target_trace += measure_terms["measure_target_trace"].item()
        total_response_smoothness += measure_terms["response_smoothness"].item()
        total_response_signature_norm += measure_terms["response_signature_norm"].item()
        total_measure_density_entropy += measure_terms["measure_density_entropy"].item()
        total_measure_tilt_alignment += measure_terms["measure_tilt_alignment"].item()
        total_response_operator_trace += measure_terms["response_operator_trace"].item()
        total_response_operator_effective_rank += measure_terms["response_operator_effective_rank"].item()
        total_response_operator_anisotropy += measure_terms["response_operator_anisotropy"].item()
        total_response_operator_asymmetry += measure_terms["response_operator_asymmetry"].item()
        total_response_drift_alignment += measure_terms["response_drift_alignment"].item()
        total_tangent_projection += measure_terms["tangent_projection"].item()
        total_tangent_observation_residual += measure_terms["tangent_observation_residual"].item()
        total_tangent_drift_residual += measure_terms["tangent_drift_residual"].item()
        total_tangent_diffusion_residual += measure_terms["tangent_diffusion_residual"].item()
        total_tangent_bundle_compatibility += measure_terms["tangent_bundle_compatibility"].item()
        total_tangent_frame_orthogonality += measure_terms["tangent_frame_orthogonality"].item()
        total_tangent_projector_trace += measure_terms["tangent_projector_trace"].item()
        total_tangent_spectrum_alignment += measure_terms["tangent_spectrum_alignment"].item()
        total_tangent_shape_alignment += measure_terms["tangent_shape_alignment"].item()
        total_tangent_nontriviality += measure_terms["tangent_nontriviality"].item()
        total_tangent_anisotropy_gap += measure_terms["tangent_anisotropy_gap"].item()
        total_pred_tangent_effective_rank += measure_terms["pred_tangent_effective_rank"].item()
        total_target_tangent_effective_rank += measure_terms["target_tangent_effective_rank"].item()
        total_pred_tangent_anisotropy += measure_terms["pred_tangent_anisotropy"].item()
        total_target_tangent_anisotropy += measure_terms["target_tangent_anisotropy"].item()
        total_pred_tangent_trace += measure_terms["pred_tangent_trace"].item()
        total_target_tangent_trace += measure_terms["target_tangent_trace"].item()
        total_pred_tangent_spectral_gap += measure_terms["pred_tangent_spectral_gap"].item()
        total_target_tangent_spectral_gap += measure_terms["target_tangent_spectral_gap"].item()
        total_measure_tilt_overreach += measure_terms["measure_tilt_overreach"].item()
        total_generator_base_trace += measure_terms["generator_base_trace"].item()
        total_generator_delta_trace += measure_terms["generator_delta_trace"].item()
        total_generator_delta_drift_norm += measure_terms["generator_delta_drift_norm"].item()
        total_generator_delta_diffusion_norm += measure_terms["generator_delta_diffusion_norm"].item()
        total_generator_delta_tilt_abs_mean += measure_terms["generator_delta_tilt_abs_mean"].item()
        total_generator_delta_budget += measure_terms["generator_delta_budget"].item()
        total_identity_acc += identity_acc.item()
        total_semantic_acc += semantic_acc.item()
        total_identity_ortho += identity_ortho.item()
        total_energy_gap += energy_gap_value.item()
        total_one_step += one_step_latent.item()
        steps += 1
        if measure_active and neighborhood_bank_size > 0 and out.latents.size(1) > 1:
            smoothness_snapshot = build_smoothness_neighborhood_reference(
                model=model,
                latents=out.latents.detach(),
                video=video,
                cond_embed=out.cond_embed.detach(),
                signature_mode=cfg.model.response_signature_mode,
                decoded=out.recon.detach(),
            )
            smoothness_reference = append_smoothness_neighborhood_reference(
                smoothness_reference,
                smoothness_snapshot,
                max_size=neighborhood_bank_size,
            )
            if target_model is not None:
                target_cond_embed_snapshot = target_model.condition_encoder(condition)
                target_latents_snapshot = target_model.encode_video(video, cond_embed=target_cond_embed_snapshot)
                target_decoded_snapshot = target_model.decode_video(target_latents_snapshot, target_cond_embed_snapshot)
                geometry_snapshot = build_geometry_neighborhood_reference(
                    model=target_model,
                    latents=target_latents_snapshot,
                    video=video,
                    cond_embed=target_cond_embed_snapshot,
                    decoded=target_decoded_snapshot,
                    geometry_knn=cfg.loss.response_geometry_knn,
                    geometry_temperature=cfg.loss.response_geometry_temperature,
                    jet_ridge=cfg.loss.response_jet_ridge,
                    jet_center_weight=cfg.loss.response_jet_center_weight,
                )
            else:
                geometry_snapshot = build_geometry_neighborhood_reference(
                    model=model,
                    latents=out.latents.detach(),
                    video=video,
                    cond_embed=out.cond_embed.detach(),
                    decoded=out.recon.detach(),
                    geometry_knn=cfg.loss.response_geometry_knn,
                    geometry_temperature=cfg.loss.response_geometry_temperature,
                    jet_ridge=cfg.loss.response_jet_ridge,
                    jet_center_weight=cfg.loss.response_jet_center_weight,
                )
            geometry_reference = append_geometry_neighborhood_reference(
                geometry_reference,
                geometry_snapshot,
                max_size=neighborhood_bank_size,
            )

    metrics = {
        "recon": total_recon / max(steps, 1),
        "loc": total_loc / max(steps, 1),
        "dyn": total_dyn / max(steps, 1),
        "cond": total_cond / max(steps, 1),
        "identity": total_identity / max(steps, 1),
        "semantic_proto": total_semantic_proto / max(steps, 1),
        "semantic_center": total_semantic_center / max(steps, 1),
        "semantic_proto_sep": total_semantic_proto_sep / max(steps, 1),
        "cond_acc": total_cond_acc / max(steps, 1),
        "cond_true_prob": total_cond_true_prob / max(steps, 1),
        "cond_entropy_norm": total_cond_entropy_norm / max(steps, 1),
        "cond_support_ratio": total_cond_support_ratio / max(steps, 1),
        "cond_true_rank": total_cond_true_rank / max(steps, 1),
        "cond_k90": total_cond_k90 / max(steps, 1),
        "cond_true_in90": total_cond_true_in90 / max(steps, 1),
        "support_refine": total_support_refine / max(steps, 1),
        "support_p_true_hinge": total_support_p_true_hinge / max(steps, 1),
        "support_margin_hinge": total_support_margin_hinge / max(steps, 1),
        "support_ratio_hinge": total_support_ratio_hinge / max(steps, 1),
        "support_gate_mean": total_support_gate_mean / max(steps, 1),
        "local_drift": total_local_drift / max(steps, 1),
        "local_diffusion": total_local_diffusion / max(steps, 1),
        "measure_stationarity": total_measure_stationarity / max(steps, 1),
        "measure_linear_stationarity": total_measure_linear_stationarity / max(steps, 1),
        "measure_quadratic_stationarity": total_measure_quadratic_stationarity / max(steps, 1),
        "measure_trig_stationarity": total_measure_trig_stationarity / max(steps, 1),
        "measure_trace_alignment": total_measure_trace_alignment / max(steps, 1),
        "measure_pred_trace": total_measure_pred_trace / max(steps, 1),
        "measure_target_trace": total_measure_target_trace / max(steps, 1),
        "response_smoothness": total_response_smoothness / max(steps, 1),
        "response_signature_norm": total_response_signature_norm / max(steps, 1),
        "measure_density_entropy": total_measure_density_entropy / max(steps, 1),
        "measure_tilt_alignment": total_measure_tilt_alignment / max(steps, 1),
        "response_operator_trace": total_response_operator_trace / max(steps, 1),
        "response_operator_effective_rank": total_response_operator_effective_rank / max(steps, 1),
        "response_operator_anisotropy": total_response_operator_anisotropy / max(steps, 1),
        "response_operator_asymmetry": total_response_operator_asymmetry / max(steps, 1),
        "response_drift_alignment": total_response_drift_alignment / max(steps, 1),
        "tangent_projection": total_tangent_projection / max(steps, 1),
        "tangent_observation_residual": total_tangent_observation_residual / max(steps, 1),
        "tangent_drift_residual": total_tangent_drift_residual / max(steps, 1),
        "tangent_diffusion_residual": total_tangent_diffusion_residual / max(steps, 1),
        "tangent_bundle_compatibility": total_tangent_bundle_compatibility / max(steps, 1),
        "tangent_frame_orthogonality": total_tangent_frame_orthogonality / max(steps, 1),
        "tangent_projector_trace": total_tangent_projector_trace / max(steps, 1),
        "tangent_spectrum_alignment": total_tangent_spectrum_alignment / max(steps, 1),
        "tangent_shape_alignment": total_tangent_shape_alignment / max(steps, 1),
        "tangent_nontriviality": total_tangent_nontriviality / max(steps, 1),
        "tangent_anisotropy_gap": total_tangent_anisotropy_gap / max(steps, 1),
        "pred_tangent_effective_rank": total_pred_tangent_effective_rank / max(steps, 1),
        "target_tangent_effective_rank": total_target_tangent_effective_rank / max(steps, 1),
        "pred_tangent_anisotropy": total_pred_tangent_anisotropy / max(steps, 1),
        "target_tangent_anisotropy": total_target_tangent_anisotropy / max(steps, 1),
        "pred_tangent_trace": total_pred_tangent_trace / max(steps, 1),
        "target_tangent_trace": total_target_tangent_trace / max(steps, 1),
        "pred_tangent_spectral_gap": total_pred_tangent_spectral_gap / max(steps, 1),
        "target_tangent_spectral_gap": total_target_tangent_spectral_gap / max(steps, 1),
        "measure_tilt_overreach": total_measure_tilt_overreach / max(steps, 1),
        "generator_base_trace": total_generator_base_trace / max(steps, 1),
        "generator_delta_trace": total_generator_delta_trace / max(steps, 1),
        "generator_delta_drift_norm": total_generator_delta_drift_norm / max(steps, 1),
        "generator_delta_diffusion_norm": total_generator_delta_diffusion_norm / max(steps, 1),
        "generator_delta_tilt_abs_mean": total_generator_delta_tilt_abs_mean / max(steps, 1),
        "generator_delta_budget": total_generator_delta_budget / max(steps, 1),
        "identity_acc": total_identity_acc / max(steps, 1),
        "semantic_acc": total_semantic_acc / max(steps, 1),
        "identity_ortho": total_identity_ortho / max(steps, 1),
        "energy_gap": total_energy_gap / max(steps, 1),
        "one_step_latent": total_one_step / max(steps, 1),
    }
    if (
        cfg.train.query_eval_enabled
        and condition_catalog is not None
        and isinstance(loader.dataset, FolderVideoDataset)
    ):
        metrics.update(
            evaluate_query_responsive_execution(
                model=model,
                dataset=loader.dataset,
                condition_catalog=condition_catalog,
                device=device,
                alpha=cfg.train.query_eval_alpha,
                obs_alpha=cfg.train.query_eval_obs_alpha,
                plan_core_alpha=cfg.train.query_eval_plan_core_alpha,
                posterior_temperature=cfg.train.query_eval_posterior_temperature,
                max_samples=cfg.train.query_eval_max_samples,
                catalog_readout_mode=cfg.train.condition_catalog_readout_mode,
            )
        )
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.train.device)
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(cfg)
    model = build_model(cfg, device)
    measure_target_teacher = build_measure_target_teacher(model, cfg.train.measure_target_ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.amp and device.type == "cuda"))

    sample_batch = next(iter(val_loader))
    condition_catalog = build_condition_catalog(train_ds, val_ds) if isinstance(train_ds, FolderVideoDataset) else None
    condition_catalog_tensor = condition_catalog.tensor.to(device) if condition_catalog is not None else None

    best_recon_val = math.inf
    best_support_score = -math.inf
    best_measure_score = -math.inf
    best_query_score: tuple[float, float, float] | None = None
    history: list[dict[str, float]] = []

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = metrics_dir / "history.jsonl"
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()

    print(f"device={device}")
    print(f"train_batches={len(train_loader)} val_batches={len(val_loader)}")

    for epoch in range(cfg.train.epochs):
        train_metrics = train_one_epoch(
            model=model,
            target_model=measure_target_teacher,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            epoch=epoch,
            condition_catalog=condition_catalog,
            condition_catalog_tensor=condition_catalog_tensor,
            max_train_steps=args.max_train_steps,
        )
        val_metrics = evaluate(
            model,
            measure_target_teacher if cfg.train.measure_target_use_teacher_eval else None,
            val_loader,
            cfg,
            device,
            condition_catalog=condition_catalog,
            condition_catalog_tensor=condition_catalog_tensor,
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_loc={train_metrics['loc']:.4f} "
            f"train_dyn={train_metrics['dyn']:.4f} "
            f"train_cond={train_metrics['cond']:.4f} "
            f"train_acc={train_metrics['cond_acc']:.3f} "
            f"train_ptrue={train_metrics['cond_true_prob']:.3f} "
            f"train_in90={train_metrics['cond_true_in90']:.3f} "
            f"train_srh={train_metrics['support_ratio_hinge']:.3f} "
            f"train_meas={train_metrics['measure_stationarity']:.3f} "
            f"train_sem_acc={train_metrics['semantic_acc']:.3f} "
            f"train_id_acc={train_metrics['identity_acc']:.3f} "
            f"val_recon={val_metrics['recon']:.4f} "
            f"val_loc={val_metrics['loc']:.4f} "
            f"val_dyn={val_metrics['dyn']:.4f} "
            f"val_acc={val_metrics['cond_acc']:.3f} "
            f"val_ptrue={val_metrics['cond_true_prob']:.3f} "
            f"val_in90={val_metrics['cond_true_in90']:.3f} "
            f"val_sr={val_metrics['cond_support_ratio']:.3f} "
            f"val_srh={val_metrics['support_ratio_hinge']:.3f} "
            f"val_meas={val_metrics['measure_stationarity']:.3f} "
            f"val_sem_acc={val_metrics['semantic_acc']:.3f} "
            f"val_id_acc={val_metrics['identity_acc']:.3f} "
            f"val_gap={val_metrics['energy_gap']:.4f}"
            + (
                f" val_qmse={val_metrics['query_exec_mse']:.4f} "
                f"val_qtop1={val_metrics['query_support_top1_mse']:.4f} "
                f"val_qmatch={val_metrics['query_match_true']:.3f} "
                f"val_qfb={val_metrics['query_fallback_rate']:.3f}"
                if "query_exec_mse" in val_metrics
                else ""
            )
        )

        record = {"epoch": epoch}
        record.update({f"train_{k}": v for k, v in train_metrics.items()})
        record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(record)
        append_jsonl(metrics_jsonl, record)
        write_history_csv(metrics_dir / "history.csv", history)
        save_history_plot(metrics_dir / "history.png", history)

        if epoch % cfg.train.sample_every == 0:
            save_samples(output_dir, model, sample_batch, device, epoch)

        checkpoint = {
            "model": model.state_dict(),
            "target_model": measure_target_teacher.state_dict() if measure_target_teacher is not None else None,
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "val": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        recon_score = val_metrics["recon"] + val_metrics["dyn"]
        if recon_score < best_recon_val:
            best_recon_val = recon_score
            torch.save(checkpoint, output_dir / "best_recon.pt")

        measure_score, measure_selection_info = compute_measure_checkpoint_score(
            val_metrics=val_metrics,
            cfg=cfg,
        )
        checkpoint["selection_score_measure"] = measure_selection_info
        if measure_score > best_measure_score:
            best_measure_score = measure_score
            torch.save(checkpoint, output_dir / "best_measure.pt")
            if cfg.train.checkpoint_selection_mode == "measure":
                torch.save(checkpoint, output_dir / "best.pt")

        if condition_catalog is None:
            if recon_score < best_support_score or best_support_score == -math.inf:
                best_support_score = recon_score
                checkpoint["selection_score"] = -recon_score
                if cfg.train.checkpoint_selection_mode != "measure":
                    torch.save(checkpoint, output_dir / "best.pt")
        else:
            support_score = (
                val_metrics["cond_true_prob"]
                + val_metrics["cond_true_in90"]
                - val_metrics["cond_support_ratio"]
            )
            if support_score > best_support_score:
                best_support_score = support_score
                checkpoint["selection_score"] = support_score
                torch.save(checkpoint, output_dir / "best_support.pt")
                if cfg.train.checkpoint_selection_mode == "support":
                    torch.save(checkpoint, output_dir / "best.pt")

            if cfg.train.query_eval_enabled and "query_exec_mse" in val_metrics:
                if cfg.train.checkpoint_selection_mode == "query_responsive":
                    query_score = (
                        val_metrics["query_match_true"] - val_metrics["query_fallback_rate"],
                        -val_metrics["query_exec_mse"],
                        support_score,
                    )
                    selection_info = {
                        "mode": "query_responsive_legacy",
                        "match_minus_fallback": query_score[0],
                        "neg_query_exec_mse": query_score[1],
                        "support_score": query_score[2],
                    }
                else:
                    query_score, selection_info = compute_query_checkpoint_score(
                        val_metrics=val_metrics,
                        support_score=support_score,
                        cfg=cfg,
                    )
                    selection_info["mode"] = cfg.train.checkpoint_selection_mode

                if best_query_score is None or query_score > best_query_score:
                    best_query_score = query_score
                    checkpoint["selection_score_query"] = selection_info
                    torch.save(checkpoint, output_dir / "best_query.pt")
                    if cfg.train.checkpoint_selection_mode in {"query_responsive", "query_balanced"}:
                        torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
