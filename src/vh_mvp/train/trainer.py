from __future__ import annotations

import argparse
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
    classification_accuracy,
    compute_stage_weights,
    dynamics_loss,
    gap_loss,
    latent_representation_loss,
    local_linearity_loss,
    nce_condition_loss,
    prototype_alignment_loss,
    prototype_separation_loss,
    reconstruction_loss,
    regularization_loss,
    support_refinement_loss,
)
from vh_mvp.models import VideoDynamicsMVP
from vh_mvp.support import summarize_condition_distribution


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
        )
        val_ds = SyntheticVideoDataset(
            size=cfg.data.val_size,
            seq_len=cfg.data.seq_len,
            image_size=cfg.data.image_size,
            seed=cfg.seed + 100_000,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition_catalog is None or condition_catalog_tensor is None:
        return model.condition_logits_and_targets(latents, condition)
    logits = model.condition_candidate_logits(latents, condition_catalog_tensor)
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


def append_jsonl(path: Path, record: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
            )
            cond_nce = nce_condition_loss(logits, cond_targets)
            cond_acc = classification_accuracy(logits, cond_targets)
            cond_dist = summarize_condition_distribution(logits, cond_targets)
            support_terms = support_refinement_loss(
                logits,
                cond_targets,
                p_true_floor=cfg.loss.support_p_true_floor,
                margin_floor=cfg.loss.support_margin_floor,
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
            neg_align = model.condition_alignment_energy(out.latents, negative_embed)
            pos_align = model.condition_alignment_energy(out.latents, out.cond_embed)
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
            sem=f"{total_metrics['semantic_acc'] / steps:.3f}",
            id=f"{total_metrics['identity_acc'] / steps:.3f}",
            idw=f"{total_metrics['identity_scale'] / steps:.2f}",
        )

    return {key: value / max(steps, 1) for key, value in total_metrics.items()}


@torch.no_grad()
def evaluate(
    model: VideoDynamicsMVP,
    loader: DataLoader,
    cfg: AppConfig,
    device: torch.device,
    condition_catalog: ConditionCatalog | None = None,
    condition_catalog_tensor: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
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
    total_identity_acc = 0.0
    total_semantic_acc = 0.0
    total_identity_ortho = 0.0
    total_energy_gap = 0.0
    total_one_step = 0.0
    steps = 0

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
        )
        cond_loss_value = nce_condition_loss(logits, cond_targets)
        cond_acc = classification_accuracy(logits, cond_targets)
        cond_dist = summarize_condition_distribution(logits, cond_targets)
        support_terms = support_refinement_loss(
            logits,
            cond_targets,
            p_true_floor=cfg.loss.support_p_true_floor,
            margin_floor=cfg.loss.support_margin_floor,
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
        pos_align = model.condition_alignment_energy(out.latents, out.cond_embed)
        neg_align = model.condition_alignment_energy(out.latents, negative_embed)
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
        total_identity_acc += identity_acc.item()
        total_semantic_acc += semantic_acc.item()
        total_identity_ortho += identity_ortho.item()
        total_energy_gap += energy_gap_value.item()
        total_one_step += one_step_latent.item()
        steps += 1

    return {
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
        "identity_acc": total_identity_acc / max(steps, 1),
        "semantic_acc": total_semantic_acc / max(steps, 1),
        "identity_ortho": total_identity_ortho / max(steps, 1),
        "energy_gap": total_energy_gap / max(steps, 1),
        "one_step_latent": total_one_step / max(steps, 1),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.train.device)
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(cfg)
    model = build_model(cfg, device)
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
            f"val_sem_acc={val_metrics['semantic_acc']:.3f} "
            f"val_id_acc={val_metrics['identity_acc']:.3f} "
            f"val_gap={val_metrics['energy_gap']:.4f}"
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

        if condition_catalog is None:
            if recon_score < best_support_score or best_support_score == -math.inf:
                best_support_score = recon_score
                checkpoint["selection_score"] = -recon_score
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
                torch.save(checkpoint, output_dir / "best.pt")
                torch.save(checkpoint, output_dir / "best_support.pt")
