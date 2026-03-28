from __future__ import annotations

import torch
import torch.nn.functional as F

from vh_mvp.support import posterior_from_logits


def reconstruction_loss(recon: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, video)


def latent_representation_loss(latents: torch.Tensor) -> torch.Tensor:
    deltas = latents[:, 1:] - latents[:, :-1]
    smooth = (deltas**2).mean()
    accel = ((deltas[:, 1:] - deltas[:, :-1]) ** 2).mean() if deltas.size(1) > 1 else latents.new_tensor(0.0)
    return smooth + 0.5 * accel


def local_linearity_loss(
    model,
    latents: torch.Tensor,
    video: torch.Tensor,
    cond_embed: torch.Tensor,
    eps: float = 1e-2,
) -> torch.Tensor:
    if latents.size(1) < 2:
        return latents.new_tensor(0.0)

    batch, steps, latent_dim = latents.shape
    z_start = latents[:, :-1].reshape(-1, latent_dim)
    z_next = latents[:, 1:].reshape(-1, latent_dim)
    delta_z = z_next - z_start
    cond_seq = cond_embed.unsqueeze(1).expand(batch, steps - 1, cond_embed.size(-1)).reshape(-1, cond_embed.size(-1))

    base = model.frame_decoder(z_start, cond_seq)
    perturbed = model.frame_decoder(z_start + eps * delta_z, cond_seq)
    linear_delta = (perturbed - base) / eps
    true_delta = (video[:, 1:] - video[:, :-1]).reshape(-1, *video.shape[2:])
    return F.mse_loss(linear_delta, true_delta)


def dynamics_loss(
    model,
    latents: torch.Tensor,
    video: torch.Tensor,
    cond_embed: torch.Tensor,
    short_span_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, steps, _ = latents.shape
    decoded = model.decode_video(latents, cond_embed)
    total = latents.new_tensor(0.0)
    weight_sum = latents.new_tensor(0.0)
    per_sample = latents.new_zeros(batch)
    delta_reg = latents.new_tensor(0.0)

    for i in range(steps - 1):
        max_roll = steps - i - 1
        rollout, deltas = model.rollout_from(latents[:, i], cond_embed, max_roll)
        if deltas.numel() > 0:
            delta_reg = delta_reg + (deltas**2).mean()
        for span in range(1, max_roll + 1):
            j = i + span
            weight = 1.0 / (span**short_span_bias)
            pred_frame = model.decode_video(rollout[:, span - 1 : span], cond_embed)[:, 0]
            prev_frame = decoded[:, j - 1]
            pred_delta = pred_frame - prev_frame
            true_delta = video[:, j] - video[:, j - 1]
            residual = ((pred_delta - true_delta) ** 2).flatten(1).mean(dim=1)
            total = total + weight * residual.mean()
            weight_sum = weight_sum + weight
            per_sample = per_sample + weight * residual

    total = total / weight_sum.clamp_min(1e-6)
    per_sample = per_sample / weight_sum.clamp_min(1e-6)
    delta_reg = delta_reg / max(steps - 1, 1)
    return total, per_sample, delta_reg


def nce_condition_loss(logits: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
    if labels is None:
        labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def support_refinement_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    p_true_floor: float,
    margin_floor: float,
    support_ratio_ceiling: float,
    gate_p_true: float,
    gate_margin: float,
    gate_temperature: float,
) -> dict[str, torch.Tensor]:
    probs = posterior_from_logits(logits)
    p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    probs_wo_true = probs.clone()
    probs_wo_true.scatter_(1, labels.unsqueeze(1), -1.0)
    p_second = probs_wo_true.max(dim=1).values if probs.size(1) > 1 else torch.zeros_like(p_true)
    margin = p_true - p_second

    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    support_ratio = torch.exp(entropy) / float(logits.size(1))

    zero = logits.new_tensor(0.0)
    p_true_hinge = F.relu(p_true_floor - p_true).mean() if p_true_floor > 0.0 else zero
    margin_hinge = F.relu(margin_floor - margin).mean() if margin_floor > 0.0 else zero

    gate_threshold_p = gate_p_true if gate_p_true > 0.0 else max(p_true_floor, 0.0)
    gate_threshold_m = gate_margin if gate_margin > 0.0 else max(margin_floor, 0.0)
    gate_temperature = max(gate_temperature, 1e-4)
    confidence_gate = torch.sigmoid((p_true.detach() - gate_threshold_p) / gate_temperature)
    if probs.size(1) > 1:
        confidence_gate = confidence_gate * torch.sigmoid((margin.detach() - gate_threshold_m) / gate_temperature)

    support_ratio_hinge = zero
    if support_ratio_ceiling < 1.0:
        support_ratio_hinge = (confidence_gate * F.relu(support_ratio - support_ratio_ceiling)).mean()

    return {
        "support_p_true_hinge": p_true_hinge,
        "support_margin_hinge": margin_hinge,
        "support_ratio_hinge": support_ratio_hinge,
        "support_gate_mean": confidence_gate.mean(),
    }


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == labels).float().mean()


def prototype_alignment_loss(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    feature_unit = F.normalize(features, dim=-1, eps=1e-6)
    prototype_unit = F.normalize(prototypes, dim=-1, eps=1e-6)
    assigned = prototype_unit[labels]
    return (1.0 - (feature_unit * assigned).sum(dim=-1)).mean()


def prototype_separation_loss(prototypes: torch.Tensor) -> torch.Tensor:
    if prototypes.size(0) < 2:
        return prototypes.new_tensor(0.0)
    prototype_unit = F.normalize(prototypes, dim=-1, eps=1e-6)
    gram = prototype_unit @ prototype_unit.T
    mask = ~torch.eye(prototypes.size(0), dtype=torch.bool, device=prototypes.device)
    return (gram[mask] ** 2).mean()


def gap_loss(
    energy_pos: torch.Tensor,
    energy_neg: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    return F.relu(margin + energy_pos - energy_neg).mean()


def regularization_loss(
    cond_delta_norm: torch.Tensor,
    temporal_delta_smoothness: torch.Tensor,
    delta_reg_weight: float,
    delta_temporal_weight: float,
) -> torch.Tensor:
    return delta_reg_weight * cond_delta_norm + delta_temporal_weight * temporal_delta_smoothness


def compute_stage_weights(epoch: int, cfg) -> dict[str, float]:
    beta = 0.0
    gamma = 0.0
    eta = cfg.loss.reg_weight

    if epoch >= cfg.train.stage1_epochs:
        beta = cfg.loss.dyn_weight
    if epoch >= cfg.train.stage2_epochs:
        gamma = cfg.loss.cond_weight
    if epoch >= cfg.train.stage3_epochs:
        beta = cfg.loss.dyn_weight
        gamma = cfg.loss.cond_weight

    return {
        "base": cfg.loss.base_weight,
        "rep": cfg.loss.rep_weight,
        "dyn": beta,
        "cond": gamma,
        "reg": eta if gamma > 0.0 else eta * 0.25,
        "gap": cfg.loss.gap_weight if gamma > 0.0 else 0.0,
    }
