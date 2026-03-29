from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vh_mvp.support import build_condition_inference_posterior, candidate_sets_from_posterior


def summarize_encoded_video(encoded: torch.Tensor) -> torch.Tensor:
    if encoded.ndim != 5:
        raise ValueError(f"Expected encoded video to have shape [B, T, C, H, W], got {tuple(encoded.shape)}")
    pooled = encoded.mean(dim=(-1, -2))
    first = pooled[:, 0]
    mean = pooled.mean(dim=1)
    last = pooled[:, -1]
    delta = last - first
    if pooled.size(1) > 1:
        velocity = pooled[:, 1:] - pooled[:, :-1]
        mean_velocity = velocity.mean(dim=1)
    else:
        mean_velocity = torch.zeros_like(mean)
    return torch.cat([first, mean, last, delta, mean_velocity], dim=-1)


def build_condition_probe(
    *,
    input_dim: int,
    num_classes: int,
    probe_type: str,
    hidden_dim: int,
) -> nn.Module:
    if probe_type == "linear":
        return nn.Linear(input_dim, num_classes)
    if probe_type == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    raise ValueError(f"Unsupported probe_type: {probe_type}")


@dataclass
class ProbeTrainResult:
    probe: nn.Module
    history: list[dict[str, float]]
    best_val_acc: float


def train_condition_probe(
    *,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    probe_type: str,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> ProbeTrainResult:
    probe = build_condition_probe(
        input_dim=train_features.size(1),
        num_classes=int(train_targets.max().item()) + 1,
        probe_type=probe_type,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    val_features = val_features.to(device)
    val_targets = val_targets.to(device)

    loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    history: list[dict[str, float]] = []
    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0
        for feat, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = probe(feat)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * feat.size(0)
            total_correct += float((logits.argmax(dim=1) == target).float().sum().item())
            total_count += feat.size(0)

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_features)
            val_acc = float((val_logits.argmax(dim=1) == val_targets).float().mean().item())

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(total_count, 1),
                "train_acc": total_correct / max(total_count, 1),
                "val_acc": val_acc,
            }
        )

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    return ProbeTrainResult(probe=probe, history=history, best_val_acc=best_acc)


def protocol_b_selection_metrics(
    *,
    future_mse: torch.Tensor,
    true_idx: int,
    posterior_logits: torch.Tensor,
    alpha: float,
    temperature: float,
) -> dict[str, float]:
    posterior = build_condition_inference_posterior(posterior_logits.unsqueeze(0), temperature=temperature)
    candidate_set = candidate_sets_from_posterior(posterior, [alpha])[alpha]
    members = candidate_set.member_indices()[0]
    top1_idx = int(posterior.top1_idx[0].item())
    exec_idx = top1_idx
    set_best_idx = min(members, key=lambda idx: float(future_mse[idx].item()))
    oracle_idx = int(future_mse.argmin().item())
    return {
        "query_direct_mse": float(future_mse[true_idx].item()),
        "query_support_top1_mse": float(future_mse[top1_idx].item()),
        "query_exec_mse": float(future_mse[exec_idx].item()),
        "query_set_best_mse": float(future_mse[set_best_idx].item()),
        "query_oracle_mse": float(future_mse[oracle_idx].item()),
        "query_exec_gap_to_oracle": float(future_mse[exec_idx].item() - future_mse[oracle_idx].item()),
        "query_exec_gain_vs_top1": 0.0,
        "query_match_true": float(exec_idx == true_idx),
        "query_exec_set_size": float(len(members)),
        "query_fallback_rate": 0.0,
    }
