from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def posterior_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return F.softmax(logits / temperature, dim=-1)


def condition_key(condition: tuple[int, ...]) -> str:
    return ",".join(str(v) for v in condition)


@dataclass
class CandidatePosterior:
    logits: torch.Tensor
    probs: torch.Tensor
    sorted_probs: torch.Tensor
    sorted_idx: torch.Tensor
    top1_idx: torch.Tensor
    top1_prob: torch.Tensor


@dataclass
class CandidateSet:
    alpha: float
    k_alpha: torch.Tensor
    mask: torch.Tensor
    probs: torch.Tensor

    @property
    def num_candidates(self) -> int:
        return int(self.mask.size(1))

    def member_indices(self) -> list[list[int]]:
        members: list[list[int]] = []
        for row in self.mask:
            members.append(torch.nonzero(row, as_tuple=False).squeeze(1).tolist())
        return members

    def mass(self) -> torch.Tensor:
        return (self.probs * self.mask.float()).sum(dim=-1)

    def masked_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~self.mask, float("-inf"))


@dataclass
class ConditionInferencePosterior:
    logits: torch.Tensor
    probs: torch.Tensor
    sorted_probs: torch.Tensor
    sorted_idx: torch.Tensor
    top1_idx: torch.Tensor
    top1_prob: torch.Tensor

    def as_candidate_posterior(self) -> CandidatePosterior:
        return CandidatePosterior(
            logits=self.logits,
            probs=self.probs,
            sorted_probs=self.sorted_probs,
            sorted_idx=self.sorted_idx,
            top1_idx=self.top1_idx,
            top1_prob=self.top1_prob,
        )


@dataclass
class MeasureReadout:
    alpha: float
    log_weights: torch.Tensor
    weights: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_idx: torch.Tensor
    top1_idx: torch.Tensor
    top1_weight: torch.Tensor
    k_alpha: torch.Tensor
    mask: torch.Tensor

    def member_indices(self) -> list[list[int]]:
        members: list[list[int]] = []
        for row in self.mask:
            members.append(torch.nonzero(row, as_tuple=False).squeeze(1).tolist())
        return members

    def mass(self) -> torch.Tensor:
        return (self.weights * self.mask.float()).sum(dim=-1)

    def restricted_argmax(self, scores: torch.Tensor) -> torch.Tensor:
        return masked_argmax(scores, self.mask)


@dataclass
class MeasureExecutionSelection:
    selected_idx: torch.Tensor
    condition_set: CandidateSet
    condition_weights: torch.Tensor
    rollout_readout: MeasureReadout

    def member_indices(self) -> list[list[int]]:
        return self.rollout_readout.member_indices()

    def mass(self) -> torch.Tensor:
        return self.rollout_readout.mass()


@dataclass
class QueryResponsiveSelection:
    selected_idx: torch.Tensor
    exec_mask: torch.Tensor
    obs_set: CandidateSet
    plan_core: CandidateSet
    used_plan_core_fallback: torch.Tensor

    def member_indices(self) -> list[list[int]]:
        members: list[list[int]] = []
        for row in self.exec_mask:
            members.append(torch.nonzero(row, as_tuple=False).squeeze(1).tolist())
        return members


def _ranked_distribution(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    top1_prob, top1_idx = probs.max(dim=-1)
    return sorted_probs, sorted_idx, top1_prob, top1_idx


def _candidate_set_masks(
    *,
    sorted_probs: torch.Tensor,
    sorted_idx: torch.Tensor,
    num_candidates: int,
    alphas: list[float],
) -> dict[float, tuple[torch.Tensor, torch.Tensor]]:
    cumsum = sorted_probs.cumsum(dim=-1)
    rank_positions = torch.arange(num_candidates, device=sorted_probs.device).unsqueeze(0)
    results: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
    for alpha in alphas:
        threshold = torch.full_like(cumsum[:, :1], alpha)
        k_alpha = torch.searchsorted(cumsum, threshold, right=False).squeeze(1) + 1
        k_alpha = torch.clamp(k_alpha, min=1, max=num_candidates)
        sorted_mask = rank_positions < k_alpha.unsqueeze(1)
        mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
        mask.scatter_(1, sorted_idx, sorted_mask)
        results[alpha] = k_alpha, mask
    return results


def build_condition_inference_posterior(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> ConditionInferencePosterior:
    probs = posterior_from_logits(logits, temperature=temperature)
    sorted_probs, sorted_idx, top1_prob, top1_idx = _ranked_distribution(probs)
    return ConditionInferencePosterior(
        logits=logits,
        probs=probs,
        sorted_probs=sorted_probs,
        sorted_idx=sorted_idx,
        top1_idx=top1_idx,
        top1_prob=top1_prob,
    )


def build_candidate_posterior(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> CandidatePosterior:
    return build_condition_inference_posterior(logits, temperature=temperature).as_candidate_posterior()


def candidate_sets_from_posterior(
    posterior: CandidatePosterior | ConditionInferencePosterior,
    alphas: list[float],
) -> dict[float, CandidateSet]:
    results: dict[float, CandidateSet] = {}
    masks = _candidate_set_masks(
        sorted_probs=posterior.sorted_probs,
        sorted_idx=posterior.sorted_idx,
        num_candidates=posterior.probs.size(-1),
        alphas=alphas,
    )
    for alpha, (k_alpha, mask) in masks.items():
        results[alpha] = CandidateSet(
            alpha=alpha,
            k_alpha=k_alpha,
            mask=mask,
            probs=posterior.probs,
        )
    return results


def alpha_candidate_sets(
    probs: torch.Tensor,
    alphas: list[float],
) -> dict[float, dict[str, torch.Tensor]]:
    posterior = build_candidate_posterior(torch.log(probs.clamp_min(1e-12)), temperature=1.0)
    sets = candidate_sets_from_posterior(posterior, alphas)
    return {
        alpha: {
            "k_alpha": candidate_set.k_alpha,
            "mask": candidate_set.mask,
        }
        for alpha, candidate_set in sets.items()
    }


def measure_readouts_from_log_weights(
    log_weights: torch.Tensor,
    *,
    alphas: list[float],
    temperature: float = 1.0,
) -> dict[float, MeasureReadout]:
    if log_weights.ndim == 1:
        log_weights = log_weights.unsqueeze(0)
    weights = posterior_from_logits(log_weights, temperature=temperature)
    sorted_weights, sorted_idx, top1_weight, top1_idx = _ranked_distribution(weights)
    masks = _candidate_set_masks(
        sorted_probs=sorted_weights,
        sorted_idx=sorted_idx,
        num_candidates=weights.size(-1),
        alphas=alphas,
    )
    results: dict[float, MeasureReadout] = {}
    for alpha, (k_alpha, mask) in masks.items():
        results[alpha] = MeasureReadout(
            alpha=alpha,
            log_weights=log_weights,
            weights=weights,
            sorted_weights=sorted_weights,
            sorted_idx=sorted_idx,
            top1_idx=top1_idx,
            top1_weight=top1_weight,
            k_alpha=k_alpha,
            mask=mask,
        )
    return results


def measure_readout_from_log_weights(
    log_weights: torch.Tensor,
    *,
    alpha: float = 0.90,
    temperature: float = 1.0,
) -> MeasureReadout:
    return measure_readouts_from_log_weights(
        log_weights,
        alphas=[alpha],
        temperature=temperature,
    )[alpha]


def masked_argmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or mask.ndim != 2:
        raise ValueError("masked_argmax expects 2D scores and 2D mask tensors.")
    if scores.shape != mask.shape:
        raise ValueError(f"shape mismatch: scores={scores.shape}, mask={mask.shape}")
    if not bool(mask.any(dim=1).all().item()):
        raise ValueError("Each row must contain at least one valid candidate.")
    return scores.masked_fill(~mask, float("-inf")).argmax(dim=1)


def query_responsive_selection(
    obs_posterior: CandidatePosterior | ConditionInferencePosterior,
    plan_posterior: CandidatePosterior | ConditionInferencePosterior,
    *,
    obs_alpha: float = 0.90,
    plan_core_alpha: float = 0.50,
) -> QueryResponsiveSelection:
    obs_set = candidate_sets_from_posterior(obs_posterior, [obs_alpha])[obs_alpha]
    plan_core = candidate_sets_from_posterior(plan_posterior, [plan_core_alpha])[plan_core_alpha]
    exec_mask = obs_set.mask & plan_core.mask
    used_plan_core_fallback = ~exec_mask.any(dim=1)
    exec_mask = torch.where(used_plan_core_fallback.unsqueeze(1), plan_core.mask, exec_mask)
    selected_idx = masked_argmax(obs_posterior.probs, exec_mask)
    return QueryResponsiveSelection(
        selected_idx=selected_idx,
        exec_mask=exec_mask,
        obs_set=obs_set,
        plan_core=plan_core,
        used_plan_core_fallback=used_plan_core_fallback,
    )


def query_measure_execution(
    obs_posterior: ConditionInferencePosterior,
    rollout_log_weights_by_condition: torch.Tensor,
    *,
    obs_alpha: float = 0.90,
    readout_alpha: float = 0.90,
    temperature: float = 1.0,
) -> MeasureExecutionSelection:
    if rollout_log_weights_by_condition.ndim == 2:
        rollout_log_weights_by_condition = rollout_log_weights_by_condition.unsqueeze(0)
    if rollout_log_weights_by_condition.ndim != 3:
        raise ValueError(
            "rollout_log_weights_by_condition must have shape [B, C, K] or [C, K], "
            f"got {tuple(rollout_log_weights_by_condition.shape)}"
        )
    batch, num_conditions, _ = rollout_log_weights_by_condition.shape
    if obs_posterior.probs.ndim != 2 or obs_posterior.probs.size(0) != batch:
        raise ValueError(
            "obs_posterior batch shape must match rollout_log_weights_by_condition batch shape, "
            f"got posterior={tuple(obs_posterior.probs.shape)} and rollout={tuple(rollout_log_weights_by_condition.shape)}"
        )
    if obs_posterior.probs.size(1) != num_conditions:
        raise ValueError(
            "condition count mismatch between obs_posterior and rollout_log_weights_by_condition, "
            f"got {obs_posterior.probs.size(1)} vs {num_conditions}"
        )
    condition_set = candidate_sets_from_posterior(obs_posterior, [obs_alpha])[obs_alpha]
    condition_weights = obs_posterior.probs * condition_set.mask.float()
    condition_weights = condition_weights / condition_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    rollout_weights = posterior_from_logits(rollout_log_weights_by_condition, temperature=temperature)
    mixed_rollout_weights = (condition_weights.unsqueeze(-1) * rollout_weights).sum(dim=1)
    mixed_rollout_weights = mixed_rollout_weights / mixed_rollout_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    rollout_readout = measure_readout_from_log_weights(
        mixed_rollout_weights.clamp_min(1e-12).log(),
        alpha=readout_alpha,
        temperature=1.0,
    )
    return MeasureExecutionSelection(
        selected_idx=rollout_readout.top1_idx,
        condition_set=condition_set,
        condition_weights=condition_weights,
        rollout_readout=rollout_readout,
    )


def summarize_condition_distribution(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.90,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    posterior = build_condition_inference_posterior(logits, temperature=temperature)
    probs = posterior.probs
    p_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    entropy_norm = entropy / math.log(max(logits.size(1), 2))
    eff_support_ratio = torch.exp(entropy) / float(logits.size(1))
    rank_by_class = torch.argsort(posterior.sorted_idx, dim=-1)
    true_rank = rank_by_class.gather(1, targets.unsqueeze(1)).squeeze(1).float() + 1.0
    candidate_set = candidate_sets_from_posterior(posterior, [alpha])[alpha]
    k_alpha = candidate_set.k_alpha.float()
    true_in_alpha = (true_rank <= k_alpha).float()
    return {
        "cond_true_prob": p_true.mean(),
        "cond_entropy_norm": entropy_norm.mean(),
        "cond_support_ratio": eff_support_ratio.mean(),
        "cond_true_rank": true_rank.mean(),
        "cond_k90": k_alpha.mean(),
        "cond_true_in90": true_in_alpha.mean(),
    }
