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


def build_candidate_posterior(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> CandidatePosterior:
    probs = posterior_from_logits(logits, temperature=temperature)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    top1_prob, top1_idx = probs.max(dim=-1)
    return CandidatePosterior(
        logits=logits,
        probs=probs,
        sorted_probs=sorted_probs,
        sorted_idx=sorted_idx,
        top1_idx=top1_idx,
        top1_prob=top1_prob,
    )


def candidate_sets_from_posterior(
    posterior: CandidatePosterior,
    alphas: list[float],
) -> dict[float, CandidateSet]:
    cumsum = posterior.sorted_probs.cumsum(dim=-1)
    rank_positions = torch.arange(posterior.probs.size(-1), device=posterior.probs.device).unsqueeze(0)
    results: dict[float, CandidateSet] = {}
    for alpha in alphas:
        threshold = torch.full_like(cumsum[:, :1], alpha)
        k_alpha = torch.searchsorted(cumsum, threshold, right=False).squeeze(1) + 1
        k_alpha = torch.clamp(k_alpha, min=1, max=posterior.probs.size(-1))
        sorted_mask = rank_positions < k_alpha.unsqueeze(1)
        mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
        mask.scatter_(1, posterior.sorted_idx, sorted_mask)
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


def summarize_condition_distribution(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.90,
) -> dict[str, torch.Tensor]:
    posterior = build_candidate_posterior(logits)
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
