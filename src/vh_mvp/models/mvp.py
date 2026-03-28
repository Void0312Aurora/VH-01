from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from vh_mvp.data import CONDITION_CARDINALITIES


class ConditionEncoder(nn.Module):
    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        embed_dim = max(8, math.ceil(cond_dim / len(CONDITION_CARDINALITIES)))
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embed_dim)
            for cardinality in CONDITION_CARDINALITIES.values()
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * len(CONDITION_CARDINALITIES), cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        chunks = []
        for idx, emb in enumerate(self.embeddings):
            chunks.append(emb(condition[:, idx]))
        return self.proj(torch.cat(chunks, dim=-1))


class FrameEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 4 * 4, latent_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames)


class FrameDecoder(nn.Module):
    def __init__(self, out_channels: int, base_channels: int, latent_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, base_channels * 4 * 4 * 4),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latents: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(torch.cat([latents, cond_embed], dim=-1))
        hidden = hidden.view(hidden.size(0), -1, 4, 4)
        return self.net(hidden)


class DynamicsMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ForwardOutput:
    recon: torch.Tensor
    latents: torch.Tensor
    cond_embed: torch.Tensor


class VideoDynamicsMVP(nn.Module):
    def __init__(
        self,
        channels: int,
        base_channels: int,
        latent_dim: int,
        cond_dim: int,
        hidden_dim: int,
        condition_score_mode: str = "distance",
        energy_hidden_dim: int = 128,
        identity_num_classes: int = 0,
        identity_hidden_dim: int = 128,
        semantic_num_classes: int = 0,
        semantic_temperature: float = 0.2,
    ) -> None:
        super().__init__()
        if condition_score_mode not in {"distance", "energy"}:
            raise ValueError(f"Unsupported condition_score_mode: {condition_score_mode}")
        self.latent_dim = latent_dim
        self.condition_score_mode = condition_score_mode
        self.identity_num_classes = identity_num_classes
        self.semantic_num_classes = semantic_num_classes
        self.semantic_temperature = semantic_temperature
        self.condition_encoder = ConditionEncoder(cond_dim)
        self.frame_encoder = FrameEncoder(channels, base_channels, latent_dim)
        self.frame_decoder = FrameDecoder(channels, base_channels, latent_dim, cond_dim)
        self.base_dynamics = DynamicsMLP(latent_dim, hidden_dim, latent_dim)
        self.cond_delta = DynamicsMLP(latent_dim + cond_dim, hidden_dim, latent_dim)
        self.cond_projection = nn.Linear(cond_dim, latent_dim)
        self.condition_energy_head = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, energy_hidden_dim),
            nn.SiLU(),
            nn.Linear(energy_hidden_dim, energy_hidden_dim),
            nn.SiLU(),
            nn.Linear(energy_hidden_dim, 1),
        )
        if identity_num_classes > 0:
            self.identity_residual_head = nn.Sequential(
                nn.Linear(latent_dim, identity_hidden_dim),
                nn.SiLU(),
                nn.Linear(identity_hidden_dim, latent_dim),
            )
            self.identity_classifier = nn.Sequential(
                nn.Linear(latent_dim, identity_hidden_dim),
                nn.SiLU(),
                nn.Linear(identity_hidden_dim, identity_num_classes),
            )
        else:
            self.identity_residual_head = None
            self.identity_classifier = None
        if semantic_num_classes > 0:
            self.semantic_prototypes = nn.Parameter(torch.randn(semantic_num_classes, latent_dim) * 0.02)
        else:
            self.register_parameter("semantic_prototypes", None)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = video.shape
        flat = video.view(batch * steps, channels, height, width)
        latents = self.frame_encoder(flat)
        return latents.view(batch, steps, self.latent_dim)

    def decode_video(self, latents: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        batch, steps, latent_dim = latents.shape
        cond_seq = cond_embed.unsqueeze(1).expand(batch, steps, cond_embed.size(-1))
        flat_latents = latents.reshape(batch * steps, latent_dim)
        flat_cond = cond_seq.reshape(batch * steps, cond_embed.size(-1))
        recon = self.frame_decoder(flat_latents, flat_cond)
        return recon.view(batch, steps, *recon.shape[1:])

    def forward(self, video: torch.Tensor, condition: torch.Tensor) -> ForwardOutput:
        cond_embed = self.condition_encoder(condition)
        latents = self.encode_video(video)
        recon = self.decode_video(latents, cond_embed)
        return ForwardOutput(recon=recon, latents=latents, cond_embed=cond_embed)

    def decompose_summary(
        self,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary = latents.mean(dim=1)
        if self.identity_residual_head is None:
            zero = torch.zeros_like(summary)
            return summary, zero, summary
        identity_residual = self.identity_residual_head(summary)
        semantic_summary = summary - identity_residual
        return semantic_summary, identity_residual, summary

    def identity_logits(self, latents: torch.Tensor) -> torch.Tensor:
        if self.identity_classifier is None:
            raise RuntimeError("Identity head is not enabled for this model.")
        _, identity_residual, _ = self.decompose_summary(latents)
        return self.identity_classifier(identity_residual)

    def semantic_logits(self, latents: torch.Tensor) -> torch.Tensor:
        if self.semantic_prototypes is None:
            raise RuntimeError("Semantic prototype head is not enabled for this model.")
        semantic_summary, _, _ = self.decompose_summary(latents)
        semantic_unit = F.normalize(semantic_summary, dim=-1, eps=1e-6)
        prototype_unit = F.normalize(self.semantic_prototypes, dim=-1, eps=1e-6)
        return semantic_unit @ prototype_unit.T / self.semantic_temperature

    def step_dynamics(self, z: torch.Tensor, cond_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.base_dynamics(z)
        delta = self.cond_delta(torch.cat([z, cond_embed], dim=-1))
        next_z = z + base + delta
        return next_z, delta

    def rollout_from(
        self,
        z_start: torch.Tensor,
        cond_embed: torch.Tensor,
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = z_start
        states = []
        deltas = []
        for _ in range(steps):
            current, delta = self.step_dynamics(current, cond_embed)
            states.append(current)
            deltas.append(delta)
        return torch.stack(states, dim=1), torch.stack(deltas, dim=1)

    def condition_alignment_energy(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
    ) -> torch.Tensor:
        summary, _, _ = self.decompose_summary(latents)
        if self.condition_score_mode == "energy":
            energy_input = torch.cat([summary, cond_embed], dim=-1)
            return self.condition_energy_head(energy_input).squeeze(-1)
        target = self.cond_projection(cond_embed)
        return ((summary - target) ** 2).mean(dim=-1)

    def condition_candidate_logits(
        self,
        latents: torch.Tensor,
        condition_candidates: torch.Tensor,
    ) -> torch.Tensor:
        cond_embed = self.condition_encoder(condition_candidates)
        summary, _, _ = self.decompose_summary(latents)
        if self.condition_score_mode == "energy":
            batch = summary.size(0)
            num_candidates = cond_embed.size(0)
            summary_grid = summary.unsqueeze(1).expand(batch, num_candidates, summary.size(-1))
            cond_grid = cond_embed.unsqueeze(0).expand(batch, num_candidates, cond_embed.size(-1))
            flat_input = torch.cat([summary_grid, cond_grid], dim=-1).reshape(batch * num_candidates, -1)
            energies = self.condition_energy_head(flat_input).view(batch, num_candidates)
            return -energies
        target = self.cond_projection(cond_embed)
        distances = torch.cdist(summary, target, p=2.0) ** 2
        return -distances

    def condition_logits_and_targets(
        self,
        latents: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unique_condition, inverse = torch.unique(condition, dim=0, return_inverse=True)
        logits = self.condition_candidate_logits(latents, unique_condition)
        return logits, inverse
