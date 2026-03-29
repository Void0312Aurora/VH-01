from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from vh_mvp.data import CONDITION_CARDINALITIES, CONDITION_KEYS


@dataclass
class BaselineOutput:
    recon: torch.Tensor
    future: torch.Tensor


class DiscreteConditionEncoder(nn.Module):
    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        if cond_dim <= 0:
            raise ValueError("cond_dim must be positive.")
        per_key_dim = max(cond_dim // len(CONDITION_KEYS), 4)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(CONDITION_CARDINALITIES[key], per_key_dim) for key in CONDITION_KEYS]
        )
        self.proj = nn.Sequential(
            nn.Linear(per_key_dim * len(CONDITION_KEYS), cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        parts = [embedding(condition[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        return self.proj(torch.cat(parts, dim=-1))


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gates = self.gates(torch.cat([x, h_prev], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConditionalConvLSTMBaseline(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        image_size: int = 32,
        cond_dim: int = 64,
        base_channels: int = 32,
        latent_channels: int = 64,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4.")
        self.channels = channels
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.latent_size = image_size // 4

        self.condition_encoder = DiscreteConditionEncoder(cond_dim)

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.frame_decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.cond_to_latent = nn.Linear(cond_dim, latent_channels * 2)
        self.cond_to_input = nn.Linear(cond_dim, latent_channels)
        self.init_hidden = nn.Conv2d(latent_channels * 2, hidden_channels, kernel_size=3, padding=1)
        self.transition = ConvLSTMCell(
            input_channels=latent_channels * 2,
            hidden_channels=hidden_channels,
            kernel_size=3,
        )
        self.hidden_to_delta = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def encode_frames(self, video: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = video.shape
        flat = video.view(batch * steps, channels, height, width)
        latent = self.frame_encoder(flat)
        return latent.view(batch, steps, self.latent_channels, self.latent_size, self.latent_size)

    def _conditioned_latent(self, latent: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.cond_to_latent(cond_embed).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return latent * (1.0 + gamma) + beta

    def decode_latents(self, latents: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = latents.shape
        cond_embed = cond_embed.unsqueeze(1).expand(batch, steps, -1)
        conditioned = self._conditioned_latent(
            latents.reshape(batch * steps, channels, height, width),
            cond_embed.reshape(batch * steps, cond_embed.size(-1)),
        )
        recon = self.frame_decoder(conditioned)
        return recon.view(batch, steps, self.channels, self.image_size, self.image_size)

    def rollout_from_first(
        self,
        first_latent: torch.Tensor,
        cond_embed: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        batch = first_latent.size(0)
        cond_plane = self.cond_to_input(cond_embed).view(batch, self.latent_channels, 1, 1).expand(
            -1, -1, self.latent_size, self.latent_size
        )
        h = torch.tanh(self.init_hidden(torch.cat([first_latent, cond_plane], dim=1)))
        c = torch.zeros_like(h)
        current = first_latent
        outputs = []
        for _ in range(steps):
            h, c = self.transition(torch.cat([current, cond_plane], dim=1), (h, c))
            delta = self.hidden_to_delta(h)
            current = current + delta
            outputs.append(current)
        if not outputs:
            return first_latent.new_zeros(batch, 0, self.latent_channels, self.latent_size, self.latent_size)
        return torch.stack(outputs, dim=1)

    def forward(self, video: torch.Tensor, condition: torch.Tensor) -> BaselineOutput:
        cond_embed = self.condition_encoder(condition)
        latents = self.encode_frames(video)
        recon = self.decode_latents(latents, cond_embed)
        future_latents = self.rollout_from_first(latents[:, 0], cond_embed, steps=video.size(1) - 1)
        future = self.decode_latents(future_latents, cond_embed)
        return BaselineOutput(recon=recon, future=future)
