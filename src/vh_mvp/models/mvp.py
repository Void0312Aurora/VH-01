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


def _make_two_layer_head(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


@dataclass
class ForwardOutput:
    recon: torch.Tensor
    latents: torch.Tensor
    cond_embed: torch.Tensor


@dataclass
class BaseMeasure:
    state: torch.Tensor
    log_base_density: torch.Tensor


@dataclass
class ConditionalTilt:
    cond_embed: torch.Tensor
    log_tilt: torch.Tensor


@dataclass
class ConditionalMeasure:
    base_measure: BaseMeasure
    conditional_tilt: ConditionalTilt
    log_total_density: torch.Tensor

    @property
    def state(self) -> torch.Tensor:
        return self.base_measure.state

    @property
    def log_base_density(self) -> torch.Tensor:
        return self.base_measure.log_base_density

    @property
    def log_tilt(self) -> torch.Tensor:
        return self.conditional_tilt.log_tilt

    def normalized_weights(self, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        log_weights = self.log_total_density.squeeze(-1)
        return torch.softmax(log_weights / temperature, dim=-1)


@dataclass
class BaseGeneratorContext:
    state: torch.Tensor
    response_context: torch.Tensor | None
    tangent_structure: dict[str, torch.Tensor] | None
    base_measure: BaseMeasure


@dataclass
class BaseLocalGenerator:
    context: BaseGeneratorContext
    drift: torch.Tensor
    diffusion_matrix: torch.Tensor
    tangent_core_cov: torch.Tensor | None = None

    @property
    def state(self) -> torch.Tensor:
        return self.context.state

    @property
    def response_context(self) -> torch.Tensor | None:
        return self.context.response_context

    @property
    def tangent_structure(self) -> dict[str, torch.Tensor] | None:
        return self.context.tangent_structure

    @property
    def base_measure(self) -> BaseMeasure:
        return self.context.base_measure

    def trace(self) -> torch.Tensor:
        return self.diffusion_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


@dataclass
class ConditionalGeneratorDelta:
    cond_embed: torch.Tensor
    conditional_tilt: ConditionalTilt
    drift: torch.Tensor
    diffusion_matrix: torch.Tensor
    tangent_core_cov: torch.Tensor | None = None

    @property
    def log_tilt(self) -> torch.Tensor:
        return self.conditional_tilt.log_tilt

    def trace(self) -> torch.Tensor:
        return self.diffusion_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


@dataclass
class GeneratorContext:
    state: torch.Tensor
    cond_embed: torch.Tensor
    response_context: torch.Tensor | None
    tangent_structure: dict[str, torch.Tensor] | None
    conditional_measure: ConditionalMeasure


@dataclass
class LocalGenerator:
    context: GeneratorContext
    drift: torch.Tensor
    diffusion_matrix: torch.Tensor
    tangent_core_cov: torch.Tensor | None = None
    base_generator: BaseLocalGenerator | None = None
    conditional_delta: ConditionalGeneratorDelta | None = None

    @property
    def state(self) -> torch.Tensor:
        return self.context.state

    @property
    def cond_embed(self) -> torch.Tensor:
        return self.context.cond_embed

    @property
    def response_context(self) -> torch.Tensor | None:
        return self.context.response_context

    @property
    def tangent_structure(self) -> dict[str, torch.Tensor] | None:
        return self.context.tangent_structure

    @property
    def conditional_measure(self) -> ConditionalMeasure:
        return self.context.conditional_measure

    @property
    def tangent_projector(self) -> torch.Tensor | None:
        if self.tangent_structure is None:
            return None
        return self.tangent_structure["projector"]

    @property
    def base_measure(self) -> BaseMeasure:
        if self.base_generator is not None:
            return self.base_generator.base_measure
        return self.conditional_measure.base_measure

    def density_weights(self, temperature: float = 1.0) -> torch.Tensor:
        return self.conditional_measure.normalized_weights(temperature=temperature)

    def trace(self) -> torch.Tensor:
        return self.diffusion_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    def apply_linear(self, directions: torch.Tensor) -> torch.Tensor:
        return self.drift @ directions.T

    def apply_quadratic(self, directions: torch.Tensor) -> torch.Tensor:
        projected_state = self.state @ directions.T
        projected_drift = self.apply_linear(directions)
        projected_diffusion = torch.einsum("bde,kd,ke->bk", self.diffusion_matrix, directions, directions)
        return 2.0 * projected_state * projected_drift + projected_diffusion

    def apply_trig(self, directions: torch.Tensor, trig_scale: float) -> torch.Tensor:
        trig_scale = max(trig_scale, 1e-4)
        projected_state = self.state @ directions.T
        projected_drift = self.apply_linear(directions)
        projected_diffusion = torch.einsum("bde,kd,ke->bk", self.diffusion_matrix, directions, directions)
        return (
            trig_scale * torch.cos(trig_scale * projected_state) * projected_drift
            - 0.5 * (trig_scale**2) * torch.sin(trig_scale * projected_state) * projected_diffusion
        )

    def apply_radial(self) -> torch.Tensor:
        diffusion_diag = self.diffusion_matrix.diagonal(dim1=-2, dim2=-1)
        return 2.0 * (self.state * self.drift).sum(dim=1) + diffusion_diag.sum(dim=1)


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
        chart_hidden_dim: int = 128,
        chart_num_experts: int = 1,
        chart_mode: str = "pointwise_residual",
        chart_residual_scale: float = 0.1,
        chart_temporal_hidden_dim: int = 128,
        chart_temporal_kernel_size: int = 3,
        encoder_condition_mode: str = "residual_temporal",
        encoder_condition_hidden_dim: int = 128,
        encoder_condition_scale: float = 0.1,
        state_cov_proj_dim: int = 0,
        response_signature_dim: int = 0,
        response_context_dim: int = 0,
        tangent_dim: int = 0,
        local_measure_hidden_dim: int = 128,
        local_measure_rank: int = 8,
        local_measure_eps: float = 1e-4,
        local_diffusion_mode: str = "legacy",
        local_diffusion_geometry_mode: str = "ambient",
        local_diffusion_condition_mode: str = "joint",
        measure_density_mode: str = "joint",
    ) -> None:
        super().__init__()
        if condition_score_mode not in {"distance", "energy"}:
            raise ValueError(f"Unsupported condition_score_mode: {condition_score_mode}")
        if chart_mode not in {"pointwise_residual", "temporal_residual", "gated_temporal"}:
            raise ValueError(f"Unsupported chart_mode: {chart_mode}")
        if encoder_condition_mode not in {"none", "residual_temporal"}:
            raise ValueError(f"Unsupported encoder_condition_mode: {encoder_condition_mode}")
        if local_diffusion_mode not in {"legacy", "trace_scaled"}:
            raise ValueError(f"Unsupported local_diffusion_mode: {local_diffusion_mode}")
        if local_diffusion_geometry_mode not in {"ambient", "tangent"}:
            raise ValueError(f"Unsupported local_diffusion_geometry_mode: {local_diffusion_geometry_mode}")
        if chart_num_experts <= 0:
            raise ValueError("chart_num_experts must be positive")
        if local_diffusion_condition_mode not in {"joint", "base_only"}:
            raise ValueError(f"Unsupported local_diffusion_condition_mode: {local_diffusion_condition_mode}")
        if measure_density_mode not in {"joint", "tilted"}:
            raise ValueError(f"Unsupported measure_density_mode: {measure_density_mode}")
        if tangent_dim < 0 or tangent_dim > latent_dim:
            raise ValueError("tangent_dim must be between 0 and latent_dim")
        if local_diffusion_geometry_mode == "tangent" and tangent_dim <= 0:
            raise ValueError("tangent_dim must be positive when local_diffusion_geometry_mode='tangent'")
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.condition_score_mode = condition_score_mode
        self.identity_num_classes = identity_num_classes
        self.semantic_num_classes = semantic_num_classes
        self.semantic_temperature = semantic_temperature
        self.chart_num_experts = chart_num_experts
        self.chart_mode = chart_mode
        self.chart_residual_scale = chart_residual_scale
        self.encoder_condition_mode = encoder_condition_mode
        self.encoder_condition_scale = encoder_condition_scale
        self.state_cov_proj_dim = state_cov_proj_dim
        self.response_context_dim = response_context_dim
        self.tangent_dim = tangent_dim
        self.local_measure_rank = local_measure_rank
        self.local_measure_eps = local_measure_eps
        self.local_diffusion_mode = local_diffusion_mode
        self.local_diffusion_geometry_mode = local_diffusion_geometry_mode
        self.local_diffusion_condition_mode = local_diffusion_condition_mode
        self.measure_density_mode = measure_density_mode
        self.condition_encoder = ConditionEncoder(cond_dim)
        self.frame_encoder = FrameEncoder(channels, base_channels, latent_dim)
        self.frame_decoder = FrameDecoder(channels, base_channels, latent_dim, cond_dim)
        self.base_dynamics = DynamicsMLP(latent_dim, hidden_dim, latent_dim)
        self.cond_delta = DynamicsMLP(latent_dim + cond_dim, hidden_dim, latent_dim)
        self.cond_projection = nn.Linear(cond_dim, latent_dim)
        temporal_padding = chart_temporal_kernel_size // 2
        if encoder_condition_mode == "residual_temporal":
            self.encoder_condition_temporal_net = nn.Sequential(
                nn.Conv1d(
                    latent_dim + cond_dim,
                    encoder_condition_hidden_dim,
                    kernel_size=chart_temporal_kernel_size,
                    padding=temporal_padding,
                ),
                nn.SiLU(),
                nn.Conv1d(
                    encoder_condition_hidden_dim,
                    encoder_condition_hidden_dim,
                    kernel_size=chart_temporal_kernel_size,
                    padding=temporal_padding,
                ),
                nn.SiLU(),
                nn.Conv1d(encoder_condition_hidden_dim, latent_dim, kernel_size=1),
            )
            self.encoder_condition_gate_head = nn.Sequential(
                nn.Linear(latent_dim + cond_dim, encoder_condition_hidden_dim),
                nn.SiLU(),
                nn.Linear(encoder_condition_hidden_dim, latent_dim),
            )
            self.encoder_condition_bias_head = nn.Sequential(
                nn.Linear(cond_dim, encoder_condition_hidden_dim),
                nn.SiLU(),
                nn.Linear(encoder_condition_hidden_dim, latent_dim),
            )
        else:
            self.encoder_condition_temporal_net = None
            self.encoder_condition_gate_head = None
            self.encoder_condition_bias_head = None
        if state_cov_proj_dim > 0:
            self.state_cov_proj = nn.Linear(latent_dim, state_cov_proj_dim, bias=False)
            summary_feature_dim = latent_dim * 5 + state_cov_proj_dim * state_cov_proj_dim
        else:
            self.state_cov_proj = None
            summary_feature_dim = latent_dim * 5
        trajectory_point_feature_dim = chart_hidden_dim * 2 + latent_dim * 3
        self.trajectory_point_temporal_net = nn.Sequential(
            nn.Conv1d(
                latent_dim,
                chart_hidden_dim,
                kernel_size=chart_temporal_kernel_size,
                padding=temporal_padding,
            ),
            nn.SiLU(),
            nn.Conv1d(
                chart_hidden_dim,
                chart_hidden_dim,
                kernel_size=chart_temporal_kernel_size,
                padding=temporal_padding,
            ),
            nn.SiLU(),
        )
        self.trajectory_point_head = _make_two_layer_head(
            trajectory_point_feature_dim,
            chart_hidden_dim,
            latent_dim,
        )
        if response_context_dim > 0:
            if response_signature_dim <= 0:
                raise ValueError("response_signature_dim must be positive when response_context_dim > 0")
            self.response_context_head = nn.Sequential(
                nn.Linear(response_signature_dim, local_measure_hidden_dim),
                nn.SiLU(),
                nn.Linear(local_measure_hidden_dim, response_context_dim),
            )
        else:
            self.response_context_head = None
        if tangent_dim > 0:
            self.tangent_frame_head = _make_two_layer_head(
                latent_dim * 2,
                local_measure_hidden_dim,
                latent_dim * tangent_dim,
            )
        else:
            self.tangent_frame_head = None
        if chart_num_experts == 1:
            self.chart_state_head = _make_two_layer_head(summary_feature_dim, chart_hidden_dim, latent_dim)
            self.chart_state_experts = None
            self.chart_state_gate_head = None
        else:
            self.chart_state_head = None
            self.chart_state_experts = nn.ModuleList(
                _make_two_layer_head(summary_feature_dim, chart_hidden_dim, latent_dim) for _ in range(chart_num_experts)
            )
            self.chart_state_gate_head = nn.Sequential(
                nn.Linear(summary_feature_dim, chart_hidden_dim),
                nn.SiLU(),
                nn.Linear(chart_hidden_dim, chart_num_experts),
            )
        self.chart_latent_head = nn.Sequential(
            nn.Linear(latent_dim, chart_hidden_dim),
            nn.SiLU(),
            nn.Linear(chart_hidden_dim, latent_dim),
        )
        self.chart_temporal_net = nn.Sequential(
            nn.Conv1d(latent_dim, chart_temporal_hidden_dim, kernel_size=chart_temporal_kernel_size, padding=temporal_padding),
            nn.SiLU(),
            nn.Conv1d(
                chart_temporal_hidden_dim,
                latent_dim,
                kernel_size=chart_temporal_kernel_size,
                padding=temporal_padding,
            ),
        )
        self.chart_temporal_gate_head = nn.Sequential(
            nn.Linear(latent_dim * 3, chart_temporal_hidden_dim),
            nn.SiLU(),
            nn.Linear(chart_temporal_hidden_dim, latent_dim),
        )
        self.condition_energy_head = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, energy_hidden_dim),
            nn.SiLU(),
            nn.Linear(energy_hidden_dim, energy_hidden_dim),
            nn.SiLU(),
            nn.Linear(energy_hidden_dim, 1),
        )
        diffusion_context_dim = latent_dim + response_context_dim
        if local_diffusion_condition_mode == "joint":
            diffusion_context_dim += cond_dim
        factor_out_dim = latent_dim * local_measure_rank
        if local_diffusion_geometry_mode == "tangent":
            factor_out_dim = tangent_dim * local_measure_rank
        self.local_diffusion_factor_head = nn.Sequential(
            nn.Linear(diffusion_context_dim, local_measure_hidden_dim),
            nn.SiLU(),
            nn.Linear(local_measure_hidden_dim, local_measure_hidden_dim),
            nn.SiLU(),
            nn.Linear(local_measure_hidden_dim, factor_out_dim),
        )
        self.local_diffusion_scale_head = nn.Sequential(
            nn.Linear(diffusion_context_dim, local_measure_hidden_dim),
            nn.SiLU(),
            nn.Linear(local_measure_hidden_dim, local_measure_hidden_dim),
            nn.SiLU(),
            nn.Linear(local_measure_hidden_dim, 1),
        )
        if measure_density_mode == "joint":
            self.measure_log_density_head = _make_two_layer_head(latent_dim + cond_dim, local_measure_hidden_dim, 1)
            self.measure_base_log_density_head = None
            self.measure_tilt_head = None
        else:
            self.measure_log_density_head = None
            self.measure_base_log_density_head = _make_two_layer_head(latent_dim, local_measure_hidden_dim, 1)
            self.measure_tilt_head = _make_two_layer_head(latent_dim + cond_dim, local_measure_hidden_dim, 1)
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

    def _encode_video_base(self, video: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = video.shape
        flat = video.view(batch * steps, channels, height, width)
        latents = self.frame_encoder(flat)
        return latents.view(batch, steps, self.latent_dim)

    def _encode_video_with_condition(
        self,
        base_latents: torch.Tensor,
        cond_embed: torch.Tensor,
    ) -> torch.Tensor:
        if self.encoder_condition_mode == "none":
            return base_latents
        if (
            self.encoder_condition_temporal_net is None
            or self.encoder_condition_gate_head is None
            or self.encoder_condition_bias_head is None
        ):
            raise RuntimeError("Conditional encoder modules are unexpectedly missing.")
        batch, steps, _ = base_latents.shape
        cond_seq = cond_embed.unsqueeze(1).expand(batch, steps, cond_embed.size(-1))
        temporal_input = torch.cat([base_latents, cond_seq], dim=-1).transpose(1, 2)
        residual = self.encoder_condition_temporal_net(temporal_input).transpose(1, 2)
        gate = torch.sigmoid(
            self.encoder_condition_gate_head(
                torch.cat([base_latents, cond_seq], dim=-1).reshape(batch * steps, -1)
            )
        ).view(batch, steps, self.latent_dim)
        cond_bias = self.encoder_condition_bias_head(cond_embed).unsqueeze(1)
        return base_latents + self.encoder_condition_scale * gate * (residual + cond_bias)

    def encode_video(
        self,
        video: torch.Tensor,
        cond_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_latents = self._encode_video_base(video)
        if cond_embed is None:
            return base_latents
        return self._encode_video_with_condition(base_latents, cond_embed)

    def decode_video(self, latents: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        batch, steps, latent_dim = latents.shape
        cond_seq = cond_embed.unsqueeze(1).expand(batch, steps, cond_embed.size(-1))
        flat_latents = latents.reshape(batch * steps, latent_dim)
        flat_cond = cond_seq.reshape(batch * steps, cond_embed.size(-1))
        recon = self.frame_decoder(flat_latents, flat_cond)
        return recon.view(batch, steps, *recon.shape[1:])

    def forward(self, video: torch.Tensor, condition: torch.Tensor) -> ForwardOutput:
        cond_embed = self.condition_encoder(condition)
        latents = self.encode_video(video, cond_embed=cond_embed)
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

    def chart_residual(self, latents: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, steps, latent_dim = latents.shape
        flat = latents.reshape(batch * steps, latent_dim)
        pointwise_residual = self.chart_latent_head(flat).view(batch, steps, latent_dim)
        aux = {
            "chart_gate_mean": latents.new_tensor(0.0),
            "chart_gate_abs_mean": latents.new_tensor(0.0),
        }
        if self.chart_mode == "temporal_residual":
            temporal_residual = self.chart_temporal_net(latents.transpose(1, 2)).transpose(1, 2)
            residual = pointwise_residual + temporal_residual
        elif self.chart_mode == "gated_temporal":
            temporal_residual = self.chart_temporal_net(latents.transpose(1, 2)).transpose(1, 2)
            gate_input = torch.cat([latents, pointwise_residual, temporal_residual], dim=-1)
            gate = torch.sigmoid(self.chart_temporal_gate_head(gate_input.reshape(batch * steps, -1))).view(
                batch, steps, latent_dim
            )
            residual = pointwise_residual + gate * temporal_residual
            aux["chart_gate_mean"] = gate.mean()
            aux["chart_gate_abs_mean"] = gate.abs().mean()
        else:
            residual = pointwise_residual
        return residual, aux

    def chart_latents(self, latents: torch.Tensor) -> torch.Tensor:
        residual, _ = self.chart_residual(latents)
        return latents + self.chart_residual_scale * residual

    def chart_diagnostics(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        residual, aux = self.chart_residual(latents)
        chart_latents = latents + self.chart_residual_scale * residual
        shift = chart_latents - latents
        aux["chart_shift_l2"] = shift.square().mean().sqrt()
        aux["chart_shift_abs_mean"] = shift.abs().mean()
        return aux

    def _trajectory_state_features(self, latents: torch.Tensor) -> torch.Tensor:
        chart_latents = self.chart_latents(latents)
        summary = chart_latents.mean(dim=1)
        batch, steps, latent_dim = latents.shape
        if steps < 2:
            zero = torch.zeros_like(summary)
            features_list = [summary, zero, zero, zero, zero]
            if self.state_cov_proj is not None:
                cov_zero = torch.zeros(
                    batch,
                    self.state_cov_proj_dim * self.state_cov_proj_dim,
                    device=latents.device,
                    dtype=latents.dtype,
                )
                features_list.append(cov_zero)
            features = torch.cat(features_list, dim=-1)
            return features

        delta = chart_latents[:, 1:] - chart_latents[:, :-1]
        velocity = delta.mean(dim=1)
        accel = (
            (delta[:, 1:] - delta[:, :-1]).mean(dim=1)
            if delta.size(1) > 1
            else torch.zeros(batch, latent_dim, device=latents.device, dtype=latents.dtype)
        )
        endpoint = chart_latents[:, -1] - chart_latents[:, 0]
        spread = chart_latents.std(dim=1, unbiased=False)
        features_list = [summary, velocity, accel, endpoint, spread]
        if self.state_cov_proj is not None:
            projected_delta = self.state_cov_proj(delta.reshape(batch * (steps - 1), latent_dim)).view(
                batch,
                steps - 1,
                self.state_cov_proj_dim,
            )
            centered_proj = projected_delta - projected_delta.mean(dim=1, keepdim=True)
            projected_cov = torch.matmul(centered_proj.transpose(1, 2), centered_proj) / float(centered_proj.size(1))
            features_list.append(projected_cov.reshape(batch, -1))
        features = torch.cat(features_list, dim=-1)
        return features

    def _trajectory_state_from_features(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux = {
            "chart_expert_entropy": features.new_tensor(0.0),
            "chart_expert_max_weight": features.new_tensor(1.0),
        }
        if self.chart_num_experts == 1:
            if self.chart_state_head is None:
                raise RuntimeError("chart_state_head is unexpectedly missing.")
            return self.chart_state_head(features), aux
        if self.chart_state_gate_head is None or self.chart_state_experts is None:
            raise RuntimeError("Chart expert modules are unexpectedly missing.")
        gate_logits = self.chart_state_gate_head(features)
        gate = torch.softmax(gate_logits, dim=-1)
        expert_outputs = torch.stack([expert(features) for expert in self.chart_state_experts], dim=1)
        aux["chart_expert_entropy"] = -(gate * gate.clamp_min(1e-12).log()).sum(dim=-1).mean()
        aux["chart_expert_max_weight"] = gate.max(dim=-1).values.mean()
        return torch.einsum("be,bed->bd", gate, expert_outputs), aux

    def _trajectory_point_features(self, latents: torch.Tensor) -> torch.Tensor:
        chart_latents = self.chart_latents(latents)
        temporal_features = self.trajectory_point_temporal_net(chart_latents.transpose(1, 2))
        temporal_mean = temporal_features.mean(dim=-1)
        temporal_max = temporal_features.amax(dim=-1)
        start = chart_latents[:, 0]
        end = chart_latents[:, -1]
        summary = chart_latents.mean(dim=1)
        return torch.cat([temporal_mean, temporal_max, start, end, summary], dim=-1)

    def trajectory_point(self, latents: torch.Tensor) -> torch.Tensor:
        features = self._trajectory_point_features(latents)
        return self.trajectory_point_head(features)

    def trajectory_point_diagnostics(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        point = self.trajectory_point(latents)
        return {
            "trajectory_point_norm": point.norm(dim=-1).mean(),
            "trajectory_point_std": point.std(dim=0, unbiased=False).mean(),
        }

    def trajectory_summary_context(self, latents: torch.Tensor) -> torch.Tensor:
        features = self._trajectory_state_features(latents)
        summary_context, _ = self._trajectory_state_from_features(features)
        return summary_context

    def trajectory_summary_diagnostics(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._trajectory_state_features(latents)
        _, aux = self._trajectory_state_from_features(features)
        return aux

    def trajectory_state(self, latents: torch.Tensor) -> torch.Tensor:
        # Compatibility alias: active geometry paths now treat the trajectory point
        # as the default state anchor rather than the summary-only context.
        return self.trajectory_point(latents)

    def trajectory_state_diagnostics(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        aux = self.trajectory_summary_diagnostics(latents)
        aux.update(self.trajectory_point_diagnostics(latents))
        return aux

    def trajectory_tangent_frame(
        self,
        latents: torch.Tensor,
        *,
        point: torch.Tensor | None = None,
        summary_context: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.tangent_frame_head is None:
            return None
        if point is None:
            point = self.trajectory_point(latents)
        if summary_context is None:
            summary_context = self.trajectory_summary_context(latents)
        features = torch.cat([point, summary_context], dim=-1)
        raw_frame = self.tangent_frame_head(features).view(latents.size(0), self.latent_dim, self.tangent_dim)
        frame, _ = torch.linalg.qr(raw_frame.float(), mode="reduced")
        return frame[:, :, : self.tangent_dim].to(dtype=raw_frame.dtype)

    def trajectory_tangent_projector(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor | None:
        frame = self.trajectory_tangent_frame(latents)
        if frame is None:
            return None
        return frame @ frame.transpose(-1, -2)

    def trajectory_tangent_diagnostics(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = latents.new_tensor(0.0)
        frame = self.trajectory_tangent_frame(latents)
        if frame is None:
            return {
                "tangent_frame_orthogonality": zero,
                "tangent_projector_trace": zero,
            }
        gram = frame.transpose(-1, -2) @ frame
        eye = torch.eye(self.tangent_dim, device=latents.device, dtype=latents.dtype).unsqueeze(0)
        return {
            "tangent_frame_orthogonality": (gram - eye).square().mean().sqrt(),
            "tangent_projector_trace": gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean(),
        }

    def local_tangent_structure(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        response_context: torch.Tensor | None = None,
        *,
        state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if self.local_diffusion_geometry_mode != "tangent":
            return None
        point = state if state is not None else self.trajectory_point(latents)
        frame = self.trajectory_tangent_frame(latents, point=point)
        if frame is None:
            raise RuntimeError("Tangent frame is required for tangent-geometry diffusion mode.")
        context = self.local_diffusion_context(
            latents,
            cond_embed,
            response_context=response_context,
            state=state,
        )
        core_factor = self.local_diffusion_factor_head(context).view(
            latents.size(0),
            self.tangent_dim,
            self.local_measure_rank,
        )
        if self.local_diffusion_mode == "trace_scaled":
            raw_norm = core_factor.square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
            scale = torch.exp(self.local_diffusion_scale_head(context)).clamp_min(self.local_measure_eps)
            core_factor = core_factor / raw_norm * scale.sqrt().unsqueeze(-1)
        eye = torch.eye(self.tangent_dim, device=latents.device, dtype=latents.dtype).unsqueeze(0)
        core_cov = core_factor @ core_factor.transpose(-1, -2) + self.local_measure_eps * eye
        return {
            "frame": frame,
            "projector": frame @ frame.transpose(-1, -2),
            "core_factor": core_factor,
            "core_cov": core_cov,
        }

    def local_measure_context(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        include_condition: bool = True,
    ) -> torch.Tensor:
        if state is None:
            state = self.trajectory_point(latents)
        if not include_condition:
            return state
        return torch.cat([state, cond_embed], dim=-1)

    def local_diffusion_context(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        response_context: torch.Tensor | None = None,
        *,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        include_condition = self.local_diffusion_condition_mode == "joint"
        context = self.local_measure_context(latents, cond_embed, state=state, include_condition=include_condition)
        if self.response_context_head is None:
            return context
        if response_context is None:
            response_features = context.new_zeros(context.size(0), self.response_context_dim)
        else:
            response_features = self.response_context_head(response_context)
        return torch.cat([context, response_features], dim=-1)

    def step_dynamics(self, z: torch.Tensor, cond_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_base, _ = self.base_step_dynamics(z)
        delta = self.conditional_step_delta(z, cond_embed)
        next_z = next_base + delta
        return next_z, delta

    def zero_cond_embed(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.cond_dim, device=device, dtype=dtype)

    def base_step_dynamics(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.base_dynamics(z)
        return z + base, base

    def conditional_step_delta(self, z: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        return self.cond_delta(torch.cat([z, cond_embed], dim=-1))

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

    def trajectory_drift(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
    ) -> torch.Tensor:
        return self.trajectory_base_drift(latents) + self.trajectory_conditional_drift_delta(latents, cond_embed)

    def trajectory_base_drift(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        if latents.size(1) < 2:
            return torch.zeros(latents.size(0), latents.size(-1), device=latents.device, dtype=latents.dtype)
        current = latents[:, :-1].reshape(-1, latents.size(-1))
        current_chart = self.chart_latents(current.unsqueeze(1)).squeeze(1)
        pred_next, _ = self.base_step_dynamics(current)
        pred_next_chart = self.chart_latents(pred_next.unsqueeze(1)).squeeze(1)
        drift = (pred_next_chart - current_chart).view(latents.size(0), latents.size(1) - 1, latents.size(-1))
        return drift.mean(dim=1)

    def trajectory_conditional_drift_delta(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
    ) -> torch.Tensor:
        if latents.size(1) < 2:
            return torch.zeros(latents.size(0), latents.size(-1), device=latents.device, dtype=latents.dtype)
        current = latents[:, :-1].reshape(-1, latents.size(-1))
        cond_seq = cond_embed.unsqueeze(1).expand(-1, latents.size(1) - 1, -1).reshape(-1, cond_embed.size(-1))
        base_next, _ = self.base_step_dynamics(current)
        cond_delta = self.conditional_step_delta(current, cond_seq)
        full_next = base_next + cond_delta
        base_next_chart = self.chart_latents(base_next.unsqueeze(1)).squeeze(1)
        full_next_chart = self.chart_latents(full_next.unsqueeze(1)).squeeze(1)
        drift = (full_next_chart - base_next_chart).view(latents.size(0), latents.size(1) - 1, latents.size(-1))
        return drift.mean(dim=1)

    def local_diffusion_factor(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        response_context: torch.Tensor | None = None,
        *,
        state: torch.Tensor | None = None,
        tangent_structure: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.local_diffusion_geometry_mode == "tangent":
            if tangent_structure is None:
                tangent_structure = self.local_tangent_structure(
                    latents,
                    cond_embed,
                    response_context=response_context,
                    state=state,
                )
            if tangent_structure is None:
                raise RuntimeError("Tangent structure is required for tangent-geometry diffusion mode.")
            return tangent_structure["frame"] @ tangent_structure["core_factor"]
        context = self.local_diffusion_context(
            latents,
            cond_embed,
            response_context=response_context,
            state=state,
        )
        factor = self.local_diffusion_factor_head(context).view(
            latents.size(0),
            self.latent_dim,
            self.local_measure_rank,
        )
        if self.local_diffusion_mode == "trace_scaled":
            raw_norm = factor.square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
            scale = torch.exp(self.local_diffusion_scale_head(context)).clamp_min(self.local_measure_eps)
            factor = factor / raw_norm * scale.sqrt().unsqueeze(-1)
        return factor

    def local_tangent_covariance(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        response_context: torch.Tensor | None = None,
        *,
        state: torch.Tensor | None = None,
        tangent_structure: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if self.local_diffusion_geometry_mode != "tangent":
            return None
        if tangent_structure is None:
            tangent_structure = self.local_tangent_structure(
                latents,
                cond_embed,
                response_context=response_context,
                state=state,
            )
        if tangent_structure is None:
            return None
        return tangent_structure["core_cov"]

    def local_diffusion_matrix(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        response_context: torch.Tensor | None = None,
        *,
        state: torch.Tensor | None = None,
        tangent_structure: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.local_diffusion_geometry_mode == "tangent":
            if tangent_structure is None:
                tangent_structure = self.local_tangent_structure(
                    latents,
                    cond_embed,
                    response_context=response_context,
                    state=state,
                )
            if tangent_structure is None:
                raise RuntimeError("Tangent structure is required for tangent-geometry diffusion mode.")
            frame = tangent_structure["frame"]
            core_cov = tangent_structure["core_cov"]
            return frame @ core_cov @ frame.transpose(-1, -2)
        factor = self.local_diffusion_factor(
            latents,
            cond_embed,
            response_context=response_context,
            state=state,
            tangent_structure=tangent_structure,
        )
        diffusion = factor @ factor.transpose(-1, -2)
        eye = torch.eye(
            self.latent_dim,
            device=diffusion.device,
            dtype=diffusion.dtype,
        ).unsqueeze(0)
        return diffusion + self.local_measure_eps * eye

    def local_diffusion_diag(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        response_context: torch.Tensor | None = None,
        *,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diffusion = self.local_diffusion_matrix(
            latents,
            cond_embed,
            response_context=response_context,
            state=state,
        )
        return diffusion.diagonal(dim1=-2, dim2=-1)

    def base_measure(
        self,
        latents: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        cond_embed: torch.Tensor | None = None,
    ) -> BaseMeasure:
        if state is None:
            state = self.trajectory_point(latents)
        if self.measure_density_mode == "joint":
            if cond_embed is None:
                cond_embed = self.zero_cond_embed(state.size(0), device=state.device, dtype=state.dtype)
            if self.measure_log_density_head is None:
                raise RuntimeError("measure_log_density_head is unexpectedly missing.")
            joint_context = self.local_measure_context(latents, cond_embed, state=state, include_condition=True)
            log_base_density = self.measure_log_density_head(joint_context)
        else:
            if self.measure_base_log_density_head is None:
                raise RuntimeError("measure_base_log_density_head is unexpectedly missing.")
            log_base_density = self.measure_base_log_density_head(state)
        return BaseMeasure(
            state=state,
            log_base_density=log_base_density,
        )

    def conditional_tilt(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
    ) -> ConditionalTilt:
        if state is None:
            state = self.trajectory_point(latents)
        zero = state.new_zeros(state.size(0), 1)
        if self.measure_density_mode == "joint":
            if self.measure_log_density_head is None:
                raise RuntimeError("measure_log_density_head is unexpectedly missing.")
            zero_cond_embed = torch.zeros_like(cond_embed)
            full_context = self.local_measure_context(latents, cond_embed, state=state, include_condition=True)
            base_context = self.local_measure_context(latents, zero_cond_embed, state=state, include_condition=True)
            log_tilt = self.measure_log_density_head(full_context) - self.measure_log_density_head(base_context)
        else:
            if self.measure_tilt_head is None:
                raise RuntimeError("measure_tilt_head is unexpectedly missing.")
            tilt_context = self.local_measure_context(latents, cond_embed, state=state, include_condition=True)
            log_tilt = self.measure_tilt_head(tilt_context)
        return ConditionalTilt(
            cond_embed=cond_embed,
            log_tilt=log_tilt,
        )

    def conditional_measure(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
    ) -> ConditionalMeasure:
        if state is None:
            state = self.trajectory_point(latents)
        base_measure = self.base_measure(latents, state=state, cond_embed=cond_embed)
        conditional_tilt = self.conditional_tilt(latents, cond_embed, state=state)
        return ConditionalMeasure(
            base_measure=base_measure,
            conditional_tilt=conditional_tilt,
            log_total_density=base_measure.log_base_density + conditional_tilt.log_tilt,
        )

    def local_generator(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        response_context: torch.Tensor | None = None,
        *,
        state: torch.Tensor | None = None,
    ) -> LocalGenerator:
        if state is None:
            state = self.trajectory_point(latents)
        zero_cond_embed = torch.zeros_like(cond_embed)
        base_measure = self.base_measure(latents, state=state, cond_embed=zero_cond_embed)
        conditional_tilt = self.conditional_tilt(latents, cond_embed, state=state)
        conditional_measure = ConditionalMeasure(
            base_measure=base_measure,
            conditional_tilt=conditional_tilt,
            log_total_density=base_measure.log_base_density + conditional_tilt.log_tilt,
        )
        tangent_structure = self.local_tangent_structure(
            latents,
            cond_embed,
            response_context=response_context,
            state=state,
        )
        base_tangent_structure = self.local_tangent_structure(
            latents,
            zero_cond_embed,
            response_context=response_context,
            state=state,
        )
        diffusion_matrix = self.local_diffusion_matrix(
            latents,
            cond_embed,
            response_context=response_context,
            state=state,
            tangent_structure=tangent_structure,
        )
        base_diffusion_matrix = self.local_diffusion_matrix(
            latents,
            zero_cond_embed,
            response_context=response_context,
            state=state,
            tangent_structure=base_tangent_structure,
        )
        tangent_core_cov = self.local_tangent_covariance(
            latents,
            cond_embed,
            response_context=response_context,
            state=state,
            tangent_structure=tangent_structure,
        )
        base_tangent_core_cov = self.local_tangent_covariance(
            latents,
            zero_cond_embed,
            response_context=response_context,
            state=state,
            tangent_structure=base_tangent_structure,
        )
        base_drift = self.trajectory_base_drift(latents)
        conditional_drift_delta = self.trajectory_conditional_drift_delta(latents, cond_embed)
        context = GeneratorContext(
            state=state,
            cond_embed=cond_embed,
            response_context=response_context,
            tangent_structure=tangent_structure,
            conditional_measure=conditional_measure,
        )
        base_context = BaseGeneratorContext(
            state=state,
            response_context=response_context,
            tangent_structure=base_tangent_structure,
            base_measure=base_measure,
        )
        base_generator = BaseLocalGenerator(
            context=base_context,
            drift=base_drift,
            diffusion_matrix=base_diffusion_matrix,
            tangent_core_cov=base_tangent_core_cov,
        )
        conditional_delta = ConditionalGeneratorDelta(
            cond_embed=cond_embed,
            conditional_tilt=conditional_tilt,
            drift=conditional_drift_delta,
            diffusion_matrix=diffusion_matrix - base_diffusion_matrix,
            tangent_core_cov=(
                None
                if tangent_core_cov is None or base_tangent_core_cov is None
                else tangent_core_cov - base_tangent_core_cov
            ),
        )
        return LocalGenerator(
            context=context,
            drift=base_drift + conditional_drift_delta,
            diffusion_matrix=diffusion_matrix,
            tangent_core_cov=tangent_core_cov,
            base_generator=base_generator,
            conditional_delta=conditional_delta,
        )

    def measure_log_density_components(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        measure = self.conditional_measure(latents, cond_embed, state=state)
        return measure.log_base_density, measure.log_tilt, measure.log_total_density

    def measure_log_density(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.conditional_measure(latents, cond_embed, state=state).log_total_density

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
