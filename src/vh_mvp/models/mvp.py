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
        self.condition_score_mode = condition_score_mode
        self.identity_num_classes = identity_num_classes
        self.semantic_num_classes = semantic_num_classes
        self.semantic_temperature = semantic_temperature
        self.chart_num_experts = chart_num_experts
        self.chart_mode = chart_mode
        self.chart_residual_scale = chart_residual_scale
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
        if state_cov_proj_dim > 0:
            self.state_cov_proj = nn.Linear(latent_dim, state_cov_proj_dim, bias=False)
            state_feature_dim = latent_dim * 5 + state_cov_proj_dim * state_cov_proj_dim
        else:
            self.state_cov_proj = None
            state_feature_dim = latent_dim * 5
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
                state_feature_dim,
                local_measure_hidden_dim,
                latent_dim * tangent_dim,
            )
        else:
            self.tangent_frame_head = None
        if chart_num_experts == 1:
            self.chart_state_head = _make_two_layer_head(state_feature_dim, chart_hidden_dim, latent_dim)
            self.chart_state_experts = None
            self.chart_state_gate_head = None
        else:
            self.chart_state_head = None
            self.chart_state_experts = nn.ModuleList(
                _make_two_layer_head(state_feature_dim, chart_hidden_dim, latent_dim) for _ in range(chart_num_experts)
            )
            self.chart_state_gate_head = nn.Sequential(
                nn.Linear(state_feature_dim, chart_hidden_dim),
                nn.SiLU(),
                nn.Linear(chart_hidden_dim, chart_num_experts),
            )
        self.chart_latent_head = nn.Sequential(
            nn.Linear(latent_dim, chart_hidden_dim),
            nn.SiLU(),
            nn.Linear(chart_hidden_dim, latent_dim),
        )
        temporal_padding = chart_temporal_kernel_size // 2
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

    def trajectory_state(self, latents: torch.Tensor) -> torch.Tensor:
        features = self._trajectory_state_features(latents)
        state, _ = self._trajectory_state_from_features(features)
        return state

    def trajectory_state_diagnostics(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._trajectory_state_features(latents)
        _, aux = self._trajectory_state_from_features(features)
        return aux

    def trajectory_tangent_frame(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.tangent_frame_head is None:
            return None
        features = self._trajectory_state_features(latents)
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
        frame = self.trajectory_tangent_frame(latents)
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
            state = self.trajectory_state(latents)
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

    def trajectory_drift(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
    ) -> torch.Tensor:
        if latents.size(1) < 2:
            return torch.zeros(latents.size(0), latents.size(-1), device=latents.device, dtype=latents.dtype)
        current = latents[:, :-1].reshape(-1, latents.size(-1))
        cond_seq = cond_embed.unsqueeze(1).expand(-1, latents.size(1) - 1, -1).reshape(-1, cond_embed.size(-1))
        pred_next, _ = self.step_dynamics(current, cond_seq)
        current_chart = self.chart_latents(current.unsqueeze(1)).squeeze(1)
        pred_next_chart = self.chart_latents(pred_next.unsqueeze(1)).squeeze(1)
        drift = (pred_next_chart - current_chart).view(latents.size(0), latents.size(1) - 1, latents.size(-1))
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

    def measure_log_density_components(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state is None:
            state = self.trajectory_state(latents)
        zero = state.new_zeros(state.size(0), 1)
        if self.measure_density_mode == "joint":
            if self.measure_log_density_head is None:
                raise RuntimeError("measure_log_density_head is unexpectedly missing.")
            joint_context = self.local_measure_context(latents, cond_embed, state=state, include_condition=True)
            total = self.measure_log_density_head(joint_context)
            return total, zero, total
        if self.measure_base_log_density_head is None or self.measure_tilt_head is None:
            raise RuntimeError("Tilted measure density heads are unexpectedly missing.")
        base = self.measure_base_log_density_head(state)
        tilt_context = self.local_measure_context(latents, cond_embed, state=state, include_condition=True)
        tilt = self.measure_tilt_head(tilt_context)
        return base, tilt, base + tilt

    def measure_log_density(
        self,
        latents: torch.Tensor,
        cond_embed: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, _, total = self.measure_log_density_components(latents, cond_embed, state=state)
        return total

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
