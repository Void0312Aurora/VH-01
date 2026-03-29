from __future__ import annotations

from dataclasses import dataclass

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
    _, energy_triangle, mask, delta_reg = _compute_response_triangle_components(
        model=model,
        latents=latents,
        video=video,
        cond_embed=cond_embed,
        include_residual_triangle=False,
        include_delta_reg=True,
    )
    if steps < 2:
        zero = latents.new_tensor(0.0)
        return zero, latents.new_zeros(batch), zero

    span_weights = latents.new_tensor(
        [1.0 / ((span_idx + 1) ** short_span_bias) for span_idx in range(steps - 1)]
    )
    weight_triangle = mask.to(dtype=latents.dtype) * span_weights.view(-1, 1)
    weight_sum = weight_triangle.sum().clamp_min(1e-6)
    weighted_energy = energy_triangle * weight_triangle.unsqueeze(0)
    per_sample = weighted_energy.sum(dim=(1, 2)) / weight_sum
    total = per_sample.mean()
    return total, per_sample, delta_reg


def _compute_response_triangle_components(
    model,
    latents: torch.Tensor,
    video: torch.Tensor,
    cond_embed: torch.Tensor,
    *,
    decoded: torch.Tensor | None = None,
    include_residual_triangle: bool,
    include_delta_reg: bool,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, steps, *_ = video.shape
    if decoded is None:
        decoded = model.decode_video(latents, cond_embed)
    frame_shape = video.shape[2:]
    if steps < 2:
        residual_triangle = latents.new_zeros(batch, 1, 1, *frame_shape) if include_residual_triangle else None
        energy_triangle = latents.new_zeros(batch, 1, 1)
        mask = torch.zeros(1, 1, dtype=torch.bool, device=latents.device)
        delta_reg = latents.new_tensor(0.0)
        return residual_triangle, energy_triangle, mask, delta_reg

    span_count = steps - 1
    residual_triangle = (
        latents.new_zeros(batch, span_count, span_count, *frame_shape)
        if include_residual_triangle
        else None
    )
    energy_triangle = latents.new_zeros(batch, span_count, span_count)
    mask = torch.zeros(span_count, span_count, dtype=torch.bool, device=latents.device)
    delta_reg = latents.new_tensor(0.0)

    for start in range(steps - 1):
        max_roll = steps - start - 1
        rollout, deltas = model.rollout_from(latents[:, start], cond_embed, max_roll)
        if include_delta_reg and deltas.numel() > 0:
            delta_reg = delta_reg + (deltas**2).mean()
        for span in range(1, max_roll + 1):
            end = start + span
            pred_frame = model.decode_video(rollout[:, span - 1 : span], cond_embed)[:, 0]
            prev_frame = decoded[:, end - 1]
            pred_delta = pred_frame - prev_frame
            true_delta = video[:, end] - video[:, end - 1]
            residual = pred_delta - true_delta
            if residual_triangle is not None:
                residual_triangle[:, span - 1, start] = residual
            energy_triangle[:, span - 1, start] = residual.square().flatten(1).mean(dim=1)
            mask[span - 1, start] = True

    if include_delta_reg:
        delta_reg = delta_reg / max(steps - 1, 1)
    return residual_triangle, energy_triangle, mask, delta_reg


def nce_condition_loss(logits: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
    if labels is None:
        labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def support_refinement_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    posterior_temperature: float,
    p_true_floor: float,
    p_true_ceiling: float,
    margin_floor: float,
    margin_ceiling: float,
    support_ratio_floor: float,
    support_ratio_ceiling: float,
    gate_p_true: float,
    gate_margin: float,
    gate_temperature: float,
) -> dict[str, torch.Tensor]:
    probs = posterior_from_logits(logits, temperature=posterior_temperature)
    p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    probs_wo_true = probs.clone()
    probs_wo_true.scatter_(1, labels.unsqueeze(1), -1.0)
    p_second = probs_wo_true.max(dim=1).values if probs.size(1) > 1 else torch.zeros_like(p_true)
    margin = p_true - p_second

    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    support_ratio = torch.exp(entropy) / float(logits.size(1))

    zero = logits.new_tensor(0.0)
    p_true_hinge = zero
    if p_true_floor > 0.0:
        p_true_hinge = p_true_hinge + F.relu(p_true_floor - p_true).mean()
    margin_hinge = zero
    if margin_floor > 0.0:
        margin_hinge = margin_hinge + F.relu(margin_floor - margin).mean()

    gate_threshold_p = gate_p_true if gate_p_true > 0.0 else max(p_true_floor, 0.0)
    gate_threshold_m = gate_margin if gate_margin > 0.0 else max(margin_floor, 0.0)
    gate_temperature = max(gate_temperature, 1e-4)
    confidence_gate = torch.sigmoid((p_true.detach() - gate_threshold_p) / gate_temperature)
    if probs.size(1) > 1:
        confidence_gate = confidence_gate * torch.sigmoid((margin.detach() - gate_threshold_m) / gate_temperature)

    if p_true_ceiling < 1.0:
        p_true_hinge = p_true_hinge + (
            confidence_gate * F.relu(p_true - p_true_ceiling)
        ).mean()
    if margin_ceiling < 1.0:
        margin_hinge = margin_hinge + (
            confidence_gate * F.relu(margin - margin_ceiling)
        ).mean()

    support_ratio_hinge = zero
    if support_ratio_floor > 0.0:
        support_ratio_hinge = support_ratio_hinge + (
            confidence_gate * F.relu(support_ratio_floor - support_ratio)
        ).mean()
    if support_ratio_ceiling < 1.0:
        support_ratio_hinge = support_ratio_hinge + (
            confidence_gate * F.relu(support_ratio - support_ratio_ceiling)
        ).mean()

    return {
        "support_p_true_hinge": p_true_hinge,
        "support_margin_hinge": margin_hinge,
        "support_ratio_hinge": support_ratio_hinge,
        "support_gate_mean": confidence_gate.mean(),
    }


@dataclass
class ResponseTriangleBundle:
    residual_triangle: torch.Tensor
    energy_triangle: torch.Tensor
    mask: torch.Tensor


@dataclass
class GeometryNeighborhoodReference:
    points: torch.Tensor | None
    response_channels: torch.Tensor | None
    tangent_frames: torch.Tensor | None
    tangent_frame_valid: torch.Tensor | None

    def size(self) -> int:
        if self.points is None:
            return 0
        return int(self.points.size(0))


@dataclass
class SmoothnessNeighborhoodReference:
    signatures: torch.Tensor | None
    state: torch.Tensor | None
    drift: torch.Tensor | None
    diffusion_flat: torch.Tensor | None
    log_density: torch.Tensor | None

    def size(self) -> int:
        if self.signatures is None:
            return 0
        return int(self.signatures.size(0))


@dataclass
class KNNNeighborhood:
    knn_idx: torch.Tensor | None
    weights: torch.Tensor | None
    distances: torch.Tensor | None
    reference_mask: torch.Tensor | None
    candidate_count: int
    reference_count: int


@dataclass
class ResponseInvariantTarget:
    descriptor_triangle: torch.Tensor
    signed_triangle: torch.Tensor
    magnitude_triangle: torch.Tensor
    mask: torch.Tensor
    response_channels: torch.Tensor
    operator: torch.Tensor
    eigvals: torch.Tensor
    trace: torch.Tensor
    effective_rank: torch.Tensor
    anisotropy: torch.Tensor
    asymmetry: torch.Tensor
    spectral_gap: torch.Tensor
    scale_profile: torch.Tensor
    tangent_frame: torch.Tensor | None
    tangent_projector: torch.Tensor | None
    tangent_drift: torch.Tensor | None
    tangent_cov: torch.Tensor | None
    identifiable_tangent_cov: torch.Tensor | None
    support_tilt: torch.Tensor | None
    graph_tau: torch.Tensor | None
    neighbor_idx: torch.Tensor | None
    neighbor_weights: torch.Tensor | None
    transport: torch.Tensor | None
    identifiable_effective_rank: torch.Tensor
    identifiable_anisotropy: torch.Tensor
    neighbor_pool_size: torch.Tensor
    reference_pool_size: torch.Tensor
    reference_neighbor_ratio: torch.Tensor

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class LocalGeneratorTarget:
    signatures: torch.Tensor
    source_point: torch.Tensor
    source_summary_context: torch.Tensor
    source_measure: object | None
    invariant_target: ResponseInvariantTarget | None
    drift_target: torch.Tensor
    diffusion_target: torch.Tensor
    full_diffusion_target: torch.Tensor
    bootstrap_drift_target: torch.Tensor
    bootstrap_diffusion_target: torch.Tensor
    target_tangent_drift: torch.Tensor
    target_tangent_cov: torch.Tensor | None
    identifiable_tangent_cov: torch.Tensor | None
    target_tangent_frame: torch.Tensor | None
    target_tangent_projector: torch.Tensor | None
    target_neighbor_idx: torch.Tensor | None
    target_neighbor_weights: torch.Tensor | None
    target_transport: torch.Tensor | None
    response_identifiable_effective_rank: torch.Tensor
    response_identifiable_anisotropy: torch.Tensor
    tilt_target: torch.Tensor | None
    response_operator_trace: torch.Tensor
    response_operator_effective_rank: torch.Tensor
    response_operator_anisotropy: torch.Tensor
    response_operator_asymmetry: torch.Tensor
    response_drift_alignment: torch.Tensor
    geometry_neighbor_pool_size: torch.Tensor
    geometry_reference_pool_size: torch.Tensor
    geometry_reference_neighbor_ratio: torch.Tensor

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def as_dict(self) -> dict[str, object]:
        return {
            "signatures": self.signatures,
            "source_point": self.source_point,
            "source_summary_context": self.source_summary_context,
            "source_measure": self.source_measure,
            "invariant_target": self.invariant_target,
            "drift_target": self.drift_target,
            "diffusion_target": self.diffusion_target,
            "full_diffusion_target": self.full_diffusion_target,
            "bootstrap_drift_target": self.bootstrap_drift_target,
            "bootstrap_diffusion_target": self.bootstrap_diffusion_target,
            "target_tangent_drift": self.target_tangent_drift,
            "target_tangent_cov": self.target_tangent_cov,
            "identifiable_tangent_cov": self.identifiable_tangent_cov,
            "target_tangent_frame": self.target_tangent_frame,
            "target_tangent_projector": self.target_tangent_projector,
            "target_neighbor_idx": self.target_neighbor_idx,
            "target_neighbor_weights": self.target_neighbor_weights,
            "target_transport": self.target_transport,
            "response_identifiable_effective_rank": self.response_identifiable_effective_rank,
            "response_identifiable_anisotropy": self.response_identifiable_anisotropy,
            "tilt_target": self.tilt_target,
            "response_operator_trace": self.response_operator_trace,
            "response_operator_effective_rank": self.response_operator_effective_rank,
            "response_operator_anisotropy": self.response_operator_anisotropy,
            "response_operator_asymmetry": self.response_operator_asymmetry,
            "response_drift_alignment": self.response_drift_alignment,
            "geometry_neighbor_pool_size": self.geometry_neighbor_pool_size,
            "geometry_reference_pool_size": self.geometry_reference_pool_size,
            "geometry_reference_neighbor_ratio": self.geometry_reference_neighbor_ratio,
        }


DEFAULT_RESPONSE_DESCRIPTOR_SPATIAL_SIZE = 4
DEFAULT_RESPONSE_DESCRIPTOR_INCLUDE_ABS = True
DEFAULT_RESPONSE_DESCRIPTOR_INCLUDE_SQUARE = True


def _truncate_reference_tensor(tensor: torch.Tensor | None, max_size: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    if max_size <= 0 or tensor.size(0) <= max_size:
        return tensor
    return tensor[-max_size:]


def _concat_reference_tensors(
    current: torch.Tensor | None,
    update: torch.Tensor | None,
    *,
    max_size: int,
) -> torch.Tensor | None:
    if current is None:
        return _truncate_reference_tensor(update, max_size)
    if update is None:
        return _truncate_reference_tensor(current, max_size)
    merged = torch.cat([current, update], dim=0)
    return _truncate_reference_tensor(merged, max_size)


@torch.no_grad()
def append_geometry_neighborhood_reference(
    reference: GeometryNeighborhoodReference | None,
    update: GeometryNeighborhoodReference | None,
    *,
    max_size: int,
) -> GeometryNeighborhoodReference | None:
    if max_size <= 0 or update is None or update.size() <= 0:
        return reference if max_size > 0 else None
    if reference is None:
        return GeometryNeighborhoodReference(
            points=_truncate_reference_tensor(update.points, max_size),
            response_channels=_truncate_reference_tensor(update.response_channels, max_size),
            tangent_frames=_truncate_reference_tensor(update.tangent_frames, max_size),
            tangent_frame_valid=_truncate_reference_tensor(update.tangent_frame_valid, max_size),
        )
    return GeometryNeighborhoodReference(
        points=_concat_reference_tensors(reference.points, update.points, max_size=max_size),
        response_channels=_concat_reference_tensors(
            reference.response_channels,
            update.response_channels,
            max_size=max_size,
        ),
        tangent_frames=_concat_reference_tensors(
            reference.tangent_frames,
            update.tangent_frames,
            max_size=max_size,
        ),
        tangent_frame_valid=_concat_reference_tensors(
            reference.tangent_frame_valid,
            update.tangent_frame_valid,
            max_size=max_size,
        ),
    )


@torch.no_grad()
def append_smoothness_neighborhood_reference(
    reference: SmoothnessNeighborhoodReference | None,
    update: SmoothnessNeighborhoodReference | None,
    *,
    max_size: int,
) -> SmoothnessNeighborhoodReference | None:
    if max_size <= 0 or update is None or update.size() <= 0:
        return reference if max_size > 0 else None
    if reference is None:
        return SmoothnessNeighborhoodReference(
            signatures=_truncate_reference_tensor(update.signatures, max_size),
            state=_truncate_reference_tensor(update.state, max_size),
            drift=_truncate_reference_tensor(update.drift, max_size),
            diffusion_flat=_truncate_reference_tensor(update.diffusion_flat, max_size),
            log_density=_truncate_reference_tensor(update.log_density, max_size),
        )
    return SmoothnessNeighborhoodReference(
        signatures=_concat_reference_tensors(reference.signatures, update.signatures, max_size=max_size),
        state=_concat_reference_tensors(reference.state, update.state, max_size=max_size),
        drift=_concat_reference_tensors(reference.drift, update.drift, max_size=max_size),
        diffusion_flat=_concat_reference_tensors(
            reference.diffusion_flat,
            update.diffusion_flat,
            max_size=max_size,
        ),
        log_density=_concat_reference_tensors(reference.log_density, update.log_density, max_size=max_size),
    )


@torch.no_grad()
def response_descriptor_dim(
    channels: int,
    *,
    spatial_size: int = DEFAULT_RESPONSE_DESCRIPTOR_SPATIAL_SIZE,
    include_abs: bool = DEFAULT_RESPONSE_DESCRIPTOR_INCLUDE_ABS,
    include_square: bool = DEFAULT_RESPONSE_DESCRIPTOR_INCLUDE_SQUARE,
) -> int:
    component_count = 1 + int(include_abs) + int(include_square)
    return max(channels * spatial_size * spatial_size * component_count, 1)


@torch.no_grad()
def response_signature_dim(
    seq_len: int,
    mode: str = "span_stats",
    *,
    channels: int | None = None,
    spatial_size: int = DEFAULT_RESPONSE_DESCRIPTOR_SPATIAL_SIZE,
    include_abs: bool = DEFAULT_RESPONSE_DESCRIPTOR_INCLUDE_ABS,
    include_square: bool = DEFAULT_RESPONSE_DESCRIPTOR_INCLUDE_SQUARE,
) -> int:
    if seq_len < 2:
        if mode in {"span_stats", "full_triangle"}:
            return 1
        if channels is None:
            raise ValueError(f"channels must be provided for response_signature mode: {mode}")
        descriptor_width = response_descriptor_dim(
            channels,
            spatial_size=spatial_size,
            include_abs=include_abs,
            include_square=include_square,
        )
        if mode == "descriptor_span_stats":
            return 2 * descriptor_width
        if mode == "descriptor_full_triangle":
            return descriptor_width
        raise ValueError(f"Unsupported response_signature mode: {mode}")
    if mode == "span_stats":
        return max(2 * (seq_len - 1), 2)
    if mode == "full_triangle":
        return max(seq_len * (seq_len - 1) // 2, 1)
    if mode in {"descriptor_span_stats", "descriptor_full_triangle"}:
        if channels is None:
            raise ValueError(f"channels must be provided for response_signature mode: {mode}")
        descriptor_width = response_descriptor_dim(
            channels,
            spatial_size=spatial_size,
            include_abs=include_abs,
            include_square=include_square,
        )
        if mode == "descriptor_span_stats":
            return max(2 * (seq_len - 1) * descriptor_width, 2 * descriptor_width)
        return max((seq_len * (seq_len - 1) // 2) * descriptor_width, descriptor_width)
    raise ValueError(f"Unsupported response_signature mode: {mode}")


@torch.no_grad()
def response_triangle_bundle(
    model,
    latents: torch.Tensor,
    video: torch.Tensor,
    cond_embed: torch.Tensor,
    decoded: torch.Tensor | None = None,
) -> ResponseTriangleBundle:
    residual_triangle, energy_triangle, mask, _ = _compute_response_triangle_components(
        model=model,
        latents=latents,
        video=video,
        cond_embed=cond_embed,
        decoded=decoded,
        include_residual_triangle=True,
        include_delta_reg=False,
    )
    return ResponseTriangleBundle(
        residual_triangle=residual_triangle,
        energy_triangle=energy_triangle,
        mask=mask,
    )


@torch.no_grad()
def response_triangle(
    model,
    latents: torch.Tensor,
    video: torch.Tensor,
    cond_embed: torch.Tensor,
    decoded: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bundle = response_triangle_bundle(
        model=model,
        latents=latents,
        video=video,
        cond_embed=cond_embed,
        decoded=decoded,
    )
    return bundle.energy_triangle, bundle.mask


@torch.no_grad()
def response_descriptor_triangle_from_bundle(
    bundle: ResponseTriangleBundle,
    *,
    spatial_size: int = DEFAULT_RESPONSE_DESCRIPTOR_SPATIAL_SIZE,
    include_abs: bool = DEFAULT_RESPONSE_DESCRIPTOR_INCLUDE_ABS,
    include_square: bool = DEFAULT_RESPONSE_DESCRIPTOR_INCLUDE_SQUARE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_triangle = bundle.residual_triangle
    mask = bundle.mask
    batch, span_count, max_start, channels, _, _ = residual_triangle.shape
    output_size = (spatial_size, spatial_size)
    component_count = 1 + int(include_abs) + int(include_square)
    descriptor_dim = channels * spatial_size * spatial_size * component_count
    descriptor_triangle = residual_triangle.new_zeros(batch, span_count, max_start, descriptor_dim)
    signed_triangle = residual_triangle.new_zeros(batch, span_count, max_start)
    magnitude_triangle = residual_triangle.new_zeros(batch, span_count, max_start)

    for span_idx in range(span_count):
        valid_count = int(mask[span_idx].sum().item())
        if valid_count <= 0:
            continue
        span_residual = residual_triangle[:, span_idx, :valid_count]
        flat_residual = span_residual.reshape(batch * valid_count, channels, *span_residual.shape[-2:])
        signed_pool = F.adaptive_avg_pool2d(flat_residual, output_size).flatten(1)
        components = [signed_pool]
        if include_abs:
            components.append(F.adaptive_avg_pool2d(flat_residual.abs(), output_size).flatten(1))
        if include_square:
            components.append(F.adaptive_avg_pool2d(flat_residual.square(), output_size).flatten(1))
        descriptor = torch.cat(components, dim=1).view(batch, valid_count, -1)
        descriptor_triangle[:, span_idx, :valid_count] = descriptor
        signed_triangle[:, span_idx, :valid_count] = signed_pool.mean(dim=1).view(batch, valid_count)
        magnitude_triangle[:, span_idx, :valid_count] = descriptor.norm(dim=-1)

    return descriptor_triangle, signed_triangle, magnitude_triangle


@torch.no_grad()
def response_descriptor_from_bundle(
    bundle: ResponseTriangleBundle,
    *,
    spatial_size: int = DEFAULT_RESPONSE_DESCRIPTOR_SPATIAL_SIZE,
    include_abs: bool = DEFAULT_RESPONSE_DESCRIPTOR_INCLUDE_ABS,
    include_square: bool = DEFAULT_RESPONSE_DESCRIPTOR_INCLUDE_SQUARE,
) -> torch.Tensor:
    descriptor_triangle, _, _ = response_descriptor_triangle_from_bundle(
        bundle,
        spatial_size=spatial_size,
        include_abs=include_abs,
        include_square=include_square,
    )
    batch = descriptor_triangle.size(0)
    pooled_channels: list[torch.Tensor] = []
    for span_idx in range(descriptor_triangle.size(1)):
        valid_count = int(bundle.mask[span_idx].sum().item())
        if valid_count <= 0:
            continue
        pooled_channels.append(descriptor_triangle[:, span_idx, :valid_count].reshape(batch, -1))
    if not pooled_channels:
        return descriptor_triangle.new_zeros(batch, 1)
    return torch.cat(pooled_channels, dim=1)


@torch.no_grad()
def _response_signature_from_descriptor_triangle(
    descriptor_triangle: torch.Tensor,
    signed_triangle: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    batch, span_count, _, descriptor_width = descriptor_triangle.shape
    if span_count < 1:
        if mode == "descriptor_span_stats":
            return descriptor_triangle.new_zeros(batch, 2 * descriptor_width)
        if mode == "descriptor_full_triangle":
            return descriptor_triangle.new_zeros(batch, descriptor_width)
        return signed_triangle.new_zeros(batch, 1)

    if mode not in {"span_stats", "full_triangle", "descriptor_span_stats", "descriptor_full_triangle"}:
        raise ValueError(f"Unsupported response_signature mode: {mode}")

    components: list[torch.Tensor] = []
    for span_idx in range(span_count):
        valid_count = int(mask[span_idx].sum().item())
        if valid_count <= 0:
            continue
        span_descriptor = descriptor_triangle[:, span_idx, :valid_count]
        span_tensor = signed_triangle[:, span_idx, :valid_count]
        if mode == "descriptor_full_triangle":
            components.append(span_descriptor.reshape(batch, valid_count * descriptor_width))
        elif mode == "descriptor_span_stats":
            components.append(span_descriptor.mean(dim=1))
            if valid_count > 1:
                components.append(span_descriptor.std(dim=1, unbiased=False))
            else:
                components.append(torch.zeros_like(span_descriptor[:, 0]))
        elif mode == "full_triangle":
            components.append(span_tensor)
        else:
            components.append(span_tensor.mean(dim=1, keepdim=True))
            if span_tensor.size(1) > 1:
                components.append(span_tensor.std(dim=1, unbiased=False, keepdim=True))
            else:
                components.append(torch.zeros_like(span_tensor[:, :1]))
    if not components:
        if mode == "descriptor_span_stats":
            return descriptor_triangle.new_zeros(batch, max(2 * span_count * descriptor_width, 2 * descriptor_width))
        if mode == "descriptor_full_triangle":
            return descriptor_triangle.new_zeros(batch, descriptor_width)
        return signed_triangle.new_zeros(batch, response_signature_dim(span_count + 1, mode))
    return torch.cat(components, dim=1)


@torch.no_grad()
def _response_operator_from_descriptor_triangle(
    descriptor_triangle: torch.Tensor,
    magnitude_triangle: torch.Tensor,
    mask: torch.Tensor,
    *,
    signed_triangle: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    batch, span_count, max_start, descriptor_dim = descriptor_triangle.shape
    row_dim = max_start * descriptor_dim
    row_vectors = descriptor_triangle.new_zeros(batch, span_count, row_dim)
    row_mask = torch.zeros(span_count, row_dim, dtype=descriptor_triangle.dtype, device=descriptor_triangle.device)

    for span_idx in range(span_count):
        valid_count = int(mask[span_idx].sum().item())
        if valid_count <= 0:
            continue
        width = valid_count * descriptor_dim
        row_vectors[:, span_idx, :width] = descriptor_triangle[:, span_idx, :valid_count].reshape(batch, width)
        row_mask[span_idx, :width] = 1.0

    row_mask_f = row_mask.unsqueeze(0)
    valid_counts = row_mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    row_mean = (row_vectors * row_mask_f).sum(dim=-1, keepdim=True) / valid_counts
    centered = (row_vectors - row_mean) * row_mask_f
    row_scale = centered.square().sum(dim=-1, keepdim=True).div(valid_counts).sqrt().clamp_min(eps)
    normalized = centered / row_scale
    normalized = normalized * row_mask_f
    operator = normalized @ normalized.transpose(-1, -2)
    normalizer = row_mask_f.sum(dim=-1).amax(dim=-1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
    operator = operator / normalizer
    operator = 0.5 * (operator + operator.transpose(-1, -2))
    eigvals = _sorted_eigvalsh(operator).clamp_min(0.0)
    trace = eigvals.sum(dim=-1)
    spectral_mass = eigvals / trace.unsqueeze(-1).clamp_min(eps)
    entropy = -(spectral_mass * spectral_mass.clamp_min(eps).log()).sum(dim=-1)
    effective_rank = torch.exp(entropy)
    anisotropy = torch.log(eigvals.clamp_min(eps)).std(dim=-1, unbiased=False)
    start_positions = torch.linspace(-1.0, 1.0, max_start, device=descriptor_triangle.device, dtype=descriptor_triangle.dtype)
    asymmetry_source = signed_triangle if signed_triangle is not None else magnitude_triangle
    mask_f = mask.to(dtype=descriptor_triangle.dtype).unsqueeze(0)
    asymmetry_num = (asymmetry_source * mask_f * start_positions.view(1, 1, -1)).sum(dim=(-1, -2))
    asymmetry_den = (asymmetry_source.abs() * mask_f).sum(dim=(-1, -2)).clamp_min(eps)
    asymmetry = asymmetry_num / asymmetry_den
    return {
        "operator": operator,
        "eigvals": eigvals,
        "trace": trace,
        "effective_rank": effective_rank,
        "anisotropy": anisotropy,
        "asymmetry": asymmetry,
    }


@torch.no_grad()
def build_response_invariant_target(
    source_point: torch.Tensor,
    response_bundle: ResponseTriangleBundle,
    *,
    tangent_dim: int,
    geometry_knn: int,
    geometry_temperature: float,
    jet_ridge: float,
    jet_center_weight: float,
    base_log_density_target: torch.Tensor | None = None,
    tilt_target_mode: str = "none",
    tau_ridge: float = 1e-3,
    tau_mean_penalty: float = 1.0,
    tau_drift_scale: float = 0.25,
    geometry_reference: GeometryNeighborhoodReference | None = None,
) -> ResponseInvariantTarget:
    descriptor_triangle, signed_triangle, magnitude_triangle = response_descriptor_triangle_from_bundle(response_bundle)
    descriptor_rows: list[torch.Tensor] = []
    scale_rows: list[torch.Tensor] = []
    for span_idx in range(descriptor_triangle.size(1)):
        valid_count = int(response_bundle.mask[span_idx].sum().item())
        if valid_count <= 0:
            continue
        descriptor_rows.append(descriptor_triangle[:, span_idx, :valid_count].reshape(descriptor_triangle.size(0), -1))
        span_scale = magnitude_triangle[:, span_idx, :valid_count].mean(dim=1, keepdim=True)
        scale_rows.append(span_scale)
    response_channels = (
        torch.cat(descriptor_rows, dim=1)
        if descriptor_rows
        else descriptor_triangle.new_zeros(descriptor_triangle.size(0), 1)
    )
    scale_profile = (
        torch.cat(scale_rows, dim=1)
        if scale_rows
        else descriptor_triangle.new_zeros(descriptor_triangle.size(0), 1)
    )
    scale_profile = scale_profile / scale_profile.sum(dim=1, keepdim=True).clamp_min(1e-6)

    response_operator = _response_operator_from_descriptor_triangle(
        descriptor_triangle,
        magnitude_triangle,
        response_bundle.mask,
        signed_triangle=signed_triangle,
    )
    eigvals = response_operator["eigvals"]
    if eigvals.size(1) > 1:
        spectral_gap = (eigvals[:, 0] - eigvals[:, 1]) / response_operator["trace"].clamp_min(1e-6)
    else:
        spectral_gap = eigvals[:, 0] / response_operator["trace"].clamp_min(1e-6)

    response_jet = _local_response_jet_bundle(
        source_point,
        response_channels,
        tangent_dim=tangent_dim,
        knn=geometry_knn,
        temperature=geometry_temperature,
        ridge=jet_ridge,
        center_weight=jet_center_weight,
        reference_states=None if geometry_reference is None else geometry_reference.points,
        reference_response_channels=None if geometry_reference is None else geometry_reference.response_channels,
        reference_tangent_frames=None if geometry_reference is None else geometry_reference.tangent_frames,
        reference_tangent_frame_valid=None if geometry_reference is None else geometry_reference.tangent_frame_valid,
    )

    support_tilt = torch.log(response_operator["trace"].clamp_min(1e-6)).unsqueeze(-1)
    support_tilt = support_tilt + 0.25 * response_operator["anisotropy"].unsqueeze(-1)
    support_tilt = support_tilt + 0.25 * response_operator["asymmetry"].abs().unsqueeze(-1)
    support_tilt = (support_tilt - support_tilt.mean(dim=0, keepdim=True)) / support_tilt.std(
        dim=0,
        unbiased=False,
        keepdim=True,
    ).clamp_min(1e-6)
    if response_jet["support_tilt"] is not None:
        support_tilt = response_jet["support_tilt"]

    graph_tau_target = None
    if tilt_target_mode == "graph_tau" and base_log_density_target is not None and response_jet["frame"] is not None:
        graph_tau_bundle = _solve_graph_tau_target(
            source_point,
            base_log_density_target,
            jet_bundle=response_jet,
            temperature=geometry_temperature,
            ridge=tau_ridge,
            mean_penalty=tau_mean_penalty,
            drift_scale=tau_drift_scale,
        )
        graph_tau_target = graph_tau_bundle["tau"]

    return ResponseInvariantTarget(
        descriptor_triangle=descriptor_triangle,
        signed_triangle=signed_triangle,
        magnitude_triangle=magnitude_triangle,
        mask=response_bundle.mask,
        response_channels=response_channels,
        operator=response_operator["operator"],
        eigvals=eigvals,
        trace=response_operator["trace"],
        effective_rank=response_operator["effective_rank"],
        anisotropy=response_operator["anisotropy"],
        asymmetry=response_operator["asymmetry"],
        spectral_gap=spectral_gap,
        scale_profile=scale_profile,
        tangent_frame=response_jet["frame"],
        tangent_projector=response_jet["projector"],
        tangent_drift=response_jet["tangent_drift"],
        tangent_cov=response_jet["tangent_cov"],
        identifiable_tangent_cov=response_jet["identifiable_tangent_cov"],
        support_tilt=support_tilt,
        graph_tau=graph_tau_target,
        neighbor_idx=response_jet["neighbor_idx"],
        neighbor_weights=response_jet["neighbor_weights"],
        transport=response_jet["transport"],
        identifiable_effective_rank=response_jet["effective_rank"],
        identifiable_anisotropy=response_jet["anisotropy"],
        neighbor_pool_size=response_jet["neighbor_pool_size"],
        reference_pool_size=response_jet["reference_pool_size"],
        reference_neighbor_ratio=response_jet["reference_neighbor_ratio"],
    )


@torch.no_grad()
def build_geometry_neighborhood_reference(
    model,
    latents: torch.Tensor,
    video: torch.Tensor,
    cond_embed: torch.Tensor,
    *,
    decoded: torch.Tensor | None = None,
    geometry_knn: int = 3,
    geometry_temperature: float = 0.5,
    jet_ridge: float = 1e-3,
    jet_center_weight: float = 1.0,
) -> GeometryNeighborhoodReference:
    if latents.size(0) == 0:
        return GeometryNeighborhoodReference(
            points=None,
            response_channels=None,
            tangent_frames=None,
            tangent_frame_valid=None,
        )
    point = model.trajectory_point(latents).detach()
    bundle = response_triangle_bundle(
        model=model,
        latents=latents,
        video=video,
        cond_embed=cond_embed,
        decoded=decoded,
    )
    response_channels = response_descriptor_from_bundle(bundle).detach()
    tangent_frames = None
    tangent_frame_valid = None
    tangent_dim = int(getattr(model, "tangent_dim", 0))
    if tangent_dim > 0:
        tangent_frames = point.new_zeros(point.size(0), point.size(1), tangent_dim)
        tangent_frame_valid = torch.zeros(point.size(0), device=point.device, dtype=torch.bool)
        response_jet = _local_response_jet_bundle(
            point,
            response_channels,
            tangent_dim=tangent_dim,
            knn=geometry_knn,
            temperature=geometry_temperature,
            ridge=jet_ridge,
            center_weight=jet_center_weight,
        )
        jet_frame = response_jet["frame"]
        if jet_frame is not None:
            tangent_frames = jet_frame.detach()
            tangent_frame_valid = torch.ones(point.size(0), device=point.device, dtype=torch.bool)
    return GeometryNeighborhoodReference(
        points=point,
        response_channels=response_channels,
        tangent_frames=tangent_frames,
        tangent_frame_valid=tangent_frame_valid,
    )


@torch.no_grad()
def build_smoothness_neighborhood_reference(
    model,
    latents: torch.Tensor,
    video: torch.Tensor,
    cond_embed: torch.Tensor,
    *,
    signature_mode: str,
    decoded: torch.Tensor | None = None,
) -> SmoothnessNeighborhoodReference:
    if latents.size(0) == 0:
        return SmoothnessNeighborhoodReference(
            signatures=None,
            state=None,
            drift=None,
            diffusion_flat=None,
            log_density=None,
        )
    state = model.trajectory_point(latents)
    signatures = response_signature(
        model=model,
        latents=latents,
        video=video,
        cond_embed=cond_embed,
        decoded=decoded,
        mode=signature_mode,
    )
    generator = model.local_generator(
        latents,
        cond_embed,
        response_context=signatures,
        state=state,
    )
    return SmoothnessNeighborhoodReference(
        signatures=signatures.detach(),
        state=state.detach(),
        drift=generator.drift.detach(),
        diffusion_flat=generator.diffusion_matrix.detach().flatten(1),
        log_density=generator.conditional_measure.log_total_density.detach().squeeze(-1),
    )


@torch.no_grad()
def _build_knn_graph(
    points: torch.Tensor,
    *,
    knn: int,
    temperature: float,
    reference_points: torch.Tensor | None = None,
) -> KNNNeighborhood:
    batch = points.size(0)
    reference_count = 0 if reference_points is None else int(reference_points.size(0))
    candidate_count = batch + reference_count
    if candidate_count < 2 or knn <= 0:
        return KNNNeighborhood(
            knn_idx=None,
            weights=None,
            distances=None,
            reference_mask=None,
            candidate_count=candidate_count,
            reference_count=reference_count,
        )
    num_neighbors = min(knn, candidate_count - 1)
    candidate_points = points.detach()
    if reference_points is not None and reference_points.numel() > 0:
        candidate_points = torch.cat(
            [
                candidate_points,
                reference_points.detach().to(device=points.device, dtype=points.dtype),
            ],
            dim=0,
        )
    distance = torch.cdist(points.detach(), candidate_points, p=2.0)
    inf = torch.full((batch,), float("inf"), device=distance.device, dtype=distance.dtype)
    distance = distance.clone()
    distance[torch.arange(batch, device=distance.device), torch.arange(batch, device=distance.device)] = inf
    knn_dist, knn_idx = torch.topk(distance, k=num_neighbors, largest=False, dim=1)
    weights = torch.softmax(-knn_dist / max(temperature, 1e-4), dim=1)
    return KNNNeighborhood(
        knn_idx=knn_idx,
        weights=weights,
        distances=knn_dist,
        reference_mask=knn_idx >= batch,
        candidate_count=candidate_count,
        reference_count=reference_count,
    )


def _quadratic_feature_dim(dim: int) -> int:
    return dim * (dim + 1) // 2


def _quadratic_features(coords: torch.Tensor) -> torch.Tensor:
    dim = coords.size(-1)
    pieces: list[torch.Tensor] = []
    for i in range(dim):
        pieces.append(coords[..., i : i + 1] * coords[..., i : i + 1])
        for j in range(i + 1, dim):
            pieces.append(coords[..., i : i + 1] * coords[..., j : j + 1])
    if not pieces:
        return coords.new_zeros(*coords.shape[:-1], 0)
    return torch.cat(pieces, dim=-1)


def _quadratic_coeffs_to_hessian(coeffs: torch.Tensor, dim: int) -> torch.Tensor:
    hessian = coeffs.new_zeros(coeffs.size(0), dim, dim)
    idx = 0
    for i in range(dim):
        hessian[:, i, i] = 2.0 * coeffs[:, idx]
        idx += 1
        for j in range(i + 1, dim):
            hessian[:, i, j] = coeffs[:, idx]
            hessian[:, j, i] = coeffs[:, idx]
            idx += 1
    return hessian


@torch.no_grad()
def _local_response_jet_bundle(
    states: torch.Tensor,
    response_channels: torch.Tensor,
    *,
    tangent_dim: int,
    knn: int,
    temperature: float,
    ridge: float,
    center_weight: float,
    reference_states: torch.Tensor | None = None,
    reference_response_channels: torch.Tensor | None = None,
    reference_tangent_frames: torch.Tensor | None = None,
    reference_tangent_frame_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    if response_channels.dim() > 2:
        response_channels = response_channels.reshape(response_channels.size(0), -1)
    elif response_channels.dim() == 1:
        response_channels = response_channels.unsqueeze(-1)

    batch, state_dim = states.shape
    zero_scalar = states.new_tensor(0.0)
    reference_count = 0 if reference_states is None else int(reference_states.size(0))
    neighbor_pool_size = states.new_tensor(float(batch + reference_count))
    reference_pool_size = states.new_tensor(float(reference_count))
    if batch < 1 or batch + reference_count < 2 or tangent_dim <= 0:
        return {
            "frame": None,
            "projector": None,
            "neighbor_idx": None,
            "neighbor_weights": None,
            "transport": None,
            "tangent_drift": None,
            "tangent_cov": None,
            "identifiable_tangent_cov": None,
            "support_tilt": None,
            "effective_rank": zero_scalar,
            "anisotropy": zero_scalar,
            "neighbor_pool_size": neighbor_pool_size,
            "reference_pool_size": reference_pool_size,
            "reference_neighbor_ratio": zero_scalar,
        }

    neighborhood = _build_knn_graph(
        states,
        knn=knn,
        temperature=temperature,
        reference_points=reference_states,
    )
    knn_idx = neighborhood.knn_idx
    knn_weights = neighborhood.weights
    if knn_idx is None or knn_weights is None:
        return {
            "frame": None,
            "projector": None,
            "neighbor_idx": None,
            "neighbor_weights": None,
            "transport": None,
            "tangent_drift": None,
            "tangent_cov": None,
            "identifiable_tangent_cov": None,
            "support_tilt": None,
            "effective_rank": zero_scalar,
            "anisotropy": zero_scalar,
            "neighbor_pool_size": neighbor_pool_size,
            "reference_pool_size": reference_pool_size,
            "reference_neighbor_ratio": zero_scalar,
        }

    tangent_dim = min(tangent_dim, state_dim)
    num_neighbors = knn_idx.size(1)
    candidate_states = states
    if reference_states is not None and reference_states.numel() > 0:
        candidate_states = torch.cat(
            [
                states,
                reference_states.to(device=states.device, dtype=states.dtype),
            ],
            dim=0,
        )
    neighbor_states = candidate_states.index_select(0, knn_idx.reshape(-1)).view(batch, num_neighbors, state_dim)
    delta = neighbor_states - states.unsqueeze(1)
    cov = torch.einsum("bk,bkd,bke->bde", knn_weights, delta, delta)
    with torch.autocast(device_type=states.device.type, enabled=False):
        eigvals, eigvecs = torch.linalg.eigh(cov.float())
    frame = torch.flip(eigvecs, dims=(-1,))[..., :tangent_dim].to(dtype=states.dtype)
    projector = frame @ frame.transpose(-1, -2)
    coords = torch.einsum("bkd,bdm->bkm", delta, frame)

    candidate_responses = response_channels
    if reference_response_channels is not None and reference_response_channels.numel() > 0:
        candidate_responses = torch.cat(
            [
                response_channels,
                reference_response_channels.to(device=states.device, dtype=response_channels.dtype),
            ],
            dim=0,
        )
    neighbor_responses = candidate_responses.index_select(0, knn_idx.reshape(-1)).view(batch, num_neighbors, -1)
    channel_count = response_channels.size(1)
    quad_dim = _quadratic_feature_dim(tangent_dim)
    feature_dim = 1 + tangent_dim + quad_dim
    eye = torch.eye(feature_dim, device=states.device, dtype=torch.float32)
    tangent_eye = torch.eye(tangent_dim, device=states.device, dtype=states.dtype)

    tangent_drift = states.new_zeros(batch, tangent_dim)
    tangent_cov = states.new_zeros(batch, tangent_dim, tangent_dim)
    identifiable_tangent_cov = states.new_zeros(batch, tangent_dim, tangent_dim)
    support_tilt = states.new_zeros(batch, 1)
    effective_rank_values = states.new_zeros(batch)
    anisotropy_values = states.new_zeros(batch)

    for b in range(batch):
        local_coords = coords[b]
        local_quad = _quadratic_features(local_coords)
        neighbor_design = torch.cat(
            [
                torch.ones(num_neighbors, 1, device=states.device, dtype=states.dtype),
                local_coords,
                local_quad,
            ],
            dim=-1,
        )
        center_design = states.new_zeros(1, feature_dim)
        center_design[:, 0] = 1.0
        design = torch.cat([center_design, neighbor_design], dim=0)
        responses = torch.cat([response_channels[b : b + 1], neighbor_responses[b]], dim=0)
        weights = torch.cat(
            [
                states.new_tensor([center_weight]),
                knn_weights[b],
            ],
            dim=0,
        )
        sqrt_weights = weights.sqrt().unsqueeze(-1)
        design_w = design.float() * sqrt_weights.float()
        responses_w = responses.float() * sqrt_weights.float()
        reg = ridge * eye
        reg[0, 0] = 0.0
        normal = (design_w.transpose(0, 1) @ design_w + reg).float()
        rhs = (design_w.transpose(0, 1) @ responses_w).float()
        with torch.autocast(device_type=states.device.type, enabled=False):
            beta = torch.linalg.solve(normal, rhs)

        intercept = beta[0].to(dtype=states.dtype)
        linear = beta[1 : 1 + tangent_dim].transpose(0, 1).to(dtype=states.dtype)
        quad = beta[1 + tangent_dim :].transpose(0, 1).to(dtype=states.dtype)
        hessian = _quadratic_coeffs_to_hessian(quad, tangent_dim)

        channel_scale = (
            intercept.abs()
            + linear.norm(dim=-1)
            + hessian.flatten(1).norm(dim=-1)
        ).clamp_min(1e-6)
        channel_weights = channel_scale / channel_scale.sum().clamp_min(1e-6)

        tangent_drift[b] = (channel_weights.unsqueeze(-1) * linear).sum(dim=0)
        channel_metric = hessian.transpose(-1, -2) @ hessian
        metric = (channel_weights.view(channel_count, 1, 1) * channel_metric).sum(dim=0)
        metric = 0.5 * (metric + metric.transpose(-1, -2))
        metric = metric + 1e-6 * tangent_eye
        tangent_cov[b] = metric

        metric_trace = metric.diagonal(dim1=-2, dim2=-1).sum()
        normalized_metric = metric / metric_trace.clamp_min(1e-6)
        identifiable_tangent_cov[b] = normalized_metric

        metric_eigs = _sorted_eigvalsh(normalized_metric.unsqueeze(0)).squeeze(0).clamp_min(1e-8)
        metric_weights = metric_eigs / metric_eigs.sum().clamp_min(1e-6)
        metric_entropy = -(metric_weights * metric_weights.clamp_min(1e-8).log()).sum()
        effective_rank_values[b] = torch.exp(metric_entropy)
        anisotropy_values[b] = torch.log(metric_eigs).std(unbiased=False)
        support_tilt[b, 0] = torch.log(channel_scale.mean()) + 0.25 * torch.log(metric_trace.clamp_min(1e-6))

    support_tilt = (support_tilt - support_tilt.mean(dim=0, keepdim=True)) / support_tilt.std(
        dim=0,
        unbiased=False,
        keepdim=True,
    ).clamp_min(1e-6)

    transport = None
    neighbor_idx_out = None
    neighbor_weights_out = None
    reference_neighbor_ratio = zero_scalar
    if neighborhood.reference_mask is not None:
        reference_neighbor_ratio = neighborhood.reference_mask.float().mean().to(dtype=states.dtype)
    candidate_frames = frame
    candidate_frame_valid = torch.ones(batch, device=states.device, dtype=torch.bool)
    if reference_tangent_frames is not None and reference_tangent_frames.numel() > 0:
        candidate_frames = torch.cat(
            [
                frame,
                reference_tangent_frames.to(device=states.device, dtype=states.dtype),
            ],
            dim=0,
        )
        if reference_tangent_frame_valid is None:
            reference_valid = torch.ones(
                reference_tangent_frames.size(0),
                device=states.device,
                dtype=torch.bool,
            )
        else:
            reference_valid = reference_tangent_frame_valid.to(device=states.device, dtype=torch.bool)
        candidate_frame_valid = torch.cat([candidate_frame_valid, reference_valid], dim=0)

    neighbor_frame = candidate_frames.index_select(0, knn_idx.reshape(-1)).view(batch, num_neighbors, state_dim, tangent_dim)
    neighbor_frame_valid = candidate_frame_valid.index_select(0, knn_idx.reshape(-1)).view(batch, num_neighbors)
    if bool(neighbor_frame_valid.any().item()):
        center_frame = frame.unsqueeze(1).expand(-1, num_neighbors, -1, -1)
        overlap = torch.matmul(neighbor_frame.transpose(-1, -2).float(), center_frame.float())
        with torch.autocast(device_type=states.device.type, enabled=False):
            u, _, vh = torch.linalg.svd(overlap.float(), full_matrices=False)
        transport = (u @ vh).to(dtype=states.dtype)
        identity_transport = torch.eye(
            tangent_dim,
            device=states.device,
            dtype=states.dtype,
        ).view(1, 1, tangent_dim, tangent_dim)
        transport = torch.where(
            neighbor_frame_valid.unsqueeze(-1).unsqueeze(-1),
            transport,
            identity_transport,
        )
        neighbor_weights_out = knn_weights * neighbor_frame_valid.to(dtype=knn_weights.dtype)
        neighbor_weights_out = neighbor_weights_out / neighbor_weights_out.sum(dim=1, keepdim=True).clamp_min(1e-6)
        neighbor_idx_out = knn_idx

    return {
        "frame": frame,
        "projector": projector,
        "neighbor_idx": neighbor_idx_out,
        "neighbor_weights": neighbor_weights_out,
        "transport": transport,
        "tangent_drift": tangent_drift,
        "tangent_cov": tangent_cov,
        "identifiable_tangent_cov": identifiable_tangent_cov,
        "support_tilt": support_tilt,
        "effective_rank": effective_rank_values.mean(),
        "anisotropy": anisotropy_values.mean(),
        "neighbor_pool_size": neighbor_pool_size,
        "reference_pool_size": reference_pool_size,
        "reference_neighbor_ratio": reference_neighbor_ratio,
    }


@torch.no_grad()
def _solve_graph_tau_target(
    states: torch.Tensor,
    base_log_density: torch.Tensor,
    *,
    jet_bundle: dict[str, torch.Tensor | None],
    temperature: float,
    ridge: float,
    mean_penalty: float,
    drift_scale: float,
) -> dict[str, torch.Tensor | None]:
    knn_idx = jet_bundle.get("neighbor_idx")
    knn_weights = jet_bundle.get("neighbor_weights")
    frame = jet_bundle.get("frame")
    identifiable_tangent_cov = jet_bundle.get("identifiable_tangent_cov")
    tangent_drift = jet_bundle.get("tangent_drift")
    zero_scalar = states.new_tensor(0.0)
    if (
        knn_idx is None
        or knn_weights is None
        or frame is None
        or identifiable_tangent_cov is None
        or states.size(0) < 2
    ):
        return {
            "tau": None,
            "residual": zero_scalar,
            "correction_norm": zero_scalar,
        }

    batch, state_dim = states.shape
    tangent_dim = frame.size(-1)
    num_neighbors = knn_idx.size(1)
    base_logits = base_log_density.squeeze(-1).float()
    mu0 = torch.softmax(base_logits, dim=0)

    neighbor_states = states.index_select(0, knn_idx.reshape(-1)).view(batch, num_neighbors, state_dim)
    delta = neighbor_states - states.unsqueeze(1)
    coords = torch.einsum("bkd,bdm->bkm", delta, frame).float()

    eye_tangent = torch.eye(tangent_dim, device=states.device, dtype=torch.float32).unsqueeze(0)
    geom_cov = identifiable_tangent_cov.float() + 1e-4 * eye_tangent
    with torch.autocast(device_type=states.device.type, enabled=False):
        geom_precision = torch.linalg.inv(geom_cov)
    quad = torch.einsum("bkm,bmn,bkn->bk", coords, geom_precision, coords)
    geom_logits = -0.5 * quad / max(temperature, 1e-4)
    if tangent_drift is not None and drift_scale != 0.0:
        drift_scores = (coords * tangent_drift.float().unsqueeze(1)).sum(dim=-1)
        geom_logits = geom_logits + float(drift_scale) * drift_scores
    geom_weights = torch.softmax(geom_logits, dim=1)

    p_base = torch.zeros(batch, batch, device=states.device, dtype=torch.float32)
    p_geom = torch.zeros(batch, batch, device=states.device, dtype=torch.float32)
    p_base.scatter_(1, knn_idx, knn_weights.float())
    p_geom.scatter_(1, knn_idx, geom_weights)

    # Only solve for the conditional increment relative to the base graph generator.
    # This keeps tau focused on the condition-induced imbalance instead of absorbing
    # whatever non-stationarity the base density/head already has on its own.
    residual = p_base.transpose(0, 1) @ mu0 - p_geom.transpose(0, 1) @ mu0
    identity = torch.eye(batch, device=states.device, dtype=torch.float32)
    a_base = p_base.transpose(0, 1) - identity
    ones = torch.ones(batch, 1, device=states.device, dtype=torch.float32)
    normal = a_base.transpose(0, 1) @ a_base
    normal = normal + float(ridge) * identity + float(mean_penalty) * (ones @ ones.transpose(0, 1))
    rhs = (a_base.transpose(0, 1) @ residual).float()
    with torch.autocast(device_type=states.device.type, enabled=False):
        nu = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)

    tau = nu / mu0.clamp_min(1e-6)
    weighted_mean = (tau * mu0).sum()
    tau = tau - weighted_mean

    correction_residual = a_base @ nu - residual
    return {
        "tau": tau.to(dtype=states.dtype).unsqueeze(-1),
        "residual": correction_residual.norm().to(dtype=states.dtype),
        "correction_norm": nu.norm().to(dtype=states.dtype),
    }


@torch.no_grad()
def local_measure_targets(
    model,
    latents: torch.Tensor,
    video: torch.Tensor,
    cond_embed: torch.Tensor,
    *,
    diffusion_target_mode: str = "full",
    signature_mode: str = "span_stats",
    measure_target_mode: str = "chart_moments",
    measure_target_blend: float = 0.5,
    drift_target_mode: str = "bootstrap",
    drift_target_blend: float = 0.5,
    tilt_target_mode: str = "none",
    tilt_target_blend: float = 0.5,
    geometry_knn: int = 8,
    geometry_temperature: float = 0.5,
    jet_ridge: float = 1e-3,
    jet_center_weight: float = 1.0,
    tau_ridge: float = 1e-3,
    tau_mean_penalty: float = 1.0,
    tau_drift_scale: float = 0.25,
    signatures: torch.Tensor | None = None,
    decoded: torch.Tensor | None = None,
    tangent_frame: torch.Tensor | None = None,
    target_model=None,
    target_cond_embed: torch.Tensor | None = None,
    geometry_reference: GeometryNeighborhoodReference | None = None,
) -> LocalGeneratorTarget:
    if latents.size(1) < 2:
        zero_scalar = latents.new_tensor(0.0)
        zero_vector = latents.new_zeros(latents.size(0), latents.size(-1))
        zero_matrix = latents.new_zeros(latents.size(0), latents.size(-1), latents.size(-1))
        return LocalGeneratorTarget(
            signatures=latents.new_zeros(
                latents.size(0),
                response_signature_dim(latents.size(1), signature_mode, channels=video.size(2)),
            ),
            source_point=zero_vector,
            source_summary_context=zero_vector,
            source_measure=None,
            invariant_target=None,
            drift_target=zero_vector,
            diffusion_target=zero_matrix,
            full_diffusion_target=zero_matrix,
            bootstrap_drift_target=zero_vector,
            bootstrap_diffusion_target=zero_matrix,
            target_tangent_drift=zero_vector,
            target_tangent_cov=None,
            identifiable_tangent_cov=None,
            target_tangent_frame=None,
            target_tangent_projector=None,
            target_neighbor_idx=None,
            target_neighbor_weights=None,
            target_transport=None,
            response_identifiable_effective_rank=zero_scalar,
            response_identifiable_anisotropy=zero_scalar,
            tilt_target=None,
            response_operator_trace=zero_scalar,
            response_operator_effective_rank=zero_scalar,
            response_operator_anisotropy=zero_scalar,
            response_operator_asymmetry=zero_scalar,
            response_drift_alignment=zero_scalar,
            geometry_neighbor_pool_size=latents.new_tensor(0.0),
            geometry_reference_pool_size=latents.new_tensor(0.0),
            geometry_reference_neighbor_ratio=zero_scalar,
        )

    source_model = target_model if target_model is not None else model
    source_cond_embed = target_cond_embed if target_cond_embed is not None else cond_embed
    source_latents = latents
    source_decoded = decoded
    if target_model is not None:
        source_latents = source_model.encode_video(video, cond_embed=source_cond_embed)
        source_decoded = source_model.decode_video(source_latents, source_cond_embed)
    source_point = source_model.trajectory_point(source_latents)
    source_summary_context = source_model.trajectory_summary_context(source_latents)

    response_bundle = None
    triangle = None
    triangle_mask = None
    needs_response_structure = (
        (measure_target_mode != "chart_moments")
        or (drift_target_mode != "bootstrap")
        or (tilt_target_mode in {"response_support", "graph_tau", "hybrid"})
    )
    if signatures is None or needs_response_structure:
        response_bundle = response_triangle_bundle(
            model=source_model,
            latents=source_latents,
            video=video,
            cond_embed=source_cond_embed,
            decoded=source_decoded,
        )
        triangle = response_bundle.energy_triangle
        triangle_mask = response_bundle.mask
    if signatures is None:
        if triangle is None or triangle_mask is None:
            signatures = response_signature(
                model=source_model,
                latents=source_latents,
                video=video,
                cond_embed=source_cond_embed,
                decoded=source_decoded,
                mode=signature_mode,
            )
        else:
            descriptor_triangle, signed_triangle, _ = response_descriptor_triangle_from_bundle(response_bundle)
            signatures = _response_signature_from_descriptor_triangle(
                descriptor_triangle,
                signed_triangle,
                triangle_mask,
                signature_mode,
            )

    chart_latents = source_model.chart_latents(source_latents)
    chart_delta = chart_latents[:, 1:] - chart_latents[:, :-1]
    bootstrap_drift_target = chart_delta.mean(dim=1)
    centered_delta = chart_delta - bootstrap_drift_target.unsqueeze(1)
    bootstrap_full_diffusion_target = torch.matmul(centered_delta.transpose(1, 2), centered_delta) / float(centered_delta.size(1))

    student_tangent_frame = tangent_frame
    if needs_response_structure and student_tangent_frame is None:
        student_tangent_frame = model.trajectory_tangent_frame(
            latents,
            point=model.trajectory_point(latents),
            summary_context=model.trajectory_summary_context(latents),
        )
    source_tangent_frame = None
    if needs_response_structure:
        source_tangent_frame = source_model.trajectory_tangent_frame(
            source_latents,
            point=source_point,
            summary_context=source_summary_context,
        )
    target_tangent_dim = 0
    if student_tangent_frame is not None:
        target_tangent_dim = student_tangent_frame.size(-1)
    elif source_tangent_frame is not None:
        target_tangent_dim = source_tangent_frame.size(-1)

    drift_target = bootstrap_drift_target
    full_diffusion_target = bootstrap_full_diffusion_target
    target_tangent_drift = None
    target_tangent_cov = None
    identifiable_tangent_cov = None
    target_tangent_frame = None
    target_tangent_projector = None
    target_neighbor_idx = None
    target_neighbor_weights = None
    target_transport = None
    if student_tangent_frame is not None:
        target_tangent_cov = student_tangent_frame.transpose(-1, -2) @ bootstrap_full_diffusion_target @ student_tangent_frame
    response_operator_trace = latents.new_tensor(0.0)
    response_operator_effective_rank = latents.new_tensor(0.0)
    response_operator_anisotropy = latents.new_tensor(0.0)
    response_operator_asymmetry = latents.new_tensor(0.0)
    response_drift_alignment = latents.new_tensor(0.0)
    response_identifiable_effective_rank = latents.new_tensor(0.0)
    response_identifiable_anisotropy = latents.new_tensor(0.0)
    geometry_neighbor_pool_size = latents.new_tensor(float(source_point.size(0)))
    geometry_reference_pool_size = latents.new_tensor(0.0)
    geometry_reference_neighbor_ratio = latents.new_tensor(0.0)

    if measure_target_mode not in {"chart_moments", "response_invariant_bootstrap", "hybrid", "response_jet"}:
        raise ValueError(f"Unsupported measure_target_mode: {measure_target_mode}")
    if drift_target_mode not in {"bootstrap", "response_asymmetry", "hybrid", "response_jet"}:
        raise ValueError(f"Unsupported drift_target_mode: {drift_target_mode}")
    if tilt_target_mode not in {"none", "teacher_tilt", "response_support", "graph_tau", "hybrid"}:
        raise ValueError(f"Unsupported tilt_target_mode: {tilt_target_mode}")

    tilt_target = None
    teacher_tilt_target = None
    base_log_density_target = None
    source_measure = None
    invariant_target = None
    if tilt_target_mode in {"teacher_tilt", "hybrid", "graph_tau"} or source_model.measure_density_mode == "tilted":
        source_measure = source_model.conditional_measure(
            source_latents,
            source_cond_embed,
            state=source_point,
        )
        base_log_density_target = source_measure.log_base_density
        teacher_tilt_target = source_measure.log_tilt
        total_log_density_target = source_measure.log_total_density
        if source_model.measure_density_mode != "tilted":
            teacher_tilt_target = None

    if needs_response_structure and response_bundle is not None and triangle is not None and triangle_mask is not None:
        invariant_target = build_response_invariant_target(
            source_point,
            response_bundle,
            tangent_dim=target_tangent_dim,
            geometry_knn=geometry_knn,
            geometry_temperature=geometry_temperature,
            jet_ridge=jet_ridge,
            jet_center_weight=jet_center_weight,
            base_log_density_target=base_log_density_target,
            tilt_target_mode=tilt_target_mode,
            tau_ridge=tau_ridge,
            tau_mean_penalty=tau_mean_penalty,
            tau_drift_scale=tau_drift_scale,
            geometry_reference=geometry_reference,
        )
        response_operator_trace = invariant_target.trace.mean()
        response_operator_effective_rank = invariant_target.effective_rank.mean()
        response_operator_anisotropy = invariant_target.anisotropy.mean()
        response_operator_asymmetry = invariant_target.asymmetry.mean()
        geometry_neighbor_pool_size = invariant_target.neighbor_pool_size
        geometry_reference_pool_size = invariant_target.reference_pool_size
        geometry_reference_neighbor_ratio = invariant_target.reference_neighbor_ratio
        response_support_target = invariant_target.support_tilt

        if tilt_target_mode == "response_support":
            tilt_target = response_support_target
        elif tilt_target_mode == "graph_tau":
            tilt_target = invariant_target.graph_tau if invariant_target.graph_tau is not None else response_support_target
        elif tilt_target_mode == "hybrid":
            if teacher_tilt_target is None:
                tilt_target = response_support_target
            else:
                tilt_blend = float(min(max(tilt_target_blend, 0.0), 1.0))
                tilt_target = (1.0 - tilt_blend) * teacher_tilt_target + tilt_blend * response_support_target

        if invariant_target.tangent_frame is not None:
            target_tangent_frame = invariant_target.tangent_frame
            target_tangent_projector = invariant_target.tangent_projector
            target_neighbor_idx = invariant_target.neighbor_idx
            target_neighbor_weights = invariant_target.neighbor_weights
            target_transport = invariant_target.transport
            response_identifiable_effective_rank = invariant_target.identifiable_effective_rank
            response_identifiable_anisotropy = invariant_target.identifiable_anisotropy

        if student_tangent_frame is not None:
            if source_tangent_frame is not None:
                source_bootstrap_tangent_drift = torch.einsum(
                    "bij,bj->bi",
                    source_tangent_frame.transpose(-1, -2),
                    bootstrap_drift_target,
                )
            else:
                source_bootstrap_tangent_drift = torch.einsum(
                    "bij,bj->bi",
                    student_tangent_frame.transpose(-1, -2),
                    bootstrap_drift_target,
                )
            bootstrap_tangent_drift = source_bootstrap_tangent_drift
            asym_tangent_drift = torch.zeros_like(bootstrap_tangent_drift)
            drift_scale = source_bootstrap_tangent_drift.norm(dim=-1, keepdim=True)
            asym_value = invariant_target.asymmetry.unsqueeze(-1)
            asym_tangent_drift[:, :1] = torch.sign(asym_value) * drift_scale * asym_value.abs().clamp_max(1.0)
            if drift_target_mode == "response_jet" and invariant_target.tangent_drift is not None:
                target_tangent_drift = invariant_target.tangent_drift
                if target_tangent_frame is not None:
                    drift_target = torch.einsum("bij,bj->bi", target_tangent_frame, target_tangent_drift)
                else:
                    drift_target = torch.einsum("bij,bj->bi", student_tangent_frame, target_tangent_drift)
                bootstrap_in_target_frame = (
                    torch.einsum("bij,bj->bi", target_tangent_frame.transpose(-1, -2), bootstrap_drift_target)
                    if target_tangent_frame is not None
                    else bootstrap_tangent_drift
                )
                response_drift_alignment = F.mse_loss(bootstrap_in_target_frame, target_tangent_drift)
            elif drift_target_mode == "response_asymmetry":
                target_tangent_drift = asym_tangent_drift
            elif drift_target_mode == "hybrid":
                drift_blend = float(min(max(drift_target_blend, 0.0), 1.0))
                target_tangent_drift = (1.0 - drift_blend) * bootstrap_tangent_drift + drift_blend * asym_tangent_drift
            else:
                target_tangent_drift = bootstrap_tangent_drift
            if drift_target_mode != "response_jet" or invariant_target.tangent_drift is None:
                drift_target = torch.einsum("bij,bj->bi", student_tangent_frame, target_tangent_drift)
                response_drift_alignment = F.mse_loss(bootstrap_tangent_drift[:, :1], asym_tangent_drift[:, :1])

        if measure_target_mode != "chart_moments" and student_tangent_frame is not None:
            if measure_target_mode == "response_jet" and invariant_target.tangent_cov is not None:
                target_tangent_cov = invariant_target.tangent_cov
                identifiable_tangent_cov = invariant_target.identifiable_tangent_cov
                if target_tangent_frame is not None:
                    full_diffusion_target = (
                        target_tangent_frame @ target_tangent_cov @ target_tangent_frame.transpose(-1, -2)
                    )
            else:
                tangent_dim = student_tangent_frame.size(-1)
                identifiable_tangent_cov = invariant_target.identifiable_tangent_cov
                if identifiable_tangent_cov is None:
                    target_eigs = invariant_target.eigvals[:, : min(tangent_dim, invariant_target.eigvals.size(-1))]
                    spectral_weights = target_eigs / target_eigs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    if tangent_dim > spectral_weights.size(1):
                        spectral_weights = F.pad(spectral_weights, (0, tangent_dim - spectral_weights.size(1)))
                    spectral_weights = spectral_weights.clamp_min(1e-8)
                    spectral_weights = spectral_weights / spectral_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    identifiable_tangent_cov = torch.diag_embed(spectral_weights)
                    response_identifiable_effective_rank = torch.exp(
                        -(spectral_weights * spectral_weights.clamp_min(1e-8).log()).sum(dim=-1)
                    ).mean()
                    response_identifiable_anisotropy = torch.log(spectral_weights.clamp_min(1e-8)).std(
                        dim=-1,
                        unbiased=False,
                    ).mean()
                else:
                    response_identifiable_effective_rank = invariant_target.identifiable_effective_rank
                    response_identifiable_anisotropy = invariant_target.identifiable_anisotropy
                bootstrap_trace = bootstrap_full_diffusion_target.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                invariant_trace = identifiable_tangent_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-8)
                invariant_tangent_cov = identifiable_tangent_cov * (bootstrap_trace / invariant_trace).view(-1, 1, 1)
                bootstrap_tangent_cov = (
                    student_tangent_frame.transpose(-1, -2) @ bootstrap_full_diffusion_target @ student_tangent_frame
                )

                if measure_target_mode == "response_invariant_bootstrap":
                    target_tangent_cov = invariant_tangent_cov
                else:
                    blend = float(min(max(measure_target_blend, 0.0), 1.0))
                    target_tangent_cov = (1.0 - blend) * bootstrap_tangent_cov + blend * invariant_tangent_cov
                full_diffusion_target = (
                    student_tangent_frame @ target_tangent_cov @ student_tangent_frame.transpose(-1, -2)
                )
    elif tilt_target_mode == "teacher_tilt":
        tilt_target = teacher_tilt_target

    diffusion_target = full_diffusion_target
    if diffusion_target_mode == "diag":
        diffusion_target = torch.diag_embed(diffusion_target.diagonal(dim1=-2, dim2=-1))
    elif diffusion_target_mode != "full":
        raise ValueError(f"Unsupported diffusion_target_mode: {diffusion_target_mode}")

    if target_tangent_drift is None:
        if student_tangent_frame is None:
            tangent_drift_target_out = bootstrap_drift_target
        else:
            tangent_drift_target_out = torch.einsum(
                "bij,bj->bi",
                student_tangent_frame.transpose(-1, -2),
                drift_target,
            )
    else:
        tangent_drift_target_out = target_tangent_drift

    return LocalGeneratorTarget(
        signatures=signatures,
        source_point=source_point,
        source_summary_context=source_summary_context,
        source_measure=source_measure,
        invariant_target=invariant_target,
        drift_target=drift_target,
        diffusion_target=diffusion_target,
        full_diffusion_target=full_diffusion_target,
        bootstrap_drift_target=bootstrap_drift_target,
        bootstrap_diffusion_target=bootstrap_full_diffusion_target,
        target_tangent_drift=tangent_drift_target_out,
        target_tangent_cov=target_tangent_cov,
        identifiable_tangent_cov=identifiable_tangent_cov,
        target_tangent_frame=target_tangent_frame,
        target_tangent_projector=target_tangent_projector,
        target_neighbor_idx=target_neighbor_idx,
        target_neighbor_weights=target_neighbor_weights,
        target_transport=target_transport,
        response_identifiable_effective_rank=response_identifiable_effective_rank,
        response_identifiable_anisotropy=response_identifiable_anisotropy,
        tilt_target=tilt_target,
        response_operator_trace=response_operator_trace,
        response_operator_effective_rank=response_operator_effective_rank,
        response_operator_anisotropy=response_operator_anisotropy,
        response_operator_asymmetry=response_operator_asymmetry,
        response_drift_alignment=response_drift_alignment,
        geometry_neighbor_pool_size=geometry_neighbor_pool_size,
        geometry_reference_pool_size=geometry_reference_pool_size,
        geometry_reference_neighbor_ratio=geometry_reference_neighbor_ratio,
    )


@torch.no_grad()
def response_signature(
    model,
    latents: torch.Tensor,
    video: torch.Tensor,
    cond_embed: torch.Tensor,
    decoded: torch.Tensor | None = None,
    mode: str = "span_stats",
) -> torch.Tensor:
    batch, steps, *_ = video.shape
    if steps < 2:
        return latents.new_zeros(batch, response_signature_dim(steps, mode, channels=video.size(2)))
    bundle = response_triangle_bundle(
        model=model,
        latents=latents,
        video=video,
        cond_embed=cond_embed,
        decoded=decoded,
    )
    descriptor_triangle, signed_triangle, _ = response_descriptor_triangle_from_bundle(bundle)
    return _response_signature_from_descriptor_triangle(
        descriptor_triangle,
        signed_triangle,
        bundle.mask,
        mode,
    )


def local_neighbor_smoothness_loss(
    signatures: torch.Tensor,
    fields: list[torch.Tensor],
    *,
    knn: int,
    temperature: float,
    reference_signatures: torch.Tensor | None = None,
    reference_fields: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    batch = signatures.size(0)
    reference_count = 0 if reference_signatures is None else int(reference_signatures.size(0))
    if batch < 1 or batch + reference_count < 2 or knn <= 0:
        return signatures.new_tensor(0.0)
    if reference_fields is not None and len(reference_fields) != len(fields):
        raise ValueError("reference_fields must match fields length")
    neighborhood = _build_knn_graph(
        signatures,
        knn=knn,
        temperature=temperature,
        reference_points=reference_signatures,
    )
    knn_idx = neighborhood.knn_idx
    weights = neighborhood.weights
    if knn_idx is None or weights is None:
        return signatures.new_tensor(0.0)

    total = signatures.new_tensor(0.0)
    for field_idx, field in enumerate(fields):
        flat_field = field if field.ndim == 2 else field.unsqueeze(1)
        candidate_field = flat_field
        if reference_fields is not None:
            reference_field = reference_fields[field_idx]
            if reference_field is not None and reference_field.numel() > 0:
                reference_field = reference_field if reference_field.ndim == 2 else reference_field.unsqueeze(1)
                candidate_field = torch.cat(
                    [
                        flat_field,
                        reference_field.to(device=flat_field.device, dtype=flat_field.dtype),
                    ],
                    dim=0,
                )
        neighbor_field = candidate_field.index_select(0, knn_idx.reshape(-1)).view(batch, knn_idx.size(1), -1)
        diff = flat_field.unsqueeze(1) - neighbor_field
        total = total + (weights.unsqueeze(-1) * diff.square()).sum(dim=1).mean()

    return total / max(len(fields), 1)


def _measure_test_directions(
    dim: int,
    num_directions: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_directions <= 0:
        raise ValueError("num_directions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    directions = torch.randn(num_directions, dim, generator=generator, dtype=torch.float32)
    directions = F.normalize(directions, dim=-1, eps=1e-6).to(device=device, dtype=dtype)
    return directions


def _sorted_eigvalsh(matrix: torch.Tensor) -> torch.Tensor:
    eigvals = torch.linalg.eigvalsh(matrix.float())
    eigvals = torch.flip(eigvals, dims=(-1,))
    return eigvals


def identification_nontriviality_loss(
    pred_eigs: torch.Tensor,
    target_eigs: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_eigs = pred_eigs.clamp_min(eps)
    target_eigs = target_eigs.clamp_min(eps)

    pred_log = torch.log(pred_eigs)
    target_log = torch.log(target_eigs)
    pred_anisotropy = pred_log.std(dim=-1, unbiased=False)
    target_anisotropy = target_log.std(dim=-1, unbiased=False)

    pred_weights = pred_eigs / pred_eigs.sum(dim=-1, keepdim=True).clamp_min(eps)
    target_weights = target_eigs / target_eigs.sum(dim=-1, keepdim=True).clamp_min(eps)
    pred_effective_rank = torch.exp(-(pred_weights * pred_weights.clamp_min(eps).log()).sum(dim=-1))
    target_effective_rank = torch.exp(-(target_weights * target_weights.clamp_min(eps).log()).sum(dim=-1))

    if pred_eigs.size(-1) > 1:
        pred_gap = (pred_eigs[..., 0] - pred_eigs[..., 1]) / pred_eigs.sum(dim=-1).clamp_min(eps)
        target_gap = (target_eigs[..., 0] - target_eigs[..., 1]) / target_eigs.sum(dim=-1).clamp_min(eps)
    else:
        pred_gap = pred_eigs[..., 0] / pred_eigs.sum(dim=-1).clamp_min(eps)
        target_gap = target_eigs[..., 0] / target_eigs.sum(dim=-1).clamp_min(eps)

    anisotropy_floor = F.relu(target_anisotropy - pred_anisotropy)
    rank_ceiling = F.relu(pred_effective_rank - target_effective_rank)
    gap_floor = F.relu(target_gap - pred_gap)
    loss = anisotropy_floor.mean() + rank_ceiling.mean() + gap_floor.mean()
    return loss, {
        "pred_effective_rank": pred_effective_rank.mean(),
        "target_effective_rank": target_effective_rank.mean(),
        "pred_anisotropy": pred_anisotropy.mean(),
        "target_anisotropy": target_anisotropy.mean(),
        "pred_spectral_gap": pred_gap.mean(),
        "target_spectral_gap": target_gap.mean(),
    }


def tilt_overreach_loss(
    tilt_log_density: torch.Tensor,
    *,
    geometry_signal: torch.Tensor,
    geometry_residual: torch.Tensor,
) -> torch.Tensor:
    signal_gate = torch.tanh(geometry_signal.detach().clamp_min(0.0))
    residual_gate = torch.tanh(geometry_residual.detach().clamp_min(0.0))
    return tilt_log_density.abs().mean() * signal_gate * residual_gate


def local_measure_loss(
    model,
    latents: torch.Tensor,
    video: torch.Tensor,
    cond_embed: torch.Tensor,
    *,
    signature_knn: int,
    signature_temperature: float,
    geometry_knn: int,
    geometry_temperature: float,
    jet_ridge: float,
    jet_center_weight: float,
    tau_ridge: float,
    tau_mean_penalty: float,
    tau_drift_scale: float,
    density_temperature: float,
    test_num_directions: int,
    trig_scale: float,
    diffusion_target_mode: str = "full",
    measure_target_mode: str = "chart_moments",
    measure_target_blend: float = 0.5,
    drift_target_mode: str = "bootstrap",
    drift_target_blend: float = 0.5,
    tilt_target_mode: str = "none",
    tilt_target_blend: float = 0.5,
    signature_mode: str = "span_stats",
    signatures: torch.Tensor | None = None,
    decoded: torch.Tensor | None = None,
    target_model=None,
    target_cond_embed: torch.Tensor | None = None,
    geometry_reference: GeometryNeighborhoodReference | None = None,
    smoothness_reference: SmoothnessNeighborhoodReference | None = None,
) -> dict[str, torch.Tensor]:
    if latents.size(1) < 2:
        zero = latents.new_tensor(0.0)
        return {
            "local_drift": zero,
            "local_diffusion": zero,
            "measure_stationarity": zero,
            "measure_linear_stationarity": zero,
            "measure_quadratic_stationarity": zero,
            "measure_trig_stationarity": zero,
            "measure_trace_alignment": zero,
            "response_smoothness": zero,
            "response_signature_norm": zero,
            "measure_density_entropy": zero,
            "measure_tilt_abs_mean": zero,
            "measure_tilt_alignment": zero,
            "measure_pred_trace": zero,
            "measure_target_trace": zero,
            "tangent_projection": zero,
            "tangent_observation_residual": zero,
            "tangent_drift_residual": zero,
            "tangent_diffusion_residual": zero,
            "tangent_bundle_compatibility": zero,
            "tangent_frame_orthogonality": zero,
            "tangent_projector_trace": zero,
            "tangent_spectrum_alignment": zero,
            "tangent_shape_alignment": zero,
            "tangent_nontriviality": zero,
            "tangent_anisotropy_gap": zero,
            "pred_tangent_effective_rank": zero,
            "target_tangent_effective_rank": zero,
            "pred_tangent_anisotropy": zero,
            "target_tangent_anisotropy": zero,
            "pred_tangent_trace": zero,
            "target_tangent_trace": zero,
            "pred_tangent_spectral_gap": zero,
            "target_tangent_spectral_gap": zero,
            "response_operator_trace": zero,
            "response_operator_effective_rank": zero,
            "response_operator_anisotropy": zero,
            "response_operator_asymmetry": zero,
            "response_identifiable_effective_rank": zero,
            "response_identifiable_anisotropy": zero,
            "response_drift_alignment": zero,
            "measure_tilt_overreach": zero,
            "generator_base_trace": zero,
            "generator_delta_trace": zero,
            "generator_delta_drift_norm": zero,
            "generator_delta_diffusion_norm": zero,
            "generator_delta_tilt_abs_mean": zero,
            "generator_delta_budget": zero,
            "response_geometry_pool_size": zero,
            "response_geometry_reference_pool_size": zero,
            "response_geometry_reference_neighbor_ratio": zero,
            "response_smoothness_pool_size": zero,
        }

    state = model.trajectory_point(latents)
    target_bundle = local_measure_targets(
        model=model,
        latents=latents,
        video=video,
        cond_embed=cond_embed,
        diffusion_target_mode=diffusion_target_mode,
        signature_mode=signature_mode,
        measure_target_mode=measure_target_mode,
        measure_target_blend=measure_target_blend,
        drift_target_mode=drift_target_mode,
        drift_target_blend=drift_target_blend,
        tilt_target_mode=tilt_target_mode,
        tilt_target_blend=tilt_target_blend,
        geometry_knn=geometry_knn,
        geometry_temperature=geometry_temperature,
        jet_ridge=jet_ridge,
        jet_center_weight=jet_center_weight,
        tau_ridge=tau_ridge,
        tau_mean_penalty=tau_mean_penalty,
        tau_drift_scale=tau_drift_scale,
        signatures=signatures,
        decoded=decoded,
        target_model=target_model,
        target_cond_embed=target_cond_embed,
        geometry_reference=geometry_reference,
    )
    signatures = target_bundle.signatures
    generator = model.local_generator(
        latents,
        cond_embed,
        response_context=signatures,
        state=state,
    )
    drift = generator.drift
    tangent_structure = generator.tangent_structure
    diffusion_matrix = generator.diffusion_matrix
    tangent_core_cov = generator.tangent_core_cov
    diffusion_diag = diffusion_matrix.diagonal(dim1=-2, dim2=-1)
    conditional_measure = generator.conditional_measure
    tilt_log_density = conditional_measure.log_tilt
    log_density = conditional_measure.log_total_density.squeeze(-1)
    base_generator = generator.base_generator
    conditional_delta = generator.conditional_delta
    chart_latents = model.chart_latents(latents)
    chart_delta = chart_latents[:, 1:] - chart_latents[:, :-1]

    drift_target = target_bundle.drift_target
    diffusion_target = target_bundle.diffusion_target
    full_diffusion_target = target_bundle.full_diffusion_target
    target_tangent_cov = target_bundle.target_tangent_cov
    identifiable_tangent_cov = target_bundle.identifiable_tangent_cov
    target_tangent_frame = target_bundle.target_tangent_frame
    target_tangent_projector = target_bundle.target_tangent_projector
    target_neighbor_idx = target_bundle.target_neighbor_idx
    target_neighbor_weights = target_bundle.target_neighbor_weights
    target_transport = target_bundle.target_transport
    tilt_target = target_bundle.tilt_target
    invariant_target = target_bundle.invariant_target

    drift_loss = F.mse_loss(drift, drift_target)
    diffusion_loss = F.mse_loss(diffusion_matrix, diffusion_target)
    tilt_alignment = latents.new_tensor(0.0) if tilt_target is None else F.mse_loss(tilt_log_density, tilt_target)

    density_weights = generator.density_weights(temperature=density_temperature)
    directions = _measure_test_directions(
        state.size(1),
        test_num_directions,
        device=state.device,
        dtype=state.dtype,
    )
    projected_state = generator.state @ directions.T
    projected_drift = generator.apply_linear(directions)
    projected_diffusion = torch.einsum("bde,kd,ke->bk", generator.diffusion_matrix, directions, directions)

    linear_moment = (density_weights.unsqueeze(1) * generator.apply_linear(directions)).sum(dim=0)
    quadratic_moment = (density_weights.unsqueeze(1) * generator.apply_quadratic(directions)).sum(dim=0)
    trig_moment = (density_weights.unsqueeze(1) * generator.apply_trig(directions, trig_scale)).sum(dim=0)
    radial_moment = (density_weights * generator.apply_radial()).sum()
    pred_trace = generator.trace()
    target_trace = full_diffusion_target.diagonal(dim1=-2, dim2=-1).sum(dim=1)
    trace_alignment = (
        torch.log(pred_trace.clamp_min(1e-8)) - torch.log(target_trace.clamp_min(1e-8))
    ).abs().mean()
    stationarity_loss = (
        linear_moment.square().mean()
        + quadratic_moment.square().mean()
        + trig_moment.square().mean()
        + radial_moment.square()
    )

    smoothness_loss = local_neighbor_smoothness_loss(
        signatures=signatures,
        fields=[state, drift, diffusion_matrix.flatten(1), log_density],
        knn=signature_knn,
        temperature=signature_temperature,
        reference_signatures=None if smoothness_reference is None else smoothness_reference.signatures,
        reference_fields=None
        if smoothness_reference is None
        else [
            smoothness_reference.state,
            smoothness_reference.drift,
            smoothness_reference.diffusion_flat,
            smoothness_reference.log_density,
        ],
    )
    smoothness_pool_size = signatures.new_tensor(
        float(signatures.size(0) + (0 if smoothness_reference is None else smoothness_reference.size()))
    )
    tangent_projector = tangent_structure["projector"] if tangent_structure is not None else model.trajectory_tangent_projector(latents)
    tangent_diag = model.trajectory_tangent_diagnostics(latents)
    if tangent_projector is None:
        tangent_observation_residual = latents.new_tensor(0.0)
        tangent_drift_residual = latents.new_tensor(0.0)
        tangent_diffusion_residual = latents.new_tensor(0.0)
        tangent_bundle_compatibility = latents.new_tensor(0.0)
        tangent_spectrum_alignment = latents.new_tensor(0.0)
        tangent_shape_alignment = latents.new_tensor(0.0)
        tangent_nontriviality = latents.new_tensor(0.0)
        tangent_anisotropy_gap = latents.new_tensor(0.0)
        pred_tangent_effective_rank = latents.new_tensor(0.0)
        target_tangent_effective_rank = latents.new_tensor(0.0)
        pred_tangent_anisotropy = latents.new_tensor(0.0)
        target_tangent_anisotropy = latents.new_tensor(0.0)
        pred_tangent_trace = latents.new_tensor(0.0)
        target_tangent_trace = latents.new_tensor(0.0)
        pred_tangent_spectral_gap = latents.new_tensor(0.0)
        target_tangent_spectral_gap = latents.new_tensor(0.0)
    else:
        tangent_observation = torch.einsum("bij,btj->bti", tangent_projector, chart_delta)
        tangent_observation_residual = F.mse_loss(tangent_observation, chart_delta)
        tangent_drift = torch.einsum("bij,bj->bi", tangent_projector, drift)
        tangent_drift_residual = F.mse_loss(tangent_drift, drift)
        tangent_diffusion = tangent_projector @ diffusion_matrix @ tangent_projector
        tangent_diffusion_residual = F.mse_loss(tangent_diffusion, diffusion_matrix)
        if target_tangent_projector is not None:
            tangent_projection = F.mse_loss(tangent_projector, target_tangent_projector)
        else:
            tangent_projection = (
                tangent_observation_residual + tangent_drift_residual + tangent_diffusion_residual
            ) / 3.0
        if (
            tangent_structure is not None
            and target_neighbor_idx is not None
            and target_neighbor_weights is not None
            and target_transport is not None
        ):
            pred_frame = tangent_structure["frame"]
            candidate_pred_frame = pred_frame
            if geometry_reference is not None and geometry_reference.tangent_frames is not None:
                candidate_pred_frame = torch.cat(
                    [
                        pred_frame,
                        geometry_reference.tangent_frames.to(device=pred_frame.device, dtype=pred_frame.dtype),
                    ],
                    dim=0,
                )
            neighbor_pred_frame = candidate_pred_frame.index_select(0, target_neighbor_idx.reshape(-1)).view(
                pred_frame.size(0),
                target_neighbor_idx.size(1),
                pred_frame.size(1),
                pred_frame.size(2),
            )
            transported_neighbor_frame = torch.matmul(neighbor_pred_frame, target_transport)
            tangent_bundle_compatibility = (
                target_neighbor_weights.unsqueeze(-1).unsqueeze(-1)
                * (pred_frame.unsqueeze(1) - transported_neighbor_frame).square()
            ).sum(dim=1).mean()
        else:
            tangent_bundle_compatibility = local_neighbor_smoothness_loss(
                signatures=signatures,
                fields=[tangent_projector.flatten(1)],
                knn=signature_knn,
                temperature=signature_temperature,
            )
        if tangent_core_cov is None or tangent_structure is None:
            tangent_spectrum_alignment = latents.new_tensor(0.0)
            tangent_shape_alignment = latents.new_tensor(0.0)
            tangent_nontriviality = latents.new_tensor(0.0)
            tangent_anisotropy_gap = latents.new_tensor(0.0)
            pred_tangent_effective_rank = latents.new_tensor(0.0)
            target_tangent_effective_rank = latents.new_tensor(0.0)
            pred_tangent_anisotropy = latents.new_tensor(0.0)
            target_tangent_anisotropy = latents.new_tensor(0.0)
            pred_tangent_trace = latents.new_tensor(0.0)
            target_tangent_trace = latents.new_tensor(0.0)
            pred_tangent_spectral_gap = latents.new_tensor(0.0)
            target_tangent_spectral_gap = latents.new_tensor(0.0)
        else:
            if target_tangent_cov is None:
                frame = tangent_structure["frame"]
                target_tangent_cov = frame.transpose(-1, -2) @ full_diffusion_target @ frame
            pred_tangent_eigs = _sorted_eigvalsh(tangent_core_cov)
            target_tangent_eigs = _sorted_eigvalsh(target_tangent_cov)
            pred_log_spectrum = torch.log(pred_tangent_eigs.clamp_min(1e-8))
            target_log_spectrum = torch.log(target_tangent_eigs.clamp_min(1e-8))
            tangent_spectrum_alignment = F.mse_loss(pred_log_spectrum, target_log_spectrum)
            pred_tangent_anisotropy = pred_log_spectrum.std(dim=-1, unbiased=False).mean()
            target_tangent_anisotropy = target_log_spectrum.std(dim=-1, unbiased=False).mean()
            pred_tangent_trace = pred_tangent_eigs.sum(dim=-1).mean()
            target_tangent_trace = target_tangent_eigs.sum(dim=-1).mean()
            pred_weights = pred_tangent_eigs / pred_tangent_eigs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            pred_entropy = -(pred_weights * pred_weights.clamp_min(1e-8).log()).sum(dim=-1)
            pred_tangent_effective_rank = torch.exp(pred_entropy).mean()
            pred_tangent_spectral_gap = (
                (pred_tangent_eigs[:, 0] - pred_tangent_eigs[:, 1]) / pred_tangent_eigs.sum(dim=-1).clamp_min(1e-8)
                if pred_tangent_eigs.size(1) > 1
                else pred_tangent_eigs[:, 0] / pred_tangent_eigs.sum(dim=-1).clamp_min(1e-8)
            ).mean()
            target_tangent_spectral_gap = (
                (target_tangent_eigs[:, 0] - target_tangent_eigs[:, 1]) / target_tangent_eigs.sum(dim=-1).clamp_min(1e-8)
                if target_tangent_eigs.size(1) > 1
                else target_tangent_eigs[:, 0] / target_tangent_eigs.sum(dim=-1).clamp_min(1e-8)
            ).mean()
            if identifiable_tangent_cov is None:
                tangent_shape_alignment = latents.new_tensor(0.0)
                tangent_nontriviality = latents.new_tensor(0.0)
                tangent_anisotropy_gap = latents.new_tensor(0.0)
                target_tangent_effective_rank = latents.new_tensor(0.0)
            else:
                identifiable_tangent_eigs = _sorted_eigvalsh(identifiable_tangent_cov)
                identifiable_log_spectrum = torch.log(identifiable_tangent_eigs.clamp_min(1e-8))
                pred_centered_log_spectrum = pred_log_spectrum - pred_log_spectrum.mean(dim=-1, keepdim=True)
                identifiable_centered_log_spectrum = identifiable_log_spectrum - identifiable_log_spectrum.mean(
                    dim=-1,
                    keepdim=True,
                )
                tangent_shape_alignment = F.mse_loss(
                    pred_centered_log_spectrum,
                    identifiable_centered_log_spectrum,
                )
                target_tangent_effective_rank = target_bundle.response_identifiable_effective_rank
                tangent_nontriviality, nontriviality_stats = identification_nontriviality_loss(
                    pred_tangent_eigs,
                    identifiable_tangent_eigs,
                )
                tangent_anisotropy_gap = (
                    pred_tangent_anisotropy - target_bundle.response_identifiable_anisotropy
                ).abs()
                if not torch.is_tensor(target_tangent_effective_rank):
                    target_tangent_effective_rank = latents.new_tensor(float(target_tangent_effective_rank))
                pred_tangent_effective_rank = nontriviality_stats["pred_effective_rank"]
                target_tangent_effective_rank = nontriviality_stats["target_effective_rank"]
                pred_tangent_anisotropy = nontriviality_stats["pred_anisotropy"]
                target_tangent_anisotropy = nontriviality_stats["target_anisotropy"]
                pred_tangent_spectral_gap = nontriviality_stats["pred_spectral_gap"]
                target_tangent_spectral_gap = nontriviality_stats["target_spectral_gap"]
    if tangent_projector is None:
        tangent_projection = latents.new_tensor(0.0)

    if invariant_target is None:
        measure_tilt_overreach = latents.new_tensor(0.0)
    else:
        geometry_signal = (
            invariant_target.anisotropy.mean()
            + invariant_target.spectral_gap.mean()
            + invariant_target.asymmetry.abs().mean()
        )
        geometry_residual = (
            drift_loss.detach()
            + diffusion_loss.detach()
            + tangent_projection.detach()
            + tangent_spectrum_alignment.detach()
            + tangent_shape_alignment.detach()
        )
        measure_tilt_overreach = tilt_overreach_loss(
            tilt_log_density,
            geometry_signal=geometry_signal,
            geometry_residual=geometry_residual,
        )

    density_entropy = -(density_weights * density_weights.clamp_min(1e-12).log()).sum()
    if base_generator is None or conditional_delta is None:
        generator_base_trace = latents.new_tensor(0.0)
        generator_delta_trace = latents.new_tensor(0.0)
        generator_delta_drift_norm = latents.new_tensor(0.0)
        generator_delta_diffusion_norm = latents.new_tensor(0.0)
        generator_delta_tilt_abs_mean = latents.new_tensor(0.0)
        generator_delta_budget = latents.new_tensor(0.0)
    else:
        generator_base_trace = base_generator.trace().mean()
        generator_delta_trace = conditional_delta.trace().abs().mean()
        generator_delta_drift_norm = conditional_delta.drift.square().mean().sqrt()
        generator_delta_diffusion_norm = conditional_delta.diffusion_matrix.square().mean().sqrt()
        generator_delta_tilt_abs_mean = conditional_delta.log_tilt.abs().mean()
        generator_delta_budget = conditional_delta.drift.square().mean() + conditional_delta.diffusion_matrix.square().mean()

    return {
        "local_drift": drift_loss,
        "local_diffusion": diffusion_loss,
        "measure_stationarity": stationarity_loss,
        "measure_linear_stationarity": linear_moment.square().mean(),
        "measure_quadratic_stationarity": quadratic_moment.square().mean(),
        "measure_trig_stationarity": trig_moment.square().mean(),
        "measure_trace_alignment": trace_alignment,
        "response_smoothness": smoothness_loss,
        "response_signature_norm": signatures.norm(dim=1).mean(),
        "measure_density_entropy": density_entropy,
        "measure_tilt_abs_mean": tilt_log_density.abs().mean(),
        "measure_tilt_alignment": tilt_alignment,
        "measure_pred_trace": pred_trace.mean(),
        "measure_target_trace": target_trace.mean(),
        "tangent_projection": tangent_projection,
        "tangent_observation_residual": tangent_observation_residual,
        "tangent_drift_residual": tangent_drift_residual,
        "tangent_diffusion_residual": tangent_diffusion_residual,
        "tangent_bundle_compatibility": tangent_bundle_compatibility,
        "tangent_frame_orthogonality": tangent_diag["tangent_frame_orthogonality"],
        "tangent_projector_trace": tangent_diag["tangent_projector_trace"],
        "tangent_spectrum_alignment": tangent_spectrum_alignment,
        "tangent_shape_alignment": tangent_shape_alignment,
        "tangent_nontriviality": tangent_nontriviality,
        "tangent_anisotropy_gap": tangent_anisotropy_gap,
        "pred_tangent_effective_rank": pred_tangent_effective_rank,
        "target_tangent_effective_rank": target_tangent_effective_rank,
        "pred_tangent_anisotropy": pred_tangent_anisotropy,
        "target_tangent_anisotropy": target_tangent_anisotropy,
        "pred_tangent_trace": pred_tangent_trace,
        "target_tangent_trace": target_tangent_trace,
        "pred_tangent_spectral_gap": pred_tangent_spectral_gap,
        "target_tangent_spectral_gap": target_tangent_spectral_gap,
        "response_operator_trace": target_bundle.response_operator_trace,
        "response_operator_effective_rank": target_bundle.response_operator_effective_rank,
        "response_operator_anisotropy": target_bundle.response_operator_anisotropy,
        "response_operator_asymmetry": target_bundle.response_operator_asymmetry,
        "response_identifiable_effective_rank": target_bundle.response_identifiable_effective_rank,
        "response_identifiable_anisotropy": target_bundle.response_identifiable_anisotropy,
        "response_drift_alignment": target_bundle.response_drift_alignment,
        "measure_tilt_overreach": measure_tilt_overreach,
        "generator_base_trace": generator_base_trace,
        "generator_delta_trace": generator_delta_trace,
        "generator_delta_drift_norm": generator_delta_drift_norm,
        "generator_delta_diffusion_norm": generator_delta_diffusion_norm,
        "generator_delta_tilt_abs_mean": generator_delta_tilt_abs_mean,
        "generator_delta_budget": generator_delta_budget,
        "response_geometry_pool_size": target_bundle.geometry_neighbor_pool_size,
        "response_geometry_reference_pool_size": target_bundle.geometry_reference_pool_size,
        "response_geometry_reference_neighbor_ratio": target_bundle.geometry_reference_neighbor_ratio,
        "response_smoothness_pool_size": smoothness_pool_size,
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
