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


@torch.no_grad()
def response_signature_dim(seq_len: int, mode: str = "span_stats") -> int:
    if seq_len < 2:
        return 1
    if mode == "span_stats":
        return max(2 * (seq_len - 1), 2)
    if mode == "full_triangle":
        return max(seq_len * (seq_len - 1) // 2, 1)
    raise ValueError(f"Unsupported response_signature mode: {mode}")


@torch.no_grad()
def response_triangle(
    model,
    latents: torch.Tensor,
    video: torch.Tensor,
    cond_embed: torch.Tensor,
    decoded: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, steps, *_ = video.shape
    if steps < 2:
        triangle = latents.new_zeros(batch, 1, 1)
        mask = torch.zeros(1, 1, dtype=torch.bool, device=latents.device)
        return triangle, mask

    if decoded is None:
        decoded = model.decode_video(latents, cond_embed)

    span_count = steps - 1
    triangle = latents.new_zeros(batch, span_count, span_count)
    mask = torch.zeros(span_count, span_count, dtype=torch.bool, device=latents.device)
    for span in range(1, steps):
        span_residuals: list[torch.Tensor] = []
        for start in range(steps - span):
            rollout, _ = model.rollout_from(latents[:, start], cond_embed, span)
            pred_frame = model.decode_video(rollout[:, span - 1 : span], cond_embed)[:, 0]
            prev_frame = decoded[:, start + span - 1]
            pred_delta = pred_frame - prev_frame
            true_delta = video[:, start + span] - video[:, start + span - 1]
            residual = ((pred_delta - true_delta) ** 2).flatten(1).mean(dim=1)
            span_residuals.append(residual)
        span_tensor = torch.stack(span_residuals, dim=1)
        triangle[:, span - 1, : span_tensor.size(1)] = span_tensor
        mask[span - 1, : span_tensor.size(1)] = True
    return triangle, mask


@torch.no_grad()
def _response_signature_from_triangle(
    triangle: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    batch, span_count, _ = triangle.shape
    if span_count < 1:
        return triangle.new_zeros(batch, 1)

    if mode not in {"span_stats", "full_triangle"}:
        raise ValueError(f"Unsupported response_signature mode: {mode}")

    components: list[torch.Tensor] = []
    for span_idx in range(span_count):
        valid_count = int(mask[span_idx].sum().item())
        if valid_count <= 0:
            continue
        span_tensor = triangle[:, span_idx, :valid_count]
        if mode == "full_triangle":
            components.append(span_tensor)
        else:
            components.append(span_tensor.mean(dim=1, keepdim=True))
            if span_tensor.size(1) > 1:
                components.append(span_tensor.std(dim=1, unbiased=False, keepdim=True))
            else:
                components.append(torch.zeros_like(span_tensor[:, :1]))
    if not components:
        return triangle.new_zeros(batch, response_signature_dim(span_count + 1, mode))
    return torch.cat(components, dim=1)


@torch.no_grad()
def _response_operator_from_triangle(
    triangle: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    mask_f = mask.to(dtype=triangle.dtype).unsqueeze(0)
    valid_counts = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    row_mean = (triangle * mask_f).sum(dim=-1, keepdim=True) / valid_counts
    centered = (triangle - row_mean) * mask_f
    row_scale = centered.square().sum(dim=-1, keepdim=True).div(valid_counts).sqrt().clamp_min(eps)
    normalized = centered / row_scale
    normalized = normalized * mask_f
    operator = normalized @ normalized.transpose(-1, -2)
    normalizer = mask_f.sum(dim=-1).amax(dim=-1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
    operator = operator / normalizer
    operator = 0.5 * (operator + operator.transpose(-1, -2))
    eigvals = _sorted_eigvalsh(operator).clamp_min(0.0)
    trace = eigvals.sum(dim=-1)
    spectral_mass = eigvals / trace.unsqueeze(-1).clamp_min(eps)
    entropy = -(spectral_mass * spectral_mass.clamp_min(eps).log()).sum(dim=-1)
    effective_rank = torch.exp(entropy)
    anisotropy = torch.log(eigvals.clamp_min(eps)).std(dim=-1, unbiased=False)
    start_positions = torch.linspace(-1.0, 1.0, triangle.size(-1), device=triangle.device, dtype=triangle.dtype)
    asymmetry_num = (triangle * mask_f * start_positions.view(1, 1, -1)).sum(dim=(-1, -2))
    asymmetry_den = (triangle.abs() * mask_f).sum(dim=(-1, -2)).clamp_min(eps)
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
def _flatten_response_channels(
    triangle: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    batch = triangle.size(0)
    channels: list[torch.Tensor] = []
    for span_idx in range(triangle.size(1)):
        valid_count = int(mask[span_idx].sum().item())
        if valid_count <= 0:
            continue
        channels.append(triangle[:, span_idx, :valid_count])
    if not channels:
        return triangle.new_zeros(batch, 1)
    return torch.cat(channels, dim=1)


@torch.no_grad()
def _build_knn_graph(
    points: torch.Tensor,
    *,
    knn: int,
    temperature: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    batch = points.size(0)
    if batch < 2 or knn <= 0:
        return None, None, None
    num_neighbors = min(knn, batch - 1)
    distance = torch.cdist(points.detach(), points.detach(), p=2.0)
    inf = torch.full_like(distance.diagonal(), float("inf"))
    distance = distance.clone()
    distance.diagonal().copy_(inf)
    knn_dist, knn_idx = torch.topk(distance, k=num_neighbors, largest=False, dim=1)
    weights = torch.softmax(-knn_dist / max(temperature, 1e-4), dim=1)
    return knn_idx, weights, knn_dist


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
) -> dict[str, torch.Tensor | None]:
    if response_channels.dim() > 2:
        response_channels = response_channels.reshape(response_channels.size(0), -1)
    elif response_channels.dim() == 1:
        response_channels = response_channels.unsqueeze(-1)

    batch, state_dim = states.shape
    zero_scalar = states.new_tensor(0.0)
    if batch < 2 or tangent_dim <= 0:
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
        }

    knn_idx, knn_weights, _ = _build_knn_graph(states, knn=knn, temperature=temperature)
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
        }

    tangent_dim = min(tangent_dim, state_dim)
    num_neighbors = knn_idx.size(1)
    neighbor_states = states.index_select(0, knn_idx.reshape(-1)).view(batch, num_neighbors, state_dim)
    delta = neighbor_states - states.unsqueeze(1)
    cov = torch.einsum("bk,bkd,bke->bde", knn_weights, delta, delta)
    with torch.autocast(device_type=states.device.type, enabled=False):
        eigvals, eigvecs = torch.linalg.eigh(cov.float())
    frame = torch.flip(eigvecs, dims=(-1,))[..., :tangent_dim].to(dtype=states.dtype)
    projector = frame @ frame.transpose(-1, -2)
    coords = torch.einsum("bkd,bdm->bkm", delta, frame)

    neighbor_responses = response_channels.index_select(0, knn_idx.reshape(-1)).view(batch, num_neighbors, -1)
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

    neighbor_frame = frame.index_select(0, knn_idx.reshape(-1)).view(batch, num_neighbors, state_dim, tangent_dim)
    center_frame = frame.unsqueeze(1).expand(-1, num_neighbors, -1, -1)
    overlap = torch.matmul(neighbor_frame.transpose(-1, -2).float(), center_frame.float())
    with torch.autocast(device_type=states.device.type, enabled=False):
        u, _, vh = torch.linalg.svd(overlap.float(), full_matrices=False)
    transport = (u @ vh).to(dtype=states.dtype)

    return {
        "frame": frame,
        "projector": projector,
        "neighbor_idx": knn_idx,
        "neighbor_weights": knn_weights,
        "transport": transport,
        "tangent_drift": tangent_drift,
        "tangent_cov": tangent_cov,
        "identifiable_tangent_cov": identifiable_tangent_cov,
        "support_tilt": support_tilt,
        "effective_rank": effective_rank_values.mean(),
        "anisotropy": anisotropy_values.mean(),
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

    residual = mu0 - p_geom.transpose(0, 1) @ mu0
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
    weighted_var = ((tau.square()) * mu0).sum().clamp_min(1e-6)
    tau = tau / weighted_var.sqrt()

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
) -> dict[str, torch.Tensor]:
    if latents.size(1) < 2:
        zero_scalar = latents.new_tensor(0.0)
        zero_vector = latents.new_zeros(latents.size(0), latents.size(-1))
        zero_matrix = latents.new_zeros(latents.size(0), latents.size(-1), latents.size(-1))
        return {
            "signatures": latents.new_zeros(latents.size(0), response_signature_dim(latents.size(1), signature_mode)),
            "drift_target": zero_vector,
            "diffusion_target": zero_matrix,
            "full_diffusion_target": zero_matrix,
            "bootstrap_drift_target": zero_vector,
            "bootstrap_diffusion_target": zero_matrix,
            "target_tangent_drift": zero_vector,
            "target_tangent_cov": None,
            "identifiable_tangent_cov": None,
            "target_tangent_frame": None,
            "target_tangent_projector": None,
            "target_neighbor_idx": None,
            "target_neighbor_weights": None,
            "target_transport": None,
            "response_identifiable_effective_rank": zero_scalar,
            "response_identifiable_anisotropy": zero_scalar,
            "tilt_target": None,
            "response_operator_trace": zero_scalar,
            "response_operator_effective_rank": zero_scalar,
            "response_operator_anisotropy": zero_scalar,
            "response_operator_asymmetry": zero_scalar,
            "response_drift_alignment": zero_scalar,
        }

    source_model = target_model if target_model is not None else model
    source_cond_embed = target_cond_embed if target_cond_embed is not None else cond_embed
    source_latents = latents
    source_decoded = decoded
    if target_model is not None:
        source_latents = source_model.encode_video(video)
        source_decoded = source_model.decode_video(source_latents, source_cond_embed)
    source_state = source_model.trajectory_state(source_latents)

    triangle = None
    triangle_mask = None
    needs_response_structure = (
        (measure_target_mode != "chart_moments")
        or (drift_target_mode != "bootstrap")
        or (tilt_target_mode in {"response_support", "hybrid"})
    )
    if signatures is None or needs_response_structure:
        triangle, triangle_mask = response_triangle(
            model=source_model,
            latents=source_latents,
            video=video,
            cond_embed=source_cond_embed,
            decoded=source_decoded,
        )
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
            signatures = _response_signature_from_triangle(triangle, triangle_mask, signature_mode)

    chart_latents = source_model.chart_latents(source_latents)
    chart_delta = chart_latents[:, 1:] - chart_latents[:, :-1]
    bootstrap_drift_target = chart_delta.mean(dim=1)
    centered_delta = chart_delta - bootstrap_drift_target.unsqueeze(1)
    bootstrap_full_diffusion_target = torch.matmul(centered_delta.transpose(1, 2), centered_delta) / float(centered_delta.size(1))

    student_tangent_frame = tangent_frame
    if needs_response_structure and student_tangent_frame is None:
        student_tangent_frame = model.trajectory_tangent_frame(latents)
    source_tangent_frame = None
    if needs_response_structure:
        source_tangent_frame = source_model.trajectory_tangent_frame(source_latents)
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

    if measure_target_mode not in {"chart_moments", "response_invariant_bootstrap", "hybrid", "response_jet"}:
        raise ValueError(f"Unsupported measure_target_mode: {measure_target_mode}")
    if drift_target_mode not in {"bootstrap", "response_asymmetry", "hybrid", "response_jet"}:
        raise ValueError(f"Unsupported drift_target_mode: {drift_target_mode}")
    if tilt_target_mode not in {"none", "teacher_tilt", "response_support", "graph_tau", "hybrid"}:
        raise ValueError(f"Unsupported tilt_target_mode: {tilt_target_mode}")

    tilt_target = None
    teacher_tilt_target = None
    base_log_density_target = None
    if tilt_target_mode in {"teacher_tilt", "hybrid", "graph_tau"} or source_model.measure_density_mode == "tilted":
        base_log_density_target, teacher_tilt_target, total_log_density_target = source_model.measure_log_density_components(
            source_latents,
            source_cond_embed,
            state=source_state,
        )
        if source_model.measure_density_mode != "tilted":
            base_log_density_target = total_log_density_target
            teacher_tilt_target = None

    response_operator = None
    response_channels = None
    response_jet = None
    graph_tau_target = None
    if needs_response_structure and triangle is not None and triangle_mask is not None:
        response_channels = _flatten_response_channels(triangle, triangle_mask)
        if target_tangent_dim > 0:
            response_jet = _local_response_jet_bundle(
                source_state,
                response_channels,
                tangent_dim=target_tangent_dim,
                knn=geometry_knn,
                temperature=geometry_temperature,
                ridge=jet_ridge,
                center_weight=jet_center_weight,
            )
            if tilt_target_mode == "graph_tau" and base_log_density_target is not None:
                graph_tau_bundle = _solve_graph_tau_target(
                    source_state,
                    base_log_density_target,
                    jet_bundle=response_jet,
                    temperature=geometry_temperature,
                    ridge=tau_ridge,
                    mean_penalty=tau_mean_penalty,
                    drift_scale=tau_drift_scale,
                )
                graph_tau_target = graph_tau_bundle["tau"]
        response_operator = _response_operator_from_triangle(triangle, triangle_mask)
        response_operator_trace = response_operator["trace"].mean()
        response_operator_effective_rank = response_operator["effective_rank"].mean()
        response_operator_anisotropy = response_operator["anisotropy"].mean()
        response_operator_asymmetry = response_operator["asymmetry"].mean()

        response_support_target = torch.log(response_operator["trace"].clamp_min(1e-6)).unsqueeze(-1)
        response_support_target = response_support_target + 0.25 * response_operator["anisotropy"].unsqueeze(-1)
        response_support_target = response_support_target + 0.25 * response_operator["asymmetry"].abs().unsqueeze(-1)
        response_support_target = (
            response_support_target - response_support_target.mean(dim=0, keepdim=True)
        ) / response_support_target.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        if response_jet is not None and response_jet["support_tilt"] is not None:
            response_support_target = response_jet["support_tilt"]

        if tilt_target_mode == "response_support":
            tilt_target = response_support_target
        elif tilt_target_mode == "graph_tau":
            tilt_target = graph_tau_target if graph_tau_target is not None else response_support_target
        elif tilt_target_mode == "hybrid":
            if teacher_tilt_target is None:
                tilt_target = response_support_target
            else:
                tilt_blend = float(min(max(tilt_target_blend, 0.0), 1.0))
                tilt_target = (1.0 - tilt_blend) * teacher_tilt_target + tilt_blend * response_support_target

        if response_jet is not None:
            target_tangent_frame = response_jet["frame"]
            target_tangent_projector = response_jet["projector"]
            target_neighbor_idx = response_jet["neighbor_idx"]
            target_neighbor_weights = response_jet["neighbor_weights"]
            target_transport = response_jet["transport"]
            response_identifiable_effective_rank = response_jet["effective_rank"]
            response_identifiable_anisotropy = response_jet["anisotropy"]

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
            asym_value = response_operator["asymmetry"].unsqueeze(-1)
            asym_tangent_drift[:, :1] = torch.sign(asym_value) * drift_scale * asym_value.abs().clamp_max(1.0)
            if drift_target_mode == "response_jet" and response_jet is not None and response_jet["tangent_drift"] is not None:
                target_tangent_drift = response_jet["tangent_drift"]
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
            if drift_target_mode != "response_jet" or response_jet is None:
                drift_target = torch.einsum("bij,bj->bi", student_tangent_frame, target_tangent_drift)
                response_drift_alignment = F.mse_loss(bootstrap_tangent_drift[:, :1], asym_tangent_drift[:, :1])

        if measure_target_mode != "chart_moments" and student_tangent_frame is not None:
            if measure_target_mode == "response_jet" and response_jet is not None and response_jet["tangent_cov"] is not None:
                target_tangent_cov = response_jet["tangent_cov"]
                identifiable_tangent_cov = response_jet["identifiable_tangent_cov"]
                if target_tangent_frame is not None:
                    full_diffusion_target = (
                        target_tangent_frame @ target_tangent_cov @ target_tangent_frame.transpose(-1, -2)
                    )
            else:
                tangent_dim = student_tangent_frame.size(-1)
                target_eigs = response_operator["eigvals"][:, : min(tangent_dim, response_operator["eigvals"].size(-1))]
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
                bootstrap_trace = bootstrap_full_diffusion_target.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                invariant_diag = bootstrap_trace.unsqueeze(-1) * spectral_weights
                invariant_tangent_cov = torch.diag_embed(invariant_diag.clamp_min(1e-8))
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

    return {
        "signatures": signatures,
        "drift_target": drift_target,
        "diffusion_target": diffusion_target,
        "full_diffusion_target": full_diffusion_target,
        "bootstrap_drift_target": bootstrap_drift_target,
        "bootstrap_diffusion_target": bootstrap_full_diffusion_target,
        "target_tangent_drift": tangent_drift_target_out,
        "target_tangent_cov": target_tangent_cov,
        "identifiable_tangent_cov": identifiable_tangent_cov,
        "target_tangent_frame": target_tangent_frame,
        "target_tangent_projector": target_tangent_projector,
        "target_neighbor_idx": target_neighbor_idx,
        "target_neighbor_weights": target_neighbor_weights,
        "target_transport": target_transport,
        "response_identifiable_effective_rank": response_identifiable_effective_rank,
        "response_identifiable_anisotropy": response_identifiable_anisotropy,
        "tilt_target": tilt_target,
        "response_operator_trace": response_operator_trace,
        "response_operator_effective_rank": response_operator_effective_rank,
        "response_operator_anisotropy": response_operator_anisotropy,
        "response_operator_asymmetry": response_operator_asymmetry,
        "response_drift_alignment": response_drift_alignment,
    }


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
        return latents.new_zeros(batch, response_signature_dim(steps, mode))
    triangle, mask = response_triangle(
        model=model,
        latents=latents,
        video=video,
        cond_embed=cond_embed,
        decoded=decoded,
    )
    return _response_signature_from_triangle(triangle, mask, mode)


def local_neighbor_smoothness_loss(
    signatures: torch.Tensor,
    fields: list[torch.Tensor],
    *,
    knn: int,
    temperature: float,
) -> torch.Tensor:
    batch = signatures.size(0)
    if batch < 2 or knn <= 0:
        return signatures.new_tensor(0.0)

    num_neighbors = min(knn, batch - 1)
    distance = torch.cdist(signatures.detach(), signatures.detach(), p=2.0)
    inf = torch.full_like(distance.diagonal(), float("inf"))
    distance = distance.clone()
    distance.diagonal().copy_(inf)
    knn_dist, knn_idx = torch.topk(distance, k=num_neighbors, largest=False, dim=1)
    weights = torch.softmax(-knn_dist / max(temperature, 1e-4), dim=1)

    total = signatures.new_tensor(0.0)
    for field in fields:
        flat_field = field if field.ndim == 2 else field.unsqueeze(1)
        neighbor_field = flat_field.index_select(0, knn_idx.reshape(-1)).view(batch, num_neighbors, -1)
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
            "tangent_anisotropy_gap": zero,
            "pred_tangent_effective_rank": zero,
            "target_tangent_effective_rank": zero,
            "pred_tangent_anisotropy": zero,
            "target_tangent_anisotropy": zero,
            "pred_tangent_trace": zero,
            "target_tangent_trace": zero,
            "response_operator_trace": zero,
            "response_operator_effective_rank": zero,
            "response_operator_anisotropy": zero,
            "response_operator_asymmetry": zero,
            "response_identifiable_effective_rank": zero,
            "response_identifiable_anisotropy": zero,
            "response_drift_alignment": zero,
        }

    state = model.trajectory_state(latents)
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
    )
    signatures = target_bundle["signatures"]
    drift = model.trajectory_drift(latents, cond_embed)
    tangent_structure = model.local_tangent_structure(
        latents,
        cond_embed,
        response_context=signatures,
        state=state,
    )
    diffusion_matrix = model.local_diffusion_matrix(
        latents,
        cond_embed,
        response_context=signatures,
        state=state,
        tangent_structure=tangent_structure,
    )
    tangent_core_cov = model.local_tangent_covariance(
        latents,
        cond_embed,
        response_context=signatures,
        state=state,
        tangent_structure=tangent_structure,
    )
    diffusion_diag = diffusion_matrix.diagonal(dim1=-2, dim2=-1)
    _, tilt_log_density, total_log_density = model.measure_log_density_components(
        latents,
        cond_embed,
        state=state,
    )
    log_density = total_log_density.squeeze(-1)
    chart_latents = model.chart_latents(latents)
    chart_delta = chart_latents[:, 1:] - chart_latents[:, :-1]

    drift_target = target_bundle["drift_target"]
    diffusion_target = target_bundle["diffusion_target"]
    full_diffusion_target = target_bundle["full_diffusion_target"]
    target_tangent_cov = target_bundle["target_tangent_cov"]
    identifiable_tangent_cov = target_bundle["identifiable_tangent_cov"]
    target_tangent_frame = target_bundle["target_tangent_frame"]
    target_tangent_projector = target_bundle["target_tangent_projector"]
    target_neighbor_idx = target_bundle["target_neighbor_idx"]
    target_neighbor_weights = target_bundle["target_neighbor_weights"]
    target_transport = target_bundle["target_transport"]
    tilt_target = target_bundle["tilt_target"]

    drift_loss = F.mse_loss(drift, drift_target)
    diffusion_loss = F.mse_loss(diffusion_matrix, diffusion_target)
    tilt_alignment = latents.new_tensor(0.0) if tilt_target is None else F.mse_loss(tilt_log_density, tilt_target)

    density_weights = posterior_from_logits(log_density.unsqueeze(0), temperature=density_temperature).squeeze(0)
    directions = _measure_test_directions(
        state.size(1),
        test_num_directions,
        device=state.device,
        dtype=state.dtype,
    )
    projected_state = state @ directions.T
    projected_drift = drift @ directions.T
    projected_diffusion = torch.einsum("bde,kd,ke->bk", diffusion_matrix, directions, directions)

    linear_moment = (density_weights.unsqueeze(1) * projected_drift).sum(dim=0)
    quadratic_moment = (
        density_weights.unsqueeze(1)
        * (2.0 * projected_state * projected_drift + projected_diffusion)
    ).sum(dim=0)
    trig_scale = max(trig_scale, 1e-4)
    trig_moment = (
        density_weights.unsqueeze(1)
        * (
            trig_scale * torch.cos(trig_scale * projected_state) * projected_drift
            - 0.5 * (trig_scale**2) * torch.sin(trig_scale * projected_state) * projected_diffusion
        )
    ).sum(dim=0)
    radial_moment = (
        density_weights * (2.0 * (state * drift).sum(dim=1) + diffusion_diag.sum(dim=1))
    ).sum()
    pred_trace = diffusion_diag.sum(dim=1)
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
        tangent_anisotropy_gap = latents.new_tensor(0.0)
        pred_tangent_effective_rank = latents.new_tensor(0.0)
        target_tangent_effective_rank = latents.new_tensor(0.0)
        pred_tangent_anisotropy = latents.new_tensor(0.0)
        target_tangent_anisotropy = latents.new_tensor(0.0)
        pred_tangent_trace = latents.new_tensor(0.0)
        target_tangent_trace = latents.new_tensor(0.0)
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
            neighbor_pred_frame = pred_frame.index_select(0, target_neighbor_idx.reshape(-1)).view(
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
            tangent_anisotropy_gap = latents.new_tensor(0.0)
            pred_tangent_effective_rank = latents.new_tensor(0.0)
            target_tangent_effective_rank = latents.new_tensor(0.0)
            pred_tangent_anisotropy = latents.new_tensor(0.0)
            target_tangent_anisotropy = latents.new_tensor(0.0)
            pred_tangent_trace = latents.new_tensor(0.0)
            target_tangent_trace = latents.new_tensor(0.0)
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
            if identifiable_tangent_cov is None:
                tangent_shape_alignment = latents.new_tensor(0.0)
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
                target_tangent_effective_rank = target_bundle["response_identifiable_effective_rank"]
                tangent_anisotropy_gap = (
                    pred_tangent_anisotropy - target_bundle["response_identifiable_anisotropy"]
                ).abs()
                if not torch.is_tensor(target_tangent_effective_rank):
                    target_tangent_effective_rank = latents.new_tensor(float(target_tangent_effective_rank))
    if tangent_projector is None:
        tangent_projection = latents.new_tensor(0.0)

    density_entropy = -(density_weights * density_weights.clamp_min(1e-12).log()).sum()

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
        "tangent_anisotropy_gap": tangent_anisotropy_gap,
        "pred_tangent_effective_rank": pred_tangent_effective_rank,
        "target_tangent_effective_rank": target_tangent_effective_rank,
        "pred_tangent_anisotropy": pred_tangent_anisotropy,
        "target_tangent_anisotropy": target_tangent_anisotropy,
        "pred_tangent_trace": pred_tangent_trace,
        "target_tangent_trace": target_tangent_trace,
        "response_operator_trace": target_bundle["response_operator_trace"],
        "response_operator_effective_rank": target_bundle["response_operator_effective_rank"],
        "response_operator_anisotropy": target_bundle["response_operator_anisotropy"],
        "response_operator_asymmetry": target_bundle["response_operator_asymmetry"],
        "response_identifiable_effective_rank": target_bundle["response_identifiable_effective_rank"],
        "response_identifiable_anisotropy": target_bundle["response_identifiable_anisotropy"],
        "response_drift_alignment": target_bundle["response_drift_alignment"],
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
