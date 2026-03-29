from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    kind: str = "synthetic"
    synthetic_mode: str = "base"
    train_size: int = 4096
    val_size: int = 512
    seq_len: int = 8
    image_size: int = 32
    channels: int = 3
    num_workers: int = 4
    root: str = ""
    manifest_path: str = ""
    val_manifest_path: str = ""


@dataclass
class ModelConfig:
    latent_dim: int = 64
    cond_dim: int = 64
    hidden_dim: int = 128
    base_channels: int = 32
    condition_score_mode: str = "distance"
    energy_hidden_dim: int = 128
    identity_num_classes: int = 0
    identity_hidden_dim: int = 128
    semantic_num_classes: int = 0
    semantic_temperature: float = 0.2
    chart_hidden_dim: int = 128
    chart_num_experts: int = 1
    chart_mode: str = "pointwise_residual"
    chart_residual_scale: float = 0.1
    chart_temporal_hidden_dim: int = 128
    chart_temporal_kernel_size: int = 3
    state_cov_proj_dim: int = 0
    response_context_dim: int = 0
    response_signature_mode: str = "span_stats"
    tangent_dim: int = 0
    local_measure_hidden_dim: int = 128
    local_measure_rank: int = 8
    local_measure_eps: float = 1e-4
    local_diffusion_mode: str = "legacy"
    local_diffusion_geometry_mode: str = "ambient"
    local_diffusion_condition_mode: str = "joint"
    measure_density_mode: str = "joint"


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-5
    amp: bool = True
    log_every: int = 20
    sample_every: int = 1
    gradient_clip_norm: float = 1.0
    device: str = "auto"
    stage1_epochs: int = 2
    stage2_epochs: int = 4
    stage3_epochs: int = 8
    identity_start_epoch: int = 0
    identity_warmup_epochs: int = 0
    query_eval_enabled: bool = False
    query_eval_alpha: float = 0.90
    query_eval_obs_alpha: float = 0.90
    query_eval_plan_core_alpha: float = 0.50
    query_eval_posterior_temperature: float = 1.0
    query_eval_max_samples: int = 80
    checkpoint_selection_mode: str = "support"
    query_checkpoint_fallback_budget: float = 0.05
    query_checkpoint_exec_weight: float = 1.0
    query_checkpoint_match_weight: float = 0.20
    query_checkpoint_support_weight: float = 0.05
    query_checkpoint_gap_weight: float = 0.25
    measure_target_ema_decay: float = 0.0
    measure_target_use_teacher_eval: bool = True


@dataclass
class LossConfig:
    base_weight: float = 1.0
    rep_weight: float = 0.05
    loc_weight: float = 0.0
    loc_eps: float = 1e-2
    dyn_weight: float = 0.5
    cond_weight: float = 0.25
    reg_weight: float = 0.01
    gap_weight: float = 0.25
    condition_dyn_weight: float = 0.5
    delta_reg_weight: float = 0.5
    delta_temporal_weight: float = 0.5
    identity_weight: float = 0.0
    identity_ortho_weight: float = 0.0
    semantic_proto_weight: float = 0.0
    semantic_center_weight: float = 0.0
    semantic_proto_sep_weight: float = 0.0
    hard_negative_prob: float = 0.75
    max_negative_edits: int = 2
    support_p_true_weight: float = 0.0
    support_margin_weight: float = 0.0
    support_ratio_weight: float = 0.0
    support_p_true_floor: float = 0.0
    support_margin_floor: float = 0.0
    support_ratio_ceiling: float = 1.0
    support_gate_p_true: float = 0.0
    support_gate_margin: float = 0.0
    support_gate_temperature: float = 0.05
    local_drift_weight: float = 0.0
    local_diffusion_weight: float = 0.0
    measure_stationarity_weight: float = 0.0
    response_smoothness_weight: float = 0.0
    response_signature_knn: int = 4
    response_signature_temperature: float = 0.5
    response_geometry_knn: int = 8
    response_geometry_temperature: float = 0.5
    response_jet_ridge: float = 1e-3
    response_jet_center_weight: float = 1.0
    response_tau_ridge: float = 1e-3
    response_tau_mean_penalty: float = 1.0
    response_tau_drift_scale: float = 0.25
    measure_density_temperature: float = 1.0
    measure_test_num_directions: int = 8
    measure_trig_scale: float = 1.0
    measure_trace_weight: float = 0.0
    tangent_projection_weight: float = 0.0
    tangent_compatibility_weight: float = 0.0
    tangent_spectrum_weight: float = 0.0
    tangent_shape_weight: float = 0.0
    diffusion_target_mode: str = "full"
    measure_target_mode: str = "chart_moments"
    measure_target_blend: float = 0.5
    drift_target_mode: str = "bootstrap"
    drift_target_blend: float = 0.5
    measure_tilt_target_weight: float = 0.0
    tilt_target_mode: str = "none"
    tilt_target_blend: float = 0.5


@dataclass
class AppConfig:
    seed: int = 7
    output_dir: str = "runs/mvp"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> AppConfig:
    config = AppConfig()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _update_dataclass(config, raw)
