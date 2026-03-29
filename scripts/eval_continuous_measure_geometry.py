from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vh_mvp.config import load_config
from vh_mvp.data import FolderVideoDataset, SyntheticVideoDataset
from vh_mvp.losses import local_measure_loss, local_measure_targets, response_signature_dim
from vh_mvp.models import VideoDynamicsMVP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate continuous local-measure geometry metrics.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_model_from_config(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[VideoDynamicsMVP, VideoDynamicsMVP | None, object]:
    cfg = load_config(config_path)
    model = VideoDynamicsMVP(
        channels=cfg.data.channels,
        base_channels=cfg.model.base_channels,
        latent_dim=cfg.model.latent_dim,
        cond_dim=cfg.model.cond_dim,
        hidden_dim=cfg.model.hidden_dim,
        condition_score_mode=cfg.model.condition_score_mode,
        energy_hidden_dim=cfg.model.energy_hidden_dim,
        identity_num_classes=cfg.model.identity_num_classes,
        identity_hidden_dim=cfg.model.identity_hidden_dim,
        semantic_num_classes=cfg.model.semantic_num_classes,
        semantic_temperature=cfg.model.semantic_temperature,
        chart_hidden_dim=cfg.model.chart_hidden_dim,
        chart_num_experts=cfg.model.chart_num_experts,
        chart_mode=cfg.model.chart_mode,
        chart_residual_scale=cfg.model.chart_residual_scale,
        chart_temporal_hidden_dim=cfg.model.chart_temporal_hidden_dim,
        chart_temporal_kernel_size=cfg.model.chart_temporal_kernel_size,
        state_cov_proj_dim=cfg.model.state_cov_proj_dim,
        response_signature_dim=response_signature_dim(cfg.data.seq_len, cfg.model.response_signature_mode),
        response_context_dim=cfg.model.response_context_dim,
        tangent_dim=cfg.model.tangent_dim,
        local_measure_hidden_dim=cfg.model.local_measure_hidden_dim,
        local_measure_rank=cfg.model.local_measure_rank,
        local_measure_eps=cfg.model.local_measure_eps,
        local_diffusion_mode=cfg.model.local_diffusion_mode,
        local_diffusion_geometry_mode=cfg.model.local_diffusion_geometry_mode,
        local_diffusion_condition_mode=cfg.model.local_diffusion_condition_mode,
        measure_density_mode=cfg.model.measure_density_mode,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    target_model = None
    if checkpoint.get("target_model") is not None:
        target_model = VideoDynamicsMVP(
            channels=cfg.data.channels,
            base_channels=cfg.model.base_channels,
            latent_dim=cfg.model.latent_dim,
            cond_dim=cfg.model.cond_dim,
            hidden_dim=cfg.model.hidden_dim,
            condition_score_mode=cfg.model.condition_score_mode,
            energy_hidden_dim=cfg.model.energy_hidden_dim,
            identity_num_classes=cfg.model.identity_num_classes,
            identity_hidden_dim=cfg.model.identity_hidden_dim,
            semantic_num_classes=cfg.model.semantic_num_classes,
            semantic_temperature=cfg.model.semantic_temperature,
            chart_hidden_dim=cfg.model.chart_hidden_dim,
            chart_num_experts=cfg.model.chart_num_experts,
            chart_mode=cfg.model.chart_mode,
            chart_residual_scale=cfg.model.chart_residual_scale,
            chart_temporal_hidden_dim=cfg.model.chart_temporal_hidden_dim,
            chart_temporal_kernel_size=cfg.model.chart_temporal_kernel_size,
            state_cov_proj_dim=cfg.model.state_cov_proj_dim,
            response_signature_dim=response_signature_dim(cfg.data.seq_len, cfg.model.response_signature_mode),
            response_context_dim=cfg.model.response_context_dim,
            tangent_dim=cfg.model.tangent_dim,
            local_measure_hidden_dim=cfg.model.local_measure_hidden_dim,
            local_measure_rank=cfg.model.local_measure_rank,
            local_measure_eps=cfg.model.local_measure_eps,
            local_diffusion_mode=cfg.model.local_diffusion_mode,
            local_diffusion_geometry_mode=cfg.model.local_diffusion_geometry_mode,
            local_diffusion_condition_mode=cfg.model.local_diffusion_condition_mode,
            measure_density_mode=cfg.model.measure_density_mode,
        ).to(device)
        target_model.load_state_dict(checkpoint["target_model"], strict=False)
        target_model.eval()
    return model, target_model, cfg


def build_dataset(cfg, split: str):
    if cfg.data.kind == "synthetic":
        if split == "train":
            return SyntheticVideoDataset(
                size=cfg.data.train_size,
                seq_len=cfg.data.seq_len,
                image_size=cfg.data.image_size,
                seed=cfg.seed,
                synthetic_mode=cfg.data.synthetic_mode,
            )
        return SyntheticVideoDataset(
            size=cfg.data.val_size,
            seq_len=cfg.data.seq_len,
            image_size=cfg.data.image_size,
            seed=cfg.seed + 100_000,
            synthetic_mode=cfg.data.synthetic_mode,
        )
    if cfg.data.kind == "folder":
        manifest_path = cfg.data.manifest_path if split == "train" else (cfg.data.val_manifest_path or cfg.data.manifest_path)
        return FolderVideoDataset(
            root=cfg.data.root,
            manifest_path=manifest_path,
            seq_len=cfg.data.seq_len,
            image_size=cfg.data.image_size,
        )
    raise ValueError(f"Unsupported data.kind: {cfg.data.kind}")


def save_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(
    *,
    model: VideoDynamicsMVP,
    target_model: VideoDynamicsMVP | None,
    cfg,
    dataset,
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    total_weight = 0
    aggregate = {
        "measure_stationarity": 0.0,
        "measure_linear_stationarity": 0.0,
        "measure_quadratic_stationarity": 0.0,
        "measure_trig_stationarity": 0.0,
        "response_smoothness": 0.0,
        "local_drift": 0.0,
        "local_diffusion": 0.0,
        "diffusion_diag_mse": 0.0,
        "diffusion_offdiag_mse": 0.0,
        "pred_trace_mean": 0.0,
        "target_trace_mean": 0.0,
        "raw_target_trace_mean": 0.0,
        "pred_offdiag_energy": 0.0,
        "target_offdiag_energy": 0.0,
        "raw_target_offdiag_energy": 0.0,
        "pred_min_eig_mean": 0.0,
        "measure_density_entropy": 0.0,
        "measure_tilt_abs_mean": 0.0,
        "measure_tilt_alignment": 0.0,
        "response_operator_trace": 0.0,
        "response_operator_effective_rank": 0.0,
        "response_operator_anisotropy": 0.0,
        "response_operator_asymmetry": 0.0,
        "response_drift_alignment": 0.0,
        "chart_shift_l2": 0.0,
        "chart_shift_abs_mean": 0.0,
        "chart_gate_mean": 0.0,
        "chart_gate_abs_mean": 0.0,
        "chart_expert_entropy": 0.0,
        "chart_expert_max_weight": 0.0,
        "tangent_projection": 0.0,
        "tangent_observation_residual": 0.0,
        "tangent_drift_residual": 0.0,
        "tangent_diffusion_residual": 0.0,
        "tangent_bundle_compatibility": 0.0,
        "tangent_frame_orthogonality": 0.0,
        "tangent_projector_trace": 0.0,
        "tangent_spectrum_alignment": 0.0,
        "tangent_shape_alignment": 0.0,
        "tangent_anisotropy_gap": 0.0,
        "pred_tangent_effective_rank": 0.0,
        "target_tangent_effective_rank": 0.0,
        "pred_tangent_anisotropy": 0.0,
        "target_tangent_anisotropy": 0.0,
        "pred_tangent_trace": 0.0,
        "target_tangent_trace": 0.0,
        "response_identifiable_effective_rank": 0.0,
        "response_identifiable_anisotropy": 0.0,
    }
    rows: list[dict[str, float]] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        video = batch["video"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)

        out = model(video, condition)
        target_cond_embed = None
        if target_model is not None:
            target_cond_embed = target_model.condition_encoder(condition)
        target_terms = local_measure_targets(
            model=model,
            latents=out.latents,
            video=video,
            cond_embed=out.cond_embed,
            diffusion_target_mode=cfg.loss.diffusion_target_mode,
            signature_mode=cfg.model.response_signature_mode,
            measure_target_mode=cfg.loss.measure_target_mode,
            measure_target_blend=cfg.loss.measure_target_blend,
            decoded=out.recon,
            drift_target_mode=cfg.loss.drift_target_mode,
            drift_target_blend=cfg.loss.drift_target_blend,
            tilt_target_mode=cfg.loss.tilt_target_mode,
            tilt_target_blend=cfg.loss.tilt_target_blend,
            geometry_knn=cfg.loss.response_geometry_knn,
            geometry_temperature=cfg.loss.response_geometry_temperature,
            jet_ridge=cfg.loss.response_jet_ridge,
            jet_center_weight=cfg.loss.response_jet_center_weight,
            tau_ridge=cfg.loss.response_tau_ridge,
            tau_mean_penalty=cfg.loss.response_tau_mean_penalty,
            tau_drift_scale=cfg.loss.response_tau_drift_scale,
            target_model=target_model,
            target_cond_embed=target_cond_embed,
        )
        signatures = target_terms["signatures"]
        chart_diag = model.chart_diagnostics(out.latents)
        state_diag = model.trajectory_state_diagnostics(out.latents)
        tangent_diag = model.trajectory_tangent_diagnostics(out.latents)
        measure_terms = local_measure_loss(
            model,
            out.latents,
            video,
            out.cond_embed,
            signature_knn=cfg.loss.response_signature_knn,
            signature_temperature=cfg.loss.response_signature_temperature,
            geometry_knn=cfg.loss.response_geometry_knn,
            geometry_temperature=cfg.loss.response_geometry_temperature,
            jet_ridge=cfg.loss.response_jet_ridge,
            jet_center_weight=cfg.loss.response_jet_center_weight,
            tau_ridge=cfg.loss.response_tau_ridge,
            tau_mean_penalty=cfg.loss.response_tau_mean_penalty,
            tau_drift_scale=cfg.loss.response_tau_drift_scale,
            density_temperature=cfg.loss.measure_density_temperature,
            test_num_directions=cfg.loss.measure_test_num_directions,
            trig_scale=cfg.loss.measure_trig_scale,
            diffusion_target_mode=cfg.loss.diffusion_target_mode,
            measure_target_mode=cfg.loss.measure_target_mode,
            measure_target_blend=cfg.loss.measure_target_blend,
            drift_target_mode=cfg.loss.drift_target_mode,
            drift_target_blend=cfg.loss.drift_target_blend,
            tilt_target_mode=cfg.loss.tilt_target_mode,
            tilt_target_blend=cfg.loss.tilt_target_blend,
            signature_mode=cfg.model.response_signature_mode,
            signatures=signatures,
            decoded=out.recon,
            target_model=target_model,
            target_cond_embed=target_cond_embed,
        )

        target_cov = target_terms["diffusion_target"]
        raw_target_cov = target_terms["bootstrap_diffusion_target"]
        pred_cov = model.local_diffusion_matrix(out.latents, out.cond_embed, response_context=signatures)

        diag_target = target_cov.diagonal(dim1=-2, dim2=-1)
        diag_pred = pred_cov.diagonal(dim1=-2, dim2=-1)
        diag_mse = (diag_pred - diag_target).square().mean()

        offdiag_mask = (~torch.eye(pred_cov.size(-1), dtype=torch.bool, device=pred_cov.device)).unsqueeze(0)
        offdiag_diff = (pred_cov - target_cov).masked_select(offdiag_mask)
        offdiag_target = target_cov.masked_select(offdiag_mask)
        offdiag_pred = pred_cov.masked_select(offdiag_mask)
        offdiag_mse = offdiag_diff.square().mean()

        pred_trace = diag_pred.sum(dim=-1)
        target_trace = diag_target.sum(dim=-1)
        raw_target_trace = raw_target_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        raw_target_offdiag = raw_target_cov.masked_select(offdiag_mask)
        pred_min_eig = torch.linalg.eigvalsh(pred_cov).amin(dim=-1)

        batch_size_actual = video.size(0)
        total_weight += batch_size_actual
        batch_metrics = {
            "batch_idx": float(batch_idx),
            "measure_stationarity": float(measure_terms["measure_stationarity"].item()),
            "measure_linear_stationarity": float(measure_terms["measure_linear_stationarity"].item()),
            "measure_quadratic_stationarity": float(measure_terms["measure_quadratic_stationarity"].item()),
            "measure_trig_stationarity": float(measure_terms["measure_trig_stationarity"].item()),
            "response_smoothness": float(measure_terms["response_smoothness"].item()),
            "local_drift": float(measure_terms["local_drift"].item()),
            "local_diffusion": float(measure_terms["local_diffusion"].item()),
            "diffusion_diag_mse": float(diag_mse.item()),
            "diffusion_offdiag_mse": float(offdiag_mse.item()),
            "pred_trace_mean": float(pred_trace.mean().item()),
            "target_trace_mean": float(target_trace.mean().item()),
            "raw_target_trace_mean": float(raw_target_trace.mean().item()),
            "pred_offdiag_energy": float(offdiag_pred.square().mean().item()),
            "target_offdiag_energy": float(offdiag_target.square().mean().item()),
            "raw_target_offdiag_energy": float(raw_target_offdiag.square().mean().item()),
            "pred_min_eig_mean": float(pred_min_eig.mean().item()),
            "measure_density_entropy": float(measure_terms["measure_density_entropy"].item()),
            "measure_tilt_abs_mean": float(measure_terms["measure_tilt_abs_mean"].item()),
            "measure_tilt_alignment": float(measure_terms["measure_tilt_alignment"].item()),
            "response_operator_trace": float(measure_terms["response_operator_trace"].item()),
            "response_operator_effective_rank": float(measure_terms["response_operator_effective_rank"].item()),
            "response_operator_anisotropy": float(measure_terms["response_operator_anisotropy"].item()),
            "response_operator_asymmetry": float(measure_terms["response_operator_asymmetry"].item()),
            "response_drift_alignment": float(measure_terms["response_drift_alignment"].item()),
            "chart_shift_l2": float(chart_diag["chart_shift_l2"].item()),
            "chart_shift_abs_mean": float(chart_diag["chart_shift_abs_mean"].item()),
            "chart_gate_mean": float(chart_diag["chart_gate_mean"].item()),
            "chart_gate_abs_mean": float(chart_diag["chart_gate_abs_mean"].item()),
            "chart_expert_entropy": float(state_diag["chart_expert_entropy"].item()),
            "chart_expert_max_weight": float(state_diag["chart_expert_max_weight"].item()),
            "tangent_projection": float(measure_terms["tangent_projection"].item()),
            "tangent_observation_residual": float(measure_terms["tangent_observation_residual"].item()),
            "tangent_drift_residual": float(measure_terms["tangent_drift_residual"].item()),
            "tangent_diffusion_residual": float(measure_terms["tangent_diffusion_residual"].item()),
            "tangent_bundle_compatibility": float(measure_terms["tangent_bundle_compatibility"].item()),
            "tangent_frame_orthogonality": float(tangent_diag["tangent_frame_orthogonality"].item()),
            "tangent_projector_trace": float(tangent_diag["tangent_projector_trace"].item()),
            "tangent_spectrum_alignment": float(measure_terms["tangent_spectrum_alignment"].item()),
            "tangent_shape_alignment": float(measure_terms["tangent_shape_alignment"].item()),
            "tangent_anisotropy_gap": float(measure_terms["tangent_anisotropy_gap"].item()),
            "pred_tangent_effective_rank": float(measure_terms["pred_tangent_effective_rank"].item()),
            "target_tangent_effective_rank": float(measure_terms["target_tangent_effective_rank"].item()),
            "pred_tangent_anisotropy": float(measure_terms["pred_tangent_anisotropy"].item()),
            "target_tangent_anisotropy": float(measure_terms["target_tangent_anisotropy"].item()),
            "pred_tangent_trace": float(measure_terms["pred_tangent_trace"].item()),
            "target_tangent_trace": float(measure_terms["target_tangent_trace"].item()),
            "response_identifiable_effective_rank": float(measure_terms["response_identifiable_effective_rank"].item()),
            "response_identifiable_anisotropy": float(measure_terms["response_identifiable_anisotropy"].item()),
        }
        rows.append(batch_metrics)

        for key in aggregate:
            aggregate[key] += batch_metrics[key] * batch_size_actual

    denom = max(total_weight, 1)
    summary = {key: value / denom for key, value in aggregate.items()}
    summary["num_samples"] = float(total_weight)
    summary["num_batches"] = float(len(rows))
    return summary, rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, target_model, cfg = build_model_from_config(args.config, args.checkpoint, device)
    dataset = build_dataset(cfg, args.split)
    summary, rows = evaluate(
        model=model,
        target_model=target_model,
        cfg=cfg,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        max_batches=args.max_batches,
    )

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_csv(metrics_dir / f"{args.split}_continuous_measure_batches.csv", rows)
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
