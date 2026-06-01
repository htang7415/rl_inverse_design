"""Reusable Step 5 run execution helpers."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml

from .config import (
    build_run_config,
    filter_step5_target_rows_by_condition,
    resolve_step5_generation_budget,
    resolve_step5_sampling_num_steps,
)
from .conditional_sampling import create_conditional_sampler, sample_conditional_with_class_prior
from .dataset import ConditionScaler, build_inference_condition_bundle_from_target_row
from .dpo import train_s4_dpo_alignment
from .evaluation import (
    aggregate_round_metrics,
    aggregate_target_row_metrics,
    build_generated_samples_frame,
    build_method_metrics,
    evaluate_generated_samples,
    load_step5_evaluator,
    summarize_target_rows,
)
from .frozen_sampling import (
    create_raw_step1_sampler,
    load_step1_diffusion,
    resolve_class_sampling_prior,
    sample_raw_step1_unconditional,
    sample_unconditional_with_class_prior,
)
from .guided_sampler import GuidedConditionalSampler, GuidedSampler
from .plotting import plot_generated_chi_vs_target, plot_per_target_success, plot_success_gate_funnel
from .rl_trainer import train_s4_rl_alignment
from .supervised import build_s2_components_from_step1, load_step5_checkpoint_into_modules
from .train_s2 import S2TrainingArtifacts, train_s2_supervised_run
from src.utils.config import as_yamlable
from src.utils.reproducibility import save_run_metadata, seed_everything
from src.utils.reporting import append_log_message, save_artifact_manifest, write_initial_log

_HPO_SHARED_S4_WARM_START_CACHE: Dict[str, Dict[str, Any]] = {}


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _resolve_cross_target_duplicate_rejection_enabled(resolved) -> bool:
    chi_cfg = resolved.base_config.get("chi_training", {})
    decode_cfg = (
        chi_cfg.get("step5_inverse_design", {})
        if isinstance(chi_cfg.get("step5_inverse_design", {}), dict)
        else {}
    )
    if not decode_cfg:
        legacy_cfg = chi_cfg.get("step5_class_inverse_design", {})
        if isinstance(legacy_cfg, dict):
            decode_cfg = legacy_cfg
    default_value = bool(decode_cfg.get("decode_constraint_reject_duplicate_canonical_across_targets", False))
    overrides = decode_cfg.get("decode_constraint_reject_duplicate_canonical_across_targets_overrides", {})
    if not isinstance(overrides, dict):
        return default_value
    target_class = str(resolved.c_target).strip().lower()
    for raw_key, raw_value in overrides.items():
        if str(raw_key).strip().lower() != target_class:
            continue
        return bool(raw_value)
    return default_value


def _is_missing_config_value(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none"})


def _resolve_s0_target_condition(resolved, run_cfg: Dict[str, object]) -> Tuple[Optional[float], Optional[float]]:
    if str(run_cfg.get("run_name", "")) != "S0_raw_unconditional":
        return None, None
    if not bool(resolved.step5.get("s0_use_configured_target_condition", False)):
        return None, None

    raw_temperature = resolved.step5.get("target_temperature")
    raw_phi = resolved.step5.get("target_phi")
    if _is_missing_config_value(raw_temperature) and _is_missing_config_value(raw_phi):
        chi_cfg = resolved.base_config.get("chi_training", {})
        design_cfg = (
            chi_cfg.get("step5_inverse_design", {})
            if isinstance(chi_cfg.get("step5_inverse_design", {}), dict)
            else {}
        )
        if not design_cfg:
            legacy_cfg = chi_cfg.get("step5_class_inverse_design", {})
            if isinstance(legacy_cfg, dict):
                design_cfg = legacy_cfg
        raw_temperature = design_cfg.get("target_temperature")
        raw_phi = design_cfg.get("target_phi")

    if _is_missing_config_value(raw_temperature) and _is_missing_config_value(raw_phi):
        return None, None
    if _is_missing_config_value(raw_temperature) or _is_missing_config_value(raw_phi):
        raise ValueError(
            "S0 configured target-condition filtering requires both target_temperature and target_phi."
        )
    return float(raw_temperature), float(raw_phi)


def _resolve_hpo_sampling_objective_column(metric_name: str, evaluation_df: pd.DataFrame) -> Optional[str]:
    """Map an HPO objective metric to a per-sample hit column for intermediate pruning."""

    candidates_by_metric = {
        "mean_success_hit_rate_discovery": ["success_hit_discovery", "success_hit"],
        "mean_success_hit_rate": ["success_hit"],
        "mean_property_success_hit_rate_discovery": [
            "property_success_hit_discovery",
            "property_success_hit",
        ],
        "mean_property_success_hit_rate": ["property_success_hit"],
    }
    for column in candidates_by_metric.get(str(metric_name).strip(), ["success_hit_discovery", "success_hit"]):
        if column in evaluation_df.columns:
            return str(column)
    return None


def create_run_dirs(run_dir: Path, *, create_checkpoints_dir: bool = True) -> Dict[str, Path]:
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    checkpoints_dir = run_dir / "checkpoints"
    paths = [run_dir, metrics_dir, figures_dir]
    if create_checkpoints_dir:
        paths.append(checkpoints_dir)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "metrics_dir": metrics_dir,
        "figures_dir": figures_dir,
        "checkpoints_dir": checkpoints_dir,
    }


def save_run_config_snapshot(
    run_dirs: Dict[str, Path],
    *,
    run_cfg: Dict[str, object],
    resolved,
    config_path: str,
    extra_context: Optional[Dict[str, Any]] = None,
) -> None:
    snapshot = {
        "model_size": resolved.model_size,
        "split_mode": resolved.split_mode,
        "classification_split_mode": resolved.classification_split_mode,
        "c_target": resolved.c_target,
        "run_config": run_cfg,
        "paths": {
            "method_root": str(resolved.method_root),
            "step4_reg_metrics_dir": str(resolved.step4_reg_metrics_dir),
            "step4_cls_metrics_dir": str(resolved.step4_cls_metrics_dir),
        },
    }
    if extra_context:
        snapshot["extra_context"] = extra_context
    with open(run_dirs["run_dir"] / "config_snapshot.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(as_yamlable(snapshot), handle, sort_keys=False)

    seed_info = seed_everything(int(resolved.step5["random_seed"]), deterministic=True)
    save_run_metadata(run_dirs["run_dir"], config_path, seed_info)
    write_initial_log(
        step_dir=run_dirs["run_dir"],
        step_name="step5_inverse_design",
        context={
            "config_path": config_path,
            "model_size": resolved.model_size,
            "run_name": run_cfg["run_name"],
            "canonical_family": run_cfg["canonical_family"],
            "split_mode": resolved.split_mode,
            "c_target": resolved.c_target,
            **(extra_context or {}),
        },
    )


def build_s4_warm_start_run_cfg(resolved, run_cfg: Dict[str, object]) -> Dict[str, object]:
    source_name = str(run_cfg["s4"].get("supervised_train_config_source", "s2")).strip().lower()
    if source_name != "s2":
        raise NotImplementedError(
            f"Unsupported Step 5 S4 warm-start source: {source_name}. Only 's2' is implemented."
        )
    if "S2_conditional" in resolved.enabled_runs:
        warm_cfg = build_run_config(resolved, "S2_conditional")
    else:
        warm_cfg = deepcopy(resolved.step5)
        warm_cfg["run_name"] = "S2_conditional"
        warm_cfg["canonical_family"] = "S2"
        warm_cfg["c_target"] = resolved.c_target
    if isinstance(run_cfg.get("s2"), dict):
        warm_cfg["s2"] = deepcopy(run_cfg["s2"])
    supervised_override = run_cfg.get("s4", {}).get("supervised_train_override")
    if isinstance(supervised_override, dict):
        if isinstance(supervised_override.get("s2"), dict):
            warm_cfg["s2"].update(deepcopy(supervised_override["s2"]))
        else:
            warm_cfg["s2"].update(deepcopy(supervised_override))
    warm_cfg["run_name"] = f"{run_cfg['run_name']}__warm_start"
    return warm_cfg


def _clone_module_state_dict(module: Optional[torch.nn.Module]) -> Optional[Dict[str, torch.Tensor]]:
    if module is None:
        return None
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _shared_s4_warm_start_cache_key(
    *,
    resolved,
    warm_run_cfg: Dict[str, object],
    warm_run_dir: Path,
    device: str,
) -> str:
    payload = {
        "method_root": str(resolved.method_root),
        "warm_run_dir": str(warm_run_dir),
        "model_size": str(resolved.model_size),
        "split_mode": str(resolved.split_mode),
        "c_target": str(resolved.c_target),
        "device": str(device),
        "s2_cfg": as_yamlable(dict(warm_run_cfg.get("s2", {}))),
    }
    return json.dumps(payload, sort_keys=True)


def _cache_s4_warm_start_payload(warm_start_artifacts: S2TrainingArtifacts) -> Dict[str, Any]:
    scaler = warm_start_artifacts.scaler
    return {
        "model_state_dict": _clone_module_state_dict(warm_start_artifacts.diffusion_model),
        "aux_state_dict": _clone_module_state_dict(warm_start_artifacts.aux_heads),
        "condition_scaler": {
            "temperature_min": float(scaler.temperature_min),
            "temperature_max": float(scaler.temperature_max),
            "phi_min": float(scaler.phi_min),
            "phi_max": float(scaler.phi_max),
            "chi_goal_min": float(scaler.chi_goal_min),
            "chi_goal_max": float(scaler.chi_goal_max),
        },
        "history_df": warm_start_artifacts.history_df.copy(),
        "augmentation_diag_df": warm_start_artifacts.augmentation_diag_df.copy(),
        "batch_mix_counts": deepcopy(warm_start_artifacts.batch_mix_counts),
    }


def _load_warm_start_artifacts_from_cache(
    *,
    resolved,
    warm_run_cfg: Dict[str, object],
    warm_dirs: Dict[str, Path],
    device: str,
    cache_payload: Dict[str, Any],
) -> S2TrainingArtifacts:
    tokenizer, diffusion_model, aux_heads, step1_checkpoint_path, backbone_finetune_info = build_s2_components_from_step1(
        resolved,
        device=device,
        run_cfg=warm_run_cfg,
    )
    diffusion_model.load_state_dict(cache_payload["model_state_dict"])
    if aux_heads is not None and cache_payload.get("aux_state_dict") is not None:
        aux_heads.load_state_dict(cache_payload["aux_state_dict"])
    scaler_payload = dict(cache_payload["condition_scaler"])
    scaler = ConditionScaler(
        temperature_min=float(scaler_payload["temperature_min"]),
        temperature_max=float(scaler_payload["temperature_max"]),
        phi_min=float(scaler_payload["phi_min"]),
        phi_max=float(scaler_payload["phi_max"]),
        chi_goal_min=float(scaler_payload["chi_goal_min"]),
        chi_goal_max=float(scaler_payload["chi_goal_max"]),
    )
    return S2TrainingArtifacts(
        tokenizer=tokenizer,
        diffusion_model=diffusion_model,
        aux_heads=aux_heads,
        checkpoint_path=warm_dirs["checkpoints_dir"] / "conditional_diffusion_best.pt",
        last_checkpoint_path=warm_dirs["checkpoints_dir"] / "conditional_diffusion_last.pt",
        step1_checkpoint_path=step1_checkpoint_path,
        scaler=scaler,
        history_df=cache_payload["history_df"].copy(),
        augmentation_diag_df=cache_payload["augmentation_diag_df"].copy(),
        batch_mix_counts=deepcopy(cache_payload["batch_mix_counts"]),
        backbone_finetune_info=backbone_finetune_info,
    )


def _use_shared_s4_warm_start(run_cfg: Dict[str, object], *, extra_context: Optional[Dict[str, Any]]) -> bool:
    policy = str(run_cfg.get("s4", {}).get("warm_start_policy", "")).strip().lower()
    if policy != "shared_family_producer":
        return False
    context = extra_context or {}
    if bool(context.get("hpo_mode", False)):
        return bool(context.get("allow_shared_s4_warm_start", False))
    return True


def _resolve_s4_warm_dirs(
    *,
    resolved,
    run_cfg: Dict[str, object],
    local_run_dir: Path,
    extra_context: Optional[Dict[str, Any]],
) -> Dict[str, Path]:
    create_checkpoints_dir = not bool((extra_context or {}).get("skip_disk_checkpoints", False))
    if _use_shared_s4_warm_start(run_cfg, extra_context=extra_context):
        producer_name = str(run_cfg["s4"].get("warm_start_producer_name", "S4_supervised_warm_start")).strip()
        return create_run_dirs(
            resolved.method_root / producer_name,
            create_checkpoints_dir=create_checkpoints_dir,
        )
    return create_run_dirs(local_run_dir / "_warm_start", create_checkpoints_dir=create_checkpoints_dir)


def _load_warm_start_artifacts_from_checkpoint(
    *,
    resolved,
    warm_run_cfg: Dict[str, object],
    warm_dirs: Dict[str, Path],
    device: str,
) -> S2TrainingArtifacts:
    checkpoint_path = warm_dirs["checkpoints_dir"] / "conditional_diffusion_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")

    tokenizer, diffusion_model, aux_heads, step1_checkpoint_path, backbone_finetune_info = build_s2_components_from_step1(
        resolved,
        device=device,
        run_cfg=warm_run_cfg,
    )
    payload = load_step5_checkpoint_into_modules(
        checkpoint_path=checkpoint_path,
        diffusion_model=diffusion_model,
        aux_heads=aux_heads,
        device=device,
    )
    scaler_payload = dict(payload.get("condition_scaler", {}))
    scaler = ConditionScaler(
        temperature_min=float(scaler_payload["temperature_min"]),
        temperature_max=float(scaler_payload["temperature_max"]),
        phi_min=float(scaler_payload["phi_min"]),
        phi_max=float(scaler_payload["phi_max"]),
        chi_goal_min=float(scaler_payload["chi_goal_min"]),
        chi_goal_max=float(scaler_payload["chi_goal_max"]),
    )
    history_path = warm_dirs["metrics_dir"] / "supervised_training_history.csv"
    augmentation_path = warm_dirs["metrics_dir"] / "chi_target_augmentation_eligibility.csv"
    batch_mix_path = warm_dirs["metrics_dir"] / "train_batch_mix_resolved.json"
    history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    augmentation_diag_df = pd.read_csv(augmentation_path) if augmentation_path.exists() else pd.DataFrame()
    if batch_mix_path.exists():
        with open(batch_mix_path, "r", encoding="utf-8") as handle:
            batch_mix_counts = json.load(handle)
    else:
        batch_mix_counts = {}

    return S2TrainingArtifacts(
        tokenizer=tokenizer,
        diffusion_model=diffusion_model,
        aux_heads=aux_heads,
        checkpoint_path=checkpoint_path,
        last_checkpoint_path=warm_dirs["checkpoints_dir"] / "conditional_diffusion_last.pt",
        step1_checkpoint_path=step1_checkpoint_path,
        scaler=scaler,
        history_df=history_df,
        augmentation_diag_df=augmentation_diag_df,
        batch_mix_counts=batch_mix_counts,
        backbone_finetune_info=backbone_finetune_info,
    )


def _load_s2_training_artifacts_from_existing_checkpoint(
    *,
    resolved,
    run_cfg: Dict[str, object],
    checkpoint_path: Path,
    metrics_dir: Optional[Path],
    device: str,
) -> S2TrainingArtifacts:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Reusable S2 checkpoint not found: {checkpoint_path}")

    tokenizer, diffusion_model, aux_heads, step1_checkpoint_path, backbone_finetune_info = build_s2_components_from_step1(
        resolved,
        device=device,
        run_cfg=run_cfg,
    )
    payload = load_step5_checkpoint_into_modules(
        checkpoint_path=checkpoint_path,
        diffusion_model=diffusion_model,
        aux_heads=aux_heads,
        device=device,
    )
    scaler_payload = dict(payload.get("condition_scaler", {}))
    scaler = ConditionScaler(
        temperature_min=float(scaler_payload["temperature_min"]),
        temperature_max=float(scaler_payload["temperature_max"]),
        phi_min=float(scaler_payload["phi_min"]),
        phi_max=float(scaler_payload["phi_max"]),
        chi_goal_min=float(scaler_payload["chi_goal_min"]),
        chi_goal_max=float(scaler_payload["chi_goal_max"]),
    )
    history_df = pd.DataFrame()
    augmentation_diag_df = pd.DataFrame()
    batch_mix_counts: Dict[str, int] = {}
    if metrics_dir is not None:
        history_path = metrics_dir / "supervised_training_history.csv"
        augmentation_path = metrics_dir / "chi_target_augmentation_eligibility.csv"
        batch_mix_path = metrics_dir / "train_batch_mix_resolved.json"
        history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        augmentation_diag_df = pd.read_csv(augmentation_path) if augmentation_path.exists() else pd.DataFrame()
        if batch_mix_path.exists():
            with open(batch_mix_path, "r", encoding="utf-8") as handle:
                batch_mix_counts = json.load(handle)

    return S2TrainingArtifacts(
        tokenizer=tokenizer,
        diffusion_model=diffusion_model,
        aux_heads=aux_heads,
        checkpoint_path=checkpoint_path,
        last_checkpoint_path=checkpoint_path.parent / "conditional_diffusion_last.pt",
        step1_checkpoint_path=step1_checkpoint_path,
        scaler=scaler,
        history_df=history_df,
        augmentation_diag_df=augmentation_diag_df,
        batch_mix_counts=batch_mix_counts,
        backbone_finetune_info=backbone_finetune_info,
    )


def _resolve_reusable_s2_artifact_paths(
    *,
    extra_context: Optional[Dict[str, Any]],
) -> tuple[Optional[Path], Optional[Path]]:
    context = extra_context or {}
    run_dir_raw = str(context.get("reuse_s2_run_dir", "") or "").strip()
    if run_dir_raw:
        run_dir = Path(run_dir_raw)
        return run_dir / "checkpoints" / "conditional_diffusion_best.pt", run_dir / "metrics"

    checkpoint_raw = str(context.get("reuse_s2_checkpoint_path", "") or "").strip()
    if not checkpoint_raw:
        return None, None
    checkpoint_path = Path(checkpoint_raw)
    metrics_raw = str(context.get("reuse_s2_metrics_dir", "") or "").strip()
    metrics_dir = Path(metrics_raw) if metrics_raw else checkpoint_path.parent.parent / "metrics"
    return checkpoint_path, metrics_dir


def _resolve_reusable_s4_checkpoint_path(
    *,
    run_cfg: Dict[str, object],
    extra_context: Optional[Dict[str, Any]],
) -> Optional[Path]:
    context = extra_context or {}
    checkpoint_raw = str(context.get("reuse_s4_checkpoint_path", "") or "").strip()
    if checkpoint_raw:
        return Path(checkpoint_raw)

    run_dir_raw = str(context.get("reuse_s4_run_dir", "") or "").strip()
    if not run_dir_raw:
        return None

    mode = str(context.get("reuse_s4_checkpoint_mode", "best") or "best").strip().lower()
    if mode not in {"best", "last"}:
        raise ValueError("reuse_s4_checkpoint_mode must be one of {'best', 'last'}.")
    alignment_mode = str(run_cfg.get("s4", {}).get("alignment_mode", "")).strip().lower()
    if alignment_mode not in {"rl", "ppo", "grpo"}:
        raise ValueError("reuse_s4_run_dir is supported only for S4 rl/ppo/grpo runs.")
    return Path(run_dir_raw) / "checkpoints" / f"aligned_{alignment_mode}_{mode}.pt"


def _load_s4_aligned_checkpoint_for_sampling(
    *,
    checkpoint_path: Path,
    policy_model: torch.nn.Module,
    device: str,
) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Reusable S4 aligned checkpoint not found: {checkpoint_path}")
    return load_step5_checkpoint_into_modules(
        checkpoint_path=checkpoint_path,
        diffusion_model=policy_model,
        aux_heads=None,
        device=device,
    )


def _prepare_s4_warm_start(
    *,
    resolved,
    run_cfg: Dict[str, object],
    run_dirs: Dict[str, Path],
    config_path: str,
    device: str,
    extra_context: Optional[Dict[str, Any]],
    pruning_callback: Optional[Callable[..., None]],
) -> Tuple[Dict[str, object], Dict[str, Path], S2TrainingArtifacts]:
    warm_run_cfg = build_s4_warm_start_run_cfg(resolved, run_cfg)
    using_shared_warm_start = _use_shared_s4_warm_start(run_cfg, extra_context=extra_context)
    skip_disk_checkpoints = bool((extra_context or {}).get("skip_disk_checkpoints", False))
    warm_dirs = _resolve_s4_warm_dirs(
        resolved=resolved,
        run_cfg=run_cfg,
        local_run_dir=run_dirs["run_dir"],
        extra_context=extra_context,
    )
    cache_key = None
    if using_shared_warm_start and skip_disk_checkpoints:
        cache_key = _shared_s4_warm_start_cache_key(
            resolved=resolved,
            warm_run_cfg=warm_run_cfg,
            warm_run_dir=warm_dirs["run_dir"],
            device=device,
        )
        cache_payload = _HPO_SHARED_S4_WARM_START_CACHE.get(cache_key)
        if cache_payload is not None:
            warm_start_artifacts = _load_warm_start_artifacts_from_cache(
                resolved=resolved,
                warm_run_cfg=warm_run_cfg,
                warm_dirs=warm_dirs,
                device=device,
                cache_payload=cache_payload,
            )
            return warm_run_cfg, warm_dirs, warm_start_artifacts
    checkpoint_path = warm_dirs["checkpoints_dir"] / "conditional_diffusion_best.pt"
    if checkpoint_path.exists():
        warm_start_artifacts = _load_warm_start_artifacts_from_checkpoint(
            resolved=resolved,
            warm_run_cfg=warm_run_cfg,
            warm_dirs=warm_dirs,
            device=device,
        )
        return warm_run_cfg, warm_dirs, warm_start_artifacts

    save_run_config_snapshot(
        warm_dirs,
        run_cfg=warm_run_cfg,
        resolved=resolved,
        config_path=config_path,
        extra_context={"parent_run_name": run_cfg["run_name"], **(extra_context or {})},
    )
    warm_start_artifacts = train_s2_supervised_run(
        resolved=resolved,
        run_cfg=warm_run_cfg,
        run_dirs=warm_dirs,
        device=device,
        skip_disk_checkpoints=bool(skip_disk_checkpoints),
        pruning_callback=(None if using_shared_warm_start else pruning_callback),
        pruning_stage="warm_start",
    )
    if cache_key is not None:
        _HPO_SHARED_S4_WARM_START_CACHE[cache_key] = _cache_s4_warm_start_payload(warm_start_artifacts)
    return warm_run_cfg, warm_dirs, warm_start_artifacts


def run_single_target_sampling(
    *,
    run_cfg: Dict[str, object],
    resolved,
    target_row: pd.Series,
    tokenizer,
    diffusion_model,
    prior,
    device: str,
    generation_budget: int,
    evaluator=None,
    s2_scaler=None,
    seen_canonical_smiles: Optional[set[str]] = None,
    sampling_state: Optional[Dict[str, object]] = None,
) -> Tuple[List[str], Dict[str, int], Dict[str, object]]:
    if run_cfg["run_name"] == "S0_raw_unconditional":
        sampler = create_raw_step1_sampler(
            diffusion_model=diffusion_model,
            tokenizer=tokenizer,
            resolved=resolved,
            device=device,
        )
        smiles, sample_meta = sample_raw_step1_unconditional(
            sampler=sampler,
            tokenizer=tokenizer,
            resolved=resolved,
            num_samples=int(generation_budget),
            show_progress=False,
            source_lengths=list(getattr(prior, "fallback_source_lengths", []) or []),
        )
        return smiles, {"training_soluble_oracle_calls": 0, "training_chi_oracle_calls": 0}, sample_meta

    if str(run_cfg["canonical_family"]) == "S1":
        if evaluator is None:
            raise ValueError("S1 guided frozen sampling requires a loaded Step 5 evaluator.")
        s1_cfg = run_cfg["s1"]
        sampler = GuidedSampler(
            diffusion_model=diffusion_model,
            tokenizer=tokenizer,
            num_steps=resolve_step5_sampling_num_steps(resolved.step5, resolved.base_config),
            temperature=float(resolved.step5["sampling_temperature"]),
            top_k=resolved.base_config.get("sampling", {}).get("top_k"),
            top_p=resolved.base_config.get("sampling", {}).get("top_p"),
            target_stars=int(resolved.base_config.get("sampling", {}).get("target_stars", 2)),
            use_constraints=bool(resolved.base_config.get("sampling", {}).get("use_constraints", True)),
            device=device,
            evaluator=evaluator,
            target_row=target_row.to_dict(),
            best_of_k=int(s1_cfg["best_of_k"]),
            guidance_start_frac=float(s1_cfg["guidance_start_frac"]),
            sol_log_prob_floor=float(s1_cfg["sol_log_prob_floor"]),
            w_sol=float(s1_cfg["w_sol"]),
            w_chi=float(s1_cfg["w_chi"]),
            w_sa=float(s1_cfg.get("w_sa", 0.0)),
            w_sa_continuous=float(s1_cfg.get("w_sa_continuous", 0.0)),
            invalid_reward_penalty=float(s1_cfg.get("invalid_reward_penalty", -10.0)),
        )
        sampler.set_class_token_bias_start_frac(float(resolved.step5.get("class_token_bias_start_frac", 0.0)))
        if prior.class_token_logit_bias is not None:
            sampler.set_class_token_logit_bias(prior.class_token_logit_bias)
        smiles, sample_meta = sample_unconditional_with_class_prior(
            sampler=sampler,
            tokenizer=tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(generation_budget),
            show_progress=False,
            seen_canonical_smiles=seen_canonical_smiles,
            sampling_state=sampling_state,
        )
        return smiles, sampler.get_guidance_stats(), sample_meta

    if str(run_cfg["canonical_family"]) == "S2":
        if s2_scaler is None:
            raise ValueError("S2 conditional sampling requires the fitted ConditionScaler.")
        condition_bundle = torch.tensor(
            build_inference_condition_bundle_from_target_row(
                target_row.to_dict(),
                scaler=s2_scaler,
                soluble=1,
            ),
            dtype=torch.float32,
            device=device,
        )
        sampler = create_conditional_sampler(
            diffusion_model=diffusion_model,
            tokenizer=tokenizer,
            resolved=resolved,
            prior=prior,
            condition_bundle=condition_bundle,
            cfg_scale=float(run_cfg["s2"]["cfg_scale"]),
            device=device,
        )
        smiles, sample_meta = sample_conditional_with_class_prior(
            sampler=sampler,
            tokenizer=tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(generation_budget),
            show_progress=False,
            seen_canonical_smiles=seen_canonical_smiles,
            sampling_state=sampling_state,
        )
        return smiles, {"training_soluble_oracle_calls": 0, "training_chi_oracle_calls": 0}, sample_meta

    if str(run_cfg["canonical_family"]) == "S3":
        if evaluator is None:
            raise ValueError("S3 conditional guided sampling requires a loaded Step 5 evaluator.")
        if s2_scaler is None:
            raise ValueError("S3 conditional guided sampling requires the fitted ConditionScaler.")
        s3_cfg = run_cfg["s3"]
        condition_bundle = torch.tensor(
            build_inference_condition_bundle_from_target_row(
                target_row.to_dict(),
                scaler=s2_scaler,
                soluble=1,
            ),
            dtype=torch.float32,
            device=device,
        )
        sampler = GuidedConditionalSampler(
            diffusion_model=diffusion_model,
            tokenizer=tokenizer,
            num_steps=resolve_step5_sampling_num_steps(resolved.step5, resolved.base_config),
            temperature=float(resolved.step5["sampling_temperature"]),
            top_k=resolved.base_config.get("sampling", {}).get("top_k"),
            top_p=resolved.base_config.get("sampling", {}).get("top_p"),
            target_stars=int(resolved.base_config.get("sampling", {}).get("target_stars", 2)),
            use_constraints=bool(resolved.base_config.get("sampling", {}).get("use_constraints", True)),
            device=device,
            condition_bundle=condition_bundle,
            cfg_scale=float(s3_cfg["cfg_scale"]),
            evaluator=evaluator,
            target_row=target_row.to_dict(),
            best_of_k=int(s3_cfg["best_of_k"]),
            guidance_start_frac=float(s3_cfg["guidance_start_frac"]),
            sol_log_prob_floor=float(s3_cfg["sol_log_prob_floor"]),
            w_sol=float(s3_cfg["w_sol"]),
            w_chi=float(s3_cfg["w_chi"]),
            w_sa=float(s3_cfg.get("w_sa", 0.0)),
            w_sa_continuous=float(s3_cfg.get("w_sa_continuous", 0.0)),
            invalid_reward_penalty=float(s3_cfg.get("invalid_reward_penalty", -10.0)),
        )
        sampler.set_class_token_bias_start_frac(float(resolved.step5.get("class_token_bias_start_frac", 0.0)))
        if prior.class_token_logit_bias is not None:
            sampler.set_class_token_logit_bias(prior.class_token_logit_bias)
        smiles, sample_meta = sample_conditional_with_class_prior(
            sampler=sampler,
            tokenizer=tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(generation_budget),
            show_progress=False,
            seen_canonical_smiles=seen_canonical_smiles,
            sampling_state=sampling_state,
        )
        return smiles, sampler.get_guidance_stats(), sample_meta

    if str(run_cfg["canonical_family"]) == "S4":
        if s2_scaler is None:
            raise ValueError("S4 sampling requires the fitted ConditionScaler from the warm start.")
        condition_bundle = torch.tensor(
            build_inference_condition_bundle_from_target_row(
                target_row.to_dict(),
                scaler=s2_scaler,
                soluble=1,
            ),
            dtype=torch.float32,
            device=device,
        )
        sampler = create_conditional_sampler(
            diffusion_model=diffusion_model,
            tokenizer=tokenizer,
            resolved=resolved,
            prior=prior,
            condition_bundle=condition_bundle,
            cfg_scale=float(run_cfg["s4"]["cfg_scale"]),
            device=device,
        )
        smiles, sample_meta = sample_conditional_with_class_prior(
            sampler=sampler,
            tokenizer=tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(generation_budget),
            show_progress=False,
            seen_canonical_smiles=seen_canonical_smiles,
            sampling_state=sampling_state,
        )
        return smiles, {"training_soluble_oracle_calls": 0, "training_chi_oracle_calls": 0}, sample_meta

    raise NotImplementedError(f"Unsupported Step 5 run: {run_cfg['run_name']}")


def _resolve_round_generation_budgets(
    *,
    generation_budget: int,
    num_rounds: int,
    preserve_total_generation_across_rounds: bool,
) -> List[int]:
    """Return the per-round sampling budget for one target row."""

    generation_budget = int(generation_budget)
    num_rounds = int(num_rounds)
    if generation_budget <= 0:
        raise ValueError("generation_budget must be >= 1.")
    if num_rounds <= 0:
        raise ValueError("num_rounds must be >= 1.")
    if not preserve_total_generation_across_rounds:
        return [generation_budget for _ in range(num_rounds)]
    if generation_budget < num_rounds:
        raise ValueError(
            "step5.generation_budget must be >= step5.num_sampling_rounds when "
            "step5.preserve_total_generation_across_rounds is true."
        )
    base_budget, extra_rounds = divmod(generation_budget, num_rounds)
    return [base_budget + (1 if idx < extra_rounds else 0) for idx in range(num_rounds)]


def execute_step5_run(
    *,
    resolved,
    run_name: str,
    device: str,
    config_path: str,
    run_cfg: Optional[Dict[str, object]] = None,
    run_dir: Optional[Path] = None,
    shared_evaluator=None,
    target_rows_df: Optional[pd.DataFrame] = None,
    generation_budget: Optional[int] = None,
    sampling_seeds: Optional[List[int]] = None,
    num_rounds: Optional[int] = None,
    save_figures: bool = True,
    extra_context: Optional[Dict[str, Any]] = None,
    pruning_callback: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    extra_context = extra_context or {}
    run_wall_start = time.perf_counter()
    run_cfg = run_cfg or build_run_config(resolved, run_name)
    run_dir = run_dir or (resolved.method_root / str(run_cfg["run_name"]))
    explicit_target_rows = target_rows_df is not None
    target_rows_df = target_rows_df.copy() if explicit_target_rows else resolved.target_family_df.copy()
    s0_target_condition_filter: Optional[Dict[str, Any]] = None
    s0_target_temperature, s0_target_phi = _resolve_s0_target_condition(resolved, run_cfg)
    if s0_target_temperature is not None and s0_target_phi is not None:
        try:
            target_rows_df = filter_step5_target_rows_by_condition(
                target_rows_df,
                target_temperature=float(s0_target_temperature),
                target_phi=float(s0_target_phi),
            )
        except ValueError:
            if not explicit_target_rows:
                raise
            target_rows_df = filter_step5_target_rows_by_condition(
                resolved.target_family_df,
                target_temperature=float(s0_target_temperature),
                target_phi=float(s0_target_phi),
            )
        s0_target_condition_filter = {
            "target_temperature": float(s0_target_temperature),
            "target_phi": float(s0_target_phi),
            "matched_target_rows": int(len(target_rows_df)),
        }
    raw_max_target_rows = resolved.step5.get("max_target_rows")
    if not explicit_target_rows and raw_max_target_rows not in {None, "", "null"}:
        max_target_rows = int(raw_max_target_rows)
        if max_target_rows <= 0:
            raise ValueError("step5.max_target_rows must be >= 1 when provided.")
        target_rows_df = target_rows_df.head(max_target_rows).copy()
        if target_rows_df.empty:
            raise ValueError("step5.max_target_rows selected zero target rows.")
    generation_budget = int(
        generation_budget
        if generation_budget is not None
        else resolve_step5_generation_budget(run_cfg, resolved.c_target)
    )
    sampling_seeds = list(sampling_seeds if sampling_seeds is not None else resolved.step5["sampling_seeds"])
    num_rounds = int(num_rounds if num_rounds is not None else resolved.step5["num_sampling_rounds"])
    if num_rounds > len(sampling_seeds):
        raise ValueError("num_rounds exceeds the number of provided sampling seeds.")
    preserve_total_generation_across_rounds = bool(
        run_cfg.get("preserve_total_generation_across_rounds", False)
    )
    round_generation_budgets = _resolve_round_generation_budgets(
        generation_budget=generation_budget,
        num_rounds=num_rounds,
        preserve_total_generation_across_rounds=preserve_total_generation_across_rounds,
    )
    generation_budget_mode = (
        "total_across_rounds" if preserve_total_generation_across_rounds else "per_round"
    )

    skip_disk_checkpoints = bool(extra_context.get("skip_disk_checkpoints", False))
    run_dirs = create_run_dirs(run_dir, create_checkpoints_dir=not skip_disk_checkpoints)
    save_run_config_snapshot(
        run_dirs,
        run_cfg=run_cfg,
        resolved=resolved,
        config_path=config_path,
        extra_context=extra_context,
    )
    canonical_family = str(run_cfg["canonical_family"])
    append_log_message(
        run_dirs["run_dir"],
        (
            f"Run start | run={run_cfg['run_name']} family={canonical_family} "
            f"generation_budget={int(generation_budget)} generation_budget_mode={generation_budget_mode} "
            f"num_rounds={int(num_rounds)} round_generation_budgets={round_generation_budgets} "
            f"target_rows={int(len(target_rows_df))}"
            + (
                f" s0_target_condition=T{float(s0_target_condition_filter['target_temperature']):.6g},"
                f"phi{float(s0_target_condition_filter['target_phi']):.6g}"
                if s0_target_condition_filter is not None
                else ""
            )
        ),
        echo=True,
    )
    evaluator = shared_evaluator
    s2_scaler = None
    model_setup_start = time.perf_counter()

    if canonical_family in {"S2", "S3"}:
        reuse_checkpoint_path, reuse_metrics_dir = _resolve_reusable_s2_artifact_paths(extra_context=extra_context)
        if reuse_checkpoint_path is not None:
            training_artifacts = _load_s2_training_artifacts_from_existing_checkpoint(
                resolved=resolved,
                run_cfg=run_cfg,
                checkpoint_path=reuse_checkpoint_path,
                metrics_dir=reuse_metrics_dir,
                device=device,
            )
            _write_json(
                {
                    "reused_s2_checkpoint_path": str(reuse_checkpoint_path),
                    "reused_s2_metrics_dir": (str(reuse_metrics_dir) if reuse_metrics_dir is not None else None),
                },
                run_dirs["metrics_dir"] / "reused_s2_checkpoint.json",
            )
        else:
            training_artifacts = train_s2_supervised_run(
                resolved=resolved,
                run_cfg=run_cfg,
                run_dirs=run_dirs,
                device=device,
                skip_disk_checkpoints=skip_disk_checkpoints,
                pruning_callback=pruning_callback,
                pruning_stage="s2",
            )
        tokenizer = training_artifacts.tokenizer
        diffusion_model = training_artifacts.diffusion_model
        prior = resolve_class_sampling_prior(resolved, run_cfg, tokenizer, metrics_dir=run_dirs["metrics_dir"])
        s2_scaler = training_artifacts.scaler
    elif canonical_family == "S4" and str(run_cfg["s4"]["alignment_mode"]).strip().lower() == "dpo":
        warm_run_cfg, warm_dirs, warm_start_artifacts = _prepare_s4_warm_start(
            resolved=resolved,
            run_cfg=run_cfg,
            run_dirs=run_dirs,
            config_path=config_path,
            device=device,
            extra_context=extra_context,
            pruning_callback=pruning_callback,
        )
        prior = resolve_class_sampling_prior(
            resolved,
            run_cfg,
            warm_start_artifacts.tokenizer,
            metrics_dir=run_dirs["metrics_dir"],
        )
        pair_source = str(run_cfg["s4"]["dpo"]["pair_source"]).strip().lower()
        checkpoint_mode = str(run_cfg["s4"]["dpo"].get("checkpoint_selection_mode", "val_dpo_loss")).strip().lower()
        if (
            pair_source in {"target_row_synthetic", "chi_aware_plus_target_row_synthetic"}
            or checkpoint_mode == "proxy_property_success_hit_rate"
        ) and evaluator is None:
            evaluator = load_step5_evaluator(resolved, device=device)
        dpo_artifacts = train_s4_dpo_alignment(
            resolved=resolved,
            run_cfg=run_cfg,
            run_dirs=run_dirs,
            warm_start=warm_start_artifacts,
            prior=prior,
            evaluator=evaluator,
            device=device,
            target_rows_df=target_rows_df,
            skip_disk_checkpoints=skip_disk_checkpoints,
            pruning_callback=pruning_callback,
        )
        tokenizer = dpo_artifacts.tokenizer
        diffusion_model = dpo_artifacts.policy_model
        s2_scaler = dpo_artifacts.scaler
        _write_json(
            {
                "warm_start_run_name": warm_run_cfg["run_name"],
                "warm_start_dir": str(warm_dirs["run_dir"]),
                "warm_start_best_checkpoint": (None if skip_disk_checkpoints else str(warm_start_artifacts.checkpoint_path)),
                "aligned_best_checkpoint": (None if skip_disk_checkpoints else str(dpo_artifacts.checkpoint_path)),
                "disk_checkpoints_saved": bool(not skip_disk_checkpoints),
            },
            run_dirs["metrics_dir"] / "s4_alignment_summary.json",
        )
    elif canonical_family == "S4" and str(run_cfg["s4"]["alignment_mode"]).strip().lower() in {"rl", "ppo", "grpo"}:
        warm_run_cfg, warm_dirs, warm_start_artifacts = _prepare_s4_warm_start(
            resolved=resolved,
            run_cfg=run_cfg,
            run_dirs=run_dirs,
            config_path=config_path,
            device=device,
            extra_context=extra_context,
            pruning_callback=pruning_callback,
        )
        prior = resolve_class_sampling_prior(
            resolved,
            run_cfg,
            warm_start_artifacts.tokenizer,
            metrics_dir=run_dirs["metrics_dir"],
        )
        reuse_s4_checkpoint_path = _resolve_reusable_s4_checkpoint_path(
            run_cfg=run_cfg,
            extra_context=extra_context,
        )
        if reuse_s4_checkpoint_path is not None:
            checkpoint_payload = _load_s4_aligned_checkpoint_for_sampling(
                checkpoint_path=reuse_s4_checkpoint_path,
                policy_model=warm_start_artifacts.diffusion_model,
                device=device,
            )
            tokenizer = warm_start_artifacts.tokenizer
            diffusion_model = warm_start_artifacts.diffusion_model
            append_log_message(
                run_dirs["run_dir"],
                (
                    f"S4 sampling-only checkpoint loaded | run={run_cfg['run_name']} "
                    f"checkpoint={reuse_s4_checkpoint_path}"
                ),
                echo=True,
            )
            _write_json(
                {
                    "warm_start_run_name": warm_run_cfg["run_name"],
                    "warm_start_dir": str(warm_dirs["run_dir"]),
                    "warm_start_best_checkpoint": (
                        None if skip_disk_checkpoints else str(warm_start_artifacts.checkpoint_path)
                    ),
                    "aligned_best_checkpoint": str(reuse_s4_checkpoint_path),
                    "reused_s4_checkpoint_path": str(reuse_s4_checkpoint_path),
                    "reuse_s4_checkpoint_step_idx": checkpoint_payload.get("step_idx"),
                    "reuse_s4_checkpoint_mode": "sampling_only",
                    "disk_checkpoints_saved": bool(not skip_disk_checkpoints),
                },
                run_dirs["metrics_dir"] / "s4_alignment_summary.json",
            )
        else:
            if evaluator is None:
                evaluator = load_step5_evaluator(resolved, device=device)
            rl_artifacts = train_s4_rl_alignment(
                resolved=resolved,
                run_cfg=run_cfg,
                run_dirs=run_dirs,
                warm_start=warm_start_artifacts,
                prior=prior,
                evaluator=evaluator,
                device=device,
                target_rows_df=target_rows_df,
                skip_disk_checkpoints=skip_disk_checkpoints,
                pruning_callback=pruning_callback,
            )
            tokenizer = rl_artifacts.tokenizer
            diffusion_model = rl_artifacts.policy_model
            _write_json(
                {
                    "warm_start_run_name": warm_run_cfg["run_name"],
                    "warm_start_dir": str(warm_dirs["run_dir"]),
                    "warm_start_best_checkpoint": (
                        None if skip_disk_checkpoints else str(warm_start_artifacts.checkpoint_path)
                    ),
                    "aligned_best_checkpoint": (
                        None if skip_disk_checkpoints else str(rl_artifacts.checkpoint_path)
                    ),
                    "disk_checkpoints_saved": bool(not skip_disk_checkpoints),
                },
                run_dirs["metrics_dir"] / "s4_alignment_summary.json",
            )
        s2_scaler = warm_start_artifacts.scaler
    else:
        tokenizer, diffusion_model, checkpoint_path = load_step1_diffusion(resolved, device=device)
        s0_raw_step1_baseline = str(run_cfg["run_name"]) == "S0_raw_unconditional"
        prior = resolve_class_sampling_prior(
            resolved,
            run_cfg,
            tokenizer,
            metrics_dir=(None if s0_raw_step1_baseline else run_dirs["metrics_dir"]),
        )
        _write_json({"checkpoint_path": str(checkpoint_path)}, run_dirs["metrics_dir"] / "step1_checkpoint.json")
        if s0_raw_step1_baseline:
            _write_json(
                {
                    "sampling_mode": "step1_unconditional",
                    "class_aware_decode_constraints_enabled": False,
                    "class_token_bias_enabled": False,
                    "family_sampling_mode": "none",
                    "enforce_class_match": False,
                    "enforce_backbone_class_match": False,
                    "enforce_star_ok_acceptance": False,
                    "target_condition_filter": s0_target_condition_filter,
                },
                run_dirs["metrics_dir"] / "s0_raw_step1_sampling.json",
            )

    sampling_wall_time_seconds = 0.0
    evaluation_wall_time_seconds = 0.0
    generated_frames: List[pd.DataFrame] = []
    evaluation_frames: List[pd.DataFrame] = []
    round_oracle_rows: List[Dict[str, int]] = []
    sampling_meta_rows: List[Dict[str, Any]] = []
    sample_id_start = 1
    hpo_objective_metric = str(extra_context.get("hpo_objective_metric", "mean_success_hit_rate_discovery")).strip()
    hpo_sampling_pruning_enabled = bool(extra_context.get("hpo_enable_sampling_pruning", False))
    hpo_objective_sum = 0.0
    hpo_objective_count = 0
    hpo_sampling_step = 0
    reject_duplicate_canonical_across_targets = _resolve_cross_target_duplicate_rejection_enabled(resolved)
    cycle_backbone_template_cores_across_targets = bool(
        getattr(prior, "cycle_backbone_template_cores_across_targets", False)
    )

    if canonical_family in {"S1", "S3"} and evaluator is None:
        evaluator = load_step5_evaluator(resolved, device=device)
    model_setup_wall_time_seconds = time.perf_counter() - model_setup_start

    for round_id, (sampling_seed, round_generation_budget) in enumerate(
        zip(sampling_seeds[:num_rounds], round_generation_budgets),
        start=1,
    ):
        append_log_message(
            run_dirs["run_dir"],
            (
                f"Sampling round start | run={run_cfg['run_name']} "
                f"round={int(round_id)}/{int(num_rounds)} seed={int(sampling_seed)} "
                f"generation_budget={int(round_generation_budget)} "
                f"target_rows={int(len(target_rows_df))}"
            ),
            echo=True,
        )
        seed_everything(int(sampling_seed), deterministic=True)
        round_soluble_calls = 0
        round_chi_calls = 0
        round_generated_count = 0
        round_valid_count = 0
        round_property_hit_count = 0
        round_success_hit_count = 0
        round_seen_canonical_smiles: Optional[set[str]] = set() if reject_duplicate_canonical_across_targets else None
        round_sampling_state: Optional[Dict[str, object]] = (
            {}
            if cycle_backbone_template_cores_across_targets
            else None
        )

        for target_idx, (_, target_row) in enumerate(target_rows_df.iterrows(), start=1):
            append_log_message(
                run_dirs["run_dir"],
                (
                    f"Target start | run={run_cfg['run_name']} round={int(round_id)} "
                    f"target={int(target_idx)}/{int(len(target_rows_df))} "
                    f"target_row_id={int(target_row['target_row_id'])} "
                    f"T={float(target_row['temperature']):.2f} "
                    f"phi={float(target_row['phi']):.2f} "
                    f"chi_target={float(target_row['chi_target']):.4f}"
                ),
                echo=True,
            )
            target_sampling_start = time.perf_counter()
            smiles, guidance_stats, sample_meta = run_single_target_sampling(
                run_cfg=run_cfg,
                resolved=resolved,
                target_row=target_row,
                tokenizer=tokenizer,
                diffusion_model=diffusion_model,
                prior=prior,
                evaluator=evaluator,
                device=device,
                s2_scaler=s2_scaler,
                generation_budget=int(round_generation_budget),
                seen_canonical_smiles=round_seen_canonical_smiles,
                sampling_state=round_sampling_state,
            )
            target_sampling_wall_time_seconds = time.perf_counter() - target_sampling_start
            sampling_wall_time_seconds += float(target_sampling_wall_time_seconds)
            generated_df = build_generated_samples_frame(
                smiles,
                target_row=target_row,
                round_id=round_id,
                sampling_seed=int(sampling_seed),
                run_name=str(run_cfg["run_name"]),
                canonical_family=str(run_cfg["canonical_family"]),
                sample_id_start=sample_id_start,
            )
            sample_id_start += int(len(generated_df))
            if evaluator is None:
                evaluator = load_step5_evaluator(resolved, device=device)
            target_evaluation_start = time.perf_counter()
            evaluation_df = evaluate_generated_samples(generated_df, evaluator)
            target_evaluation_wall_time_seconds = time.perf_counter() - target_evaluation_start
            evaluation_wall_time_seconds += float(target_evaluation_wall_time_seconds)
            generated_count = int(len(generated_df))
            valid_count = int(evaluation_df["valid_ok"].astype(int).sum()) if "valid_ok" in evaluation_df.columns else 0
            property_hit_count = (
                int(evaluation_df["property_success_hit"].astype(int).sum())
                if "property_success_hit" in evaluation_df.columns
                else 0
            )
            success_hit_count = (
                int(evaluation_df["success_hit"].astype(int).sum())
                if "success_hit" in evaluation_df.columns
                else 0
            )
            round_generated_count += generated_count
            round_valid_count += valid_count
            round_property_hit_count += property_hit_count
            round_success_hit_count += success_hit_count
            generated_frames.append(generated_df)
            evaluation_frames.append(evaluation_df)
            round_soluble_calls += int(guidance_stats.get("training_soluble_oracle_calls", 0))
            round_chi_calls += int(guidance_stats.get("training_chi_oracle_calls", 0))
            sampling_meta_rows.append(
                {
                    "run_name": str(run_cfg["run_name"]),
                    "canonical_family": str(run_cfg["canonical_family"]),
                    "round_id": int(round_id),
                    "sampling_seed": int(sampling_seed),
                    "target_row_id": int(target_row["target_row_id"]),
                    "target_row_key": str(target_row["target_row_key"]),
                    "c_target": str(target_row["c_target"]),
                    "temperature": float(target_row["temperature"]),
                    "phi": float(target_row["phi"]),
                    "chi_target": float(target_row["chi_target"]),
                    "configured_generation_budget": int(generation_budget),
                    "round_generation_budget": int(round_generation_budget),
                    "generation_budget_mode": generation_budget_mode,
                    "target_sampling_wall_time_seconds": float(target_sampling_wall_time_seconds),
                    "target_evaluation_wall_time_seconds": float(target_evaluation_wall_time_seconds),
                    **sample_meta,
                }
            )
            append_log_message(
                run_dirs["run_dir"],
                (
                    f"Target done | run={run_cfg['run_name']} round={int(round_id)} "
                    f"target={int(target_idx)}/{int(len(target_rows_df))} "
                    f"generated={generated_count} valid={valid_count} "
                    f"property_hits={property_hit_count} success_hits={success_hit_count}"
                ),
                echo=True,
            )
            if pruning_callback is not None and hpo_sampling_pruning_enabled:
                objective_column = _resolve_hpo_sampling_objective_column(hpo_objective_metric, evaluation_df)
                if objective_column is not None:
                    hpo_objective_sum += float(
                        pd.to_numeric(evaluation_df[objective_column], errors="coerce").fillna(0.0).sum()
                    )
                    hpo_objective_count += int(len(evaluation_df))
                elif generated_count > 0:
                    hpo_objective_count += int(generated_count)
                hpo_sampling_step += 1
                cumulative_objective = (
                    float(hpo_objective_sum) / float(hpo_objective_count)
                    if hpo_objective_count > 0
                    else 0.0
                )
                pruning_callback(
                    stage="sampling",
                    step=int(hpo_sampling_step),
                    value=float(cumulative_objective),
                    metrics={
                        "objective_metric": hpo_objective_metric,
                        "sampling_objective_column": objective_column,
                        "pruning_metric": str(objective_column or hpo_objective_metric),
                        "cumulative_sampling_objective": float(cumulative_objective),
                        "cumulative_sampling_samples": int(hpo_objective_count),
                        "round_id": int(round_id),
                        "target_idx": int(target_idx),
                        "target_rows": int(len(target_rows_df)),
                        "generated": int(generated_count),
                        "valid": int(valid_count),
                        "property_hits": int(property_hit_count),
                        "success_hits": int(success_hit_count),
                    },
                )

        round_oracle_rows.append(
            {
                "run_name": str(run_cfg["run_name"]),
                "canonical_family": str(run_cfg["canonical_family"]),
                "round_id": round_id,
                "sampling_seed": int(sampling_seed),
                "training_soluble_oracle_calls": int(round_soluble_calls),
                "training_chi_oracle_calls": int(round_chi_calls),
            }
        )
        append_log_message(
            run_dirs["run_dir"],
            (
                f"Sampling round complete | run={run_cfg['run_name']} round={int(round_id)}/{int(num_rounds)} "
                f"generated={int(round_generated_count)} valid={int(round_valid_count)} "
                f"property_hits={int(round_property_hit_count)} success_hits={int(round_success_hit_count)} "
                f"train_sol_oracles={int(round_soluble_calls)} train_chi_oracles={int(round_chi_calls)}"
            ),
            echo=True,
        )

    generated_samples_df = pd.concat(generated_frames, ignore_index=True) if generated_frames else pd.DataFrame()
    evaluation_results_df = pd.concat(evaluation_frames, ignore_index=True) if evaluation_frames else pd.DataFrame()
    if "success_hit" in evaluation_results_df.columns:
        success_hit_mask = evaluation_results_df["success_hit"].astype(int) == 1
        success_hit_polymers_df = evaluation_results_df.loc[success_hit_mask].copy()
    else:
        success_hit_polymers_df = evaluation_results_df.iloc[0:0].copy()
    success_hit_column_renames = {
        "chi_pred_target": "predicted_chi",
        "chi_pred_std_target": "predicted_chi_std",
        "class_prob": "predicted_water_soluble_probability",
        "class_prob_std": "predicted_water_soluble_probability_std",
        "class_logit": "predicted_water_soluble_logit",
        "soluble_ok": "predicted_water_soluble_label",
        "chi_ok": "predicted_chi_ok",
        "class_ok": "predicted_class_match_label",
        "property_success_hit": "property_success_hit_label",
        "property_success_hit_discovery": "property_success_hit_discovery_label",
        "success_hit": "success_hit_label",
    }
    applicable_success_hit_renames = {
        source_col: renamed_col
        for source_col, renamed_col in success_hit_column_renames.items()
        if source_col in success_hit_polymers_df.columns and renamed_col not in success_hit_polymers_df.columns
    }
    if applicable_success_hit_renames:
        success_hit_polymers_df = success_hit_polymers_df.rename(
            columns=applicable_success_hit_renames
        )
    success_hit_polymers_df = success_hit_polymers_df.drop(
        columns=["success_hit_discovery"],
        errors="ignore",
    )
    target_row_metrics_df = aggregate_target_row_metrics(evaluation_results_df)
    target_row_summary_df = summarize_target_rows(target_row_metrics_df)
    round_metrics_df = aggregate_round_metrics(evaluation_results_df, target_row_metrics_df)
    round_oracle_df = pd.DataFrame(round_oracle_rows)
    sampling_meta_df = pd.DataFrame(sampling_meta_rows)
    if not round_oracle_df.empty:
        round_metrics_df = round_metrics_df.drop(
            columns=["training_soluble_oracle_calls", "training_chi_oracle_calls"],
            errors="ignore",
        ).merge(
            round_oracle_df,
            on=["run_name", "canonical_family", "round_id", "sampling_seed"],
            how="left",
        )

    method_metrics = build_method_metrics(round_metrics_df, target_row_summary_df)
    class_aware_sampling_enabled = str(run_cfg["run_name"]) != "S0_raw_unconditional"
    method_family_sampling_mode = str(prior.family_sampling_mode) if class_aware_sampling_enabled else "none"
    method_family_sampling_scope = str(prior.family_sampling_scope) if class_aware_sampling_enabled else "none"
    method_center_min_frac = float(prior.center_min_frac) if class_aware_sampling_enabled else 0.0
    method_center_max_frac = float(prior.center_max_frac) if class_aware_sampling_enabled else 0.0
    method_spans_per_sample = int(prior.spans_per_sample) if class_aware_sampling_enabled else 0
    method_backbone_min_gap = int(prior.backbone_template_min_gap_tokens) if class_aware_sampling_enabled else 0
    method_backbone_core_count = int(len(prior.backbone_template_cores)) if class_aware_sampling_enabled else 0
    method_backbone_max_core_length = (
        float(max((len(tokenizer.tokenize(core)) for core in prior.backbone_template_cores), default=0))
        if class_aware_sampling_enabled
        else 0.0
    )
    method_enforce_star_ok_acceptance = (
        int(bool(prior.enforce_star_ok_acceptance)) if class_aware_sampling_enabled else 0
    )
    method_class_token_bias_enabled = (
        int(bool(prior.class_token_logit_bias is not None)) if class_aware_sampling_enabled else 0
    )
    method_class_token_bias_strength = float(prior.class_token_bias_strength) if class_aware_sampling_enabled else 0.0
    method_metrics.update(
        {
            "run_name": str(run_cfg["run_name"]),
            "canonical_family": str(run_cfg["canonical_family"]),
            "class_aware_sampling_enabled": int(bool(class_aware_sampling_enabled)),
            "s0_target_condition_filter_enabled": int(bool(s0_target_condition_filter is not None)),
            "s0_target_temperature": (
                float(s0_target_condition_filter["target_temperature"])
                if s0_target_condition_filter is not None
                else None
            ),
            "s0_target_phi": (
                float(s0_target_condition_filter["target_phi"])
                if s0_target_condition_filter is not None
                else None
            ),
            "family_sampling_mode": method_family_sampling_mode,
            "family_sampling_scope": method_family_sampling_scope,
            "family_sampling_center_min_frac": method_center_min_frac,
            "family_sampling_center_max_frac": method_center_max_frac,
            "family_sampling_spans_per_sample": method_spans_per_sample,
            "backbone_template_min_gap_tokens": method_backbone_min_gap,
            "backbone_template_core_count": method_backbone_core_count,
            "backbone_template_max_core_token_length": method_backbone_max_core_length,
            "enforce_star_ok_acceptance": method_enforce_star_ok_acceptance,
            "class_token_bias_enabled": method_class_token_bias_enabled,
            "class_token_bias_strength": method_class_token_bias_strength,
            "mean_training_soluble_oracle_calls": (
                float(round_metrics_df["training_soluble_oracle_calls"].mean())
                if "training_soluble_oracle_calls" in round_metrics_df.columns and not round_metrics_df.empty
                else 0.0
            ),
            "mean_training_chi_oracle_calls": (
                float(round_metrics_df["training_chi_oracle_calls"].mean())
                if "training_chi_oracle_calls" in round_metrics_df.columns and not round_metrics_df.empty
                else 0.0
            ),
            "mean_class_match_acceptance_rate": (
                float(sampling_meta_df["class_match_acceptance_rate"].mean())
                if "class_match_acceptance_rate" in sampling_meta_df.columns and not sampling_meta_df.empty
                else 0.0
            ),
            "mean_class_match_oversampling_ratio": (
                float(sampling_meta_df["class_match_oversampling_ratio"].mean())
                if "class_match_oversampling_ratio" in sampling_meta_df.columns and not sampling_meta_df.empty
                else 0.0
            ),
            "mean_total_raw_samples_drawn": (
                float(sampling_meta_df["total_raw_samples_drawn"].mean())
                if "total_raw_samples_drawn" in sampling_meta_df.columns and not sampling_meta_df.empty
                else 0.0
            ),
            "model_setup_wall_time_seconds": float(model_setup_wall_time_seconds),
            "sampling_wall_time_seconds": float(sampling_wall_time_seconds),
            "evaluation_wall_time_seconds": float(evaluation_wall_time_seconds),
            "total_wall_time_seconds": float(time.perf_counter() - run_wall_start),
            "generated_samples": int(len(generated_samples_df)),
            "evaluated_samples": int(len(evaluation_results_df)),
            "target_rows": int(len(target_rows_df)),
            "generation_budget": int(generation_budget),
            "generation_budget_mode": generation_budget_mode,
            "round_generation_budgets": [int(value) for value in round_generation_budgets],
            "total_generation_budget_per_target": int(sum(round_generation_budgets)),
            "num_sampling_rounds": int(num_rounds),
        }
    )

    _write_frame(generated_samples_df, run_dirs["metrics_dir"] / "generated_samples.csv")
    _write_frame(evaluation_results_df, run_dirs["metrics_dir"] / "evaluation_results.csv")
    _write_frame(success_hit_polymers_df, run_dirs["metrics_dir"] / "success_hit_polymers.csv")
    _write_frame(round_metrics_df, run_dirs["metrics_dir"] / "round_metrics.csv")
    _write_frame(target_row_metrics_df, run_dirs["metrics_dir"] / "target_row_metrics.csv")
    _write_frame(target_row_summary_df, run_dirs["metrics_dir"] / "target_row_summary.csv")
    _write_frame(sampling_meta_df, run_dirs["metrics_dir"] / "sampling_metadata.csv")
    _write_json(method_metrics, run_dirs["metrics_dir"] / "method_metrics.json")
    if not sampling_meta_df.empty:
        _write_json(
            {
                "mean_class_match_acceptance_rate": float(sampling_meta_df["class_match_acceptance_rate"].mean())
                if "class_match_acceptance_rate" in sampling_meta_df.columns
                else 0.0,
                "mean_class_match_oversampling_ratio": float(sampling_meta_df["class_match_oversampling_ratio"].mean())
                if "class_match_oversampling_ratio" in sampling_meta_df.columns
                else 0.0,
                "class_aware_sampling_enabled": int(bool(class_aware_sampling_enabled)),
                "s0_target_condition_filter_enabled": int(bool(s0_target_condition_filter is not None)),
                "s0_target_temperature": (
                    float(s0_target_condition_filter["target_temperature"])
                    if s0_target_condition_filter is not None
                    else None
                ),
                "s0_target_phi": (
                    float(s0_target_condition_filter["target_phi"])
                    if s0_target_condition_filter is not None
                    else None
                ),
                "family_sampling_mode": method_family_sampling_mode,
                "family_sampling_scope": method_family_sampling_scope,
                "family_sampling_center_min_frac": method_center_min_frac,
                "family_sampling_center_max_frac": method_center_max_frac,
                "backbone_template_min_gap_tokens": method_backbone_min_gap,
                "backbone_template_max_core_token_length": method_backbone_max_core_length,
                "enforce_star_ok_acceptance": method_enforce_star_ok_acceptance,
                "mean_total_raw_samples_drawn": float(sampling_meta_df["total_raw_samples_drawn"].mean())
                if "total_raw_samples_drawn" in sampling_meta_df.columns
                else 0.0,
                "max_total_raw_samples_drawn": float(sampling_meta_df["total_raw_samples_drawn"].max())
                if "total_raw_samples_drawn" in sampling_meta_df.columns
                else 0.0,
            },
            run_dirs["metrics_dir"] / "sampling_metadata_summary.json",
        )

    if save_figures and not evaluation_results_df.empty:
        plot_success_gate_funnel(
            evaluation_results_df,
            run_dirs["figures_dir"] / "candidate_screening_funnel.png",
            font_size=int(resolved.step5["figure_font_size"]),
        )
        plot_generated_chi_vs_target(
            evaluation_results_df,
            run_dirs["figures_dir"] / "generated_chi_vs_target.png",
            font_size=int(resolved.step5["figure_font_size"]),
        )
    if save_figures and not target_row_summary_df.empty:
        plot_per_target_success(
            target_row_summary_df,
            run_dirs["figures_dir"] / "per_target_success_hit_rate.png",
            font_size=int(resolved.step5["figure_font_size"]),
        )

    save_artifact_manifest(
        step_dir=run_dirs["run_dir"],
        metrics_dir=run_dirs["metrics_dir"],
        figures_dir=run_dirs["figures_dir"],
    )
    append_log_message(
        run_dirs["run_dir"],
        (
            f"Run complete | run={run_cfg['run_name']} "
            f"total_generated={int(len(generated_samples_df))} "
            f"total_evaluated={int(len(evaluation_results_df))} "
            f"rounds_completed={int(num_rounds)}"
        ),
        echo=True,
    )
    return {
        "run_cfg": run_cfg,
        "run_dirs": run_dirs,
        "method_metrics": method_metrics,
        "evaluation_results_df": evaluation_results_df,
        "round_metrics_df": round_metrics_df,
        "target_row_summary_df": target_row_summary_df,
    }
