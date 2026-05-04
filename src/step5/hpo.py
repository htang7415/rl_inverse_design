"""Optuna HPO helpers for Step 5."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import shutil
import tempfile
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

from .config import _deep_merge, build_run_config, resolve_step5_hpo_generation_budget
from .plotting import plot_hpo_best_metric_curve, plot_hpo_best_success_curve
from .run_core import execute_step5_run
from .study_families import STUDY_BASE_RUNS
from src.utils.model_scales import get_model_config

try:
    import optuna
except ImportError:  # pragma: no cover - dependency is optional at import time
    optuna = None


def _require_optuna():
    if optuna is None:
        raise ImportError("Step 5 HPO requires optuna to be installed in the current environment.")


def _hpo_root(resolved) -> Path:
    root_dirname = str(resolved.step5_hpo.get("root_dirname", "step5_hpo")).strip() or "step5_hpo"
    return resolved.results_dir / root_dirname / resolved.split_mode / resolved.c_target


def _study_root(resolved, *, study_family: str) -> Path:
    return _hpo_root(resolved) / str(study_family)


def _best_params_path(resolved, *, study_family: str) -> Path:
    return _study_root(resolved, study_family=study_family) / "best_params.yaml"


def _resolve_hpo_runtime_config(resolved, *, study_family: str) -> Dict[str, Any]:
    hpo_cfg = dict(resolved.step5_hpo)
    runtime_overrides = dict(hpo_cfg.get("method_runtime_overrides", {}).get(study_family, {}) or {})
    if runtime_overrides:
        hpo_cfg.update(runtime_overrides)
    return hpo_cfg


def _resolve_optuna_timeout_seconds(budgets: Dict[str, Any]) -> int | None:
    if "timeout_hours_medium" not in budgets:
        raise KeyError(
            "Missing method_budgets.<study_family>.timeout_hours_medium for Step 5 HPO. "
            "Set it explicitly in configs/config5.yaml."
        )
    raw_timeout_hours = budgets.get("timeout_hours_medium")
    if raw_timeout_hours is None:
        raise ValueError(
            "Step 5 HPO timeout_hours_medium must be set explicitly per study family; null is not allowed."
        )
    timeout_hours = float(raw_timeout_hours)
    if not math.isfinite(timeout_hours) or timeout_hours <= 0.0:
        return None
    if timeout_hours > 30.0:
        raise ValueError(
            "Step 5 HPO timeout_hours_medium exceeds the supported 30-hour limit. "
            f"Received {timeout_hours}."
        )
    return max(1, int(round(timeout_hours * 3600.0)))


def _resolve_optional_positive_int(raw_value: Any) -> int | None:
    if raw_value in {None, "", "null"}:
        return None
    value = int(raw_value)
    return int(value) if value > 0 else None


def _resolve_optional_positive_float(raw_value: Any) -> float | None:
    if raw_value in {None, "", "null"}:
        return None
    value = float(raw_value)
    return float(value) if math.isfinite(value) and value > 0.0 else None


def _tighten_step5_decode_class_cap(
    step5_decode_cfg: Dict[str, Any],
    *,
    key: str,
    target_class: str,
    cap_value: int,
) -> None:
    overrides_key = f"{key}_overrides"
    current_cap = _resolve_optional_positive_int(step5_decode_cfg.get(key))
    step5_decode_cfg[key] = int(
        cap_value if current_cap is None else min(int(current_cap), int(cap_value))
    )
    class_overrides = dict(step5_decode_cfg.get(overrides_key, {}) or {})
    current_target_cap = _resolve_optional_positive_int(class_overrides.get(target_class))
    class_overrides[target_class] = int(
        cap_value if current_target_cap is None else min(int(current_target_cap), int(cap_value))
    )
    step5_decode_cfg[overrides_key] = class_overrides


def _build_optuna_pruner(resolved):
    pruner_name = str(resolved.step5_hpo.get("pruner", "median")).strip().lower()
    if pruner_name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(resolved.step5_hpo.get("pruner_n_startup_trials", 2)),
            n_warmup_steps=int(resolved.step5_hpo.get("pruner_n_warmup_steps", 1)),
        )
    if pruner_name == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=int(resolved.step5_hpo.get("pruner_min_resource", 1)),
            reduction_factor=int(resolved.step5_hpo.get("pruner_reduction_factor", 2)),
            min_early_stopping_rate=int(resolved.step5_hpo.get("pruner_min_early_stopping_rate", 0)),
        )
    if pruner_name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=int(resolved.step5_hpo.get("pruner_min_resource", 1)),
            max_resource=str(resolved.step5_hpo.get("pruner_max_resource", "auto")),
            reduction_factor=int(resolved.step5_hpo.get("pruner_reduction_factor", 2)),
        )
    raise ValueError(f"Unsupported Step 5 HPO pruner: {pruner_name}")


def _storage_uri(resolved, *, study_family: str | None = None) -> str:
    raw = str(resolved.step5_hpo.get("storage_uri", "")).strip()
    if not raw:
        default_root = _hpo_root(resolved) / study_family if study_family else _hpo_root(resolved)
        return f"sqlite:///{default_root / 'optuna.db'}"
    substituted = (
        raw.replace("<model_size>", str(resolved.model_size))
        .replace("<split_mode>", str(resolved.split_mode))
        .replace("<c_target>", str(resolved.c_target))
    )
    if study_family:
        substituted = substituted.replace("<study_family>", study_family)
        if "<study_family>" not in raw and substituted.startswith("sqlite:///") and substituted.endswith(".db"):
            db_path = Path(substituted[len("sqlite:///") :])
            substituted = f"sqlite:///{db_path.parent / study_family / db_path.name}"
    return substituted


def _backbone_num_layers(resolved) -> int:
    return int(get_model_config(resolved.model_size, resolved.base_config, model_type="sequence")["num_layers"])


def _finetune_last_layer_choices(resolved) -> list[object]:
    n_layers = _backbone_num_layers(resolved)
    anchors = {
        0,
        max(1, n_layers // 4),
        max(1, n_layers // 2),
        max(1, (3 * n_layers) // 4),
        int(n_layers),
    }
    ordered = sorted(int(value) for value in anchors)
    return ordered + ["full"]


def _update_s2_training_params(run_cfg: Dict[str, object], params: Dict[str, Any], *, prefix: str = "") -> None:
    lr_key = f"{prefix}learning_rate"
    dropout_key = f"{prefix}condition_dropout_rate"
    augmentation_key = f"{prefix}chi_target_augmentation_rate"
    mix_key = f"{prefix}train_batch_mix_d_chi"

    updates: Dict[str, Any] = {}
    if lr_key in params:
        updates["learning_rate"] = float(params[lr_key])
    if dropout_key in params:
        updates["condition_dropout_rate"] = float(params[dropout_key])
    if augmentation_key in params:
        updates["chi_target_augmentation_rate"] = float(params[augmentation_key])
    if mix_key in params:
        d_chi = float(params[mix_key])
        updates["train_batch_mix"] = {
            "d_chi": d_chi,
            "d_water": float(1.0 - d_chi),
        }
    if updates:
        run_cfg["s2"].update(updates)


def _update_s2_model_params(run_cfg: Dict[str, object], params: Dict[str, Any], *, prefix: str = "") -> None:
    variant_key = f"{prefix}variant"
    metric_key = f"{prefix}checkpoint_selection_metric"
    soluble_weight_key = f"{prefix}mt_aux_soluble_loss_weight"
    chi_weight_key = f"{prefix}mt_aux_chi_loss_weight"

    if variant_key in params:
        variant = str(params[variant_key])
        run_cfg["s2"]["variant"] = variant
        if variant != "mt":
            run_cfg["s2"]["checkpoint_selection_metric"] = "auto"
            return
    if metric_key in params:
        run_cfg["s2"]["checkpoint_selection_metric"] = str(params[metric_key])
    if soluble_weight_key in params or chi_weight_key in params:
        mt_aux = dict(run_cfg["s2"].get("mt_aux", {}))
        if soluble_weight_key in params:
            mt_aux["soluble_loss_weight"] = float(params[soluble_weight_key])
        if chi_weight_key in params:
            mt_aux["chi_loss_weight"] = float(params[chi_weight_key])
        run_cfg["s2"]["mt_aux"] = mt_aux


def _suggest_s2_training_params(trial, params: Dict[str, Any], *, prefix: str = "") -> None:
    params[f"{prefix}learning_rate"] = trial.suggest_float(
        f"{prefix}learning_rate", 1.0e-4, 6.0e-4, log=True
    )
    params[f"{prefix}condition_dropout_rate"] = trial.suggest_float(
        f"{prefix}condition_dropout_rate", 0.05, 0.20
    )
    params[f"{prefix}chi_target_augmentation_rate"] = trial.suggest_float(
        f"{prefix}chi_target_augmentation_rate", 0.0, 0.25
    )
    params[f"{prefix}train_batch_mix_d_chi"] = trial.suggest_float(
        f"{prefix}train_batch_mix_d_chi", 0.60, 0.90
    )


def _suggest_s2_model_params(trial, params: Dict[str, Any], *, prefix: str = "") -> None:
    params[f"{prefix}variant"] = trial.suggest_categorical(f"{prefix}variant", ["pure", "mt"])
    if str(params[f"{prefix}variant"]) == "mt":
        params[f"{prefix}checkpoint_selection_metric"] = trial.suggest_categorical(
            f"{prefix}checkpoint_selection_metric",
            ["val_total_loss", "val_aux_soluble_loss", "val_aux_chi_loss"],
        )
        params[f"{prefix}mt_aux_soluble_loss_weight"] = trial.suggest_float(
            f"{prefix}mt_aux_soluble_loss_weight",
            0.05,
            1.5,
            log=True,
        )
        params[f"{prefix}mt_aux_chi_loss_weight"] = trial.suggest_float(
            f"{prefix}mt_aux_chi_loss_weight",
            0.01,
            0.25,
            log=True,
        )


def _apply_hpo_runtime_overrides(
    resolved,
    run_cfg: Dict[str, object],
    *,
    study_family: str,
) -> Tuple[object, Dict[str, object]]:
    hpo_cfg = _resolve_hpo_runtime_config(resolved, study_family=study_family)
    resolved = replace(resolved, base_config=deepcopy(resolved.base_config))
    sampling_cfg = resolved.base_config.setdefault("sampling", {})
    chi_cfg = resolved.base_config.setdefault("chi_training", {})
    step5_decode_cfg = chi_cfg.setdefault("step5_inverse_design", {})
    target_class = str(resolved.c_target).strip().lower()
    s2_hpo_max_steps = int(hpo_cfg.get("hpo_s2_max_steps", 0) or 0)
    rl_hpo_num_steps = int(hpo_cfg.get("hpo_rl_num_steps", 0) or 0)
    hpo_sampling_batch_size = int(hpo_cfg.get("hpo_sampling_batch_size", 0) or 0)
    hpo_class_match_attempts_max = int(hpo_cfg.get("hpo_decode_class_match_sampling_attempts_max", 0) or 0)
    hpo_class_match_oversample_factor = float(
        hpo_cfg.get("hpo_decode_class_match_oversample_factor", 0.0) or 0.0
    )
    hpo_class_match_max_request_size = _resolve_optional_positive_int(
        hpo_cfg.get("hpo_decode_class_match_max_request_size")
    )
    hpo_class_match_max_total_raw_samples = _resolve_optional_positive_int(
        hpo_cfg.get("hpo_decode_class_match_max_total_raw_samples")
    )
    raw_hpo_partial_fill_ratio = hpo_cfg.get("hpo_decode_partial_quota_min_fill_ratio", None)
    hpo_partial_fill_ratio = (
        None
        if raw_hpo_partial_fill_ratio in {None, "", "null"}
        else float(raw_hpo_partial_fill_ratio)
    )
    hpo_s4_trajectories_per_batch = int(hpo_cfg.get("hpo_s4_trajectories_per_batch", 0) or 0)
    hpo_s4_rl_diffusion_num_steps = int(hpo_cfg.get("hpo_s4_rl_diffusion_num_steps", 0) or 0)
    hpo_s4_replay_batch_size = int(hpo_cfg.get("hpo_s4_replay_batch_size", 0) or 0)
    hpo_s2_val_checks = int(hpo_cfg.get("hpo_s2_val_checks", 0) or 0)
    hpo_rl_proxy_eval_checks = int(hpo_cfg.get("hpo_rl_proxy_eval_checks", 0) or 0)

    if hpo_sampling_batch_size > 0:
        current_batch_size = int(sampling_cfg.get("batch_size", hpo_sampling_batch_size))
        sampling_cfg["batch_size"] = int(min(current_batch_size, hpo_sampling_batch_size))

    if hpo_class_match_attempts_max > 0:
        step5_decode_cfg["decode_constraint_class_match_sampling_attempts_max"] = int(
            hpo_class_match_attempts_max
        )
        attempts_overrides = dict(
            step5_decode_cfg.get("decode_constraint_class_match_sampling_attempts_max_overrides", {}) or {}
        )
        attempts_overrides[target_class] = int(hpo_class_match_attempts_max)
        step5_decode_cfg["decode_constraint_class_match_sampling_attempts_max_overrides"] = attempts_overrides

    if hpo_class_match_oversample_factor > 0.0:
        step5_decode_cfg["decode_constraint_class_match_oversample_factor"] = float(
            hpo_class_match_oversample_factor
        )
        oversample_overrides = dict(
            step5_decode_cfg.get("decode_constraint_class_match_oversample_factor_overrides", {}) or {}
        )
        oversample_overrides[target_class] = float(hpo_class_match_oversample_factor)
        step5_decode_cfg["decode_constraint_class_match_oversample_factor_overrides"] = oversample_overrides

    if hpo_class_match_max_request_size is not None:
        _tighten_step5_decode_class_cap(
            step5_decode_cfg,
            key="decode_constraint_class_match_max_request_size",
            target_class=target_class,
            cap_value=int(hpo_class_match_max_request_size),
        )

    if hpo_class_match_max_total_raw_samples is not None:
        _tighten_step5_decode_class_cap(
            step5_decode_cfg,
            key="decode_constraint_class_match_max_total_raw_samples",
            target_class=target_class,
            cap_value=int(hpo_class_match_max_total_raw_samples),
        )

    if hpo_partial_fill_ratio is not None:
        step5_decode_cfg["decode_constraint_allow_partial_quota_return"] = True
        allow_partial_overrides = dict(
            step5_decode_cfg.get("decode_constraint_allow_partial_quota_return_overrides", {}) or {}
        )
        allow_partial_overrides[target_class] = True
        step5_decode_cfg["decode_constraint_allow_partial_quota_return_overrides"] = allow_partial_overrides
        current_fill_ratio = step5_decode_cfg.get("decode_constraint_partial_quota_min_fill_ratio", None)
        if current_fill_ratio in {None, "", "null"}:
            step5_decode_cfg["decode_constraint_partial_quota_min_fill_ratio"] = float(hpo_partial_fill_ratio)
        else:
            step5_decode_cfg["decode_constraint_partial_quota_min_fill_ratio"] = float(
                min(float(current_fill_ratio), float(hpo_partial_fill_ratio))
            )
        partial_fill_ratio_overrides = dict(
            step5_decode_cfg.get("decode_constraint_partial_quota_min_fill_ratio_overrides", {}) or {}
        )
        current_target_fill_ratio = partial_fill_ratio_overrides.get(target_class, None)
        if current_target_fill_ratio in {None, "", "null"}:
            partial_fill_ratio_overrides[target_class] = float(hpo_partial_fill_ratio)
        else:
            partial_fill_ratio_overrides[target_class] = float(
                min(float(current_target_fill_ratio), float(hpo_partial_fill_ratio))
            )
        step5_decode_cfg["decode_constraint_partial_quota_min_fill_ratio_overrides"] = (
            partial_fill_ratio_overrides
        )

    if "s2" in run_cfg and s2_hpo_max_steps > 0:
        capped_steps = min(int(run_cfg["s2"]["max_steps"]), int(s2_hpo_max_steps))
        run_cfg["s2"]["max_steps"] = int(capped_steps)
        target_s2_val_checks = int(hpo_s2_val_checks if hpo_s2_val_checks > 0 else 4)
        run_cfg["s2"]["val_check_interval_steps"] = int(
            min(
                int(run_cfg["s2"]["val_check_interval_steps"]),
                max(100, capped_steps // max(1, target_s2_val_checks)),
            )
        )

    if study_family in {"S4_rl", "S4_ppo", "S4_grpo"} and rl_hpo_num_steps > 0:
        capped_rl_steps = int(min(int(run_cfg["s4"]["rl_num_steps"]), int(rl_hpo_num_steps)))
        run_cfg["s4"]["rl_num_steps"] = int(capped_rl_steps)
        target_rl_proxy_eval_checks = int(
            hpo_rl_proxy_eval_checks if hpo_rl_proxy_eval_checks > 0 else 1
        )
        run_cfg["s4"]["rl_proxy_eval_interval_steps"] = int(
            min(
                int(run_cfg["s4"]["rl_proxy_eval_interval_steps"]),
                max(1, capped_rl_steps // max(1, target_rl_proxy_eval_checks)),
            )
        )
    if study_family in {"S4_rl", "S4_ppo", "S4_grpo"} and hpo_s4_trajectories_per_batch > 0:
        current_trajectories = int(run_cfg["s4"].get("trajectories_per_batch", hpo_s4_trajectories_per_batch))
        run_cfg["s4"]["trajectories_per_batch"] = int(min(current_trajectories, hpo_s4_trajectories_per_batch))
    if study_family in {"S4_rl", "S4_ppo", "S4_grpo"} and hpo_s4_rl_diffusion_num_steps > 0:
        current_diffusion_steps = int(run_cfg["s4"].get("rl_diffusion_num_steps", hpo_s4_rl_diffusion_num_steps))
        run_cfg["s4"]["rl_diffusion_num_steps"] = int(min(current_diffusion_steps, hpo_s4_rl_diffusion_num_steps))
    if study_family in {"S4_rl", "S4_ppo", "S4_grpo"} and hpo_s4_replay_batch_size > 0:
        current_replay_batch_size = int(run_cfg["s4"].get("replay_batch_size", hpo_s4_replay_batch_size))
        run_cfg["s4"]["replay_batch_size"] = int(min(current_replay_batch_size, hpo_s4_replay_batch_size))
    if study_family == "S4_dpo":
        dpo_cfg = dict(run_cfg["s4"].get("dpo", {}))
        raw_hpo_dpo_checkpoint_mode = hpo_cfg.get("hpo_s4_dpo_checkpoint_selection_mode", None)
        if raw_hpo_dpo_checkpoint_mode not in {None, "", "null"}:
            dpo_cfg["checkpoint_selection_mode"] = str(raw_hpo_dpo_checkpoint_mode).strip().lower()
        hpo_dpo_proxy_eval_interval_epochs = _resolve_optional_positive_int(
            hpo_cfg.get("hpo_s4_dpo_proxy_eval_interval_epochs")
        )
        if hpo_dpo_proxy_eval_interval_epochs is not None:
            dpo_cfg["proxy_eval_interval_epochs"] = int(hpo_dpo_proxy_eval_interval_epochs)
        run_cfg["s4"]["dpo"] = dpo_cfg

    return resolved, run_cfg


def _apply_trial_params(resolved, run_cfg: Dict[str, object], params: Dict[str, Any]):
    resolved_trial = resolved
    if "finetune_last_layers" in params:
        finetune_value = params["finetune_last_layers"]
        run_cfg["s2"]["finetune_last_layers"] = None if finetune_value == "full" else int(finetune_value)

    family = str(run_cfg["canonical_family"])
    if family == "S1":
        run_cfg["s1"].update(
            {
                "best_of_k": int(params["best_of_k"]),
                "guidance_start_frac": float(params["guidance_start_frac"]),
                "w_sol": float(params["w_sol"]),
                "w_chi": float(params["w_chi"]),
            }
        )
        if "sol_log_prob_floor" in params:
            run_cfg["s1"]["sol_log_prob_floor"] = float(params["sol_log_prob_floor"])
    elif family == "S2":
        _update_s2_model_params(run_cfg, params)
        _update_s2_training_params(run_cfg, params)
        run_cfg["s2"].update(
            {
                "cfg_scale": float(params["cfg_scale"]),
            }
        )
    elif family == "S3":
        _update_s2_model_params(run_cfg, params, prefix="s2_")
        _update_s2_training_params(run_cfg, params, prefix="s2_")
        run_cfg["s3"].update(
            {
                "cfg_scale": float(params["cfg_scale"]),
                "best_of_k": int(params["best_of_k"]),
                "guidance_start_frac": float(params["guidance_start_frac"]),
                "w_sol": float(params["w_sol"]),
                "w_chi": float(params["w_chi"]),
            }
        )
        if "sol_log_prob_floor" in params:
            run_cfg["s3"]["sol_log_prob_floor"] = float(params["sol_log_prob_floor"])
    elif family == "S4":
        alignment_mode = str(run_cfg["s4"]["alignment_mode"]).strip().lower()
        run_cfg["s4"]["cfg_scale"] = float(params["cfg_scale"])
        if "rl_prompt_source" in params:
            run_cfg["s4"]["rl_prompt_source"] = str(params["rl_prompt_source"])
        if alignment_mode in {"rl", "ppo", "grpo"}:
            reward_weights = deepcopy(run_cfg["s4"]["reward_weights"])
            reward_weights.update(
                {
                    "w_success": float(params["w_success"]),
                    "w_sol": float(params["w_sol"]),
                    "w_chi": float(params["w_chi"]),
                    "w_sa": float(params["w_sa"]),
                    "w_sa_continuous": float(params["w_sa_continuous"]),
                }
            )
            run_cfg["s4"]["reward_weights"] = reward_weights
            if "reward_curriculum_transition_frac" in params:
                reward_curriculum = deepcopy(run_cfg["s4"].get("reward_curriculum", {}))
                reward_curriculum["enabled"] = True
                reward_curriculum["transition_frac"] = float(params["reward_curriculum_transition_frac"])
                reward_curriculum["success_final_scale"] = float(params["reward_curriculum_success_final_scale"])
                reward_curriculum["dense_final_scale"] = float(params["reward_curriculum_dense_final_scale"])
                run_cfg["s4"]["reward_curriculum"] = reward_curriculum
            reward_shaping = deepcopy(run_cfg["s4"].get("reward_shaping", {}))
            if "reward_shaping_mode" in params:
                mode = str(params["reward_shaping_mode"])
                reward_shaping["solubility_term_mode"] = mode
            run_cfg["s4"]["reward_shaping"] = reward_shaping
            if "sol_log_prob_floor" in params:
                sol_log_prob_floor = float(params["sol_log_prob_floor"])
                run_cfg["s4"]["sol_log_prob_floor"] = sol_log_prob_floor
            run_cfg["s4"]["kl_weight"] = float(params["kl_weight"])
            run_cfg["s4"]["learning_rate"] = float(params["rl_learning_rate"])
            if alignment_mode in {"ppo", "grpo"}:
                run_cfg["s4"]["policy_update_epochs"] = int(params["policy_update_epochs"])
                run_cfg["s4"]["ppo_clip_eps"] = float(params["ppo_clip_eps"])
                run_cfg["s4"]["normalize_advantages"] = True
            if alignment_mode == "grpo":
                run_cfg["s4"]["grpo_group_size"] = int(params["grpo_group_size"])
        elif alignment_mode == "dpo":
            dpo_cfg = deepcopy(run_cfg["s4"]["dpo"])
            dpo_cfg.update(
                {
                    "offline_pair_budget": int(params["offline_pair_budget"]),
                    "beta": float(params["beta"]),
                    "learning_rate": float(params["learning_rate"]),
                    "batch_size": int(params["batch_size"]),
                    "num_epochs": int(params["num_epochs"]),
                }
            )
            if "d_water_budget_fraction" in params:
                dpo_cfg["d_water_budget_fraction"] = float(params["d_water_budget_fraction"])
            if "pair_source" in params:
                dpo_cfg["pair_source"] = str(params["pair_source"])
            if "synthetic_candidates_per_target" in params:
                dpo_cfg["synthetic_candidates_per_target"] = int(params["synthetic_candidates_per_target"])
            run_cfg["s4"]["dpo"] = dpo_cfg
        else:
            raise ValueError(f"Unsupported Step 5 HPO alignment mode: {alignment_mode}")
    return resolved_trial, run_cfg


def _hpo_choice_list(hpo_cfg: Dict[str, Any], key: str, default: list[Any]) -> list[Any]:
    raw_value = hpo_cfg.get(key, None)
    if raw_value is None or raw_value == "" or raw_value == "null":
        return list(default)
    choices = list(raw_value) if isinstance(raw_value, (list, tuple)) else [raw_value]
    if not choices:
        raise ValueError(f"step5_hpo.{key} must contain at least one choice when set.")
    return choices


def _grpo_group_size_choices(
    *,
    resolved,
    study_family: str,
    run_cfg: Dict[str, object],
) -> list[int]:
    hpo_cfg = _resolve_hpo_runtime_config(resolved, study_family=study_family)
    s4_cfg = dict(run_cfg.get("s4", {}) or {})
    trajectories_per_batch = int(
        hpo_cfg.get(
            "hpo_s4_trajectories_per_batch",
            s4_cfg.get("trajectories_per_batch", 8),
        )
        or 8
    )
    default_choices = [4, 8, 16]
    choices = [
        int(choice)
        for choice in default_choices
        if int(choice) <= int(trajectories_per_batch)
        and int(trajectories_per_batch) % int(choice) == 0
    ]
    return choices if choices else [max(1, int(trajectories_per_batch))]


def suggest_trial_params(trial, *, study_family: str, resolved, run_cfg: Dict[str, object]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    if study_family == "S1":
        params["best_of_k"] = trial.suggest_categorical("best_of_k", [2, 4])
        params["guidance_start_frac"] = trial.suggest_float("guidance_start_frac", 0.0, 0.85)
        params["w_sol"] = trial.suggest_float("w_sol", 0.25, 4.0, log=True)
        params["w_chi"] = trial.suggest_float("w_chi", 0.5, 4.0, log=True)
        params["sol_log_prob_floor"] = trial.suggest_float("sol_log_prob_floor", -12.0, -4.0)
    elif study_family == "S2":
        params["variant"] = trial.suggest_categorical("variant", ["pure", "mt"])
        if params["variant"] == "mt":
            params["checkpoint_selection_metric"] = trial.suggest_categorical(
                "checkpoint_selection_metric",
                ["val_total_loss", "val_aux_soluble_loss", "val_aux_chi_loss"],
            )
            params["mt_aux_soluble_loss_weight"] = trial.suggest_float(
                "mt_aux_soluble_loss_weight", 0.05, 1.5, log=True
            )
            params["mt_aux_chi_loss_weight"] = trial.suggest_float(
                "mt_aux_chi_loss_weight", 0.01, 0.25, log=True
            )
        params["finetune_last_layers"] = trial.suggest_categorical(
            "finetune_last_layers", _finetune_last_layer_choices(resolved)
        )
        _suggest_s2_training_params(trial, params)
        params["cfg_scale"] = trial.suggest_float("cfg_scale", 0.0, 3.0)
    elif study_family == "S3":
        _suggest_s2_model_params(trial, params, prefix="s2_")
        params["finetune_last_layers"] = trial.suggest_categorical(
            "finetune_last_layers", _finetune_last_layer_choices(resolved)
        )
        _suggest_s2_training_params(trial, params, prefix="s2_")
        params["cfg_scale"] = trial.suggest_float("cfg_scale", 0.0, 3.0)
        params["best_of_k"] = trial.suggest_categorical("best_of_k", [2, 4])
        params["guidance_start_frac"] = trial.suggest_float("guidance_start_frac", 0.0, 0.85)
        params["w_sol"] = trial.suggest_float("w_sol", 0.25, 4.0, log=True)
        params["w_chi"] = trial.suggest_float("w_chi", 0.5, 4.0, log=True)
        params["sol_log_prob_floor"] = trial.suggest_float("sol_log_prob_floor", -12.0, -4.0)
    elif study_family == "S4_rl":
        params["w_success"] = trial.suggest_float("w_success", 0.5, 4.0, log=True)
        params["w_sol"] = trial.suggest_float("w_sol", 0.25, 4.0, log=True)
        params["w_chi"] = trial.suggest_float("w_chi", 0.5, 4.0, log=True)
        params["w_sa"] = trial.suggest_float("w_sa", 0.0, 1.5)
        params["w_sa_continuous"] = trial.suggest_float("w_sa_continuous", 0.0, 2.0)
        params["kl_weight"] = trial.suggest_float("kl_weight", 1.0e-3, 5.0e-2, log=True)
        params["rl_learning_rate"] = trial.suggest_float("rl_learning_rate", 1.0e-6, 1.0e-4, log=True)
        params["cfg_scale"] = trial.suggest_float("cfg_scale", 0.0, 2.0)
        params["rl_prompt_source"] = trial.suggest_categorical(
            "rl_prompt_source",
            ["benchmark_target_rows", "train_exact_step3_distribution"],
        )
        params["reward_shaping_mode"] = trial.suggest_categorical(
            "reward_shaping_mode",
            ["log_prob", "logit_margin"],
        )
        params["sol_log_prob_floor"] = trial.suggest_float("sol_log_prob_floor", -12.0, -4.0)
        params["reward_curriculum_transition_frac"] = trial.suggest_float(
            "reward_curriculum_transition_frac", 0.2, 0.6
        )
        params["reward_curriculum_success_final_scale"] = trial.suggest_float(
            "reward_curriculum_success_final_scale", 1.0, 3.0
        )
        params["reward_curriculum_dense_final_scale"] = trial.suggest_float(
            "reward_curriculum_dense_final_scale", 0.25, 1.0
        )
    elif study_family == "S4_ppo":
        params["w_success"] = trial.suggest_float("w_success", 0.5, 4.0, log=True)
        params["w_sol"] = trial.suggest_float("w_sol", 0.25, 4.0, log=True)
        params["w_chi"] = trial.suggest_float("w_chi", 0.5, 4.0, log=True)
        params["w_sa"] = trial.suggest_float("w_sa", 0.0, 1.5)
        params["w_sa_continuous"] = trial.suggest_float("w_sa_continuous", 0.0, 2.0)
        params["kl_weight"] = trial.suggest_float("kl_weight", 1.0e-3, 5.0e-2, log=True)
        params["rl_learning_rate"] = trial.suggest_float("rl_learning_rate", 1.0e-6, 1.0e-4, log=True)
        params["cfg_scale"] = trial.suggest_float("cfg_scale", 0.0, 2.0)
        params["rl_prompt_source"] = trial.suggest_categorical(
            "rl_prompt_source",
            ["benchmark_target_rows", "train_exact_step3_distribution"],
        )
        params["reward_shaping_mode"] = trial.suggest_categorical(
            "reward_shaping_mode",
            ["log_prob", "logit_margin"],
        )
        params["sol_log_prob_floor"] = trial.suggest_float("sol_log_prob_floor", -12.0, -4.0)
        params["reward_curriculum_transition_frac"] = trial.suggest_float(
            "reward_curriculum_transition_frac", 0.2, 0.6
        )
        params["reward_curriculum_success_final_scale"] = trial.suggest_float(
            "reward_curriculum_success_final_scale", 1.0, 3.0
        )
        params["reward_curriculum_dense_final_scale"] = trial.suggest_float(
            "reward_curriculum_dense_final_scale", 0.25, 1.0
        )
        params["policy_update_epochs"] = trial.suggest_categorical("policy_update_epochs", [1, 2])
        params["ppo_clip_eps"] = trial.suggest_float("ppo_clip_eps", 0.10, 0.30)
    elif study_family == "S4_grpo":
        params["w_success"] = trial.suggest_float("w_success", 0.5, 4.0, log=True)
        params["w_sol"] = trial.suggest_float("w_sol", 0.25, 4.0, log=True)
        params["w_chi"] = trial.suggest_float("w_chi", 0.5, 4.0, log=True)
        params["w_sa"] = trial.suggest_float("w_sa", 0.0, 1.5)
        params["w_sa_continuous"] = trial.suggest_float("w_sa_continuous", 0.0, 2.0)
        params["kl_weight"] = trial.suggest_float("kl_weight", 1.0e-3, 5.0e-2, log=True)
        params["rl_learning_rate"] = trial.suggest_float("rl_learning_rate", 1.0e-6, 1.0e-4, log=True)
        params["cfg_scale"] = trial.suggest_float("cfg_scale", 0.0, 2.0)
        params["rl_prompt_source"] = trial.suggest_categorical(
            "rl_prompt_source",
            ["benchmark_target_rows", "train_exact_step3_distribution"],
        )
        params["reward_shaping_mode"] = trial.suggest_categorical(
            "reward_shaping_mode",
            ["log_prob", "logit_margin"],
        )
        params["sol_log_prob_floor"] = trial.suggest_float("sol_log_prob_floor", -12.0, -4.0)
        params["reward_curriculum_transition_frac"] = trial.suggest_float(
            "reward_curriculum_transition_frac", 0.2, 0.6
        )
        params["reward_curriculum_success_final_scale"] = trial.suggest_float(
            "reward_curriculum_success_final_scale", 1.0, 3.0
        )
        params["reward_curriculum_dense_final_scale"] = trial.suggest_float(
            "reward_curriculum_dense_final_scale", 0.25, 1.0
        )
        params["policy_update_epochs"] = trial.suggest_categorical("policy_update_epochs", [1, 2])
        params["ppo_clip_eps"] = trial.suggest_float("ppo_clip_eps", 0.10, 0.30)
        params["grpo_group_size"] = trial.suggest_categorical(
            "grpo_group_size",
            _grpo_group_size_choices(resolved=resolved, study_family=study_family, run_cfg=run_cfg),
        )
    elif study_family == "S4_dpo":
        dpo_hpo_cfg = _resolve_hpo_runtime_config(resolved, study_family=study_family)
        params["offline_pair_budget"] = trial.suggest_categorical(
            "offline_pair_budget",
            _hpo_choice_list(dpo_hpo_cfg, "hpo_s4_dpo_offline_pair_budget_choices", [5000, 10000]),
        )
        params["beta"] = trial.suggest_float("beta", 0.01, 0.2, log=True)
        params["learning_rate"] = trial.suggest_float("learning_rate", 1.0e-5, 1.0e-4, log=True)
        params["batch_size"] = trial.suggest_categorical(
            "batch_size",
            _hpo_choice_list(dpo_hpo_cfg, "hpo_s4_dpo_batch_size_choices", [32, 64]),
        )
        params["num_epochs"] = trial.suggest_categorical(
            "num_epochs",
            _hpo_choice_list(dpo_hpo_cfg, "hpo_s4_dpo_num_epochs_choices", [4, 6]),
        )
        dpo_cfg_scale_min = float(dpo_hpo_cfg.get("hpo_s4_dpo_cfg_scale_min", 0.5) or 0.5)
        if dpo_cfg_scale_min >= 2.0:
            raise ValueError(
                "step5_hpo.hpo_s4_dpo_cfg_scale_min must be < 2.0 for the current DPO search space."
            )
        params["cfg_scale"] = trial.suggest_float("cfg_scale", dpo_cfg_scale_min, 2.0)
        params["d_water_budget_fraction"] = trial.suggest_categorical(
            "d_water_budget_fraction",
            _hpo_choice_list(dpo_hpo_cfg, "hpo_s4_dpo_d_water_budget_fraction_choices", [0.2, 0.3, 0.5]),
        )
        params["pair_source"] = trial.suggest_categorical(
            "pair_source",
            _hpo_choice_list(
                dpo_hpo_cfg,
                "hpo_s4_dpo_pair_source_choices",
                ["chi_aware_plus_target_row_synthetic", "target_row_synthetic"],
            ),
        )
        pair_source = str(params["pair_source"]).strip().lower()
        if pair_source in {"target_row_synthetic", "chi_aware_plus_target_row_synthetic"}:
            params["synthetic_candidates_per_target"] = trial.suggest_categorical(
                "synthetic_candidates_per_target",
                _hpo_choice_list(dpo_hpo_cfg, "hpo_s4_dpo_synthetic_candidates_per_target_choices", [8, 16]),
            )
    else:
        raise ValueError(f"Unsupported Step 5 HPO study family: {study_family}")
    return params


def _trial_tie_break_key(trial, *, tie_epsilon: float) -> tuple:
    attrs = trial.user_attrs
    selection_objective = float(attrs.get("selection_objective_value", trial.value or float("-inf")))
    raw_objective = float(trial.value or float("-inf"))

    def _tie_bucket(value: float) -> float:
        if not math.isfinite(float(value)):
            return float("-inf")
        return float(round(float(value) / max(tie_epsilon, 1.0e-12)))

    return (
        _tie_bucket(selection_objective),
        _tie_bucket(raw_objective),
        float(attrs.get("mean_soluble_ok", float("-inf"))),
        float(attrs.get("mean_chi_ok", float("-inf"))),
        float(attrs.get("mean_chi_band_ok", float("-inf"))),
        -float(attrs.get("oracle_call_cost", float("inf"))),
        -float(attrs.get("wall_time_sec", float("inf"))),
    )


def _select_best_completed_trial(study, *, tie_epsilon: float):
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]
    if not completed:
        raise ValueError("No completed Optuna trials are available for Step 5 HPO.")
    effective_values = [
        float(trial.user_attrs.get("selection_objective_value", trial.value))
        for trial in completed
    ]
    best_value = max(effective_values)
    if not math.isfinite(best_value):
        contenders = list(completed)
    else:
        contenders = [
            trial
            for trial in completed
            if abs(float(trial.user_attrs.get("selection_objective_value", trial.value)) - best_value)
            <= float(tie_epsilon)
        ]
    contenders.sort(key=lambda trial: _trial_tie_break_key(trial, tie_epsilon=float(tie_epsilon)), reverse=True)
    return contenders[0]


def _compute_selection_objective_value(
    *,
    objective_value: float,
    mean_class_match_acceptance_rate: float,
    mean_total_raw_samples_drawn: float,
    hpo_cfg: Dict[str, Any],
) -> tuple[float, bool, list[str]]:
    selection_value = float(objective_value)
    notes: list[str] = []
    raw_min_acceptance = hpo_cfg.get("efficiency_min_class_match_acceptance_rate", None)
    if raw_min_acceptance not in {None, "", "null"}:
        min_acceptance = float(raw_min_acceptance)
        if min_acceptance > 0.0:
            if (
                not math.isfinite(float(mean_class_match_acceptance_rate))
                or float(mean_class_match_acceptance_rate) <= 0.0
            ):
                selection_value = float("-inf")
                notes.append("class_match_acceptance_rate_missing_or_nonpositive")
            elif float(mean_class_match_acceptance_rate) < float(min_acceptance):
                selection_value *= max(
                    0.0,
                    float(mean_class_match_acceptance_rate) / float(min_acceptance),
                )
                notes.append(
                    "class_match_acceptance_rate_below_floor:"
                    f"{float(mean_class_match_acceptance_rate):.6g}<{float(min_acceptance):.6g}"
                )
    raw_max_total = hpo_cfg.get("efficiency_max_mean_total_raw_samples_drawn", None)
    if raw_max_total not in {None, "", "null"}:
        max_total_raw_samples = float(raw_max_total)
        if (
            max_total_raw_samples > 0.0
            and math.isfinite(float(mean_total_raw_samples_drawn))
            and float(mean_total_raw_samples_drawn) > max_total_raw_samples
        ):
            selection_value *= max(
                0.0,
                float(max_total_raw_samples) / float(mean_total_raw_samples_drawn),
            )
            notes.append(
                "mean_total_raw_samples_drawn_above_ceiling:"
                f"{float(mean_total_raw_samples_drawn):.6g}>{float(max_total_raw_samples):.6g}"
            )
    return float(selection_value), len(notes) == 0, notes


def _trial_summary_path(study_root: Path, trial_number: int) -> Path:
    return study_root / "trials" / f"trial_{int(trial_number):04d}.json"


def _write_trial_summary(
    study_root: Path,
    *,
    trial_number: int,
    study_family: str,
    run_name: str,
    params: Dict[str, Any],
    objective_value: float | None,
    objective_metric: str,
    mean_property_success_hit_rate: float | None,
    mean_property_success_hit_rate_discovery: float | None,
    mean_success_hit_rate: float | None,
    mean_success_hit_rate_discovery: float | None,
    mean_class_match_acceptance_rate: float | None = None,
    mean_total_raw_samples_drawn: float | None = None,
    selection_objective_value: float | None = None,
    selection_efficiency_eligible: bool | None = None,
    selection_efficiency_notes: list[str] | None = None,
    wall_time_sec: float | None = None,
    model_setup_wall_time_sec: float | None = None,
    sampling_wall_time_sec: float | None = None,
    evaluation_wall_time_sec: float | None = None,
    generated_samples: int | None = None,
    target_rows: int | None = None,
    state: str = "UNKNOWN",
) -> None:
    payload = {
        "trial_number": int(trial_number),
        "study_family": str(study_family),
        "run_name": str(run_name),
        "state": str(state),
        "objective_metric": str(objective_metric),
        "objective_value": (float(objective_value) if objective_value is not None else None),
        "mean_property_success_hit_rate": (
            float(mean_property_success_hit_rate) if mean_property_success_hit_rate is not None else None
        ),
        "mean_property_success_hit_rate_discovery": (
            float(mean_property_success_hit_rate_discovery)
            if mean_property_success_hit_rate_discovery is not None
            else None
        ),
        "mean_success_hit_rate": (float(mean_success_hit_rate) if mean_success_hit_rate is not None else None),
        "mean_success_hit_rate_discovery": (
            float(mean_success_hit_rate_discovery) if mean_success_hit_rate_discovery is not None else None
        ),
        "mean_class_match_acceptance_rate": (
            float(mean_class_match_acceptance_rate) if mean_class_match_acceptance_rate is not None else None
        ),
        "mean_total_raw_samples_drawn": (
            float(mean_total_raw_samples_drawn) if mean_total_raw_samples_drawn is not None else None
        ),
        "selection_objective_value": (
            float(selection_objective_value) if selection_objective_value is not None else None
        ),
        "selection_efficiency_eligible": (
            bool(selection_efficiency_eligible)
            if selection_efficiency_eligible is not None
            else None
        ),
        "selection_efficiency_notes": (
            [str(note) for note in selection_efficiency_notes]
            if selection_efficiency_notes is not None
            else None
        ),
        "wall_time_sec": (float(wall_time_sec) if wall_time_sec is not None else None),
        "model_setup_wall_time_sec": (
            float(model_setup_wall_time_sec) if model_setup_wall_time_sec is not None else None
        ),
        "sampling_wall_time_sec": (
            float(sampling_wall_time_sec) if sampling_wall_time_sec is not None else None
        ),
        "evaluation_wall_time_sec": (
            float(evaluation_wall_time_sec) if evaluation_wall_time_sec is not None else None
        ),
        "generated_samples": (int(generated_samples) if generated_samples is not None else None),
        "target_rows": (int(target_rows) if target_rows is not None else None),
        "hyperparameters": dict(params),
    }
    trial_path = _trial_summary_path(study_root, int(trial_number))
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trial_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_search_space(
    path: Path,
    *,
    study_family: str,
    pair_source: str,
) -> None:
    payload = {
        "study_family": study_family,
        "pair_source": pair_source,
    }
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def run_optuna_study(
    *,
    resolved,
    study_family: str,
    config_path: str,
    base_config_path: str,
    model_size: str | None,
    device: str,
    refit_best: bool = True,
    fresh_study: bool = False,
) -> Dict[str, Any]:
    _require_optuna()
    if not bool(resolved.step5_hpo.get("enabled", False)):
        raise ValueError("step5_hpo.enabled is false; enable it before running Step 5 HPO.")
    if resolved.hpo_target_df.empty:
        raise ValueError("Resolved Step 5 HPO target table is empty.")
    if study_family not in STUDY_BASE_RUNS:
        raise ValueError(f"Unsupported Step 5 HPO study family: {study_family}")

    base_run_name = STUDY_BASE_RUNS[study_family]
    base_run_cfg = build_run_config(resolved, base_run_name)
    family_hpo_cfg = _resolve_hpo_runtime_config(resolved, study_family=study_family)
    budgets = dict(resolved.step5_hpo.get("method_budgets", {}).get(study_family, {}))
    n_trials = int(budgets.get("n_trials", 120))
    timeout_seconds = _resolve_optuna_timeout_seconds(budgets)
    default_startup_trials = int(resolved.step5_hpo.get("n_startup_trials", 20))
    effective_startup_trials = int(
        min(
            int(budgets.get("n_startup_trials", default_startup_trials)),
            max(1, n_trials - 1) if n_trials > 1 else 1,
        )
    )
    study_root = _study_root(resolved, study_family=study_family)
    if fresh_study and study_root.exists():
        shutil.rmtree(study_root)
    study_root.mkdir(parents=True, exist_ok=True)
    _write_search_space(
        study_root / "search_space.yaml",
        study_family=study_family,
        pair_source=str(base_run_cfg.get("s4", {}).get("dpo", {}).get("pair_source", "")),
    )
    resolved.hpo_target_df.to_csv(study_root / "d_hpo_family.csv", index=False)
    validation_diagnostics = (
        resolved.config_snapshot.get("derived", {}).get("validation_bucket_diagnostics", {})
        if isinstance(resolved.config_snapshot.get("derived", {}), dict)
        else {}
    )
    with open(study_root / "hpo_target_overlap_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(validation_diagnostics.get("hpo_target_overlap", {}), handle, indent=2)

    sampler_name = str(resolved.step5_hpo.get("sampler", "tpe")).strip().lower()
    if sampler_name != "tpe":
        raise ValueError(f"Unsupported Step 5 HPO sampler: {sampler_name}")

    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        seed=int(resolved.step5_hpo.get("sampler_seed", 42)),
        n_startup_trials=int(effective_startup_trials),
    )
    pruner = _build_optuna_pruner(resolved)
    study_name = f"step5_{study_family}_{resolved.c_target}_{resolved.model_size}"
    storage_uri = _storage_uri(resolved, study_family=study_family)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_uri,
        load_if_exists=not fresh_study,
    )

    existing_trial_count = int(len(study.trials))
    remaining_trials = max(0, int(n_trials) - existing_trial_count)
    hpo_generation_budget = int(resolve_step5_hpo_generation_budget(family_hpo_cfg, resolved.c_target))
    hpo_num_rounds = int(family_hpo_cfg["hpo_num_rounds"])
    hpo_sampling_seeds = [int(x) for x in family_hpo_cfg["hpo_sampling_seeds"]]
    tie_epsilon = float(resolved.step5_hpo.get("tie_epsilon", 1.0e-4))
    objective_metric_name = str(
        resolved.step5_hpo.get("objective_metric", "mean_success_hit_rate_discovery")
    ).strip()
    supported_objective_metrics = {
        "mean_success_hit_rate_discovery",
        "mean_success_hit_rate",
        "mean_property_success_hit_rate_discovery",
        "mean_property_success_hit_rate",
    }
    if objective_metric_name not in supported_objective_metrics:
        raise ValueError(
            "Unsupported step5_hpo.objective_metric="
            f"{objective_metric_name!r}. Choose from: {sorted(supported_objective_metrics)}"
        )
    share_s4_warm_start_in_hpo = bool(resolved.step5_hpo.get("reuse_shared_s4_warm_start", True))
    skip_disk_checkpoints_in_hpo = bool(resolved.step5_hpo.get("skip_disk_checkpoints", True))
    enable_sampling_pruning = bool(family_hpo_cfg.get("enable_sampling_pruning", True))
    trial_wall_time_limit_seconds = _resolve_optional_positive_float(
        family_hpo_cfg.get("trial_wall_time_limit_seconds", None)
    )
    stage_offsets = {
        "s2": 0,
        "warm_start": 1_000_000,
        "dpo": 2_000_000,
        "rl": 3_000_000,
        "sampling": 4_000_000,
    }

    def objective(trial) -> float:
        def pruning_callback(*, stage: str, step: int, value: float, metrics: Dict[str, Any]) -> None:
            if (
                trial_wall_time_limit_seconds is not None
                and time.time() - start > float(trial_wall_time_limit_seconds)
            ):
                raise optuna.TrialPruned(
                    f"Pruned Step 5 {study_family} trial {int(trial.number)} after "
                    f"{float(trial_wall_time_limit_seconds):.1f}s wall-time limit"
                )
            if not math.isfinite(float(value)):
                return
            trial.report(float(value), step=stage_offsets.get(str(stage), 0) + int(step))
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned Step 5 {study_family} trial {int(trial.number)} at stage={stage} step={int(step)}"
                )

        run_cfg = deepcopy(base_run_cfg)
        params = suggest_trial_params(trial, study_family=study_family, resolved=resolved, run_cfg=run_cfg)
        resolved_trial, run_cfg = _apply_trial_params(resolved, run_cfg, params)
        resolved_trial, run_cfg = _apply_hpo_runtime_overrides(resolved_trial, run_cfg, study_family=study_family)
        run_cfg["run_name"] = f"{base_run_cfg['run_name']}__trial_{int(trial.number):04d}"
        start = time.time()
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"step5_hpo_{study_family}_trial_{int(trial.number):04d}_"
            ) as tmp_dir:
                result = execute_step5_run(
                    resolved=resolved_trial,
                    run_name=run_cfg["run_name"],
                    run_cfg=run_cfg,
                    device=device,
                    config_path=config_path,
                    run_dir=Path(tmp_dir),
                    shared_evaluator=None,
                    target_rows_df=resolved_trial.hpo_target_df,
                    generation_budget=hpo_generation_budget,
                    sampling_seeds=hpo_sampling_seeds,
                    num_rounds=hpo_num_rounds,
                    save_figures=False,
                    pruning_callback=pruning_callback,
                    extra_context={
                        "base_config_path": base_config_path,
                        "model_size": model_size,
                        "study_family": study_family,
                        "trial_number": int(trial.number),
                        "hpo_mode": True,
                        "skip_disk_checkpoints": bool(skip_disk_checkpoints_in_hpo),
                        "hpo_objective_metric": objective_metric_name,
                        "hpo_enable_sampling_pruning": bool(enable_sampling_pruning),
                        "allow_shared_s4_warm_start": bool(
                            share_s4_warm_start_in_hpo and study_family.startswith("S4_")
                        ),
                        "trial_params": params,
                    },
                )
        except optuna.TrialPruned:
            _write_trial_summary(
                study_root,
                trial_number=int(trial.number),
                study_family=study_family,
                run_name=str(run_cfg["run_name"]),
                params=params,
                objective_value=None,
                objective_metric=objective_metric_name,
                mean_property_success_hit_rate=None,
                mean_property_success_hit_rate_discovery=None,
                mean_success_hit_rate=None,
                mean_success_hit_rate_discovery=None,
                mean_class_match_acceptance_rate=None,
                mean_total_raw_samples_drawn=None,
                selection_objective_value=None,
                selection_efficiency_eligible=None,
                selection_efficiency_notes=None,
                wall_time_sec=float(time.time() - start),
                state="PRUNED",
            )
            raise
        except Exception:
            _write_trial_summary(
                study_root,
                trial_number=int(trial.number),
                study_family=study_family,
                run_name=str(run_cfg["run_name"]),
                params=params,
                objective_value=None,
                objective_metric=objective_metric_name,
                mean_property_success_hit_rate=None,
                mean_property_success_hit_rate_discovery=None,
                mean_success_hit_rate=None,
                mean_success_hit_rate_discovery=None,
                mean_class_match_acceptance_rate=None,
                mean_total_raw_samples_drawn=None,
                selection_objective_value=None,
                selection_efficiency_eligible=None,
                selection_efficiency_notes=None,
                wall_time_sec=float(time.time() - start),
                state="FAIL",
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        wall_time = time.time() - start
        metrics = result["method_metrics"]
        eval_df = result["evaluation_results_df"]
        mean_property_success_hit_rate = float(metrics.get("mean_property_success_hit_rate", float("nan")))
        mean_property_success_hit_rate_discovery = float(
            metrics.get("mean_property_success_hit_rate_discovery", mean_property_success_hit_rate)
        )
        mean_success_hit_rate = float(metrics["mean_success_hit_rate"])
        mean_success_hit_rate_discovery = float(
            metrics.get("mean_success_hit_rate_discovery", mean_success_hit_rate)
        )
        if objective_metric_name not in metrics:
            warnings.warn(
                f"Step 5 HPO objective metric {objective_metric_name!r} is missing from method_metrics; "
                "recording this trial as -inf.",
                RuntimeWarning,
                stacklevel=2,
            )
            objective_value = float("nan")
        else:
            objective_value = float(metrics.get(objective_metric_name, float("nan")))
        if not math.isfinite(objective_value):
            warnings.warn(
                f"Step 5 HPO objective metric {objective_metric_name!r} is non-finite; "
                "recording this trial as -inf.",
                RuntimeWarning,
                stacklevel=2,
            )
            objective_value = float("-inf")
        trial.report(objective_value, step=9_999_999)
        mean_class_match_acceptance_rate = float(
            metrics.get("mean_class_match_acceptance_rate", float("nan"))
        )
        mean_total_raw_samples_drawn = float(
            metrics.get("mean_total_raw_samples_drawn", float("nan"))
        )
        model_setup_wall_time_sec = float(metrics.get("model_setup_wall_time_seconds", float("nan")))
        sampling_wall_time_sec = float(metrics.get("sampling_wall_time_seconds", float("nan")))
        evaluation_wall_time_sec = float(metrics.get("evaluation_wall_time_seconds", float("nan")))
        generated_samples = int(metrics.get("generated_samples", len(eval_df)))
        target_rows = int(metrics.get("target_rows", 0))
        selection_objective_value, selection_efficiency_eligible, selection_efficiency_notes = (
            _compute_selection_objective_value(
                objective_value=float(objective_value),
                mean_class_match_acceptance_rate=mean_class_match_acceptance_rate,
                mean_total_raw_samples_drawn=mean_total_raw_samples_drawn,
                hpo_cfg=family_hpo_cfg,
            )
        )
        trial.set_user_attr("mean_chi_ok", float(eval_df["chi_ok"].astype(float).mean()) if not eval_df.empty else float("nan"))
        trial.set_user_attr(
            "mean_chi_band_ok",
            float(eval_df["chi_band_ok"].astype(float).mean()) if "chi_band_ok" in eval_df.columns and not eval_df.empty else float("nan"),
        )
        trial.set_user_attr("mean_soluble_ok", float(eval_df["soluble_ok"].astype(float).mean()) if not eval_df.empty else float("nan"))
        trial.set_user_attr(
            "oracle_call_cost",
            float(metrics.get("mean_training_soluble_oracle_calls", 0.0)) + float(metrics.get("mean_training_chi_oracle_calls", 0.0)),
        )
        trial.set_user_attr("objective_metric", objective_metric_name)
        trial.set_user_attr("mean_property_success_hit_rate_reporting", mean_property_success_hit_rate)
        trial.set_user_attr("mean_property_success_hit_rate_discovery", mean_property_success_hit_rate_discovery)
        trial.set_user_attr("mean_success_hit_rate_reporting", mean_success_hit_rate)
        trial.set_user_attr("mean_success_hit_rate_discovery", mean_success_hit_rate_discovery)
        trial.set_user_attr(
            "mean_class_match_acceptance_rate",
            mean_class_match_acceptance_rate,
        )
        trial.set_user_attr(
            "mean_total_raw_samples_drawn",
            mean_total_raw_samples_drawn,
        )
        trial.set_user_attr("selection_objective_value", float(selection_objective_value))
        trial.set_user_attr("selection_efficiency_eligible", bool(selection_efficiency_eligible))
        trial.set_user_attr("selection_efficiency_notes", list(selection_efficiency_notes))
        trial.set_user_attr("wall_time_sec", float(wall_time))
        trial.set_user_attr("model_setup_wall_time_sec", model_setup_wall_time_sec)
        trial.set_user_attr("sampling_wall_time_sec", sampling_wall_time_sec)
        trial.set_user_attr("evaluation_wall_time_sec", evaluation_wall_time_sec)
        trial.set_user_attr("generated_samples", int(generated_samples))
        trial.set_user_attr("target_rows", int(target_rows))
        trial.set_user_attr("run_name", str(run_cfg["run_name"]))
        _write_trial_summary(
            study_root,
            trial_number=int(trial.number),
            study_family=study_family,
            run_name=str(run_cfg["run_name"]),
            params=params,
            objective_value=objective_value,
            objective_metric=objective_metric_name,
            mean_property_success_hit_rate=mean_property_success_hit_rate,
            mean_property_success_hit_rate_discovery=mean_property_success_hit_rate_discovery,
            mean_success_hit_rate=mean_success_hit_rate,
            mean_success_hit_rate_discovery=mean_success_hit_rate_discovery,
            mean_class_match_acceptance_rate=mean_class_match_acceptance_rate,
            mean_total_raw_samples_drawn=mean_total_raw_samples_drawn,
            selection_objective_value=float(selection_objective_value),
            selection_efficiency_eligible=bool(selection_efficiency_eligible),
            selection_efficiency_notes=list(selection_efficiency_notes),
            wall_time_sec=float(wall_time),
            model_setup_wall_time_sec=model_setup_wall_time_sec,
            sampling_wall_time_sec=sampling_wall_time_sec,
            evaluation_wall_time_sec=evaluation_wall_time_sec,
            generated_samples=int(generated_samples),
            target_rows=int(target_rows),
            state="COMPLETE",
        )
        return objective_value

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=timeout_seconds,
            catch=(RuntimeError, ValueError),
        )
    try:
        best_trial = _select_best_completed_trial(study, tie_epsilon=tie_epsilon)
    except ValueError:
        best_trial = None

    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    trials_df = trials_df.rename(columns={"number": "trial_number", "value": "objective_value"})
    attr_rows = [
        {
            "trial_number": int(trial.number),
            "objective_metric": str(trial.user_attrs.get("objective_metric", objective_metric_name)),
            "mean_property_success_hit_rate": trial.user_attrs.get("mean_property_success_hit_rate_reporting"),
            "mean_property_success_hit_rate_discovery": trial.user_attrs.get("mean_property_success_hit_rate_discovery"),
            "mean_success_hit_rate": trial.user_attrs.get("mean_success_hit_rate_reporting"),
            "mean_success_hit_rate_discovery": trial.user_attrs.get("mean_success_hit_rate_discovery"),
            "mean_class_match_acceptance_rate": trial.user_attrs.get("mean_class_match_acceptance_rate"),
            "mean_total_raw_samples_drawn": trial.user_attrs.get("mean_total_raw_samples_drawn"),
            "selection_objective_value": trial.user_attrs.get("selection_objective_value"),
            "selection_efficiency_eligible": trial.user_attrs.get("selection_efficiency_eligible"),
            "wall_time_sec": trial.user_attrs.get("wall_time_sec"),
            "model_setup_wall_time_sec": trial.user_attrs.get("model_setup_wall_time_sec"),
            "sampling_wall_time_sec": trial.user_attrs.get("sampling_wall_time_sec"),
            "evaluation_wall_time_sec": trial.user_attrs.get("evaluation_wall_time_sec"),
            "generated_samples": trial.user_attrs.get("generated_samples"),
            "target_rows": trial.user_attrs.get("target_rows"),
        }
        for trial in study.trials
    ]
    if attr_rows:
        trials_df = trials_df.merge(trials_df.__class__(attr_rows), on="trial_number", how="left")
    trials_df.to_csv(study_root / "trials.csv", index=False)
    plot_hpo_best_success_curve(
        trials_df,
        study_root / "figures" / "hpo_best_success_hit_rate.png",
    )
    plot_hpo_best_metric_curve(
        trials_df,
        study_root / "figures" / "hpo_best_property_success_hit_rate.png",
        metric_candidates=[
            "mean_property_success_hit_rate_discovery",
            "mean_property_success_hit_rate",
        ],
        output_column="best_property_success_hit_rate_so_far",
        ylabel="Best property success hit rate so far",
    )
    plot_hpo_best_metric_curve(
        trials_df,
        study_root / "figures" / "hpo_best_objective_value.png",
        metric_candidates=["objective_value"],
        output_column="best_objective_value_so_far",
        ylabel="Best objective value so far",
    )
    trial_records = [
        {
            "trial_number": int(trial.number),
            "state": str(trial.state.name),
            "objective_metric": str(trial.user_attrs.get("objective_metric", objective_metric_name)),
            "objective_value": (float(trial.value) if trial.value is not None else None),
            "mean_property_success_hit_rate": (
                float(trial.user_attrs["mean_property_success_hit_rate_reporting"])
                if "mean_property_success_hit_rate_reporting" in trial.user_attrs
                else None
            ),
            "mean_property_success_hit_rate_discovery": (
                float(trial.user_attrs["mean_property_success_hit_rate_discovery"])
                if "mean_property_success_hit_rate_discovery" in trial.user_attrs
                else None
            ),
            "mean_success_hit_rate": (
                float(trial.user_attrs["mean_success_hit_rate_reporting"])
                if "mean_success_hit_rate_reporting" in trial.user_attrs
                else None
            ),
            "mean_success_hit_rate_discovery": (
                float(trial.user_attrs["mean_success_hit_rate_discovery"])
                if "mean_success_hit_rate_discovery" in trial.user_attrs
                else None
            ),
            "mean_class_match_acceptance_rate": (
                float(trial.user_attrs["mean_class_match_acceptance_rate"])
                if "mean_class_match_acceptance_rate" in trial.user_attrs
                else None
            ),
            "mean_total_raw_samples_drawn": (
                float(trial.user_attrs["mean_total_raw_samples_drawn"])
                if "mean_total_raw_samples_drawn" in trial.user_attrs
                else None
            ),
            "selection_objective_value": (
                float(trial.user_attrs["selection_objective_value"])
                if "selection_objective_value" in trial.user_attrs
                else None
            ),
            "selection_efficiency_eligible": (
                bool(trial.user_attrs["selection_efficiency_eligible"])
                if "selection_efficiency_eligible" in trial.user_attrs
                else None
            ),
            "selection_efficiency_notes": (
                [str(note) for note in trial.user_attrs["selection_efficiency_notes"]]
                if "selection_efficiency_notes" in trial.user_attrs
                else None
            ),
            "wall_time_sec": (
                float(trial.user_attrs["wall_time_sec"])
                if "wall_time_sec" in trial.user_attrs
                else None
            ),
            "model_setup_wall_time_sec": (
                float(trial.user_attrs["model_setup_wall_time_sec"])
                if "model_setup_wall_time_sec" in trial.user_attrs
                else None
            ),
            "sampling_wall_time_sec": (
                float(trial.user_attrs["sampling_wall_time_sec"])
                if "sampling_wall_time_sec" in trial.user_attrs
                else None
            ),
            "evaluation_wall_time_sec": (
                float(trial.user_attrs["evaluation_wall_time_sec"])
                if "evaluation_wall_time_sec" in trial.user_attrs
                else None
            ),
            "generated_samples": (
                int(trial.user_attrs["generated_samples"])
                if "generated_samples" in trial.user_attrs
                else None
            ),
            "target_rows": (
                int(trial.user_attrs["target_rows"])
                if "target_rows" in trial.user_attrs
                else None
            ),
            "hyperparameters": dict(trial.params),
        }
        for trial in study.trials
    ]
    with open(study_root / "trials.json", "w", encoding="utf-8") as handle:
        json.dump(trial_records, handle, indent=2)
    if best_trial is not None:
        with open(study_root / "best_params.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "study_family": study_family,
                    "configured_n_trials": int(n_trials),
                    "effective_timeout_hours": (
                        float(timeout_seconds) / 3600.0 if timeout_seconds is not None else None
                    ),
                    "existing_trial_count_at_start": int(existing_trial_count),
                    "remaining_trials_executed": int(remaining_trials),
                    "objective_metric": objective_metric_name,
                    "best_trial_number": int(best_trial.number),
                    "best_objective_value": float(best_trial.value),
                    "best_selection_objective_value": float(
                        best_trial.user_attrs.get("selection_objective_value", best_trial.value)
                    ),
                    "selection_efficiency_eligible": bool(
                        best_trial.user_attrs.get("selection_efficiency_eligible", True)
                    ),
                    "selection_efficiency_notes": [
                        str(note)
                        for note in best_trial.user_attrs.get("selection_efficiency_notes", [])
                    ],
                    "best_mean_property_success_hit_rate": float(
                        best_trial.user_attrs.get("mean_property_success_hit_rate_reporting", best_trial.value)
                    ),
                    "best_mean_property_success_hit_rate_discovery": float(
                        best_trial.user_attrs.get("mean_property_success_hit_rate_discovery", best_trial.value)
                    ),
                    "best_mean_success_hit_rate": float(
                        best_trial.user_attrs.get("mean_success_hit_rate_reporting", best_trial.value)
                    ),
                    "best_mean_success_hit_rate_discovery": float(
                        best_trial.user_attrs.get("mean_success_hit_rate_discovery", best_trial.value)
                    ),
                    "best_mean_class_match_acceptance_rate": float(
                        best_trial.user_attrs.get("mean_class_match_acceptance_rate", float("nan"))
                    ),
                    "best_mean_total_raw_samples_drawn": float(
                        best_trial.user_attrs.get("mean_total_raw_samples_drawn", float("nan"))
                    ),
                    "best_wall_time_sec": float(
                        best_trial.user_attrs.get("wall_time_sec", float("nan"))
                    ),
                    "best_model_setup_wall_time_sec": float(
                        best_trial.user_attrs.get("model_setup_wall_time_sec", float("nan"))
                    ),
                    "best_sampling_wall_time_sec": float(
                        best_trial.user_attrs.get("sampling_wall_time_sec", float("nan"))
                    ),
                    "best_evaluation_wall_time_sec": float(
                        best_trial.user_attrs.get("evaluation_wall_time_sec", float("nan"))
                    ),
                    "best_generated_samples": int(
                        best_trial.user_attrs.get("generated_samples", 0)
                    ),
                    "best_target_rows": int(best_trial.user_attrs.get("target_rows", 0)),
                    "best_params": dict(best_trial.params),
                },
                handle,
                sort_keys=False,
            )

    refit_result = None
    if refit_best and best_trial is not None:
        tuned_run_cfg = deepcopy(base_run_cfg)
        resolved_refit, tuned_run_cfg = _apply_trial_params(resolved, tuned_run_cfg, dict(best_trial.params))
        tuned_run_cfg["run_name"] = f"{base_run_cfg['run_name']}_optuna"
        refit_run_dir = resolved_refit.method_root / tuned_run_cfg["run_name"]
        if fresh_study and refit_run_dir.exists():
            shutil.rmtree(refit_run_dir)
        refit_result = execute_step5_run(
            resolved=resolved_refit,
            run_name=tuned_run_cfg["run_name"],
            run_cfg=tuned_run_cfg,
            device=device,
            config_path=config_path,
            run_dir=refit_run_dir,
            shared_evaluator=None,
            target_rows_df=None,
            generation_budget=None,
            sampling_seeds=None,
            num_rounds=None,
            save_figures=True,
            extra_context={
                "base_config_path": base_config_path,
                "model_size": model_size,
                "study_family": study_family,
                "hpo_refit": True,
                "source_trial_number": int(best_trial.number),
                "source_trial_params": dict(best_trial.params),
            },
        )

    return {
        "study": study,
        "best_trial": best_trial,
        "study_root": study_root,
        "refit_result": refit_result,
    }


def refit_best_trial(
    *,
    resolved,
    study_family: str,
    config_path: str,
    base_config_path: str,
    model_size: str | None,
    device: str,
    fresh_refit: bool = False,
    use_hpo_runtime_caps: bool = False,
    run_cfg_override: Dict[str, Any] | None = None,
    generation_budget: int | None = None,
    sampling_seeds: List[int] | None = None,
    num_rounds: int | None = None,
    extra_context: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if study_family not in STUDY_BASE_RUNS:
        raise ValueError(f"Unsupported Step 5 HPO study family: {study_family}")

    best_params_path = _best_params_path(resolved, study_family=study_family)
    if not best_params_path.exists():
        return None

    with open(best_params_path, "r", encoding="utf-8") as handle:
        best_payload = yaml.safe_load(handle) or {}
    best_params = dict(best_payload.get("best_params", {}) or {})
    if not best_params:
        return None

    base_run_name = STUDY_BASE_RUNS[study_family]
    base_run_cfg = build_run_config(resolved, base_run_name)
    tuned_run_cfg = deepcopy(base_run_cfg)
    resolved_refit, tuned_run_cfg = _apply_trial_params(resolved, tuned_run_cfg, best_params)
    if use_hpo_runtime_caps:
        resolved_refit, tuned_run_cfg = _apply_hpo_runtime_overrides(
            resolved_refit,
            tuned_run_cfg,
            study_family=study_family,
        )
    if run_cfg_override:
        tuned_run_cfg = _deep_merge(tuned_run_cfg, deepcopy(run_cfg_override))
    tuned_run_cfg["run_name"] = f"{base_run_cfg['run_name']}_optuna"
    refit_run_dir = resolved_refit.method_root / tuned_run_cfg["run_name"]
    if fresh_refit and refit_run_dir.exists():
        shutil.rmtree(refit_run_dir)

    refit_context = {
        "base_config_path": base_config_path,
        "model_size": model_size,
        "study_family": study_family,
        "hpo_refit": True,
        "source_trial_number": best_payload.get("best_trial_number"),
        "source_trial_params": best_params,
        "source_best_params_path": str(best_params_path),
        "use_hpo_runtime_caps": bool(use_hpo_runtime_caps),
    }
    if extra_context:
        refit_context.update(extra_context)

    refit_result = execute_step5_run(
        resolved=resolved_refit,
        run_name=tuned_run_cfg["run_name"],
        run_cfg=tuned_run_cfg,
        device=device,
        config_path=config_path,
        run_dir=refit_run_dir,
        shared_evaluator=None,
        target_rows_df=None,
        generation_budget=generation_budget,
        sampling_seeds=sampling_seeds,
        num_rounds=num_rounds,
        save_figures=True,
        extra_context=refit_context,
    )
    return {
        "study_family": study_family,
        "best_params_path": best_params_path,
        "refit_result": refit_result,
    }
