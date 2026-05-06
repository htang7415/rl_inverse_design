#!/usr/bin/env python
"""Refit Step 5 runs from previously saved Optuna best_params.yaml artifacts."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import re
import sys
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.step5.config import (
    _deep_merge,
    apply_step5_hpo_output_suffix,
    apply_step5_output_suffix,
    apply_step5_target_condition_filter,
    load_step5_config,
)
from src.step5.hpo import STUDY_BASE_RUNS, refit_best_trial


def _parse_sampling_seeds(value: str | None) -> List[int] | None:
    if value in {None, "", "null"}:
        return None
    seeds = [chunk.strip() for chunk in str(value).split(",") if chunk.strip()]
    if not seeds:
        raise ValueError("sampling_seeds must contain at least one integer when provided.")
    return [int(seed) for seed in seeds]


def _validate_positive_int(name: str, value: int | None) -> None:
    if value is not None and int(value) <= 0:
        raise ValueError(f"{name} must be >= 1 when provided.")


def _sanitize_condition_value(value: float) -> str:
    text = str(float(value))
    text = text.replace("+", "_").replace("-", "m").replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _target_condition_suffix(target_temperature: float, target_phi: float) -> str:
    return f"_T{_sanitize_condition_value(target_temperature)}_phi{_sanitize_condition_value(target_phi)}"


def _build_run_cfg_override(args) -> Dict[str, object]:
    override: Dict[str, object] = {}
    s1_override: Dict[str, object] = {}
    if args.s1_best_of_k is not None:
        s1_override["best_of_k"] = int(args.s1_best_of_k)
    if args.s1_guidance_start_frac is not None:
        s1_override["guidance_start_frac"] = float(args.s1_guidance_start_frac)
    if args.s1_w_sol is not None:
        s1_override["w_sol"] = float(args.s1_w_sol)
    if args.s1_w_chi is not None:
        s1_override["w_chi"] = float(args.s1_w_chi)
    if args.s1_w_sa is not None:
        s1_override["w_sa"] = float(args.s1_w_sa)
    if args.s1_w_sa_continuous is not None:
        s1_override["w_sa_continuous"] = float(args.s1_w_sa_continuous)
    if args.s1_sol_log_prob_floor is not None:
        s1_override["sol_log_prob_floor"] = float(args.s1_sol_log_prob_floor)
    if s1_override:
        override["s1"] = s1_override

    s2_override: Dict[str, object] = {}
    if args.s2_max_steps is not None:
        s2_override["max_steps"] = int(args.s2_max_steps)
    if args.s2_val_check_interval_steps is not None:
        s2_override["val_check_interval_steps"] = int(args.s2_val_check_interval_steps)
    if args.s2_early_stopping_patience_checks is not None:
        s2_override["early_stopping_patience_checks"] = int(args.s2_early_stopping_patience_checks)
    if s2_override:
        override["s2"] = s2_override

    s4_override: Dict[str, object] = {}
    if args.rl_num_steps is not None:
        s4_override["rl_num_steps"] = int(args.rl_num_steps)
    if args.rl_proxy_eval_interval_steps is not None:
        s4_override["rl_proxy_eval_interval_steps"] = int(args.rl_proxy_eval_interval_steps)
    if args.rl_proxy_num_targets is not None:
        s4_override["rl_proxy_num_targets"] = int(args.rl_proxy_num_targets)
    if args.rl_proxy_generation_budget is not None:
        s4_override["rl_proxy_generation_budget"] = int(args.rl_proxy_generation_budget)
    if args.trajectories_per_batch is not None:
        s4_override["trajectories_per_batch"] = int(args.trajectories_per_batch)
    if args.rl_diffusion_num_steps is not None:
        s4_override["rl_diffusion_num_steps"] = int(args.rl_diffusion_num_steps)
    if args.replay_batch_size is not None:
        s4_override["replay_batch_size"] = int(args.replay_batch_size)
    if args.s4_cfg_scale is not None:
        s4_override["cfg_scale"] = float(args.s4_cfg_scale)

    dpo_override: Dict[str, object] = {}
    if args.dpo_num_epochs is not None:
        dpo_override["num_epochs"] = int(args.dpo_num_epochs)
    if args.dpo_checkpoint_selection_mode is not None:
        dpo_override["checkpoint_selection_mode"] = str(args.dpo_checkpoint_selection_mode).strip().lower()
    if args.dpo_proxy_eval_interval_epochs is not None:
        dpo_override["proxy_eval_interval_epochs"] = int(args.dpo_proxy_eval_interval_epochs)
    if args.dpo_beta is not None:
        dpo_override["beta"] = float(args.dpo_beta)
    if args.dpo_pair_source is not None:
        dpo_override["pair_source"] = str(args.dpo_pair_source).strip().lower()
    if args.dpo_synthetic_candidates_per_target is not None:
        dpo_override["synthetic_candidates_per_target"] = int(args.dpo_synthetic_candidates_per_target)
    if dpo_override:
        s4_override["dpo"] = dpo_override
    if s4_override:
        override["s4"] = s4_override
    return override


def _apply_base_config_override(resolved, args):
    override: Dict[str, object] = {}
    if args.sampling_batch_size is not None:
        override.setdefault("sampling", {})["batch_size"] = int(args.sampling_batch_size)

    decode_override: Dict[str, object] = {}
    if args.class_match_sampling_attempts_max is not None:
        decode_override["decode_constraint_class_match_sampling_attempts_max"] = int(
            args.class_match_sampling_attempts_max
        )
    if args.class_match_oversample_factor is not None:
        decode_override["decode_constraint_class_match_oversample_factor"] = float(
            args.class_match_oversample_factor
        )
    if args.class_match_max_request_size is not None:
        decode_override["decode_constraint_class_match_max_request_size"] = int(
            args.class_match_max_request_size
        )
    if args.class_match_max_total_raw_samples is not None:
        decode_override["decode_constraint_class_match_max_total_raw_samples"] = int(
            args.class_match_max_total_raw_samples
        )
    if args.partial_quota_min_fill_ratio is not None:
        decode_override["decode_constraint_allow_partial_quota_return"] = True
        decode_override["decode_constraint_partial_quota_min_fill_ratio"] = float(
            args.partial_quota_min_fill_ratio
        )
    if decode_override:
        override.setdefault("chi_training", {})["step5_inverse_design"] = decode_override

    if not override:
        return resolved, {}

    from dataclasses import replace

    snapshot = deepcopy(resolved.config_snapshot)
    snapshot["runtime_base_config_overrides"] = override
    return replace(
        resolved,
        base_config=_deep_merge(resolved.base_config, override),
        config_snapshot=snapshot,
    ), override


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit Step 5 best Optuna trials.")
    parser.add_argument("--config", default="configs/config5.yaml")
    parser.add_argument("--base_config", default="configs/config.yaml")
    parser.add_argument("--model_size", default=None)
    parser.add_argument(
        "--c_target",
        "--polymer_family",
        dest="c_target",
        default=None,
        help="Override step5.c_target polymer-family target.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--study_families",
        default="S1,S2,S3,S4_rl,S4_ppo,S4_grpo,S4_dpo",
        help="Comma-separated study families to refit.",
    )
    parser.add_argument(
        "--force_enable",
        action="store_true",
        help="Force-enable step5_hpo for this invocation without editing configs/config5.yaml.",
    )
    parser.add_argument(
        "--fresh_refit",
        action="store_true",
        help="Delete existing tuned Step 5 method outputs before rerunning the refit.",
    )
    parser.add_argument(
        "--use_hpo_runtime_caps",
        action="store_true",
        help="Reuse the HPO trial runtime caps during best-trial refit.",
    )
    parser.add_argument(
        "--generation_budget",
        type=int,
        default=None,
        help=(
            "Override samples per target row. When "
            "step5.preserve_total_generation_across_rounds is true, this is distributed across rounds."
        ),
    )
    parser.add_argument("--num_rounds", type=int, default=None, help="Override number of sampling rounds.")
    parser.add_argument("--sampling_seeds", default=None, help="Comma-separated sampling seed override.")
    parser.add_argument("--target_temperature", type=float, default=None, help="Run one exact target temperature.")
    parser.add_argument("--target_phi", type=float, default=None, help="Run one exact target polymer fraction.")
    parser.add_argument("--method_root_suffix", default=None, help="Optional suffix for Step 5 output roots.")
    parser.add_argument(
        "--hpo_root_suffix",
        default=None,
        help="Optional suffix for the HPO c_target output root. Defaults to method_root_suffix or target condition.",
    )
    parser.add_argument("--s1_best_of_k", type=int, default=None)
    parser.add_argument("--s1_guidance_start_frac", type=float, default=None)
    parser.add_argument("--s1_w_sol", type=float, default=None)
    parser.add_argument("--s1_w_chi", type=float, default=None)
    parser.add_argument("--s1_w_sa", type=float, default=None)
    parser.add_argument("--s1_w_sa_continuous", type=float, default=None)
    parser.add_argument("--s1_sol_log_prob_floor", type=float, default=None)
    parser.add_argument("--s2_max_steps", type=int, default=None)
    parser.add_argument("--s2_val_check_interval_steps", type=int, default=None)
    parser.add_argument("--s2_early_stopping_patience_checks", type=int, default=None)
    parser.add_argument("--rl_num_steps", type=int, default=None)
    parser.add_argument("--rl_proxy_eval_interval_steps", type=int, default=None)
    parser.add_argument("--rl_proxy_num_targets", type=int, default=None)
    parser.add_argument("--rl_proxy_generation_budget", type=int, default=None)
    parser.add_argument("--trajectories_per_batch", type=int, default=None)
    parser.add_argument("--rl_diffusion_num_steps", type=int, default=None)
    parser.add_argument("--replay_batch_size", type=int, default=None)
    parser.add_argument("--s4_cfg_scale", type=float, default=None)
    parser.add_argument("--dpo_num_epochs", type=int, default=None)
    parser.add_argument(
        "--dpo_checkpoint_selection_mode",
        choices=["val_dpo_loss", "proxy_property_success_hit_rate"],
        default=None,
    )
    parser.add_argument("--dpo_proxy_eval_interval_epochs", type=int, default=None)
    parser.add_argument("--dpo_beta", type=float, default=None)
    parser.add_argument(
        "--dpo_pair_source",
        choices=[
            "label_water_miscibility",
            "chi_aware_label_bucketed",
            "target_row_synthetic",
            "chi_aware_plus_target_row_synthetic",
        ],
        default=None,
    )
    parser.add_argument("--dpo_synthetic_candidates_per_target", type=int, default=None)
    parser.add_argument("--sampling_batch_size", type=int, default=None)
    parser.add_argument("--class_match_sampling_attempts_max", type=int, default=None)
    parser.add_argument("--class_match_oversample_factor", type=float, default=None)
    parser.add_argument("--class_match_max_request_size", type=int, default=None)
    parser.add_argument("--class_match_max_total_raw_samples", type=int, default=None)
    parser.add_argument("--partial_quota_min_fill_ratio", type=float, default=None)
    args = parser.parse_args()

    _validate_positive_int("generation_budget", args.generation_budget)
    _validate_positive_int("num_rounds", args.num_rounds)
    _validate_positive_int("s1_best_of_k", args.s1_best_of_k)
    _validate_positive_int("s2_max_steps", args.s2_max_steps)
    _validate_positive_int("s2_val_check_interval_steps", args.s2_val_check_interval_steps)
    _validate_positive_int("s2_early_stopping_patience_checks", args.s2_early_stopping_patience_checks)
    _validate_positive_int("rl_num_steps", args.rl_num_steps)
    _validate_positive_int("rl_proxy_eval_interval_steps", args.rl_proxy_eval_interval_steps)
    _validate_positive_int("rl_proxy_num_targets", args.rl_proxy_num_targets)
    _validate_positive_int("rl_proxy_generation_budget", args.rl_proxy_generation_budget)
    _validate_positive_int("trajectories_per_batch", args.trajectories_per_batch)
    _validate_positive_int("rl_diffusion_num_steps", args.rl_diffusion_num_steps)
    _validate_positive_int("replay_batch_size", args.replay_batch_size)
    _validate_positive_int("dpo_num_epochs", args.dpo_num_epochs)
    _validate_positive_int("dpo_proxy_eval_interval_epochs", args.dpo_proxy_eval_interval_epochs)
    _validate_positive_int("dpo_synthetic_candidates_per_target", args.dpo_synthetic_candidates_per_target)
    _validate_positive_int("sampling_batch_size", args.sampling_batch_size)
    _validate_positive_int("class_match_sampling_attempts_max", args.class_match_sampling_attempts_max)
    _validate_positive_int("class_match_max_request_size", args.class_match_max_request_size)
    _validate_positive_int("class_match_max_total_raw_samples", args.class_match_max_total_raw_samples)
    if (args.target_temperature is None) != (args.target_phi is None):
        raise ValueError("--target_temperature and --target_phi must be provided together.")
    if args.s1_guidance_start_frac is not None and not (0.0 <= float(args.s1_guidance_start_frac) <= 1.0):
        raise ValueError("s1_guidance_start_frac must be in [0, 1] when provided.")
    for name in ("s1_w_sol", "s1_w_chi", "s1_w_sa", "s1_w_sa_continuous"):
        value = getattr(args, name)
        if value is not None and float(value) < 0.0:
            raise ValueError(f"{name} must be >= 0 when provided.")
    if args.class_match_oversample_factor is not None and float(args.class_match_oversample_factor) <= 0.0:
        raise ValueError("class_match_oversample_factor must be > 0 when provided.")
    if args.partial_quota_min_fill_ratio is not None and not (0.0 < float(args.partial_quota_min_fill_ratio) <= 1.0):
        raise ValueError("partial_quota_min_fill_ratio must be in (0, 1] when provided.")
    if args.s4_cfg_scale is not None and float(args.s4_cfg_scale) <= 0.0:
        raise ValueError("s4_cfg_scale must be > 0 when provided.")
    if args.dpo_beta is not None and float(args.dpo_beta) <= 0.0:
        raise ValueError("dpo_beta must be > 0 when provided.")

    resolved = load_step5_config(
        config_path=args.config,
        base_config_path=args.base_config,
        model_size=args.model_size,
        c_target_override=args.c_target,
        force_hpo_enabled=bool(args.force_enable),
    )
    resolved = apply_step5_target_condition_filter(
        resolved,
        target_temperature=args.target_temperature,
        target_phi=args.target_phi,
    )
    hpo_root_suffix = args.hpo_root_suffix
    if hpo_root_suffix is None:
        hpo_root_suffix = args.method_root_suffix
    if hpo_root_suffix is None and args.target_temperature is not None and args.target_phi is not None:
        hpo_root_suffix = _target_condition_suffix(args.target_temperature, args.target_phi)
    resolved = apply_step5_hpo_output_suffix(resolved, hpo_root_suffix=hpo_root_suffix)
    resolved = apply_step5_output_suffix(resolved, method_root_suffix=args.method_root_suffix)
    resolved, base_config_override = _apply_base_config_override(resolved, args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    requested = [item.strip() for item in str(args.study_families).split(",") if item.strip()]
    unknown = [item for item in requested if item not in STUDY_BASE_RUNS]
    if unknown:
        raise ValueError(f"Unknown Step 5 study families: {unknown}")

    sampling_seeds = _parse_sampling_seeds(args.sampling_seeds)
    run_cfg_override = _build_run_cfg_override(args)
    extra_context = {}
    if args.target_temperature is not None and args.target_phi is not None:
        extra_context["target_temperature"] = float(args.target_temperature)
        extra_context["target_phi"] = float(args.target_phi)
    if base_config_override:
        extra_context["base_config_runtime_overrides"] = base_config_override
    if run_cfg_override:
        extra_context["run_cfg_runtime_overrides"] = run_cfg_override
    if args.generation_budget is not None:
        extra_context["generation_budget_override"] = int(args.generation_budget)
    if args.num_rounds is not None:
        extra_context["num_rounds_override"] = int(args.num_rounds)
    if sampling_seeds is not None:
        extra_context["sampling_seeds_override"] = sampling_seeds
    if args.method_root_suffix:
        extra_context["method_root_suffix"] = str(args.method_root_suffix)

    for study_family in requested:
        result = refit_best_trial(
            resolved=resolved,
            study_family=study_family,
            config_path=args.config,
            base_config_path=args.base_config,
            model_size=args.model_size,
            device=device,
            fresh_refit=bool(args.fresh_refit),
            use_hpo_runtime_caps=bool(args.use_hpo_runtime_caps),
            run_cfg_override=run_cfg_override,
            generation_budget=args.generation_budget,
            sampling_seeds=sampling_seeds,
            num_rounds=args.num_rounds,
            extra_context=extra_context,
        )
        if result is None:
            print(f"[step5_hpo_refit] {study_family} skipped: no best_params.yaml with refittable params")
            continue
        print(
            f"[step5_hpo_refit] {study_family} refit_complete "
            f"best_params={result['best_params_path']}"
        )


if __name__ == "__main__":
    main()
