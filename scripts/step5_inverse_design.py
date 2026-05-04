#!/usr/bin/env python
"""Step 5 inverse-design driver."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.step5.config import (
    _deep_merge,
    apply_step5_output_suffix,
    apply_step5_target_condition_filter,
    build_run_config,
    load_step5_config,
)
from src.step5.dataset import build_step5_split_leakage_audit
from src.step5.evaluation import load_step5_evaluator
from src.step5.run_core import execute_step5_run
from src.utils.config import as_yamlable


SUPPORTED_RUNS = {
    "S0_raw_unconditional",
    "S1_guided_frozen",
    "S2_conditional",
    "S2_cfg_0p0",
    "S2_cfg_2p0",
    "S2_mt",
    "S3_conditional_guided",
    "S3_cfg_2p0",
    "S4_dpo",
    "S4_rl_finetuned",
    "S4_ppo",
    "S4_grpo",
}


def _parse_sampling_seeds(value: str | None) -> List[int] | None:
    if value is None:
        return None
    seeds = [chunk.strip() for chunk in str(value).split(",") if chunk.strip()]
    if not seeds:
        raise ValueError("sampling_seeds must contain at least one integer when provided.")
    return [int(seed) for seed in seeds]


def _validate_positive_int(name: str, value: int | None) -> None:
    if value is not None and int(value) <= 0:
        raise ValueError(f"{name} must be >= 1 when provided.")


def _build_step5_override(args) -> Dict[str, object]:
    override: Dict[str, object] = {}

    if args.sampling_num_steps is not None:
        override["sampling_num_steps"] = int(args.sampling_num_steps)

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
    if args.dpo_num_epochs is not None:
        s4_override["dpo"] = {"num_epochs": int(args.dpo_num_epochs)}
    if s4_override:
        override["s4"] = s4_override

    return override


def _build_base_config_override(args) -> Dict[str, object]:
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

    return override


def _apply_cli_resolved_overrides(
    resolved,
    *,
    step5_override: Dict[str, object],
    base_config_override: Dict[str, object],
    method_root_suffix: str | None,
):
    if not step5_override and not base_config_override and not method_root_suffix:
        return resolved

    step5_cfg = deepcopy(resolved.step5)
    if step5_override:
        step5_cfg = _deep_merge(step5_cfg, step5_override)

    base_config = deepcopy(resolved.base_config)
    if base_config_override:
        base_config = _deep_merge(base_config, base_config_override)

    snapshot = deepcopy(resolved.config_snapshot)
    snapshot["step5"] = as_yamlable(step5_cfg)
    if base_config_override:
        snapshot["runtime_base_config_overrides"] = as_yamlable(base_config_override)

    updated = replace(
        resolved,
        base_config=base_config,
        step5=step5_cfg,
        config_snapshot=snapshot,
    )
    return apply_step5_output_suffix(updated, method_root_suffix=method_root_suffix)


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _prepare_shared_artifacts(resolved, *, config_path: str) -> None:
    shared_dir = resolved.method_root / "_shared"
    metrics_dir = shared_dir / "metrics"
    shared_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(shared_dir / "config_snapshot.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(as_yamlable(resolved.config_snapshot), handle, sort_keys=False)

    _write_frame(resolved.target_base_df, metrics_dir / "d_target_base.csv")
    _write_frame(resolved.target_family_df, metrics_dir / "d_target_family.csv")
    _write_frame(resolved.rl_proxy_df, metrics_dir / "rl_proxy_targets.csv")
    if not resolved.hpo_target_df.empty:
        _write_frame(resolved.hpo_target_df, metrics_dir / "d_hpo_family.csv")
    _write_frame(resolved.chi_split_df, metrics_dir / "d_chi_with_split.csv")
    validation_diagnostics = (
        resolved.config_snapshot.get("derived", {}).get("validation_bucket_diagnostics", {})
        if isinstance(resolved.config_snapshot.get("derived", {}), dict)
        else {}
    )
    _write_json(validation_diagnostics, metrics_dir / "validation_target_diagnostics.json")
    split_audit_df, split_audit_summary = build_step5_split_leakage_audit(resolved)
    _write_frame(split_audit_df, metrics_dir / "split_consistency_audit.csv")
    _write_json(split_audit_summary, metrics_dir / "split_consistency_audit_summary.json")
    if (
        bool(resolved.step5.get("fail_on_split_leakage", True))
        and int(split_audit_summary.get("train_eval_leakage_key_count", 0)) > 0
    ):
        raise ValueError(
            "Step 5 split consistency audit found cross-source train/eval leakage. "
            f"See {metrics_dir / 'split_consistency_audit.csv'}."
        )

    run_rows = [build_run_config(resolved, run_name) for run_name in resolved.enabled_runs]
    run_manifest = pd.DataFrame(
        [
            {
                "run_name": row["run_name"],
                "canonical_family": row["canonical_family"],
                "cfg_scale": row.get("s2", {}).get(
                    "cfg_scale",
                    row.get("s3", {}).get("cfg_scale", row.get("s4", {}).get("cfg_scale")),
                ),
                "alignment_mode": row.get("s4", {}).get("alignment_mode", ""),
            }
            for row in run_rows
        ]
    )
    _write_frame(run_manifest, metrics_dir / "enabled_runs.csv")
    _write_json(
        {
            "config_path": config_path,
            "c_target": resolved.c_target,
            "split_mode": resolved.split_mode,
            "classification_split_mode": resolved.classification_split_mode,
            "model_size": resolved.model_size,
            "num_target_rows": int(len(resolved.target_family_df)),
            "num_rl_proxy_rows": int(len(resolved.rl_proxy_df)),
            "num_hpo_rows": int(len(resolved.hpo_target_df)),
            "enabled_runs": resolved.enabled_runs,
            "method_root": str(resolved.method_root),
        },
        metrics_dir / "prepare_summary.json",
    )


def _resolve_requested_runs(resolved, runs_arg: str | None, allow_partial: bool) -> List[str]:
    if runs_arg:
        requested = [run.strip() for run in runs_arg.split(",") if run.strip()]
        unknown = [run for run in requested if run not in resolved.enabled_runs]
        if unknown:
            raise ValueError(f"Requested runs are not enabled in config5: {unknown}")
        selected = requested
    else:
        selected = list(resolved.enabled_runs)

    unsupported = [run for run in selected if run not in SUPPORTED_RUNS]
    if unsupported and not allow_partial:
        raise NotImplementedError(
            "This implementation increment supports only the currently wired Step 5 runs. "
            f"Unsupported selected runs: {unsupported}. "
            "Use --runs with a supported subset or pass --allow_partial."
        )
    if unsupported and allow_partial:
        print(f"Skipping unsupported runs in this increment: {unsupported}")
        selected = [run for run in selected if run in SUPPORTED_RUNS]
    if not selected:
        raise ValueError("No supported runs selected.")
    return selected


def _execute_run(
    *,
    resolved,
    run_name: str,
    device: str,
    config_path: str,
    shared_evaluator=None,
    run_dir=None,
    target_rows_df=None,
    generation_budget: int | None = None,
    sampling_seeds: List[int] | None = None,
    num_rounds: int | None = None,
    save_figures: bool = True,
    extra_context: Dict | None = None,
) -> None:
    execute_step5_run(
        resolved=resolved,
        run_name=run_name,
        device=device,
        config_path=config_path,
        shared_evaluator=shared_evaluator,
        run_dir=run_dir,
        target_rows_df=target_rows_df,
        generation_budget=generation_budget,
        sampling_seeds=sampling_seeds,
        num_rounds=num_rounds,
        save_figures=save_figures,
        extra_context=extra_context,
    )


def _run_requires_shared_evaluator(run_cfg: Dict[str, object]) -> bool:
    canonical_family = str(run_cfg.get("canonical_family", "")).strip()
    if canonical_family in {"S1", "S3"}:
        return True
    if canonical_family != "S4":
        return False
    alignment_mode = str(run_cfg.get("s4", {}).get("alignment_mode", "")).strip().lower()
    if alignment_mode in {"rl", "ppo", "grpo"}:
        return True
    if alignment_mode == "dpo":
        pair_source = str(run_cfg.get("s4", {}).get("dpo", {}).get("pair_source", "")).strip().lower()
        return pair_source in {"target_row_synthetic", "chi_aware_plus_target_row_synthetic"}
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 5 inverse design.")
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
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--runs", default=None, help="Comma-separated subset of enabled runs.")
    parser.add_argument("--allow_partial", action="store_true", help="Skip unsupported runs for development.")
    parser.add_argument("--generation_budget", type=int, default=None, help="Override samples per target row.")
    parser.add_argument("--num_rounds", type=int, default=None, help="Override number of sampling rounds.")
    parser.add_argument(
        "--target_temperature",
        type=float,
        default=None,
        help="Restrict Step 5 execution to this exact target temperature.",
    )
    parser.add_argument(
        "--target_phi",
        type=float,
        default=None,
        help="Restrict Step 5 execution to this exact target polymer fraction.",
    )
    parser.add_argument("--s2_max_steps", type=int, default=None, help="Override Step 5 S2 supervised max_steps.")
    parser.add_argument(
        "--s2_val_check_interval_steps",
        type=int,
        default=None,
        help="Override Step 5 S2 supervised val_check_interval_steps.",
    )
    parser.add_argument(
        "--s2_early_stopping_patience_checks",
        type=int,
        default=None,
        help="Override Step 5 S2 supervised early_stopping_patience_checks.",
    )
    parser.add_argument("--rl_num_steps", type=int, default=None, help="Override Step 5 S4 RL rl_num_steps.")
    parser.add_argument("--dpo_num_epochs", type=int, default=None, help="Override Step 5 S4 DPO num_epochs.")
    parser.add_argument("--sampling_batch_size", type=int, default=None, help="Override raw sampling batch size.")
    parser.add_argument("--sampling_num_steps", type=int, default=None, help="Override final sampling diffusion steps.")
    parser.add_argument(
        "--class_match_sampling_attempts_max",
        type=int,
        default=None,
        help="Override target-class constrained sampling retry count.",
    )
    parser.add_argument(
        "--class_match_oversample_factor",
        type=float,
        default=None,
        help="Override target-class constrained sampling oversample factor.",
    )
    parser.add_argument(
        "--class_match_max_request_size",
        type=int,
        default=None,
        help="Cap raw samples requested in any one class-match retry.",
    )
    parser.add_argument(
        "--class_match_max_total_raw_samples",
        type=int,
        default=None,
        help="Cap total raw samples drawn for one target-condition quota.",
    )
    parser.add_argument(
        "--partial_quota_min_fill_ratio",
        type=float,
        default=None,
        help="Allow partial target quotas once this fill ratio is reached.",
    )
    parser.add_argument(
        "--sampling_seeds",
        default=None,
        help="Comma-separated sampling seeds override used with --num_rounds or for deterministic smoke runs.",
    )
    parser.add_argument(
        "--max_target_rows",
        type=int,
        default=None,
        help="Limit execution to the first N target rows for fast smoke tests.",
    )
    parser.add_argument(
        "--skip_disk_checkpoints",
        action="store_true",
        help="Keep temporary checkpoints in memory only during testing.",
    )
    parser.add_argument(
        "--no_figures",
        action="store_true",
        help="Skip figure generation for faster smoke tests.",
    )
    parser.add_argument(
        "--run_dir_suffix",
        default=None,
        help="Optional suffix appended to each run directory name to isolate smoke outputs.",
    )
    parser.add_argument(
        "--method_root_suffix",
        default=None,
        help="Optional suffix appended to the Step 5 method root so shared and run outputs stay isolated.",
    )
    parser.add_argument(
        "--benchmark_root_suffix",
        dest="method_root_suffix",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    _validate_positive_int("generation_budget", args.generation_budget)
    _validate_positive_int("num_rounds", args.num_rounds)
    _validate_positive_int("s2_max_steps", args.s2_max_steps)
    _validate_positive_int("s2_val_check_interval_steps", args.s2_val_check_interval_steps)
    _validate_positive_int("s2_early_stopping_patience_checks", args.s2_early_stopping_patience_checks)
    _validate_positive_int("rl_num_steps", args.rl_num_steps)
    _validate_positive_int("dpo_num_epochs", args.dpo_num_epochs)
    _validate_positive_int("sampling_batch_size", args.sampling_batch_size)
    _validate_positive_int("sampling_num_steps", args.sampling_num_steps)
    _validate_positive_int("class_match_sampling_attempts_max", args.class_match_sampling_attempts_max)
    _validate_positive_int("class_match_max_request_size", args.class_match_max_request_size)
    _validate_positive_int("class_match_max_total_raw_samples", args.class_match_max_total_raw_samples)
    if (args.target_temperature is None) != (args.target_phi is None):
        raise ValueError("--target_temperature and --target_phi must be provided together.")
    if args.class_match_oversample_factor is not None and float(args.class_match_oversample_factor) <= 0.0:
        raise ValueError("class_match_oversample_factor must be > 0 when provided.")
    if args.partial_quota_min_fill_ratio is not None and not (0.0 < float(args.partial_quota_min_fill_ratio) <= 1.0):
        raise ValueError("partial_quota_min_fill_ratio must be in (0, 1] when provided.")

    resolved = load_step5_config(
        config_path=args.config,
        base_config_path=args.base_config,
        model_size=args.model_size,
        c_target_override=args.c_target,
    )
    resolved = apply_step5_target_condition_filter(
        resolved,
        target_temperature=args.target_temperature,
        target_phi=args.target_phi,
    )
    step5_override = _build_step5_override(args)
    base_config_override = _build_base_config_override(args)
    resolved = _apply_cli_resolved_overrides(
        resolved,
        step5_override=step5_override,
        base_config_override=base_config_override,
        method_root_suffix=args.method_root_suffix,
    )
    _prepare_shared_artifacts(resolved, config_path=args.config)

    print(f"Prepared Step 5 shared artifacts under: {resolved.method_root / '_shared'}")
    print(f"c_target={resolved.c_target} | num_target_rows={len(resolved.target_family_df)}")
    print("Enabled runs:")
    for run_name in resolved.enabled_runs:
        print(f"  - {run_name}")

    if args.prepare_only:
        return

    selected_runs = _resolve_requested_runs(resolved, args.runs, args.allow_partial)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sampling_seeds = _parse_sampling_seeds(args.sampling_seeds)
    if args.max_target_rows is not None and int(args.max_target_rows) <= 0:
        raise ValueError("max_target_rows must be >= 1 when provided.")
    target_rows_df = None
    if args.max_target_rows is not None:
        target_rows_df = resolved.target_family_df.head(int(args.max_target_rows)).copy()
        if target_rows_df.empty:
            raise ValueError("max_target_rows selected zero target rows.")

    extra_context = {}
    if args.skip_disk_checkpoints:
        extra_context["skip_disk_checkpoints"] = True
    if args.max_target_rows is not None:
        extra_context["max_target_rows"] = int(args.max_target_rows)
    if args.target_temperature is not None and args.target_phi is not None:
        extra_context["target_temperature"] = float(args.target_temperature)
        extra_context["target_phi"] = float(args.target_phi)
    if args.generation_budget is not None:
        extra_context["generation_budget_override"] = int(args.generation_budget)
    if args.num_rounds is not None:
        extra_context["num_rounds_override"] = int(args.num_rounds)
    if args.s2_max_steps is not None:
        extra_context["s2_max_steps_override"] = int(args.s2_max_steps)
    if args.s2_val_check_interval_steps is not None:
        extra_context["s2_val_check_interval_steps_override"] = int(args.s2_val_check_interval_steps)
    if args.s2_early_stopping_patience_checks is not None:
        extra_context["s2_early_stopping_patience_checks_override"] = int(args.s2_early_stopping_patience_checks)
    if args.rl_num_steps is not None:
        extra_context["rl_num_steps_override"] = int(args.rl_num_steps)
    if args.dpo_num_epochs is not None:
        extra_context["dpo_num_epochs_override"] = int(args.dpo_num_epochs)
    if args.sampling_batch_size is not None:
        extra_context["sampling_batch_size_override"] = int(args.sampling_batch_size)
    if args.sampling_num_steps is not None:
        extra_context["sampling_num_steps_override"] = int(args.sampling_num_steps)
    if args.class_match_sampling_attempts_max is not None:
        extra_context["class_match_sampling_attempts_max_override"] = int(args.class_match_sampling_attempts_max)
    if args.class_match_oversample_factor is not None:
        extra_context["class_match_oversample_factor_override"] = float(args.class_match_oversample_factor)
    if args.class_match_max_request_size is not None:
        extra_context["class_match_max_request_size_override"] = int(args.class_match_max_request_size)
    if args.class_match_max_total_raw_samples is not None:
        extra_context["class_match_max_total_raw_samples_override"] = int(args.class_match_max_total_raw_samples)
    if args.partial_quota_min_fill_ratio is not None:
        extra_context["partial_quota_min_fill_ratio_override"] = float(args.partial_quota_min_fill_ratio)
    if sampling_seeds is not None:
        extra_context["sampling_seeds_override"] = sampling_seeds
    if args.no_figures:
        extra_context["save_figures"] = False
    if args.run_dir_suffix:
        extra_context["run_dir_suffix"] = str(args.run_dir_suffix)
    if args.method_root_suffix:
        extra_context["method_root_suffix"] = str(args.method_root_suffix)
    extra_context = extra_context or None

    selected_run_cfgs = [build_run_config(resolved, run_name) for run_name in selected_runs]
    shared_evaluator = None
    if any(_run_requires_shared_evaluator(run_cfg) for run_cfg in selected_run_cfgs):
        shared_evaluator = load_step5_evaluator(resolved, device=device)

    for run_cfg in selected_run_cfgs:
        run_dir = None
        if args.run_dir_suffix:
            run_dir = resolved.method_root / f"{run_cfg['run_name']}{args.run_dir_suffix}"
        _execute_run(
            resolved=resolved,
            run_name=str(run_cfg["run_name"]),
            device=device,
            config_path=args.config,
            shared_evaluator=shared_evaluator,
            run_dir=run_dir,
            target_rows_df=target_rows_df,
            generation_budget=args.generation_budget,
            sampling_seeds=sampling_seeds,
            num_rounds=args.num_rounds,
            save_figures=not args.no_figures,
            extra_context=extra_context,
        )


if __name__ == "__main__":
    main()
