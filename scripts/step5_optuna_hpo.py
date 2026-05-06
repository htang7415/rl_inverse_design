#!/usr/bin/env python
"""Run Step 5 Optuna HPO studies."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.step5.config import (
    apply_step5_hpo_output_suffix,
    apply_step5_target_condition_filter,
    load_step5_config,
)
from src.step5.hpo import STUDY_BASE_RUNS, run_optuna_study


def _sanitize_condition_value(value: float) -> str:
    text = str(float(value))
    text = text.replace("+", "_").replace("-", "m").replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _target_condition_suffix(target_temperature: float, target_phi: float) -> str:
    return f"_T{_sanitize_condition_value(target_temperature)}_phi{_sanitize_condition_value(target_phi)}"


def _resolve_target_condition(args: argparse.Namespace, resolved) -> tuple[float | None, float | None]:
    if (args.target_temperature is None) != (args.target_phi is None):
        raise ValueError("--target_temperature and --target_phi must be provided together.")
    if args.target_temperature is not None and args.target_phi is not None:
        return float(args.target_temperature), float(args.target_phi)

    step5_inverse_cfg = (
        resolved.base_config.get("chi_training", {}).get("step5_inverse_design", {})
        if isinstance(resolved.base_config.get("chi_training", {}), dict)
        else {}
    )
    default_temperature = step5_inverse_cfg.get("target_temperature", None)
    default_phi = step5_inverse_cfg.get("target_phi", None)
    if default_temperature is None and default_phi is None:
        return None, None
    if default_temperature is None or default_phi is None:
        raise ValueError("base_config chi_training.step5_inverse_design target_temperature and target_phi must be set together.")
    return float(default_temperature), float(default_phi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 5 Optuna HPO.")
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
        help="Comma-separated study families to run.",
    )
    parser.add_argument("--skip_refit", action="store_true", help="Skip the best-trial full-budget refit.")
    parser.add_argument(
        "--force_enable",
        action="store_true",
        help="Force-enable step5_hpo for this invocation without editing configs/config5.yaml.",
    )
    parser.add_argument(
        "--fresh_study",
        action="store_true",
        help="Delete existing per-family Step 5 HPO study artifacts before rerunning.",
    )
    parser.add_argument("--target_temperature", type=float, default=None, help="HPO target temperature.")
    parser.add_argument("--target_phi", type=float, default=None, help="HPO target polymer volume fraction.")
    parser.add_argument(
        "--hpo_root_suffix",
        default=None,
        help="Optional suffix for the HPO c_target output root. Defaults to the target condition suffix.",
    )
    args = parser.parse_args()

    resolved = load_step5_config(
        config_path=args.config,
        base_config_path=args.base_config,
        model_size=args.model_size,
        c_target_override=args.c_target,
        force_hpo_enabled=bool(args.force_enable),
    )
    target_temperature, target_phi = _resolve_target_condition(args, resolved)
    hpo_root_suffix = args.hpo_root_suffix
    if hpo_root_suffix is None and target_temperature is not None and target_phi is not None:
        hpo_root_suffix = _target_condition_suffix(target_temperature, target_phi)
    resolved = apply_step5_target_condition_filter(
        resolved,
        target_temperature=target_temperature,
        target_phi=target_phi,
        align_hpo_target_df=True,
    )
    resolved = apply_step5_hpo_output_suffix(resolved, hpo_root_suffix=hpo_root_suffix)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    requested = [item.strip() for item in str(args.study_families).split(",") if item.strip()]
    unknown = [item for item in requested if item not in STUDY_BASE_RUNS]
    if unknown:
        raise ValueError(f"Unknown Step 5 study families: {unknown}")

    for study_family in requested:
        result = run_optuna_study(
            resolved=resolved,
            study_family=study_family,
            config_path=args.config,
            base_config_path=args.base_config,
            model_size=args.model_size,
            device=device,
            refit_best=not args.skip_refit,
            fresh_study=bool(args.fresh_study),
        )
        best_trial = result["best_trial"]
        if best_trial is None:
            print(f"[step5_hpo] {study_family} completed with no COMPLETE trials")
            continue
        print(f"[step5_hpo] {study_family} best_trial={int(best_trial.number)} best_value={float(best_trial.value):.6f}")


if __name__ == "__main__":
    main()
