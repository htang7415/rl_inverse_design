#!/usr/bin/env python
"""Compare Step 5 inverse-design runs for one target class."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.step5.config import apply_step5_output_suffix, load_step5_config
from src.step5.study_families import STUDY_BASE_RUNS
from src.step5.plotting import (
    plot_alignment_training_curves,
    plot_chi_vs_target_compare,
    plot_dpo_training_curves,
    plot_overall_success_all_runs,
    plot_overall_success_by_family,
    plot_per_target_difficulty_ranked,
    plot_per_target_success_compare,
    plot_supervised_training_curves,
    plot_success_gate_funnel_compare,
    plot_success_vs_oracle_budget,
)
from src.utils.config import as_yamlable
from src.utils.reporting import save_artifact_manifest, write_initial_log


REQUIRED_RUN_FILES = [
    "metrics/method_metrics.json",
    "metrics/round_metrics.csv",
    "metrics/target_row_summary.csv",
    "metrics/evaluation_results.csv",
]
OPTIONAL_RUN_FILES = {
    "sampling_metadata_df": "metrics/sampling_metadata.csv",
    "supervised_history_df": "metrics/supervised_training_history.csv",
    "rl_history_df": "metrics/rl_training_history.csv",
    "ppo_history_df": "metrics/ppo_training_history.csv",
    "grpo_history_df": "metrics/grpo_training_history.csv",
    "dpo_history_df": "metrics/dpo_training_history.csv",
}

COMPARE_METRIC_REGISTRY = {
    "full_discovery_success_hit_rate": {
        "method_mean": "mean_success_hit_rate_discovery",
        "method_std": "std_success_hit_rate_discovery",
        "method_macro": "macro_average_row_mean_success_hit_rate_discovery",
        "target_mean": "mean_success_hit_discovery_rate",
        "target_std": "std_success_hit_discovery_rate",
        "label": "Full discovery success hit rate",
    },
    "full_reporting_success_hit_rate": {
        "method_mean": "mean_success_hit_rate",
        "method_std": "std_success_hit_rate",
        "method_macro": "macro_average_row_mean_success_hit_rate",
        "target_mean": "mean_success_hit_rate",
        "target_std": "std_success_hit_rate",
        "label": "Full reporting success hit rate",
    },
    "property_discovery_success_hit_rate": {
        "method_mean": "mean_property_success_hit_rate_discovery",
        "method_std": "std_property_success_hit_rate_discovery",
        "method_macro": "macro_average_row_mean_property_success_hit_rate_discovery",
        "target_mean": "mean_property_success_hit_discovery_rate",
        "target_std": "std_property_success_hit_discovery_rate",
        "label": "Property discovery success hit rate",
    },
    "property_reporting_success_hit_rate": {
        "method_mean": "mean_property_success_hit_rate",
        "method_std": "std_property_success_hit_rate",
        "method_macro": "macro_average_row_mean_property_success_hit_rate",
        "target_mean": "mean_property_success_hit_rate",
        "target_std": "std_property_success_hit_rate",
        "label": "Property reporting success hit rate",
    },
}


def _resolve_selected_runs(resolved, runs_arg: str | None, *, allow_partial: bool = False) -> List[str]:
    existing_run_dirs = {
        path.name
        for path in resolved.method_root.iterdir()
        if path.is_dir() and path.name != "_shared"
    } if resolved.method_root.exists() else set()
    if runs_arg:
        requested = [item.strip() for item in str(runs_arg).split(",") if item.strip()]
        allowed = set(resolved.enabled_runs) | existing_run_dirs
        unknown = [run for run in requested if run not in allowed]
        if unknown:
            if allow_partial:
                return requested
            raise ValueError(
                "Requested Step 5_1 runs are neither enabled in config5 nor present under the "
                f"Step 5 method root: {unknown}"
            )
        return requested
    if bool(resolved.step5_1.get("compare_all_enabled_runs", True)):
        if bool(resolved.step5_hpo.get("enabled", False)):
            preferred_runs: List[str] = []
            if "S0_raw_unconditional" in resolved.enabled_runs and "S0_raw_unconditional" in existing_run_dirs:
                preferred_runs.append("S0_raw_unconditional")
            for base_run_name in STUDY_BASE_RUNS.values():
                tuned_name = f"{base_run_name}_optuna"
                if tuned_name in existing_run_dirs:
                    preferred_runs.append(tuned_name)
            if preferred_runs:
                return preferred_runs
        return [run for run in resolved.enabled_runs if run in existing_run_dirs]
    return [run for run in resolved.enabled_runs if (resolved.method_root / run).exists()]


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _base_run_label(run_name: str, canonical_family: str) -> str:
    name = str(run_name)
    if name.startswith("S0"):
        return "S0"
    if name.startswith("S1"):
        return "S1"
    if "S4_rl" in name:
        return "S4r"
    if "S4_ppo" in name:
        return "S4p"
    if "S4_grpo" in name:
        return "S4g"
    if "S4_dpo" in name:
        return "S4d"
    if name.startswith("S3"):
        return "S3"
    if name.startswith("S2"):
        return "S2"
    return str(canonical_family) or "Run"


def _build_run_label_map(run_comparison_df: pd.DataFrame) -> pd.DataFrame:
    if run_comparison_df.empty:
        return pd.DataFrame(columns=["run_name", "canonical_family", "run_label"])
    rows = []
    for _, row in run_comparison_df.sort_values(["canonical_family", "run_name"], kind="mergesort").iterrows():
        rows.append(
            {
                "run_name": str(row["run_name"]),
                "canonical_family": str(row["canonical_family"]),
                "base_label": _base_run_label(str(row["run_name"]), str(row["canonical_family"])),
            }
        )
    label_df = pd.DataFrame(rows)
    counts = label_df["base_label"].value_counts().to_dict()
    seen: Dict[str, int] = {}
    labels: List[str] = []
    for base_label in label_df["base_label"].astype(str).tolist():
        seen[base_label] = int(seen.get(base_label, 0)) + 1
        if int(counts.get(base_label, 0)) > 1:
            suffix = chr(ord("a") + seen[base_label] - 1)
            labels.append(f"{base_label}{suffix}")
        else:
            labels.append(base_label)
    label_df["run_label"] = labels
    return label_df[["run_name", "canonical_family", "run_label"]].copy()


def _attach_run_labels(df: pd.DataFrame, run_label_map_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or run_label_map_df.empty or "run_name" not in df.columns:
        return df
    return df.drop(columns=["run_label"], errors="ignore").merge(
        run_label_map_df[["run_name", "run_label"]],
        on="run_name",
        how="left",
    )


def _build_target_label_map(per_target_run_df: pd.DataFrame) -> pd.DataFrame:
    if per_target_run_df.empty:
        return pd.DataFrame(columns=["target_row_id", "target_row_key", "target_label"])
    target_cols = [
        col
        for col in ["target_row_id", "target_row_key", "c_target", "temperature", "phi", "chi_target"]
        if col in per_target_run_df.columns
    ]
    target_df = per_target_run_df[target_cols].drop_duplicates("target_row_id")
    sort_cols = [col for col in ["temperature", "phi", "target_row_id"] if col in target_df.columns]
    target_df = target_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    width = max(2, len(str(max(1, len(target_df)))))
    target_df["target_label"] = [f"T{idx + 1:0{width}d}" for idx in range(len(target_df))]
    return target_df


def _attach_target_labels(df: pd.DataFrame, target_label_map_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or target_label_map_df.empty or "target_row_id" not in df.columns:
        return df
    return df.drop(columns=["target_label"], errors="ignore").merge(
        target_label_map_df[["target_row_id", "target_label"]],
        on="target_row_id",
        how="left",
    )


def _attach_best_run_labels(canonical_family_df: pd.DataFrame, run_label_map_df: pd.DataFrame) -> pd.DataFrame:
    if canonical_family_df.empty or run_label_map_df.empty:
        return canonical_family_df
    best_label_map = run_label_map_df[["run_name", "run_label"]].rename(
        columns={"run_name": "best_run_name", "run_label": "best_run_label"}
    )
    return canonical_family_df.drop(columns=["best_run_label"], errors="ignore").merge(
        best_label_map,
        on="best_run_name",
        how="left",
    )


def _collect_evaluation_compare_df(
    run_payloads: List[Dict[str, object]],
    run_label_map_df: pd.DataFrame,
    target_label_map_df: pd.DataFrame,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    keep_cols = [
        "run_name",
        "canonical_family",
        "target_row_id",
        "target_row_key",
        "c_target",
        "temperature",
        "phi",
        "chi_target",
        "chi_pred_target",
        "success_hit",
        "property_success_hit",
    ]
    for payload in run_payloads:
        df = payload["evaluation_results_df"]
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        sub = df[[col for col in keep_cols if col in df.columns]].copy()
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = _attach_run_labels(out, run_label_map_df)
    out = _attach_target_labels(out, target_label_map_df)
    return out


def _collect_history_df(
    run_payloads: List[Dict[str, object]],
    run_label_map_df: pd.DataFrame,
    history_keys: List[str],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    label_map = run_label_map_df.set_index("run_name")["run_label"].to_dict() if not run_label_map_df.empty else {}
    for payload in run_payloads:
        run_name = str(payload["run_name"])
        run_label = str(label_map.get(run_name, run_name))
        for key in history_keys:
            df = payload.get(key, pd.DataFrame())
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            sub = df.copy()
            sub["run_name"] = run_name
            sub["run_label"] = run_label
            frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _resolve_compare_metric(resolved, run_payloads: List[Dict[str, object]]) -> Tuple[str, Dict[str, str]]:
    requested = str(resolved.step5_1.get("compare_metric", "full_discovery_success_hit_rate")).strip().lower()
    metric = COMPARE_METRIC_REGISTRY.get(requested)
    if metric is None:
        raise ValueError(
            f"Unsupported step5_1.compare_metric={requested!r}. "
            f"Choose from: {sorted(COMPARE_METRIC_REGISTRY.keys())}"
        )

    def _payload_has_metric(payload: Dict[str, object], spec: Dict[str, str]) -> bool:
        method_metrics = dict(payload["method_metrics"])
        return spec["method_mean"] in method_metrics and spec["method_std"] in method_metrics

    if run_payloads and all(_payload_has_metric(payload, metric) for payload in run_payloads):
        return requested, metric

    fallback_name = "full_reporting_success_hit_rate"
    if not all(_payload_has_metric(payload, COMPARE_METRIC_REGISTRY[fallback_name]) for payload in run_payloads):
        fallback_name = "property_reporting_success_hit_rate"
    return fallback_name, COMPARE_METRIC_REGISTRY[fallback_name]


def _load_run_outputs(
    resolved,
    *,
    run_name: str,
) -> Dict[str, object]:
    run_dir = resolved.method_root / run_name
    missing = [path for path in REQUIRED_RUN_FILES if not (run_dir / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Run {run_name} is missing required Step 5 outputs: {missing}")

    with open(run_dir / "metrics" / "method_metrics.json", "r", encoding="utf-8") as handle:
        method_metrics = json.load(handle)
    round_metrics_df = pd.read_csv(run_dir / "metrics" / "round_metrics.csv")
    target_row_summary_df = pd.read_csv(run_dir / "metrics" / "target_row_summary.csv")
    evaluation_results_df = pd.read_csv(run_dir / "metrics" / "evaluation_results.csv")
    optional_payloads: Dict[str, object] = {}
    for key, rel_path in OPTIONAL_RUN_FILES.items():
        path = run_dir / rel_path
        optional_payloads[key] = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "method_metrics": method_metrics,
        "round_metrics_df": round_metrics_df,
        "target_row_summary_df": target_row_summary_df,
        "evaluation_results_df": evaluation_results_df,
        **optional_payloads,
    }


def _build_run_comparison_row(
    run_payload: Dict[str, object],
    *,
    compare_metric_name: str,
    compare_metric: Dict[str, str],
) -> Dict[str, object]:
    method_metrics = dict(run_payload["method_metrics"])
    round_metrics_df = run_payload["round_metrics_df"]
    evaluation_results_df = run_payload["evaluation_results_df"]
    sampling_metadata_df = run_payload.get("sampling_metadata_df", pd.DataFrame())

    row = {
        "run_name": str(method_metrics["run_name"]),
        "canonical_family": str(method_metrics["canonical_family"]),
        "mean_success_hit_rate": _safe_float(method_metrics.get("mean_success_hit_rate")),
        "std_success_hit_rate": _safe_float(method_metrics.get("std_success_hit_rate")),
        "macro_average_row_mean_success_hit_rate": _safe_float(method_metrics.get("macro_average_row_mean_success_hit_rate")),
        "mean_success_hit_rate_discovery": _safe_float(method_metrics.get("mean_success_hit_rate_discovery")),
        "std_success_hit_rate_discovery": _safe_float(method_metrics.get("std_success_hit_rate_discovery")),
        "macro_average_row_mean_success_hit_rate_discovery": _safe_float(
            method_metrics.get("macro_average_row_mean_success_hit_rate_discovery")
        ),
        "mean_success_hit_rate_strict": _safe_float(method_metrics.get("mean_success_hit_rate_strict")),
        "std_success_hit_rate_strict": _safe_float(method_metrics.get("std_success_hit_rate_strict")),
        "macro_average_row_mean_success_hit_rate_strict": _safe_float(
            method_metrics.get("macro_average_row_mean_success_hit_rate_strict")
        ),
        "mean_success_hit_rate_loose": _safe_float(method_metrics.get("mean_success_hit_rate_loose")),
        "std_success_hit_rate_loose": _safe_float(method_metrics.get("std_success_hit_rate_loose")),
        "macro_average_row_mean_success_hit_rate_loose": _safe_float(
            method_metrics.get("macro_average_row_mean_success_hit_rate_loose")
        ),
        "mean_success_hit_rate_discovery_strict": _safe_float(
            method_metrics.get("mean_success_hit_rate_discovery_strict")
        ),
        "std_success_hit_rate_discovery_strict": _safe_float(
            method_metrics.get("std_success_hit_rate_discovery_strict")
        ),
        "macro_average_row_mean_success_hit_rate_discovery_strict": _safe_float(
            method_metrics.get("macro_average_row_mean_success_hit_rate_discovery_strict")
        ),
        "mean_success_hit_rate_discovery_loose": _safe_float(
            method_metrics.get("mean_success_hit_rate_discovery_loose")
        ),
        "std_success_hit_rate_discovery_loose": _safe_float(
            method_metrics.get("std_success_hit_rate_discovery_loose")
        ),
        "macro_average_row_mean_success_hit_rate_discovery_loose": _safe_float(
            method_metrics.get("macro_average_row_mean_success_hit_rate_discovery_loose")
        ),
        "mean_benchmark_soluble_oracle_calls": _safe_float(method_metrics.get("mean_benchmark_soluble_oracle_calls", 0.0), 0.0),
        "mean_benchmark_chi_oracle_calls": _safe_float(method_metrics.get("mean_benchmark_chi_oracle_calls", 0.0), 0.0),
        "mean_training_soluble_oracle_calls": _safe_float(method_metrics.get("mean_training_soluble_oracle_calls", 0.0), 0.0),
        "mean_training_chi_oracle_calls": _safe_float(method_metrics.get("mean_training_chi_oracle_calls", 0.0), 0.0),
        "mean_class_match_acceptance_rate": _safe_float(method_metrics.get("mean_class_match_acceptance_rate")),
        "mean_class_match_oversampling_ratio": _safe_float(method_metrics.get("mean_class_match_oversampling_ratio")),
        "mean_total_raw_samples_drawn": _safe_float(method_metrics.get("mean_total_raw_samples_drawn")),
        "success_metric_mode": str(method_metrics.get("success_metric_mode", "")),
        "discovery_success_metric_mode": str(method_metrics.get("discovery_success_metric_mode", "")),
        "comparison_metric_name": compare_metric_name,
        "comparison_metric_label": str(compare_metric["label"]),
        "comparison_mean_success_hit_rate": _safe_float(method_metrics.get(compare_metric["method_mean"])),
        "comparison_std_success_hit_rate": _safe_float(method_metrics.get(compare_metric["method_std"])),
        "comparison_macro_average_row_mean_success_hit_rate": _safe_float(method_metrics.get(compare_metric["method_macro"])),
    }
    for gate in [
        "valid_ok",
        "novel_ok",
        "star_ok",
        "sa_ok_reporting",
        "sa_ok_discovery",
        "sa_ok",
        "soluble_ok",
        "class_ok",
        "class_ok_loose",
        "class_ok_strict",
        "chi_ok",
        "chi_band_ok",
    ]:
        col = f"mean_{gate}_rate"
        row[col] = float(round_metrics_df[col].mean()) if col in round_metrics_df.columns and not round_metrics_df.empty else float("nan")

    row["num_rounds"] = int(round_metrics_df["round_id"].nunique()) if not round_metrics_df.empty else 0
    row["num_generated_samples"] = int(len(evaluation_results_df))
    row["num_sampling_metadata_rows"] = int(len(sampling_metadata_df)) if isinstance(sampling_metadata_df, pd.DataFrame) else 0
    row["mean_total_oracle_calls"] = (
        row["mean_benchmark_soluble_oracle_calls"]
        + row["mean_benchmark_chi_oracle_calls"]
        + row["mean_training_soluble_oracle_calls"]
        + row["mean_training_chi_oracle_calls"]
    )
    if isinstance(sampling_metadata_df, pd.DataFrame) and not sampling_metadata_df.empty:
        if "class_match_acceptance_rate" in sampling_metadata_df.columns:
            row["mean_class_match_acceptance_rate"] = float(sampling_metadata_df["class_match_acceptance_rate"].mean())
        if "class_match_oversampling_ratio" in sampling_metadata_df.columns:
            row["mean_class_match_oversampling_ratio"] = float(sampling_metadata_df["class_match_oversampling_ratio"].mean())
        if "total_raw_samples_drawn" in sampling_metadata_df.columns:
            row["mean_total_raw_samples_drawn"] = float(sampling_metadata_df["total_raw_samples_drawn"].mean())
    return row


def _build_per_target_run_comparison(
    run_payloads: List[Dict[str, object]],
    *,
    compare_metric_name: str,
    compare_metric: Dict[str, str],
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for payload in run_payloads:
        df = payload["target_row_summary_df"].copy()
        sampling_metadata_df = payload.get("sampling_metadata_df", pd.DataFrame())
        keep = [
            "run_name",
            "canonical_family",
            "target_row_id",
            "target_row_key",
            "c_target",
            "temperature",
            "phi",
            "chi_target",
            "mean_valid_ok_rate",
            "mean_novel_ok_rate",
            "mean_star_ok_rate",
            "mean_sa_ok_reporting_rate",
            "mean_sa_ok_discovery_rate",
            "mean_sa_ok_rate",
            "mean_soluble_ok_rate",
            "mean_chi_ok_rate",
            "mean_chi_band_ok_rate",
            "mean_property_success_hit_discovery_rate",
            "std_property_success_hit_discovery_rate",
            "mean_property_success_hit_rate",
            "std_property_success_hit_rate",
            "mean_success_hit_discovery_rate",
            "std_success_hit_discovery_rate",
            "mean_success_hit_rate",
            "std_success_hit_rate",
            "num_rounds",
        ]
        keep = [column for column in keep if column in df.columns]
        sub = df[keep].copy()
        sub["comparison_metric_name"] = compare_metric_name
        sub["comparison_metric_label"] = str(compare_metric["label"])
        sub["comparison_mean_success_hit_rate"] = (
            sub[compare_metric["target_mean"]].astype(float)
            if compare_metric["target_mean"] in sub.columns
            else sub["mean_success_hit_rate"].astype(float)
        )
        sub["comparison_std_success_hit_rate"] = (
            sub[compare_metric["target_std"]].astype(float)
            if compare_metric["target_std"] in sub.columns
            else sub["std_success_hit_rate"].astype(float)
        )
        if isinstance(sampling_metadata_df, pd.DataFrame) and not sampling_metadata_df.empty and "target_row_id" in sampling_metadata_df.columns:
            agg_dict = {}
            for column in [
                "class_match_acceptance_rate",
                "class_match_oversampling_ratio",
                "total_raw_samples_drawn",
                "class_match_sampling_attempts",
            ]:
                if column in sampling_metadata_df.columns:
                    agg_dict[column] = "mean"
            if agg_dict:
                sampling_target_df = (
                    sampling_metadata_df.groupby("target_row_id", dropna=False)
                    .agg(agg_dict)
                    .reset_index()
                    .rename(
                        columns={
                            "class_match_acceptance_rate": "mean_class_match_acceptance_rate",
                            "class_match_oversampling_ratio": "mean_class_match_oversampling_ratio",
                            "total_raw_samples_drawn": "mean_total_raw_samples_drawn",
                            "class_match_sampling_attempts": "mean_class_match_sampling_attempts",
                        }
                    )
                )
                sub = sub.merge(sampling_target_df, on="target_row_id", how="left")
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["target_row_id", "run_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_difficulty_summary(per_target_run_df: pd.DataFrame) -> pd.DataFrame:
    if per_target_run_df.empty:
        return pd.DataFrame()
    group_cols = ["target_row_id", "target_row_key", "c_target", "temperature", "phi", "chi_target"]
    rows: List[Dict[str, object]] = []
    for keys, sub in per_target_run_df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        row["num_runs"] = int(sub["run_name"].nunique())
        row["comparison_metric_name"] = (
            str(sub["comparison_metric_name"].iloc[0])
            if "comparison_metric_name" in sub.columns
            else "property_reporting_success_hit_rate"
        )
        row["comparison_metric_label"] = str(sub["comparison_metric_label"].iloc[0]) if "comparison_metric_label" in sub.columns else "Reporting success hit rate"
        row["mean_success_hit_rate_across_runs"] = float(sub["comparison_mean_success_hit_rate"].mean())
        row["std_success_hit_rate_across_runs"] = float(sub["comparison_mean_success_hit_rate"].std(ddof=0))
        row["mean_soluble_ok_rate_across_runs"] = float(sub["mean_soluble_ok_rate"].mean())
        row["mean_chi_ok_rate_across_runs"] = float(sub["mean_chi_ok_rate"].mean())
        if "mean_class_match_acceptance_rate" in sub.columns:
            row["mean_class_match_acceptance_rate_across_runs"] = float(sub["mean_class_match_acceptance_rate"].mean())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(
        ["mean_success_hit_rate_across_runs", "target_row_id"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["difficulty_rank"] = range(1, len(out) + 1)
    return out


def _build_canonical_family_comparison(run_comparison_df: pd.DataFrame) -> pd.DataFrame:
    if run_comparison_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for canonical_family, sub in run_comparison_df.groupby("canonical_family", sort=True):
        ordered = sub.sort_values(
            ["comparison_mean_success_hit_rate", "run_name"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        best = ordered.iloc[0]
        rows.append(
            {
                "canonical_family": str(canonical_family),
                "num_runs_compared": int(len(sub)),
                "best_run_name": str(best["run_name"]),
                "comparison_metric_name": str(best["comparison_metric_name"]),
                "comparison_metric_label": str(best["comparison_metric_label"]),
                "best_mean_success_hit_rate": float(best["comparison_mean_success_hit_rate"]),
                "best_std_success_hit_rate": float(best["comparison_std_success_hit_rate"]),
                "family_mean_success_hit_rate": float(sub["comparison_mean_success_hit_rate"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["best_mean_success_hit_rate", "canonical_family"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _write_compare_outputs(
    *,
    resolved,
    compare_root: Path,
    selected_runs: List[str],
    run_comparison_df: pd.DataFrame,
    per_target_run_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    canonical_family_df: pd.DataFrame,
    run_label_map_df: pd.DataFrame,
    target_label_map_df: pd.DataFrame,
    evaluation_compare_df: pd.DataFrame,
    supervised_history_df: pd.DataFrame,
    alignment_history_df: pd.DataFrame,
    dpo_history_df: pd.DataFrame,
    partial_compare: bool,
    skipped_runs: List[str],
    config_path: str,
) -> None:
    metrics_dir = compare_root / "metrics"
    figures_dir = compare_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_comparison_df.to_csv(metrics_dir / "run_comparison.csv", index=False)
    per_target_run_df.to_csv(metrics_dir / "per_target_run_comparison.csv", index=False)
    difficulty_df.to_csv(metrics_dir / "per_target_difficulty_summary.csv", index=False)
    canonical_family_df.to_csv(metrics_dir / "canonical_family_comparison.csv", index=False)
    run_label_map_df.to_csv(metrics_dir / "run_label_map.csv", index=False)
    target_label_map_df.to_csv(metrics_dir / "target_label_map.csv", index=False)
    if not evaluation_compare_df.empty:
        evaluation_compare_df.to_csv(metrics_dir / "chi_vs_target_compare_points.csv", index=False)

    best_run = run_comparison_df.iloc[0].to_dict() if not run_comparison_df.empty else {}
    payload = {
        "config_path": config_path,
        "compare_root": str(compare_root),
        "c_target": resolved.c_target,
        "split_mode": resolved.split_mode,
        "model_size": resolved.model_size,
        "compare_metric": (
            str(run_comparison_df["comparison_metric_name"].iloc[0])
            if "comparison_metric_name" in run_comparison_df.columns and not run_comparison_df.empty
            else str(resolved.step5_1.get("compare_metric", "full_reporting_success_hit_rate"))
        ),
        "compare_metric_label": (
            str(run_comparison_df["comparison_metric_label"].iloc[0])
            if "comparison_metric_label" in run_comparison_df.columns and not run_comparison_df.empty
            else "Full reporting success hit rate"
        ),
        "partial_compare": bool(partial_compare),
        "selected_runs": selected_runs,
        "skipped_runs": skipped_runs,
        "run_label_map": run_label_map_df.to_dict(orient="records"),
        "method_independence_note": (
            "S4_rl, S4_ppo, S4_grpo, and S4_dpo share the configured supervised S4 warm start "
            "by design so the comparison isolates the alignment algorithm."
        ),
        "best_run_overall": best_run,
        "best_run_by_family": canonical_family_df.to_dict(orient="records"),
    }
    with open(metrics_dir / "run_comparison.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    snapshot = {
        "model_size": resolved.model_size,
        "split_mode": resolved.split_mode,
        "classification_split_mode": resolved.classification_split_mode,
        "c_target": resolved.c_target,
        "config_path": config_path,
        "compare_all_enabled_runs": bool(resolved.step5_1.get("compare_all_enabled_runs", True)),
        "summarize_by_canonical_family": bool(resolved.step5_1.get("summarize_by_canonical_family", True)),
        "compare_metric": str(resolved.step5_1.get("compare_metric", "full_discovery_success_hit_rate")),
        "selected_runs": selected_runs,
        "skipped_runs": skipped_runs,
        "partial_compare": bool(partial_compare),
        "method_root": str(resolved.method_root),
        "compare_root": str(compare_root),
    }
    with open(compare_root / "config_snapshot.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(as_yamlable(snapshot), handle, sort_keys=False)

    font_size = int(resolved.step5["figure_font_size"])
    if not run_comparison_df.empty:
        plot_overall_success_all_runs(
            run_comparison_df,
            figures_dir / "overall_success_hit_rate_all_runs.png",
            font_size=font_size,
        )
        plot_success_gate_funnel_compare(
            run_comparison_df,
            figures_dir / "success_gate_funnel_compare_all_runs.png",
            font_size=font_size,
        )
        plot_success_vs_oracle_budget(
            run_comparison_df,
            figures_dir / "success_hit_vs_oracle_budget_all_runs.png",
            font_size=font_size,
        )
        if not evaluation_compare_df.empty:
            plot_chi_vs_target_compare(
                evaluation_compare_df,
                figures_dir / "chi_vs_target_compare_all_runs.png",
                font_size=font_size,
            )
    if not canonical_family_df.empty:
        plot_overall_success_by_family(
            canonical_family_df,
            figures_dir / "overall_success_hit_rate_by_family.png",
            font_size=font_size,
        )
    if not per_target_run_df.empty:
        plot_per_target_success_compare(
            per_target_run_df,
            figures_dir / "per_target_success_hit_rate_compare_all_runs.png",
            font_size=font_size,
        )
    if not difficulty_df.empty:
        plot_per_target_difficulty_ranked(
            difficulty_df,
            figures_dir / "per_target_difficulty_ranked.png",
            font_size=font_size,
        )
    if not supervised_history_df.empty:
        plot_supervised_training_curves(
            supervised_history_df,
            figures_dir / "supervised_training_curves.png",
            font_size=font_size,
        )
    if not alignment_history_df.empty:
        plot_alignment_training_curves(
            alignment_history_df,
            figures_dir / "s4_alignment_training_curves.png",
            font_size=font_size,
        )
    if not dpo_history_df.empty:
        plot_dpo_training_curves(
            dpo_history_df,
            figures_dir / "dpo_training_curves.png",
            font_size=font_size,
        )

    write_initial_log(
        compare_root,
        "step5_1_compare_inverse_design",
        context={
            "config_path": config_path,
            "model_size": resolved.model_size,
            "split_mode": resolved.split_mode,
            "c_target": resolved.c_target,
            "partial_compare": bool(partial_compare),
            "selected_runs": ",".join(selected_runs),
            "skipped_runs": ",".join(skipped_runs),
        },
    )
    save_artifact_manifest(compare_root, metrics_dir, figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Step 5 inverse-design runs.")
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
    parser.add_argument("--runs", default=None, help="Comma-separated subset of enabled runs to compare.")
    parser.add_argument("--allow_partial", action="store_true", help="Skip missing/incomplete runs for development.")
    parser.add_argument(
        "--method_root_suffix",
        default=None,
        help="Optional suffix appended to the Step 5 method and comparison roots.",
    )
    args = parser.parse_args()

    resolved = load_step5_config(
        config_path=args.config,
        base_config_path=args.base_config,
        model_size=args.model_size,
        c_target_override=args.c_target,
    )
    resolved = apply_step5_output_suffix(resolved, method_root_suffix=args.method_root_suffix)
    selected_runs = _resolve_selected_runs(resolved, args.runs, allow_partial=bool(args.allow_partial))

    run_payloads: List[Dict[str, object]] = []
    skipped_runs: List[str] = []
    for run_name in selected_runs:
        try:
            run_payloads.append(_load_run_outputs(resolved, run_name=run_name))
        except FileNotFoundError:
            if not args.allow_partial:
                raise
            skipped_runs.append(run_name)

    if not run_payloads:
        raise ValueError("No completed Step 5 runs are available for Step 5_1 comparison.")

    compare_metric_name, compare_metric = _resolve_compare_metric(resolved, run_payloads)
    run_rows = [
        _build_run_comparison_row(
            payload,
            compare_metric_name=compare_metric_name,
            compare_metric=compare_metric,
        )
        for payload in run_payloads
    ]
    run_comparison_df = pd.DataFrame(run_rows).sort_values(
        ["comparison_mean_success_hit_rate", "run_name"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    run_comparison_df["rank"] = range(1, len(run_comparison_df) + 1)
    run_label_map_df = _build_run_label_map(run_comparison_df)
    run_comparison_df = _attach_run_labels(run_comparison_df, run_label_map_df)

    per_target_run_df = _build_per_target_run_comparison(
        run_payloads,
        compare_metric_name=compare_metric_name,
        compare_metric=compare_metric,
    )
    per_target_run_df = _attach_run_labels(per_target_run_df, run_label_map_df)
    target_label_map_df = _build_target_label_map(per_target_run_df)
    per_target_run_df = _attach_target_labels(per_target_run_df, target_label_map_df)
    difficulty_df = _build_difficulty_summary(per_target_run_df)
    difficulty_df = _attach_target_labels(difficulty_df, target_label_map_df)
    canonical_family_df = _build_canonical_family_comparison(run_comparison_df)
    canonical_family_df = _attach_best_run_labels(canonical_family_df, run_label_map_df)
    evaluation_compare_df = _collect_evaluation_compare_df(
        run_payloads,
        run_label_map_df,
        target_label_map_df,
    )
    supervised_history_df = _collect_history_df(
        run_payloads,
        run_label_map_df,
        ["supervised_history_df"],
    )
    alignment_history_df = _collect_history_df(
        run_payloads,
        run_label_map_df,
        ["rl_history_df", "ppo_history_df", "grpo_history_df"],
    )
    dpo_history_df = _collect_history_df(
        run_payloads,
        run_label_map_df,
        ["dpo_history_df"],
    )

    compare_root = resolved.compare_root
    _write_compare_outputs(
        resolved=resolved,
        compare_root=compare_root,
        selected_runs=[payload["run_name"] for payload in run_payloads],
        run_comparison_df=run_comparison_df,
        per_target_run_df=per_target_run_df,
        difficulty_df=difficulty_df,
        canonical_family_df=canonical_family_df,
        run_label_map_df=run_label_map_df,
        target_label_map_df=target_label_map_df,
        evaluation_compare_df=evaluation_compare_df,
        supervised_history_df=supervised_history_df,
        alignment_history_df=alignment_history_df,
        dpo_history_df=dpo_history_df,
        partial_compare=bool(args.allow_partial and skipped_runs),
        skipped_runs=skipped_runs,
        config_path=args.config,
    )

    print(f"Step 5_1 comparison written to: {compare_root}")
    print(f"Compared runs ({len(run_payloads)}): {[payload['run_name'] for payload in run_payloads]}")
    if skipped_runs:
        print(f"Skipped runs ({len(skipped_runs)}): {skipped_runs}")


if __name__ == "__main__":
    main()
