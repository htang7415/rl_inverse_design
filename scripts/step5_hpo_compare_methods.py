#!/usr/bin/env python3
"""Compare Step 5 HPO running-best metrics across methods."""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.step5.config import apply_step5_hpo_output_suffix, load_step5_config
from src.step5.hpo import get_hpo_root
from src.step5.plotting import apply_step5_plot_style
from src.utils.figure_style import save_publication_figure

METHOD_LABELS = {
    "S1": "S1: Guided",
    "S2": "S2: Conditional",
    "S3": "S3: Guided conditional",
    "S4_dpo": "S4: RL-DPO",
    "S4_grpo": "S4: RL-GRPO",
    "S4_ppo": "S4: RL-PPO",
    "S4_rl": "S4: RL",
}


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


def _load_trials_df(study_root: Path) -> pd.DataFrame:
    db_path = study_root / "optuna.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing Optuna database: {db_path}")
    query = """
    WITH attr AS (
        SELECT
            trial_id,
            MAX(CASE WHEN "key" = 'mean_success_hit_rate_reporting' THEN value_json END)
                AS success_hit_rate,
            MAX(CASE WHEN "key" = 'mean_success_hit_rate_discovery' THEN value_json END)
                AS success_hit_rate_discovery
        FROM trial_user_attributes
        GROUP BY trial_id
    )
    SELECT
        t.number AS trial_number,
        t.state AS state,
        attr.success_hit_rate,
        attr.success_hit_rate_discovery
    FROM trials t
    LEFT JOIN attr ON attr.trial_id = t.trial_id
    ORDER BY t.number
    """
    with sqlite3.connect(db_path) as conn:
        trials_df = pd.read_sql_query(query, conn)
    return trials_df


def _running_best(values: Iterable[float]) -> list[float]:
    best_so_far = float("nan")
    running_best: list[float] = []
    for value in values:
        if np.isfinite(value):
            if not np.isfinite(best_so_far):
                best_so_far = float(value)
            else:
                best_so_far = max(float(best_so_far), float(value))
        running_best.append(float(best_so_far) if np.isfinite(best_so_far) else float("nan"))
    return running_best


def _build_plot_frame(study_root: Path, metric_cols: list[str]) -> pd.DataFrame:
    trials_df = _load_trials_df(study_root)
    available_metric_cols = [col for col in metric_cols if col in trials_df.columns]
    frame = trials_df.loc[:, ["trial_number", *available_metric_cols]].copy()
    frame["trial_number"] = pd.to_numeric(frame["trial_number"], errors="coerce")
    if available_metric_cols:
        metric_values = pd.to_numeric(frame[available_metric_cols[0]], errors="coerce")
        for metric_col in available_metric_cols[1:]:
            metric_values = metric_values.combine_first(pd.to_numeric(frame[metric_col], errors="coerce"))
        frame["metric_value"] = metric_values
    else:
        frame["metric_value"] = np.nan
    frame = frame.loc[frame["trial_number"].notna()].sort_values("trial_number", kind="mergesort").reset_index(drop=True)
    frame["running_best"] = _running_best(frame["metric_value"].to_numpy(dtype=float))
    return frame


def _plot_metric(study_root: Path, metric_cols: list[str], ylabel: str, output_name: str, *, font_size: int, dpi: int) -> None:
    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.set_box_aspect(1.0)

    plotted_any = False
    ymax = 0.0
    for method_dir in sorted(path for path in study_root.iterdir() if path.is_dir()):
        try:
            frame = _build_plot_frame(method_dir, metric_cols=metric_cols)
        except Exception:
            continue
        if frame.empty or not np.isfinite(frame["running_best"].to_numpy(dtype=float)).any():
            continue
        plotted_any = True
        trial_numbers = frame["trial_number"].to_numpy(dtype=int)
        running_best = frame["running_best"].to_numpy(dtype=float)
        ymax = max(ymax, float(np.nanmax(running_best)))
        ax.plot(
            trial_numbers,
            running_best,
            linewidth=1.8,
            marker="o",
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=METHOD_LABELS.get(method_dir.name, method_dir.name),
        )

    if plotted_any:
        ax.set_ylim(0.0, max(1.0, min(1.08, ymax * 1.12 if ymax > 0.0 else 1.0)))
        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=int(font_size),
            ncol=1,
            columnspacing=0.9,
            handlelength=1.5,
        )
    else:
        ax.text(0.5, 0.5, "No completed trials", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, axis="y", linestyle=(0, (3, 3)), linewidth=0.65, color="#D8DEE4", alpha=0.9)
    fig.tight_layout(rect=(0, 0, 0.76, 1) if plotted_any else None)
    save_publication_figure(fig, study_root / output_name, dpi=dpi, write_pdf=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Step 5 HPO running-best curves across methods.")
    parser.add_argument(
        "--study-root",
        type=Path,
        default=None,
        help="Directory containing per-method Step 5 HPO studies. If omitted, resolve from the Step 5 config.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config5.yaml"))
    parser.add_argument("--base_config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--model_size", default="small")
    parser.add_argument("--c_target", default="")
    parser.add_argument("--target_temperature", type=float, default=None)
    parser.add_argument("--target_phi", type=float, default=None)
    parser.add_argument(
        "--hpo_root_suffix",
        default=None,
        help="Optional suffix for the HPO c_target output root. Defaults to the target condition suffix.",
    )
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    if args.study_root is not None:
        study_root = args.study_root.resolve()
    else:
        resolved = load_step5_config(
            config_path=args.config,
            base_config_path=args.base_config,
            model_size=args.model_size,
            c_target_override=(args.c_target or None),
        )
        target_temperature, target_phi = _resolve_target_condition(args, resolved)
        hpo_root_suffix = args.hpo_root_suffix
        if hpo_root_suffix is None and target_temperature is not None and target_phi is not None:
            hpo_root_suffix = _target_condition_suffix(target_temperature, target_phi)
        resolved = apply_step5_hpo_output_suffix(resolved, hpo_root_suffix=hpo_root_suffix)
        study_root = get_hpo_root(resolved).resolve()
    study_root.mkdir(parents=True, exist_ok=True)

    _plot_metric(
        study_root,
        metric_cols=["success_hit_rate_discovery", "success_hit_rate"],
        ylabel="Success hit rate",
        output_name="hpo_compare_best_success_hit_rate.png",
        font_size=int(args.font_size),
        dpi=int(args.dpi),
    )
    for stale_name in [
        "hpo_compare_best_mean_property_success_hit_rate.png",
        "hpo_compare_best_mean_success_hit_rate.png",
        "hpo_compare_best_mean_success_hit_rate_discovery.png",
    ]:
        stale_output = study_root / stale_name
        if stale_output.exists():
            stale_output.unlink()


if __name__ == "__main__":
    main()
