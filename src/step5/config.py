"""Config and target-table utilities for Step 5."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from src.chi.data import SplitConfig, add_split_column, load_chi_dataset, make_split_assignments
from src.chi.inverse_design_common import load_soluble_targets
from src.evaluation.class_decode_constraints import load_decode_constraint_source_smiles
from src.evaluation.polymer_class import BACKBONE_CLASS_MATCH_CLASSES, PolymerClassifier
from src.utils.config import load_config
from src.utils.model_scales import get_results_dir
from src.utils.chemistry import canonicalize_smiles


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_step5_bundle_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a Step 5 config, optionally merging a small overlay into a base config."""

    config_path = Path(config_path)
    config = load_config(str(config_path))
    if not isinstance(config, dict):
        return config
    extends = config.get("extends", config.get("base_step5_config"))
    if extends in {None, "", "null"}:
        return config
    base_path = Path(str(extends))
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    override = {
        key: deepcopy(value)
        for key, value in config.items()
        if key not in {"extends", "base_step5_config"}
    }
    return _deep_merge(_load_step5_bundle_config(base_path), override)


def _as_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_as_serializable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _format_float(value: float) -> str:
    return np.format_float_positional(float(value), trim="-")


def _condition_key(temperature: float, phi: float) -> Tuple[str, str]:
    return (_format_float(temperature), _format_float(phi))


def _select_evenly_spaced_indices(total_count: int, select_count: int) -> List[int]:
    total_count = int(total_count)
    select_count = int(select_count)
    if total_count <= 0 or select_count <= 0:
        return []
    if select_count >= total_count:
        return list(range(total_count))
    requested = np.rint(np.linspace(0, total_count - 1, num=select_count)).astype(int).tolist()
    selected: List[int] = []
    used: set[int] = set()
    for target in requested:
        if target not in used:
            selected.append(int(target))
            used.add(int(target))
            continue
        for delta in range(1, total_count):
            left = int(target - delta)
            if left >= 0 and left not in used:
                selected.append(left)
                used.add(left)
                break
            right = int(target + delta)
            if right < total_count and right not in used:
                selected.append(right)
                used.add(right)
                break
    return sorted(selected)


def _first_existing(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    for path in paths:
        if path.exists():
            return path
    if not paths:
        raise ValueError("No candidate paths provided.")
    return paths[0]


def _build_target_row_key(c_target: str, temperature: float, phi: float, chi_target: float) -> str:
    return (
        f"{c_target}|T={_format_float(temperature)}"
        f"|phi={_format_float(phi)}|chi={_format_float(chi_target)}"
    )


def _resolve_class_numeric_overrides(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    resolved: Dict[str, float] = {}
    for key, value in raw.items():
        class_name = str(key).strip().lower()
        if not class_name:
            continue
        try:
            resolved[class_name] = float(value)
        except (TypeError, ValueError):
            continue
    return resolved


def resolve_step5_generation_budget(step5_cfg: Dict[str, Any], c_target: str) -> int:
    """Resolve the per-class Step 5 generation budget."""
    base_budget = int(step5_cfg["generation_budget"])
    overrides = _resolve_class_numeric_overrides(step5_cfg.get("generation_budget_by_class", {}))
    return int(round(overrides.get(str(c_target).strip().lower(), float(base_budget))))


def resolve_step5_hpo_generation_budget(hpo_cfg: Dict[str, Any], c_target: str) -> int:
    """Resolve the per-class Step 5 HPO generation budget."""
    base_budget = int(hpo_cfg["hpo_generation_budget"])
    overrides = _resolve_class_numeric_overrides(hpo_cfg.get("hpo_generation_budget_by_class", {}))
    return int(round(overrides.get(str(c_target).strip().lower(), float(base_budget))))


def resolve_step5_sampling_num_steps(step5_cfg: Dict[str, Any], base_config: Dict[str, Any]) -> int:
    """Resolve sampler reverse-diffusion steps without changing checkpoint shapes."""
    base_steps = int(base_config["diffusion"]["num_steps"])
    raw_steps = step5_cfg.get("sampling_num_steps", None)
    if raw_steps in {None, "", "null"}:
        return int(base_steps)
    return int(max(1, min(base_steps, int(raw_steps))))


def resolve_step5_sa_thresholds(step5_cfg: Dict[str, Any], c_target: str) -> Dict[str, float]:
    """Resolve reporting and discovery SA thresholds for a target class."""
    target_key = str(c_target).strip().lower()
    reporting_default = float(step5_cfg["target_sa_max"])
    reporting_overrides = _resolve_class_numeric_overrides(step5_cfg.get("reporting_target_sa_max_by_class", {}))
    discovery_overrides = _resolve_class_numeric_overrides(step5_cfg.get("discovery_target_sa_max_by_class", {}))
    reporting = float(reporting_overrides.get(target_key, reporting_default))
    discovery = float(discovery_overrides.get(target_key, reporting))
    return {
        "reporting": reporting,
        "discovery": discovery,
    }


def select_step5_proxy_target_rows(
    source_df: pd.DataFrame,
    *,
    num_targets: int,
) -> pd.DataFrame:
    """Select a small deterministic proxy slice from benchmark target rows.

    This is used by S4 checkpoint selection so the proxy objective stays aligned
    with the actual inverse-design target distribution instead of a disjoint
    validation table.
    """

    work = source_df.reset_index(drop=True).copy()
    if work.empty:
        return work
    limit = max(1, min(int(num_targets), int(len(work))))
    if limit >= len(work):
        return work

    candidate_indices = np.linspace(0, len(work) - 1, num=limit, dtype=int)
    seen: set[int] = set()
    ordered_indices: List[int] = []
    for idx in candidate_indices.tolist():
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        ordered_indices.append(idx)
    if len(ordered_indices) < limit:
        for idx in range(len(work)):
            if idx in seen:
                continue
            seen.add(idx)
            ordered_indices.append(int(idx))
            if len(ordered_indices) >= limit:
                break
    return work.iloc[ordered_indices].reset_index(drop=True)


@dataclass(frozen=True)
class ExactChiTargetLookup:
    """Exact-match Step 3 chi-target lookup."""

    mapping: Dict[Tuple[str, str], float]
    source_path: Path
    property_rule_mapping: Dict[Tuple[str, str], str]
    q025_mapping: Dict[Tuple[str, str], float]
    q975_mapping: Dict[Tuple[str, str], float]

    def lookup(
        self,
        temperature: float,
        phi: float,
        *,
        warn_on_missing: bool = False,
    ) -> Optional[float]:
        value = self.mapping.get(_condition_key(temperature, phi))
        if value is None and warn_on_missing:
            warnings.warn(
                (
                    "Missing exact Step 3 chi_target lookup for "
                    f"(T={temperature}, phi={phi}) in {self.source_path}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        return value

    def lookup_row(
        self,
        temperature: float,
        phi: float,
        *,
        warn_on_missing: bool = False,
    ) -> Optional[Dict[str, Any]]:
        key = _condition_key(temperature, phi)
        chi_target = self.mapping.get(key)
        if chi_target is None:
            if warn_on_missing:
                warnings.warn(
                    (
                        "Missing exact Step 3 chi_target lookup for "
                        f"(T={temperature}, phi={phi}) in {self.source_path}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
            return None
        q025 = self.q025_mapping.get(key, np.nan)
        q975 = self.q975_mapping.get(key, np.nan)
        return {
            "chi_target": float(chi_target),
            "property_rule": str(self.property_rule_mapping.get(key, "upper_bound")).strip().lower(),
            "chi_target_boot_q025": float(q025) if np.isfinite(q025) else np.nan,
            "chi_target_boot_q975": float(q975) if np.isfinite(q975) else np.nan,
        }


@dataclass(frozen=True)
class ResolvedStep5Config:
    """Fully resolved Step 5 configuration."""

    base_config: Dict[str, Any]
    step5: Dict[str, Any]
    step5_1: Dict[str, Any]
    step5_hpo: Dict[str, Any]
    model_size: Optional[str]
    split_mode: str
    classification_split_mode: str
    c_target: str
    enabled_runs: List[str]
    available_target_classes: List[str]
    polymer_patterns: Dict[str, str]
    results_dir: Path
    results_dir_nosplit: Path
    base_results_dir: Path
    method_root: Path
    compare_root: Path
    step4_reg_dir: Path
    step4_cls_dir: Path
    step4_reg_metrics_dir: Path
    step4_cls_metrics_dir: Path
    step3_targets_path: Path
    chi_lookup: ExactChiTargetLookup
    target_base_df: pd.DataFrame
    target_family_df: pd.DataFrame
    rl_proxy_df: pd.DataFrame
    hpo_target_df: pd.DataFrame
    chi_split_df: pd.DataFrame
    chi_train_stats: Dict[str, float]
    class_support_stats: Dict[str, Any]
    config_snapshot: Dict[str, Any]


def apply_step5_output_suffix(
    resolved: ResolvedStep5Config,
    *,
    method_root_suffix: str | None,
) -> ResolvedStep5Config:
    """Return a resolved config whose Step 5 output roots have a suffix."""

    if method_root_suffix in {None, "", "null"}:
        return resolved

    suffix = str(method_root_suffix)
    method_root = resolved.method_root.parent / f"{resolved.method_root.name}{suffix}"
    compare_root = resolved.compare_root.parent / f"{resolved.compare_root.name}{suffix}"

    snapshot = deepcopy(resolved.config_snapshot)
    paths = snapshot.setdefault("paths", {})
    if isinstance(paths, dict):
        paths["method_root"] = str(method_root)
        paths["compare_root"] = str(compare_root)

    return replace(
        resolved,
        method_root=method_root,
        compare_root=compare_root,
        config_snapshot=snapshot,
    )


def filter_step5_target_rows_by_condition(
    target_family_df: pd.DataFrame,
    *,
    target_temperature: float | None,
    target_phi: float | None,
    atol: float = 1.0e-6,
) -> pd.DataFrame:
    """Filter Step 5 benchmark rows to one exact target condition."""

    if target_temperature is None and target_phi is None:
        return target_family_df.copy()
    if target_temperature is None or target_phi is None:
        raise ValueError("target_temperature and target_phi must be provided together.")

    work = target_family_df.copy()
    if work.empty:
        raise ValueError("Step 5 target table is empty; cannot apply a target-condition filter.")
    required = {"temperature", "phi"}
    if not required.issubset(work.columns):
        raise ValueError(f"Step 5 target table must contain {sorted(required)} columns.")

    temperature_match = np.isclose(
        work["temperature"].astype(float),
        float(target_temperature),
        atol=float(atol),
        rtol=0.0,
    )
    phi_match = np.isclose(
        work["phi"].astype(float),
        float(target_phi),
        atol=float(atol),
        rtol=0.0,
    )
    mask = temperature_match & phi_match
    out = work.loc[mask].reset_index(drop=True).copy()
    if out.empty:
        available = (
            work[["temperature", "phi"]]
            .drop_duplicates()
            .sort_values(["temperature", "phi"])
            .head(30)
            .to_dict(orient="records")
        )
        raise ValueError(
            "No Step 5 target row matches "
            f"target_temperature={float(target_temperature)} target_phi={float(target_phi)}. "
            f"Available conditions include: {available}"
        )
    return out


def apply_step5_target_condition_filter(
    resolved: ResolvedStep5Config,
    *,
    target_temperature: float | None,
    target_phi: float | None,
    atol: float = 1.0e-6,
) -> ResolvedStep5Config:
    """Return a resolved config restricted to one target temperature/phi row."""

    if target_temperature is None and target_phi is None:
        return resolved

    target_rows = filter_step5_target_rows_by_condition(
        resolved.target_family_df,
        target_temperature=target_temperature,
        target_phi=target_phi,
        atol=atol,
    )
    step5_cfg = deepcopy(resolved.step5)
    step5_cfg["target_temperature"] = float(target_temperature)
    step5_cfg["target_phi"] = float(target_phi)

    snapshot = deepcopy(resolved.config_snapshot)
    snapshot["step5"] = _as_serializable(step5_cfg)
    derived = snapshot.setdefault("derived", {})
    if isinstance(derived, dict):
        derived["num_target_rows"] = int(len(target_rows))
        derived["target_condition_filter"] = {
            "target_temperature": float(target_temperature),
            "target_phi": float(target_phi),
            "atol": float(atol),
            "matched_target_row_ids": [
                int(value)
                for value in target_rows.get("target_row_id", pd.Series(dtype=int)).tolist()
            ],
        }

    return replace(
        resolved,
        step5=step5_cfg,
        target_family_df=target_rows,
        config_snapshot=snapshot,
    )


def _resolve_step4_dirs(
    base_config: Dict[str, Any],
    *,
    model_size: Optional[str],
    split_mode: str,
) -> Tuple[Path, Path, Path, Path, Path, Path]:
    base_results_dir = Path(base_config["paths"]["results_dir"])
    results_dir = Path(get_results_dir(model_size, base_results_dir, split_mode))
    results_dir_nosplit = Path(get_results_dir(model_size, base_results_dir, None))

    reg_candidates = [
        results_dir_nosplit / "step4_1_regression" / split_mode,
        results_dir / "step4_1_regression" / split_mode,
        results_dir_nosplit / "step4_chi_training" / "step4_1_regression" / split_mode,
        results_dir / "step4_chi_training" / split_mode / "step4_1_regression",
    ]
    cls_candidates = [
        results_dir_nosplit / "step4_2_classification",
        results_dir / "step4_2_classification",
        results_dir_nosplit / "step4_chi_training" / "step4_2_classification",
        results_dir / "step4_chi_training" / split_mode / "step4_2_classification",
    ]

    step4_reg_dir = _first_existing(reg_candidates)
    step4_cls_dir = _first_existing(cls_candidates)
    return (
        results_dir,
        results_dir_nosplit,
        base_results_dir,
        step4_reg_dir,
        step4_cls_dir,
        step4_reg_dir / "metrics",
    )


def _resolve_step4_metrics_dirs(
    base_config: Dict[str, Any],
    *,
    model_size: Optional[str],
    split_mode: str,
) -> Tuple[Path, Path, Path, Path, Path, Path, Path]:
    (
        results_dir,
        results_dir_nosplit,
        base_results_dir,
        step4_reg_dir,
        step4_cls_dir,
        step4_reg_metrics_dir,
    ) = _resolve_step4_dirs(base_config, model_size=model_size, split_mode=split_mode)
    step4_cls_metrics_dir = step4_cls_dir / "metrics"
    return (
        results_dir,
        results_dir_nosplit,
        base_results_dir,
        step4_reg_dir,
        step4_cls_dir,
        step4_reg_metrics_dir,
        step4_cls_metrics_dir,
    )


def _resolve_split_ratios(base_config: Dict[str, Any]) -> Dict[str, float]:
    chi_cfg = base_config.get("chi_training", {})
    shared_cfg = chi_cfg.get("shared", {}) if isinstance(chi_cfg.get("shared"), dict) else {}
    split_cfg = shared_cfg.get("split", {}) if isinstance(shared_cfg.get("split"), dict) else {}
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Step 5 split ratios must sum to 1.0, got {train_ratio}+{val_ratio}+{test_ratio}={total_ratio}"
        )
    return {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
    }


def _load_or_build_chi_split_df(
    *,
    base_config: Dict[str, Any],
    step4_reg_dir: Path,
    step4_reg_metrics_dir: Path,
    split_mode: str,
    random_seed: int,
) -> pd.DataFrame:
    split_candidates = [
        step4_reg_metrics_dir / "chi_dataset_with_split.csv",
        step4_reg_dir / "shared" / f"chi_dataset_with_split_step4_1_{split_mode}.csv",
        step4_reg_dir / "shared" / "chi_dataset_with_split.csv",
    ]
    split_path = next((path for path in split_candidates if path.exists()), None)
    if split_path is not None:
        df = pd.read_csv(split_path)
        if "split" not in df.columns:
            raise ValueError(f"Split-aware chi dataset missing 'split' column: {split_path}")
        return df

    warnings.warn(
        (
            "Step 4 split-aware chi dataset not found; rebuilding deterministic "
            "Step 5 D_chi split locally from the explicit 8:1:1 split ratios."
        ),
        RuntimeWarning,
        stacklevel=2,
    )
    dataset_path = base_config["chi_training"]["shared"]["dataset_path"]
    split_ratios = _resolve_split_ratios(base_config)
    chi_df = load_chi_dataset(dataset_path)
    assignments = make_split_assignments(
        chi_df,
        SplitConfig(
            split_mode=split_mode,
            train_ratio=split_ratios["train_ratio"],
            val_ratio=split_ratios["val_ratio"],
            test_ratio=split_ratios["test_ratio"],
            seed=int(random_seed),
        ),
    )
    return add_split_column(chi_df, assignments)


def _build_lookup_and_target_tables(
    *,
    base_config: Dict[str, Any],
    step5_cfg: Dict[str, Any],
    results_dir: Path,
    base_results_dir: Path,
    split_mode: str,
) -> Tuple[Path, ExactChiTargetLookup, pd.DataFrame, pd.DataFrame]:
    target_base_df, path_used = load_soluble_targets(
        targets_csv=None,
        results_dir=results_dir,
        base_results_dir=base_results_dir,
        split_mode=split_mode,
    )
    target_base_df = target_base_df.rename(columns={"target_chi": "chi_target"}).copy()
    target_base_df["temperature"] = target_base_df["temperature"].astype(float)
    target_base_df["phi"] = target_base_df["phi"].astype(float)
    target_base_df["chi_target"] = target_base_df["chi_target"].astype(float)
    lookup = ExactChiTargetLookup(
        mapping={
            _condition_key(float(row["temperature"]), float(row["phi"])): float(row["chi_target"])
            for _, row in target_base_df.iterrows()
        },
        source_path=Path(path_used),
        property_rule_mapping={
            _condition_key(float(row["temperature"]), float(row["phi"])): str(
                row.get("property_rule", "upper_bound")
            ).strip().lower()
            for _, row in target_base_df.iterrows()
        },
        q025_mapping={
            _condition_key(float(row["temperature"]), float(row["phi"])): float(value)
            for _, row in target_base_df.iterrows()
            for value in [pd.to_numeric(row.get("chi_target_boot_q025", np.nan), errors="coerce")]
            if np.isfinite(value)
        },
        q975_mapping={
            _condition_key(float(row["temperature"]), float(row["phi"])): float(value)
            for _, row in target_base_df.iterrows()
            for value in [pd.to_numeric(row.get("chi_target_boot_q975", np.nan), errors="coerce")]
            if np.isfinite(value)
        },
    )

    c_target = str(step5_cfg["c_target"]).strip().lower()
    rows: List[Dict[str, Any]] = []
    for idx, row in target_base_df.sort_values(["temperature", "phi"]).reset_index(drop=True).iterrows():
        temperature = float(row["temperature"])
        phi = float(row["phi"])
        chi_target = float(row["chi_target"])
        property_rule = str(row.get("property_rule", "upper_bound")).strip().lower()
        target_row_id = int(idx + 1)
        target_row_key = _build_target_row_key(c_target, temperature, phi, chi_target)
        row_payload = {
            "target_row_id": target_row_id,
            "target_id": target_row_id,
            "target_row_key": target_row_key,
            "c_target": c_target,
            "target_polymer_class": c_target,
            "temperature": temperature,
            "phi": phi,
            "chi_target": chi_target,
            "target_chi": chi_target,
            "property_rule": property_rule,
        }
        for optional_col in ("chi_target_boot_q025", "chi_target_boot_q975"):
            optional_value = pd.to_numeric(row.get(optional_col, np.nan), errors="coerce")
            row_payload[optional_col] = float(optional_value) if np.isfinite(optional_value) else np.nan
        rows.append(row_payload)
    target_family_df = pd.DataFrame(rows)
    return Path(path_used), lookup, target_base_df, target_family_df


def _build_validation_target_tables(
    *,
    chi_split_df: pd.DataFrame,
    c_target: str,
    chi_lookup: ExactChiTargetLookup,
    rl_proxy_num_targets: int,
    hpo_enabled: bool,
    hpo_num_targets: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    val_df = chi_split_df.loc[chi_split_df["split"].astype(str) == "val"].copy()
    if val_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "coverage_warning": "no_val_rows",
                "proxy_target_gap_warning": False,
                "proxy_target_gap_warning_threshold": 0.3,
            },
            pd.DataFrame(),
        )

    val_df["canonical_smiles"] = val_df["SMILES"].astype(str).map(canonicalize_smiles)
    val_df["canonical_smiles"] = val_df["canonical_smiles"].where(
        val_df["canonical_smiles"].notna(),
        val_df["SMILES"].astype(str),
    )
    val_df["step3_chi_target"] = [
        chi_lookup.lookup(float(t), float(p), warn_on_missing=False)
        for t, p in zip(val_df["temperature"], val_df["phi"])
    ]
    eligible_df = val_df.loc[val_df["step3_chi_target"].notna()].copy()
    if eligible_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "coverage_warning": "no_exact_step3_match",
                "proxy_target_gap_warning": False,
                "proxy_target_gap_warning_threshold": 0.3,
            },
            pd.DataFrame(),
        )

    ordered_buckets = (
        eligible_df[["temperature", "phi"]]
        .drop_duplicates()
        .sort_values(["temperature", "phi"])
        .reset_index(drop=True)
    )
    total_bucket_count = int(len(ordered_buckets))
    rl_bucket_count = min(int(rl_proxy_num_targets), total_bucket_count)
    hpo_bucket_count = (
        min(int(hpo_num_targets), max(0, total_bucket_count - rl_bucket_count))
        if hpo_enabled
        else 0
    )
    combined_bucket_count = int(rl_bucket_count + hpo_bucket_count)
    combined_bucket_indices = _select_evenly_spaced_indices(total_bucket_count, combined_bucket_count)
    rl_selection_positions = set(
        _select_evenly_spaced_indices(len(combined_bucket_indices), rl_bucket_count)
    )
    rl_bucket_indices = [
        combined_bucket_indices[pos]
        for pos in range(len(combined_bucket_indices))
        if pos in rl_selection_positions
    ]
    hpo_bucket_indices = [
        combined_bucket_indices[pos]
        for pos in range(len(combined_bucket_indices))
        if pos not in rl_selection_positions
    ]

    rl_buckets = ordered_buckets.iloc[rl_bucket_indices].copy()
    hpo_buckets = ordered_buckets.iloc[hpo_bucket_indices].copy()

    def _bucket_rows(bucket_df: pd.DataFrame, *, source_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows: List[Dict[str, Any]] = []
        drift_rows: List[Dict[str, Any]] = []
        for idx, bucket in bucket_df.reset_index(drop=True).iterrows():
            bucket_rows = eligible_df.loc[
                np.isclose(eligible_df["temperature"].astype(float), float(bucket["temperature"]))
                & np.isclose(eligible_df["phi"].astype(float), float(bucket["phi"]))
            ].copy()
            bucket_rows["abs_step3_gap"] = (
                pd.to_numeric(bucket_rows["chi"], errors="coerce")
                - pd.to_numeric(bucket_rows["step3_chi_target"], errors="coerce")
            ).abs()
            bucket_rows = bucket_rows.sort_values(
                ["abs_step3_gap", "canonical_smiles", "Polymer", "row_id"]
            ).reset_index(drop=True)
            selected = bucket_rows.iloc[0]
            chi_target = float(selected["chi"])
            step3_target = float(selected["step3_chi_target"])
            abs_step3_gap = abs(chi_target - step3_target)
            target_row_id = int(idx + 1)
            rows.append(
                {
                    "target_row_id": target_row_id,
                    "target_id": target_row_id,
                    "target_row_key": _build_target_row_key(
                        c_target,
                        float(selected["temperature"]),
                        float(selected["phi"]),
                        chi_target,
                    ),
                    "c_target": c_target,
                    "target_polymer_class": c_target,
                    "temperature": float(selected["temperature"]),
                    "phi": float(selected["phi"]),
                    "chi_target": chi_target,
                    "target_chi": chi_target,
                    "property_rule": "upper_bound",
                    "proxy_source": source_name,
                    "source_row_id": int(selected["row_id"]),
                    "source_polymer": str(selected["Polymer"]),
                    "source_canonical_smiles": str(selected["canonical_smiles"]),
                    "step3_chi_target": step3_target,
                    "abs_step3_gap": abs_step3_gap,
                }
            )
            drift_rows.append(
                {
                    "proxy_source": source_name,
                    "temperature": float(selected["temperature"]),
                    "phi": float(selected["phi"]),
                    "proxy_chi_target": chi_target,
                    "step3_chi_target": step3_target,
                    "abs_step3_gap": abs_step3_gap,
                }
            )
        return pd.DataFrame(rows), pd.DataFrame(drift_rows)

    rl_proxy_df, rl_drift_df = _bucket_rows(rl_buckets, source_name="rl_proxy")
    hpo_target_df, hpo_drift_df = _bucket_rows(hpo_buckets, source_name="hpo") if hpo_enabled else (pd.DataFrame(), pd.DataFrame())
    drift_df = pd.concat([rl_drift_df, hpo_drift_df], ignore_index=True) if not rl_drift_df.empty or not hpo_drift_df.empty else pd.DataFrame()
    rl_mean_gap = float(rl_drift_df["abs_step3_gap"].mean()) if not rl_drift_df.empty else np.nan
    rl_max_gap = float(rl_drift_df["abs_step3_gap"].max()) if not rl_drift_df.empty else np.nan
    hpo_mean_gap = float(hpo_drift_df["abs_step3_gap"].mean()) if not hpo_drift_df.empty else np.nan
    hpo_max_gap = float(hpo_drift_df["abs_step3_gap"].max()) if not hpo_drift_df.empty else np.nan
    gap_threshold = 0.3
    diagnostics = {
        "eligible_validation_buckets": int(len(ordered_buckets)),
        "rl_proxy_bucket_count": int(rl_bucket_count),
        "hpo_bucket_count": int(hpo_bucket_count),
        "hpo_enabled": bool(hpo_enabled),
        "bucket_selection_policy": "evenly_spaced_disjoint_validation_buckets",
        "rl_proxy_mean_abs_step3_gap": rl_mean_gap,
        "rl_proxy_max_abs_step3_gap": rl_max_gap,
        "hpo_mean_abs_step3_gap": hpo_mean_gap,
        "hpo_max_abs_step3_gap": hpo_max_gap,
        "proxy_target_gap_warning_threshold": float(gap_threshold),
        "proxy_target_gap_warning": bool(
            (np.isfinite(rl_mean_gap) and rl_mean_gap > gap_threshold)
            or (np.isfinite(hpo_mean_gap) and hpo_mean_gap > gap_threshold)
        ),
        "coverage_warning": None,
    }
    if hpo_enabled and hpo_bucket_count < int(hpo_num_targets):
        diagnostics["coverage_warning"] = (
            f"Requested hpo_num_targets={int(hpo_num_targets)} but only {int(hpo_bucket_count)} "
            "disjoint validation buckets remained after RL proxy reservation."
        )
    return rl_proxy_df, hpo_target_df, diagnostics, drift_df


def _condition_key_set(df: pd.DataFrame) -> set[tuple[float, float]]:
    if df.empty or not {"temperature", "phi"}.issubset(df.columns):
        return set()
    return {
        (round(float(row["temperature"]), 8), round(float(row["phi"]), 8))
        for row in df[["temperature", "phi"]].dropna().to_dict(orient="records")
    }


def _build_hpo_target_overlap_diagnostics(
    *,
    target_family_df: pd.DataFrame,
    rl_proxy_df: pd.DataFrame,
    hpo_target_df: pd.DataFrame,
) -> Dict[str, Any]:
    benchmark_keys = _condition_key_set(target_family_df)
    rl_proxy_keys = _condition_key_set(rl_proxy_df)
    hpo_keys = _condition_key_set(hpo_target_df)
    hpo_benchmark_overlap = hpo_keys & benchmark_keys
    hpo_rl_proxy_overlap = hpo_keys & rl_proxy_keys
    hpo_count = int(len(hpo_keys))
    benchmark_count = int(len(benchmark_keys))
    return {
        "benchmark_condition_count": benchmark_count,
        "hpo_condition_count": hpo_count,
        "rl_proxy_condition_count": int(len(rl_proxy_keys)),
        "hpo_benchmark_overlap_count": int(len(hpo_benchmark_overlap)),
        "hpo_benchmark_overlap_fraction": (
            float(len(hpo_benchmark_overlap)) / float(hpo_count) if hpo_count else 0.0
        ),
        "hpo_rl_proxy_overlap_count": int(len(hpo_rl_proxy_overlap)),
        "hpo_targets_disjoint_from_rl_proxy": bool(len(hpo_rl_proxy_overlap) == 0),
        "target_overlap_policy": (
            "HPO targets are selected from validation buckets disjoint from the RL proxy buckets. "
            "They are not automatically removed from the final benchmark target table; overlap with "
            "benchmark conditions is documented here so small target tables are not split artificially."
        ),
        "benchmark_has_more_conditions_than_hpo": bool(benchmark_count > hpo_count),
    }


def _compute_chi_train_stats(chi_split_df: pd.DataFrame) -> Dict[str, float]:
    train_df = chi_split_df.loc[chi_split_df["split"].astype(str) == "train"].copy()
    if train_df.empty:
        raise ValueError("No train rows found in split-aware D_chi dataframe.")
    return {
        "temperature_min": float(train_df["temperature"].min()),
        "temperature_max": float(train_df["temperature"].max()),
        "phi_min": float(train_df["phi"].min()),
        "phi_max": float(train_df["phi"].max()),
        "chi_goal_min": float(train_df["chi"].min()),
        "chi_goal_max": float(train_df["chi"].max()),
    }


def _compute_class_support_stats(
    *,
    c_target: str,
    polymer_patterns: Dict[str, str],
    source_smiles: Iterable[str],
) -> Dict[str, Any]:
    source_smiles = [str(smi) for smi in source_smiles]
    classifier = PolymerClassifier(patterns=polymer_patterns)
    positive_loose = 0
    positive_strict = 0
    target_is_backbone = bool(str(c_target).strip().lower() in BACKBONE_CLASS_MATCH_CLASSES)
    for smiles in source_smiles:
        try:
            if classifier.classify(smiles).get(c_target, False):
                positive_loose += 1
            if classifier.classify_backbone(smiles).get(c_target, False):
                positive_strict += 1
        except Exception:
            continue
    return {
        "c_target": c_target,
        "target_class_backbone_defined": bool(target_is_backbone),
        "source_corpus_size": int(len(source_smiles)),
        "source_positive_count": int(positive_loose),
        "source_positive_count_loose": int(positive_loose),
        "source_positive_count_strict": int(positive_strict),
        "token_bias_available": bool(positive_loose >= 10),
        "sparse_class_warning": bool(positive_loose < 20),
        "strict_sparse_class_warning": bool(positive_strict < 20),
    }


def _validate_step5_config(
    *,
    step5_cfg: Dict[str, Any],
    polymer_patterns: Dict[str, str],
) -> None:
    split_mode = str(step5_cfg["split_mode"]).strip().lower()
    classification_split_mode = str(step5_cfg["classification_split_mode"]).strip().lower()
    if split_mode not in {"polymer", "random"}:
        raise ValueError("step5.split_mode must be one of {'polymer', 'random'}")
    if classification_split_mode not in {"polymer", "random"}:
        raise ValueError("step5.classification_split_mode must be one of {'polymer', 'random'}")

    c_target = str(step5_cfg["c_target"]).strip().lower()
    available = [str(x).strip().lower() for x in step5_cfg.get("available_target_classes", [])]
    if c_target not in available:
        raise ValueError(
            f"step5.c_target={c_target!r} is not in available_target_classes={available}"
        )
    if c_target not in polymer_patterns:
        raise ValueError(
            f"step5.c_target={c_target!r} is not defined in config.yaml polymer_classes"
        )
    seeds = list(step5_cfg["sampling_seeds"])
    if len(seeds) != int(step5_cfg["num_sampling_rounds"]):
        raise ValueError("sampling_seeds length must equal num_sampling_rounds")
    enabled_runs = list(step5_cfg["enabled_runs"])
    if not enabled_runs:
        raise ValueError("step5.enabled_runs is empty")


def build_run_config(resolved: ResolvedStep5Config, run_name: str) -> Dict[str, Any]:
    if run_name not in resolved.enabled_runs:
        raise ValueError(f"Run {run_name!r} is not enabled in step5.enabled_runs")
    run_cfg = deepcopy(resolved.step5)
    override = deepcopy(run_cfg.get("run_overrides", {}).get(run_name, {}))
    canonical_family = str(
        override.get("canonical_family", run_name.split("_", 1)[0])
    ).strip()
    run_cfg = _deep_merge(run_cfg, override)
    run_cfg["run_name"] = run_name
    run_cfg["canonical_family"] = canonical_family
    run_cfg["c_target"] = resolved.c_target
    return _apply_step5_class_overrides(run_cfg, c_target=resolved.c_target)


def _apply_step5_class_overrides(value: Any, *, c_target: str) -> Any:
    if isinstance(value, dict):
        resolved = {
            str(key): _apply_step5_class_overrides(item, c_target=c_target)
            for key, item in value.items()
        }
        for key, item in list(resolved.items()):
            if not key.endswith("_by_class_overrides") or not isinstance(item, dict):
                continue
            base_key = key[: -len("_by_class_overrides")]
            if c_target in item:
                override_value = _apply_step5_class_overrides(item[c_target], c_target=c_target)
                if isinstance(resolved.get(base_key), dict) and isinstance(override_value, dict):
                    resolved[base_key] = _deep_merge(resolved[base_key], override_value)
                else:
                    resolved[base_key] = override_value
        return resolved
    if isinstance(value, list):
        return [_apply_step5_class_overrides(item, c_target=c_target) for item in value]
    return value


def _build_snapshot(
    *,
    base_config: Dict[str, Any],
    step5_cfg: Dict[str, Any],
    step5_1_cfg: Dict[str, Any],
    hpo_cfg: Dict[str, Any],
    model_size: Optional[str],
    paths: Dict[str, Path],
    target_family_df: pd.DataFrame,
    rl_proxy_df: pd.DataFrame,
    hpo_target_df: pd.DataFrame,
    chi_train_stats: Dict[str, float],
    class_support_stats: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model_size": model_size,
        "paths": _as_serializable(paths),
        "step5": _as_serializable(step5_cfg),
        "step5_1": _as_serializable(step5_1_cfg),
        "step5_hpo": _as_serializable(hpo_cfg),
        "derived": {
            "num_target_rows": int(len(target_family_df)),
            "num_rl_proxy_rows": int(len(rl_proxy_df)),
            "num_hpo_rows": int(len(hpo_target_df)),
            "chi_train_stats": _as_serializable(chi_train_stats),
            "class_support_stats": _as_serializable(class_support_stats),
            "validation_bucket_diagnostics": _as_serializable(diagnostics),
            "base_results_dir": str(base_config["paths"]["results_dir"]),
        },
    }


def load_step5_config(
    *,
    config_path: str = "configs/config5.yaml",
    base_config_path: str = "configs/config.yaml",
    model_size: Optional[str] = None,
    force_hpo_enabled: bool = False,
    c_target_override: Optional[str] = None,
) -> ResolvedStep5Config:
    base_config = load_config(base_config_path)
    step5_bundle = _load_step5_bundle_config(config_path)
    base_config_overrides = deepcopy(step5_bundle.get("base_config_overrides", {}))
    if base_config_overrides:
        base_config = _deep_merge(base_config, base_config_overrides)
    step5_cfg = deepcopy(step5_bundle.get("step5", {}))
    step5_1_cfg = deepcopy(step5_bundle.get("step5_1", {}))
    hpo_cfg = deepcopy(step5_bundle.get("step5_hpo", {}))
    if force_hpo_enabled:
        hpo_cfg["enabled"] = True
    if not step5_cfg:
        raise ValueError(f"No step5 block found in {config_path}")
    if c_target_override:
        step5_cfg["c_target"] = str(c_target_override).strip().lower()

    polymer_patterns = {
        str(key).strip().lower(): str(value)
        for key, value in base_config.get("polymer_classes", {}).items()
    }
    _validate_step5_config(step5_cfg=step5_cfg, polymer_patterns=polymer_patterns)

    split_mode = str(step5_cfg["split_mode"]).strip().lower()
    classification_split_mode = str(step5_cfg["classification_split_mode"]).strip().lower()
    c_target = str(step5_cfg["c_target"]).strip().lower()
    step5_cfg = _apply_step5_class_overrides(step5_cfg, c_target=c_target)
    enabled_runs = [str(name) for name in step5_cfg["enabled_runs"]]
    available_target_classes = [str(x).strip().lower() for x in step5_cfg["available_target_classes"]]

    (
        results_dir,
        results_dir_nosplit,
        base_results_dir,
        step4_reg_dir,
        step4_cls_dir,
        step4_reg_metrics_dir,
        step4_cls_metrics_dir,
    ) = _resolve_step4_metrics_dirs(base_config, model_size=model_size, split_mode=split_mode)
    method_root = (
        results_dir / "step5_inverse_design" / split_mode / c_target
    )
    compare_root = (
        results_dir / "step5_1_inverse_design_compare" / split_mode / c_target
    )

    step3_targets_path, chi_lookup, target_base_df, target_family_df = _build_lookup_and_target_tables(
        base_config=base_config,
        step5_cfg=step5_cfg,
        results_dir=results_dir,
        base_results_dir=base_results_dir,
        split_mode=split_mode,
    )
    chi_split_df = _load_or_build_chi_split_df(
        base_config=base_config,
        step4_reg_dir=step4_reg_dir,
        step4_reg_metrics_dir=step4_reg_metrics_dir,
        split_mode=split_mode,
        random_seed=int(step5_cfg["random_seed"]),
    )
    rl_proxy_df, hpo_target_df, diagnostics, drift_df = _build_validation_target_tables(
        chi_split_df=chi_split_df,
        c_target=c_target,
        chi_lookup=chi_lookup,
        rl_proxy_num_targets=int(step5_cfg["s4"]["rl_proxy_num_targets"]),
        hpo_enabled=bool(hpo_cfg.get("enabled", False)),
        hpo_num_targets=int(hpo_cfg.get("hpo_num_targets", 0)),
    )
    diagnostics["hpo_target_overlap"] = _build_hpo_target_overlap_diagnostics(
        target_family_df=target_family_df,
        rl_proxy_df=rl_proxy_df,
        hpo_target_df=hpo_target_df,
    )
    chi_train_stats = _compute_chi_train_stats(chi_split_df)
    training_source_smiles = load_decode_constraint_source_smiles(Path(base_config["paths"]["data_dir"]))
    class_support_stats = _compute_class_support_stats(
        c_target=c_target,
        polymer_patterns=polymer_patterns,
        source_smiles=training_source_smiles,
    )
    if class_support_stats["sparse_class_warning"]:
        warnings.warn(
            (
                f"Low-support c_target={c_target!r}: "
                f"source_positive_count_loose={class_support_stats['source_positive_count_loose']} "
                f"source_positive_count_strict={class_support_stats['source_positive_count_strict']}"
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    paths = {
        "results_dir": results_dir,
        "results_dir_nosplit": results_dir_nosplit,
        "base_results_dir": base_results_dir,
        "method_root": method_root,
        "compare_root": compare_root,
        "step3_targets_path": step3_targets_path,
        "step4_reg_dir": step4_reg_dir,
        "step4_cls_dir": step4_cls_dir,
        "step4_reg_metrics_dir": step4_reg_metrics_dir,
        "step4_cls_metrics_dir": step4_cls_metrics_dir,
    }
    snapshot = _build_snapshot(
        base_config=base_config,
        step5_cfg=step5_cfg,
        step5_1_cfg=step5_1_cfg,
        hpo_cfg=hpo_cfg,
        model_size=model_size,
        paths=paths,
        target_family_df=target_family_df,
        rl_proxy_df=rl_proxy_df,
        hpo_target_df=hpo_target_df,
        chi_train_stats=chi_train_stats,
        class_support_stats=class_support_stats,
        diagnostics=diagnostics,
    )
    if not drift_df.empty:
        snapshot["derived"]["validation_target_drift"] = _as_serializable(
            {
                "mean_abs_step3_gap": float(drift_df["abs_step3_gap"].mean()),
                "max_abs_step3_gap": float(drift_df["abs_step3_gap"].max()),
            }
        )

    return ResolvedStep5Config(
        base_config=base_config,
        step5=step5_cfg,
        step5_1=step5_1_cfg,
        step5_hpo=hpo_cfg,
        model_size=model_size,
        split_mode=split_mode,
        classification_split_mode=classification_split_mode,
        c_target=c_target,
        enabled_runs=enabled_runs,
        available_target_classes=available_target_classes,
        polymer_patterns=polymer_patterns,
        results_dir=results_dir,
        results_dir_nosplit=results_dir_nosplit,
        base_results_dir=base_results_dir,
        method_root=method_root,
        compare_root=compare_root,
        step4_reg_dir=step4_reg_dir,
        step4_cls_dir=step4_cls_dir,
        step4_reg_metrics_dir=step4_reg_metrics_dir,
        step4_cls_metrics_dir=step4_cls_metrics_dir,
        step3_targets_path=step3_targets_path,
        chi_lookup=chi_lookup,
        target_base_df=target_base_df,
        target_family_df=target_family_df,
        rl_proxy_df=rl_proxy_df,
        hpo_target_df=hpo_target_df,
        chi_split_df=chi_split_df,
        chi_train_stats=chi_train_stats,
        class_support_stats=class_support_stats,
        config_snapshot=snapshot,
    )
