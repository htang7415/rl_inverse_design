"""Offline DPO pair construction and training for Step 5 S4_dpo."""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.tokenizer import PSmilesTokenizer
from src.utils.chemistry import canonicalize_smiles, check_validity, count_stars, has_terminal_connection_stars
from src.utils.reproducibility import seed_everything
from src.utils.reporting import append_log_message

from .conditional_sampling import create_conditional_sampler, sample_conditional_with_class_prior
from .config import ResolvedStep5Config, select_step5_proxy_target_rows
from .dataset import (
    ConditionScaler,
    build_inference_condition_bundle,
    build_inference_condition_bundle_from_target_row,
    build_step5_supervised_frames,
)
from .evaluation import build_generated_samples_frame, evaluate_generated_samples
from .frozen_sampling import ClassConstrainedSamplingQuotaError, ResolvedClassSamplingPrior
from .rewards import compute_success_shaped_rewards
from .supervised import build_optimizer_and_scheduler, load_step5_checkpoint_into_modules
from .train_s2 import S2TrainingArtifacts


COND_COL_PREFIX = "cond_"
LOGGER = logging.getLogger(__name__)


@dataclass
class DpoTrainingArtifacts:
    """Artifacts returned by Step 5 S4_dpo alignment."""

    tokenizer: PSmilesTokenizer
    policy_model: torch.nn.Module
    reference_model: torch.nn.Module
    checkpoint_path: Path
    last_checkpoint_path: Path
    scaler: ConditionScaler
    train_pairs_df: pd.DataFrame
    val_pairs_df: pd.DataFrame
    history_df: pd.DataFrame


class Step5DpoDataset(Dataset):
    """Offline DPO pair dataset."""

    def __init__(self, df: pd.DataFrame, *, tokenizer: PSmilesTokenizer):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.cond_cols = _resolve_condition_columns(self.df)
        if not self.cond_cols:
            raise ValueError("Step 5 DPO dataset is missing condition bundle columns.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        chosen = self.tokenizer.encode(
            str(row["chosen_smiles"]),
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True,
        )
        rejected = self.tokenizer.encode(
            str(row["rejected_smiles"]),
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True,
        )
        return {
            "condition_bundle": torch.tensor(row[self.cond_cols].to_numpy(dtype=np.float32), dtype=torch.float32),
            "chosen_input_ids": torch.tensor(chosen["input_ids"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen["attention_mask"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected["input_ids"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected["attention_mask"], dtype=torch.long),
        }


def step5_dpo_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in batch[0].keys()}


def _stable_hash_hex(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _canonical_sort_key(smiles: str) -> str:
    canonical = canonicalize_smiles(str(smiles))
    return canonical if canonical else str(smiles)


def _resolve_condition_columns(df: pd.DataFrame) -> List[str]:
    cond_cols = [str(column) for column in df.columns if str(column).startswith(COND_COL_PREFIX)]
    return sorted(cond_cols, key=lambda column: int(str(column).split("_", 1)[1]))


def _bundle_to_row(bundle: np.ndarray) -> Dict[str, float]:
    bundle_arr = np.asarray(bundle, dtype=np.float32).reshape(-1)
    return {f"{COND_COL_PREFIX}{idx}": float(bundle_arr[idx]) for idx in range(int(bundle_arr.shape[0]))}


def _build_pair_condition_bundle(
    *,
    temperature: float,
    phi: float,
    chi_goal: float,
    property_rule: str,
    chi_goal_lower: float,
    chi_goal_upper: float,
    scaler: ConditionScaler,
    chosen_row: pd.Series | Dict[str, Any],
) -> np.ndarray:
    del property_rule, chi_goal_lower, chi_goal_upper, chosen_row
    bundle = build_inference_condition_bundle(
        temperature=float(temperature),
        phi=float(phi),
        chi_goal=float(chi_goal),
        scaler=scaler,
        soluble=1,
    )
    return bundle


def _compute_star_ok(smiles: str) -> int:
    smiles = str(smiles)
    return int(
        check_validity(smiles)
        and count_stars(smiles) == 2
        and has_terminal_connection_stars(smiles, expected_stars=2)
    )


def _filter_star_ok_rows(df: pd.DataFrame, *, smiles_col: str = "SMILES") -> Tuple[pd.DataFrame, Dict[str, int]]:
    if df.empty:
        return df.copy(), {"input_rows": 0, "star_ok_rows": 0}
    work = df.copy()
    work["_star_ok"] = work[smiles_col].astype(str).map(_compute_star_ok).astype(int)
    filtered = work.loc[work["_star_ok"] == 1].drop(columns=["_star_ok"]).reset_index(drop=True)
    return filtered, {"input_rows": int(len(df)), "star_ok_rows": int(len(filtered))}


def _base_pair_record(
    *,
    source_name: str,
    bucket_key: str,
    prompt_temperature: float,
    prompt_phi: float,
    prompt_chi_goal: float,
    condition_bundle: np.ndarray,
    chosen_row: pd.Series,
    rejected_row: pd.Series,
) -> Dict[str, Any]:
    chosen_sort = _canonical_sort_key(str(chosen_row["SMILES"]))
    rejected_sort = _canonical_sort_key(str(rejected_row["SMILES"]))
    pair_key = (
        f"{source_name}|{bucket_key}|"
        f"{chosen_sort}|{int(chosen_row['row_id'])}|"
        f"{rejected_sort}|{int(rejected_row['row_id'])}"
    )
    return {
        "source_name": source_name,
        "bucket_key": bucket_key,
        "prompt_temperature": float(prompt_temperature) if np.isfinite(prompt_temperature) else np.nan,
        "prompt_phi": float(prompt_phi) if np.isfinite(prompt_phi) else np.nan,
        "prompt_chi_goal": float(prompt_chi_goal) if np.isfinite(prompt_chi_goal) else np.nan,
        "chosen_row_id": int(chosen_row["row_id"]),
        "rejected_row_id": int(rejected_row["row_id"]),
        "chosen_polymer": str(chosen_row["Polymer"]),
        "rejected_polymer": str(rejected_row["Polymer"]),
        "chosen_smiles": str(chosen_row["SMILES"]),
        "rejected_smiles": str(rejected_row["SMILES"]),
        "chosen_canonical_smiles": chosen_sort,
        "rejected_canonical_smiles": rejected_sort,
        "chosen_water_miscible": int(chosen_row["water_miscible"]),
        "rejected_water_miscible": int(rejected_row["water_miscible"]),
        "chosen_chi_observed": float(chosen_row["chi"]) if np.isfinite(chosen_row["chi"]) else np.nan,
        "rejected_chi_observed": float(rejected_row["chi"]) if np.isfinite(rejected_row["chi"]) else np.nan,
        "pair_key": pair_key,
        "pair_hash": _stable_hash_hex(pair_key),
        **_bundle_to_row(condition_bundle),
    }


def _build_d_water_pairs(
    train_d_water: pd.DataFrame,
    *,
    scaler: ConditionScaler,
    max_pairs: Optional[int] = None,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    filtered, star_stats = _filter_star_ok_rows(train_d_water)
    positives = filtered.loc[filtered["water_miscible"].astype(int) == 1].copy()
    negatives = filtered.loc[filtered["water_miscible"].astype(int) == 0].copy()
    if positives.empty or negatives.empty:
        return pd.DataFrame(), star_stats

    positives["canonical_smiles"] = positives["SMILES"].astype(str).map(_canonical_sort_key)
    negatives["canonical_smiles"] = negatives["SMILES"].astype(str).map(_canonical_sort_key)
    positives = positives.sort_values(["canonical_smiles", "Polymer", "row_id"]).reset_index(drop=True)
    negatives = negatives.sort_values(["canonical_smiles", "Polymer", "row_id"]).reset_index(drop=True)

    possible_pair_count = int(len(positives) * len(negatives))
    pair_limit = possible_pair_count if max_pairs is None else max(0, min(int(max_pairs), possible_pair_count))
    star_stats = {
        **star_stats,
        "possible_pair_count": possible_pair_count,
        "selected_pair_count": int(pair_limit),
        "sampled_pair_selection": int(pair_limit < possible_pair_count),
    }
    if pair_limit <= 0:
        return pd.DataFrame(), star_stats

    if pair_limit < possible_pair_count:
        rng = np.random.default_rng(int(random_seed))
        flat_indices = np.sort(
            rng.choice(possible_pair_count, size=int(pair_limit), replace=False)
        )
    else:
        flat_indices = np.arange(possible_pair_count, dtype=int)

    rows: List[Dict[str, Any]] = []
    n_negatives = int(len(negatives))
    condition_bundle = _build_pair_condition_bundle(
        temperature=np.nan,
        phi=np.nan,
        chi_goal=np.nan,
        property_rule="upper_bound",
        chi_goal_lower=np.nan,
        chi_goal_upper=np.nan,
        scaler=scaler,
        chosen_row=positives.iloc[0],
    )
    for flat_idx in flat_indices.tolist():
        chosen_idx = int(flat_idx // n_negatives)
        rejected_idx = int(flat_idx % n_negatives)
        chosen_row = positives.iloc[chosen_idx]
        rejected_row = negatives.iloc[rejected_idx]
        rows.append(
            _base_pair_record(
                source_name="d_water",
                bucket_key="d_water|missing_fields",
                prompt_temperature=np.nan,
                prompt_phi=np.nan,
                prompt_chi_goal=np.nan,
                condition_bundle=condition_bundle,
                chosen_row=chosen_row,
                rejected_row=rejected_row,
            )
        )
    if not rows:
        return pd.DataFrame(), star_stats
    pair_df = pd.DataFrame(rows).sort_values(
        ["source_name", "bucket_key", "chosen_canonical_smiles", "rejected_canonical_smiles", "chosen_row_id", "rejected_row_id"]
    ).reset_index(drop=True)
    return pair_df, star_stats


def _build_d_chi_pairs(
    train_d_chi: pd.DataFrame,
    *,
    scaler: ConditionScaler,
    chi_lookup,
    pair_source: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    work, star_stats = _filter_star_ok_rows(train_d_chi)
    work["canonical_smiles"] = work["SMILES"].astype(str).map(_canonical_sort_key)
    rows: List[Dict[str, Any]] = []
    gap_rows: List[Dict[str, Any]] = []

    for (temperature, phi), bucket in work.groupby(["temperature", "phi"], dropna=False):
        if not np.isfinite(temperature) or not np.isfinite(phi):
            continue
        chi_target_row = chi_lookup.lookup_row(float(temperature), float(phi), warn_on_missing=False)
        if chi_target_row is None:
            continue
        chi_target = float(chi_target_row["chi_target"])
        property_rule = str(chi_target_row.get("property_rule", "upper_bound"))
        chi_target_lower = pd.to_numeric(chi_target_row.get("chi_target_boot_q025", np.nan), errors="coerce")
        chi_target_upper = pd.to_numeric(chi_target_row.get("chi_target_boot_q975", np.nan), errors="coerce")
        bucket = bucket.sort_values(["canonical_smiles", "Polymer", "row_id"]).reset_index(drop=True)
        if pair_source == "label_water_miscibility":
            chosen_df = bucket.loc[bucket["water_miscible"].astype(int) == 1].copy()
            rejected_df = bucket.loc[bucket["water_miscible"].astype(int) == 0].copy()
        elif pair_source == "chi_aware_label_bucketed":
            chosen_mask = (bucket["water_miscible"].astype(int) == 1) & (
                bucket["chi"].astype(float) < float(chi_target)
            )
            chosen_df = bucket.loc[chosen_mask].copy()
            rejected_df = bucket.loc[~chosen_mask].copy()
        else:
            raise NotImplementedError(f"Unsupported Step 5 DPO pair source: {pair_source}")

        if chosen_df.empty or rejected_df.empty:
            continue

        for _, row in bucket.iterrows():
            chi_observed = float(row["chi"]) if np.isfinite(row["chi"]) else np.nan
            gap_rows.append(
                {
                    "temperature": float(temperature),
                    "phi": float(phi),
                    "chi_target": float(chi_target),
                    "row_id": int(row["row_id"]),
                    "water_miscible": int(row["water_miscible"]),
                    "chi_observed": chi_observed,
                    "abs_prompt_gap": abs(float(chi_target) - chi_observed) if np.isfinite(chi_observed) else np.nan,
                }
            )

        bucket_key = f"d_chi|T={float(temperature):.6f}|phi={float(phi):.6f}"
        for _, chosen_row in chosen_df.iterrows():
            condition_bundle = _build_pair_condition_bundle(
                temperature=float(temperature),
                phi=float(phi),
                chi_goal=float(chi_target),
                property_rule=property_rule,
                chi_goal_lower=chi_target_lower,
                chi_goal_upper=chi_target_upper,
                scaler=scaler,
                chosen_row=chosen_row,
            )
            for _, rejected_row in rejected_df.iterrows():
                rows.append(
                    _base_pair_record(
                        source_name="d_chi",
                        bucket_key=bucket_key,
                        prompt_temperature=float(temperature),
                        prompt_phi=float(phi),
                        prompt_chi_goal=float(chi_target),
                        condition_bundle=condition_bundle,
                        chosen_row=chosen_row,
                        rejected_row=rejected_row,
                    )
                )

    pair_df = pd.DataFrame(rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(
            [
                "source_name",
                "bucket_key",
                "chosen_canonical_smiles",
                "rejected_canonical_smiles",
                "chosen_row_id",
                "rejected_row_id",
            ]
        ).reset_index(drop=True)
    gap_df = pd.DataFrame(gap_rows)
    return pair_df, gap_df, star_stats


def _synthetic_pair_record(
    *,
    target_row: pd.Series,
    condition_bundle: np.ndarray,
    chosen_row: pd.Series,
    rejected_row: pd.Series,
) -> Dict[str, Any]:
    chosen_smiles = str(chosen_row["smiles"])
    rejected_smiles = str(rejected_row["smiles"])
    chosen_sort = str(chosen_row.get("canonical_smiles") or _canonical_sort_key(chosen_smiles))
    rejected_sort = str(rejected_row.get("canonical_smiles") or _canonical_sort_key(rejected_smiles))
    pair_key = (
        f"target_row_synthetic|{str(target_row['target_row_key'])}|"
        f"{chosen_sort}|{int(chosen_row['sample_id'])}|"
        f"{rejected_sort}|{int(rejected_row['sample_id'])}"
    )
    return {
        "source_name": "target_row_synthetic",
        "bucket_key": str(target_row["target_row_key"]),
        "prompt_temperature": float(target_row["temperature"]),
        "prompt_phi": float(target_row["phi"]),
        "prompt_chi_goal": float(target_row["chi_target"]),
        "chosen_row_id": int(chosen_row["sample_id"]),
        "rejected_row_id": int(rejected_row["sample_id"]),
        "chosen_polymer": str(chosen_row.get("smiles", "")),
        "rejected_polymer": str(rejected_row.get("smiles", "")),
        "chosen_smiles": chosen_smiles,
        "rejected_smiles": rejected_smiles,
        "chosen_canonical_smiles": chosen_sort,
        "rejected_canonical_smiles": rejected_sort,
        "chosen_water_miscible": int(chosen_row.get("soluble_ok", 0)),
        "rejected_water_miscible": int(rejected_row.get("soluble_ok", 0)),
        "chosen_chi_observed": float(chosen_row.get("chi_pred_target", np.nan)),
        "rejected_chi_observed": float(rejected_row.get("chi_pred_target", np.nan)),
        "chosen_reward_score": float(chosen_row["reward_score"]),
        "rejected_reward_score": float(rejected_row["reward_score"]),
        "pair_key": pair_key,
        "pair_hash": _stable_hash_hex(pair_key),
        **_bundle_to_row(condition_bundle),
    }


def _build_target_row_synthetic_pairs(
    *,
    resolved: ResolvedStep5Config,
    run_cfg: Dict[str, object],
    warm_start: S2TrainingArtifacts,
    prior: ResolvedClassSamplingPrior,
    evaluator,
    device: str,
    metrics_dir: Path,
    target_rows_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dpo_cfg = dict(run_cfg["s4"]["dpo"])
    synthetic_per_target = int(dpo_cfg["synthetic_candidates_per_target"])
    reward_weights = dict(run_cfg["s4"]["reward_weights"])
    candidate_frames: List[pd.DataFrame] = []
    pair_rows: List[Dict[str, Any]] = []
    sample_id_start = 1
    base_seed = int(resolved.step5["random_seed"])

    pair_target_df = (
        target_rows_df.copy().reset_index(drop=True)
        if target_rows_df is not None and not target_rows_df.empty
        else resolved.target_family_df.copy().reset_index(drop=True)
    )

    for target_offset, (_, target_row) in enumerate(pair_target_df.iterrows()):
        seed_everything(base_seed + 10_000 + target_offset, deterministic=True)
        condition_bundle = torch.tensor(
            build_inference_condition_bundle_from_target_row(
                target_row.to_dict(),
                scaler=warm_start.scaler,
                soluble=1,
            ),
            dtype=torch.float32,
            device=device,
        )
        sampler = create_conditional_sampler(
            diffusion_model=warm_start.diffusion_model,
            tokenizer=warm_start.tokenizer,
            resolved=resolved,
            prior=prior,
            condition_bundle=condition_bundle,
            cfg_scale=float(run_cfg["s4"]["cfg_scale"]),
            device=device,
        )
        smiles, _sample_meta = sample_conditional_with_class_prior(
            sampler=sampler,
            tokenizer=warm_start.tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=synthetic_per_target,
            show_progress=False,
        )
        sample_df = build_generated_samples_frame(
            smiles,
            target_row=target_row,
            round_id=0,
            sampling_seed=base_seed + 10_000 + target_offset,
            run_name=str(run_cfg["run_name"]),
            canonical_family=str(run_cfg["canonical_family"]),
            sample_id_start=sample_id_start,
        )
        sample_id_start += len(sample_df)
        eval_df = evaluate_generated_samples(sample_df, evaluator).copy()
        reward_tensor, _reward_metrics = compute_success_shaped_rewards(
            eval_df,
            reward_weights=reward_weights,
            sol_log_prob_floor=float(run_cfg["s4"]["sol_log_prob_floor"]),
            reward_shaping=run_cfg["s4"].get("reward_shaping", {}),
        )
        eval_df["reward_score"] = reward_tensor.detach().cpu().numpy()
        eval_df["reward_rank_key"] = (
            -pd.to_numeric(eval_df["reward_score"], errors="coerce").fillna(-1.0e9)
        )
        eval_df = eval_df.sort_values(
            ["reward_rank_key", "canonical_smiles", "sample_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        eval_df["synthetic_rank"] = np.arange(1, len(eval_df) + 1, dtype=int)
        candidate_frames.append(eval_df)

        bundle_np = build_inference_condition_bundle_from_target_row(
            target_row.to_dict(),
            scaler=warm_start.scaler,
            soluble=1,
        )
        for chosen_idx in range(len(eval_df)):
            chosen_row = eval_df.iloc[chosen_idx]
            for rejected_idx in range(chosen_idx + 1, len(eval_df)):
                rejected_row = eval_df.iloc[rejected_idx]
                if float(chosen_row["reward_score"]) <= float(rejected_row["reward_score"]):
                    continue
                pair_rows.append(
                    _synthetic_pair_record(
                        target_row=target_row,
                        condition_bundle=bundle_np,
                        chosen_row=chosen_row,
                        rejected_row=rejected_row,
                    )
                )

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(
            [
                "source_name",
                "bucket_key",
                "chosen_reward_score",
                "rejected_reward_score",
                "chosen_canonical_smiles",
                "rejected_canonical_smiles",
                "chosen_row_id",
                "rejected_row_id",
            ],
            ascending=[True, True, False, True, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    candidates_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    if not candidates_df.empty:
        candidates_df.to_csv(metrics_dir / "dpo_synthetic_candidate_scores.csv", index=False)
    return pair_df, candidates_df


def _allocate_bucket_counts(total_budget: int, available_counts: Dict[str, int]) -> Dict[str, int]:
    if total_budget <= 0 or not available_counts:
        return {bucket_key: 0 for bucket_key in available_counts}
    bucket_keys = sorted(available_counts)
    base = total_budget // len(bucket_keys)
    remainder = total_budget % len(bucket_keys)
    counts = {}
    for idx, bucket_key in enumerate(bucket_keys):
        counts[bucket_key] = min(int(available_counts[bucket_key]), base + (1 if idx < remainder else 0))
    return counts


def _take_budgeted_pairs(
    *,
    d_water_pairs: pd.DataFrame,
    d_chi_pairs: pd.DataFrame,
    offline_pair_budget: int,
    d_water_budget_fraction: float,
) -> pd.DataFrame:
    budget = max(0, int(offline_pair_budget))
    if budget == 0:
        return pd.DataFrame(columns=list(d_water_pairs.columns if not d_water_pairs.empty else d_chi_pairs.columns))

    selected_parts: List[pd.DataFrame] = []
    remaining_parts: List[pd.DataFrame] = []
    water_available = int(len(d_water_pairs))
    chi_available = int(len(d_chi_pairs))
    water_fraction = float(np.clip(float(d_water_budget_fraction), 0.0, 1.0))

    if water_available > 0 and chi_available > 0:
        water_target = int(np.floor(float(budget) * water_fraction))
        water_target = max(0, min(int(budget), water_target))
        chi_target = budget - water_target
    elif chi_available > 0:
        water_target = 0
        chi_target = budget
    else:
        water_target = budget
        chi_target = 0

    if water_available > 0:
        water_take = min(water_available, water_target)
        selected_parts.append(d_water_pairs.iloc[:water_take].copy())
        if water_take < water_available:
            remaining_parts.append(d_water_pairs.iloc[water_take:].copy())

    if chi_available > 0:
        chi_groups = {
            bucket_key: bucket_df.reset_index(drop=True)
            for bucket_key, bucket_df in d_chi_pairs.groupby("bucket_key", sort=True)
        }
        allocated = _allocate_bucket_counts(chi_target, {k: len(v) for k, v in chi_groups.items()})
        for bucket_key in sorted(chi_groups):
            bucket_df = chi_groups[bucket_key]
            take = min(len(bucket_df), int(allocated.get(bucket_key, 0)))
            if take > 0:
                selected_parts.append(bucket_df.iloc[:take].copy())
            if take < len(bucket_df):
                remaining_parts.append(bucket_df.iloc[take:].copy())

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    selected_count = int(len(selected))
    if selected_count < budget and remaining_parts:
        remaining = pd.concat(remaining_parts, ignore_index=True)
        remaining = remaining.sort_values(
            ["source_name", "bucket_key", "chosen_canonical_smiles", "rejected_canonical_smiles", "chosen_row_id", "rejected_row_id"]
        ).reset_index(drop=True)
        backfill = remaining.iloc[: max(0, budget - selected_count)].copy()
        selected = pd.concat([selected, backfill], ignore_index=True)

    if not selected.empty:
        selected = selected.drop_duplicates(subset=["pair_key"]).sort_values(
            ["source_name", "bucket_key", "chosen_canonical_smiles", "rejected_canonical_smiles", "chosen_row_id", "rejected_row_id"]
        ).reset_index(drop=True)
    return selected


def _take_budgeted_single_source_pairs(
    *,
    pair_df: pd.DataFrame,
    offline_pair_budget: int,
) -> pd.DataFrame:
    budget = max(0, int(offline_pair_budget))
    if budget == 0 or pair_df.empty:
        return pair_df.iloc[:0].copy()

    groups = {
        bucket_key: bucket.reset_index(drop=True)
        for bucket_key, bucket in pair_df.groupby("bucket_key", sort=True)
    }
    allocated = _allocate_bucket_counts(budget, {bucket_key: len(bucket) for bucket_key, bucket in groups.items()})
    selected_parts: List[pd.DataFrame] = []
    remaining_parts: List[pd.DataFrame] = []
    for bucket_key in sorted(groups):
        bucket_df = groups[bucket_key]
        take = min(len(bucket_df), int(allocated.get(bucket_key, 0)))
        if take > 0:
            selected_parts.append(bucket_df.iloc[:take].copy())
        if take < len(bucket_df):
            remaining_parts.append(bucket_df.iloc[take:].copy())

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if len(selected) < budget and remaining_parts:
        remaining = pd.concat(remaining_parts, ignore_index=True)
        remaining = remaining.sort_values(
            [
                "source_name",
                "bucket_key",
                "chosen_reward_score",
                "rejected_reward_score",
                "chosen_canonical_smiles",
                "rejected_canonical_smiles",
                "chosen_row_id",
                "rejected_row_id",
            ],
            ascending=[True, True, False, True, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        selected = pd.concat([selected, remaining.iloc[: max(0, budget - len(selected))].copy()], ignore_index=True)

    if not selected.empty:
        selected = selected.drop_duplicates(subset=["pair_key"]).sort_values(
            [
                "source_name",
                "bucket_key",
                "chosen_reward_score",
                "rejected_reward_score",
                "chosen_canonical_smiles",
                "rejected_canonical_smiles",
                "chosen_row_id",
                "rejected_row_id",
            ],
            ascending=[True, True, False, True, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return selected


def _stable_val_mask(df: pd.DataFrame, *, val_fraction: float) -> pd.Series:
    if df.empty or val_fraction <= 0.0:
        return pd.Series(False, index=df.index)
    marks = pd.Series(False, index=df.index)
    for _, bucket_df in df.groupby(["source_name", "bucket_key"], sort=True):
        bucket_size = int(len(bucket_df))
        if bucket_size <= 1:
            continue
        n_val = int(round(bucket_size * float(val_fraction)))
        n_val = min(max(1, n_val), bucket_size - 1)
        ranked = bucket_df.assign(_hash_rank=bucket_df["pair_hash"]).sort_values("_hash_rank")
        marks.loc[ranked.index[:n_val]] = True
    return marks


def _unique_ratio(unique_count: int, total_count: int) -> float:
    if int(total_count) <= 0:
        return 0.0
    return float(unique_count) / float(total_count)


def _pair_source_bucket_counts(df: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    columns = [
        "source_name",
        "bucket_key",
        f"{prefix}_pair_count",
        f"{prefix}_chosen_unique_count",
        f"{prefix}_rejected_unique_count",
        f"{prefix}_chosen_unique_ratio",
        f"{prefix}_rejected_unique_ratio",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        df.groupby(["source_name", "bucket_key"], as_index=False)
        .agg(
            **{
                f"{prefix}_pair_count": ("pair_key", "nunique"),
                f"{prefix}_chosen_unique_count": ("chosen_canonical_smiles", "nunique"),
                f"{prefix}_rejected_unique_count": ("rejected_canonical_smiles", "nunique"),
            }
        )
        .sort_values(["source_name", "bucket_key"])
        .reset_index(drop=True)
    )
    grouped[f"{prefix}_chosen_unique_ratio"] = [
        _unique_ratio(unique_count, total_count)
        for unique_count, total_count in zip(
            grouped[f"{prefix}_chosen_unique_count"],
            grouped[f"{prefix}_pair_count"],
        )
    ]
    grouped[f"{prefix}_rejected_unique_ratio"] = [
        _unique_ratio(unique_count, total_count)
        for unique_count, total_count in zip(
            grouped[f"{prefix}_rejected_unique_count"],
            grouped[f"{prefix}_pair_count"],
        )
    ]
    return grouped[columns]


def _build_dpo_pair_source_bucket_diagnostics(
    *,
    available_pairs: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
) -> pd.DataFrame:
    diagnostics = _pair_source_bucket_counts(available_pairs, prefix="raw")
    for frame in (
        _pair_source_bucket_counts(selected_pairs, prefix="selected"),
        _pair_source_bucket_counts(train_pairs, prefix="train"),
        _pair_source_bucket_counts(val_pairs, prefix="val"),
    ):
        diagnostics = diagnostics.merge(frame, on=["source_name", "bucket_key"], how="outer")
    if diagnostics.empty:
        return diagnostics
    count_cols = [col for col in diagnostics.columns if col.endswith("_count")]
    ratio_cols = [col for col in diagnostics.columns if col.endswith("_ratio")]
    diagnostics[count_cols] = diagnostics[count_cols].fillna(0).astype(int)
    diagnostics[ratio_cols] = diagnostics[ratio_cols].fillna(0.0).astype(float)
    return diagnostics.sort_values(["source_name", "bucket_key"]).reset_index(drop=True)


def build_dpo_pair_splits(
    *,
    resolved: ResolvedStep5Config,
    run_cfg: Dict[str, object],
    scaler: ConditionScaler,
    metrics_dir: Path,
    warm_start: Optional[S2TrainingArtifacts] = None,
    prior: Optional[ResolvedClassSamplingPrior] = None,
    evaluator=None,
    device: Optional[str] = None,
    target_rows_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Construct offline DPO pair splits for the configured pair source."""

    dpo_cfg = dict(run_cfg["s4"]["dpo"])
    pair_source = str(dpo_cfg["pair_source"]).strip().lower()
    candidate_df = pd.DataFrame()
    chi_prompt_gap_df = pd.DataFrame()
    d_water_star_stats = {"input_rows": 0, "star_ok_rows": 0}
    d_chi_star_stats = {"input_rows": 0, "star_ok_rows": 0}
    offline_pair_budget = int(dpo_cfg["offline_pair_budget"])
    available_pair_parts: List[pd.DataFrame] = []
    if pair_source in {"target_row_synthetic", "chi_aware_plus_target_row_synthetic"}:
        if warm_start is None or prior is None or evaluator is None or not device:
            raise ValueError(
                "target_row_synthetic DPO pairs require warm_start, prior, evaluator, and device."
            )
        synthetic_pairs, candidate_df = _build_target_row_synthetic_pairs(
            resolved=resolved,
            run_cfg=run_cfg,
            warm_start=warm_start,
            prior=prior,
            evaluator=evaluator,
            device=device,
            metrics_dir=metrics_dir,
            target_rows_df=target_rows_df,
        )
        if not synthetic_pairs.empty:
            available_pair_parts.append(synthetic_pairs)
        selected_synthetic_pairs = _take_budgeted_single_source_pairs(
            pair_df=synthetic_pairs,
            offline_pair_budget=(
                offline_pair_budget
                if pair_source == "target_row_synthetic"
                else max(0, offline_pair_budget - max(1, offline_pair_budget // 10))
            ),
        )
        if pair_source == "target_row_synthetic":
            selected_pairs = selected_synthetic_pairs
        else:
            frames = build_step5_supervised_frames(resolved)
            d_water_pairs, d_water_star_stats = _build_d_water_pairs(
                frames["train_d_water"],
                scaler=scaler,
                max_pairs=max(0, offline_pair_budget - int(len(selected_synthetic_pairs))),
                random_seed=int(resolved.step5["random_seed"]),
            )
            if not d_water_pairs.empty:
                available_pair_parts.append(d_water_pairs)
            d_chi_pairs, chi_prompt_gap_df, d_chi_star_stats = _build_d_chi_pairs(
                frames["train_d_chi"],
                scaler=scaler,
                chi_lookup=resolved.chi_lookup,
                pair_source="chi_aware_label_bucketed",
            )
            if not d_chi_pairs.empty:
                available_pair_parts.append(d_chi_pairs)
            selected_static_pairs = _take_budgeted_pairs(
                d_water_pairs=d_water_pairs,
                d_chi_pairs=d_chi_pairs,
                offline_pair_budget=max(0, offline_pair_budget - int(len(selected_synthetic_pairs))),
                d_water_budget_fraction=float(dpo_cfg.get("d_water_budget_fraction", 0.5)),
            )
            selected_pairs = pd.concat(
                [selected_synthetic_pairs, selected_static_pairs],
                ignore_index=True,
                sort=False,
            )
    else:
        frames = build_step5_supervised_frames(resolved)
        d_water_pairs, d_water_star_stats = _build_d_water_pairs(
            frames["train_d_water"],
            scaler=scaler,
            max_pairs=offline_pair_budget,
            random_seed=int(resolved.step5["random_seed"]),
        )
        if not d_water_pairs.empty:
            available_pair_parts.append(d_water_pairs)
        d_chi_pair_source = pair_source if pair_source == "chi_aware_label_bucketed" else "label_water_miscibility"
        d_chi_pairs, chi_prompt_gap_df, d_chi_star_stats = _build_d_chi_pairs(
            frames["train_d_chi"],
            scaler=scaler,
            chi_lookup=resolved.chi_lookup,
            pair_source=d_chi_pair_source,
        )
        if not d_chi_pairs.empty:
            available_pair_parts.append(d_chi_pairs)
        selected_pairs = _take_budgeted_pairs(
            d_water_pairs=d_water_pairs,
            d_chi_pairs=d_chi_pairs,
            offline_pair_budget=offline_pair_budget,
            d_water_budget_fraction=float(dpo_cfg.get("d_water_budget_fraction", 0.5)),
        )
    if selected_pairs.empty:
        raise ValueError("No Step 5 DPO pairs could be constructed from the current training split.")

    selected_pairs = selected_pairs.reset_index(drop=True)
    selected_pairs["pair_id"] = np.arange(1, len(selected_pairs) + 1, dtype=int)
    val_mask = _stable_val_mask(selected_pairs, val_fraction=float(dpo_cfg["val_pair_fraction"]))
    train_pairs = selected_pairs.loc[~val_mask].reset_index(drop=True)
    val_pairs = selected_pairs.loc[val_mask].reset_index(drop=True)
    available_pairs = (
        pd.concat(available_pair_parts, ignore_index=True, sort=False)
        if available_pair_parts
        else selected_pairs.iloc[:0].copy()
    )

    pair_summary = (
        selected_pairs.groupby(["source_name", "bucket_key"], as_index=False)
        .size()
        .rename(columns={"size": "pair_count"})
        .sort_values(["source_name", "bucket_key"])
        .reset_index(drop=True)
    )
    pair_summary["train_pair_count"] = pair_summary["bucket_key"].map(
        train_pairs.groupby("bucket_key").size().to_dict()
    ).fillna(0).astype(int)
    pair_summary["val_pair_count"] = pair_summary["bucket_key"].map(
        val_pairs.groupby("bucket_key").size().to_dict()
    ).fillna(0).astype(int)
    pair_source_bucket_diagnostics = _build_dpo_pair_source_bucket_diagnostics(
        available_pairs=available_pairs,
        selected_pairs=selected_pairs,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
    )

    selected_pairs.to_csv(metrics_dir / "dpo_pairs_selected.csv", index=False)
    train_pairs.to_csv(metrics_dir / "dpo_pairs_train.csv", index=False)
    val_pairs.to_csv(metrics_dir / "dpo_pairs_val.csv", index=False)
    pair_summary.to_csv(metrics_dir / "dpo_pair_counts_by_source_bucket.csv", index=False)
    pair_source_bucket_diagnostics.to_csv(
        metrics_dir / "dpo_pair_source_bucket_diagnostics.csv",
        index=False,
    )
    if not chi_prompt_gap_df.empty:
        chi_prompt_gap_df.to_csv(metrics_dir / "dpo_prompt_gap_d_chi_rows.csv", index=False)
    if not candidate_df.empty:
        candidate_df.to_csv(metrics_dir / "dpo_synthetic_candidates_selected.csv", index=False)

    realized_total = int(len(selected_pairs))
    raw_total_pairs = int(available_pairs["pair_key"].nunique()) if not available_pairs.empty else 0
    selected_chosen_unique_count = (
        int(selected_pairs["chosen_canonical_smiles"].nunique()) if not selected_pairs.empty else 0
    )
    selected_rejected_unique_count = (
        int(selected_pairs["rejected_canonical_smiles"].nunique()) if not selected_pairs.empty else 0
    )
    diagnostics = {
        "pair_source": pair_source,
        "offline_pair_budget": offline_pair_budget,
        "raw_total_pairs": raw_total_pairs,
        "d_water_budget_fraction": float(dpo_cfg.get("d_water_budget_fraction", 0.5)),
        "realized_total_pairs": realized_total,
        "realized_train_pairs": int(len(train_pairs)),
        "realized_val_pairs": int(len(val_pairs)),
        "selected_chosen_unique_count": selected_chosen_unique_count,
        "selected_rejected_unique_count": selected_rejected_unique_count,
        "selected_chosen_unique_ratio": _unique_ratio(selected_chosen_unique_count, realized_total),
        "selected_rejected_unique_ratio": _unique_ratio(selected_rejected_unique_count, realized_total),
        "shortfall_pairs": max(0, offline_pair_budget - realized_total),
        "thin_signal_warning": bool(realized_total < 500),
        "realized_d_water_pairs": int((selected_pairs["source_name"] == "d_water").sum()),
        "realized_d_chi_pairs": int((selected_pairs["source_name"] == "d_chi").sum()),
        "realized_target_row_synthetic_pairs": int((selected_pairs["source_name"] == "target_row_synthetic").sum()),
        "synthetic_candidate_count": int(len(candidate_df)),
        "star_ok_pair_filter_enabled": bool(pair_source != "target_row_synthetic"),
        "d_water_input_rows": int(d_water_star_stats["input_rows"]),
        "d_water_star_ok_rows": int(d_water_star_stats["star_ok_rows"]),
        "d_chi_input_rows": int(d_chi_star_stats["input_rows"]),
        "d_chi_star_ok_rows": int(d_chi_star_stats["star_ok_rows"]),
    }
    if diagnostics["shortfall_pairs"] > 0:
        LOGGER.warning(
            "Step 5 DPO pair budget shortfall: requested=%s realized=%s shortfall=%s",
            offline_pair_budget,
            realized_total,
            diagnostics["shortfall_pairs"],
        )
    if diagnostics["thin_signal_warning"]:
        LOGGER.warning(
            "Step 5 DPO thin alignment signal: realized_total_pairs=%s (< 500).",
            realized_total,
        )
    append_log_message(
        metrics_dir.parent,
        (
            "DPO pair diagnostics | "
            f"pair_source={pair_source} raw_pairs={raw_total_pairs} "
            f"selected_pairs={realized_total} buckets={int(len(pair_source_bucket_diagnostics))} "
            f"selected_chosen_unique_ratio={diagnostics['selected_chosen_unique_ratio']:.4f} "
            f"selected_rejected_unique_ratio={diagnostics['selected_rejected_unique_ratio']:.4f}"
        ),
        echo=True,
    )
    with open(metrics_dir / "dpo_pair_construction_summary.json", "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    return train_pairs, val_pairs, pair_summary, diagnostics


def _single_step_logprob_t1(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    condition_bundle: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    batch_size = int(input_ids.shape[0])
    timesteps = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    logits = model.classifier_free_guidance_logits_impl(
        input_ids,
        timesteps,
        attention_mask,
        condition_bundle=condition_bundle,
        cfg_scale=float(cfg_scale),
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    active_mask = attention_mask.bool() & (input_ids != model.pad_token_id)
    return (token_log_probs * active_mask.float()).sum(dim=1)


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _compute_dpo_loss(
    *,
    policy_model,
    reference_model,
    batch: Dict[str, torch.Tensor],
    cfg_scale: float,
    beta: float,
) -> Dict[str, torch.Tensor]:
    chosen_policy = _single_step_logprob_t1(
        policy_model,
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
        condition_bundle=batch["condition_bundle"],
        cfg_scale=cfg_scale,
    )
    rejected_policy = _single_step_logprob_t1(
        policy_model,
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
        condition_bundle=batch["condition_bundle"],
        cfg_scale=cfg_scale,
    )
    with torch.no_grad():
        chosen_ref = _single_step_logprob_t1(
            reference_model,
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            condition_bundle=batch["condition_bundle"],
            cfg_scale=cfg_scale,
        )
        rejected_ref = _single_step_logprob_t1(
            reference_model,
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            condition_bundle=batch["condition_bundle"],
            cfg_scale=cfg_scale,
        )

    delta_theta = chosen_policy - rejected_policy
    delta_ref = chosen_ref - rejected_ref
    margin = float(beta) * (delta_theta - delta_ref)
    loss = -F.logsigmoid(margin).mean()
    return {
        "loss": loss,
        "margin_mean": margin.mean().detach(),
        "policy_delta_mean": delta_theta.mean().detach(),
        "ref_delta_mean": delta_ref.mean().detach(),
        "preference_accuracy": (margin > 0).float().mean().detach(),
    }


def _evaluate_dpo_validation(
    *,
    policy_model,
    reference_model,
    val_loader: Optional[DataLoader],
    device: str,
    cfg_scale: float,
    beta: float,
) -> Dict[str, float]:
    if val_loader is None:
        return {
            "val_dpo_loss": float("nan"),
            "val_margin_mean": float("nan"),
            "val_preference_accuracy": float("nan"),
            "val_num_pairs": 0.0,
        }

    policy_model.eval()
    totals = {
        "n_pairs": 0,
        "loss": 0.0,
        "margin_mean": 0.0,
        "preference_accuracy": 0.0,
    }
    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch_to_device(batch, device)
            metrics = _compute_dpo_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                batch=batch,
                cfg_scale=cfg_scale,
                beta=beta,
            )
            batch_size = int(batch["condition_bundle"].shape[0])
            totals["n_pairs"] += batch_size
            totals["loss"] += float(metrics["loss"].item()) * batch_size
            totals["margin_mean"] += float(metrics["margin_mean"].item()) * batch_size
            totals["preference_accuracy"] += float(metrics["preference_accuracy"].item()) * batch_size

    denom = max(1, int(totals["n_pairs"]))
    return {
        "val_dpo_loss": totals["loss"] / denom,
        "val_margin_mean": totals["margin_mean"] / denom,
        "val_preference_accuracy": totals["preference_accuracy"] / denom,
        "val_num_pairs": float(totals["n_pairs"]),
    }


def _evaluate_dpo_proxy_success_metrics(
    *,
    resolved: ResolvedStep5Config,
    run_cfg: Dict[str, object],
    policy_model: torch.nn.Module,
    warm_start: S2TrainingArtifacts,
    prior: ResolvedClassSamplingPrior,
    evaluator,
    device: str,
    eval_step: int,
    target_rows_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    sample_id_start = 1
    proxy_target_df = select_step5_proxy_target_rows(
        (
            target_rows_df.copy()
            if target_rows_df is not None and not target_rows_df.empty
            else resolved.target_family_df.copy()
        ),
        num_targets=int(run_cfg["s4"]["rl_proxy_num_targets"]),
    )
    for _, target_row in proxy_target_df.iterrows():
        condition_bundle = torch.tensor(
            build_inference_condition_bundle_from_target_row(
                target_row.to_dict(),
                scaler=warm_start.scaler,
                soluble=1,
            ),
            dtype=torch.float32,
            device=device,
        )
        sampler = create_conditional_sampler(
            diffusion_model=policy_model,
            tokenizer=warm_start.tokenizer,
            resolved=resolved,
            prior=prior,
            condition_bundle=condition_bundle,
            cfg_scale=float(run_cfg["s4"]["cfg_scale"]),
            device=device,
        )
        try:
            smiles, _metadata = sample_conditional_with_class_prior(
                sampler=sampler,
                tokenizer=warm_start.tokenizer,
                prior=prior,
                resolved=resolved,
                num_samples=int(run_cfg["s4"]["rl_proxy_generation_budget"]),
                show_progress=False,
            )
        except ClassConstrainedSamplingQuotaError as exc:
            smiles = list(exc.partial_smiles)
            _metadata = dict(exc.metadata)
            _metadata["quota_shortfall_exception_fallback"] = True
        if not smiles:
            continue
        sample_df = build_generated_samples_frame(
            smiles,
            target_row=target_row,
            round_id=int(eval_step),
            sampling_seed=int(eval_step),
            run_name=str(run_cfg["run_name"]),
            canonical_family=str(run_cfg["canonical_family"]),
            sample_id_start=sample_id_start,
        )
        sample_id_start += len(sample_df)
        rows.append(evaluate_generated_samples(sample_df, evaluator))
    if not rows:
        return {
            "proxy_property_success_hit_rate_reporting": float("nan"),
            "proxy_property_success_hit_rate_discovery": float("nan"),
        }, pd.DataFrame()
    eval_df = pd.concat(rows, ignore_index=True)
    property_reporting = (
        float(eval_df["property_success_hit"].astype(float).mean())
        if "property_success_hit" in eval_df.columns
        else float("nan")
    )
    property_discovery = (
        float(eval_df["property_success_hit_discovery"].astype(float).mean())
        if "property_success_hit_discovery" in eval_df.columns
        else property_reporting
    )
    return {
        "proxy_property_success_hit_rate_reporting": property_reporting,
        "proxy_property_success_hit_rate_discovery": property_discovery,
    }, eval_df


def _save_dpo_checkpoint(
    *,
    checkpoint_path: Path,
    policy_model,
    reference_checkpoint_path: Path,
    optimizer,
    scheduler,
    epoch_idx: int,
    best_val_dpo_loss: float,
    run_cfg: Dict[str, object],
    warm_start: S2TrainingArtifacts,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": policy_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch_idx": int(epoch_idx),
        "best_val_dpo_loss": float(best_val_dpo_loss),
        "run_name": str(run_cfg["run_name"]),
        "alignment_mode": "dpo",
        "reference_checkpoint_path": str(reference_checkpoint_path),
        "warm_start_checkpoint_path": str(warm_start.checkpoint_path),
        "condition_scaler": {
            "temperature_min": float(warm_start.scaler.temperature_min),
            "temperature_max": float(warm_start.scaler.temperature_max),
            "phi_min": float(warm_start.scaler.phi_min),
            "phi_max": float(warm_start.scaler.phi_max),
            "chi_goal_min": float(warm_start.scaler.chi_goal_min),
            "chi_goal_max": float(warm_start.scaler.chi_goal_max),
        },
    }
    torch.save(payload, checkpoint_path)


def _clone_policy_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def train_s4_dpo_alignment(
    *,
    resolved: ResolvedStep5Config,
    run_cfg: Dict[str, object],
    run_dirs: Dict[str, Path],
    warm_start: S2TrainingArtifacts,
    prior: Optional[ResolvedClassSamplingPrior] = None,
    evaluator=None,
    device: str,
    target_rows_df: Optional[pd.DataFrame] = None,
    pruning_callback=None,
    skip_disk_checkpoints: bool = False,
) -> DpoTrainingArtifacts:
    """Train the Step 5 offline DPO branch from a supervised warm start."""

    dpo_cfg = dict(run_cfg["s4"]["dpo"])
    if str(dpo_cfg.get("logprob_mode", "single_step_t1")).strip().lower() != "single_step_t1":
        raise NotImplementedError(
            "Step 5 DPO currently implements only dpo.logprob_mode='single_step_t1'."
        )
    if str(dpo_cfg.get("pair_refresh_mode", "offline_fixed")).strip().lower() != "offline_fixed":
        raise NotImplementedError(
            "Step 5 DPO currently implements only dpo.pair_refresh_mode='offline_fixed'."
        )
    checkpoint_mode = str(dpo_cfg.get("checkpoint_selection_mode", "val_dpo_loss")).strip().lower()
    if checkpoint_mode not in {"val_dpo_loss", "proxy_property_success_hit_rate"}:
        raise NotImplementedError(
            "Step 5 DPO currently supports only dpo.checkpoint_selection_mode in "
            "{'val_dpo_loss', 'proxy_property_success_hit_rate'}."
        )
    proxy_eval_interval_epochs = max(1, int(dpo_cfg.get("proxy_eval_interval_epochs", 1)))
    if checkpoint_mode == "proxy_property_success_hit_rate" and (prior is None or evaluator is None):
        raise ValueError("Step 5 DPO proxy checkpoint selection requires prior and evaluator.")
    train_pairs_df, val_pairs_df, _pair_summary_df, diagnostics = build_dpo_pair_splits(
        resolved=resolved,
        run_cfg=run_cfg,
        scaler=warm_start.scaler,
        metrics_dir=run_dirs["metrics_dir"],
        warm_start=warm_start,
        prior=prior,
        evaluator=evaluator,
        device=device,
        target_rows_df=target_rows_df,
    )

    policy_model = deepcopy(warm_start.diffusion_model).to(device)
    reference_model = deepcopy(warm_start.diffusion_model).to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    train_dataset = Step5DpoDataset(train_pairs_df, tokenizer=warm_start.tokenizer)
    val_dataset = Step5DpoDataset(val_pairs_df, tokenizer=warm_start.tokenizer) if not val_pairs_df.empty else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(dpo_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=step5_dpo_collate_fn,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(dpo_cfg["batch_size"]),
            shuffle=False,
            num_workers=0,
            collate_fn=step5_dpo_collate_fn,
        )

    configured_num_epochs = int(dpo_cfg["num_epochs"])
    append_log_message(
        run_dirs["run_dir"],
        (
            f"DPO train start | run={run_cfg['run_name']} train_pairs={int(len(train_pairs_df))} "
            f"val_pairs={int(len(val_pairs_df))} configured_epochs={configured_num_epochs} "
            f"checkpoint_mode={checkpoint_mode}"
        ),
        echo=True,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        modules={"policy_model": policy_model},
        learning_rate=float(dpo_cfg["learning_rate"]),
        weight_decay=float(dpo_cfg["weight_decay"]),
        warmup_steps=int(dpo_cfg["warmup_steps"]),
        max_steps=max(1, int(len(train_loader) * configured_num_epochs)),
        warmup_schedule=str(dpo_cfg["warmup_schedule"]),
        lr_schedule=str(dpo_cfg["lr_schedule"]),
    )

    best_checkpoint_path = run_dirs["checkpoints_dir"] / "aligned_dpo_best.pt"
    last_checkpoint_path = run_dirs["checkpoints_dir"] / "aligned_dpo_last.pt"
    history_rows: List[Dict[str, Any]] = []
    proxy_rows: List[Dict[str, Any]] = []
    best_val_dpo_loss = float("inf")
    best_proxy_success = float("-inf")
    beta_value = float(dpo_cfg["beta"])
    effective_num_epochs = configured_num_epochs
    saturation_guard_triggered = False
    best_policy_state: Optional[Dict[str, torch.Tensor]] = None
    proxy_objective_metric = "proxy_property_success_hit_rate_discovery"
    best_checkpoint_metric_name = "val_dpo_loss" if checkpoint_mode == "val_dpo_loss" else proxy_objective_metric
    best_checkpoint_metric_value = float("inf") if checkpoint_mode == "val_dpo_loss" else float("-inf")

    def _should_run_proxy_eval(epoch_idx: int, total_epochs: int) -> bool:
        return int(epoch_idx) == int(total_epochs) or (int(epoch_idx) % int(proxy_eval_interval_epochs)) == 0

    if checkpoint_mode == "proxy_property_success_hit_rate":
        proxy_metrics, proxy_eval_df = _evaluate_dpo_proxy_success_metrics(
            resolved=resolved,
            run_cfg=run_cfg,
            policy_model=policy_model,
            warm_start=warm_start,
            prior=prior,
            evaluator=evaluator,
            device=device,
            eval_step=0,
            target_rows_df=target_rows_df,
        )
        proxy_objective_value = float(proxy_metrics.get(proxy_objective_metric, float("nan")))
        proxy_rows.append(
            {
                "epoch_idx": 0,
                "proxy_num_samples": int(len(proxy_eval_df)),
                **proxy_metrics,
            }
        )
        if np.isfinite(proxy_objective_value):
            best_proxy_success = float(proxy_objective_value)
            best_checkpoint_metric_value = float(proxy_objective_value)
            if skip_disk_checkpoints:
                best_policy_state = _clone_policy_state_dict(policy_model)
            else:
                _save_dpo_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    policy_model=policy_model,
                    reference_checkpoint_path=warm_start.checkpoint_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch_idx=0,
                    best_val_dpo_loss=best_val_dpo_loss,
                    run_cfg=run_cfg,
                    warm_start=warm_start,
                )

    epoch_counter = 0
    epoch_idx = 1
    while epoch_idx <= effective_num_epochs:
        policy_model.train()
        train_loss_sum = 0.0
        train_margin_sum = 0.0
        train_acc_sum = 0.0
        train_count = 0
        current_beta = float(beta_value)

        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            metrics = _compute_dpo_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                batch=batch,
                cfg_scale=float(run_cfg["s4"]["cfg_scale"]),
                beta=current_beta,
            )
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            batch_size = int(batch["condition_bundle"].shape[0])
            train_count += batch_size
            train_loss_sum += float(metrics["loss"].item()) * batch_size
            train_margin_sum += float(metrics["margin_mean"].item()) * batch_size
            train_acc_sum += float(metrics["preference_accuracy"].item()) * batch_size

        val_metrics = _evaluate_dpo_validation(
            policy_model=policy_model,
            reference_model=reference_model,
            val_loader=val_loader,
            device=device,
            cfg_scale=float(run_cfg["s4"]["cfg_scale"]),
            beta=current_beta,
        )
        denom = max(1, train_count)
        history_row = {
            "epoch_idx": int(epoch_idx),
            "train_dpo_loss": train_loss_sum / denom,
            "train_margin_mean": train_margin_sum / denom,
            "train_preference_accuracy": train_acc_sum / denom,
            "dpo_beta": current_beta,
            "effective_num_epochs": int(effective_num_epochs),
            "saturation_guard_triggered": 0,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "proxy_eval_interval_epochs": int(proxy_eval_interval_epochs),
            "proxy_eval_skipped": 0,
            **val_metrics,
        }
        epoch_counter = epoch_idx
        current_val = float(val_metrics["val_dpo_loss"]) if np.isfinite(val_metrics["val_dpo_loss"]) else float(
            history_row["train_dpo_loss"]
        )
        if np.isfinite(current_val):
            best_val_dpo_loss = min(best_val_dpo_loss, current_val)
        if checkpoint_mode == "proxy_property_success_hit_rate":
            history_row["proxy_eval_skipped"] = int(
                not _should_run_proxy_eval(int(epoch_idx), int(effective_num_epochs))
            )
            if not bool(history_row["proxy_eval_skipped"]):
                proxy_metrics, proxy_eval_df = _evaluate_dpo_proxy_success_metrics(
                    resolved=resolved,
                    run_cfg=run_cfg,
                    policy_model=policy_model,
                    warm_start=warm_start,
                    prior=prior,
                    evaluator=evaluator,
                    device=device,
                    eval_step=int(epoch_idx),
                    target_rows_df=target_rows_df,
                )
                proxy_objective_value = float(proxy_metrics.get(proxy_objective_metric, float("nan")))
                proxy_rows.append(
                    {
                        "epoch_idx": int(epoch_idx),
                        "proxy_num_samples": int(len(proxy_eval_df)),
                        **proxy_metrics,
                    }
                )
                history_row.update(proxy_metrics)
                if np.isfinite(proxy_objective_value) and proxy_objective_value > best_proxy_success:
                    best_proxy_success = float(proxy_objective_value)
                    best_checkpoint_metric_value = float(proxy_objective_value)
                    if skip_disk_checkpoints:
                        best_policy_state = _clone_policy_state_dict(policy_model)
                    else:
                        _save_dpo_checkpoint(
                            checkpoint_path=best_checkpoint_path,
                            policy_model=policy_model,
                            reference_checkpoint_path=warm_start.checkpoint_path,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch_idx=epoch_idx,
                            best_val_dpo_loss=current_val,
                            run_cfg=run_cfg,
                            warm_start=warm_start,
                        )
                if pruning_callback is not None and np.isfinite(proxy_objective_value):
                    pruning_callback(
                        stage="dpo",
                        step=int(epoch_idx),
                        value=float(proxy_objective_value),
                        metrics={
                            **history_row,
                            proxy_objective_metric: float(proxy_objective_value),
                            "pruning_metric": str(proxy_objective_metric),
                        },
                    )
        else:
            if pruning_callback is not None and np.isfinite(current_val):
                pruning_callback(
                    stage="dpo",
                    step=int(epoch_idx),
                    value=-float(current_val),
                    metrics={**history_row, "pruning_metric": "val_dpo_loss"},
                )
            if current_val < best_checkpoint_metric_value:
                best_checkpoint_metric_value = float(current_val)
                if skip_disk_checkpoints:
                    best_policy_state = _clone_policy_state_dict(policy_model)
                else:
                    _save_dpo_checkpoint(
                        checkpoint_path=best_checkpoint_path,
                        policy_model=policy_model,
                        reference_checkpoint_path=warm_start.checkpoint_path,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch_idx=epoch_idx,
                        best_val_dpo_loss=best_val_dpo_loss,
                        run_cfg=run_cfg,
                        warm_start=warm_start,
                    )
        if (not saturation_guard_triggered) and epoch_idx == 1:
            train_loss_value = float(history_row["train_dpo_loss"])
            val_acc_value = float(history_row["val_preference_accuracy"])
            if (
                (np.isfinite(val_acc_value) and val_acc_value > 0.95)
                or (np.isfinite(train_loss_value) and train_loss_value < 1.0e-4)
            ):
                saturation_guard_triggered = True
                beta_value = max(current_beta * 0.5, 1.0e-4)
                effective_num_epochs = max(effective_num_epochs, configured_num_epochs * 2)
                history_row["saturation_guard_triggered"] = 1
                LOGGER.warning(
                    "Step 5 DPO saturation guard triggered for run=%s: val_accuracy=%.4f train_loss=%.6g beta %.6g -> %.6g epochs %d -> %d",
                    str(run_cfg["run_name"]),
                    val_acc_value,
                    train_loss_value,
                    current_beta,
                    beta_value,
                    configured_num_epochs,
                    effective_num_epochs,
                )
        history_row["next_epoch_dpo_beta"] = float(beta_value)
        history_row["effective_num_epochs"] = int(effective_num_epochs)
        history_rows.append(history_row)
        proxy_value = float(history_row.get(proxy_objective_metric, float("nan")))
        proxy_text = f"{proxy_value:.4f}" if np.isfinite(proxy_value) else "nan"
        append_log_message(
            run_dirs["run_dir"],
            (
                f"DPO epoch | run={run_cfg['run_name']} epoch={int(epoch_idx)}/{int(effective_num_epochs)} "
                f"train_loss={float(history_row['train_dpo_loss']):.4f} "
                f"val_loss={float(history_row['val_dpo_loss']):.4f} "
                f"val_acc={float(history_row['val_preference_accuracy']):.4f} "
                f"proxy={proxy_text} beta={float(current_beta):.4g} "
                f"next_beta={float(beta_value):.4g} guard={int(history_row['saturation_guard_triggered'])}"
            ),
            echo=True,
        )
        epoch_idx += 1

    if not skip_disk_checkpoints:
        _save_dpo_checkpoint(
            checkpoint_path=last_checkpoint_path,
            policy_model=policy_model,
            reference_checkpoint_path=warm_start.checkpoint_path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_idx=epoch_counter,
            best_val_dpo_loss=best_val_dpo_loss,
            run_cfg=run_cfg,
            warm_start=warm_start,
        )
    if skip_disk_checkpoints:
        if best_policy_state is not None:
            policy_model.load_state_dict(best_policy_state)
    elif best_checkpoint_path.exists():
        load_step5_checkpoint_into_modules(
            checkpoint_path=best_checkpoint_path,
            diffusion_model=policy_model,
            aux_heads=None,
            device=device,
        )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dirs["metrics_dir"] / "dpo_training_history.csv", index=False)
    proxy_history_df = pd.DataFrame(proxy_rows)
    if not proxy_history_df.empty:
        proxy_history_df.to_csv(run_dirs["metrics_dir"] / "dpo_proxy_history.csv", index=False)
    with open(run_dirs["metrics_dir"] / "dpo_training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_name": str(run_cfg["run_name"]),
                "pair_source": diagnostics["pair_source"],
                "best_val_dpo_loss": float(best_val_dpo_loss),
                "best_checkpoint_metric_name": best_checkpoint_metric_name,
                "best_checkpoint_metric_value": (
                    float(best_checkpoint_metric_value) if np.isfinite(best_checkpoint_metric_value) else None
                ),
                "best_proxy_metric_value": (
                    float(best_proxy_success) if np.isfinite(best_proxy_success) else None
                ),
                "proxy_objective_metric": proxy_objective_metric,
                "checkpoint_selection_mode": checkpoint_mode,
                "proxy_eval_interval_epochs": int(proxy_eval_interval_epochs),
                "num_epochs": int(effective_num_epochs),
                "configured_num_epochs": int(configured_num_epochs),
                "configured_beta": float(dpo_cfg["beta"]),
                "final_beta": float(beta_value),
                "saturation_guard_triggered": bool(saturation_guard_triggered),
                "train_pairs": int(len(train_pairs_df)),
                "val_pairs": int(len(val_pairs_df)),
                "num_proxy_evals": int(len(proxy_history_df)),
                "best_checkpoint_path": (None if skip_disk_checkpoints else str(best_checkpoint_path)),
                "last_checkpoint_path": (None if skip_disk_checkpoints else str(last_checkpoint_path)),
                "disk_checkpoints_saved": bool(not skip_disk_checkpoints),
            },
            handle,
            indent=2,
        )
    append_log_message(
        run_dirs["run_dir"],
        (
            f"DPO train complete | run={run_cfg['run_name']} epochs_completed={int(epoch_counter)} "
            f"configured_epochs={int(configured_num_epochs)} effective_epochs={int(effective_num_epochs)} "
            f"best_checkpoint_metric={best_checkpoint_metric_name} "
            f"best_checkpoint_metric_value="
            f"{(float(best_checkpoint_metric_value) if np.isfinite(best_checkpoint_metric_value) else float('nan')):.4f} "
            f"best_val_dpo_loss={float(best_val_dpo_loss):.4f}"
        ),
        echo=True,
    )

    return DpoTrainingArtifacts(
        tokenizer=warm_start.tokenizer,
        policy_model=policy_model,
        reference_model=reference_model,
        checkpoint_path=(best_checkpoint_path if not skip_disk_checkpoints else last_checkpoint_path),
        last_checkpoint_path=last_checkpoint_path,
        scaler=warm_start.scaler,
        train_pairs_df=train_pairs_df,
        val_pairs_df=val_pairs_df,
        history_df=history_df,
    )
