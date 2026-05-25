"""On-policy RL alignment for Step 5 S4_rl."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from src.utils.reporting import append_log_message

from .conditional_sampling import create_conditional_sampler, sample_conditional_with_class_prior
from .config import select_step5_proxy_target_rows
from .dataset import (
    ConditionScaler,
    build_inference_condition_bundle,
    build_inference_condition_bundle_from_target_row,
    build_step5_supervised_frames,
)
from .evaluation import evaluate_generated_samples
from .frozen_sampling import (
    ClassConstrainedSamplingQuotaError,
    ResolvedClassSamplingPrior,
    _accepted_target_class_indices,
    _quota_shortfall_is_tolerable,
    _resolve_class_match_request_size,
)
from .rewards import compute_success_shaped_rewards
from .supervised import build_optimizer_and_scheduler, load_step5_checkpoint_into_modules
from .train_s2 import S2TrainingArtifacts
from .trajectory import (
    TrajectoryConditionalSampler,
    sample_trajectories_with_class_prior,
    select_sampling_trajectory_rows,
)
from src.evaluation.polymer_class import BACKBONE_CLASS_MATCH_CLASSES, PolymerClassifier
from src.utils.chemistry import canonicalize_smiles


@dataclass
class RlTrainingArtifacts:
    """Artifacts returned by Step 5 S4_rl alignment."""

    tokenizer: object
    policy_model: torch.nn.Module
    reference_model: torch.nn.Module
    checkpoint_path: Path
    last_checkpoint_path: Path
    scaler: ConditionScaler
    history_df: pd.DataFrame
    proxy_history_df: pd.DataFrame


def _build_rollout_frame(
    *,
    smiles: List[str],
    prompt_df: pd.DataFrame,
    sample_id_start: int,
    run_name: str,
    canonical_family: str,
    step_idx: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for offset, (smiles_value, prompt_row) in enumerate(zip(smiles, prompt_df.to_dict(orient="records"))):
        row = {
            "sample_id": int(sample_id_start + offset),
            "target_row_id": int(prompt_row["target_row_id"]),
            "target_row_key": str(prompt_row["target_row_key"]),
            "round_id": int(step_idx),
            "sampling_seed": int(prompt_row.get("sampling_seed", 0)),
            "run_name": str(run_name),
            "canonical_family": str(canonical_family),
            "c_target": str(prompt_row["c_target"]),
            "temperature": float(prompt_row["temperature"]),
            "phi": float(prompt_row["phi"]),
            "chi_target": float(prompt_row["chi_target"]),
            "property_rule": str(prompt_row.get("property_rule", "upper_bound")),
            "smiles": str(smiles_value),
        }
        for optional_col in ("chi_target_boot_q025", "chi_target_boot_q975"):
            optional_value = pd.to_numeric(prompt_row.get(optional_col, np.nan), errors="coerce")
            if np.isfinite(optional_value):
                row[optional_col] = float(optional_value)
        if "prompt_group_id" in prompt_row:
            row["prompt_group_id"] = int(prompt_row["prompt_group_id"])
        if "prompt_member_id" in prompt_row:
            row["prompt_member_id"] = int(prompt_row["prompt_member_id"])
        rows.append(row)
    return pd.DataFrame(rows)


def _build_generic_prompt_source_df(
    resolved,
    *,
    exact_step3_targets: bool,
) -> pd.DataFrame:
    frames = build_step5_supervised_frames(resolved)
    train_d_chi = frames["train_d_chi"].copy()
    train_d_chi = train_d_chi.loc[train_d_chi["water_miscible"].astype(int) == 1].copy()
    if train_d_chi.empty:
        raise ValueError("No water-miscible D_chi training rows available for generic RL prompt sampling.")
    if exact_step3_targets:
        train_d_chi["step3_chi_target"] = [
            resolved.chi_lookup.lookup(float(temp), float(phi), warn_on_missing=False)
            if np.isfinite(temp) and np.isfinite(phi)
            else None
            for temp, phi in zip(train_d_chi["temperature"], train_d_chi["phi"])
        ]
        train_d_chi = train_d_chi.loc[train_d_chi["step3_chi_target"].notna()].copy()
        if train_d_chi.empty:
            raise ValueError(
                "No exact Step 3-matched D_chi training rows available for RL prompt sampling."
            )
    train_d_chi = train_d_chi.sort_values(["temperature", "phi", "row_id"]).reset_index(drop=True)
    out = train_d_chi.copy()
    out["target_row_id"] = out["row_id"].astype(int) + (2_000_000 if exact_step3_targets else 1_000_000)
    key_prefix = "train_step3_row" if exact_step3_targets else "generic_train_row"
    out["target_row_key"] = out["row_id"].map(lambda row_id: f"{key_prefix}_{int(row_id)}")
    out["c_target"] = resolved.c_target
    out["chi_target"] = (
        out["step3_chi_target"].astype(float) if exact_step3_targets else out["chi"].astype(float)
    )
    out["property_rule"] = "upper_bound"
    if exact_step3_targets:
        step3_bounds = resolved.target_base_df[["temperature", "phi"]].copy()
        for optional_col in ("chi_target_boot_q025", "chi_target_boot_q975"):
            if optional_col in resolved.target_base_df.columns:
                step3_bounds[optional_col] = pd.to_numeric(
                    resolved.target_base_df[optional_col],
                    errors="coerce",
                )
        step3_bounds = step3_bounds.drop_duplicates(["temperature", "phi"]).reset_index(drop=True)
        out = out.merge(step3_bounds, on=["temperature", "phi"], how="left")
    keep_cols = ["target_row_id", "target_row_key", "c_target", "temperature", "phi", "chi_target", "property_rule"]
    keep_cols.extend(
        [optional_col for optional_col in ("chi_target_boot_q025", "chi_target_boot_q975") if optional_col in out.columns]
    )
    return out[keep_cols].copy()


def _build_prompt_source_df(resolved, run_cfg: Dict[str, object]) -> pd.DataFrame:
    prompt_source = str(run_cfg["s4"]["rl_prompt_source"]).strip().lower()
    if prompt_source == "benchmark_target_rows":
        return resolved.target_family_df.reset_index(drop=True).copy()
    if prompt_source == "generic_condition_distribution":
        return _build_generic_prompt_source_df(resolved, exact_step3_targets=False)
    if prompt_source == "train_exact_step3_distribution":
        return _build_generic_prompt_source_df(resolved, exact_step3_targets=True)
    raise NotImplementedError(f"Unsupported Step 5 rl_prompt_source: {prompt_source}")


def _sample_prompt_rows_from_source(
    prompt_source_df: pd.DataFrame,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if prompt_source_df.empty:
        raise ValueError("Step 5 RL prompt source is empty.")
    indices = rng.integers(0, len(prompt_source_df), size=int(batch_size))
    return prompt_source_df.iloc[indices].copy().reset_index(drop=True)


def _resolve_on_policy_alignment_mode(s4_cfg: Dict[str, object]) -> str:
    mode = str(s4_cfg.get("alignment_mode", "rl")).strip().lower()
    if mode not in {"rl", "ppo", "grpo"}:
        raise NotImplementedError(
            f"Unsupported Step 5 on-policy alignment mode: {mode!r}. "
            "Expected one of {'rl', 'ppo', 'grpo'}."
        )
    return mode


def _resolve_policy_update_epochs(s4_cfg: Dict[str, object], *, mode: str) -> int:
    default_epochs = 1 if str(mode).strip().lower() == "rl" else 4
    return max(1, int(s4_cfg.get("policy_update_epochs", default_epochs)))


def _resolve_stepwise_reward_weights(
    s4_cfg: Dict[str, object],
    *,
    step_idx: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    base_reward_weights = {
        str(key): float(value)
        for key, value in dict(s4_cfg.get("reward_weights", {})).items()
    }
    total_steps = max(1, int(s4_cfg.get("rl_num_steps", 1)))
    schedule_cfg = s4_cfg.get("reward_curriculum", {})
    if not isinstance(schedule_cfg, dict) or not bool(schedule_cfg.get("enabled", False)):
        return base_reward_weights, {
            "reward_curriculum_enabled": 0.0,
            "reward_curriculum_progress": 0.0,
            "reward_success_scale": 1.0,
            "reward_dense_scale": 1.0,
        }

    progress = 1.0 if total_steps <= 1 else float(step_idx - 1) / float(total_steps - 1)
    transition_frac = float(schedule_cfg.get("transition_frac", 0.4))
    transition_frac = min(max(transition_frac, 1.0e-8), 1.0)
    ramp = min(max(progress / transition_frac, 0.0), 1.0)
    success_final_scale = max(0.0, float(schedule_cfg.get("success_final_scale", 1.0)))
    dense_final_scale = max(0.0, float(schedule_cfg.get("dense_final_scale", 1.0)))
    success_scale = 1.0 + (success_final_scale - 1.0) * ramp
    dense_scale = 1.0 + (dense_final_scale - 1.0) * ramp

    step_reward_weights = deepcopy(base_reward_weights)
    if "w_success" in step_reward_weights:
        step_reward_weights["w_success"] = float(step_reward_weights["w_success"]) * float(success_scale)
    for key in ("w_sol", "w_chi", "w_sa", "w_sa_continuous"):
        if key in step_reward_weights:
            step_reward_weights[key] = float(step_reward_weights[key]) * float(dense_scale)

    return step_reward_weights, {
        "reward_curriculum_enabled": 1.0,
        "reward_curriculum_progress": float(progress),
        "reward_success_scale": float(success_scale),
        "reward_dense_scale": float(dense_scale),
    }


def _expand_prompt_groups(prompt_df: pd.DataFrame, *, group_size: int) -> pd.DataFrame:
    if group_size < 2:
        raise ValueError("GRPO requires grpo_group_size >= 2.")
    repeated = prompt_df.loc[prompt_df.index.repeat(int(group_size))].copy().reset_index(drop=True)
    repeated["prompt_group_id"] = np.repeat(np.arange(len(prompt_df), dtype=int), int(group_size))
    repeated["prompt_member_id"] = np.tile(np.arange(int(group_size), dtype=int), len(prompt_df))
    return repeated


def _sample_rollout_prompt_rows(
    prompt_source_df: pd.DataFrame,
    *,
    s4_cfg: Dict[str, object],
    rng: np.random.Generator,
) -> pd.DataFrame:
    mode = _resolve_on_policy_alignment_mode(s4_cfg)
    trajectories_per_batch = int(s4_cfg["trajectories_per_batch"])
    if mode == "grpo":
        group_size = int(s4_cfg.get("grpo_group_size", 4))
        if trajectories_per_batch % group_size != 0:
            raise ValueError(
                "Step 5 GRPO requires trajectories_per_batch to be divisible by grpo_group_size. "
                f"Got trajectories_per_batch={trajectories_per_batch}, grpo_group_size={group_size}."
            )
        base_prompt_df = _sample_prompt_rows_from_source(
            prompt_source_df,
            batch_size=int(trajectories_per_batch // group_size),
            rng=rng,
        ).reset_index(drop=True)
        return _expand_prompt_groups(base_prompt_df, group_size=group_size)

    prompt_df = _sample_prompt_rows_from_source(
        prompt_source_df,
        batch_size=trajectories_per_batch,
        rng=rng,
    ).reset_index(drop=True)
    prompt_df["prompt_group_id"] = np.arange(len(prompt_df), dtype=int)
    prompt_df["prompt_member_id"] = 0
    return prompt_df


def _prompt_df_to_condition_tensor(
    prompt_df: pd.DataFrame,
    *,
    scaler: ConditionScaler,
    device: str,
) -> torch.Tensor:
    bundles = [
        build_inference_condition_bundle(
            temperature=float(row["temperature"]),
            phi=float(row["phi"]),
            chi_goal=float(row["chi_target"]),
            scaler=scaler,
            soluble=1,
        )
        for row in prompt_df.to_dict(orient="records")
    ]
    return torch.tensor(np.asarray(bundles, dtype=np.float32), dtype=torch.float32, device=device)


def _create_trajectory_sampler(
    *,
    diffusion_model,
    tokenizer,
    resolved,
    prior: ResolvedClassSamplingPrior,
    condition_bundle: torch.Tensor,
    cfg_scale: float,
    num_steps: int,
    device: str,
) -> TrajectoryConditionalSampler:
    sampling_cfg = resolved.base_config.get("sampling", {})
    sampler = TrajectoryConditionalSampler(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        num_steps=int(num_steps),
        temperature=float(resolved.step5["sampling_temperature"]),
        top_k=sampling_cfg.get("top_k"),
        top_p=sampling_cfg.get("top_p"),
        target_stars=int(sampling_cfg.get("target_stars", 2)),
        use_constraints=bool(sampling_cfg.get("use_constraints", True)),
        device=device,
        condition_bundle=condition_bundle,
        cfg_scale=float(cfg_scale),
    )
    sampler.set_class_token_bias_start_frac(float(resolved.step5.get("class_token_bias_start_frac", 0.0)))
    if prior.class_token_logit_bias is not None:
        sampler.set_class_token_logit_bias(prior.class_token_logit_bias)
    sampler.set_forbidden_tokens(prior.forbidden_tokens)
    return sampler


def _sample_on_policy_rollouts(
    *,
    prompt_df: pd.DataFrame,
    policy_model,
    tokenizer,
    resolved,
    prior: ResolvedClassSamplingPrior,
    scaler: ConditionScaler,
    cfg_scale: float,
    num_steps: int,
    device: str,
    min_train_fill_ratio: float,
    partial_stop_attempt_fraction: float,
) -> Tuple[pd.DataFrame, List[str], List, Dict[str, object]]:
    """Sample one accepted rollout per prompt row while preserving prompt alignment."""

    pending_prompt_df = prompt_df.reset_index(drop=True).copy()
    accepted_prompt_parts: List[pd.DataFrame] = []
    accepted_smiles: List[str] = []
    accepted_trajectories: List = []
    accepted_raw_count = 0
    accepted_prompt_aligned_count = 0
    total_drawn = 0
    attempts = 0
    seen_canonical_smiles: set[str] = set()
    max_attempts = max(1, int(prior.class_match_sampling_attempts_max))
    requested_count = int(len(prompt_df))
    train_fill_ratio = min(max(float(min_train_fill_ratio), 0.0), 1.0)
    min_training_accepts = max(1, int(np.ceil(float(max(1, requested_count)) * train_fill_ratio)))
    partial_stop_attempt = max(
        1,
        int(np.ceil(float(max_attempts) * min(max(float(partial_stop_attempt_fraction), 0.0), 1.0))),
    )
    stopped_at_trainable_partial = False
    filter_rejection_counts = {
        "target_class_candidate_count": 0,
        "star_filter_rejected_count": 0,
        "sidechain_backbone_hybrid_rejected_count": 0,
        "unexpected_atoms_rejected_count": 0,
        "forbidden_token_rejected_count": 0,
        "duplicate_canonical_rejected_count": 0,
    }
    # Preserve the shared DiT family constraints in the rollout sampler, but keep
    # prompt-level class acceptance in this function so batch size stays aligned
    # with the prompt dataframe. The generic class-quota helper can oversample
    # beyond the prompt batch, which breaks prompt-conditioned trajectory replay.
    sampling_prior = replace(
        prior,
        enforce_class_match=False,
        enforce_backbone_class_match=False,
    )
    classifier = PolymerClassifier(patterns=resolved.polymer_patterns) if prior.enforce_class_match else None
    last_raw_meta: Dict[str, object] = {}
    stopped_at_quota_exhaustion_partial = False
    quota_exhaustion_reason: Optional[str] = None

    while not pending_prompt_df.empty:
        attempts += 1
        if attempts > max_attempts:
            accepted_count = int(len(accepted_smiles))
            if _quota_shortfall_is_tolerable(
                prior=prior,
                accepted_count=accepted_count,
                requested_count=requested_count,
            ):
                quota_exhaustion_reason = "max_attempts_partial_quota_tolerated"
                break
            if accepted_count >= min_training_accepts:
                stopped_at_trainable_partial = True
                quota_exhaustion_reason = "max_attempts_trainable_partial"
                break
            if accepted_count > 0:
                stopped_at_quota_exhaustion_partial = True
                quota_exhaustion_reason = "max_attempts_partial_below_min_train_fill"
                break
            raise RuntimeError(
                "Step 5 on-policy sampling could not satisfy the target-class quota. "
                f"target_class={prior.target_class!r} accepted={accepted_count} requested={requested_count} "
                f"after {max_attempts} attempts."
            )
        remaining = int(len(pending_prompt_df))
        request_size_total, request_debug = _resolve_class_match_request_size(
            prior=prior,
            remaining=int(remaining),
            attempts=int(attempts),
            total_drawn=int(total_drawn),
            accepted_raw_count=int(accepted_raw_count),
        )
        per_prompt_draws = max(1, int(np.ceil(float(request_size_total) / float(max(1, remaining)))))
        expanded_prompt_df = pending_prompt_df.loc[pending_prompt_df.index.repeat(per_prompt_draws)].copy()
        expanded_prompt_df = expanded_prompt_df.reset_index(drop=True)
        expanded_prompt_df["_pending_row_idx"] = np.repeat(
            np.arange(len(pending_prompt_df), dtype=int),
            per_prompt_draws,
        )

        condition_bundle = _prompt_df_to_condition_tensor(
            expanded_prompt_df,
            scaler=scaler,
            device=device,
        )
        trajectory_sampler = _create_trajectory_sampler(
            diffusion_model=policy_model,
            tokenizer=tokenizer,
            resolved=resolved,
            prior=sampling_prior,
            condition_bundle=condition_bundle,
            cfg_scale=cfg_scale,
            num_steps=int(num_steps),
            device=device,
        )
        raw_smiles, raw_trajectories, raw_meta = sample_trajectories_with_class_prior(
            sampler=trajectory_sampler,
            tokenizer=tokenizer,
            prior=sampling_prior,
            resolved=resolved,
            num_samples=int(len(expanded_prompt_df)),
            show_progress=False,
        )
        last_raw_meta = raw_meta
        total_drawn += int(len(raw_smiles))

        if prior.enforce_class_match and classifier is not None:
            local_seen_canonical_smiles = set(seen_canonical_smiles)
            accepted_idx, _attempt_filter_stats = _accepted_target_class_indices(
                raw_smiles,
                prior=prior,
                tokenizer=tokenizer,
                classifier=classifier,
                seen_canonical_smiles=local_seen_canonical_smiles,
            )
        else:
            local_seen_canonical_smiles = set(seen_canonical_smiles)
            accepted_idx, _attempt_filter_stats = _accepted_target_class_indices(
                raw_smiles,
                prior=prior,
                tokenizer=tokenizer,
                classifier=classifier,
                seen_canonical_smiles=local_seen_canonical_smiles,
            )
        for key, value in _attempt_filter_stats.items():
            filter_rejection_counts[key] = int(filter_rejection_counts.get(key, 0)) + int(value)
        accepted_raw_count += int(len(accepted_idx))

        chosen_raw_idx: List[int] = []
        chosen_pending_idx: List[int] = []
        chosen_pending_idx_set: set[int] = set()
        for raw_idx in accepted_idx:
            pending_idx = int(expanded_prompt_df.iloc[int(raw_idx)]["_pending_row_idx"])
            if pending_idx in chosen_pending_idx_set:
                continue
            chosen_pending_idx_set.add(pending_idx)
            chosen_pending_idx.append(pending_idx)
            chosen_raw_idx.append(int(raw_idx))
            canonical_smiles = canonicalize_smiles(str(raw_smiles[int(raw_idx)]))
            if canonical_smiles:
                seen_canonical_smiles.add(canonical_smiles)

        accepted_prompt_aligned_count += int(len(chosen_raw_idx))
        if chosen_raw_idx:
            accepted_prompt_parts.append(pending_prompt_df.iloc[chosen_pending_idx].copy())
            accepted_smiles.extend([raw_smiles[idx] for idx in chosen_raw_idx])
            accepted_trajectories.extend(
                select_sampling_trajectory_rows(raw_trajectories, keep_indices=chosen_raw_idx)
            )

        accepted_set = set(int(idx) for idx in chosen_pending_idx)
        pending_prompt_df = pending_prompt_df.iloc[
            [idx for idx in range(len(pending_prompt_df)) if idx not in accepted_set]
        ].reset_index(drop=True)
        last_raw_meta.update(
            {
                "requested_num_samples": int(request_size_total),
                "per_prompt_draws": int(per_prompt_draws),
                "expanded_prompt_batch_size": int(len(expanded_prompt_df)),
                **request_debug,
            }
        )
        if (
            not pending_prompt_df.empty
            and len(accepted_smiles) >= min_training_accepts
            and attempts >= partial_stop_attempt
        ):
            stopped_at_trainable_partial = True
            quota_exhaustion_reason = "partial_stop_attempt_trainable_partial"
            break

    accepted_prompt_df = (
        pd.concat(accepted_prompt_parts, ignore_index=True).reset_index(drop=True)
        if accepted_prompt_parts
        else prompt_df.iloc[:0].copy()
    )
    effective_attempts = min(int(attempts), int(max_attempts))
    fill_fraction = float(len(accepted_smiles)) / float(requested_count) if requested_count > 0 else 1.0
    metadata = {
        "num_samples": int(requested_count),
        "returned_num_samples": int(len(accepted_smiles)),
        "remaining_num_samples": max(0, int(requested_count) - int(len(accepted_smiles))),
        "fill_fraction": float(fill_fraction),
        "quota_satisfied": bool(len(accepted_smiles) == requested_count),
        "quota_shortfall_tolerated": bool(
            len(accepted_smiles) < requested_count
            and _quota_shortfall_is_tolerable(
                prior=prior,
                accepted_count=len(accepted_smiles),
                requested_count=requested_count,
            )
        ),
        "quota_shortfall_relaxed_for_training": bool(
            len(accepted_smiles) < requested_count
            and requested_count > 0
            and len(accepted_smiles) >= min_training_accepts
            and not _quota_shortfall_is_tolerable(
                prior=prior,
                accepted_count=len(accepted_smiles),
                requested_count=requested_count,
            )
        ),
        "quota_shortfall_trainable": bool(
            len(accepted_smiles) < requested_count and len(accepted_smiles) >= min_training_accepts
        ),
        "quota_shortfall_below_min_train": bool(
            len(accepted_smiles) < requested_count
            and 0 < len(accepted_smiles) < min_training_accepts
        ),
        "stopped_at_trainable_partial": bool(stopped_at_trainable_partial),
        "stopped_at_quota_exhaustion_partial": bool(stopped_at_quota_exhaustion_partial),
        "quota_exhaustion_reason": quota_exhaustion_reason,
        "min_train_fill_ratio": float(train_fill_ratio),
        "min_training_accepts": int(min_training_accepts),
        "partial_stop_attempt": int(partial_stop_attempt),
        "num_trajectory_batches": int(len(accepted_trajectories)),
        "total_raw_samples_drawn": int(total_drawn),
        "accepted_raw_target_class_samples": int(accepted_raw_count),
        "accepted_prompt_aligned_samples": int(accepted_prompt_aligned_count),
        "class_match_sampling_attempts": int(effective_attempts),
        "class_match_acceptance_rate": (
            float(accepted_raw_count) / float(total_drawn) if total_drawn > 0 else 0.0
        ),
        "class_match_oversampling_ratio": (
            float(total_drawn) / float(requested_count) if requested_count > 0 else 0.0
        ),
        "filter_rejection_counts": {
            str(key): int(value)
            for key, value in filter_rejection_counts.items()
        },
        "spans_per_sample": int(prior.spans_per_sample),
        "motif_count": int(len(prior.motifs)),
        "motif_source": prior.motif_source,
        "backbone_template_enabled": bool(prior.backbone_template_enabled),
        "backbone_template_core_count": int(len(prior.backbone_template_cores)),
        "backbone_template_source": prior.backbone_template_source,
        "backbone_template_layout": "contiguous_scaffold" if prior.backbone_template_enabled else "disabled",
        "length_prior_count": int(last_raw_meta.get("length_prior_count", len(prior.length_prior_lengths))),
        "length_prior_source": last_raw_meta.get("length_prior_source", prior.length_prior_source),
        "class_token_bias_enabled": bool(prior.class_token_logit_bias is not None),
        "class_token_bias_strength": float(prior.class_token_bias_strength),
        "enforce_class_match": bool(prior.enforce_class_match),
        "enforce_backbone_class_match": bool(prior.enforce_backbone_class_match),
        "class_match_mode": (
            "strict_backbone"
            if prior.target_class in BACKBONE_CLASS_MATCH_CLASSES
            and prior.enforce_backbone_class_match
            else "loose"
        ),
    }
    return accepted_prompt_df, accepted_smiles, accepted_trajectories, metadata


def _save_rl_checkpoint(
    *,
    checkpoint_path: Path,
    policy_model,
    reference_checkpoint_path: Path,
    optimizer,
    scheduler,
    step_idx: int,
    best_proxy_metric_value: float,
    alignment_mode: str,
    run_cfg: Dict[str, object],
    warm_start: S2TrainingArtifacts,
    proxy_metric_name: str = "proxy_property_success_hit_rate_discovery",
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": policy_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step_idx": int(step_idx),
        "best_proxy_metric_value": float(best_proxy_metric_value),
        "best_proxy_metric_name": str(proxy_metric_name),
        "run_name": str(run_cfg["run_name"]),
        "alignment_mode": str(alignment_mode),
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


def _clone_model_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _compute_on_policy_advantages(
    rewards: torch.Tensor,
    *,
    prompt_df: pd.DataFrame,
    mode: str,
    normalize_advantages: bool,
    group_epsilon: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    rewards = rewards.detach().to(dtype=torch.float32)
    metrics: Dict[str, float] = {
        "advantage_mean": float("nan"),
        "advantage_std": float("nan"),
        "advantage_abs_mean": float("nan"),
        "advantage_nonzero_frac": float("nan"),
        "prompt_group_count": float("nan"),
        "grpo_zero_advantage_group_frac": float("nan"),
        "grpo_group_reward_std_mean": float("nan"),
    }
    if rewards.numel() == 0:
        return rewards.clone(), metrics

    if str(mode).strip().lower() == "grpo":
        if "prompt_group_id" not in prompt_df.columns:
            raise ValueError("GRPO rollout prompt_df must include prompt_group_id.")
        group_ids = torch.tensor(prompt_df["prompt_group_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        advantages = torch.zeros_like(rewards)
        unique_ids = torch.unique(group_ids, sorted=True)
        zero_advantage_groups = 0
        group_reward_stds: List[float] = []
        for group_id in unique_ids.tolist():
            mask = group_ids == int(group_id)
            group_rewards = rewards[mask]
            if group_rewards.numel() <= 1:
                advantages[mask] = 0.0
                zero_advantage_groups += 1
                group_reward_stds.append(0.0)
                continue
            group_mean = group_rewards.mean()
            group_std = group_rewards.std(unbiased=False)
            group_reward_stds.append(float(group_std.item()))
            if float(group_std.item()) <= float(group_epsilon):
                advantages[mask] = 0.0
                zero_advantage_groups += 1
            else:
                advantages[mask] = (group_rewards - group_mean) / (group_std + float(group_epsilon))
        metrics["prompt_group_count"] = float(len(unique_ids))
        metrics["grpo_zero_advantage_group_frac"] = (
            float(zero_advantage_groups) / float(len(unique_ids)) if len(unique_ids) > 0 else float("nan")
        )
        metrics["grpo_group_reward_std_mean"] = (
            float(np.mean(group_reward_stds)) if group_reward_stds else float("nan")
        )
    else:
        advantages = rewards - rewards.mean()
        if normalize_advantages and rewards.numel() > 1:
            adv_std = advantages.std(unbiased=False)
            if float(adv_std.item()) > float(group_epsilon):
                advantages = advantages / (adv_std + float(group_epsilon))

    metrics["advantage_mean"] = float(advantages.mean().item())
    metrics["advantage_std"] = float(advantages.std(unbiased=False).item()) if advantages.numel() > 1 else 0.0
    metrics["advantage_abs_mean"] = float(advantages.abs().mean().item()) if advantages.numel() else float("nan")
    metrics["advantage_nonzero_frac"] = (
        float((advantages.abs() > 1.0e-8).float().mean().item()) if advantages.numel() else float("nan")
    )
    return advantages, metrics


def _precompute_old_logprob_chunks(
    trajectory_sampler: TrajectoryConditionalSampler,
    trajectories: List,
    *,
    mode: str,
    batch_chunk_size: int,
) -> List[Optional[torch.Tensor]]:
    if str(mode).strip().lower() not in {"ppo", "grpo"}:
        return [None] * len(trajectories)
    old_chunks: List[Optional[torch.Tensor]] = []
    with torch.no_grad():
        for trajectory in trajectories:
            batch_size = int(trajectory.final_ids.shape[0])
            if batch_chunk_size > 0 and batch_size > batch_chunk_size:
                chunk_logprobs: List[torch.Tensor] = []
                for start_idx in range(0, batch_size, batch_chunk_size):
                    end_idx = min(batch_size, start_idx + batch_chunk_size)
                    chunk_list = select_sampling_trajectory_rows(
                        [trajectory],
                        keep_indices=range(start_idx, end_idx),
                    )
                    if len(chunk_list) != 1:
                        raise ValueError("Expected exactly one chunk when slicing a single Step 5 trajectory batch.")
                    replay = trajectory_sampler.replay_trajectory_logprob(chunk_list[0], grad_enabled=False)
                    chunk_logprobs.append(replay["trajectory_logprob"].detach().to(device="cpu"))
                old_chunks.append(torch.cat(chunk_logprobs, dim=0))
            else:
                replay = trajectory_sampler.replay_trajectory_logprob(trajectory, grad_enabled=False)
                old_chunks.append(replay["trajectory_logprob"].detach().to(device="cpu"))
    return old_chunks


def _evaluate_proxy_success_metrics(
    *,
    resolved,
    run_cfg: Dict[str, object],
    policy_model,
    tokenizer,
    scaler: ConditionScaler,
    prior: ResolvedClassSamplingPrior,
    evaluator,
    device: str,
    step_idx: int,
    num_steps: int,
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
                scaler=scaler,
                soluble=1,
            ),
            dtype=torch.float32,
            device=device,
        )
        sampler = create_conditional_sampler(
            diffusion_model=policy_model,
            tokenizer=tokenizer,
            resolved=resolved,
            prior=prior,
            condition_bundle=condition_bundle,
            cfg_scale=float(run_cfg["s4"]["cfg_scale"]),
            device=device,
            num_steps=int(num_steps),
        )
        try:
            smiles, _metadata = sample_conditional_with_class_prior(
                sampler=sampler,
                tokenizer=tokenizer,
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
        sample_df = pd.DataFrame(
            {
                "sample_id": np.arange(sample_id_start, sample_id_start + len(smiles), dtype=int),
                "target_row_id": int(target_row["target_row_id"]),
                "target_row_key": str(target_row["target_row_key"]),
                "round_id": int(step_idx),
                "sampling_seed": int(step_idx),
                "run_name": str(run_cfg["run_name"]),
                "canonical_family": str(run_cfg["canonical_family"]),
                "c_target": str(target_row["c_target"]),
                "temperature": float(target_row["temperature"]),
                "phi": float(target_row["phi"]),
                "chi_target": float(target_row["chi_target"]),
                "property_rule": str(target_row.get("property_rule", "upper_bound")),
                "smiles": smiles,
            }
        )
        for optional_col in ("chi_target_boot_q025", "chi_target_boot_q975"):
            optional_value = pd.to_numeric(target_row.get(optional_col, np.nan), errors="coerce")
            if np.isfinite(optional_value):
                sample_df[optional_col] = float(optional_value)
        sample_id_start += len(smiles)
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


def train_s4_rl_alignment(
    *,
    resolved,
    run_cfg: Dict[str, object],
    run_dirs: Dict[str, Path],
    warm_start: S2TrainingArtifacts,
    prior: ResolvedClassSamplingPrior,
    evaluator,
    device: str,
    target_rows_df: Optional[pd.DataFrame] = None,
    pruning_callback: Optional[Callable[..., None]] = None,
    skip_disk_checkpoints: bool = False,
) -> RlTrainingArtifacts:
    """Train the Step 5 on-policy branch from a supervised warm start."""

    s4_cfg = dict(run_cfg["s4"])
    alignment_mode = _resolve_on_policy_alignment_mode(s4_cfg)
    training_prior = prior
    rl_diffusion_num_steps = int(
        s4_cfg.get("rl_diffusion_num_steps", resolved.base_config["diffusion"]["num_steps"])
    )
    if str(s4_cfg.get("rl_checkpoint_selection_mode", "proxy_property_success_hit_rate")).strip().lower() not in {
        "proxy_property_success_hit_rate",
        "final_checkpoint",
    }:
        raise NotImplementedError(
            "Step 5 RL currently supports only rl_checkpoint_selection_mode in "
            "{'proxy_property_success_hit_rate', 'final_checkpoint'}."
        )

    policy_model = deepcopy(warm_start.diffusion_model).to(device)
    reference_model = deepcopy(warm_start.diffusion_model).to(device)
    if hasattr(policy_model.backbone, "set_gradient_checkpointing"):
        policy_model.backbone.set_gradient_checkpointing(bool(s4_cfg.get("gradient_checkpointing", True)))
    if hasattr(reference_model.backbone, "set_gradient_checkpointing"):
        reference_model.backbone.set_gradient_checkpointing(False)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    policy_update_epochs = _resolve_policy_update_epochs(s4_cfg, mode=alignment_mode)
    optimizer, scheduler = build_optimizer_and_scheduler(
        modules={"policy_model": policy_model},
        learning_rate=float(s4_cfg["learning_rate"]),
        weight_decay=float(s4_cfg["weight_decay"]),
        warmup_steps=int(s4_cfg["warmup_steps"]),
        max_steps=int(s4_cfg["rl_num_steps"]) * int(policy_update_epochs),
        warmup_schedule=str(s4_cfg["warmup_schedule"]),
        lr_schedule=str(s4_cfg["lr_schedule"]),
    )

    rng = np.random.default_rng(int(resolved.step5["random_seed"]))
    prompt_source_df = _build_prompt_source_df(resolved, run_cfg)
    best_checkpoint_path = run_dirs["checkpoints_dir"] / f"aligned_{alignment_mode}_best.pt"
    last_checkpoint_path = run_dirs["checkpoints_dir"] / f"aligned_{alignment_mode}_last.pt"
    history_rows: List[Dict[str, Any]] = []
    proxy_rows: List[Dict[str, Any]] = []
    best_proxy_success = float("-inf")
    best_policy_state: Optional[Dict[str, torch.Tensor]] = None
    checkpoint_mode = str(s4_cfg.get("rl_checkpoint_selection_mode", "proxy_property_success_hit_rate")).strip().lower()
    proxy_objective_metric = "proxy_property_success_hit_rate_discovery"
    sample_id_start = 1
    clip_eps = float(s4_cfg.get("ppo_clip_eps", 0.2))
    normalize_advantages = bool(s4_cfg.get("normalize_advantages", alignment_mode in {"ppo", "grpo"}))
    grpo_group_size = int(s4_cfg.get("grpo_group_size", 4))
    grpo_advantage_epsilon = float(s4_cfg.get("grpo_advantage_epsilon", 1.0e-6))
    replay_batch_size = int(s4_cfg.get("replay_batch_size", 0) or 0)
    min_train_fill_ratio = min(max(float(s4_cfg.get("rl_min_train_fill_ratio", 0.50)), 0.0), 1.0)
    partial_stop_attempt_fraction = min(
        max(float(s4_cfg.get("rl_partial_rollout_stop_attempt_frac", 0.50)), 0.0),
        1.0,
    )
    low_fill_stop_ratio = min(max(float(s4_cfg.get("rl_low_fill_stop_ratio", 0.25)), 0.0), 1.0)
    low_fill_patience = max(0, int(s4_cfg.get("rl_low_fill_patience", 3)))
    consecutive_low_fill_steps = 0
    append_log_message(
        run_dirs["run_dir"],
        (
            f"RL train start | run={run_cfg['run_name']} mode={alignment_mode} "
            f"rl_num_steps={int(s4_cfg['rl_num_steps'])} policy_update_epochs={int(policy_update_epochs)} "
            f"prompt_source={str(s4_cfg['rl_prompt_source'])} "
            f"min_train_fill_ratio={float(min_train_fill_ratio):.3f}"
        ),
        echo=True,
    )

    completed_steps = 0
    stop_reason: Optional[str] = None
    for step_idx in range(1, int(s4_cfg["rl_num_steps"]) + 1):
        prompt_df = _sample_rollout_prompt_rows(
            prompt_source_df,
            s4_cfg=s4_cfg,
            rng=rng,
        ).reset_index(drop=True)
        prompt_df["sampling_seed"] = int(step_idx)
        accepted_prompt_df, smiles, trajectories, rollout_meta = _sample_on_policy_rollouts(
            prompt_df=prompt_df,
            policy_model=policy_model,
            tokenizer=warm_start.tokenizer,
            resolved=resolved,
            prior=training_prior,
            scaler=warm_start.scaler,
            cfg_scale=float(s4_cfg["cfg_scale"]),
            num_steps=int(rl_diffusion_num_steps),
            device=device,
            min_train_fill_ratio=float(min_train_fill_ratio),
            partial_stop_attempt_fraction=float(partial_stop_attempt_fraction),
        )

        rollout_df = _build_rollout_frame(
            smiles=smiles,
            prompt_df=accepted_prompt_df,
            sample_id_start=sample_id_start,
            run_name=str(run_cfg["run_name"]),
            canonical_family=str(run_cfg["canonical_family"]),
            step_idx=step_idx,
        )
        sample_id_start += len(rollout_df)
        evaluation_df = evaluate_generated_samples(rollout_df, evaluator).sort_values("sample_id").reset_index(drop=True)
        reward_weights, reward_schedule_metrics = _resolve_stepwise_reward_weights(
            s4_cfg,
            step_idx=int(step_idx),
        )
        rewards, reward_metrics = compute_success_shaped_rewards(
            evaluation_df,
            reward_weights=reward_weights,
            sol_log_prob_floor=float(s4_cfg["sol_log_prob_floor"]),
            reward_shaping=s4_cfg.get("reward_shaping", {}),
        )
        advantages, advantage_metrics = _compute_on_policy_advantages(
            rewards,
            prompt_df=accepted_prompt_df,
            mode=alignment_mode,
            normalize_advantages=normalize_advantages,
            group_epsilon=grpo_advantage_epsilon,
        )
        condition_dim = (
            int(trajectories[0].condition_bundle.shape[1])
            if trajectories and trajectories[0].condition_bundle.ndim == 2
            else 7
        )
        logprob_sampler = _create_trajectory_sampler(
            diffusion_model=policy_model,
            tokenizer=warm_start.tokenizer,
            resolved=resolved,
            prior=training_prior,
            condition_bundle=torch.empty((0, condition_dim), dtype=torch.float32, device=device),
            cfg_scale=float(s4_cfg["cfg_scale"]),
            num_steps=int(rl_diffusion_num_steps),
            device=device,
        )
        old_logprob_chunks = _precompute_old_logprob_chunks(
            logprob_sampler,
            trajectories,
            mode=alignment_mode,
            batch_chunk_size=int(replay_batch_size),
        )

        denom = max(1, len(evaluation_df))
        epoch_losses: List[float] = []
        epoch_policy_objectives: List[float] = []
        epoch_kl_means: List[float] = []
        epoch_logprob_means: List[float] = []
        epoch_ratio_means: List[float] = []
        epoch_clip_fractions: List[float] = []

        for _update_epoch in range(1, int(policy_update_epochs) + 1):
            optimizer.zero_grad(set_to_none=True)
            total_policy_term = torch.tensor(0.0, dtype=torch.float32, device=device)
            total_kl_term = torch.tensor(0.0, dtype=torch.float32, device=device)
            total_logprob_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
            ratio_sum = 0.0
            ratio_count = 0
            clipped_count = 0
            offset = 0
            for traj_idx, trajectory in enumerate(trajectories):
                batch_size = int(trajectory.final_ids.shape[0])
                old_logprob_full = old_logprob_chunks[traj_idx]
                start_idx = 0
                base_chunk_size = batch_size if replay_batch_size <= 0 else replay_batch_size
                while start_idx < batch_size:
                    current_chunk_size = min(batch_size - start_idx, base_chunk_size)
                    while True:
                        end_idx = min(batch_size, start_idx + current_chunk_size)
                        chunk_list = select_sampling_trajectory_rows(
                            [trajectory],
                            keep_indices=range(start_idx, end_idx),
                        )
                        if len(chunk_list) != 1:
                            raise ValueError("Expected exactly one chunk when slicing a single Step 5 trajectory batch.")
                        trajectory_chunk = chunk_list[0]
                        chunk_size = int(trajectory_chunk.final_ids.shape[0])
                        advantage_chunk = advantages[offset + start_idx : offset + start_idx + chunk_size].to(
                            device=device,
                            dtype=torch.float32,
                        )
                        try:
                            replay = logprob_sampler.replay_trajectory_logprob(trajectory_chunk, grad_enabled=True)
                            current_logprob = replay["trajectory_logprob"]
                            kl_stats = logprob_sampler.compute_trajectory_kl(
                                trajectory_chunk,
                                reference_diffusion_model=reference_model,
                            )
                            break
                        except torch.cuda.OutOfMemoryError:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if current_chunk_size <= 1:
                                raise
                            reduced_chunk_size = max(1, current_chunk_size // 2)
                            append_log_message(
                                run_dirs["run_dir"],
                                (
                                    "RL replay/KL OOM; reducing replay chunk size "
                                    f"from {int(current_chunk_size)} to {int(reduced_chunk_size)} "
                                    f"at step={int(step_idx)} trajectory_batch={int(traj_idx)}."
                                ),
                                echo=True,
                            )
                            current_chunk_size = reduced_chunk_size
                    total_logprob_sum = total_logprob_sum + current_logprob.sum()
                    if alignment_mode == "rl":
                        policy_term_chunk = -(advantage_chunk.detach() * current_logprob).sum()
                    else:
                        if old_logprob_full is None:
                            raise ValueError(f"Missing old logprob chunk for alignment_mode={alignment_mode}.")
                        old_logprob = old_logprob_full[start_idx:end_idx].to(device=device, dtype=torch.float32)
                        log_ratio = torch.clamp(current_logprob - old_logprob, min=-20.0, max=20.0)
                        ratio = torch.exp(log_ratio)
                        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                        surrogate = ratio * advantage_chunk.detach()
                        surrogate_clipped = clipped_ratio * advantage_chunk.detach()
                        policy_term_chunk = -torch.minimum(surrogate, surrogate_clipped).sum()
                        ratio_sum += float(ratio.detach().sum().item())
                        ratio_count += int(ratio.numel())
                        clipped_count += int(
                            (torch.abs(ratio.detach() - clipped_ratio.detach()) > 1.0e-8).sum().item()
                        )
                    total_policy_term = total_policy_term + policy_term_chunk
                    total_kl_term = total_kl_term + kl_stats["trajectory_kl"].sum()
                    start_idx = end_idx
                offset += batch_size

            loss = (total_policy_term + float(s4_cfg["kl_weight"]) * total_kl_term) / float(denom)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(float(loss.item()))
            epoch_policy_objectives.append(float((-total_policy_term / float(denom)).item()))
            epoch_kl_means.append(float((total_kl_term / float(denom)).item()))
            epoch_logprob_means.append(float((total_logprob_sum / float(denom)).item()))
            if ratio_count > 0:
                epoch_ratio_means.append(float(ratio_sum / float(ratio_count)))
                epoch_clip_fractions.append(float(clipped_count) / float(ratio_count))

        history_row = {
            "step_idx": int(step_idx),
            "rollout_batch_size": int(len(evaluation_df)),
            "alignment_mode": str(alignment_mode),
            "policy_update_epochs": int(policy_update_epochs),
            "loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "baseline_reward": float(rewards.mean().item()) if len(rewards) else float("nan"),
            "trajectory_logprob_mean": float(np.mean(epoch_logprob_means)) if epoch_logprob_means else float("nan"),
            "policy_objective_mean": float(np.mean(epoch_policy_objectives)) if epoch_policy_objectives else float("nan"),
            "trajectory_kl_mean": float(np.mean(epoch_kl_means)) if epoch_kl_means else float("nan"),
            "ratio_mean": float(np.mean(epoch_ratio_means)) if epoch_ratio_means else float("nan"),
            "clip_fraction": float(np.mean(epoch_clip_fractions)) if epoch_clip_fractions else 0.0,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "w_success": float(reward_weights.get("w_success", 0.0)),
            "w_sol": float(reward_weights.get("w_sol", 0.0)),
            "w_chi": float(reward_weights.get("w_chi", 0.0)),
            "w_sa": float(reward_weights.get("w_sa", 0.0)),
            "w_sa_continuous": float(reward_weights.get("w_sa_continuous", 0.0)),
            "kl_weight": float(s4_cfg["kl_weight"]),
            "ppo_clip_eps": (float(clip_eps) if alignment_mode in {"ppo", "grpo"} else float("nan")),
            "normalize_advantages": int(bool(normalize_advantages)),
            "grpo_group_size": (int(grpo_group_size) if alignment_mode == "grpo" else 1),
            **reward_schedule_metrics,
            **reward_metrics,
            **advantage_metrics,
            **{
                "class_match_sampling_attempts": int(rollout_meta.get("class_match_sampling_attempts", 0)),
                "class_match_acceptance_rate": float(
                    rollout_meta.get("class_match_acceptance_rate", float("nan"))
                ),
                "class_match_oversampling_ratio": float(
                    rollout_meta.get("class_match_oversampling_ratio", float("nan"))
                ),
                "rollout_fill_fraction": float(rollout_meta.get("fill_fraction", float("nan"))),
                "quota_satisfied": int(bool(rollout_meta.get("quota_satisfied", False))),
                "quota_shortfall_trainable": int(bool(rollout_meta.get("quota_shortfall_trainable", False))),
                "quota_shortfall_below_min_train": int(
                    bool(rollout_meta.get("quota_shortfall_below_min_train", False))
                ),
                "stopped_at_trainable_partial": int(bool(rollout_meta.get("stopped_at_trainable_partial", False))),
                "stopped_at_quota_exhaustion_partial": int(
                    bool(rollout_meta.get("stopped_at_quota_exhaustion_partial", False))
                ),
                "min_training_accepts": int(rollout_meta.get("min_training_accepts", 0)),
                "accepted_prompt_aligned_samples": int(
                    rollout_meta.get("accepted_prompt_aligned_samples", len(evaluation_df))
                ),
                "total_raw_samples_drawn": int(rollout_meta.get("total_raw_samples_drawn", len(evaluation_df))),
                "rl_diffusion_num_steps": int(rl_diffusion_num_steps),
                "motif_count": int(rollout_meta.get("motif_count", 0)),
                "class_token_bias_enabled": int(bool(rollout_meta.get("class_token_bias_enabled", False))),
                "backbone_template_enabled": int(bool(rollout_meta.get("backbone_template_enabled", False))),
            },
        }

        should_eval_proxy = (
            checkpoint_mode == "proxy_property_success_hit_rate"
            and (
                step_idx == 1
                or step_idx % int(s4_cfg["rl_proxy_eval_interval_steps"]) == 0
                or step_idx == int(s4_cfg["rl_num_steps"])
            )
        )
        if should_eval_proxy:
            proxy_metrics, proxy_eval_df = _evaluate_proxy_success_metrics(
                resolved=resolved,
                run_cfg=run_cfg,
                policy_model=policy_model,
                tokenizer=warm_start.tokenizer,
                scaler=warm_start.scaler,
                prior=training_prior,
                evaluator=evaluator,
                device=device,
                step_idx=step_idx,
                num_steps=int(rl_diffusion_num_steps),
                target_rows_df=target_rows_df,
            )
            proxy_objective_value = float(proxy_metrics.get(proxy_objective_metric, float("nan")))
            proxy_rows.append(
                {
                    "step_idx": int(step_idx),
                    **proxy_metrics,
                    "proxy_num_samples": int(len(proxy_eval_df)),
                }
            )
            history_row.update(proxy_metrics)
            if np.isfinite(proxy_objective_value) and proxy_objective_value > best_proxy_success:
                best_proxy_success = float(proxy_objective_value)
                if skip_disk_checkpoints:
                    best_policy_state = _clone_model_state_dict(policy_model)
                else:
                    _save_rl_checkpoint(
                        checkpoint_path=best_checkpoint_path,
                        policy_model=policy_model,
                        reference_checkpoint_path=warm_start.checkpoint_path,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step_idx=step_idx,
                        best_proxy_metric_value=best_proxy_success,
                        alignment_mode=alignment_mode,
                        run_cfg=run_cfg,
                        warm_start=warm_start,
                        proxy_metric_name=proxy_objective_metric,
                    )
            if pruning_callback is not None and np.isfinite(proxy_objective_value):
                pruning_callback(
                    stage="rl",
                    step=int(step_idx),
                    value=float(proxy_objective_value),
                    metrics={
                        **history_row,
                        proxy_objective_metric: float(proxy_objective_value),
                        "pruning_metric": str(proxy_objective_metric),
                    },
                )

        history_rows.append(history_row)
        proxy_value = float(history_row.get(proxy_objective_metric, float("nan")))
        proxy_text = f"{proxy_value:.4f}" if np.isfinite(proxy_value) else "nan"
        append_log_message(
            run_dirs["run_dir"],
            (
                f"RL step | run={run_cfg['run_name']} mode={alignment_mode} "
                f"step={int(step_idx)}/{int(s4_cfg['rl_num_steps'])} "
                f"rollout_batch={int(len(evaluation_df))} reward={float(history_row['baseline_reward']):.4f} "
                f"loss={float(history_row['loss']):.4f} proxy={proxy_text} "
                f"lr={float(history_row['learning_rate']):.4g} "
                f"fill={float(history_row['rollout_fill_fraction']):.3f}"
            ),
            echo=True,
        )
        completed_steps = int(step_idx)
        rollout_fill_fraction = float(history_row.get("rollout_fill_fraction", float("nan")))
        if (
            low_fill_patience > 0
            and np.isfinite(rollout_fill_fraction)
            and rollout_fill_fraction < float(low_fill_stop_ratio)
        ):
            consecutive_low_fill_steps += 1
        else:
            consecutive_low_fill_steps = 0
        if low_fill_patience > 0 and consecutive_low_fill_steps >= int(low_fill_patience):
            stop_reason = (
                "low_rollout_fill_fraction:"
                f"{rollout_fill_fraction:.4f}<{float(low_fill_stop_ratio):.4f}"
                f"_for_{int(consecutive_low_fill_steps)}_steps"
            )
            append_log_message(
                run_dirs["run_dir"],
                (
                    f"RL early stop | run={run_cfg['run_name']} mode={alignment_mode} "
                    f"step={int(step_idx)} reason={stop_reason}"
                ),
                echo=True,
            )
            break

    if not skip_disk_checkpoints:
        _save_rl_checkpoint(
            checkpoint_path=last_checkpoint_path,
            policy_model=policy_model,
            reference_checkpoint_path=warm_start.checkpoint_path,
            optimizer=optimizer,
            scheduler=scheduler,
            step_idx=int(completed_steps),
            best_proxy_metric_value=best_proxy_success,
            alignment_mode=alignment_mode,
            run_cfg=run_cfg,
            warm_start=warm_start,
            proxy_metric_name=proxy_objective_metric,
        )

    if skip_disk_checkpoints and checkpoint_mode == "proxy_property_success_hit_rate":
        if best_policy_state is not None:
            policy_model.load_state_dict(best_policy_state)
    elif checkpoint_mode == "proxy_property_success_hit_rate" and best_checkpoint_path.exists():
        load_step5_checkpoint_into_modules(
            checkpoint_path=best_checkpoint_path,
            diffusion_model=policy_model,
            aux_heads=None,
            device=device,
        )
    elif skip_disk_checkpoints and checkpoint_mode == "final_checkpoint":
        pass
    elif checkpoint_mode == "final_checkpoint":
        load_step5_checkpoint_into_modules(
            checkpoint_path=last_checkpoint_path,
            diffusion_model=policy_model,
            aux_heads=None,
            device=device,
        )

    history_df = pd.DataFrame(history_rows)
    proxy_history_df = pd.DataFrame(proxy_rows)
    history_df.to_csv(run_dirs["metrics_dir"] / f"{alignment_mode}_training_history.csv", index=False)
    if not proxy_history_df.empty:
        proxy_history_df.to_csv(run_dirs["metrics_dir"] / f"{alignment_mode}_proxy_history.csv", index=False)
    with open(run_dirs["metrics_dir"] / f"{alignment_mode}_training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_name": str(run_cfg["run_name"]),
                "alignment_mode": str(alignment_mode),
                "rl_prompt_source": str(s4_cfg["rl_prompt_source"]),
                "proxy_objective_metric": proxy_objective_metric,
                "best_proxy_metric_value": float(best_proxy_success),
                "checkpoint_selection_mode": checkpoint_mode,
                "best_checkpoint_path": (None if skip_disk_checkpoints else str(best_checkpoint_path)),
                "last_checkpoint_path": (None if skip_disk_checkpoints else str(last_checkpoint_path)),
                "num_history_rows": int(len(history_df)),
                "num_proxy_evals": int(len(proxy_history_df)),
                "steps_completed": int(len(history_df)),
                "configured_steps": int(s4_cfg["rl_num_steps"]),
                "stop_reason": stop_reason,
                "policy_update_epochs": int(policy_update_epochs),
                "ppo_clip_eps": (float(clip_eps) if alignment_mode in {"ppo", "grpo"} else None),
                "grpo_group_size": (int(grpo_group_size) if alignment_mode == "grpo" else None),
                "rl_diffusion_num_steps": int(rl_diffusion_num_steps),
                "disk_checkpoints_saved": bool(not skip_disk_checkpoints),
            },
            handle,
            indent=2,
        )
    append_log_message(
        run_dirs["run_dir"],
        (
            f"RL train complete | run={run_cfg['run_name']} mode={alignment_mode} "
            f"steps_completed={int(len(history_df))} best_proxy={float(best_proxy_success):.4f}"
            + (f" stop_reason={stop_reason}" if stop_reason else "")
        ),
        echo=True,
    )

    return RlTrainingArtifacts(
        tokenizer=warm_start.tokenizer,
        policy_model=policy_model,
        reference_model=reference_model,
        checkpoint_path=(
            best_checkpoint_path
            if (not skip_disk_checkpoints and best_checkpoint_path.exists())
            else last_checkpoint_path
        ),
        last_checkpoint_path=last_checkpoint_path,
        scaler=warm_start.scaler,
        history_df=history_df,
        proxy_history_df=proxy_history_df,
    )
