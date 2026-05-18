"""Shared family-aware sampling helpers for Step 5 decoding."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.chi.embeddings import load_backbone_from_step1
from src.evaluation.class_decode_constraints import (
    compute_class_token_logit_bias,
    load_decode_constraint_source_smiles,
    resolve_class_backbone_template_cores,
    resolve_class_decode_length_prior,
    resolve_class_decode_motifs,
)
from src.evaluation.polymer_class import BACKBONE_CLASS_MATCH_CLASSES, PolymerClassifier
from src.model.diffusion import DiscreteMaskingDiffusion
from src.sampling.sampler import ConstrainedSampler
from src.step5.config import ResolvedStep5Config, resolve_step5_sampling_num_steps
from src.data.tokenizer import PSmilesTokenizer
from src.utils.chemistry import (
    canonicalize_smiles,
    check_validity,
    count_stars,
    has_terminal_connection_stars,
)


@dataclass
class ResolvedClassSamplingPrior:
    """Resolved class-aware sampling priors shared across Step 5 samplers."""

    target_class: str
    family_sampling_mode: str
    family_sampling_scope: str
    source_smiles: List[str]
    motifs: List[str]
    motif_source: str
    motif_token_ids: List[List[int]]
    spans_per_sample: int
    center_min_frac: float
    center_max_frac: float
    length_prior_lengths: List[int]
    length_prior_source: Optional[str]
    length_prior_min_tokens: Optional[int]
    length_prior_max_tokens: Optional[int]
    fallback_source_lengths: List[int]
    class_token_logit_bias: Optional[List[float]]
    class_token_bias_strength: float
    backbone_template_enabled: bool
    backbone_template_cores: List[str]
    backbone_template_source: Optional[str]
    backbone_template_token_ids: List[List[int]]
    backbone_template_min_gap_tokens: int
    backbone_template_terminal_star_anchor: bool
    cycle_backbone_template_cores_across_targets: bool
    enforce_class_match: bool
    enforce_backbone_class_match: bool
    class_match_sampling_attempts_max: int
    class_match_oversample_factor: float
    class_match_min_request_size: int
    class_match_max_request_size: Optional[int]
    class_match_max_total_raw_samples: Optional[int]
    allow_partial_quota_return: bool
    partial_quota_min_fill_ratio: float
    enforce_star_ok_acceptance: bool
    reject_duplicate_canonical_acceptance: bool
    reject_sidechain_backbone_hybrids: bool
    allowed_atomic_symbols: Optional[List[str]]
    forbidden_tokens: Optional[List[str]]
    forbidden_token_ids: List[int]


class ClassConstrainedSamplingQuotaError(RuntimeError):
    """Raised when class-constrained final sampling cannot fill the requested quota."""

    def __init__(
        self,
        *,
        target_class: str,
        accepted_smiles: List[str],
        requested_num_samples: int,
        attempts_max: int,
        metadata: Dict[str, object],
        last_raw_smiles: Optional[List[str]] = None,
    ):
        message = (
            "Step 5 class-constrained sampling could not satisfy the target-class quota. "
            f"target_class={target_class!r} accepted={len(accepted_smiles)} requested={int(requested_num_samples)} "
            f"after {int(attempts_max)} attempts."
        )
        super().__init__(message)
        self.target_class = str(target_class)
        self.accepted_smiles = [str(smiles) for smiles in accepted_smiles]
        self.partial_smiles = list(self.accepted_smiles)
        self.requested_num_samples = int(requested_num_samples)
        self.attempts_max = int(attempts_max)
        self.metadata = dict(metadata)
        self.last_raw_smiles = [str(smiles) for smiles in (last_raw_smiles or [])]


def _normalize_family_sampling_mode(raw_mode: object) -> str:
    mode = str(raw_mode if raw_mode is not None else "motif").strip().lower()
    if mode not in {"none", "motif", "backbone_template", "sidechain_scaffold"}:
        raise ValueError(
            "decode constraint family_sampling_mode must be one of "
            "{'none', 'motif', 'backbone_template', 'sidechain_scaffold'}"
        )
    return mode


def _normalize_family_sampling_scope(raw_scope: object) -> str:
    scope = str(raw_scope if raw_scope is not None else "final_only").strip().lower()
    if scope not in {"final_only", "train_rollout_and_final"}:
        raise ValueError(
            "decode constraint family_sampling_scope must be one of "
            "{'final_only', 'train_rollout_and_final'}"
        )
    return scope


def _resolve_class_override(
    raw_overrides: object,
    *,
    target_class: str,
    default_value,
    cast,
):
    if not isinstance(raw_overrides, dict):
        return default_value
    for raw_key, raw_value in raw_overrides.items():
        key = str(raw_key).strip().lower()
        if key != str(target_class).strip().lower():
            continue
        return cast(raw_value)
    return default_value


def _resolve_optional_class_int_override(
    raw_overrides: object,
    *,
    target_class: str,
    default_value: Optional[int],
) -> Optional[int]:
    if not isinstance(raw_overrides, dict):
        return default_value
    for raw_key, raw_value in raw_overrides.items():
        key = str(raw_key).strip().lower()
        if key != str(target_class).strip().lower():
            continue
        if raw_value in {None, "", "null"}:
            return None
        return int(raw_value)
    return default_value


def _normalize_optional_string_list(raw_value: object, *, config_name: str) -> Optional[List[str]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and raw_value.strip().lower() in {"", "null"}:
        return None
    if isinstance(raw_value, str):
        tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    elif isinstance(raw_value, (list, tuple, set)):
        tokens = [str(token).strip() for token in raw_value if str(token).strip()]
    else:
        raise ValueError(
            f"{config_name} must be null, a comma-delimited string, or a list"
        )
    if not tokens:
        return None
    seen: set[str] = set()
    normalized: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _normalize_optional_symbol_list(raw_value: object) -> Optional[List[str]]:
    return _normalize_optional_string_list(
        raw_value,
        config_name="decode constraint allowed atomic symbol overrides",
    )


def _resolve_optional_class_symbol_list_override(
    raw_overrides: object,
    *,
    target_class: str,
    default_value: Optional[List[str]],
) -> Optional[List[str]]:
    if not isinstance(raw_overrides, dict):
        return default_value
    for raw_key, raw_value in raw_overrides.items():
        key = str(raw_key).strip().lower()
        if key != str(target_class).strip().lower():
            continue
        return _normalize_optional_symbol_list(raw_value)
    return default_value


def _resolve_optional_class_string_list_override(
    raw_overrides: object,
    *,
    target_class: str,
    default_value: Optional[List[str]],
    config_name: str,
) -> Optional[List[str]]:
    if not isinstance(raw_overrides, dict):
        return default_value
    for raw_key, raw_value in raw_overrides.items():
        key = str(raw_key).strip().lower()
        if key != str(target_class).strip().lower():
            continue
        return _normalize_optional_string_list(raw_value, config_name=config_name)
    return default_value


def _prepare_forbidden_token_ids(
    tokenizer: PSmilesTokenizer,
    forbidden_tokens: Optional[List[str]],
) -> List[int]:
    if not forbidden_tokens:
        return []
    seen: set[int] = set()
    token_ids: List[int] = []
    for token in forbidden_tokens:
        token_id = tokenizer.vocab.get(str(token))
        if token_id is None:
            continue
        token_id = int(token_id)
        if token_id in seen:
            continue
        seen.add(token_id)
        token_ids.append(token_id)
    return token_ids


def _resolve_forbidden_tokens_from_step5_cfg(
    step5_cfg: Dict[str, object],
    *,
    target_class: str,
) -> Optional[List[str]]:
    forbidden_tokens_default = _normalize_optional_string_list(
        step5_cfg.get("decode_constraint_forbidden_tokens"),
        config_name="decode constraint forbidden tokens",
    )
    return _resolve_optional_class_string_list_override(
        step5_cfg.get("decode_constraint_forbidden_tokens_overrides", {}),
        target_class=target_class,
        default_value=forbidden_tokens_default,
        config_name="decode constraint forbidden token overrides",
    )


def _resolve_base_config_forbidden_tokens(
    base_config: Dict[str, object],
    *,
    target_class: str,
) -> Optional[List[str]]:
    chi_cfg = base_config.get("chi_training", {})
    step5_cfg = (
        chi_cfg.get("step5_inverse_design", {})
        if isinstance(chi_cfg, dict) and isinstance(chi_cfg.get("step5_inverse_design", {}), dict)
        else {}
    )
    if not step5_cfg and isinstance(chi_cfg, dict):
        legacy_step5_cfg = chi_cfg.get("step5_class_inverse_design", {})
        if isinstance(legacy_step5_cfg, dict):
            step5_cfg = legacy_step5_cfg
    return _resolve_forbidden_tokens_from_step5_cfg(
        step5_cfg,
        target_class=target_class,
    )


def _prepare_decode_constraint_token_ids(
    tokenizer: PSmilesTokenizer,
    fragments: List[str],
) -> List[List[int]]:
    token_ids: List[List[int]] = []
    for fragment in fragments:
        tokens = tokenizer.tokenize(str(fragment))
        if not tokens or "".join(tokens) != str(fragment):
            continue
        ids = [tokenizer.vocab.get(token, tokenizer.unk_token_id) for token in tokens]
        if any(token_id == tokenizer.unk_token_id for token_id in ids):
            continue
        if tokenizer.get_star_token_id() in ids:
            continue
        token_ids.append(ids)
    return token_ids


def _build_decode_constraint_spans(
    *,
    motif_token_ids: List[List[int]],
    lengths: List[int],
    center_min_frac: float,
    center_max_frac: float,
    seq_length: int,
) -> tuple[List[List[int]], List[int], List[int]]:
    if not motif_token_ids:
        raise ValueError("motif_token_ids is empty")
    if not lengths:
        return [], [], []
    if not (0.0 <= center_min_frac <= center_max_frac <= 1.0):
        raise ValueError("decode constraint center fractions must satisfy 0 <= min <= max <= 1")

    max_motif_len = max(len(ids) for ids in motif_token_ids)
    adjusted_lengths = [
        min(seq_length, max(int(raw_length), max_motif_len + 4))
        for raw_length in lengths
    ]

    chosen_spans: List[List[int]] = []
    start_positions: List[int] = []
    for effective_length in adjusted_lengths:
        fitting = [ids for ids in motif_token_ids if len(ids) <= max(1, effective_length - 2)]
        if not fitting:
            raise ValueError(
                f"No decode-time motif fits effective sequence length {effective_length}. "
                "Increase allowed lengths or shorten motifs."
            )
        motif_ids = fitting[np.random.randint(0, len(fitting))]
        max_start = int(effective_length) - 1 - len(motif_ids)
        if max_start < 1:
            raise ValueError(
                f"Motif length {len(motif_ids)} does not fit sequence length {effective_length}"
            )
        center_frac = (
            center_min_frac
            if center_min_frac == center_max_frac
            else float(np.random.uniform(center_min_frac, center_max_frac))
        )
        center_target = center_frac * float(max(1, effective_length - 1))
        candidate_start = int(round(center_target - (0.5 * len(motif_ids))))
        start = max(1, min(max_start, candidate_start))
        chosen_spans.append(motif_ids)
        start_positions.append(start)

    return chosen_spans, start_positions, adjusted_lengths


def _place_multi_spans_for_length(
    *,
    motifs: List[List[int]],
    effective_length: int,
    center_min_frac: float,
    center_max_frac: float,
    min_gap_tokens: int = 2,
) -> List[int]:
    if not motifs:
        return []

    usable_token_count = max(0, int(effective_length) - 2)
    required_token_count = sum(len(motif_ids) for motif_ids in motifs) + (
        max(0, len(motifs) - 1) * int(min_gap_tokens)
    )
    if required_token_count > usable_token_count:
        raise ValueError(
            f"Could not place {len(motifs)} decode-time motifs within effective length {effective_length}"
        )

    if len(motifs) == 1:
        motif_ids = motifs[0]
        max_start = int(effective_length) - 1 - len(motif_ids)
        center_frac = (
            center_min_frac
            if center_min_frac == center_max_frac
            else float(np.random.uniform(center_min_frac, center_max_frac))
        )
        center_target = center_frac * float(max(1, effective_length - 1))
        candidate_start = int(round(center_target - (0.5 * len(motif_ids))))
        return [max(1, min(max_start, candidate_start))]

    anchors = np.linspace(center_min_frac, center_max_frac, len(motifs) + 2)[1:-1]
    future_requirements: List[int] = [0] * len(motifs)
    running_requirement = 0
    for idx in range(len(motifs) - 1, -1, -1):
        future_requirements[idx] = running_requirement
        running_requirement += len(motifs[idx]) + int(min_gap_tokens)

    starts: List[int] = []
    prev_end = 1
    for idx, (anchor_frac, motif_ids) in enumerate(zip(anchors, motifs)):
        max_start = int(effective_length) - 1 - len(motif_ids) - future_requirements[idx]
        candidate_start = int(
            round(anchor_frac * float(max(1, effective_length - 1)) - (0.5 * len(motif_ids)))
        )
        min_start = 1 if idx == 0 else prev_end + int(min_gap_tokens)
        if max_start < min_start:
            raise ValueError(
                f"Could not place {len(motifs)} decode-time motifs within effective length {effective_length}"
            )
        start = max(min_start, min(max_start, candidate_start))
        starts.append(start)
        prev_end = start + len(motif_ids)
    return starts


def _build_decode_constraint_multi_spans(
    *,
    motif_token_ids: List[List[int]],
    lengths: List[int],
    center_min_frac: float,
    center_max_frac: float,
    seq_length: int,
    spans_per_sample: int,
    min_gap_tokens: int = 2,
) -> tuple[List[List[List[int]]], List[List[int]], List[int]]:
    if spans_per_sample < 1:
        raise ValueError(f"spans_per_sample must be >= 1, got {spans_per_sample}")
    if spans_per_sample == 1:
        spans, starts, adjusted = _build_decode_constraint_spans(
            motif_token_ids=motif_token_ids,
            lengths=lengths,
            center_min_frac=center_min_frac,
            center_max_frac=center_max_frac,
            seq_length=seq_length,
        )
        return [[span] for span in spans], [[start] for start in starts], adjusted

    max_motif_len = max(len(ids) for ids in motif_token_ids)
    adjusted_lengths = [
        min(seq_length, max(int(raw_length), max_motif_len + 4))
        for raw_length in lengths
    ]

    chosen_spans: List[List[List[int]]] = []
    start_positions: List[List[int]] = []
    for effective_length in adjusted_lengths:
        fitting = [ids for ids in motif_token_ids if len(ids) <= max(1, effective_length - 2)]
        if not fitting:
            raise ValueError(
                f"No decode-time motif fits effective sequence length {effective_length}. "
                "Increase allowed lengths or shorten motifs."
            )
        min_len = min(len(ids) for ids in fitting)
        shortest_pool = [ids for ids in fitting if len(ids) <= (min_len + 1)]
        usable_pool = shortest_pool if shortest_pool else fitting
        max_spans_fit = max(
            1,
            int((max(1, effective_length - 2) + int(min_gap_tokens)) // (min_len + int(min_gap_tokens))),
        )
        sample_span_count = max(1, min(int(spans_per_sample), int(max_spans_fit)))

        sample_spans: List[List[int]] | None = None
        sample_starts: List[int] | None = None
        available_token_count = max(0, int(effective_length) - 2)
        for current_span_count in range(sample_span_count, 0, -1):
            pool = usable_pool if current_span_count > 1 else fitting
            for _ in range(32):
                candidate_spans = [
                    list(pool[np.random.randint(0, len(pool))])
                    for _ in range(current_span_count)
                ]
                required_token_count = sum(len(ids) for ids in candidate_spans) + (
                    max(0, current_span_count - 1) * int(min_gap_tokens)
                )
                if required_token_count > available_token_count:
                    continue
                try:
                    candidate_starts = _place_multi_spans_for_length(
                        motifs=candidate_spans,
                        effective_length=int(effective_length),
                        center_min_frac=float(center_min_frac),
                        center_max_frac=float(center_max_frac),
                        min_gap_tokens=int(min_gap_tokens),
                    )
                except ValueError:
                    continue
                sample_spans = candidate_spans
                sample_starts = candidate_starts
                break
            if sample_spans is not None and sample_starts is not None:
                break
        if sample_spans is None or sample_starts is None:
            raise ValueError(
                f"Could not construct {sample_span_count} decode-time motifs within effective length {effective_length}"
            )
        chosen_spans.append(sample_spans)
        start_positions.append(sample_starts)

    return chosen_spans, start_positions, adjusted_lengths


def _build_backbone_template_multi_spans(
    *,
    backbone_template_token_ids: List[List[int]],
    lengths: List[int],
    center_min_frac: float,
    center_max_frac: float,
    seq_length: int,
    star_token_id: int,
    backbone_gap_token_id: int,
    min_gap_tokens: int,
    anchor_terminal_stars: bool,
    sampling_state: Optional[Dict[str, object]] = None,
) -> tuple[List[List[List[int]]], List[List[int]], List[int]]:
    if not backbone_template_token_ids:
        raise ValueError("backbone_template_token_ids is empty")
    if star_token_id < 0:
        raise ValueError("Tokenizer does not define a '*' token for backbone-template sampling")
    if backbone_gap_token_id < 0:
        raise ValueError("Tokenizer does not define a valid backbone spacer token for backbone-template sampling")
    if not lengths:
        return [], [], []

    max_core_len = max(len(ids) for ids in backbone_template_token_ids)
    min_required_length = max_core_len + 4 + (2 * int(min_gap_tokens))
    adjusted_lengths = [
        min(seq_length, max(int(raw_length), int(min_required_length)))
        for raw_length in lengths
    ]

    chosen_spans: List[List[List[int]]] = []
    start_positions: List[List[int]] = []
    star_span = [int(star_token_id)]
    gap_span = [int(backbone_gap_token_id)] * max(0, int(min_gap_tokens))
    for effective_length in adjusted_lengths:
        fitting_indices = [
            idx
            for idx, ids in enumerate(backbone_template_token_ids)
            if len(ids) <= max(1, int(effective_length) - 4 - (2 * int(min_gap_tokens)))
        ]
        if not fitting_indices:
            raise ValueError(
                f"No backbone-template core fits effective sequence length {effective_length}. "
                "Increase allowed lengths or shorten the template core."
            )
        selected_index: int
        if sampling_state is not None:
            used_raw = sampling_state.get("used_backbone_template_core_indices", [])
            if isinstance(used_raw, (list, tuple, set)):
                used_indices = {int(value) for value in used_raw}
            else:
                used_indices = set()
            available_indices = [idx for idx in fitting_indices if idx not in used_indices]
            if not available_indices:
                used_indices = set()
                available_indices = list(fitting_indices)
            selected_index = int(available_indices[np.random.randint(0, len(available_indices))])
            used_indices.add(selected_index)
            sampling_state["used_backbone_template_core_indices"] = sorted(used_indices)
        else:
            selected_index = int(fitting_indices[np.random.randint(0, len(fitting_indices))])
        core_ids = list(backbone_template_token_ids[selected_index])
        center_frac = (
            center_min_frac
            if center_min_frac == center_max_frac
            else float(np.random.uniform(center_min_frac, center_max_frac))
        )
        if anchor_terminal_stars:
            left_span = star_span + gap_span
            right_span = gap_span + star_span
            left_start = 1
            right_start = int(effective_length) - 1 - len(right_span)
            min_core_start = left_start + len(left_span)
            max_core_start = right_start - len(core_ids)
            if max_core_start < min_core_start:
                raise ValueError(
                    f"Anchored backbone-template core length {len(core_ids)} does not fit sequence length {effective_length}"
                )
            candidate_core_start = int(
                round(center_frac * float(max(1, effective_length - 1)) - (0.5 * len(core_ids)))
            )
            core_start = max(min_core_start, min(max_core_start, candidate_core_start))
            chosen_spans.append([left_span, core_ids, right_span])
            start_positions.append([left_start, core_start, right_start])
            continue

        scaffold_ids = star_span + gap_span + core_ids + gap_span + star_span
        max_start = int(effective_length) - 1 - len(scaffold_ids)
        if max_start < 1:
            raise ValueError(
                f"Backbone-template scaffold length {len(scaffold_ids)} does not fit sequence length {effective_length}"
            )
        candidate_start = int(round(center_frac * float(max(1, effective_length - 1)) - (0.5 * len(scaffold_ids))))
        scaffold_start = max(1, min(max_start, candidate_start))
        chosen_spans.append([scaffold_ids])
        start_positions.append([scaffold_start])

    return chosen_spans, start_positions, adjusted_lengths


def load_step1_diffusion(
    resolved: ResolvedStep5Config,
    *,
    device: str,
) -> tuple[PSmilesTokenizer, DiscreteMaskingDiffusion, Path]:
    """Load tokenizer + Step 1 diffusion model for S0/S1."""

    try:
        tokenizer, backbone, checkpoint_path = load_backbone_from_step1(
            config=resolved.base_config,
            model_size=resolved.model_size,
            split_mode=resolved.split_mode,
            checkpoint_path=None,
            device=device,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Step 5 frozen sampling requires an existing Step 1 backbone checkpoint and tokenizer. "
            f"model_size={resolved.model_size!r}, split_mode={resolved.split_mode!r}. "
            f"Original error: {exc}"
        ) from exc
    diffusion = DiscreteMaskingDiffusion(
        backbone=backbone,
        num_steps=resolved.base_config["diffusion"]["num_steps"],
        beta_min=resolved.base_config["diffusion"]["beta_min"],
        beta_max=resolved.base_config["diffusion"]["beta_max"],
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ).to(device)
    diffusion.eval()
    return tokenizer, diffusion, checkpoint_path


def _token_lengths_from_smiles(
    tokenizer: PSmilesTokenizer,
    smiles_list: Iterable[str],
) -> List[int]:
    lengths: List[int] = []
    for smiles in smiles_list:
        token_len = len(tokenizer.tokenize(str(smiles))) + 2
        token_len = max(2, min(int(tokenizer.max_length), int(token_len)))
        lengths.append(token_len)
    return lengths


def resolve_class_sampling_prior(
    resolved: ResolvedStep5Config,
    run_cfg: Dict[str, object],
    tokenizer: PSmilesTokenizer,
    *,
    metrics_dir: Optional[Path] = None,
) -> ResolvedClassSamplingPrior:
    """Resolve shared Step 5-style class-aware sampling priors."""

    chi_cfg = resolved.base_config.get("chi_training", {})
    step5_cfg = (
        chi_cfg.get("step5_inverse_design", {})
        if isinstance(chi_cfg.get("step5_inverse_design", {}), dict)
        else {}
    )
    if not step5_cfg:
        legacy_step5_cfg = chi_cfg.get("step5_class_inverse_design", {})
        if isinstance(legacy_step5_cfg, dict):
            step5_cfg = legacy_step5_cfg
    target_class = resolved.c_target
    source_smiles = load_decode_constraint_source_smiles(Path(resolved.base_config["paths"]["data_dir"]))
    decode_constraint_enabled = bool(step5_cfg.get("decode_constraint_enabled", True))
    resolution_strategy = str(
        step5_cfg.get("decode_constraint_resolution_strategy", "configured_or_defaults")
    ).strip().lower()
    configured_bank_path_raw = step5_cfg.get("decode_constraint_motif_bank_json")
    configured_bank_path = (
        Path(configured_bank_path_raw).resolve()
        if configured_bank_path_raw not in {None, "", "null"}
        else None
    )
    resolve_source_smiles = (
        source_smiles if resolution_strategy == "configured_or_local_mined_or_defaults" else []
    )
    motifs, motif_source = resolve_class_decode_motifs(
        target_class=target_class,
        tokenizer=tokenizer,
        source_smiles=resolve_source_smiles,
        patterns=resolved.polymer_patterns,
        configured_bank_path=configured_bank_path,
        max_motifs=int(step5_cfg.get("decode_constraint_max_motifs", 6)),
        resolution_strategy=resolution_strategy,
    )
    motif_token_ids = _prepare_decode_constraint_token_ids(tokenizer, motifs)
    use_class_length_prior = bool(step5_cfg.get("decode_constraint_use_class_length_prior", True))
    length_prior_lengths: List[int] = []
    length_prior_source: Optional[str] = None
    if use_class_length_prior:
        length_prior_lengths, length_prior_source = resolve_class_decode_length_prior(
            target_class=target_class,
            tokenizer=tokenizer,
            source_smiles=source_smiles,
            patterns=resolved.polymer_patterns,
            max_length=int(tokenizer.max_length),
        )
    raw_length_prior_min_tokens = step5_cfg.get("decode_constraint_length_prior_min_tokens")
    length_prior_min_tokens_default = (
        None
        if raw_length_prior_min_tokens in {None, "", "null"}
        else int(raw_length_prior_min_tokens)
    )
    length_prior_min_tokens = _resolve_optional_class_int_override(
        step5_cfg.get("decode_constraint_length_prior_min_tokens_overrides", {}),
        target_class=target_class,
        default_value=length_prior_min_tokens_default,
    )
    raw_length_prior_max_tokens = step5_cfg.get("decode_constraint_length_prior_max_tokens")
    length_prior_max_tokens_default = (
        None
        if raw_length_prior_max_tokens in {None, "", "null"}
        else int(raw_length_prior_max_tokens)
    )
    length_prior_max_tokens = _resolve_optional_class_int_override(
        step5_cfg.get("decode_constraint_length_prior_max_tokens_overrides", {}),
        target_class=target_class,
        default_value=length_prior_max_tokens_default,
    )
    fallback_source_lengths = _token_lengths_from_smiles(tokenizer, source_smiles)
    class_token_bias_strength = _resolve_class_override(
        step5_cfg.get("decode_constraint_class_token_bias_strength_overrides", {}),
        target_class=target_class,
        default_value=float(run_cfg.get("class_token_bias_strength", 1.5)),
        cast=float,
    )
    enforce_class_match = bool(step5_cfg.get("decode_constraint_enforce_class_match", False))
    enforce_backbone_class_match = bool(step5_cfg.get("decode_constraint_enforce_backbone_class_match", False))
    class_match_sampling_attempts_max = _resolve_class_override(
        step5_cfg.get("decode_constraint_class_match_sampling_attempts_max_overrides", {}),
        target_class=target_class,
        default_value=int(step5_cfg.get("decode_constraint_class_match_sampling_attempts_max", 12)),
        cast=int,
    )
    class_match_oversample_factor = _resolve_class_override(
        step5_cfg.get("decode_constraint_class_match_oversample_factor_overrides", {}),
        target_class=target_class,
        default_value=float(step5_cfg.get("decode_constraint_class_match_oversample_factor", 2.0)),
        cast=float,
    )
    class_match_min_request_size = _resolve_class_override(
        step5_cfg.get("decode_constraint_class_match_min_request_size_overrides", {}),
        target_class=target_class,
        default_value=int(step5_cfg.get("decode_constraint_class_match_min_request_size", 1)),
        cast=int,
    )
    raw_class_match_max_request_size = step5_cfg.get("decode_constraint_class_match_max_request_size")
    class_match_max_request_size_default = (
        None
        if raw_class_match_max_request_size in {None, "", "null"}
        else int(raw_class_match_max_request_size)
    )
    class_match_max_request_size = _resolve_optional_class_int_override(
        step5_cfg.get("decode_constraint_class_match_max_request_size_overrides", {}),
        target_class=target_class,
        default_value=class_match_max_request_size_default,
    )
    raw_class_match_max_total_raw_samples = step5_cfg.get(
        "decode_constraint_class_match_max_total_raw_samples"
    )
    class_match_max_total_raw_samples_default = (
        None
        if raw_class_match_max_total_raw_samples in {None, "", "null"}
        else int(raw_class_match_max_total_raw_samples)
    )
    class_match_max_total_raw_samples = _resolve_optional_class_int_override(
        step5_cfg.get("decode_constraint_class_match_max_total_raw_samples_overrides", {}),
        target_class=target_class,
        default_value=class_match_max_total_raw_samples_default,
    )
    allow_partial_quota_return = _resolve_class_override(
        step5_cfg.get("decode_constraint_allow_partial_quota_return_overrides", {}),
        target_class=target_class,
        default_value=bool(step5_cfg.get("decode_constraint_allow_partial_quota_return", False)),
        cast=bool,
    )
    partial_quota_min_fill_ratio = _resolve_class_override(
        step5_cfg.get("decode_constraint_partial_quota_min_fill_ratio_overrides", {}),
        target_class=target_class,
        default_value=float(step5_cfg.get("decode_constraint_partial_quota_min_fill_ratio", 1.0)),
        cast=float,
    )
    enforce_star_ok_acceptance = _resolve_class_override(
        step5_cfg.get("decode_constraint_enforce_star_ok_acceptance_overrides", {}),
        target_class=target_class,
        default_value=bool(step5_cfg.get("decode_constraint_enforce_star_ok_acceptance", False)),
        cast=bool,
    )
    reject_duplicate_canonical_acceptance = bool(
        step5_cfg.get("decode_constraint_reject_duplicate_canonical_acceptance", True)
    )
    reject_sidechain_backbone_hybrids = _resolve_class_override(
        step5_cfg.get("decode_constraint_reject_sidechain_backbone_hybrid_overrides", {}),
        target_class=target_class,
        default_value=bool(step5_cfg.get("decode_constraint_reject_sidechain_backbone_hybrid", False)),
        cast=bool,
    )
    allowed_atomic_symbols_default = _normalize_optional_symbol_list(
        step5_cfg.get("decode_constraint_allowed_atomic_symbols")
    )
    allowed_atomic_symbols = _resolve_optional_class_symbol_list_override(
        step5_cfg.get("decode_constraint_allowed_atomic_symbols_overrides", {}),
        target_class=target_class,
        default_value=allowed_atomic_symbols_default,
    )
    forbidden_tokens = _resolve_forbidden_tokens_from_step5_cfg(
        step5_cfg,
        target_class=target_class,
    )
    forbidden_token_ids = _prepare_forbidden_token_ids(tokenizer, forbidden_tokens)
    family_sampling_scope = _normalize_family_sampling_scope(
        step5_cfg.get("decode_constraint_family_sampling_scope", "final_only")
    )
    default_family_sampling_mode = _normalize_family_sampling_mode(
        step5_cfg.get("decode_constraint_family_sampling_default_mode", "motif")
    )
    raw_mode_overrides = step5_cfg.get("decode_constraint_family_sampling_mode_overrides", {})
    family_sampling_mode_overrides: Dict[str, str] = {}
    if isinstance(raw_mode_overrides, dict):
        for raw_key, raw_value in raw_mode_overrides.items():
            key = str(raw_key).strip().lower()
            if not key:
                continue
            family_sampling_mode_overrides[key] = _normalize_family_sampling_mode(raw_value)
    configured_template_classes_raw = step5_cfg.get("decode_constraint_backbone_template_classes", [])
    if isinstance(configured_template_classes_raw, str):
        configured_template_classes = {
            token.strip().lower()
            for token in configured_template_classes_raw.split(",")
            if token.strip()
        }
    else:
        configured_template_classes = {
            str(token).strip().lower()
            for token in list(configured_template_classes_raw)
            if str(token).strip()
        }
    backbone_template_globally_enabled = bool(step5_cfg.get("decode_constraint_backbone_template_enabled", True))
    center_min_frac = _resolve_class_override(
        step5_cfg.get("decode_constraint_center_min_frac_overrides", {}),
        target_class=target_class,
        default_value=float(step5_cfg.get("decode_constraint_center_min_frac", 0.25)),
        cast=float,
    )
    center_max_frac = _resolve_class_override(
        step5_cfg.get("decode_constraint_center_max_frac_overrides", {}),
        target_class=target_class,
        default_value=float(step5_cfg.get("decode_constraint_center_max_frac", 0.75)),
        cast=float,
    )
    explicit_family_mode = family_sampling_mode_overrides.get(target_class)
    family_sampling_mode = explicit_family_mode or default_family_sampling_mode
    if not decode_constraint_enabled:
        family_sampling_mode = "none"
    elif explicit_family_mode is None and (
        backbone_template_globally_enabled
        and target_class in BACKBONE_CLASS_MATCH_CLASSES
        and target_class in configured_template_classes
    ):
        family_sampling_mode = "backbone_template"

    class_token_logit_bias = None
    if family_sampling_mode != "none":
        class_token_logit_bias = compute_class_token_logit_bias(
            target_class=target_class,
            tokenizer=tokenizer,
            source_smiles=source_smiles,
            patterns=resolved.polymer_patterns,
            bias_strength=class_token_bias_strength,
        )

    template_enabled = family_sampling_mode == "backbone_template"
    backbone_template_cores: List[str] = []
    backbone_template_source: Optional[str] = None
    backbone_template_token_ids: List[List[int]] = []
    if template_enabled:
        if not backbone_template_globally_enabled or target_class not in BACKBONE_CLASS_MATCH_CLASSES:
            family_sampling_mode = "motif" if motif_token_ids else "none"
            template_enabled = False
        elif explicit_family_mode is None and target_class not in configured_template_classes:
            family_sampling_mode = "motif" if motif_token_ids else "none"
            template_enabled = False
    if template_enabled:
        backbone_template_max_templates = _resolve_class_override(
            step5_cfg.get("decode_constraint_backbone_template_max_templates_overrides", {}),
            target_class=target_class,
            default_value=int(step5_cfg.get("decode_constraint_backbone_template_max_templates", 3)),
            cast=int,
        )
        backbone_template_cores, backbone_template_source = resolve_class_backbone_template_cores(
            target_class=target_class,
            tokenizer=tokenizer,
            max_templates=int(backbone_template_max_templates),
        )
        backbone_template_token_ids = _prepare_decode_constraint_token_ids(tokenizer, backbone_template_cores)
        template_enabled = bool(backbone_template_token_ids)
        if not template_enabled:
            family_sampling_mode = "motif" if motif_token_ids else "none"
    backbone_template_min_gap_tokens = _resolve_class_override(
        step5_cfg.get("decode_constraint_backbone_template_min_gap_tokens_overrides", {}),
        target_class=target_class,
        default_value=int(step5_cfg.get("decode_constraint_backbone_template_min_gap_tokens", 1)),
        cast=int,
    )
    backbone_template_terminal_star_anchor = _resolve_class_override(
        step5_cfg.get("decode_constraint_backbone_template_terminal_star_anchor_overrides", {}),
        target_class=target_class,
        default_value=bool(step5_cfg.get("decode_constraint_backbone_template_terminal_star_anchor", False)),
        cast=bool,
    )
    cycle_backbone_template_cores_across_targets = _resolve_class_override(
        step5_cfg.get("decode_constraint_cycle_backbone_template_cores_across_targets_overrides", {}),
        target_class=target_class,
        default_value=bool(step5_cfg.get("decode_constraint_cycle_backbone_template_cores_across_targets", False)),
        cast=bool,
    )

    prior = ResolvedClassSamplingPrior(
        target_class=target_class,
        family_sampling_mode=str(family_sampling_mode),
        family_sampling_scope=str(family_sampling_scope),
        source_smiles=source_smiles,
        motifs=motifs,
        motif_source=motif_source,
        motif_token_ids=motif_token_ids,
        spans_per_sample=int(step5_cfg.get("decode_constraint_spans_per_sample", 2)),
        center_min_frac=float(center_min_frac),
        center_max_frac=float(center_max_frac),
        length_prior_lengths=length_prior_lengths,
        length_prior_source=length_prior_source,
        length_prior_min_tokens=length_prior_min_tokens,
        length_prior_max_tokens=length_prior_max_tokens,
        fallback_source_lengths=fallback_source_lengths,
        class_token_logit_bias=class_token_logit_bias,
        class_token_bias_strength=class_token_bias_strength,
        backbone_template_enabled=bool(template_enabled),
        backbone_template_cores=backbone_template_cores,
        backbone_template_source=backbone_template_source,
        backbone_template_token_ids=backbone_template_token_ids,
        backbone_template_min_gap_tokens=backbone_template_min_gap_tokens,
        backbone_template_terminal_star_anchor=bool(backbone_template_terminal_star_anchor),
        cycle_backbone_template_cores_across_targets=bool(cycle_backbone_template_cores_across_targets),
        enforce_class_match=bool(enforce_class_match),
        enforce_backbone_class_match=bool(enforce_backbone_class_match),
        class_match_sampling_attempts_max=int(class_match_sampling_attempts_max),
        class_match_oversample_factor=float(class_match_oversample_factor),
        class_match_min_request_size=max(1, int(class_match_min_request_size)),
        class_match_max_request_size=(
            max(1, int(class_match_max_request_size))
            if class_match_max_request_size is not None
            else None
        ),
        class_match_max_total_raw_samples=(
            max(1, int(class_match_max_total_raw_samples))
            if class_match_max_total_raw_samples is not None
            else None
        ),
        allow_partial_quota_return=bool(allow_partial_quota_return),
        partial_quota_min_fill_ratio=float(partial_quota_min_fill_ratio),
        enforce_star_ok_acceptance=bool(enforce_star_ok_acceptance),
        reject_duplicate_canonical_acceptance=bool(reject_duplicate_canonical_acceptance),
        reject_sidechain_backbone_hybrids=bool(reject_sidechain_backbone_hybrids),
        allowed_atomic_symbols=list(allowed_atomic_symbols) if allowed_atomic_symbols else None,
        forbidden_tokens=list(forbidden_tokens) if forbidden_tokens else None,
        forbidden_token_ids=forbidden_token_ids,
    )

    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        backbone_core_token_lengths = [
            int(len(tokenizer.tokenize(core)))
            for core in prior.backbone_template_cores
        ]
        with open(metrics_dir / "decode_constraint_motif_bank_resolved.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "target_class": target_class,
                    "family_sampling_mode": prior.family_sampling_mode,
                    "family_sampling_scope": prior.family_sampling_scope,
                    "source": motif_source,
                    "motifs": motifs,
                },
                handle,
                indent=2,
            )
        if length_prior_lengths:
            with open(metrics_dir / "decode_constraint_length_prior_resolved.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "target_class": target_class,
                        "source": length_prior_source,
                        "lengths": length_prior_lengths,
                        "min_tokens": prior.length_prior_min_tokens,
                        "max_tokens": prior.length_prior_max_tokens,
                    },
                    handle,
                    indent=2,
                )
        if class_token_logit_bias is not None:
            with open(metrics_dir / "decode_constraint_class_token_bias.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "target_class": target_class,
                        "bias_strength": class_token_bias_strength,
                        "bias": class_token_logit_bias,
                    },
                    handle,
                )
        with open(metrics_dir / "decode_constraint_backbone_template_resolved.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "target_class": target_class,
                    "family_sampling_mode": prior.family_sampling_mode,
                    "family_sampling_scope": prior.family_sampling_scope,
                    "enabled": bool(prior.backbone_template_enabled),
                    "source": prior.backbone_template_source,
                    "min_gap_tokens": int(prior.backbone_template_min_gap_tokens),
                    "terminal_star_anchor": bool(prior.backbone_template_terminal_star_anchor),
                    "cycle_cores_across_targets": bool(prior.cycle_backbone_template_cores_across_targets),
                    "center_min_frac": float(prior.center_min_frac),
                    "center_max_frac": float(prior.center_max_frac),
                    "cores": prior.backbone_template_cores,
                    "core_token_lengths": backbone_core_token_lengths,
                    "max_core_token_length": max(backbone_core_token_lengths) if backbone_core_token_lengths else 0,
                    "enforce_class_match": bool(prior.enforce_class_match),
                    "enforce_backbone_class_match": bool(prior.enforce_backbone_class_match),
                    "class_match_sampling_attempts_max": int(prior.class_match_sampling_attempts_max),
                    "class_match_oversample_factor": float(prior.class_match_oversample_factor),
                    "class_match_min_request_size": int(prior.class_match_min_request_size),
                    "enforce_star_ok_acceptance": bool(prior.enforce_star_ok_acceptance),
                    "reject_duplicate_canonical_acceptance": bool(prior.reject_duplicate_canonical_acceptance),
                    "reject_sidechain_backbone_hybrids": bool(prior.reject_sidechain_backbone_hybrids),
                    "allowed_atomic_symbols": prior.allowed_atomic_symbols,
                    "forbidden_tokens": prior.forbidden_tokens,
                    "forbidden_token_ids": prior.forbidden_token_ids,
                },
                handle,
                indent=2,
            )

    return prior


def _sample_lengths(
    *,
    prior: ResolvedClassSamplingPrior,
    tokenizer: PSmilesTokenizer,
    num_samples: int,
    sampling_cfg: Dict[str, object],
) -> List[int]:
    if prior.length_prior_lengths:
        sampled = np.random.choice(
            np.asarray(prior.length_prior_lengths, dtype=np.int32),
            size=int(num_samples),
            replace=True,
        )
        lengths = [int(x) for x in sampled.tolist()]
    elif bool(sampling_cfg.get("variable_length", False)):
        min_len = int(sampling_cfg.get("variable_length_min_tokens", 12))
        max_len = int(min(int(tokenizer.max_length), sampling_cfg.get("variable_length_max_tokens", 100)))
        lengths = [int(np.random.randint(min_len, max_len + 1)) for _ in range(int(num_samples))]
    elif prior.fallback_source_lengths:
        sampled = np.random.choice(
            np.asarray(prior.fallback_source_lengths, dtype=np.int32),
            size=int(num_samples),
            replace=True,
        )
        lengths = [int(x) for x in sampled.tolist()]
    else:
        lengths = [int(tokenizer.max_length)] * int(num_samples)

    motif_buffer = max((len(ids) for ids in prior.motif_token_ids), default=0) + 4
    if prior.backbone_template_enabled and prior.backbone_template_token_ids:
        motif_buffer = max(
            motif_buffer,
            max((len(ids) for ids in prior.backbone_template_token_ids), default=0)
            + 4
            + (2 * int(prior.backbone_template_min_gap_tokens)),
        )
    min_length = max(2, int(prior.length_prior_min_tokens or 0), int(motif_buffer))
    max_length = int(tokenizer.max_length)
    if prior.length_prior_max_tokens is not None:
        max_length = min(max_length, int(prior.length_prior_max_tokens))
    max_length = max(int(min_length), int(max_length))
    return [
        min(int(tokenizer.max_length), max(int(min_length), min(int(max_length), int(length))))
        for length in lengths
    ]


def create_constrained_sampler(
    *,
    diffusion_model: DiscreteMaskingDiffusion,
    tokenizer: PSmilesTokenizer,
    resolved: ResolvedStep5Config,
    prior: ResolvedClassSamplingPrior,
    device: str,
) -> ConstrainedSampler:
    """Create a Step 5 sampler with shared class-prior settings."""

    sampler = create_raw_step1_sampler(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        resolved=resolved,
        device=device,
    )
    sampler.set_class_token_bias_start_frac(float(resolved.step5.get("class_token_bias_start_frac", 0.0)))
    if prior.class_token_logit_bias is not None:
        sampler.set_class_token_logit_bias(prior.class_token_logit_bias)
    sampler.set_forbidden_tokens(prior.forbidden_tokens)
    return sampler


def create_raw_step1_sampler(
    *,
    diffusion_model: DiscreteMaskingDiffusion,
    tokenizer: PSmilesTokenizer,
    resolved: ResolvedStep5Config,
    device: str,
) -> ConstrainedSampler:
    """Create the plain Step 1 sampler without Step 5 class-aware priors."""

    sampling_cfg = resolved.base_config.get("sampling", {})
    sampler = ConstrainedSampler(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        num_steps=resolve_step5_sampling_num_steps(resolved.step5, resolved.base_config),
        temperature=float(resolved.step5["sampling_temperature"]),
        top_k=sampling_cfg.get("top_k"),
        top_p=sampling_cfg.get("top_p"),
        target_stars=int(sampling_cfg.get("target_stars", 2)),
        use_constraints=bool(sampling_cfg.get("use_constraints", True)),
        device=device,
    )
    sampler.set_forbidden_tokens(
        _resolve_base_config_forbidden_tokens(
            resolved.base_config,
            target_class=resolved.c_target,
        )
    )
    return sampler


def _sample_raw_step1_lengths(
    *,
    tokenizer: PSmilesTokenizer,
    num_samples: int,
    sampling_cfg: Dict[str, object],
    source_lengths: Optional[List[int]] = None,
) -> List[int]:
    if bool(sampling_cfg.get("variable_length", False)):
        min_len = int(sampling_cfg.get("variable_length_min_tokens", 12))
        max_len = int(min(int(tokenizer.max_length), sampling_cfg.get("variable_length_max_tokens", 100)))
        lengths = [int(np.random.randint(min_len, max_len + 1)) for _ in range(int(num_samples))]
    elif source_lengths:
        sampled = np.random.choice(
            np.asarray(source_lengths, dtype=np.int32),
            size=int(num_samples),
            replace=True,
        )
        lengths = [int(value) for value in sampled.tolist()]
    else:
        lengths = [int(tokenizer.max_length)] * int(num_samples)
    return [
        min(int(tokenizer.max_length), max(2, int(length)))
        for length in lengths
    ]


def sample_raw_step1_unconditional(
    *,
    sampler: ConstrainedSampler,
    tokenizer: PSmilesTokenizer,
    resolved: ResolvedStep5Config,
    num_samples: int,
    show_progress: bool = True,
    source_lengths: Optional[List[int]] = None,
) -> Tuple[List[str], Dict[str, object]]:
    """Sample the Step 1 model directly, without Step 5 class-aware decoding."""

    sampling_start_time = time.perf_counter()
    sampling_cfg = resolved.base_config.get("sampling", {})
    lengths = _sample_raw_step1_lengths(
        tokenizer=tokenizer,
        num_samples=int(num_samples),
        sampling_cfg=sampling_cfg,
        source_lengths=source_lengths,
    )
    _, smiles = sampler.sample_batch(
        num_samples=int(num_samples),
        seq_length=int(tokenizer.max_length),
        batch_size=int(sampling_cfg.get("batch_size", 128)),
        show_progress=show_progress,
        lengths=lengths,
    )
    elapsed_seconds = float(time.perf_counter() - sampling_start_time)
    return smiles, {
        "num_samples": int(num_samples),
        "returned_num_samples": int(len(smiles)),
        "accepted_num_samples": int(len(smiles)),
        "remaining_num_samples": max(0, int(num_samples) - int(len(smiles))),
        "quota_satisfied": bool(len(smiles) >= int(num_samples)),
        "sampling_mode": "step1_unconditional",
        "class_aware_decode_constraints_enabled": False,
        "family_sampling_mode": "none",
        "family_sampling_scope": "none",
        "spans_per_sample": 0,
        "motif_count": 0,
        "motif_source": None,
        "backbone_template_enabled": False,
        "backbone_template_core_count": 0,
        "backbone_template_source": None,
        "backbone_template_min_gap_tokens": 0,
        "backbone_template_terminal_star_anchor": False,
        "backbone_template_max_core_token_length": 0,
        "backbone_template_layout": "disabled",
        "center_min_frac": 0.0,
        "center_max_frac": 0.0,
        "length_prior_count": int(len(source_lengths or [])),
        "length_prior_source": "step1_source_smiles" if source_lengths else None,
        "length_prior_min_tokens": None,
        "length_prior_max_tokens": None,
        "sampled_length_min": int(min(lengths)) if lengths else None,
        "sampled_length_max": int(max(lengths)) if lengths else None,
        "sampled_length_mean": float(np.mean(lengths)) if lengths else None,
        "class_token_bias_enabled": False,
        "class_token_bias_strength": 0.0,
        "enforce_class_match": False,
        "enforce_backbone_class_match": False,
        "enforce_star_ok_acceptance": False,
        "reject_duplicate_canonical_acceptance": False,
        "reject_sidechain_backbone_hybrids": False,
        "allowed_atomic_symbols": None,
        "forbidden_tokens": _resolve_base_config_forbidden_tokens(
            resolved.base_config,
            target_class=resolved.c_target,
        ),
        "forbidden_token_ids": sorted(getattr(sampler, "forbidden_token_ids", set())),
        "class_match_mode": "disabled",
        "class_match_sampling_attempts": 0,
        "class_match_acceptance_rate": 0.0,
        "class_match_oversampling_ratio": 1.0 if int(num_samples) > 0 else 0.0,
        "class_match_min_request_size": None,
        "class_match_max_request_size": None,
        "class_match_max_total_raw_samples": None,
        "allow_partial_quota_return": False,
        "partial_quota_min_fill_ratio": 0.0,
        "total_raw_samples_drawn": int(num_samples),
        "accepted_raw_target_class_samples": 0,
        "class_match_total_wall_time_seconds": elapsed_seconds,
        "class_match_total_raw_sampling_wall_time_seconds": elapsed_seconds,
        "class_match_total_filter_wall_time_seconds": 0.0,
        "class_match_wall_time_seconds_per_raw_draw": (
            elapsed_seconds / float(num_samples) if int(num_samples) > 0 else None
        ),
        "class_match_raw_sampling_wall_time_seconds_per_raw_draw": (
            elapsed_seconds / float(num_samples) if int(num_samples) > 0 else None
        ),
    }


def _sample_raw_smiles_with_prior(
    *,
    sampler: ConstrainedSampler,
    tokenizer: PSmilesTokenizer,
    prior: ResolvedClassSamplingPrior,
    resolved: ResolvedStep5Config,
    num_samples: int,
    show_progress: bool,
    sampling_state: Optional[Dict[str, object]] = None,
) -> Tuple[List[str], Dict[str, object]]:
    sampling_cfg = resolved.base_config.get("sampling", {})
    lengths = _sample_lengths(
        prior=prior,
        tokenizer=tokenizer,
        num_samples=num_samples,
        sampling_cfg=sampling_cfg,
    )

    if prior.family_sampling_mode == "backbone_template" and prior.backbone_template_enabled and prior.backbone_template_token_ids:
        backbone_gap_token_id = int(tokenizer.vocab.get("C", tokenizer.unk_token_id))
        if backbone_gap_token_id == int(tokenizer.unk_token_id):
            raise ValueError("Tokenizer vocabulary must define token 'C' for backbone-template sampling")
        multi_spans, multi_span_starts, lengths = _build_backbone_template_multi_spans(
            backbone_template_token_ids=prior.backbone_template_token_ids,
            lengths=lengths,
            center_min_frac=prior.center_min_frac,
            center_max_frac=prior.center_max_frac,
            seq_length=int(tokenizer.max_length),
            star_token_id=int(tokenizer.get_star_token_id()),
            backbone_gap_token_id=backbone_gap_token_id,
            min_gap_tokens=int(prior.backbone_template_min_gap_tokens),
            anchor_terminal_stars=bool(prior.backbone_template_terminal_star_anchor),
            sampling_state=(
                sampling_state
                if prior.cycle_backbone_template_cores_across_targets
                else None
            ),
        )
        _, smiles = sampler.sample_batch_with_multiple_fixed_spans(
            num_samples=num_samples,
            seq_length=int(tokenizer.max_length),
            span_token_ids=multi_spans,
            span_start_positions=multi_span_starts,
            batch_size=int(sampling_cfg.get("batch_size", 128)),
            show_progress=show_progress,
            lengths=lengths,
        )
    elif prior.family_sampling_mode == "sidechain_scaffold":
        raise NotImplementedError("sidechain_scaffold family sampling mode is not implemented yet.")
    elif prior.family_sampling_mode == "motif" and prior.motif_token_ids:
        multi_spans, multi_span_starts, lengths = _build_decode_constraint_multi_spans(
            motif_token_ids=prior.motif_token_ids,
            lengths=lengths,
            center_min_frac=prior.center_min_frac,
            center_max_frac=prior.center_max_frac,
            seq_length=int(tokenizer.max_length),
            spans_per_sample=prior.spans_per_sample,
        )
        if prior.spans_per_sample == 1:
            _, smiles = sampler.sample_batch_with_fixed_spans(
                num_samples=num_samples,
                seq_length=int(tokenizer.max_length),
                span_token_ids=[sample_spans[0] for sample_spans in multi_spans],
                span_start_positions=[sample_starts[0] for sample_starts in multi_span_starts],
                batch_size=int(sampling_cfg.get("batch_size", 128)),
                show_progress=show_progress,
                lengths=lengths,
            )
        else:
            _, smiles = sampler.sample_batch_with_multiple_fixed_spans(
                num_samples=num_samples,
                seq_length=int(tokenizer.max_length),
                span_token_ids=multi_spans,
                span_start_positions=multi_span_starts,
                batch_size=int(sampling_cfg.get("batch_size", 128)),
                show_progress=show_progress,
                lengths=lengths,
            )
    else:
        _, smiles = sampler.sample_batch(
            num_samples=num_samples,
            seq_length=int(tokenizer.max_length),
            batch_size=int(sampling_cfg.get("batch_size", 128)),
            show_progress=show_progress,
            lengths=lengths,
        )

    return smiles, {
        "length_prior_count": int(len(prior.length_prior_lengths)),
        "length_prior_source": prior.length_prior_source,
        "sampled_lengths": [int(length) for length in lengths],
    }


def _unexpected_atomic_symbols(
    smiles: str,
    *,
    allowed_atomic_symbols: Optional[List[str]],
) -> List[str]:
    if not allowed_atomic_symbols:
        return []
    try:
        from rdkit import Chem

        smiles_clean = str(smiles).replace("*", "[*]")
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return []
        allowed = set(str(symbol) for symbol in allowed_atomic_symbols)
        return sorted({atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() not in allowed})
    except Exception:
        return []


def _contains_forbidden_token(
    smiles: str,
    *,
    tokenizer: PSmilesTokenizer,
    forbidden_tokens: Optional[List[str]],
) -> bool:
    if not forbidden_tokens:
        return False
    forbidden = {str(token) for token in forbidden_tokens}
    try:
        return any(token in forbidden for token in tokenizer.tokenize(str(smiles)))
    except Exception:
        smiles_str = str(smiles)
        return any(token in smiles_str for token in forbidden)


def _accepted_target_class_indices(
    smiles_list: List[str],
    *,
    prior: ResolvedClassSamplingPrior,
    tokenizer: PSmilesTokenizer,
    classifier: Optional[PolymerClassifier],
    seen_canonical_smiles: set[str],
) -> tuple[List[int], Dict[str, int]]:
    use_backbone = bool(prior.target_class in BACKBONE_CLASS_MATCH_CLASSES and prior.enforce_backbone_class_match)
    accepted: List[int] = []
    stats = {
        "target_class_candidate_count": 0,
        "star_filter_rejected_count": 0,
        "sidechain_backbone_hybrid_rejected_count": 0,
        "unexpected_atoms_rejected_count": 0,
        "forbidden_token_rejected_count": 0,
        "duplicate_canonical_rejected_count": 0,
    }
    for idx, smiles in enumerate(smiles_list):
        smiles_str = str(smiles)
        valid_ok = bool(check_validity(smiles_str))
        try:
            if prior.enforce_class_match and classifier is not None:
                matches = classifier.classify_backbone(smiles_str) if use_backbone else classifier.classify(smiles_str)
            else:
                matches = {prior.target_class: True}
        except Exception:
            matches = {}
        class_ok = bool(matches.get(prior.target_class, False))
        if not class_ok:
            continue
        stats["target_class_candidate_count"] += 1
        if prior.enforce_star_ok_acceptance:
            if not (
                valid_ok
                and count_stars(smiles_str) == 2
                and has_terminal_connection_stars(smiles_str, expected_stars=2)
            ):
                stats["star_filter_rejected_count"] += 1
                continue
        if prior.reject_sidechain_backbone_hybrids and prior.target_class not in BACKBONE_CLASS_MATCH_CLASSES:
            backbone_matches = classifier.classify_backbone(smiles_str) if (valid_ok and classifier is not None) else {}
            hybrid_backbone_families = [
                str(name)
                for name, matched in backbone_matches.items()
                if matched and name in BACKBONE_CLASS_MATCH_CLASSES
            ]
            if hybrid_backbone_families:
                stats["sidechain_backbone_hybrid_rejected_count"] += 1
                continue
        if prior.allowed_atomic_symbols:
            unexpected_symbols = _unexpected_atomic_symbols(
                smiles_str,
                allowed_atomic_symbols=prior.allowed_atomic_symbols,
            )
            if unexpected_symbols:
                stats["unexpected_atoms_rejected_count"] += 1
                continue
        if _contains_forbidden_token(
            smiles_str,
            tokenizer=tokenizer,
            forbidden_tokens=prior.forbidden_tokens,
        ):
            stats["forbidden_token_rejected_count"] += 1
            continue
        canonical_smiles = canonicalize_smiles(smiles_str) if valid_ok else None
        if (
            prior.reject_duplicate_canonical_acceptance
            and canonical_smiles is not None
            and canonical_smiles in seen_canonical_smiles
        ):
            stats["duplicate_canonical_rejected_count"] += 1
            continue
        if canonical_smiles is not None:
            seen_canonical_smiles.add(canonical_smiles)
        if class_ok:
            accepted.append(int(idx))
    return accepted, stats


def _build_class_sampling_metadata(
    *,
    tokenizer: PSmilesTokenizer,
    prior: ResolvedClassSamplingPrior,
    num_samples: int,
    returned_smiles: List[str],
    accepted_smiles: List[str],
    total_drawn: int,
    accepted_raw_count: int,
    attempts: int,
    last_raw_meta: Dict[str, object],
    quota_satisfied: bool,
    attempt_log: Optional[List[Dict[str, object]]] = None,
    last_raw_smiles: Optional[List[str]] = None,
    last_raw_accepted_indices: Optional[List[int]] = None,
    filter_rejection_counts: Optional[Dict[str, int]] = None,
    total_wall_time_seconds: Optional[float] = None,
    total_raw_sampling_wall_time_seconds: Optional[float] = None,
    total_filter_wall_time_seconds: Optional[float] = None,
) -> Dict[str, object]:
    metadata = {
        "num_samples": int(num_samples),
        "returned_num_samples": int(len(returned_smiles)),
        "accepted_num_samples": int(len(accepted_smiles)),
        "remaining_num_samples": max(0, int(num_samples) - int(len(returned_smiles))),
        "quota_satisfied": bool(quota_satisfied),
        "total_raw_samples_drawn": int(total_drawn),
        "accepted_raw_target_class_samples": int(accepted_raw_count),
        "class_match_sampling_attempts": int(attempts),
        "class_match_acceptance_rate": (
            float(accepted_raw_count) / float(total_drawn) if total_drawn > 0 else 0.0
        ),
        "class_match_oversampling_ratio": (
            float(total_drawn) / float(num_samples) if int(num_samples) > 0 else 0.0
        ),
        "class_match_total_wall_time_seconds": (
            float(total_wall_time_seconds) if total_wall_time_seconds is not None else None
        ),
        "class_match_total_raw_sampling_wall_time_seconds": (
            float(total_raw_sampling_wall_time_seconds)
            if total_raw_sampling_wall_time_seconds is not None
            else None
        ),
        "class_match_total_filter_wall_time_seconds": (
            float(total_filter_wall_time_seconds)
            if total_filter_wall_time_seconds is not None
            else None
        ),
        "class_match_wall_time_seconds_per_raw_draw": (
            float(total_wall_time_seconds) / float(total_drawn)
            if total_wall_time_seconds is not None and total_drawn > 0
            else None
        ),
        "class_match_raw_sampling_wall_time_seconds_per_raw_draw": (
            float(total_raw_sampling_wall_time_seconds) / float(total_drawn)
            if total_raw_sampling_wall_time_seconds is not None and total_drawn > 0
            else None
        ),
        "class_match_min_request_size": int(prior.class_match_min_request_size),
        "class_match_max_request_size": (
            int(prior.class_match_max_request_size)
            if prior.class_match_max_request_size is not None
            else None
        ),
        "class_match_max_total_raw_samples": (
            int(prior.class_match_max_total_raw_samples)
            if prior.class_match_max_total_raw_samples is not None
            else None
        ),
        "allow_partial_quota_return": bool(prior.allow_partial_quota_return),
        "partial_quota_min_fill_ratio": float(prior.partial_quota_min_fill_ratio),
        "enforce_star_ok_acceptance": bool(prior.enforce_star_ok_acceptance),
        "reject_duplicate_canonical_acceptance": bool(prior.reject_duplicate_canonical_acceptance),
        "reject_sidechain_backbone_hybrids": bool(prior.reject_sidechain_backbone_hybrids),
        "allowed_atomic_symbols": list(prior.allowed_atomic_symbols) if prior.allowed_atomic_symbols else None,
        "forbidden_tokens": list(prior.forbidden_tokens) if prior.forbidden_tokens else None,
        "forbidden_token_ids": [int(token_id) for token_id in prior.forbidden_token_ids],
        "family_sampling_mode": str(prior.family_sampling_mode),
        "family_sampling_scope": str(prior.family_sampling_scope),
        "spans_per_sample": int(prior.spans_per_sample),
        "motif_count": int(len(prior.motifs)),
        "motif_source": prior.motif_source,
        "backbone_template_enabled": bool(prior.backbone_template_enabled),
        "backbone_template_core_count": int(len(prior.backbone_template_cores)),
        "backbone_template_source": prior.backbone_template_source,
        "backbone_template_min_gap_tokens": int(prior.backbone_template_min_gap_tokens),
        "backbone_template_terminal_star_anchor": bool(prior.backbone_template_terminal_star_anchor),
        "backbone_template_max_core_token_length": (
            max((len(tokenizer.tokenize(core)) for core in prior.backbone_template_cores), default=0)
        ),
        "backbone_template_layout": (
            "terminal_star_anchored_scaffold"
            if prior.backbone_template_enabled and prior.backbone_template_terminal_star_anchor
            else ("contiguous_scaffold" if prior.backbone_template_enabled else "disabled")
        ),
        "center_min_frac": float(prior.center_min_frac),
        "center_max_frac": float(prior.center_max_frac),
        "length_prior_count": int(last_raw_meta.get("length_prior_count", len(prior.length_prior_lengths))),
        "length_prior_source": last_raw_meta.get("length_prior_source", prior.length_prior_source),
        "length_prior_min_tokens": (
            int(prior.length_prior_min_tokens) if prior.length_prior_min_tokens is not None else None
        ),
        "length_prior_max_tokens": (
            int(prior.length_prior_max_tokens) if prior.length_prior_max_tokens is not None else None
        ),
        "class_token_bias_enabled": bool(prior.class_token_logit_bias is not None),
        "class_token_bias_strength": float(prior.class_token_bias_strength),
        "enforce_class_match": bool(prior.enforce_class_match),
        "enforce_backbone_class_match": bool(prior.enforce_backbone_class_match),
        "class_match_mode": (
            "strict_backbone"
            if prior.target_class in BACKBONE_CLASS_MATCH_CLASSES and prior.enforce_backbone_class_match
            else "loose"
        ),
    }
    if filter_rejection_counts is not None:
        metadata["filter_rejection_counts"] = {
            str(key): int(value)
            for key, value in dict(filter_rejection_counts).items()
        }
    if attempt_log is not None and not quota_satisfied:
        metadata["attempt_log"] = [dict(row) for row in attempt_log]
    if last_raw_smiles is not None and not quota_satisfied:
        metadata["last_raw_smiles"] = [str(smiles) for smiles in last_raw_smiles]
        metadata["last_raw_batch_size"] = int(len(last_raw_smiles))
    if last_raw_accepted_indices is not None and not quota_satisfied:
        metadata["last_raw_accepted_indices"] = [int(idx) for idx in last_raw_accepted_indices]
        metadata["last_raw_accepted_count"] = int(len(last_raw_accepted_indices))
    if not quota_satisfied:
        metadata["partial_accepted_smiles"] = [str(smiles) for smiles in accepted_smiles]
    return metadata


def _smoothed_class_match_acceptance_rate(
    *,
    total_drawn: int,
    accepted_raw_count: int,
) -> float:
    """Conservative smoothed acceptance estimate for adaptive class-match retries."""

    return float(accepted_raw_count + 0.5) / float(total_drawn + 1.0)


def _quota_shortfall_is_tolerable(
    *,
    prior: ResolvedClassSamplingPrior,
    accepted_count: int,
    requested_count: int,
) -> bool:
    """Allow near-complete batches to return partial results instead of hard-failing."""

    if not bool(prior.allow_partial_quota_return):
        return False
    if int(requested_count) <= 0:
        return True
    min_fill_ratio = min(max(float(prior.partial_quota_min_fill_ratio), 0.0), 1.0)
    min_accepted = int(np.ceil(float(requested_count) * min_fill_ratio))
    return int(accepted_count) >= max(1, min_accepted)


def _handle_class_sampling_quota_shortfall(
    *,
    tokenizer: PSmilesTokenizer,
    prior: ResolvedClassSamplingPrior,
    num_samples: int,
    accepted_smiles: List[str],
    total_drawn: int,
    accepted_raw_count: int,
    attempts: int,
    last_raw_meta: Dict[str, object],
    attempt_log: List[Dict[str, object]],
    last_raw_smiles: List[str],
    last_raw_accepted_indices: List[int],
    filter_rejection_counts: Dict[str, int],
    total_wall_time_seconds: float,
    total_raw_sampling_wall_time_seconds: float,
    total_filter_wall_time_seconds: float,
    exhaustion_reason: str,
):
    metadata = _build_class_sampling_metadata(
        tokenizer=tokenizer,
        prior=prior,
        num_samples=int(num_samples),
        returned_smiles=accepted_smiles,
        accepted_smiles=accepted_smiles,
        total_drawn=total_drawn,
        accepted_raw_count=accepted_raw_count,
        attempts=attempts,
        last_raw_meta=last_raw_meta,
        quota_satisfied=False,
        attempt_log=attempt_log,
        last_raw_smiles=last_raw_smiles,
        last_raw_accepted_indices=last_raw_accepted_indices,
        filter_rejection_counts=filter_rejection_counts,
        total_wall_time_seconds=total_wall_time_seconds,
        total_raw_sampling_wall_time_seconds=total_raw_sampling_wall_time_seconds,
        total_filter_wall_time_seconds=total_filter_wall_time_seconds,
    )
    metadata["quota_exhaustion_reason"] = str(exhaustion_reason)
    if _quota_shortfall_is_tolerable(
        prior=prior,
        accepted_count=len(accepted_smiles),
        requested_count=int(num_samples),
    ):
        metadata["quota_shortfall_tolerated"] = True
        metadata["quota_shortfall_count"] = max(0, int(num_samples) - int(len(accepted_smiles)))
        return list(accepted_smiles), metadata
    raise ClassConstrainedSamplingQuotaError(
        target_class=prior.target_class,
        accepted_smiles=accepted_smiles,
        requested_num_samples=int(num_samples),
        attempts_max=max(1, int(attempts)),
        metadata=metadata,
        last_raw_smiles=last_raw_smiles,
    )


def _resolve_class_match_request_size(
    *,
    prior: ResolvedClassSamplingPrior,
    remaining: int,
    attempts: int,
    total_drawn: int,
    accepted_raw_count: int,
) -> tuple[int, Dict[str, float | int | str]]:
    base_request_size = int(remaining)
    request_strategy = "remaining_only"
    attempts_left_including_current = max(
        1,
        int(prior.class_match_sampling_attempts_max) - int(attempts) + 1,
    )
    target_accepts_this_attempt = int(remaining)
    smoothed_acceptance_rate = float("nan")
    clipped_by_max_request_size = False

    if prior.enforce_class_match:
        base_request_size = max(
            int(remaining),
            int(np.ceil(float(remaining) * float(prior.class_match_oversample_factor))),
            int(prior.class_match_min_request_size),
        )
        request_strategy = "static_floor"
        target_accepts_this_attempt = max(
            1,
            int(np.ceil(float(remaining) / float(attempts_left_including_current))),
        )

    request_size = int(base_request_size)
    if prior.enforce_class_match and total_drawn > 0 and attempts > 1:
        smoothed_acceptance_rate = _smoothed_class_match_acceptance_rate(
            total_drawn=int(total_drawn),
            accepted_raw_count=int(accepted_raw_count),
        )
        expected_accepts_from_base = float(base_request_size) * float(smoothed_acceptance_rate)
        if expected_accepts_from_base + 1.0e-9 < float(target_accepts_this_attempt):
            adaptive_request_size = int(
                np.ceil(
                    (float(target_accepts_this_attempt) / max(float(smoothed_acceptance_rate), 1.0e-6))
                    * 1.15
                )
            )
            request_size = max(int(base_request_size), int(adaptive_request_size))
            request_strategy = "adaptive_acceptance_tail"

    if prior.enforce_class_match and prior.class_match_max_request_size is not None:
        capped_request_size = min(int(request_size), int(prior.class_match_max_request_size))
        clipped_by_max_request_size = int(capped_request_size) < int(request_size)
        request_size = int(capped_request_size)

    return int(request_size), {
        "base_request_size": int(base_request_size),
        "request_strategy": str(request_strategy),
        "attempts_left_including_current": int(attempts_left_including_current),
        "target_accepts_this_attempt": int(target_accepts_this_attempt),
        "smoothed_acceptance_rate": float(smoothed_acceptance_rate),
        "max_request_size": (
            int(prior.class_match_max_request_size)
            if prior.class_match_max_request_size is not None
            else None
        ),
        "max_total_raw_samples": (
            int(prior.class_match_max_total_raw_samples)
            if prior.class_match_max_total_raw_samples is not None
            else None
        ),
        "clipped_by_max_request_size": bool(clipped_by_max_request_size),
    }


def sample_with_class_prior(
    *,
    sampler: ConstrainedSampler,
    tokenizer: PSmilesTokenizer,
    prior: ResolvedClassSamplingPrior,
    resolved: ResolvedStep5Config,
    num_samples: int,
    show_progress: bool = True,
    seen_canonical_smiles: Optional[set[str]] = None,
    sampling_state: Optional[Dict[str, object]] = None,
) -> Tuple[List[str], Dict[str, object]]:
    """Sample polymers with Step 5 class priors using a sampler-compatible backend.

    This decode path is shared by frozen and conditional samplers, so family-aware
    template/scaffold changes here affect every final-generation Step 5 method.
    """
    sampler.set_forbidden_tokens(prior.forbidden_tokens)
    accepted_smiles: List[str] = []
    accepted_raw_count = 0
    total_drawn = 0
    attempts = 0
    need_classifier = bool(prior.enforce_class_match or prior.reject_sidechain_backbone_hybrids)
    classifier = PolymerClassifier(patterns=resolved.polymer_patterns) if need_classifier else None
    last_raw_meta: Dict[str, object] = {}
    last_raw_smiles: List[str] = []
    last_raw_accepted_indices: List[int] = []
    attempt_log: List[Dict[str, object]] = []
    seen_canonical_smiles = seen_canonical_smiles if seen_canonical_smiles is not None else set()
    filter_rejection_counts = {
        "target_class_candidate_count": 0,
        "star_filter_rejected_count": 0,
        "sidechain_backbone_hybrid_rejected_count": 0,
        "unexpected_atoms_rejected_count": 0,
        "forbidden_token_rejected_count": 0,
        "duplicate_canonical_rejected_count": 0,
    }
    sampling_start_time = time.perf_counter()
    total_raw_sampling_wall_time_seconds = 0.0
    total_filter_wall_time_seconds = 0.0

    while len(accepted_smiles) < int(num_samples):
        attempt_start_time = time.perf_counter()
        attempts += 1
        if attempts > int(prior.class_match_sampling_attempts_max):
            return _handle_class_sampling_quota_shortfall(
                tokenizer=tokenizer,
                prior=prior,
                num_samples=int(num_samples),
                accepted_smiles=accepted_smiles,
                total_drawn=total_drawn,
                accepted_raw_count=accepted_raw_count,
                attempts=int(prior.class_match_sampling_attempts_max),
                last_raw_meta=last_raw_meta,
                attempt_log=attempt_log,
                last_raw_smiles=last_raw_smiles,
                last_raw_accepted_indices=last_raw_accepted_indices,
                filter_rejection_counts=filter_rejection_counts,
                total_wall_time_seconds=(time.perf_counter() - sampling_start_time),
                total_raw_sampling_wall_time_seconds=total_raw_sampling_wall_time_seconds,
                total_filter_wall_time_seconds=total_filter_wall_time_seconds,
                exhaustion_reason="attempts_max",
            )
        remaining = int(num_samples) - len(accepted_smiles)
        remaining_raw_budget = None
        if prior.class_match_max_total_raw_samples is not None:
            remaining_raw_budget = max(
                0,
                int(prior.class_match_max_total_raw_samples) - int(total_drawn),
            )
            if remaining_raw_budget <= 0:
                return _handle_class_sampling_quota_shortfall(
                    tokenizer=tokenizer,
                    prior=prior,
                    num_samples=int(num_samples),
                    accepted_smiles=accepted_smiles,
                    total_drawn=total_drawn,
                    accepted_raw_count=accepted_raw_count,
                    attempts=max(0, int(attempts - 1)),
                    last_raw_meta=last_raw_meta,
                    attempt_log=attempt_log,
                    last_raw_smiles=last_raw_smiles,
                    last_raw_accepted_indices=last_raw_accepted_indices,
                    filter_rejection_counts=filter_rejection_counts,
                    total_wall_time_seconds=(time.perf_counter() - sampling_start_time),
                    total_raw_sampling_wall_time_seconds=total_raw_sampling_wall_time_seconds,
                    total_filter_wall_time_seconds=total_filter_wall_time_seconds,
                    exhaustion_reason="max_total_raw_samples",
                )
        request_size, request_debug = _resolve_class_match_request_size(
            prior=prior,
            remaining=int(remaining),
            attempts=int(attempts),
            total_drawn=int(total_drawn),
            accepted_raw_count=int(accepted_raw_count),
        )
        clipped_by_remaining_raw_budget = False
        if remaining_raw_budget is not None:
            bounded_request_size = min(int(request_size), int(remaining_raw_budget))
            clipped_by_remaining_raw_budget = int(bounded_request_size) < int(request_size)
            request_size = int(bounded_request_size)
        if request_size <= 0:
            return _handle_class_sampling_quota_shortfall(
                tokenizer=tokenizer,
                prior=prior,
                num_samples=int(num_samples),
                accepted_smiles=accepted_smiles,
                total_drawn=total_drawn,
                accepted_raw_count=accepted_raw_count,
                attempts=max(0, int(attempts - 1)),
                last_raw_meta=last_raw_meta,
                attempt_log=attempt_log,
                last_raw_smiles=last_raw_smiles,
                last_raw_accepted_indices=last_raw_accepted_indices,
                filter_rejection_counts=filter_rejection_counts,
                total_wall_time_seconds=(time.perf_counter() - sampling_start_time),
                total_raw_sampling_wall_time_seconds=total_raw_sampling_wall_time_seconds,
                total_filter_wall_time_seconds=total_filter_wall_time_seconds,
                exhaustion_reason="max_total_raw_samples",
            )
        retry_context_setter = getattr(sampler, "set_retry_sampling_context", None)
        if callable(retry_context_setter):
            retry_context_setter(
                request_strategy=str(request_debug["request_strategy"]),
                attempts=int(attempts),
                remaining=int(remaining),
                smoothed_acceptance_rate=float(request_debug["smoothed_acceptance_rate"]),
            )
        raw_sampling_start_time = time.perf_counter()
        raw_smiles, raw_meta = _sample_raw_smiles_with_prior(
            sampler=sampler,
            tokenizer=tokenizer,
            prior=prior,
            resolved=resolved,
            num_samples=int(request_size),
            show_progress=show_progress and attempts == 1,
            sampling_state=sampling_state,
        )
        raw_sampling_wall_time_seconds = time.perf_counter() - raw_sampling_start_time
        total_raw_sampling_wall_time_seconds += float(raw_sampling_wall_time_seconds)
        last_raw_meta = raw_meta
        last_raw_smiles = [str(smiles) for smiles in raw_smiles]
        total_drawn += int(len(raw_smiles))
        filter_start_time = time.perf_counter()
        if prior.enforce_class_match and classifier is not None:
            accepted_idx, attempt_filter_stats = _accepted_target_class_indices(
                raw_smiles,
                prior=prior,
                tokenizer=tokenizer,
                classifier=classifier,
                seen_canonical_smiles=seen_canonical_smiles,
            )
            last_raw_accepted_indices = [int(idx) for idx in accepted_idx]
            for key, value in attempt_filter_stats.items():
                filter_rejection_counts[key] = int(filter_rejection_counts.get(key, 0)) + int(value)
            accepted_raw_count += int(len(accepted_idx))
            accepted_batch = [raw_smiles[idx] for idx in accepted_idx]
            accepted_smiles.extend(accepted_batch)
        else:
            accepted_idx, attempt_filter_stats = _accepted_target_class_indices(
                raw_smiles,
                prior=prior,
                tokenizer=tokenizer,
                classifier=classifier,
                seen_canonical_smiles=seen_canonical_smiles,
            )
            last_raw_accepted_indices = [int(idx) for idx in accepted_idx]
            for key, value in attempt_filter_stats.items():
                filter_rejection_counts[key] = int(filter_rejection_counts.get(key, 0)) + int(value)
            accepted_raw_count += int(len(accepted_idx))
            accepted_batch = [raw_smiles[idx] for idx in accepted_idx]
            accepted_smiles.extend(accepted_batch)
        filter_wall_time_seconds = time.perf_counter() - filter_start_time
        total_filter_wall_time_seconds += float(filter_wall_time_seconds)
        attempt_log.append(
            {
                "attempt": int(attempts),
                "remaining_before_attempt": int(remaining),
                "base_request_size": int(request_debug["base_request_size"]),
                "request_size": int(request_size),
                "request_strategy": str(request_debug["request_strategy"]),
                "attempts_left_including_current": int(request_debug["attempts_left_including_current"]),
                "target_accepts_this_attempt": int(request_debug["target_accepts_this_attempt"]),
                "smoothed_acceptance_rate": float(request_debug["smoothed_acceptance_rate"]),
                "max_request_size": request_debug["max_request_size"],
                "max_total_raw_samples": request_debug["max_total_raw_samples"],
                "remaining_raw_budget_before_attempt": (
                    int(remaining_raw_budget) if remaining_raw_budget is not None else None
                ),
                "request_size_clipped_by_max_request_size": bool(
                    request_debug["clipped_by_max_request_size"]
                ),
                "request_size_clipped_by_raw_budget": bool(clipped_by_remaining_raw_budget),
                "raw_draw_count": int(len(raw_smiles)),
                "target_class_candidates_in_attempt": int(attempt_filter_stats.get("target_class_candidate_count", 0)),
                "accepted_in_attempt": int(len(accepted_batch)),
                "accepted_cumulative": int(len(accepted_smiles)),
                "acceptance_rate_in_attempt": (
                    float(len(accepted_batch)) / float(len(raw_smiles)) if raw_smiles else 0.0
                ),
                "star_filter_rejected_in_attempt": int(attempt_filter_stats.get("star_filter_rejected_count", 0)),
                "sidechain_backbone_hybrid_rejected_in_attempt": int(
                    attempt_filter_stats.get("sidechain_backbone_hybrid_rejected_count", 0)
                ),
                "unexpected_atoms_rejected_in_attempt": int(
                    attempt_filter_stats.get("unexpected_atoms_rejected_count", 0)
                ),
                "forbidden_token_rejected_in_attempt": int(
                    attempt_filter_stats.get("forbidden_token_rejected_count", 0)
                ),
                "duplicate_canonical_rejected_in_attempt": int(
                    attempt_filter_stats.get("duplicate_canonical_rejected_count", 0)
                ),
                "raw_sampling_wall_time_seconds": float(raw_sampling_wall_time_seconds),
                "filter_wall_time_seconds": float(filter_wall_time_seconds),
                "attempt_wall_time_seconds": float(time.perf_counter() - attempt_start_time),
                "raw_sampling_wall_time_seconds_per_raw_draw": (
                    float(raw_sampling_wall_time_seconds) / float(len(raw_smiles))
                    if raw_smiles
                    else None
                ),
            }
        )

    smiles = accepted_smiles[: int(num_samples)]

    metadata = _build_class_sampling_metadata(
        tokenizer=tokenizer,
        prior=prior,
        num_samples=int(num_samples),
        returned_smiles=smiles,
        accepted_smiles=accepted_smiles,
        total_drawn=total_drawn,
        accepted_raw_count=accepted_raw_count,
        attempts=attempts,
        last_raw_meta=last_raw_meta,
        quota_satisfied=True,
        attempt_log=attempt_log,
        last_raw_smiles=last_raw_smiles,
        last_raw_accepted_indices=last_raw_accepted_indices,
        filter_rejection_counts=filter_rejection_counts,
        total_wall_time_seconds=(time.perf_counter() - sampling_start_time),
        total_raw_sampling_wall_time_seconds=total_raw_sampling_wall_time_seconds,
        total_filter_wall_time_seconds=total_filter_wall_time_seconds,
    )
    return smiles, metadata


def sample_unconditional_with_class_prior(
    *,
    sampler: ConstrainedSampler,
    tokenizer: PSmilesTokenizer,
    prior: ResolvedClassSamplingPrior,
    resolved: ResolvedStep5Config,
    num_samples: int,
    show_progress: bool = True,
    seen_canonical_smiles: Optional[set[str]] = None,
    sampling_state: Optional[Dict[str, object]] = None,
) -> Tuple[List[str], Dict[str, object]]:
    """Backward-compatible alias for S0/S1 class-aware sampling."""

    return sample_with_class_prior(
        sampler=sampler,
        tokenizer=tokenizer,
        prior=prior,
        resolved=resolved,
        num_samples=num_samples,
        show_progress=show_progress,
        seen_canonical_smiles=seen_canonical_smiles,
        sampling_state=sampling_state,
    )
