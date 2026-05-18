"""Conditional sampling helpers for Step 5 S2/S3/S4."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from src.sampling.sampler import ConstrainedSampler
from src.step5.conditional_diffusion import ConditionalDiscreteMaskingDiffusion
from src.step5.config import ResolvedStep5Config, resolve_step5_sampling_num_steps
from src.step5.frozen_sampling import ResolvedClassSamplingPrior, sample_with_class_prior
from src.data.tokenizer import PSmilesTokenizer


def _resolve_step5_class_override(
    step5_cfg: Dict[str, object],
    *,
    key: str,
    target_class: str,
    default_value: float,
) -> float:
    overrides = step5_cfg.get(f"{key}_overrides", {})
    if isinstance(overrides, dict):
        target_key = str(target_class).strip().lower()
        for raw_key, raw_value in overrides.items():
            if str(raw_key).strip().lower() != target_key:
                continue
            return float(raw_value)
    return float(step5_cfg.get(key, default_value))


class ConditionalConstrainedSampler(ConstrainedSampler):
    """Constrained sampler that uses Step 5 conditional CFG logits."""

    diffusion_model: ConditionalDiscreteMaskingDiffusion

    def __init__(
        self,
        *,
        condition_bundle: torch.Tensor,
        cfg_scale: float,
        cfg_scale_early_frac: float = 1.0,
        cfg_scale_ramp_start_frac: float = 0.0,
        cfg_scale_retry_multiplier: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if condition_bundle.dim() == 1:
            condition_bundle = condition_bundle.unsqueeze(0)
        if condition_bundle.dim() != 2 or int(condition_bundle.shape[-1]) <= 0:
            raise ValueError(
                f"condition_bundle must have shape [batch, dim] or [dim], got {tuple(condition_bundle.shape)}"
            )
        self.condition_dim = int(condition_bundle.shape[-1])
        self.condition_bundle = condition_bundle.detach().to(
            device=device_from_kwargs(kwargs),
            dtype=torch.float32,
        )
        self.cfg_scale = float(cfg_scale)
        self.cfg_scale_early_frac = float(cfg_scale_early_frac)
        self.cfg_scale_ramp_start_frac = float(cfg_scale_ramp_start_frac)
        self.cfg_scale_retry_multiplier = float(cfg_scale_retry_multiplier)
        self.cfg_scale_current_attempt_multiplier = 1.0
        if not (0.0 <= self.cfg_scale_early_frac <= 1.0):
            raise ValueError(
                f"cfg_scale_early_frac must be in [0, 1], got {self.cfg_scale_early_frac}"
            )
        if not (0.0 <= self.cfg_scale_ramp_start_frac <= 1.0):
            raise ValueError(
                f"cfg_scale_ramp_start_frac must be in [0, 1], got {self.cfg_scale_ramp_start_frac}"
            )
        if self.cfg_scale_retry_multiplier <= 0.0:
            raise ValueError(
                f"cfg_scale_retry_multiplier must be > 0, got {self.cfg_scale_retry_multiplier}"
            )

    def set_retry_sampling_context(
        self,
        *,
        request_strategy: str,
        attempts: int,
        remaining: int,
        smoothed_acceptance_rate: float,
    ) -> None:
        del attempts, remaining, smoothed_acceptance_rate
        self.cfg_scale_current_attempt_multiplier = (
            float(self.cfg_scale_retry_multiplier)
            if str(request_strategy).strip().lower() == "adaptive_acceptance_tail"
            else 1.0
        )

    def _cfg_scale_progress_multiplier(self, step_progress: float) -> float:
        early = float(self.cfg_scale_early_frac)
        start = float(self.cfg_scale_ramp_start_frac)
        progress = float(step_progress)
        if early >= 1.0:
            return 1.0
        if progress <= start:
            return early
        ramp = (progress - start) / max(1.0e-8, 1.0 - start)
        return float(early + ((1.0 - early) * max(0.0, min(1.0, ramp))))

    def _effective_cfg_scale(self, step_progress: float) -> float:
        return (
            float(self.cfg_scale)
            * float(self.cfg_scale_current_attempt_multiplier)
            * float(self._cfg_scale_progress_multiplier(step_progress))
        )

    def _condition_for_batch(self, batch_size: int) -> torch.Tensor:
        if self.condition_bundle.shape[0] == batch_size:
            return self.condition_bundle
        if self.condition_bundle.shape[0] == 1:
            return self.condition_bundle.expand(batch_size, -1)
        raise ValueError(
            f"Conditional sampler condition bundle has batch dimension {self.condition_bundle.shape[0]}, "
            f"but current batch_size is {batch_size}"
        )

    def _sample_from_ids(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, List[str]]:
        self.diffusion_model.eval()
        batch_size = ids.shape[0]
        cond = self._condition_for_batch(batch_size)

        final_logits = None
        steps = range(self.num_steps, 0, -1)
        if show_progress:
            from tqdm import tqdm

            steps = tqdm(steps, desc="Conditional sampling")

        for t in steps:
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            step_progress = self._step_progress_frac(int(t))

            with torch.no_grad():
                logits = self.diffusion_model.classifier_free_guidance_logits(
                    ids,
                    timesteps,
                    attention_mask,
                    condition_bundle=cond,
                    cfg_scale=self._effective_cfg_scale(step_progress),
                )

            logits = logits / self.temperature
            if self.use_constraints:
                logits = self._apply_star_constraint(logits, ids, max_stars=self.target_stars)
                logits = self._apply_exact_star_budget_constraint(logits, ids, target_stars=self.target_stars)
                logits = self._apply_position_aware_paren_constraints(logits, ids)
                logits = self._apply_ring_constraints(logits, ids)
                logits = self._apply_bond_placement_constraints(logits, ids)
            logits = self._apply_class_token_bias(logits, fixed_mask=fixed_mask, step_progress=step_progress)
            logits = self._apply_sampling_filters(logits)
            logits = self._apply_special_token_constraints(logits, ids)
            logits = self._ensure_valid_logits(logits)

            probs = self._logits_to_probs(logits)
            is_masked = (ids == self.mask_id) & (~fixed_mask)
            unmask_prob = 1.0 / t

            for i in range(batch_size):
                masked_pos = torch.where(is_masked[i])[0]
                if len(masked_pos) == 0:
                    continue

                num_unmask = max(1, int(len(masked_pos) * unmask_prob))
                unmask_indices = torch.randperm(len(masked_pos), device=self.device)[:num_unmask]
                unmask_positions = masked_pos[unmask_indices]

                for pos in unmask_positions:
                    sampled = torch.multinomial(probs[i, pos], 1)
                    ids[i, pos] = sampled
                    sampled_token = int(sampled.item())

                    if self.use_constraints:
                        if sampled_token == self.star_id:
                            non_mask = ids[i] != self.mask_id
                            current_stars = ((ids[i] == self.star_id) & non_mask).sum().item()
                            if current_stars >= self.target_stars:
                                remaining_mask = (ids[i] == self.mask_id) & (~fixed_mask[i])
                                logits[i, remaining_mask, self.star_id] = float("-inf")
                                probs[i] = self._logits_to_probs(logits[i])
                        elif sampled_token in self.bond_ids:
                            next_pos = pos + 1
                            if next_pos < len(ids[i]) and ids[i, next_pos] == self.mask_id:
                                for bond_id in self.bond_ids:
                                    logits[i, next_pos, bond_id] = float("-inf")
                                probs[i] = self._logits_to_probs(logits[i])
                        elif sampled_token == self.open_paren_id:
                            next_pos = pos + 1
                            if next_pos < len(ids[i]) and ids[i, next_pos] == self.mask_id:
                                logits[i, next_pos, self.close_paren_id] = float("-inf")
                                probs[i] = self._logits_to_probs(logits[i])

            if t == 1:
                final_logits = logits

        if self.use_constraints:
            ids = self._fix_ring_closures(ids, final_logits, fixed_mask=fixed_mask)
            ids = self._fix_bond_placement(ids, final_logits, fixed_mask=fixed_mask)
            ids = self._fix_paren_balance(ids, final_logits, fixed_mask=fixed_mask)
            ids = self._fix_star_count(ids, final_logits, target_stars=self.target_stars, fixed_mask=fixed_mask)
            ids = self._fix_ring_closures(ids, final_logits, fixed_mask=fixed_mask)
            ids = self._fix_paren_balance(ids, final_logits, fixed_mask=fixed_mask)

        smiles_list = self.tokenizer.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)
        return ids, smiles_list


def device_from_kwargs(kwargs: Dict[str, object]) -> str:
    device = kwargs.get("device", "cpu")
    return str(device)


def create_conditional_sampler(
    *,
    diffusion_model: ConditionalDiscreteMaskingDiffusion,
    tokenizer: PSmilesTokenizer,
    resolved: ResolvedStep5Config,
    prior: ResolvedClassSamplingPrior,
    condition_bundle: torch.Tensor,
    cfg_scale: float,
    device: str,
    num_steps: int | None = None,
) -> ConditionalConstrainedSampler:
    sampling_cfg = resolved.base_config.get("sampling", {})
    step5_cfg = dict(resolved.step5)
    target_class = str(getattr(resolved, "c_target", "") or step5_cfg.get("c_target", "")).strip().lower()
    sampler = ConditionalConstrainedSampler(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        num_steps=int(
            resolve_step5_sampling_num_steps(resolved.step5, resolved.base_config)
            if num_steps is None
            else num_steps
        ),
        temperature=float(resolved.step5["sampling_temperature"]),
        top_k=sampling_cfg.get("top_k"),
        top_p=sampling_cfg.get("top_p"),
        target_stars=int(sampling_cfg.get("target_stars", 2)),
        use_constraints=bool(sampling_cfg.get("use_constraints", True)),
        device=device,
        condition_bundle=condition_bundle,
        cfg_scale=float(cfg_scale),
        cfg_scale_early_frac=_resolve_step5_class_override(
            step5_cfg,
            key="conditional_cfg_scale_early_frac",
            target_class=target_class,
            default_value=1.0,
        ),
        cfg_scale_ramp_start_frac=_resolve_step5_class_override(
            step5_cfg,
            key="conditional_cfg_scale_ramp_start_frac",
            target_class=target_class,
            default_value=0.0,
        ),
        cfg_scale_retry_multiplier=_resolve_step5_class_override(
            step5_cfg,
            key="conditional_cfg_scale_retry_multiplier",
            target_class=target_class,
            default_value=1.0,
        ),
    )
    sampler.set_class_token_bias_start_frac(float(resolved.step5.get("class_token_bias_start_frac", 0.0)))
    if prior.class_token_logit_bias is not None:
        sampler.set_class_token_logit_bias(prior.class_token_logit_bias)
    sampler.set_forbidden_tokens(prior.forbidden_tokens)
    return sampler


def sample_conditional_with_class_prior(
    *,
    sampler: ConditionalConstrainedSampler,
    tokenizer: PSmilesTokenizer,
    prior: ResolvedClassSamplingPrior,
    resolved: ResolvedStep5Config,
    num_samples: int,
    show_progress: bool = True,
    seen_canonical_smiles: set[str] | None = None,
    sampling_state: Dict[str, object] | None = None,
) -> Tuple[List[str], Dict[str, object]]:
    """Conditional Step 5 sampling through the shared family-aware decode path."""
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
