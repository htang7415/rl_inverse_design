"""Guided sampler for Step 5 S1."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from src.sampling.sampler import ConstrainedSampler
from src.step5.conditional_sampling import ConditionalConstrainedSampler
from src.step5.evaluation import Step5Evaluator
from src.step5.rewards import score_guidance_batch


def _resolve_guided_sample_mask(
    *,
    is_masked: torch.Tensor,
    initial_mask_counts: torch.Tensor,
    step_progress: float,
    guidance_start_frac: float,
    best_of_k: int,
) -> torch.Tensor:
    """Enable guidance once a sample is sufficiently complete.

    Short class-constrained sequences can finish unmasking before a late
    diffusion-step fraction such as 0.5 or 0.75 is ever reached. Use the
    per-sample editable-token completion ratio as an additional progress signal
    so guidance still activates on short, heavily constrained generations.
    """

    masked_counts = is_masked.sum(dim=1)
    if int(best_of_k) < 2:
        return torch.zeros_like(masked_counts, dtype=torch.bool)

    editable_counts = initial_mask_counts.to(dtype=torch.float32).clamp(min=1.0)
    completion_progress = 1.0 - (masked_counts.to(dtype=torch.float32) / editable_counts)
    guidance_progress = torch.maximum(
        completion_progress,
        torch.full_like(completion_progress, float(step_progress)),
    )
    return (masked_counts > 0) & (guidance_progress >= float(guidance_start_frac))


class GuidedSampler(ConstrainedSampler):
    """Frozen-model guided sampler with late oracle guidance."""

    def __init__(
        self,
        *,
        evaluator: Step5Evaluator,
        target_row: Dict[str, object],
        best_of_k: int,
        guidance_start_frac: float,
        sol_log_prob_floor: float,
        w_sol: float,
        w_chi: float,
        w_sa: float = 0.0,
        w_sa_continuous: float = 0.0,
        invalid_reward_penalty: float = -10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.target_row = target_row
        self.best_of_k = int(best_of_k)
        self.guidance_start_frac = float(guidance_start_frac)
        self.sol_log_prob_floor = float(sol_log_prob_floor)
        self.w_sol = float(w_sol)
        self.w_chi = float(w_chi)
        self.w_sa = float(w_sa)
        self.w_sa_continuous = float(w_sa_continuous)
        self.invalid_reward_penalty = float(invalid_reward_penalty)
        self.training_oracle_calls_soluble = 0
        self.training_oracle_calls_chi = 0
        self.guidance_no_valid_fallback_steps = 0

    def reset_guidance_stats(self) -> None:
        self.training_oracle_calls_soluble = 0
        self.training_oracle_calls_chi = 0
        self.guidance_no_valid_fallback_steps = 0

    def get_guidance_stats(self) -> Dict[str, int]:
        return {
            "training_soluble_oracle_calls": int(self.training_oracle_calls_soluble),
            "training_chi_oracle_calls": int(self.training_oracle_calls_chi),
            "guidance_no_valid_fallback_steps": int(self.guidance_no_valid_fallback_steps),
        }

    def _apply_within_step_constraint_updates(
        self,
        logits_row: torch.Tensor,
        probs_row: torch.Tensor,
        ids_row: torch.Tensor,
        fixed_mask_row: torch.Tensor,
        sampled_token: int,
        pos: int,
    ) -> torch.Tensor:
        if not self.use_constraints:
            return probs_row

        if sampled_token == self.star_id:
            non_mask = ids_row != self.mask_id
            current_stars = ((ids_row == self.star_id) & non_mask).sum().item()
            if current_stars >= self.target_stars:
                remaining_mask = (ids_row == self.mask_id) & (~fixed_mask_row)
                logits_row[remaining_mask, self.star_id] = float("-inf")
                probs_row = self._logits_to_probs(logits_row)
        elif sampled_token in self.bond_ids:
            next_pos = pos + 1
            if next_pos < len(ids_row) and ids_row[next_pos] == self.mask_id:
                for bond_id in self.bond_ids:
                    logits_row[next_pos, bond_id] = float("-inf")
                probs_row = self._logits_to_probs(logits_row)
        elif sampled_token == self.open_paren_id:
            next_pos = pos + 1
            if next_pos < len(ids_row) and ids_row[next_pos] == self.mask_id:
                logits_row[next_pos, self.close_paren_id] = float("-inf")
                probs_row = self._logits_to_probs(logits_row)
        return probs_row

    def _sample_from_ids(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, List[str]]:
        self.diffusion_model.eval()
        backbone = self.diffusion_model.backbone
        batch_size = ids.shape[0]
        final_logits = None
        steps = range(self.num_steps, 0, -1)
        initial_mask_counts = ((ids == self.mask_id) & (~fixed_mask)).sum(dim=1).to(dtype=torch.float32)

        if show_progress:
            from tqdm import tqdm

            steps = tqdm(steps, desc="Guided sampling")

        for t in steps:
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            step_progress = self._step_progress_frac(int(t))
            with torch.no_grad():
                logits = backbone(ids, timesteps, attention_mask)

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
            probs = self._logits_to_probs(logits)

            is_masked = (ids == self.mask_id) & (~fixed_mask)
            unmask_prob = 1.0 / t
            guided_mask = _resolve_guided_sample_mask(
                is_masked=is_masked,
                initial_mask_counts=initial_mask_counts,
                step_progress=step_progress,
                guidance_start_frac=self.guidance_start_frac,
                best_of_k=self.best_of_k,
            )

            candidate_blocks: List[torch.Tensor] = []
            attention_blocks: List[torch.Tensor] = []
            unmask_positions_by_sample: Dict[int, torch.Tensor] = {}

            for i in range(batch_size):
                masked_pos = torch.where(is_masked[i])[0]
                if len(masked_pos) == 0:
                    continue
                num_unmask = max(1, int(len(masked_pos) * unmask_prob))
                unmask_indices = torch.randperm(len(masked_pos), device=self.device)[:num_unmask]
                unmask_positions = masked_pos[unmask_indices]
                if not bool(guided_mask[i].item()):
                    for pos in unmask_positions:
                        sampled = torch.multinomial(probs[i, pos], 1)
                        ids[i, pos] = sampled
                        probs[i] = self._apply_within_step_constraint_updates(
                            logits[i],
                            probs[i],
                            ids[i],
                            fixed_mask[i],
                            int(sampled.item()),
                            int(pos.item()),
                        )
                    continue
                unmask_positions_by_sample[i] = unmask_positions

                candidate_ids = ids[i].unsqueeze(0).repeat(self.best_of_k, 1)
                for pos in masked_pos:
                    sampled = torch.multinomial(probs[i, pos], self.best_of_k, replacement=True)
                    candidate_ids[:, int(pos.item())] = sampled
                candidate_blocks.append(candidate_ids)
                attention_blocks.append(attention_mask[i].unsqueeze(0).repeat(self.best_of_k, 1))

            if candidate_blocks:
                provisional_ids = torch.cat(candidate_blocks, dim=0)
                provisional_attention = torch.cat(attention_blocks, dim=0)
                scored = score_guidance_batch(
                    provisional_ids,
                    provisional_attention,
                    target_row=self.target_row,
                    evaluator=self.evaluator,
                    tokenizer=self.tokenizer,
                    sol_log_prob_floor=self.sol_log_prob_floor,
                    w_sol=self.w_sol,
                    w_chi=self.w_chi,
                    invalid_reward_penalty=self.invalid_reward_penalty,
                    w_sa=self.w_sa,
                    w_sa_continuous=self.w_sa_continuous,
                )
                self.training_oracle_calls_soluble += int(scored["oracle_calls_soluble"])
                self.training_oracle_calls_chi += int(scored["oracle_calls_chi"])

                rewards = scored["reward"]
                valid_mask = scored["valid_mask"].bool()
                reward_offset = 0
                for i, unmask_positions in unmask_positions_by_sample.items():
                    sample_rewards = rewards[reward_offset : reward_offset + self.best_of_k]
                    sample_valid = valid_mask[reward_offset : reward_offset + self.best_of_k]
                    if sample_valid.any():
                        valid_rewards = sample_rewards[sample_valid]
                        valid_indices = torch.nonzero(sample_valid, as_tuple=False).flatten()
                        best_idx = int(valid_indices[int(torch.argmax(valid_rewards).item())].item())
                        chosen_ids = provisional_ids[reward_offset + best_idx]
                    else:
                        self.guidance_no_valid_fallback_steps += 1
                        chosen_ids = ids[i].clone()
                        for pos in unmask_positions:
                            sampled = torch.multinomial(probs[i, pos], 1)
                            chosen_token = int(sampled.item())
                            chosen_ids[int(pos.item())] = chosen_token
                            probs[i] = self._apply_within_step_constraint_updates(
                                logits[i],
                                probs[i],
                                chosen_ids,
                                fixed_mask[i],
                                chosen_token,
                                int(pos.item()),
                            )
                    reward_offset += self.best_of_k
                    for pos in unmask_positions:
                        chosen_token = int(chosen_ids[int(pos.item())].item())
                        ids[i, int(pos.item())] = chosen_token
                        probs[i] = self._apply_within_step_constraint_updates(
                            logits[i],
                            probs[i],
                            ids[i],
                            fixed_mask[i],
                            chosen_token,
                            int(pos.item()),
                        )

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


class GuidedConditionalSampler(ConditionalConstrainedSampler):
    """Conditional guided sampler for Step 5 S3."""

    def __init__(
        self,
        *,
        evaluator: Step5Evaluator,
        target_row: Dict[str, object],
        best_of_k: int,
        guidance_start_frac: float,
        sol_log_prob_floor: float,
        w_sol: float,
        w_chi: float,
        w_sa: float = 0.0,
        w_sa_continuous: float = 0.0,
        invalid_reward_penalty: float = -10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.target_row = target_row
        self.best_of_k = int(best_of_k)
        self.guidance_start_frac = float(guidance_start_frac)
        self.sol_log_prob_floor = float(sol_log_prob_floor)
        self.w_sol = float(w_sol)
        self.w_chi = float(w_chi)
        self.w_sa = float(w_sa)
        self.w_sa_continuous = float(w_sa_continuous)
        self.invalid_reward_penalty = float(invalid_reward_penalty)
        self.training_oracle_calls_soluble = 0
        self.training_oracle_calls_chi = 0
        self.guidance_no_valid_fallback_steps = 0

    def reset_guidance_stats(self) -> None:
        self.training_oracle_calls_soluble = 0
        self.training_oracle_calls_chi = 0
        self.guidance_no_valid_fallback_steps = 0

    def get_guidance_stats(self) -> Dict[str, int]:
        return {
            "training_soluble_oracle_calls": int(self.training_oracle_calls_soluble),
            "training_chi_oracle_calls": int(self.training_oracle_calls_chi),
            "guidance_no_valid_fallback_steps": int(self.guidance_no_valid_fallback_steps),
        }

    def _apply_within_step_constraint_updates(
        self,
        logits_row: torch.Tensor,
        probs_row: torch.Tensor,
        ids_row: torch.Tensor,
        fixed_mask_row: torch.Tensor,
        sampled_token: int,
        pos: int,
    ) -> torch.Tensor:
        if not self.use_constraints:
            return probs_row

        if sampled_token == self.star_id:
            non_mask = ids_row != self.mask_id
            current_stars = ((ids_row == self.star_id) & non_mask).sum().item()
            if current_stars >= self.target_stars:
                remaining_mask = (ids_row == self.mask_id) & (~fixed_mask_row)
                logits_row[remaining_mask, self.star_id] = float("-inf")
                probs_row = self._logits_to_probs(logits_row)
        elif sampled_token in self.bond_ids:
            next_pos = pos + 1
            if next_pos < len(ids_row) and ids_row[next_pos] == self.mask_id:
                for bond_id in self.bond_ids:
                    logits_row[next_pos, bond_id] = float("-inf")
                probs_row = self._logits_to_probs(logits_row)
        elif sampled_token == self.open_paren_id:
            next_pos = pos + 1
            if next_pos < len(ids_row) and ids_row[next_pos] == self.mask_id:
                logits_row[next_pos, self.close_paren_id] = float("-inf")
                probs_row = self._logits_to_probs(logits_row)
        return probs_row

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
        initial_mask_counts = ((ids == self.mask_id) & (~fixed_mask)).sum(dim=1).to(dtype=torch.float32)

        if show_progress:
            from tqdm import tqdm

            steps = tqdm(steps, desc="Conditional guided sampling")

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
            guided_mask = _resolve_guided_sample_mask(
                is_masked=is_masked,
                initial_mask_counts=initial_mask_counts,
                step_progress=step_progress,
                guidance_start_frac=self.guidance_start_frac,
                best_of_k=self.best_of_k,
            )

            candidate_blocks: List[torch.Tensor] = []
            attention_blocks: List[torch.Tensor] = []
            unmask_positions_by_sample: Dict[int, torch.Tensor] = {}

            for i in range(batch_size):
                masked_pos = torch.where(is_masked[i])[0]
                if len(masked_pos) == 0:
                    continue
                num_unmask = max(1, int(len(masked_pos) * unmask_prob))
                unmask_indices = torch.randperm(len(masked_pos), device=self.device)[:num_unmask]
                unmask_positions = masked_pos[unmask_indices]
                if not bool(guided_mask[i].item()):
                    for pos in unmask_positions:
                        sampled = torch.multinomial(probs[i, pos], 1)
                        ids[i, pos] = sampled
                        probs[i] = self._apply_within_step_constraint_updates(
                            logits[i],
                            probs[i],
                            ids[i],
                            fixed_mask[i],
                            int(sampled.item()),
                            int(pos.item()),
                        )
                    continue
                unmask_positions_by_sample[i] = unmask_positions

                candidate_ids = ids[i].unsqueeze(0).repeat(self.best_of_k, 1)
                for pos in masked_pos:
                    sampled = torch.multinomial(probs[i, pos], self.best_of_k, replacement=True)
                    candidate_ids[:, int(pos.item())] = sampled
                candidate_blocks.append(candidate_ids)
                attention_blocks.append(attention_mask[i].unsqueeze(0).repeat(self.best_of_k, 1))

            if candidate_blocks:
                provisional_ids = torch.cat(candidate_blocks, dim=0)
                provisional_attention = torch.cat(attention_blocks, dim=0)
                scored = score_guidance_batch(
                    provisional_ids,
                    provisional_attention,
                    target_row=self.target_row,
                    evaluator=self.evaluator,
                    tokenizer=self.tokenizer,
                    sol_log_prob_floor=self.sol_log_prob_floor,
                    w_sol=self.w_sol,
                    w_chi=self.w_chi,
                    invalid_reward_penalty=self.invalid_reward_penalty,
                    w_sa=self.w_sa,
                    w_sa_continuous=self.w_sa_continuous,
                )
                self.training_oracle_calls_soluble += int(scored["oracle_calls_soluble"])
                self.training_oracle_calls_chi += int(scored["oracle_calls_chi"])

                rewards = scored["reward"]
                valid_mask = scored["valid_mask"].bool()
                reward_offset = 0
                for i, unmask_positions in unmask_positions_by_sample.items():
                    sample_rewards = rewards[reward_offset : reward_offset + self.best_of_k]
                    sample_valid = valid_mask[reward_offset : reward_offset + self.best_of_k]
                    if sample_valid.any():
                        valid_rewards = sample_rewards[sample_valid]
                        valid_indices = torch.nonzero(sample_valid, as_tuple=False).flatten()
                        best_idx = int(valid_indices[int(torch.argmax(valid_rewards).item())].item())
                        chosen_ids = provisional_ids[reward_offset + best_idx]
                    else:
                        self.guidance_no_valid_fallback_steps += 1
                        chosen_ids = ids[i].clone()
                        for pos in unmask_positions:
                            sampled = torch.multinomial(probs[i, pos], 1)
                            chosen_token = int(sampled.item())
                            chosen_ids[int(pos.item())] = chosen_token
                            probs[i] = self._apply_within_step_constraint_updates(
                                logits[i],
                                probs[i],
                                chosen_ids,
                                fixed_mask[i],
                                chosen_token,
                                int(pos.item()),
                            )
                    reward_offset += self.best_of_k
                    for pos in unmask_positions:
                        chosen_token = int(chosen_ids[int(pos.item())].item())
                        ids[i, int(pos.item())] = chosen_token
                        probs[i] = self._apply_within_step_constraint_updates(
                            logits[i],
                            probs[i],
                            ids[i],
                            fixed_mask[i],
                            chosen_token,
                            int(pos.item()),
                        )

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
