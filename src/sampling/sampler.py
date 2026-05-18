"""Constrained sampler for polymer generation."""

import torch
import torch.nn.functional as F
from typing import Optional, List, Sequence, Tuple
import numpy as np
from tqdm import tqdm


class ConstrainedSampler:
    """Sampler for polymer generation with optional constraints.

    When use_constraints is True, enforces SMILES syntax constraints and
    exactly two '*' tokens. Special tokens are always forbidden at MASK positions.
    """

    def __init__(
        self,
        diffusion_model,
        tokenizer,
        num_steps: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        target_stars: int = 2,
        use_constraints: bool = True,
        device: str = 'cuda'
    ):
        """Initialize sampler.

        Args:
            diffusion_model: Trained discrete masking diffusion model.
            tokenizer: Tokenizer instance.
            num_steps: Number of diffusion steps.
            temperature: Sampling temperature.
            top_k: Keep only top-k tokens per position during sampling (None disables).
            top_p: Keep minimal token set with cumulative probability <= top-p (None disables).
            target_stars: Target number of '*' tokens for p-SMILES outputs.
            use_constraints: Whether to apply chemistry constraints during sampling.
            device: Device for computation.
        """
        self.diffusion_model = diffusion_model
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.temperature = temperature
        self.top_k = int(top_k) if top_k is not None else None
        self.top_p = float(top_p) if top_p is not None else None
        self.target_stars = int(target_stars)
        self.use_constraints = use_constraints
        self.device = device
        self.class_token_logit_bias: Optional[torch.Tensor] = None
        self.class_token_bias_start_frac: float = 0.0
        self.forbidden_token_ids: set[int] = set()

        # Get special token IDs
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.star_id = tokenizer.get_star_token_id()

        # Token categories for syntax constraints
        self.open_paren_id = tokenizer.vocab.get('(', -1)
        self.close_paren_id = tokenizer.vocab.get(')', -1)
        self.bond_ids = {tokenizer.vocab.get(b, -1) for b in ['-', '=', '#', '/', '\\']} - {-1}
        self.ring_digit_ids = {tokenizer.vocab.get(str(d), -1) for d in range(10)} - {-1}
        self.ring_percent_ids = {v for k, v in tokenizer.vocab.items() if k.startswith('%')}

        # Build set of bracket tokens (tokens that start with '[' and end with ']')
        self.bracket_token_ids = {
            v for k, v in tokenizer.vocab.items()
            if k.startswith('[') and k.endswith(']') and k not in tokenizer.SPECIAL_TOKENS
        }

        # Atom tokens (for determining what can follow bonds)
        self.atom_ids = {
            tokenizer.vocab.get(a, -1) for a in
            ['B', 'C', 'N', 'O', 'F', 'P', 'S', 'I', 'H', 'c', 'n', 'o', 's', 'p',
             'Cl', 'Br', 'Si', 'Na', 'Li', 'Ca', 'Mg', 'Al', '*']
        } - {-1}
        self.atom_ids.update(self.bracket_token_ids)  # Bracket tokens are also atoms
        self.fallback_non_star_id = self._resolve_fallback_non_star_id()

    def set_class_token_logit_bias(self, bias: Optional[List[float]]) -> None:
        """Set per-token logit bias to steer sampling toward a target class.

        Args:
            bias: List of length vocab_size with additive logit biases,
                  or None to disable.
        """
        if bias is None:
            self.class_token_logit_bias = None
        else:
            self.class_token_logit_bias = torch.tensor(
                bias, dtype=torch.float32, device=self.device
            )

    def set_class_token_bias_start_frac(self, start_frac: float) -> None:
        """Set late-activation threshold for class-token logit bias."""
        start_frac = float(start_frac)
        if not (0.0 <= start_frac <= 1.0):
            raise ValueError(f"class_token_bias_start_frac must be in [0, 1], got {start_frac}")
        self.class_token_bias_start_frac = start_frac

    def set_forbidden_token_ids(self, token_ids: Optional[Sequence[int]]) -> None:
        """Set token IDs that must never be sampled at mutable positions."""
        vocab_size = int(self.tokenizer.vocab_size)
        self.forbidden_token_ids = {
            int(token_id)
            for token_id in (token_ids or [])
            if 0 <= int(token_id) < vocab_size
        }
        self.fallback_non_star_id = self._resolve_fallback_non_star_id()

    def set_forbidden_tokens(self, tokens: Optional[Sequence[str]]) -> None:
        """Set tokenizer strings that must never be sampled at mutable positions."""
        token_ids = [
            int(self.tokenizer.vocab[str(token)])
            for token in (tokens or [])
            if str(token) in self.tokenizer.vocab
        ]
        self.set_forbidden_token_ids(token_ids)

    def _class_token_bias_scale(self, step_progress: float) -> float:
        """Linearly ramp class-token bias from ``start_frac`` to the final step."""
        if self.class_token_logit_bias is None:
            return 0.0
        start_frac = float(self.class_token_bias_start_frac)
        progress = float(step_progress)
        if progress < start_frac:
            return 0.0
        if start_frac >= 1.0:
            return 1.0
        return float(min(1.0, max(0.0, (progress - start_frac) / max(1.0e-8, 1.0 - start_frac))))

    def _apply_class_token_bias(
        self,
        logits: torch.Tensor,
        *,
        fixed_mask: torch.Tensor,
        step_progress: float,
    ) -> torch.Tensor:
        """Apply late-ramped class-token bias at non-fixed positions."""
        scale = self._class_token_bias_scale(float(step_progress))
        if scale <= 0.0 or self.class_token_logit_bias is None:
            return logits
        bias_mask = (~fixed_mask).unsqueeze(-1).float()
        return logits + (scale * self.class_token_logit_bias.unsqueeze(0).unsqueeze(0) * bias_mask)

    def _step_progress_frac(self, t: int) -> float:
        """Map reverse-diffusion step ``t`` to progress fraction in [0, 1]."""
        if self.num_steps <= 1:
            return 1.0
        return 1.0 - (float(t) - 1.0) / float(self.num_steps - 1)

    def _resolve_fallback_non_star_id(self) -> int:
        """Choose a stable fallback token ID that is not special and not '*'."""
        preferred = self.tokenizer.vocab.get('C', -1)
        forbidden = {self.star_id, self.mask_id, self.pad_id, self.bos_id, self.eos_id}
        forbidden.update(getattr(self, "forbidden_token_ids", set()))
        if preferred >= 0 and preferred not in forbidden:
            return int(preferred)

        vocab_size = self.tokenizer.vocab_size
        for tok in range(vocab_size):
            if tok not in forbidden:
                return int(tok)
        return 0

    def _best_replacement_token(
        self,
        pos_logits: torch.Tensor,
        forbidden_tokens: set[int]
    ) -> Optional[int]:
        """Return the best token ID after masking forbidden IDs.

        Returns None when no valid replacement is available.
        """
        candidate = pos_logits.clone()
        forbidden_tokens = set(forbidden_tokens)
        forbidden_tokens.update(getattr(self, "forbidden_token_ids", set()))
        vocab_size = candidate.shape[0]
        for tok in forbidden_tokens:
            if 0 <= tok < vocab_size:
                candidate[tok] = float('-inf')
        if not torch.isfinite(candidate).any():
            return None
        return int(candidate.argmax().item())

    def _count_stars(self, ids: torch.Tensor) -> torch.Tensor:
        """Count '*' tokens in each sequence.

        Args:
            ids: Token IDs of shape [batch, seq_len].

        Returns:
            Counts of shape [batch].
        """
        return (ids == self.star_id).sum(dim=1)

    def _analyze_syntax_state(self, seq: torch.Tensor) -> dict:
        """Analyze syntax state of partially generated sequence.

        Examines unmasked tokens to determine current SMILES syntax state.
        Used to determine which tokens are valid for remaining masked positions.

        Args:
            seq: Token IDs of shape [seq_len].

        Returns:
            Dictionary with:
            - paren_depth: int (unclosed parentheses count)
            - open_rings: set (ring numbers currently open)
            - has_unclosed_structure: bool (any unclosed parens)
        """
        paren_depth = 0
        open_rings = set()

        for i in range(len(seq)):
            token_id = seq[i].item()

            # Skip masked and special tokens
            if token_id in [self.mask_id, self.pad_id, self.bos_id, self.eos_id]:
                continue

            # Track parentheses
            if token_id == self.open_paren_id:
                paren_depth += 1
            elif token_id == self.close_paren_id:
                paren_depth = max(0, paren_depth - 1)

            # Track ring closures (digits 0-9)
            if token_id in self.ring_digit_ids:
                if token_id in open_rings:
                    open_rings.remove(token_id)
                else:
                    open_rings.add(token_id)

            # Track ring closures (%10, %11, etc.)
            if token_id in self.ring_percent_ids:
                if token_id in open_rings:
                    open_rings.remove(token_id)
                else:
                    open_rings.add(token_id)

        return {
            'paren_depth': paren_depth,
            'open_rings': open_rings,
            'has_unclosed_structure': paren_depth > 0
        }

    def _get_forbidden_tokens(self, syntax_state: dict, num_masked: int) -> set:
        """Return token IDs that are forbidden given current syntax state.

        Args:
            syntax_state: Dictionary from _analyze_syntax_state.
            num_masked: Number of remaining masked positions.

        Returns:
            Set of token IDs that should be forbidden.
        """
        forbidden = set()
        paren_depth = syntax_state['paren_depth']

        # Rule 1: Cannot close more parentheses than are open
        if paren_depth == 0:
            if self.close_paren_id >= 0:
                forbidden.add(self.close_paren_id)

        # Rule 2: Cannot open '(' if not enough positions left to close existing parens
        # Need at least 1 position for each unclosed paren
        if num_masked <= paren_depth:
            if self.open_paren_id >= 0:
                forbidden.add(self.open_paren_id)

        # Rule 3: Limit total open parens to half of remaining positions
        # Each '(' needs a matching ')', so max opens = num_masked // 2
        max_allowed_opens = num_masked // 2
        if paren_depth >= max_allowed_opens:
            if self.open_paren_id >= 0:
                forbidden.add(self.open_paren_id)

        # Rule 4: Forbid special tokens from being sampled
        forbidden.add(self.mask_id)
        forbidden.add(self.pad_id)
        forbidden.add(self.bos_id)
        forbidden.add(self.eos_id)

        return forbidden

    def _apply_special_token_constraints(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forbid special and configured tokens from being sampled at MASK positions."""
        is_masked = current_ids == self.mask_id
        forbidden_tokens = [
            self.mask_id,
            self.pad_id,
            self.bos_id,
            self.eos_id,
            *sorted(getattr(self, "forbidden_token_ids", set())),
        ]
        for tok in forbidden_tokens:
            if tok >= 0:
                logits[:, :, tok].masked_fill_(is_masked, float('-inf'))
        return logits

    def _apply_syntax_constraints(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply SMILES syntax constraints to logits.

        For each sequence in batch:
        1. Analyze current unmasked tokens to get syntax state
        2. Determine forbidden tokens for remaining masked positions
        3. Set logits of forbidden tokens to -inf

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].

        Returns:
            Modified logits with invalid tokens masked out.
        """
        batch_size, seq_len, vocab_size = logits.shape

        for i in range(batch_size):
            # Count masked positions
            num_masked = (current_ids[i] == self.mask_id).sum().item()

            # Analyze syntax state from unmasked tokens
            syntax_state = self._analyze_syntax_state(current_ids[i])

            # Get forbidden tokens
            forbidden = self._get_forbidden_tokens(syntax_state, num_masked)

            # Apply to all masked positions
            mask_positions = current_ids[i] == self.mask_id
            for token_id in forbidden:
                if token_id >= 0:
                    logits[i, mask_positions, token_id] = float('-inf')

        return logits

    def _apply_position_aware_paren_constraints(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply position-aware parenthesis constraints to logits.

        Unlike global constraints, this method considers the POSITION of each
        masked token to determine valid parenthesis placements:
        - ')' can only be placed if there's an unclosed '(' to its LEFT
        - '(' can only be placed if there's room to close it to its RIGHT

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].

        Returns:
            Modified logits with invalid tokens masked out.
        """
        if self.open_paren_id < 0 and self.close_paren_id < 0:
            return logits

        is_masked = current_ids == self.mask_id
        if not is_masked.any():
            return logits

        open_mask = current_ids == self.open_paren_id
        close_mask = current_ids == self.close_paren_id
        depth_delta = open_mask.int() - close_mask.int()
        depth_prefix = depth_delta.cumsum(dim=1)
        depth_at_pos = torch.cat(
            [torch.zeros_like(depth_prefix[:, :1]), depth_prefix[:, :-1]],
            dim=1
        )

        masked_from_pos = torch.flip(
            torch.cumsum(torch.flip(is_masked.int(), dims=[1]), dim=1),
            dims=[1]
        )

        if self.close_paren_id >= 0:
            invalid_close = is_masked & (depth_at_pos <= 0)
            logits[:, :, self.close_paren_id].masked_fill_(invalid_close, float('-inf'))

        if self.open_paren_id >= 0:
            invalid_open = is_masked & (depth_at_pos + 2 > masked_from_pos)
            logits[:, :, self.open_paren_id].masked_fill_(invalid_open, float('-inf'))

        for tok in [self.mask_id, self.pad_id, self.bos_id, self.eos_id]:
            if tok >= 0:
                logits[:, :, tok].masked_fill_(is_masked, float('-inf'))

        return logits

    def _apply_ring_constraints(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply ring closure constraints to logits.

        Ensures each ring digit (1-9, %10-%99) appears exactly twice.
        - Forbids ring digits that are already closed (appeared twice)
        - If too many open rings, forbids opening new ones

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].

        Returns:
            Modified logits with invalid ring tokens masked out.
        """
        all_ring_ids = sorted(self.ring_digit_ids | self.ring_percent_ids)
        if not all_ring_ids:
            return logits

        ring_ids = torch.tensor(all_ring_ids, device=current_ids.device, dtype=torch.long)
        ring_mask = current_ids.unsqueeze(-1) == ring_ids
        ring_counts = ring_mask.sum(dim=1)

        open_rings = ring_counts == 1
        closed_rings = ring_counts >= 2

        num_masked = (current_ids == self.mask_id).sum(dim=1)
        open_counts = open_rings.sum(dim=1)
        forbid_open_new = num_masked <= open_counts

        forbidden_ring = closed_rings | (forbid_open_new.unsqueeze(1) & ~open_rings)

        masked_positions = current_ids == self.mask_id
        ring_logits = logits.index_select(2, ring_ids)
        forbidden = masked_positions.unsqueeze(-1) & forbidden_ring.unsqueeze(1)
        ring_logits = ring_logits.masked_fill(forbidden, float('-inf'))
        logits.index_copy_(2, ring_ids, ring_logits)

        return logits

    def _apply_bond_placement_constraints(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply bond placement constraints to logits.

        Bonds (=, #, /, \) must be preceded by an atom, not by:
        - Open parenthesis (
        - Another bond
        - Start of sequence
        - Close parenthesis )

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].

        Returns:
            Modified logits with invalid bond placements masked out.
        """
        if not self.bond_ids:
            return logits

        is_masked = current_ids == self.mask_id
        if not is_masked.any():
            return logits

        batch_size, seq_len = current_ids.shape
        indices = torch.arange(seq_len, device=current_ids.device).unsqueeze(0).expand(batch_size, -1)
        valid_idx = torch.where(is_masked, torch.full_like(indices, -1), indices)
        prev_idx = torch.cummax(valid_idx, dim=1).values

        prev_idx_clamped = prev_idx.clamp(min=0)
        prev_ids = current_ids.gather(1, prev_idx_clamped)
        prev_ids = torch.where(prev_idx >= 0, prev_ids, torch.full_like(prev_ids, -1))

        invalid_prev = prev_idx < 0
        if self.bos_id >= 0:
            invalid_prev = invalid_prev | (prev_ids == self.bos_id)
        if self.eos_id >= 0:
            invalid_prev = invalid_prev | (prev_ids == self.eos_id)
        if self.pad_id >= 0:
            invalid_prev = invalid_prev | (prev_ids == self.pad_id)
        if self.open_paren_id >= 0:
            invalid_prev = invalid_prev | (prev_ids == self.open_paren_id)
        if self.close_paren_id >= 0:
            invalid_prev = invalid_prev | (prev_ids == self.close_paren_id)

        bond_ids = sorted(self.bond_ids)
        bond_ids_tensor = torch.tensor(bond_ids, device=current_ids.device, dtype=current_ids.dtype)
        prev_is_bond = (prev_ids.unsqueeze(-1) == bond_ids_tensor).any(dim=-1)
        invalid_prev = invalid_prev | prev_is_bond

        invalid_bond = is_masked & invalid_prev
        bond_logits = logits.index_select(2, bond_ids_tensor)
        bond_logits = bond_logits.masked_fill(invalid_bond.unsqueeze(-1), float('-inf'))
        logits.index_copy_(2, bond_ids_tensor, bond_logits)

        if self.open_paren_id >= 0 and self.close_paren_id >= 0:
            invalid_close = is_masked & (prev_ids == self.open_paren_id)
            logits[:, :, self.close_paren_id].masked_fill_(invalid_close, float('-inf'))

        return logits

    def _apply_star_constraint(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor,
        max_stars: int = 2
    ) -> torch.Tensor:
        """Apply constraint to limit number of '*' tokens.

        Args:
            logits: Logits of shape [batch, seq_len, vocab_size].
            current_ids: Current token IDs of shape [batch, seq_len].
            max_stars: Maximum allowed '*' tokens.

        Returns:
            Modified logits.
        """
        if self.star_id < 0:
            return logits

        # Count current stars (excluding MASK positions)
        non_mask = current_ids != self.mask_id
        current_stars = ((current_ids == self.star_id) & non_mask).sum(dim=1)

        # For sequences with >= max_stars, set star logit to -inf at MASK positions
        blocked = current_stars >= max_stars
        if blocked.any():
            mask_positions = current_ids == self.mask_id
            logits[:, :, self.star_id].masked_fill_(mask_positions & blocked.unsqueeze(1), float('-inf'))

        return logits

    def _apply_exact_star_budget_constraint(
        self,
        logits: torch.Tensor,
        current_ids: torch.Tensor,
        target_stars: int = 2
    ) -> torch.Tensor:
        """Enforce an exact star budget over remaining masked positions.

        Rules:
        - If no stars are needed, forbid '*' in all remaining masked positions.
        - If stars needed >= remaining masked positions, force '*' in all masked positions.
        """
        if self.star_id < 0:
            return logits

        is_masked = current_ids == self.mask_id
        remaining_masked = is_masked.sum(dim=1)
        current_stars = (current_ids == self.star_id).sum(dim=1)
        needed_stars = target_stars - current_stars

        batch_size = current_ids.shape[0]
        for i in range(batch_size):
            mask_positions = is_masked[i]
            n_masked = int(remaining_masked[i].item())
            if n_masked <= 0:
                continue

            needed = int(needed_stars[i].item())
            if needed <= 0:
                logits[i, mask_positions, self.star_id] = float('-inf')
                continue

            if needed >= n_masked:
                logits[i, mask_positions, :] = float('-inf')
                logits[i, mask_positions, self.star_id] = 0.0

        return logits

    def _apply_sampling_filters(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply top-k/top-p filtering to logits."""
        filtered = logits
        original = logits

        if self.top_k is not None and self.top_k > 0 and self.top_k < filtered.shape[-1]:
            topk_vals = torch.topk(filtered, self.top_k, dim=-1).values
            kth = topk_vals[..., -1].unsqueeze(-1)
            filtered = filtered.masked_fill(filtered < kth, float('-inf'))

        if self.top_p is not None and 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_remove = cumulative_probs > self.top_p
            sorted_remove[..., 0] = False

            remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool)
            remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
            filtered = filtered.masked_fill(remove_mask, float('-inf'))

        # Guard: keep at least one valid token per position if filters over-prune.
        has_valid = torch.isfinite(filtered).any(dim=-1, keepdim=True)
        filtered = torch.where(has_valid, filtered, original)
        return filtered

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to valid probability vectors with robust fallbacks."""
        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        sums = probs.sum(dim=-1, keepdim=True)
        valid_rows = sums > 0

        finite = torch.isfinite(logits)
        finite_count = finite.sum(dim=-1, keepdim=True)
        uniform_over_finite = finite.float() / finite_count.clamp(min=1).float()

        fallback_token = self.star_id if self.star_id >= 0 else 0
        hard_fallback = torch.zeros_like(probs)
        hard_fallback[..., fallback_token] = 1.0
        row_fallback = torch.where(finite_count > 0, uniform_over_finite, hard_fallback)

        normalized = probs / sums.clamp(min=1e-12)
        return torch.where(valid_rows, normalized, row_fallback)

    def _ensure_valid_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Guarantee at least one finite token logit at every position.

        Constraint stacks can occasionally prune an entire token row to ``-inf``.
        Sampling tolerates that through probability fallbacks, but log-softmax-
        based objectives such as RL KL and replay diagnostics need finite logits.
        """
        finite = torch.isfinite(logits)
        valid_rows = finite.any(dim=-1, keepdim=True)
        if bool(valid_rows.all()):
            return logits

        repaired = logits.clone()
        fallback = torch.full_like(repaired, float("-inf"))
        fallback[..., self.fallback_non_star_id] = 0.0
        return torch.where(valid_rows, repaired, fallback)

    def _fix_star_count(
        self,
        ids: torch.Tensor,
        logits: torch.Tensor,
        target_stars: int = 2,
        fixed_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fix the number of '*' tokens in final sequences.

        Args:
            ids: Token IDs of shape [batch, seq_len].
            logits: Final logits of shape [batch, seq_len, vocab_size].
            target_stars: Target number of '*' tokens.
            fixed_mask: Boolean mask of fixed (protected) positions to skip.

        Returns:
            Fixed token IDs.
        """
        batch_size, seq_len = ids.shape
        fixed_ids = ids.clone()

        for i in range(batch_size):
            star_mask = fixed_ids[i] == self.star_id
            num_stars = star_mask.sum().item()

            if num_stars > target_stars:
                # Keep only the top-k most probable star positions
                star_positions = torch.where(star_mask)[0]
                # Only remove stars from non-fixed positions
                if fixed_mask is not None:
                    removable = star_positions[~fixed_mask[i, star_positions]]
                else:
                    removable = star_positions
                star_probs = logits[i, star_positions, self.star_id]

                # Get indices of stars to keep (highest probability)
                _, keep_indices = torch.topk(star_probs, min(target_stars, len(star_positions)))
                keep_positions = {
                    int(p.item()) for p in star_positions[keep_indices]
                }

                # Replace extra stars with second-best token (only non-fixed)
                for pos in removable:
                    pos_idx = int(pos.item())
                    if pos_idx not in keep_positions:
                        best_token = self._best_replacement_token(
                            pos_logits=logits[i, pos_idx],
                            forbidden_tokens={
                                self.star_id,
                                self.mask_id,
                                self.pad_id,
                                self.bos_id,
                                self.eos_id,
                                self.open_paren_id,
                                self.close_paren_id,
                                *self.bond_ids,
                                *self.ring_digit_ids,
                                *self.ring_percent_ids,
                            },
                        )
                        if best_token is None:
                            best_token = self.fallback_non_star_id
                        fixed_ids[i, pos_idx] = best_token

            elif num_stars < target_stars:
                # Find best positions to add stars
                needed = target_stars - num_stars

                # Get star probabilities at all non-special, non-fixed positions
                valid_mask = (
                    (fixed_ids[i] != self.bos_id) &
                    (fixed_ids[i] != self.eos_id) &
                    (fixed_ids[i] != self.pad_id) &
                    (fixed_ids[i] != self.star_id)
                )
                if fixed_mask is not None:
                    valid_mask = valid_mask & ~fixed_mask[i]
                valid_positions = torch.where(valid_mask)[0]

                if len(valid_positions) >= needed:
                    star_probs = logits[i, valid_positions, self.star_id]
                    _, best_indices = torch.topk(star_probs, needed)
                    best_positions = valid_positions[best_indices]

                    for pos in best_positions:
                        fixed_ids[i, pos] = self.star_id
                elif needed > 0:
                    eligible_mask = (
                        (fixed_ids[i] != self.bos_id) &
                        (fixed_ids[i] != self.eos_id) &
                        (fixed_ids[i] != self.pad_id)
                    )
                    if fixed_mask is not None:
                        eligible_mask = eligible_mask & ~fixed_mask[i]
                    eligible_positions = torch.where(eligible_mask)[0]
                    if len(eligible_positions) > 0:
                        k = min(needed, len(eligible_positions))
                        star_probs = logits[i, eligible_positions, self.star_id]
                        _, best_indices = torch.topk(star_probs, k)
                        best_positions = eligible_positions[best_indices]
                        for pos in best_positions:
                            fixed_ids[i, pos] = self.star_id

        return fixed_ids

    def _fix_paren_balance(
        self,
        ids: torch.Tensor,
        logits: torch.Tensor,
        fixed_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fix unbalanced parentheses in final sequences.

        Two-pass algorithm:
        1. Left-to-right: Replace any ')' that has no matching '(' to its left
        2. Right-to-left: Replace any '(' that has no matching ')' to its right

        Positions marked in fixed_mask are never modified.

        Args:
            ids: Token IDs of shape [batch, seq_len].
            logits: Final logits of shape [batch, seq_len, vocab_size].
            fixed_mask: Boolean mask of fixed (protected) positions to skip.

        Returns:
            Fixed token IDs with balanced parentheses.
        """
        batch_size, seq_len = ids.shape
        fixed_ids = ids.clone()

        for i in range(batch_size):
            # First pass (left-to-right): remove ')' that have no matching '('
            depth = 0
            for j in range(seq_len):
                token_id = fixed_ids[i, j].item()
                if token_id == self.open_paren_id:
                    depth += 1
                elif token_id == self.close_paren_id:
                    if depth > 0:
                        depth -= 1
                    elif fixed_mask is not None and fixed_mask[i, j]:
                        pass  # Cannot modify fixed position
                    else:
                        # No matching '(' to the left - replace with best alternative
                        best_token = self._best_replacement_token(
                            pos_logits=logits[i, j],
                            forbidden_tokens={
                                self.close_paren_id,
                                self.open_paren_id,
                                self.mask_id,
                                self.pad_id,
                                self.bos_id,
                                self.eos_id,
                                self.star_id,
                            },
                        )
                        if best_token is not None:
                            fixed_ids[i, j] = best_token

            # Second pass (right-to-left): remove unclosed '('
            depth = 0
            for j in range(seq_len - 1, -1, -1):
                token_id = fixed_ids[i, j].item()
                if token_id == self.close_paren_id:
                    depth += 1
                elif token_id == self.open_paren_id:
                    if depth > 0:
                        depth -= 1
                    elif fixed_mask is not None and fixed_mask[i, j]:
                        pass  # Cannot modify fixed position
                    else:
                        # No matching ')' to the right - replace with best alternative
                        best_token = self._best_replacement_token(
                            pos_logits=logits[i, j],
                            forbidden_tokens={
                                self.close_paren_id,
                                self.open_paren_id,
                                self.mask_id,
                                self.pad_id,
                                self.bos_id,
                                self.eos_id,
                                self.star_id,
                            },
                        )
                        if best_token is not None:
                            fixed_ids[i, j] = best_token

        return fixed_ids

    def _fix_ring_closures(
        self,
        ids: torch.Tensor,
        logits: torch.Tensor,
        fixed_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fix unpaired ring closures in final sequences.

        Scans each sequence and replaces ring digits that appear an odd number
        of times (unpaired) with the next-best non-ring token.
        Positions marked in fixed_mask are never modified.

        Args:
            ids: Token IDs of shape [batch, seq_len].
            logits: Final logits of shape [batch, seq_len, vocab_size].
            fixed_mask: Boolean mask of fixed (protected) positions to skip.

        Returns:
            Fixed token IDs with paired ring closures.
        """
        batch_size, seq_len = ids.shape
        fixed_ids = ids.clone()

        # Combine all ring token IDs
        all_ring_ids = self.ring_digit_ids | self.ring_percent_ids

        for i in range(batch_size):
            # Count occurrences and positions of each ring digit
            ring_positions = {}  # ring_id -> list of positions
            for j in range(seq_len):
                token_id = fixed_ids[i, j].item()
                if token_id in all_ring_ids:
                    if token_id not in ring_positions:
                        ring_positions[token_id] = []
                    ring_positions[token_id].append(j)

            # Fix rings with odd counts (replace last non-fixed occurrence)
            for ring_id, positions in ring_positions.items():
                if len(positions) % 2 != 0:  # Odd count - unpaired
                    # Find the last non-fixed position to replace
                    target_pos = None
                    for candidate_pos in reversed(positions):
                        if fixed_mask is None or not fixed_mask[i, candidate_pos]:
                            target_pos = candidate_pos
                            break
                    if target_pos is None:
                        continue  # All positions are fixed, skip

                    forbidden = set(all_ring_ids)
                    forbidden.update(
                        {
                            self.mask_id,
                            self.pad_id,
                            self.bos_id,
                            self.eos_id,
                            self.open_paren_id,
                            self.close_paren_id,
                            self.star_id,
                        }
                    )
                    best_token = self._best_replacement_token(
                        pos_logits=logits[i, target_pos],
                        forbidden_tokens=forbidden,
                    )
                    if best_token is not None:
                        fixed_ids[i, target_pos] = best_token

        return fixed_ids

    def _fix_bond_placement(
        self,
        ids: torch.Tensor,
        logits: torch.Tensor,
        fixed_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Repair obvious invalid bond placements in final sequences.

        Positions marked in fixed_mask are never modified.
        """
        if not self.bond_ids:
            return ids

        batch_size, seq_len = ids.shape
        fixed_ids = ids.clone()
        bond_ids = set(self.bond_ids)

        for i in range(batch_size):
            for j in range(seq_len):
                token_id = int(fixed_ids[i, j].item())
                if token_id not in bond_ids:
                    continue
                # Never modify fixed positions (e.g. decode-constraint motif spans)
                if fixed_mask is not None and fixed_mask[i, j]:
                    continue

                prev_id = int(fixed_ids[i, j - 1].item()) if j > 0 else -1
                next_id = int(fixed_ids[i, j + 1].item()) if (j + 1) < seq_len else -1

                invalid_prev = (
                    j == 0
                    or prev_id in bond_ids
                    or prev_id in {self.bos_id, self.eos_id, self.pad_id, self.open_paren_id, self.close_paren_id}
                )
                invalid_next = (
                    (j + 1) >= seq_len
                    or next_id in bond_ids
                    or next_id in {self.eos_id, self.pad_id, self.close_paren_id}
                )

                if not (invalid_prev or invalid_next):
                    continue

                forbidden = set(bond_ids)
                forbidden.update(
                    {
                        self.mask_id,
                        self.pad_id,
                        self.bos_id,
                        self.eos_id,
                        self.star_id,
                        self.open_paren_id,
                        self.close_paren_id,
                    }
                )
                forbidden.update(self.ring_digit_ids)
                forbidden.update(self.ring_percent_ids)
                best_token = self._best_replacement_token(
                    pos_logits=logits[i, j],
                    forbidden_tokens=forbidden,
                )
                if best_token is not None:
                    fixed_ids[i, j] = best_token

        return fixed_ids

    def _sample_from_ids(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Run reverse diffusion sampling starting from provided token IDs.

        Args:
            ids: Initial token IDs of shape [batch, seq_len].
            attention_mask: Attention mask of shape [batch, seq_len].
            fixed_mask: Boolean mask marking fixed (non-sampled) positions.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (token_ids, smiles_strings).
        """
        self.diffusion_model.eval()
        backbone = self.diffusion_model.backbone
        batch_size = ids.shape[0]

        final_logits = None
        steps = range(self.num_steps, 0, -1)
        if show_progress:
            steps = tqdm(steps, desc="Sampling")

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
            # Apply class-enriched token bias at non-fixed positions
            logits = self._apply_class_token_bias(logits, fixed_mask=fixed_mask, step_progress=step_progress)
            logits = self._apply_sampling_filters(logits)
            logits = self._apply_special_token_constraints(logits, ids)

            probs = self._logits_to_probs(logits)
            is_masked = (ids == self.mask_id) & (~fixed_mask)
            unmask_prob = 1.0 / t

            for i in range(batch_size):
                masked_pos = torch.where(is_masked[i])[0]
                if len(masked_pos) == 0:
                    continue

                num_unmask = max(1, int(len(masked_pos) * unmask_prob))
                unmask_indices = torch.randperm(len(masked_pos))[:num_unmask]
                unmask_positions = masked_pos[unmask_indices]

                # Sample tokens for these positions SEQUENTIALLY with constraint updates
                for pos in unmask_positions:
                    sampled = torch.multinomial(probs[i, pos], 1)
                    ids[i, pos] = sampled

                    sampled_token = sampled.item()

                    if self.use_constraints:
                        if sampled_token == self.star_id:
                            non_mask = ids[i] != self.mask_id
                            current_stars = ((ids[i] == self.star_id) & non_mask).sum().item()

                            if current_stars >= self.target_stars:
                                remaining_mask = (ids[i] == self.mask_id) & (~fixed_mask[i])
                                logits[i, remaining_mask, self.star_id] = float('-inf')
                                probs[i] = self._logits_to_probs(logits[i])

                        elif sampled_token in self.bond_ids:
                            next_pos = pos + 1
                            if next_pos < len(ids[i]) and ids[i, next_pos] == self.mask_id:
                                for bond_id in self.bond_ids:
                                    logits[i, next_pos, bond_id] = float('-inf')
                                probs[i] = self._logits_to_probs(logits[i])

                        elif sampled_token == self.open_paren_id:
                            next_pos = pos + 1
                            if next_pos < len(ids[i]) and ids[i, next_pos] == self.mask_id:
                                logits[i, next_pos, self.close_paren_id] = float('-inf')
                                probs[i] = self._logits_to_probs(logits[i])

            if t == 1:
                final_logits = logits

        if self.use_constraints:
            ids = self._fix_ring_closures(ids, final_logits, fixed_mask=fixed_mask)
            ids = self._fix_bond_placement(ids, final_logits, fixed_mask=fixed_mask)
            ids = self._fix_paren_balance(ids, final_logits, fixed_mask=fixed_mask)
            ids = self._fix_star_count(ids, final_logits, target_stars=self.target_stars, fixed_mask=fixed_mask)
            # Final cleanup after token replacements.
            ids = self._fix_ring_closures(ids, final_logits, fixed_mask=fixed_mask)
            ids = self._fix_paren_balance(ids, final_logits, fixed_mask=fixed_mask)
        smiles_list = self.tokenizer.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)

        return ids, smiles_list

    def sample_with_lengths(
        self,
        lengths: List[int],
        max_length: Optional[int] = None,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sample polymers with per-sample lengths (includes BOS/EOS).

        Args:
            lengths: Sequence lengths INCLUDING BOS/EOS for each sample.
            max_length: Optional hard cap for sequence length.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (token_ids, smiles_strings).
        """
        if not lengths:
            return torch.empty((0, 0), dtype=torch.long, device=self.device), []

        lengths = [max(2, int(l)) for l in lengths]
        seq_length = max(lengths)
        if max_length is not None and seq_length > max_length:
            raise ValueError(f"Max length {max_length} is smaller than required {seq_length}")

        batch_size = len(lengths)
        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device
        )
        attention_mask = torch.zeros_like(ids)
        fixed_mask = torch.zeros_like(ids, dtype=torch.bool)

        ids[:, 0] = self.bos_id
        fixed_mask[:, 0] = True

        for i, length in enumerate(lengths):
            eos_pos = length - 1
            ids[i, eos_pos] = self.eos_id
            fixed_mask[i, eos_pos] = True
            attention_mask[i, :length] = 1

            if length < seq_length:
                ids[i, length:] = self.pad_id
                fixed_mask[i, length:] = True

        return self._sample_from_ids(ids, attention_mask, fixed_mask, show_progress)

    def sample(
        self,
        batch_size: int,
        seq_length: int,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sample new polymers with exactly two '*' tokens.

        Args:
            batch_size: Number of samples to generate.
            seq_length: Sequence length.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (token_ids, smiles_strings).
        """
        self.diffusion_model.eval()

        # Initialize with fully masked sequence (except BOS/EOS)
        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device
        )
        ids[:, 0] = self.bos_id
        ids[:, -1] = self.eos_id

        # Create attention mask
        attention_mask = torch.ones_like(ids)

        fixed_mask = ids != self.mask_id

        return self._sample_from_ids(ids, attention_mask, fixed_mask, show_progress)

    def sample_batch(
        self,
        num_samples: int,
        seq_length: int,
        batch_size: int = 256,
        show_progress: bool = True,
        lengths: Optional[List[int]] = None
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Sample multiple batches of polymers.

        Args:
            num_samples: Total number of samples.
            seq_length: Sequence length.
            batch_size: Batch size for sampling.
            show_progress: Whether to show progress.
            lengths: Optional list of per-sample lengths (includes BOS/EOS).

        Returns:
            Tuple of (all_ids, all_smiles).
        """
        all_ids = []
        all_smiles = []

        if lengths is not None and len(lengths) != num_samples:
            raise ValueError("lengths must match num_samples")

        num_batches = (num_samples + batch_size - 1) // batch_size
        sample_idx = 0

        for batch_idx in tqdm(range(num_batches), desc="Batch sampling", disable=not show_progress):
            current_batch_size = min(batch_size, num_samples - sample_idx)

            if lengths is None:
                ids, smiles = self.sample(
                    current_batch_size,
                    seq_length,
                    show_progress=False
                )
            else:
                batch_lengths = lengths[sample_idx:sample_idx + current_batch_size]
                ids, smiles = self.sample_with_lengths(
                    batch_lengths,
                    max_length=seq_length,
                    show_progress=False
                )

            all_ids.append(ids)
            all_smiles.extend(smiles)
            sample_idx += current_batch_size

        return all_ids, all_smiles

    def sample_with_fixed_spans(
        self,
        span_token_ids: Sequence[Sequence[int]],
        span_start_positions: Sequence[int],
        seq_length: int,
        lengths: Optional[Sequence[int]] = None,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sample by fixing one interior token span per sequence.

        Args:
            span_token_ids: Token-id spans to keep fixed for each sample.
            span_start_positions: Start positions for each fixed span.
            seq_length: Shared padded sequence length.
            lengths: Optional per-sample sequence lengths including BOS/EOS.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (token_ids, smiles_strings).
        """
        batch_size = len(span_token_ids)
        if len(span_start_positions) != batch_size:
            raise ValueError("span_start_positions must match span_token_ids")
        if lengths is not None and len(lengths) != batch_size:
            raise ValueError("lengths must match span_token_ids")

        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device
        )
        attention_mask = torch.ones_like(ids)
        fixed_mask = torch.zeros_like(ids, dtype=torch.bool)

        ids[:, 0] = self.bos_id
        fixed_mask[:, 0] = True

        effective_lengths: List[int] = []
        if lengths is None:
            ids[:, -1] = self.eos_id
            fixed_mask[:, -1] = True
            effective_lengths = [int(seq_length)] * batch_size
        else:
            attention_mask = torch.zeros_like(ids)
            for i, raw_length in enumerate(lengths):
                length = max(2, int(raw_length))
                if length > seq_length:
                    raise ValueError(f"length {length} exceeds seq_length={seq_length}")
                effective_lengths.append(length)
                eos_pos = length - 1
                ids[i, eos_pos] = self.eos_id
                fixed_mask[i, eos_pos] = True
                attention_mask[i, :length] = 1
                if length < seq_length:
                    ids[i, length:] = self.pad_id
                    fixed_mask[i, length:] = True

        for i, (span_ids, start_pos) in enumerate(zip(span_token_ids, span_start_positions)):
            span = list(int(token_id) for token_id in span_ids)
            if not span:
                continue
            effective_length = effective_lengths[i]
            if effective_length < 2:
                raise ValueError(f"effective length must be >= 2, got {effective_length}")
            end_pos = int(start_pos) + len(span)
            if int(start_pos) < 1 or end_pos > (effective_length - 1):
                raise ValueError(
                    f"Fixed span [{start_pos}, {end_pos}) does not fit sequence length {effective_length}"
                )
            ids[i, int(start_pos):end_pos] = torch.tensor(span, dtype=torch.long, device=self.device)
            fixed_mask[i, int(start_pos):end_pos] = True

        return self._sample_from_ids(ids, attention_mask, fixed_mask, show_progress)

    def sample_with_multiple_fixed_spans(
        self,
        span_token_ids: Sequence[Sequence[Sequence[int]]],
        span_start_positions: Sequence[Sequence[int]],
        seq_length: int,
        lengths: Optional[Sequence[int]] = None,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sample by fixing multiple interior token spans per sequence."""
        batch_size = len(span_token_ids)
        if len(span_start_positions) != batch_size:
            raise ValueError("span_start_positions must match span_token_ids")
        if lengths is not None and len(lengths) != batch_size:
            raise ValueError("lengths must match span_token_ids")

        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device
        )
        attention_mask = torch.ones_like(ids)
        fixed_mask = torch.zeros_like(ids, dtype=torch.bool)

        ids[:, 0] = self.bos_id
        fixed_mask[:, 0] = True

        effective_lengths: List[int] = []
        if lengths is None:
            ids[:, -1] = self.eos_id
            fixed_mask[:, -1] = True
            effective_lengths = [int(seq_length)] * batch_size
        else:
            attention_mask = torch.zeros_like(ids)
            for i, raw_length in enumerate(lengths):
                length = max(2, int(raw_length))
                if length > seq_length:
                    raise ValueError(f"length {length} exceeds seq_length={seq_length}")
                effective_lengths.append(length)
                eos_pos = length - 1
                ids[i, eos_pos] = self.eos_id
                fixed_mask[i, eos_pos] = True
                attention_mask[i, :length] = 1
                if length < seq_length:
                    ids[i, length:] = self.pad_id
                    fixed_mask[i, length:] = True

        for i, (sample_spans, sample_starts) in enumerate(zip(span_token_ids, span_start_positions)):
            if len(sample_spans) != len(sample_starts):
                raise ValueError("Each sample's span list must match its start-position list")
            effective_length = effective_lengths[i]
            prev_end = 1
            for span_ids, start_pos in zip(sample_spans, sample_starts):
                span = list(int(token_id) for token_id in span_ids)
                if not span:
                    continue
                if effective_length < 2:
                    raise ValueError(f"effective length must be >= 2, got {effective_length}")
                end_pos = int(start_pos) + len(span)
                if int(start_pos) < 1 or end_pos > (effective_length - 1):
                    raise ValueError(
                        f"Fixed span [{start_pos}, {end_pos}) does not fit sequence length {effective_length}"
                    )
                if int(start_pos) < prev_end:
                    raise ValueError(
                        f"Overlapping fixed spans detected for sample {i}: "
                        f"start={start_pos}, previous_end={prev_end}"
                    )
                ids[i, int(start_pos):end_pos] = torch.tensor(span, dtype=torch.long, device=self.device)
                fixed_mask[i, int(start_pos):end_pos] = True
                prev_end = end_pos

        return self._sample_from_ids(ids, attention_mask, fixed_mask, show_progress)

    def sample_batch_with_multiple_fixed_spans(
        self,
        num_samples: int,
        seq_length: int,
        span_token_ids: Sequence[Sequence[Sequence[int]]],
        span_start_positions: Sequence[Sequence[int]],
        batch_size: int = 256,
        show_progress: bool = True,
        lengths: Optional[Sequence[int]] = None
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Batch wrapper for multi-span fixed constrained sampling."""
        if len(span_token_ids) != num_samples:
            raise ValueError("span_token_ids must match num_samples")
        if len(span_start_positions) != num_samples:
            raise ValueError("span_start_positions must match num_samples")
        if lengths is not None and len(lengths) != num_samples:
            raise ValueError("lengths must match num_samples")

        all_ids = []
        all_smiles = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        sample_idx = 0

        for _ in tqdm(range(num_batches), desc="Batch sampling", disable=not show_progress):
            current_batch_size = min(batch_size, num_samples - sample_idx)
            batch_span_ids = span_token_ids[sample_idx:sample_idx + current_batch_size]
            batch_span_starts = span_start_positions[sample_idx:sample_idx + current_batch_size]
            batch_lengths = None if lengths is None else lengths[sample_idx:sample_idx + current_batch_size]

            ids, smiles = self.sample_with_multiple_fixed_spans(
                span_token_ids=batch_span_ids,
                span_start_positions=batch_span_starts,
                seq_length=seq_length,
                lengths=batch_lengths,
                show_progress=False,
            )
            all_ids.append(ids)
            all_smiles.extend(smiles)
            sample_idx += current_batch_size

        return all_ids, all_smiles

    def sample_batch_with_fixed_spans(
        self,
        num_samples: int,
        seq_length: int,
        span_token_ids: Sequence[Sequence[int]],
        span_start_positions: Sequence[int],
        batch_size: int = 256,
        show_progress: bool = True,
        lengths: Optional[Sequence[int]] = None
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Batch wrapper for fixed-span constrained sampling."""
        if len(span_token_ids) != num_samples:
            raise ValueError("span_token_ids must match num_samples")
        if len(span_start_positions) != num_samples:
            raise ValueError("span_start_positions must match num_samples")
        if lengths is not None and len(lengths) != num_samples:
            raise ValueError("lengths must match num_samples")

        all_ids = []
        all_smiles = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        sample_idx = 0

        for _ in tqdm(range(num_batches), desc="Batch sampling", disable=not show_progress):
            current_batch_size = min(batch_size, num_samples - sample_idx)
            batch_span_ids = span_token_ids[sample_idx:sample_idx + current_batch_size]
            batch_span_starts = span_start_positions[sample_idx:sample_idx + current_batch_size]
            batch_lengths = None if lengths is None else lengths[sample_idx:sample_idx + current_batch_size]

            ids, smiles = self.sample_with_fixed_spans(
                span_token_ids=batch_span_ids,
                span_start_positions=batch_span_starts,
                seq_length=seq_length,
                lengths=batch_lengths,
                show_progress=False,
            )
            all_ids.append(ids)
            all_smiles.extend(smiles)
            sample_idx += current_batch_size

        return all_ids, all_smiles

    def sample_variable_length(
        self,
        num_samples: int,
        length_range: Tuple[int, int] = (20, 100),
        batch_size: int = 256,
        samples_per_length: int = 16,
        show_progress: bool = True
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Sample polymers with variable sequence lengths.

        Each small batch uses a different randomly chosen sequence length
        from the specified range, producing diverse SMILES lengths.

        Args:
            num_samples: Total number of samples.
            length_range: (min_length, max_length) for sequence lengths.
            batch_size: Maximum batch size for GPU memory.
            samples_per_length: Number of samples per length (controls diversity).
                               Smaller values = more length diversity.
            show_progress: Whether to show progress.

        Returns:
            Tuple of (all_ids, all_smiles).
        """
        all_ids = []
        all_smiles = []

        min_len, max_len = length_range
        # Use smaller internal batch size for more length diversity
        internal_batch_size = min(batch_size, samples_per_length)
        num_batches = (num_samples + internal_batch_size - 1) // internal_batch_size

        for batch_idx in tqdm(range(num_batches), desc="Variable length sampling", disable=not show_progress):
            current_batch_size = min(internal_batch_size, num_samples - len(all_smiles))

            # Random sequence length for this batch
            seq_length = np.random.randint(min_len, max_len + 1)

            ids, smiles = self.sample(
                current_batch_size,
                seq_length,
                show_progress=False
            )

            all_ids.append(ids)
            all_smiles.extend(smiles)

        return all_ids, all_smiles

    def sample_conditional(
        self,
        batch_size: int,
        seq_length: int,
        prefix_ids: Optional[torch.Tensor] = None,
        suffix_ids: Optional[torch.Tensor] = None,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Sample with optional prefix/suffix conditioning.

        Args:
            batch_size: Number of samples.
            seq_length: Sequence length.
            prefix_ids: Fixed prefix tokens.
            suffix_ids: Fixed suffix tokens.
            show_progress: Whether to show progress.

        Returns:
            Tuple of (token_ids, smiles_strings).
        """
        # Initialize
        ids = torch.full(
            (batch_size, seq_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device
        )
        ids[:, 0] = self.bos_id
        ids[:, -1] = self.eos_id

        # Apply prefix/suffix constraints
        fixed_mask = torch.zeros_like(ids, dtype=torch.bool)
        fixed_mask[:, 0] = True  # BOS
        fixed_mask[:, -1] = True  # EOS

        if prefix_ids is not None:
            prefix_len = prefix_ids.shape[1]
            ids[:, 1:1+prefix_len] = prefix_ids
            fixed_mask[:, 1:1+prefix_len] = True

        if suffix_ids is not None:
            suffix_len = suffix_ids.shape[1]
            ids[:, -1-suffix_len:-1] = suffix_ids
            fixed_mask[:, -1-suffix_len:-1] = True

        attention_mask = torch.ones_like(ids)

        return self._sample_from_ids(ids, attention_mask, fixed_mask, show_progress)
