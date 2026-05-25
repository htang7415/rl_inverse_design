import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.step5 import rl_trainer
from src.step5.frozen_sampling import ResolvedClassSamplingPrior


def _prior(**overrides):
    values = {
        "target_class": "polyamide",
        "family_sampling_mode": "none",
        "family_sampling_scope": "target",
        "source_smiles": [],
        "motifs": [],
        "motif_source": "none",
        "motif_token_ids": [],
        "spans_per_sample": 1,
        "center_min_frac": 0.0,
        "center_max_frac": 1.0,
        "length_prior_lengths": [],
        "length_prior_source": None,
        "length_prior_min_tokens": None,
        "length_prior_max_tokens": None,
        "fallback_source_lengths": [],
        "class_token_logit_bias": None,
        "class_token_bias_strength": 0.0,
        "backbone_template_enabled": False,
        "backbone_template_cores": [],
        "backbone_template_source": None,
        "backbone_template_token_ids": [],
        "backbone_template_min_gap_tokens": 0,
        "backbone_template_terminal_star_anchor": False,
        "cycle_backbone_template_cores_across_targets": False,
        "enforce_class_match": False,
        "enforce_backbone_class_match": False,
        "class_match_sampling_attempts_max": 2,
        "class_match_oversample_factor": 1.0,
        "class_match_min_request_size": 1,
        "class_match_max_request_size": None,
        "class_match_max_total_raw_samples": None,
        "allow_partial_quota_return": False,
        "partial_quota_min_fill_ratio": 1.0,
        "enforce_star_ok_acceptance": False,
        "reject_duplicate_canonical_acceptance": False,
        "reject_sidechain_backbone_hybrids": False,
        "allowed_atomic_symbols": None,
        "forbidden_tokens": None,
        "forbidden_token_ids": [],
    }
    values.update(overrides)
    return ResolvedClassSamplingPrior(**values)


def _prompt_df(n_rows=4):
    return pd.DataFrame(
        {
            "target_row_id": range(n_rows),
            "target_row_key": [f"target_{idx}" for idx in range(n_rows)],
            "c_target": ["polyamide"] * n_rows,
            "temperature": [293.15] * n_rows,
            "phi": [0.2] * n_rows,
            "chi_target": [0.1] * n_rows,
            "property_rule": ["upper_bound"] * n_rows,
        }
    )


def _stub_sampler(monkeypatch, batches):
    calls = {"n": 0}

    monkeypatch.setattr(
        rl_trainer,
        "_prompt_df_to_condition_tensor",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        rl_trainer,
        "_create_trajectory_sampler",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        rl_trainer,
        "select_sampling_trajectory_rows",
        lambda rows, keep_indices: [rows[int(idx)] for idx in keep_indices],
    )

    def fake_sample_trajectories_with_class_prior(**kwargs):
        batch_idx = calls["n"]
        calls["n"] += 1
        smiles = list(batches[batch_idx]) if batch_idx < len(batches) else []
        trajectories = [f"trajectory_{batch_idx}_{idx}" for idx in range(len(smiles))]
        return smiles, trajectories, {}

    monkeypatch.setattr(
        rl_trainer,
        "sample_trajectories_with_class_prior",
        fake_sample_trajectories_with_class_prior,
    )
    return calls


def test_on_policy_sampler_returns_nonzero_partial_below_min_train(monkeypatch):
    calls = _stub_sampler(monkeypatch, batches=[["C"], ["CC"]])

    accepted_prompt_df, smiles, trajectories, metadata = rl_trainer._sample_on_policy_rollouts(
        prompt_df=_prompt_df(4),
        policy_model=object(),
        tokenizer=object(),
        resolved=object(),
        prior=_prior(class_match_sampling_attempts_max=2),
        scaler=object(),
        cfg_scale=1.0,
        num_steps=1,
        device="cpu",
        min_train_fill_ratio=0.75,
        partial_stop_attempt_fraction=1.0,
    )

    assert calls["n"] == 2
    assert len(accepted_prompt_df) == 2
    assert smiles == ["C", "CC"]
    assert trajectories == ["trajectory_0_0", "trajectory_1_0"]
    assert metadata["quota_satisfied"] is False
    assert metadata["quota_shortfall_below_min_train"] is True
    assert metadata["quota_shortfall_trainable"] is False
    assert metadata["stopped_at_quota_exhaustion_partial"] is True
    assert metadata["quota_exhaustion_reason"] == "max_attempts_partial_below_min_train_fill"


def test_on_policy_sampler_still_fails_when_quota_has_no_accepted_samples(monkeypatch):
    _stub_sampler(monkeypatch, batches=[[]])

    with pytest.raises(RuntimeError, match="accepted=0 requested=4"):
        rl_trainer._sample_on_policy_rollouts(
            prompt_df=_prompt_df(4),
            policy_model=object(),
            tokenizer=object(),
            resolved=object(),
            prior=_prior(class_match_sampling_attempts_max=1),
            scaler=object(),
            cfg_scale=1.0,
            num_steps=1,
            device="cpu",
            min_train_fill_ratio=0.50,
            partial_stop_attempt_fraction=1.0,
        )
