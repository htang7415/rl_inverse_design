# Water-Soluble Polymer Discovery

This project studies how generative models can design water-soluble polymers
instead of only screening known structures. The goal is to generate polymer
repeat units that are chemically valid, novel, compatible with a target polymer
family, and likely to mix favorably with water under a specified condition.

## Motivation

Useful water-soluble polymers must satisfy several constraints at once:
valid chemistry, feasible synthesis, family consistency, novelty, and favorable
polymer-water interaction behavior. A model that only imitates known polymers
may generate realistic structures but still miss the desired physical target.
This project treats polymer generation as a goal-directed inverse-design
problem.

## Method

The approach combines:

- masked diffusion generation for polymer repeat units
- physical property modeling for polymer-water interaction behavior
- conditional generation for target-aware design
- reward and preference alignment for improving generated candidates

The generator first learns polymer syntax, then learns to respond to target
conditions, and finally is aligned toward candidates that pass chemistry,
property, novelty, and feasibility checks.

## Theory And Math

The key physical signal is the Flory-Huggins interaction parameter `chi`, which
describes polymer-water mixing behavior:

```text
chi = chi(T, phi; p)
```

where `p` is the polymer, `T` is temperature, and `phi` is polymer volume
fraction. Lower `chi` generally means more favorable mixing.

The modeled interaction form is:

```text
chi(T, phi; p)
= (a0 + a1 / T + a2 log(T) + a3 T)
  (1 + b1(1 - phi) + b2(1 - phi)^2)
```

The inverse-design objective is:

```text
x* ~ p_theta(x | c)
```

where `x` is a generated polymer candidate and `c` is the design request, such
as polymer family, temperature, composition, and desired `chi` behavior.

The project connects imitation learning:

```text
maximize log p_theta(x_known)
```

with reward-driven design:

```text
maximize E[R(x, c)]
```

where `R` rewards validity, novelty, family match, solubility, target-property
success, and feasibility.

## Data And Architecture

The model uses a large unlabeled polymer collection to learn polymer syntax and
a smaller labeled set to learn water-mixing and miscibility behavior.

The core architecture is a masked diffusion transformer. It denoises corrupted
polymer sequences instead of generating strictly left to right, which helps it
capture long-range structural consistency, attachment-site balance, and
family-specific motifs.

## How To Use

Install the project in your environment:

```bash
pip install -e .
```

Prepare data and run the core training/evaluation workflow:

```bash
bash scripts/run_step0.sh
bash scripts/run_steps1_4.sh small
```

Run a short smoke test for inverse design:

```bash
bash scripts/run_step5_smoke.sh small
```

Run inverse design for one polymer family:

```bash
bash scripts/run_step5.sh small polyimide
bash scripts/run_step5_1.sh small polyimide
```

Run the hyperparameter-search workflow:

```bash
bash scripts/run_step5_hpo_and_5_1.sh small polyimide
```

Common polymer-family targets include:

```text
polyimide, polyamide, polyester, polyether, polyacrylate, polystyrene, polysulfone
```

Run the focused unit test:

```bash
pytest tests/test_step5_rl_rollout_quota.py
```

## What The Results Show

The experiments compare raw generation, guided generation, conditional
generation, and aligned generation. The key question is whether each added form
of feedback improves target satisfaction without destroying validity, novelty,
or feasibility.

A strong result is not just a high predicted property score. It is a candidate
set that remains chemically plausible, novel, target-family consistent,
physically aligned with the requested condition, and practical enough for
further scientific evaluation.

## Conclusion

This project frames polymer discovery as physically constrained inverse design.
It shows how a diffusion generator can move from learning what polymers look
like toward proposing candidates for a requested behavior. The broader goal is
to study generative models as controllable design policies for scientific
discovery.
