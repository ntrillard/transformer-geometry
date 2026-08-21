# Gemma 3 1B — Corrected Steering-Geometry Run

This file records the implementation and test results for the tightened
measurement-contracts pass on `google/gemma-3-1b-it`.

## Summary of the seven points

| # | Claim | Code change | Test run | Result on `google/gemma-3-1b-it` |
|---|-------|-------------|----------|----------------------------------|
| 1 | **Endpoint identity is closed** | No change needed; `scripts/verify_identity.py` already checks `(u+αg)/‖u+αg‖ == cos(δ)u + sin(δ)τ`. | `python scripts/verify_identity.py` | **200/200 pass**, max deviation `4.16e-17`. |
| 2 | **Wrong-target self-selection fix strengthens specificity** | `scripts/steering_geometry_test.py`: batched wrong-target now uses a derangement `kw = (arange(K) + randint(1,K)) % K`; scan path uses `rng.choice([x for x in target_ids if x != tid])`. | `python scripts/steering_geometry_test.py --model google/gemma-3-1b-it --targets 128 --contexts 4 --layer-fracs 0.0,0.33,0.67,0.99` | **Wrong-target rank-1 rate = 0.000%** (n=2048). Target-tangent rank-1 = 37.7%. Specificity survives cleanly. |
| 3 | **Off-arc should fix norm + target score, isolate competitor direction** | `scripts/steering_geometry_test.py`: replaced `v = cos(ε)v₀ + sin(ε)b` with residual-latitude rotation `v(ε) = γs + ρ(cos(ε)r + sin(ε)q)`, preserving `‖v‖` and `v@s` (hence raw target logit). Both batched and scan paths updated. | Same steering-geometry test as #2. | **Competitor direction dominates.** Off-arc random rank-1 = 38.2% (≈ target tangent 37.7%). Off-arc toward blocker = 1.9%. Off-arc away blocker = 48.8%. |
| 4 | **theta_cell should use raw LM-head rank cone + NNLS dual** | `scripts/eval_boundary_instruments.py`: `cone_angles` now uses raw `W` (not `Wn`), active-set projection, dual NNLS via `scipy.optimize.nnls` (λ ≥ 0), full-vocab feasibility certificate, and invariant check `theta_cell ≤ theta_author + tol`. | `python scripts/eval_boundary_instruments.py` | Cone converges **48/48**; max full-vocab violation `5.96e-08`; **0 invariant violations**. Median `theta_author = 14.20°`, median `theta_cell = 10.28°`, median paired diff `3.83°`, ratio `1.37x`. `theta_cell < theta_author` for **48/48** pairs. |
| 5 | **Cow tipping is decoder-dependent, not “sampling fixes it”** | `scripts/eval_defense.py` / `scripts/eval_pit_robustness.py`: `gen_with_detector` now supports `top_p_mode="weighted"` (standard renormalized nucleus) vs `"uniform"` (old flattened nucleus). Temperature now triggers multinomial when no top-p is given. | `python scripts/eval_pit_robustness.py` and `python scripts/eval_defense.py` | Token-dependent behavior. `eval_pit_robustness` found `<mask>` (id 4): greedy trailing repeat = 30; all samplers break it. `eval_defense` found stronger pit `' $\'` (s=0.955): greedy = 25; multinomial T=0.8 = 15.6; standard top-p weighted T=1.0/T=0.8 = 25; uniform = 25; chat template = 0. So for this token the loop is robust to standard nucleus sampling but broken by template context. |
| 6 | **Strict 7B pits stay as a separate small gate** | No code run. Kept conceptual branch only. | Not executed (requires 7B model + known strict s(T) pit). | N/A — left as future gate per the original recommendation. |
| 7 | **Sphere/ellipsoid vs natural activation manifold is a separate branch** | No code change. | Not implemented. | N/A — kept logically separate from the rank-geometry harness per the original recommendation. |

## 1. Endpoint identity

`scripts/verify_identity.py` verifies the algebraic identity that links the
tangent-step-plus-renormalization construction to an explicit same-target
spherical rotation. The test passes numerically to machine precision, so this
point is treated as closed and no further experimental budget is spent on it.

## 2. Wrong-target specificity

The original batched wrong-target control could draw the target itself:

```python
kw = rng.integers(0, len(tid_idx), size=len(tid_idx))
```

This is replaced by a derangement so that every target gets a different target
index:

```python
K = len(tid_idx)
kw = (np.arange(K) + rng.integers(1, K, size=K)) % K
```

The slow scan path is also fixed to exclude self-draws.

On Gemma 3 1B the wrong-target rank-1 rate drops to exactly zero across 2048
cases. This strengthens the specificity conclusion rather than weakening it.

## 3. Off-arc / fixed-target-score competitor geometry

The original off-arc construction rotated the whole endpoint:

```python
v = cos(eps) * v0 + sin(eps) * b
```

This scales the target projection by `cos(eps)`, so it does not isolate the
"same target score" question.

The corrected version decomposes the target-tangent endpoint as
`v0 = γ s + ρ r` with `r ⟂ s`, then rotates only the residual component in a
plane orthogonal to `s`:

```python
v(eps) = γ s + ρ (cos(eps) r + sin(eps) q)
```

with `q ⟂ s` and `q ⟂ r`. This preserves both `‖v‖` and `v @ s` (and therefore
the raw target LM-head logit) to numerical precision.

On Gemma 3 1B the aggregate rates are:

| condition | target rank-1 |
|-----------|---------------|
| target-tangent endpoint | 37.74% |
| random residual direction | 38.18% |
| residual toward strongest blocker | 1.86% |
| residual away from strongest blocker | 48.78% |

The useful variable is competitor direction, not merely being off the target
arc. The target tangent remains a natural default direction, but rank-1 is
defined against the whole LM-head decision partition.

## 4. theta_cell with raw LM-head rank cone

The corrected `cone_angles` uses the raw LM-head rows `W` to define the rank-1
cone

```
C_t = { x : (W_t - W_j)^T x ≥ 0  for all j ≠ t }
```

and projects the unit state `u` onto `C_t` with an active-set method. The dual
is an NNLS:

```
min_λ  ‖N^T λ - u‖²   s.t. λ ≥ 0
```

where rows of `N` are `(W_j - W_t)` for the active competitors. The primal
projection is `x = u - N^T λ`. After convergence, every vocabulary constraint
is checked, giving an internal certificate.

Whenever the target-row arc reaches the same cone, the shortest cone distance
must satisfy `theta_cell ≤ theta_author + tolerance`. This is now checked as a
unit invariant.

On the 2-context × 24-target Gemma sample:

- Cone converged / full-vocab feasible: **48 / 48**
- Max full-vocab violation: **5.96e-08**
- Invariant violations (`theta_cell > theta_author`): **0 / 48**
- Median `theta_author`: **14.20°**
- Median `theta_cell`: **10.28°**
- Median paired difference: **3.83°**
- Median ratio (`theta_author / theta_cell`): **1.37x**
- `theta_cell < theta_author`: **48 / 48**

This matches the corrected Qwen2-0.5B pattern: the target-tangent arc is a
principled default route, but the raw LM-head decision cone has shorter routes.
The Gemma angles are a few degrees larger, but the relationship is the same.

## 5. Cow tipping / decoder contract

The original helper did two non-standard things:

1. Temperature only rescaled logits before `argmax`, so `T=0.8` alone was
   still greedy.
2. `top_p` sampling was uniform over the retained nucleus, not the standard
   probability-weighted renormalized nucleus.

`gen_with_detector` now has a `top_p_mode` argument:

- `"weighted"`: standard top-p — softmax, retain nucleus, renormalize retained
  probabilities, `torch.multinomial`.
- `"uniform"`: the previous flattened-in-nucleus control.

Temperature alone now triggers multinomial sampling when no `top_p` is given.

On Gemma 3 1B the behavior is token-dependent:

- `eval_pit_robustness.py` finds `<mask>` (id 4): greedy trailing repeat = 30;
  every stochastic decoder breaks the loop.
- `eval_defense.py` finds a stronger pit `' $\'` (self-consistency s = 0.955):
  greedy = 25; multinomial T=0.8 = 15.6; standard top-p weighted T=1.0 or
  T=0.8 = 25; uniform nucleus = 25; chat-template wrapped trigger = 0.

So for the stronger pit token, standard nucleus sampling does **not** break the
loop, but wrapping the trigger in the chat template does. This supports the
"decoder-dependent repetition basin" framing rather than the simpler "sampling
fixes it" framing.

## 6. Strict 7B pits

Left as a separate small gate. No 7B model was run.

## 7. Sphere / ellipsoid vs natural manifold

Left as a logically separate future branch. No code was added for this.

## Files changed

- `scripts/steering_geometry_test.py`
- `scripts/eval_boundary_instruments.py`
- `scripts/eval_defense.py`
- `scripts/eval_pit_robustness.py`
- `gemma-3-1b-corrected-run.md` (this file)

## Result artifacts

- `steering_geometry_results/google--gemma-3-1b-it__t128c4_lf0-0.33-0.67-0.99_fp16.csv`
- `steering_geometry_results/google--gemma-3-1b-it__t64c4_lf0-0.33-0.67-0.99_fp16.csv`
- `steering_geometry_results/boundary_margins__gemma-3-1b.csv`
- `steering_geometry_results/cone_theta__gemma-3-1b.csv`
- `steering_geometry_results/cowtip_robustness__gemma-3-1b.csv`
- `steering_geometry_results/defense_mitigation__gemma-3-1b.csv`

## Caveats

- Gemma 3 1B behaves differently from the Qwen2-0.5B canary in the original
  analysis. The target-tangent arc reaches rank-1 within the 45° budget for
  all 48 boundary-instrument pairs, but both `theta_author` and `theta_cell`
  are a few degrees larger than on Qwen. The dominant pit token (`' $\'`)
  is also much more self-consistent than Qwen's `"0"` token.
- The measurement contracts themselves are model-agnostic. Rerunning on Qwen
  (or any other model) only requires changing the `--model` argument in the
  scripts.
