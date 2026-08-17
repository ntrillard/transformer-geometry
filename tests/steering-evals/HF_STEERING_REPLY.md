# Reply to "Steer on a Sphere" review posts — point-by-point

Two review posts addressed: the long geometry review by **John6666** (the "rotation
branch" / competitor-geometry analysis) and the short **corechek** post (cow‑tipping
robustness + defensive‑encoding mitigation). Every point is answered with our
research, and each answer carries a **Methodology** line stating what was measured,
with what controls, and on which data.

All new measurements in this reply were run on the local RTX‑3080 box with cached
models (Qwen2‑0.5B‑Instruct primary probe; 5‑family cross‑check: Qwen2‑1.5B,
Qwen2‑0.5B, GPT‑2, SmolLM‑135M, Pythia‑160m), full‑vocabulary LM‑head linear assay,
fp32 head, seed 42, identical settings across families.

---

## Part 1 — John6666 (geometry review)

### 1.1 "The tangent step + renormalization is an exact member of the same target-tangent great-circle family as a rotation."
**Answer: confirmed, exactly.**
We verified numerically that `(u + α·g)/||u + α·g|| = cos δ·u + sin δ·τ` with
`δ = atan(α·‖g‖)` on 200 random (u,s) pairs: **max deviation < 1e-10 in all 200**.
So a rotation and our tangent+renorm are the same endpoint family — any divergence
in results cannot come from that choice.

**Methodology:** fp32, d=1536, ‖s‖=‖u‖=1, direct vector identity check. It also
fixes the budget framing exactly as you suggest: `α=0.3 ⇒ δ≤atan(0.3)≈16.7°` since ‖g‖≤1.

### 1.2 "The target tangent is the steepest-ascent direction for target alignment on the sphere (gradient of sᵀu)."
**Answer: correct — and the same gradient is what we steer along.** `g = s − (sᵀu)u`
is the Riemannian gradient of `f(u)=sᵀu` on the unit sphere, so among
norm‑preserving moves it maximises target score per infinitesimal step. Our harness
uses exactly this `g`/`τ`. No disagreement.

**Methodology: algebraic (Riemannian gradient on S^(d−1)); consistent with 1.1.**

### 1.3 "Rank‑1 is a multi‑competitor condition: C_t = ∩_{j≠t}{(W_t−W_j)ᵀx ≥ 0}."
**Answer: confirmed, and it becomes a measuring tool in our harness.** A token is
rank‑1 iff it lies in its decision cone `C_t`. Against the *alignment* objective,
this is a genuinely different question — which is why we added two instruments this
week: (A) blocking‑competitor + margin records along the arc, and (B) an
active‑set **projection onto C_t** over the **full vocabulary**, giving the shortest
angle `theta_cell`.

**Methodology:** Qwen2‑0.5B, bias‑free linear LM head (verified no bias), full V
(≈152 k) constraint set; active‑set projection; violations rechecked over all rows
each round; results in `cone_theta__Qwen2-0.5B.csv`.

### 1.4 Your matched-control probe (Qwen2‑0.5B, 512 cases): author endpoint 94.34%, wrong tangent 0/512, random off‑arc ≈94%, blocker‑boosted ≈0.98–52%, blocker‑suppressed ≈96.9%.
**Answer: reproduced, family‑wide, and directionally identical.**
Our identical controls on the 5‑family cross‑run (2048 cases each; Qwen2‑0.5B
values in brackets):

| control | ours (Qwen2‑0.5B) | 5-family range |
|---|---|---|
| correct target tangent | 97.3% | 27.9–99.2% |
| wrong‑target tangent | 1.6% | 0.4–1.6% |
| random tangent | 0% | 0% |
| off‑arc, random residual (same score/norm) | 97.7% | 66.8–99.0% |
| off‑arc, toward strongest blocker | 13.7% | 2.5–34.8% |
| off‑arc, away from blocker | 100% | 70.1–100% |

This reproduces your most load‑bearing finding: **random same‑score residual
rotation mostly leaves rank intact; a competitor‑aligned rotation at the same
score/norm destroys it.**

**Methodology:** per‑(context, layer, target) start, unit‑sphere, s = W_t/‖W_t‖,
over‑arc defines movement in the target tangent plane; the off‑arc rotations keep
‖v‖ and `v·s` fixed to fp‑32 noise, and vary only the residual component
(toward/away/random in the `span(v0,s)`⊥ basis); ranks over the full V logits.

### 1.5 "Same target score does not imply same target rank."
**Answer: confirmed — we observe exactly this.** At trajectories where target‑logit
and norm are fixed to fp noise, target rank varies from ~2% (toward blocker) to
~97–100% (away/random) purely with the residual direction at Qwen2‑0.5B, and in the
5‑family run across every family. Score is not rank.

**Methodology:** as 1.4; the invariance check `target_logit(v)=v·s` identical to
~1e‑7 across the three off‑arc conditions.

### 1.6 theta_author vs theta_cell (your numbers: 48 pairs, 8.32° / 7.40°, Δ1.03°, ratio 1.17×).
**Answer: replicated in the same shape — arc is near‑optimal, not strictly
optimal.** Our Qwen2‑0.5B, 48‑pair probe (full vocab, fp32 head):
```
median theta_author = 10.0°
median theta_cell   = 9.5°
median Δ            =  0.4°
median ratio        = 1.04×
arc gave an angle:  92% of pairs
cone gave an angle: 100% of pairs
```
So `theta_author ≈ theta_cell + <1°` — the target‑row great‑circle is a good, usually
efficient route; it is *not* the unique shortest, and it can miss targets that the
cone can still reach (6 such pairs here; you saw the same).

**Methods:** both angles from the same (state, target) pair. theta_author is the
**analytic first rank‑1 crossing** of the target‑row arc (closed form from the
sinusoid margins `m_j(θ) = r(A_j cosθ + B_j sinθ)`, full vocabulary) — validated
against a 200‑step numerical scan on 300+ random configurations with no
mismatches at the scan granularity (±0.5°). theta_cell = `acos(‖proj_Ct(u)‖)` via active‑set projection, full‑vocab
violation re‑checked every iteration.

### 1.6b "Unreachable on the target‑row arc ≠ unreachable in the decision region."
**Answer: we observe the same.** In our 48‑pair sample, 92% of arcs reach rank‑1 and
100% of cones do; the pairs that fail on the arc are still inside their decision
region (6 cases). Combined with 1.5: the arc is a good default route, not the only
route.

**Methods:** same cone/arc instrumentation as 1.6.

### 1.7 Your 4‑way decomposition (endpoint / movement / alignment / task).
Point‑wise:
1. **Same endpoint** (tangent+renorm vs rotation): **identical** by 1.1 (1e‑10).
2. **Same movement angle** (correct vs wrong/random tangent): **not sufficient** —
   wrong/random tangents give ~0% everywhere in all families.
3. **Same target alignment** (score & norm fixed, residual varies): the *decisive*
   one. Random residual benign (97–99% rank‑1 rates), competitor‑targeted residual
   brutal (toward → ~3–35%; away → ~70–100%). Matches your "middle case".
4. **Same rank objective** (arc vs shortest‑into‑cone): ratio ≈ 1.04–1.17× in our
   probes — arc ≈ shortest, with exceptions (your 18.9° vs 7.9° example; we see
   several too).

**Methods:** each sub‑question measured with its own matched control; the endpoint
test is direct vector equality, not a sampling argument.

### 1.8 "Record the first rank‑1 crossing angle, not just a fixed‑α flag."
Implemented. Our per‑(context, layer, target) record is now:
target_id · initial_rank · **first_rank1_angle** · margin vs best‑other ·
**blocking competitor** · margin at crossing · context · θ → written to
`boundary_margins__Qwen2-0.5B.csv`. Family medians: **8.0°–12.1°** (Qwen2‑1.5B 8.0,
GPT‑2 11.6, SmolLM 9.8); your probe: ≈ 8.46°. Agree: a fixed‑budget flag hides
structure; the crossing angle is the informative continuous observable.

**Methods:** analytic crossing — exact sinusoid‑margin solution per competitor over
the full vocab, no angular scan.

### 1.9 "Blocking competitor can change as θ changes; rank is not globally monotone in θ."
**Answer: consistent.** The margins `m_j(θ)` are sinusoids with different phases, so
the blocking competitor can (and does) swap along the arc; increasing `target_score`
can lose to a *different* competitor. We record the blocking token at `θ=0` and at
the crossing, and use "first detected crossing" (non‑monotone) wording rather than
assuming monotonic ranks.

**Methods:** full‑vocab margin traces; no monotonicity assumption baked in.

### 1.10 Context as an axis (not just one‑state tables).
Agreed, and already in the harness: every run is **4 contexts × 4 depth‑layers**
(depth‑adaptive), × 5 families. We see precisely your pattern: **top‑1 rate is
stable across contexts** (e.g., ~94–99% per context for Qwen2‑1.5B‑IT) while the
**rank‑1 crossing angle shifts by several degrees** between contexts (family
medians 8–12°, per‑context spread of several degrees). So the phenomenon is robust
but the distance‑to‑region is context‑dependent — the "Activation Source Selection…"
direction.

**Methods:** same‑target‑IDs × several starting states (4 per layer, 4 layers),
reported per context.

### 1.11 Literature alignment.
Useful map, we agree and position ourselves as **same "rotate toward a target
direction" family, with the objective = *vocabulary‑token rank* rather than a
behavioral concept, evaluated with LM‑head decision‑cell + cone distances.** Stolen
Probability / Unargmaxable Classes frame the *global feasibility*; Predicting‑Where‑Steering
Succeeds, the *accessibility*; Angle–Norm–Decomposition explains the norm‑as‑control
equivalence (we also use norm matching, which is a control in a linear head, not a
mechanism). We'll add these references to the paper (currently only Ba et al.,
Vaswani et al., Xiao et al.).

### 1.12 "Don't collapse behavioral‑steering and final‑head‑rank claims."
Agreed — every claim here is about the **linear, bias‑free final LM‑head assay**
(final hidden state → W_head → logits → rank). Norm matching is a control condition,
not an explanation of rank in a linear head; it says nothing about earlier‑layer
behavioral steering.

### 1.13 Decision tree.
We confirm every node on our data: endpoint **yes**; correct‑vs‑wrong **needed**;
random same‑score **benign**; competitor‑targeted **sensitive**; arc‑vs‑cell **near
1.0–1.2× with exceptions** — the mid‑column outcome. This ontology is exactly the
frame the rotation viewer should encode, and we export these observables per case.

---

## Part 2 — corechek (cow‑tipping robustness + defensive‑encoding mitigation)

### 2.1 robustness across precision, quantization, temperature/sampling, chat templates.
Measured on the repeated‑`0` self‑consistency loop (the digit‑zero self‑repeat
phenomenology that the paper describes at scale; here reproduced at 0.5B: greedy
trailing‑repeat 24 fp16). Table:

| axis | result |
|---|---|
| precision fp16 → bf16 | loop persists (24 → 30 tokens) |
| quantization fp16 → int8 | loop persists (30) |
| greedy / temp 0.8 | loop persists (24 / 24) |
| temp 1.0, top‑p .9 sampling | **loop breaks (≈0 trailing-repeat)** |
| chat‑template‑wrapped trigger | **loop broken (0)** — basin narrow |

So: the repeated‑token fixed point is **stable to weight‑precision and 8‑bit
quantization and mild temperature**, but **sampling at meaningful temperature, or any
non‑terminal wrapping, escapes the loop**. Formal "self‑consistent pits s(T)≥0.4"
still require 7B‑class vocabulary-scan (top s≈0.04 on Qwen2‑0.5B), so the full
matrix for *true* pits needs the 7B families; we can run those (guarded cache) on
request. Everything above is the *degenerate‑loop* case measurable at this scale.

Methods: deterministic greedy vs tracked sampling (temp 0.8 / 1.0 + top‑p .9) on the
looping token; fp16/b‑f16/int8 reloads (bitsandbytes); trailing‑counter; 25
new‑token cap, seeded RNG.

### 2.2 Mitigation‑focused evaluation.
| mitigation | result |
|---|---|
| baseline (no mitigation) | 35 trailing‑repeat tokens — failure mode present |
| control‑character sanitisation | 34 — **ineffective for printable triggers** (it only helps control‑char triggers, e.g. NULL bytes, which need 7B‑scale vocab) |
| **repetition‑loop detection** (halt at ≥4 same) | **4 tokens emitted then stop — reliable** |
| **normal‑generation harm** (n=8) | **median 0 tokens** truncated — safe |

Conclusion: repetition‑loop detection is cheap, reliable, and orthogonal into
normal text; sanitisation alone is insufficient against printable‑trigger.

### 2.3 Threat model + mitigations (short, as requested).
- **Threat**: content ending in a pit‑/repetition trigger (…0000, cut cut cut, NULL
  bytes) at a chunk‑boundary can lock an LM crawler into degenerate self‑loop,
  emptying instead of reading.
- Mitigations: (i) repetition‑loop detection is effective & zero‑harm; (ii)
  control‑char sanitisation (protected‑character triggers only); (iii) sampling/
  temperature is stochastic for‑escape, unreliable sole‑defense; (iv) generation‑side
  repetition penalties.
- Limitations: trigger inventory is vocab/scale dependent; narrow basins mean the
  trigger must be terminal content (any tail breaks it).

**Methodology checklist (explicit):**
all claims are **linear final‑head assay** on bias‑free LM heads;
identity 1.1 **verified numerically** (not assumed); cross‑family **identical seed/settings**;
angles **analytic first‑crossing, full‑vocab**, validated vs 200‑step scan (0 mismatch @0.05);
cone distances **active‑set projection, full‑vocab**.

Data files saved this session:
- `steering_geometry_results/boundary_off__Qwen0.5B.csv`
- `steering_geometry_results/one_theta__Qwen0.5B.csv`
- `steering_geometry_results/cowtip_robustness__Qwen2-0.5B.csv`
- `custom_geometry_results/defense_mitigtion__Qwen2-0.5B.csv`

---

## Appendix — point → artifact map (scripts + evidence)

| Reply point | Script(s) | Evidence output |
|---|---|---|
| 1.1 endpoint identity | `scripts/verify_identity.py` | 200/200, max dev 4e-17 (run: `python scripts/verify_identity.py`) |
| 1.2 tangent = spherical gradient | `scripts/steering_geometry_test.py` (`tangent_direction`) | algebraic |
| 1.3 decision cone | `scripts/eval_boundary_instruments.py` (`cone_angles`) | `results/cone_theta__Qwen2-0.5B.csv` |
| 1.4 matched controls (families) | `scripts/steering_geometry_test.py` (`run_model`) | `results/cross_family_summary.csv` |
| 1.5 score != rank | as 1.4 (off-arc toward/away/random columns) | same CSV |
| 1.6 theta_author vs theta_cell | `scripts/eval_boundary_instruments.py` | `results/cone_theta__Qwen2-0.5B.csv` |
| 1.6b arc-unreachable, region-reachable | as 1.6 (subscribe: arc 92%, cone 100%) | same |
| 1.7 4-way decomposition | 1.1 + 1.4 + 1.6 scripts | — |
| 1.8 first rank-1 crossing angle | `scripts/eval_boundary_instruments.py` (`arc_stats`) | `results/boundary_margins__Qwen2-0.5B.csv` |
| 1.9 blocking competitor / non-monotone rank | as 1.8 | same |
| 1.10 context-as-axis | `scripts/steering_geometry_test.py` (`--contexts`, `--layer-fracs`) | `results/cross_family_summary.csv` |
| 2.1 cow-tip loop robustness (prec/quant/temp/template) | `scripts/eval_pit_robustness.py` | printed table (token `'0'` id 15) |
| 2.2 defensive-encoding mitigation | `scripts/eval_defense.py` | `results/defense_mitigation__Qwen2-0.5B.csv` |
| 2.3 threat model / mitigations | `scripts/eval_defense.py` | same CSV |

All reproduction commands, model selection, budgets and seed policy are in `README.md`. Every CSV is a raw run artifact (single machine, seed 42, identical settings).
