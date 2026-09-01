# Appendix: "Meaning lives in the orthogonal complement" — a staged falsification

**Status: CLOSED (negative, Type B).** Do not spend more compute on this line.

**Question.** Is the semantic concept steerable through the component of the
hidden state orthogonal to the token-row shell — i.e. can we inject a single
hidden-state direction `d_per` (the residual of a thematic contrast after
removing its projection onto the rank-`r` token-shell subspace) and read it
back as *coherent semantic transport* in generated text?

**Geometry setup.** Token rows of the LM head `W` are normalized, PCA'd
(`svd_lowrank`, q=300). The shell rank is `r≈231` (90% energy). The
complement direction is

```
d_per = d - Uc (Ucᵀ d),   d = mean_state(target) - mean_state(neutral)
```

projected into the full residual space, with `shell-leak = ‖Uc(Ucᵀ d_per)‖ = 0.0000`.

**Controls** (all in the *correct* geometry, per prior review): random
control is a rotated random vector **projected into the complement**
(`shell-leak=0.0000`), not a full Haar draw (which leaks); `-d_per` sign
control at every α; class-defining vocabets (ANCHORS / ENUM_CLUSTER /
HELD_OUT); string-level HELD_OUT for text, token-ID logits for the per-α
diagnostic; a-priori coherence criterion (max_run<6, distinct-1>0.6).

---

## Stage 1 — cheap mechanistic logit scan (8 s, no sampling)

For a batch of fixed prefixes, at each dose, measure `ΔL_held`, `ΔL_anchor`,
`ΔL_neutral` on the **readout** for `+d_per`, random-complement, `-d_per`
vs the natural baseline.

```
α     +dper held   rand held   -dper held   held−neutral(+dper)
.04   +0.40        -0.02        -0.40         +0.445
.10   +1.00         0.00        -1.00         +1.112
.16   +1.60        -0.07        -1.60         +1.778
.22   +2.20        -0.10        -2.20         +2.445
.28   +2.80        -0.13        -2.80         +3.111
.34   +3.40        -0.16        -3.40         +3.778
.40   +4.00        -0.19        -4.00         +4.444
```

- `ΔL_held(+d_per) = 10.0·α` — perfectly linear, r=10.0.
- **Random ≈ 0** at every dose (≤0.19, no shell leak). **Sign control is an
  exact mirror.** → The effect is genuinely carried by `d_per`, not by
  generic perturbation.
- But `ΔL_anchor = 1.1 × ΔL_held` at every dose (e.g. +4.40 anchor vs +4.00
  held at α=0.40). **It is a royal-region readout direction, not a
  held-out-selective one.**

## Stage 2 — gradient screen, K=4 at the promising doses

```
α     +dper  supA   rand   -dper   dLogitH+  dLogitB+  dLogitN+  minHrank  topPentry
.28   1/4    1/4    0/4    0/4     +328      +381      -38        1        0.32
.34   0/4    1/4    0/4    0/4     +382      +444      -44        1        0.29
.40   0/4    0/4    0/4    0/4     +426      +495      -49        4        0.10
```

Per-step rank / nucleus-entry tracking (the decisive metric) shows:

- **`minHrank=1`** at α=.28/.34 — a HELD_OUT word (crown/reign/kingdom/…)
  reaches **rank 1** in sampling logits at some step. Not "rank 400".
- **`topPentry≈0.30`** — a HELD_OUT word is inside the top-p nucleus ~30% of
  steps, while random ≈ 0 (its `dLogitH+` deltas are *negative*, −30…−43).
- **Yet transport ≤ 1/4** across all doses; isolated single-word hits
  (matching baseline noise). Pre-registered falsification criterion
  (`exists α: R/4 ≥ .5 & R>Q & M<R`) fails everywhere → **NO-TRANSPORT**.

## The evidence hierarchy (what is established)

```
complement direction → logit movement → top-p access ⇏ coherent semantic transport
```

Multi-modal support:

| Observation | Reading |
|---|---|
| random-complement ≈ no held-out effect (0 leak) | not generic perturbation |
| `+d_per` / `-d_per` are near-exact mirrors | structured direction, not noise |
| HELD_OUT reaches rank 1 at α=.28/.34 | real readout influence |
| HELD_OUT enters the nucleus | causal access to selection mass |
| no coherent transport emerges | fails generative control |
| `ΔL_anchor > ΔL_held` at every dose | royal-region, not held-out-selective |
| raising α worsens HELD_OUT rank (→4) / topPentry (→.10) | anchor over-commitment |
| anchor suppression does not rescue transport | not an anchor-averaging artifact |

## Defensible statement (reviewer-approved wording)

> **Whatever information about the target concept is encoded in the
> orthogonal complement is not sufficient, in this intervention geometry, to
> produce reliable semantic transport.** The complement has causal access to
> the readout, but the accessible direction is not an independently
> controllable semantic variable.

What we deliberately do **not** claim: (a) "meaning does not live in the
complement" — the experiment cannot establish that; (b) "the complement is
merely noise" — the sign control and random-complement comparison
demonstrate structured, reproducible information.

## Conceptual payoff: representation ≠ readout ≠ control

The line separates three notions that are **not interchangeable**:

| quantity | complement direction `d_per` |
|---|---|
| representation | ✅ encoded (royal contrast present) |
| readout influence | ✅ sign-asymmetric, random-zero logit effect |
| generative semantic control | ❌ no coherent transport |

This motivates the pivot: *why does logit-space contrast produce semantic
transport when hidden-state directions into this complement do not?*

---

*Files:* `logit_scan.py` (Stage 1), `falsify_orth3.py` (Stage 2, incl. the
anchor-suppressed `supA` condition and per-step rank / nucleus tracking).
Superseded: `falsify_orth.py` (over-aggregated, contaminated random control),
`falsify_orth2.py` (slow, no KV cache).

---

# Part II — What transports: row(W) cannot represent the generative signal

## The pivot question (after the ortho-complement falsification)

The complement direction `d_per` fails to transport even with correct
controls. The *working* mechanism is **logit-space contrast** (per-sentence
z-scored next-token logit difference, top-200 masked, added each step).
The open question: **what distinguishes the working logit contrast from a
hidden-state contrast projected through the output map `W`?**

## 1. Pure vector geometry (fantasy task, Qwen2-1.5B)

```
cos(dL_z_full, W d_per)        = +0.854    (the DIFFUSE contrast == hidden dir)
cos(dL_static(top-200), Wd_per)= +0.111    (the SPIKES are ~orthogonal to it)
spearman(dL_z, Wd_per)         = +0.817
sign-agreement                 = 80.7%
top-k overlap (K=200)          = 76 / 200  (38%)
```

The top-k masking is **not a neutral sparsification**: it selects a logit
subspace nearly orthogonal to `W d_per`. The transport-carrying spikes live
in the 99% residual.

## 2. Causal decomposition (decomp_gen.py, fantasy, canonical config)

```
dL_static = dL_parallel + dL_perp      (dL_parallel = proj onto Wd_per)

||dL_parallel|| ≈ 11.4   (~11% of dL)
||dL_perp||     ≈ 101.5  (~99%)
||dL_full||     ≈ 102.1

             transport   minHrank
dL_parallel   0/6         216    (hidden-representable -> does NOT transport)
dL_perp       3/6           0    (readout residual      -> does transport)
dL_full       2/6           0    (= the residual, ~99%)
random(matched norm) 0/6   260    (not magnitude)
```

- `dL_perp` matches/exceeds `dL_full` (3/6 vs 2/6, with better coherence:
  maxrun 1.7 vs 14.5). Phrased carefully: the residual matched or exceeded
  the full intervention in this 6-seed test with better repetition metrics.
- `dL_parallel ≈ random` (0/6, held-out never rank<216). The hidden-reachable
  slice is generatively inert.
- The small parallel component appears to *hurt* coherence when added back
  (dL_full maxrun jumps to 14.5) — suspected generic repetition attractor,
  not semantic.

## 3. Subspace hierarchy (subspace_hierarchy.py) — the sharp boundary

Project `dL_static` onto progressively richer hidden-reachable subspaces
(all **rescaled to the reference norm** so dose is identical):

| subspace | dim | transport | cos(proj, dL_full) |
|---|---|---|---|
| L0 `W d_per` | 1 | 0/6 | +0.113 |
| L1 shell image `W(Uc)` | 231 | 1/6 | +0.097 |
| L2 full `row(W)` | 1536 | 0/6 | +0.195 |
| L3 full logit space (= ref) | 152k | **2/6** | +1.000 |

**Interpretation (outcome A, sharp representability boundary):**
even the *full* rank-1536 hidden-to-logit image projects onto the success
vector with cosine only **+0.20, and does not transport**. This is not a
direction-selection issue — the generatively useful signal is essentially
**orthogonal to all of row(W)**. The success vector lives in the residual
`row(W)⊥` relative to the model's linear readout.

## 4. Defensible claim (reviewer-approved wording)

> The semantic transport signal is present in logit space, but the tested
> hidden-state direction — indeed **any** hidden-state direction, i.e. the
> whole linear image `row(W)` — cannot reproduce the portion of that signal
> responsible for generation.

Or: **the generatively effective operation exploits structure created by the
nonlinear/selective post-readout construction** (per-sentence z-normalization
+ top-k mask on the *output distributions*), not a hidden-state direction.
This is the "exploits post-readout transformation" statement.

Caveats kept explicit: the residual may reflect token-specific
nonlinear/contextual effects, information unavailable to averaged hidden
states, the z-normalization procedure, higher-order interactions, and/or the
specific TGT/NEU construction. We do **not** claim a general "semantic
residual"; we claim the narrower result above for these experiments.

*Files:* `vec_compare.py` (geometry), `decomp_gen.py` (causal
decomposition), `subspace_hierarchy.py` (representability boundary).
