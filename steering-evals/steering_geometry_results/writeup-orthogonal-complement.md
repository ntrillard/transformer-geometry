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

---

# Part III — Mechanism matrix: which operation is causal?

## Question

The decomposition showed the transport signal is ~99% outside `row(W)`. But
is that *escape* itself the mechanism, or is it a side effect of something
else? Separate the candidate operations with a factorial:

- **Normalization**: raw / centered / z-scored / per-sentence z-sum
- **Mask**: none / top-25 / top-50 / top-200 (positive top-k selection)
- **Controls** (all at top-200, norm-matched to the working reference):
  random coords (own values), random coords (sorted top-200 magnitudes),
  real top-200 coords with permuted values, real coords with equal weight,
  and the full `row(W)` projection of the working vector.

Every condition is **rescaled to the same effective logit norm** (`N_REF`),
so dose cannot explain any difference.

## Phase A — row-space fraction vs K

```
K         R_row(perz)      residual (1-R)
1         0.011            98.9%
10        0.015            98.5%
25        0.018            98.2%
50        0.022            97.8%
100       0.027            97.3%
200       0.038            96.2%
500       0.065            93.5%
1000      0.097            90.3%
5000      0.252            74.8%
V (full)  0.999             0.1%
```

Masking *creates* the row-space escape: the top-200 vector is **96% outside
row(W)**; the unmasked vector is **99.9% inside**. `zs` and `perz` give
nearly identical fractions (normalization barely moves geometry).

## Phase B — the causal table (SEEDS=6, fantasy)

```
cond       transport  dLogP_H  dLogP_U  maxrun  dist1  R_row
raw        0/6   +0.44  -0.58    0.0  0.68  1.000
raw_t25    0/6   -7.53  -7.19   33.0  0.67  0.018
raw_t50    0/6   -7.57  -7.59   60.2  0.43  0.022
raw_t200   2/6   -2.03  -2.42   14.5  0.56  0.038   ← transports
cent_t200  1/6   -1.74  -2.38   14.8  0.47  0.038
zs_t200    1/6   -1.74  -2.38   14.8  0.47  0.038
perz_t200  2/6   -2.00  -2.49   14.5  0.55  0.038   ← transports
rand200    0/6   -5.76  -6.84   83.3  0.25  0.012   ← degenerate
magmatch200 0/6  -3.68  -3.81   52.7  0.35  0.011   ← degenerate
shuffle200 0/6   -1.92  -2.58   17.7  0.44  0.037   ← dead
equal200  0/6    -1.90  -2.54   25.7  0.43  0.038   ← dead
rowW_proj 0/6    +0.79  -0.29    0.0  0.62  1.000   ← dead (back in rowW)
```

## What is causal

1. **Normalization is disposable.** raw/cent/zs/perz at top-200 are all
   ~1-2/6. The working "per-sentence z" is not the special ingredient; the
   raw contrast works equally once top-k-selected.

2. **Top-200 *coordinate* selection is necessary.** t25/t50 are dead and
   degenerate (maxrun 33-60 repetition); no-mask (full row-space vector) is
   dead. Only K≈200 carries the winner set the counterfactual needs.

3. **Row-space escape is necessary but NOT sufficient.** All top-200
   conditions sit ~96% outside row(W) — including the four dead controls.
   Escape alone explains nothing.

4. **The missing ingredient is the *coordinated* object: correct
   vocabulary coordinates × ranked magnitudes × out-of-rowW.**
   - `rand200`/`magmatch200` (wrong coords): dead + degenerate.
   - `shuffle200` (right coords, permuted values): dead.
   - `equal200` (right coords, equal weight): dead (no ranking gradient).
   - `raw_t200`/`perz_t200` (right coords, ranked magnitudes): transport.

5. **`rowW_proj` is the clean negative control**: projecting the working
   vector into `row(W)` kills transport (0/6) while keeping maxrun 0 — the
   signal itself is inert inside the hidden-reachable subspace, active
   outside it.

## Mechanism statement (reviewer-approved)

> The generatively effective operation is: take the vocabulary readout
> contrast, **select its top-200 positive coordinates with their ranked
> magnitudes, and apply the resulting sparse vector as a logit offset**.
> This vector necessarily escapes `row(W)` (masking breaks representability),
> and it is that *coordinated* sparse object — not any individual axis
> (normalization, sparsity, magnitudes, row-escape) — that transports.
> Random/equal/permuted/sparse/row-constrained variants all fail.

*File:* `mechanism_matrix.py`.

---

## Part IV — K × λ causal surface: sparsity and row-space interact

The mechanism statement isolates a *coordinated* object, but does not yet
separate its two axes. We cleanly manipulate both while holding dose fixed:

**Design.** For each K, `d_K = topk_pos(perz, K)` defines the sparse
coordinate set. We decompose `d_K = P_row d_K + P_\perp d_K` and set
`vec(K, λ) = (1-λ)·(d_K − P_row d_K) + λ·P_row d_K`, renormalized to the
canonical working norm `N_REF = 102.076` in every cell (fixed dose).
`λ=0` is the pure out-of-row residual; `λ=1` the pure row(W) projection.

**Compact diagnostic** (SEEDS=4, NTOK=60):

```
K     λ=0       λ=0.5     λ=1
100   0/4       0/4       0/4
200   2/4       1/4       0/4      <- localized transport
500   0/4       0/4       1/4
```

The effect is a **localized island around K=200**, not a generic function
of row-space distance. At K=200, cos(dK, dL_ref) ≈ 0.98-1.0 and replacing
the residual by its row-space projection monotonically suppresses
transport (2/4 → 1/4 → 0/4).

**Full replication** (SEEDS=30, NTOK=120), K=200 only:

```
λ       transport   medMinR   R_row   cos_ref
0.00    4/30        2         0.000   +0.981
0.25    4/30        1         0.004   +0.992
0.50    3/30        0         0.038   +1.000
0.75    6/30        0         0.263   +0.942
1.00    0/30        23        1.000   +0.195
```

**Causal reading (careful, not overclaimed).** The clean claim is *not* a
smooth monotonic dose-response — λ=0.75 peaking at 6/30 is within binomial
noise (n=30). The causal variable is **existence vs. absence of the
out-of-row component**: **λ=1, the pure row-space projection, is the unique
collapsed cell** (0/30, medMinR→23, cos_ref→+0.195), while every λ<1 that
retains a *nonzero* residual transports at ~3-6/30 with medMinR≈0 and
cos_ref≈0.94-1.0.

This is a **behavioral discontinuity at the row-space boundary**, not a
continuous dose-response and not a nonlinearity in the interpolation
(the λ mixture itself is linear; only the transport behavior jumps at
λ=1). Projecting the sparse chosen vector back into `row(W)` (the only
`λ=1` manipulation) crosses out of the reachable set and destroys the
effect; any retained residual fraction preserves it, at a roughly flat
3-6/30 plateau. This is itself a negative control: an *amount* of
out-of-row escape does not predict efficacy — either the coordinate/
ranking pattern escapes `row(W)` (works) or it is pulled fully inside
(fails).

**Net mechanism (updated).**

> The generatively effective object is a sparse (K≈200), positively-masked,
> ranked-magnitude vocabulary contrast vector, applied as a logit offset.
> Its coordinate/ranking pattern is the semantic carrier; escaping `row(W)`
> (so it is not cancelled by the hidden-reachable subspace) is *necessary*;
> but the *amount* of escape, beyond merely nonzero, does not scale efficacy.

**Net mechanism (updated).**

> The generatively effective object is a sparse (K≈200), positively-masked,
> ranked-magnitude vocabulary contrast vector, applied as a logit offset.
> Its coordinate/ranking pattern is the semantic carrier; escaping `row(W)`
> (so it is not cancelled by the hidden-reachable subspace) is *necessary*;
> but the *amount* of escape, beyond merely nonzero, does not scale efficacy.

---

## Part V — Semantic vs. lexical (`neighbor_probe.py`, SEEDS=30)

The K×λ surface and mechanism matrix established *what* the intervention
does. This section asks **whether it transports a concept or merely forces
lexical coordinates**. We hold out three probe classes, all filtered so
their tokens are **NOT** among the boosted top-200 coordinates (zero direct
additive boost):

```
class   UNSTEERED          STEERED
lex     0/30  rank 45      27/30  rank 0      <- boosted coords: huge lift
sem     0/30  rank 188     0/30  rank 172     <- unboosted semantic neigh: flat
unr     18/30 rank 0       16/30 rank 1       <- unboosted unrelated: flat
```

- Positive control (LEX = tokens decoded from the actual boosted top-200):
  emission 0/30 → 27/30, best rank 45 → 0. The harness lifts precisely what
  it directly boosts.
- Semantic neighbors outside the mask (fiend/wraith/ghoul/apparition/
  sorcery/incantation/tyranny, all single-token, all NOT in top-200):
  **0/30 → 0/30, best rank 188 → 172.** No generalization to unboosted
  concept coordinates.
- Unrelated frequency-matched control: unchanged (18/30 → 16/30).

**Net claim (honest).** The technique is *sparse ranked **lexical** logit
steering*: it selectively raises a coordinated set of vocabulary coordinates
and generation enters that lexical region. It does **not** demonstrably
transport a *concept* to unboosted coordinates. This is a cleaner, more
defensible mechanism than semantic transport — and the honest way to describe
the method to a reviewer.

*Files:* `mechanism_matrix.py` (K×λ surface), `neighbor_probe.py` (lexical-vs-semantic).
