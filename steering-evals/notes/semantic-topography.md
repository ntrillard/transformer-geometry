# Idea log — LM-head semantic topography (T1b)

**Status**: measured, results in `steering_geometry_results/`. Seed of the
chord/inversion steering family. Do NOT lose.

## The core find (T1b)

Local neighborhoods of the LM head's rows (the token "prototypes" on the state
sphere) are **semantically coherent** — the row space already contains a
semantic topographic map:

```
apple  ->  Apple / apple / Apple / 苹果 / apples / APPLE      (30-NN)
Paris  ->  Paris / 巴黎 / paris / France / French / London
king   ->  King / king / kings / KING / queen / 国王 / Queen
ocean  ->  Ocean / oceans / Ocean / 海洋 / sea / Sea / 海水
red    ->  Red / 红 / Red / red / -red / _red / RED / 红色
```

- Scale: nearest-neighbor rows sit ~48-70 deg apart (form/identity variants
  ~45 deg; same-class words ~75 deg; unrelated pairs ~80 deg).
- **Identity variants** (Apple/APPLE/apples): ~45 deg. **Cross-lingual**
  (apple <-> 苹果): present in neighborhoods (~5/30 of 30-NN are CJK). **Same
  class** (apple/banana/rice): ~75 deg, forming SEPARATED CAPS (intra med 75.0
  vs inter med 86.6 deg, ratio 0.866, +11.6 deg separation).
- Neighborhood purity: ~9/30 of the 30-NN of a seed word are same-word
  variants + CJK translations (30%); another chunk are semantically related.

## Why a global Kohonen lattice FAILS on this manifold (measured)

A 16x16 spherical batch SOM (256 neurons, Gaussian neighborhood, radius decay)
fit to 20k Qwen head rows:

- 214/256 neurons EMPTY; one neuron owns ~130k of 151,936 rows.
- quantization error +0.108 cosine ~= 84 deg (captures almost nothing).
- prototype spacing ~70 deg while the data's own 1-NN scale is ~48 deg
  (kernel window paradox: any radius below the 48 deg NN scale sees nothing,
  above it averages everything to the centroid; a data-space-kernel control
  also collapses to 1 cell / 100%).
- density profile (full vocab): 1-NN 48.0 / 5-NN 58.4 / 30-NN 62.8 / 100-NN
  63.1 deg — near-uniform high-dim packing, NO low-dim manifold to tile.

**Conclusion**: the useful topography is already IN the head rows (a
semantic map you get for free by cosine-KNN); a Kohonen fit merely re-derives
it worse. A 2-D grid prior is a topological mismatch for this sphere.

## Why it matters (the chord/inversion family this seeded)

- The semantic classes (food/animal/city/color/nature/number) are the
  "chords" (note stacks) of the steering reframing.
- Chord CENTER steering fails (spread threshold ~50 deg; corr(spread,reach)
  = -0.84); chord INVERSION steering (aim at the best-positioned family
  note from the current state) resolves the family cone at ~89.6% vs 22.9%.
- Generation-level "Chord-Inversion Steer": single-shot = free topical
  conditioning (100% first-token adherence, diversity 0.74 ~ baseline);
  cadence k>=3 = a topical dosage dial; persistent = pit.

## Scripts / data

- `eval_semantic_map.py`            (P1-P4: scales, purity, class caps, cross-model)
- `eval_som_failure.py`             (density profile + grid vs data kernel collapse)
- `eval_kohonen_sphere.py`          (T1 global SOM, T1b neighborhoods, T2/T3 row vs centroid)
- `eval_chord_steering.py`          + `chord_steering.csv`, `chord_interference.csv`
- `eval_chord_inversion.py`         + `chord_inversion.csv`
- `eval_topic_steering.py`          + `topic_steering.csv`  (generation primitive)
- `steering_geometry_results/`      all CSVs

## Cross-model separability ratio (intra/inter): Qwen 0.866, GPT-2 0.827,
Pythia 0.847, Gemma 0.815 — the semantic-cap structure is a universal head
property, not a Qwen quirk.

## EQ, the equator confirmation (MOST RECENT)

**Concise form**: the semantic topography lives on the EQUATOR — the
BOS-axis is a pure rotation axis, and the semantic map is a pure longitude
map.

Demonstrated on Qwen2-0.5B (position-0 hidden state of 'Once upon a time',
final layer = the BOS/latitude axis of the paper):

```
token-row angle to BOS axis: median 91.46 deg  (p10 89.2, p90 93.3)
-> full vocab within +-3 deg of perpendicular to BOS: EQUATORIAL

class caps (food/animal/color/city/nature/number):
  intra-med / inter-med (full space):      75.0 / 86.6 deg  (ratio 0.866)
  intra-med / inter-med (BOS-orthogonal):  75.0 / 86.6 deg  (ratio 0.866)
-> IDENTICAL after stripping the BOS axis: semantic structure is equatorial
```

Meaning:
- BOS = latitude = context/gating (the paper's BOS section, norm growth);
  tokens = longitude = content.  The equator is not a metaphor: it is where
  the semantics measurably live.
- The chord/inversion steering is an EQUATORIAL operation (rotates within
  the content plane, leaves latitude alone) -- likely WHY it preserves
  fluency (diversity 0.74 ~ baseline).
- The pit axis is the POLAR one: persistent steering collapses via the
  model's own norm/latitude dynamics.

Open thread: does the equator + semantic map survive at MID-STACK layers,
or is the longitude structure a final-head artifact? (Test E below).

### Files used / produced (this whole arc)

REPO (this work, uncommitted):
- notes/semantic-topography.md        (this note)
- scripts/eval_semantic_map.py        (class caps, purity, scales)
- scripts/eval_som_failure.py         (density profile, SOM collapse)
- scripts/eval_kohonen_sphere.py      (T1/T1b/T2/T3)
- scripts/eval_chord_steering.py      + chord_steering.csv, chord_interference.csv
- scripts/eval_chord_inversion.py     + chord_inversion.csv
- scripts/eval_topic_steering.py      + topic_steering.csv  (generation primitive)
- scripts/eval_nb_quick.py            (class enrichment, mutual-NN, CJK share)
- scripts/eval_equator_fast.py        (equator battery: E1-E3, ~5s each)
- steering_geometry_results/*.csv

### FAST equator battery (E1-E3, ~5s each, eval_equator_fast.py)

E1  equator universality (token-row angle to the position-0/BOS axis):
      Qwen 91.6 (eq) | GPT-2 97.4 (eq) | Pythia 18.6 deg (NOT eq)
      Verified Pythia at 4 prompts (18.6-20.1 deg): real architecture
      difference, not a choice artifact.  NEW cross-model law: the models
      with viable steering reach (Qwen 98%, GPT-2 91%) have an EQUATORIAL
      vocabulary; the worst steered (Pythia 28%) is POLAR w.r.t. position-0.
      -> spherical organization of the vocab predicts steering reach.

E2  chord polar tilt (Qwen): centroids sit 1.2-2.2 deg MORE polar than
      their single rows (nature: -2.2, number: -1.4).  Averaging cancels
      azimuthal spread and leaves a small polar component, but all centroids
      stay in the equatorial band (84.6-87.1 deg).  Chords are ~equatorial.

E3  latitude conservation: a 17-deg inversion arc moves state latitude only
      ~2 deg (51.6 -> 53.2 med, max delta 2.53).  Inversion steering is
      nearly pure azimuthal (content) motion -- corroborates why it
      preserves fluency (leaves the latitude/context channel alone).

PAPER + REPO (pre-existing, read-only for this arc):
- paper/paper_steer.tex  (BOS Axis section, theta_cell, cross-family table)
- scripts/steering_geometry_test.py   (canonical harness: get_states, tangent dir)
- scripts/eval_practical_steering.py  (gen(), sphere_hook)
- scripts/verify_identity.py          (rotation identity)

### Open threads

- distinct family words per generation is low (1.1-1.6/10): add used-note
  suppression for a diversity knob.
- Apply at mid-stack layers (does the semantic map survive depth?) ->
  Test E below is the first cut.
- 0.5B-scale cross-check (gpt2/pythia/gemma generation).
#JZ|- 0.5B-scale cross-check (gpt2/pythia/gemma generation).

---

## Assumption-free checks (S1-S4, eval_som_sweep.py, Qwen2-0.5B)

Addresses two challenges: why 16x16, and the neighborhoods/chords were
assumed semantic without proof.  Every claim re-tested WITHOUT labels.

S1  The LM head is TIED to the input embedding (tie_word_embeddings=True
    for Qwen2-0.5B; same memory, first-row cosine 1.000000).  The semantic
    map on head rows IS the embedding space - no head-specific mechanism
    needed.  Topographic structure is inherited from the tokenizer-world
    distribution (embedding learns geometry directly from co-occurrence).

S2  Lattice sweep (why 16x16 - because any 2D grid fails the same way):
    sides 4/8/16/24/32 -> 16..1024 neurons + 1D-ring(256) control.
    Data 1-NN scale = 61.7 deg (the tiling scale a SOM must beat).
      side  neurons  quant-err  empty%  max-memb%  proto-NN  vs-data
        4     16      64.4     0.0     40.7       19.2    sub-data
        8     64      64.5     9.4     44.8       12.0    sub-data
       16    256      64.4    48.8     33.6        4.0    sub-data
       24    576      64.6    83.3     65.3       65.6    OK
       32   1024      64.4    88.7     50.7       62.6    OK
      1D-ring 256   qe 64.6   empty 0.0%  max-memb 3.9%
    => quant error is ~64 deg at EVERY grid size (matches the data 1-NN
    scale; a SOM cannot beat the intrinsic angular resolution of the data,
    it only repackages it).  Bigger grids redistribute the same 64-deg
    quota; more empty neurons (up to 89%) while max membership stays
    ~40-65% (one prototype still owns half the mass).  The 1-D ring spreads
    membership evenly (3.9% max) at the SAME 64.6 deg error - 2D topology
    is irrelevant, error is set by local density, not tiling.  16x16 was
    not the problem; collapsing is the norm at every topology.

S3  Label-free auto-clusters confirm the inversion law: spherical k-means
    (k=30, no semantics assumed) -> 5 usable clusters on Qwen:
      family  size   spread   center-res  inversion-res
         6    7457   45.4       25%         100%
        20    4980   46.8       75%         100%
         0    1024   70.8        0%         100%
         9     895   70.5      100%         100%
        26     583   69.1        0%         100%
      AVG                  ~25-40%     100.0%   (run-variance: k-means init
                              unseeded; center 0-100 per family,
                              inversion 100 in every run)
    => inversion resolves EVERY auto-clustered family 100%, center-steering
    only 40% (its weak/corrupt spread dependence: center aiming ignores the
    decisive note - the member closest to the current state; spread alone
    is not the right predictor).  The chord idea is label-free.

    PROMPT-ROBUSTNESS (29 diverse prompts: facts/story/question/instruction/
    code/math/CJK/European/register-edge/1-token punctuation):
      overall cells (family x prompt): center 30.3%  vs  inversion 100.0%
      inversion resolves ALL 5 families on 29/29 prompts
      inversion >= center on 29/29 prompts
      per-family center 0-55% (prompt-dependent), inversion 100% flat
    => the inversion law is PROMPT-INDEPENDENT, center failure is not a
      small-prompt artifact.

S4  Geometric NN pairs are behaviorally interchangeable (correlated logits
    across 64 contexts, no labels):
      (t, geometric-NN(t)): mean logit-corr +0.233  (n0-free +0.333)
      (t, random):          mean +0.095  med +0.099
      NN-beats-random: 72.5% of tokens
    => geometric adjacency => functional adjacency: tokens adjacent on the
    sphere produce correlated model output.  No semantics assumed anywhere.
    PROMPT-ROBUSTNESS (29 diverse prompts): NN +0.272 vs random +0.132
    (med +0.096), NN-beats-random 82.5% - the behavioral-interchangeability
    law also survives every register/language/1-token context.

---

## CROSS-MODEL (Gemma-3-1B, GPT-2, Pythia-160M; eval_som_sweep.py [model])

The MUST-DO follow-up is DONE.  Full 4-model battery, same script, same 29
prompts (sizes: Qwen 0.5B/151936 rows, Gemma-3-1B 262144, GPT-2 50257,
Pythia 50254):

                        Qwen2-0.5B    Gemma-3-1B    GPT-2       Pythia-160M
    S1 head tied?        TIED          TIED         TIED        SEPARATE
    S2 data 1-NN scale   61.7 deg      75.2 deg     61.3 deg    19.9 deg
       quant-err         ~64-65        ~74-76       ~60-63      ~18-19
       (== 1-NN scale    at EVERY lattice size + ring on all 4)
    S3 inversion         98.6%         92-99%       100%        93-97%
       center            33.8%         12-29%       67.6%       16-24%
       inv>=center       29/29         29/29        29/29       29/29
    S4 NN / rand corr    +0.35/+0.19   +0.29/+0.14  +0.95/+0.92 +0.70/+0.50
       NN beats rand     80%           77.5%        75%         72.5%
    S5 equator (BOS)     91.5 deg      92.4 deg     98.2 deg    19.1 deg
    (reach @45deg)      (100%)        (98.4%)      (100%)      (96.1%)

Laws - now measured on 4 families (not Qwen-only):
1. quant-err is pinned at the data's own 1-NN scale at EVERY lattice size and
   the 1D ring, on all 4 models, across a 20x spread of that scale (19.9 - 75.2
   deg).  No topology tiles the shell; the error is set by local density.
2. The INVERSION law holds everywhere: ~92-100% families resolved vs ~12-68%
   for center steering, and inversion >= center on 29/29 prompts on all 4 -
   INCLUDING polar Pythia (16.6-24.1% center).  The chord recipe transfers.
3. The EQUATOR law: equatorial vocabs are the steered ones (Qwen 91.5, Gemma
   92.4, GPT-2 98.2 -> reach 98-100%); Pythia is POLAR (19.1 deg) yet still
   reaches 96% @45deg budget - the cheaper-the-equator relation holds at fixed
   budget; Pythia just needs a larger arc.
4. S4 behavioral interchangeability holds on all 4 (NN > random on 72-80% of
   tokens).  GPT-2/Pythia have higher baselines (tying + smaller vocabs) but
   the NN edge is consistent.  NOTE: S4 metrics must be de-poled for polar
   models (project candidate rows off the BOS axis) or every correlation
   collapses to ~1.0 - the shared pole component dominates.
5. S1: TIED embeddings (Qwen/Gemma/GPT-2) confirm the semantic map IS the
   embedding space.  Pythia is the untied outlier (tie_word_embeddings=False),
   and its rows come from embed_out - yet the same laws hold, so the map is a
   property of trained semantic geometry, not of tying per se.

Gibble: gemma's tiny spread families resolve 93-100% but occasionally one
family dips to 69% (tight cluster, one note very close to state) - the
per-family variance is real; the 29/29 always-inversion-advantage is the
robust claim.
---

## TOPIC PATHS (eval_topic_path.py, Qwen2-0.5B, ~4s)

T1  TOPIC RING: the 6 class centroids sit at DISTINCT azimuths around the
    equator (de-poled, PCA-projected to 2D):
        city 15.1  animal 109.9  food 123.1  nature 129.2  color 209.7
        number 278.8   (deg)
        circular order: city -> animal -> food -> nature -> color -> number
    Pairwise equatorial distances 64-90 deg.  There IS a topic topology:
    shortest path between topics = azimuthal distance on the ring.
T2  CHORD WALK food -> city (4deg steps, re-aim at target's best member):
        start ' there' (number) -> enter food ' honey' x2 -> ' London' (city)
        step 3 and locked.  The walk is BALLISTIC, not gradual: it stays on
        the start topic until it crosses the topic decision boundary, then
        jumps to the target and holds.  Path between topics = crossing the
        piecewise decision-cell partition, consistent with competitor
        geometry (HF thread #17) rather than a smooth longitude drift.
T3  OPEN (one 32deg arc) vs CLOSED (8x4deg re-aim) both land in city
    (' London') on Qwen.  Open-loop fine when target within reach; re-aim
    buys safety for hard/edge targets (todo: test farthest pair number-city).
