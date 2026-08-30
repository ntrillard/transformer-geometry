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