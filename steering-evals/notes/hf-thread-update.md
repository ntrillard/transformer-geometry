# HF thread — topical neighborhoods & chord-inversion steering (Aug 30, 2026)

Result-led, file-for-file reproducible (`ntrillard/transformer-geometry`,
`steering-evals/`). Two findings: the LM head's rows are a functional,
label-free topical map; and steering a token family works by aiming at its
best-positioned member, not its center.

---

## Finding 1 — the topical neighborhoods

**`scripts/eval_kohonen_sphere.py` (test T1b)** — cosine-KNN neighborhoods of
30K sampled head rows, Qwen2-0.5B:

```
apple  ->  Apple / Apple / apples / 苹果 / APPLE
Paris  ->  Paris / 巴黎 / France / French / London
king   ->  King / queen / kings / 国王 / KING
ocean  ->  Ocean / oceans / 海洋 / sea / Sea / 海水
```

- identity variants ~45°; same-class ~75° intra vs 86.6° inter; CJK ≈16% of
  the 30-NN.

**`scripts/eval_nb_quick.py`** — enrichment of the *hand-labeled* class sets:
number/color/city enrich 3.2–6.3× random in the neighborhoods, food/animal
0.4–0.8 — the neighborhoods genuinely pick up the semantic classes.

**`scripts/eval_semantic_map.py`** — the class-cap structure is cross-model:
intra/inter separability 0.866 (Qwen), 0.827 (GPT-2), 0.847 (Pythia), 0.815
(Gemma-3-1B).

**`scripts/eval_som_sweep.py` (S1, S4, S5)** — what the map is, without labels:

- **S1** — Qwen/Gemma/GPT-2 heads are *tied* to the input embedding: the
  neighborhood map IS the embedding space, not a head-specific artifact.
- **S4** — geometric NN token pairs have correlated logits across 29 diverse
  prompts: NN corr +0.29–0.95 vs random +0.14–0.92; NN beats random on
  72–80% of tokens in all 4 models → geometry ⇒ function.
- **S5** — the map is equatorial (median row-to-BOS 91.5° Qwen / 92.4° Gemma /
  98.2° GPT-2 vs 19.1° Pythia-polar) → the map is longitudinal (content); the
  pole is context.

Why cosine-KNN and not a fitted map: **`scripts/eval_som_failure.py`** +
`eval_som_sweep.py` **S2** showed SOM quantization error is pinned at the
data's own 1-NN angular scale at every lattice size (16→1024) and even a 1D
ring — 19.9° (Pythia), 61–62° (Qwen/GPT-2), 75.2° (Gemma). The shell has no
low-dim manifold to tile; the useful map is already in the rows, free.

## Finding 2 — chord-inversion steering

**`scripts/eval_chord_steering.py`** (+ `results/chord_steering.csv`,
`chord_interference.csv`) — steering a token family at its **centroid**
(center-steering) fails as family spread grows: threshold ~50°,
corr(spread, reach) = −0.84. The centroid is the point farthest from every
member.

**`scripts/eval_chord_inversion.py`** (+ `results/chord_inversion.csv`) —
instead aim at the **best-positioned member** of the family (the note closest
to the current state) → the family cone resolves **89.6% vs 22.9%**.

**`scripts/eval_som_sweep.py` (S3)** — the recipe is label-free and
cross-model: spherical k-means clusters (no word classes assumed) resolve on
Qwen2-0.5B 98.6%, Gemma-3-1B 92–100%, GPT-2 100%, Pythia-160M 93–97% vs
12–68% center-steering, and inversion ≥ center on **29/29 diverse prompts ×
4 models**, including polar Pythia.

**`scripts/eval_topic_steering.py`** (+ `results/topic_steering.csv`) — the
generation primitive: a single 17° inversion arc = free topical conditioning
(100% first-token adherence, diversity 0.74 ≈ baseline); cadence k≥3 = a
topical dose dial; persistent-steering = the known pit failure mode.

**`scripts/eval_equator_fast.py` (E3)** — why it stays fluent: inversion arcs
conserve latitude (~2° change), i.e. nearly pure content-plane motion that
leaves the context/latitude channel alone.

---

## Reproduce

```bash
cd steering-evals/scripts
python eval_kohonen_sphere.py        # T1b neighborhoods
python eval_nb_quick.py              # class enrichment
python eval_semantic_map.py          # cross-model class caps
python eval_chord_steering.py        # center-steering law
python eval_chord_inversion.py       # inversion recipe
python eval_topic_steering.py        # generation primitive
python eval_som_sweep.py [model]     # S1-S5: tying, lattice, label-free
                                     #   inversion, NN interchange, equator
                                     #   (~20 s/model; gemma needs HF token)
```

Idea log: `notes/semantic-topography.md`; all numbers backed by `results/*.csv`.

## Scope

- Per-family inversion 93–100% with rare dips on tight clusters (one run 69%);
  the robust claim is inversion ≥ center on 29/29 prompts × 4/4 models.
- Polar models (Pythia) must de-pole the S4 rows off the BOS axis or
  correlations saturate ~1.0.
- Open: does the neighbor map + inversion law survive at mid-stack layers; the
  persistent-steering pit boundary on the polar model.