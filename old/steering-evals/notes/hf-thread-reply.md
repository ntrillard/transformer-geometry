# Reply to John6666 (#17) — complementary measurements on the LM-head row geometry

Draft reply for the HF thread. Post #17 proposes a decomposition (generic
high-dimensional geometry as the null model; training as the source of
departure) and supports it with a Pythia checkpoint trajectory. This reply
contributes observations from the token-row side of the same object, obtained
without a training-trajectory apparatus. All numbers are reproducible from the
files cited inline (`ntrillard/transformer-geometry`, `steering-evals/`).

## 1. The random null does not account for the measured row structure

The decomposition is a useful working frame. Measured on four families
(Qwen2-0.5B, Gemma-3-1B, GPT-2, Pythia-160M), three row-side properties
appear to sit outside the iid-Gaussian null:

1. **Behavioral correlation of geometric neighbors** (`eval_som_sweep.py`,
   S4). For 40 uniformly sampled printable tokens, the Pearson correlation
   between a token's logits and those of its geometric nearest-neighbor row,
   computed across 29 diverse prompts, is +0.29 to +0.95 for neighbors and
   +0.14 to +0.92 for random pairings; neighbors exceed random on 72–80% of
   tokens on all four models. Under the null the two quantities would be
   statistically indistinguishable and near zero.

2. **Cross-model semantic separability** (`eval_semantic_map.py`). Six
   hand-labeled classes (food, animal, color, city, nature, number) have
   intra-class median row angles below inter-class medians in the ratio
   0.866 / 0.827 / 0.847 / 0.815 (Qwen / GPT-2 / Pythia / Gemma). Random rows
   carry no class ordering, so this structure is at least partly a product of
   training.

3. **The readout is the embedding matrix** (`eval_som_sweep.py`, S1). Qwen,
   Gemma and GPT-2 tie the language-model head to the input embedding
   (`tie_word_embeddings=True`; the corresponding output rows coincide in
   memory). The row geometry in question is therefore the co-occurrence
   structure of the embedding matrix itself — the same object whose evolution
   the checkpoint trajectory tracks from the state side.

## 2. An azimuthal coordinate alongside the radial one

The reach measures reported in #17 — critical angles and per-arc
accessibility — characterize a radial separation (state relative to the
target cone) at each depth. A second, azimuthal organization also appears to
be present. After projecting the six class centroids off the BOS/latitude
axis and onto their two leading principal components, the centroids fall at
distinct longitudes (`eval_topic_path.py`, T1):

```
city 15° -> animal 110° -> food 123° -> nature 129° -> color 210° -> number 279°
```

with pairwise equatorial distances in the 64–90° range. If this organization
survives across depths, a steering intervention has a two-axis reading: a
radial component (whether an arc of length θ reaches the target cone, the
quantity the trajectory results characterize) and an azimuthal component
(where a topic sits on the ring, computable from the embedding matrix and a
single forward pass at any layer).

## 3. The chord walk is consistent with the competitor/decision-cell account

As a step-level test of the row geometry, we translate a state toward a
target family's best-positioned member in 4° increments (`eval_topic_path.py`,
T2). The top-1 token remains on the starting topic through the approach, then
jumps to the target and locks at the boundary; the transition is abrupt
rather than gradual. This is the rank-competitor mechanism reported in #17
(competitor-relative residual orientation dominating rank at fixed target
score), observed from the row side: the intervening structure is the
decision-cell partition rather than the target-tangent gradient. On Qwen,
both the single-arc (open-loop) and re-aimed (closed-loop) forms reach the
target for every class pair tested, including the farthest (number–city,
89.9°); re-aiming appears to matter primarily for low-spread families (one
Gemma cluster resolved at 69%), which is the regime in which the toward-
blocker effect in the table is strongest.

## 4. Possible joint directions

Each of the following appears cheap relative to another cross-family sweep:

- Whether the azimuthal organization (Section 2) is present at initialization
  or emerges during training. Comparing the ring across Pythia checkpoints,
  with the same null ladder (iid rows, label-permuted rows, observed rows),
  would give the row-side analogue of the accessibility trajectory.
- Whether the depth-dependent localization reported in #17 (late-training
  accessibility concentrated in the final block) also appears in the row map
  or is state-side only. Cross-reading block-depth accessibility against the
  ring at matching checkpoints would separate the two.
- A frequency/register control on the ring order, to check whether the
  city–animal–food–nature–color–number sequence reflects semantics or
  token-frequency structure.

## References and reproducibility

- `eval_som_sweep.py` — S1 (head tying), S4 (neighbor behavioral
  correlation), S5 (equator/BOS projection).
- `eval_semantic_map.py` — class separability ratios.
- `eval_topic_path.py` — topic ring (T1) and chord walk (T2/T3).
- `notes/semantic-topography.md` — cumulative results and caveats.

```bash
cd steering-evals/scripts
python eval_topic_path.py [model] [start] [target]   # ~4 s
python eval_som_sweep.py [model]                     # ~20 s
```

If the checkpoint experiment is pursued, we would be glad to contribute the
azimuthal measurements; prompt and seed hashes are available on request.