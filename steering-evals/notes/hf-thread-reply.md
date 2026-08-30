# Reply to John6666 (#17) — topical map + chord walk against your null framing

Draft for the HF thread. Woven into it: your null/decomposition framing, our
four-model topical map as "what training bought", the topic ring giving the
picture an azimuthal coordinate, and the chord walk matching your competitor
geometry. All numbers reproducible from the named files.

---

> Rather than "training is not involved", the random matrix is a null model
> for the part that comes for free — then ask where the trained model departs.

Agreed, and that framing fits what we've been measuring on the **token-side
geometry** (the LM-head rows / embedding map, as opposed to the state-side θ
you probed). I read your Pythia trajectory as the state-side departure; here's
the row-side departure, which we now have on four families (Qwen2-0.5B,
Gemma-3-1B, GPT-2, Pythia-160M — all offline in a ~20 s battery,
`eval_som_sweep.py` S1/S4/S5).

**Where the map is trained, not free:**

1. **Geometric neighbor rows are behaviorally correlated** (S4): for 40
   random printable tokens, the logit correlation between a token and its
   geometric nearest-neighbor row across 29 diverse prompts is NN +0.29–0.95
   vs random +0.14–0.92, with NN beating random on 72–80% of tokens on all
   four models. Your iid-Gaussian null would put both near 0 — the NN signal
   is a trained, non-random structure.
2. **The semantic-class geometry is cross-model, not Qwen-quirk**
   (`eval_semantic_map.py`): the intra/inter row-angle separability for six
   hand-labeled/frequency-balanced classes is 0.866 / 0.827 / 0.847 / 0.815.
   Identical logic to your cross-checkpoint compatibility check — a trained
   structure, since random rows would show no class ordering.
3. **The readout has a concrete object**: S1 shows the LM heads of Qwen,
   Gemma and GPT-2 are *tied* to the input embedding. So the row-side
   geometry is the embedding matrix itself — the thing whose co-occurrence
   structure training writes.

**An azimuthal coordinate for your radial picture.** The "steering reach"
you measure (θ_critical, accessibility-per-arc) is the radial axis of the
state to the target cone. We found the *second* axis: the class centroids sit
at distinct azimuths around the equator (`eval_topic_path.py` T1:

```
city 15° -> animal 110° -> food 123° -> nature 129° -> color 210° -> number 279°
```

pairwise equatorial distances 64–90°). So a steering act has a two-axis
decomposition: *can I reach it* (your θ_critical vs arc) and *where is it*
(topic longitude on the ring). At any layer you can compute both from the
embedding matrix + one forward pass.

**The chord walk is exactly your competitor picture.** Stepping toward a
topic family (`eval_topic_path.py` T2, 4° re-aimed rotation toward the
family's best-positioned member): top-1 stays pinned on the start topic
through the arc, then jumps to the target and locks at the decision boundary
— it never slides gradually. That's the "competitor-relative residual
orientation dominates rank" result, seen from the token side: the decision-cell
partition is the thing the arc crosses, not the target-logit gradient.
(Open vs closed loop both land on Qwen across pairs including the farthest
89.9°; re-aiming only matters for tight low-spread families, e.g. one Gemma
cluster dipped to 69% — the mechanism you'd expect from your "toward-blocker
destroys" table.)

**Natural joint next steps** (all cheap, none need a sweep):

- Run T1 (the topic ring) along your Pythia trajectory: does the ring
  *appear* over training, or is it present from step0 like your
  initialization reach? That is the row-side analogue of your
  "departure-from-null over depth" result, with basically the same null
  ladder (iid rows -> random-shuffle row labels -> real rows).
- Your accessibility-vs-depth table (block 0→11) crossed with the topic ring:
  is the late-training localization (final block 100%) also a *row-side*
  localization, or purely state-side?
- The largest point I'd keep open: whether the ring order (city-animal-food-
  nature-color-number) is a frequency artifact or semantic. A frequency-
  balanced random-sample control would say.

Everything above is in `ntrillard/transformer-geometry`, `steering-evals/`:
`eval_som_sweep.py` (S1/S4/S5), `eval_semantic_map.py`, `eval_topic_path.py`;
results in `notes/semantic-topography.md`. Happy to share the exact
prompt/seed hashes.

---

Reproduction blurb (for the post footer):

```bash
cd steering-evals/scripts
python eval_topic_path.py [model] [start] [target]   # topic ring + chord walk (~4 s)
python eval_som_sweep.py [model]                     # S1-S5 battery (~20 s)