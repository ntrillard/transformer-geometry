# HF thread update — "Steer on a Sphere" (Aug 30, 2026)

Update to the steering-on-a-sphere discussion. Since the last post, the
semantic-topography arc went from Qwen-only to a 4-model law, and the two
hardest claims ("are the neighborhoods actually semantic?" and "why did 16x16
fail?") are now measured without any assumptions.

TL;DR — four laws, all reproduced on **Qwen2-0.5B, Gemma-3-1B, GPT-2, and
Pythia-160M**, ~20 s per model on one 3080:

| Law | What it says |
|---|---|
| 1. The map IS the embeddings | Qwen / Gemma / GPT-2 LM heads are **tied** to the input embedding — the semantic map on head rows is literally the embedding space, no head-specific machinery |
| 2. You can't tile the shell | SOM/lattice quantization error is pinned at the data's own 1-NN angular scale at **every** lattice size (16 → 1024 neurons) and even a 1D ring. Scales measured: 19.9° (Pythia), 61.3–61.7° (Qwen/GPT-2), 75.2° (Gemma). Grid only repackages the same error + add empty cells |
| 3. Aim at the member, not the center | Label-free families (spherical k-means, no semantics assumed) resolve under **inversion steering** (aim at the family member closest to the current state): ~92–100% vs ~12–68% for center steering. Inversion ≥ center on **29/29 diverse prompts, on all 4 models** — including polar Pythia |
| 4. Equator law, now complete | Median token-row angle to the BOS axis: Qwen 91.5°, Gemma 92.4°, GPT-2 98.2° (equatorial → cheap steering, reach 98–100% @45°) vs **Pythia 19.1° (polar)** — yet Pythia still reaches 96% @45°, it just needs a larger arc. Equatorial = budget-lean, not a hard cutoff |

## What these give you

- **A steering recipe that transfers across families**: don't aim at a family's
  average (it's the point farthest from every member), aim at the
  best-positioned member. One 17° rotation from any of 29 diverse contexts
  (facts, code, CJK, even `?` alone) resolves the topic family. Single-shot
  topical conditioning at baseline fluency/diversity; cadence k≥3 is the dose
  dial; persistent = degenerate pit (the polar axis).
- **Two methods retired with numbers**: global SOM/topographic lattices on the
  head shell (structural failure, all sizes/topologies), and center-of-class
  steering (works only when the family happens to face you).
- **A cheap pre-flight metric**: measure your model's row-to-BOS angle and its
  data 1-NN scale before building any steering tooling. Two lines, seconds.

## Methodology notes (so it's checkable)

- All numbers from `eval_som_sweep.py` in the repo
  (`ntrillard/transformer-geometry`): S1 head-tying, S2 lattice sweep, S3
  label-free inversion vs center, S4 logit interchangeability of geometric
  neighbors vs random (NN beats random on 72–80% of tokens everywhere), S5
  equator angle. 29 prompts: facts, stories, questions, instructions, code,
  math, CJK, German/French, register edges, 1-token punctuation.
- For **polar** models the S4 correlation must be computed after projecting the
  candidate rows off the BOS axis — otherwise the shared pole component drives
  every correlation to ~1.0 and the metric is meaningless.
- Honest scope: per-family inversion is 93–100% with occasional dips on very
  tight clusters (one family, one run: 69%); the robust claim is that inversion
  ≥ center on 29/29 prompts and 4/4 models. The equator numbers used
  position-0 of "Once upon a time" as the BOS/probe axis (same as the paper's
  BOS-axis section).

## Still open

- Mid-stack layers: does the semantic map + inversion law survive at
  non-final layers (the paper says final-layer steering is near-universal)?
- The persistent-steering pit boundary on the polar model.

Links: repo
[ntrillard/transformer-geometry](https://github.com/ntrillard/transformer-geometry)
· preprint `paper/paper_steer.pdf` · notes
`steering-evals/notes/semantic-topography.md` · battery
`steering-evals/scripts/eval_som_sweep.py`