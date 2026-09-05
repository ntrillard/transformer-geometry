<div align="center">
  <img src="paper/steeronasphere.png" width="180">

  # Steer on a Sphere
  ### messing with transformer outputs using geometry

  [![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21954871-blue)](https://doi.org/10.5281/zenodo.21954871)
  [![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
</div>

---

## What is this?

A playground for bending what a language model writes — using geometry, not prompts, not finetuning.

Hidden states live roughly on a sphere, every token points somewhere on that sphere, and nudging the hidden state toward a token's direction raises that token's probability. Push far enough and it shows up in the output.

Detail is in the evals and writeups.

## Try it

All the fun stuff runs on `Qwen/Qwen2-1.5B`, no quantization, just hooks:

```bash
pip install -r requirements.txt

# baseline — what the model writes alone, no tricks
python3 gen_pure.py Qwen/Qwen2-1.5B "The office was quiet after hours" 120 0

# the reliable one — plants words as real tokens + a short settle blend
HF_TOKEN=$TOKEN python3 gen_blendtraj.py Qwen/Qwen2-1.5B \
    "The office was quiet after hours" "sheep,sushi,elevator"
# env: LAM=0.4 SETTLE=8 HOLD_ANGLE=4 PLANT0=20 SEED=0

# pure geometry — no input edits at all, just rotates the readout
# MODE=emit = push until the word appears once, then let go
G_ANGLE=8 G_LAN=0.9 WINDOW=12 SW0=20 SEED=0 MODE=emit python3 old/gen_geom.py \
    Qwen/Qwen2-1.5B "The waves crashed gently on the beach" "computer,lantern,trumpet"
    Qwen/Qwen2-1.5B "The waves crashed gently on the beach" "computer,lantern,trumpet"
```

What those scripts do, in plain English:

| File | What it does |
|---|---|
| `gen_pure.py` | Plain sampling. Zero hooks. This is "what would the model do anyway?" |
| `gen_blendtraj.py` | Drops each target word into context as a real token, runs a short 8-step blend of natural + slightly-steered readouts so the splice doesn't break, then hands control back. Most coherent output. |
| `old/gen_geom.py` | No input edits. Just rotates the readout toward the target until it pops out once, then stops. Sometimes misses honestly, and pushing too hard loops — but when it lands the grammar is nice ("diamond ribs", "marble chests"). |

Example — office scene with `sheep,sushi,elevator`:

> The office was quiet after hours . The rest of the employees had left to go to their homes.
> It was a Saturday night and **sheep**ish Adam Frost sat in his office using his mouse to type.
> He was just like the other developers, waiting for someone to send him the file **sushi**.ico
> that he needed for the next game he was working on... Just as **elevator** music played
> in the background, the file finally came.

All three words land mid-sentence without breaking grammar. That's the whole point — older force-inject tricks made the model "snap back" ("sushi a year ago", "Two people elevator people"). Planting + a short settle window fixes most of that.

## What actually works (short version)

Full story in `steering-evals/steering_geometry_results/` — here's the tl;dr from a bunch of late-night runs (documented in the HF thread):

- **Plant + settle is the workhorse.** Real token in context does ~100% of the work. The blend (LAM=0.4, 8 steps) just keeps the splice smooth. More blending / fancier branches only blur things. Archived the failures in `old/`.
- **Pure geometry works too, with one rule: stop after one hit.** Push until the word is sampled once, then let go. Same rotation held too long = loops. Too weak = misses honestly. The small bit of natural logits mixed in (10-20%) is what keeps the story from sliding into quiz-template land.
- **Topic steering is real but different.** Can't plant a whole vibe, so this one uses a logit-space contrast (mean target-next-token minus neutral, top-200, added at low weight). It's a dial: too low = nothing, just right = scene bends mid-narrative ("beach → evil creatures from the surf"), too high = hijack loop.
- **There's also fun pit-token stuff** (some tokens like `0` love to repeat themselves forever). Defenses + false-positive tables live in [`THREAT_MODEL.md`](THREAT_MODEL.md) if you're into that.

Honest limits: single-token words can get outcompeted by the story, 2-token words (cactus, ketchup) don't fit the single-direction trick, and one prompt (the farm one) collapses into a quiz template even with zero steering — so that's the model, not us.

## Repo map

```
.
├── gen_pure.py                   # baseline, no hooks
├── gen_blendtraj.py              # plant words + settle blend (start here)
├── old/                          # experimental + probe scripts (gen_geom, falsify_orth*, pivot*, mechanism_matrix, ...)
├── old/                          # failed variants, kept for honesty
├── steering-evals/
│   ├── scripts/                  # measurement harness
│   └── steering_geometry_results/ # CSVs + longer writeups
├── paper/                        # preprint tex + pdf (see note below)
├── THREAT_MODEL.md               # pit tokens + mitigations
├── CITATION.cff
└── LICENSE                       # CC BY 4.0
```

Repro for the measurement side:

```bash
cd steering-evals/scripts
python verify_identity.py
python steering_geometry_test.py --model Qwen/Qwen2-0.5B-Instruct \
    --targets 64 --contexts 4 --layer-fracs 0.0,0.33,0.67,0.99
```

Older sweep scripts were removed in the Aug-21 cleanup — check git history if you really want them.

## Longer writeups

- `writeup-blendtraj.md` — plant + settle, what mattered and what didn't
- `writeup-geom.md` / `writeup-geom-many.md` — pure geometry across 9 scenes (8/9 coherent, 24/27 words vs 0/27 unsteered)
- `writeup-sentence-concept.md` — concept centroids + full-sentence/topic steering, incl. the 6 things that didn't work
- HF thread — [Steer on a Sphere](https://discuss.huggingface.co/t/steer-on-a-sphere-geometric-control-of-transformer-outputs/178732/20) — where most of the debugging happened in public

---

Note: there's a preprint describing an earlier version of this ([`paper/paper_steer.pdf`](paper/paper_steer.pdf), Zenodo [10.5281/zenodo.21954871](https://doi.org/10.5281/zenodo.21954871)). The repo has moved on since then — trust the scripts + writeups above over the preprint text.
