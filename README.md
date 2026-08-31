<div align="center">
  <img src="paper/steeronasphere.png" width="180">

  # Steer on a Sphere
  ### Geometric Control of Transformer Outputs

  **N. Trillard** — August 21, 2026

  [![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21954871-blue)](https://doi.org/10.5281/zenodo.21954871)
  [![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
</div>

---

## How It Works

Every transformer layer uses RMSNorm, which constrains hidden states near a sphere of radius $\|\gamma_l\|$ — a learned value, not $\sqrt{d}$.

The LM head gives every token a direction on that sphere. **A tangent step toward any token's direction raises its logit with mathematical guarantee and reaches rank 1 on 97–99% of reachable tokens (Qwen family), with the wrong-target control at ~0%.**

```
  Hidden state h
       │
       ▼
  g_t = W_t − (W_t·ĥ)ĥ    ← tangent toward token t
       │
       ▼
  h' ← normalize on sphere
       │
       ▼
  logits → t ranks #1 within θ° arc
```

---

## Key Results

**Steering geometry** (64 targets × 2 contexts × 4 depth-adaptive layers, fp16, seed 42):

| Model | Arc-reach | Wrong-target | Toward blocker | Median first rank-1 angle |
|---|---:|---:|---:|---:|
| Qwen2-1.5B | 99.2% | 0.0% | 34.8% | 8.0° |
| Qwen2-0.5B | 97.3% | 0.0% | 13.7% | 10.6° |
| GPT-2 | 90.8% | 0.0% | 5.9% | 11.6° |
| SmolLM-135M | 67.2% | 0.0% | 2.5% | 9.8° |
| Pythia-160M | 27.9% | 0.0% | 9.0% | 2.6° |

- Rank acquisition is governed by the LM-head decision partition, not just target alignment: rotating the residual *toward* the strongest blocker collapses success to 3–35%, *away* restores ~100%, at fixed norm and fixed target score.
- No mid-arc loss: any target that becomes rank-1 along the arc stays rank-1 at the endpoint (verified on all 2,560 cases).
- The 17° table above is the fixed-budget (α=0.3) view. Sweeping the budget shows the spread is a *shift*, not a ceiling: every family reaches ≥96% rank-1 within a 45° budget (Qwen/GPT-2 100%, Gemma-3-1B 98.4%, Pythia-160M 96.1%); wrong-target/random controls stay 0% at every budget. Pythia mid-layers need ~29° (its entry angles are 28.7–30.7° across layers 0–7) vs 2.4° at the final layer.
- Shortest route: cone projection gives `theta_cell ≤ theta_author` (e.g. 8.45° vs 10.02° on Qwen2-0.5B).
- Natural activations sit near a spherical shell (norm std ≈ 7% of mean); actual next-token steps are 2–2.6× larger than a controlled tangent step.

**Cow tipping** — self-reinforcing pit tokens (`0`, `<mask>`, NULL byte, …):

| Finding | Result |
|---|---|
| Qwen2-0.5B `"0"` basin | decoder-dependent; sampling breaks it |
| Qwen2.5-7B strict pit | survives greedy + nucleus sampling; broken only by chat-template context |
| Best defenses (shallow pits) | repetition cap, output collapse, entropy floor, pit-away steering — 0 false positives |
| Strict 7B pit | resists sampling AND single-layer steering at α=1.0; only truncation guards + context work |

Full mitigation/false-positive matrices: [`THREAT_MODEL.md`](THREAT_MODEL.md).

---

## Narrative Steering (Qwen2-1.5B)

Production scripted-steering tools live at the repo root. Both are minimal, hook-only, run on
`Qwen/Qwen2-1.5B` (bf16, no quantization).

| File | Purpose |
|---|---|
| `gen_pure.py` | **Unsteered baseline** - model + multinomial sampling only (zero hooks). Establishes what the model writes alone for any prompt/seed. |
| `gen_blendtraj.py` | **Production steerer** - plant each target word as a REAL token in context (space-prefixed single token so it doesn't fuse), then a settle window blending two readout series (natural + a small hold rotation) before handing back to free generation. |
| `gen_geom.py` | **Pure-geometric control** - rotation-only readout steering, NO input edit; words may honestly miss. Measures what geometry alone can do: sub-threshold = narrative no-op, rank-1 = degenerate loops, never grammatical. |

**The winning configuration** (SETTLE=8 is the coherent sweet spot):

```bash
HF_TOKEN=$TOKEN python3 gen_blendtraj.py Qwen/Qwen2-1.5B \
    "The office was quiet after hours" "sheep,sushi,elevator"
# env: LAM=0.4 SETTLE=8 HOLD_ANGLE=4 PLANT0=20 SEED=0
```

Example output (all three out-of-place words woven in grammatically):

> The office was quiet after hours . The rest of the employees had left to go to their homes.
> It was a Saturday night and **sheep**ishly I found my way in the door. It was a sleepy
> evening and just like the five horsemen; I was the one who had to **sushi** for dinner and
> sleep in bed later than my co-workers... One **elevator** ride later I was seeing the snowy
> white of a Maine coon...

**Key findings from the exploration** (see `steering-evals/steering_geometry_results/writeup-blendtraj.md`):

1. **The planted real token is 100% of the steering** - the model writes *with* the word in
   context naturally ("sheep**ishly**", "sushi noodles", "elevator maintenance invoice").
2. **With blending off (LAM=0), window length is irrelevant** - SETTLE 8/14/20/30 are all
   byte-identical; the settle branch is pure natural sampling.
3. **With blending on, SETTLE=8 is most coherent**; 14-20 develop more but loosen the splice;
   30 over-extends (maintenance-spam tangent).
4. **More blending is not better** - every added state (pre-insert memories, counterfactual
   branches, rolling post-insert memories, 3+/5-way simplexes) is stale relative to the live
   context and blurs the readout at the splice. The two-series blend at LAM=0.4 is the ceiling.
   All experimental variants are archived in `old/`.
5. **Rotation-only steering cannot produce grammatical usage** (`gen_geom.py`, the "isn't
   this cheating?" control) - at honest angles (θ=6, eff≈3°) the bias is a narrative no-op
   (0/9 words); at rank-1 angles (θ=8-10°) the model degenerate-loops the token ("sheep
   sheep sheep...") because it was never in context. The planted token is the grammatical
   foothold; the geometry moves the state. See `writeup-geom.md`.
---

## Repository Layout

```
.
├── paper/                          # preprint (tex + pdf + figure)
├── THREAT_MODEL.md                 # threat model + mitigation/false-positive tables
├── gen_pure.py                     # unsteered baseline (model + multinomial, zero hooks)
├── gen_blendtraj.py                # production steerer (plant real tokens + settle blend)
├── old/                            # archived experimental steerers (gen_steer, gen_blend*, ...)
├── steering-evals/
│   ├── scripts/                    # canonical measurement harness
│   │   ├── verify_identity.py            # endpoint identity check (200/200, <1e-10)
│   │   ├── steering_geometry_test.py     # specificity + residual-latitude controls + angles
│   │   ├── steering_geometry_test_offarc.py  # null-space off-arc variant (cross-family table)
│   │   ├── eval_boundary_instruments.py  # theta_author vs theta_cell cone projection
│   │   ├── eval_manifold_geometry.py     # sphere vs natural-manifold probe
│   │   ├── eval_defense.py               # pit scan + defensive encoding
│   │   ├── eval_pit_robustness.py        # decoder-contract robustness matrix
│   │   ├── eval_7b_strict_pit_gate.py    # strict-pit gate for Qwen2.5-7B (--quant nf4)
│   │   ├── eval_multi_goal_steering.py   # reach-vs-budget curves + multi-goal battery
│   │   └── eval_threat_model.py          # mitigations + false positives (+ --quant for 7B)
│   └── steering_geometry_results/  # raw CSVs + writeups (incl. writeup-blendtraj.md)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Reproduce

```bash
pip install -r requirements.txt
cd steering-evals/scripts

python verify_identity.py                                        # endpoint identity
python steering_geometry_test.py --model Qwen/Qwen2-0.5B-Instruct \
       --targets 64 --contexts 4 --layer-fracs 0.0,0.33,0.67,0.99
python eval_boundary_instruments.py --model Qwen/Qwen2-0.5B-Instruct
python eval_manifold_geometry.py --model Qwen/Qwen2-0.5B-Instruct --layer-idx 8
python eval_multi_goal_steering.py --model Qwen/Qwen2-0.5B-Instruct \
       --targets 64 --contexts 2 --layer-fracs 0.0,0.33,0.67,0.99 --plain-prompts \
       --budget 45
python eval_pit_robustness.py --model Qwen/Qwen2-0.5B-Instruct --seeds 64
python eval_threat_model.py --model Qwen/Qwen2.5-7B-Instruct --quant nf4 --pit-id 15
```

Legacy exploration tooling (rotation sweeps, original proof scripts) was removed in
the 2026-08-21 consolidation; recover it from git history if needed.

## Paper

📄 [`paper/paper_steer.pdf`](paper/paper_steer.pdf) — full preprint (v2, Aug 21 2026)

> Trillard, N. (2026). *Steer on a Sphere: Geometric Control of Transformer Outputs*. Zenodo. [10.5281/zenodo.21954871](https://doi.org/10.5281/zenodo.21954871)

---

Preprint. The geometric picture is approximate, not a theorem. Steering is a white-box traversal primitive. [CC BY 4.0](LICENSE).
