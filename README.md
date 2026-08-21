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
| Qwen2-1.5B | 99.2% | 1.6% | 34.8% | 8.0° |
| Qwen2-0.5B | 97.3% | 1.6% | 13.7% | 10.6° |
| GPT-2 | 90.8% | 1.0% | 5.9% | 11.6° |
| SmolLM-135M | 67.2% | 0.4% | 2.5% | 9.8° |
| Pythia-160M | 27.9% | 0.4% | 9.0% | 2.6° |

- Rank acquisition is governed by the LM-head decision partition, not just target alignment: rotating the residual *toward* the strongest blocker collapses success to 3–35%, *away* restores ~100%, at fixed norm and fixed target score.
- No mid-arc loss: any target that becomes rank-1 along the arc stays rank-1 at the endpoint (verified on all 2,560 cases).
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

## Repository Layout

```
.
├── paper/                          # preprint (tex + pdf + figure)
├── THREAT_MODEL.md                 # threat model + mitigation/false-positive tables
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
│   │   └── eval_threat_model.py          # mitigations + false positives (+ --quant for 7B)
│   └── steering_geometry_results/  # raw CSVs backing every paper number
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
python eval_pit_robustness.py --model Qwen/Qwen2-0.5B-Instruct --seeds 64
python eval_threat_model.py --model Qwen/Qwen2.5-7B-Instruct --quant nf4 --pit-id 15
```

Legacy exploration tooling (rotation sweeps, original proof scripts) was removed in
the 2026-08-21 consolidation; recover it from git history if needed.

## Paper

📄 [`paper/paper_steer.pdf`](paper/paper_steer.pdf) — full preprint

> Trillard, N. (2026). *Steer on a Sphere: Geometric Control of Transformer Outputs*. Zenodo. [10.5281/zenodo.21954871](https://doi.org/10.5281/zenodo.21954871)

---

Preprint. The geometric picture is approximate, not a theorem. Steering is a white-box traversal primitive. [CC BY 4.0](LICENSE).
