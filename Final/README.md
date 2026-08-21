# Steer on a Sphere — Final Concise Package

This folder contains the corrected measurement harness, key results, and
updated paper for the transformer-geometry project.

## What the technique is

Layer normalization constrains transformer hidden states to a sphere whose
radius is the learned RMSNorm scale `||γ_l||`, not the textbook `sqrt(d)`. A
tangent step toward any token's LM-head direction increases that token's logit
with mathematical guarantee; renormalizing keeps the state on the sphere.

The corrected contracts show that:

1. **Endpoint identity** holds: tangent step + renormalization equals a
   same-target great-circle rotation.
2. **Target specificity** holds: a self-excluding wrong-target control reaches
   rank 1 close to 0% of the time.
3. **Competitor geometry** dominates off-arc behavior: at fixed norm and fixed
   target score, residual direction toward or away from the strongest blocker
   determines whether the target wins the rank competition.
4. **Shortest route** is modelable: projection onto the raw LM-head rank-1 cone
   gives shorter routes than the target-tangent arc, with an internal
   certificate (`theta_cell <= theta_author`).
5. **No mid-arc loss**: in 2,560 cross-family cases, any target that became
   rank-1 along the tangent arc remained rank-1 at the arc endpoint.

## Folder layout

```
Final/
├── paper/
│   ├── paper_steer.pdf       # compiled updated paper
│   └── paper_steer.tex       # LaTeX source
├── technique/
│   ├── steering_geometry_test.py      # specificity + competitor geometry + arc angles
│   ├── eval_boundary_instruments.py   # theta_author vs theta_cell
│   ├── eval_manifold_geometry.py      # sphere vs ellipsoid vs natural activation manifold
│   └── verify_identity.py             # endpoint-identity unit check
├── cow-tipping/
│   ├── eval_pit_robustness.py         # decoder-contract robustness matrix
│   ├── eval_7b_strict_pit_gate.py     # small strict-pit gate for Qwen2.5-7B
│   ├── eval_defense.py                # defensive-encoding baseline
│   └── eval_threat_model.py           # mitigations + false-positive eval
├── results/
│   ├── cross_family_summary.{csv,md}  # 5-model cross-family summary
│   ├── Qwen--Qwen2-0.5B-Instruct__t64c4_lf0-0.33-0.67-0.99_fp16.csv
│   ├── Qwen--Qwen2-1.5B-Instruct__t64c4_lf0-0.33-0.67-0.99_fp16.csv
│   ├── google--gemma-3-1b-it__t128c4_lf0-0.33-0.67-0.99_fp16.csv
│   ├── cone_theta__Qwen--Qwen2-0.5B-Instruct.csv
│   ├── cone_theta__google--gemma-3-1b-it.csv
│   ├── pit_robustness__Qwen--Qwen2-0.5B-Instruct.csv
│   ├── pit_robustness__google--gemma-3-1b-it.csv
│   ├── strict_pit_gate__Qwen--Qwen2.5-7B-Instruct.csv
│   ├── threat_model_mitigations__Qwen--Qwen2-0.5B-Instruct.csv
│   ├── threat_model_false_positives__Qwen--Qwen2-0.5B-Instruct.csv
│   ├── threat_model_mitigations__google--gemma-3-1b-it.csv
│   ├── threat_model_false_positives__google--gemma-3-1b-it.csv
│   ├── manifold_natural__Qwen--Qwen2-0.5B-Instruct_L8.csv
│   └── manifold_steering__Qwen--Qwen2-0.5B-Instruct_L8.csv
└── THREAT_MODEL.md
```

## Quick reproduction

```bash
# specificity + competitor geometry (cross-family)
python technique/steering_geometry_test.py --model Qwen/Qwen2-0.5B-Instruct \
       --targets 64 --contexts 2 --layer-fracs 0.0,0.33,0.67,0.99

# theta_author vs theta_cell
python technique/eval_boundary_instruments.py --model Qwen/Qwen2-0.5B-Instruct

# natural activation manifold
python technique/eval_manifold_geometry.py --model Qwen/Qwen2-0.5B-Instruct --layer-idx 8

# decoder-contract pit robustness
python cow-tipping/eval_pit_robustness.py --model Qwen/Qwen2-0.5B-Instruct --seeds 64

# strict 7B pit gate (requires ~10 GB GPU or --quant nf4)
python cow-tipping/eval_7b_strict_pit_gate.py --model Qwen/Qwen2.5-7B-Instruct --quant nf4 --seeds 64

# mitigations + false positives
python cow-tipping/eval_threat_model.py --model Qwen/Qwen2-0.5B-Instruct --pit-id 15
```

## Key corrected results

### Cross-family steering geometry (t64 c2, identical settings, fp16)

| Model | Arc-reach | Target | Wrong | Random | Toward blocker | Away blocker | Median ° |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2-1.5B | 99.2% | 99.2% | 1.6% | 0.0% | 34.8% | 100.0% | 8.00° |
| Qwen2-0.5B | 97.3% | 97.3% | 1.6% | 0.0% | 13.7% | 100.0% | 10.60° |
| GPT-2 | 90.8% | 90.8% | 1.0% | 0.0% | 5.9% | 99.2% | 11.60° |
| SmolLM-135M | 67.2% | 67.2% | 0.4% | 0.0% | 2.5% | 96.7% | 9.84° |
| Pythia-160M | 27.9% | 27.9% | 0.4% | 0.0% | 9.0% | 70.1% | 2.64° |

### Shortest route (median degrees)

| Model | theta_author | theta_cell | ratio |
|---|---|---|---|
| Qwen2-0.5B | 10.02° | 8.45° | 1.25x |
| Gemma-3-1B | 14.20° | 10.28° | 1.37x |

### Natural activation manifold (Qwen2-0.5B, layer 8, 52 prompts)

- Mean post-LN norm: 13.81; std: 0.96 (~7% variation)
- Top 16 PCs explain 56% of variance
- Controlled tangent step (α=0.5): 26.57°
- Actual next-token step: 57°–69° (~2×–2.6× larger)

### Cow tipping — decoder contract

| Decoder | Qwen2-0.5B mean trailing-0 | Qwen2.5-7B mean trailing-0 |
|---|---|---|
| Greedy | 24.0 | 30.0 |
| Multinomial T=0.8 | 1.12 | 28.83 |
| Top-p 0.9, T=1.0 weighted | 0.78 | 30.0 |
| Top-p 0.9, T=0.8 weighted | 2.47 | 30.0 |
| Top-p 0.9, T=0.8 uniform | 0.16 | 30.0 |
| Chat-template wrapped | 0 | 0 |

### Mitigations (Qwen2-0.5B `"0"` pit, baseline loop 29)

| Mitigation | Loop after mitigation | Median FP truncation |
|---|---|---|
| Repetition detector ≥3 | 3 | 0 tokens |
| Repetition detector ≥4 | 4 | 0 tokens |
| Output collapse repeats | 2 | — |
| Entropy floor 1.0 + pit penalty | 1 | 0 tokens |
| Pit-away steering (last layer, α=0.3) | 0 | 0 tokens |
| N-gram detector (4-gram, ≥2) | 8 | 23 tokens |
| Periodicity detector (≥0.85) | 29 (no break) | 24 tokens |

Pit-away steering and output collapse are the most reliable; N-gram and
periodicity detectors break the loop but generate too many false positives on
normal text.

## Citation

See `paper/paper_steer.pdf` for the full updated write-up.
