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
   rank 1 exactly 0% of the time.
3. **Competitor geometry** dominates off-arc behavior: at fixed norm and fixed
   target score, residual direction toward or away from the strongest blocker
   determines whether the target wins the rank competition.
4. **Shortest route** is modelable: projection onto the raw LM-head rank-1 cone
   gives shorter routes than the target-tangent arc, with an internal
   certificate (`theta_cell <= theta_author`).

## Folder layout

```
Final/
├── paper/
│   ├── paper_steer.pdf       # compiled updated paper
│   └── paper_steer.tex       # LaTeX source
├── technique/
│   ├── steering_geometry_test.py      # specificity + competitor geometry + arc angles
│   ├── eval_boundary_instruments.py   # theta_author vs theta_cell
│   └── verify_identity.py             # endpoint-identity unit check
├── cow-tipping/
│   ├── eval_pit_robustness.py         # decoder-contract robustness matrix
│   ├── eval_7b_strict_pit_gate.py     # small strict-pit gate for Qwen2.5-7B
│   ├── eval_defense.py                # defensive-encoding baseline
│   └── eval_threat_model.py           # mitigations + false-positive eval
├── results/
│   ├── Qwen--Qwen2-0.5B-Instruct__t64c4_lf0-0.33-0.67-0.99_fp16.csv
│   ├── Qwen--Qwen2-1.5B-Instruct__t64c4_lf0-0.33-0.67-0.99_fp16.csv
│   ├── google--gemma-3-1b-it__t128c4_lf0-0.33-0.67-0.99_fp16.csv
│   ├── cone_theta__Qwen--Qwen2-0.5B-Instruct.csv
│   ├── cone_theta__google--gemma-3-1b-it.csv
│   ├── pit_robustness__Qwen--Qwen2-0.5B-Instruct.csv
│   ├── pit_robustness__google--gemma-3-1b-it.csv
│   ├── strict_pit_gate__Qwen--Qwen2.5-7B-Instruct.csv
│   ├── threat_model_mitigations__Qwen--Qwen2-0.5B-Instruct.csv
│   └── threat_model_false_positives__Qwen--Qwen2-0.5B-Instruct.csv
└── THREAT_MODEL.md
```

## Quick reproduction

```bash
# specificity + competitor geometry
python technique/steering_geometry_test.py --model Qwen/Qwen2-0.5B-Instruct --targets 64 --contexts 4 --layer-fracs 0.0,0.33,0.67,0.99

# theta_author vs theta_cell
python technique/eval_boundary_instruments.py --model Qwen/Qwen2-0.5B-Instruct

# decoder-contract pit robustness
python cow-tipping/eval_pit_robustness.py --model Qwen/Qwen2-0.5B-Instruct --seeds 64

# strict 7B pit gate (requires ~10 GB GPU or --quant nf4)
python cow-tipping/eval_7b_strict_pit_gate.py --model Qwen/Qwen2.5-7B-Instruct --quant nf4 --seeds 64

# mitigations + false positives
python cow-tipping/eval_threat_model.py --model Qwen/Qwen2-0.5B-Instruct --pit-id 15
```

## Key corrected results

### Steering geometry (target rank-1 rates)

| Model | Target tangent | Wrong target | Random tangent | Toward blocker | Away blocker |
|---|---|---|---|---|---|
| Qwen2-0.5B | 95.8% | 0.0% | 0.0% | 15.9% | 100.0% |
| Qwen2-1.5B | 99.0% | 0.0% | 0.0% | 37.0% | 100.0% |
| Gemma-3-1B | 37.7% | 0.0% | 0.0% | 1.9% | 48.8% |

### Shortest route (median degrees)

| Model | theta_author | theta_cell | ratio |
|---|---|---|---|
| Qwen2-0.5B | 10.02° | 8.45° | 1.25x |
| Gemma-3-1B | 14.20° | 10.28° | 1.37x |

### Cow tipping — decoder contract

| Decoder | Qwen2-0.5B mean trailing-0 | Qwen2.5-7B mean trailing-0 |
|---|---|---|
| Greedy | 24.0 | 30.0 |
| Multinomial T=0.8 | 1.12 | 28.83 |
| Top-p 0.9, T=1.0 weighted | 0.78 | 30.0 |
| Top-p 0.9, T=0.8 weighted | 2.47 | 30.0 |
| Top-p 0.9, T=0.8 uniform | 0.16 | 30.0 |
| Chat-template wrapped | 0 | 0 |

The 0.5B loop is a decoder-dependent repetition basin; the 7B strict pit
survives greedy and standard nucleus sampling and is broken only by the chat
template context.

## Citation

See `paper/paper_steer.pdf` for the full updated write-up.
