# steering-evals — reproducible evidence for the "Steer on a Sphere" review replies

Standalone, clean, **no secrets** copy of the scripts and raw run artifacts behind the
two forum replies (`HF_STEERING_REPLY.md`, the full point-by-point answer with a
methodology line per point; `hf_post_reply.txt`, the concise paste-ready version).

## What each script does + how to run

Requirements: `torch`, `transformers` (>=4.40), `numpy`, `pandas`, `bitsandbytes` (for
`--quant int8`). Models are loaded from the HF cache (`local_files_only=False`) —
the scripts download on first use (a [`_disk_guard`](/scripts/steering_geometry_test.py) refuses
to download if free disk < 8 GB or the repo needs more than fits with a 6 GB margin).

```bash
pip install -r requirements.txt

# 1.1 endpoint identity  (tangent+renorm == same-target rotation)
python scripts/verify_identity.py                 # -> 200/200, max dev ~4e-17

# 1.2..1.10  main geometry harness (controls, families, first rank-1 angle)
python scripts/steering_geometry_test.py --model Qwen/Qwen2-0.5B-Instruct \
       --layer-fracs 0.0,0.33,0.67,0.99 --targets 64 --contexts 2
#   (any cached model; --model Qwen/Qwen2-1.5B-Instruct, openai-community/gpt2,
#    EleutherAI/pythia-160m, HuggingFaceTB/SmolLM-135M-Instruct, ...)

# 1.3, 1.6, 1.8, 1.9: boundary instruments (blocking competitor + margin; theta_cell)
python scripts/eval_boundary_instruments.py        # 48 pairs, < ~1 min

# 2.1: cow-tip repetition-loop robustness matrix (prec/quant/temp/template)
python scripts/eval_pit_robustness.py             # finds '0' id 15 loop, prints matrix

# 2.2 / 2.3: defensive-encoding mitigation eval
python scripts/eval_defense.py
```

## Methodology commitments (kept deliberately literal)

1. **Linear final-head assay.** All rank claims are `final hidden state → W_head → logits/rank`
   on bias-free LM heads (verified bias-free on the tested models), fp32 head eval,
   full vocabulary.
2. **Identity is verified, never assumed** (1.1: numeric 200/200, <1e-10).
3. **Cross-family runs share one seed (42) and identical settings**; layer indices
   are depth-adaptive fractions so different stacks are compared at the same depths.
4. **Angles are analytic** (first rank-1 crossing from the sinusoid margin
   `m_j(θ)=r(A_j cosθ + B_j sinθ)`), validated against a 200-step scan with no
   mismatches at the scan granularity; ranks are treated as non-monotone
   ("first detected crossing", not arg monotonicity).
5. **theta_cell** (shortest angle into the target's decision cone) is computed by
   active-set projection over the full vocabulary, violations rechecked every round.
6. **Robustness/mitigation numbers** come from counted generation runs on real
   models (not a formula), with precision/quantization/sampling/template as
   controlled axes, and a "normal-generation harm" baseline for the detector.
7. Reproducibility verified: each script re-ran end-to-end on a single RTX 3080
   box and reproduced the numbers in the replies (see the point→artifact map in
   `HF_STEERING_REPLY.md`).

## Layout

```
.
├── HF_STEERING_REPLY.md        # full point-by-point answer + methodology + artifact map
├── hf_post_reply.txt           # concise post-ready reply
├── README.md
├── scripts/
│   ├── verify_identity.py
│   ├── steering_geometry_test.py      # geometry harness (dtype/quant/layer-fracs knobs)
│   ├── eval_boundary_instruments.py   # blocking/margin + theta_cell
│   ├── eval_defense.py                # cow-tip scan + mitigation matrix
│   └── eval_pit_robustness.py         # robustness matrix (2.1)
└── results/
    ├── cross_family_summary.{csv,md}
    ├── boundary_margins__Qwen2-0.5B.csv
    ├── cone_theta__Qwen2-0.5B.csv
    ├── cowtip_robustness__Qwen2-0.5B.csv
    └── defense_mitigation__Qwen2-0.5B.csv
```

Security: this repo intentionally contains **no credentials**. If you fork the
parent research repo, note its git remote had a token embedded in the URL — rotate
that credential and use the CLI credential helper instead.