<div align="center">

# Steer on a Sphere
### Geometric Control of Transformer Outputs

**N. Trillard** — August 15, 2026

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21954871-blue)](https://doi.org/10.5281/zenodo.21954871)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

</div>

---

## About

Transformer hidden states live on a sphere, and that geometry can be *used*.

We show that layer normalization constrains hidden states to a sphere of radius $\|\gamma_l\|$ per layer, that a single tangent step reaches **any** of the 152,064 vocabulary tokens with 91–98% reliability, and that certain self-reinforcing tokens ("pits") trap the model in indefinite repetition — exploitable as a **defensive encoding** against LLM scraping.

## Findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **The Sphere** — hidden states float near a sphere of radius $\|\gamma_l\|$ | 4 model families |
| 2 | **Tangent Traversal** — one tangent step reaches 91–98% of tokens at rank 1, 100% never lower | 4 models × 1,000 tokens |
| 3 | **Cow Tipping** — self-reinforcing tokens lock into repetition ($p=0.974$, $\cos\approx1$) | 7 pits, 15-step permanence |
| 4 | **Defensive Encoding** — NULL bytes / repeated tokens collapse scrapers | 3 encodings verified |
| 5 | **Edge of Chaos** — Lyapunov rate clusters near $\lambda\cdot L\approx0.5$ | 13 architectures |

## Paper

- 📄 [`paper/paper_steer.pdf`](paper/paper_steer.pdf) — compiled PDF
- 📝 [`paper/paper_steer.tex`](paper/paper_steer.tex) — LaTeX source

**Cite as:**

> Trillard, N. (2026). *Steer on a Sphere: Geometric Control of Transformer Outputs*. Zenodo. https://doi.org/10.5281/zenodo.21954871

## Code

| File | Purpose |
|------|---------|
| `pit_engine.py` | Reverse-engineer self-consistent tokens ("pits") from weights + encode into data |
| `steer_sphere_proof.py` | Sphere steering: tangent step + hidden-state hooking |
| `sphere_test_suite.py` | Batch geometry verification (norms, contraction, Lyapunov) |
| `safety_toolkit.py` | Geometric diagnostics (λ health check, steer-away) |

```bash
pip install -r requirements.txt
python pit_engine.py --model Qwen/Qwen2.5-7B-Instruct --scan
```

## Notes

Preprint. The geometric picture is approximate and presented as a framework, not a theorem. Steering is a white-box traversal primitive that does not distinguish censored from uncensored paths.

## License

[CC BY 4.0](LICENSE)
