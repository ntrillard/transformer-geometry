<div align="center">
  <img src="paper/steeronasphere.png" width="180">

  # Steer on a Sphere
  ### Geometric Control of Transformer Outputs

  **N. Trillard** — August 15, 2026

  [![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21954871-blue)](https://doi.org/10.5281/zenodo.21954871)
  [![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
</div>

---

## How It Works

Every transformer layer uses RMSNorm, which forces hidden states onto a sphere of radius $\|\gamma_l\|$ — a learned value, not $\sqrt{d}$.

The LM head gives every token a direction on that sphere. **A single tangent step reaches any of them.**

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
  logits → 91–98% chance t ranks #1
```

**Why this matters:** anyone with weight access can steer generation toward any token — no training, no data, no retraining.

---

## Cow Tipping

Some tokens are self-reinforcing: feed them to the model and it repeats them forever.

| Token | Triggers a loop of... | Real-world example |
|-------|----------------------|-------------------|
| `0` (digit) | `000000...` | Phone number `000-000-0000` |
| NULL byte | `\x00\x00\x00...` | Invisible page footer |
| `cut` | `cut cut cut...` | Repeated delimiter |
| `ere` | `ereereere...` | Common substring |

**Defensive encoding:** put a pit trigger at the end of your page → any LLM scraper falls into a repetition loop. Invisible to humans (NULL bytes), catastrophic for crawlers.

---

## Key Numbers

| Measurement | Result |
|-------------|--------|
| Tokens reachable at rank 1 | 91–98% across 4 model families |
| Rank improvement guaranteed | 100% (never lowers) |
| Cow tipping permanence | 15/15 steps locked |
| Defensive encodings verified | 3 (NULL, cut, phone) |
| Edge of chaos clustering | 13 architectures mapped |

---

## Code

```bash
pip install -r requirements.txt

# Find all pits on any model
python pit_engine.py --model Qwen/Qwen2.5-7B-Instruct --scan

# Reproduce sphere steering
python steer_sphere_proof.py

# Run geometry verification suite
python sphere_test_suite.py
```

| File | Purpose |
|------|---------|
| `pit_engine.py` | Reverse-engineer pits from weights + defensive encoding |
| `steer_sphere_proof.py` | Tangent traversal on the sphere |
| `sphere_test_suite.py` | Batch geometry verification |
| `safety_toolkit.py` | Diagnostics (λ health, steer-away) |

---

## Paper

📄 [`paper/paper_steer.pdf`](paper/paper_steer.pdf) — full preprint
📝 [`paper/paper_steer.tex`](paper/paper_steer.tex) — LaTeX source

**Cite as:**

> Trillard, N. (2026). *Steer on a Sphere: Geometric Control of Transformer Outputs*. Zenodo. [10.5281/zenodo.21954871](https://doi.org/10.5281/zenodo.21954871)

---

Preprint. The geometric picture is approximate, not a theorem. Steering is a white-box traversal primitive. [CC BY 4.0](LICENSE).