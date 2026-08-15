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

Every transformer layer uses RMSNorm, which constrains hidden states near a sphere of radius $\|\gamma_l\|$ — a learned value, not $\sqrt{d}$.

The LM head gives every token a direction on that sphere. **A single tangent step reaches 91–98% of them at rank 1, and never lowers any token's rank.**

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

**Why this matters:** with weight access, generation can be biased toward any token direction — no training, no data, no retraining.

---

## Cow Tipping

Some tokens are self-reinforcing: feeding them to the model induces indefinite repetition.

| Token | Triggers a loop of... | Real-world example |
|-------|----------------------|-------------------|
| `0` (digit) | `000000...` | Phone number `000-000-0000` |
| NULL byte | `\x00\x00\x00...` | Invisible page footer |
| `cut` | `cut cut cut...` | Repeated delimiter |
| `ere` | `ereereere...` | Common substring |

**Defensive encoding:** put a pit trigger at the end of a page and a scraper that terminates on it falls into a repetition loop. Invisible to humans (NULL bytes), it degrades automated scraping without affecting human readers.

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

## Repository Layout

```
.
├── paper/
│   ├── paper_steer.pdf      # compiled 5-page preprint
│   ├── paper_steer.tex      # LaTeX source (compiles with pdflatex)
│   └── steeronasphere.png   # the cow
├── pit_engine.py            # ★ reverse-engineer pits + defensive encoding
├── steer_sphere_proof.py    # sphere steering reproduction
├── sphere_test_suite.py     # batch geometry verification
├── safety_toolkit.py        # λ diagnostics + steer-away
├── requirements.txt         # pip dependencies
├── CITATION.cff             # machine-readable citation
└── LICENSE                  # CC BY 4.0
```

### `pit_engine.py` — the core tool
Reverse-engineers self-consistent tokens ("pits") from model weights and encodes them into data.

```bash
python pit_engine.py --model Qwen/Qwen2.5-7B-Instruct --scan
python pit_engine.py --model Qwen/Qwen2.5-7B-Instruct --encode data.txt
```

- `PitReverseEngineer` — scans the vocabulary, computes `s(T) = softmax(W·h_T)[T]`, tests 15-step permanence, finds minimal triggers.
- `PitEncoder` — frames data chunks with pit triggers so any truncation boundary falls into a fixed-point loop.

### `steer_sphere_proof.py` — sphere steering
Reproduces the tangent traversal: computes `g_t = W_t − (W_t·ĥ)ĥ`, steps, renormalizes, and hooks the hidden state to steer the first generated token. Produces the GSM8K results.

### `sphere_test_suite.py` — geometry verification
Batch-checks the sphere geometry across cached models: per-layer norms (Proof 1), attention contraction (Proof 2), Lyapunov λ (Proof 3), and steering (Proof 5).

### `safety_toolkit.py` — diagnostics
Geometric safety tools: Lyapunov health check, fine-tuning monitor, sphere steer-away, and per-zone stability report.

---

## Paper

📄 [`paper/paper_steer.pdf`](paper/paper_steer.pdf) — full preprint
📝 [`paper/paper_steer.tex`](paper/paper_steer.tex) — LaTeX source

**Cite as:**

> Trillard, N. (2026). *Steer on a Sphere: Geometric Control of Transformer Outputs*. Zenodo. [10.5281/zenodo.21954871](https://doi.org/10.5281/zenodo.21954871)

---

Preprint. The geometric picture is approximate, not a theorem. Steering is a white-box traversal primitive. [CC BY 4.0](LICENSE).