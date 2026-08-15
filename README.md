# Steer on a Sphere: Geometric Control of Transformer Outputs

**N. Trillard — August 15, 2026**

Code and data accompanying the paper *Steer on a Sphere: Geometric Control of Transformer Outputs*.

## Paper

- [`paper/paper_steer.pdf`](paper/paper_steer.pdf) — compiled paper
- [`paper/paper_steer.tex`](paper/paper_steer.tex) — LaTeX source

## Findings

1. **The Sphere.** Post-normalization hidden states float near a sphere of radius $\|\gamma_l\|$ per layer (not the textbook $\sqrt{d}$). The radius varies across architectures at the same dimension.

2. **Tangent Traversal.** A single tangent step along the LM head direction reaches any of the 152,064 vocabulary tokens with 91--98\% reliability and never lowers the target token's rank (100\% improvement). Verified across Qwen2.5-7B, DeepSeek-7B, Mistral-7B, and Qwen2-1.5B.

3. **Cow Tipping.** Some tokens are self-reinforcing fixed points: feeding them to the model induces indefinite self-generation. The digit-zero token (`000`, `000-000-0000`) locks with $p = 0.974$ and $\cos \approx 1.0$. Seven pits discovered on Qwen2.5-7B; the phenomenon generalizes across models with vocabulary-specific tokens.

4. **Defensive Encoding.** Pits can be embedded into data as a defense: any model that reads a chunk terminating in a pit trigger falls into a repetition loop. NULL bytes (`\x00\x00\x00`), repeated sub-tokens (`cut cut cut`), and terminal phone numbers (`000-000-0000`) are all verified to collapse generation. Because pits are derived from model weights, they can be computed for any architecture without retraining.

5. **Edge of Chaos.** A theory of perturbation dynamics: the per-layer Lyapunov exponent clusters near $\lambda \cdot L \approx 0.5$ for the Gemma, GPT-2, and mid-Qwen families (6 of 13 tested architectures), consistent with edge-of-chaos dynamics.

## Code

### `pit_engine.py`
Reverse-engineer self-consistent tokens ("pits") from model weights and encode them into data as a defensive measure.

```bash
python pit_engine.py --model Qwen/Qwen2.5-7B-Instruct --scan
python pit_engine.py --model Qwen/Qwen2.5-7B-Instruct --encode data.txt
```

- `PitReverseEngineer` — full vocabulary scan, self-consistency scores, permanence testing, isolation analysis.
- `PitEncoder` — frames data with pit triggers so truncation boundaries land inside a pit basin.

### `steer_sphere_proof.py`
Reproduce sphere steering: tangent step, renormalization, hidden-state hooking, and generation. Produces the GSM8K steering results.

### `sphere_test_suite.py`
Batch verification of the sphere geometry (per-layer norms, attention contraction, Lyapunov profiles, steering) across cached models.

### `safety_toolkit.py`
Geometric diagnostics: Lyapunov health check, fine-tuning monitor, sphere steer-away, and per-zone stability report.

## Install

```bash
pip install -r requirements.txt
```

## Notes

This is a preprint. The geometric picture is approximate and is presented as a framework organizing measurements, not a theorem.
Steering is a white-box traversal primitive that does not distinguish censored from uncensored paths.

## License

CC BY 4.0