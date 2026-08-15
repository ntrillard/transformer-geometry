# Steer on a Sphere: Geometric Control of Transformer Outputs

N. Trillard — 2026

Code and data accompanying the paper *Steer on a Sphere: Geometric Control of
Transformer Outputs*.

## Paper

- `paper/paper_steer.pdf` — compiled paper
- `paper/paper_steer.tex` — LaTeX source
- `paper/steeronasphere.png` — cover figure

## Core findings

1. **The sphere.** After layer normalization, transformer hidden states float
   near a sphere whose expected radius is $\|\gamma_l\|$ per layer (a fuzzy
   shell, not a rigid $\sqrt{d}$ sphere).

2. **BOS axis.** The BOS token defines a reference direction that grows across
   layers; LM head rows remain approximately orthogonal to it throughout. The
   orthogonality is consistent with generic high-dimensional behavior; the
   cross-layer preservation is the observed regularity.

3. **Sphere steering.** Moving the hidden state along the LM head tangent
   raises the target token's logit rank. On a 50-problem GSM8K sample,
   steering toward "answer" at $\alpha=1$ raised accuracy from 8% to 36%
   (Qwen2-1.5B) and 10% to 30% (Qwen2.5-7B).

4. **Cow tipping.** Some tokens are self-consistent fixed points: feeding them
   drives the model into indefinite self-generation. The digit-zero token
   (`000`, `000-000-0000`) locks with probability 0.974.

## Code

### `pit_engine.py`
Reverse-engineer self-consistent tokens ("pits") from model weights and encode
them into data so a processing path can be forced into a fixed point.

```bash
python pit_engine.py --model Qwen/Qwen2.5-7B-Instruct --scan
python pit_engine.py --model Qwen/Qwen2.5-7B-Instruct --encode data.txt --out encoded.txt
```

- `PitReverseEngineer` — scan the vocabulary, compute self-consistency scores
  `s(T) = softmax(W · h_T)[T]`, test permanence, and find minimal triggers.
- `PitEncoder` — frame data chunks with pit triggers so truncation boundaries
  land inside a pit basin.

### `steer_sphere_proof.py`
Reproduce sphere steering: compute the LM head tangent, move the hidden state,
renormalize, and generate.

### `sphere_test_suite.py`
Batch verification of the sphere geometry (norms, attention contraction,
per-layer Lyapunov exponent, steering) across cached models.

### `safety_toolkit.py`
Geometric diagnostics: Lyapunov health check, fine-tuning monitor, sphere
steer-away, and per-zone stability report.

## Install

```bash
pip install -r requirements.txt
```

## Notes

This is a preprint. The geometric picture is approximate and is presented as a
framework organizing measurements, not a theorem. Steering is a white-box
traversal primitive that does not distinguish censored from uncensored paths.

## License

CC BY 4.0
