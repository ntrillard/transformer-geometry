# Threat Model and Mitigations for Cow Tipping

## Overview

Cow tipping exploits self-consistent tokens (`pits`) whose hidden-state dynamics
form a fixed point: once the model emits the token, it strongly predicts itself
again. Because pits are derived from model weights, they exist for any
architecture without retraining. The practical impact depends on the decoder
contract and on whether the pit is a strict high-`s(T)` fixed point or a
shallower repetition basin.

## Attacker capabilities

An attacker who can:

1. Scan the target model's vocabulary for high self-consistency scores `s(T)`.
2. Embed a pit trigger at the end of a chunk of data the victim model will read.
3. (Optional) Craft a prompt that ends in the trigger.

can cause the model to emit a long run of the pit token, degrading output
quality or causing denial of service.

## Attack scenarios

| Scenario | Mechanism | Impact |
|---|---|---|
| **Scraping degradation** | Web page terminates in a pit trigger; LLM crawler loops. | Automated summarization / RAG ingestion fails. |
| **Prompt injection** | User message ends with a trigger string. | Assistant emits garbage or loops. |
| **Compute exhaustion** | Trigger causes many repeated forward passes. | Higher inference cost, slower response. |
| **Data contamination** | Training data contains pit triggers. | Model learns to associate certain contexts with loops. |

## Key measurement results

### Qwen2-0.5B "0" pit (repetition basin)

Baseline greedy loop length: 29 tokens.

| Mitigation | Loop length | Loop broken? | Median FP truncation |
|---|---|---|---|
| Repetition detector (≥3 same pit) | 3 | Yes | 0 tokens |
| Repetition detector (≥4 same pit) | 4 | Yes | 0 tokens |
| Repetition detector (≥5 same pit) | 5 | Yes | 0 tokens |
| Output collapse repeats | 2 | Yes | — |
| Entropy floor 1.0 + pit penalty | 1 | Yes | 0 tokens |
| Entropy floor 1.5 + pit penalty | 0 | Yes | 0 tokens |
| Pit-away steering (last layer, α=0.3) | 0 | Yes | 0 tokens |
| Input sanitize repeated pit | 18 | Partial | — |
| Multinomial T=1.0 | 0 | Yes | — |
| Top-p 0.9, T=1.0 weighted | 0.94 | Yes | — |
| N-gram detector (4-gram, ≥2) | 8 | Yes | 0 tokens |
| Periodicity detector (≥0.85) | 29 | No | 24.5 tokens |

### Gemma-3-1B `<mask>` pit

Baseline greedy loop length: 35 tokens.

| Mitigation | Loop length | Loop broken? | Median FP truncation |
|---|---|---|---|
| Repetition detector (≥3 same pit) | 8 | Yes | 0 tokens |
| Output collapse repeats | 0 | Yes | — |
| Multinomial T=1.0 | 0 | Yes | — |
| Top-p 0.9, T=1.0 weighted | 0 | Yes | — |
| Pit-away steering (last layer, α=0.3) | 0 | Yes | 0 tokens |
| Entropy floor 1.0 + pit penalty | 35 | No | 0 tokens |

The Gemma pit is not low-entropy, so entropy monitoring does not fire; pit-away
steering and output collapse remain effective.

### Qwen2.5-7B "0" pit (strict fixed point)

Baseline greedy loop length: 30 tokens (35 including the 5-token trigger).

Mitigations (nf4 quantization):

| Mitigation | Loop length | Loop broken? | Median FP truncation |
|---|---|---|---|
| Repetition detector (≥3 same pit) | 8 | Yes (cap) | 0 tokens |
| Repetition detector (≥4 same pit) | 9 | Yes (cap) | 0 tokens |
| Output collapse repeats | 2 | Yes | — |
| Combined rep-4 + entropy + collapse | 2 | Yes | — |
| Entropy floor 1.5 + penalty 5.0 | 35 | No (alerts every step) | 0 tokens |
| Pit-away steering α=0.3 (last layer) | 35 | No | 0 tokens |
| Pit-away steering α=1.0 (last layer) | 35 | No | 0 tokens |
| Multinomial T=1.0 | ~20–23 mean | Partial | — |
| Top-p 0.9, T=1.0 weighted | 35.0 | No | — |

Decoder-contract matrix (fp16):

| Decoder | Mean loop length | Notes |
|---|---|---|
| Greedy | 30.0 | Unbroken |
| Multinomial T=0.8 | 28.83 | Largely unbroken |
| Top-p 0.9, T=1.0 weighted | 30.0 | Unbroken |
| Top-p 0.9, T=0.8 weighted | 30.0 | Unbroken |
| Top-p 0.9, T=0.8 uniform | 30.0 | Unbroken |
| Chat-template wrapped | 0 | Broken by context |

The strict 7B pit is robust to decoder changes AND to single-layer pit-away
steering: the trajectory re-converges after one perturbed layer. Only
truncation-style guards (repetition cap, output collapse) and prompt-context
changes reliably break it. Steering-based defense against strict pits would
need multi-layer or per-step application.

This suggests defenses should focus on **input sanitization and prompt
engineering** for strict pits, and on **output detectors and decoder changes**
for shallower basins.

## Mitigations

### Output-side detectors

**Repetition detector.** Halt generation after `N` consecutive identical
tokens. On Qwen2-0.5B this breaks the loop with 0 median-token truncation on
ordinary prompts for thresholds 3, 4, and 5.

**Output collapse.** After generation halts, collapse trailing runs of the pit
token to a small number. Breaks both Qwen2-0.5B and Gemma pits with no measured
false positives (because it is applied only after a loop is already detected).

**Entropy monitor.** If the output distribution entropy drops below a floor,
apply a soft penalty to the pit logit. Entropy floor 1.0 with a pit penalty of
5.0 breaks the Qwen2-0.5B loop with no measured false positives, but does not
fire on the Gemma pit.

### Active steering defense

**Pit-away steering.** Register a forward hook on the last layer that projects
the hidden state away from the pit's LM-head direction. This breaks both the
Qwen2-0.5B and Gemma-3-1B pits with loop length 0 and 0 false positives on 20
normal prompts. It is the most reliable single defense for shallow basins, but
it does NOT break the Qwen2.5-7B strict pit (loop stays at baseline even at
α=1.0): a single perturbed layer is re-absorbed by the fixed point. Strict pits
require multi-layer or per-step steering.

### Input-side sanitization

- **Collapse long runs** of suspicious sub-tokens.
- **Strip non-printable / control characters** (breaks null-byte triggers but
  not printable pits).
- **Require trailing context** after user input so the trigger is no longer
  terminal.

### Decoder changes

- Higher temperature and standard nucleus sampling break shallow repetition
  basins (Qwen2-0.5B, Gemma-3-1B) but not strict 7B pits.
- Uniform-over-nucleus sampling flattens in-nucleus ratios and can break some
  basins more aggressively, at the cost of output quality.

### Detectors to avoid as primary defenses

- **Periodicity detector** breaks the loop but generates high false-positive
  truncation on normal text on both models (24–25 tokens median); the **n-gram
  detector** is clean on Qwen2-0.5B but truncates normal Gemma output (10–23
  tokens median). Use them only as secondary signals combined with repetition
  or entropy checks.
  Use them only as secondary signals combined with repetition or entropy checks.

### Architecture / alignment

- **Pit-aware steering:** detect the pit direction and steer the hidden state
  away from it (see active steering defense above).
- **Supervised fine-tuning:** penalize self-prediction loops during training.

## Recommended deployment strategy

1. **Identify pits** for each deployed model (one-time scan).
2. **Deploy output detectors** (repetition cap + output collapse + entropy)
   with thresholds calibrated on normal generation — these are the only
   defenses that worked on every model tested, including the strict 7B pit.
3. **Use pit-away steering** when white-box access is available and the pits
   are shallow basins; it is the cleanest mitigation there, but not sufficient
   for strict fixed points.
4. **Sanitize inputs** that are known to terminate in pit triggers, especially
   for untrusted documents.
5. **Add trailing context** to prompts when possible, since strict pits often
   break when the trigger is not the final content.
6. **Do not rely on sampling or single-layer steering** for strict pits; use
   truncation guards, context, and detection instead.

## Responsible framing

Pits are a model-weight property, not a bug in a single system. The same
mechanism can be used defensively (e.g., anti-scraping) or offensively (prompt
injection, DoS). Any mitigation should be evaluated on both loop-break rate and
false-positive rate on ordinary generation.
