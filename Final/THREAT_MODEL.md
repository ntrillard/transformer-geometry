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

| Mitigation | Mean loop length | Loop broken? |
|---|---|---|
| Repetition detector (≥3 same pit) | 3 | Yes |
| Repetition detector (≥4 same pit) | 4 | Yes |
| Repetition detector (≥5 same pit) | 5 | Yes |
| Entropy floor 0.5 | 6 | Partial |
| Entropy floor 1.0 | 1 | Yes |
| Entropy floor 1.5 | 2 | Yes |
| Input sanitize repeated pit | 18 | Partial |
| Multinomial T=1.0 | 0 | Yes |
| Top-p 0.9, T=1.0 weighted | 0.94 | Yes |

### Qwen2.5-7B "0" pit (strict fixed point)

Baseline greedy loop length: 30 tokens.

| Decoder | Mean loop length | Notes |
|---|---|---|
| Greedy | 30.0 | Unbroken |
| Multinomial T=0.8 | 28.83 | Largely unbroken |
| Top-p 0.9, T=1.0 weighted | 30.0 | Unbroken |
| Top-p 0.9, T=0.8 weighted | 30.0 | Unbroken |
| Top-p 0.9, T=0.8 uniform | 30.0 | Unbroken |
| Chat-template wrapped | 0 | Broken by context |

The strict 7B pit is robust to decoder changes but fragile to prompt context.
This suggests defenses should focus on **input sanitization and prompt
engineering** for strict pits, and on **output detectors and decoder changes**
for shallower basins.

## Mitigations

### Output-side detectors

**Repetition detector.** Halt generation after `N` consecutive identical
tokens. On Qwen2-0.5B this breaks the loop with 0 median-token truncation on
ordinary prompts for thresholds 3, 4, and 5.

**Entropy monitor.** If the output distribution entropy drops below a floor,
intervene (e.g., force the second-highest logit). Entropy floor 1.0 breaks the
0.5B loop with no measured false positives on 20 normal prompts.

### Input-side sanitization

- **Collapse long runs** of suspicious sub-tokens.
- **Strip non-printable / control characters** (breaks null-byte triggers but
  not printable pits).
- **Require trailing context** after user input so the trigger is no longer
  terminal.

### Decoder changes

- Higher temperature and standard nucleus sampling break shallow repetition
  basins (Qwen2-0.5B) but not strict 7B pits.
- Uniform-over-nucleus sampling flattens in-nucleus ratios and can break some
  basins more aggressively, at the cost of output quality.

### Architecture / alignment

- **Pit-aware steering:** detect the pit direction and steer the hidden state
  away from it.
- **Supervised fine-tuning:** penalize self-prediction loops during training.

## Recommended deployment strategy

1. **Identify pits** for each deployed model (one-time scan).
2. **Deploy output detectors** (repetition + entropy) with thresholds calibrated
   on normal generation.
3. **Sanitize inputs** that are known to terminate in pit triggers, especially
   for untrusted documents.
4. **Add trailing context** to prompts when possible, since strict pits often
   break when the trigger is not the final content.
5. **Do not rely solely on sampling** for strict pits; use context and
   detection instead.

## Responsible framing

Pits are a model-weight property, not a bug in a single system. The same
mechanism can be used defensively (e.g., anti-scraping) or offensively (prompt
injection, DoS). Any mitigation should be evaluated on both loop-break rate and
false-positive rate on ordinary generation.
