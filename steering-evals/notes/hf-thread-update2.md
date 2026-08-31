# HF thread — topic-switching controller on Qwen2-1.5B (Aug 31, 2026)

Sequential multi-topic generation with a calibrated graft + continuous
herding controller. One file, at repo root: `eval_switch_big.py`
(`ntrillard/transformer-geometry`, run from root, results land in
`steering-evals/steering_geometry_results/`). Qwen2-1.5B base, bf16,
no quant, `"Tell a story"`, NTOK=64, switches at 0/16/32/48
(city→animal→food→nature), seeds 0/1.

---

## Why a controller and not a single graft

A one-shot steer to a token family gives first-token adherence but no
sequence control — the model drifts back to its own distribution
immediately. Sequential switching needs the intervention to *stay*
available per segment, and naive persistence (planting + anti-blocking
the planted token every step) produces word-salad:

```
HARD plant+block:  "berlin hotel new york / york fall gardening projects /
                    calendar for dog walking miami florida / bear air vac…"
```

Topic hits 8/8, but the text is a keyword list, not narrative.

## The free baseline is honest — and low

Every run pairs steered vs the model's own continuation (no hooks):

```
FREE seed0:  "I do not have personal preferences. however, i can provide
              a general story about being an entrepreneur…"
FREE seed1:  "I am not capable of emotions. however, i can create a story…"
```

Qwen2-1.5B-base on `"Tell a story"` collapses into a
refusal/constraint template. `FREE 0/2` quality on both seeds. The
controller's job is to win against *that*, not against a good writer.

## Three things that were actually wrong, fixed

1. **Tokenization of graft targets matters more than geometry.**
   `' tokyo'` tokenizes to `[' to','ky','o']` — the first token is the
   function word `" to"`, which corrupts *every* graft toward Tokyo
   ("to __ the area of new products"). Probe first, then swap:
   `tokyo → oslo`. Wrapping matrix (env `WRAP`): `before` (leading
   space, 1 token/word) = clean graft; `none` = fragment collapse
   (`par`→`parabole`); `after`/`both` = word splits + trailing-space
   token → digit/number lock.
2. **Static angles are wrong per context.** `best_angle()` recalibrates
   at the live hidden state: sweep θ → min angle that makes the closest
   member word rank-1, +2° margin. Grafts land real topic words
   (`berlin`, `horse`, `sushi bar`, `tree`), not whatever token
   happens to surface.
3. **Proportional herding fights the model all 64 steps.** Each drift
   step re-asserts the model's own distribution, so a constant small
   nudge never accumulates into the topic basin (correction-steps 64,
   seed-1 collapse into `:)`). Replaced with **integral control**:
   the injected angle accumulates toward the family centroid while the
   story drifts, and *releases completely* once the model emits topic
   on its own (rolling 5-token window ≥2 family tokens, or family
   probability ≥ 0.02).

## Result (hybrid SOFT mode)

```text
quality-good  STEERED 2/2   FREE 0/2
hijacks       0/4, 0/4 (no refusal/QA escape valves)
corrections   64 → 18/20 per seed
hits          topic words land at every switch
```

Seed 1 (the previously-collapsing one):

> berliners berlin, google it and see new ideas. If you londonize the
> polarising words in the headline "proder catas bear—oh dear!" sushi,
> "urban creativity" could refer to any search term you imagine: hot
> water pipes, city with deer on its street corner or Yuletide T-tree

Seed 0 (wordier but topic-complete):

> berlin workshop / Berlin berlin story. Training arts training,
> workshops… Bear sushi across chelsea… water polo players… sun people

All four planted topics appear in both seeds with no repetition
collapse and no template escape. Steering beats free on the only bar
that matters here (free = refusal template), at the cost of adjective
density — the model's own prose is still the ceiling at NTOK=64.

## Reproduce

```bash
cd /path/to/transformer-geometry
HF_TOKEN=<tok> SW_SOFT=1 SW_META=1 python3 eval_switch_big.py \
    Qwen/Qwen2-1.5B "Tell a story"
```

`SW_META=1` zeros template/refusal/QA escape-valve tokens seen in the
top-64 during the first 3 steps after a switch. Knobs: `PEN` (0.3),
`SOFT_ACC_START/STEP/MAX`, `SOFT_TARGET`, `SOFT_TARGET_ALIGN`,
`SOFT_HYST`; `WRAP=before` default.

## Scope

- Single model (Qwen2-1.5B) + one prompt so far; 3B run pending.
- "Quality" here is repetition/word-dup rate + freedom-from-template,
  not literary merit.
- The win is *over the free arm*, which is a refusal template — the
  controller makes topic-switched prose, not a great writer.
- Open: longer segments per switch (16 tokens is tight), the 3B model,
  and whether release-on-topic holds across prompt families.