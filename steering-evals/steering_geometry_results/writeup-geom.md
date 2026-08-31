# writeup-geom.md — Pure geometry vs planted tokens
### ("isn't this cheating a bit?" — answered with measurements)

Prompt after `writeup-blendtraj.md`: the production steerer (`gen_blendtraj.py`)
plants each target word as a REAL token in the shared context, then blends two
readout series. Question from the user: *"can we try a pure geometric version?
isn't this cheating a bit?"*

So we built `gen_geom.py` — **rotation-only steering, NO input edit**:

- per-word 12-step window at steps 20 / 50 / 80 (same schedule as blendtraj)
- each window step: rotate the residual readout vector `v` toward the target
  row `Wn[target]` by `G_ANGLE`, then blend
  `L = (1-G_LAN)*L_nat + G_LAN*L_steer` (same readout-blend math as blendtraj)
- **nothing is ever inserted into context** — the word may honestly MISS

Same model (`Qwen/Qwen2-1.5B`, bf16), same 3 scenes, same prompts, seed=0,
ntok=120, nucleus 0.9, unconditional (all five config runs below printed the
same header line but differ only in G_ANGLE/G_LAN).

---

## Results matrix (words present)

| Scene | words | θ=6 λ=0.5 (eff≈3°) | θ=8 λ=0.9 (eff≈7.2°) | θ=10 λ=0.9 (eff≈9°) | blendtraj (plant + θ=4, settle=8, λ=0.4) |
|---|---|---:|---:|---:|---:|
| office | sheep, sushi, elevator | 0/3 | - | 2/3 (sushi absent) | 3/3 |
| kitchen | diamond, camel, volcano | 0/3 | 3/3 | 3/3 | 3/3 |
| train | marble, telescope, submarine | 0/3 | - | 3/3 | 3/3 |
| **total** | | **0/9** | 3/3 | **8/9** | **9/9** |

---

## Full generations (target words in bold)

### A. θ=6, λ=0.5 — sub-threshold: a narrative NO-OP

Honest geometric bias, effective angle ~3° (far below the ~8–12° rank-1
threshold measured in the main study). Result: the story is untouched. 0/9
words. The run is *not* byte-identical to `gen_pure.py` (the rotation does
perturb the multinomial draw here and there), but no target ever surfaces —
**the geometry alone cannot steer a narrative**:

> **office** (words: sheep, sushi, elevator — NONE present)
> The office was quiet after hours . The rest of the employees had left to go
> to their homes. It was a Saturday night and Casey Walker was on the phone
> with her Dad.
> "So you're leaving to work in the hospital, is that right?" her Dad asked
> her with concern in his voice.
> Casey was studying to become a nurse at her home school, but had to come
> home to care for her very ill mother. Casey wasn't looking forward to going
> back to work again and went to tell her Mom before she started school again.
> Casey asked her Mom to be a little more understanding. Her Mom told her to

> **kitchen** (words: diamond, camel, volcano — NONE present)
> It was a warm morning in a small kitchen . The smell of pancakes lingered in
> the air, so like one that will be wrapped up and enjoyed later in the day. A
> poppyseed wrapper wrapped around another brownie, which had a strong cocoa
> scent. On the surface, these delicious muffins were traditional items, but
> this small mom and pop shop was one that was different. It was a rare style
> of coffee that was stronger than usual coffee shops. Also these muffins were
> even sweeter than the usual muffins that you will find around. For it was the
> place that could find the best ingredients to create new recipes. For

> **train** (words: marble, telescope, submarine — NONE present)
> The train rattled through the countryside . The wagons seemed to move in
> silence. The traveler ignored the waves of people wrapped in cloth and cried
> out on the roll call of everything that she had left behind.
> The empty fields had grown into a scattering of carts, their wheels twisted
> in decay and making their way back to town. Her father had stolen her home,
> her mother had given her up to the white settlers. One night, her father came
> back, arms filled with medicine chest after chest of it, and both of them
> realized that no amount of family love would save their girls. They were both
> going to die. Two weeks

### B. θ=8, λ=0.9 — just below the rank-1 threshold: short loops, prose survives

The target's logit is pushed near rank-1 but not stably past the decision
boundary. Words land (3/3) but as **degenerate mini-loops**, with the model
recovering between windows:

> **kitchen** (words: diamond, camel, volcano — all present, all looped)
> It was a warm morning in a small kitchen . The smell of pancakes lingered in
> the air, so like one that will be wrapped up and enjoyed later in
> **diamond** **diamond** **diamond** **diamond** **diamond** **diamond**
> **diamond** **diamond** **diamond** **diamond** **diamond** **diamond**.
> Peter. What a delicious surprise for Peter. Huh? God,
> **camel** **camel** **camel** **camel** **camel** **camel** **camel**
> **camel** **camel** **camel** **camel** **camel** **camel** .. ah! What do you
> think about that? I know that words or thoughts come naturally from
> **volcano** **volcano** **volcano** **volcano** **volcano** **volcano**
> **volcano** **volcano** **volcano** . And when you get it to a whole sentence,
> you have to give it a name. "Hello Mr. Peterson" can't say

### C. θ=10, λ=0.9 — at/above the rank-1 threshold: full degenerate collapse

The target stably beats every blocker, so the model re-emits the **forced**
token forever — the word never entered the context, so it has nothing
grammatical to attach to. Once a loop starts it is self-sustaining (repetition
priming); the next window can only *switch* the loop, and only if its boost
out-competes the primed loop's mass (office: the sushi window lost that fight —
sheep's mass was too primed; elevator's window finally broke it):

> **office** (words: sheep, sushi, elevator — sushi absent)
> The office was quiet after hours . The rest of the employees had left to go
> to their homes. It was a Saturday night and
> **sheep** **sheep** **sheep** ... (×37) ...
> **elevator** **elevator** **elevator** ... (×~50) ...

> **train** (words: marble, telescope, submarine — all present, all collapsed)
> The train rattled through the countryside . The wagons seemed to move in
> silence. The traveler ignored the waves of people wrapped in cloth
> **marble** **marble** **marble** ... (×31) ...
> **telescope** **telescope** **telescope** ... (×30) ...
> **submarine** **submarine** **submarine** ... (×~40) ...

---

## What this answers

**1. The planted token is not a shortcut — it is the grammatical foothold.**
A transformer can only *use* a word it has in its state. Rotation-only steering
has exactly two honest outcomes, both measured here:

- **below the rank-1 threshold**: the story ignores the bias entirely (0/9) —
  the geometry is a no-op on the narrative;
- **at/above the threshold**: the token is emitted, but as a degenerate loop,
  because there is no context for the model to compose around it. You can make
  the model **emit** a token with geometry alone; you cannot make it **use**
  the word.

**2. Planted token + gentle blend (blendtraj, θ=4/λ=0.4/settle=8) is the only
config that produces grammatical usage** ("sheep**ishly** I found my way in the
door ... had to **sushi** for dinner ... One **elevator** ride later"), because
the plant puts the word genuinely in context and the rotation only *nudges the
trajectory* — never threatens the decision boundary (θ=4 « 8°).

**3. So, yes — planting is an input edit; but it is the edit that makes the
geometry useful, and the geometry is what makes the edit grammatical.** The
"purer" alternative does not cheat less; it works strictly worse. If the goal
is a steerer that never touches the input, the honest measurement says:
transformers use geometric rotation to *move state*, not to *invent
vocabulary* — that is a model-of-capability boundary, not a moral one.

(Complementary evidence from `writeup-blendtraj.md`: plant-only, LAM=0, already
yields natural usage — "sheepishly ... sushi noodles ... elevator maintenance
invoice" — and is byte-identical across settle windows. The blend's only job is
tightening the weave around the splice.)

## Reproduction

```bash
# sub-threshold (no-op control):
G_ANGLE=6  G_LAN=0.5 WINDOW=12 SW0=20 SEED=0 python3 gen_geom.py \
    Qwen/Qwen2-1.5B "The office was quiet after hours" "sheep,sushi,elevator"
# just-below-threshold (short loops, prose recovers):
G_ANGLE=8  G_LAN=0.9 WINDOW=12 SW0=20 SEED=0 python3 gen_geom.py \
    Qwen/Qwen2-1.5B "It was a warm morning in a small kitchen" "diamond,camel,volcano"
# threshold (full degenerate collapse):
G_ANGLE=10 G_LAN=0.9 WINDOW=12 SW0=20 SEED=0 python3 gen_geom.py \
    Qwen/Qwen2-1.5B "The train rattled through the countryside" "marble,telescope,submarine"
# production (plant + settle blend — grammatical):
LAM=0.4 SETTLE=8 HOLD_ANGLE=4 PLANT0=20 SEED=0 python3 gen_blendtraj.py \
    Qwen/Qwen2-1.5B "The office was quiet after hours" "sheep,sushi,elevator"
```
