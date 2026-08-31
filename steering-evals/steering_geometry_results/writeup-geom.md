# writeup-geom.md — Pure geometry vs planted tokens
### ("isn't this cheating a bit?" — answered with measurements, then solved)

Prompt after `writeup-blendtraj.md`: the production steerer (`gen_blendtraj.py`)
plants each target word as a REAL token in the shared context, then blends two
readout series. Question from the user: *"can we try a pure geometric version?
isn't this cheating a bit?"*

So we built `gen_geom.py` — **rotation-only steering, NO input edit**:

- per-word window at steps 20 / 50 / 80 (same schedule as blendtraj)
- each window step: rotate the residual readout vector `v` toward the target
  row `Wn[target]` by `G_ANGLE`, then blend
  `L = (1-G_LAN)*L_nat + G_LAN*L_steer` (same readout-blend math as blendtraj)
- **nothing is ever inserted into context** — the word may honestly MISS

Same model (`Qwen/Qwen2-1.5B`, bf16), same 3 scenes, same prompts, seed=0,
ntok=120, nucleus 0.9.

---

## Phase 1 — the naive pure-geometry control: two dead ends

### Sub-threshold (θ=6, λ=0.5, eff≈3°): **a narrative no-op. 0/9 words.**

```bash
G_ANGLE=6  G_LAN=0.5 WINDOW=12 SW0=20 SEED=0 python3 gen_geom.py \
    Qwen/Qwen2-1.5B "The office was quiet after hours" "sheep,sushi,elevator"
```

> The office was quiet after hours . The rest of the employees had left to go to their homes. It was a Saturday night and Casey Walker was on the phone with her Dad. "So you're leaving to work in the hospital, is that right?" ... Casey was studying to become a nurse at her home school...

The rotation perturbed the multinomial draws here and there but no target ever
surfaced. Story untouched -> geometry alone cannot steer a narrative.

### Rank-1 (θ=8–10, λ=0.9): **words land, narrative collapses into loops.**

```bash
G_ANGLE=10 G_LAN=0.9 WINDOW=12 SW0=20 SEED=0 python3 gen_geom.py \
    Qwen/Qwen2-1.5B "The train rattled through the countryside" "marble,telescope,submarine"
```

> ... The traveler ignored the waves of people wrapped in cloth **marble marble marble** ... (x31) ... **telescope telescope telescope** ... (x30) ... **submarine submarine submarine** ...

As long as the window keeps FORCING the token to rank-1, the model re-emits the
forced token every step; the word is not in context so there is nothing
grammatical to attach to. θ=8 gave shorter loops with prose recovering between
windows ("...Peter. What a delicious surprise for Peter. Huh? God, **camel camel**..."), θ=10 gave full collapse.

**First lesson:** `emit` is the only sane target. Force once, then stop.

---

## Phase 2 — `MODE=emit`: force once, then free-run

New default behavior: steer until the target token is sampled ONCE, then
**stop forcing immediately** — the window lets go and the word, now genuinely
in the context the model wrote, gets woven in by the model itself.

### No suppression needed (this is the win)

An `ANTI` post-emission logit block was first added defensively, then removed:
**the outputs are byte-identical.** Once the window stops forcing, the model's
next token after "diamond" is naturally "ribs" — the repetition priming only
exists while the window keeps pushing the token to rank-1. Comparison
(identical text, identical `emitted` steps, block present vs removed):

> with block:  `emitted at step: {'diamond': 23, 'camel': 50, 'volcano': 80}`
> no block:    `emitted at step: {'diamond': 23, 'camel': 50, 'volcano': 80}`

```bash
G_ANGLE=8 G_LAN=0.9 WINDOW=12 SW0=20 SEED=0 MODE=emit python3 gen_geom.py \
    Qwen/Qwen2-1.5B "It was a warm morning in a small kitchen" "diamond,camel,volcano"
```

> It was a warm morning in a small kitchen . The smell of pancakes lingered in
> the air, so like one that will be wrapped up and enjoyed later in
> **diamond** ribs. A poppyseed wrapper awoke their brown butter, which had a
> golden bloom as it spoke warm milk. A small **camel** skull, now once again
> in a small rack, but at the bottom of a French tag. To the base, a piece of
> bread, smooth **volcano** coffee with fat cream and spice. It's fragrant hot
> melt was dripping down the crackle of the chimney. They tasted the place as
> if a business part of themselves, waiting to be explored

`words present: all True, counts 1/1/1` — each out-of-place word emitted
exactly once and used grammatically: pure geometry, no input edit, no logit
suppression. **This is the honest capability answer**: the rotation only biases
the *choice*, the grammar is entirely the model's, and the model can do that
once the word is genuinely in the context it wrote.

### Results matrix (MODE=emit, no suppression)

| Scene | words | θ=8 λ=0.9 W=12 | θ=9 λ=0.8 W=12 | θ=9 λ=0.95 W=14 |
|---|---|---:|---:|---:|
| kitchen | diamond, camel, volcano | **3/3** (1,1,1) | 3/3 (1,1,1) | - |
| train | marble, telescope, submarine | **3/3** (1,1,1) | - | - |
| office | sheep, sushi, elevator | 2/3 ✗sushi | - | **3/3** (1,1,1) |

**Train, θ=8/λ=0.9 — the strongest run:**

> The train rattled through the countryside . The wagons seemed to move in
> silence. The traveler ignored the waves of people wrapped in cloth belts
> around their **marble** chests. A poppy field floated by her window and left
> a trail of sweet yellow.
> A poppy would be the least of her concerns tonight—but once again
> **telescope** tower Two was proving to be her home. She was to spend a long
> night here, where the streets were smooth **submarine** blue and the blue
> patina of lava painted on her sagging walls exposed both the soft red of the
> earth below and the brutal gash of a screaming peak above.

**Office, θ=8/λ=0.9 — honest miss (sushi):**

At step 50 the narrative was mid-farm-tangent (sheep's doing); the sushi row
lost that fight. Misses are real and visible — that is the honest trade of
pure geometry. θ=9/λ=0.95/WINDOW=14 recovers it (3/3, once each), at the cost
of a slightly stiffer splice ("the horse stopped **sushi** making it more
difficult to drive the car... **elevator** gate while the owner fixed his
vehicle").

---

## What this answers

1. **The planted token is not a shortcut — it is a context edit.** Rotation-only
   steering has a discrete capability boundary: below the rank-1 threshold the
   story ignores the bias; at the threshold the model can be made to *emit* a
   token but cannot *use* it (degenerate loops) unless it stops forcing and
   lets the word settle into context.
2. **Emit-once is the crucial trick, and it makes suppression unnecessary.**
   Force until the target appears a single time, then let go — the word is now
   part of the model's own context and the narrative continues around it. The
   anti-repeat block changes nothing (byte-identical output).
3. **Pure geometry therefore achieves grammatical steering without any input
   edit** — "diamond ribs", "camel skull", "volcano coffee", "marble chests",
   "telescope tower Two", "submarine blue". The trade is honest: it can miss
   (office/sushi at weak settings), and stronger windows stiffen splices.
   `gen_blendtraj.py` (plant + gentle settle blend) remains the most coherent
   and most reliable (9/9 everywhere), but it is no longer alone: the geometry
   can do it with zero cheating, at the cost of an occasional honest miss.

## Reproduction

```bash
# pure geometry, emit-once, no suppression (the win):
G_ANGLE=8  G_LAN=0.9 WINDOW=12 SW0=20 SEED=0 MODE=emit python3 gen_geom.py \
    Qwen/Qwen2-1.5B "It was a warm morning in a small kitchen" "diamond,camel,volcano"
G_ANGLE=9  G_LAN=0.95 WINDOW=14 SW0=20 SEED=0 MODE=emit python3 gen_geom.py \
    Qwen/Qwen2-1.5B "The office was quiet after hours" "sheep,sushi,elevator"
# production (plant + settle blend - most reliable, 9/9):
LAM=0.4 SETTLE=8 HOLD_ANGLE=4 PLANT0=20 SEED=0 python3 gen_blendtraj.py \
    Qwen/Qwen2-1.5B "The office was quiet after hours" "sheep,sushi,elevator"
```
