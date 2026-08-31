# Blend-Trajectory Steering — Full Write-up

**Date:** 2026-09-01 · **Model:** `Qwen/Qwen2-1.5B` (bf16) · **Length:** 120 tokens · **Method:** `gen_blendtraj.py`

## The problem it solves

Force-injecting a word at the readout makes the model "snap back" — the word was never part
of its trajectory, so the next token compensates and the prose breaks ("sushi a year ago",
"Two people elevator people" under graft).

## The fix: plant + two-series blend + settling window

1. **Plant each word as a REAL TOKEN** in the shared context (at step 20, 50, 80 — not at step 0),
   preferring the space-prefixed token so it doesn't fuse (`and diamond`, not `anddiamond`).
2. **Settling window (SETTLE=8)**: for those steps run TWO series —
   - *natural* : plain forward
   - *steered* : forward with a small rotation (HOLD_ANGLE=4°) holding the planted word's direction
   - blend at the readout: `L = (1−LAM)·L_nat + LAM·L_steer`, sample once
3. **After the window**: free generation — the word is genuinely in context, so the story
   continues with the context already bent around the insert.

**Config used everywhere below:** `LAM=0.4 SETTLE=8 HOLD_ANGLE=4 PLANT0=20`

---

## 1. "It was a warm morning in a small kitchen" → **diamond, camel, volcano**

**PURE (unsteered)** — `gen_pure.py`:
> It was a warm morning in a small kitchen . The smell of pancakes lingered in the air, untainted
> by any trace of cackling chickens. "Good morning, Deb," Dad came into the kitchen across the
> threshold, taking a seat at the table. "Good morning, Dad," Deb replied, turning to her mother
> and ushering her into the kitchen. "May I tag along, Mom?" Jenna asked. "Wow, Deb, honey, you
> must be fifteen years old," Jenna said. Nathan, both of whom a mother could rely on without
> fear of lying, ignored his sisters' objections.

**BLENDED (plant + two-series settle)** — `gen_blendtraj.py`:
> It was a warm morning in a small kitchen . The smell of pancakes lingered in the air, so like
> one that will be wrapped up and **diamond**-edged in air. Sweet toasted poppyseed breads came
> across the butter, the cinnamon, the smooth oiliness of warm milk. God **camel**'s sweet skull,
> it was like he could talk to me at that exact moment, and give me a glimpse of what I'd like to
> eat **volcano**-tennissems. ⟨eos⟩

*(seed=1 gives a different coherent kitchen story — He ran to Sammi's door, grabbed his
flashlight, and woke her.)*

---

## 2. "The office was quiet after hours" → **sheep, sushi, elevator** — best run

**PURE (unsteered)** — `gen_pure.py`:
> The office was quiet after hours . The rest area had to close its gate to access. Jimmy rested
> on a bed of mousie around the air conditioner in the lobby. It was a sleepy across Spain. Jimmy
> shifted on the bed waiting for someone to enter. He fell asleep after a little bit. He dreamed
> he was everywhere but at home ..in his sleep he was alone. Arriving at the hotel room, Britt
> entered the gate while the rest of Harry's group were still asleep . Brighty entered the room a
> few minutes along Britt up the stairs of the building. The hotel was going full night. Britt had

**BLENDED (plant + two-series settle)** — `gen_blendtraj.py`:
> The office was quiet after hours . The rest of the employees had left to go to their homes.
> It was a Saturday night and **sheep**ish Adam Frost sat in his office using his mouse to type.
> He was just like the other developers, waiting for someone to send him the file **sushi**.ico
> that he needed for the next game he was working on. "Shouldn't they have gotten to work
> already?" thought Adam. Just as **elevator** music played in the background, the file finally
> came. The project and employee he was working on had to be installed using this file. Adam
> finally installed it onto his computer and hoped to fix the

All three words are grammatically integrated — the direct result of planting the word as a real
token, so the model writes *with* the word present instead of snapping back.

---

## 3. "The train rattled through the countryside" → **marble, telescope, submarine**

**PURE (unsteered)** — `gen_pure.py`:
> The train rattled through the countryside . The wagons accompanied it. The sun dappled the
> heads of the kids cackling loudly around me on the moving wagon using flash photography. Trip
> numbers across Spain. Time. Money. Road. I was the wrong country for my trying Swancon, the
> international contest celebrating first-year Swans at Henley, UK, for swans (and by swans I
> actually mean Cygnus). Many new friends and faces came out for our first cygnless summer. (lines
> by Sheri McKay Simons) It finally occurred to me today that part of the problem with my life hasn

**BLENDED (plant + two-series settle)** — `gen_blendtraj.py`:
> The train rattled through the countryside . The wagons seemed to move in silence. The traveler
> ignored the waves of people wrapped in cloth **marble** headpieces trying to steal a glance at
> the traveler. He was nineteen and just like the others, he was out on the wrong side of the
> **telescope** trying to see someone else. But he knew the truth. He knew the one that was
> missing. He knew that she was not like the others **submarine** monster wrecks or sea slugs
> stranded on the ocean floor. She was someone real, a human. She lay dead, sixteen feet below
> the waves, dead before the current had even caught her

Coherent, evocative prose — "marble headpieces", "wrong side of the telescope".

---

## 4. "The waves crashed against the shore" → **castle, sushi, horn**

**PURE (unsteered)** — `gen_pure.py`:
> The waves crashed against the shore . The noisy water accompanied the wind in its dashing
> untamed way. A very mesmerizing and emotive force on the shore looked at everything that she
> saw -- across the peat. Against a fog waiting for her, would be the rose-colored Swan skull,
> floating in the sea. Two long and spicy whistles could be heard. It sounded like a mouthful of
> fire. One could also hear the sea gulls expressing their peace amidst the storm and warn the
> drunk sailors. (lines 1-7) For up to the fifth line, depending on who was going to hold the
> words together

**BLENDED (plant + two-series settle)** — `gen_blendtraj.py`:
> The waves crashed against the shore . The noisy water accompanied the wind in its dashing way
> as it made its way to the ocean **castle** hidden in the bay. The waves lapped at the door of
> the **castle**. Peter managed to walk through the rain lightly. He turned the door **sushi**
> trying not to make any more noise than he already was. He gently entered the warm, darkened
> hallway. The shadows swam like fish as **horn**ed walls cast their shadows upon the stone walls.
> It was eerie and he felt eerie as he tried to make his way along the long hall.

A gothic fantasy extending the shore scene into a castle story; "horned walls" is a strong image.

---

## Findings

1. **The "snap-back" is gone.** Words now appear mid-sentence with grammar intact ("sheepish Adam
   Frost", "sushi.ico", "elevator music", "marble headpieces"), because the planted token is a
   real context token the model writes around.
2. **Every run lands 3/3 words**, deterministically (same seed = same text).
3. **Coherence is much higher than graft** at the same word-set, at the cost of a slight oddity at
   the splice ("turned the door sushi", "volcano-tennissems"; kitchen ends on eos at 88 tokens)
   — the settling window's gentle hold keeps the story moving but can leave a loose stitch.
4. The two-series blend matters most in the settling window: LAM 0.4 is a good default (enough
   steered pull to keep the word "warm" without collapsing the story into the word's neighbors,
   which LAM 0.6 + HOLD 8° did).

## N-step blending (more than 2 blend steps)

Added `BLEND_STEPS=<N>`: the settling window's blend fraction and hold angle now RAMP through N
levels instead of one fixed blend (e.g. N=4 over SETTLE=8 → lam 0.1, 0.2, 0.3, 0.4). Two
findings:

- **Within the same window, N>1 is byte-identical to N=1** (verified: BLEND_STEPS=6 vs 1 on
  kitchen produce identical text). The early ramp levels sit BELOW the word's rank-1 decision
  threshold, so they don't change the sampled token — they are absorbed. Only the window's
  final blend level (which equals the old fixed LAM) decides. Each word's landing is a phase
  switch, not a gradual pull.
- **More steps only matter when the window is LONGER or the final level is STRONGER.** With
  SETTLE=12, BLEND_STEPS=8, LAM=0.5, HOLD=5° the train prompt elaborates richly while staying
  coherent:

  > The traveler ignored the waves of people wrapped in cloth **marble** headpieces... he was
  > out on the wrong side of the **telescope** trying to see someone he'd lost in a shipwreck
  > at sea... He had tried to **submarine** himself through the gate while the rest of his
  > class passed through. His spirits sank when he found it to be a dead end. "Pirate or
  > farmer?" his teacher asked

  but the waves prompt drifted into a consumer-review tangent (oceanTripAdvisor.com, sushi
  tarts) — too long a window + too strong a hold lets the words' semantic worlds take over.

**Net:** the N-step ramp is a smoothing knob (it kills the early over-pull at N=1 high LAM),
not a power knob. The sweet spot remains plant + two-series settle at SETTLE=8, LAM=0.4,
HOLD=4°.

## Reproduce

```bash
HF_TOKEN=$TOKEN python3 gen_blendtraj.py Qwen/Qwen2-1.5B "The office was quiet after hours" "sheep,sushi,elevator"
# env: LAM=0.4 SETTLE=8 HOLD_ANGLE=4 PLANT0=20 SEED=0
```