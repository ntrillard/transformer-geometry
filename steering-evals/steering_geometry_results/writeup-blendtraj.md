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

> It was a warm morning in a small kitchen . The smell of pancakes lingered in the air, so like
> one that will be wrapped up and **diamond**-edged in air. Sweet toasted poppyseed breads came
> across the butter, the cinnamon, the smooth oiliness of warm milk. God **camel**'s sweet skull,
> it was like he could talk to me at that exact moment, and give me a glimpse of what I'd like to
> eat **volcano**-tennissems. ⟨eos⟩

*(seed=1 gives a different coherent kitchen story — He ran to Sammi's door, grabbed his
flashlight, and woke her.)*

---

## 2. "The office was quiet after hours" → **sheep, sushi, elevator** — best run

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

## Reproduce

```bash
HF_TOKEN=$TOKEN python3 gen_blendtraj.py Qwen/Qwen2-1.5B "The office was quiet after hours" "sheep,sushi,elevator"
# env: LAM=0.4 SETTLE=8 HOLD_ANGLE=4 PLANT0=20 SEED=0
```