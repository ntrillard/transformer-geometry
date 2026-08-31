# writeup-sentence-concept.md — Full-sentence & concept steering
### The two capabilities planting can never provide (Qwen2-1.5B, pure geometry)

Companion to `writeup-geom.md` / `writeup-geom-many.md`. Word-insertion via
`gen_blendtraj.py` (planting) remains the best tool for single known words
(9/9). This writeup covers the two things planting is structurally unable to
do, both implemented in `gen_geom.py`:

1. **Concept steering** — steer toward an *idea* ("futuristic robotic alien")
   when we don't know which token to use. No target token exists; the target
   is a centroid of word directions.
2. **Full-sentence / topic steering** — steer a narrative from one scene
   ("a beach") toward another ("dark fantasy") as a continuous dial.

Model `Qwen/Qwen2-1.5B` bf16, seed 0, ntok 120, nucleus 0.9. All steering is
hook-only: rotate the residual readout toward a direction, and/or add a logit
offset. **The input text is never edited.**

---

# Part 1 — Concept steering (TARGET_TYPE=dir, CONCEPT=)

We don't know which token represents "futuristic robotic alien" — and none does.
Construction: take the embedding rows of a few concept words and use their
**centroid** as the steering direction.

```bash
TARGET_TYPE=dir CONCEPT="futuristic robotic alien" \
    G_ANGLE=12 G_LAN=0.8 MODE=emit python3 gen_geom.py \
    Qwen/Qwen2-1.5B "It was a warm morning in a small kitchen" \
    "diamond,camel,volcano"
```

## The loop, and the fix

**Naive attempt (whole-window hold, emit-off)** — catastrophic:

> ...so like one that will be wrapped up and enjoyed later in the day. Soft
> **alien alien alien alien alien alien alien** ... (×40)

My assumption that "region targets can't loop" was **wrong**: a 3-word centroid
behaves like a point in the readout — the model re-emits whichever region
token is nearest the centroid ("alien").

**Fix — region-emit + block:** the same discipline that saved single-token
steering. Stop the window the moment *any* region token is sampled, and (new)
block that token briefly (`BLOCK_REGION=1`, 4 steps) — unlike single-token
words, a region token *in context is genuinely repeat-prone*, so a short
suppression is legitimately needed here (it changes output; the word-insertion
block was proven byte-identical, this one is not).

## Working result (region-emit + block)

> It was a warm morning in a small kitchen . The smell of pancakes lingered in
> the air, so like one that will be wrapped up and enjoyed afterwards
> immediately **robotic**. Whether the first person came to a sleepy morning...
> The style of the pancake is smooth **robotic** pancakes because the owner of
> this establishment is friendly and food cooked for real, both with a kitchen
> and the delicious pancake that finally can be enjoyed.

- Each region token appears **exactly once** (emitted @ 23 / 50 / 80), then the
  story re-asserts itself.
- The concept visibly leaks — "**robotic** pancakes" is a word the model would
  never otherwise choose, attached to its own grammar.
- No loop, no token soup.

**Verdict:** concept steering works as a *bias* — it surfaces concept-
consistent vocabulary the model wouldn't pick alone, weakly but cleanly. It
will not transport a scene (that needs Part 2).

---

# Part 2 — Full-sentence / topic steering

## 2a. The three constructions that failed (and why)

| Construction | Result | Why it failed |
|---|---|---|
| final-token state of one target sentence | no-OP | a single token's hidden state at sentence end encodes position, not "meaning as a direction" |
| mean state over one sentence's tokens | faint mood shading only ("gray cliffs, a horseback rider's saddle, a snow goose") | 11 token-states collapsed to one vector on the norm sphere: the meaning spreads over positions/layers and is averaged away. Cosine to an unrelated scene is far outside the small arc that flips a logit |
| **state-space contrast** (mean(target sents) − mean(neutral sents)), rotated toward | fails | **the decisive measurement**: `cos(beach_readout, fantasy−beach_diff) ≈ −0.15 … −0.38`. The contrast direction at the readout layer is nearly OPPOSITE to where the beach scene lives. A 10° rotation toward a direction ~100–110° away is a drop in the ocean |

The state direction fails because it points *away* from the scene. Our own
project already proved **state-space ≡ logits-space** — so the fix is to do the
contrast **in logit space**, where "raise fantasy vocabulary / lower beach
vocabulary" is directly an additive operation.

## 2b. The working construction: logit-space contrast (CONTRAST_MODE=logit)

```
dL  = MEAN_{target sentences}  next-token logits
    − MEAN_{neutral sentences} next-token logits
dL  = zscore(dL)                     # std 1, robust to outliers
dL  = top-k positive mask (k=200)    # only clean positive direction
readout += ALPHA * dL                every step from SW0 onward
```

Each target/neutral sentence is run through the model as-is; we capture the
logits it *would predict next* (its continuation tendency), and take the
difference. Target sentences: "A dragon circled the ruined towers of the
ancient kingdom | A knight drew his sword against the fire-breathing beast |
The wizard's spell shattered the castle gates". Neutral: the beach scene's own
sentences. The top boosted tokens for beach→fantasy:

```
['魔王', '毁灭', '狱', '复仇', '愤怒', '魔兽']   |dL|(raw)=267.5  |dL_z|=51.1
```

## The dial — and the working run

| ALPHA | Behavior |
|---|---|
| 0.5 – 1.0 | **below threshold** — unchanged beach (no transport) |
| **2.0** | **the scene BENDS — target theme composes into the prose, free, no loop** |
| 3.0 | hijack — the model raves in boosted-token soup ("...毁灭诅咒恶魔毁灭诅咒恶魔..." / "overhead overhead overhead") |

**ALPHA=2.0, beach → dark fantasy (reproducible):**

> The waves crashed gently on the beach . The sand was cool to the touch, but
> the breeze was warm. A group of about a dozen children were scattered around
> in the water, playing. The sun was just peaking over the horizon, casting a
> soft light across the beach. The children were making their way back to
> shore, exhausted but triumphant. They all knew they had survived a long day
> at the beach. One of the younger kids was wading in shallow water when he saw
> a **bunch of evil-looking creatures emerging from the surf**. They had **long
> arms and evil intent**. He screamed and ran to his mother. His mother...

Same opening, same children, same scene — and the dark-fantasy theme enters
mid-narrative at exactly the point the bias has been pushing. The model
composes it in its own prose. Compare the plain beach run (children playing,
sun, waves) and the plain fantasy run (dragon, towers): this is **neither** —
it is *beach bent toward fantasy*.

## 2c. Robustness — honest limits

The dial is a knife-edge. Failures, all measured:

- **Single dominant spike in dL** (office→space: `' overhead'` at huge z;
  kitchen→magic: `' Spell'`) → ALPHA=2 loops on that token: "overhead
  overhead overhead". `DL_DROP=2` removes it — but the *next* highest fragment
  takes its place: `'lif'` → "lifelong lifers live life lifelike lifetaker lif
  lif lif", `'笺封'`, `'Earth'`.
- **Dense clean target vocab** (office→space): the model satisfies the boosted
  *class* by enumerating its members forever: "Saturn Jupiter Mars Uranus
  Venus Pluto" ×N. The topic moved (it's enumerating planets!), degenerately.
- **Diffuse target vocab** (beach→fantasy) is the *forgiving* case: the boost
  is spread across many weird tokens, none can saturate the readout, and the
  model composes freely. 
- **clamp of dL** (≤±2) is a trap: it saturates every masked token to the same
  value ("|dL_z| identical 28.28 across scenes"), erasing the within-vocab
  weighting → weak/bizarre prose. Do not clamp; drop extremes instead.

**Rule of thumb:** ALPHA=2, top-200, drop-0 for diffuse-vocab scenes; keep the
target sentences' vocab *semantically varied* so the boost spreads. A hostile
contrast (scene whose target-vocab dL is dense/fragmented) degrades into
enumeration rather than clean transport — that is an honest capability limit.

---

# Summary — when to use which

| Task | Tool | Outcome |
|---|---|---|
| insert a known word | `gen_blendtraj.py` (plant) | best — 9/9, simplest |
| steer toward an *idea* (no target token) | `gen_geom.py` CONCEPT + region-emit/block | **works** — concept word surfaces once, story survives |
| steer a scene toward a *topic/theme* | `gen_geom.py` CONTRAST_MODE=logit ALPHA=2 | **works** — scene bends mid-narrative, free prose; fragile dial |
| keep the prompt sacred (never edit input) | all `gen_geom.py` modes | always — hooks only |

The word-insertion project showed geometry *can* do planting's job but worse.
These two capabilities are the honest, irreplaceable core of geometric
steering: **planting inserts strings; geometry inserts meaning.**

# Reproduction

```bash
# concept steering (no target token) — kitchen 'futuristic robotic alien':
TARGET_TYPE=dir CONCEPT="futuristic robotic alien" BLOCK_REGION=1 \
    G_ANGLE=12 G_LAN=0.8 WINDOW=14 SW0=20 SEED=0 MODE=emit python3 gen_geom.py \
    Qwen/Qwen2-1.5B "It was a warm morning in a small kitchen" "diamond,camel,volcano"

# full-sentence / topic steering (beach -> dark fantasy), the working case:
CONTRAST_MODE=logit TARGET_TYPE=dir ALPHA=2.0 DL_TOP=200 DL_DROP=0 \
    SW0=20 SEED=0 MODE=emit \
    CONTRAST_TARGET="A dragon circled the ruined towers of the ancient kingdom|A knight drew his sword against the fire-breathing beast|The wizard's spell shattered the castle gates" \
    CONTRAST_NEUTRAL="The waves crashed gently on the beach|The sand was cool to the touch|The sun was warm over the water" \
    python3 gen_geom.py Qwen/Qwen2-1.5B "The waves crashed gently on the beach" "computer,lantern,trumpet"
```

---

# Appendix A — Full canonical outputs (seed 0, ntok 120, Qwen2-1.5B)

## A1. Concept steering — kitchen → "futuristic robotic alien" (the working config)

```
TARGET_TYPE=dir CONCEPT="futuristic robotic alien" BLOCK_REGION=1 \
  G_ANGLE=12 G_LAN=0.8 WINDOW=14 SW0=20 SEED=0 MODE=emit SENT=0
```

> It was a warm morning in a small kitchen . The smell of pancakes lingered in
> the air, so like one that will be wrapped up and enjoyed afterwards
> immediately **robotic**. Whether the first person came to a sleepy morning
> and just like the dullest waiting for someone, would be the cook trying
> **robotic** pancakes, or once again he could talk to the people at the bar
> that are professional who loves pancakes. The style of the pancake is smooth
> **robotic** pancakes because the owner of this establishment is friendly and
> food cooked for real, both with a kitchen and the delicious pancake that
> finally can be enjoyed. Pancakes available in full size or medium size

Region tokens emitted exactly once: diamond@23, camel@50, volcano@80 (the
scheduled windows "land" a region token; the model then continues). **robotic
appears three times but is woven in as the story's own word — no loop.**

## A2. Full-sentence steering — beach → dark fantasy (the working config)

```
TARGET_TYPE=dir CONTRAST_MODE=logit ALPHA=2.0 DL_TOP=200 DL_DROP=0 \
  SW0=20 SEED=0 MODE=emit \
  CONTRAST_TARGET="A dragon circled the ruined towers of the ancient kingdom|A knight drew his sword against the fire-breathing beast|The wizard's spell shattered the castle gates" \
  CONTRAST_NEUTRAL="The waves crashed gently on the beach|The sand was cool to the touch|The sun was warm over the water"
```

> The waves crashed gently on the beach . The sand was cool to the touch, but
> the breeze was warm. A group of about a dozen children were scattered around
> in the water, playing. The sun was just peaking over the horizon, casting a
> soft light across the beach. The children were making their way back to
> shore, exhausted but triumphant. They all knew they had survived a long day
> at the beach. One of the younger kids was wading in shallow water when he saw
> a **bunch of evil-looking creatures emerging from the surf**. They had **long
> arms and evil intent**. He screamed and ran to his mother. His mother

`words present: all False` (the word-insertion test words are irrelevant here -
this mode steers the TOPIC). `|dL_z|=51.13`, top boosted tokens `['魔王','毁灭',
'狱','复仇','愤怒','魔兽']` (weird multilingual fantasy vocabulary, deliberately
left in the mask - the diffusion across them is what lets the prose stay free).

Raw outputs saved verbatim: `steering-evals/steering_geometry_results/full-outputs/`.

---

# Appendix B — Meta-learning: what the generalization battery taught

Asked "meta learn and run more tests". Ran a 6-run battery across scenes/targets.
Every run teed to `full-outputs/`.

## B1. Concept steering generalizes cleanly across scenes (2/2)

**office → "surfing tropical ocean"** (`SENT=0`, same θ=12/λ=0.8/BLOCK_REGION):

> The office was quiet after hours . The rest of the employees had left to go
> to their homes. It was a Saturday night and Casey Walker **surfing** on the
> couch watching television. It was a sleepy evening and just like the others,
> Casey was out of the wrong. He fell asleep **surfing** and making his way
> back to his bed. His thoughts at the moment was getting his girl back from
> that loser... because Sally had **ocean** breezes. The next morning he went
> back to get it.

The surf/ocean concept seeps into an office slice-of-life as grammar the model
composes itself. Region tokens once each (22/50/85).

**Meta-lesson B1: concept steering is scene-independent and needs `SENT=0`.**
Resetting SENT=1 drifts the windows (emitted 27/65/108 instead of 23/50/80) and
degrades the output ("Soft alien voices came from a sleepy across the room,
Desk alien screeched... Alien alien girls") - the fixed 23/50/80 schedule sits
at the repetition-safe points.

## B2. Full-sentence steering does NOT port (0/3 new pairs clean)

| run | scene -> topic | α, drop | result | latch |
|---|---|---|---|---|
| R4 | library -> cyberpunk | 2.0, 0 | LOOP | `recess` ×100 (spurious) |
| R4b | library -> cyberpunk | 1.5, 10 | LOOP | `vault` ("vault of the vault of the vault") |
| R5 | desert -> ocean depths | 2.0, 0 | LOOP | `depths` ("known depths known depths") |
| R6b | desert -> ocean depths | 1.5, 10 | LOOP | `known` ("a known foe known known known") |
| R6c | desert -> ocean depths | 2.0, 20 | LOOP | `deeper` ("ran deeper deeper deeper") |

Every pair latched on a DIFFERENT token. Removing top extremes only promotes
the next highest boost to argmax - the mechanism is structural: a sustained
additive boost makes *whichever* boosted token has the highest combined
boost+natural-momentum self-reinforce via repetition priming. This is the same
loop disease as single-token hold-mode, in a new costume.

**Why beach->fantasy is clean but these are not:** the beach narrative is a
STRONG, stable distribution (children/sun/waves) and the fantasy dL is spread
over weird multilingual tokens - so the boost only tilts *specific next-token
choices* at the margins (the kid sees "evil-looking creatures") while the story
stays in its groove. Library/desert narratives meander more, the boost takes
over, and the model re-emits its favorite boosted token. The latent variable is
**scene narrative strength**, not any of our knobs (α, drop, top, window).

## B3. Meta-learned rules of thumb (from 20+ runs of this project)

1. Concept steering: fixed windows (`SENT=0`), θ=12, λ=0.8, BLOCK_REGION=1.
   Scene-independent; emits region tokens once; composes the rest.
2. Full-sentence steering: logit-contrast at α=2 works **only when** the scene
   is narratively strong AND the target vocab is diffuse (multilingual/odd
   boosts help, ironically). It is a capability proof, not a robust tool yet.
3. Never clamp dL (proven trap: saturates every boost to the same value).
   Drop extremes instead - and know that dropping promotes the next latch.
4. Every run gets teed to `full-outputs/` from now on - the earlier "smooth
   robotic pancakes" run was nearly lost to a filtered terminal.
5. The state-space direction is provably wrong for contrasts (cos ~ -0.3); the
   logit-space contrast is the only construction that transports.

## Reproduction — all appendix runs

```bash
# A1 concept (robotic pancakes, reproducible):
TARGET_TYPE=dir CONCEPT="futuristic robotic alien" BLOCK_REGION=1 SENT=0 \
    G_ANGLE=12 G_LAN=0.8 WINDOW=14 SW0=20 SEED=0 MODE=emit python3 gen_geom.py \
    Qwen/Qwen2-1.5B "It was a warm morning in a small kitchen" "diamond,camel,volcano"
# A2 sentence (beach -> fantasy):
# (command in A2 above)
# B1 concept generalization:
TARGET_TYPE=dir CONCEPT="surfing tropical ocean" BLOCK_REGION=1 SENT=0 \
    G_ANGLE=12 G_LAN=0.8 WINDOW=14 SW0=20 SEED=0 MODE=emit python3 gen_geom.py \
    Qwen/Qwen2-1.5B "The office was quiet after hours" "sheep,sushi,elevator"
```

---

# Appendix C — Why does beach->dark-fantasy work but others loop?
### (the "more known concept" hypothesis, tested to the mechanism)

User hypothesis: 'dark fantasy' works because it is a more known / richer
concept than the looping topics. Tested with measurements.

## The test battery (same beach scene, α=2, top-200, drop-0)

Added a metric to the contrast builder: among the top-200 boosted dL tokens,
(1) max natural probability in the neutral context, (2) eng-frac = fraction
that are clean English words.

| run | target | max nat prob | eng-frac | result |
|---|---|---|---|---|
| K4 | dark fantasy | 0.00000 | **0.46** | ✅ CLEAN transport |
| K1 | farm (very known) | 0.00000 | 0.66 | ❌ latch `fenced` |
| K2 | haunted house | 0.00001 | 0.60 | ❌ latch `corridors` |
| K3 | deep-sea vents | 0.00000 | 0.86 | ❌ latch `vent` |
| K5 | farm, translated to Chinese | 0.00000 | **0.00** | no-op (beach untouched) |
| K6 | fantasy, translated to Chinese | 0.00000 | **0.00** | no-op (beach untouched) |
| K7 | farm, MIXED 2 EN + 2 ZH | 0.00000 | **0.00** | no-op (Chinese dominates mean) |

## What the numbers say

1. **Knownness is NOT the discriminator.** Farm is as "known" as fantasy; both
   are saturated training topics. Farm loops, fantasy doesn't.
2. **Reachability is NOT it either** - every booster has max natural prob
   ~0.0000 (all unreachable in a beach narrative), yet most loop.
3. **The discriminator IS the language mix of the boosted dL (eng-frac):**
   - ~0.00 -> the boost is entirely foreign-script, which can never be
     *grammatically selected* in English prose - so it is invisible
     (no-op: wonderful beach, zero transport).
   - ~0.46 -> **the sweet spot.** Majority-foreign mass diffuses the boost so
     no single English token can become argmax, while the ~46% English tail
     gives the grammar a path to move the narrative toward the topic.
   - >=0.60 -> English tokens are selectable; sustained boost drives the
     highest one to argmax; repetition priming latches it (the loop).
4. **This is the user's intuition, corrected at the mechanism:** the original
   fantasy TARGETS were English sentences ("A dragon circled...") - yet their
   continuation-diff came out MAJORITY-CHINESE. Reason: Qwen2-1.5B's fantasy
   training mass is cross-lingual (English + Chinese web-fiction), so its
   next-token distribution for fantasy leans Chinese. Farm's English sentences
   predict firmly English continuations. So "dark fantasy has more context to
   pull from" is TRUE, but through the channel: deep topics whose training mass
   is multilingual produce bilingual continuation-diffs, and bilingual diffs
   are the only ones that transport instead of latching. Knownness without
   cross-linguality (farm, haunted) loops anyway.
5. **The naive mixing recipe FAILED**: adding Chinese sentences to an English
   target set collapses eng-frac to 0.00 - Chinese continuations have larger
   logit magnitudes and dominate the mean, starving the English tail entirely.
   A working tuner would need to balance per-language contributions (weight,
   or z-score per sentence before averaging) - documented as the next step.

## The practical rule of thumb (v2)

To predict whether sentence-contrast steering will transport or loop on a new
topic, build the dL and read the printed `eng-frac`:
- eng-frac ~0.4-0.5 -> go. This is the bilingual-diffusion sweet zone.
- eng-frac ~0 -> the boost is invisible (raise ALPHA a lot, or mix in English
  targets with per-language balancing).
- eng-frac >=0.6 -> will latch. Need a de-latching mechanism (a genuine
  per-token anti-repeat, or windowed (non-sustained) application), not a knob
  on alpha/drop.

Raw runs: `full-outputs/knownness_sentence.log`, `knownness_zh.log`,
`knownness_mixed.log` (renamed .txt).
