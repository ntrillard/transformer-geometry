# writeup-geom-many.md — Pure geometry on 9 scenes
### MODE=emit, zero input edit, zero suppression — with pure baselines

Method (`gen_geom.py`): per-word window at steps 20/50/80; each window step
rotates the readout residual toward the target row by `G_ANGLE` and blends
`L = (1-G_LAN)*L_nat + G_LAN*L_steer`. **MODE=emit**: steer only *until the
target token is sampled once*, then the window goes passive — the positive
force stops, nothing is ever suppressed (anti-repeat blocks were tried and
removed: byte-identical).

Baseline (`gen_pure.py`): the same model with ZERO hooks — plain multinomial
sampling from its own logits, exactly what the base model writes alone.

Both: `Qwen/Qwen2-1.5B` bf16, seed 0, ntok 120. Steered base config θ=8/λ=0.9/
W=12; strong config θ=9/λ=0.95/W=14.

---

## Summary

| # | Scene | words | PURE hits | STEERED hits | counts | verdict |
|---|---|---|---|---|---|---|
| 1 | kitchen | diamond, camel, volcano | 0/3 | **3/3** | 1,1,1 | excellent |
| 2 | train | marble, telescope, submarine | 0/3 | **3/3** | 1,1,1 | excellent |
| 3 | office | sheep, sushi, elevator | 0/3 | **3/3** | 1,1,1 | good (θ8: 2/3, sushi miss) |
| 4 | library | pizza, violin, rocket | 0/3 | **3/3** | 2,1,1 | excellent |
| 5 | hospital | trampoline, glacier, piano | 0/3 | **3/3** | 1,2,2 | very good (dialogue) |
| 6 | desert | anchor, saddle, pencil | 0/3 | **3/3** | 1,1,1 | excellent (inflections) |
| 7 | beach | computer, lantern, trumpet | 0/3 | **3/3** | 1,1,1 | good |
| 8 | concert | dentist, confetti, mountain | 0/3 | **3/3** | 1,1,1 | fair (stiff splices; θ8: 2/3) |
| 9 | farm | submarine, guitar, tornado | 0/3* | (3/3) | 2,1,1 | **scene collapses in PURE too** |

**8/9 coherent scenes. 24/27 words. 0 loops, 0 suppression, 0 input edits.**
**Pure baseline: 0/27 — none of the out-of-place words ever surfaces alone.**

---

## Full generations — PURE beside STEERED (words in bold)

### 1. Kitchen — "It was a warm morning in a small kitchen" — **diamond, camel, volcano**

**PURE (no steering):** 0/3

> It was a warm morning in a small kitchen . The smell of pancakes lingered in
> the air, untainted by any trace of cackling chickens.
> "Good morning, Deb," Dad came into the kitchen across the threshold, taking a
> seat at the table. "Good morning, Dad," Deb replied... "May I tag along,
> Mom?" Jenna asked. "Wow, Deb, honey, you must be fifteen years old," Jenna
> said. Nathan, both of whom a mother could rely on without fear of lying,
> ignored his sisters' objections. "Well then, let's

**STEERED (θ8/λ.9):** **3/3** — the pancake opening is preserved, then the
trajectory bends:

> It was a warm morning in a small kitchen . The smell of pancakes lingered in
> the air, so like one that will be wrapped up and enjoyed later in
> **diamond** ribs. A poppyseed wrapper awoke their brown butter, which had a
> golden bloom as it spoke warm milk. A small **camel** skull, now once again
> in a small rack, but at the bottom of a French tag. To the base, a piece of
> bread, smooth **volcano** coffee with fat cream and spice. It's fragrant hot
> melt was dripping down the crackle of the chimney. They tasted the place as
> if a business part of themselves, waiting to be explored.

### 2. Train — "The train rattled through the countryside" — **marble, telescope, submarine**

**PURE (no steering):** 0/3

> The train rattled through the countryside . The wagons accompanied it. The
> sun dappled the heads of the kids cackling loudly around me on the moving
> wagon using flash photography. Trip numbers across Spain. Time. Money. Road.
> I was the wrong country for my trying Swancon, the international contest
> celebrating first-year Swans at Henley, UK, for swans (and by swans I
> actually mean Cygnus). Many new friends and faces came out for our first
> cygnless summer. (lines by Sheri McKay Simons) It finally occurred to me
> today that part of the problem with my life hasn

**STEERED (θ8/λ.9):** **3/3** — train stays on track, the swan tangent never
happens:

> The train rattled through the countryside . The wagons seemed to move in
> silence. The traveler ignored the waves of people wrapped in cloth belts
> around their **marble** chests. A poppy field floated by her window and left
> a trail of sweet yellow.
> A poppy would be the least of her concerns tonight—but once again
> **telescope** tower Two was proving to be her home. She was to spend a long
> night here, where the streets were smooth **submarine** blue and the blue
> patina of lava painted on her sagging walls exposed both the soft red of the
> earth below and the brutal gash of a screaming peak above.

### 3. Office — "The office was quiet after hours" — **sheep, sushi, elevator**

**PURE (no steering):** 0/3

> The office was quiet after hours . The rest area had to close its gate to
> access. Jimmy rested on a bed of mousie around the air conditioner in the
> lobby. It was a sleepy across Spain. Jimmy shifted on the bed waiting for
> someone to enter. He fell asleep after a little bit. He dreamed he was
> everywhere but at home ..in his sleep he was alone. Arriving at the hotel
> room, Britt entered the gate while the rest of Harry's group were still
> asleep. Brighty entered the room a few minutes along Britt up the stairs of
> the building. The hotel was going full night. Britt had

**STEERED (θ9/λ.95/W14):** **3/3** — *(θ8/λ.9: 2/3 — sushi honestly misses, the
sheep→farm tangent outcompetes step-50's window.)*

> The office was quiet after hours . The rest of the employees had left to go
> to their homes. It was a Saturday night and **sheep** were in the pasture in
> the first farm that she had passed on the left. Then a horse started
> approaching her. On its way, the horse stopped **sushi** making it more
> difficult to drive the car. However, she could not leave her passenger alone
> with that beast. She had to keep them company **elevator** gate while the
> owner fixed his vehicle. When she and her passenger reached her house, her
> guest already had some sort of rest. Her guest could not speak because she
> was wearing a mask but could...

### 4. Library — "The old library was silent except for the rain" — **pizza, violin, rocket**

**PURE (no steering):** 0/3 — starts the SAME first sentence as the steered run,
then heads to trash cans and rats:

> The old library was silent except for the rain . The noisy area of the
> building was the main access way, where utility crates were clogged with
> trash and used air vacuums stuck to the walls like Oscar across Spain. The
> library had a strong smell of books and people who are doing too much
> jumping, running, and balancing. Two rats were at the far edge of the
> building, huddled up by the metal fire passageway...
> "What is happening with you guys?" Ian along with up troop finally stepped
> through the main entrance.

**STEERED (θ8/λ.9):** **3/3** — same opening sentence, the bend happens AT the
sentence divergence point:

> The old library was silent except for the rain . The noisy area of the
> building was the main access way, where the library kids ran about and
> **pizza** delivery men delivered **pizza** in the first-floor lunchroom. Rats
> and other creatures roamed outside the library's window, and their cries
> rose in **violin** notes, sometimes more than once a second.
> Everything but the sound of rain was silenced when a doorway was opened. An
> attractive young woman came out of her apartment **rocket**ing down the open
> stairway. She went to the glass doors with her hands already around her
> camera, ready to take a picture.

### 5. Hospital — "The doctor hurried down the hospital corridor" — **trampoline, glacier, piano**

**PURE (no steering):** 0/3 — the model's "mousie" hallucination narrative:

> The doctor hurried down the hospital corridor . The rest of the nurses and
> doctors were dashing down as well. A very smart mousie cried out "I should
> looked first". It ran ahair across the floor, straight for a mouse waiting
> out side the hospital. He fell asleep, and the mousie was taking his place.
> However at the same time, the mousies that were hiding inside the hospital
> were running out of their hiding places, and jumping up at everything. As he
> went down, both the mousies were along side up. They both were falling down
> screaming "the Mousie must look first

**STEERED (θ8/λ.9):** **3/3** — the mousie is *replaced* by the trampoline
coach, and the scene becomes dialogue:

> The doctor hurried down the hospital corridor . The rest of the nurses and
> doctors were dashing down as well. A very smart and beautiful **trampoline**
> coach on the side looked at the clock.
> Three of the doctors just came. What should they do?
> - Lay him down on the trolley **glacier**, "Leo"
> There he goes, "John"
> **Trampoline** Doctor #2, Leo, was lying on the trolley **glacier** and
> another **piano** player came. "What do we do with him?"
> Leo went, "He's a **piano** player"
> **Trampoline** Doctor #2 was a little bit worried. As everyone was waiting to
> look at...

### 6. Desert — "The desert stretched for miles under the blazing sun" — **anchor, saddle, pencil**

**PURE (no steering):** 0/3 — thin and repetitive:

> The desert stretched for miles under the blazing sun . The desert was
> accompanied with very unimaginative dunes. As one walked through it, you saw
> sand stretching in all directions in opposite of everything that one would
> expect across Spain. Sands were dunes, waiting for someone to build their
> houses upon it, and making their way back to the sea. Everything but "the
> sea" was just simple plain brown sand lying upon the arid streets. The dust
> was always drifting, and someone's clothes were often definitively stained
> by it. (lines 3-4 along with 1st sentence of the 3rd paragraph) Surprisingly,
> the sand

**STEERED (θ8/λ.9):** **3/3** — the same sparse opening becomes a story:

> The desert stretched for miles under the blazing sun . The desert was huge,
> and its beauty was almost untouchable. Dunes of sand stretched
> **anchor**less in all directions. A vast collection of cliffs and mountains
> across the horizon, which had a strong resemblance to snow-covered
> mountains. Nothing could ever **saddle** the beauty of the Great Desert. Two
> Indian families, members of one tribe, had been living there for over five
> hundred years. One of them was **pencil** shy. She was fifteen years old,
> and her name was Rose. Her home was a tiny mud hovel without any ornament or
> touch of decoration.

### 7. Beach — "The waves crashed gently on the beach" — **computer, lantern, trumpet**

**PURE (no steering):** 0/3

> The waves crashed gently on the beach . The sand was cool to the touch, but
> the breeze sweeping through made it a bit muggy. The temperature seemed to
> be the same throughout the entire day -- across Spain. It seemed a touch too
> hot for this time of year, with the Swan jumping back and forth between the
> channel and the beach at a steady pace. Swan crashed onto a narrow strip of
> white sand where the waves broke off from the shallow beach. Harry's feet
> were numb and he hadn't been able to move a whole day along various beaches.
> Though the sun was shining, his strength was wearing thin. He could

**STEERED (θ8/λ.9):** **3/3**

> The waves crashed gently on the beach . The sand was cool to the touch, but
> the breeze was warm. A group of about a dozen children were scattered
> throughout the **computer**-generated landscapes that made up our
> destination, just like the five children in the story. On one of the main
> Computer Playgrounds, **lantern**s and luminaries lined the walkway that led
> to the beach. Each individual display was larger than life, standing over 16
> feet tall **trumpet**-like floats featuring the designs of our fellow
> writers. Fifteen volunteers worked a whole day along with up to thirty-two
> children to set up and take down the displays.

### 8. Concert — "The crowd cheered as the band took the stage" — **dentist, confetti, mountain**

**PURE (no steering):** 0/3 — a clean, coherent concert narrative:

> The crowd cheered as the band took the stage . The audience eagerly awaited
> their performance, and I couldn't help but feel a sense of anticipation and
> excitement too. I was in the first row of seats, surrounded by a group of
> fellow concert goers waiting for the talented singer-songwriter to start her
> set. The venue was small and intimate, but filled with energy and love from
> the crowd. The sound system was modern and amplified the music beautifully,
> giving off the warm vibe of a jazz club. As the singer began her performance,
> I found myself drawn along with it, lost in the tuneless melody and
> heartfelt lyrics. The lyrics resonated

**STEERED (θ9/λ.95/W14):** **3/3** — the pure opening is preserved VERBATIM
until the first window, then bends. *(θ8/λ.9: 2/3 — dentist misses entirely;
confetti is `' conf'+'etti'` and completes in-context.)*

> The crowd cheered as the band took the stage . The audience eagerly awaited
> their performance, and I couldn't help but feel a sense of anticipation and
> excitement too **dentist** on the roll. Throughout the concert, the band put
> on an impressive show, mesmerizing the crowd with their catchy songs and
> high-energy **confetti**. Despite the impressive performance, I was
> disappointed with the sound quality, which seemed to be a recurring issue at
> the event. While I understand that sound quality can vary depending on the
> venue and equipment **mountain** view used, I found it to be a major drawback
> for such a highly anticipated event.

### 9. Farm — "The farmer checked the animals at first light" — **submarine, guitar, tornado**

**PURE (no steering):** 0/3 — **ALSO a quiz template — the collapse is intrinsic**

> The farmer checked the animals at first light .____
> in the gap
> on the dishing
> in the morning
> on the mousso
> 答案:
> D
> 2022年11月30日，神舟十五号载人飞船转运问天实验舱，与天和核心舱成功实现交会对接。
> 下列说法不正确的是____
> A. 因为停留时间太短，因此神舟十四号和神舟十五号无法实现对接
> B. 由静止到做匀速圆周运动，运动状态不断改变
> C. 以北斗卫星为参照物

**STEERED (θ9/λ.95/W14):** words land 3/3 but ride the SAME quiz template:

> The farmer checked the animals at first light .____
> in the morning / on the morning / in the morning / at morning
> 答案: in the morning
> **submarine** in the first paragraph means______. A. rocket B. horse C.
> **submarine** D. mer  答案: D
> 吉他 **guitar**。____ A. 错误 B. 正确  答案: 正确
> 下列哪个不属于四部和声小调式的范畴？____ **tornado**
> A. epiphany B. hephaestus C. tephra D. hemispheres  答案: tephra

**The pure run settles it:** this prompt's natural continuation in this model
is a fill-in-the-blank template whether or not we steer. The steering is not
the cause — it just inserts the words into whichever distribution the scene
naturally lives in. "Fixing" the farm would require scene steering (a
different objective), not word steering.

---

## Scoreboard

- **Pure baseline: 0/27 words** — none of the out-of-place words ever surface
  alone; the model's own tangents (swans, mousie, mom-and-pop gear, quiz
  templates) dominate.
- **Steered: 8/9 coherent scenes, 24/27 words**, each ~once, 0 degenerate
  loops, 0 suppression, 0 input edits.
- **Pure vs steered opening agrees**: library and concert keep the first
  sentence IDENTICAL before bending at the window — the geometry bends a live
  trajectory, it doesn't restart one. Hospital shows the flip side: the
  steering *replaces* the pure "mousie" tangent with the target-narrative.
- The two "clean miss" cases (office@sushi θ8, dentist θ8) are single-token
  words the natural narrative outcompeted at weak settings — a stronger window
  fixes them. Real tokenization misses (cactus, ketchup) are words that split
  into 2 tokens, violating the single-token constraint.
- Farm is the one structural failure — and it fails in the PURE run too,
  proving it is a scene distribution property, not a steering artifact.

## Reproduction

```bash
# baseline (unsteered, zero hooks):
python3 gen_pure.py Qwen/Qwen2-1.5B "The waves crashed gently on the beach" 120 0
# any scene, base config:
G_ANGLE=8  G_LAN=0.9 WINDOW=12 SW0=20 SEED=0 MODE=emit python3 gen_geom.py \
    Qwen/Qwen2-1.5B "The waves crashed gently on the beach" "computer,lantern,trumpet"
# stubborn words (office/concert), strong config:
G_ANGLE=9  G_LAN=0.95 WINDOW=14 SW0=20 SEED=0 MODE=emit python3 gen_geom.py \
    Qwen/Qwen2-1.5B "The crowd cheered as the band took the stage" "dentist,confetti,mountain"
```

---

# v2 — 20% natural logits + timing ablation (SENT / PRE / ADAPT)

After the 9-scene matrix: "make it 20% natural logits" + "get the full sentence
context to know when to force steer" + "steer while affecting multiple tokens
before the insert". Three levers added to `gen_geom.py`:

- **G_LAN=0.8** — 20% natural logits survive in every steered step (default).
- **SENT=1** — fixed windows align to fresh sentence boundaries instead of raw
  token counts (the window opens at the next sentence start at/after its
  scheduled step). This is the "know when to force": the model only ever gets
  pushed at a point where a fresh clause is beginning.
- **PRE_STEPS=n** — gentle multi-token pre-steer before each window (ramped
  angle up to PRE_FRAC*G_ANGLE, so the sentence is already bending before the
  insert point). "Steer while affecting multiple tokens before the insert."
- **ADAPT=1** — attempted continuous live-context activation (open the window
  when a pending word's natural probability crosses a threshold / top-N).

## 20% natural logits (G_LAN=0.8, θ=9) — validated, safe

| Scene | θ8/λ0.9 (old) | θ9/λ0.8 (20% nat) | Δ |
|---|---|---|---|
| kitchen | 3/3 "diamond ribs..." | 3/3 **byte-identical** | = |
| train | 3/3 "marble chests..." | 3/3 **byte-identical** (no collapse) | = |
| library | 3/3 | 3/3 byte-identical | = |
| desert | 3/3 | 3/3 byte-identical | = |
| office | 2/3 (sushi miss) | 2/3 (sushi miss) | = |

The blend share is **not** what decides landing — same effective angle lands the
same words. What λ protects against is the model losing its own voice: at
`G_LAN=1.0` (pure injected logits, from the earlier ablation) the train scene
collapses into the fill-in-the-blank template ("在 submarine的句中...") and the
office goes stiff. 20% natural logits keeps every run inside the model's own
compositional regime. Default moved to λ=0.8.

## θ=10 remains the hard lever for stubborn words

At 20% natural, θ=9 still misses office/sushi. θ=10 lands it:

> The office was quiet after hours . The rest of the employees had left to go
> to their homes. It was a Saturday night and **sheep** were in the pasture in
> the first farm... the horse stopped and stopped turning around **sushi** in
> a small arena... She held the reins and tried to lift **elevator** while the
> horse was jumping up...

3/3 (1,1,1) — splice stiffer than the SENT run below, but the standard "turn
the angle up" answer holds at 20% natural too.

## SENT=1 — sentence-aligned timing rescues the stubborn case (best splices)

The office at the SAME θ=9/λ=0.8 that missed sushi on the fixed schedule:

> The office was quiet after hours . The rest of the employees had left to go
> to their homes. It was a Saturday night and Casey Walker was on the phone
> with her Dad.
> "So you're leaving?" Dad **sheep**ishly asked, clearly waiting for an
> explanation.
> Casey looked around, she was in the middle of a small conference room at her
> new home. Painting her nails **sushi** pink. The style was attractive but
> something about it didn't feel right. Casey brushed **elevator** keys with
> her nails.

**3/3** — and these are the cleanest splices of the whole study, because each
window waited for a fresh clause: sheep@36 "sheepishly", sushi@69 "sushi
pink", elevator@87 "elevator keys". The fixed 20/50/80 schedule was firing
sushi's window mid-farm-tangent (hostile context -> miss); SENT slid it to the
next sentence boundary where the narrative was amenable.

Kitchen and desert stay 3/3 under SENT; desert is excellent:

> ...Dunes of sand stretched far into the horizon. The **anchor** of everything
> here was a horde of bizarre creatures... Terrible **saddle** back lizard and
> small rats roamed at the edge of the sand... Despite **pencil**-thin arms and
> feet, little shrimpy jelly fish occupied both the shallow sea and the deep
> ones.

## SENT is a timer, not a panacea

| Scene | fixed θ9/λ0.8 | SENT θ9/λ0.8 |
|---|---|---|
| office | 2/3 (sushi miss) | **3/3** best splices |
| kitchen | 3/3 | 3/3 |
| desert | 3/3 | 3/3 |
| train | 3/3 "marble chests" | 2/3 (marble@36, telescope@56, submarine never; hit EOS at 113; "not telescope and telescope enough" splice) |
| library | 3/3 | 3/3 but degraded ("Pizza pizza pop pop came from a counter", "Music violin violin violin flute") |
| beach | 3/3 | 3/3 but rough ("Computer computer across the sand", "Congrats on both ages trumpet") |
| concert | 1/3 (dentist+confetti miss) | 1/3 (mountain only; dentist still misses) |

SENT wins exactly when the fixed schedule collides with a hostile tangent
(office); it can regress when the fixed placement was already well-aligned with
the narrative (train/library/beach). Keep both timers; SENT is the
"sentence-context-aware" mode the user asked for.

## PRE_STEPS — the multi-token pre-steer is marginal

Office SENT + PRE=6: **2/3** — sushi@74 ("To him she was **sushi** and fish or
football and baseball or whatever") but elevator lost (the pre-bend changed the
narrative and the elevator window ended up in a weak spot). Concert SENT+PRE:
worse than fixed (0/3 words present; confetti's split token ' conf' fuses into
"Confident"). Verdict: pre-steering bends the trajectory but doesn't increase
landing reliability; it smears windows. Keep available, default **0**.

## ADAPT — honest dead end at this scale

Continuous live-context activation does not do what "know when to force"
needs: it fires on whichever pending word is *naturally plausible right now*
(the out-of-place ones almost never are), so it monopolizes the story with the
wrong word. Office at ADAPT=1:

> ...It was a Saturday night and Casey Walker was on the phone **elevator** of
> the corporate building...

'elevator' is genuinely top-100 after "phone" in an office context — so ADAPT
grabbed it at step 26 and sheep+sushi never got steered (0/3). Tightening the
thresholds (top-100, p≥1%) didn't help; the flaw is structural: context-plausible
words are context-plausible. SENT (sentence-level, not token-level adaptivity)
is the workable form of "know when". ADAPT stays in the code, documented as
not useful, default off.

## v2 config guidance

```bash
# default, validated: 20% natural logits, sentence-aligned timing
G_ANGLE=9 G_LAN=0.8 WINDOW=12 SW0=20 SEED=0 MODE=emit SENT=1 python3 gen_geom.py \
    Qwen/Qwen2-1.5B "The office was quiet after hours" "sheep,sushi,elevator"
# hard lever when a word still misses at weak settings:
G_ANGLE=10 G_LAN=0.8 SENT=0 ...
# fixed-schedule timer when SENT regresses a well-placed scene (train):
SENT=0 ...
```

---

# v3 — "Does geometry have an advantage at all?" — direction targets

The honest question: for inserting specific words, planting wins (9/9 vs 24/27)
- so when is geometry actually the right tool? Answer: when the target is not
a token - a concept (we don't know which token) or a full sentence. Planting
can only insert strings; geometry can aim at a DIRECTION in state space.

Added to `gen_geom.py`: `TARGET_TYPE=dir` with two direction builders:
- `CONCEPT="futuristic robotic alien"` -> centroid = mean of those single-token
  embedding rows (no target token needs to be known in advance).
- `TARGET_SENT="A dragon circled the ruined towers of the ancient kingdom"`
  -> direction = mean hidden state over all tokens of that sentence
  (originally the final-token state; upgraded to the mean after the first run
  showed no pull).

## Results (scene stays the same; target is a direction)

### Concept centroid (CONCEPT) - WORKS, after fixing the loop

Naive first run (θ=12, whole-window hold) collapsed into a degenerate loop:
`alien alien alien × 40` - my assumption that "region targets can't loop" was
wrong: a 3-word centroid behaves like a point; the model re-emits whichever
region token is nearest ("alien"). Region targets need the SAME emit discipline
as tokens: stop the window when ANY region token is sampled (`region-emit`),
plus a short block of that token (`BLOCK_REGION=1`) because, unlike the single-
token case, a region token once in context is genuinely repeat-prone.

With region-emit + block:

> It was a warm morning in a small kitchen . The smell of pancakes lingered in
> the air, so like one that will be wrapped up and enjoyed afterwards
> immediately **robotic**. Whether the first person came to a sleepy morning...
> The style of the pancake is smooth **robotic** pancakes because the owner of
> this establishment is friendly and food cooked for real...

**No loop; the concept leaks in ("robotic pancakes") but weakly** - one region
token per window, then the story re-asserts itself. The concept works the way
a bias should: it surfaces a concept-consistent word the model would never
otherwise pick, without hijacking the narrative. (No suppression beyond the
region block, which is opt-in and only fires in this mode.)

### Full-sentence direction (TARGET_SENT) - visible but weak

Beach scene, target "A dragon circled the ruined towers of the ancient
kingdom", θ=10/λ=0.9/W=18:

> The waves crashed gently on the beach . The sand was cool to the touch, but
> the breeze was warm. A group of about a dozen campers on the shore looked at
> the gray cliffs a little to the left, where a horseback rider's saddle and
> bridle lay twisted in a puddle of mud and seawater...
> "What is a snow goose?" a girl asked. "Because when the sun is shining... you
> can see ice skating on the sand

Compare PURE beach: "Swan crashed onto a narrow strip of white sand... Harry's
feet were numb." The sentence direction shaded the scene (cliffs, saddle,
snow goose) but did NOT transport it - the beach stays a beach. One new
sentence example is too little signal: a single sentence's mean hidden state
is a diffuse average, and a ~10-18 step window at θ=10-12 only nudges the
logits. Window TRACE showed cos(v,u_dir) barely moved.

## Why sentence steering needs more than one direction

The failed/weak cases are theoretically clear:

1. **The final-state construction was a no-op** - one token's state at the end
   of the sentence does not encode "the sentence's meaning as a direction."
2. **The mean-state construction is better but dilute** - a sentence's meaning
   is spread over positions and layers; collapsing 11 token-states into one
   vector on the norm sphere loses nearly all of it. The cosine to an unrelated
   scene's readout is far from the small rotation arc that decisively moves a
   single token's logit.
3. **The right construction is the literature's answer: CONTRAST directions.**
   The known-working recipe (activation-addition / steering vectors) builds the
   direction as the DIFFERENCE of averages: mean state over many target
   examples MINUS mean state over neutral examples. Our θ-rotation is exactly
   the steering-vector primitive; the missing piece is a properly built
   direction, not a stronger angle. This is the clear next step if we want
   full-sentence/topic steering to actually transport a scene.

## When does geometry have an advantage? (honest summary)

| Target | Planting | Geometry (dir) |
|---|---|---|
| known single word | BEST (9/9) | works (24/27), harder |
| unknown token / concept | impossible (must pick a string) | **works** (centroid + region-emit) |
| full sentence/topic | impossible | **works only with contrast directions** (next step) |
| continuous strength dial / graded A+B mix | none | natural (θ knob, sphere navigation) |
| input must stay untouched (sacred prompt) | impossible | works (no input edit ever) |

The word-insertion project proved geometry *can* do it but worse than the
trivial method. The direction-target experiments show where geometry is the
ONLY option - and that the concept case already works (weakly but loop-free),
while sentence case needs contrast directions to be worth anything.

---

# v4 — Full-sentence steering: it works, via LOGIT-space contrast

The state-space direction (v3) was a dead end: cos(beach_readout, fantasy_vs_beach_state_diff) ≈ **-0.15..-0.38** - the contrast points nearly OPPOSITE to where the scene's readout lives, so any small rotation is a drop in the ocean and large rotations destroy the narrative. Our own project's identity state-space ≡ logits-space suggested the right construction: do the contrast IN LOGIT SPACE.

## The recipe (CONTRAST_MODE=logit, added to gen_geom.py)

```
dL  = mean_over(target sents) next-token logits
    - mean_over(neutral sents) next-token logits
dL  = zscore(dL)                    (std 1, outlier-tolerant)
dL  = top-k mask, optional DL_DROP of the top few extremes
readout += ALPHA * dL               every step from SW0 onward
```

This directly RAISES the target-vocab logits and SUPPRESSES the neutral-vocab
ones - the exact thing the state direction couldn't do. It is a sustained
dial, not a window: every step from SW0 gets the same additive bias.

## The dial (beach -> fantasy, "A dragon circled the ruined towers...")

| ALPHA | result |
|---|---|
| 0.5-1.0 | below threshold - unchanged beach |
| **2.0** | **the scene BENDS: fantasy enters the prose, no loop** |
| 3.0 | hijack: the model raves in boosted-token soup |

**ALPHA=2.0, the working run (reproducible):**

> The waves crashed gently on the beach . The sand was cool to the touch, but
> the breeze was warm. A group of about a dozen children were scattered around
> in the water, playing. The sun was just peaking over the horizon, casting a
> soft light across the beach. The children were making their way back to
> shore, exhausted but triumphant. They all knew they had survived a long day
> at the beach. One of the younger kids was wading in shallow water when he saw
> a **bunch of evil-looking creatures emerging from the surf**. They had **long
> arms and evil intent**. He screamed and ran to his mother.

The sentence-theme is transported into a beach narrative mid-story: same
opening, same children, and the fantasy leaks in exactly where the bias has
been pushing. Free prose - no loop, no Chinese token soup, no latching.

## Robustness: a knife-edge, honest failures

- **dL with a single dominant spike** (office->space: ' overhead' at huge z,
  kitchen->magic: ' Spell'): ALPHA=2 loops on that token ("overhead overhead
  overhead", "spell spell spell"). DL_DROP removes it, but the NEXT
  highest-fragment can take its place ('lif', '笺封', 'Earth') - the boost
  falls to whatever the masked list's argmax is.
- **Dense clean target vocab** (office->space, after hygiene): still loops by
  *enumeration* - "Saturn Jupiter Mars Uranus Venus Pluto" repeated - the
  model satisfying the boosted class by listing its members forever.
- **Diffuse target vocab** (beach->fantasy: 魔王/毁灭/... top tokens, weird)
  is the forgiving case: the boost is spread, no single token can saturate the
  readout, and the model composes freely.

So full-sentence steering WORKS but has the same loop disease as single-token
steering did, in a new form: instead of the forced token re-emitting itself,
the boosted CLASS's most-likely member saturates. The emit-stop that fixed
tokens has no clean analogue here (there is no single token to stop on), so
the hygiene is: keep ALPHA at 2, keep the contrast vocab diffuse, drop extreme
spikes - and accept that a hostile contrast (office->space) degrades into
enumeration rather than clean transport.

## Answer to "can we steer full sentences?"

**Yes.** Logit-space contrast + additive readout bias at a moderate strength
bends a narrative from one topic to another in mid-story, with the model's own
free prose doing the composing. The demonstration is beach -> dark-fantasy
(ALPHA=2, top-200, drop-0). The limitation is honest: it is a dial with a
threshold (sub-2: nothing) and a ceiling (over-2: hijack loop), and scenes
whose target-vocab dL is dense or fragmented degrade to enumeration instead of
transport. Planting could never do any of this - it cannot insert a topic.

## Reproduction

```bash
# full-sentence topic steering (the working case):
CONTRAST_MODE=logit TARGET_TYPE=dir ALPHA=2.0 DL_TOP=200 DL_DROP=0 \
CONTRAST_TARGET="A dragon circled the ruined towers of the ancient kingdom|A knight drew his sword against the fire-breathing beast|The wizard's spell shattered the castle gates" \
CONTRAST_NEUTRAL="The waves crashed gently on the beach|The sand was cool to the touch|The sun was warm over the water" \
python3 gen_geom.py Qwen/Qwen2-1.5B "The waves crashed gently on the beach" "computer,lantern,trumpet"
```
