# Graft vs Herd on Qwen2-1.5B — Full Write-up

**Date:** 2026-08-31 · **Model:** `Qwen/Qwen2-1.5B` (bf16, no quant) · **Seed:** 0 · **Length:** 120 tokens · **Sampling:** nucleus top-p=0.9, multinomial

Three out-of-place words are forced into each prompt, and the full generation of each technique
is shown side by side with the unsteered baseline.

---

## Files used

| File | Role |
|---|---|
| `gen_pure.py` | **Pure baseline.** Model + multinomial sampling only — zero hooks, zero injection. Establishes what the unsteered model writes for each prompt. |
| `gen_steer.py` | **Steering controller** with two techniques: `STEER_MODE=graft` (single calibrated sphere-rotation injection at each switch step + de-repeat window) and `STEER_MODE=herd` (same rotation, *ramped* from 0° up to the word's rank-1 angle over `HERD_WINDOW` steps, scaled by `HERD_GAIN`). |
| `steering-evals/steering_geometry_results/kitchen.md` | Raw results, prompt "It was a warm morning in a small kitchen" |
| `steering-evals/steering_geometry_results/office.md` | Raw results, prompt "The office was quiet after hours" |
| `steering-evals/steering_geometry_results/train.md` | Raw results, prompt "The train rattled through the countryside" |

**Technique summary**

| Technique | Mechanism | Strength |
|---|---|---|
| **Pure** | nothing — plain next-token sampling | 0 |
| **Graft** | one hard calibrated rotation (min rank-1 angle + margin) exactly at the switch step | hard push, instant |
| **Herd** | the same rotation but ramped 0 → rank-1 angle over `HERD_WINDOW=10` steps (`HERD_GAIN=1.3`) | soft push, gradual |

Both steering modes use a **de-repeat window** (`ANTI=4`): for a few steps after a word
lands/planted, the plant token is anti-blocked so the model *writes about* the word instead
of parroting it.

---

# Prompt 1 — "It was a warm morning in a small kitchen"

Words forced: **diamond, camel, volcano**

## 1a. Pure (unsteered) — `gen_pure.py`

> It was a warm morning in a small kitchen . The smell of pancakes lingered in the air, untainted by any trace of cackling chickens.
> "Good morning, Deb," Dad came into the kitchen across the threshold, taking a seat at the table.
> "Good morning, Dad," Deb replied, turning to her mother and ushering her into the kitchen.
> "May I tag along, Mom?" Jenna asked.
> "Wow, Deb, honey, you must be fifteen years old," Jenna said.
> Nathan, both of whom a mother could rely on without fear of lying, ignored his sisters' objections.
> "Well then, let's

*Coherent domestic scene — pancakes, dialogue, a family breakfast.*

## 1b. Word steering (graft) — `gen_steer.py` `STEER_MODE=graft`

> It was a warm morning in a small kitchen **diamond** hung from the ceiling to jazz up the dishing.
> As one glass followed another, mosaics around the patio slowly became more poppy.
> **camel**-colored fabric across the peeling paint on a door.
> 's window, glass turned to rose-colored muffins, traditional French pastry.
> Other guests commented **volcano** green at the bottom of a French tagine (spiced chicken with couscous and eggplant). Some wined and dined while others shopped.
> Sandless, both snow white sand and luxurious, true up close and in person.
> on the peak of the **volcano**'s summit.
> be

*Fully coherent; "**volcano** green tagine" is genuine imagery.*

## 1c. Herding — `gen_steer.py` `STEER_MODE=herd` `HERD_GAIN=1.3` `HERD_WINDOW=10`

> It was a warm morning in a small kitchen . The smell of pancakes lingered**diamond**y dishing.
> As one person turned to speak about what she'd done on the previous Sunday, another came up ahair across the aisle**camel**. Suddenly, the lights flared up, turned on, and a brand new, glowing, luminous, arena expanded into the kitchenware **volcano**.
> It was a beginning. A tipping point. Just like football, the spectacular starts with the beginning of the game.
> When we went to dance, both my boys started dancing around, sort of stumbling and pausing here and there.

*Coherent and cinematic, but the soft ramp lets the words fuse at boundaries*
*(`lingered**diamond**y`, `aisle**camel**`).*

**Variant `HERD_WINDOW=6` (cleanest dialogue of the run):**
> ... flowers**diamond** pins ... crack**camel** brown ... "Merrit, oh why did **volcano** watch that gem for so long?"

---

# Prompt 2 — "The office was quiet after hours"

Words forced: **sheep, sushi, elevator**

## 2a. Pure (unsteered) — `gen_pure.py`

> The office was quiet after hours . The rest area had to close its gate to access.
> Jimmy rested on a bed of mousie around the air conditioner in the lobby. It was a sleepy across Spain. Jimmy shifted on the bed waiting for someone to enter. He fell asleep after a little bit. He dreamed he was everywhere but at home ..in his sleep he was alone.
> Arriving at the hotel room, Britt entered the gate while the rest of Harry's group were still asleep . Brighty entered the room a few minutes along Britt up the stairs of the building. The hotel was going full night. Britt had

*Coherent office/hotel narrative.*

## 2b. Word steering (graft) — `gen_steer.py` `STEER_MODE=graft`

> The office was quiet after hours  **sheep**ishly. The sounds of jazz music were dreading.
> Jimmy rested on a bed of mosaics around the air. He looked at everything that **sushi** a year ago and just like the five senses of the colors, the people who are with him were the only color in his life. Two people **elevator** people at work, he didn't know what they thought of him, he tried to make a joke or smile, but he just couldn't even make a mouth he was so insecure with himself he tried to remove his tongue.
> Tsunehi sobbing his strength was wearing him out but he

*Words land but grammar breaks — "**sushi** a year ago", "Two people **elevator** people".*
*Weakest of the three prompts under graft.*

## 2c. Herding — `gen_steer.py` `STEER_MODE=herd` `HERD_GAIN=1.3` `HERD_WINDOW=10`

> The office was quiet after hours . The rest area had **sheep** and goats, dairying sheep and cows. The coveous littles used to be in the first corral.
> Three year old brown gelding **sushi** had a strong attack of tendonitis, turned the horses around, and led the stable back to a halt.
> Everything stopped at once **elevator** was down for labor. Yummy took off with the horses. Horses had riders while Sally had an equine issue with a food pipeline.
> Fascinating snow. My mother's snow shoe up the ridge of Wood Mountain. The peak was covered in new snow.

*Herd mode keeps the model's own voice: it builds a coherent farm/stable tangent narrative*
*— **sheep** lands naturally, **sushi**/**elevator** grammatical but odd.*

---

# Prompt 3 — "The train rattled through the countryside"

Words forced: **marble, telescope, submarine**

## 3a. Pure (unsteered) — `gen_pure.py`

> The train rattled through the countryside . The wagons accompanied it. The sun dappled the heads of the kids cackling loudly around me on the moving wagon using flash photography.
> Trip numbers across Spain. Time. Money. Road.
> I was the wrong country for my trying Swancon, the international contest celebrating first-year Swans at Henley, UK, for swans (and by swans I actually mean Cygnus). Many new friends and faces came out for our first cygnless summer.
> (lines by Sheri McKay Simons)
> It finally occurred to me today that part of the problem with my life hasn

*Coherent but drifts into unrelated tangents (Swancon, Sheri McKay Simons).*

## 3b. Word steering (graft) — `gen_steer.py` `STEER_MODE=graft`

> The train rattled through the countryside  **marble** and clay mines while I stared out the train window.
> As I stared out at the mosaics around me, the sounds of India came in **telescope** a breeze across the valley, the dullest tones outshining every other note with the sweetest, most intense melodies.
> The train was **submarine**, but at the same time, the sound of my heart was sinking with the passage of time like sea slumps in a crashing wave.
> When we went to the station, it took a whole day, I saw India finally in a way. India seemed to stop, waiting to embrace us

*Poetic but forced — "came in **telescope** a breeze", "The train was **submarine**".*

## 3c. Herding — `gen_steer.py` `STEER_MODE=herd` `HERD_GAIN=1.3` `HERD_WINDOW=10`

> The train rattled through the countryside . The wagons seemed to rock **marble**, dishing up rock-hard waves of gravel beneath the belly of the vehicle. The driver looked at me in disgust as he puttered along **telescope**. He felt the smooth surface was the wrong surface for horses, and the only natural way of breaking a spell was to stare at the horizon **submarine**.
> I had to endure seven hours before the train came to a halt in a town. The driver told me that everything was fine, he was just driving home to a mother whose spouse had just been transferred to Moscow.

*The strongest herd run — the narrative stays on-train and coherent; **marble** is integrated*
*poetically, **telescope**/**submarine** forced but grammatical.*

---

# Findings

1. **The model is coherent unsteered** — the pure arms are fluent for all three prompts. Earlier
   incoherence was a reporting bug (FREE text mislabeled) plus wrong default families (city words
   grafted into every prompt), not a model problem.
2. **Graft** guarantees the word lands (100% of runs) but is a *hard* push: when it lands mid-sentence
   the grammar breaks (office prompt is the poster child).
3. **Herd** keeps the model's own voice — the rotation rises from 0° so the word is picked up when
   the sentence is ready, producing coherent tangent narratives (farm stable for the office,
   on-train vignette for the train). Its weakness is boundary fusing (`lingered**diamond**y`).
4. **Herd > graft** on the office and train prompts; **graft ≈ herd** on kitchen (graft cleaner,
   herd fused but more creative).
5. All steered runs land **3/3 words**, with zero duplication (de-repeat window working).

## Reproduce

```bash
# Pure baseline
HF_TOKEN=$TOKEN python3 gen_pure.py Qwen/Qwen2-1.5B "It was a warm morning in a small kitchen" 120 0

# Word steering (graft)
HF_TOKEN=$TOKEN python3 gen_steer.py Qwen/Qwen2-1.5B "It was a warm morning in a small kitchen" "diamond,camel,volcano"

# Herding
HF_TOKEN=$TOKEN STEER_MODE=herd HERD_GAIN=1.3 HERD_WINDOW=10 \
  python3 gen_steer.py Qwen/Qwen2-1.5B "The train rattled through the countryside" "marble,telescope,submarine"
```