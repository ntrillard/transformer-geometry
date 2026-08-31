# Prompt: "It was a warm morning in a small kitchen"

Words: **diamond, camel, volcano** (out-of-place in a kitchen)
Run: seed=0, ntok=120, NUCLEUS=0.9

---

## PURE (unsteered) — produced by `gen_pure.py`

> It was a warm morning in a small kitchen . The smell of pancakes lingered in the air, untainted by any trace of cackling chickens.
> "Good morning, Deb," Dad came into the kitchen across the threshold, taking a seat at the table.
> "Good morning, Dad," Deb replied, turning to her mother and ushering her into the kitchen.
> "May I tag along, Mom?" Jenna asked.
> "Wow, Deb, honey, you must be fifteen years old," Jenna said.
> Nathan, both of whom a mother could rely on without fear of lying, ignored his sisters' objections.
> "Well then, let's
> (120 tokens, eos=NO)

---

## STEERED graft — produced by `gen_steer.py` (STEER_MODE=graft, default)

Single calibrated rotation at each switch step + de-repeat window (ANTI=4). Words present: all True.

> It was a warm morning in a small kitchen **diamond** hung from the ceiling to jazz up the dishing.
> As one glass followed another, mosaics around the patio slowly became more poppy.
> **camel**-colored fabric across the peeling paint on a door.
> 's window, glass turned to rose-colored muffins, traditional French pastry.
> Other guests commented **volcano** green at the bottom of a French tagine (spiced chicken with couscous and eggplant). Some wined and dined while others shopped.
> Sandless, both snow white sand and luxurious, true up close and in person.
> on the peak of the **volcano**'s summit.
> be
> (120 tokens, eos=NO)

Verdict: fully coherent; all three out-of-place words integrated grammatically.

---

## STEERED herd — produced by `gen_steer.py` (STEER_MODE=herd, HERD_GAIN=1.3, HERD_WINDOW=10)

Ramped calibrated rotation 0..rank-1-angle over the window, de-repeat after the word lands. Words present: all True.

> It was a warm morning in a small kitchen . The smell of pancakes lingered**diamond**y dishing.
> As one person turned to speak about what she'd done on the previous Sunday, another came up ahair across the aisle**camel**. Suddenly, the lights flared up, turned on, and a brand new, glowing, luminous, arena expanded into the kitchenware **volcano**.
> It was a beginning. A tipping point. Just like football, the spectacular starts with the beginning of the game.
> When we went to dance, both my boys started dancing around, sort of stumbling and pausing here and there.
> (120 tokens, eos=NO)

Variant HERD_WINDOW=6: "flowers**diamond** pins... crack**camel** brown... 'Merrit, oh why did **volcano** watch that gem for so long?'" — cleanest dialogue of the run.
