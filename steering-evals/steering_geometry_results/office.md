# Prompt: "The office was quiet after hours"

Words: **sheep, sushi, elevator** (out-of-place in an office)
Run: seed=0, ntok=120, NUCLEUS=0.9

---

## PURE (unsteered) — produced by `gen_pure.py`

> The office was quiet after hours . The rest area had to close its gate to access.
> Jimmy rested on a bed of mousie around the air conditioner in the lobby. It was a sleepy across Spain. Jimmy shifted on the bed waiting for someone to enter. He fell asleep after a little bit. He dreamed he was everywhere but at home ..in his sleep he was alone.
> Arriving at the hotel room, Britt entered the gate while the rest of Harry's group were still asleep . Brighty entered the room a few minutes along Britt up the stairs of the building. The hotel was going full night. Britt had
> (120 tokens, eos=NO)

---

## STEERED graft — produced by `gen_steer.py` (STEER_MODE=graft, default)

Single calibrated rotation at each switch step + de-repeat window (ANTI=4). Words present: sheep True, sushi True, elevator True.

> The office was quiet after hours  **sheep**ishly. The sounds of jazz music were dreading.
> Jimmy rested on a bed of mosaics around the air. He looked at everything that **sushi** a year ago and just like the five senses of the colors, the people who are with him were the only color in his life. Two people **elevator** people at work, he didn't know what they thought of him, he tried to make a joke or smile, but he just couldn't even make a mouth he was so insecure with himself he tried to remove his tongue.
> Tsunehi sobbing his strength was wearing him out but he
> (120 tokens, eos=NO)

Verdict: words land but grammar breaks ("sushi a year ago", "Two people elevator people"). Weakest of the three prompts.

---

## STEERED herd — produced by `gen_steer.py` (STEER_MODE=herd, HERD_GAIN=1.3, HERD_WINDOW=10)

Ramped calibrated rotation 0..rank-1-angle over the window, de-repeat after the word lands. Words present: all True.

> The office was quiet after hours . The rest area had **sheep** and goats, dairying sheep and cows. The coveous littles used to be in the first corral.
> Three year old brown gelding **sushi** had a strong attack of tendonitis, turned the horses around, and led the stable back to a halt.
> Everything stopped at once **elevator** was down for labor. Yummy took off with the horses. Horses had riders while Sally had an equine issue with a food pipeline.
> Fascinating snow. My mother's snow shoe up the ridge of Wood Mountain. The peak was covered in new snow.
> (120 tokens, eos=NO)

Verdict: herd mode keeps the model's own voice — it builds a coherent farm/stable tangent narrative; sheep lands naturally, sushi/elevator grammatical but odd.