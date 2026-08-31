# Prompt: "The train rattled through the countryside"

Words: **marble, telescope, submarine** (out-of-place on a train)
Run: seed=0, ntok=120, NUCLEUS=0.9

---

## PURE (unsteered) — produced by `gen_pure.py`

> The train rattled through the countryside . The wagons accompanied it. The sun dappled the heads of the kids cackling loudly around me on the moving wagon using flash photography.
> Trip numbers across Spain. Time. Money. Road.
> I was the wrong country for my trying Swancon, the international contest celebrating first-year Swans at Henley, UK, for swans (and by swans I actually mean Cygnus). Many new friends and faces came out for our first cygnless summer.
> (lines by Sheri McKay Simons)
> It finally occurred to me today that part of the problem with my life hasn
> (120 tokens, eos=NO)

---

## STEERED graft — produced by `gen_steer.py` (STEER_MODE=graft, default)

Single calibrated rotation at each switch step + de-repeat window (ANTI=4). Words present: all True.

> The train rattled through the countryside  **marble** and clay mines while I stared out the train window.
> As I stared out at the mosaics around me, the sounds of India came in **telescope** a breeze across the valley, the dullest tones outshining every other note with the sweetest, most intense melodies.
> The train was **submarine**, but at the same time, the sound of my heart was sinking with the passage of time like sea slumps in a crashing wave.
> When we went to the station, it took a whole day, I saw India finally in a way. India seemed to stop, waiting to embrace us
> (120 tokens, eos=NO)

Verdict: poetic but forced ("came in telescope a breeze", "The train was submarine").

---

## STEERED herd — produced by `gen_steer.py` (STEER_MODE=herd, HERD_GAIN=1.3, HERD_WINDOW=10)

Ramped calibrated rotation 0..rank-1-angle over the window, de-repeat after the word lands. Words present: all True.

> The train rattled through the countryside . The wagons seemed to rock **marble**, dishing up rock-hard waves of gravel beneath the belly of the vehicle. The driver looked at me in disgust as he puttered along **telescope**. He felt the smooth surface was the wrong surface for horses, and the only natural way of breaking a spell was to stare at the horizon **submarine**.
> I had to endure seven hours before the train came to a halt in a town. The driver told me that everything was fine, he was just driving home to a mother whose spouse had just been transferred to Moscow.
> (120 tokens, eos=NO)

Verdict: the strongest herd run — narrative stays on-train and coherent; marble is integrated poetically, telescope/submarine forced but grammatical.