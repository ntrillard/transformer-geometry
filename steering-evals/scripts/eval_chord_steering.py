#!/usr/bin/env python3
"""Steering as a NOTE IN A CHORD: composite harmonic target directions.

  note            = LM-head row W_t (unit direction)             [a token]
  chord           = a set of notes T = {t_1..t_m}                [semantic class]
  chord direction = C = normalize(sum_i W_ti)  (constructive      [the chord
                    interference if consonant; cancellation if    "rings" if the
                    dissonant/spread)                              notes reinforce]
  resolution      = entering the FAMILY cone { x : (W_ti-W_j).x >= 0
                    for all ti in T, j outside } -- ANY note of the
                    chord beating the whole vocabulary.           [chord resolves]
  key / tonic     = where the state already sits; the chord
                    resolves in some keys (states) and not others.

Tests (Qwen2-0.5B, cached rows + real states):
  T1 interference : ||sum W_ti|| and angular cap for consonant (semantic-class)
                    chords vs random chords -- the geometric "ring".
  T2 chord reach   : rank-1 (any note beats the full vocab) after steering
                    toward C, 4 depths x 2 contexts, 17/45 deg budgets.
                    vs best-single-note reach and random-chord control.
  T3 tonic         : under chord steering, WHICH note ranks 1 -- is it the
                    note most aligned with C (the "root")?
  T4 generation    : PERSISTENT steer toward a single note (known pit:
                    apple x48, div .10) vs toward a 10-note chord (food).
                    Does a topical chord steer give varied topical text?

Run: python eval_chord_steering.py
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import steering_geometry_test as M
from eval_practical_steering import gen

OUT = Path("../steering_geometry_results")

CLASSES = {
    'food':   ['apple', 'banana', 'bread', 'cheese', 'chicken', 'grape',
               'honey', 'milk', 'rice', 'soup'],
    'animal': ['dog', 'cat', 'horse', 'lion', 'bird', 'wolf', 'tiger',
               'fish', 'snake', 'rabbit'],
    'color':  ['red', 'blue', 'green', 'black', 'white', 'yellow',
               'pink', 'purple', 'brown', 'gray'],
    'city':   ['Paris', 'London', 'Tokyo', 'Berlin', 'Rome', 'Moscow',
               'Cairo', 'Delhi', 'Seoul', 'Madrid'],
    'nature': ['ocean', 'tree', 'mountain', 'river', 'forest', 'flower',
               'stone', 'cloud', 'star', 'moon'],
    'number': ['one', 'two', 'three', 'four', 'five', 'six', 'seven',
               'eight', 'nine', 'ten'],
}
SEED = 0


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    model, tok = M.load_model('Qwen/Qwen2-0.5B-Instruct', dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    vocab = model.config.vocab_size
    W = W[:vocab]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    V, d = Wn.shape
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    word2id = {}
    for w in sorted({x for cls in CLASSES.values() for x in cls}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    print(f"note rows available: {len(word2id)}/{sum(len(c) for c in CLASSES.values())}")

    # build chords: semantic classes (consonant) + random controls (dissonant)
    chords = {}
    for cls, words in CLASSES.items():
        ids = [word2id[w] for w in words if w in word2id]
        if len(ids) >= 6:
            chords[f'{cls}'] = np.array(ids)
    # random chords (matched size) = dissonant control
    for k in range(6):
        chords[f'rand{k}'] = rng.choice(V, size=10, replace=False)

    def chord_dir(ids):
        c = Wn[ids].mean(0)
        n = np.linalg.norm(c)
        return c / n, n

    # ============ T1: interference (the chord "rings") ============
    print("\n[T1] interference: ||sum notes|| (unit rows) and angular spread")
    rows_t1 = []
    for name, ids in chords.items():
        C, rho = chord_dir(ids)
        Cc = np.clip(Wn[ids] @ C, -1, 1)
        spread = np.degrees(np.arccos(Cc.mean()))       # mean note-to-center angle
        top_note = int(np.argmax(Cc))
        rows_t1.append(dict(chord=name, norm=rho, mean_cos=float(Cc.mean()),
                            spread_deg=float(spread),
                            top_note_cos=float(Cc.max())))
    t1 = pd.DataFrame(rows_t1)
    real = t1[~t1.chord.str.startswith('rand')]
    rnd = t1[t1.chord.str.startswith('rand')]
    print(f"   consonant chords : norm med {real.norm.median():.3f}  "
          f"mean-note-cos med {real.mean_cos.median():+.3f}  "
          f"spread med {real.spread_deg.median():.1f} deg")
    print(f"   random chords    : norm med {rnd.norm.median():.3f}  "
          f"mean-note-cos med {rnd.mean_cos.median():+.3f}  "
          f"spread med {rnd.spread_deg.median():.1f} deg")
    # interference ratio
    print(f"   norm ratio consonant/random: {real.norm.median() / rnd.norm.median():.2f}x")

    # ============ T2/T3: chord-reach geometry ============
    states = M.get_states(model, tok, ["The capital of France is",
                                       "Once upon a time"],
                          sorted({int(round(f * (model.config.num_hidden_layers - 1)))
                                  for f in (0.0, 0.33, 0.67, 0.99)}))
    Wt = torch.as_tensor(W, device=dev)
    rows = []
    for name, ids in chords.items():
        C, _ = chord_dir(ids)
        is_real = not name.startswith('rand')
        for ctx, llayer in states.items():
            for l, h in llayer.items():
                u = h / np.linalg.norm(h)
                tau = M.tangent_direction(u, C)
                for budget in (17, 45):
                    v = M.rotate_toward(u, tau, math.radians(budget))
                    L = v @ W.T
                    family = L[ids]
                    outsider = np.delete(L, ids).max() if V - len(ids) else -np.inf
                    best_note = int(np.argmax(family))
                    chord_rank1 = bool(family.max() > outsider)
                    # single-note baseline: max over notes of note-reach
                    note_reach = []
                    for ti in ids:
                        nt = tau_for_note(u, Wn[ti])
                        vn = M.rotate_toward(u, nt, math.radians(budget))
                        Ln = vn @ W.T
                        note_reach.append(bool(Ln[ti] > np.delete(Ln, ti).max()))
                    rows.append(dict(chord=name, real=is_real, ctx=ctx, layer=l,
                                     budget=budget,
                                     chord_rank1=chord_rank1,
                                     best_note_id=int(ids[best_note]),
                                     best_note_logit=float(family[best_note]),
                                     margin=float(family.max() - outsider),
                                     any_note_reach=bool(np.any(note_reach)),
                                     all_note_reach=float(np.mean(note_reach))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "chord_steering.csv", index=False)
    print("\n[T2] chord-reach @17 (any note beats full vocab after steering toward C):")
    d17 = df[df.budget == 17]
    for name, g in d17.groupby('chord'):
        print(f"   {name:8s} chord-reach@17={g.chord_rank1.mean()*100:5.1f}%  "
              f"best-note-reach={g.any_note_reach.mean()*100:5.1f}%  "
              f"mean-single-note={g.all_note_reach.mean()*100:5.1f}%  "
              f"margin(med)={g.margin.median():+.3f}")
    rg = df[df.real & (df.budget == 17)]
    rr = df[~df.real & (df.budget == 17)]
    print(f"\n   consonant: chord-reach {rg.chord_rank1.mean()*100:5.1f}%  "
          f"random: {rr.chord_rank1.mean()*100:5.1f}%")
    print(f"   chord >= best-single-note?  "
          f"{(rg.chord_rank1 >= rg.any_note_reach).mean()*100:.1f}% of cases")

    # T3: tonic (is the steered-argmax note the one most aligned with C?)
    print("\n[T3] tonic: is the note that ranks 1 after chord-steering the "
          "'root' (most C-aligned note)?")
    tonal = []
    for name, ids in chords.items():
        C, _ = chord_dir(ids)
        align = np.argsort(-np.clip(Wn[ids] @ C, -1, 1))
        for ctx, llayer in states.items():
            for l, h in llayer.items():
                u = h / np.linalg.norm(h)
                tau = M.tangent_direction(u, C)
                v = M.rotate_toward(u, tau, math.radians(17))
                L = v @ W.T
                fam = L[ids]
                if fam.max() > np.delete(L, ids).max():     # chord resolved
                    argmax_note = int(ids[int(np.argmax(fam))])
                    root = int(ids[align[0]])
                    tonal.append(int(argmax_note) == root)
    if tonal:
        print(f"   argmax-note == C-root note in {np.mean(tonal)*100:.1f}% of resolved cases")

    # ============ T4: generation, single note vs chord ============
    print("\n[T4] persistent generation: single note 'apple' vs food-chord")
    pids = tok("Once upon a time", add_special_tokens=False,
               return_tensors="pt").input_ids.to(model.device)
    li = model.config.num_hidden_layers - 1
    food_ids = np.array([word2id[w] for w in CLASSES['food'] if w in word2id])
    C_food, _ = chord_dir(food_ids)
    s_apple = Wn[word2id['apple']]

    def hook_for(dn):
        dn_t = torch.as_tensor(dn, device=model.device, dtype=model.dtype)

        def hook(module, inp, out):
            out2 = out.clone()
            h = out2[0, -1, :].float()
            hn = h / h.norm()
            g = dn_t.float() - (dn_t.float() @ hn) * hn
            g = g / max(g.norm().item(), 1e-8)
            h2 = h + 0.3 * h.norm() * g
            out2[0, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
            return out2
        return hook

    food_words = set(CLASSES['food'])
    for name, tgt, family_words in (("note 'apple'", s_apple, {'apple'}),
                                    ("chord 'food'", C_food, food_words)):
        occs, divs, distincts, maxrun = [], [], [], []
        for sd in range(4):
            text, conf = gen(model, tok, pids, 48, hook_fn=hook_for(tgt),
                             layer=li, duty=1, top_p=0.9, seed=sd)
            low = text.lower()
            occ = sum(low.count(w) for w in family_words)
            toks = tok(text, add_special_tokens=False).input_ids
            div = len(set(toks)) / max(len(toks), 1)
            distinct = sum(1 for w in family_words if w in low)
            # longest run of a single token
            run, cur, last = 0, 0, None
            for tid in toks:
                if tid == last:
                    cur += 1
                else:
                    cur = 1
                run = max(run, cur)
                last = tid
            occs.append(occ); divs.append(div)
            distincts.append(distinct); maxrun.append(run)
            if sd == 0:
                print(f"   {name:14s} seed0: {text[:88]!r}")
        print(f"   {name:14s} x{np.mean(occs):5.1f}  distinct-family "
              f"{np.mean(distincts):.1f}/10  diversity {np.mean(divs):.2f}  "
              f"max-token-run {np.mean(maxrun):.0f}")

    # save
    t1.to_csv(OUT / "chord_interference.csv", index=False)
    print(f"\n[{time.time()-t0:.0f}s] saved chord_steering.csv + chord_interference.csv")


def tau_for_note(u, note):
    return M.tangent_direction(u, note)


if __name__ == "__main__":
    main()