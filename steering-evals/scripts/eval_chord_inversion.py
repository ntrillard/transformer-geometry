#!/usr/bin/env python3
"""Chord INVERSION steering: aim at a NOTE, keep the FAMILY goal.

T2 found: steering toward the chord CENTER (cap-center C = normalize(SUM rows))
rarely resolves the family cone (color 75%, number 62.5%, others 0%; spread
threshold ~50 deg; corr(spread,reach) = -0.84), while steering toward a single
family note reaches rank-1 ~100% of the time.  Hypothosis:

  C is useful as a FAMILY IDENTIFIER (root/topic), useless as a STEERING TARGET
  (it lands in a no-man's-land).  To resolve a chord: aim at the best-
  POSITIONED note for the current state (the "inversion" -- resolve through the
  note nearest from the tonic), score the result by FAMILY-cone entry.

This script tests, for 6 consonant chord-families x 2 contexts x 4 layers:
  targets:   center C, root (most C-aligned note), best-logit note (from the
             state), random family note
  budgets:   17 / 45 deg
  metrics:   family-resolved (ANY note beats full vocab), which note is argmax,
             family coverage (# notes in vocab top-10), entry angle to the
             family cone, entry angle to the aimed note's own cell.

Predicted results:
  P1  inversion (best-logit note) resolves the FAMILY at ~100% @17 across ALL
      chords -- the chord-spread specificity vanishes for note-steering.
  P2  root steering (C-aligned note) does NOT match inversion (C is not a
      steering target; "aim at the best note, not the most central").
  P3  inversion family-coverage (family notes in top-10) vs center coverage:
      does note-aim keep the family ranked, or collapse to one winner?
  P4  inversion needs LESS budget than center (earlier family-cone entry).

Run: python eval_chord_inversion.py
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import steering_geometry_test as M

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


def family_entry_angle(u, fam_ids, Wn, W, max_budget=45):
    """First arc angle (deg) where ANY family note beats the full vocab."""
    tau = None
    # analytic: beat-all window per note -> family resolves when any note is
    # above the best OUTSIDER. Compute over a scan for robustness here.
    best = None
    for deg in np.linspace(0.5, max_budget, 200):
        v = M.rotate_toward(u, M.tangent_direction(u, Wn[fam_ids].mean(0)),
                            math.radians(deg))
        L = v @ W.T
        outsider = np.delete(L, fam_ids).max()
        if L[fam_ids].max() > outsider:
            best = deg
            break
    return best


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    model, tok = M.load_model('Qwen/Qwen2-0.5B-Instruct', dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    vocab = model.config.vocab_size
    W = W[:vocab]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    V, d = Wn.shape

    word2id = {}
    for w in sorted({x for cls in CLASSES.values() for x in cls}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])

    states = M.get_states(model, tok, ["The capital of France is",
                                       "Once upon a time"],
                          sorted({int(round(f * (model.config.num_hidden_layers - 1)))
                                  for f in (0.0, 0.33, 0.67, 0.99)}))

    rows = []
    for cls, words in CLASSES.items():
        fam = np.array([word2id[w] for w in words if w in word2id])
        if len(fam) < 6:
            continue
        S = Wn[fam]
        C = S.mean(0); C = C / np.linalg.norm(C)
        align = S @ C
        root = fam[int(np.argmax(align))]
        for ctx, llayer in states.items():
            for l, h in llayer.items():
                u = h / np.linalg.norm(h)
                start_logits = u @ Wn.T
                best_n = fam[int(np.argmax(start_logits[fam]))]   # best-positioned note
                rnd_n = fam[int(rng.integers(len(fam)))]
                targets = {'center': C, 'root': Wn[root],
                           'best_note': Wn[best_n],
                           'rand_note': Wn[rnd_n]}
                for budget in (17, 45):
                    for tname, tgt in targets.items():
                        tau = M.tangent_direction(u, tgt)
                        v = M.rotate_toward(u, tau, math.radians(budget))
                        L = v @ W.T
                        family = L[fam]
                        outsider = np.delete(L, fam).max()
                        resolved = float(family.max() > outsider)
                        argmax_note = int(fam[int(np.argmax(family))])
                        # family coverage: within vocab top-10 how many family?
                        top10 = np.argsort(-L)[:10]
                        cov = float(np.isin(top10, fam).sum())
                        # entry angle for CENTER target only (budget scan)
                        entry = None
                        if tname == 'center' and budget == 17:
                            entry = family_entry_angle(u, fam, Wn, W)
                        rows.append(dict(cls=cls, ctx=ctx, layer=l, budget=budget,
                                         target=tname, resolved=resolved,
                                         argmax_note=str(argmax_note),
                                         root=str(root), best_note=str(best_n),
                                         family_top10=cov, entry_deg=entry))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "chord_inversion.csv", index=False)
    print(f"[{time.time()-t0:.0f}s] saved chord_inversion.csv\n")

    print("== Family-resolution rate by target and chord (17 deg budget) ==")
    d17 = df[df.budget == 17]
    piv = d17.pivot_table(index='cls', columns='target', values='resolved',
                          aggfunc='mean')
    print(piv.round(3).to_string())
    print("\n  overall by target:")
    print(d17.groupby('target').resolved.mean().round(3).to_string())

    print("\n== P1: best_note (inversion) vs center, per chord ==")
    c17 = d17[d17.target == 'center'].groupby('cls').resolved.mean()
    b17 = d17[d17.target == 'best_note'].groupby('cls').resolved.mean()
    cmp = pd.DataFrame({'center': c17, 'inversion(best_note)': b17})
    print(cmp.round(3).to_string())

    print("\n== P2: root vs best_note (is C-aligned note the best steer target?) ==")
    r = d17[d17.target == 'root'].groupby('cls').resolved.mean()
    print(pd.DataFrame({'root': r, 'best_note': b17}).round(3).to_string())

    print("\n== P3: family coverage (family notes in vocab-top-10) @17 ==")
    cov = d17.groupby('target').family_top10.mean()
    print(cov.round(2).to_string())

    print("\n== P4: entry angle to family cone (center target, scan) ==")
    ent = d17[d17.entry_deg.notna()].groupby('cls').entry_deg.median()
    print(ent.round(1).to_string())


if __name__ == "__main__":
    main()