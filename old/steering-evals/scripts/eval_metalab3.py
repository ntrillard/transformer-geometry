#!/usr/bin/env python3
"""eval_metalab3.py — anti-last coherence x angle SURFACE (the full map).

Builds on metalab2 (anti-last@7deg coherent 9/12; failure zone = high
gap/rank). This maps coherence (>=2/3 seeds, fixed metric) vs steer angle
per prompt to answer: (1) do the 2/3-seed marginal prompts reach 3/3 at
some angle (adaptive helps)? (2) are the failures (tower/austr/bern)
failures at EVERY angle?

12 prompts x angles {4,7,10,13} x 3 seeds x 5 tok. Careful, no known
error patterns: rep4 fixed (0.0 for <4), .cpu() before numpy, hooks in
try/finally, single-row CSV.
Run: HF_TOKEN=<tok> timeout 18 python3 -u eval_metalab3.py
"""
import csv
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

import steering_geometry_test as SGT
from eval_nb_quick import CLASSES

MODEL = 'google/gemma-3-1b-pt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 5
SEEDS = [0, 1, 2]
TARGET = 'city'
OUT = Path('../steering_geometry_results/metalab3.csv')
ANGLES = [4.0, 7.0, 10.0]

PROMPTS = [
    ('ask',  'If you ask me which European city is the most beautiful, I would say that'),
    ('japan','The capital of Japan is'),
    ('spain','The capital of Spain is'),
    ('tower','The tallest tower in the world is in'),
    ('austr','The biggest city in Australia is'),
    ('visitnl','I love to visit new places, and my favorite city is'),
    ('olymp','The Olympic Games were held in'),
    ('france','The capital of France is'),
    ('paris','I visited Paris last summer and it was'),
    ('traveld','I would love to travel to'),
    ('bern','The capital of Switzerland is'),
    ('concert','The concert was held in'),
]


def rep4(toks):
    if len(toks) < 4:
        return 0.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return sum(1 for i in range(len(toks) - 3) if n4[i] in n4[i + 1:]) \
        / (len(toks) - 3)


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach().float()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()
    fam = [int(tok(' ' + w, add_special_tokens=False).input_ids[0])
           for w in CLASSES[TARGET]]
    fset = set(fam)
    Wff = Wn[fam].float()

    def closest(vv):
        u = vv / vv.norm()
        return fam[int((Wff @ u).argmax())]

    def rot_to_angle(vv, tid, theta):
        a = math.radians(theta)
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(a) + g * math.sin(a)) * vv.norm()

    def nat_vL(ids):
        cv = {}
        hk = model.model.norm.register_forward_hook(
            lambda m, i, o: cv.__setitem__('v', o[0, -1, :].float()))
        try:
            with torch.no_grad():
                L = model(ids).logits[0, -1].float()
        finally:
            hk.remove()
        return cv['v'], L

    def sample(L):
        p = torch.softmax(L.float(), 0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk) / (q * msk).sum()
        return int(torch.multinomial(qq, 1))

    def coherent(toks, txt):
        if not any(t in toks for t in fset):
            return False
        if '<eos>' in txt:
            return False
        if rep4(toks) != 0.0:
            return False
        mr = max((sum(1 for _ in grp) for _, grp in
                  itertools.groupby(toks)), default=0)
        if mr > 2:
            return False
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        if any(c > 2 for c in counts.values()):
            return False
        return True

    def shot_anti(ids0, vp, tgt, sd):
        torch.manual_seed(sd)
        ids = ids0.clone()
        toks = []
        for step in range(NTOK):
            hs = []
            try:
                if step == 0:
                    def inj(m, i, o, p=vp):
                        out = o.clone()
                        out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                        device=out.device)
                        return out
                    hs.append(model.model.norm.register_forward_hook(inj))
                if step >= 1:
                    def anti(m, i, o, tid=tgt):
                        out = o.clone()
                        out[0, -1, tid] = -30.0
                        return out
                    hs.append(model.lm_head.register_forward_hook(anti))
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
            finally:
                for h in hs:
                    h.remove()
            nxt = sample(L)
            toks.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        return toks

    rows = []
    print(f"[{MODEL}] anti-last coherence x angle surface "
          f"(angles {ANGLES}, {len(SEEDS)} seeds)")
    print("  %-9s %5s %6s | %s | best" % ('prompt', 'gap', 'f_rank',
                                          ' '.join(f'{int(a):>3}(n/Y)'
                                                   for a in ANGLES)))
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        order = Ln.argsort(descending=True).tolist()
        f_rank = order.index(fam[int(Ln[fam].argmax())]) + 1
        tgt = closest(vf)
        per_angle = []
        for a in ANGLES:
            vp = rot_to_angle(vf, tgt, a)
            oks = []
            seed_str = ''
            for sd in SEEDS:
                toks = shot_anti(ids0, vp, tgt, sd)
                c = coherent(toks, tok.decode(toks))
                seed_str += 'Y' if c else 'n'
                oks.append(c)
            ncoh = sum(oks)
            lab = 1 if ncoh >= 2 else 0
            per_angle.append((a, ncoh, seed_str, lab))
            rows.append(dict(prompt=pname, gap=round(gap, 2),
                             f_rank=f_rank, angle=a, seeds=seed_str,
                             n_coherent=ncoh, robust=lab))
        best = max(per_angle, key=lambda x: x[1])
        best_a = best[0]
        print("  %-9s %5.1f %6d | %s | %d@%gdeg"
              % (pname[:9], gap, f_rank,
                 ' '.join(f'{int(pa)}:{pn}' for pa, pn, _, _ in
                          per_angle), best[1], best_a), flush=True)

    # summary
    print("\n-- surface summary --")
    marg = []
    for pname in [p[0] for p in PROMPTS]:
        sel = [r for r in rows if r['prompt'] == pname]
        best = max(sel, key=lambda r: r['n_coherent'])
        if best['n_coherent'] == 3:
            print(f"  {pname}: reaches 3/3 at {best['angle']}g")
        else:
            print(f"  {pname}: best {best['n_coherent']}/3 at any angle "
                  f"(never 3/3)")
        if 2 <= best['n_coherent'] < 3:
            marg.append(pname)
    print(f"  marginal (2/3 best, could a sweeper lift?): {marg or 'none'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()