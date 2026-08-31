#!/usr/bin/env python3
"""eval_metalab4.py — (1) confirm the hard-limit boundary (gap/rank
beyond which anti-last coherence is impossible), (2) attack the 2/3
"soft ceiling" with a LONGER anti window.

Follows metalab3 (anti-last@10deg: gap<13 & rank<640 -> coherent; the
2/3-marginals japan/olymp/france/paris/traveld never reach 3/3 at any
angle). This probe:
  A) boundary: for the known failures (tower gap13.5/rank640,
     austr14.0/1096, bern13.3/3458) confirm they're hard at 10deg, and
     find a gradient of gaps/ranks to locate the actual threshold.
  B) soft ceiling: for the 2/3 marginals, try anti-WINDOW=2 (suppress
     the planted token for TWO steps instead of one) at 10deg - does a
     longer window lift them to 3/3?

One model, no template. Run: HF_TOKEN=<tok> timeout 18 python3 -u
eval_metalab4.py
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
OUT = Path('../steering_geometry_results/metalab4.csv')
ANGLE = 10.0

# (name, prompt, want) want=1 for the soft-ceiling attack, 0 for boundary
PROMPTS = [
    ('ask',  'If you ask me which European city is the most beautiful, I would say that', 1),
    ('japan','The capital of Japan is', 1),
    ('spain','The capital of Spain is', 1),
    ('tower','The tallest tower in the world is in', 0),
    ('austr','The biggest city in Australia is', 0),
    ('visitnl','I love to visit new places, and my favorite city is', 1),
    ('olymp','The Olympic Games were held in', 1),
    ('france','The capital of France is', 1),
    ('paris','I visited Paris last summer and it was', 1),
    ('traveld','I would love to travel to', 1),
    ('bern','The capital of Switzerland is', 0),
    ('concert','The concert was held in', 1),
]
WINDOWS = [1, 2]   # anti suppression window


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

    def shot_anti(ids0, vp, tgt, sd, window):
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
                if 1 <= step <= window:
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
    print(f"[{MODEL}] anti-last@10deg: hard boundary + anti-window "
          f"attack ({len(SEEDS)} seeds, window in {WINDOWS})")
    print("  %-9s %5s %6s | %s | best" % ('prompt', 'gap', 'rank',
                                          ' '.join(f'w{w}(n/Y)'
                                                   for w in WINDOWS)))
    for pname, pr, _ in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        order = Ln.argsort(descending=True).tolist()
        f_rank = order.index(fam[int(Ln[fam].argmax())]) + 1
        tgt = closest(vf)
        vp = rot_to_angle(vf, tgt, ANGLE)
        per_w = []
        for w in WINDOWS:
            oks = []
            seed_str = ''
            for sd in SEEDS:
                toks = shot_anti(ids0, vp, tgt, sd, w)
                c = coherent(toks, tok.decode(toks))
                seed_str += 'Y' if c else 'n'
                oks.append(c)
            ncoh = sum(oks)
            per_w.append((w, ncoh, seed_str))
            rows.append(dict(prompt=pname, gap=round(gap, 2),
                             f_rank=f_rank, angle=ANGLE, window=w,
                             seeds=seed_str, n_coherent=ncoh,
                             robust=1 if ncoh >= 2 else 0))
        best = max(per_w, key=lambda x: x[1])
        print("  %-9s %5.1f %6d | %s | %d@w%d"
              % (pname[:9], gap, f_rank,
                 ' '.join(f'w{w}:{s}' for w, _, s in per_w),
                 best[1], best[0]), flush=True)

    # summary
    print("\n-- boundary + window summary --")
    for pname in [p[0] for p in PROMPTS]:
        sel = [r for r in rows if r['prompt'] == pname]
        best = max(sel, key=lambda r: r['n_coherent'])
        print(f"  {pname}: best {best['n_coherent']}/3 "
              f"(win={best['window']})  gap={best['gap']} rank={best['f_rank']}")
    n_w1 = sum(1 for r in rows if r['window'] == 1 and r['n_coherent'] >= 2)
    n_w2 = sum(1 for r in rows if r['window'] == 2 and r['n_coherent'] >= 2)
    print(f"\n  window1 coherent>=2: {n_w1}/12   window2: {n_w2}/12")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()