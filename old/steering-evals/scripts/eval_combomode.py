#!/usr/bin/env python3
"""eval_combomode.py — META-LEARN the READOUT-COMBINATION mode (big leap).

Grid over the COMBINATION control space (operator x blend weight) at the
readout (final norm -> head): each cell feeds the head a normalized mix
of native v and law-rotated v'. Label each cell with SEED-ROBUST REAL
coherence = strict-coherent on >=2/3 independent rolls (strict: plant,
div>=0.7, rep4==0, maxrun<=2, no <eos>, no token>2x). Meta-summarize
which (operator, lam) is a robust-coherence REGIME vs which flips
between 'native loop' and 'topic loop'.

Operators:
  lin : normalize(lam*v + (1-lam)*v')          (chord/linear interp)
  slerp: geodesic blend at fraction (1-lam)    (spherical interp)
  geom : elementwise geometric mean w/ sign     (multiplicative)
  maxsel: elementwise pick larger-magnitude     (winner-take-all)
Budget(20s): 2 prompts x 4 ops x 2 lam x 2 seeds x 6 tok.
One model, no template.
Run: HF_TOKEN=<tok> timeout 18 python3 -u eval_combomode.py
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
NTOK = 6
SEEDS = [0, 1]
TARGET = 'city'
OUT = Path('../steering_geometry_results/combomode.csv')

PROMPTS = [
    ('ask', 'If you ask me which European city is the most beautiful, I would say that'),
    ('fr',  'The capital of France is'),
]
OPS = ['lin', 'slerp', 'geom', 'maxsel']
LAMS = [0.4, 0.6]


def rep4(toks):
    if len(toks) < 4:
        return 1.0
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

    def rot(vv, tid, alpha):
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(alpha) + g * math.sin(alpha)) * vv.norm()

    def nat_vL(ids):
        cv = {}
        hk = model.model.norm.register_forward_hook(
            lambda m, i, o: cv.__setitem__('v', o[0, -1, :].float()))
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        hk.remove()
        return cv['v'], L

    def sample(L):
        p = torch.softmax(L.float(), 0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk) / (q * msk).sum()
        return int(torch.multinomial(qq, 1))

    def combine(v, r, op, lam):
        n = v.float(); r = r.float()
        nn = n / n.norm(); rn = r / r.norm()
        if op == 'lin':
            out = lam * n + (1 - lam) * r
        elif op == 'slerp':
            t = 1 - lam
            coso = (nn * rn).sum().clamp(-1.0, 1.0)
            om = torch.acos(coso); so = torch.sin(om)
            if so.abs() < 1e-6:
                out = n
            else:
                out = (torch.sin((1 - t) * om) / so) * n + \
                      (torch.sin(t * om) / so) * r
        elif op == 'geom':
            mag = torch.sqrt(n.abs() * r.abs())
            sgn = torch.where((n + r) == 0, torch.ones_like(n), n + r)
            sgn = torch.sign(sgn)
            out = sgn * mag
        elif op == 'maxsel':
            out = torch.where(n.abs() >= r.abs(), n, r)
        else:
            raise ValueError(op)
        return out / out.norm() * n.norm()

    def strict_ok(toks, txt):
        if not any(t in toks for t in fset):
            return False
        if '<eos>' in txt:
            return False
        if len(set(toks)) / len(toks) < 0.7:
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

    rows = []
    agg = {}
    print(f"[{MODEL}] readout COMBINATION grid (op x lam), seed-robust "
          f"coherence ({len(SEEDS)} seeds, strict)")
    print(f"  {'prompt':<5}{'op':>7}{'lam':>5} | per-seed robust  text")
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        alpha = 2 * (gap / 97.0) + 0.02
        tgt = closest(vf)
        vp = rot(vf, tgt, alpha)
        for op in OPS:
            for lam in LAMS:
                vin = combine(vf, vp, op, lam)
                oks = []
                str_seeds = ''
                txts = []
                for sd in SEEDS:
                    torch.manual_seed(sd)
                    ids = ids0.clone()
                    toks = []
                    for step in range(NTOK):
                        if step == 0:
                            def inj(m, i, o, p=vin):
                                out = o.clone()
                                out[0, -1, :] = torch.as_tensor(
                                    p, dtype=out.dtype, device=out.device)
                                return out
                            h = model.model.norm.register_forward_hook(inj)
                            with torch.no_grad():
                                L = model(ids).logits[0, -1].float()
                            h.remove()
                        else:
                            with torch.no_grad():
                                L = model(ids).logits[0, -1].float()
                        nxt = sample(L)
                        toks.append(nxt)
                        ids = torch.cat([ids, torch.tensor([[nxt]],
                                                           device=DEV)], dim=1)
                    txt = tok.decode(toks)
                    o = strict_ok(toks, txt)
                    oks.append(o)
                    str_seeds += 'Y' if o else 'n'
                    txts.append(txt)
                robust = 1 if sum(oks) >= 2 else 0
                rows.append(dict(prompt=pname, op=op, lam=lam,
                                 per_seed=str_seeds,
                                 n_coherent=sum(oks), robust=robust,
                                 text0=txts[0][:26], text1=txts[1][:26]))
                agg.setdefault((op, lam), [0, 0])
                agg[(op, lam)][0] += robust
                agg[(op, lam)][1] += 1
                print("  %-5s %7s %5.2f | %s  %d   %s / %s"
                      % (pname[:5], op, lam, str_seeds, robust,
                         txts[0][:18].strip(), txts[1][:18].strip()),
                      flush=True)

    print("\n-- REGIME SUMMARY: (op, lam) -> robust across prompts --")
    for (op, lam), (nr, tot) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"  {op:>7} lam={lam:.1f}: robust {nr}/{tot}  "
              f"({'REGIME' if nr == tot and tot > 0 else ''})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()