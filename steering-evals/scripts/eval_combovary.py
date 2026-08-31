#!/usr/bin/env python3
"""eval_combovary.py — combine expected & rotated states in MANY WAYS at
the readout (final norm -> head), all normalized, then long free-run to watch.

Readout combination idea (eval_combo): feed the head a normalized mix of
the NATIVE final state v and the law-rotated v' instead of either alone.
This probe sweeps DIFFERENT combination operators to see if the choice of
operator changes behavior (the linear one just interpolated native<->rot).

Operators (all produce a vector at ||v|| scale fed to lm_head at step 0):
  readout  : v' (plain rotation, reference)
  lin_05   : normalize(0.5*v + 0.5*v')            (chord/linear)
  slerp_05 : spherical geodesic midpoint of v,v'  (constant angular vel)
  geom_05  : elementwise geometric mean of |v|,|v'| with sign       (multiplicative)
  maxsel   : elementwise pick the larger-magnitude component (winner-take-all)
  overshoot: normalize(v + 1.5*(v'-v))            (past the target)

Then FREE-RUN. Watch for any operator that breaks the native<->rotated
dichotomy (i.e. produces something other than 'ante loop' or 'topic loop').
Run: HF_TOKEN=<tok> timeout 30 python3 -u eval_combovary.py
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
NTOK = 18
SEEDS = [0]
TARGET = 'city'
OUT = Path('../steering_geometry_results/combovary.csv')

PROMPTS = [
    ('ask', 'If you ask me which European city is the most beautiful, I would say that'),
    ('fr',  'The capital of France is'),
]
MODES = ['readout', 'lin_05', 'slerp_05', 'geom_05', 'maxsel', 'overshoot']


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

    def combine(view, mode):
        """produce the readout vector for step 0 from native v and rot vp."""
        n = view['v'].float()
        r = view['vp'].float()
        nn = n / n.norm()
        rn = r / r.norm()
        if mode == 'readout':
            out = r
        elif mode == 'lin_05':
            out = 0.5 * n + 0.5 * r
        elif mode == 'overshoot':
            out = n + 1.5 * (r - n)
        elif mode == 'maxsel':
            sel = torch.where(n.abs() >= r.abs(), n, r)
            out = sel
        elif mode == 'geom_05':
            mag = torch.sqrt(n.abs() * r.abs())       # geometric mean of magnitudes
            sgn = torch.sign(n + r) + 0.0
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            out = sgn * mag
        elif mode == 'slerp_05':
            t = 0.5
            coso = (nn * rn).sum().clamp(-1.0, 1.0)
            om = torch.acos(coso)
            so = torch.sin(om)
            if so.abs() < 1e-6:
                out = n
            else:
                a = torch.sin((1 - t) * om) / so
                b = torch.sin(t * om) / so
                out = a * n + b * r
        else:
            raise ValueError(mode)
        return out / out.norm() * n.norm()

    rows = []
    print(f"[{MODEL}] readout combination OPERATORS, {NTOK}-tok free-run")
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        alpha = 2 * (gap / 97.0) + 0.02
        tgt = closest(vf)
        vp = rot(vf, tgt, alpha)
        print(f"\n==== {pname}: {pr!r} gap={gap:.1f} -> "
              f"{tok.decode([tgt])!r} ====")
        for mode in MODES:
            view = {'v': vf, 'vp': vp}
            vin = combine(view, mode)
            torch.manual_seed(0)
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
                ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)],
                                dim=1)
            x = sum(1 for t in toks if t in fset)
            plant = 1.0 if any(t in toks[:8] for t in fset) else 0.0
            div = len(set(toks)) / len(toks)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(toks)), default=0)
            rp = rep4(toks)
            txt = tok.decode(toks)
            rows.append(dict(prompt=pname, mode=mode, gap=round(gap, 3),
                             alpha=round(alpha, 3), plant=plant, xtgt=x,
                             div=round(div, 3), maxrun=mr,
                             rep4=round(rp, 3), text=txt))
            print(f"\n--- [{mode:>9}] plant={plant:.0f} xtgt={x} "
                  f"div={div:.2f} maxrun={mr} rep4={rp:.2f} ---", flush=True)
            print(f"    {txt}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()