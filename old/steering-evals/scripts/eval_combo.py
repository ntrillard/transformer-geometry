#!/usr/bin/env python3
"""eval_combo.py — combine the EXPECTED state with the ROTATED state
RIGHT BEFORE THE HEAD READS (final two steps only).

User's idea, scoped to the readout: grab the natural final-norm state
v_final (the "expected" state) and the law-budget rotated state v'
(toward the closest city row). At the final norm -> head boundary, feed
a NORMALIZED COMBINATION of them instead of either alone:
    combo(lam) = normalize( lam*v_final + (1-lam)*v' ) * ||v_final||
Both operands are final-norm-scale vectors the head is trained to read,
so combining them stays in-distribution (no 26-layer re-injection).
All intervention happens at the final two steps (final norm output that
feeds lm_head). After the step-0 intervention, FREE-RUN long to watch.

Modes (all feed the head directly from the final-norm boundary):
  baseline  : v_final (native)
  readout   : v' (plain rotation, committed recipe) - reference
  combo_bal : lam=0.5 (balanced expected+rotated)
  combo_nat : lam=0.7 (bias expected)
  combo_rot : lam=0.3 (bias rotated)
Run: HF_TOKEN=<tok> timeout 30 python3 -u eval_combo.py
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
NTOK = 30
SEEDS = [0]
TARGET = 'city'
OUT = Path('../steering_geometry_results/combo.csv')

PROMPTS = [
    ('ask', 'If you ask me which European city is the most beautiful, I would say that'),
    ('fr',  'The capital of France is'),
]
MODES = [('baseline', None), ('readout', None),
         ('combo_bal', 0.5), ('combo_nat', 0.7), ('combo_rot', 0.3)]


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

    def f0(mode, lam, ids0, vf, vp):
        """step-0 logits fed with the chosen readout vector (or native)."""
        if mode == 'baseline':
            with torch.no_grad():
                return model(ids0).logits[0, -1].float()
        if mode == 'readout':
            vin = vp
        else:
            combo = (lam * vf + (1 - lam) * vp).float()
            combo = combo / combo.norm() * vf.norm()
            vin = combo
        def inj(m, i, o, p=vin):
            out = o.clone()
            out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                            device=out.device)
            return out
        h = model.model.norm.register_forward_hook(inj)
        with torch.no_grad():
            L = model(ids0).logits[0, -1].float()
        h.remove()
        return L

    rows = []
    print(f"[{MODEL}] readout COMBINATION (expected+rotated, normalized) "
          f"-> head, {NTOK}-tok free-run")
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
        for mode, lam in MODES:
            torch.manual_seed(0)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                if step == 0:
                    L = f0(mode, lam, ids, vf, vp)
                else:
                    with torch.no_grad():
                        L = model(ids).logits[0, -1].float()
                nxt = sample(L)
                toks.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)],
                                dim=1)
            x = sum(1 for t in toks if t in fset)
            plant = 1.0 if any(t in toks[:10] for t in fset) else 0.0
            div = len(set(toks)) / len(toks)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(toks)), default=0)
            rp = rep4(toks)
            txt = tok.decode(toks)
            rows.append(dict(prompt=pname, mode=mode, lam=lam,
                             gap=round(gap, 3), alpha=round(alpha, 3),
                             plant=plant, xtgt=x, div=round(div, 3),
                             maxrun=mr, rep4=round(rp, 3), text=txt))
            print(f"\n--- [{mode:>9} lam={lam}] plant={plant:.0f} "
                  f"xtgt={x} div={div:.2f} maxrun={mr} rep4={rp:.2f} ---",
                  flush=True)
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