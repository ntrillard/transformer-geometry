#!/usr/bin/env python3
"""eval_antidote.py — the big-leap follow-up to eval_metamode.

eval_metamode inferred (single-step, one model, 5s): on base Gemma the
readout steer has ONE mode - ATTRACTOR-GRAFT. 10/10 prompts, gaps
5.5-19.8, ctx 4-16: a single law-budget steer (2*(gap/97)+0.02 toward
closest family member) collapses the distribution onto that one family
token, post_max 0.91-1.00. Inferred: no 'steer-safe' class exists on
this model; naive baseline 1.00 unbeatable.

Diagnosis from the recipe corpus: graft is the PLANTING MECHANISM
(guaranteed), and the committed recipe (1356f2d) already converts a
graft into prose with anti-last (suppress the planted target token).
The UNTESTED corner: does ONE shot + anti-last give coherent+topical
text at LONG context on the BASE model? (longwalk showed per-step walk
= 'Paris Paris Paris', shot-anti was never tried there.)

Modes (all at the readout):
  walk     : per-step steer toward closest family member (longwalk
             control - known collapse)
  shot_anti: ONE law-budget steer at step 0 toward closest family
             member, then from step 1 FREE-RUN with the planted target
             token logit zeroed at the lm_head each step (anti-last).

Prompts: same as longwalk base rows (baselines already in that CSV):
  short 'The capital of France is'
  long  'If you ask me which European city is the most beautiful...'
NTOK=8, 1 seed, 2 modes -> ~32 forwards, well under 10s.
CSV persisted. No chat template.
Run: HF_TOKEN=<tok> timeout 15 python3 -u eval_antidote.py
"""
import csv
import itertools
import math
import time
from pathlib import Path

import torch

import steering_geometry_test as SGT
from eval_nb_quick import CLASSES

MODEL = 'google/gemma-3-1b-pt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 8
SEEDS = [0]
TARGET = 'city'
OUT = Path('../steering_geometry_results/antidote.csv')

PROMPTS = [
    ('short', 'The capital of France is'),
    ('long', 'If you ask me which European city is the most beautiful, '
             'I would say that'),
]


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return sum(1 for i in range(len(toks) - 3)
               if n4[i] in n4[i + 1:]) / (len(toks) - 3)


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

    def sample(L):
        p = torch.softmax(L.float(), dim=0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk) / (q * msk).sum()
        return int(torch.multinomial(qq, 1))

    def nat_state(ids):
        cv = {}
        hk = model.model.norm.register_forward_hook(
            lambda m, i, o: cv.__setitem__('v', o[0, -1, :].float()))
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        hk.remove()
        return cv['v'], L

    rows = []
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_state(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        alpha = 2 * (gap / 97.0) + 0.02
        tgt = closest(vf)
        vp = rot(vf, tgt, alpha)
        print(f"\n[{pname}] {pr[:44]!r} native={tok.decode([nat_top])!r} "
              f"gap={gap:.1f} alpha={alpha:.3f} tgt={tok.decode([tgt])!r}")

        for mode in ['walk', 'shot_anti']:
            for sd in SEEDS:
                torch.manual_seed(sd)
                ids = ids0.clone()
                toks = []
                for step in range(NTOK):
                    if mode == 'walk':
                        vv, _ = nat_state(ids)
                        tg = closest(vv)
                        vv2 = rot(vv, tg, alpha)

                        def inj(m, i, o, p=vv2):
                            out = o.clone()
                            out[0, -1, :] = torch.as_tensor(
                                p, dtype=out.dtype, device=out.device)
                            return out

                        hi = model.model.norm.register_forward_hook(inj)
                        with torch.no_grad():
                            L = model(ids).logits[0, -1].float()
                        hi.remove()
                        nxt = sample(L)
                    else:
                        hooks = []
                        if step == 0:
                            def inj(m, i, o, p=vp):
                                out = o.clone()
                                out[0, -1, :] = torch.as_tensor(
                                    p, dtype=out.dtype, device=out.device)
                                return out

                            hooks.append(model.model.norm.
                                         register_forward_hook(inj))

                        if step >= 1:
                            def anti(m, i, o, tid=tgt):
                                out = o.clone()
                                out[0, -1, tid] = -30.0
                                return out

                            hooks.append(model.lm_head.
                                         register_forward_hook(anti))
                        with torch.no_grad():
                            L = model(ids).logits[0, -1].float()
                        for h in hooks:
                            h.remove()
                        nxt = sample(L)
                    toks.append(nxt)
                    ids = torch.cat([ids, torch.tensor(
                        [[nxt]], device=DEV)], dim=1)
                x = sum(1 for t in toks if t in fset)
                dis = len({t for t in toks if t in fset})
                plant = 1.0 if any(t in toks[:6] for t in fset) else 0.0
                div = len(set(toks)) / len(toks)
                mr = max((sum(1 for _ in grp) for _, grp in
                          itertools.groupby(toks)), default=0)
                rp = rep4(toks)
                txt = tok.decode(toks)
                rows.append(dict(model=MODEL, prompt=pname, mode=mode,
                                 seed=sd, plant=plant, xtgt=x, dis=dis,
                                 div=round(div, 3), maxrun=mr,
                                 rep4=round(rp, 3), text=txt))
                print(f"  {mode:>9} plant={plant:.0f} xtgt={x} dis={dis} "
                      f"div={div:.2f} maxrun={mr} rep4={rp:.2f}  "
                      f"{txt[:56]!r}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()