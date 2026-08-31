#!/usr/bin/env python3
"""eval_adcon2.py — generalize the coherent-angle setpoint law to NEW
prompts with 50-token generations.

Builds on eval_adcon (fit: angle = 0.308*gap + 4.45). Here:
  PART A: extend the angle-vs-gap fit with a WIDER prompt pool, return the
          fitted closed-form setpoint angle(gap).
  PART B: evaluate the fitted law on NEW prompts (NOT in the fit pool),
          at 50 tokens, with a MIDWAY (step 25) re-steer of the topic.
          Score each long gen (coherence text + metrics) and report whether
          the setpoint law holds on unseen prompts.

One model, no template. Run: HF_TOKEN=<tok> timeout 38 python3 -u
eval_adcon2.py
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
TARGET = 'city'
OUT = Path('../steering_geometry_results/adcon2.csv')

FITPOOL = [   # prompt for the angle-vs-gap fit
    ('ask',   'If you ask me which European city is the most beautiful, I would say that'),
    ('japan', 'The capital of Japan is'),
    ('olymp', 'The Olympic Games were held in'),
    ('france','The capital of France is'),
]
ONPROBE = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

NEWLONG = [  # NEW prompts (unseen in fit) for the 50-token gen
    ('spain',  'The capital of Spain is'),
    ('tower',  'The tallest tower in the world is in'),
    ('austr',  'The biggest city in Australia is'),
    ('visitnl','I love to visit new places, and my favorite city is'),
]
NTOK = 50
MID = 25
SEEDS = [0, 1]


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

    def nat_vL(ids):
        cv = {}
        hk = model.model.norm.register_forward_hook(
            lambda m, i, o: cv.__setitem__('v', o[0, -1, :].float()))
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        hk.remove()
        return cv['v'], L

    def rot_to_angle(vv, tid, theta):
        a = math.radians(theta)
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(a) + g * math.sin(a)) * vv.norm()

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

    # ---------- PART A: widen the fit ----------
    print(f"[{MODEL}] PART A: WIDER fit of onset angle vs gap", flush=True)
    fitpts = []
    for pname, pr in FITPOOL:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        gap = float(Ln[int(Ln.argmax())] - Ln[fam].max())
        tgt = closest(vf)
        onset = None
        for a in ONPROBE:
            vin = rot_to_angle(vf, tgt, a)
            # quick 2-seed coherence probe at this angle
            ok = []
            for sd in SEEDS:
                torch.manual_seed(sd)
                ids = ids0.clone(); toks = []
                for step in range(3):
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
                ok.append(coherent(toks, tok.decode(toks)))
            if all(ok):
                onset = a
                break
        fitpts.append((gap, onset))
        print("  %-7s gap=%5.1f  onset=%s" % (pname, gap,
                                              onset if onset else 'none'),
              flush=True)
    pts = [(g, o) for g, o in fitpts if o is not None]
    if len(pts) >= 2:
        gs = np.array([p[0] for p in pts]); os_ = np.array([p[1] for p in pts])
        k, b = np.polyfit(gs, os_, 1)
        corr = np.corrcoef(gs, os_)[0, 1]
        print(f"  FIT: angle(gap) = {k:.3f}*gap + {b:.3f}  "
              f"(n={len(pts)}, corr={corr:.2f})")
    else:
        k, b = 0.0, 7.0
        corr = 0.0
        print(f"  FIT fallback (only {len(pts)} onset points): angle=7")

    def setpoint(gap):
        a = k * gap + b
        return min(max(a, 4.0), 12.0)

    # ---------- PART B: 50-token gens on NEW prompts ----------
    print(f"\n[PART B] 50-token generns on NEW prompts, fitted "
          f"angle(gap), midway@{MID} re-steer")
    rows = []
    for pname, pr in NEWLONG:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        gap = float(Ln[int(Ln.argmax())] - Ln[fam].max())
        tgt = closest(vf)
        a0 = setpoint(gap)
        print(f"\n==== {pname}: {pr!r} gap={gap:.2f} -> "
              f"{tok.decode([tgt])!r}  a0={a0:.1f}deg ====", flush=True)
        for sd in SEEDS:
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                hs = []
                if step == 0 or step == MID:
                    if step == 0:
                        vv = rot_to_angle(vf, tgt, a0)
                    else:
                        vv2, _ = nat_vL(ids)
                        vv = rot_to_angle(vv2, closest(vv2), a0)
                    def inj(m, i, o, p=vv):
                        out = o.clone()
                        out[0, -1, :] = torch.as_tensor(
                            p, dtype=out.dtype, device=out.device)
                        return out
                    hs.append(model.model.norm.register_forward_hook(inj))
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
                for h in hs:
                    h.remove()
                nxt = sample(L)
                toks.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)],
                                dim=1)
            x = sum(1 for t in toks if t in fset)
            div = len(set(toks)) / len(toks)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(toks)), default=0)
            rp = rep4(toks)
            txt = tok.decode(toks)
            cint = int(coherent(toks, txt))
            rows.append(dict(prompt=pname, gap=round(gap, 2),
                             setpoint_angle=round(a0, 2), seed=sd, xtgt=x,
                             div=round(div, 3), maxrun=mr,
                             rep4=round(rp, 3), coherent=cint, text=txt))
            print(f"  --- {pname} sd{sd} a0={a0:.1f} xtgt={x} "
                  f"div={div:.2f} rep4={rp:.2f} coherent={cint} ---",
                  flush=True)
            print(f"      {txt}", flush=True)

    # summary
    print("\n-- NEW-prompt generalization (fitted law, 50 tok) --")
    for pname in [p for p, _ in NEWLONG]:
        sel = [r for r in rows if r['prompt'] == pname]
        nc = sum(r['coherent'] for r in sel)
        print(f"  {pname}: coherent {nc}/{len(sel)}  "
              f"(div range {min(r['div'] for r in sel):.2f}-"
              f"{max(r['div'] for r in sel):.2f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()