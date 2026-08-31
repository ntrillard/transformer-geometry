#!/usr/bin/env python3
"""eval_closest_note.py — the OLD way vs our hard recipe, on Gemma.

The user's challenge: the old chord methods steered toward the family
member CLOSEST to the current state (NOT the centroid, NOT a fixed hard
push), at GENTLE budgets (17 deg = 0.30 rad single-shot, 8 deg = 0.14
persistent), with NO anti-last. Commit 6cb9801 reported 'coherent
topical text' on Qwen with that. We never tested this against our hard
one-shot law recipe (a=0.378-0.477 + anti-last). Test on Gemma.

Modes (all target the FOOD family, prompt 'For dinner I made'):
  law-once      : steer 'chicken' a=2*gap/97+0.02=0.378 once + anti-last  [ours]
  close17-once  : step0 steer toward family member closest to the STATE at
                  a=0.30 (17 deg), then free sampling  [old single-shot]
  close08-pers  : EVERY step steer toward closest-to-state member at a=0.14
                  (8 deg), no anti-last  [old persistent]
  close15-pers  : same but a=0.15
  cent08-pers   : persistent toward the CENTROID at a=0.14 (old center)
  law-soft      : steer 'chicken' a=0.20 once + anti-last  [gentler ours]

closest-to-state = argmax over family rows of (row . current_unit_state),
recomputed each step (old inversion). anti-last uses the modern rot-away.

metrics: plant (any food word in first 10), x, dis, div, #SEP (native ' I'
count), rep4, maxrun + the decoded sample (the actual readability).
CSV: steering_geometry_results/closest_note_gemma.csv

Run: timeout 90 python3 -u eval_closest_note.py  # GEMMA-3-1B
"""
import csv
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
NTOK = 16
SEEDS = 1
A_REP = 0.15
FOOD = ['apple', 'banana', 'bread', 'cheese', 'chicken', 'grape',
        'honey', 'milk', 'rice', 'soup']
OUT = Path('../steering_geometry_results/closest_note_gemma.csv')


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()

    word2id = {}
    for w in FOOD:
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    food_ids = sorted(word2id.values())
    fset = set(food_ids)
    Wf = Wn[food_ids].float()

    ids0 = tok(PROMPT, add_special_tokens=False,
               return_tensors='pt').input_ids.to(DEV)
    cf = {}

    def hook_c(m, i, o):
        cf['v'] = o[0, -1, :].float()

    h = model.model.norm.register_forward_hook(hook_c)
    with torch.no_grad():
        L0 = model(ids0).logits[0, -1].float()
    h.remove()
    native = int(L0.argmax())
    vf = cf['v'].float()
    vfn = vf / vf.norm()
    nname = tok.decode([native])
    print(f"[{MODEL}] {PROMPT!r} native={nname!r}  food family n="
          f"{len(food_ids)}  closest-note vs law, NTOK={NTOK}")

    def closest_member(vv):
        """family member row most aligned with the (unit) state."""
        u = vv / vv.norm()
        return food_ids[int((Wf @ u).argmax())]

    def rot_toward(vv, tid, amt):
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(amt) + g * math.sin(amt)) * vv.norm()

    def anti(vv, tid, amt=A_REP):
        vv1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tauv = Wb - (vv1 @ Wb) * vv1
        gl = -tauv / tauv.norm()
        return (vv1 * math.cos(amt) + gl * math.sin(amt)) * vv.norm()

    def gen(mode):
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                vv = vf
                if mode == 'law-once' and step == 0:
                    t = word2id['chicken']
                    gap = float(L0[native] - L0[t])
                    a = 2 * gap / 97.0 + 0.02
                    vv = rot_toward(vv, t, a)
                elif mode == 'law-soft' and step == 0:
                    vv = rot_toward(vv, word2id['chicken'], 0.20)
                elif mode == 'close17-once' and step == 0:
                    t = closest_member(vv)
                    vv = rot_toward(vv, t, 0.30)
                elif mode == 'close08-pers':
                    t = closest_member(vv)
                    vv = rot_toward(vv, t, 0.14)
                elif mode == 'close15-pers':
                    t = closest_member(vv)
                    vv = rot_toward(vv, t, 0.15)
                elif mode == 'cent08-pers':
                    C = Wf.mean(0)
                    C = C / C.norm()
                    tau = C - (vv / vv.norm() @ C) * (vv / vv.norm())
                    g = tau / tau.norm()
                    vv = ((vv / vv.norm()) * math.cos(0.14) +
                          g * math.sin(0.14)) * vv.norm()
                # anti-last only for the modern modes
                if mode.startswith('law') and toks:
                    vv = anti(vv, toks[-1])

                def inj(m, i, o, p=vv):
                    out = o.clone()
                    out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                    device=out.device)
                    return out

                hi = model.model.norm.register_forward_hook(inj)
                try:
                    with torch.no_grad():
                        L = model(ids).logits[0, -1].float()
                finally:
                    hi.remove()
                p = torch.softmax(L.float(), dim=0)
                q = p.clone(); order = q.argsort(descending=True)
                k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
                msk = torch.zeros_like(q); msk[order[:k]] = 1
                qq = (q * msk) / (q * msk).sum()
                nxt = int(torch.multinomial(qq, 1))
                toks.append(int(nxt))
                ids = torch.cat([ids,
                                 torch.tensor([[nxt]], device=ids.device)],
                                dim=1)
            allres.append(toks)
        return allres

    modes = ['law-once', 'close17-once', 'close08-pers', 'close15-pers',
             'cent08-pers', 'law-soft']
    print(f"  {'mode':>13} {'plant':>6} {'x':>4} {'dis':>4} {'div':>5} "
          f"{'maxrun':>6} {'rep4':>5} {'#SEP':>5}  sample")
    rows = []
    for md in modes:
        gs = gen(md)
        for g in gs:
            x = sum(1 for t in g if t in fset)
            dis = len({t for t in g if t in fset})
            plant = 1.0 if any(t in g[:10] for t in fset) else 0.0
            div = len(set(g)) / len(g)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(g)), default=0)
            rp = rep4(g)
            nsep = sum(1 for t in g if t == native)
            rows.append(dict(model=MODEL, prompt=PROMPT, mode=md,
                             plant=plant, x=x, dis=dis, div=round(div, 3),
                             maxrun=mr, rep4=round(rp, 3), nsep=nsep,
                             sample=tok.decode(g)))
            print(f"  {md:>13} {plant:>6.1f} {x:>4d} {dis:>4d} "
                  f"{div:>5.2f} {mr:>6d} {rp:>5.2f} {nsep:>5}  "
                  f"{tok.decode(g)[:42]!r}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()