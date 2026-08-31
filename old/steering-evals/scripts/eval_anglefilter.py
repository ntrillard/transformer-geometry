#!/usr/bin/env python3
"""eval_anglefilter.py — generate many blends, FILTER to the ~9 deg
coherent setpoint (the angle-invariant from eval_transition), then test
WHICH operators are coherent AT that fixed angle.

Big-leap question: eval_blendall found coherence needs the state to stay
ON the native->rotated arc (linear/rot family), and eval_transition found
~9 deg is the coherent angle. Here we GENERATE a wide set of blend
candidates, KEEP only those whose effective angle lands in [8,10] deg
(the filter), and measure seed-robust coherence of exactly those. If ALL
9-deg-on-arc blends are coherent -> '9 deg is a sufficient+operator-
independent setpoint'. If only SOME -> operator still matters at the
setpoint.

Budget(20s): 2 coherent-capable prompts, ~24 blend candidates filtered,
2 seeds x 5 tok. One model, no template.
Run: HF_TOKEN=<tok> timeout 18 python3 -u eval_anglefilter.py
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
SEEDS = [0, 1]
TARGET = 'city'
OUT = Path('../steering_geometry_results/anglefilter.csv')
LO, HI = 8.0, 10.0          # the coherent angle filter window

PROMPTS = [
    ('ask',  'If you ask me which European city is the most beautiful, I would say that'),
    ('japan', 'The capital of Japan is'),
]


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

    def ang(v1, v2):
        return float(torch.acos(((v1 / v1.norm()) * (v2 / v2.norm())).sum()
                                .clamp(-1, 1)) * 180 / math.pi)

    def rot_to_angle(vv, tid, theta):
        a = math.radians(theta)
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(a) + g * math.sin(a)) * vv.norm()

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

    # candidate generators: produce a readout vector from native vf + target
    # (the blend arg is a config; we then FILTER by the effective angle)
    def candidates(vf, tid):
        cands = []  # (label, vector)
        nrm = vf.norm()
        v1 = vf / vf.norm()
        Wb = Wn[tid].float()
        tauv = Wb - (v1 @ Wb) * v1
        g = (tauv / tauv.norm()).float()
        # rotations at many angles
        for deg in np.arange(4.0, 12.01, 0.5):
            a = math.radians(float(deg))
            v = (v1 * math.cos(a) + g * math.sin(a)) * nrm
            cands.append((f'rot{deg:.1f}', v))
        # linear blends at many weights (will yield angles < full rot angle)
        for w in np.arange(0.3, 1.001, 0.05):
            r = rot_to_angle(vf, tid, 20.0)   # reference rotated at 20 deg
            v = (float(w) * vf + (1 - float(w)) * r)
            v = v / v.norm() * nrm
            cands.append((f'lin{w:.2f}', v))
        # slerp blends
        for t in np.arange(0.3, 1.001, 0.1):
            coso = (v1 * (g)).sum() if False else ((v1) * (r := rot_to_angle(
                vf, tid, 20.0)) / (rot_to_angle(vf, tid, 20.0).norm())).sum()
            # simpler: slerp between v1 and rotated20
            r = rot_to_angle(vf, tid, 20.0); rn = r / r.norm()
            c = (v1 * rn).sum().clamp(-1, 1); om = torch.acos(c); so = math.sin(om)
            v = (math.sin((1 - t) * om) / so) * vf + \
                (math.sin(t * om) / so) * r
            v = v / v.norm() * nrm
            cands.append((f'slerp{t:.1f}', v))
        return cands

    rows = []
    print(f"[{MODEL}] blend candidates GENERATED then FILTERED to "
          f"angle in [{LO},{HI}] deg ({SEEDS} seeds)")
    print(f"  filter window: {LO}-{HI} deg (the coherent setpoint)")
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        tgt = closest(vf)
        cands = candidates(vf, tgt)
        print(f"\n==== {pname}: {pr!r} gap={gap:.1f} -> "
              f"{tok.decode([tgt])!r}  ({len(cands)} candidates) ====")
        kept = [c for c in cands if LO <= ang(vf, c[1]) <= HI]
        print(f"  [filtered] {len(kept)}/{len(cands)} blend candidates at "
              f"~9deg:")
        for lab, v in kept:
            a_eff = ang(vf, v)
            oks = []
            for sd in SEEDS:
                torch.manual_seed(sd)
                ids = ids0.clone()
                toks = []
                for step in range(NTOK):
                    if step == 0:
                        def inj(m, i, o, p=v):
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
                oks.append(strict_ok(toks, tok.decode(toks)))
            robust = int(sum(oks) >= 2)
            rows.append(dict(prompt=pname, label=lab, angle=round(a_eff, 2),
                             robust=robust,
                             seeds=''.join('Y' if o else 'n' for o in oks)))
            print("    %-10s ang=%5.2f seeds=%s robust=%d"
                  % (lab, a_eff, ''.join('Y' if o else 'n' for o in oks),
                     robust), flush=True)

    # summary: at the 9-deg setpoint, is coherence universal or operator-dependent?
    for pname in [p for p, _ in PROMPTS]:
        sel = [r for r in rows if r['prompt'] == pname]
        if not sel:
            continue
        nr = sum(r['robust'] for r in sel)
        print(f"\n  {pname}: {nr}/{len(sel)} filtered-9deg blends coherent "
              f"({'UNIVERSAL @ 9deg' if nr == len(sel) else 'operator-dependent'})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()