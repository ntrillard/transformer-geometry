#!/usr/bin/env python3
"""eval_coherent_angles.py — PERFECT the coherent-angle recipe: generate
text with DIFFERENT blends at the FOUND per-prompt coherent angle.

anglesweep (fixed metric) found robust coherent angles per prompt:
  ask  -> 8 deg (3/3 seeds), japan -> 7 deg (2/3). This probe takes that
  FOUND angle and produces FREE-RUN TEXT using three different blend
  constructions that all land at that angle:
    rot   : on-arc rotation by theta
    lin   : linear blend tuned so effective angle == theta
    slerp : spherical blend tuned so effective angle == theta
  Longer NTOK so you can actually SEE the prose quality at the coherent
  angle (not just a binary metric), across seeds.

One model, no template. Run: HF_TOKEN=<tok> timeout 20 python3 -u
eval_coherent_angles.py
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
NTOK = 16
SEEDS = [0, 1, 2]
TARGET = 'city'
OUT = Path('../steering_geometry_results/coherent_angles.csv')

# the FOUND cohesive-angle per prompt (from eval_anglesweep)
CONFIG = [
    ('ask',   'If you ask me which European city is the most beautiful, I would say that', 8.0),
    ('japan', 'The capital of Japan is', 7.0),
]
BLENDS = ['rot', 'lin', 'slerp']


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

    def match_blend(vv, r20, theta_target, nrm, kind):
        """blend vv toward r20 so effective angle == theta_target."""
        rn = r20 / r20.norm(); v1 = vv / vv.norm()
        coso = (v1 * rn).sum().clamp(-1, 1); om = torch.acos(coso)
        so = math.sin(om)
        best = None; best_err = 1e9
        for lam in np.arange(0.0, 1.001, 0.005):
            if kind == 'lin':
                c = (lam * vv + (1 - lam) * r20)
            else:
                t = 1 - lam
                c = (math.sin((1 - t) * om) / (so + 1e-12)) * vv + \
                    (math.sin(t * om) / (so + 1e-12)) * r20
            c = c / c.norm() * nrm
            err = abs(ang(vv, c) - theta_target)
            if err < best_err:
                best_err = err; best = (lam, c)
        return best

    rows = []
    print(f"[{MODEL}] coherent-angle ({NTOK}-tok free-run) text, "
          f"{len(SEEDS)} seeds, 3 blends")
    for pname, pr, theta in CONFIG:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        gap = float(Ln[int(Ln.argmax())] - Ln[fam].max())
        tgt = closest(vf)
        nrm = vf.norm()
        r20 = rot_to_angle(vf, tgt, 20.0)
        print(f"\n==== {pname}: {pr!r} gap={gap:.1f} -> "
              f"{tok.decode([tgt])!r}  COHERENT ANGLE={theta} deg ====")
        for bname in BLENDS:
            if bname == 'rot':
                vin = rot_to_angle(vf, tgt, theta)
                blab = f'rot{theta}'
            else:
                lam, c = match_blend(vf, r20, theta, nrm, bname)
                vin = c
                blab = f'{bname}@{theta}(lam={lam:.2f})'
            a_eff = ang(vf, vin)
            print(f"\n--- [{bname:>5} {blab}] ang_eff={a_eff:.1f} ---",
                  flush=True)
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
                    ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)],
                                    dim=1)
                x = sum(1 for t in toks if t in fset)
                div = len(set(toks)) / len(toks)
                mr = max((sum(1 for _ in grp) for _, grp in
                          itertools.groupby(toks)), default=0)
                rp = rep4(toks)
                txt = tok.decode(toks)
                vlam = None
                if 'lam=' in blab:
                    try:
                        vlam = round(float(blab.split('lam=')[1][:-1]), 3)
                    except Exception:
                        vlam = None
                rows.append(dict(prompt=pname, theta=theta, blend=bname,
                                 angle_eff=round(a_eff, 2), seed=sd,
                                 plant=float(any(t in toks[:6]
                                                 for t in fset)),
                                 xtgt=x, div=round(div, 3), maxrun=mr,
                                 rep4=round(rp, 3), text=txt))
                print(f"    sd{sd}: plant={any(t in toks[:6] for t in fset)}"
                      f" xtgt={x} div={div:.2f} maxrun={mr} rep4={rp:.2f}",
                      flush=True)
                print(f"        {txt}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()