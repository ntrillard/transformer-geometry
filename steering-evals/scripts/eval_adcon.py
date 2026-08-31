#!/usr/bin/env python3
"""eval_adcon.py — (A) FIT the per-prompt coherent angle vs a cheap feature
(gap) to get a CLOSED-FORM setpoint, then (B) validate on multiple LONG
generations with the topic steered MIDWAY using that same method.

Part A (fit): for a prompt pool, measure gap (native logit - family max,
one sort) and find the ON-SET coherent angle (first probe angle where BOTH
seeds are coherent) with the FIXED coherence metric. Fit angle = a + b*gap
(closed form -> 'find the correct angle from the original final state'
automated).

Part B (long gen + midway steer): using the fitted angle(gap), generate
LONG free-run text; at the MIDWAY point re-apply the coherent-angle steer
(same method, toward closest family) to keep/steer the topic. 2 prompts,
40-token generations, print text live.

One model, no template. Run: HF_TOKEN=<tok> timeout 38 python3 -u
eval_adcon.py
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
OUT = Path('../steering_geometry_results/adcon.csv')

POOL = [   # (name, prompt) for the angle-vs-gap fit
    ('ask',   'If you ask me which European city is the most beautiful, I would say that'),
    ('japan', 'The capital of Japan is'),
    ('eif',   'The Eiffel Tower is located in'),
    ('olymp', 'The Olympic Games were held in'),
    ('visit', 'People from all over the world visit'),
    ('france','The capital of France is'),
]
ONPROBE = [6.0, 7.0, 8.0, 9.0, 10.0]   # coarse onset probe angles

LONG = [   # long generations with midway steer
    ('ask',  'If you ask me which European city is the most beautiful, I would say that'),
    ('japan', 'The capital of Japan is'),
]
NTOKinLONG = 40
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

    def gen(ids0, vecs_at_steps, ntok, seeds):
        """vecs_at_steps: dict step->vector to inject at that step's final
        norm (before sampling)."""
        results = []
        for sd in seeds:
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(ntok):
                hs = []
                if step in vecs_at_steps:
                    vv = vecs_at_steps[step]
                    def inj(m, i, o, p=vv):
                        out = o.clone()
                        out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                        device=out.device)
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
            results.append(toks)
        return results

    # ============ PART A: fit onset angle vs gap ============
    print(f"[{MODEL}] PART A: fit coherent ON-SET angle vs gap "
          f"(2 seeds, NTOK probe)")
    fitpts = []
    for pname, pr in POOL:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        gap = float(Ln[int(Ln.argmax())] - Ln[fam].max())
        tgt = closest(vf)
        onset = None
        for a in ONPROBE:
            vin = rot_to_angle(vf, tgt, a)
            toks = gen(ids0, {0: vin}, 4, SEEDS)
            c = [coherent(t, tok.decode(t)) for t in toks]
            if all(c):
                onset = a
                break
        fitpts.append((gap, onset))
        print("  %-7s gap=%5.1f  onset=%s" % (pname, gap,
                                              onset if onset else 'none'),
              flush=True)
    # linear fit on points with an onset
    pts = [(g, o) for g, o in fitpts if o is not None]
    if len(pts) >= 2:
        gs = np.array([p[0] for p in pts]); os_ = np.array([p[1] for p in pts])
        k, b = np.polyfit(gs, os_, 1)
        print(f"  FIT: angle(gap) = {k:.3f}*gap + {b:.3f}  "
              f"(n={len(pts)}, corr={np.corrcoef(gs, os_)[0,1]:.2f})")
    else:
        k, b = 0.0, 8.0
        print(f"  FIT: insufficient onset points ({len(pts)}); "
              f"fallback angle=8")

    def setpoint(gap):
        a = k * gap + b
        return min(max(a, 5.0), 12.0)

    # ============ PART B: LONG generations, MIDWAY steer ============
    print(f"\n[PART B] LONG {NTOKinLONG}-token generations, topic steered "
          f"MIDWAY with fitted angle(gap)")
    for pname, pr in LONG:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        gap = float(Ln[int(Ln.argmax())] - Ln[fam].max())
        tgt = closest(vf)
        a0 = setpoint(gap)
        # step0 steer + midway steer (recompute fresh final state + target)
        def steervec(ids):
            vv, _ = nat_vL(ids)
            tt = closest(vv)
            return rot_to_angle(vv, tt, setpoint(
                float(Ln[int(Ln.argmax())] - Ln[fam].max()))) \
                if False else rot_to_angle(vv, tt, a0)
        mid = NTOKinLONG // 2
        tok0 = gen(ids0, {0: rot_to_angle(vf, tgt, a0)}, 1, [0])[0]
        ids_mid = torch.cat([ids0, torch.tensor([[tok0[0]]], device=DEV)],
                            dim=1)
        # simpler: build vecs: step0 = steer(a0); mid = steer(a0) at mid
        # compute mid vector from the midway context
        # we'll recompute mid steer using a fresh forward at step mid-1 is
        # complex; approximate: use step0 steer at step 0, and at `mid`
        # re-inject same-class steer using a fresh nat forward of the
        # context-so-far is not available in gen(); instead do 2-phase:
        # run first half free (steer at 0), then read state, steer at mid.
        rows = []
        for sd in SEEDS:
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            vmid = None
            for step in range(NTOKinLONG):
                hs = []
                if step == 0 or step == mid:
                    if step == 0:
                        vv = rot_to_angle(vf, tgt, a0)
                    else:  # midway: re-steer to current closest family
                        vv2, _ = nat_vL(ids)
                        tt2 = closest(vv2)
                        vv = rot_to_angle(vv2, tt2, a0)
                    def inj(m, i, o, p=vv):
                        out = o.clone()
                        out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                        device=out.device)
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
            rows.append(dict(prompt=pname, fitted_angle=round(a0, 2),
                             gap=round(gap, 2), seed=sd, xtgt=x,
                             div=round(div, 3), maxrun=mr,
                             rep4=round(rp, 3), text=txt))
            print(f"\n--- {pname} sd{sd} (a0={a0:.1f}deg, midway@{mid}) "
                  f"xtgt={x} div={div:.2f} rep4={rp:.2f} ---", flush=True)
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