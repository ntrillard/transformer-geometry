#!/usr/bin/env python3
"""eval_why2.py — TEST the mechanism: anti-last coherence depends on the
TOPIC NOT RE-LATCHING after the anti window expires, and a SUSTAINED anti
(suppress topic all the way) is the fix.

From eval_why: coherent vs collapse, the pivot (step-1 = is/was/.) is
identical; the difference is the TRAJECTORY. The observed loops are
'is Cairo is Cairo' - the topic token re-latches as soon as anti stops
(window=2 => free at step 3). MECHANISM CHECK + FIX:
  A) measure P(relatch) = P(step2 = topic) under window anti
     (one forward: context [T, pivot], anti active) - predicts collapse?
  B) sustained anti (window=ALL) vs window=2: does it kill the loops?
Room even if the pivot differs, the re-latch test is on the first pivot.

One model, no template. Run: HF_TOKEN=<tok> timeout 18 python3 -u
eval_why2.py
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
OUT = Path('../steering_geometry_results/why2.csv')
ANGLE = 10.0

# mixed known set (both coherent and collapse) to test the fix
PROMPTS = [
    ('ask',   'If you ask me which European city is the most beautiful, I would say that'),
    ('spain', 'The capital of Spain is'),
    ('japan', 'The capital of Japan is'),
    ('mona',  'The Mona Lisa is displayed in'),
    ('novel', 'The story takes place in'),
    ('hockey','The hockey championship was held in'),
    ('austr', 'The biggest city in Australia is'),
    ('itcap', 'The capital of Italy is'),
    ('decap', 'The capital of Germany is'),
    ('france','The capital of France is'),
]


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
                if 1 <= step <= window:   # window=large => sustained anti
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
    print(f"[{MODEL}] WHY2: sustained-anti fix + relatch predictor "
          f"({len(PROMPTS)} prompts, {len(SEEDS)} seeds)")
    print("  %-8s %6s %6s %6s | %s | %s" % ('prompt', 'pRe', 'w2',
                                            'wALL', 'coherent(w2/ALL?',
                                            'seeds_w2/ALL'))
    feat_p, y_w2, y_all = [], [], []
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        tgt = closest(vf)
        vp = rot_to_angle(vf, tgt, ANGLE)

        # -- relatch predictor: after [T, pivot] with anti, P(T at step2) --
        # (pivot = the argmax under anti at step1, deterministic)
        hs = [model.model.norm.register_forward_hook(
            lambda m, i, o, p=vp: inj2(o, p))]
        try:
            with torch.no_grad():
                L0 = model(ids0).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        tok0 = int(L0.argmax())
        ids1 = torch.cat([ids0, torch.tensor([[tok0]], device=DEV)], dim=1)
        hs = [model.lm_head.register_forward_hook(
            lambda m, i, o, tid=tgt: anti2(o, tid))]
        try:
            with torch.no_grad():
                L1 = model(ids1).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        pivot = int(L1.argmax())
        ids2 = torch.cat([ids1, torch.tensor([[pivot]], device=DEV)], dim=1)
        hs = [model.lm_head.register_forward_hook(
            lambda m, i, o, tid=tgt: anti2(o, tid))]
        try:
            with torch.no_grad():
                L2 = model(ids2).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        p_relatch = float(torch.softmax(L2, 0)[tgt])

        def run(window):
            # recompute cleanly
            oks = []
            seed_str = ''
            for sd in SEEDS:
                toks = shot_anti(ids0, vp, tgt, sd, window)
                c = coherent(toks, tok.decode(toks))
                oks.append(c)
                seed_str += 'Y' if c else 'n'
            return sum(oks) >= 2, seed_str

        c2, s2 = run(2)
        call, sall = run(99)
        rows.append(dict(prompt=pname, gap=round(gap, 2),
                         p_relatch=round(p_relatch, 3),
                         win2=int(c2), win_all=int(call),
                         seeds_w2=s2, seeds_all=sall,
                         pivot=tok.decode([pivot]).strip()))
        feat_p.append(p_relatch)
        y_w2.append(1 if c2 else 0)
        y_all.append(1 if call else 0)
        print("  %-8s %6.2f %6d %6d | %s | %s"
              % (pname[:8], p_relatch, int(c2), int(call), s2, sall),
              flush=True)

    y_w2 = np.array(y_w2, dtype=float)
    y_all = np.array(y_all, dtype=float)
    P = np.array(feat_p, dtype=float)
    n = len(y_w2)
    print(f"\n  coherent w2: {int(y_w2.sum())}/{n}   wALL: "
          f"{int(y_all.sum())}/{n}")
    if P.std() > 0 and y_w2.std() > 0:
        print(f"  corr(P_relatch -> coherent@w2) = "
              f"{np.corrcoef(P, y_w2)[0, 1]:+.3f}")
    ups = [i for i in range(n) if y_all[i] > y_w2[i]]
    downs = [i for i in range(n) if y_all[i] < y_w2[i]]
    print(f"  sustained-anti lifted: {[PROMPTS[i][0] for i in ups]}")
    print(f"  sustained-anti hurt  : {[PROMPTS[i][0] for i in downs]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


def inj2(out, p):
    out = out.clone()
    out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype, device=out.device)
    return out


def anti2(out, tid):
    out = out.clone()
    out[0, -1, tid] = -30.0
    return out


if __name__ == "__main__":
    main()