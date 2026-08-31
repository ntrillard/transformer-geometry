#!/usr/bin/env python3
"""eval_metamode4.py — scale-check the DYNAMIC anti-last recovery mode.

Follow-up to metamode3 (2084572, on disk only here): does the dynamic
recovery widening hold across MORE prompts? Dynamic = after graft plants
the topic token in context + anti suppresses it, one free-run step later
is the distribution soft (ent>=3.0 and p_max<0.5) or collapsed?

Only the DYNAMIC branch is measured (drop static) to fit more prompts in
<=10s. Features (natural forward): gap, f_rank, ent_nat, p_max, isfun,
ctxlen, cosang. Label from one steered+anti dynamic forward.

No commit - run on disk only.
Run: HF_TOKEN=<tok> timeout 10 python3 -u eval_metamode4.py
"""
import csv
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
OUT = Path('../steering_geometry_results/metamode4.csv')

PROMPTS = [
    'The capital of France is',
    'The capital of Japan is',
    'The capital of Italy is',
    'The Eiffel Tower is located in',
    'The Statue of Liberty is in',
    'The tallest building in the world is in',
    'My favorite city in the world is',
    'The best city in the world is',
    'I visited Paris last summer and it was',
    'If you ask me which European city is the most beautiful, I would say that',
    'The most populous city in the United States is',
    'The Olympic Games were held in',
]

FUNCTION = {'the', 'a', 'an', 'i', 'is', 'was', 'to', 'of', 'in', 'that',
            'it', 'this', 'and', 'but', 'for', 'with', 'on', 'at', 'my',
            'his', 'her', 'there', 'were', 'are', 'be', 'as', 'by', 'from',
            'you', 'we', 'they', 'he', 'she', 'me', 'us', 'them', 'our',
            'their', 'then', 'so', 'will', 'would', 's', 't', 'do', 'does'}


def entnp(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


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

    def forward(ids, norm_inject=None, anti_id=None):
        hooks = []
        if norm_inject is not None:
            def inj(m, i, o, p=norm_inject):
                out = o.clone()
                out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                device=out.device)
                return out
            hooks.append(model.model.norm.register_forward_hook(inj))
        if anti_id is not None:
            def anti(m, i, o, tid=anti_id):
                out = o.clone()
                out[0, -1, tid] = -30.0
                return out
            hooks.append(model.lm_head.register_forward_hook(anti))
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        for h in hooks:
            h.remove()
        return L

    feats, y = [], []
    print(f"[{MODEL}] family={TARGET}  dynamic anti-last recovery, "
          f"{len(PROMPTS)} prompts")
    print("  %-44s ctx  gap  f-rk  entN p__N | dyA dyL" % 'prompt')
    for sd, pr in enumerate(PROMPTS):
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        order = Ln.argsort(descending=True).tolist()
        f_rank = order.index(fam[int(Ln[fam].argmax())]) + 1
        pnat = torch.softmax(Ln.float().cpu(), 0).numpy()
        ent_nat = entnp(pnat)
        p_max = float(pnat.max())
        gap = float(Ln[nat_top] - Ln[fam].max())
        cosang = float((vf / vf.norm()) @ Wn[closest(vf)])
        isfun = 1 if tok.decode([nat_top]).strip().lower() in FUNCTION else 0
        ctxlen = ids0.shape[1]
        feats.append([gap, f_rank, ent_nat, p_max, isfun, ctxlen, cosang])

        alpha = 2 * (gap / 97.0) + 0.02
        tgt = closest(vf)
        vp = rot(vf, tgt, alpha)
        Ls = forward(ids0, norm_inject=vp)
        nxt0 = int(Ls.argmax())
        ids1 = torch.cat([ids0, torch.tensor([[nxt0]], device=DEV)], dim=1)
        L1 = forward(ids1, anti_id=tgt)
        p1 = torch.softmax(L1.float().cpu(), 0).numpy()
        ent_dy = entnp(p1); p_dy = float(p1.max())
        lab = 1 if (ent_dy >= 3.0 and p_dy < 0.5) else 0
        y.append(lab)
        print("  %-44s %3d %4.1f %5d %5.2f %4.2f | %5.2f %3d"
              % (pr[:44], ctxlen, gap, f_rank, ent_nat, p_max,
                 ent_dy, lab), flush=True)

    X = np.array(feats, dtype=float); y = np.array(y, dtype=float)
    names = ['gap', 'f_rank', 'ent', 'p_max', 'isfun', 'ctxlen', 'cosang']
    n = len(y); nd = int(y.sum())
    print(f"\n  dynamic recovery: {nd}/{n}")
    best = (None, -1, None, None)
    for j, nm in enumerate(names):
        c = (np.corrcoef(X[:, j], y)[0, 1] if X[:, j].std() > 0
             else float('nan'))
        accs = []
        for th in np.unique(X[:, j]):
            accs.append(sum(1 for i in range(n)
                            if (X[i, j] <= th) == y[i]) / n)
        acc = max(accs); th = np.unique(X[:, j])[int(np.argmax(accs))]
        side = '<=' if np.mean(y[X[:, j] <= th]) >= np.mean(y[X[:, j] > th]) \
            else '>'
        print(f"    {nm:>7}: corr={c:+.3f}  LOO={acc:.2f} @{nm}{side}{th:.3f}")
        if acc > best[1]:
            best = (nm, acc, th, side)
    bname, bacc, bth, bside = best
    naive = max(y.mean(), 1 - y.mean())
    print(f"  inferred detector {bname}{bside}{bth:.3f} LOO={bacc:.2f} "
          f"naive={naive:.2f}")

    rows = [dict(prompt=p, **{nm: X[i, j] for j, nm in enumerate(names)},
                 dynamic_label=int(y[i])) for i, p in enumerate(PROMPTS)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()