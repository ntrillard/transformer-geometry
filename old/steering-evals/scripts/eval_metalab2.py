#!/usr/bin/env python3
"""eval_metalab2.py — LEARNED anti-last coherent controller.

Train a detector on a subset of prompts, evaluate the anti-last
controller on HELD-OUT prompts:
  features (cheap, natural+steered forward): gap, f_rank, ent_nat,
  cosang, ent_anti (post-anti entropy -> strongest predictor in
  eval_metalab, corr +0.41).
  label (fixed metric): 6-tok shot_anti free-run coherent on >=2/3 seeds.
Detector: best single-feature threshold fit on TRAIN only.
Controller on HELD-OUT: steer+anti only when detector says recoverable;
measure coherent-hit-rate vs always-steer and never-steer.

CARE: all helpers use the FIXED coherence metric (rep4 returns 0.0 for
<4 tokens), numpy conversions via .cpu(), hooks cleaned in finally.

One model, no template. Run: HF_TOKEN=<tok> timeout 18 python3 -u
eval_metalab2.py
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
SEEDS = [0, 1, 2]
TARGET = 'city'
OUT = Path('../steering_geometry_results/metalab2.csv')
ANGLE = 7.0

PROMPTS = [
    ('ask',  'If you ask me which European city is the most beautiful, I would say that'),
    ('japan','The capital of Japan is'),
    ('spain','The capital of Spain is'),
    ('tower','The tallest tower in the world is in'),
    ('austr','The biggest city in Australia is'),
    ('visitnl','I love to visit new places, and my favorite city is'),
    ('olymp','The Olympic Games were held in'),
    ('france','The capital of France is'),
    ('paris','I visited Paris last summer and it was'),
    ('traveld','I would love to travel to'),
    ('bern','The capital of Switzerland is'),
    ('concert','The concert was held in'),
]


def rep4(toks):
    if len(toks) < 4:
        return 0.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return sum(1 for i in range(len(toks) - 3) if n4[i] in n4[i + 1:]) \
        / (len(toks) - 3)


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

    def shot_anti_rollout(ids0, vp, tgt, sd):
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
                if step >= 1:
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
    print(f"[{MODEL}] learned anti-last controller "
          f"({len(PROMPTS)} prompts, NTOK={NTOK}, angle={ANGLE})")
    print("  %-9s %5s %5s %5s %5s %5s | %5s | %s"
          % ('prompt', 'gap', 'frk', 'entN', 'cos', 'entA',
             'label', 'seeds'))
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        order = Ln.argsort(descending=True).tolist()
        f_rank = order.index(fam[int(Ln[fam].argmax())]) + 1
        ent_nat = entnp(torch.softmax(Ln.float().cpu(), 0).numpy())
        cosang = float((vf / vf.norm()) @ Wn[closest(vf)])
        tgt = closest(vf)
        vp = rot_to_angle(vf, tgt, ANGLE)
        # ent_anti proxy (steered forward, zero target, softmax entropy)
        hs = [model.model.norm.register_forward_hook(
            lambda m, i, o, p=vp: inj2(o, p))]
        try:
            with torch.no_grad():
                Ls = model(ids0).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        LsA = Ls.float().cpu().numpy().copy()
        LsA[tgt] = -float('inf')
        ent_anti = entnp(torch.softmax(torch.as_tensor(LsA), 0).numpy())
        # real label (>=2/3 seeds)
        ok = []
        seed_str = ''
        for sd in SEEDS:
            toks = shot_anti_rollout(ids0, vp, tgt, sd)
            c = coherent(toks, tok.decode(toks))
            seed_str += 'Y' if c else 'n'
            ok.append(c)
        lab = 1 if sum(ok) >= 2 else 0
        rows.append(dict(prompt=pname, gap=round(gap, 2), f_rank=f_rank,
                         ent_nat=round(ent_nat, 2), cosang=round(cosang, 3),
                         ent_anti=round(ent_anti, 2), label=lab,
                         seeds=seed_str))
        print("  %-9s %5.1f %5d %5.2f %5.2f %5.2f | %5d | %s"
              % (pname[:9], gap, f_rank, ent_nat, cosang, ent_anti,
                 lab, seed_str), flush=True)

    X = np.array([[r['gap'], r['f_rank'], r['ent_nat'], r['cosang'],
                   r['ent_anti']] for r in rows], dtype=float)
    y = np.array([r['label'] for r in rows], dtype=float)
    FNAMES = ['gap', 'f_rank', 'ent_nat', 'cosang', 'ent_anti']
    n = len(y)
    print(f"\n  coherent labels: {int(y.sum())}/{n}")

    # fit detector on FIRST 2/3 (train), evaluate on LAST 1/3 (held-out)
    ntr = int(round(n * 2 / 3))
    tr_idx = list(range(ntr))
    te_idx = list(range(ntr, n))
    best = (None, -1, None, None)
    for j, nm in enumerate(FNAMES):
        accs = []
        for th in np.unique(X[tr_idx, j]):
            accs.append(sum(1 for i in tr_idx
                            if (X[i, j] <= th) == y[i]) / len(tr_idx))
        acc = max(accs); th = np.unique(X[tr_idx, j])[int(np.argmax(accs))]
        tr = np.zeros(n, dtype=bool); tr[tr_idx] = True
        mask = X[:, j] <= th
        left = y[mask & tr].mean() if (mask & tr).any() else 0.0
        gr = (~mask) & tr
        left_gt = y[gr].mean() if gr.any() else 0.0
        side = '<=' if left >= left_gt else '>'
        print(f"  train {nm:>7}: acc={acc:.2f} @{nm}{side}{th:.3f}")
        if acc > best[1]:
            best = (nm, acc, th, side)
    bname, bacc, bth, bside = best
    print(f"  detector on train: {bname}{bside}{bth:.3f} "
          f"(train acc={bacc:.2f})")

    # evaluate controller on held-out
    def predict(i):
        v = X[i, FNAMES.index(bname)]
        return (v <= bth) if bside == '<=' else (v > bth)

    always = sum(y[te_idx])
    never = sum(1 for i in te_idx if X[i, 1] <= 2)
    det = sum(1 for i in te_idx
              if (predict(i) and y[i] == 1) or
              (not predict(i) and X[i, 1] <= 2))
    nt = len(te_idx)
    print(f"\n-- held-out controller (n={nt}) --")
    print(f"  always-steer : {always}/{nt}")
    print(f"  never-steer  : {never}/{nt}")
    print(f"  learned      : {det}/{nt}   (rule {bname}{bside}{bth:.2f})")
    print(f"  improvement vs always: {int(det) - int(always):+d}")

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


if __name__ == "__main__":
    main()