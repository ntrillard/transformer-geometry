#!/usr/bin/env python3
"""eval_metalab.py — RE-META-LEARN the anti-last coherence mode with the
FIXED coherence metric (the rep4 short-window bug polluted all of
metamode2-6). The big leap: with a correct label, does a cheap feature
finally SEPARATE coherent-free-run from collapse?

Recipe: ONE steer at step 0 to the fitted/7deg coherent angle toward the
closest family member, then ANTI-LAST (suppress that planted token at the
lm_head from step 1) -- the recipe that produced 'Paris is the most
beautiful'. Label (FIXED metric): coherent free-run 8-tok, >=2/3 seeds.

Features (cheap, natural forward; all ONE sort/softmax away):
  gap, f_rank (family rank), ent_nat (natural softmax entropy), cosang
  (alignment v to closest family), isfun, ctxlen.

Meta-fit: corr + LOO best separator per feature -> coherent. Also test the
OLD proxy (post-anti next-token entropy, ent_anti) corr vs coherence --
with the correct label, does the proxy PREDICT now (inversion was an
artifact of the broken metric)?

One model, no template. Run: HF_TOKEN=<tok> timeout 18 python3 -u
eval_metalab.py
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
NTOK = 8
SEEDS = [0, 1, 2]
TARGET = 'city'
OUT = Path('../steering_geometry_results/metalab.csv')
ANGLE = 7.0            # coherent angle (fixed, from onset fit)

PROMPTS = [
    ('ask',  'If you ask me which European city is the most beautiful, I would say that'),
    ('japan','The capital of Japan is'),
    ('spain','The capital of Spain is'),
    ('tower','The tallest tower in the world is in'),
    ('austr','The biggest city in Australia is'),
    ('visitnl','I love to visit new places, and my favorite city is'),
    ('olymp','The Olympic Games were held in'),
    ('france','The capital of France is'),
]
FUNCTION = {'the', 'a', 'an', 'i', 'is', 'was', 'to', 'of', 'in', 'that',
            'it', 'this', 'and', 'but', 'for', 'with', 'on', 'at', 'my',
            'his', 'her', 'there', 'were', 'are', 'be', 'as', 'by', 'from',
            'you', 'we', 'they', 'he', 'she', 'me', 'us', 'them', 'our',
            'their', 'then', 'so', 'will', 'would', 's', 't', 'do', 'does'}


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
            for h in hs:
                h.remove()
            nxt = sample(L)
            toks.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        return toks

    rows = []
    feats, ent_anti_all, y = [], [], []
    print(f"[{MODEL}] RE-meta-learn anti-last mode with FIXED metric "
          f"(coherent = {NTOK}-tok free-run, >=2/3 seeds, 7deg steer)")
    print("  %-8s %5s %5s %5s %5s %5s | %5s %5s | %s"
          % ('prompt', 'gap', 'frk', 'ent', 'cos', 'isf',
             'entA*', 'label', 'seeds'))
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
        isfun = 1 if tok.decode([nat_top]).strip().lower() in FUNCTION else 0
        ctxlen = ids0.shape[1]
        tgt = closest(vf)
        vp = rot_to_angle(vf, tgt, ANGLE)
        # OLD proxy: post-anti next-token entropy (steered forward + anti)
        Ls = None
        def inj(m, i, o, p=vp):
            out = o.clone()
            out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                            device=out.device)
            return out
        h = model.model.norm.register_forward_hook(inj)
        with torch.no_grad():
            Ls = model(ids0).logits[0, -1].float()
        h.remove()
        LsA = Ls.float().cpu().numpy().copy()
        LsA[tgt] = -float('inf')
        ent_anti = entnp(torch.softmax(torch.as_tensor(LsA), 0).numpy())
        # REAL label: shot_anti 8-tok free-run, >=2/3 seeds
        ok = [coherent(shot_anti_rollout(ids0, vp, tgt, sd),
                       tok.decode(shot_anti_rollout(ids0, vp, tgt, sd)))
              for sd in SEEDS]
        # (recompute per sd cleanly)
        ok = []
        seed_str = ''
        for sd in SEEDS:
            toks = shot_anti_rollout(ids0, vp, tgt, sd)
            seed_str += 'Y' if coherent(toks, tok.decode(toks)) else 'n'
            ok.append(coherent(toks, tok.decode(toks)))
        lab = 1 if sum(ok) >= 2 else 0
        feats.append([gap, f_rank, ent_nat, cosang, isfun, ctxlen])
        ent_anti_all.append(ent_anti)
        y.append(lab)
        rows.append(dict(prompt=pname, gap=round(gap, 2), f_rank=f_rank,
                         ent_nat=round(ent_nat, 2), cosang=round(cosang, 3),
                         isfun=isfun, ctxlen=ctxlen,
                         ent_anti=round(ent_anti, 2), label=lab, seeds=seed_str))
        print("  %-8s %5.1f %5d %5.2f %5.2f %5d | %5.2f %5d | %s"
              % (pname[:8], gap, f_rank, ent_nat, cosang, isfun,
                 ent_anti, lab, seed_str), flush=True)

    X = np.array(feats, dtype=float)
    P = np.array(ent_anti_all, dtype=float)
    y = np.array(y, dtype=float)
    n = len(y); nc = int(y.sum())
    names = ['gap', 'f_rank', 'ent_nat', 'cosang', 'isfun', 'ctxlen']
    print(f"\n  coherent: {nc}/{n}")
    best = (None, -1, None, None)
    for j, nm in enumerate(names):
        c = (np.corrcoef(X[:, j], y)[0, 1] if X[:, j].std() > 0 and
             y.std() > 0 else float('nan'))
        accs = []
        for th in np.unique(X[:, j]):
            accs.append(sum(1 for i in range(n)
                            if (X[i, j] <= th) == y[i]) / n)
        acc = max(accs); th = np.unique(X[:, j])[int(np.argmax(accs))]
        left = np.mean(y[X[:, j] <= th]) if (X[:, j] <= th).any() else 0
        side = '<=' if left >= np.mean(y[X[:, j] > th] if
                                       (X[:, j] > th).any() else y) else '>'
        print(f"  {nm:>7}: corr={c:+.3f}  LOO={acc:.2f} @{nm}{side}{th:.3f}")
        if acc > best[1]:
            best = (nm, acc, th, side)
    if P.std() > 0 and y.std() > 0:
        print(f"  OLD proxy (ent_anti) corr with corrected coherence = "
              f"{np.corrcoef(P, y)[0, 1]:+.3f}")
    bname, bacc, bth, bside = best
    naive = max(y.mean(), 1 - y.mean())
    print(f"  inferred mode detector {bname}{bside}{bth:.3f} "
          f"LOO={bacc:.2f} naive={naive:.2f}")
    if bacc > naive + 0.15 and y.std() > 0:
        print(f"  -> {bname} SEPARATES the mode (a real learned gate)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()