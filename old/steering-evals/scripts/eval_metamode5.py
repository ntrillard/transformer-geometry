#!/usr/bin/env python3
"""eval_metamode5.py — meta-learn the COHERENCE mode with REAL labels.

THE CORRECTION: metamode2/3/4 labeled 'recovery' with a single-step
next-token entropy proxy (post-anti ent>=3 & p_max<0.5). eval_realroll
proved that proxy is ANTI-predictive of true 8-token coherence:
  - fr (gap 6.0): proxy GOOD -> real COLLAPSE ('Paris...France is...')
  - jpn (7.2):    proxy GOOD -> real collapse/mix
  - ask (8.8):    proxy BAD  -> real div 1.00 3/3 seeds (BEST output)
So this probe meta-learns the mode that matters using REAL rollout
coherence as the label (run shot_anti, measure div/rep4/maxrun), not a
proxy. Then correlate the same cheap features + the old proxy to see
what ACTUALLY predicts coherence (and whether ent_anti is inverted).

Cheap per prompt (tight budget, 1 seed, 6-token rollout):
  natural : features gap, ent_nat, cosang, ctxlen
  steered : ent_anti (the OLD dynamic proxy value)
  real    : shot_anti 6-token rollout -> coherent = rep4==0 and div>=0.7
Label = real coherence. Fit feature -> coherence (corr + LOO stump) for
the cheap features AND the old proxy. Report the BIG-LEAP finding.

One model, no template, <=10s.
Run: HF_TOKEN=<tok> timeout 10 python3 -u eval_metamode5.py
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
NTOK = 6
TARGET = 'city'
OUT = Path('../steering_geometry_results/metamode5.csv')

# hand-picked to SPAN the real regimes (from eval_realroll + prior):
#   ask-like (recoverable) vs fr-like (collapse)
PROMPTS = [
    ('ask',  'If you ask me which European city is the most beautiful, I would say that'),
    ('fr',   'The capital of France is'),
    ('jpn',  'The capital of Japan is'),
    ('olym', 'The Olympic Games were held in'),
    ('trav', 'I would love to travel to'),
    ('visit','People from all over the world visit'),
    ('eif',  'The Eiffel Tower is located in'),
    ('best', 'My favorite city in the world is'),
]


def rep4(toks):
    if len(toks) < 4:
        return 1.0
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

    def sample(L):
        p = torch.softmax(L.float(), 0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk) / (q * msk).sum()
        return int(torch.multinomial(qq, 1))

    def rollout(ids0, vp, tgt, sd=0):
        torch.manual_seed(sd)
        ids = ids0.clone()
        toks = []
        for step in range(NTOK):
            hooks = []
            if step == 0:
                def inj(m, i, o, p=vp):
                    out = o.clone()
                    out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                    device=out.device)
                    return out
                hooks.append(model.model.norm.register_forward_hook(inj))
            if step >= 1:
                def anti(m, i, o, tid=tgt):
                    out = o.clone()
                    out[0, -1, tid] = -30.0
                    return out
                hooks.append(model.lm_head.register_forward_hook(anti))
            with torch.no_grad():
                L = model(ids).logits[0, -1].float()
            for h in hooks:
                h.remove()
            nxt = sample(L)
            toks.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        return toks

    feats, proxy, y = [], [], []
    print(f"[{MODEL}] REAL-label coherence meta-fit  "
          f"(coherent = rep4==0 and div>=0.7, {NTOK}-tok rollout)")
    print("  %-28s gap entN cos ctx | entA(proxy) div  rep4 maxrun | label"
          % 'prompt')
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        pnat = torch.softmax(Ln.float().cpu(), 0).numpy()
        ent_nat = entnp(pnat)
        cosang = float((vf / vf.norm()) @ Wn[closest(vf)])
        ctxlen = ids0.shape[1]
        feats.append([gap, ent_nat, cosang, ctxlen])

        alpha = 2 * (gap / 97.0) + 0.02
        tgt = closest(vf)
        vp = rot(vf, tgt, alpha)
        # old dynamic proxy: steered forward then anti, next-token entropy
        hooks = [model.model.norm.register_forward_hook(
            lambda m, i, o, p=vp: _inj(o, p))]
        with torch.no_grad():
            Ls = model(ids0).logits[0, -1].float()
        for h in hooks:
            h.remove()
        LsA = Ls.float().cpu().numpy().copy()
        LsA[tgt] = -float('inf')
        ent_anti = entnp(torch.softmax(torch.as_tensor(LsA), 0).numpy())
        # sample next token from grafted dist, free-run one step + anti
        nxt0 = int(Ls.argmax())
        ids1 = torch.cat([ids0, torch.tensor([[nxt0]], device=DEV)], dim=1)
        hooks = [model.lm_head.register_forward_hook(
            lambda m, i, o, tid=tgt: _anti(o, tid))]
        with torch.no_grad():
            L1 = model(ids1).logits[0, -1].float()
        for h in hooks:
            h.remove()
        p1 = torch.softmax(L1.float().cpu(), 0).numpy()
        proxy.append(entnp(p1))   # dynamic post-anti entropy (OLD proxy)

        toks = rollout(ids0, vp, tgt)
        x = sum(1 for t in toks if t in fset)
        div = len(set(toks)) / len(toks)
        rp = rep4(toks)
        mr = max((sum(1 for _ in grp) for _, grp in
                  itertools.groupby(toks)), default=0)
        lab = 1 if (rp == 0.0 and div >= 0.7) else 0
        y.append(lab)
        txt = tok.decode(toks)
        print("  %-28s %4.1f %5.2f %4.2f %3d | %5.2f  %4.2f %4.2f %5d  "
              "%s  %s" % (pname[:26], gap, ent_nat, cosang, ctxlen,
                          ent_anti, div, rp, mr, 'COH' if lab else 'bad',
                          txt[:34]), flush=True)

    X = np.array(feats, dtype=float)
    P = np.array(proxy, dtype=float)
    y = np.array(y, dtype=float)
    n = len(y); ncoh = int(y.sum())
    names = ['gap', 'ent_nat', 'cosang', 'ctxlen']
    print(f"\n  real-coherent: {ncoh}/{n}")
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
        print(f"  {nm:>8}: corr={c:+.3f}  LOO={acc:.2f} @{nm}{side}{th:.3f}")
        if acc > best[1]:
            best = (nm, acc, th, side)
    # the OLD proxy correlation (the inversion test)
    if P.std() > 0 and y.std() > 0:
        c_proxy = np.corrcoef(P, y)[0, 1]
        print(f"  ent_anti(OLD proxy): corr with REAL coherence = "
              f"{c_proxy:+.3f}   <= 0 means the old proxy is INVERTED")
    bname, bacc, bth, bside = best
    naive = max(y.mean(), 1 - y.mean())
    print(f"  inferred real-coherence detector {bname}{bside}{bth:.3f} "
          f"LOO={bacc:.2f} naive={naive:.2f}")

    rows = [dict(prompt=pname, gap=X[i, 0], ent_nat=X[i, 1],
                 cosang=X[i, 2], ctxlen=int(X[i, 3]),
                 ent_anti=round(float(P[i]), 3),
                 coherent=int(y[i])) for i, (pname, _) in enumerate(PROMPTS)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


def _inj(out, p):
    out = out.clone()
    out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype, device=out.device)
    return out


def _anti(out, tid):
    out = out.clone()
    out[0, -1, tid] = -30.0
    return out


if __name__ == "__main__":
    main()