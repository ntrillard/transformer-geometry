#!/usr/bin/env python3
"""eval_metamode2.py — META-LEARN THE ANTI-LAST RECOVERY MODE (big-leap probe).

eval_metamode (13fcc3e) was degenerate: on base Gemma EVERY single-step
graft collapsed the distribution (constant label -> corr=nan, nothing to
learn). The big-leap follow-up is to meta-learn the mode that DOES vary:
**does the anti-last antidote (graft then suppress the planted token)
recover a SOFT free-running distribution (recoverable mode -> the
'Paris is the most beautiful, but that' prose) or stay a DEGENERATE
collapse (harmful mode)?**

Cheap per-prompt measurement (2 forwards each, NO generation loop):
  F1 natural forward  : features gap, f_rank (family rank), ent, p_max,
                        isfun (native is function word), ctxlen, cosang.
  F2 steered forward  : graft = law-budget rotation toward closest family
                        member; then compute post-anti distribution IN
                        NUMPY (zero the grafted target logit, softmax).
  recovery label      : good if ent_anti >= 3.0 AND p_max_anti < 0.5
                        (anti unlocks a broad soft free-run; the Qwen
                        'grammar survives' cell and the antidote prose).

Then META-FIT feature -> recovery mode (corr + LOO best-separator stump +
ridge logistic if sklearn), INFER the single best mode detector, and
SIMULATE policies:
  always-steer : every prompt graft+anti; success iff recovery-good
  never-steer  : success iff native already on-family (no steer needed)
  predicted    : steer only when detector predicts recovery-good
Big-leap table: predicted strictly dominates never (adds topic where
safe) and beats always (avoids collapse) IF a feature separates cleanly.

One model, no chat template, next-token only -> ~25x2=50 forwards,
well under 10s.
Run: HF_TOKEN=<tok> timeout 10 python3 -u eval_metamode2.py
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
OUT = Path('../steering_geometry_results/metamode2.csv')

PROMPTS = [
    'The capital of France is',
    'The capital of Japan is',
    'The Eiffel Tower is located in',
    'The Statue of Liberty is in',
    'The tallest building in the world is in',
    'My favorite city in the world is',
    'I visited Paris last summer and it was',
    'If you ask me which European city is the most beautiful, I would say that',
    'The most populous city in the United States is',
    'The Olympic Games were held in',
    'Once upon a time there was a',
    'For dinner I made',
    'A good place to live is',
    'The best place to go on vacation is',
    'People from all over the world visit',
    'I would love to travel to',
    'The airport in this country is in',
    'The concert was held in',
    'The conference will take place in',
    "The team's biggest match is in",
    'Birds fly south in the winter to',
    'The weather is nice today in',
    'I woke up this morning and decided to',
    'She told me that the meeting was in',
    'It happened near the big city of',
]

FUNCTION = {'the', 'a', 'an', 'i', 'is', 'was', 'to', 'of', 'in', 'that',
            'it', 'this', 'and', 'but', 'for', 'with', 'on', 'at', 'my',
            'his', 'her', 'there', 'were', 'are', 'be', 'as', 'by', 'from',
            'you', 'we', 'they', 'he', 'she', 'me', 'us', 'them', 'our',
            'their', 'then', 'so', 'will', 'would', 's', 't', 'do', 'does'}


def ent(p):
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

    feats, labs = [], []
    print(f"[{MODEL}] family={TARGET} n={len(fam)}  "
          f"recovery-good = ent_anti>=3.0 and p_max_anti<0.5")
    hdr = "  %-46s ctx g-%-5s f-rk  ent  p__x isfun cos  | rcy entA p__xA"
    print("  %-46s %3s %-6s %4s %5s %5s %5s %5s | %s %5s %5s" % (
          'prompt', 'ctx', 'gap', 'f-rk', 'ent', 'p_max', 'isfun', 'cos',
          'rcy', 'entA', 'p_maxA'))

    for pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        order = Ln.argsort(descending=True).tolist()
        f_rank = order.index(fam[int(Ln[fam].argmax())]) + 1
        pnat = torch.softmax(Ln.float().cpu(), 0).numpy()
        ent_nat = ent(pnat)
        p_max = float(pnat.max())
        gap = float(Ln[nat_top] - Ln[fam].max())
        cosang = float((vf / vf.norm()) @ Wn[closest(vf)])
        isfun = 1 if tok.decode([nat_top]).strip().lower() in FUNCTION else 0
        ctxlen = ids0.shape[1]

        # steered forward (graft)
        alpha = 2 * (gap / 97.0) + 0.02
        tgt = closest(vf)
        vp = rot(vf, tgt, alpha)

        def inj(m, i, o, p=vp):
            out = o.clone()
            out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                            device=out.device)
            return out

        hi = model.model.norm.register_forward_hook(inj)
        with torch.no_grad():
            Ls = model(ids0).logits[0, -1].float()
        hi.remove()

        # post-anti distribution in numpy
        Ls_anti = Ls.float().cpu().numpy().copy()
        Ls_anti[tgt] = -float('inf')
        pa = torch.softmax(torch.as_tensor(Ls_anti), 0).numpy()
        ent_anti = ent(pa)
        p_max_anti = float(pa.max())
        lab = 1 if (ent_anti >= 3.0 and p_max_anti < 0.5) else 0
        feats.append([gap, f_rank, ent_nat, p_max, isfun, ctxlen, cosang])
        labs.append(lab)
        print("  %-46s %3d %-6s %4d %5.2f %5.2f %5d %5.2f |  %s  "
              "%5.2f %5.2f" % (
                  pr[:44], ctxlen, f"{gap:.1f}", f_rank, ent_nat, p_max,
                  isfun, cosang, 'good' if lab else 'bad ',
                  ent_anti, p_max_anti), flush=True)

    X = np.array(feats, dtype=float)
    y = np.array(labs, dtype=float)
    names = ['gap', 'f_rank', 'ent', 'p_max', 'isfun', 'ctxlen', 'cosang']
    n = len(y)
    ng = int(y.sum())

    print("\n-- meta-fit: feature -> recovery mode (corr + LOO separator)")
    best = (None, -1, None)
    for j, nm in enumerate(names):
        c = np.corrcoef(X[:, j], y)[0, 1] if X[:, j].std() > 0 else float('nan')
        accs = []
        for th in np.unique(X[:, j]):
            sc = 0.0
            for i in range(n):
                pred = 1 if X[i, j] <= th else 0
                sc += (pred == y[i])
            accs.append(sc / n)
        acc = max(accs)
        th = np.unique(X[:, j])[int(np.argmax(accs))]
        side = '<=' if np.mean(y[X[:, j] <= th]) > np.mean(y[X[:, j] > th]) \
            else '>'
        print(f"  {nm:>7}: corr={c:+.3f}  best LOO acc={acc:.2f} "
              f"@ {nm}{side}{th:.3f}")
        if acc > best[1]:
            best = (nm, acc, th, side)
    bname, bacc, bth, bside = best
    naive = max(ng / n, 1 - ng / n)
    print(f"  inferred mode detector: {bname}{bside}{bth:.3f} -> "
          f"'recoverable' | LOO acc={bacc:.2f} vs naive={naive:.2f} "
          f"(classes {ng} good / {n - ng} bad)")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        preds = []
        for tr, tek in LeaveOneOut().split(Xs):
            c2 = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs')
            c2.fit(Xs[tr], y[tr])
            preds.append(int(c2.predict(Xs[tek])[0]))
        sk_acc = np.mean(np.array(preds) == y)
        print(f"  ridge-logistic LOO acc={sk_acc:.2f}")
    except Exception:
        sk_acc = bacc

    # policy simulation
    nat_onfam = [1 if rows['f_rank'] <= 2 else 0 for rows in
                 [dict(zip(names, r)) for r in X]] if False else []
    # native on-family: f_rank<=2 (best family token near top)
    never = sum(1 for i in range(n) if X[i, 1] <= 2)
    always = ng
    pred = [1 if (X[i, names.index(bname)] <= bth if bside == '<=' else
                  X[i, names.index(bname)] > bth) else 0 for i in range(n)]
    pmod = sum(1 for i in range(n)
               if (pred[i] == 1 and y[i] == 1) or
               (pred[i] == 0 and X[i, 1] <= 2))
    print("\n-- policy simulation (success = topical AND coherent)")
    print(f"  always-steer(graft+anti): {always}/{n}")
    print(f"  never-steer (native on-family, f_rank<=2): {never}/{n}")
    print(f"  predicted ('{bname}{bside}{bth:.2f}'):        {pmod}/{n}")
    print(f"  BIG LEAP  predicted vs never: {'+' if pmod >= never else ''}"
          f"{pmod - never}   vs always: "
          f"{'+' if pmod >= always else ''}{pmod - always}")

    # persist
    rows = [dict(prompt=p, **{nm: X[i, j] for j, nm in enumerate(names)},
                 label=int(labs[i]), ent_anti=None, p_max_anti=None)
            for i, p in enumerate(PROMPTS)]
    rows.append(dict(prompt='__META__', gap=0, f_rank=int(bacc * 100),
                     ent=float(sk_acc), p_max=float(naive), isfun=int(never),
                     ctxlen=int(always), cosang=float(pmod),
                     label=int(ng), ent_anti=None, p_max_anti=None))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()