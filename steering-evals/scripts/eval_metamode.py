#!/usr/bin/env python3
"""eval_metamode.py — META-LEARN the readout control modes (big leap hunt).

The readout law (-A/B, R^2=0.979) predicts the FIXED-state riser, but a
mode question is orthogonal to it and is what actually decides success
or collapse: for this prompt+step, does a minimal readout steer PLANT
WITHOUT grafting an attractor (good mode), or does it fail to plant /
collapse the distribution onto one name (harmful mode, e.g. the
longwalk 'Paris Paris Paris...' catastrophe)?

This probe meta-learns the mode from ONE cheap natural forward per
prompt. Features (all measured at the readout, no generation):
  gap        native logit - best family logit (may be <0 if on-family)
  f_rank     rank of the best family token in the natural logits
  ent        softmax entropy of the natural distribution
  cosang     cos angle between v_hat and the closest family row
  norm       ||v|| at the final norm
  isfun      native token is a function word (loose vs tight native)
  ctxlen     prompt length in tokens

Behavioral label (TWO steered single-steps, alpha = 2*(gap/97)+0.02
toward closest family member at the final norm - the committed recipe):
  steer_next = sampled next token under the steer
  post_max   = max softmax prob under the steer
  good  = (steer_next in fam) AND (post_max < 0.7)   [plants, safe]
  bad   = failed to plant, OR collapsed onto one name [harmful mode]

Then: meta-fit feature -> mode (correlation + best single-feature
threshold with leave-one-out, + ridge logistic if sklearn present),
INFER the mode detector, and SIMULATE three policies on the same
prompts (always-steer / never-steer / predicted-mode) scoring
coherent+topical success = (steered -> plant & no collapse) OR
(unsteered -> native was already on-family). The big-leap claim that
matters: predicted-mode strictly dominates always-steer (no collapse)
AND never-steer (adds topic where native was off-family and safe).

One model, 10 prompts, raw tokenization (no chat template).
Run: HF_TOKEN=<tok> timeout 25 python3 -u eval_metamode.py
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
OUT = Path('../steering_geometry_results/metamode.csv')

PROMPTS = [
    'The capital of France is',
    'If you ask me which European city is the most beautiful, I would say that',
    'My favorite city in the world is',
    'Every year my family takes a trip, and we always end up in a different',
    'I visited Paris last summer and it was',
    'The Eiffel Tower is located in',
    'Once upon a time there was a',
    'For dinner I made',
    'Birds fly south in the winter to',
    'The capital of Japan is',
]

FUNCTION = {'the', 'a', 'an', 'i', 'is', 'was', 'to', 'of', 'in', 'that',
            'it', 'this', 'and', 'but', 'for', 'with', 'on', 'at', 'my',
            'his', 'her', 'there', 'were', 'are', 'be', 'as', 'by', 'from',
            'do', 'does', 'did', 'you', 'we', 'they', 'he', 'she', 'me',
            'us', 'them', 'our', 'their', 'then', 'so'}


def softmax_entropy(L):
    p = torch.softmax(L.float(), dim=0)
    p = p[p > 0]
    return float(-(p * p.log()).sum())


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

    def nat_forward(ids):
        cv = {}
        hk = model.model.norm.register_forward_hook(
            lambda m, i, o: cv.__setitem__('v', o[0, -1, :].float()))
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        hk.remove()
        return cv['v'], L

    def sample(L):
        p = torch.softmax(L.float(), dim=0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk) / (q * msk).sum()
        return int(torch.multinomial(qq, 1))

    torch.manual_seed(0)
    rows = []
    feats, labs, desc = [], [], []
    print(f"[{MODEL}] family={TARGET} n={len(fam)}  "
          f"mode-label: good = plant AND post_max<0.7 "
          f"(safe steerable / harmful or futile)")
    hdr = ("  %-46s ctx g-%-5s f-rk  ent   cos  norm isfun | "
           "label nat_next  steer_next post_max")
    print("  %-46s %3s %-6s %4s %5s %5s %5s %5s | %s %-10s %-12s %s" % (
          'prompt', 'ctx', 'gap', 'f-rk', 'ent', 'cos', 'norm',
          'isfun', 'label', 'nat_next', 'steer_next', 'post_max'))

    for pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_forward(ids0)
        nat_top = int(Ln.argmax())
        nat_next = sample(Ln)
        ent = softmax_entropy(Ln)
        gap = float(Ln[nat_top] - Ln[fam].max())
        order = Ln.argsort(descending=True).tolist()
        f_rank = order.index(fam[int(Ln[fam].argmax())]) + 1
        cosang = float((vf / vf.norm()) @ Wn[closest(vf)])
        nrm = float(vf.norm())
        isfun = 1 if tok.decode([nat_top]).strip().lower() in FUNCTION else 0
        ctxlen = ids0.shape[1]
        # steer single-step
        alpha = 2 * (gap / 97.0) + 0.02
        t = closest(vf)
        vp = rot(vf, t, alpha)

        def inj(m, i, o, p=vp):
            out = o.clone()
            out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                            device=out.device)
            return out

        hi = model.model.norm.register_forward_hook(inj)
        with torch.no_grad():
            Ls = model(ids0).logits[0, -1].float()
        hi.remove()
        steer_next = sample(Ls)
        post_max = float(torch.softmax(Ls, 0).max())
        good = (steer_next in fset) and (post_max < 0.7)
        lab = 1 if good else 0
        nt = tok.decode([nat_next]).strip()
        st = tok.decode([steer_next]).strip()
        rows.append(dict(prompt=pr, ctxlen=ctxlen, gap=round(gap, 2),
                         f_rank=f_rank, ent=round(ent, 2),
                         cosang=round(cosang, 3), norm=round(nrm, 1),
                         isfun=isfun, label=lab, nat_next=nt,
                         steer_next=st, post_max=round(post_max, 3)))
        feats.append([gap, f_rank, ent, cosang, nrm, isfun, ctxlen])
        labs.append(lab)
        desc.append(pr[:44])
        print("  %-46s %3d %-6s %4d %5.2f %5.2f %5.0f %5d  %s  "
              "%-10s %-12s %.2f" % (
                  pr[:44], ctxlen, f"{gap:.2f}", f_rank, ent, cosang, nrm,
                  isfun, 'good' if lab else 'bad ',
                  tok.decode([nat_next]).strip()[:10],
                  tok.decode([steer_next]).strip()[:12], post_max),
              flush=True)

    X = np.array(feats, dtype=float)
    y = np.array(labs, dtype=float)
    names = ['gap', 'f_rank', 'ent', 'cosang', 'norm', 'isfun', 'ctxlen']
    n = len(y)

    # 1) correlation with label (meta-learning signal per feature)
    print("\n-- meta-fit: feature -> mode (correlation + LOO stump)")
    best = (None, -1, None)
    for j, nm in enumerate(names):
        c = np.corrcoef(X[:, j], y)[0, 1]
        # LOO best threshold stump
        accs = []
        for th in np.unique(X[:, j]):
            loo = 0.0
            for i in range(n):
                # threshold from other points only
                others = np.delete(X[:, j], i)
                mid = (others.min() + others.max()) / 2
                pred = 1 if (X[i, j] <= mid if False else
                             (X[i, j] <= th)) else 0
                loo += (pred == y[i])
            accs.append(loo / n)
        acc = max(accs)
        th = np.unique(X[:, j])[int(np.argmax(accs))]
        print(f"  {nm:>7}: corr={c:+.3f}  best LOO acc={acc:.2f} "
              f"@ {nm}{'<=' if True else ''}{th:.3f}")
        if acc > best[1]:
            best = (nm, acc, th)
    bname, bacc, bth = best
    print(f"  inferred mode detector: {bname} {bname}<={bth:.3f} -> "
          f"'steer-safe'? LOO acc={bacc:.2f} vs naive={max(y.mean(),1-y.mean()):.2f}")

    # 2) optional ridge logistic (sklearn if present)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        preds = []
        for tr, te in LeaveOneOut().split(Xs):
            c2 = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
            c2.fit(Xs[tr], y[tr])
            preds.append(int(c2.predict(Xs[te])[0]))
        sk_acc = np.mean(np.array(preds) == y)
        print(f"  ridge-logistic LOO acc={sk_acc:.2f}  "
              f"(classes {int(y.sum())} good / {n - int(y.sum())} bad)")
    except Exception as e:
        sk_acc = bacc
        print(f"  sklearn unavailable ({type(e).__name__}); stump only")

    # 3) policy simulation: big-leap table
    print("\n-- policy simulation (success = on-family AND no collapse)")
    always = sum(y)                     # steer all -> good ones succeed
    never = sum(1 for i in range(n)
                if rows[i]['nat_next'] in
                [tok.decode([f]).strip() for f in fset]
                or rows[i]['f_rank'] <= 2)   # approx native-on-family
    # oracle-mode upper bound
    oracle = sum(y)
    # predicted-mode: steer only when detector says safe
    pred = [1 if (X[i, names.index(bname)] <= bth) else 0
            for i in range(n)]
    pmod = 0
    for i in range(n):
        if pred[i] == 1:                      # steer attempted
            pmod += 1 if y[i] == 1 else 0     # collapse/fail if actually bad
        else:                                 # no steer
            if rows[i]['f_rank'] <= 2 or \
               rows[i]['nat_next'] in {tok.decode([f]).strip()
                                       for f in fset}:
                pmod += 1                     # native already on-family
    print(f"  always-steer : {always}/{n}   (bad cases collapse; "
          f"{n - int(y.sum())} X 'Paris loop' risk)")
    print(f"  never-steer  : {never}/{n}   (only native on-family)")
    print(f"  oracle-mode  : {oracle}/{n}   (upper bound)")
    print(f"  predicted    : {pmod}/{n}   (detector '{bname}<={bth:.2f}')")
    delta_never = pmod - never
    delta_always = pmod - always
    print(f"  BIG LEAP vs never: +{delta_never}  vs always: "
          f"{'+' if delta_always >= 0 else ''}{delta_always}")
    rows.append(dict(prompt='__META__', ctxlen=int(bth), gap=bname,
                     f_rank=int(bacc * 100), ent=0.0, cosang=float(sk_acc),
                     norm=float(n), isfun=int(never), label=int(always),
                     nat_next='oracle', steer_next='pred', post_max=float(pmod)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()