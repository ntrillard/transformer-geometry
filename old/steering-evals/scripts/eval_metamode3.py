#!/usr/bin/env python3
"""eval_metamode3.py — META-LEARN STATIC vs DYNAMIC anti-last recovery.

The big leap hunt. eval_metamode2 (795d45b) inferred gap<=5.58 gates
anti-last recovery -- but from a STATIC next-token proxy (post-anti
distribution at the SAME prompt position). The 8-token rollout in
antidote.csv at gap 8.8 produced the BEST prose ('Paris is the most
beautiful, but that', div 1.00) even though the static proxy marked it
bad (ent_anti 2.72). Hypothesis: RECOVERY IS DYNAMICAL -- after graft
plants the topic token and anti suppresses it, the model free-runs the
NEXT step and its own grammar rebuilds a soft distribution. If dynamic
recovery far exceeds static, the steerable envelope is much wider than
the static 'gap gates everything' ceiling claimed.

Per prompt (3 forwards each, one model, no template):
  F1 natural  : features gap, ent_nat, p_max, f_rank, isfun, ctx, cosang
  F2 steered  : graft = law-budget rot toward closest family member;
                compute STATIC post-anti (zero grafted logit, softmax)
                -> static_label
                sample next token nxt0 from the grafted distribution
  F3 dynamic  : forward context = prompt + [nxt0] with the grafted token
                suppressed at lm_head (-30); measure distribution
                -> dynamic_label (ent>=3 and p_max<0.5)

META-FIT (feature -> static mode, feature -> dynamic mode) via corr +
LOO best-separator per feature. The BIG-LEAP table:
  static_recovery  = # prompts static-good
  dynamic_recovery = # prompts dynamic-good  (the real rollout property)
  envelope_widening = dynamic - static (should be large and NOT gated by
                       the static gap boundary if recovery is dynamic)

Time: ~15 prompts x 3 forwards.
Run: HF_TOKEN=<tok> timeout 10 python3 -u eval_metamode3.py
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
OUT = Path('../steering_geometry_results/metamode3.csv')

PROMPTS = [
    'The capital of France is',
    'The Eiffel Tower is located in',
    'My favorite city in the world is',
    'I visited Paris last summer and it was',
    'If you ask me which European city is the most beautiful, I would say that',
    'The Olympic Games were held in',
    'Once upon a time there was a',
    'For dinner I made',
    'People from all over the world visit',
    'I would love to travel to',
    'The concert was held in',
    'Birds fly south in the winter to',
]

FUNCTION = {'the', 'a', 'an', 'i', 'is', 'was', 'to', 'of', 'in', 'that',
            'it', 'this', 'and', 'but', 'for', 'with', 'on', 'at', 'my',
            'his', 'her', 'there', 'were', 'are', 'be', 'as', 'by', 'from',
            'you', 'we', 'they', 'he', 'she', 'me', 'us', 'them', 'our',
            'their', 'then', 'so', 'will', 'would', 's', 't', 'do', 'does'}


def entnp(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def sample_tok(L, temp=1.0, seed=0):
    p = torch.softmax(L.float(), 0)
    q = p.clone(); order = q.argsort(descending=True)
    k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
    msk = torch.zeros_like(q); msk[order[:k]] = 1
    qq = (q * msk) / (q * msk).sum()
    g = torch.Generator(device=DEV).manual_seed(seed)
    return int(torch.multinomial(qq, 1, generator=g))


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
        """forward with optional norm inject and/or lm_head anti suppress."""
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

    feats, y_static, y_dynamic = [], [], []
    print(f"[{MODEL}] family={TARGET}  static = post-anti at prompt; "
          f"dynamic = post-anti one free-run step later")
    print("  %-46s ctx  gap  f-rk  entN p__N | stA stL | dyA dyL" % 'prompt')

    for sd_seed, pr in enumerate(PROMPTS):
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
        # static post-anti
        LsA = Ls.float().cpu().numpy().copy()
        LsA[tgt] = -float('inf')
        psa = torch.softmax(torch.as_tensor(LsA), 0).numpy()
        ent_st = entnp(psa); p_st = float(psa.max())
        y_static.append(1 if (ent_st >= 3.0 and p_st < 0.5) else 0)
        # dynamic: sample from grafted distribution, then free-run + anti
        nxt0 = sample_tok(Ls, seed=sd_seed)
        ids1 = torch.cat([ids0, torch.tensor([[nxt0]], device=DEV)], dim=1)
        L1 = forward(ids1, anti_id=tgt)
        p1 = torch.softmax(L1.float().cpu(), 0).numpy()
        ent_dy = entnp(p1); p_dy = float(p1.max())
        y_dynamic.append(1 if (ent_dy >= 3.0 and p_dy < 0.5) else 0)
        print("  %-46s %3d %4.1f %5d %5.2f %4.2f | %5.2f %3d | %5.2f %3d"
              % (pr[:44], ctxlen, gap, f_rank, ent_nat, p_max,
                 ent_st, y_static[-1], ent_dy, y_dynamic[-1]), flush=True)

    X = np.array(feats, dtype=float)
    y1 = np.array(y_static, dtype=float)
    y2 = np.array(y_dynamic, dtype=float)
    names = ['gap', 'f_rank', 'ent', 'p_max', 'isfun', 'ctxlen', 'cosang']
    n = len(y1)
    ns, nd = int(y1.sum()), int(y2.sum())

    print("\n-- static mode (proxy) vs dynamic mode (rollout property)")
    print(f"  static  recovery : {ns}/{n}")
    print(f"  dynamic recovery : {nd}/{n}   (envelope widening {nd - ns:+d})")

    def fit(yi, tag):
        print(f"  -- {tag} mode --")
        best = (None, -1, None, None)
        for j, nm in enumerate(names):
            c = (np.corrcoef(X[:, j], yi)[0, 1]
                 if X[:, j].std() > 0 else float('nan'))
            accs = []
            for th in np.unique(X[:, j]):
                sc = sum(1 for i in range(n) if (X[i, j] <= th) == yi[i])
                accs.append(sc / n)
            acc = max(accs); th = np.unique(X[:, j])[int(np.argmax(accs))]
            left = np.mean(yi[X[:, j] <= th]) if (X[:, j] <= th).any() else 0
            side = '<=' if left >= np.mean(yi[X[:, j] > th] if
                                           (X[:, j] > th).any() else yi) else '>'
            print(f"    {nm:>7}: corr={c:+.3f}  LOO={acc:.2f} "
                  f"@{nm}{side}{th:.3f}")
            if acc > best[1]:
                best = (nm, acc, th, side)
        bname, bacc, bth, bside = best
        naive = max(yi.mean(), 1 - yi.mean())
        print(f"    inferred detector {bname}{bside}{bth:.3f} LOO={bacc:.2f} "
              f"naive={naive:.2f}")
        return bname, bacc, bth, bside

    fit(y1, 'static')
    fit(y2, 'dynamic')

    # BIG LEAP: does dynamic recovery extend beyond the static gap boundary?
    s_gapb = 5.578  # static boundary from metamode2
    high_gap = [i for i in range(n) if X[i, 0] > s_gapb]
    dyn_high = sum(y2[i] for i in high_gap)
    print(f"\n-- BIG LEAP: dynamic recovery ABOVE the static gap boundary "
          f"(gap>{s_gapb})")
    print(f"  prompts above boundary: {len(high_gap)}, "
          f"of which dynamic-recoverable: {dyn_high}/{len(high_gap)}")
    print(f"  => if {dyn_high}/{len(high_gap)} recover dynamically, the "
          f"'gap gates recovery' ceiling is broken (recovery is dynamic)")

    rows = [dict(prompt=p, **{nm: X[i, j] for j, nm in enumerate(names)},
                 static_label=int(y1[i]), dynamic_label=int(y2[i]))
            for i, p in enumerate(PROMPTS)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()