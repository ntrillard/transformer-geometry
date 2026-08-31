#!/usr/bin/env python3
"""eval_metamode6.py — META-LEARN THE SEED-ROBUST COHERENCE MODE (big leap).

metamode5 showed cheap features barely separate single-seed 'coherence'
and the entropy proxy is inverted (corr -0.14 vs real). The BIG-LEAP
frame: a steering mode is a REAL property only if it is coherent across
INDEPENDENT seeds (3 seeds here) -> label = coherent on >=2/3 rollouts,
with a STRICT criterion that kills the lenient false-positives that
polluted metamode5 ('Tokyo Japa music is music', 'Paris<eos>Paris').

strict coherent := plant AND div>=0.7 AND rep4==0 AND maxrun<=2 AND
no '<eos>' AND no single token appears >2 times.

Features (cheap, natural forward): gap, cosang (family alignment),
ent_nat, ctxlen. Also old proxy ent_anti for the inversion check.
Meta-fit feature -> seed-robust-coherence (corr + LOO stump). The big-
leap question: is there a cheap feature (candidate cosang) that CLEANLY
separates seed-robust-coherent from the rest? If yes it is the controllable
CHEAP TRIGGER for coherent anti-last steering.

Budget: 5 prompts x (3 seeds x 6-tok rollout + 2 forwards) ~ under 10s.
One model, no template.
Run: HF_TOKEN=<tok> timeout 10 python3 -u eval_metamode6.py
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
SEEDS = [0, 1, 2]
TARGET = 'city'
OUT = Path('../steering_geometry_results/metamode6.csv')

PROMPTS = [
    ('ask',  'If you ask me which European city is the most beautiful, I would say that'),  # robust coherent (realroll)
    ('fr',   'The capital of France is'),                                                    # collapse
    ('jpn',  'The capital of Japan is'),                                                     # mixed
    ('trav', 'I would love to travel to'),                                                   # eos
    ('best', 'My favorite city in the world is'),                                            # loop
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

    def rollout(ids0, vp, tgt, sd):
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

    def strict_ok(toks, txt):
        if not any(t in toks for t in fset):      # plant
            return False
        if '<eos>' in txt:
            return False
        if len(set(toks)) / len(toks) < 0.7:      # div
            return False
        if rep4(toks) != 0.0:                     # no repeated 4-gram
            return False
        mr = max((sum(1 for _ in grp) for _, grp in
                  itertools.groupby(toks)), default=0)
        if mr > 2:                                # no long run
            return False
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        if any(c > 2 for c in counts.values()):   # no token >2x
            return False
        return True

    feats, ent_anti_all, ncoh, rows = [], [], [], []
    print(f"[{MODEL}] seed-robust coherence (3 seeds, strict criterion)")
    print("  %-26s gap cos  entN ctx | entA | 1/3 seeds COH | robust(y)"
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
        feats.append([gap, cosang, ent_nat, ctxlen])

        alpha = 2 * (gap / 97.0) + 0.02
        tgt = closest(vf)
        vp = rot(vf, tgt, alpha)
        # old proxy ent_anti
        hooks = [model.model.norm.register_forward_hook(
            lambda m, i, o, p=vp: (_inj(o, p)))]
        with torch.no_grad():
            Ls = model(ids0).logits[0, -1].float()
        for h in hooks:
            h.remove()
        LsA = Ls.float().cpu().numpy().copy()
        LsA[tgt] = -float('inf')
        ent_a = entnp(torch.softmax(torch.as_tensor(LsA), 0).numpy())
        ent_anti_all.append(ent_a)

        ok = []
        str_seeds = ''
        for sd in SEEDS:
            toks = rollout(ids0, vp, tgt, sd)
            txt = tok.decode(toks)
            o = strict_ok(toks, txt)
            ok.append(o)
            str_seeds += 'Y' if o else 'n'
        robust = 1 if sum(ok) >= 2 else 0
        ncoh.append(robust)
        rows.append(dict(prompt=pname, gap=round(gap, 3),
                         cosang=round(cosang, 3), ent_nat=round(ent_nat, 3),
                         ctxlen=ctxlen, ent_anti=round(ent_a, 3),
                         seeds=str_seeds, n_coherent=sum(ok),
                         robust=robust))
        print("  %-26s %4.1f %4.2f %5.2f %3d | %4.2f |  %s   %d"
              % (pname[:24], gap, cosang, ent_nat, ctxlen, ent_a,
                 str_seeds, robust), flush=True)

    X = np.array(feats, dtype=float)
    ya = np.array(ent_anti_all, dtype=float)
    y = np.array(ncoh, dtype=float)
    names = ['gap', 'cosang', 'ent_nat', 'ctxlen']
    n = len(y); nr = int(y.sum())
    print(f"\n  seed-robust coherent: {nr}/{n}")
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
    if ya.std() > 0 and y.std() > 0:
        print(f"  ent_anti(OLD proxy) corr with robust coherence = "
              f"{np.corrcoef(ya, y)[0, 1]:+.3f}  (inversion check)")
    bname, bacc, bth, bside = best
    naive = max(y.mean(), 1 - y.mean())
    print(f"  inferred robust-mode detector {bname}{bside}{bth:.3f} "
          f"LOO={bacc:.2f} naive={naive:.2f}")

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


if __name__ == "__main__":
    main()