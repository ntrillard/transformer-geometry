#!/usr/bin/env python3
"""eval_why.py — test the MECHANISM of anti-last coherence.

WHY does anti-last (graft + suppress planted token) produce coherent
free-run on some prompts and collapse on others? HYPOTHESIS: after the
graft plants topic token T, suppressing T at step 1 forces the model to
sample from its BACKUP distribution. If that backup is GRAMMATICAL
(function words/be-verbs -> grammar machinery active -> the model can
free-run prose), the run is coherent. If the backup is junk (rare
tokens, <eos>, sparse), it collapses.

Testable predictor (ONE extra forward, no generation loop): after
steered+anti, read step-1 distribution L1: top1 token, is it a function
word, p_top1, entropy L1. CORRELATE with real coherence label
(fixed metric, >=2/3 seeds).

If a step-1 L1 property separates coherent from collapse -> that is the
mechanism AND a contextual predictor (a 'why it works' that transfers).

One model, no template. Run: HF_TOKEN=<tok> timeout 18 python3 -u
eval_why.py
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
NTOK = 4
SEEDS = [0, 1]
TARGET = 'city'
OUT = Path('../steering_geometry_results/why.csv')
ANGLE = 10.0

# mix of known coherent (1) and collapse (0) and new prompts
PROMPTS = [
    ('ask',   'If you ask me which European city is the most beautiful, I would say that'),
    ('japan', 'The capital of Japan is'),
    ('spain', 'The capital of Spain is'),
    ('tower', 'The tallest tower in the world is in'),
    ('austr', 'The biggest city in Australia is'),
    ('visitnl','I love to visit new places, and my favorite city is'),
    ('olymp', 'The Olympic Games were held in'),
    ('france','The capital of France is'),
    ('paris', 'I visited Paris last summer and it was'),
    ('itcap', 'The capital of Italy is'),
    ('mona',  'The Mona Lisa is displayed in'),
    ('novel', 'The story takes place in'),
    ('hockey','The hockey championship was held in'),
    ('decap', 'The capital of Germany is'),
]

FUNCTION = {'the', 'a', 'an', 'i', 'is', 'was', 'to', 'of', 'in', 'that',
            'it', 'this', 'and', 'but', 'for', 'with', 'on', 'at', 'my',
            'his', 'her', 'there', 'were', 'are', 'be', 'as', 'by', 'from',
            'you', 'we', 'they', 'he', 'she', 'me', 'us', 'them', 'our',
            'their', 'then', 'so', 'will', 'would', 's', 't', 'do', 'does',
            'is', 'the', 'a', 'an', 'of', 'in', 'that', 'to'}


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

    def shot_anti(ids0, vp, tgt, sd, window=2):
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
                if 1 <= step <= window:
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
    print(f"[{MODEL}] WHY anti-last works: step-1 backup-continuation "
          f"predictor vs real coherence ({len(PROMPTS)} prompts)")
    print("  %-8s %5s %5s %5s %6s %5s | %s | %s"
          % ('prompt', 'gap', 'p1', 'ent1', 'isfn', 'tok1', 'pred',
             'coh_seeds'))
    feats, y = [], []
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        tgt = closest(vf)
        vp = rot_to_angle(vf, tgt, ANGLE)
        # STEP-1 backup distribution: graft at step0, sample T, append,
        # forward with anti -> L1
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
        p1 = torch.softmax(L1.float(), 0)
        top1 = int(p1.argmax())
        ent1 = entnp(p1.cpu().numpy())
        p1v = float(p1[top1])
        isfn = 1 if tok.decode([top1]).strip().lower() in FUNCTION else 0
        tok1_s = tok.decode([top1]).strip()
        # predictor: backup is grammatical if top1 is a function word
        pred = isfn
        # real label
        oks = []
        seed_str = ''
        for sd in SEEDS:
            toks = shot_anti(ids0, vp, tgt, sd)
            c = coherent(toks, tok.decode(toks))
            seed_str += 'Y' if c else 'n'
            oks.append(c)
        lab = 1 if sum(oks) >= 2 else 0
        feats.append([gap, p1v, ent1, isfn])
        y.append(lab)
        rows.append(dict(prompt=pname, gap=round(gap, 2), step1_top=tok1_s,
                         p_top1=round(p1v, 3), ent1=round(ent1, 2),
                         is_function=int(isfn), predictor=int(pred),
                         label=int(lab), seeds=seed_str))
        print("  %-8s %5.1f %5.2f %5.2f %6d %6s | %3d | %s"
              % (pname[:8], gap, p1v, ent1, isfn, tok1_s[:6], pred,
                 seed_str), flush=True)

    X = np.array(feats, dtype=float)
    y = np.array(y, dtype=float)
    n = len(y); nc = int(y.sum())
    names = ['gap', 'p_top1', 'ent1', 'isfn']
    print(f"\n  real coherent: {nc}/{n}")
    for j, nm in enumerate(names):
        c = (np.corrcoef(X[:, j], y)[0, 1] if X[:, j].std() > 0 and
             y.std() > 0 else float('nan'))
        accs = []
        for th in np.unique(X[:, j]):
            accs.append(sum(1 for i in range(n)
                            if (X[i, j] <= th) == y[i]) / n)
        acc = max(accs) if accs else 0.0
        print(f"  {nm:>6}: corr={c:+.3f}  LOO={acc:.2f}")
    # the predictor (isfn==1 -> coherent)
    acc_pred = float((X[:, 3] == y).mean())
    print(f"  PREDICTOR (step1-top is function word -> coherent): "
          f"acc={acc_pred:.2f} (naive={max(y.mean(), 1-y.mean()):.2f})")
    # direct table
    for i in range(n):
        if int(X[i, 3]) == int(y[i]):
            print(f"    CORRECT: {PROMPTS[i][0]} (isfn={int(X[i,3])} "
                  f"label={int(y[i])})")
        else:
            print(f"    WRONG  : {PROMPTS[i][0]} (isfn={int(X[i,3])} "
                  f"label={int(y[i])})")

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