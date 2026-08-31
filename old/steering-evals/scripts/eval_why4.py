#!/usr/bin/env python3
"""eval_why4.py — TEST the copular-island mechanism + successor fix.

why1-3 eliminated: pivot is grammar both ways (why1), planted topic never
re-latches (why2), family never fills (why3, pFam=0). The remaining
failure mode: the PIVOT re-samples itself - with the topic blocked, the
model's top mass at every step is the same copula, and without a content
word it sticks in a loop ('is is is').

TWO tests:
  A) PREDICTOR: under window-2 single-anti, is the step-2 argmax the
     SAME token as the pivot ('is')? (self-repeat = copular island =
     collapse). Measured with ONE forward, no rollout.
  B) FIX: at the anti steps, also BOOST the model's own top non-family
     content token (+7.0 logits) - give the copula a successor. If the
     island is the cause, grammar coherence should jump.

3 seeds / arm. One model, no template.
Run: HF_TOKEN=<tok> timeout 25 python3 -u eval_why4.py
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
SEEDS = [0, 1, 2]
TARGET = 'city'
OUT = Path('../steering_geometry_results/why4.csv')
ANGLE = 10.0
BOOST = 7.0

PROMPTS = [
    ('ask',    'If you ask me which European city is the most beautiful, I would say that'),
    ('spain',  'The capital of Spain is'),
    ('japan',  'The capital of Japan is'),
    ('mona',   'The Mona Lisa is displayed in'),
    ('novel',  'The story takes place in'),
    ('hockey', 'The hockey championship was held in'),
    ('austr',  'The biggest city in Australia is'),
    ('itcap',  'The capital of Italy is'),
    ('decap',  'The capital of Germany is'),
    ('france', 'The capital of France is'),
    ('olymp',  'The Olympic Games were held in'),
    ('paris',  'I visited Paris last summer and it was'),
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

    def clean_toks(toks, txt):
        if '<eos>' in txt:
            return False
        if rep4(toks) != 0.0:
            return False
        mr = max((sum(1 for _ in grp) for _, grp in
                  itertools.groupby(toks)), default=0)
        if mr > 1:
            return False
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        if any(c > 1 for c in counts.values()):
            return False
        return True

    def topic_ok(toks, txt):
        if not any(t in toks for t in fset):
            return False
        return clean_toks(toks, txt)

    def shot_anti(ids0, vp, tgt, sd, boost_c):
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
                if 1 <= step <= 2:
                    if boost_c is None:
                        def anti(m, i, o, tid=tgt):
                            out = o.clone()
                            out[0, -1, tid] = -30.0
                            return out
                        hs.append(model.lm_head.register_forward_hook(anti))
                    else:
                        def antiB(m, i, o, tid=tgt, cid=boost_c):
                            out = o.clone()
                            out[0, -1, tid] = -30.0
                            out[0, -1, cid] += BOOST
                            return out
                        hs.append(model.lm_head.register_forward_hook(antiB))
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
    print(f"[{MODEL}] WHY4: copular-island (self-repeat) predictor + "
          f"successor-boost fix ({len(PROMPTS)} prompts, "
          f"{len(SEEDS)} seeds)")
    print("  %-8s %5s %6s %4s | %s | %s | %s | %s"
          % ('prompt', 'gap', 'pSelf', 'pr', 'pivot', 'base',
             'boost', 'boost_tpc'))
    y_base, y_boost, feat = [], [], []
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        tgt = closest(vf)
        vp = rot_to_angle(vf, tgt, ANGLE)

        # pivot + successor content token (single forward with graft+anti)
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

        # content successor: top non-family, != pivot, prefer non-function
        p1 = torch.softmax(L1.float(), 0)
        p1[fam] = 0.0
        order = p1.argsort(descending=True)
        boost_c = None
        for c in order.tolist():
            if c == pivot:
                continue
            if tok.decode([c]).strip().lower() in FUNCTION:
                continue
            boost_c = c
            break
        if boost_c is None:
            for c in order.tolist():
                if c != pivot:
                    boost_c = c
                    break

        # SELF-REPEAT PREDICTOR: step-2 argmax under anti == pivot?
        ids2 = torch.cat([ids1, torch.tensor([[pivot]], device=DEV)], dim=1)
        hs = [model.lm_head.register_forward_hook(
            lambda m, i, o, tid=tgt: anti2(o, tid))]
        try:
            with torch.no_grad():
                L2 = model(ids2).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        pSelf = float(torch.softmax(L2, 0)[pivot])
        selfrep = int(L2.argmax() == pivot)

        def run(boost_c):
            st = ''; gg = ''
            for sd in SEEDS:
                toks = shot_anti(ids0, vp, tgt, sd, boost_c)
                txt = tok.decode(toks)
                st += 'Y' if topic_ok(toks, txt) else 'n'
                gg += 'Y' if clean_toks(toks, txt) else 'n'
            return st, gg

        sb, gb = run(None)
        sbo, gbo = run(boost_c)
        y_base.append(1 if gb.count('Y') >= 2 else 0)
        y_boost.append(1 if gbo.count('Y') >= 2 else 0)
        feat.append(selfrep)
        rows.append(dict(prompt=pname, gap=round(gap, 2),
                         p_selfrep=round(pSelf, 3), selfrep=int(selfrep),
                         pivot=tok.decode([pivot]).strip(),
                         successor=tok.decode([boost_c]).strip(),
                         base_topic=sb, base_grammar=gb,
                         boost_topic=sbo, boost_grammar=gbo))
        print("  %-8s %5.1f %6.2f %4d | %-6s | %s | %s | %s"
              % (pname[:8], gap, pSelf, selfrep,
                 tok.decode([pivot]).strip(),
                 sb + '/' + gb, sbo, gbo), flush=True)

    n = len(y_base)
    yb = np.array(y_base, dtype=float)
    ybo = np.array(y_boost, dtype=float)
    f = np.array(feat, dtype=float)
    print(f"\n  base grammar-coherent   : {int(yb.sum())}/{n}")
    print(f"  boost grammar-coherent  : {int(ybo.sum())}/{n}")
    if f.std() > 0 and yb.std() > 0:
        acc = float((f == yb).mean())
        print(f"  SELF-REPEAT predictor acc: {acc:.2f} "
              f"(naive={max(yb.mean(), 1 - yb.mean()):.2f})")
    up = [i for i in range(n) if ybo[i] > yb[i]]
    dn = [i for i in range(n) if ybo[i] < yb[i]]
    print(f"  boost lifted: {[PROMPTS[i][0] for i in up]}")
    print(f"  boost hurt  : {[PROMPTS[i][0] for i in dn]}")

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