#!/usr/bin/env python3
"""eval_agentic.py — Task 1 (option 1): mechanism forward-prediction.

Mechanism (from the why series): anti-last coherence depends on the model
having a non-degenerate continuation PAST the planted topic. The copular
family ("The capital of X is") fails because the prompt ends inside a
copula: after graft-city + anti, the only high-mass token is the same
copula -> 'is [...] is' island. CONCRETE/AGENTIC prompts end in a
preposition ("fly to", "held in") where the model has rich content after
the city -> the mechanism PREDICTS near-universal coherence here.

Test: 8 agentic prompts, anti-last @10deg, window 2, 3 seeds, fixed
coherence (topic + grammar). If >6/8 coherent, the mechanism's forward
prediction holds.

One model, no template. Run: HF_TOKEN=<tok> timeout 20 python3 -u
eval_agentic.py
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
OUT = Path('../steering_geometry_results/agentic.csv')
ANGLE = 10.0

PROMPTS = [
    ('fly',   'Tonight I am flying to'),
    ('meet',  'After work I will meet my friends in'),
    ('fest',  'The music festival is held in'),
    ('conf',  'The annual conference will take place in'),
    ('award', 'The award ceremony was hosted in'),
    ('vac',   'We spent our summer vacation in'),
    ('train', 'The express train from Madrid stops in'),
    ('live',  'My cousin lives in'),
]


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

    def run_steer(ids0, vp, tgt, sd):
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
    print(f"[{MODEL}] AGENTIC family: anti-last @10deg win2, 3 seeds "
          f"({len(PROMPTS)} prompts)")
    print("  %-7s %5s | %s | %s" % ('prompt', 'gap', 'seeds', 'out'))
    ncoh = 0
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        tgt = closest(vf)
        vp = rot_to_angle(vf, tgt, ANGLE)
        st = ''
        example = ''
        for sd in SEEDS:
            toks = run_steer(ids0, vp, tgt, sd)
            txt = tok.decode(toks)
            c = topic_ok(toks, txt)
            st += 'Y' if c else 'n'
            if c and example == '':
                example = txt
        ncoh += 1 if st.count('Y') >= 2 else 0
        rows.append(dict(prompt=pname, gap=round(gap, 2), seeds=st,
                         example=example))
        print("  %-7s %5.1f | %s | %s" % (pname[:7], gap, st,
                                          example[:42]), flush=True)

    n = len(PROMPTS)
    print(f"\n  coherent (2/3 seeds): {ncoh}/{n}  "
          f"prediction: mechanism says >6/8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()