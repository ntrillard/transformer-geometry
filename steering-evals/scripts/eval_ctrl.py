#!/usr/bin/env python3
"""eval_ctrl.py — DECODER-level fix test of the revised mechanism.

Trace showed: the 1B base natively degenerates into token repetition at
NTOK=4 (free runs are 'is is is is', 'that that that that' — no city
even). The steered 'collapse' ('Cairo is is is') is the same repetition
sickness AFTER the topic is planted. Steering reliably plants the city;
coherence is decided by the DECODER.

Test: repetition-penalized sampler (damp already-sampled tokens in the
nucleus) × {free, steer} vs steer-without-penalty baseline.
PREDICTION: steer + rep-penalty > 7/10 coherent on the 'failing' family.

One model, no template. Run: HF_TOKEN=<tok> timeout 25 python3 -u
eval_ctrl.py
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
OUT = Path('../steering_geometry_results/ctrl.csv')
ANGLE = 10.0
PEN = 0.05          # multiplicative damp per prior occurrence

PROMPTS = [
    ('ask',    'If you ask me which European city is the most beautiful, I would say that'),
    ('mona',   'The Mona Lisa is displayed in'),
    ('austr',  'The biggest city in Australia is'),
    ('itcap',  'The capital of Italy is'),
    ('decap',  'The capital of Germany is'),
    ('france', 'The capital of France is'),
    ('fly',    'Tonight I am flying to'),
    ('fest',   'The music festival is held in'),
    ('conf',   'The annual conference will take place in'),
    ('vac',    'We spent our summer vacation in'),
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

    def sample(L, prefix):
        p = torch.softmax(L.float(), 0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk)
        for t in set(prefix):
            c = prefix.count(t)
            if c:
                qq[t] = qq[t] * (PEN ** c)
        qq = qq / qq.sum()
        return int(torch.multinomial(qq, 1))

    def run(ids0, vp, tgt, sd, steer, pen):
        torch.manual_seed(sd)
        ids = ids0.clone()
        toks = []
        for step in range(NTOK):
            hs = []
            try:
                if steer and step == 0:
                    def inj(m, i, o, p=vp):
                        out = o.clone()
                        out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                        device=out.device)
                        return out
                    hs.append(model.model.norm.register_forward_hook(inj))
                if steer and 1 <= step <= 2:
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
            nxt = sample(L, toks if pen else [])
            toks.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        return toks

    def clean_toks(toks, txt):
        if '<eos>' in txt:
            return False
        if rep4(toks) != 0.0:
            return False
        # token-level: runs and id-duplicates
        mr = max((sum(1 for _ in grp) for _, grp in
                  itertools.groupby(toks)), default=0)
        if mr > 1:
            return False
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        if any(c > 1 for c in counts.values()):
            return False
        # word-level: distinct decoded words (no 'London London' variants)
        wds = [w for w in txt.split() if w]
        if len(set(wds)) != len(wds):
            return False
        if any(wds.count(w) > 1 for w in set(wds)):
            return False
        return True

    def topic_ok(toks, txt):
        if not any(t in toks for t in fset):
            return False
        return clean_toks(toks, txt)

    rows = []
    print(f"[{MODEL}] CTRL: rep-penalty sampler x free/steer vs "
          f"steer-base ({len(PROMPTS)} prompts, 3 seeds, PEN={PEN})")
    print("  %-7s %5s | %-6s | %-6s | %-6s | %s"
          % ('prompt', 'gap', 'freeP', 'steerP', 'steerB', 'steerP_out'))
    y = {'freeP': 0, 'steerP': 0, 'steerB': 0}
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        tgt = closest(vf)
        vp = rot_to_angle(vf, tgt, ANGLE)
        outs = {}
        for arm, (steer, pen) in [('freeP', (False, True)),
                                  ('steerP', (True, True)),
                                  ('steerB', (True, False))]:
            st = ''
            ex = ''
            for sd in SEEDS:
                toks = run(ids0, vp, tgt, sd, steer, pen)
                txt = tok.decode(toks)
                c = topic_ok(toks, txt)
                st += 'Y' if c else 'n'
                if c and ex == '':
                    ex = txt
            outs[arm] = (st, ex)
        nsp = outs['steerP'][0].count('Y')
        if nsp >= 2:
            y['steerP'] += 1
        if outs['freeP'][0].count('Y') >= 2:
            y['freeP'] += 1
        if outs['steerB'][0].count('Y') >= 2:
            y['steerB'] += 1
        rows.append(dict(prompt=pname, gap=round(gap, 2),
                         freeP=outs['freeP'][0], steerP=outs['steerP'][0],
                         steerB=outs['steerB'][0],
                         steerP_out=outs['steerP'][1]))
        print("  %-7s %5.1f | %-6s | %-6s | %-6s | %s"
              % (pname[:7], gap, outs['freeP'][0], outs['steerP'][0],
                 outs['steerB'][0], outs['steerP'][1][:44]), flush=True)

    n = len(PROMPTS)
    print(f"\n  coherent (2/3): free+pen {y['freeP']}/{n} | "
          f"steer+pen {y['steerP']}/{n} | steer-base {y['steerB']}/{n}")
    print(f"  prediction: steer+pen > 7/10")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()