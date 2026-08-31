#!/usr/bin/env python3
"""eval_long2_qwen.py — same controller stress on Qwen2-1.5B base.
NTOK=8.

Controller: graft + anti-last @10deg (win2) + rep-penalty nucleus
decoding (PEN=0.05). Validated 5/10 vs 1/10 base at NTOK=4 on short
templates. Now: 10 long natural prompts (15-30 tok), NTOK=8, 2 seeds.

Criterion (calibrated for 8-tok prose): topic present, no <eos>,
rep4==0, no run>2, no decoded word occurring >2x.

One model, no template. Run: HF_TOKEN=<tok> timeout 30 python3 -u
eval_long2.py
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

MODEL = 'Qwen/Qwen2-1.5B'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 8
SEEDS = [0, 1]
TARGET = 'city'
OUT = Path('../steering_geometry_results/long2.csv')
ANGLE = 10.0
PEN = 0.05

PROMPTS = [
    ('goldg',  'The city that is famous for the Golden Gate Bridge and its steep hills is'),
    ('flight', 'After the meeting in New York, I had to catch a flight to'),
    ('blog',   'My travel blog about Europe begins with a story from the beautiful old town of'),
    ('cousin', 'My cousin moved to a new country last year, and now she lives in the bustling capital city of'),
    ('olymp',  'The summer Olympic Games that everyone remembers from my childhood were held in'),
    ('museum', 'The museum in the center of the city where the famous artist was born is located in'),
    ('bridge', 'The longest bridge in the world that connects two major islands is located in'),
    ('eiffel', 'If you ever visit Europe and want to see the Eiffel Tower, you should travel to'),
    ('film',   'The famous film director received the award and mentioned that the movie was shot entirely in'),
    ('grandma','My grandmother always tells stories about the city where she grew up during the war; it is a small town in'),
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
    if not hasattr(model.model, 'norm'):
        raise RuntimeError(f'{MODEL}: no model.model.norm (expected Qwen2/Gemma)')
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

    def coherent(toks, txt):
        if '<eos>' in txt:
            return False
        if rep4(toks) != 0.0:
            return False
        mr = max((sum(1 for _ in grp) for _, grp in
                  itertools.groupby(toks)), default=0)
        if mr > 2:
            return False
        wds = [w.lower() for w in txt.split() if w]
        freq = {}
        for w in wds:
            freq[w] = freq.get(w, 0) + 1
        if any(c > 2 for c in freq.values()):
            return False
        if not any(t in toks for t in fset):
            return False
        return True

    rows = []
    print(f"[{MODEL}] LONG2: different+longer prompts, NTOK={NTOK}, "
          f"{len(SEEDS)} seeds, PEN={PEN}")
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
                c = coherent(toks, txt)
                st += 'Y' if c else 'n'
                if c and ex == '':
                    ex = txt
            outs[arm] = (st, ex)
        y['freeP'] += 1 if outs['freeP'][0].count('Y') >= 2 else 0
        y['steerP'] += 1 if outs['steerP'][0].count('Y') >= 2 else 0
        y['steerB'] += 1 if outs['steerB'][0].count('Y') >= 2 else 0
        rows.append(dict(prompt=pname, gap=round(gap, 2),
                         freeP=outs['freeP'][0], steerP=outs['steerP'][0],
                         steerB=outs['steerB'][0],
                         steerP_out=outs['steerP'][1]))
        print("  %-7s %5.1f | %-6s | %-6s | %-6s | %s"
              % (pname[:7], gap, outs['freeP'][0], outs['steerP'][0],
                 outs['steerB'][0], outs['steerP'][1][:48]), flush=True)

    n = len(PROMPTS)
    print(f"\n  coherent (2/2 seeds): freeP {y['freeP']}/{n} | "
          f"steerP {y['steerP']}/{n} | steerB {y['steerB']}/{n}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()