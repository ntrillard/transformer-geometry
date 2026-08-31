#!/usr/bin/env python3
"""eval_switch_qwen.py — topic switching on Qwen2-1.5B base.

Generate several sentences on topic A, MID-GENERATION steer to a
completely different topic B, then C, then D. Controller: readout graft
+ anti-last @10deg + mild rep-penalty decode. NTOK=64 with 4 switches
every 16 tokens (city@0, animal@16, food@32, nature@48) so the model
has 16 tokens to stabilize between steerings.

Each switch: capture current readout vector, rotate 10deg toward the NEW
topic's closest word, inject, anti the new topic for a 2-token window.

One model, no template. Run: HF_TOKEN=<tok> timeout 180 python3 -u
eval_switch.py
"""
import csv
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'Qwen/Qwen2-1.5B'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 64
SEEDS = [0, 1]
ANGLE = 10.0
PEN = 0.5
OUT = Path('../steering_geometry_results/switch_qwen.csv')
PROMPT = 'The whole group sat down and began to discuss'
SWITCHES = {0: 'city', 16: 'animal', 32: 'food', 48: 'nature'}
SEG_N = 16
FAMILIES = {
    'city':   ['paris', 'london', 'berlin', 'madrid', 'tokyo'],
    'animal': ['cat', 'dog', 'bird', 'bear', 'horse'],
    'food':   ['pizza', 'sushi', 'pasta', 'burger'],
    'nature': ['forest', 'rice', 'water', 'sun', 'tree'],
}


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
        raise RuntimeError(f'{MODEL}: no model.model.norm readout surface')
    W = model.lm_head.weight.detach().float()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()

    famids = {}
    names = {}
    for fam, words in FAMILIES.items():
        ids = []
        for w in words:
            ids1 = tok(' ' + w, add_special_tokens=False).input_ids
            if len(ids1) == 1:
                ids.append(int(ids1[0]))
                names[int(ids1[0])] = w
        assert ids, f'family {fam} all multi-token'
        famids[fam] = ids
        print(f"  family {fam:>6}: {[names[i] for i in ids]}")

    def closest_to_fam(vv, fam):
        u = vv / vv.norm()
        s = Wn[famids[fam]].float() @ u
        return famids[fam][int(s.argmax())]

    def rot_to_angle(vv, tid, theta):
        a = math.radians(theta)
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(a) + g * math.sin(a)) * vv.norm()

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

    def forward(ids, inj_p=None, anti_t=None):
        hs = []
        try:
            if inj_p is not None:
                def inj(m, i, o, p=inj_p):
                    out = o.clone()
                    out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                    device=out.device)
                    return out
                hs.append(model.model.norm.register_forward_hook(inj))
            if anti_t is not None:
                def anti(m, i, o, tid=anti_t):
                    out = o.clone()
                    out[0, -1, tid] = -30.0
                    return out
                hs.append(model.lm_head.register_forward_hook(anti))
            with torch.no_grad():
                return model(ids).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()

    def capture_v(ids):
        vc = {}
        hk = model.model.norm.register_forward_hook(
            lambda m, i, o: vc.__setitem__('v', o[0, -1, :].float()))
        try:
            with torch.no_grad():
                model(ids).logits[0, -1].float()
        finally:
            hk.remove()
        return vc['v']

    def run_schedule(sd):
        torch.manual_seed(sd)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        sampled = []
        hits = {}
        last_switch = -10
        last_tgt = None
        for step in range(NTOK):
            inj_p = None
            anti_t = None
            if step in SWITCHES:
                fam = SWITCHES[step]
                v = capture_v(ids)
                tgt = closest_to_fam(v, fam)
                vp = rot_to_angle(v, tgt, ANGLE)
                inj_p = vp
                last_switch = step
                last_tgt = tgt
            elif 1 <= step - last_switch <= 2 and last_tgt is not None:
                anti_t = last_tgt
            L = forward(ids, inj_p=inj_p, anti_t=anti_t)
            nxt = sample(L, sampled)
            if step in SWITCHES:
                fam = SWITCHES[step]
                hits[step] = (fam, names[last_tgt], nxt in famids[fam])
            sampled.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        return sampled, hits

    rows = []
    steps = sorted(SWITCHES)
    ltr = {st: chr(ord('A') + i) for i, st in enumerate(steps)}
    print(f"\n[{MODEL}] SWITCH-LONG: {PROMPT!r}  NTOK={NTOK} "
          f"switches={SWITCHES}")
    for sd in SEEDS:
        toks, hits = run_schedule(sd)
        txt = tok.decode(toks)
        segs = [tok.decode(toks[i:i + SEG_N]) for i in
                range(0, NTOK, SEG_N)]
        print(f"\n  seed {sd}:")
        for i, st in enumerate(steps):
            fam, word, hit = hits[st]
            print(f"    switch@{st:>2} {fam:>6} -> {word:<6} "
                  f"{'HIT' if hit else 'miss'}  | "
                  f"{segs[i].strip()[:74]}")
        rows.append(dict(seed=sd, full=txt,
                         **{f'{ltr[st]}_fam': hits[st][0] for st in steps},
                         **{f'{ltr[st]}_word': hits[st][1] for st in steps},
                         **{f'{ltr[st]}_hit': hits[st][2] for st in steps}))
        print(f"    FULL: {PROMPT} {txt[:180]}")
    for st in steps:
        ok = sum(1 for r in rows if r[f'{ltr[st]}_hit'])
        print(f"\n  switch@{st} ({SWITCHES[st]}): planted, "
              f"hit={ok}/{len(SEEDS)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()