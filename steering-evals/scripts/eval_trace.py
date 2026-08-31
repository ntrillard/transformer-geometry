#!/usr/bin/env python3
"""eval_trace.py — SEE the mechanism. Dump real free-run vs steered
4-token continuations for recovering, copular-collapsing, and agentic
prompts, plus per-run validity flags. No CSV, print only.

Run: HF_TOKEN=<tok> timeout 20 python3 -u eval_trace.py
"""
import itertools
import math
import time

import numpy as np
import torch

import steering_geometry_test as SGT
from eval_nb_quick import CLASSES

MODEL = 'google/gemma-3-1b-pt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 4
SEED = 1
TARGET = 'city'
ANGLE = 10.0

PROMPTS = [
    # recoverers (steer-coherent in earlier probes)
    ('ask',     'If you ask me which European city is the most beautiful, I would say that'),
    ('holiday', 'My dream vacation destination is'),
    ('univ',    'The oldest university is in'),
    ('mona',    'The Mona Lisa is displayed in'),
    ('novel',   'The story takes place in'),
    ('hockey',  'The hockey championship was held in'),
    # copular collapsers
    ('austr',   'The biggest city in Australia is'),
    ('itcap',   'The capital of Italy is'),
    ('decap',   'The capital of Germany is'),
    ('france',  'The capital of France is'),
    # agentic family (steer failed here)
    ('fly',     'Tonight I am flying to'),
    ('fest',    'The music festival is held in'),
    ('conf',    'The annual conference will take place in'),
    ('vac',     'We spent our summer vacation in'),
    ('live',    'My cousin lives in'),
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

    def run(sd, ids0, vp, tgt, steer):
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
            nxt = sample(L)
            toks.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        return toks

    def flags(toks, txt):
        fl = []
        if '<eos>' in txt:
            fl.append('EOS')
        if not any(t in toks for t in fset):
            fl.append('noTpc')
        if rep4(toks) != 0.0:
            fl.append('rep4')
        mr = max((sum(1 for _ in grp) for _, grp in
                  itertools.groupby(toks)), default=0)
        if mr > 1:
            fl.append('run%d' % mr)
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        if any(c > 1 for c in counts.values()):
            fl.append('dup')
        return ' '.join(fl) if fl else 'ok'

    print(f"[{MODEL}] TRACE: free vs steered, seed={SEED}, NTOK={NTOK} "
          f"({len(PROMPTS)} prompts)")
    print("  %-8s %6s | %-38s | %s" % ('prompt', 'gap',
                                       'free', 'steered'))
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        tgt = closest(vf)
        vp = rot_to_angle(vf, tgt, ANGLE)

        tf = run(SEED, ids0, None, None, steer=False)
        tf_txt = tok.decode(tf, skip_special_tokens=False)
        ts = run(SEED, ids0, vp, tgt, steer=True)
        ts_txt = tok.decode(ts, skip_special_tokens=False)

        print("  %-8s %6.1f | %-38s | %s" % (
            pname[:8], gap,
            (tf_txt[:44] + ' [' + flags(tf, tf_txt) + ']'),
            (ts_txt[:44] + ' [' + flags(ts, ts_txt) + ']')), flush=True)

    print(f"\n  flags legend: EOS | noTpc=no city in out | rep4 | "
          f"runN=trigram run | dup=repeated token")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()