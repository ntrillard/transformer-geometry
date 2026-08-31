#!/usr/bin/env python3
"""eval_realroll.py — REAL free-run rollouts to check if the proxy
'gap/dynamic' modes actually predict coherent prose (ground truth).

metamode2/3/4 used next-token entropy proxies for 'recovery'. They
self-contradicted (dynamic widening 7/12 in metamode3 vs 4/12 in
metamode4). The only honest test: run the REAL shot_anti 8-token
rollout on a handful of boundary prompts and measure TRUE coherence
(div, rep4) - then compare against the entropy proxy label.

For each prompt: shot_anti (ONE law-budget steer at step 0 toward
closest family member, then free-run with the planted target token
suppressed at lm_head from step 1) -> 8-token text. Also record the
dynamic entropy label for the same prompt (from metamode logic).

No commit. Run: HF_TOKEN=<tok> timeout 10 python3 -u eval_realroll.py
"""
import csv
import itertools
import math
import time
from pathlib import Path

import torch

import steering_geometry_test as SGT
from eval_nb_quick import CLASSES

MODEL = 'google/gemma-3-1b-pt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 8
SEEDS = [0, 1, 2]
TARGET = 'city'
OUT = Path('../steering_geometry_results/realroll.csv')

PROMPTS = [
    ('fr',   'The capital of France is'),          # gap ~6.0  (proxies: recovers)
    ('jpn',  'The capital of Japan is'),           # gap ~7.2
    ('ask',  'If you ask me which European city is the most beautiful, I would say that'),  # gap 8.8 (antidote: div 1.00)
]
SEEDFIX = 0


def rep4(toks):
    if len(toks) < 8:
        return 1.0
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

    def rot(vv, tid, alpha):
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(alpha) + g * math.sin(alpha)) * vv.norm()

    def sample(L):
        p = torch.softmax(L.float(), 0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk) / (q * msk).sum()
        return int(torch.multinomial(qq, 1))

    def nat_state(ids):
        cv = {}
        hk = model.model.norm.register_forward_hook(
            lambda m, i, o: cv.__setitem__('v', o[0, -1, :].float()))
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        hk.remove()
        return cv['v'], L

    rows = []
    print(f"[{MODEL}] REAL shot_anti 8-token rollouts, 3 seeds")
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_state(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        alpha = 2 * (gap / 97.0) + 0.02
        tgt = closest(vf)
        vp = rot(vf, tgt, alpha)
        print(f"\n[{pname}] {pr[:50]!r} gap={gap:.1f} tgt={tok.decode([tgt])!r}")
        for sd in SEEDS:
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                hooks = []
                if step == 0:
                    def inj(m, i, o, p=vp):
                        out = o.clone()
                        out[0, -1, :] = torch.as_tensor(
                            p, dtype=out.dtype, device=out.device)
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
                ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)],
                                dim=1)
            x = sum(1 for t in toks if t in fset)
            div = len(set(toks)) / len(toks)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(toks)), default=0)
            rp = rep4(toks)
            txt = tok.decode(toks)
            rows.append(dict(prompt=pname, gap=round(gap, 2), seed=sd,
                             plant=float(any(t in toks[:6] for t in fset)),
                             xtgt=x, div=round(div, 3), maxrun=mr,
                             rep4=round(rp, 3), text=txt))
            print(f"  sd{sd} xtgt={x} div={div:.2f} maxrun={mr} "
                  f"rep4={rp:.2f}  {txt[:60]!r}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()