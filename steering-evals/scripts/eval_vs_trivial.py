#!/usr/bin/env python3
"""eval_vs_trivial.py — does the GEOMETRY beat TRIVIAL logit manipulation?

Question (asked on b7aa483's honesty check): our whole readout arc might
reduce to 'rotate hidden state once + penalize the just-sampled token'.
The trivial decode-time equivalents are straight logit edits. If they tie
or beat us, the geometry is a rediscovery of a cheap trick.

Modes, all plant 'chicken' on 'For dinner I made', 12 tok x 1 seed,
top_p 0.9, SAME metrics:
  recipe    : steer once@0 at law budget 2*gap/97+0.02 + state-space
              anti-last every step (rot 0.15 away from last token)
              [our champion]
  lipersist : logit-boost chicken by +L every step, no penalty
              [trivial 'make it say chicken']  -> expect loop
  liboost   : logit-boost +L every step AND repet-penalty nu on the
              just-sampled token [trivial analog of recipe]
  lioonce   : boost once@0 + repet-penalty (mirror recipe's one-shot)

  L = gap+3   (chicken must beat native by 3 logits; the trivial budget)
  nu = 1.15   (divide the last token's logit - standard repetition pen.)
Metrics: plant, x (# target words), dis, div, maxrun, rep4, sample.
KEY QUESTION: does liboost tie recipe? If yes, geometry := logit edits.

Writes steering_geometry_results/geo_vs_trivial_gemma.csv
"""
import csv
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
TARGET = 'chicken'
NTOK = 12
SEEDS = 1
A_REP = 0.15
NU = 1.15
OUT = Path('../steering_geometry_results/geo_vs_trivial_gemma.csv')


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach().float()

    tid_t = int(tok(' ' + TARGET, add_special_tokens=False).input_ids[0])
    capl = [int(c) for c in tok(' ' + TARGET.capitalize(),
                                add_special_tokens=False).input_ids]
    ids0 = tok(PROMPT, add_special_tokens=False,
               return_tensors='pt').input_ids.to(DEV)
    cf = {}

    def hook_c(m, i, o):
        cf['v'] = o[0, -1, :].float()

    h = model.model.norm.register_forward_hook(hook_c)
    with torch.no_grad():
        L0 = model(ids0).logits[0, -1].float()
    h.remove()
    native = int(L0.argmax())
    vf = cf['v'].float()
    vfn = vf / vf.norm()
    gap_t = float(L0[native] - L0[tid_t])
    A_ATT = 2 * gap_t / 97.0 + 0.02
    L_BOOST = gap_t + 3.0
    Wt = W[tid_t].float()
    tau_t = Wt - (vfn @ Wt) * vfn
    g_t = tau_t / tau_t.norm()
    nname = tok.decode([native])
    print(f"[{MODEL}] {PROMPT!r} tgt={TARGET!r} native={nname!r} "
          f"gap={gap_t:.2f} a_att={A_ATT:.3f} L_boost={L_BOOST:.1f} "
          f"nu={NU}")

    def anti_geo(vv, tid, amt=A_REP):
        """state-space anti: rotate away from the just-sampled token."""
        vv1 = vv / vv.norm()
        Wb = W[tid].float()
        tauv = Wb - (vv1 @ Wb) * vv1
        gl = -tauv / tauv.norm()
        return (vv1 * math.cos(amt) + gl * math.sin(amt)) * vv.norm()

    def sample(L, last=None, nu=NU):
        if last is not None:
            L = L.clone()
            L[last] = L[last] / nu
        p = torch.softmax(L.float(), dim=0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk) / (q * msk).sum()
        return int(torch.multinomial(qq, 1))

    def gen(mode):
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                if mode == 'recipe':
                    # steer once@0 + state-space anti-last
                    vv = vf
                    if step == 0:
                        vv = (vfn * math.cos(A_ATT) +
                              g_t * math.sin(A_ATT)) * vf.norm()
                    if toks:
                        vv = anti_geo(vv, toks[-1])

                    def inj(m, i, o, p=vv):
                        out = o.clone()
                        out[0, -1, :] = torch.as_tensor(
                            p, dtype=out.dtype, device=out.device)
                        return out

                    hi = model.model.norm.register_forward_hook(inj)
                    try:
                        with torch.no_grad():
                            L = model(ids).logits[0, -1].float()
                    finally:
                        hi.remove()
                else:
                    # TRIVIAL: pure logit edits, no hooks at all
                    with torch.no_grad():
                        L = model(ids).logits[0, -1].float().clone()
                    if mode in ('lipersist', 'liboost'):
                        L[tid_t] += L_BOOST
                    elif mode == 'lioonce' and step == 0:
                        L[tid_t] += L_BOOST
                    last = toks[-1] if toks else None
                    use_nu = mode in ('liboost', 'lioonce')
                    nxt = sample(L, last if use_nu else None)
                    toks.append(int(nxt))
                    ids = torch.cat(
                        [ids, torch.tensor([[nxt]], device=ids.device)],
                        dim=1)
                    continue
                # recipe path: sample with no logit edit
                nxt = sample(L)
                toks.append(int(nxt))
                ids = torch.cat([ids,
                                 torch.tensor([[nxt]], device=ids.device)],
                                dim=1)
            allres.append(toks)
        return allres

    modes = ['recipe', 'lipersist', 'liboost', 'lioonce']
    print(f"  {'mode':>10} {'plant':>6} {'x':>4} {'dis':>4} {'div':>5} "
          f"{'maxrun':>6} {'rep4':>5}  sample")
    rows = []
    for md in modes:
        gs = gen(md)
        for g in gs:
            fset = {tid_t} | set(capl)
            x = sum(1 for t in g if t in fset)
            dis = len({t for t in g if t in fset})
            plant = 1.0 if any(t in g[:10] for t in fset) else 0.0
            div = len(set(g)) / len(g)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(g)), default=0)
            rp = rep4(g)
            rows.append(dict(model=MODEL, prompt=PROMPT, mode=md,
                             plant=plant, x=x, dis=dis, div=round(div, 3),
                             maxrun=mr, rep4=round(rp, 3),
                             sample=tok.decode(g)))
            print(f"  {md:>10} {plant:>6.1f} {x:>4d} {dis:>4d} "
                  f"{div:>5.2f} {mr:>6d} {rp:>5.2f}  "
                  f"{tok.decode(g)[:38]!r}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()