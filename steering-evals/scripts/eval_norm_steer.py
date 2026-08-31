#!/usr/bin/env python3
"""eval_norm_steer.py — the untested corner: mid-depth steering with
READOUT-SCALE perturbation (the normalization prescription).

ALL prior mid steering (eval_midaster, eval_norm_rescale, depth walks)
injected a state-norm-preserving ROTATION: perturbation size ~ alpha *
||v_d|| ~ 0.4 * 3762 ~ 1500 units against a readout norm of ~100. The
norm-ratio scandal (eval_norm_rescale) proved transfer depends on the
injected state's SCALE. Here: perturb the mid state ADDITIVELY by an
ABSOLUTE epsilon at the FINAL-norm scale (||v_final|| ~ 100), let the
natural stack transform it (no readout firewall). Grammar directions in
the residual stream stay dominant by construction -> maybe coherence.

Modes (target family 'city', prompt 'The capital of France is', base
model google/gemma-3-1b-pt, 12 tok):
  base           : no steer
  midonce        : L10 additive steer once at step 0
                     eps_abs = 2*(gap/97)*||v_final|| toward closest
                     city member
  midwalk        : L10 additive steer EVERY step (eps_abs per step,
                     gap recomputed... approximated by fixed eps_abs)
  rotwalk        : old CONTROL: norm-preserving rotation at L10 every
                     step (the known grammar-killer)

metrics: plant (any city word first 10), xtgt, dis, rep4, div, maxrun,
#SEP + the DECODED TEXT. CSV persisted.
Run: HF_TOKEN=<tok> timeout 150 python3 -u eval_norm_steer.py
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
PROMPT = 'The capital of France is'
NTOK = 12
SEEDS = 1
DEPTH = 10          # L10 (0-based L9), pre-sink
OUT = Path('../steering_geometry_results/norm_steer_gemma.csv')


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()

    fam = [int(tok(' ' + w, add_special_tokens=False).input_ids[0])
           for w in CLASSES['city']]
    fset = set(fam)
    Wff = Wn[fam].float()

    ids0 = tok(PROMPT, add_special_tokens=False,
               return_tensors='pt').input_ids.to(DEV)
    cf = {}

    def hook_c(m, i, o):
        cf['v'] = o[0, -1, :].float()

    # capture natural states: final norm + L10
    caps = {}
    hooks = [
        model.model.layers[DEPTH - 1].register_forward_hook(
            lambda m, i, o: caps.__setitem__('d', o[0, -1, :].float())),
        model.model.norm.register_forward_hook(
            lambda m, i, o: caps.__setitem__('f', o[0, -1, :].float())),
    ]
    with torch.no_grad():
        L0 = model(ids0).logits[0, -1].float()
    for h in hooks:
        h.remove()
    native = int(L0.argmax())
    vf = caps['f'].float()
    nrmf = float(vf.norm())
    vd = caps['d'].float()
    nrmd = float(vd.norm())
    gap_target = float(L0[native] - L0[fam].max())
    eps_abs = 2 * (gap_target / 97.0) * nrmf
    print(f"[{MODEL}] {PROMPT!r} native={tok.decode([native])!r} "
          f"||v_d||={nrmd:.0f} ||v_f||={nrmf:.0f} ratio={nrmd / nrmf:.1f} "
          f"gap={gap_target:.2f} eps_abs={eps_abs:.1f}")
    print(f"  city fam n={len(fam)}  eps = 2*(gap/97)*||vf|| = "
          f"law-budget * final-norm  (NOT a rotation of the mid state)")

    def closest(vv):
        u = vv / vv.norm()
        return fam[int((Wff @ u).argmax())]

    def gen(mode):
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                if mode == 'base':
                    with torch.no_grad():
                        L = model(ids).logits[0, -1].float()
                    toks.append(int(torch.softmax(L, 0).argmax()))
                    ids = torch.cat(
                        [ids, torch.tensor([[toks[-1]]], device=DEV)],
                        dim=1)
                    continue
                # capture the natural mid state fresh (for closest)
                c2 = {}
                hk = model.model.layers[DEPTH - 1].register_forward_hook(
                    lambda m, i, o: c2.__setitem__('v', o[0, -1, :].float()))
                with torch.no_grad():
                    _ = model(ids)
                hk.remove()
                vd_now = c2['v'].float()
                vdn = vd_now / vd_now.norm()
                t = closest(vd_now) if mode != 'base' else fam[0]
                Wb = Wn[t].float()
                tau = Wb - (vdn @ Wb) * vdn
                g = tau / tau.norm()

                def inj(m, i, o, p=None):
                    out = o.clone()
                    if p is None:
                        return out
                    out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                    device=out.device)
                    return out

                if mode == 'midonce' and step == 0:
                    pv = vd_now + eps_abs * g
                elif mode == 'midwalk':
                    pv = vd_now + eps_abs * g
                elif mode == 'rotwalk':
                    # old control: norm-preserving rotation
                    pv = (vdn * math.cos(eps_abs / vd_now.norm()) +
                          g * math.sin(eps_abs / vd_now.norm())) * \
                        vd_now.norm()
                else:
                    pv = None
                if pv is not None:
                    hk2 = model.model.layers[DEPTH - 1].register_forward_hook(
                        lambda m, i, o, q=pv: inj(m, i, o, q))
                else:
                    hk2 = None
                try:
                    with torch.no_grad():
                        L = model(ids).logits[0, -1].float()
                finally:
                    if hk2 is not None:
                        hk2.remove()
                p = torch.softmax(L.float(), dim=0)
                q = p.clone(); order = q.argsort(descending=True)
                k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
                msk = torch.zeros_like(q); msk[order[:k]] = 1
                qq = (q * msk) / (q * msk).sum()
                nxt = int(torch.multinomial(qq, 1))
                toks.append(int(nxt))
                ids = torch.cat([ids,
                                 torch.tensor([[nxt]], device=ids.device)],
                                dim=1)
            allres.append(toks)
        return allres

    modes = ['base', 'midonce', 'midwalk', 'rotwalk']
    print(f"  {'mode':>9} {'plant':>6} {'xtgt':>5} {'dis':>4} {'rep4':>6} "
          f"{'div':>5} {'maxrun':>6}  text")
    rows = []
    for md in modes:
        gs = gen(md)
        for g in gs:
            x = sum(1 for t in g if t in fset)
            dis = len({t for t in g if t in fset})
            plant = 1.0 if any(t in g[:10] for t in fset) else 0.0
            rp = rep4(g)
            div = len(set(g)) / len(g)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(g)), default=0)
            txt = tok.decode(g)
            rows.append(dict(model=MODEL, prompt=PROMPT, mode=md,
                             plant=plant, xtgt=x, dis=dis,
                             rep4=round(rp, 3), div=round(div, 3),
                             maxrun=mr, text=txt))
            print(f"  {md:>9} {plant:>6.1f} {x:>5d} {dis:>4d} "
                  f"{rp:>6.2f} {div:>5.2f} {mr:>6d}  {txt[:56]!r}",
                  flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()