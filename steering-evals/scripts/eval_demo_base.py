#!/usr/bin/env python3
"""eval_demo_base.py — COHERENT steered generations on the BASE model.

The whole thread (user-instigated) converged on the honest surface:
  base model google/gemma-3-1b-pt, REAL prompts (complete sentences),
  gentle walking (closest-to-state family member, ~3-5 deg/step).
This demonstrates it end-to-end AND checks whether the readout law
(alpha* = gap/97) transfers to the base model.

Part A  LAW TRANSFER: on 3 prompts, aexact (analytic crossing) vs
        apred = gap/97 for a few sample tokens (ratio ~1 = law holds).

Part B  COHERENT GENERATIONS: per prompt, baseline vs gentle walk toward
        a topic family (city/food/animal):
          baseline : no steer
          walk3    : 3 deg/step closest-to-state member
          walk5    : 5 deg/step
        metrics: plant (family word in first 10), xtgt, dis, rep4, div,
        maxrun + the DECODED TEXT (the real evidence of coherence).

Run: HF_TOKEN=<token> timeout 120 python3 -u eval_demo_base.py
CSV: steering_geometry_results/demo_base_gemma.csv
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

GEMMA_BASE = 'google/gemma-3-1b-pt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 14
SEEDS = 1
PROMPTS = [
    ('story', 'Once upon a time, there was a', 'city'),
    ('capital', 'The capital of France is', 'city'),
    ('town', 'The story begins in a small town where', 'nature'),
    ('breakfast', 'Every morning I eat', 'food'),
]
OUT = Path('../steering_geometry_results/demo_base_gemma.csv')


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def main():
    t0 = time.time()
    model, tok = SGT.load_model(GEMMA_BASE, dtype='fp16')
    W = model.lm_head.weight.detach()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()
    V = W.shape[0]

    # family member ids per class (all classes available)
    fams = {}
    for cls, words in CLASSES.items():
        ids = []
        for w in words:
            t = tok(' ' + w, add_special_tokens=False).input_ids
            if len(t) == 1:
                ids.append(int(t[0]))
        if len(ids) >= 6:
            fams[cls] = ids
    print(f"[{GEMMA_BASE}] families: "
          f"{', '.join(f'{k}({len(v)})' for k, v in fams.items())}")

    # ---------- Part A: law transfer ----------
    print("\nPart A  LAW TRANSFER (alpha* = gap/97 on the BASE model):")
    print(f"    {'prompt':>9} {'token':>7} {'gap':>6} {'apred':>6} "
          f"{'aexact':>7} {'ratio':>6}")
    law_rows = []
    probe_pairs = [
        ('story', 'city'), ('capital', 'city'), ('breakfast', 'food')]
    frames = {}
    for pname, PROMPT, pcls in PROMPTS:
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
        frames[pname] = dict(ids0=ids0, L0=L0, native=native,
                             vf=cf['v'].float(),
                             vfn=cf['v'].float() /
                             cf['v'].norm())
    for pname, tname in probe_pairs:
        fr = frames[pname]
        L0, native = fr['L0'], fr['native']
        vfn = fr['vfn']
        # pick 2 family tokens to probe
        fam = fams.get(tname, fams['city'])
        for t in fam[:2]:
            Wt = Wn[t].float()
            Wn_nat = Wn[native].float()
            A_ = vfn @ (Wt - Wn_nat)
            tau = Wt - (vfn @ Wt) * vfn
            B_ = (tau @ (Wt - Wn_nat)) / (tau.norm() + 1e-12)
            aex = abs(math.atan2(-A_, B_))
            gap = float(L0[native] - L0[t])
            apred = gap / 97.0
            ratio = aex / (apred + 1e-12)
            law_rows.append((pname, tname, tok.decode([t]), gap, apred,
                             aex, ratio))
            print(f"    {pname:>9} {tok.decode([t]):>7} {gap:>6.2f} "
                  f"{apred:>6.4f} {aex:>7.4f} {ratio:>6.2f}", flush=True)

    # ---------- Part B: coherent generations ----------
    print("\nPart B  COHERENT GENERATIONS (base Gemma, real prompts, "
          f"gentle walk, {NTOK} tok):")

    def closest_member(vv, fam):
        u = vv / vv.norm()
        return fam[int((Wn[fam].float() @ u).argmax())]

    def rot_toward(vv, tid, amt):
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(amt) + g * math.sin(amt)) * vv.norm()

    def gen(fr, fam, deg):
        ids = fr['ids0'].clone()
        vf, vfn = fr['vf'], fr['vfn']
        torch.manual_seed(0)
        toks = []
        for step in range(NTOK):
            vv = vf
            if deg is not None:
                t = closest_member(vv, fam)
                vv = rot_toward(vv, t, math.radians(deg))

            def inj(m, i, o, p=vv):
                out = o.clone()
                out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                device=out.device)
                return out

            hi = model.model.norm.register_forward_hook(inj)
            try:
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
            finally:
                hi.remove()
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
        return toks

    rows = []
    print(f"    {'prompt':>9} {'mode':>6} {'plant':>6} {'xtgt':>5} "
          f"{'rep4':>6} {'div':>5} {'maxrun':>6}  text")
    for pname, PROMPT, pcls in PROMPTS:
        fr = frames[pname]
        fam = fams[pcls]
        fset = set(fam)
        for mode, deg in (('base', None), ('walk3', 3.0), ('walk5', 5.0)):
            toks = gen(fr, fam, deg)
            x = sum(1 for t in toks if t in fset)
            dis = len({t for t in toks if t in fset})
            plant = 1.0 if any(t in toks[:10] for t in fset) else 0.0
            rp = rep4(toks)
            div = len(set(toks)) / len(toks)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(toks)), default=0)
            txt = tok.decode(toks)
            rows.append(dict(model=GEMMA_BASE, prompt=pname, mode=mode,
                             family=pcls, plant=plant, xtgt=x, dis=dis,
                             rep4=round(rp, 3), div=round(div, 3),
                             maxrun=mr, text=txt))
            print(f"    {pname:>9} {mode:>6} {plant:>6.1f} {x:>5d} "
                  f"{rp:>6.2f} {div:>5.2f} {mr:>6d}  {txt[:56]!r}",
                  flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()