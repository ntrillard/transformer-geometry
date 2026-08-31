#!/usr/bin/env python3
"""eval_gentle_walk.py — GRAMMAR-PRESERVING steering: the old gentle walk
vs the hard shots, tested explicitly for prose quality.

The old walk (3 deg/step re-aim toward the closest-to-state family
member) was scored '1/12 city' in 6cb9801 and dismissed. But its actual
text ('very talented girl from Australia named Anna Lee Smith that had
many') is GRAMMATICAL PROSE - unlike the 17 deg single-shot ('UK England
UK London British...') which is a topic word-list. The old metric
(topic-purity) threw away the property that matters for coherent text.

Test: gentle walks (1.5/3/5 deg per step, closest-family-member re-aim)
vs baseline (no steer) vs hard shots (5/17 deg once), on TWO models:
  Qwen2-0.5B  : baseline is grammatical ('...there was a very talented
                man by his name Peter Pan...') - can steering preserve it?
  Gemma-3-1B  : raw-prompt baseline loops ' I I I ...' - does ANY steer
                produce grammatical text?

metrics: plant (# target-family words in first 10), xtgt (total), dis,
div, maxrun, rep4, nsep (native token count), and the DECODED TEXT (the
real evidence). target family = 'city', both models.

CSV: steering_geometry_results/gentle_walk.csv
Run: timeout 90 python3 -u eval_gentle_walk.py
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

QUWEN = 'Qwen/Qwen2-0.5B-Instruct'
GEMMA = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
QPROMPT = 'Once upon a time, there was a'
GPROMPT = 'For dinner I made'
NTOK = 12
TARGET = 'city'
OUT = Path('../steering_geometry_results/gentle_walk.csv')


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def build_family(tok, cls='city'):
    fam = []
    word2id = {}
    for w in CLASSES[cls]:
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
            fam.append(int(ids[0]))
    return fam, word2id


def run_model(model_name, PROMPT, seeds, modes):
    model, tok = SGT.load_model(model_name, dtype='fp16')
    W = model.lm_head.weight.detach()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()
    fam, word2id = build_family(tok, TARGET)
    fset = set(fam)
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
    nname = tok.decode([native])
    print(f"\n[{model_name}] {PROMPT!r} target-family={TARGET} "
          f"native={nname!r}")

    def closest_member(vv):
        u = vv / vv.norm()
        return fam[int((Wn[fam].float() @ u).argmax())]

    def rot_toward(vv, tid, amt):
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(amt) + g * math.sin(amt)) * vv.norm()

    def gen(mode, sd):
        torch.manual_seed(sd)
        ids = ids0.clone()
        toks = []
        deg = {'walk1.5': 1.5, 'walk3': 3.0, 'walk5': 5.0}.get(mode)
        shot = {'shot5': 5.0, 'shot17': 17.0}.get(mode)
        for step in range(NTOK):
            vv = vf
            if deg is not None:
                t = closest_member(vv)
                vv = rot_toward(vv, t, math.radians(deg))
            elif shot is not None and step == 0:
                t = closest_member(vv)
                vv = rot_toward(vv, t, math.radians(shot))

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
    print(f"    {'mode':>8} {'plant':>6} {'xtgt':>5} {'dis':>4} "
          f"{'div':>5} {'maxrun':>6} {'rep4':>5} {'#SEP':>5}  text")
    for mode in modes:
        for sd in seeds:
            g = gen(mode, sd)
            x = sum(1 for t in g if t in fset)
            dis = len({t for t in g if t in fset})
            plant = 1.0 if any(t in g[:10] for t in fset) else 0.0
            div = len(set(g)) / len(g)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(g)), default=0)
            rp = rep4(g)
            nsep = sum(1 for t in g if t == native)
            txt = tok.decode(g)
            rows.append(dict(model=model_name, prompt=PROMPT, mode=mode,
                             seed=sd, plant=plant, xtgt=x, dis=dis,
                             div=round(div, 3), maxrun=mr,
                             rep4=round(rp, 3), nsep=nsep, text=txt))
            print(f"    {mode:>8} {plant:>6.1f} {x:>5d} {dis:>4d} "
                  f"{div:>5.2f} {mr:>6d} {rp:>5.2f} {nsep:>5}  "
                  f"{txt[:52]!r}", flush=True)
    return rows


def main():
    t0 = time.time()
    modes = ['baseline', 'walk1.5', 'walk3', 'walk5', 'shot5', 'shot17']
    all_rows = []
    all_rows += run_model(QUWEN, QPROMPT, [0, 7], modes)
    all_rows += run_model(GEMMA, GPROMPT, [0], modes)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()