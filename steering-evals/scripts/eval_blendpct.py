#!/usr/bin/env python3
"""eval_blendpct.py — META-SWEEP the BLEND PERCENTAGE at the readout.

User's idea: combine the EXPECTED (native) final state v and the ROTATED
v' bit-by-bit / percentage-by-percentage, feed the combination to the
head (final norm -> head only, NO re-sending through layers), and test
EACH percentage. This maps the full native<->rotated continuum finely.

At percentage pct (0..100): take pct% of each element from the steered
(rotated) vector and (100-pct)% from the native, per-dimension, then
normalize to ||v|| and feed the head:
    combo[pct] = normalize(  steered@pct + native@(100-pct)  ) * ||v||

We test pct in {0,10,20,...,100} and report plant + a seed-robust
coherence label per pct per prompt, so we find WHERE (if anywhere) the
continuum crosses from coherent-native to topical-steer without the
collapse. Also report the effective decision angle so we see the
geometry of the crossover.

One model, no template. Run: HF_TOKEN=<tok> timeout 18 python3 -u
eval_blendpct.py
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
NTOK = 6
SEEDS = [0, 1]
TARGET = 'city'
OUT = Path('../steering_geometry_results/blendpct.csv')

PROMPTS = [
    ('ask', 'If you ask me which European city is the most beautiful, I would say that'),
    ('fr',  'The capital of France is'),
]
PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def rep4(toks):
    if len(toks) < 4:
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

    def nat_vL(ids):
        cv = {}
        hk = model.model.norm.register_forward_hook(
            lambda m, i, o: cv.__setitem__('v', o[0, -1, :].float()))
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        hk.remove()
        return cv['v'], L

    def sample(L):
        p = torch.softmax(L.float(), 0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk) / (q * msk).sum()
        return int(torch.multinomial(qq, 1))

    def angle(v1, v2):
        return float(torch.acos(((v1 / v1.norm()) * (v2 / v2.norm())).sum()
                                .clamp(-1, 1)) * 180 / math.pi)

    def strict_ok(toks, txt):
        if not any(t in toks for t in fset):
            return False
        if '<eos>' in txt:
            return False
        if len(set(toks)) / len(toks) < 0.7:
            return False
        if rep4(toks) != 0.0:
            return False
        mr = max((sum(1 for _ in grp) for _, grp in
                  itertools.groupby(toks)), default=0)
        if mr > 2:
            return False
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        if any(c > 2 for c in counts.values()):
            return False
        return True

    rows = []
    print(f"[{MODEL}] BLEND % sweep at the readout (native + rotated), "
          f"head-only, {NTOK}-tok")
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        alpha = 2 * (gap / 97.0) + 0.02
        tgt = closest(vf)
        vp = rot(vf, tgt, alpha)
        print(f"\n==== {pname}: {pr!r} gap={gap:.1f} -> "
              f"{tok.decode([tgt])!r}  angrot={angle(vf,vp):.1f}deg ====")
        prev_label = None
        for pct in PCTS:
            # per-dimension combination: pct% steered, (100-pct)% native
            combo = (pct / 100.0) * vp + (1 - pct / 100.0) * vf
            combo = combo / combo.norm() * vf.norm()
            # effective angle vs native (the geometry of the crossover)
            a_eff = angle(vf, combo)
            oks = []
            str_seeds = ''
            txts = []
            for sd in SEEDS:
                torch.manual_seed(sd)
                ids = ids0.clone()
                toks = []
                for step in range(NTOK):
                    if step == 0:
                        def inj(m, i, o, p=combo):
                            out = o.clone()
                            out[0, -1, :] = torch.as_tensor(
                                p, dtype=out.dtype, device=out.device)
                            return out
                        h = model.model.norm.register_forward_hook(inj)
                        with torch.no_grad():
                            L = model(ids).logits[0, -1].float()
                        h.remove()
                    else:
                        with torch.no_grad():
                            L = model(ids).logits[0, -1].float()
                    nxt = sample(L)
                    toks.append(nxt)
                    ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)],
                                    dim=1)
                txt = tok.decode(toks)
                o = strict_ok(toks, txt)
                oks.append(o)
                str_seeds += 'Y' if o else 'n'
                txts.append(txt)
            robust = 1 if sum(oks) >= 2 else 0
            plant = 1.0 if any((sd == 0 and any(t in fset for t in
                            [x[0] if isinstance(x, list) else x for x in
                             []])) for sd in []) else None
            rows.append(dict(prompt=pname, pct=pct, angle_eff=round(a_eff, 2),
                             gap=round(gap, 3), per_seed=str_seeds,
                             robust=robust, text0=txts[0][:26],
                             text1=txts[1][:26]))
            print("  pct=%3d%%  ang=%5.1f  %s robust=%d  %s / %s"
                  % (pct, a_eff, str_seeds, robust,
                     txts[0][:20].strip(), txts[1][:20].strip()), flush=True)
            prev_label = robust

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()