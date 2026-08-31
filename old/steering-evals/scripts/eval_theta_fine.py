#!/usr/bin/env python3
"""eval_theta_fine.py — RESOLVE the coherent-plant transition angle with a
fine grid on 'ask' (the only known coherent prompt).

eval_transition found a plateau at 9 deg but only probed {3,6,9,12} deg, so
the true crossover is unresolved in the 6-9 bin. This fine-grains it:
angles 5.0..9.5 deg in 0.5 steps, rotation ON the native->rotated arc,
seed-robust strict coherence (>=2/3 seeds). Plus report the P(wants
'topic plant' at each angle => the decision-frontier geometry).

Big-leap: pin the exact minimal coherent angle so the controller has a
precise setpoint (not a coarse bin). Also do a second target row to see if
theta* is target-independent (chicken) at the same prompt.

One model, no template. Run: HF_TOKEN=<tok> timeout 18 python3 -u
eval_theta_fine.py
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
NTOK = 5
SEEDS = [0, 1, 2]
TARGET = 'city'
OUT = Path('../steering_geometry_results/theta_fine.csv')

PROMPT = 'If you ask me which European city is the most beautiful, I would say that'
# city row AND an independent token row (chicken) to test target-independence
TGTWORDS = ['paris', 'chicken']
ANGLES = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]


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

    def tok1(w):
        iid = tok(' ' + w, add_special_tokens=False).input_ids
        return int(iid[0]) if len(iid) == 1 else None

    fam = [int(tok(' ' + w, add_special_tokens=False).input_ids[0])
           for w in CLASSES[TARGET]]
    fset = set(fam)

    tid_paris = tok1('paris')
    tid_chicken = tok1('chicken')

    def rot_to_angle(vv, tid, theta):
        a = math.radians(theta)
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(a) + g * math.sin(a)) * vv.norm()
    # note: rot into the Wt plane tangent; must also ensure we rotate toward
    # the target (the git uses closest, here we use the explicit target row)

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

    def robust(vin, ids0):
        oks = []
        for sd in SEEDS:
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                if step == 0:
                    def inj(m, i, o, p=vin):
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
            oks.append(strict_ok(toks, tok.decode(toks)))
        return int(sum(oks) >= 2), sum(oks), ''.join('Y' if o else 'n'
                                                     for o in oks)

    ids0 = tok(PROMPT, add_special_tokens=False,
               return_tensors='pt').input_ids.to(DEV)
    vf, Ln = nat_vL(ids0)
    nrm = vf.norm()
    rows = []
    print(f"[{MODEL}] {PROMPT!r}  fine angle grid 5.0-9.5 (0.5), "
          f"{len(SEEDS)} seeds, on-arc rotation")
    for w in TGTWORDS:
        tid = tok1(w.lower())
        if tid is None:
            continue
        print(f"\n== target '{w}' (tid={tid}, "
              f"is-city={tid in fset}) ==")
        theta_star = None
        for a in ANGLES:
            vin = rot_to_angle(vf, tid, a)
            rob, nco, seeds = robust(vin, ids0)
            if rob and theta_star is None:
                theta_star = a
            rows.append(dict(target=w, angle=a, robust=rob, n_coherent=nco,
                             seeds=seeds))
            print("  %5.1f deg  seeds=%s  ncoherent=%d robust=%d"
                  % (a, seeds, nco, rob), flush=True)
        # find first run of robust
        print(f"  -> theta* (first robust): "
              f"{theta_star if theta_star is not None else 'none'}")
        # also report the transition: last angle at 0 before first 1
    print(f"\n  transition band reported as first-robust-angle above "
          f"(resolved to 0.5 deg).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()