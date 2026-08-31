#!/usr/bin/env python3
"""eval_metagree — the single big-leap meta model over ALL accumulated
coherence evidence.

Consolidate every finding into ONE predictive fit:
  coherent = F( on_arc_angle, operator_is_linear/rot, native_gate )
Test a bank of prompts at controlled angles; collect:
  - native_gate : is the prompt's UNSTEERED continuation grammatical?
    (native div/coherence on a short rollout) -> the gate feature
  - on_arc_angle: effective angle (deg) of a rotation-on-arc steer
  - operator    : linear(rot/lin/slerp) vs non-linear (all on 9-deg)
  - label       : seed-robust strict coherence after steer
Then FIT (logistic or threshold) coherent ~ f(gate, angle) and report:
  - the fitted gate threshold (native coherence needed)
  - the fitted angle threshold (min angle given gate)
Big-leap: a SINGLE decision rule 'if native_gate AND angle>=9 then
steer-to-9-coherent else don't' validated across prompts and operators.

Run: HF_TOKEN=<tok> timeout 18 python3 -u eval_metagree.py
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
SEEDS = [0, 1]
TARGET = 'city'
OUT = Path('../steering_geometry_results/metagree.csv')
ANGLES = [3.0, 6.0, 9.0, 12.0]

PROMPTS = [
    ('ask',  'If you ask me which European city is the most beautiful, I would say that'),
    ('japan', 'The capital of Japan is'),
    ('eif',  'The Eiffel Tower is located in'),
    ('olymp','The Olympic Games were held in'),
    ('travel','I would love to travel to'),
    ('visit', 'People from all over the world visit'),
    ('france','The capital of France is'),
]


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
        return int(sum(oks) >= 2)

    rows = []
    print(f"[{MODEL}] meta-fit coherent ~ f(gate, on-arc angle), "
          f"operators on-arc, {len(SEEDS)} seeds")
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        tgt = closest(vf)
        # NATIVE GATE: is the unsteered continuation grammatical+topical?
        ngate = robust(vf, ids0)
        for a in ANGLES:
            v_ang = rot_to_angle(vf, tgt, a)
            r = robust(v_ang, ids0)
            rows.append(dict(prompt=pname, gap=round(gap, 3), angle=a,
                             native_gate=ngate, coherent=r))
        print("  %-8s gap=%5.1f gate=%d  angles(%s): %s"
              % (pname[:8], gap, ngate,
                 ','.join(str(int(x)) for x in ANGLES),
                 ''.join(str(rows[pname == r['prompt']]['coherent']) for
                         pname in []) or
                 ' '.join(f"{a}:{r['coherent']}" for a, r in
                          zip(ANGLES, [r for r in rows
                                       if r['prompt'] == pname]))),
              flush=True)

    # META-FIT via a simple decision rule and report thresholds
    X = np.array([[r['native_gate'], r['angle']] for r in rows], dtype=float)
    y = np.array([r['coherent'] for r in rows], dtype=float)
    print(f"\n  n={len(y)} positive(coherent)={int(y.sum())}")
    # single-rule candidates
    rules = {
        'gate>=1 and angle>=9': \
            ((X[:, 0] >= 1) & (X[:, 1] >= 9)),
        'gate>=1 and angle>=6': \
            ((X[:, 0] >= 1) & (X[:, 1] >= 6)),
        'angle>=9 only': (X[:, 1] >= 9),
        'gate>=1 only': (X[:, 0] >= 1),
        'always': np.ones(len(y), dtype=bool),
    }
    print(f"  {'rule':<24}{'acc':>6}{'prec':>7}{'rec':>6}")
    best = None
    for name, pred in rules.items():
        p = pred.astype(int)
        acc = float((p == y).mean())
        tp = float((p & y.astype(bool)).sum())
        prec = tp / p.sum() if p.sum() else 0.0
        rec = tp / y.sum() if y.sum() else 0.0
        print(f"  {name:<24}{acc:>6.2f}{prec:>7.2f}{rec:>6.2f}")
        if best is None or acc > best[1]:
            best = (name, acc)
    print(f"\n  BEST single rule: '{best[0]}' acc={best[1]:.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()