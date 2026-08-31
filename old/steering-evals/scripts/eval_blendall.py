#!/usr/bin/env python3
"""eval_blendall.py — run EVERY blend method (combining native v and
rotated v') at the readout, and meta-test whether coherence is a function
of the OPERATOR or only of the EFFECTIVE ANGLE.

BIG-LEAP question: eval_transition found coherence needs ~9 deg effective
angle on coherent-native prompts. Is that enough — i.e. does EVERY blend
operator that reaches ~9 deg give the same coherence (-> 'only direction
matters', operators are equivalent), or does some blend FAMILY genuinely
deviate (non-convex op like max, or multiplicative op, produces a state
the head treats differently) -> a real operator effect?

Operators: all standard means + elementwise merges + spherical:
  pure_rot, lin(0.5)=arith, lin(0.3), lin(0.7),
  slerp05, nlerp05,
  geom(arith-geom), harm, quad, contraharm, pow3,
  max, min, absmax (pick larger |.|), 
  score (coherence) per cell from a short multi-seed rollout.
Then META-FIT: corr(effective_angle -> coherence) across ALL cells; and
flag any operator whose coherence deviates from the angle-predicted
value (the operator-effect test).

One model, no template. Run: HF_TOKEN=<tok> timeout 18 python3 -u
eval_blendall.py
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
OUT = Path('../steering_geometry_results/blendall.csv')

PROMPTS = [
    ('ask', 'If you ask me which European city is the most beautiful, I would say that'),
    ('fr',  'The capital of France is'),
]


def blend(v, r, op):
    """combine native v and rotated r -> vector (normalized to ||v||)."""
    n = v.float(); r = r.float()
    nn = n / n.norm(); rn = r / r.norm(); nrm = n.norm()
    A = nn; B = rn
    if op == 'pure_rot':
        o = r
    elif op == 'lin03':
        o = 0.7 * n + 0.3 * r
    elif op == 'lin05':       # == arithmetic mean
        o = 0.5 * n + 0.5 * r
    elif op == 'lin07':
        o = 0.3 * n + 0.7 * r
    elif op == 'slerp05':
        coso = (A * B).sum().clamp(-1, 1); om = torch.acos(coso)
        so = torch.sin(om)
        o = (torch.sin(0.5 * om) / (so + 1e-12)) * n + \
            (torch.sin(0.5 * om) / (so + 1e-12)) * r
    elif op == 'nlerp05':
        o = (A + B) / 2
    elif op == 'geom':        # geometric mean, sign-preserving (unit-mag)
        mag = torch.sqrt(A.abs() * B.abs())
        sgn = torch.sign(A + B); sgn = sgn + (sgn == 0).float()
        o = sgn * mag
    elif op == 'harm':        # harmonic mean (unit-mag, guarded)
        den = A + B
        o = torch.where(den == 0, torch.zeros_like(A),
                        2 * A * B / den)
    elif op == 'quad':        # quadratic (RMS) mean
        o = torch.sqrt((A * A + B * B) / 2)
    elif op == 'contraharm':
        den = A + B
        o = torch.where(den == 0, torch.zeros_like(A),
                        (A * A + B * B) / den)
    elif op == 'pow3':        # power mean p=3
        o = torch.sign(A + B) * torch.pow(
            (torch.abs(A) ** 3 + torch.abs(B) ** 3) / 2, 1 / 3)
    elif op == 'max':
        o = torch.max(A, B)
    elif op == 'min':
        o = torch.min(A, B)
    elif op == 'absmax':
        o = torch.where(A.abs() >= B.abs(), A, B)
    else:
        raise ValueError(op)
    return o / (o.norm() + 1e-12) * nrm


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

    def ang(v1, v2):
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

    OPS = ['pure_rot', 'lin03', 'lin05', 'lin07', 'slerp05', 'nlerp05',
           'geom', 'harm', 'quad', 'contraharm', 'pow3', 'max', 'min',
           'absmax']
    rows = []
    print(f"[{MODEL}] ALL blend operators -> readout; seed-robust "
          f"coherence, {len(SEEDS)} seeds")
    print(f"  {'op':>11} | {'ask':>10} | {'fr':>10}")
    for op in OPS:
        per_prompt = {}
        for pname, pr in PROMPTS:
            ids0 = tok(pr, add_special_tokens=False,
                       return_tensors='pt').input_ids.to(DEV)
            vf, Ln = nat_vL(ids0)
            nat_top = int(Ln.argmax())
            gap = float(Ln[nat_top] - Ln[fam].max())
            alpha = 2 * (gap / 97.0) + 0.02
            tgt = closest(vf)
            vp = rot(vf, tgt, alpha)
            vin = blend(vf, vp, op)
            a_eff = ang(vf, vin)
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
            robust = int(sum(oks) >= 2)
            per_prompt[pname] = robust
            rows.append(dict(op=op, prompt=pname, angle_eff=round(a_eff, 2),
                             gap=round(gap, 3), robust=robust,
                             seeds=''.join('Y' if o else 'n' for o in oks)))
        print("  %-11s | %10d | %10d" % (op, per_prompt['ask'],
                                         per_prompt['fr']), flush=True)

    # META-FIT: does coherence track effective angle across ALL cells?
    print("\n-- operator-effect test: coherence vs effective angle --")
    arr = [(r['angle_eff'], r['robust'], r['prompt'], r['op'])
           for r in rows if r['prompt'] == 'ask']
    angles = np.array([a for a, *_ in arr], dtype=float)
    coh = np.array([c for _, c, *_ in arr], dtype=float)
    if angles.std() > 0 and coh.std() > 0:
        c = np.corrcoef(angles, coh)[0, 1]
        print(f"  ask: corr(angle_eff -> coherence) = {c:+.3f}")
        print(f"  angle_eff range {angles.min():.1f}..{angles.max():.1f}")
    else:
        print("  (ask cells: no variance in angle or coherence)")
    # deviation: any cell coherent at low angle or incoherent at high?
    hi = [x for x in arr if x[0] >= 9.0]
    lo = [x for x in arr if x[0] < 9.0]
    print(f"  at angle>=9: coherent {sum(c for _, c, *_ in hi)}/"
          f"{len(hi)} : [{', '.join(x[3] for x in hi)}]")
    print(f"  at angle<9 : coherent {sum(c for _, c, *_ in lo)}/"
          f"{len(lo)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()