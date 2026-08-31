#!/usr/bin/env python3
"""eval_transition.py — META-INFER the coherent-plant transition angle.

blendpct found a sharp phase transition on 'ask': ~6 deg effective angle
crosses from native (no topic) to ROBUST coherent+topical, while 'fr'
has NO plateau at any angle. The big leap: is the transition angle a
predictable INVARIANT (or a clean function of a cheap feature like gap),
so we get a USABLE SETPOINT for coherent steering -- or is it
prompt-specific?

Per prompt:
  - capture native v, cast law-rotated v', gap, and a cheap native
    coherence feature (native div on a short free-run).
  - probe a few effective angles via %-blend at the readout (head only),
    label each = seed-robust strict coherence (>=2/3 seeds).
  - find the MINIMAL effective angle that achieves robust coherence
    (theta*), or None if no plateau.
Then META-FIT theta* against features (gap, native div, cosang, ctxlen).
Big-leap question: theta* ~ constant OR gap/feature-predictable?

Budget(20s): one model, no template, head-only combination.
Run: HF_TOKEN=<tok> timeout 18 python3 -u eval_transition.py
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
OUT = Path('../steering_geometry_results/transition.csv')
ANGLES = [3.0, 6.0, 9.0, 12.0]   # effective-angle probes (deg)

PROMPTS = [
    ('ask',  'If you ask me which European city is the most beautiful, I would say that'),
    ('france','The capital of France is'),
    ('japan', 'The capital of Japan is'),
    ('olymp', 'The Olympic Games were held in'),
    ('travel','I would love to travel to'),
    ('visit', 'People from all over the world visit'),
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

    def robust_at(vin, ids0):
        """seed-robust coherence (>=2/3) for a given readout vector."""
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
    print(f"[{MODEL}] transition-angle meta-inference, {len(SEEDS)} seeds")
    print("  %-8s %5s %5s %5s | " % ('prompt', 'gap', 'ndiv', 'cos') +
          " ".join(f"{a:>4}" for a in ANGLES) + "   theta*")
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        cosang = float((vf / vf.norm()) @ Wn[closest(vf)])
        # native div (cheap coherence feature)
        oks_nat = []
        for sd in SEEDS[:2]:
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for _ in range(NTOK):
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
                nxt = sample(L)
                toks.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)],
                                dim=1)
            oks_nat.append(strict_ok(toks, tok.decode(toks)))
        ndiv = int(sum(oks_nat) >= 2)

        rob = []
        for a in ANGLES:
            v_ang = rot_to_angle(vf, closest(vf), a)
            r = robust_at(v_ang, ids0)
            rob.append(r)
        # minimal angle achieving robust coherence, else None
        theta = None
        for a, r in zip(ANGLES, rob):
            if r:
                theta = a
                break
        rows.append(dict(prompt=pname, gap=round(gap, 3),
                         cosang=round(cosang, 3), native_div=ndiv,
                         r3=rob[0], r6=rob[1], r9=rob[2], r12=rob[3],
                         theta_star=(theta if theta is not None else 'None')))
        print("  %-8s %5.1f %5d %5.2f | %s   %s"
              % (pname[:8], gap, ndiv, cosang,
                 " ".join(f"{r:>4}" for r in rob),
                 theta if theta is not None else '  None'), flush=True)

    # META-FIT: theta* vs features
    data = [r for r in rows if r['theta_star'] != 'None']
    print(f"\n-- prompts with a coherent plateau: {len(data)}/{len(rows)}")
    if data:
        xs = np.array([[r['gap'], r['cosang'], r['native_div']] for r in data],
                      dtype=float)
        ys = np.array([r['theta_star'] for r in data], dtype=float)
        print(f"  theta* values: {list(ys)}")
        for j, nm in enumerate(['gap', 'cosang', 'native_div']):
            if xs[:, j].std() > 0 and ys.std() > 0:
                c = np.corrcoef(xs[:, j], ys)[0, 1]
                print(f"  theta* corr with {nm}: {c:+.3f}")
        print(f"  theta* mean={ys.mean():.1f}std={ys.std():.1f}  "
              f"-> {'~CONSTANT invariant' if ys.std() < 1.5 else 'not constant'}")
    else:
        print("  (no plateau found in any prompt)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()