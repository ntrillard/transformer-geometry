#!/usr/bin/env python3
"""eval_anglesweep.py — DENSE per-prompt angle sweep to LOCATE the working
coherence angle (or band), then confirm with extra seeds.

theta_fine/transition only sampled {3,6,9,12} - the worst way to find a
threshold. Here: for EACH prompt, sweep on-arc angles finely and plot the
coherence-vs-angle curve (coherence count 0..2 per angle, 2 seeds), find
the best contiguous band, then re-verify that band center with 3 seeds.

coherence (practical, short window): plant AND rep4==0 AND maxrun<=2 AND
no <eos> AND no token>2x (over NTOK tokens). This is the planted-grammar
definition; dropping the over-tight div>=0.7.

Budget(20s): find the dense sweep width that fits. Angles step 2deg over
4..16 (7 angles), 2 seeds, NTOK=4, 2-3 prompts, then 3-seed confirm of
best.

One model, no template. Run: HF_TOKEN=<tok> timeout 18 python3 -u
eval_anglesweep.py
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
SEEDS = [0, 1, 2]
TARGET = 'city'
OUT = Path('../steering_geometry_results/anglesweep.csv')
ANGLES = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]   # dense on-arc sweep

PROMPTS = [
    ('ask',  'If you ask me which European city is the most beautiful, I would say that'),
    ('japan', 'The capital of Japan is'),
]
VERIFY_SEEDS = [0, 1, 2]


def rep4(toks):
    if len(toks) < 4:
        return 0.0   # <4 tokens -> no 4-gram -> no repetition
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

    def rot_to_angle(vv, tid, theta):
        a = math.radians(theta)
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(a) + g * math.sin(a)) * vv.norm()

    def coherent(toks, txt):
        """practical planted-grammar coherence (short window)."""
        if not any(t in toks for t in fset):
            return False
        if '<eos>' in txt:
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

    def count_coherent(vin, ids0, seeds):
        n = 0
        txts = []
        for sd in seeds:
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
            txts.append(tok.decode(toks))
            n += int(coherent(toks, tok.decode(toks)))
        return n, txts

    rows = []
    print(f"[{MODEL}] DENSE per-prompt angle sweep, {len(SEEDS)} seeds "
          f"(then 3-seed confirm)")
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        gap = float(Ln[int(Ln.argmax())] - Ln[fam].max())
        tgt = closest(vf)
        print(f"\n==== {pname}: {pr!r} gap={gap:.1f} -> "
              f"{tok.decode([tgt])!r} ====")
        curve = []
        for a in ANGLES:
            vin = rot_to_angle(vf, tgt, a)
            n, txts = count_coherent(vin, ids0, SEEDS)
            curve.append((a, n))
            rows.append(dict(prompt=pname, angle=a, sweep_seeds=''.join(
                'Y' if i < n else 'n' for i in range(len(SEEDS))),
                n_coherent=n, phase='sweep'))
            print("  %5.1f deg : %d/%d  %s" % (a, n, len(SEEDS),
                                               txts[0][:26].strip()),
                  flush=True)
        # find best contiguous band (>=2/2 i.e. highest count)
        best_a = max(ANGLES, key=lambda a: dict(curve)[a])
        best_n = dict(curve)[best_a]
        # contiguous band where count >= 1 (any coherence)
        band = [a for a, n in curve if n >= 1]
        print(f"  best angle={best_a} ({best_n}/{len(SEEDS)}); "
              f"band(any)={band}")
        if best_n >= 1:
            # verify best angle with 3 seeds
            vin = rot_to_angle(vf, tgt, best_a)
            n3, _ = count_coherent(vin, ids0, VERIFY_SEEDS)
            rows.append(dict(prompt=pname, angle=best_a,
                             sweep_seeds=''.join('Y' if i < n3 else 'n'
                                                 for i in range(3)),
                             n_coherent=n3, phase=f'verify@best({best_a})'))
            print(f"  VERIFY {best_a} deg w/ 3 seeds: {n3}/3")

    # meta-summary: per prompt, does a working angle exist?
    print("\n-- per-prompt working-angle summary --")
    for pname in [p for p, _ in PROMPTS]:
        sel = [r for r in rows if r['prompt'] == pname]
        ver = [r for r in sel if r['phase'].startswith('verify')]
        if ver:
            print(f"  {pname}: best-angle verify = {ver[-1]['n_coherent']}/3 "
                  f"at {ver[-1]['angle']} deg")
        else:
            print(f"  {pname}: no coherent angle found in sweep")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()