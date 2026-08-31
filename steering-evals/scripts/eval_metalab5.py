#!/usr/bin/env python3
"""eval_metalab5.py — HELD-OUT generalization of the anti-last controller.

Recipe to validate (fit on 12 prompts in metalab2-4):
  gate  : steer ONLY if gap < 13 AND f_rank < 640
  steer : one 10deg rotation on-arc toward closest family member at step 0,
          then suppress that token at lm_head for steps 1..2 (window 2),
          3 seeds, 5-token free-run; coherent = >=2/3 seeds (fixed metric).
THIS RUN: a NEW prompt batch (unseen). For each prompt compute gap+rank,
apply the GATE decision, and actually run the steer to verify:
  - in-zone  (gate says steer) -> predict coherent, check
  - out-zone (gate says skip)  -> predict collapse, steer anyway to confirm
    (so we can tally gate correctness on unseen data).

One model, no template. Run: HF_TOKEN=<tok> timeout 18 python3 -u
eval_metalab5.py
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
OUT = Path('../steering_geometry_results/metalab5.csv')
ANGLE = 10.0
WIN = 2
GAP_MAX = 13.0
RANK_MAX = 640.0

PROMPTS = [   # unseen prompts for held-out validation
    ('itcap', 'The capital of Italy is'),
    ('decap', 'The capital of Germany is'),
    ('mona',  'The Mona Lisa is displayed in'),
    ('movie', 'The movie was filmed in'),
    ('holiday','My dream vacation destination is'),
    ('univ',  'The oldest university is in'),
    ('novel', 'The story takes place in'),
    ('hockey','The hockey championship was held in'),
]


def rep4(toks):
    if len(toks) < 4:
        return 0.0
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
        try:
            with torch.no_grad():
                L = model(ids).logits[0, -1].float()
        finally:
            hk.remove()
        return cv['v'], L

    def sample(L):
        p = torch.softmax(L.float(), 0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk) / (q * msk).sum()
        return int(torch.multinomial(qq, 1))

    def coherent(toks, txt):
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

    def shot_anti(ids0, vp, tgt, sd, window):
        torch.manual_seed(sd)
        ids = ids0.clone()
        toks = []
        for step in range(NTOK):
            hs = []
            try:
                if step == 0:
                    def inj(m, i, o, p=vp):
                        out = o.clone()
                        out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                        device=out.device)
                        return out
                    hs.append(model.model.norm.register_forward_hook(inj))
                if 1 <= step <= window:
                    def anti(m, i, o, tid=tgt):
                        out = o.clone()
                        out[0, -1, tid] = -30.0
                        return out
                    hs.append(model.lm_head.register_forward_hook(anti))
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
            finally:
                for h in hs:
                    h.remove()
            nxt = sample(L)
            toks.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        return toks

    rows = []
    print(f"[{MODEL}] HELD-OUT validation of anti-last controller "
          f"(gate gap<{GAP_MAX} & rank<{RANK_MAX}, 10deg, win{WIN})")
    print("  %-8s %5s %6s %5s | %5s %5s | %s"
          % ('prompt', 'gap', 'rank', 'inZ', 'pred', 'coh', 'seeds'))
    n_in, n_in_ok = 0, 0
    n_out, n_out_ok = 0, 0
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        order = Ln.argsort(descending=True).tolist()
        f_rank = order.index(fam[int(Ln[fam].argmax())]) + 1
        tgt = closest(vf)
        in_zone = (gap < GAP_MAX) and (f_rank < RANK_MAX)
        # ALWAYS run the steer to verify the gate decision
        vp = rot_to_angle(vf, tgt, ANGLE)
        oks = []
        seed_str = ''
        for sd in SEEDS:
            toks = shot_anti(ids0, vp, tgt, sd, WIN)
            c = coherent(toks, tok.decode(toks))
            seed_str += 'Y' if c else 'n'
            oks.append(c)
        ncoh = sum(oks)
        coh = 1 if ncoh >= 2 else 0
        rows.append(dict(prompt=pname, gap=round(gap, 2), f_rank=f_rank,
                         in_zone=int(in_zone), predicted=int(in_zone),
                         coherent=coh, seeds=seed_str,
                         n_coherent=ncoh, text=tok.decode(toks)))
        print("  %-8s %5.1f %6d %5d | %5d %5d | %s"
              % (pname[:8], gap, f_rank, int(in_zone), int(in_zone),
                 coh, seed_str), flush=True)
        if in_zone:
            n_in += 1
            n_in_ok += coh
        else:
            n_out += 1
            n_out_ok += coh
    # also tally native coherence (the 'skip' baseline for out-zone)
    print(f"\n  in-zone  (gate: steer):  {n_in_ok}/{n_in} coherent "
          f"(predicted all)  -> gate precision")
    print(f"  out-zone (gate: skip) :  {n_out_ok}/{n_out} would be "
          f"coherent if steered (should be ~0 to justify skip)")
    print(f"  held-out gate correctness = "
          f"{n_in_ok + (n_out - n_out_ok)}/{len(PROMPTS)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()