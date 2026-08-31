#!/usr/bin/env python3
"""eval_adaptive_theta.py — ADAPT the steer angle per-prompt from the
original final state, then build blends matching that angle.

IDEA: instead of a fixed 9deg (which theta_fine showed is a fragile
1/3-seed flicker), infer the CORRECT angle per prompt from the ORIGINAL
final state, and construct blends that land EXACTLY at that angle.

What is a defensible 'correct angle from the original final state'?
  theta_nat = angle(v_final, Wn[closest_family])   <-- the NATIVE pull
  distance to the family direction. On a prompt already near its family
  (small angle) the model is close to emitting it; on a far prompt it is
  far. The idea: steer so the post-steer state reaches a consistent
  *target* alignment to the family. Simplest: theta* = theta_nat (use the
  state's own geometry as the setpoint) and also test a scaled version
  (theta_nat, 2x theta_nat) vs the fixed 9deg baseline.

Then construct blends that MATCH the chosen angle:
  - on-arc rotation by theta  (exact match by construction)
  - linear blend that yields the same effective angle (filter: pick the
    lam whose effective angle == target within 0.3deg)
  - slerp blend at the matching fraction

Test per prompt: seed-robust coherence (>=2/3) and 1/3-seed count at
[theta_nat, 2*theta_nat, 9.0] x [rot, lin-match, slerp-match].
Big-leap question: does the prompt-adaptive theta beat the fixed 9deg?

One model, no template. Run: HF_TOKEN=<tok> timeout 18 python3 -u
eval_adaptive_theta.py
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
OUT = Path('../steering_geometry_results/adaptive_theta.csv')
ANGLES = ['nat', '2nat', 'fix9']


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

    def rot_to_angle(vv, tid, theta):
        a = math.radians(theta)
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(a) + g * math.sin(a)) * vv.norm()

    def lin_match(vv, r20, theta_target, vf_norm):
        """find lam so effective angle of lin blend == theta_target."""
        best = None; best_err = 1e9
        for lam in np.arange(0.0, 1.001, 0.005):
            c = (lam * vv + (1 - lam) * r20)
            c = c / c.norm() * vf_norm
            a = ang(vv, c)
            err = abs(a - theta_target)
            if err < best_err:
                best_err = err; best = (lam, c)
        return best

    def slerp_match(vv, r20, theta_target, vf_norm):
        rn = r20 / r20.norm(); v1 = vv / vv.norm()
        coso = (v1 * rn).sum().clamp(-1, 1); om = torch.acos(coso)
        so = math.sin(om)
        best = None; best_err = 1e9
        for t in np.arange(0.0, 1.001, 0.005):
            c = (math.sin((1 - t) * om) / so) * vv + \
                (math.sin(t * om) / so) * r20
            c = c / c.norm() * vf_norm
            a = ang(vv, c)
            err = abs(a - theta_target)
            if err < best_err:
                best_err = err; best = (t, c)
        return best

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

    rows = []
    print(f"[{MODEL}] per-prompt ADAPTIVE theta, blends matching the "
          f"angle, {len(SEEDS)} seeds")
    print(f"  {'prompt':<7}{'nat':>5}{'ang':>5} | {'mode':<14}{'angE':>5}"
          f"{'rob':>4} {'seeds':>6}  text")
    for pname, pr in PROMPTS_POOL:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        tgt = closest(vf)
        nrm = vf.norm()
        # the per-prompt angle from the original final state
        theta_nat = ang(vf, Wn[tgt].float())
        r20 = rot_to_angle(vf, tgt, 20.0)   # reference rotated for blending
        # per-prompt correct angle candidates
        theta_opts = {'nat': theta_nat, '2nat': 2 * theta_nat,
                      'fix9': 9.0}
        print(f"\n==== {pname}: {pr!r} gap={gap:.1f} -> "
              f"{tok.decode([tgt])!r}  theta_nat={theta_nat:.1f} ====")
        for aname in ANGLES:
            th = theta_opts[aname]
            # build 3 blends matching th
            blends = {}
            blends['rot'] = (f'rot', rot_to_angle(vf, tgt, th))
            lam, c = lin_match(vf, r20, th, nrm)
            blends['lin'] = (f'lin{lam:.2f}', c)
            t, c = slerp_match(vf, r20, th, nrm)
            blends['slerp'] = (f'slerp{t:.2f}', c)
            for bname, (blab, vin) in blends.items():
                a_eff = ang(vf, vin)
                rob, nco, seeds = robust(vin, ids0)
                rows.append(dict(prompt=pname, theta_name=aname,
                                 theta_target=round(th, 2), blend=bname,
                                 blend_arg=blab, angle_eff=round(a_eff, 2),
                                 robust=rob, n_coherent=nco, seeds=seeds))
                print("  %-7s %5.1f %5.1f | %-14s %5.1f %4d %s"
                      % (pname[:7], theta_nat, th, blab, a_eff, rob,
                         seeds), flush=True)

    # meta-summary: adaptive vs fixed-9
    print("\n-- adaptive vs fixed 9 deg (robust coherence) --")
    for aname in ANGLES:
        sel = [r for r in rows if r['theta_name'] == aname]
        nr = sum(r['robust'] for r in sel)
        tot = len(sel)
        print(f"  {aname:>6} (targets): robust {nr}/{tot}  "
              f"({[r['prompt'] for r in sel if r['robust']]})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


# prompt pool (include ask + several, to test adaptation)
PROMPTS_POOL = [
    ('ask',  'If you ask me which European city is the most beautiful, I would say that'),
    ('japan', 'The capital of Japan is'),
    ('eif',  'The Eiffel Tower is located in'),
]


if __name__ == "__main__":
    main()