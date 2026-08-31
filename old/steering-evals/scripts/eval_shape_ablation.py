#!/usr/bin/env python3
"""Shape ablation on REAL hidden states: why the sphere?

The paper leans on one algebraic identity (verified to 4e-17):
    (u + αg)/‖u + αg‖₂  ==  cos(δ)u + sin(δ)τ      (a great-circle rotation)
This is true for the ℓ2 ball AND ONLY the ℓ2 ball, because O(d) rotations
are isometries of exactly p=2.  For any other norm the operation is NOT a
rotation: the "arc" distorts, angle budgets stop meaning angles, and the
traversal is no longer length-preserving.

This script measures, on REAL Qwen2-0.5B hidden states at 4 depths (2 plain
prompts), with real LM-head rows as targets:

  A) Rotation-identity deviation:  ‖(u+t·g)/‖·‖_p − (cosδ·u + sinδ·τ)‖
       for p ∈ {2, ∞, 1} at the 17° and 45° budgets.  p=2 should be ~0;
       p=∞ / p=1 should be large (that is "the cube breaks the geometry").
  B) Target-logit monotonicity violation rate along the arc (p=2 ~0).
  C) Rank-1 reach@17 / @45 under each body (comparable to the paper numbers).
  D) Wrong-target specificity (~0 for every body).

Run: python eval_shape_ablation.py
"""
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as st_load

import steering_geometry_test as M

CACHE = Path.home() / '.cache/huggingface' / 'hub' / 'models--Qwen--Qwen2-0.5B-Instruct'


def load_head_rows(n_max=4000):
    snaps = sorted((CACHE / 'snapshots').glob('*'))
    st = sorted(snaps[0].glob('*.safetensors'))[0]
    d = st_load(str(st))
    key = [k for k in d if 'weight' in k and d[k].ndim == 2][0]
    W = d[key].float().numpy()[:n_max]
    return W


def pnorm1(x):
    return np.abs(x).sum()


def pnorminf(x):
    return np.abs(x).max()


def main():
    rng = np.random.default_rng(0)
    model, tok = M.load_model('Qwen/Qwen2-0.5B-Instruct', dtype='fp16')
    N = model.config.num_hidden_layers
    layers = sorted({int(round(f * (N - 1))) for f in (0.0, 0.33, 0.67, 0.99)})
    states = M.get_states(model, tok, ['The capital of France is',
                                       'Once upon a time'], layers)

    W = load_head_rows()
    d = W.shape[1]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)

    # 64 real target rows + full-vocab ranking
    tidx = rng.choice(len(Wn), size=64, replace=False)
    srows = Wn[tidx]
    Wf = np.vstack([Wn])  # full vocab for ranking

    budgets = {17: math.tan(math.radians(17)), 45: math.tan(math.radians(45))}

    def body_endpoint(uk, gn, t, p):
        """Endpoint after tangent t·gn renormalized by body p."""
        v_raw = uk + t * gn
        if p == 2:
            return v_raw / np.linalg.norm(v_raw)
        if p == np.inf:
            return v_raw / pnorminf(v_raw)
        return v_raw / pnorm1(v_raw)

    results = {p: {deg: dict(reach=0, wrong=0, viol=0, n=0, roterr=[]) for deg in budgets}
               for p in (2, np.inf, 1)}

    for ctx, llayer in states.items():
        for l, h in llayer.items():
            u = h / np.linalg.norm(h)
            for k, s in enumerate(srows):
                g = s - (s @ u) * u
                gl = np.linalg.norm(g)
                if gl < 1e-8:
                    continue
                gn = g / gl
                tau = (s - (s @ u) * u) / gl
                for deg, t in budgets.items():
                    delta = math.atan(gl * t)          # true p=2 arc angle
                    rot = math.cos(delta) * u + math.sin(delta) * tau
                    for p in (2, np.inf, 1):
                        v = body_endpoint(u, gn, t, p)
                        # A) rotation-identity deviation
                        err = float(np.linalg.norm(v - rot))
                        results[p][deg]['roterr'].append(err)
                        # C) reach
                        L = v @ Wf.T
                        own = float(v @ s)
                        rank = int((L > own).sum() + 1)
                        if rank == 1:
                            results[p][deg]['reach'] += 1
                        results[p][deg]['n'] += 1
                        # D) wrong-target: steer toward a DIFFERENT target, check s
                        k2 = (k + 7) % len(srows)
                        g2 = srows[k2] - (srows[k2] @ u) * u
                        g2l = np.linalg.norm(g2)
                        if g2l > 1e-8:
                            v2 = body_endpoint(u, g2 / g2l, t, p)
                            L2 = v2 @ Wf.T
                            own2 = float(v2 @ s)
                            if int((L2 > own2).sum() + 1) == 1:
                                results[p][deg]['wrong'] += 1
                        # B) monotonicity along 16 sub-steps
                        prev = float(u @ s)
                        for st_ in np.linspace(0.05, t, 16):
                            vv = body_endpoint(u, gn, st_, p)
                            cur = float(vv @ s)
                            if cur < prev - 1e-6:
                                results[p][deg]['viol'] += 1
                            prev = cur

    print(f"{'body':12s} {'deg':>3s} {'rot-err (med)':>13s} {'monotone-viol':>14s} "
          f"{'reach':>7s} {'wrong@tgt':>9s}  n")
    for p, name in ((2, 'sphere p=2'), (np.inf, 'cube   p=∞'), (1, 'diamond p=1')):
        for deg in (17, 45):
            r = results[p][deg]
            roterr = np.median(r['roterr'])
            print(f"{name:12s} {deg:3d} {roterr:13.3e} {r['viol'] / r['n']:13.2%} "
                  f"{r['reach'] / r['n']:6.1%} {r['wrong'] / r['n']:8.1%}  {r['n']}")


if __name__ == "__main__":
    main()