#!/usr/bin/env python3
"""eval_plane_compat.py — BIG LEAP #2: is steerability = plane-compatibility?

Follows eval_collapse_rank (state trajectory PR=2.95: 27 states span ~3
dims). If the residual flow lives on a low-dim PLANE, then:
  - target head-rows W_t lying IN the plane = easy to steer (the readout
    reads the plane position directly)
  - off-plane targets = the rotation must leave the plane -> layers
    re-absorb it (scramble), only the readout sees the off-plane part.
Test: capture V (one forward), PCA basis of the 27 states, project each
W_t -> in_plane fraction. Correlate in_plane_frac with alpha*@final
(computed analytically from the final state). If corr high -> the
collapse plane IS the steering manifold; steerability = plane fraction.

Run: timeout 60 python3 -u eval_plane_compat.py  # GEMMA-3-1B
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
TARGETS = ('chicken pizza pasta bread steak paris tokyo london cheese '
           'butter soup salad beef cake rome ocean river apple milk wine').split()


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    NL = model.config.num_hidden_layers

    tid_l = {}
    for w in TARGETS:
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            tid_l[w] = int(ids[0])
    tids = list(tid_l.values())
    tnames = list(tid_l.keys())
    Wt = lm_w[tids].detach().float().cpu().numpy().astype(np.float32)
    T = len(tids)

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    caps = {}

    def mk(li):
        def h(m, i, o):
            caps[li] = o[0, -1, :].float()
        return h

    hooks = [model.model.layers[li].register_forward_hook(mk(li))
             for li in range(NL)]
    hooks.append(model.model.norm.register_forward_hook(mk('f')))
    with torch.no_grad():
        L0 = model(ids).logits[0, -1].float()
    for h in hooks:
        h.remove()
    native = int(L0.argmax())
    Wn = lm_w[native].detach().float().cpu().numpy().astype(np.float32)

    V = np.stack([caps[li].cpu().numpy() for li in range(NL)] +
                 [caps['f'].cpu().numpy()])                # (27, d)

    # ---- PCA basis of the state trajectory ----
    Vc = V - V.mean(0)
    _, _, vt = np.linalg.svd(Vc, full_matrices=False)      # rows = principal axes
    basis = vt[:3]                                         # (3, d)
    print(f"[{MODEL}] {PROMPT!r} native={tok.decode([native])!r} "
          f"plane dims=3")
    vf = V[-1]
    vfn = vf / (np.linalg.norm(vf) + 1e-12)

    # ---- per-target: in-plane fraction + alpha*@final ----
    rows = []
    for t in range(T):
        w = Wt[t]
        proj = basis @ w                                   # (3,) comps
        in_frac = float(np.linalg.norm(proj) / (np.linalg.norm(w) + 1e-12))
        A = float(vfn @ (w - Wn))
        tau = w - (vfn @ w) * vfn
        B = float(tau @ (w - Wn)) / (np.linalg.norm(tau) + 1e-12)
        astar = math.atan2(-A, B)
        rows.append((tnames[t], in_frac, astar))
    # native too
    proj_n = basis @ Wn
    in_frac_n = float(np.linalg.norm(proj_n) /
                      (np.linalg.norm(Wn) + 1e-12))

    print(f"  native in-plane frac = {in_frac_n:.3f}")
    print(f"  {'target':>8} {'inplane':>8} {'a*@final':>9}", flush=True)
    for tn, ip, aa in rows:
        print(f"  {tn:>8} {ip:>8.3f} {aa:>+9.3f}", flush=True)

    ifps = np.array([r[1] for r in rows])
    aas = np.array([r[2] for r in rows])
    cc = np.corrcoef(ifps, aas)[0, 1]
    print(f"\n  corr(in_plane_frac, a*@final) = {cc:+.3f}")
    print(f"  corr(in_plane_frac, ABS(a*))    = "
          f"{np.corrcoef(ifps, np.abs(aas))[0, 1]:+.3f}")
    # food vs city in-plane
    foods = [i for i, tn in enumerate(tnames) if tn in
             ('chicken', 'pizza', 'pasta', 'bread', 'steak', 'cheese',
              'butter', 'soup', 'salad', 'beef', 'cake', 'apple', 'milk',
              'wine')]
    cits = [i for i, tn in enumerate(tnames) if tn in
            ('paris', 'tokyo', 'london', 'rome', 'ocean', 'river')]
    print(f"  in-plane: food={ifps[foods].mean():.3f} "
          f"city={ifps[cits].mean():.3f}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()