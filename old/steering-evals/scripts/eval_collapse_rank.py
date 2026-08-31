#!/usr/bin/env python3
"""eval_collapse_rank.py — BIG LEAP: the RANK of the information collapse.

One forward (<=10s, Gemma-3-1B). Captures all 27 states for one prompt,
21 targets. Asks THE collapse question in higher dimensions:

M[l, t] = <v_l, W_t> - <v_l, W_native>   (population matrix, 27 x T)
  - rank(M) ~ 1: the ENTIRE stack is a scalar gate — every target's
    population scales in lockstep; steerability = the readout scalar
    multiplier of the target's weight. Maximally collapsed.
  - rank(M) > 1: targets evolve through independent channels
    (food/city/syntax...) — a multidimensional collapse; WHICH channel
    survives is real structure.

Also: participation ratio (effective info channels), raw vs per-layer-
normalized SVD (is the collapse NORM-driven or DIRECTION-driven?),
gap-ratio anisotropy (pure scaling vs rotation), and PCA of the 27
states (how many dims does the trajectory itself occupy).

Run: timeout 90 python3 -u eval_collapse_rank.py  # GEMMA-3-1B
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
TARGETS = ('chicken pizza pasta bread steak paris tokyo london cheese '
           'butter soup salad beef cake rome ocean river apple milk wine').split()


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
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

    # ---- 1. population matrix M and its SVD ----
    lgt_ln = V @ Wn
    POP = (V @ Wt.T) - lgt_ln[:, None]                       # (27, T)
    u, s, vt = np.linalg.svd(POP, full_matrices=False)
    pr = float((s.sum() ** 2) / ((s ** 2).sum()))
    rank1 = int((s > 0.01 * s[0]).sum())
    print(f"[{MODEL}] {PROMPT!r} targets={T} layers={NL + 1} "
          f"native={tok.decode([native])!r}")
    print(f"  RAW POP[{NL + 1}x{T}] SVD: s = {np.round(s[:8], 1)} ...")
    print(f"    practical rank (>{0.01 * s[0]}) = {rank1} "
          f"  participation ratio = {pr:.2f}")

    # ---- 2. per-layer-normalized M (direction-only collapse) ----
    Vn = V / (np.linalg.norm(V, axis=1)[:, None] + 1e-12)
    Mn = (Vn @ Wt.T) - (Vn @ Wn)[:, None]
    un, sn, vtn = np.linalg.svd(Mn, full_matrices=False)
    prn = float((sn.sum() ** 2) / ((sn ** 2).sum()))
    rankn = int((sn > 0.01 * sn[0]).sum())
    print(f"  NORMED M SVD: s = {np.round(sn[:8], 2)} ...")
    print(f"    practical rank = {rankn}  participation ratio = {prn:.2f}")

    # ---- 3. gap-ratio anisotropy (chicken/paris across layers) ----
    gi, ci = tnames.index('chicken'), tnames.index('paris')
    gc, gcity = POP[:, gi], POP[:, ci]
    ratio = gc / (gcity + 1e-9)
    print(f"  gap(chicken)/gap(paris): std={ratio.std():.2f} "
          f"span=[{ratio.min():.1f},{ratio.max():.1f}]  "
          f"(0 std = pure scalar collapse)")

    # ---- 4. PCA of the 27 states (trajectory dim) ----
    Vc = V - V.mean(0)
    sv = np.linalg.svd(Vc, compute_uv=False)
    prv = float((sv.sum() ** 2) / ((sv ** 2).sum()))
    print(f"  STATE trajectory PCA: top3 sv={np.round(sv[:3], 1)}  "
          f"PR={prv:.2f}  (1-2 = plane; 27 = chaotic)")

    # ---- 5. the readout split: variance of survivor vs depth ----
    fd = POP[-1, :]
    print(f"  READOUT survivor: delta sd={fd.std():.1f} "
          f"span=[{fd.min():+.1f},{fd.max():+.1f}]")
    print(f"\n  CONCLUSION: collapse rank = {rank1} "
          f"({pr:.2f} effective channels)")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()