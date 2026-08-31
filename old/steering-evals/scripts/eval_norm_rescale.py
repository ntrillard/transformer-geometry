#!/usr/bin/env python3
"""eval_norm_rescale.py — PROVE the norm-ratio mechanism (5664bcb).

Steer at L10 with hidden-state norm RESCALED to a target scale before the
readout. If scramble collapses to ~1x at readout scale regardless of
depth, the mechanism is causal (norm-ratio determines transfer), not just
correlated.

Design (fast, Gemma-3-1B):
  v10 = L10 state (norm ~3762), vf = final state (norm ~100).
  For scale s in {v10-native (2.4x final?), final, 1/10 final, 1/100 final}:
    rotate v10 by alpha toward chicken with cos/sin,
    re-scale to ||v_final|| * s (same DIRECTION as final),
    then hook model.model.norm OUTPUT and inject the rescaled state
    as the actual final residual (post-norm override at readout scale).
  Measure: target rank + gap vs alpha -> behavioral slope -> stretch.
At s=1 (final norm) the L10 rotation should behave like a final push
(scramble ~1x if mechanism holds). At s=1/100 it should vanish (~0 plant).

Run: timeout 60 python3 -u eval_norm_rescale.py   # GEMMA-3-1B ONLY
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
TARGET = 'chicken'
D = 9  # L10 (0-based)
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.45]


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight

    tid = tok(' ' + TARGET, add_special_tokens=False).input_ids
    if len(tid) != 1:
        print("target not single token; abort")
        return
    tid = int(tid[0])
    capid = tok(' ' + TARGET.capitalize(), add_special_tokens=False).input_ids
    caplist = [int(c) for c in capid] if capid else [tid]
    Wt = lm_w[tid].detach().float().cpu().numpy()

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    with torch.no_grad():
        L0 = model(ids).logits[0, -1].float()
    native = int(L0.argmax())
    Wn = lm_w[native].detach().float().cpu().numpy()

    # capture L10 and final states
    caps = {}

    def mk(li):
        def h(m, i, o):
            caps[li] = o[0, -1, :].float()
        return h

    h10 = model.model.layers[D].register_forward_hook(mk('v10'))
    hf = model.model.norm.register_forward_hook(mk('vf'))
    with torch.no_grad():
        model(ids)
    h10.remove()
    hf.remove()
    v10 = caps['v10'].cpu().numpy()
    vf = caps['vf'].cpu().numpy()
    nrm10, nrmf = np.linalg.norm(v10), np.linalg.norm(vf)
    print(f"[{MODEL}] native={native!r}  L10||v||={nrm10:.1f}  "
          f"final||v||={nrmf:.1f}  ratio={nrm10 / nrmf:.1f}")

    # target-scale test: replace the FINAL residual (post-norm) with a
    # rotation of an early state, then scale to various fractions of nrmf.
    # hook model.model.norm output -> this REPLACES whatever the head reads.
    scales = [('s=final', 1.0), ('s=1/10', 0.1), ('s=1/100', 0.01),
              ('s=L10-native', nrm10 / nrmf)]
    print(f"  {'scale':>12} {'alpha':>5} {'rank_t':>6} {'gap':>7}", flush=True)
    seen = []
    for sname, sf in scales:
        for alpha in ALPHAS:
            # build the rotated+scaled forced residual
            v = v10
            vn = v / np.linalg.norm(v)
            tau = Wt - (vn @ Wt) * vn
            g = tau / (np.linalg.norm(tau) + 1e-12)
            # rotated unit
            u = vn * math.cos(alpha) + g * math.sin(alpha)
            forced = (u * (nrmf * sf)).astype(np.float32)

            def inject(m, i, o, f=forced):
                out = o.clone()
                out[0, -1, :] = torch.as_tensor(f, dtype=out.dtype,
                                                device=out.device)
                return out

            hf2 = model.model.norm.register_forward_hook(inject)
            try:
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
            finally:
                hf2.remove()
            rank = int(torch.topk(L, 30).indices.tolist().count(tid))  # 1 if in top30
            gap = float(L[tid] - L[native])
            if sname not in [k[0] for k in seen] or alpha == 0.2:
                print(f"  {sname:>12} {alpha:>5} {rank:>6} {gap:>+7.2f}",
                      flush=True)
            seen.append((sname, alpha, rank, gap))
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()