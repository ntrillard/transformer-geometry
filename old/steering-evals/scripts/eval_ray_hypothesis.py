#!/usr/bin/env python3
"""eval_ray_hypothesis.py — BIG LEAP: is the residual flow a 1D RAY?

Gemma-3-1B only, 1 forward + numpy, ~4s.

The angle-to-sink locks (89 -> 3.6 deg by L2-3) suggests the mid-stack
flow is a 1D RAY: after layer ~3, v_l ~= ||v_l|| * u_ray with u_ray fixed.
This is the MAXIMAL collapse statement: 26 layers of 'computation' = a
scalar gain on one direction + tiny orthogonal wobble.

Tests (one capture, all numpy):
  1. RAY ERROR:   err(l) = ||v_l - ||v_l|| * u_ray|| / ||v_l||,
                  u_ray = normalized v_3. Tiny err = the state IS the ray.
  2. WHERE IT BREAKS: err climbs at the readout (the final rotation off
     the ray) -> plot the err curve; the break-layer is the decision.
  3. WOBBLE DIM: PCA of the off-ray wobble {w_l} (mid layers only) ->
     how many dims of real structure exist below the readout? 1 = even
     the wobble is 1D. High = real multi-dim computation.
  4. DECISION CARRY: corr(logits(ray_proj(vf)), logits(vf)) - does the
     readout state's decision come from its ray-component or its off-ray
     part? (plane-decision analog at 1D)
  5. TANGENT GROWTH: cos(v_{l+1}-v_l, v_l) per layer -> ~1 = pure radial
     growth (scalar amp); <1 = the flow also turns.

Run: timeout 60 python3 -u eval_ray_hypothesis.py  # GEMMA-3-1B
"""
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    bias = lm_head_bias = None
    NL = model.config.num_hidden_layers

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    caps = {}

    def mk(li):
        def h(m, i, o):
            caps[li] = o[0, -1, :].float()
        return h

    hooks = [model.model.layers[li].register_forward_hook(mk(li))
             for li in range(NL)]
    hooks.append(model.model.norm.register_forward_hook(mk(NL)))
    with torch.no_grad():
        L0 = model(ids).logits[0, -1].float()
    for h in hooks:
        h.remove()
    native = int(L0.argmax())
    W = lm_w.detach()

    V = np.stack([caps[li].cpu().numpy() for li in range(NL + 1)])  # (27, d)
    vf = V[-1]
    u_ray = V[3] / (np.linalg.norm(V[3]) + 1e-12)

    # ---- 1+2: ray error per layer ----
    nrm = np.linalg.norm(V, axis=1)
    errs = np.linalg.norm(V - nrm[:, None] * u_ray, axis=1) / (nrm + 1e-12)
    print(f"[{MODEL}] {PROMPT!r} native={tok.decode([native])!r}")
    print(f"  RAY u_ray = v_3 dir.  err(l) = ||v_l - ||v_l|| u_ray||/||v_l||")
    print(f"  {'layer':>6} {'norm':>9} {'ray_err':>8} {'tangent_cos':>12}")
    for li in range(NL + 1):
        v = V[li]
        tcos = float(np.dot(V[li + 1] - v, v) /
                     (np.linalg.norm(V[li + 1] - v) * np.linalg.norm(v) + 1e-12)) \
            if li < NL else float('nan')
        print(f"  L{li:>3}   {nrm[li]:>9.1f} {errs[li]:>8.4f} {tcos:>12.3f}",
              flush=True)

    # ---- 3: wobble PCA (off-ray residual, mid layers 4..25) ----
    Wl = V[4:26] - nrm[4:26, None] * u_ray
    Wc = Wl - Wl.mean(0)
    sv = np.linalg.svd(Wc, compute_uv=False)
    pr = float((sv.sum() ** 2) / ((sv ** 2).sum() + 1e-12))
    print(f"\n  WOBBLE PCA (off-ray, L4-25): top3 sv={np.round(sv[:3], 1)} "
          f"PR={pr:.2f}")

    # ---- 4: decision carry (ray vs off-ray at the final state) ----
    def logits(v):
        return torch.mv(W, torch.as_tensor(v, dtype=torch.float16,
                                           device=DEV)).float().cpu().numpy()

    vf_on = nrm[-1] * u_ray
    vf_off = vf - vf_on
    lf, lon, loff = logits(vf), logits(vf_on), logits(vf_off)
    co = np.corrcoef(lf, lon)[0, 1]
    cx = np.corrcoef(lf, loff)[0, 1]
    sam = int(lf.argmax()) == int(lon.argmax())
    print(f"\n  DECISION CARRY: corr(full, ray_proj)={co:+.3f}  "
          f"corr(full, off-ray)={cx:+.3f}  argmax_same={sam}")
    # final rotation: how far off-ray is vf?
    ang = float(np.arccos(np.clip(np.dot(vf, u_ray) / (nrm[-1] + 1e-12),
                                  -1, 1)) * 180 / np.pi)
    print(f"  final state angle off ray = {ang:.1f} deg "
          f"(|vf|= {nrm[-1]:.1f}, remaining-plunge {nrm[-1]*0:.0f})")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()