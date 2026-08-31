#!/usr/bin/env python3
"""eval_anisotropy_depths.py — BIG LEAP: does the current's anisotropy
GROW toward the readout?

Gemma-3-1B only, 1 base forward + 6 injection forwards, <=10s.

01989fa: at L10 the native azimuth transfers 1.15x better than random
(0.195 vs 0.169) while the readout spectral map is 2.3x native
(e7c3a7a). If anisotropy = the stack ACCUMULATING direction-preference,
the ratio gain(native)/gain(rand) should RISE with depth and approach
~2.3 at the readout. If it stays flat ~1.15, the readout operator is a
special one-off.

Gives the anisotropy PROFILE of the medium, depth-resolved.

Run: timeout 60 python3 -u eval_anisotropy_depths.py  # GEMMA-3-1B
"""
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
DEPTHS = [4, 17, 23]          # 0-based; + existing L10 / final points
LAM = 0.2


def tangent_unit(v, Wt):
    """unit tangent toward Wt stripped of the v component."""
    vn = v / (np.linalg.norm(v) + 1e-12)
    t = Wt - (vn @ Wt) * vn
    n = np.linalg.norm(t)
    return t / (n + 1e-12), n


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
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
    hooks.append(model.model.norm.register_forward_hook(mk('f')))
    with torch.no_grad():
        L0 = model(ids).logits[0, -1].float()
    for h in hooks:
        h.remove()
    vf = caps['f'].cpu().numpy().astype(np.float64)
    V = np.stack([caps[li].cpu().numpy().astype(np.float64)
                  for li in range(NL)] + [vf])
    native = int(L0.argmax())
    Wnat = lm_w[native].detach().float().cpu().numpy().astype(np.float64)

    rng = np.random.default_rng(3)
    print(f"[{MODEL}] {PROMPT!r} anisotropy profile "
          f"(gain ratio native/rand per depth)")
    print(f"  {'depth':>6} {'gain_nat':>9} {'gain_rand':>9} "
          f"{'ratio':>7} {'norm':>9}")
    rows = []
    hfn = model.model.norm.register_forward_hook(
        lambda m, i, o: capf.__setitem__('v', o[0, -1, :].float()))
    capf = {}
    for d in DEPTHS:
        v_d = V[d]
        n_d = float(np.linalg.norm(v_d))
        u_n, _ = tangent_unit(v_d, Wnat)
        r = rng.standard_normal(v_d.shape[0])
        r -= (v_d @ r) / np.dot(v_d, v_d) * v_d
        r /= np.linalg.norm(r)
        gains = {}
        for name, u in (('nat', u_n), ('rand', r)):
            pert = (v_d + LAM * n_d * u).astype(np.float32)

            def inj(m, i, o, p=pert):
                out = o.clone()
                out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                device=out.device)
                return out

            hf = model.model.layers[d].register_forward_hook(inj)
            try:
                with torch.no_grad():
                    model(ids)
            finally:
                hf.remove()
            Delt = capf['v'].cpu().numpy().astype(np.float64) - vf
            gain = float(np.linalg.norm(Delt) / (LAM * n_d))
            gains[name] = gain
        ratio = gains['nat'] / (gains['rand'] + 1e-12)
        rows.append((d + 1, gains['nat'], gains['rand'], ratio, n_d))
        print(f"  L{d + 1:>3} {gains['nat']:>9.3f} {gains['rand']:>9.3f} "
              f"{ratio:>7.2f} {n_d:>9.1f}", flush=True)
    hfn.remove()

    # known points from earlier probes
    rows.append(('L10', 0.195, 0.169, 1.15, 3761.8))
    print(f"  L10 *  {0.195:>9.3f} {0.169:>9.3f} {1.15:>7.2f} {3761.8:>9.1f}")
    rows.append(('final', 2.19, 0.94, 2.33, 99.9))
    print(f"  fin *  {2.19:>9.3f} {0.94:>9.3f} {2.33:>7.2f} {99.9:>9.1f}")
    ratios = [r[3] for r in rows if isinstance(r[3], float)]
    print(f"\n  ANISOTROPY PROFILE ratio = {[f'{r:.2f}' for r in ratios]}")
    cc = float(np.corrcoef(np.arange(len(ratios)), ratios)[0, 1])
    print(f"  corr(index, ratio) = {cc:+.3f} "
          f"(rising = the stack accumulates direction-preference)")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()