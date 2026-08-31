#!/usr/bin/env python3
"""eval_azimuth_transfer.py — BIG LEAP: is the mid-stack current a SCALAR,
or does injected-direction AZIMUTH matter?

Gemma-3-1B only, 1 base forward + 6 azimuth-injection forwards, <=10s.

The plunge/current claim (f16c843, confirmed 1abda0f) is that mid-stack
steering dies from a NORM-ratio effect - the current scales all injected
directions equally (scalar). If TRUE, the surviving fraction of an
injected perturbation is the SAME for every azimuth around the state's
direction: the medium is isotropic and there is no preferred azimuth.

If FALSE (anisotropic), some injection azimuths survive the plunge much
better than others -> a PREFERRED STEERING AZIMUTH exists (free gain),
and the wobble/plane structure becomes an exploitable resource.

Design (at L10, the sweet spot; injection size fixed 0.2*||v10||):
  per azimuth u (unit, tangent: u.v10=0):
    inject v10' = v10 + 0.2||v10|| u, run to final, Delta = vf' - vf
    gain(az) = ||Delta|| / (0.2||v10||)   (the current's transfer)
    dir(az)  = |Delta.u| / ||Delta||      (arrival along own azimuth)
  6 azimuths with distinct geometry:
    chicken, paris, native tangents (token-steer directions)
    sink-component (the plunge axis tangent)
    2 random tangents (baseline)
  ISOTROPY: std(gain)/mean(gain); corr(gain, u.sink_hat) etc.

Run: timeout 60 python3 -u eval_azimuth_transfer.py  # GEMMA-3-1B
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
D = 9  # L10 (0-based)
LAM = 0.2


def tangent_unit(v, Wt):
    """unit tangent toward token row Wt, stripped of v component."""
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
    v_d = V[D]
    n_d = float(np.linalg.norm(v_d))
    sink = V[22] - V[18]
    sink_h = sink / (np.linalg.norm(sink) + 1e-12)

    native = int(L0.argmax())
    Wnat = lm_w[native].detach().float().cpu().numpy().astype(np.float64)
    Wc = lm_w[tok(' chicken', add_special_tokens=False).input_ids[0]] \
        .detach().float().cpu().numpy().astype(np.float64)
    Wp = lm_w[tok(' paris', add_special_tokens=False).input_ids[0]] \
        .detach().float().cpu().numpy().astype(np.float64)

    # build the 6 tangent azimuths
    azs = {}
    for name, Wt in (('chicken', Wc), ('paris', Wp), ('native', Wnat)):
        u, _ = tangent_unit(v_d, Wt)
        azs[name] = u
    # sink-component tangent (orthogonalize sink against v_d)
    u_sink, _ = tangent_unit(v_d, sink)
    azs['sink'] = u_sink
    # 2 random tangents (orthogonalize random gaussians against v_d)
    rng = np.random.default_rng(3)
    r1 = rng.standard_normal(v_d.shape[0])
    r1 -= (v_d @ r1) / np.dot(v_d, v_d) * v_d
    r1 /= np.linalg.norm(r1)
    r2 = rng.standard_normal(v_d.shape[0])
    r2 -= (v_d @ r2) / np.dot(v_d, v_d) * v_d
    r2 -= (r1 @ r2) * r1
    r2 /= np.linalg.norm(r2)
    azs['rand1'] = r1
    azs['rand2'] = r2

    print(f"[{MODEL}] {PROMPT!r} L{D + 1} azimuth transfer, "
          f"inject={LAM}*||v||={LAM * n_d:.0f}")
    print(f"  {'azimuth':>9} {'gain':>7} {'dir':>6} "
          f"{'in-plane':>9} {'sink·u':>7}")
    rows = []
    for name, u in azs.items():
        pert = (v_d + LAM * n_d * u).astype(np.float32)

        def inj(m, i, o, p=pert):
            out = o.clone()
            out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                            device=out.device)
            return out

        hf = model.model.layers[D].register_forward_hook(inj)
        cf = {}

        def c(m, i, o):
            cf['v'] = o[0, -1, :].float()

        hfn = model.model.norm.register_forward_hook(c)
        try:
            with torch.no_grad():
                model(ids)
        finally:
            hf.remove()
            hfn.remove()
        vfp = cf['v'].cpu().numpy().astype(np.float64)
        Delta = vfp - vf
        gain = float(np.linalg.norm(Delta) / (LAM * n_d))
        dirr = float(abs(Delta @ u) / (np.linalg.norm(Delta) + 1e-12))
        # in-plane fraction via trajectory PCA-3
        Vc = V - V.mean(0)
        _, _, vt3 = np.linalg.svd(Vc, full_matrices=False)
        inplane = float(np.linalg.norm(vt3[:3] @ u))
        su = float(u @ sink_h)
        rows.append((name, gain, dirr, inplane, su))
        print(f"  {name:>9} {gain:>7.3f} {dirr:>6.3f} {inplane:>9.3f} "
              f"{su:>+7.3f}", flush=True)

    gains = np.array([r[1] for r in rows])
    print(f"\n  ISOTROPY: std(gain)/mean(gain) = "
          f"{gains.std() / gains.mean():.3f}   "
          f"(<0.1 = scalar current, isotropic)")
    sins = np.array([r[4] for r in rows])
    print(f"  corr(gain, sink·u) = "
          f"{float(np.corrcoef(gains, sins)[0, 1]):+.3f}  "
          f"(sink-aligned azimuths survive more?)")
    inps = np.array([r[3] for r in rows])
    print(f"  corr(gain, in-plane frac) = "
          f"{float(np.corrcoef(gains, inps)[0, 1]):+.3f}")
    nat = next(r for r in rows if r[0] == 'native')
    print(f"\n  norm-ratio prediction (scalar): 1/(||v10||/||vf||) = "
          f"{1 / (n_d / np.linalg.norm(vf)):.4f} vs native gain "
          f"{nat[1]:.3f}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()