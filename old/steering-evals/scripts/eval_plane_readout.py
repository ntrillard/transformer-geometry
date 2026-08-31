#!/usr/bin/env python3
"""eval_plane_readout.py — BIG LEAP: where does the DECISION live, and
can we steer WITHOUT target rows (in-plane identity dial)?

Gemma-3-1B only, one capture + a few injected forwards, <=10s.

1. PLANE-DECISION FRACTION: split the final state vf = P_plane(vf) +
   off_plane(vf); compute logits_full = W_lm.vf, logits_plane =
   W_lm.P_plane(vf), logits_off = W_lm.off(vf). corr(full, plane) tells
   how much of the model's CHOICE is decided inside the universal 3D
   manifold. ~1.0 => the whole multi-head decision is 3D (maximal
   collapse of information into the plane).

2. SPECTRAL MAP of the readout: for the basis directions (sink_h,
   native_h2, third_h) and one OFF-plane direction, compute the gain
   g = ||W_lm . b|| / (||W_lm||_F/sqrt(d)). Non-uniform gains = the
   head's normalization law: which axes it boosts / attenuates.

3. TARGETLESS DIAL: rotate the final state IN the (sink, native) plane
   toward the native axis by theta (holding norm) -> does the native
   logit rise monotonically? If yes: steer confidence with NO target
   row - a pure plane-rotation primitive.

Run: timeout 60 python3 -u eval_plane_readout.py  # GEMMA-3-1B
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


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    lm_head = model.lm_head
    W = lm_head.weight
    bias = lm_head.bias
    NL = model.config.num_hidden_layers
    W = W.detach()  # keep on device

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
        L0_all = model(ids).logits[0, -1].float()
    for h in hooks:
        h.remove()
    L0 = L0_all.cpu().numpy()
    native = int(L0.argmax())
    vf = caps['f'].cpu().numpy()

    Wn = W[native].detach().float().cpu().numpy()

    V = np.stack([caps[li].cpu().numpy() for li in range(NL)] +
                 [caps['f'].cpu().numpy()])
    # universal plane basis: PCA-3 of the trajectory
    Vc = V - V.mean(0)
    _, _, vt = np.linalg.svd(Vc, full_matrices=False)
    B = vt[:3]  # (3, d) orthonormal
    P = B.T @ B  # (d, d) projector

    # plane split of the final state
    vf_p = P @ vf
    vf_o = vf - vf_p
    print(f"[{MODEL}] {PROMPT!r} native={tok.decode([native])!r}")
    print(f"  plane frac of vf norm = "
          f"{np.linalg.norm(vf_p) / (np.linalg.norm(vf) + 1e-12):.3f}")

    # ---- 1. plane-decision fraction ----
    def logits(v):
        vg = torch.as_tensor(v, dtype=torch.float16, device=DEV)
        lo = torch.mv(W, vg)
        return lo.float().cpu().numpy()

    lf = logits(vf)
    lp = logits(vf_p)
    lo_ = logits(vf_o)
    cp = np.corrcoef(lf, lp)[0, 1]
    co = np.corrcoef(lf, lo_)[0, 1]
    # rank agreement: argmax of full vs plane
    same_argmax = int(lf.argmax()) == int(lp.argmax())
    print(f"  PLANE-DECISION: corr(full, plane) = {cp:+.4f}  "
          f"corr(full, off) = {co:+.4f}  argmax_same = {same_argmax}")

    # ---- 2. spectral map of the readout over basis dirs ----
    sink = V[22] - V[18]
    sink_h = sink / (np.linalg.norm(sink) + 1e-12)
    nat_h = Wn / (np.linalg.norm(Wn) + 1e-12)
    nat_h2 = nat_h - (nat_h @ sink_h) * sink_h
    nat_h2 = nat_h2 / (np.linalg.norm(nat_h2) + 1e-12)
    third = B[2] - (B[2] @ sink_h) * sink_h - (B[2] @ nat_h2) * nat_h2
    third = third / (np.linalg.norm(third) + 1e-12)
    # off-plane random (in null of B)
    rng = np.random.default_rng(0)
    roff = rng.standard_normal(Wn.shape[0])
    roff = roff - P @ roff
    roff = roff / (np.linalg.norm(roff) + 1e-12)

    fro = float(torch.norm(W) / math.sqrt(W.shape[1]))
    print(f"  readout gain (normalized by Frobenius/sqrt(d) = {fro:.2f}):")
    for name, d in [('sink_h', sink_h), ('native_h2', nat_h2),
                    ('third_h', third), ('OFF-plane', roff)]:
        dg = torch.as_tensor(d, dtype=torch.float16, device=DEV)
        g = float(torch.norm(W @ dg) / fro)
        print(f"    {name:>11}: g = {g:.3f}   ({g * fro:.1f} raw)", flush=True)

    # ---- 3. targetless dial: in-plane rotation toward native ----
    a0 = float(vf @ sink_h)
    b0 = float(vf @ nat_h2)
    c0 = float(vf @ third)
    print(f"\n  TARGETLESS DIAL (in-plane rotation theta -> native axis):")
    print(f"    start coords a={a0:+.1f} b={b0:+.1f} c={c0:+.1f}")
    for th in (-0.3, 0.0, 0.2, 0.4, 0.6, 0.9):
        a = a0 * math.cos(th) - b0 * math.sin(th)
        b = a0 * math.sin(th) + b0 * math.cos(th)
        v2 = a * sink_h + b * nat_h2 + c0 * third
        v2 = v2 * (np.linalg.norm(vf) / (np.linalg.norm(v2) + 1e-12))
        L2 = logits(v2)
        print(f"    theta={th:+.1f}: native_logit={L2[native]:+.2f} "
              f"gap={L2[native] - L2.max():+.2f} "
              f"argmax={tok.decode([int(L2.argmax())])!r}", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()