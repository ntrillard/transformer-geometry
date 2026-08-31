#!/usr/bin/env python3
"""eval_head_axis.py — BIG LEAP: the decision = angular distance to the
head's PRINCIPAL axis U_1.

Gemma-3-1B only, eigh on GPU + 10 prompt forwards, <=10s.

The readout anisotropy is readout-born (0fb0ee0). The leap: the head's
top right-singular direction U_1 (the amplified residual direction) is a
FIXED object (the head doesn't change). Then maybe EVERY decision
observable is one cosine vs U_1:
  - confidence  ~ cos(vf, U_1)  (what the head amplifies)
  - margin law  ~ same (is |a*(2nd)| a function of cos(vf,U_1)?)
  - 'steerable' ~ cos(W_t, U_1)  (which tokens resonate)

Probes:
  Q1 CONTENT: what IS U_1? cos(U_1, W_native), cos(U_1, token-mean W),
     cos(U_1, sink_h); and RESONANT TOKENS = argmax(W @ U_1) - the
     tokens the head most amplifies when state ~ U_1.
  Q2 AXIS-LAW: across 10 prompts, corr(prob1, cos(vf, U_1)) and
     corr(cos(vf,U_1), |a*(2nd)|) - is confidence the alignment?
     partial vs gap12.
  Q3 STEERABILITY: cos(W_paris, U_1) vs cos(W_chicken, U_1) - does the
     harder target sit more off-axis?

Run: timeout 60 python3 -u eval_head_axis.py  # GEMMA-3-1B
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = [
    'For dinner I made', 'I went to the store and bought',
    'The recipe calls for', 'In my kitchen I have',
    'There once was a chicken', 'My favorite meal is',
    'The restaurant served', 'For lunch I had',
    'Breakfast today was', 'At the market I found',
]


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()          # (V, d) fp16
    NL = model.config.num_hidden_layers
    d = W.shape[1]

    # ---- head principal axes via eigh(W^T W) on GPU ----
    A = W.float().T @ W.float()
    evals, evecs = torch.linalg.eigh(A)
    idx = torch.argsort(evals, descending=True)[:3]
    U = evecs[:, idx].float()                  # (d, 3) principal directions
    sv = evals[idx].sqrt()
    print(f"[{MODEL}] head principal singular values: "
          f"{sv.cpu().numpy().round(1)}")

    # ---- Q1 content of U_1 ----
    # capture one trajectory for sink + lunchtime first prompt's vf
    ids0 = tok(PROMPTS[0], add_special_tokens=False,
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
        L0 = model(ids0).logits[0, -1].float()
    for h in hooks:
        h.remove()
    native = int(L0.argmax())
    V = np.stack([caps[li].cpu().numpy().astype(np.float64)
                  for li in range(NL)] + [caps['f'].cpu().numpy()])
    sink = V[22] - V[18]
    sink_h = torch.as_tensor(sink / (np.linalg.norm(sink) + 1e-12),
                             dtype=torch.float32, device=DEV)
    U1, U2, U3 = U[:, 0], U[:, 1], U[:, 2]
    Wnat = W[native].float()
    Wmean = W.float().mean(0)
    c_nat = float(U1 @ Wnat / (U1.norm() * Wnat.norm() + 1e-12))
    c_mean = float(U1 @ Wmean / (U1.norm() * Wmean.norm() + 1e-12))
    c_sink = float(U1 @ sink_h)
    print(f"  U_1 content: cos(U1, W_native)={c_nat:+.3f}  "
          f"cos(U1, token-mean)={c_mean:+.3f}  cos(U1, sink_h)={c_sink:+.3f}")

    # resonant tokens (what the head amplifies along U1)
    res1 = W.float() @ U1
    r1 = torch.topk(res1, 8).indices.cpu().tolist()
    res2 = W.float() @ U2
    r2 = torch.topk(res2, 8).indices.cpu().tolist()
    print(f"  RESONANT tokens along U_1: "
          f"{[tok.decode([i]) for i in r1]}")
    print(f"  RESONANT tokens along U_2: "
          f"{[tok.decode([i]) for i in r2]}")

    # ---- Q2 axis-law across prompts ----
    rows = []
    for pidx, PROMPT in enumerate(PROMPTS):
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        cf = {}

        def c(m, i, o):
            cf['v'] = o[0, -1, :].float()

        h = model.model.norm.register_forward_hook(c)
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        h.remove()
        vf = (cf['v'] / cf['v'].norm()).float()
        p = torch.softmax(L.float(), dim=0)
        prob1 = float(p.max())
        n1, n2 = int(p.argmax()), int(torch.topk(p, 2).indices[1])
        Wn = W[n1].float()
        Ws = W[n2].float()
        A_ = float(vf @ (Ws - Wn))
        tau = Ws - (vf @ Ws) * vf
        B_ = float(tau @ (Ws - Wn)) / (tau.norm() + 1e-12)
        a2 = math.atan2(-A_, B_)
        c1 = float(vf @ U1)
        c2 = float(vf @ U2)
        gap = float(L[n1] - L[n2])
        rows.append((prob1, abs(a2), c1, c2, gap, n1, PROMPT))
        print(f"P{pidx} {PROMPT!r:34} p1={prob1:.3f} a2={abs(a2):.4f} "
              f"c1={c1:+.3f} c2={c2:+.3f} ({tok.decode([n1])!r})",
              flush=True)

    R = np.array([r[:5] for r in rows], dtype=float)
    p1, a2, c1, c2, gap = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4]
    print(f"\n  Q2 AXIS-LAW (across {len(PROMPTS)} prompts):")
    cc_pc = float(np.corrcoef(p1, c1)[0, 1])
    cc_ac = float(np.corrcoef(a2, c1)[0, 1])
    cc_pa = float(np.corrcoef(p1, a2)[0, 1])
    print(f"    corr(prob1, cos(vf,U1))   = {cc_pc:+.3f}")
    print(f"    corr(|a*(2nd)|, cos(vf,U1)) = {cc_ac:+.3f}")
    print(f"    corr(prob1, |a*(2nd)|)    = {cc_pa:+.3f}  (margin law)")
    # partial: does c1 add beyond gap?
    from numpy.linalg import lstsq
    X = np.stack([np.ones_like(gap), gap], 1)
    resid = p1 - X @ lstsq(X, p1, rcond=None)[0]
    c_part = float(np.corrcoef(resid, c1)[0, 1])
    print(f"    PARTIAL corr(prob1, c1 | gap) = {c_part:+.3f}")

    # ---- Q3: steerability vs axis ----
    Wc = W[tok(' chicken', add_special_tokens=False).input_ids[0]].float()
    Wp = W[tok(' paris', add_special_tokens=False).input_ids[0]].float()
    for nm, Wt in (('chicken', Wc), ('paris', Wp), ('native', Wnat)):
        cc3 = float(U1 @ Wt / (U1.norm() * Wt.norm() + 1e-12))
        print(f"  cos(U1, {nm:>7}) = {cc3:+.3f}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()