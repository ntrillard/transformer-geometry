#!/usr/bin/env python3
"""eval_som_manifold.py — BIG LEAP: Kohonen SOM as the collapse microscope.

Gemma-3-1B only, ~5 forwards + 1 numpy SOM, <=10s.

Trains a self-organizing map (8x8, topology-preserving 2D collapse) on the
JOINT manifold of {27-layer residual states x LM-head rows}. Then the
information-collapse story is READ OFF THE TOPOLOGY:
  - path: where the 27 states land (BMU track) -> the plunge arc
  - QE per layer: off-manifoldness (normalization profile)
  - prompt-invariance: BMU path overlap across 3 prompts (topological
    version of the 98% plane identity)
  - state-vs-decision: grid distance from the FINAL state to the native
    head-row vs to target rows
  - STEERED probe: steer L16->chicken, project final state, does its BMU
    move TOWARD the chicken-row? (SOM as steering tracker)

Kohonen update: bmu = argmin ||x - w_j||;  w_j += lr*h(bmu,j)*(x-w_j),
h = exp(-d_grid^2 / (2 sigma^2)); lr, sigma linear-decay with epoch.

Run: timeout 60 python3 -u eval_som_manifold.py  # GEMMA-3-1B
"""
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = ['For dinner I made', 'I went to the store and bought',
           'There once was a chicken']
TARGET = 'chicken'
EXTRA = ['paris', 'pizza', 'bread', 'tokyo', 'london', ' I', '.', ' the']
GRID = 8
NP = len(PROMPTS)


def kohonen(X, g=GRID, epochs=400, lr0=0.5, sig0=2.5):
    """Classic Kohonen SOM. Returns (W (g*g, d), coords (g*g, 2))."""
    rng = np.random.default_rng(7)
    n, d = X.shape
    W = rng.standard_normal((g * g, d)) * 0.01
    W = W / (np.linalg.norm(W, axis=1)[:, None] + 1e-9)
    yy, xx = np.mgrid[:g, :g]
    coords = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
    for e in range(epochs):
        lr = lr0 * (1 - e / epochs) + 1e-3
        sig = sig0 * (1 - e / epochs) + 0.4
        for i in rng.permutation(n):
            x = X[i]
            dist = ((W - x) ** 2).sum(1)
            b = int(dist.argmin())
            dg = np.linalg.norm(coords - coords[b], axis=1)
            h = np.exp(-(dg ** 2) / (2 * sig * sig))
            W += lr * h[:, None] * (x - W)
        W = W / (np.linalg.norm(W, axis=1)[:, None] + 1e-9)
    return W, coords


def bmu(W, x):
    return int(((W - x) ** 2).sum(1).argmin())


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    NL = model.config.num_hidden_layers

    tid_l = {}
    for w in [TARGET] + EXTRA:
        ids = tok(w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            tid_l[w] = int(ids[0])

    Xstates = []
    for pidx, PROMPT in enumerate(PROMPTS):
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
        for li in range(NL + 1):
            v = caps[li].cpu().numpy().astype(np.float32)
            Xstates.append(v / (np.linalg.norm(v) + 1e-9))

    Xtgt = []
    for w in tid_l:
        Wt = lm_w[tid_l[w]].detach().float().cpu().numpy().astype(np.float32)
        Xtgt.append(Wt / (np.linalg.norm(Wt) + 1e-9))
    X = np.array(Xstates + Xtgt)
    ns = len(Xstates)

    Wsom, coords = kohonen(X)
    print(f"[{MODEL}] prompts={NP} states={ns} targets={len(Xtgt)} "
          f"SOM {GRID}x{GRID} epochs=400", flush=True)

    BMU = np.zeros((NP, NL + 1), dtype=int)
    QE = np.zeros(NL + 1)
    for p in range(NP):
        for li in range(NL + 1):
            b = bmu(Wsom, X[p * (NL + 1) + li])
            BMU[p][li] = b
            QE[li] += float(np.linalg.norm(X[p * (NL + 1) + li] - Wsom[b]))
    QE /= NP

    from scipy.cluster.vq import kmeans2
    _, lbls = kmeans2(Wsom, 4, minit='points', seed=0)

    print(f"  {'layer':>6} {'BMU P0':>9} {'QE':>6} {'region':>7}")
    for li in range(NL + 1):
        b0 = BMU[0][li]
        print(f"  L{li:>3}   ({coords[b0][0]:.0f},{coords[b0][1]:.0f}) "
              f"{QE[li]:>6.3f} R{lbls[b0]:>3}", flush=True)

    pls = []
    for p in range(NP):
        pl = sum(np.linalg.norm(coords[BMU[p][li]] - coords[BMU[p][li + 1]])
                 for li in range(NL))
        pls.append(pl)
    print(f"  path length P0={pls[0]:.1f} P1={pls[1]:.1f} P2={pls[2]:.1f}")
    agr01 = np.mean([BMU[0][li] == BMU[1][li] for li in range(NL + 1)])
    agr02 = np.mean([BMU[0][li] == BMU[2][li] for li in range(NL + 1)])
    print(f"  BMU exact-match P0vsP1={agr01:.2f} P0vsP2={agr02:.2f}")

    idx = {w: ns + i for i, w in enumerate(tid_l)}
    b_fin = BMU[0][NL]
    b_nat = bmu(Wsom, X[idx[' I']]) if ' I' in idx else -1
    b_tgt = bmu(Wsom, X[idx[TARGET]]) if TARGET in idx else -1
    d_fn = float(np.linalg.norm(coords[b_fin] - coords[b_nat]))
    d_ft = float(np.linalg.norm(coords[b_fin] - coords[b_tgt]))
    print(f"  grid dist final->native_row={d_fn:.1f}  "
          f"final->target_row={d_ft:.1f}")

    # STEER probe: inject toward chicken at L16, project final state
    ids = tok(PROMPTS[0], add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    caps16 = {}

    def c16(m, i, o):
        caps16['v'] = o[0, -1, :].float()

    h16 = model.model.layers[16].register_forward_hook(c16)
    with torch.no_grad():
        model(ids)
    h16.remove()
    v16 = caps16['v'].cpu().numpy().astype(np.float32)
    Wt = lm_w[tid_l[TARGET]].detach().float().cpu().numpy().astype(np.float32)
    vn = v16 / (np.linalg.norm(v16) + 1e-9)
    tau = Wt - (vn @ Wt) * vn
    g = tau / (np.linalg.norm(tau) + 1e-9)
    a = 0.3
    pert = (vn * np.cos(a) + g * np.sin(a)) * np.linalg.norm(v16)

    def inj(m, i, o, p=pert):
        out = o.clone()
        out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype, device=out.device)
        return out

    capf = {}

    def cf(m, i, o):
        capf['v'] = o[0, -1, :].float()

    hf = model.model.norm.register_forward_hook(cf)
    h16b = model.model.layers[16].register_forward_hook(inj)
    with torch.no_grad():
        model(ids)
    hf.remove()
    h16b.remove()
    vfs = capf['v'].cpu().numpy().astype(np.float32)
    vfs = vfs / (np.linalg.norm(vfs) + 1e-9)
    b_fs = bmu(Wsom, vfs)
    ds_t = float(np.linalg.norm(coords[b_fs] - coords[b_tgt]))
    ds_n = float(np.linalg.norm(coords[b_fs] - coords[b_nat]))
    print(f"  STEERED final BMU=({coords[b_fs][0]:.0f},{coords[b_fs][1]:.0f}) "
          f"dist->chicken={ds_t:.1f} dist->native={ds_n:.1f}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()