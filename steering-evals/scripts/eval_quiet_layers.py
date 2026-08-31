#!/usr/bin/env python3
"""eval_quiet_layers.py — FAST + META: learn to PREDICT scramble(d) from
state dynamics (the 'quiet-layer' hypothesis). GEMMA-3-1B ONLY.

Follows af71861 (scramble law) with a NOVEL internal-dynamics twist:
instead of calibrating behavioral alpha* per depth, learn scramble(d)
from cheap per-layer dynamics features captured in ONE forward:
  nrm    = ||v(l)||            state magnitude at layer l
  vel    = ||v(l+1)-v(l)||/||v(l)||   per-layer rewrite magnitude
  angl   = 1 - cos(v(l+1),v(l))       state angle change (turns)
  lleft  = layers remaining after l
And proxy alpha*(l,chicken) from the same captured states (analytic B).
Behavioral alpha*(l,chicken) table is the measured 653df80/af71861 values.

scramble(l) = behavioral_a*(l) / proxy_a*(l). Phase 2 ridge-LEARNS
log(scramble) from dynamics (NO proxy features, held-one-out = 8 depths).
Phase 3 closure: predict a*(l) = proxy(l) x exp(yhat) vs behavioral.

Run: timeout 60 python3 -u eval_quiet_layers.py  # GEMMA-3-1B ONLY
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

# 0-based layer indices for which behavioral alpha* is known
DEPTHS = [1, 5, 7, 9, 11, 13, 17, 21, 'final']
BEHAV = {1: 0.933, 5: 0.582, 7: 0.605, 9: 0.269, 11: 0.220, 13: 0.196,
         17: 0.176, 21: 0.101, 'final': 0.177}


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    NL = model.config.num_hidden_layers

    tid = tok(' ' + TARGET, add_special_tokens=False).input_ids
    if len(tid) != 1:
        print("target not single token; abort")
        return
    tid = int(tid[0])
    Wt = lm_w[tid].detach().float().cpu().numpy()
    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    with torch.no_grad():
        L0 = model(ids).logits[0, -1].float()
    native = int(L0.argmax())
    Wn = lm_w[native].detach().float().cpu().numpy()

    # ---- ONE forward: capture ALL per-layer outputs + final norm ----
    caps = {}

    def mk(li):
        def h(m, i, o):
            caps[li] = o[0, -1, :].float().cpu().numpy()
        return h

    hooks = []
    for li in range(NL):
        hooks.append(model.model.layers[li].register_forward_hook(mk(li)))
    hooks.append(model.model.norm.register_forward_hook(mk('final')))
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()

    # ---- proxy alpha* (analytic B) per depth, target=chicken ----
    rows = []
    for dpos in DEPTHS:
        d = NL if dpos == 'final' else int(dpos)
        dlbl = str(dpos) if dpos == 'final' else dpos
        v = caps[dpos if dpos != 'final' else 'final']
        vn = v / (np.linalg.norm(v) + 1e-12)
        A = float(vn @ (Wt - Wn))
        tau = Wt - (vn @ Wt) * vn
        B = float(tau @ (Wt - Wn)) / (np.linalg.norm(tau) + 1e-12)
        prox = math.atan2(-A, B)
        # dynamics features
        if dpos != 'final':
            vp = caps[dpos + 1]
            vel = float(np.linalg.norm(vp - v) / (np.linalg.norm(v) + 1e-12))
            cosd = float(np.dot(vp, v) / (np.linalg.norm(vp) * np.linalg.norm(v) + 1e-12))
            angl = 1.0 - cosd
        else:
            vel = float('nan')
            angl = float('nan')
        nrm = float(np.linalg.norm(v))
        lleft = NL - d
        beh = BEHAV[dpos]
        scram = abs(beh / prox) if prox != 0 else float('nan')
        rows.append(dict(d=dpos, dlbl=dlbl, nrm=nrm, vel=vel, angl=angl,
                         lleft=lleft, prox=prox, beh=beh, scram=scram))
        print(f"  L{dlbl:>5}: nrm={nrm:7.2f} vel={vel:5.3f} angl={angl:5.3f} "
              f"lleft={lleft:>2}  prox={prox:+.3f} beh={beh:.3f} "
              f"scram={scram:6.1f}x", flush=True)

    # ---- Phase 2: learn log(scramble) from dynamics ONLY ----
    feats = ['nrm', 'vel', 'angl', 'lleft']
    ok = [r for r in rows if r['dlbl'] != 'final' and np.isfinite(r['vel'])]
    X = np.array([[r[f] for f in feats] for r in ok])
    y = np.log(np.array([r['scram'] for r in ok]))
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    preds = np.zeros(len(ok))
    for i in range(len(ok)):
        mask = np.ones(len(ok), bool); mask[i] = False
        A = Z[mask].T @ Z[mask] + np.eye(Z.shape[1]) * 1.0
        w = np.linalg.solve(A, Z[mask].T @ y[mask])
        preds[i] = Z[i] @ w
    r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
    # full-fit weights for interpretation
    A = Z.T @ Z + np.eye(Z.shape[1]) * 1.0
    w = np.linalg.solve(A, Z.T @ y)
    std_w = w * sd
    print("\n  Phase 2: learned log(scramble) ~ dynamics (held-one-out):")
    print(f"    R2(loo) = {r2:+.3f}   (1 = dynamics fully determine scramble)")
    for f, vw in zip(feats, std_w):
        print(f"    w[{f}] = {vw:+.3f}", flush=True)

    # ---- Phase 3: closure - predict a*(d) = proxy x exp(yhat) ----
    print("\n  Phase 3: predicted behavioral a* vs measured:", flush=True)
    mae = 0.0
    for i, r in enumerate(ok):
        yhat = Z[i] @ w
        apred = r['prox'] * math.exp(yhat)
        err = apred - r['beh']
        mae += abs(err)
        print(f"    L{r['dlbl']:>3}: pred={apred:+.3f} meas={r['beh']:.3f} "
              f"err={err:+.3f}", flush=True)
    mae /= len(ok)
    print(f"    MAE(pred) = {mae:.3f} rad", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()