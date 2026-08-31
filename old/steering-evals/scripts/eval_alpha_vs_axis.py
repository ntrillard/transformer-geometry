#!/usr/bin/env python3
"""eval_alpha_vs_axis.py — BIG LEAP: steering cost alpha* = ONE coordinate
(cos(W_t, U1))? vs native-alignment as competing axis.

Gemma-3-1B only, eigh + 2 prompt forwards + vectorized a*, <=10s.

f8da699: harder targets (paris) sit less center-aligned (cos(W,U1) lower)
than easier (chicken). Test EXACTLY:
  a*(t) = atan2(-A, B) closed form (validated 0.000) for ~50 tokens,
  2 prompts (does the per-token cost curve transfer across states?).

Competing predictors of a*(t), per token:
  c1   = cos(W_t, U1)          (THE HEAD-AXIS / center hypothesis)
  cn   = cos(W_t, W_native)    (native-alignment hypothesis)
  gap0 = L0[native] - L0[t]    (raw margin)
  rank = blocking rank@final
Fit a*(t) ~ each (linear R2); report the true axis. Then: does a*(t)
transfer across prompts (rank-correlation of per-token cost between the
2 prompts)? If the center-axis explains a* AND transfers -> ONE SVD of
the head predicts the steering cost of every token, anywhere.

Run: timeout 60 python3 -u eval_alpha_vs_axis.py  # GEMMA-3-1B
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT
from eval_nb_quick import CLASSES

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = ['For dinner I made', 'I went to the store and bought']
EXTRA = ['paris', 'tokyo', 'london', 'ocean', 'river', 'milk', 'wine',
         'apple', 'cake', 'bread', 'butter']


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()          # (V, d) fp16
    d = W.shape[1]
    NL = model.config.num_hidden_layers

    # ---- head principal axis U1 ----
    A = W.float().T @ W.float()
    evals, evecs = torch.linalg.eigh(A)
    idx = int(torch.argmax(evals))
    U1 = evecs[:, idx]                         # (d,)
    print(f"[{MODEL}] U1 (head principal axis, sigma_1={evals[idx].sqrt():.1f})")

    # ---- target pool (single-token) ----
    pool = sorted({w for c in CLASSES.values() for w in c}) + EXTRA
    tids, tnames = [], []
    for w in pool:
        iids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(iids) == 1:
            tids.append(int(iids[0]))
            tnames.append(w)
    Wt = W[tids].float()                        # (T, d)
    T = len(tids)
    print(f"  targets={T}  prompts={len(PROMPTS)}")

    # per-token fixed features
    U1n = U1 / U1.norm()
    c1 = (Wt @ U1n) / (Wt.norm(dim=1) + 1e-12)  # cos(W_t, U1)

    rows_all = []
    for pi, PROMPT in enumerate(PROMPTS):
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        cf = {}

        def cc(m, i, o):
            cf['v'] = o[0, -1, :].float()

        h = model.model.norm.register_forward_hook(cc)
        with torch.no_grad():
            L0 = model(ids).logits[0, -1].float()
        h.remove()
        native = int(L0.argmax())
        vf = (cf['v'] / cf['v'].norm()).float()     # (d,)
        Wn = W[native].float()

        # vectorized a*
        A_ = torch.einsum('d,td->t', vf, Wt - Wn[None, :])     # (T,)
        proj = torch.einsum('td,d->t', Wt, vf)
        tau = Wt - proj[:, None] * vf[None, :]
        B_ = torch.einsum('td,td->t', tau, Wt - Wn[None, :]) / \
            (tau.norm(dim=1) + 1e-12)
        ast = torch.atan2(-A_, B_).float().cpu().numpy()       # (T,)
        cn = ((Wt * Wn[None, :]).sum(1) /
              (Wt.norm(dim=1) * Wn.norm() + 1e-12))
        cn = cn.float().cpu().numpy()
        gap0 = (L0[native] - L0[tids]).float().cpu().numpy()
        rank = np.array([int((L0 > L0[ti]).sum().item()) for ti in tids])
        c1n = c1.float().cpu().numpy()
        sa = (-A_ / (B_ + 1e-12)).float().cpu().numpy()   # small-angle crossing
        sa = np.where(sa < 0, sa + np.pi, sa)  # fold into [0,pi)  (B sign folded)
        rows_all.append((c1n, cn, gap0, rank, np.abs(sa), (-B_).float().cpu().numpy()))
        print(f"  P{pi} {PROMPT!r:34} native={tok.decode([native])!r}  "
              f"chicken a*={np.abs(ast[tnames.index('chicken')]):.3f} "
              f"paris a*={np.abs(ast[tnames.index('paris')]):.3f}",
              flush=True)

    # ---- regression per prompt ----
    from numpy.linalg import lstsq
    print("\n  slope structure: R2(a* vs gap/median_slope), resid-ranks:")
    for pi in range(len(PROMPTS)):
        c1n, cn, gap0, rank, aabs, slope = rows_all[pi]
        med = float(np.median(slope))
        rel = float(slope.std() / (np.abs(slope).mean() + 1e-9))
        from numpy.polynomial import polynomial as _pol
        c = _pol.polyfit(gap0, aabs, 1)
        pred = c[0] + c[1] * gap0
        r2d = 1 - float(np.sum((aabs - pred) ** 2) /
                        (np.sum((aabs - aabs.mean()) ** 2) + 1e-12))
        ratio = float(np.median(aabs / (gap0 + 1e-9)))
        print(f"  P{pi}: slope med={med:.2f} rel={rel:.3f}  ",
              f"R2(a* ~ gap)={r2d:.3f}  med(a*/gap)=1/{1/max(ratio,1e-9):.1f}", flush=True)
        print(f"  P{pi}: slope median={med:.2f}  rel-spread={rel:.3f}  ",
              f"alpha*/gap=1/{1/max(ratio,1e-9):.1f}", flush=True)
    # rank's role: residual of gap fit vs rank
    from numpy.linalg import lstsq as _ls
    for pi in range(len(PROMPTS)):
        c1n, cn, gap0, rank, aabs, slope = rows_all[pi]
        X = np.stack([np.ones_like(gap0), gap0], 1)
        b = _ls(X, aabs, rcond=None)[0]
        resid = aabs - X @ b
        cc_r = float(np.corrcoef(resid, rank)[0, 1])
        print(f"  P{pi}: corr(resid_of_gap_fit, rank) = {cc_r:+.3f} "
              f"(0 = rank adds nothing after gap)")
    for pi in range(len(PROMPTS)):
        c1n, cn, gap0, rank, aabs, slope = rows_all[pi]
        feats = {'c1(center)': c1n, 'cn(native)': cn,
                 'gap0': gap0, 'rank': rank}
        line = f"  P{pi}: "
        for name, X in feats.items():
            Xs = np.stack([np.ones_like(X), X], 1)
            b = lstsq(Xs, aabs, rcond=None)[0]
            pred = Xs @ b
            r2 = 1 - np.sum((aabs - pred) ** 2) / \
                (np.sum((aabs - aabs.mean()) ** 2) + 1e-12)
            line += f"{name}:R2={r2:+.2f}  "
        print(line)

    # ---- transfer: rank-correlation of a*(t) across prompts ----
    a0 = rows_all[0][4]
    a1 = rows_all[1][4]
    rc = float(np.corrcoef(a0, a1)[0, 1])
    from scipy.stats import spearmanr
    rs = spearmanr(a0, a1).statistic
    print(f"\n  a*(t) cross-prompt: corr={rc:+.3f}  spearman={rs:+.3f} "
          f"  (1 = token steering cost is state-independent)")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()