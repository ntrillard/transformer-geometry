#!/usr/bin/env python3
"""eval_meta_depth.py — FAST: meta-learn the DEPTH riser from internals.

Follows 653df80 (per-depth slope = analytic riser) + 9dd11fc (meta-budget
conforrnal). Key speed trick: slope(d) = B = <tau_hat(d), W_t - W_native>
is analytic from ONE forward capturing v(d) -> the FULL per-depth
meta-dataset (all targets x depths) needs only ~5 forwards. Each probe
(the capture + the learners) fits in <=10s.

The internals-discovery question: WHICH internal-state quantity (at any
depth) best predicts steering transfer efficiency (alpha*)? Features per
(depth, target): depth, layers_left, ||v(d)||, cos(v_hat, W_t), gap_proxy
=A=<v_hat, W_t-W_native>, rank_t(final), row norms, rowcos.
Label: alpha* = atan2(-A, B) (the EXACT analytic crossing, verified
rel-err 0.000 in fbb2055).

Phase A capture: per-depth residuals -> full table + corr(feature, alpha*)
per depth. Phase B learn: held-out by class; ridge+knn+conformal predict
alpha* from {all feats} and {feats WITHOUT A,B} (the true discovery -
can we predict transfer without the closed form?).

Run: timeout 90 python3 -u eval_meta_depth.py google/gemma-3-1b-it
     timeout 90 python3 -u eval_meta_depth.py Qwen/Qwen2-0.5B-Instruct
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'

if 'qwen' in MODEL.lower():
    DEPTHS = [5, 9, 13, 17, 21, 'final']   # L6/L10/L14/L18/L22
else:
    DEPTHS = [5, 9, 13, 17, 'final']       # L6/L10/L14/L18

EXTRA_TARGETS = ['paris', 'tokyo', 'chicken', 'pizza']


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    NL = model.config.num_hidden_layers

    # target pool: all CLASSES words + extras
    pool = sorted({w for c in CLASSES.values() for w in c}) + EXTRA_TARGETS
    tids, tnames, bad = [], [], []
    for w in pool:
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            tids.append(int(ids[0])); tnames.append(w)
        else:
            bad.append(w)
    tids = np.array(tids)
    print(f"[{MODEL}] targets={len(tids)} (skip {len(bad)}: {bad}) "
          f"depths={DEPTHS}", flush=True)

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    with torch.no_grad():
        L0 = model(ids).logits[0, -1].float()
    native = int(L0.argmax())
    print(f"  native={native!r} {tok.decode([native])!r}", flush=True)
    Wn = lm_w[native].detach().float().cpu().numpy()

    Wt = lm_w[tids].detach().float().cpu().numpy()      # (T, d)
    T = len(tids)
    rank_final = np.array([int((L0 > L0[ti]).sum().item()) for ti in tids])
    gap_final = np.array([float(L0[ti] - L0[native]) for ti in tids])

    def rot_alpha(out, target_row, alpha):
        v = out[:, -1, :].float().reshape(-1)
        vn = v / v.norm()
        t = target_row - (target_row @ vn.cpu().numpy()) * vn.cpu().numpy()
        t = t / (np.linalg.norm(t) + 1e-12)
        tg = torch.as_tensor(t, dtype=torch.float32, device=DEV)
        g = tg - (tg @ vn) * vn
        g = g / (g.norm() + 1e-8)
        v2 = vn * math.cos(alpha) + g * math.sin(alpha)
        out = out.clone()
        out[:, -1, :] = (v.norm() * v2).to(out.dtype)
        return out

    # ---- Phase A: capture per-depth residuals, compute features + alpha* ----
    rows = []
    for dpos in DEPTHS:
        dname = str(dpos) if dpos != 'final' else 'final'
        d = NL if dpos == 'final' else int(dpos)
        if dpos == 'final':
            layer = model.model.norm
        else:
            layer = model.model.layers[dpos]
        cap = {}

        def cap_out(m, i, o):
            cap['v'] = o[0, -1, :].float()

        h = layer.register_forward_hook(cap_out)
        with torch.no_grad():
            model(ids)
        h.remove()
        v = cap['v'].cpu().numpy()
        vn = v / (np.linalg.norm(v) + 1e-12)
        norm_v = float(np.linalg.norm(v))
        layers_left = NL - d

        # A, B, alpha* per target (vectorized via einsum)
        A = vn @ (Wt - Wn).T                      # (T,)  <v_hat, W_t - W_n>
        tau = Wt - np.outer(vn @ Wt.T, vn)         # per-target: W_t - (W_t.v)v
        tauhat = tau / (np.einsum('ij,ij->i', tau, tau)[:, None] ** 0.5 + 1e-12)
        B = np.einsum('ij,ij->i', tauhat, Wt - Wn)  # slope
        astar = np.arctan2(-A, B)                   # exact crossing
        cosprox = vn @ Wt.T / (np.linalg.norm(Wt, axis=1) + 1e-12)
        rowcos = np.einsum('ij,j->i', Wt, Wn) / (
            np.linalg.norm(Wt, axis=1) * np.linalg.norm(Wn) + 1e-12)
        for tidx in range(T):
            rows.append(dict(depth=d, dname=dname, layers_left=layers_left,
                             norm=norm_v, cos=cosprox[tidx],
                             A=A[tidx], B=B[tidx], astar=astar[tidx],
                             rank=rank_final[tidx], gap_final=gap_final[tidx],
                             row_norm_t=float(np.linalg.norm(Wt[tidx])),
                             row_norm_n=np.linalg.norm(Wn), rowcos=rowcos[tidx],
                             target=tnames[tidx]))
        # feature-label correlation at this depth
        cs = np.array([r['cos'] for r in rows if r['dname'] == dname])
        aa = np.array([r['astar'] for r in rows if r['dname'] == dname])
        print(f"  L{dname}: mean|a*|={np.abs(aa).mean():.2f} "
              f"std={aa.std():.3f} "
              f"corr(cos,a*)={np.corrcoef(cs, aa)[0, 1]:+.3f} "
              f"a*[paris]={aa[tnames.index('paris')]:.2f} "
              f"a*[chicken]={aa[tnames.index('chicken')]:.2f}", flush=True)

    # ---- Phase B: meta-learn on held-out TARGETS ----
    # split by class membership (approx: every 4th target held out)
    te_idx = list(range(3, T, 4))         # held-out target indices in tids
    tr_idx = [i for i in range(T) if i not in te_idx]

    def feat_mat(keys):
        X = np.array([[r[k] for k in keys]
                      for i, r in enumerate(rows)
                      if (i % T) in tr_idx])
        Xte = np.array([[r[k] for k in keys]
                        for i, r in enumerate(rows)
                        if (i % T) in te_idx])
        return X, Xte

    y = np.array([r['astar'] for r in rows])
    y_tr = np.array([y[i] for i in range(len(rows)) if (i % T) in tr_idx])
    y_te = np.array([y[i] for i in range(len(rows)) if (i % T) in te_idx])

    def score(name, keys):
        X, Xte = feat_mat(keys)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z, Zte = (X - mu) / sd, (Xte - mu) / sd
        # ridge
        A = Z.T @ Z + np.eye(Z.shape[1]) * 1.0
        w = np.linalg.solve(A, Z.T @ y_tr)
        pred = Zte @ w
        res = y_te - pred
        q = np.quantile(res, 0.95)
        beta = np.clip(pred + q, 0.0, 1.0)
        rel = np.mean(np.abs(beta) >= np.abs(y_te) - 1e-9)
        mae = np.abs(beta - y_te).mean()
        w_std = w * sd
        imp = np.argsort(-np.abs(w_std))[:3]
        feats = ', '.join(f"{keys[j]}({w_std[j]:+.2f})" for j in imp)
        print(f"  {name:>34} rel={rel:.2f} mae={mae:.3f} "
              f"top3={feats}", flush=True)
        return rel, mae

    print("  Phase B (learn alpha* on held-out targets):", flush=True)
    keys_all = ['depth', 'layers_left', 'norm', 'cos', 'A', 'B',
                'rank', 'gap_final', 'row_norm_t', 'row_norm_n', 'rowcos']
    keys_noclosed = ['depth', 'layers_left', 'norm', 'cos',
                     'rank', 'gap_final', 'row_norm_t', 'row_norm_n', 'rowcos']
    score('closed-form feats (A,B)', keys_all)
    score('NO closed form (discovery)', keys_noclosed)

    # ---- Phase C: per-depth importance of rank & norm ----
    print("  Phase C (per-depth learned importance of rank / norm):")
    for dpos in DEPTHS:
        dname = str(dpos) if dpos != 'final' else 'final'
        Xd = np.array([[r[k] for k in keys_noclosed]
                       for i, r in enumerate(rows)
                       if r['dname'] == dname and (i % T) in tr_idx])
        yd = np.array([y[i] for i in range(len(rows))
                       if rows[i]['dname'] == dname and (i % T) in tr_idx])
        mu, sd = Xd.mean(0), Xd.std(0) + 1e-9
        Zd = (Xd - mu) / sd
        Ad = Zd.T @ Zd + np.eye(Zd.shape[1]) * 1.0
        wd = np.linalg.solve(Ad, Zd.T @ yd)
        idxr = keys_noclosed.index('rank')
        idxn = keys_noclosed.index('norm')
        print(f"    L{dname}: w[rank]={wd[idxr]:+.2f}  w[norm]={wd[idxn]:+.2f}",
              flush=True)
    # ---- Phase D: SCRAMBLE table (behavioral/proxy ratio) ----
    # behavioral alpha* (chicken, same prompt) measured in eval_depth_slopes
    if 'qwen' in MODEL.lower():
        BEHAV = {'5': 1.41, '9': 1.20, '13': 1.05, '17': 0.98,
                 '21': 0.50, 'final': 0.029}
    else:
        BEHAV = {'5': 0.85, '9': 0.28, '13': 0.21, '17': 0.19,
                 'final': 0.18}
    print("  Phase D (scramble = behavioral/proxy, chicken):")
    for dpos in DEPTHS:
        dname = str(dpos) if dpos != 'final' else 'final'
        prox = next(r['astar'] for i, r in enumerate(rows)
                    if r['dname'] == dname and r['target'] == 'chicken')
        beh = BEHAV[dname]
        print(f"    L{dname.ljust(3) if dname.isdigit() else dname}: proxy={prox:+.3f}  behavioral={beh:+.3f}  "
              f"scramble={beh / max(prox, 1e-6):6.1f}x", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()