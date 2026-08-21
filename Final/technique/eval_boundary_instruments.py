#!/usr/bin/env python3
"""Cheap boundary instruments (John6666's two gauges), budgeted.

A) blocking competitor + margin records along the target tangent arc;
B) theta_cell: shortest angle into the target's rank-1 decision CONE,
   via active-set euclidean projection over the full vocab (capped rounds),
   vs theta_author (the arc's analytic first rank-1 crossing angle).

Run: python eval_boundary_instruments.py      (Qwen2-0.5B cached; < ~1 min)
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls

import steering_geometry_test as M

OUT = Path("steering_geometry_results")
BUDGET = math.radians(45)
MAX_T = 24
MAX_ROUNDS = 60


def _enter_angles(P, Q, budget):
    """Per-competitor interval [lo_j, hi_j] within [0,budget] where target is ahead."""
    P = np.asarray(P, np.float64); Q = np.asarray(Q, np.float64)
    R = np.hypot(P, Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.divide(P, R, out=np.zeros_like(P), where=R > 0)
        sn = np.divide(Q, R, out=np.zeros_like(P), where=R > 0)
    th = np.arctan2(sn, cs)
    lo, hi = th - math.pi / 2, th + math.pi / 2
    twopi = 2 * math.pi
    lo_j = np.full_like(P, np.inf, dtype=np.float64)
    hi_j = np.full_like(P, np.inf, dtype=np.float64)
    for k in range(0, 2):                       # competitors ahead at d=0
        L = np.clip(lo + twopi * k, 0, None); H = np.clip(hi + twopi * k, None, float(budget))
        ok = L < H
        better = (L < lo_j) | np.isinf(lo_j)
        lo_j = np.where(better & ok, L, lo_j)
        hi_j = np.where(better & ok, np.where(ok, H, np.inf), hi_j)
    pos0 = P > 1e-12
    if pos0.any():                              # already ahead at d=0: ahead [0, exit]
        for k in range(-1, 1):
            L = np.clip(lo + twopi * k, 0, None); H = np.clip(hi + twopi * k, None, float(budget))
            sel = pos0 & (L <= 1e-9) & (L < H)
            lo_j = np.where(sel, L, lo_j)
            hi_j = np.where(sel, H, hi_j)
    return lo_j, hi_j


def arc_stats(u, tau, A, B, t):
    """Blocking competitor + margins along the target-tangent arc. u unit, tau unit,
    A = u@W.T, B = tau@W.T (full vocab)."""
    At, Bt = float(A[t]), float(B[t])
    P = At - A; Q = Bt - B
    cosd, sind = math.cos(BUDGET), math.sin(BUDGET)
    r0 = int((A > At).sum() + 1)
    c0 = int(np.argmax(np.delete(A, t))); c0 += (c0 >= t)
    m_end = P * cosd + Q * sind
    c1 = int(np.argmax(np.delete(m_end, t))); c1 += (c1 >= t)
    lo_j, hi_j = _enter_angles(P, Q, BUDGET)
    lo_all, hi_all = lo_j.max(), hi_j.min()
    reached = np.isfinite(lo_all) and lo_all <= hi_all and lo_all <= float(BUDGET)
    cross = float(math.degrees(lo_all)) if reached else None
    blocker = int(np.argmax(lo_j)) if reached else None
    return dict(r0=r0, top0=c0, margin0=float(At - A[c0]),
                top_end=c1, margin_end=float(m_end.min()),
                cross_deg=cross, cross_blocker=blocker, arc_reachable=bool(reached))


@torch.no_grad()
def cone_angles(u, tids, W, max_rounds=60, atol=1e-6):
    """Batched shortest rotation into each target's rank-1 decision cone
    C_t = {x : (W_t - W_j)^T x >= 0 for all j != t} using raw LM-head rows.

    Active-set Euclidean projection of u onto each C_t.  The dual is an NNLS:
        min_lambda ||N^T lambda - u||^2  s.t. lambda >= 0
    where rows of N are (W_j - W_t) for the active competitors j.  The primal
    projection is x = u - N^T lambda.

    Returns (angles, X) where angles[k] is in degrees and X[k] is the projected
    point for target tids[k].
    """
    dev = W.device
    u_t = torch.as_tensor(u, dtype=torch.float32, device=dev)
    tids_t = torch.as_tensor(tids, dtype=torch.long, device=dev)
    K = len(tids)
    X = u_t.unsqueeze(0).expand(K, -1).clone()
    act = [[] for _ in range(K)]

    for _ in range(max_rounds):
        XW = X @ W.T                                   # (K,V)
        wtx = (W[tids_t] * X).sum(-1)                  # (K,)
        marg = wtx[:, None] - XW                       # (K,V)
        marg[torch.arange(K, device=dev), tids_t] = float("inf")
        worst = marg.min(1).values
        if float(worst.min()) > -atol:
            break
        changed = False
        for k in range(K):
            if float(worst[k]) < -atol and len(act[k]) < 64:
                v = marg[k] < -1e-5
                n = min(2, int(v.sum().item()))
                if n > 0:
                    viol_idx = torch.nonzero(v).flatten()
                    top = torch.topk(marg[k][v], k=n, largest=False).indices
                    for i in top:
                        j = int(viol_idx[i].item())
                        if j not in act[k]:
                            act[k].append(j)
                            changed = True
        if not changed:
            break
        # Per-target dual NNLS for the current active set (small -> CPU/scipy).
        for k in range(K):
            if not act[k]:
                X[k] = u_t
                continue
            N = W[act[k]] - W[tids_t[k]]               # (m,d) on GPU
            N_np = N.float().cpu().numpy().T           # (d,m)
            u_np = u_t.float().cpu().numpy()
            lam, _ = nnls(N_np, u_np)                  # lambda >= 0
            lam_t = torch.as_tensor(lam, dtype=torch.float32, device=dev)
            X[k] = u_t - N.T @ lam_t

    n = X.norm(dim=1)
    cos = (X * u_t).sum(1) / n.clamp_min(1e-9)
    angles = torch.rad2deg(torch.acos(cos.clamp(0, 1)))
    angles[n < 1e-9] = float("nan")
    return angles.cpu().numpy(), X


def main(model_name=None):
    t0 = time.perf_counter()
    if model_name is None:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--model", default="google/gemma-3-1b-it")
        args = ap.parse_args()
        model_name = args.model
    model, tok = M.load_model(model_name, dtype="fp16")
    N = model.config.num_hidden_layers
    L = N - 1
    W = model.lm_head.weight.detach().float().cpu().numpy()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    V, d = W.shape
    rng = np.random.default_rng(0)
    contexts = rng.choice(M.PROMPTS, size=2, replace=False).tolist()
    states = M.get_states(model, tok, contexts, [L])
    tids = sorted(rng.integers(0, V, size=MAX_T).tolist())
    dec = lambda t: (tok.decode([t]) or f"<{t}>").replace("\n", "\\n")

    rows = []
    for ctx in contexts:
        u = np.asarray(states[ctx][L], dtype=np.float32)
        u = u / np.linalg.norm(u)
        A = u @ W.T
        TAU = Wn[tids] - (Wn[tids] @ u)[:, None] * u
        TAU = TAU / np.linalg.norm(TAU, axis=1, keepdims=True)
        B = TAU @ W.T
        for k, t in enumerate(tids):
            rec = arc_stats(u, TAU[k], A, B[k], t)
            rec.update(context=ctx, target_id=t, target_text=dec(t))
            rows.append(rec)
    df = pd.DataFrame(rows)
    safe = model_name.replace("/", "--")
    df.to_csv(OUT / f"boundary_margins__{safe}.csv", index=False)

    # B) cone probe on the same (context, target) grid using raw LM-head rows
    st = time.perf_counter()
    W_t = torch.as_tensor(W, dtype=torch.float32, device=M.DEVICE)
    cones = []
    max_violation = 0.0
    converged = 0
    for ctx in contexts:
        u = np.asarray(states[ctx][L], dtype=np.float32)
        u = u / np.linalg.norm(u)
        A = u @ W.T
        angles, X = cone_angles(u, tids, W_t)          # theta_cell + projections
        # full-vocab feasibility certificate
        XW = X @ W_t.T                                 # (K,V)
        wtx = (W_t[tids] * X).sum(-1)                  # (K,)
        margins = wtx[:, None] - XW                    # (K,V)
        margins[torch.arange(len(tids), device=M.DEVICE), tids] = float("inf")
        feasible = margins.min(1).values > -1e-5
        converged += int(feasible.sum().item())
        if feasible.any():
            max_violation = max(max_violation, float((-margins.min(1).values[feasible]).max()))
        # compute theta_author BEFORE the per-target loop using vectorized B
        TAU_vec = Wn[tids] - (Wn[tids] @ u)[:, None] * u
        TAU_vec = TAU_vec / np.linalg.norm(TAU_vec, axis=1, keepdims=True)
        B_vec = TAU_vec @ W.T
        for k, t in enumerate(tids):
            # theta_author = arc crossing angle (analytic) within same budget
            lo_j, hi_j = _enter_angles(A[t] - A, B_vec[k][t] - B_vec[k], BUDGET)
            lo_all, hi_all = lo_j.max(), hi_j.min()

            arc = float(math.degrees(lo_all)) if (np.isfinite(lo_all) and lo_all <= hi_all
                                                  and lo_all <= float(BUDGET)) else np.nan
            cones.append(dict(context=ctx, target_id=t, target_text=dec(t),
                              theta_author=arc, theta_cell=float(angles[k])))
    cdf = pd.DataFrame(cones)
    cdf.to_csv(OUT / f"cone_theta__{safe}.csv", index=False)

    print(f"Saved {OUT/f'boundary_margins__{safe}.csv'} and "
          f"{OUT/f'cone_theta__{safe}.csv'}  ({time.perf_counter()-t0:.0f}s)")

    print("\n--- A) blocking-competitor + margin (2 contexts x %d targets) ---" % len(tids))
    g = df.groupby("context").agg({"r0": "median", "margin0": "median",
                                   "cross_deg": "median", "arc_reachable": "mean"})
    print(g.round(3).to_string())
    print("\n--- B) theta_author (arc) vs theta_cell (decision cone), %d pairs ---" % len(cdf))
    print(f"  converged / full-vocabulary feasible: {converged} / {len(tids) * len(contexts)}")
    print(f"  max full-vocab violation: {max_violation:.2e}")
    med = cdf[["theta_author", "theta_cell"]].median()
    diff = (cdf["theta_author"] - cdf["theta_cell"]).median()
    ratio = (cdf["theta_author"] / cdf["theta_cell"]).median()
    both = cdf.dropna(subset=["theta_author", "theta_cell"])
    # Invariant: if the target-row arc reaches rank 1, the cone cannot be farther.
    invariant_violations = int((both["theta_cell"] > both["theta_author"] + 1e-4).sum())
    print(f"  median theta_author={med['theta_author']:.2f}°  theta_cell={med['theta_cell']:.2f}°  "
          f"diff={diff:.2f}°  ratio={ratio:.2f}x")
    print(f"  pairs where arc gave an angle: {(~cdf['theta_author'].isna()).mean():.0%}; "
          f"cone gave an angle: {(~cdf['theta_cell'].isna()).mean():.0%}")
    print(f"  invariant violations (theta_cell > theta_author): {invariant_violations} / {len(both)}")
    ex = cdf[(cdf.theta_author.isna()) | ((cdf.theta_author - cdf.theta_cell) > 5)]
    print("  arc-unreachable-or-far-from-cone sample:", len(ex))


if __name__ == "__main__":
    main()