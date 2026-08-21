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
def cone_angles(u, tids, Wn):
    """Shortest rotation into each target's decision cone C_t = {v:(W_t-W_j).v>=0 ∀j}.
    Active-set projection of u onto C_t with full-vocab violation checks (on GPU)."""
    Wn = Wn.float()
    dev = Wn.device
    u = torch.as_tensor(u, dtype=torch.float32, device=dev)
    tids = list(tids)
    K = len(tids)
    X = u[None, :].expand(K, -1).clone()
    act = [[] for _ in range(K)]
    for _ in range(MAX_ROUNDS):
        XW = X @ Wn.T                          # (K,V) x_k . w_j
        wtx = (Wn[tids] * X).sum(-1)           # (K,)   x_k . w_t
        marg = wtx[:, None] - XW               # (K,V) target-minus-competitor margins
        marg[torch.arange(K), torch.tensor(tids)] = float("inf")
        worst = marg.min(1).values             # most-negative margin per target
        if float(worst.min()) > -1e-6:         # every target already in its cone
            break
        changed = False
        for k in range(K):
            if float(worst[k]) < -1e-6 and len(act[k]) < 64:
                v = marg[k] < -1e-5
                n = min(2, int(v.sum().item()))
                top = torch.topk(marg[k][v], k=n, largest=False).indices
                add = [int(torch.nonzero(v).flatten()[i].item()) for i in top]
                for j in add:
                    if j not in act[k]:
                        act[k].append(j); changed = True
        if not changed:
            break
        for k in range(K):
            if act[k]:
                nrm = Wn[act[k]] - Wn[tids[k]]
                G = nrm @ nrm.T + 1e-8 * torch.eye(len(act[k]), dtype=torch.float32, device=dev)
                lam = torch.linalg.solve(G, nrm @ u)
                X[k] = u - nrm.T @ lam
    angles = np.full(K, np.nan)
    for k in range(K):
        xk = X[k]; n = float(xk.norm())
        cs = 1.0 if n < 1e-9 else float((xk * u).sum()) / n
        angles[k] = math.degrees(math.acos(min(1.0, max(0.0, cs))))
    return angles


def main():
    t0 = time.perf_counter()
    model, tok = M.load_model("Qwen/Qwen2-0.5B-Instruct", dtype="fp16")
    N = model.config.num_hidden_layers
    L = N - 1
    W = model.lm_head.weight.detach().float().cpu().numpy()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    Wn_t = torch.as_tensor(Wn, dtype=torch.float32)
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
    df.to_csv(OUT / "boundary_margins__Qwen2-0.5B.csv", index=False)

    # B) cone probe on the same (context, target) grid
    st = time.perf_counter()
    cones = []
    for ctx in contexts:
        u = np.asarray(states[ctx][L], dtype=np.float32)
        u = u / np.linalg.norm(u)
        A = u @ W.T
        angles = cone_angles(u, tids, Wn_t)          # theta_cell per target
        for k, t in enumerate(tids):
            # theta_author = arc crossing angle (analytic) within same budget
            tau = Wn[t] - (Wn[t] @ u) * u
            tau = tau / np.linalg.norm(tau)
            lo_j, hi_j = _enter_angles(A[t] - A, (tau @ W.T)[t] - (tau @ W.T), BUDGET)
            lo_all, hi_all = lo_j.max(), hi_j.min()
            arc = float(math.degrees(lo_all)) if (np.isfinite(lo_all) and lo_all <= hi_all
                                                  and lo_all <= float(BUDGET)) else np.nan
            cones.append(dict(context=ctx, target_id=t, target_text=dec(t),
                              theta_author=arc, theta_cell=float(angles[k])))
    cdf = pd.DataFrame(cones)
    cdf.to_csv(OUT / "cone_theta__Qwen2-0.5B.csv", index=False)

    print(f"Saved {OUT/'boundary_margins__Qwen2-0.5B.csv'} and "
          f"{OUT/'cone_theta__Qwen2-0.5B.csv'}  ({time.perf_counter()-t0:.0f}s)")

    print("\n--- A) blocking-competitor + margin (2 contexts x %d targets) ---" % len(tids))
    g = df.groupby("context").agg({"r0": "median", "margin0": "median",
                                   "cross_deg": "median", "arc_reachable": "mean"})
    print(g.round(3).to_string())
    print("\n--- B) theta_author (arc) vs theta_cell (decision cone), %d pairs ---" % len(cdf))
    med = cdf[["theta_author", "theta_cell"]].median()
    diff = (cdf["theta_author"] - cdf["theta_cell"]).median()
    ratio = (cdf["theta_author"] / cdf["theta_cell"]).median()
    both = cdf.dropna()
    arconly = int(((~cdf["theta_author"].isna()) & cdf["theta_cell"].isna()).sum()) if False else "n/a"
    print(f"  median theta_author={med['theta_author']:.2f}°  theta_cell={med['theta_cell']:.2f}°  "
          f"diff={diff:.2f}°  ratio={ratio:.2f}x")
    print(f"  pairs where arc gave an angle: {(~cdf['theta_author'].isna()).mean():.0%}; "
          f"cone gave an angle: {(~cdf['theta_cell'].isna()).mean():.0%}")
    ex = cdf[(cdf.theta_author.isna()) | ((cdf.theta_author - cdf.theta_cell) > 5)]
    print("  arc-unreachable-or-far-from-cone sample:", len(ex))


if __name__ == "__main__":
    main()