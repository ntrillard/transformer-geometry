#!/usr/bin/env python3
"""Model-free verification of the fixed-score off-arc contract.

Reproduces, on synthetic rows with realistic low-rank structure, the two
facts established on real rows in the thread:

  1. the committed toward-blocker direction q = W_b - (W_b.s)s does NOT
     satisfy q _|_ r (r = residual axis), so renormalizing the endpoint
     rescales the s component and drifts the target score. On real-row
     geometry this reaches max |q.r| ~ 0.5 and max score drift 0.02-0.03.
  2. the exact construction q = W_b - (W_b.s)s - (W_b.r)r restores the
     fixed-score contract to machine precision, while the toward/away rank
     conclusions are UNCHANGED: essentially the same fraction of rank-1
     targets is destroyed by moving toward the blocker under either
     construction.

Also checks the 2-D separator example (u=(1,0), Wt=(1,1), Wj=(0.9,10)):
target logit rises monotonically along the arc while the competitor
overtakes -- logit ascent is guaranteed, rank ascent is not.

Run:  python verify_offarc_contract.py        (numpy only, < 5 s)
"""
import math

import numpy as np


def unit(x):
    return x / np.linalg.norm(x)


def main():
    rng = np.random.default_rng(11)
    NS = 3000
    d, rank = 1024, 64
    eps = math.radians(8)
    A = rng.standard_normal((d, rank)) / math.sqrt(rank)   # shared low-rank subspace

    max_qr_c = drift_c = max_qr_e = drift_e = 0.0
    flip_c = flip_e = n_ahead = 0

    for _ in range(NS):
        u = unit(A @ rng.standard_normal(rank) + 0.2 * rng.standard_normal(d))
        s = unit(A @ rng.standard_normal(rank) + 0.05 * rng.standard_normal(d))
        phi = math.radians(rng.uniform(15, 50))
        tau = unit(s - (s @ u) * u)
        v0 = unit(math.cos(phi) * u + math.sin(phi) * tau)   # target-tangent endpoint
        gamma = float(v0 @ s)
        rv = v0 - gamma * s
        rho = np.linalg.norm(rv)
        r = rv / rho                                         # residual axis, r _|_ s
        Wc = unit(0.9 * v0 + 0.4 * (A @ rng.standard_normal(rank))
                  + 0.1 * rng.standard_normal(d))            # strongest-blocker-style row

        q_c = unit(Wc - (Wc @ s) * s)                        # committed:  s only
        q_e = unit(Wc - (Wc @ s) * s - (Wc @ r) * r)         # exact:      s and r
        max_qr_c = max(max_qr_c, abs(q_c @ r))
        max_qr_e = max(max_qr_e, abs(q_e @ r))
        for q, tag in ((q_c, "c"), (q_e, "e")):
            v = unit(gamma * s + rho * (math.cos(eps) * r + math.sin(eps) * q))
            drift = abs(v @ s - gamma)
            if tag == "c":
                drift_c = max(drift_c, drift)
            else:
                drift_e = max(drift_e, drift)

        if v0 @ s > v0 @ Wc:                                 # rank-1 candidate at endpoint
            n_ahead += 1
            v_c = unit(gamma * s + rho * (math.cos(eps) * r + math.sin(eps) * q_c))
            v_e = unit(gamma * s + rho * (math.cos(eps) * r + math.sin(eps) * q_e))
            flip_c += int(v_c @ s - v_c @ Wc < 0)            # toward-blocker destroys rank 1
            flip_e += int(v_e @ s - v_e @ Wc < 0)

    print(f"off-arc contract (eps = {math.degrees(eps):.0f} deg, n = {NS} synthetic rows)")
    print(f"  max |q.r|         committed = {max_qr_c:.3f}    exact = {max_qr_e:.2e}")
    print(f"  max score drift   committed = {drift_c:.4f}    exact = {drift_e:.2e}")
    print(f"  rank-1 destroyed by toward-blocker: committed {flip_c}/{n_ahead}   "
          f"exact {flip_e}/{n_ahead}")
    assert max_qr_e < 1e-12 and drift_e < 1e-12, "exact construction must be machine-eps"
    assert abs(flip_c - flip_e) <= max(2, 0.02 * n_ahead), \
        "qualitative conclusion must not depend on the fix"

    # 2-D separator: logit up, rank down
    u2 = np.array([1.0, 0.0])
    Wt = np.array([1.0, 1.0])
    Wj = np.array([0.9, 10.0])
    g = Wt - (Wt @ u2) * u2
    tau2 = g / np.linalg.norm(g)
    overtake = None
    for deg in np.linspace(0, 8, 4001):
        th = math.radians(deg)
        x = math.cos(th) * u2 + math.sin(th) * tau2
        if x @ Wj > x @ Wt and overtake is None:
            overtake = deg
    print(f"\n2-D separator: target logit {u2 @ Wt:.1f} > competitor {u2 @ Wj:.1f} at start, "
          f"competitor overtakes at theta = {overtake:.3f} deg")
    assert overtake is not None and 0 < overtake < 1
    print("-> logit ascent along the target tangent is guaranteed; rank ascent is not.")


if __name__ == "__main__":
    main()