#!/usr/bin/env python3
"""Model-free unit checks of the steering-geometry harness.

Exercises the actual shipped code in steering_geometry_test.py on random,
UNTRAINED matrices. Because nothing was learned in these rows, the result
isolates the LM-head decision-partition geometry from any training/RLHF
effect:

  1. the analytic first-rank-1 angle agrees with the 200-step scan (within
     scan resolution) on a random (u, s) sample;
  2. the batched GPU path on random rows reproduces the cross-family
     qualitative signature: target tangent high rank-1, wrong-target 0%,
     random tangent ~0%, toward-blocker collapse, away-blocker restore;
  3. the committed toward-blocker branch violates the fixed-score contract
     (q not orthogonal to the residual axis r), shown here on fp32-random
     rows with the same relative magnitude as real rows.

Run:  python verify_harness_units.py        (no GPU/model required, < 30 s)
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import steering_geometry_test as M


def main():
    rng = np.random.default_rng(1234)
    d, V = 256, 2000
    W = rng.standard_normal((V, d)).astype(np.float32)
    Wn = (W / np.linalg.norm(W, axis=1, keepdims=True)).astype(np.float32)
    u = rng.standard_normal(d).astype(np.float32)
    u = u / np.linalg.norm(u)

    # 1. analytic vs 200-step scan
    t0 = time.time()
    diffs = []
    for _ in range(60):
        t = int(rng.integers(0, V))
        s = Wn[t]
        a = M.first_rank1_angle(u, s, W, t, max_delta=math.radians(17), use_scan=False)
        b = M.first_rank1_angle(u, s, W, t, max_delta=math.radians(17),
                                use_scan=True, n_steps=200)
        if a is None or b is None:
            assert (a is None) == (b is None), "reachability disagrees analytic vs scan"
            continue
        diffs.append(abs(a - b))
    step = math.degrees(math.radians(17)) / 200
    diffs = np.array(diffs)
    ok = bool((diffs <= step).all())
    print(f"[1] analytic vs 200-step scan: {len(diffs)} reachable cases, "
          f"max |diff| = {diffs.max():.3f} deg (scan step {step:.3f} deg) -> "
          f"{'OK, within scan resolution' if ok else 'MISMATCH'}")
    assert ok

    # 2. batched path on random matrices -> signature
    K = 64
    tid_idx = torch.tensor(list(range(K)))
    m = M._batched_block(torch.as_tensor(u), torch.as_tensor(W), torch.as_tensor(Wn),
                         tid_idx, math.radians(17), seed=42)
    rate = {k: float((torch.as_tensor(m[k]) == 1).float().mean()) for k in
            ("r_tan", "r_wrong", "r_rand", "r_off_random", "r_off_toward", "r_off_away")}
    print(f"[2] batched path, random rows (K={K}): "
          + "  ".join(f"{k}={100*rate[k]:.1f}%" for k in rate))
    assert rate["r_wrong"] < 0.05 and rate["r_rand"] < 0.05
    assert rate["r_tan"] > 0.5
    assert rate["r_off_away"] > rate["r_off_toward"]

    # 3. contract violation on fp32-random geometry (numpy twin of committed branch)
    NS = 2000
    max_qr = drift = 0.0
    W64 = W.astype(np.float64)
    uu = u.astype(np.float64)
    for _ in range(NS):
        S = Wn[int(rng.integers(0, V))].astype(np.float64)
        tau = M.tangent_direction(uu, S)
        v0 = math.cos(math.radians(17)) * uu + math.sin(math.radians(17)) * tau
        gamma = float(v0 @ S)
        rv = v0 - gamma * S
        rho = np.linalg.norm(rv)
        r = rv / rho
        comp = W64[int(rng.integers(0, V))]
        q = comp - (comp @ S) * S                       # committed toward construction
        q = q / np.linalg.norm(q)
        max_qr = max(max_qr, abs(q @ r))
        v = gamma * S + rho * (math.cos(math.radians(8)) * r + math.sin(math.radians(8)) * q)
        v = v / np.linalg.norm(v)
        drift = max(drift, abs(v @ S - gamma))
    print(f"[3] committed toward-branch contract on random rows: "
          f"max |q.r| = {max_qr:.3f}, max score drift = {drift:.4f} "
          f"(fixed-score contract requires ~0; real-row magnitude ~0.5 / 0.02-0.03)")
    assert max_qr > 0.05, "contract violation must be visible"

    print(f"\nall harness unit checks passed in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()