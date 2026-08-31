#!/usr/bin/env python3
"""Evidence for reply point 1.1 (and the delta-framing in 1.4/1.7).

Verifies the algebraic identity  (u + a*g)/||u + a*g|| == cos(d)*u + sin(d)*t
with  d = atan(a*||g||),  g = s - (s.u)u,  t = g/||g||,  over 200 random (u,s).
Used to confirm: tangent-step + renormalization == same-target great-circle
rotation (same endpoint family); a=0.3 => d<=atan(0.3)~16.7deg.
"""
import numpy as np

rng = np.random.default_rng(0)
worst = 0.0
misses = 0
for _ in range(200):
    u = rng.standard_normal(1536); u = u / np.linalg.norm(u)
    s = rng.standard_normal(1536); s = s / np.linalg.norm(s)
    g = s - (s @ u) * u
    tau = g / np.linalg.norm(g)
    alpha = 0.3
    endp = (u + alpha * g) / np.linalg.norm(u + alpha * g)
    delta = np.arctan(alpha * np.linalg.norm(g))
    rot = np.cos(delta) * u + np.sin(delta) * tau
    err = float(np.max(np.abs(endp - rot)))
    worst = max(worst, err)
    misses += err > 1e-10

print(f"identity check: {200-misses}/200 pass (max abs deviation {worst:.2e})")
print(f"budget framing: delta = atan(alpha*||g||) <= atan({alpha}) = "
      f"{np.degrees(np.arctan(alpha)):.1f} deg   (||g||<=1 for unit rows)")
assert misses == 0 and worst < 1e-9