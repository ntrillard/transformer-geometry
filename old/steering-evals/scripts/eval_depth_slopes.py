#!/usr/bin/env python3
"""eval_depth_slopes.py — FAST: per-depth logit-gap SLOPES = analytic riser.

Follows deb2105: slope(d) = d(gap)/dalpha at depth d predicts the plant
knee everywhere: alpha*(d) = -gap0 / slope(d). The riser should be the
1/slope(d) curve. Also tests the paris (cross-category) damping: does its
lower L10 plant come from a lower slope at that depth?

Measures, per model, per depth {2,6,8,10,12,14,18,22,final}, per target
{chicken (same-cat food), paris (cross-cat)}: gap at alpha=0..0.3 -> slope.

Run: timeout 90 python3 -u eval_depth_slopes.py google/gemma-3-1b-it
     timeout 90 python3 -u eval_depth_slopes.py Qwen/Qwen2-0.5B-Instruct
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
ALPHAS = [0.0, 0.1, 0.2, 0.3]
TARGETS = ['chicken', 'paris']


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    NL = model.config.num_hidden_layers

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    with torch.no_grad():
        L0 = model(ids).logits[0, -1].float()
    native = int(L0.argmax())

    targets = {}
    for tname in TARGETS:
        tids = tok(' ' + tname, add_special_tokens=False).input_ids
        if len(tids) != 1:
            print(f"  target {tname!r} not single token; skip")
            continue
        tid = int(tids[0])
        targets[tname] = (tid, lm_w[tid].detach().float().cpu().numpy(),
                          float(L0[tid] - L0[native]))

    def rot(out, alpha, trow):
        v = out[:, -1, :].float().reshape(-1)
        vn = v / v.norm()
        t = trow - (trow @ vn.cpu().numpy()) * vn.cpu().numpy()
        t = t / (np.linalg.norm(t) + 1e-12)
        tg = torch.as_tensor(t, dtype=torch.float32, device=DEV)
        g = tg - (tg @ vn) * vn
        g = g / (g.norm() + 1e-8)
        v2 = vn * math.cos(alpha) + g * math.sin(alpha)
        out = out.clone()
        out[:, -1, :] = (v.norm() * v2).to(out.dtype)
        return out

    depths = [('L2', model.model.layers[1]), ('L6', model.model.layers[5]),
              ('L8', model.model.layers[7]), ('L10', model.model.layers[9]),
              ('L12', model.model.layers[11]), ('L14', model.model.layers[13]),
              ('L18', model.model.layers[17]), ('L22', model.model.layers[21])]
    if NL >= 27:
        depths.append(('L26', model.model.layers[25]))
    depths.append(('final', model.model.norm))

    print(f"[{MODEL}] native={native!r} {tok.decode([native])!r}")
    hdr = f"{'depth':>6}"
    for tname in targets:
        hdr += f"  {tname:>8} {'gap0':>6} {'slope':>7} {'a*':>6}"
    print(hdr)
    for dname, layer in depths:
        line = f"{dname:>6}"
        for tname, (tid, trow, gap0) in targets.items():
            gaps = []
            for alpha in ALPHAS:
                hook = layer.register_forward_hook(
                    lambda m, i, o, a=alpha, r=trow: rot(o, a, r))
                try:
                    with torch.no_grad():
                        L = model(ids).logits[0, -1].float()
                finally:
                    hook.remove()
                gaps.append(float(L[tid] - L[native]))
            # slope from alpha=0..0.3 linear fit (fixed intercept-ish)
            slope = (gaps[-1] - gaps[0]) / (ALPHAS[-1] - ALPHAS[0])
            astar = -gap0 / max(slope, 1e-9)
            line += (f"  {tname:>8} {gap0:>+6.2f} {slope:>+7.1f} "
                     f"{astar:>6.2f}")
        print(line, flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()