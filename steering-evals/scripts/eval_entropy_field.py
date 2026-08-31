#!/usr/bin/env python3
"""eval_entropy_field.py — BIG LEAP: uncertainty is SPATIAL (the crossing
field density sets the entropy).

Gemma-3-1B only, 1 forward x 10 prompts + GPU vectorized a*, <=10s.

The margin law (44e0d9b) showed confidence = distance to the NEAREST
crossing |a*(2nd)|. The leap: the FULL uncertainty (softmax entropy) is
the DENSITY of the crossing field around the state — how MANY token
directions sit near a crossing. High entropy (perplexity spike) = the
state sits at a MULTI-WAY crossing; low entropy = alone far from every
crossing. If true, the distribution collapse is spatial: 256K
probabilities <=> the point's position relative to 256K fixed directions.

Per prompt (10): entropy H = -sum p log p (full vocab); crossing field =
a*(t) for the top-200 logit tokens (vectorized GPU closed form);
summaries: nearest |a*| (margin-law baseline), density rho(eps)=fraction
with |a*|<eps for eps in {0.02, 0.05}, and the 10th-smallest |a*|.

Across the 10 prompts: corr(H, nearest) / corr(H, rho) / corr(H, k10).
rho beating nearest => uncertainty = multi-token spatial geometry.

Run: timeout 60 python3 -u eval_entropy_field.py  # GEMMA-3-1B
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = [
    'For dinner I made', 'I went to the store and bought',
    'The recipe calls for', 'In my kitchen I have',
    'There once was a chicken', 'My favorite meal is',
    'The restaurant served', 'For lunch I had',
    'Breakfast today was', 'At the market I found',
]
KTOP = 200


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()          # (V, d) fp16 GPU
    bias = model.lm_head.bias
    NL = model.config.num_hidden_layers

    rows = []
    for pidx, PROMPT in enumerate(PROMPTS):
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        cf = {}

        def c(m, i, o):
            cf['v'] = o[0, -1, :].float()

        h = model.model.norm.register_forward_hook(c)
        with torch.no_grad():
            L0 = model(ids).logits[0, -1].float()
        h.remove()
        vf = cf['v'].float()
        p = torch.softmax(L0.double(), dim=0)
        H = float(-(p * torch.log(p + 1e-30)).sum())
        native = int(L0.argmax())

        # top-200 logit tokens
        tids = torch.topk(L0, KTOP).indices
        Wt = W[tids].float()                          # (200, d) float32
        Wn = W[native].float()
        vn = (vf / vf.norm()).float()

        # A_t = vn . (W_t - W_n)   (GPU einsum)
        A = torch.einsum('d,td->t', vn, Wt - Wn[None, :])
        # per-token tau = W_t - (W_t . vn) vn
        proj = torch.einsum('td,d->t', Wt, vn)        # (200,)
        tau = Wt - proj[:, None] * vn[None, :]
        tnorm = tau.norm(dim=1) + 1e-12
        B = torch.einsum('td,td->t', tau, Wt - Wn[None, :]) / tnorm
        ast = torch.atan2(-A, B).float().cpu().numpy()  # (200,)
        ast[tids.cpu().tolist().index(native)] = float('inf')  # exclude native
        aabs = np.abs(ast)
        nearest = float(aabs.min())
        rho2 = float((aabs < 0.02).mean())
        rho5 = float((aabs < 0.05).mean())
        k10 = float(np.sort(aabs)[9])
        rows.append((H, nearest, rho2, rho5, k10, PROMPT))
        print(f"P{pidx:>2} {PROMPT!r:32} H={H:.3f}  nearest={nearest:.4f} "
              f"rho(0.02)={rho2:.3f} rho(0.05)={rho5:.3f} "
              f"k10={k10:.4f}  n={tok.decode([native])!r}", flush=True)

    arr = np.array([r[:5] for r in rows], dtype=float)
    H, nrst, r2, r5, k10 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    print(f"\n  UNCERTAINTY-FIELD corr across {len(PROMPTS)} prompts:")
    print(f"    corr(H, nearest|a*|)    = "
          f"{float(np.corrcoef(H, nrst)[0, 1]):+.3f}  (margin law, 1D)")
    print(f"    corr(H, rho(0.02))      = "
          f"{float(np.corrcoef(H, r2)[0, 1]):+.3f}  (multi-token density)")
    print(f"    corr(H, rho(0.05))      = "
          f"{float(np.corrcoef(H, r5)[0, 1]):+.3f}")
    print(f"    corr(H, 10th-smallest)  = "
          f"{float(np.corrcoef(H, k10)[0, 1]):+.3f}")
    # partial: does rho add beyond nearest?
    from numpy.linalg import lstsq
    X = np.stack([np.ones_like(H), nrst], 1)
    resid = H - X @ lstsq(X, H, rcond=None)[0]
    c_part = float(np.corrcoef(resid, r2)[0, 1])
    print(f"    PARTIAL corr(H, rho02 | nearest) = {c_part:+.3f} "
          f"(density adds beyond the 1D law?)")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()