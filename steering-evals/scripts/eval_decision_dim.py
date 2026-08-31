#!/usr/bin/env python3
"""eval_decision_dim.py — BIG LEAP: the DECISION-RANK spectrum + tube-membership
as a steering predictor.

Gemma-3-1B only, 1 forward + numpy sweeps, <=10s.

Two linked questions closing the collapse story:
  A. DECISION RANK: for k = 2..8 principal dims of the residual trajectory,
     project the final state into the top-k subspace and measure how much
     of the DECISION survives: corr(full_logits, k-proj_logits) and
     argmax-match. The knee k* = the true decision dimension (the model's
     choice collapses to k* numbers).
  B. TUBE-MEMBERSHIP AS STEERABILITY: project each token head-row onto the
     tube basis (top-5), get the in-tube fraction. Correlate with the
     analytic alpha*@final per token. If in-tube rows steer cheap (low
     alpha*), the tube IS the steering-relevant manifold -> tube-membership
     is a learned proxy for steerability (a single cheap number per target).

Run: timeout 60 python3 -u eval_decision_dim.py  # GEMMA-3-1B
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = ['For dinner I made', 'I went to the store and bought',
          'There once was a chicken']
TOKENS = []  # membership part skipped in this universal run
K_MAX = 8
TUBE_K = 5


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()          # (V, d) fp16 GPU
    bias = model.lm_head.bias
    NL = model.config.num_hidden_layers

    print(f"[{MODEL}] DECISION-RANK per prompt (argmax match at each k):")
    print(f"  {'prompt':>36} k2/  k3/  k8")
    for pidx, PROMPT in enumerate(PROMPTS):
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        caps = {}

        def mk(li):
            def h(m, i, o):
                caps[li] = o[0, -1, :].float()
            return h

        hooks = [model.model.layers[li].register_forward_hook(mk(li))
                 for li in range(NL)]
        hooks.append(model.model.norm.register_forward_hook(mk(NL)))
        with torch.no_grad():
            L0 = model(ids).logits[0, -1].float()
        for h in hooks:
            h.remove()
        native = int(L0.argmax())
        vf = caps[NL].cpu().numpy().astype(np.float64)
        V = np.stack([caps[li].cpu().numpy().astype(np.float64)
                      for li in range(NL + 1)])     # (27, d)
        Vc = V - V.mean(0)
        _, _, vt = np.linalg.svd(Vc, full_matrices=False)
        basis = vt[:K_MAX]                           # (8, d)
        lf = torch.mv(W, torch.as_tensor(vf, dtype=torch.float16,
                                         device=DEV)).float().cpu().numpy()
        argmax_full = int(lf.argmax())
        ams, ccs = [], []
        for k in (2, 3, 8):
            Pk = basis[:k].T @ basis[:k]
            vf_k = Pk @ vf
            lk = torch.mv(W, torch.as_tensor(vf_k, dtype=torch.float16,
                                             device=DEV)).float().cpu().numpy()
            ams.append(int(lk.argmax()) == argmax_full)
            ccs.append(float(np.corrcoef(lf, lk)[0, 1]))
        print(f"  {PROMPT!r:36} {ams[0]:>2} {ccs[0]:+.2f}  {ams[1]:>2} "
              f"{ccs[1]:+.2f}  {ams[2]:>2} {ccs[2]:+.2f}   "
              f"(native={tok.decode([native])!r})", flush=True)

    # ---- B. tube-membership vs alpha* (skipped when TOKENS empty) ----
    if not TOKENS:
        print(f"[{time.time() - t0:.0f}s total]")
        return
    tids, tnames = [], []
    for w in TOKENS:
        iids = tok(w, add_special_tokens=False).input_ids
        if len(iids) == 1:
            tids.append(int(iids[0])); tnames.append(w.strip())
    Wt = W[tids].detach().float().cpu().numpy().astype(np.float64)
    Wn = W[native].detach().float().cpu().numpy().astype(np.float64)

    basis5 = basis[:TUBE_K]
    mems = np.linalg.norm(basis5 @ Wt.T, axis=0) / \
        (np.linalg.norm(Wt, axis=1) + 1e-12)
    vfn = vf / (np.linalg.norm(vf) + 1e-12)
    alphas = np.zeros(len(tids))
    for i in range(len(tids)):
        w = Wt[i]
        A = float(vfn @ (w - Wn))
        tau = w - (vfn @ w) * vfn
        B = float(tau @ (w - Wn)) / (np.linalg.norm(tau) + 1e-12)
        alphas[i] = math.atan2(-A, B)
    mem_nat = float(np.linalg.norm(basis5 @ Wn) / (np.linalg.norm(Wn) + 1e-12))

    print(f"\n  TUBE-MEMBERSHIP (k={TUBE_K}) vs alpha*@final:")
    print(f"    native mem = {mem_nat:.3f}")
    for i, tn in enumerate(tnames):
        print(f"    {tn:>8}: mem={mems[i]:.3f}  a*={alphas[i]:+.3f}",
              flush=True)
    cc_m = float(np.corrcoef(mems, alphas)[0, 1])
    cc_mabs = float(np.corrcoef(mems, np.abs(alphas))[0, 1])
    print(f"  corr(mem, a*)      = {cc_m:+.3f}")
    print(f"  corr(mem, |a*|)    = {cc_mabs:+.3f}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()