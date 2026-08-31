#!/usr/bin/env python3
"""eval_residue_structure.py — BIG LEAP: what does the OFF-FLOW residue encode?

Gemma-3-1B only, 1 forward x 4 prompts + numpy, <=10s.

Closes the 'decision is off-flow residue' thread (5e6bafe) by asking what
that residue IS:

  Q1  SHARED-L2-TEXTURE: if a big chunk of the residual logits is a
      COMMON 'language model prior' (the general continuation texture)
      it should SURVIVE across prompts. Compute, per prompt, residue =
      full_logits - proj(flow)_logits, then the pairwise correlation of
      residues across prompts. High corr = a shared off-flow prior
      (copying 'the' + prose rhythm); low = prompt-specific structure.

  Q2  SPECIFICITY: does the residue TRACK the native token? Residue of
      the FINAL state should go to the specific native row (the choice).
      Measure corr(residue_logits, onehot-ish): actually simpler - how
      many of the native token's logits come from the residue vs flow:
      share = |residue[native]| / (|residue[native]| + |flow[native]|).

  Q3  PROMPT-SWAP STABILITY: take the residue logits of prompt A and ADD
      them to the FLOW logits of prompt B (the injected decision) -> does
      the argmax become A's native? If yes: residue = the transferable
      decision; the rest is context noise.

  Q4  RESIDUE DIMENSIONALITY: PCA of the residue vectors across the 4
      prompts -> PR + top directions. Does the deciding residue itself
      collapse to 1-2 dims? (the true 'decision dim', now measured on the
      right object).

Run: timeout 60 python3 -u eval_residue_structure.py  # GEMMA-3-1B
"""
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = ['For dinner I made', 'I went to the store and bought',
           'There once was a chicken', 'My favorite meal is']


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()          # (V, d) fp16 GPU
    bias = model.lm_head.bias
    NL = model.config.num_hidden_layers
    K = 8

    ls_full, ls_flow = [], []
    nats = []
    Vsp = []
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
        nat = int(L0.argmax())
        nats.append(nat)
        vf = caps[NL].cpu().numpy().astype(np.float64)
        V = np.stack([caps[li].cpu().numpy().astype(np.float64)
                      for li in range(NL + 1)])
        Vc = V - V.mean(0)
        _, _, vt = np.linalg.svd(Vc, full_matrices=False)
        basis = vt[:K]
        Pk = basis.T @ basis
        vflow = Pk @ vf
        vres = vf - vflow
        lf = torch.mv(W, torch.as_tensor(vf, dtype=torch.float16,
                                         device=DEV)).float().cpu().numpy()
        lflow = torch.mv(W, torch.as_tensor(vflow, dtype=torch.float16,
                                            device=DEV)).float().cpu().numpy()
        lres = torch.mv(W, torch.as_tensor(vres, dtype=torch.float16,
                                           device=DEV)).float().cpu().numpy()
        ls_full.append(lf)
        ls_flow.append(lflow)
        ls_res = ls_full[-1] - ls_flow[-1]
        Vsp.append(vres)
        share = abs(ls_res[nat]) / (abs(ls_res[nat]) +
                                    abs(lflow[nat]) + 1e-12)
        print(f"P{pidx} {PROMPT!r:32} native={tok.decode([nat])!r:6} "
              f"res_share@{nat}={share:.2f}", flush=True)

    # ---- Q1: residue correlations across prompts ----
    print(f"\n  Q1 cross-prompt residue corr matrix:")
    for i in range(len(PROMPTS)):
        row = [float(np.corrcoef(ls_full[i], ls_full[i])[0, 1] * 0 +
                     np.corrcoef(ls_full[i], ls_full[i])[0, 1] /
                     (1 if i == i else 1)) for _ in range(0)]
        row = []
        for j in range(len(PROMPTS)):
            row.append(float(np.corrcoef(ls_full[i] - ls_flow[i],
                                         ls_full[j] - ls_flow[j])[0, 1])
                       if i != j else 1.0)
        print(f"    P{i}: {[f'{x:+.2f}' for x in row]}", flush=True)

    # mean off-diag
    off = [np.corrcoef(ls_full[i] - ls_flow[i],
                       ls_full[j] - ls_flow[j])[0, 1]
           for i in range(len(PROMPTS)) for j in range(i + 1, len(PROMPTS))]
    print(f"    mean resid corr across prompts = {np.mean(off):+.3f}")

    # ---- Q3: prompt-swap - add A's residue logits to B's flow logits ----
    print(f"\n  Q3 prompt-swap decision transfer (B_flow + A_res -> argmax?):")
    for ia in range(len(PROMPTS)):
        for ib in range(len(PROMPTS)):
            if ia == ib:
                continue
            ls = ls_flow[ib] + (ls_full[ia] - ls_flow[ia])
            am = int(ls.argmax())
            hit = am == nats[ia]
            print(f"    A=P{ia}({tok.decode([nats[ia]])!r}) onto "
                  f"B=P{ib}: argmax={tok.decode([am])!r}  hit={hit}",
                  flush=True)

    # ---- Q4: residue dim (PCA of the residue vectors, L2-normalized) ----
    Rs = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in Vsp])
    sv = np.linalg.svd(Rs, compute_uv=False)
    pr = float((sv.sum() ** 2) / ((sv ** 2).sum() + 1e-9))
    print(f"\n  Q4 residue-dir PCA (4x d): sv={np.round(sv, 2)}  PR={pr:.2f}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()