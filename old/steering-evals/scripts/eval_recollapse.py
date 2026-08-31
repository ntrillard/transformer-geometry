#!/usr/bin/env python3
"""eval_recollapse.py — BIG LEAP: the layers are RE-PLANARIZING machines.

Gemma-3-1B only, 1 base capture + ~6 injection forwards + 1 GPU eigh, <=10s.

Closes the collapse theory:
  1. HEAD SUBSPACE IDENTITY: top-3 eigenvectors of W_lm^T W_lm (GPU eigh)
     vs the trajectory PCA-3 -> overlap tells whether the universal plane
     IS the head's top eigenspace (layers align the residual to the
     head's amplification directions - the eigenalignment hypothesis).
  2. ALIGNMENT CURVE: al(l) = energy of v_l in head-top-3 / ||v_l||,
     per layer. Monotone rise -> the computation IS alignment-into-the-
     head-subspace. This is the 'collapse/normalize' in one curve.
  3. RE-COLLAPSE (the operator test of scramble): inject an OFF-PLANE
     unit perturbation at depth d (times 0.2*||v_d||), run to final,
     measure frac of the final-state delta that SURVIVES OFF the plane.
     Deep injection -> high survival (nothing left to replanarize) =
     scramble~1. Shallow injection -> low survival (re-collapsed) =
     scramble high. Replanarization = the scramble mechanism.

Run: timeout 60 python3 -u eval_recollapse.py  # GEMMA-3-1B
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
INJ_DEPTHS = [4, 9, 14, 20, 25]          # 0-based; last block layer (25)


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    lm_head = model.lm_head
    W = lm_head.weight.detach()           # (V, d) fp16 on GPU
    bias = lm_head.bias
    NL = model.config.num_hidden_layers

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    caps = {}

    def mk(li):
        def h(m, i, o):
            caps[li] = o[0, -1, :].float()
        return h

    hooks = [model.model.layers[li].register_forward_hook(mk(li))
             for li in range(NL)]
    hooks.append(model.model.norm.register_forward_hook(mk('f')))
    with torch.no_grad():
        L0 = model(ids).logits[0, -1].float()
    for h in hooks:
        h.remove()
    native = int(L0.argmax())
    vf = caps['f'].cpu().numpy()

    V = np.stack([caps[li].cpu().numpy() for li in range(NL)] +
                 [caps['f'].cpu().numpy()])                # (27, d)
    # trajectory PCA-3
    Vc = V - V.mean(0)
    _, _, vt = np.linalg.svd(Vc, full_matrices=False)
    B_t = vt[:3].astype(np.float32)                        # (3, d)

    # head top-3 via eigh(W^T W) on GPU
    Wf = W.float()
    A = Wf.T @ Wf                                           # (d, d)
    evals, evecs = torch.linalg.eigh(A)
    idx = torch.argsort(evals, descending=True)[:3]
    B_h = evecs[:, idx].T.float().cpu().numpy()            # (3, d)
    print(f"[{MODEL}] {PROMPT!r} native={tok.decode([native])!r}")
    print(f"  head top-3 evals: {evals[idx].sqrt().cpu().numpy().round(1)} "
          f"(sing. vals)")

    # ---- 1. overlap(plane, head-subspace) ----
    C = B_t @ B_h.T                                        # (3,3)
    ov = np.linalg.svd(C, compute_uv=False)
    print(f"  overlap(traj-PCA3, head-top3): cos = {np.round(ov, 3)}  "
          f"top={ov[0]:.3f}")

    # ---- 2. alignment curve ----
    Vg = torch.as_tensor(V, dtype=torch.float32, device=DEV)
    Bh_g = torch.as_tensor(B_h.T, dtype=torch.float32, device=DEV)  # (d,3)
    Pk = Bh_g @ Bh_g.T                                     # (d, d) projector
    proj = Vg @ Pk                                         # (27, d)
    nrml = Vg.norm(dim=1)
    al = (proj.norm(dim=1) / (nrml + 1e-9)).cpu().numpy()
    print(f"  ALIGNMENT CURVE (energy in head-top-3 / norm):")
    for li in (0, 4, 9, 14, 18, 22, 26):
        print(f"    L{li:>2}: al = {al[li]:.3f}", flush=True)
    print(f"    final: al = {al[-1]:.3f}  (1.0 = fully aligned)")

    # ---- 3. RE-COLLAPSE operator test ----
    # off-plane unit (null of B_t)
    rng = np.random.default_rng(1)
    n_off = rng.standard_normal(W.shape[1]).astype(np.float32)
    n_off -= B_t.T @ (B_t @ n_off)                          # orth to plane
    n_off = n_off / (np.linalg.norm(n_off) + 1e-12)
    print(f"  RE-COLLAPSE (off-plane injection survival at final):")
    print(f"    {'inj@L':>7} {'off_survive':>11} {'plane_survive':>13}")
    for d in INJ_DEPTHS:
        v_d = V[d]
        lam = 0.2 * float(np.linalg.norm(v_d))
        pert = (v_d + lam * n_off).astype(np.float32)

        def inject(m, i, o, p=pert):
            out = o.clone()
            out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                            device=out.device)
            return out

        h = model.model.layers[d].register_forward_hook(inject)
        capf = {}

        def cf(m, i, o):
            capf['v'] = o[0, -1, :].float()

        hf = model.model.norm.register_forward_hook(cf)
        try:
            with torch.no_grad():
                model(ids)
        finally:
            h.remove(); hf.remove()
        vfp = capf['v'].cpu().numpy()
        delta = vfp - vf
        off_frac = float(np.linalg.norm(delta - B_t.T @ (B_t @ delta)) /
                         (np.linalg.norm(delta) + 1e-12))
        print(f"    L{d:>3}      {off_frac:>11.3f} {1 - off_frac:>13.3f}",
              flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()