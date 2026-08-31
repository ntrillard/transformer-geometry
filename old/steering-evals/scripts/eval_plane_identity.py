#!/usr/bin/env python3
"""eval_plane_identity.py — BIG LEAP #3: is the collapse plane universal?

Follows eval_collapse_rank (PR 2.95: 27 states span ~3 dims) and
eval_plane_compat (targets 96% off-plane, native 27% in-plane). Now ask
WHAT the 3 axes ARE and whether they are MODEL-INTERNAL (same across
prompts -> the transformer's own attractor) or PROMPT-CONDITIONED.

3 prompts x one forward each (~8s total). Per prompt: PCA basis of the 27
states -> principal angles between prompt-planes (subspace overlap).
Then NAME the axes: cos(axis_k, v_final) (native readout dir), and
cos(axis_k, sink_vec) where sink_vec = v(L22)-v(L18) (the collapse
direction). If the top axis ~ the sink direction every prompt -> the
plane is THE collapse manifold, model-internal.

Run: timeout 60 python3 -u eval_plane_identity.py  # GEMMA-3-1B
"""
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = [
    'For dinner I made',
    'I went to the store and bought',
    'There once was a chicken',
]
NP = len(PROMPTS)


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    NL = model.config.num_hidden_layers
    layers = [model.model.layers[li] for li in range(NL)]
    layers.append(model.model.norm)

    Vlist, nativel = [], []
    for pidx, PROMPT in enumerate(PROMPTS):
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        caps = {}

        def mk(li):
            def h(m, i, o):
                caps[li] = o[0, -1, :].float()
            return h

        hooks = [layers[li].register_forward_hook(mk(li))
                 for li in range(NL + 1)]
        with torch.no_grad():
            L0 = model(ids).logits[0, -1].float()
        for h in hooks:
            h.remove()
        native = int(L0.argmax())
        nativel.append(native)
        V = np.stack([caps[li].cpu().numpy() for li in range(NL + 1)])
        Vlist.append(V)
        Vc = V - V.mean(0)
        _, s, vt = np.linalg.svd(Vc, full_matrices=False)
        prv = float((s.sum() ** 2) / ((s ** 2).sum()))
        print(f"P{pidx} {PROMPT!r:32} native={tok.decode([native])!r:8} "
              f"PR={prv:.2f}", flush=True)

    # ---- subspace overlap (principal angles via basis cross product) ----
    Bs = []
    for V in Vlist:
        Vc = V - V.mean(0)
        _, _, vt = np.linalg.svd(Vc, full_matrices=False)
        Bs.append(vt[:3])
    print("\n  principal-angle cosines between prompt-planes "
          "(1 = same plane):")
    for i in range(NP):
        for j in range(i + 1, NP):
            C = Bs[i] @ Bs[j].T          # (3,3) direction cosines
            sv = np.linalg.svd(C, compute_uv=False)
            ov = np.cos(np.arccos(np.clip(sv, -1, 1)))
            print(f"    P{i}-P{j}: cos = {ov[:3]}  (top={ov[0]:.3f})",
                  flush=True)

    # ---- name the axes: native dir + sink dir alignment ----
    print("\n  axis naming (cos(axis, dir)) per prompt:")
    for pidx, V in enumerate(Vlist):
        Vc = V - V.mean(0)
        _, _, vt = np.linalg.svd(Vc, full_matrices=False)
        B = vt[:3]
        vf = V[-1]
        vfn = vf / (np.linalg.norm(vf) + 1e-12)
        sink = V[22] - V[18]
        sinks = sink / (np.linalg.norm(sink) + 1e-12)
        c_nat = np.abs(B @ vfn)
        c_sink = np.abs(B @ sinks)
        print(f"    P{pidx}: vs native={np.round(c_nat, 3)}  "
              f"vs sinkdir={np.round(c_sink, 3)}", flush=True)

    # native readout dir vs sink dir overlap (are they the same axis?)
    cos_ns = []
    for V in Vlist:
        vf = V[-1]; vfn = vf / (np.linalg.norm(vf) + 1e-12)
        sink = V[22] - V[18]; sinks = sink / (np.linalg.norm(sink) + 1e-12)
        cos_ns.append(float(np.abs(vfn @ sinks)))
    print(f"  native_dir vs sink_dir |cos| = {np.round(cos_ns, 3)}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()