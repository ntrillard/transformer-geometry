#!/usr/bin/env python3
"""eval_logit_gap.py — FAST: logit-gap response to alpha (stiffness test).

Tests the 1d8f5a5 hypothesis: Qwen is stiff because the SAME alpha angle
produces a smaller logit-gap at the readout. Measures, per model/depth:
  gap(alpha) = logits[target] - logits[native_argmax]   (the currency)
  align      = cos(residual, target_row), ||residual||  (the armor)
Native argmax computed from the alpha=0 forward (fixed reference).
Depth: final (model.model.norm hook) and L10 (0-based 9).

Run: timeout 90 python3 -u eval_logit_gap.py google/gemma-3-1b-it
     timeout 90 python3 -u eval_logit_gap.py Qwen/Qwen2-0.5B-Instruct
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
TARGET = 'chicken'
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6]


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight

    tid = tok(' ' + TARGET, add_special_tokens=False).input_ids
    if len(tid) != 1:
        print("target not single token; abort")
        return
    tid = int(tid[0])
    Wt = lm_w[tid].detach().float().cpu().numpy()
    capid = tok(' ' + TARGET.capitalize(), add_special_tokens=False).input_ids
    cap_tid = int(capid[0]) if capid else None
    caplist = [cap_tid] if cap_tid is not None else [tid]

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)

    # native reference (alpha=0, no hook)
    with torch.no_grad():
        L0 = model(ids).logits[0, -1].float()
    native = int(L0.argmax())

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

    def run_depth(depth_name, hook_layers, alpha):
        hook = None
        if hook_layers is not None:
            hook = hook_layers.register_forward_hook(
                lambda m, i, o: rot(o, alpha, Wt))
        try:
            with torch.no_grad():
                L = model(ids).logits[0, -1].float()
                out = None
        finally:
            if hook is not None:
                hook.remove()
        # residual at the hook point for alignment (if depth=final only)
        return L

    # alignment currency at the readout: out@norm vs target row
    # grab the residual by hooking the norm output and capturing it
    resid_cap = {}

    def cap_norm(m, i, o):
        resid_cap['v'] = o[0, -1, :].float()

    h = model.model.norm.register_forward_hook(cap_norm)
    with torch.no_grad():
        L_check = model(ids).logits[0, -1].float()
    h.remove()
    v = resid_cap['v'].cpu().numpy()
    vn = v / np.linalg.norm(v)
    align_t = float(np.dot(vn, Wt / np.linalg.norm(Wt)))
    print(f"[{MODEL}] prompt={PROMPT!r} target={TARGET!r} "
          f"native={native!r} {tok.decode([native])!r}")
    print(f"  readout residual: ||v||={np.linalg.norm(v):.2f}  "
          f"cos(v,target)={align_t:.3f}  "
          f"gap0={float((L0[tid] - L0[native])):.2f}", flush=True)

    depths = [('final', model.model.norm), ('L10', model.model.layers[9])]
    for dname, layer in depths:
        print(f"  depth {dname}:", flush=True)
        for alpha in ALPHAS:
            hook = layer.register_forward_hook(
                lambda m, i, o, a=alpha: rot(o, a, Wt))
            try:
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
            finally:
                hook.remove()
            gap = float(L[tid] - L[native])
            rank = int((L.argsort(descending=True) == tid).nonzero()[0])
            pres = any((L.argsort(descending=True)[:10] == c).any() for c in caplist)
            print(f"    a={alpha:.2f}  gap={gap:+6.2f}  rank_t={rank:>3}  "
                  f"top10={pres}", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()