#!/usr/bin/env python3
"""eval_antisteer.py — the controller's REPULSION side (avoid-set control).

Gemma-3-1B only, 1 forward + ~6 rotations, <=10s.

Readout steering raises a target; ANTI-steer should lower it symmetrically:
rotate the final state AWAY from the avoid-token (along -tau_hat) by a and
measure the avoid-token's rank + logit collapse. If symmetric, the margin
law's potential is bidirectional: steer (attract) and anti-steer (repel)
are the same rotation, opposite signs -> a complete control primitive.

Tests (avoid targets: a few words from the pool):
  a = 2*gap/97 + 0.02 (repel budget), -tau direction
  metric: avoid-token rank before/after (256K-vocab rank via GPU),
          logit drop, and the NEW argmax (what surfaces when a topic word
          is suppressed - does the model fall back to native?).

Run: timeout 60 python3 -u eval_antisteer.py  # GEMMA-3-1B
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
AVOID = ['chicken', 'pizza', 'ocean']
SLOPE = 97.0


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()
    NL = model.config.num_hidden_layers

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    cf = {}

    def c(m, i, o):
        cf['v'] = o[0, -1, :].float()

    h = model.model.norm.register_forward_hook(c)
    with torch.no_grad():
        L0 = model(ids).logits[0, -1].float()
    h.remove()
    native = int(L0.argmax())
    vf = cf['v'].float()
    vfn = vf / vf.norm()
    Wn = W[native].float()

    def rank_of(L, tid):
        return int((L > L[tid]).sum().item()) + 1

    print(f"[{MODEL}] {PROMPT!r} native={tok.decode([native])!r}  "
          f"ANTI-steer ({SLOPE}-law repel budget)")
    print(f"  {'avoid':>8} {'gap':>6} {'r0':>5} {'r_anti':>7} "
          f"{'l0':>7} {'l_anti':>7} {'new_argmax':>12}")
    for w in AVOID:
        tid = int(tok(' ' + w, add_special_tokens=False).input_ids[0])
        gap0 = float(L0[native] - L0[tid])
        r0 = rank_of(L0, tid)
        a_repel = 2 * gap0 / SLOPE + 0.02
        Wt = W[tid].float()
        tau = Wt - (vfn @ Wt) * vfn
        g = tau / tau.norm()
        # ANTI: rotate opposite the tangent (toward -g)
        v_rep = (vfn * math.cos(a_repel) - g * math.sin(a_repel)) * \
            vf.norm()

        def inj(m, i, o, p=v_rep):
            out = o.clone()
            out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                            device=out.device)
            return out

        hi = model.model.norm.register_forward_hook(inj)
        try:
            with torch.no_grad():
                L_ = model(ids).logits[0, -1].float()
        finally:
            hi.remove()
        r_a = rank_of(L_, tid)
        l_a = float(L_[tid])
        l0 = float(L0[tid])
        newarg = tok.decode([int(L_.argmax())])
        print(f"  {w:>8} {gap0:>6.2f} {r0:>5} {r_a:>7} {l0:>+7.2f} "
              f"{l_a:>+7.2f} {newarg!r:>12}", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()