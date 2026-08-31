#!/usr/bin/env python3
"""eval_margin_walk.py — BIG LEAP: the margin law as a DYNAMICAL law.

Gemma-3-1B only, one natural generation (6 tok) + one steered generation,
margin tracked per step, <=10s.

Tests, per generated step t (state v_t):
  |a*(2nd)| = crossing angle toward the running 2nd-place token (the
  model's own next-margin), computed with the CLOSED FORM (atan2).

  Q1 CONSTANCY of |a*(next-margin)| over a natural walk. Flat -> the
  model keeps a CONSTANT geometric calibration while generating
  (self-steering keeps itself at fixed margin).

  Q2 STEER-SHIFT: run a second walk with a small readout steer toward
  chicken (a_inj = 0.10, final norm): does |a*(next-margin)| DROP by
  ~a_inj vs natural (the injected steer consumes the margin)? If the
  margin law is dynamical, steering = subtracting geometric margin.

  Q3 diag: gap12 per step too (linear margin) + which token is n2.

Run: timeout 60 python3 -u eval_margin_walk.py  # GEMMA-3-1B
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
TARGET = ' chicken'
NTOK = 6
A_INJ = 0.25  # > a*(chicken)=0.177 so it flips


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    NL = model.config.num_hidden_layers
    tid_t = tok(TARGET, add_special_tokens=False).input_ids
    tid_t = int(tid_t[0])
    Wt = lm_w[tid_t].detach().float().cpu().numpy().astype(np.float64)

    def margin_at(vf, n1, n2):
        vfn = vf / (np.linalg.norm(vf) + 1e-12)
        Wn = lm_w[n1].detach().float().cpu().numpy().astype(np.float64)
        Ws = lm_w[n2].detach().float().cpu().numpy().astype(np.float64)
        A = float(vfn @ (Ws - Wn))
        tau = Ws - (vfn @ Ws) * vfn
        B = float(tau @ (Ws - Wn)) / (np.linalg.norm(tau) + 1e-12)
        return math.atan2(-A, B)

    def cap_final(ids):
        cf = {}

        def c(m, i, o):
            cf['v'] = o[0, -1, :].float()
        h = model.model.norm.register_forward_hook(c)
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        h.remove()
        return L, cf['v'].cpu().numpy().astype(np.float64)

    def run(steer=False):
        torch.manual_seed(0)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        arr = []
        for _ in range(NTOK):
            L, vf = cap_final(ids)
            p = torch.softmax(L.float(), dim=0)
            order = p.argsort(descending=True)
            n1 = int(order[0])
            n2 = int(order[1])
            a2 = margin_at(vf, n1, n2)
            gap = float(L[n1] - L[n2])
            arr.append((n1, n2, abs(a2), gap))
            # choose next: greedy (n1) unless steering increases chicken
            nxt = n1
            if steer:
                # apply the final steer toward chicken EVERY step
                vfn = vf / (np.linalg.norm(vf) + 1e-12)
                tau = Wt - (vfn @ Wt) * vfn
                g = tau / (np.linalg.norm(tau) + 1e-12)
                v2 = (vfn * math.cos(A_INJ) + g * math.sin(A_INJ)) * \
                    np.linalg.norm(vf)

                def inj(m, i, o, p=v2):
                    out = o.clone()
                    out[0, -1, :] = torch.as_tensor(
                        p, dtype=out.dtype, device=out.device)
                    return out
                hh = model.model.norm.register_forward_hook(inj)
                with torch.no_grad():
                    L2 = model(ids).logits[0, -1].float()
                hh.remove()
                nxt = int(L2.argmax())
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)],
                            dim=1)
        return arr, ids

    nat, _ = run(steer=False)
    ste, _ = run(steer=True)
    print(f"[{MODEL}] margin walk: natural vs steered(+{A_INJ}), {NTOK} tok")
    print(f"  {'step':>4} {'nat n1/n2':>14} {'|a2|_nat':>9} "
          f"{'gap_nat':>7} | {'st n1/n2':>14} {'|a2|_st':>9} {'gap_st':>7}")
    diffs = []
    for t in range(NTOK):
        n1n, n2n, a2n, gn = nat[t]
        n1s, n2s, a2s, gs = ste[t]
        d = a2n - a2s if np.isfinite(a2n) and np.isfinite(a2s) else float('nan')
        if np.isfinite(d):
            diffs.append(d)
        print(f"  {t:>4} {tok.decode([n1n])!r:>5}/{tok.decode([n2n])!r:>6} "
              f"{a2n:>9.4f} {gn:>+7.2f} | "
              f"{tok.decode([n1s])!r:>5}/{tok.decode([n2s])!r:>6} "
              f"{a2s:>9.4f} {gs:>+7.2f}", flush=True)
    print(f"\n  Q1 |a*(next-margin)| NATURAL: "
          f"mean={np.mean([a2 for (_, _, a2, _) in nat]):.4f} "
          f"std={np.std([a2 for (_, _, a2, _) in nat]):.4f}  "
          f"(flat std = constant calibration)")
    print(f"  Q2 |a*(next-margin)| STEERED: "
          f"mean={np.mean([a2 for (_, _, a2, _) in ste]):.4f}")
    if diffs:
        print(f"  Q2 STEER-SHIFT: mean(|a2|_nat - |a2|_st) = "
              f"{np.mean(diffs):+.4f}  (expect ~ +{A_INJ} if steering "
              f"consumes the margin)", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()