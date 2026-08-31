#!/usr/bin/env python3
"""eval_field_nav.py — BIG LEAP: the model NAVIGATES the crossing field.

Gemma-3-1B only, one 6-token generation + vectorized field probes, <=10s.

Capstones the entropy-field law (28eef0c): if uncertainty is crossing-
density, then a generation is a POLICY over that field. Tests:

  per step t: state v_t, top-200 field, rho(0.02) = local density,
  k10 = 10th-nearest crossing. Then AFTER the token at the NEXT position
  (the state after choosing token u):
    Q1 CHOICE-DENSITY: is the density AT the next position (rho after
       moving to u) LOWER than the density at the current position?
       i.e. the model picks tokens that MOVE IT TO EMPTIER field regions
       (avoid dense crossings -> lower perplexity continuation).
    Q2 MARGIN-HOLD: k10 stays ~constant across the walk (the constant-
       calibration law, in multi-token form).
    Q3 FIELD-STEP: the chosen u's direction vs the local field: does the
       state AFTER u sit at lower rho than the CURRENT rho?
       (probe the actual next state, not the u-direction itself).

Run: timeout 60 python3 -u eval_field_nav.py  # GEMMA-3-1B
"""
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
NTOK = 6
KTOP = 200
EPS = 0.02


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()
    NL = model.config.num_hidden_layers

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    steps = []
    for _ in range(NTOK + 1):
        cf = {}

        def c(m, i, o):
            cf['v'] = o[0, -1, :].float()

        h = model.model.norm.register_forward_hook(c)
        with torch.no_grad():
            L0 = model(ids).logits[0, -1].float()
        h.remove()
        vf = (cf['v'] / cf['v'].norm()).float()
        native = int(L0.argmax())
        p = torch.softmax(L0.float(), dim=0)
        nxt = int(p.argmax())

        tids = torch.topk(L0, KTOP).indices
        Wt = W[tids].float()
        Wn = W[native].float()
        A = torch.einsum('d,td->t', vf, Wt - Wn[None, :])
        proj = torch.einsum('td,d->t', Wt, vf)
        tau = Wt - proj[:, None] * vf[None, :]
        B = torch.einsum('td,td->t', tau, Wt - Wn[None, :]) / \
            (tau.norm(dim=1) + 1e-12)
        ast = torch.atan2(-A, B).float().cpu().numpy()
        ast[tids.cpu().tolist().index(native)] = float('inf')
        aabs = np.abs(ast)
        rho = float((aabs < EPS).mean())
        k10 = float(np.sort(aabs)[9])

        steps.append((native, nxt, rho, k10, tok.decode([nxt])))

        if len(steps) > NTOK:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)

    print(f"[{MODEL}] {PROMPT!r}: field-nav over {NTOK} generated tokens")
    print(f"  {'step':>4} {'chose':>10} {'rho_now':>8} {'k10_now':>8} "
          f"{'rho_next':>9} {'k10_next':>9}")
    for t in range(NTOK):
        rn, kn = steps[t][2], steps[t][3]
        rf, kf = steps[t + 1][2], steps[t + 1][3]
        d_rho = rn - rf
        d_k = kf - kn
        print(f"  {t:>4} {steps[t][4]!r:>10} {rn:>8.4f} {kn:>8.4f} "
              f"{rf:>9.4f} {kf:>9.4f}   drho={d_rho:+.4f} "
              f"dk={d_k:+.4f}", flush=True)

    rho_cur = np.array([steps[t][2] for t in range(NTOK)])
    rho_nxt = np.array([steps[t + 1][2] for t in range(NTOK)])
    k_cur = np.array([steps[t][3] for t in range(NTOK)])
    k_nxt = np.array([steps[t + 1][3] for t in range(NTOK)])
    print(f"\n  Q1 CHOICE-DENSITY: mean(drho = rho_now - rho_next) = "
          f"{np.mean(rho_cur - rho_nxt):+.4f}")
    print(f"    (positive = the model moves to EMPTIER field regions)")
    print(f"  Q2 MARGIN-HOLD: k10 mean={np.mean(k_cur):.4f} "
          f"std={np.std(k_cur):.4f}  next-mean={np.mean(k_nxt):.4f}")
    frac_drop = np.mean(rho_nxt < rho_cur)
    print(f"    fraction of steps where rho DROPS = {frac_drop:.2f}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()