#!/usr/bin/env python3
"""eval_gap_vocab.py — BIG LEAP at scale: the gap-only controller over
the ACCESSIBLE VOCABULARY (is steerability just a ranked margin list?).

Gemma-3-1B only, 1 forward + ~16 rotational probes, <=10s.

If alpha* = gap/97, then EVERY token with gap > threshold is steerable
with a = 2*gap/97 + 0.02 at the readout. The 'accessible vocabulary' =
a ranked margin list: sort the logits by gap, top-k are your toys.

Measures (word pool ~single-token from CLASSES):
  1. RANKED MARGIN LIST: top-16 accessible targets by gap, with their
     predicted a* (gap/97) and exact a* - the controller's menu.
  2. FULL-FLEET PLANT: steer EACH of the 16 (a=2*gap/97+0.02, one-shot
     final), plant-rate at 1 seed x 5 tok. Corr(predicted_crossing,
     plant) - does the margin list PREDICT the whole fleet?
  3. GAP THRESHOLD MAP: what is the smallest gap that still plants?
     (the rank-1 boundary from the law)

Run: timeout 60 python3 -u eval_gap_vocab.py  # GEMMA-3-1B
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT
from eval_nb_quick import CLASSES

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
TOP_K = 16
SEEDS = 1
NTOK = 5
SLOPE = 97.0


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()
    NL = model.config.num_hidden_layers

    # pool of single-token words
    pool = sorted({w for c in CLASSES.values() for w in c})
    tid_p = {}
    for w in pool:
        iid = tok(' ' + w, add_special_tokens=False).input_ids
        if len(iid) == 1:
            tid_p[w] = int(iid[0])
    tids = list(tid_p.values())
    names = list(tid_p.keys())

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

    gaps = (L0[native] - L0[tids]).float().cpu().numpy()
    order = np.argsort(-gaps)

    # exact a* for the top-k (vectorized)
    Wt = W[tids].float()
    A_ = torch.einsum('d,td->t', vfn, Wt - Wn[None, :])
    proj = torch.einsum('td,d->t', Wt, vfn)
    tau = Wt - proj[:, None] * vfn[None, :]
    B_ = torch.einsum('td,td->t', tau, Wt - Wn[None, :]) / \
        (tau.norm(dim=1) + 1e-12)
    aex = torch.abs(torch.atan2(-A_, B_)).float().cpu().numpy()

    print(f"[{MODEL}] {PROMPT!r} pool={len(tids)}  "
          f"top-{TOP_K} accessible (by gap):")
    print(f"  {'#':>2} {'word':>10} {'gap':>7} {'apred':>6} {'aexact':>6} "
          f"{'plant':>6} {'applied':>7}")
    planted = []
    for rank in range(TOP_K):
        ti = order[rank]
        w = names[ti]
        gap0 = gaps[ti]
        apred = gap0 / SLOPE
        ae = aex[ti]
        a_apply = 2 * apred + 0.02
        tid = tids[ti]

        # steer
        app = 0.0
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids2 = ids.clone()
            Wti = W[tid].float()
            tau_t = Wti - (vfn @ Wti) * vfn
            g = tau_t / tau_t.norm()
            vv = (vfn * math.cos(a_apply) + g * math.sin(a_apply)) * \
                vf.norm()

            def inj(m, i, o, p=vv):
                out = o.clone()
                out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                device=out.device)
                return out

            hi = model.model.norm.register_forward_hook(inj)
            try:
                with torch.no_grad():
                    L = model(ids2).logits[0, -1].float()
            finally:
                hi.remove()
            p = torch.softmax(L.float(), dim=0)
            nxt = int(p.argmax())
            if nxt == tid:
                app = 1.0
        planted.append(app)
        print(f"  {rank + 1:>2} {w:>10} {gap0:>7.2f} {apred:>6.3f} "
              f"{ae:>6.3f} {app:>6.1f} {a_apply:>7.3f}", flush=True)

    pl = np.array(planted)
    pred_ok = [1 if (2 * gaps[order[rank]] / SLOPE + 0.02) > aex[order[rank]]
               else 0 for rank in range(TOP_K)]
    match = float(np.mean(np.array(pred_ok) == pl))
    print(f"\n  fleet plant rate = {pl.mean():.2f}  "
          f"predictor match = {match:.2f}")

    # ---- BOUNDARY SCAN: walk down the gap list until plant fails ----
    print(f"  BOUNDARY SCAN (controller, ranks {TOP_K + 1}..40):")
    bpl = list(planted)
    bgaps = [gaps[order[r]] for r in range(TOP_K)]
    for rank in range(TOP_K, min(40, len(tids))):
        ti = order[rank]
        gap0 = gaps[ti]
        a_apply = 2 * gap0 / SLOPE + 0.02
        tid = tids[ti]
        app = 0.0
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids2 = ids.clone()
            Wti = W[tid].float()
            tau_t = Wti - (vfn @ Wti) * vfn
            g = tau_t / tau_t.norm()
            vv = (vfn * math.cos(a_apply) + g * math.sin(a_apply)) * \
                vf.norm()

            def inj2(m, i, o, p=vv):
                out = o.clone()
                out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                device=out.device)
                return out

            hi = model.model.norm.register_forward_hook(inj2)
            try:
                with torch.no_grad():
                    L = model(ids2).logits[0, -1].float()
            finally:
                hi.remove()
            if int(L.argmax()) == tid:
                app = 1.0
        bpl.append(app)
        bgaps.append(gap0)
        print(f"  rank{rank + 1:>2} {names[ti]:>10} gap={gap0:>6.2f} "
              f"a_applied={a_apply:>6.3f} plant={app:>4.0f}", flush=True)
    # the boundary: where plant drops
    fst_fail = None
    for r in range(len(bpl)):
        if bpl[r] == 0:
            fst_fail = bgaps[r]
            break
    if fst_fail is None:
        print(f"  NO failure in scanned range (min gap {min(bgaps):.2f})")
    else:
        print(f"  PLANT BOUNDARY: first failure at gap {fst_fail:.2f} "
              f"(ranks above steer, below don't)")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()