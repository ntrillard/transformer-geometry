#!/usr/bin/env python3
"""eval_lowgap.py — the LAW'S LOW-GAP EDGE: does alpha* = gap/97 survive
as gap -> 0 (near-tied tokens), or does the regime have a floor?

Gemma-3-1B only, 1 forward + a handful of plant tests, <=10s.

The law was validated on the 60-word pool at gaps 15.7-24.2 (ratio
aexact/apred ~1.0, 3.4% spread). THIS probe goes where the boundary scan
could not: tokens whose logit is CLOSE to native (ranks 2..120 of the
native distribution, gaps from ~18 down to ~1).

Tests:
  1. RATIO CURVE: for each sampled token, aexact (true crossing from the
     closed form |atan2(-A,B)|) vs apred = gap/97. Bucket by gap and
     report mean ratio + std. If the law is linear everywhere, ratio ~1
     at every bucket; if it breaks at low gap (a floor or a plateau),
     the ratio bends.
  2. PLANT AT LOW GAP: apply the rank-1 budget 2*apred + 0.02 to tokens
     from each bucket (one-shot at final, 1 seed x 5 tok). Does plant
     hold as applied -> 0.02 when gap -> 0?

Run: timeout 60 python3 -u eval_lowgap.py  # GEMMA-3-1B
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
N_RANK = 120          # take ranks 2..N_RANK+1 of the native distribution
BUCKETS = [(0, 2), (2, 5), (5, 10), (10, 18), (18, 1e9)]
PLANT_PER_BUCKET = 3  # tokens tested for plant per bucket


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()

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

    # native top-(N_RANK+1) tokens (skip rank 1 = native itself)
    top_ids = L0.topk(N_RANK + 1).indices.cpu().numpy()
    tids = np.array([int(t) for t in top_ids[1:]])

    # vectorized aexact for all sampled tokens
    Wt = W[tids].float()
    A_ = torch.einsum('d,td->t', vfn, Wt - Wn[None, :])
    proj = torch.einsum('td,d->t', Wt, vfn)
    tau = Wt - proj[:, None] * vfn[None, :]
    B_ = torch.einsum('td,td->t', tau, Wt - Wn[None, :]) / \
        (tau.norm(dim=1) + 1e-12)
    aex = torch.abs(torch.atan2(-A_, B_)).float().cpu().numpy()
    gaps = (L0[native] - L0[tids]).float().cpu().numpy()
    apred = gaps / 97.0
    ratio = aex / (apred + 1e-12)
    r0 = np.array([int((L0 > L0[t]).sum().item()) + 1 for t in tids])

    print(f"[{MODEL}] {PROMPT!r} native={tok.decode([native])!r} "
          f"N={len(tids)} ranks 2..{N_RANK + 1}")
    print(f"  {'gap bucket':>12} {'n':>4} {'mean gap':>9} "
          f"{'ratio mean':>10} {'ratio std':>10} {'min aex':>9} "
          f"{'max aex':>9}")
    ratio_by_bucket = {}
    for (lo, hi) in BUCKETS:
        m = (gaps >= lo) & (gaps < hi)
        if m.sum() == 0:
            continue
        ratio_by_bucket[(lo, hi)] = tids[m]
        print(f"  {f'({lo}, {hi})':>12} {m.sum():>4} {gaps[m].mean():>9.2f}"
              f" {ratio[m].mean():>10.3f} {ratio[m].std():>10.3f} "
              f"{aex[m].min():>9.4f} {aex[m].max():>9.4f}", flush=True)

    # ---- PLANT at the rank-1 budget, sampled per bucket ----
    print(f"\n  PLANT at applied = 2*gap/97 + 0.02 (1 seed x 5 tok):")
    print(f"  {'tok':>10} {'gap':>6} {'apred':>6} {'aexact':>7} "
          f"{'applied':>8} {'r0':>5} {'plant':>6}")
    for (lo, hi) in BUCKETS:
        bucket_tids = ratio_by_bucket.get((lo, hi), [])
        if len(bucket_tids) == 0:
            continue
        # evenly sample from the bucket
        idx = np.linspace(0, len(bucket_tids) - 1, PLANT_PER_BUCKET,
                          dtype=int)
        for k in idx:
            tid = int(bucket_tids[k])
            gap0 = float(L0[native] - L0[tid])
            ae = float(aex[np.where(tids == tid)[0][0]])
            app = 2 * gap0 / 97.0 + 0.02
            # steer one-shot
            torch.manual_seed(0)
            ids2 = ids.clone()
            Wt_ = W[tid].float()
            tau_t = Wt_ - (vfn @ Wt_) * vfn
            g = tau_t / tau_t.norm()
            vv = (vfn * math.cos(app) + g * math.sin(app)) * vf.norm()

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
            plant = 1.0 if int(p.argmax()) == tid else 0.0
            print(f"  {tok.decode([tid]):>10} {gap0:>6.2f} "
                  f"{gap0 / 97.0:>6.3f} {ae:>7.4f} {app:>8.4f} "
                  f"{int((L0 > L0[tid]).sum().item()) + 1:>5} "
                  f"{plant:>6.1f}", flush=True)

    # ---- summary: law holds or breaks at low gap? ----
    low = ratio_by_bucket.get((0, 2), [])
    if len(low):
        lm = np.where(np.isin(tids, low))[0]
        rr = ratio[lm]
        print(f"\n  low-gap (<2) ratio: mean={rr.mean():.3f} "
              f"std={rr.std():.3f}  vs high-gap ratio ~1.0")
        print(f"  VERDICT: {'LAW HOLDS at low gap' if abs(rr.mean() - 1.0)
               < 0.3 else 'LAW BREAKS at low gap'}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()