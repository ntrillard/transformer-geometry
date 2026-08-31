#!/usr/bin/env python3
"""eval_law_controller.py — CLOSED-LOOP: verify the readout law (alpha* =
gap/97) during generation AND use it as the only steering controller.

Gemma-3-1B only, 1 walk (5 tok) + 6 targets x 2 seeds x 8 tok, <=10s.

Phase A SLIDING SLOPE: during a natural 5-token walk, per step:
  slope(t) = ||vf|| * |B_hat(t)| where B_hat = the closed-form B for the
  running 2nd-place token. Sliding law: slope(t) ~ 97 +- 5% (the ratio
  field alpha*/gap is state-stable -> the law is dynamical).

Phase B GAP-ONLY CONTROLLER: for 6 targets, compute gap0(t) (one sort of
the logits), predict alpha*_p = gap0/97, rotate the final state by
2*alpha*_p + 0.02 (the validated rank-1 budget), sample 2 seeds x 8 tok.
Predict planted = (alpha_applied > alpha*(exact)); compare with the actual
plant rate. If plant tracks prediction -> the law IS the controller:
one sort + one division -> steerable targets, analytically.

Run: timeout 60 python3 -u eval_law_controller.py  # GEMMA-3-1B
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = sys.argv[2] if len(sys.argv) > 2 else 'For dinner I made'
TARGETS = ['chicken', 'pizza', 'bread', 'paris', 'cake', 'ocean']
WALK = 5
SEEDS = 2
NTOK = 8


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()
    NL = model.config.num_hidden_layers

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)

    # ---------- Phase A: sliding slope during a walk ----------
    print(f"[{MODEL}] {PROMPT!r}  Phase A: sliding slope, {WALK} tokens")
    tids_l = {}
    for w in TARGETS:
        iid = tok(' ' + w, add_special_tokens=False).input_ids
        if len(iid) == 1:
            tids_l[w] = int(iid[0])
    slopes = []
    for step in range(WALK):
        cf = {}

        def c(m, i, o):
            cf['v'] = o[0, -1, :].float()

        h = model.model.norm.register_forward_hook(c)
        with torch.no_grad():
            L0 = model(ids).logits[0, -1].float()
        h.remove()
        vf = (cf['v'] / cf['v'].norm()).float()
        n1 = int(L0.argmax())
        n2 = int(torch.topk(L0, 2).indices[1])
        Wn, Ws = W[n1].float(), W[n2].float()
        A_ = float(vf @ (Ws - Wn))
        tau = Ws - (vf @ Ws) * vf
        B_ = float(float(tau @ (Ws - Wn)) / (float(tau.norm()) + 1e-12))
        slope = float(np.linalg.norm(cf['v'].cpu().numpy())) * abs(B_)
        slopes.append(slope)
        a2 = math.atan2(-A_, B_)
        print(f"  step{step}: n1={tok.decode([n1])!r} n2={tok.decode([n2])!r} "
              f"slope={slope:7.1f} a2={abs(a2):.4f}", flush=True)
        nxt = int(L0.argmax())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    sl = np.array(slopes)
    print(f"  sliding slope: mean={sl.mean():.1f} std={sl.std():.1f} "
          f"({sl.std() / sl.mean() * 100:.1f}%)")

    # ---------- Phase B: gap-only controller ----------
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

    print(f"\n  Phase B: gap-only controller (a = 2*gap/97 + 0.02):")
    print(f"  {'target':>8} {'gap':>6} {'apred':>6} {'aexact':>6} "
          f"{'pred':>5} {'plant':>5}")
    n_ok = 0
    n_tot = 0
    for w in TARGETS:
        if w not in tids_l:
            continue
        tid = tids_l[w]
        gap0 = float(L0[native] - L0[tid])
        apred = gap0 / 97.0
        # exact alpha* (closed form)
        Wt = W[tid].float()
        Wn = W[native].float()
        A_ = float(vfn @ (Wt - Wn))
        tau = Wt - (vfn @ Wt) * vfn
        B_ = float(float(tau @ (Wt - Wn)) / (float(tau.norm()) + 1e-12))
        aexact = abs(math.atan2(-A_, B_))
        a_apply = 2 * apred + 0.02
        pred = float(a_apply > aexact)

        # rotate final state by a_apply and sample
        plants = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids2 = ids.clone()
            appeared = 0.0
            for _ in range(NTOK):
                # final-norm hook rotation
                vv = (vfn * math.cos(a_apply) +
                      (lambda g: g / g.norm())(
                          Wt - (vfn @ Wt) * vfn) * math.sin(a_apply)) * \
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
                q = p.clone(); order = q.argsort(descending=True)
                k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
                msk = torch.zeros_like(q); msk[order[:k]] = 1
                qq = (q * msk) / (q * msk).sum()
                nxt = int(torch.multinomial(qq, 1))
                if nxt == tid:
                    appeared = 1.0
                ids2 = torch.cat([ids2,
                                  torch.tensor([[nxt]], device=ids2.device)],
                                 dim=1)
            plants.append(appeared)
        plant = float(np.mean(plants))
        n_ok += int(np.round(plant) == pred)
        n_tot += 1
        print(f"  {w:>8} {gap0:>6.2f} {apred:>6.3f} {aexact:>6.3f} "
              f"{pred:>5.0f} {plant:>5.2f}", flush=True)
    print(f"  controller match: {n_ok}/{n_tot}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()