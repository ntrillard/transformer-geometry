#!/usr/bin/env python3
"""eval_recipe_cross.py — does the two-constant recipe (gap/97 steer +
fixed anti-last) transfer across TARGETS and PROMPTS, and is a_rep robust?

Gemma-3-1B only, 2 prompts x 3 targets (one+anti, 16 tok) + a_rep sweep
(base combo, 12 tok), <=10s.

The recipe from 1356f2d:
   steer ONCE (a_att = 2*gap/97 + 0.02 toward W_target at the final norm)
   then anti the just-sampled token every step (a_rep = 0.15, away)
Two constants: 97 (law) and 0.15 (ad hoc). Questions:
  A. CROSS-TRANSFER: plant/rep4/div for 3 targets x 2 prompts with fixed
     0.15. If the recipe holds everywhere, the controller is complete
     with no per-target state.
  B. A_REP ROBUSTNESS: sweep 0.05..0.30 on the base combo. If 0.15 sits
     on a plateau, the anti budget is not a tuned knob (good for a law-
     complete story); if it's a razor edge, the recipe needs recalibration.

Run: timeout 60 python3 -u eval_recipe_cross.py  # GEMMA-3-1B
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = ['For dinner I made', 'I went to the store and bought']
TARGETS = ['chicken', 'paris', 'ocean']
A_REP_BASE = 0.15
NTOK_A = 16
NTOK_B = 12


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()

    def cap(t):
        return [int(c) for c in tok(' ' + t.capitalize(),
                                    add_special_tokens=False).input_ids]

    def gen(ids0, vf, tid_t, capl, a_att, a_rep, nt, seeds):
        allres = []
        for sd in range(seeds):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(nt):
                vv = vf
                if step == 0:
                    vv = (vfn * math.cos(a_att) +
                          g_t * math.sin(a_att)) * vf.norm()
                if toks:
                    vv1 = vv / vv.norm()
                    Wl = W[toks[-1]].float()
                    tauv = Wl - (vv1 @ Wl) * vv1
                    gl_ = -tauv / tauv.norm()
                    vv = (vv1 * math.cos(a_rep) +
                          gl_ * math.sin(a_rep)) * vf.norm()

                def inj(m, i, o, p=vv):
                    out = o.clone()
                    out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                    device=out.device)
                    return out

                hi = model.model.norm.register_forward_hook(inj)
                try:
                    with torch.no_grad():
                        L = model(ids).logits[0, -1].float()
                finally:
                    hi.remove()
                p = torch.softmax(L.float(), dim=0)
                q = p.clone(); order = q.argsort(descending=True)
                k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
                msk = torch.zeros_like(q); msk[order[:k]] = 1
                qq = (q * msk) / (q * msk).sum()
                nxt = int(torch.multinomial(qq, 1))
                toks.append(int(nxt))
                ids = torch.cat([ids,
                                 torch.tensor([[nxt]], device=ids.device)],
                                dim=1)
            allres.append(toks)
        return allres

    # ---- A. cross matrix ----
    print(f"[{MODEL}] A. CROSS-TRANSFER (one+anti, a_rep={A_REP_BASE})")
    print(f"  {'prompt':>12} {'tgt':>8} {'gap':>6} {'a_att':>6} "
          f"{'plant':>6} {'rep4':>6} {'div':>6} {'#tgt':>5}")
    for PROMPT in PROMPTS:
        ids0 = tok(PROMPT, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        cf = {}

        def hook_c(m, i, o):
            cf['v'] = o[0, -1, :].float()

        h = model.model.norm.register_forward_hook(hook_c)
        with torch.no_grad():
            L0 = model(ids0).logits[0, -1].float()
        h.remove()
        native = int(L0.argmax())
        vf = cf['v'].float()
        vfn = vf / vf.norm()
        Wn = W[native].float()
        for TARGET in TARGETS:
            tid_t = int(tok(' ' + TARGET,
                            add_special_tokens=False).input_ids[0])
            gap_t = float(L0[native] - L0[tid_t])
            a_att = 2 * gap_t / 97.0 + 0.02
            Wt = W[tid_t].float()
            tau_t = Wt - (vfn @ Wt) * vfn
            g_t = tau_t / tau_t.norm()
            gs = gen(ids0, vf, tid_t, cap(TARGET), a_att, A_REP_BASE,
                     NTOK_A, 1)
            plant = np.mean([1.0 if (tid_t in g[:10] or
                                     any(c in g[:10] for c in cap(TARGET)))
                             else 0.0 for g in gs])
            rp = np.mean([rep4(g) for g in gs])
            dv = np.mean([len(set(g)) / len(g) for g in gs])
            ntg = np.mean([sum(1 for x in g if x == tid_t or x in capl)
                           for g in gs]) if False else \
                np.mean([sum(1 for x in g if x == tid_t or
                             x in cap(TARGET)) for g in gs])
            print(f"  {PROMPT[:12]:>12} {TARGET:>8} {gap_t:>6.2f} "
                  f"{a_att:>6.3f} {plant:>6.2f} {rp:>6.2f} "
                  f"{dv:>6.2f} {ntg:>5.1f}", flush=True)

    # ---- B. a_rep robustness ----
    PROMPT = PROMPTS[0]
    TARGET = TARGETS[0]
    ids0 = tok(PROMPT, add_special_tokens=False,
               return_tensors='pt').input_ids.to(DEV)
    cf = {}

    def hook_c2(m, i, o):
        cf['v'] = o[0, -1, :].float()

    h = model.model.norm.register_forward_hook(hook_c2)
    with torch.no_grad():
        L0 = model(ids0).logits[0, -1].float()
    h.remove()
    native = int(L0.argmax())
    vf = cf['v'].float()
    vfn = vf / vf.norm()
    tid_t = int(tok(' ' + TARGET, add_special_tokens=False).input_ids[0])
    gap_t = float(L0[native] - L0[tid_t])
    a_att = 2 * gap_t / 97.0 + 0.02
    Wt = W[tid_t].float()
    tau_t = Wt - (vfn @ Wt) * vfn
    g_t = tau_t / tau_t.norm()
    print(f"\n  B. A_REP ROBUSTNESS ({PROMPT!r}, {TARGET}, 1 seed x "
          f"{NTOK_B} tok)")
    print(f"  {'a_rep':>6} {'plant':>6} {'rep4':>6} {'div':>6}")
    for a_rep in (0.05, 0.10, 0.15, 0.20, 0.30):
        gs = gen(ids0, vf, tid_t, cap(TARGET), a_att, a_rep, NTOK_B, 1)
        plant = np.mean([1.0 if (tid_t in g[:10] or
                                 any(c in g[:10] for c in cap(TARGET)))
                         else 0.0 for g in gs])
        rp = np.mean([rep4(g) for g in gs])
        dv = np.mean([len(set(g)) / len(g) for g in gs])
        print(f"  {a_rep:>6.2f} {plant:>6.2f} {rp:>6.2f} {dv:>6.2f}",
              flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()