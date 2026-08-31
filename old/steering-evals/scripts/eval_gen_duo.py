#!/usr/bin/env python3
"""eval_gen_duo.py — the payoff: readout-only steer+anti-steer generates
topic AND prose (loop-breaking at ONE depth).

Gemma-3-1B only, 3 generations x 12 tok, <=10s.

Earlier (e7adff5-era) combined L10+final failed (final swamps mid). NEW:
BOTH steer (target) and anti-steer (the native loop-token) at the SAME
readout hook - one depth, compatible rotations. Test:

  modes (final-norm hook each step, pulsed@3):
   native            : expected loop ' I I I...'
   steer chicken     : plant but likely loop to 'chicken...'
   steer + anti-' I' : plant AND prose? (the loop-token suppressed)

  steer a_att = 2*gap(chicken)/97 + 0.02  (the law)
  anti  a_rep = 0.15 toward AWAY from W_(' I') (native row), only in the
  combo mode.
metric: plant (chicken in first 10), rep4 (prose), div, sample.

Run: timeout 60 python3 -u eval_gen_duo.py  # GEMMA-3-1B
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
TARGET = 'chicken'
NTOK = 12
SEEDS = 3
A_REP = 0.15


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()
    NL = model.config.num_hidden_layers

    tid_t = int(tok(' ' + TARGET, add_special_tokens=False).input_ids[0])
    capl = [int(c) for c in tok(' ' + TARGET.capitalize(),
                                add_special_tokens=False).input_ids]

    ids0 = tok(PROMPT, add_special_tokens=False,
               return_tensors='pt').input_ids.to(DEV)
    cf = {}

    def c(m, i, o):
        cf['v'] = o[0, -1, :].float()

    h = model.model.norm.register_forward_hook(c)
    with torch.no_grad():
        L0 = model(ids0).logits[0, -1].float()
    h.remove()
    native = int(L0.argmax())
    vf = cf['v'].float()
    vfn = vf / vf.norm()
    Wn = W[native].float()
    gap_t = float(L0[native] - L0[tid_t])
    A_ATT = 2 * gap_t / 97.0 + 0.02
    Wt = W[tid_t].float()
    tau_t = Wt - (vfn @ Wt) * vfn
    g_t = tau_t / tau_t.norm()
    # anti direction: away from native row
    tau_n = Wn - (vfn @ Wn) * vfn
    g_n = -tau_n / tau_n.norm()     # away from native
    nname = tok.decode([native])
    print(f"[{MODEL}] {PROMPT!r} native={nname!r} a_att={A_ATT:.3f} "
          f"a_rep={A_REP}")

    def gen(mode):
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                vv = vf
                # attract: steer toward target (pulse3 = every 3rd step)
                if mode in ('steer', 'duo', 'duo2') or \
                        (mode.startswith('pulse3') and step % 3 == 0):
                    vv = (vfn * math.cos(A_ATT) + g_t * math.sin(A_ATT)) * \
                        vf.norm()
                # anti-last: suppress the just-sampled token (loop-breaker)
                if 'anti' in mode and toks:
                    vv1 = vv / vv.norm()
                    Wl = W[toks[-1]].float()
                    tauv = Wl - (vv1 @ Wl) * vv1
                    gl_ = -tauv / tauv.norm()
                    vv = (vv1 * math.cos(A_REP) + gl_ * math.sin(A_REP)) * vf.norm()

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

    print(f"  {'mode':>10} {'plant':>6} {'rep4':>6} {'div':>6}  samples")
    for mode in ('native', 'steer', 'duo', 'duo2', 'pulse3', 'pulse3+anti'):
        gs = gen(mode)
        plant = np.mean([1.0 if (tid_t in g[:10] or
                                 any(c in g[:10] for c in capl)) else 0.0
                         for g in gs])
        rp = np.mean([rep4(g) for g in gs])
        dv = np.mean([len(set(g)) / len(g) for g in gs])
        print(f"  {mode:>10} {plant:>6.2f} {rp:>6.2f} {dv:>6.2f}  "
              f"{[tok.decode(g)[:34] for g in gs]}", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()