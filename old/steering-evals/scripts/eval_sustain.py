#!/usr/bin/env python3
"""eval_sustain.py — the sustained-topic controller: a CONSTANT weak
attractor + anti-last. Does the topic persist as an essay instead of
plant-then-drift (one+anti) or loop (full steer)?

Gemma-3-1B only, 4 modes x 2 seeds x 20 tok, <=10s.

70e23dc showed steering is one-step Markov (nothing persists after the
script). The untested corner: WEAK CONTINUOUS attract every step toward
the target (recompute tangent per step), plus the anti-last loop-breaker.
Hypothesis: a small a_c sustains chicken as a recurring topic WITHOUT
the full-rotation chicken-loop.

  mode           every-step ops
  one+anti       steer once (a=2gap/97+0.02), then anti-last
  cont05+anti    attract a_c=0.05 toward target + anti-last
  cont10+anti    attract a_c=0.10 + anti-last
  cont15+anti    attract a_c=0.15 + anti-last

metrics: plant (chicken in first 10), rep4, div, #tgt (chicken count in
the full 20), sample. #tgt is the sustain score: one-shot gives ~1-2,
the loop gives ~5+ (useless if rep4 high). The win is #tgt>=4 with
rep4<=0.15 (sustained topical prose).

Run: timeout 60 python3 -u eval_sustain.py  # GEMMA-3-1B
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
NTOK = 20
SEEDS = 2
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

    tid_t = int(tok(' ' + TARGET, add_special_tokens=False).input_ids[0])
    capl = [int(c) for c in tok(' ' + TARGET.capitalize(),
                                add_special_tokens=False).input_ids]
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
    gap_t = float(L0[native] - L0[tid_t])
    A_ATT = 2 * gap_t / 97.0 + 0.02

    def toward(vv, amt):
        """rotate vv toward the target row by amt (tangent recomputed
        against the CURRENT state)."""
        v1 = vv / vv.norm()
        Wt_ = W[tid_t].float()
        tau = Wt_ - (v1 @ Wt_) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(amt) + g * math.sin(amt)) * vv.norm()

    def anti(vv, tid, amt=A_REP):
        vv1 = vv / vv.norm()
        Wb = W[tid].float()
        tauv = Wb - (vv1 @ Wb) * vv1
        gl = -tauv / tauv.norm()
        return (vv1 * math.cos(amt) + gl * math.sin(amt)) * vv.norm()

    def gen(mode):
        a_c = {'cont05+anti': 0.05, 'cont10+anti': 0.10,
               'cont15+anti': 0.15}.get(mode, 0.0)
        one = mode == 'one+anti'
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            vv = vf
            for step in range(NTOK):
                # frame invariant: start each step from the natural state
                vv = vf
                if one and step == 0:
                    vv = (vfn * math.cos(A_ATT) +
                          (W[tid_t].float() -
                           (vfn @ W[tid_t].float()) * vfn) /
                          (W[tid_t].float() -
                           (vfn @ W[tid_t].float()) * vfn).norm() *
                          math.sin(A_ATT)) * vf.norm()
                elif a_c > 0:
                    vv = toward(vv, a_c)
                vv_use = vv
                if toks:
                    vv_use = anti(vv, toks[-1])

                def inj(m, i, o, p=vv_use):
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

    print(f"[{MODEL}] {PROMPT!r} tgt={TARGET!r} a_att={A_ATT:.3f} "
          f"a_rep={A_REP} NTOK={NTOK}")
    print(f"  {'mode':>11} {'plant':>6} {'rep4':>6} {'div':>6} "
          f"{'#tgt':>5}  samples")
    for mode in ('one+anti', 'cont05+anti', 'cont10+anti', 'cont15+anti'):
        gs = gen(mode)
        plant = np.mean([1.0 if (tid_t in g[:10] or
                                 any(c in g[:10] for c in capl)) else 0.0
                         for g in gs])
        rp = np.mean([rep4(g) for g in gs])
        dv = np.mean([len(set(g)) / len(g) for g in gs])
        ntg = np.mean([sum(1 for x in g if x == tid_t or x in capl)
                       for g in gs])
        print(f"  {mode:>11} {plant:>6.2f} {rp:>6.2f} {dv:>6.2f} "
              f"{ntg:>5.1f}  {[tok.decode(g)[:48] for g in gs]}",
              flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()