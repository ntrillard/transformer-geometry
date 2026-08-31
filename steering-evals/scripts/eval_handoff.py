#!/usr/bin/env python3
"""eval_handoff.py — RUNTIME CONTROL: does the one-shot steer override an
already-planted topic (A -> B -> A switching)?

Gemma-3-1B only, 4 modes x 2 seeds x 12 tok, <=10s.

The recipe plants one topic. The open question (7fe5f6e): can the SAME
primitive HAND OFF between topics mid-generation? If steering is really
one-step Markov (70e23dc), each steer should cleanly override the last.

  mode    steer events (step, word)          expectation
  A->B    grill@0, chicken@1, fried@3, fish@4  contiguous 'fried fish' after 3
  A->B->A grill@0, chicken@1, fried@3, fish@4, chicken@7  back to chicken
  holdA   grill@0, chicken@1 (control)          B should NOT appear
  B->A    fried@0, fish@1, grilled@3, chicken@4 reverse handoff

anti-last 0.15 every step (loop-breaker). a per steer = 2*gap/97+0.02.
metrics: contiguous bigram A / B / any-word in first 10, rep4, div, samples.

Run: timeout 60 python3 -u eval_handoff.py  # GEMMA-3-1B
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
NTOK = 12
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

    def att_dir(w):
        tid = int(tok(' ' + w, add_special_tokens=False).input_ids[0])
        gap = float(L0[native] - L0[tid])
        a = 2 * gap / 97.0 + 0.02
        Wt = W[tid].float()
        tau = Wt - (vfn @ Wt) * vfn
        return tid, a, tau / tau.norm()

    def anti(vv, tid, amt=A_REP):
        vv1 = vv / vv.norm()
        Wb = W[tid].float()
        tauv = Wb - (vv1 @ Wb) * vv1
        gl = -tauv / tauv.norm()
        return (vv1 * math.cos(amt) + gl * math.sin(amt)) * vv.norm()

    def gen(schedule):
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                vv = vf
                if step in schedule:
                    tid, a, g = att_dir(schedule[step])
                    vv = (vfn * math.cos(a) + g * math.sin(a)) * vf.norm()
                if toks:
                    vv = anti(vv, toks[-1])

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

    wa1, wa2 = 'grilled', 'chicken'
    wb1, wb2 = 'fried', 'fish'
    ta1, ta2 = att_dir(wa1), att_dir(wa2)
    tb1, tb2 = att_dir(wb1), att_dir(wb2)
    A = (ta1[0], ta2[0])
    B = (tb1[0], tb2[0])

    def has_bigram(g, tids):
        for i in range(min(len(g), 10) - 1):
            if g[i] == tids[0] and g[i + 1] == tids[1]:
                return 1.0
        return 0.0

    modes = {
        'A->B': {0: wa1, 1: wa2, 3: wb1, 4: wb2},
        'A->B->A': {0: wa1, 1: wa2, 3: wb1, 4: wb2, 7: wa2},
        'holdA': {0: wa1, 1: wa2},
        'B->A': {0: wb1, 1: wb2, 3: wa1, 4: wa2},
    }
    print(f"[{MODEL}] {PROMPT!r} A=({wa1} {wa2}) B=({wb1} {wb2}) "
          f"NTOK={NTOK}")
    print(f"  {'mode':>8} {'A':>5} {'B':>5} {'anyA':>5} {'anyB':>5} "
          f"{'rep4':>6} {'div':>6}  samples")
    for name, sched in modes.items():
        gs = gen(sched)
        a_p = np.mean([has_bigram(g, A) for g in gs])
        b_p = np.mean([has_bigram(g, B) for g in gs])
        anya = np.mean([1.0 if any(t in g[:10] for t in A) else 0.0
                        for g in gs])
        anyb = np.mean([1.0 if any(t in g[:10] for t in B) else 0.0
                        for g in gs])
        rp = np.mean([rep4(g) for g in gs])
        dv = np.mean([len(set(g)) / len(g) for g in gs])
        print(f"  {name:>8} {a_p:>5.2f} {b_p:>5.2f} {anya:>5.2f} "
              f"{anyb:>5.2f} {rp:>6.2f} {dv:>6.2f}  "
              f"{[tok.decode(g)[:46] for g in gs]}", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()