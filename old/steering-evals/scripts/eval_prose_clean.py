#!/usr/bin/env python3
"""eval_prose_clean.py — can forbidden-set anti (beyond just anti-last)
clean the ' I' metronome out of the recipe's prose?

Gemma-3-1B only, 4 modes x 2 seeds x 20 tok, <=10s.

Committed recipe (1356f2d/7fe5f6e) gives plant 1.00, rep4 0, div ~0.5 but
samples still read 'chicken I bed I another I driven' - the native ' I'
re-samples as a separator (the old native loop-token). anti-last only
kills the IMMEDIATE previous token, so ' I' returns once it leaves the
1-step anti window. Probe: widen the forgotten set.

  mode           steer   anti-last  anti-2nd-last  anti-native(0.05)
  one+anti       once    0.15       -              -           (baseline)
  anti2          once    0.15       0.10           -
  nat            once    0.15       -              always 0.05
  anti2+nat      once    0.15       0.10           always 0.05

a_att = 2*gap/97 + 0.02 (the law). Metrics: plant, rep4, div, #tgt,
plus the actual samples - the question is whether prose READS (sentence-
like) vs is a topic metronome. A hard finding either way: if the ' I'
persists under all four, prose structure is OUT of rotation reach.

Run: timeout 60 python3 -u eval_prose_clean.py  # GEMMA-3-1B
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
A_REP2 = 0.10
A_NAT = 0.05


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
    Wt = W[tid_t].float()
    tau_t = Wt - (vfn @ Wt) * vfn
    g_t = tau_t / tau_t.norm()
    nname = tok.decode([native])

    def anti_vec(vv, tid, amt):
        vv1 = vv / vv.norm()
        Wb = W[tid].float()
        tauv = Wb - (vv1 @ Wb) * vv1
        gl = -tauv / tauv.norm()
        return (vv1 * math.cos(amt) + gl * math.sin(amt)) * vv.norm()

    def gen(mode):
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                vv = vf
                if step == 0:
                    vv = (vfn * math.cos(A_ATT) +
                          g_t * math.sin(A_ATT)) * vf.norm()
                if toks:
                    vv = anti_vec(vv, toks[-1], A_REP)
                    if mode in ('anti2', 'anti2+nat') and len(toks) > 1:
                        # 2nd-last distinct token
                        seen = []
                        for t in reversed(toks):
                            if t not in seen:
                                seen.append(t)
                            if len(seen) == 2:
                                break
                        vv = anti_vec(vv, seen[1], A_REP2)
                if mode in ('nat', 'anti2+nat'):
                    vv = anti_vec(vv, native, A_NAT)

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

    print(f"[{MODEL}] {PROMPT!r} tgt={TARGET!r} native={nname!r} "
          f"a_att={A_ATT:.3f} a_rep={A_REP} a_rep2={A_REP2} "
          f"a_nat={A_NAT} NTOK={NTOK}")
    print(f"  {'mode':>9} {'plant':>6} {'rep4':>6} {'div':>6} {'#tgt':>5}"
          "  samples")
    for mode in ('one+anti', 'anti2', 'nat', 'anti2+nat'):
        gs = gen(mode)
        plant = np.mean([1.0 if (tid_t in g[:10] or
                                 any(c in g[:10] for c in capl)) else 0.0
                         for g in gs])
        rp = np.mean([rep4(g) for g in gs])
        dv = np.mean([len(set(g)) / len(g) for g in gs])
        ntg = np.mean([sum(1 for x in g if x == tid_t or x in capl)
                       for g in gs])
        print(f"  {mode:>9} {plant:>6.2f} {rp:>6.2f} {dv:>6.2f} "
              f"{ntg:>5.1f}  {[tok.decode(g)[:48] for g in gs]}",
              flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()