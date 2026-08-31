#!/usr/bin/env python3
"""eval_midaster.py — the non-firewall primitive: steer at a MID layer
(L6) and let the model's own propagation reach the head (no readout
inject). Does it plant with BETTER prose than the readout recipe?

Gemma-3-1B only, 2 seeds x 4 modes x 12 tok, <=10s.

3001566: the readout-norm inject is a FIREWALL (erases all mid effect).
But mid->readout AMPLIFICATION is real (0.2 rad at L6 -> 6.5 logits at
the head when nothing overrides the readout). So steer at L6 and DON'T
touch the final norm: the model transforms the injected frame to the
head itself. The depth-arc had L6 as the HEAL layer. This is the honest
composition test the firewall prevented.

  mode          ops (all at L6 frame, final norm untouched)
  L6native      none                                    (baseline)
  L6a15         steer@0 toward chicken, a=0.15
  L6a30         steer@0 toward chicken, a=0.30
  L6a30+anti    steer@0 a=0.30 + anti-last a=0.15 every step

rotation frame at L6: the natural L6 output, tangent toward the target
row (W row projected orthogonal to the L6 state), same construction as
the readout recipe but at depth.

metrics: plant (chicken in first 10), rep4, div, #SEP = count of the
native ' I' token (the readability meter from the readout ceiling), and
samples. If mid-steer plants AND lowers #SEP below the readout recipe's
~10/20, the depth blend is achievable after all - through the model's
own transform instead of the firewall output.

Run: timeout 60 python3 -u eval_midaster.py  # GEMMA-3-1B
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
SEEDS = 2
A_REP = 0.15
MID = 6


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

    # native final (for native token & gap)
    h = model.model.norm.register_forward_hook(hook_c)
    with torch.no_grad():
        L0 = model(ids0).logits[0, -1].float()
    h.remove()
    native = int(L0.argmax())
    gap_t = float(L0[native] - L0[tid_t])
    vf = cf['v'].float()
    vfn = vf / vf.norm()
    tau_f = W[tid_t].float() - (vfn @ W[tid_t].float()) * vfn
    g_f = tau_f / tau_f.norm()
    A_ATT = 2 * gap_t / 97.0 + 0.02
    # native MID state (the injection frame)
    cf2 = {}

    def hook_m(m, i, o):
        cf2['v'] = o[0, -1, :].float()

    hm = model.model.layers[MID].register_forward_hook(hook_m)
    with torch.no_grad():
        _ = model(ids0)
    hm.remove()
    vmid = cf2['v'].float()
    nname = tok.decode([native])
    print(f"[{MODEL}] {PROMPT!r} tgt={TARGET!r} native={nname!r} "
          f"gap={gap_t:.2f} mid=L{MID} NTOK={NTOK}", flush=True)

    def rot(vv, tid, amt, away=False):
        v1 = vv / vv.norm()
        Wb = W[tid].float()
        t = Wb - (v1 @ Wb) * v1
        g = t / t.norm()
        if away:
            g = -g
        return (v1 * math.cos(amt) + g * math.sin(amt)) * vv.norm()

    def gen(mode):
        a0 = {'L6a80': 0.80, 'L6a80+anti': 0.80}.get(mode)
        anti = mode.endswith('+anti') or mode == 'ro'
        is_ro = mode == 'ro'
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                if is_ro:
                    # readout recipe: inject at the FINAL norm
                    vv = vf
                    if step == 0:
                        vv = (vfn * math.cos(A_ATT) +
                              g_f * math.sin(A_ATT)) * vf.norm()
                    if anti and toks:
                        vv = rot(vv, toks[-1], A_REP, away=True)
                else:
                    # mid-frame: inject at L6
                    vv = vmid
                    if step == 0 and a0:
                        vv = rot(vv, tid_t, a0)
                    if anti and toks:
                        vv = rot(vv, toks[-1], A_REP, away=True)

                def inj(m, i, o, p=vv):
                    out = o.clone()
                    out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                    device=out.device)
                    return out

                hook_layer = (model.model.norm if is_ro
                              else model.model.layers[MID])
                hi = hook_layer.register_forward_hook(inj)
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

    print(f"  {'mode':>10} {'plant':>6} {'rep4':>6} {'div':>6} "
          f"{'#SEP':>5}  samples")
    for mode in ('L6native', 'L6a80', 'L6a80+anti', 'ro'):
        gs = gen(mode)
        plant = np.mean([1.0 if (tid_t in g[:10] or
                                 any(c in g[:10] for c in capl)) else 0.0
                         for g in gs])
        rp = np.mean([rep4(g) for g in gs])
        dv = np.mean([len(set(g)) / len(g) for g in gs])
        nsep = np.mean([sum(1 for x in g if x == native) for g in gs])
        print(f"  {mode:>10} {plant:>6.2f} {rp:>6.2f} {dv:>6.2f} "
              f"{nsep:>5.1f}  {[tok.decode(g)[:44] for g in gs]}",
              flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()