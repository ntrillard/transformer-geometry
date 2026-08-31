#!/usr/bin/env python3
"""eval_depth_blend.py — does a soft mid-stack pulse (depth arc) improve
the READOUT recipe's prose above the readout-only ceiling?

Gemma-3-1B only, 4 modes x 2 seeds x 20 tok, <=10s.

The readout recipe (1356f2d) gives rep4 0.00 but reads as a noun-
metronome: 'chicken I bed I another I driven' — the native ' I' token
keeps resampling as a separator (the sink/native axis). The earlier
DEPTH arc had its own prose win (L10 soft pulse): does adding a soft
mid-stack pulse toward the target IMPROVE the reading (fewer native
separator tokens) on top of the readout recipe?

  mode        readout (steer@0 + anti-last)   mid L6 pulse
  ro          yes                              none            (baseline)
  ro+L6once   yes                              steer@0, a=0.20
  ro+L6p3     yes                              every 3rd, a=0.12
  ro+L6all    yes                              every step, a=0.08

mid pulse = rotate the natural L6 output toward the target row by a_pulse
(soft, does NOT swap argmax at depth). a_att = 2*gap/97 + 0.02, a_rep=0.15.

FINDING (2026-08-30): the mid-pulse firewall is STRUCTURAL.
The readout-norm hook REPLACES the final hidden state entirely; the LM
head reads only that state. Therefore ANY mid-layer manipulation is
ERASED when the readout inject is present. Verified:
  - pulse-only L6 (no readout inject): a=0.2 -> 6.5 logit move; argmax
    stays ' I' until a>=0.8, then flips to 'chicken' (mid->readout amp).
  - readout-inject + L6 pulse 0.2..1.0: max|dlogit| = 0.0 EXACTLY;
    sample stream identical to no-pulse at ALL amplitudes.
So the depth-blend null result (all four modes bit-identical) is NOT a
cooking failure - it is a firewall: mid pulses cannot compose with a
forced readout. The recipe operates at the firewall output, which is
exactly why it never drifts.

metrics: plant, rep4, div, #SEP = count of the native ' I' token in the
20 tokens (the separator meter — lower is more readable prose), + sample.

Run: timeout 60 python3 -u eval_depth_blend.py  # GEMMA-3-1B
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
MID = 6  # mid-depth layer (L6 heal from the depth riser)


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

    # capture the natural MID-layer last state for the pulse frame
    cf2 = {}
    h2 = model.model.layers[MID].register_forward_hook(
        lambda m, i, o: cf2.__setitem__('v', o[0, -1, :].float()))
    with torch.no_grad():
        _ = model(ids0)
    h2.remove()
    vmid = cf2['v'].float()
    vmidn = vmid / vmid.norm()
    # mid pulse direction toward the target row (tangent at mid frame)
    tau_m = Wt - (vmidn @ Wt) * vmidn
    g_m = tau_m / tau_m.norm()

    def mid_pulse(vm, amt):
        return (vmidn * math.cos(amt) + g_m * math.sin(amt)) * vm.norm()

    def anti(vv, tid, amt=A_REP):
        vv1 = vv / vv.norm()
        Wb = W[tid].float()
        tauv = Wb - (vv1 @ Wb) * vv1
        gl = -tauv / tauv.norm()
        return (vv1 * math.cos(amt) + gl * math.sin(amt)) * vv.norm()

    def gen(mode):
        p_once = {'ro+L6once': (0.20, 'once')}.get(mode)
        p_pulse = {'ro+L6p3': (0.12, 'p3')}.get(mode)
        p_all = {'ro+L6all': (0.08, 'all')}.get(mode)
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
                    vv = anti(vv, toks[-1])

                def inj(m, i, o, p=vv):
                    out = o.clone()
                    out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                    device=out.device)
                    return out

                hooks = [model.model.norm.register_forward_hook(inj)]
                # mid pulse
                pv = None
                if p_once and step == 0:
                    pv = mid_pulse(vmid, p_once[0])
                elif p_pulse and step % 3 == 0:
                    pv = mid_pulse(vmid, p_pulse[0])
                elif p_all:
                    pv = mid_pulse(vmid, p_all[0])
                if pv is not None:
                    hooks.append(
                        model.model.layers[MID].register_forward_hook(
                            lambda m, i, o, p=pv: (
                                lambda out: (
                                    out.__setitem__(
                                        (0, -1), torch.as_tensor(
                                            p, dtype=out.dtype,
                                            device=out.device)),
                                    out)[1])(o)))

                try:
                    with torch.no_grad():
                        L = model(ids).logits[0, -1].float()
                finally:
                    for hh in hooks:
                        hh.remove()
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
          f"a_att={A_ATT:.3f} a_rep={A_REP} mid=L{MID} NTOK={NTOK}")
    print(f"  {'mode':>9} {'plant':>6} {'rep4':>6} {'div':>6} {'#SEP':>5}"
          "  samples")
    for mode in ('ro', 'ro+L6once', 'ro+L6p3', 'ro+L6all'):
        gs = gen(mode)
        plant = np.mean([1.0 if (tid_t in g[:10] or
                                 any(c in g[:10] for c in capl)) else 0.0
                         for g in gs])
        rp = np.mean([rep4(g) for g in gs])
        dv = np.mean([len(set(g)) / len(g) for g in gs])
        nsep = np.mean([sum(1 for x in g if x == native) for g in gs])
        print(f"  {mode:>9} {plant:>6.2f} {rp:>6.2f} {dv:>6.2f} "
              f"{nsep:>5.1f}  {[tok.decode(g)[:46] for g in gs]}",
              flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()