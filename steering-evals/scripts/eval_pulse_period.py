#!/usr/bin/env python3
"""eval_pulse_period.py — how sparse can the steer pulse be and still
sustain a topic essay? (the metronome -> essay question from d7b31db)

Gemma-3-1B only, 4 modes x 2 seeds x 20 tok, <=10s.

pulse3+anti (d7b31db) emits chicken at 0,3,6... - a topic METRONOME, not
an essay. Here the steer pulse period grows (3/5/7) and one mode steers
ONCE (step 0). The anti-last keeps breaking loops in between. Hypothesis:
longer pulse gaps -> more prose fills in before the next topic ping,
until plant (chicken within first 10) and topic sustain (chicken count)
compete with prose quality (rep4, div).

  mode       steer schedule            anti last-token
  p3+anti    every 3rd step            always (break loops)
  p5+anti    every 5th step            always
  p7+anti    every 7th step            always
  one+anti   step 0 only               always

metrics: plant (chicken in first 10), rep4, div, chicken-count (topic
sustain), and the sample text. Constants from the law: a_att = 2*gap/97
+ 0.02; a_rep = 0.15 (fixed anti budget from the anti-steer probe).

Run: timeout 60 python3 -u eval_pulse_period.py  # GEMMA-3-1B
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
    NL = model.config.num_hidden_layers

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
    Wn = W[native].float()
    gap_t = float(L0[native] - L0[tid_t])
    A_ATT = 2 * gap_t / 97.0 + 0.02
    Wt = W[tid_t].float()
    tau_t = Wt - (vfn @ Wt) * vfn
    g_t = tau_t / tau_t.norm()

    def gen(period, one_shot=False):
        """period=None means steer only at step 0 (one_shot)."""
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                vv = vf
                steer_now = one_shot and step == 0
                if period is not None and step % period == 0:
                    steer_now = True
                if steer_now:
                    vv = (vfn * math.cos(A_ATT) +
                          g_t * math.sin(A_ATT)) * vf.norm()
                # anti-last: suppress the just-sampled token
                if toks:
                    vv1 = vv / vv.norm()
                    Wl = W[toks[-1]].float()
                    tauv = Wl - (vv1 @ Wl) * vv1
                    gl_ = -tauv / tauv.norm()
                    vv = (vv1 * math.cos(A_REP) +
                          gl_ * math.sin(A_REP)) * vf.norm()

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

    print(f"[{MODEL}] {PROMPT!r} target={TARGET!r} a_att={A_ATT:.3f} "
          f"a_rep={A_REP} NTOK={NTOK}")
    print(f"  {'mode':>9} {'plant':>6} {'rep4':>6} {'div':>6} "
          f"{'#tgt':>5}  samples")
    for (label, period, one_shot) in (
            ('p3+anti', 3, False),
            ('p5+anti', 5, False),
            ('p7+anti', 7, False),
            ('one+anti', None, True)):
        gs = gen(period, one_shot)
        plant = np.mean([1.0 if (tid_t in g[:10] or
                                 any(c in g[:10] for c in capl)) else 0.0
                         for g in gs])
        rp = np.mean([rep4(g) for g in gs])
        dv = np.mean([len(set(g)) / len(g) for g in gs])
        ntg = np.mean([sum(1 for x in g if x == tid_t or x in capl)
                       for g in gs])
        print(f"  {label:>9} {plant:>6.2f} {rp:>6.2f} {dv:>6.2f} "
              f"{ntg:>5.1f}  {[tok.decode(g)[:44] for g in gs]}",
              flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()