#!/usr/bin/env python3
"""eval_context_steer.py — CORRECTED: does a STEERED CONTEXT carry the
topic through the model's own attention, WITHOUT touching the readout?

Gemma-3-1B only, 4 modes x 3 seeds x 16 tok, <=10s.

v0 (uncommitted, discarded) injected the steered vector at ALL positions
INCLUDING the last. The norm hook REPLACES the last state and the LM
head reads only the last state (the 3001566 firewall) -> 'all' was
bit-identical to 'last' BY CONSTRUCTION. That null was a measurement
artifact. The real test leaves the LAST position NATURAL: the model must
compute it from attention over the steered earlier context. Per-position
tangents are unknown (the readout law is a last-position law), so the
last-position calibration is used as a crude seed - if even that crude
seed moves the needle, context persistence is real and worth refining.

  mode         last position     context (positions 0..L-2)
  last         INJECTED (champ)  natural
  ctx1         NATURAL (free)    seeded ONCE at step 0, then nothing
  ctxre        NATURAL (free)    seeded EVERY step
  ctxre+anti   anti-last each    seeded EVERY step
                 step (loop-break
                 at readout)

ctx seeding injects the law vector vv0 = rotate(vfn, toward chicken, a)
at positions 0..L-2; position L-1 is whatever the model computes. The
money question: with the readout left natural, does the model GENERATE
chicken from a steered context?

Run: timeout 60 python3 -u eval_context_steer.py  # GEMMA-3-1B
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
NTOK = 16
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

    def anti(vv, tid, amt=A_REP):
        vv1 = vv / vv.norm()
        Wb = W[tid].float()
        tauv = Wb - (vv1 @ Wb) * vv1
        gl = -tauv / tauv.norm()
        return (vv1 * math.cos(amt) + gl * math.sin(amt)) * vv.norm()

    def forward_with(ids, last_inject=None, ctx_seed=None):
        """Run one forward. last_inject: (vector) to place at position
        L-1, or None to leave natural. ctx_seed: vector to place at
        positions 0..L-2 (all but last), or None for natural context."""
        hooks = []

        if last_inject is not None:
            def inj_last(m, i, o, p=last_inject):
                out = o.clone()
                out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                device=out.device)
                return out
            hooks.append(model.model.norm.register_forward_hook(inj_last))

        if ctx_seed is not None:
            def inj_ctx(m, i, o, p=ctx_seed):
                out = o.clone()
                L = out.shape[1]
                out[0, :L - 1] = torch.as_tensor(
                    p, dtype=out.dtype, device=out.device
                ).unsqueeze(0).expand(L - 1, -1)
                return out
            hooks.append(model.model.norm.register_forward_hook(inj_ctx))

        try:
            with torch.no_grad():
                L = model(ids).logits[0, -1].float()
        finally:
            for hh in hooks:
                hh.remove()
        return L

    def gen(mode):
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            vv0 = (vfn * math.cos(A_ATT) + g_t * math.sin(A_ATT)) * \
                vf.norm()
            for step in range(NTOK):
                if mode == 'last':
                    # champion: steer at last pos once, then anti-last
                    if step == 0:
                        li = vv0
                    else:
                        li = anti(vf, toks[-1]) if toks else vf
                    L = forward_with(ids, last_inject=li, ctx_seed=None)
                elif mode == 'ctx1':
                    # context seed ONCE at step 0; natural afterwards
                    cs = vv0 if step == 0 else None
                    L = forward_with(ids, last_inject=None, ctx_seed=cs)
                elif mode == 'ctxre':
                    # context seed EVERY step; last always natural
                    L = forward_with(ids, last_inject=None, ctx_seed=vv0)
                elif mode == 'ctxre+anti':
                    # context seed every step + anti-last at readout
                    li = anti(vf, toks[-1]) if toks else None
                    L = forward_with(ids, last_inject=li, ctx_seed=vv0)

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
          f"a_att={A_ATT:.3f} a_rep={A_REP} NTOK={NTOK}")
    print(f"  {'mode':>10} {'plant':>6} {'rep4':>6} {'div':>6} {'#SEP':>5}"
          f" {'#tgt':>5}  samples")
    for mode in ('last', 'ctx1', 'ctxre', 'ctxre+anti'):
        gs = gen(mode)
        plant = np.mean([1.0 if (tid_t in g[:10] or
                                 any(c in g[:10] for c in capl)) else 0.0
                         for g in gs])
        rp = np.mean([rep4(g) for g in gs])
        dv = np.mean([len(set(g)) / len(g) for g in gs])
        nsep = np.mean([sum(1 for x in g if x == native) for g in gs])
        ntg = np.mean([sum(1 for x in g if x == tid_t or x in capl)
                       for g in gs])
        print(f"  {mode:>10} {plant:>6.2f} {rp:>6.2f} {dv:>6.2f} "
              f"{nsep:>5.1f} {ntg:>5.1f}  "
              f"{[tok.decode(g)[:46] for g in gs]}", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()