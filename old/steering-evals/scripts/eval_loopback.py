#!/usr/bin/env python3
"""eval_loopback.py — ROTATE AT THE READOUT, THEN SEND THE STATE BACK
THROUGH THE LAYERS again, so the network's own computation digests the
steering before the head reads it.

MOTIVATION: every prior recipe rotated the state AT/PAST all 26 layers
(readout), so the layers never see the steering - the readout
(dot-product) just reads the rotated endpoint. If instead we rotate the
final state and then RE-INJECT it back into layer 0's residual and let
layers 0..N process it ONCE MORE, the model's learned grammar may
reassert over the planted topic direction, producing a softer, more
grammatical next token than reading the raw rotation.

MECHANISM (how 'send it back through the layers'):
  pass1 : run model -> capture final-norm state v_final AND layer0
          residual scale n0 = ||layer0_out[last]||
  rotate: v' = R(v_final) by law-budget alpha toward closest city row
  pass2 : run the model AGAIN on the SAME prompt, but hook layer 0's
          output so the LAST TOKEN's residual slot = v' rescaled to n0.
          Layers 1..N then re-process the steered state, then final norm,
          then head -> this is 'sending it back through the layers'.
  The first sampled token uses pass2's logits; then FREE-RUN normally
  (no further steering) so you can watch the long generation.

Modes:
  baseline : no steer (native reference)
  readout  : plain rot injected at final norm (the committed recipe - layers
             never see it) -> the comparison
  loopback : rotate at final norm, then re-inject at layer0 and re-run
             layers 0..N (the new idea), ONE step
  loopback_all : loopback at EVERY step (fully closed loop)

Long free-run (NTOK) after the step-0 (or per-step) intervention.
Run: HF_TOKEN=<tok> timeout 30 python3 -u eval_loopback.py
"""
import csv
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

import steering_geometry_test as SGT
from eval_nb_quick import CLASSES

MODEL = 'google/gemma-3-1b-pt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 60
SEEDS = [0]
TARGET = 'city'
OUT = Path('../steering_geometry_results/loopback.csv')

PROMPTS = [
    ('ask', 'If you ask me which European city is the most beautiful, I would say that'),
    ('fr',  'The capital of France is'),
]
MODES = ['baseline', 'readout', 'loopback']


def rep4(toks):
    if len(toks) < 4:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return sum(1 for i in range(len(toks) - 3) if n4[i] in n4[i + 1:]) \
        / (len(toks) - 3)


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach().float()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()
    fam = [int(tok(' ' + w, add_special_tokens=False).input_ids[0])
           for w in CLASSES[TARGET]]
    fset = set(fam)
    Wff = Wn[fam].float()

    def closest(vv):
        u = vv / vv.norm()
        return fam[int((Wff @ u).argmax())]

    def rot(vv, tid, alpha):
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(alpha) + g * math.sin(alpha)) * vv.norm()

    def sample(L):
        p = torch.softmax(L.float(), 0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk) / (q * msk).sum()
        return int(torch.multinomial(qq, 1))

    def nat_vL(ids):
        cv = {}
        hk = model.model.norm.register_forward_hook(
            lambda m, i, o: cv.__setitem__('v', o[0, -1, :].float()))
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        hk.remove()
        return cv['v'], L

    # pass2 substate capture: final norm output while layer0 residual is hijacked
    def forward_loopback(ids, vp, n0):
        """re-run layers with last-token layer-0 residual = vp scaled to n0."""
        cap = {}
        hooks = [
            model.model.layers[0].register_forward_hook(
                lambda m, i, o: cap.__setitem__('l0', o[0, -1, :].clone().float())),
            model.model.norm.register_forward_hook(
                lambda m, i, o: cap.__setitem__('v', o[0, -1, :].float())),
        ]
        with torch.no_grad():
            _ = model(ids)  # natural pass to read l0 scale + final v
        for h in hooks:
            h.remove()
        n0 = float(cap['l0'].norm())
        # hijack layer0 residual for last token
        inj_val = (vp / vp.norm() * n0).to(model.dtype)

        def hijack(m, i, o, p=inj_val):
            out = o.clone()
            out[0, -1, :] = p
            return out

        cap2 = {}
        hh = model.model.layers[0].register_forward_hook(hijack)
        hvn = model.model.norm.register_forward_hook(
            lambda m, i, o: cap2.__setitem__('v', o[0, -1, :].float()))
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        hh.remove()
        hvn.remove()
        return L, cap2['v']

    print(f"[{MODEL}] family={TARGET}  {NTOK}-tok long gen, "
          f"loopback modes")
    rows = []
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        alpha = 2 * (gap / 97.0) + 0.02
        tgt = closest(vf)
        vp = rot(vf, tgt, alpha)
        print(f"\n==== {pname}: {pr!r}  gap={gap:.1f} -> "
              f"{tok.decode([tgt])!r} ====")
        for mode in MODES:
            torch.manual_seed(0)
            ids = ids0.clone()
            toks = []
            l0_norm = None
            for step in range(NTOK):
                hooks = []
                vp_step = vp
                n0 = None
                if mode == 'readout' and step == 0:
                    # inject rotated at FINAL norm (layers never see it)
                    def inj(m, i, o, p=vp_step):
                        out = o.clone()
                        out[0, -1, :] = torch.as_tensor(
                            p, dtype=out.dtype, device=out.device)
                        return out
                    hooks.append(model.model.norm.register_forward_hook(inj))
                elif mode == 'loopback' and step == 0:
                    # pass2 with layer0 re-injection
                    pass  # handled below via forward_loopback
                elif mode == 'loopback_all':
                    pass  # handled below each step
                if (mode == 'loopback' and step == 0) or \
                   (mode == 'loopback_all'):
                    L, _ = forward_loopback(ids, vp_step, l0_norm)
                else:
                    with torch.no_grad():
                        L = model(ids).logits[0, -1].float()
                for h in hooks:
                    h.remove()
                nxt = sample(L)
                toks.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)],
                                dim=1)
            x = sum(1 for t in toks if t in fset)
            plant = 1.0 if any(t in toks[:10] for t in fset) else 0.0
            div = len(set(toks)) / len(toks)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(toks)), default=0)
            rp = rep4(toks)
            txt = tok.decode(toks)
            rows.append(dict(prompt=pname, mode=mode, gap=round(gap, 3),
                             alpha=round(alpha, 3), plant=plant, xtgt=x,
                             div=round(div, 3), maxrun=mr,
                             rep4=round(rp, 3), text=txt))
            print(f"\n--- [{mode:>12}] plant={plant:.0f} xtgt={x} "
                  f"div={div:.2f} maxrun={mr} rep4={rp:.2f} ---",
                  flush=True)
            print(f"    {txt}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()