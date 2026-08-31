#!/usr/bin/env python3
"""eval_nudge_once.py — the SINGLE gentle nudge (one 2-4 deg steer at
step 0, then FREE sampling). The untested cell between 'single hard
shot' (17 deg / law 0.378, shreds grammar) and 'persistent walk'
(converges to a topic word, repeats it). Result: TOO WEAK on base
Gemma — output identical to baseline, no topic planted; the ' I' loop
on 'Every morning I eat' persists even on the base model.

  capital   'The capital of France is'   : nudge 2/3/4 deg once ->
              'I covered a being one the a the one a France headed'
              (identical to baseline, xtgt=0) — the 2-4 deg nudge does
              not even reach the city family cone.
  breakfast 'Every morning I eat'        : nudge 2/3/4 deg once ->
              'I I I I...' (rep4 0.91, xtgt=0) — the ' I' loop is
              prompt-level, not variant-level, and a small steer is
              powerless against it.

Run: HF_TOKEN=<token> timeout 120 python3 -u eval_nudge_once.py
"""
import itertools
import math

import numpy as np
import torch

import steering_geometry_test as SGT
from eval_nb_quick import CLASSES

MODEL = 'google/gemma-3-1b-pt'
NTOK = 14
PROMPTS = [('capital', 'The capital of France is', 'city'),
           ('breakfast', 'Every morning I eat', 'food')]


def rep4(t):
    n4 = [tuple(t[i:i + 4]) for i in range(len(t) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(t) - 3)])


def main():
    t0 = __import__('time').time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()

    for pname, PROMPT, pcls in PROMPTS:
        fam = [int(tok(' ' + w, add_special_tokens=False).input_ids[0])
               for w in CLASSES[pcls]]
        fset = set(fam)
        ids0 = tok(PROMPT, add_special_tokens=False,
                   return_tensors='pt').input_ids.to('cuda')
        cf = {}
        model.model.norm.register_forward_hook(
            lambda m, i, o: cf.__setitem__('v', o[0, -1, :].float()))
        with torch.no_grad():
            L0 = model(ids0).logits[0, -1].float()
        native = int(L0.argmax())
        vf = cf['v'].float()
        vfn = vf / vf.norm()
        Wff = Wn[fam].float()
        print(f"\n[{pname}] {PROMPT!r} native={tok.decode([native])!r}")
        for deg in (2.0, 3.0, 4.0):
            torch.manual_seed(0)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                vv = vf
                if step == 0:
                    u = vv / vv.norm()
                    t = fam[int((Wff @ u).argmax())]
                    Wb = Wn[t].float()
                    tau = Wb - (u @ Wb) * u
                    g = tau / tau.norm()
                    vv = (u * math.cos(math.radians(deg)) +
                          g * math.sin(math.radians(deg))) * vf.norm()

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
            x = sum(1 for t in toks if t in fset)
            print(f"  nudge{deg:.0f}deg-once: rep4={rep4(toks):.2f} "
                  f"div={len(set(toks))/len(toks):.2f} xtgt={x} "
                  f"dis={len({t for t in toks if t in fset})}  "
                  f"{tok.decode(toks)[:60]!r}", flush=True)
    print(f"\n[{__import__('time').time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()
