#!/usr/bin/env python3
"""eval_longprompts.py — does the law + recipe survive LONGER contexts?

Everything so far validated on 4-6 word prompts ('For dinner I made',
'I went to the store and bought'). Context length changes the native
basin, the native token identity, and the moat. Test the readout law and
the one-shot+anti recipe as the prompt grows:

  p_short 'For dinner I made'                          ~4 tok
  p_med   'For dinner I made grilled chicken with roasted vegetables'
                                                       ~8 tok
  p_long  'Yesterday evening for dinner I made grilled chicken with
          roasted vegetables and a fresh salad while my family
          discussed the upcoming trip'                 ~19 tok

Per prompt, target 'ocean' (single token, absent from all prompts):
  A. LAW: native token (is it still ' I'?), gap, aexact (analytic),
     apred=gap/97, ratio, MOAT (min gap over the whole vocab, ranks>=2)
  B. RECIPE: steer once @ a_att=2*gap/97+0.02 + anti-last 0.15, 10 tok,
     1 seed, top_p 0.9: plant, rep4, div, #SEP (native token count),
     sample text. Diamond question: does the ' I' separator skeleton
     persist, or does a longer native context change the prose class?

Writes steering_geometry_results/longprompts_gemma.csv
"""
import csv
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET = 'ocean'
NTOK = 10
SEEDS = 1
A_REP = 0.15
PROMPTS = [
    ('short', 'For dinner I made'),
    ('med', 'For dinner I made grilled chicken with roasted vegetables'),
    ('long', 'Yesterday evening for dinner I made grilled chicken with '
             'roasted vegetables and a fresh salad while my family '
             'discussed the upcoming trip'),
]
OUT = Path('../steering_geometry_results/longprompts_gemma.csv')


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()
    V = W.shape[0]
    tid_t = int(tok(' ' + TARGET, add_special_tokens=False).input_ids[0])
    capl = [int(c) for c in tok(' ' + TARGET.capitalize(),
                                add_special_tokens=False).input_ids]
    fset = {tid_t} | set(capl)
    print(f"[{MODEL}] target={TARGET!r} NTOK={NTOK}, "
          f"long-prompt sweep")

    def anti_geo(vv, tid, amt=A_REP):
        vv1 = vv / vv.norm()
        Wb = W[tid].float()
        tauv = Wb - (vv1 @ Wb) * vv1
        gl = -tauv / tauv.norm()
        return (vv1 * math.cos(amt) + gl * math.sin(amt)) * vv.norm()

    rows = []
    print(f"  {'prompt':>6} {'nctx':>5} {'native':>8} {'gap':>6} "
          f"{'aexact':>7} {'apred':>6} {'ratio':>6} {'moat':>6} "
          f"{'plant':>6} {'rep4':>6} {'div':>6} {'#SEP':>5}  sample")
    for pname, PROMPT in PROMPTS:
        ids0 = tok(PROMPT, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        nctx = ids0.shape[1]
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
        Wn_native = Wn[native].float()
        Wt = Wn[tid_t].float()
        # analytic closed-form crossing
        A_ = vfn @ (Wt - Wn_native)
        tau = Wt - (vfn @ Wt) * vfn
        B_ = (tau @ (Wt - Wn_native)) / (tau.norm() + 1e-12)
        aexact = abs(math.atan2(-A_, B_))
        gap = float(L0[native] - L0[tid_t])
        apred = gap / 97.0
        ratio = aexact / (apred + 1e-12)
        # moat: min gap over the whole vocab (non-native)
        mask = torch.ones(V, dtype=torch.bool, device=DEV)
        mask[native] = False
        moat = float((L0[native] - L0[mask]).min())
        # B. recipe generation
        A_ATT = 2 * apred + 0.02
        tau_t = Wt - (vfn @ Wt) * vfn
        g_t = tau_t / tau_t.norm()
        torch.manual_seed(0)
        ids = ids0.clone()
        toks = []
        for step in range(NTOK):
            vv = vf
            if step == 0:
                vv = (vfn * math.cos(A_ATT) + g_t * math.sin(A_ATT)) * \
                    vf.norm()
            if toks:
                vv = anti_geo(vv, toks[-1])

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
        plant = 1.0 if any(t in g0 for g0 in [toks]
                           for t in toks[:10] if t in fset) else 0.0
        x = sum(1 for t in toks if t in fset)
        dis = len({t for t in toks if t in fset})
        div = len(set(toks)) / len(toks)
        mr = max((sum(1 for _ in grp) for _, grp in
                  itertools.groupby(toks)), default=0)
        rp = rep4(toks)
        nsep = sum(1 for t in toks if t == native)
        sample = tok.decode(toks)
        rows.append(dict(model=MODEL, prompt=pname, nctx=nctx,
                         native=tok.decode([native]), gap=round(gap, 2),
                         aexact=round(aexact, 4),
                         apred=round(apred, 4), ratio=round(ratio, 3),
                         moat=round(moat, 2), plant=plant, x=x, dis=dis,
                         div=round(div, 3), maxrun=mr,
                         rep4=round(rp, 3), nsep=nsep, sample=sample))
        print(f"  {pname:>6} {nctx:>5} {tok.decode([native])!r:>8} "
              f"{gap:>6.2f} {aexact:>7.4f} {apred:>6.4f} "
              f"{ratio:>6.2f} {moat:>6.2f} {plant:>6.1f} {rp:>6.2f} "
              f"{div:>6.2f} {nsep:>5}  {sample[:38]!r}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()