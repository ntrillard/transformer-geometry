#!/usr/bin/env python3
"""eval_phrase.py — beyond words: plant a contiguous PHRASE (2-token
bigram) via the same law, sequential vs mean-direction targets.

Gemma-3-1B only, 2 seeds x (4 modes) x 16 tok, <=10s.

Readout ceiling (28ccb27): rotation plants/banishes WORDS but cannot
compose clauses. Next object: the PHRASE. Two phrase primitives:
  SEQ : steer step0 -> tok1, step1 -> tok2 (scripted target succession)
  MEAN: steer once toward normalize(W_tok1 + W_tok2) (phrase centroid)
then anti-last loop-breaker as in the recipe. Does 'grilled chicken'
emerge CONTIGUOUSLY? Phrases are 2 single-token words each.

  mode       schedule
  seq        steer tok1@0, tok2@1, then anti-last
  mean       steer centroid@0, then anti-last
  seq-anti   seq + anti-last every step from step 2 (loop-break after)
  mean-anti  mean + anti-last from step 1

a = 2*gap/97 + 0.02 per target token (gap of that token on the prompt).
metric: PHRASE-PLANT = contiguous bigram (tok1 tok2) anywhere in the
first 10 tokens; word plant; rep4; div; sample.

Run: timeout 60 python3 -u eval_phrase.py  # GEMMA-3-1B
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
PHRASES = [('grilled', 'chicken'), ('fried', 'chicken'),
           ('chicken', 'soup')]
NTOK = 16
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

    def gen(mode, tid1, g1, tid2, g2, gm):
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                vv = vf
                if mode == 'seq' and step == 0:
                    vv = (vfn * math.cos(gm[0]) + g1 * math.sin(gm[0])) * \
                        vf.norm()
                elif mode == 'seq' and step == 1:
                    vv = (vfn * math.cos(gm[1]) + g2 * math.sin(gm[1])) * \
                        vf.norm()
                elif mode == 'mean' and step == 0:
                    vv = (vfn * math.cos(gm[2]) + gm[3] * math.sin(gm[2])) * \
                        vf.norm()
                elif mode == 'seq-anti' and step == 0:
                    vv = (vfn * math.cos(gm[0]) + g1 * math.sin(gm[0])) * \
                        vf.norm()
                elif mode == 'seq-anti' and step == 1:
                    vv = (vfn * math.cos(gm[1]) + g2 * math.sin(gm[1])) * \
                        vf.norm()
                elif mode == 'mean-anti' and step == 0:
                    vv = (vfn * math.cos(gm[2]) + gm[3] * math.sin(gm[2])) * \
                        vf.norm()
                if toks and mode in ('seq-anti', 'mean-anti'):
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

    print(f"[{MODEL}] {PROMPT!r} native={tok.decode([native])!r} "
          f"PHRASE planting (seq/mean), NTOK={NTOK}")
    print(f"  {'phrase':>16} {'mode':>9} {'phr-plant':>9} {'t1':>4} "
          f"{'t2':>4} {'rep4':>6} {'div':>6}  samples")
    for (w1, w2) in PHRASES:
        t1, a1, g1 = att_dir(w1)
        t2, a2, g2 = att_dir(w2)
        mm = (W[t1].float() + W[t2].float())
        mmv = mm - (vfn @ mm) * vfn
        gm = [a1, a2,
              2 * (float(L0[native] - L0[t1]) + float(L0[native] - L0[t2]))
              / 2 / 97.0 + 0.02,
              mmv / mmv.norm()]
        for mode in ('seq', 'mean', 'seq-anti', 'mean-anti'):
            gs = gen(mode, t1, g1, t2, g2, gm)
            def has_bigram(g):
                for i in range(min(len(g), 10) - 1):
                    if g[i] == t1 and g[i + 1] == t2:
                        return 1.0
                return 0.0
            phr = np.mean([has_bigram(g) for g in gs])
            t1p = np.mean([1.0 if t1 in g[:10] else 0.0 for g in gs])
            t2p = np.mean([1.0 if t2 in g[:10] else 0.0 for g in gs])
            rp = np.mean([rep4(g) for g in gs])
            dv = np.mean([len(set(g)) / len(g) for g in gs])
            print(f"  {w1 + ' ' + w2:>16} {mode:>9} {phr:>9.2f} "
                  f"{t1p:>4.2f} {t2p:>4.2f} {rp:>6.2f} {dv:>6.2f}  "
                  f"{[tok.decode(g)[:40] for g in gs]}", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()