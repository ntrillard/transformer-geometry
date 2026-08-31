#!/usr/bin/env python3
"""eval_phrase3.py — 3-token phrase scripts: does tok3 need its own steer,
and does the phrase PERSIST in a longer rollout?

Gemma-3-1B only, 3 phrases x 3 modes x 2 seeds x 20 tok, <=10s.

2fce2ef: 2-token phrases plant by SEQUENTIAL steering (tok1@0, tok2@1)
+ anti-last. Next:
  P3   : steer tok1@0, tok2@1, tok3@2 (full 3-token script)
  P2   : steer tok1@0, tok2@1 only - does tok3 follow naturally?
  P2-P : steer tok1@0, tok2@1, then NO steer - does the phrase (and then
         topic) persist through 20 tokens?
Phrases are 3 single-token words (grilled/chicken/soup, etc.).
a per token = 2*gap/97 + 0.02; anti-last 0.15 all steps after the script.

metrics: TRIGRAM contiguous plant, pair plant, per-word plant, rep4,
div, #tgt (any of the 3 in first 10). The PERSISTENCE question: with the
script only at 0-2, does the model KEEP emitting the phrase's words in
the 20-token rollout (sustained topic), or drift back to native?

Run: timeout 60 python3 -u eval_phrase3.py  # GEMMA-3-1B
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
PHRASES = [('grilled', 'chicken', 'soup'),
           ('fresh', 'chicken', 'dinner'),
           ('fried', 'chicken', 'rice')]
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

    def gen(mode, dirs):
        """dirs: list of (tid, g, a) |
        mode in ('P3') steer first 3 steps; 'P2' steer first 2;
        'P2-P' steer first 2 then anti-last only (persistence)."""
        allres = []
        nscript = {'P3': 3, 'P2': 2, 'P2-P': 2}[mode]
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                vv = vf
                if step < nscript:
                    tid, a, g = dirs[step]
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

    print(f"[{MODEL}] {PROMPT!r} native={tok.decode([native])!r} "
          f"3-token word scripts, NTOK={NTOK}")
    print(f"  {'phrase':>20} {'mode':>5} {'tri':>6} {'pair':>6} "
          f"{'t1':>4} {'t2':>4} {'t3':>4} {'rep4':>6} {'div':>6} "
          f"{'#phr':>5}  samples")

    def has_run(g, tids, rlen):
        for i in range(len(g) - rlen + 1):
            if all(g[i + j] == tids[j] for j in range(rlen)):
                return 1.0
        return 0.0

    for ws in PHRASES:
        dirs = [att_dir(w) for w in ws]
        tids = [d[0] for d in dirs]
        for mode in ('P3', 'P2', 'P2-P'):
            gs = gen(mode, dirs)
            tri = np.mean([has_run(g, tids, 3) for g in gs])
            pair = np.mean([has_run(g, tids[:2], 2) for g in gs])
            tp = [np.mean([1.0 if t in g[:10] else 0.0 for g in gs])
                  for t in tids]
            rp = np.mean([rep4(g) for g in gs])
            dv = np.mean([len(set(g)) / len(g) for g in gs])
            nph = np.mean([sum(1 for x in g if x in tids) for g in gs])
            print(f"  {' '.join(ws):>20} {mode:>5} {tri:>6.2f} "
                  f"{pair:>6.2f} {tp[0]:>4.2f} {tp[1]:>4.2f} "
                  f"{tp[2]:>4.2f} {rp:>6.2f} {dv:>6.2f} {nph:>5.1f}  "
                  f"{[tok.decode(g)[:44] for g in gs]}", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()