#!/usr/bin/env python3
"""eval_realprompt_baseline.py — is Gemma's ' I' loop a HARNESS artifact?

gemma-3-1b-it is instruction-tuned. 'For dinner I made' raw = a 4-token
fragment; instruct models degenerate on fragments (loop). The Qwen runs
used 'Once upon a time, there was a' (story mode) and produced grammar.
Does Gemma ALSO write grammar on REAL prompts? If yes, the loop baseline
was ours, not the model's - and the gentle-walk recipe should produce
coherent steered text on Gemma too.

Baselines (no steering), 12 tok, top_p 0.9, seed 0:
  'For dinner I made'                  (the loop fragment, for contrast)
  'Once upon a time, there was a'      (story start, Qwen's prompt)
  'The capital of France is'
  'Today the weather is'
  'I went to the store and bought'
metrics per baseline: rep4, div, maxrun, #SEP, plant, + the DECODED TEXT.
Then: on the most grammatical baseline, run the GENTLE WALK (3 deg/step
closest-to-state city member) -> does Gemma produce topical AND coherent
steered text?

CSV: steering_geometry_results/realprompt_baseline_gemma_base.csv
Run: timeout 90 python3 -u eval_realprompt_baseline.py
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

GEMMA = 'google/gemma-3-1b-pt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 12
PROMPTS = [
    ('fragment', 'For dinner I made'),
    ('story', 'Once upon a time, there was a'),
    ('capital', 'The capital of France is'),
    ('weather', 'Today the weather is'),
    ('store', 'I went to the store and bought'),
]
TARGET = 'city'
OUT = Path('../steering_geometry_results/realprompt_baseline_gemma_base.csv')


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def main():
    t0 = time.time()
    model, tok = SGT.load_model(GEMMA, dtype='fp16')
    W = model.lm_head.weight.detach()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()
    fam = []
    for w in CLASSES[TARGET]:
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            fam.append(int(ids[0]))
    fset = set(fam)

    def gen(ids0, steer_deg=None):
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

        def closest_member(vv):
            u = vv / vv.norm()
            return fam[int((Wn[fam].float() @ u).argmax())]

        def rot_toward(vv, tid, amt):
            v1 = vv / vv.norm()
            Wb = Wn[tid].float()
            tau = Wb - (v1 @ Wb) * v1
            g = tau / tau.norm()
            return (v1 * math.cos(amt) + g * math.sin(amt)) * vv.norm()

        torch.manual_seed(0)
        ids = ids0.clone()
        toks = []
        for step in range(NTOK):
            vv = vf
            if steer_deg is not None:
                t = closest_member(vv)
                vv = rot_toward(vv, t, math.radians(steer_deg))

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
        return toks, native, L0

    rows = []
    print(f"[{GEMMA}] baseline sweep ({NTOK} tok, seed 0)")
    print(f"    {'prompt':>9} {'rep4':>6} {'div':>5} {'maxrun':>6} "
          f"{'#SEP':>5}  text")
    for pname, PROMPT in PROMPTS:
        ids0 = tok(PROMPT, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        toks, native, L0 = gen(ids0)
        rp = rep4(toks)
        div = len(set(toks)) / len(toks)
        mr = max((sum(1 for _ in grp) for _, grp in
                  itertools.groupby(toks)), default=0)
        nsep = sum(1 for t in toks if t == native)
        txt = tok.decode(toks)
        rows.append(dict(model=GEMMA, prompt=pname, prompt_text=PROMPT,
                         rep4=round(rp, 3), div=round(div, 3), maxrun=mr,
                         nsep=nsep, text=txt))
        print(f"    {pname:>9} {rp:>6.2f} {div:>5.2f} {mr:>6d} "
              f"{nsep:>5d}  {txt[:52]!r}", flush=True)

    # gentle-walk steer on the most grammatical baseline
    best = max(rows, key=lambda r: -r['rep4'])
    print(f"\n  most grammatical baseline: {best['prompt']!r} "
          f"(rep4 {best['rep4']:.2f})")
    ids0 = tok(best['prompt_text'], add_special_tokens=False,
               return_tensors='pt').input_ids.to(DEV)
    for deg in (3.0, 5.0):
        toks, native, L0 = gen(ids0, steer_deg=deg)
        x = sum(1 for t in toks if t in fset)
        dis = len({t for t in toks if t in fset})
        txt = tok.decode(toks)
        print(f"    walk{deg:.0f}deg -> rep4={rep4(toks):.2f} "
              f"city-w={x} dis={dis}  {txt[:52]!r}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()