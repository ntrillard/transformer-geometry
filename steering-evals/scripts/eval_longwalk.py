#!/usr/bin/env python3
"""eval_longwalk.py — does the ONLY coherent+topical config (Qwen-raw +
gentle walk, from eval_gentle_walk) survive LONGER prompts?

Direct follow-up to the review request ("full coherence on even longer
prompts, right now these are all failing"). The honest data says:
  - readout-only control surface (mid falsified at every scale, c3e1426)
  - alpha = 2*(gap/97)+0.02 is the normalization (readout scale)
  - FULL COHERENCE is a MODEL property: the only coherent+topical cell
    is Qwen-raw + gentle walk; Gemma trades grammar for topic at ANY
    steer, ANY depth, ANY scale.
So the one remaining question: does the Qwen walk hold its coherence
when the prompt gets LONG (15-21 tok vs its original 4-5)? And does
base Gemma (the one-model constraint) confirm the model-property claim
at long context too?

Mechanism EXACTLY as eval_gentle_walk: capture the natural final-norm
state vf ONCE, then per-step rotate vf toward the closest city family
member by walk* deg (re-aim from the frozen state, non-cumulative).

Prompts (real, long):
  Qwen  : 'Once upon a time, there was a'                    (5 tok)
          'Every year my family takes a trip, and we always end up in a different'   (~15 tok)
          'Last summer I traveled across Europe and visited many beautiful capital cities, but the one that impressed me the most was'  (~21 tok)
  Gemma-pt (HF_TOKEN): 'The capital of France is'            (4 tok)
          'If you ask me which European city is the most beautiful, I would say that'  (~17 tok)

metrics: plant (city word first 10), xtgt, dis, div, maxrun, rep4,
nsep, + DECODED TEXT. CSV persisted.
Run: HF_TOKEN=<tok> timeout 240 python3 -u eval_longwalk.py
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

QUWEN = 'Qwen/Qwen2-0.5B-Instruct'
GEMMA = 'google/gemma-3-1b-pt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 12
SEEDS = [0, 7]
TARGET = 'city'
OUT = Path('../steering_geometry_results/longwalk.csv')

PROMPTS = {
    QUWEN: [
        ('short', 'Once upon a time, there was a'),
        ('med', 'Every year my family takes a trip, and we always end up in a different'),
        ('long', 'Last summer I traveled across Europe and visited many beautiful capital cities, but the one that impressed me the most was'),
    ],
    GEMMA: [
        ('short', 'The capital of France is'),
        ('long', 'If you ask me which European city is the most beautiful, I would say that'),
    ],
}


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def build_family(tok, cls='city'):
    fam = []
    word2id = {}
    for w in CLASSES[cls]:
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
            fam.append(int(ids[0]))
    return fam, word2id


def run_model(model_name, prompts, seeds, modes):
    model, tok = SGT.load_model(model_name, dtype='fp16')
    W = model.lm_head.weight.detach()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()
    fam, word2id = build_family(tok, TARGET)
    fset = set(fam)
    print(f"\n[{model_name}] target-family={TARGET} "
          f"family-n={len(fam)}")

    def closest_member(vv):
        u = vv / vv.norm()
        return fam[int((Wn[fam].float() @ u).argmax())]

    def rot_toward(vv, tid, amt):
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(amt) + g * math.sin(amt)) * vv.norm()

    rows = []
    for pname, PROMPT in prompts:
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
        nname = tok.decode([native])
        plen = ids0.shape[1]
        print(f"\n  prompt[{pname}] ({plen} tok) native={nname!r} "
              f"logit_city_top={float(L0[fam].max()):.1f}")

        def gen(mode, sd):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            deg = {'walk3': 3.0, 'walk5': 5.0}.get(mode)
            for step in range(NTOK):
                vv = vf
                if deg is not None:
                    t = closest_member(vv)
                    vv = rot_toward(vv, t, math.radians(deg))

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
            return toks

        print(f"    {'mode':>8} {'seed':>4} {'plant':>6} {'xtgt':>5} "
              f"{'dis':>4} {'div':>5} {'maxrun':>6} {'rep4':>5}  text")
        for mode in modes:
            for sd in seeds:
                g = gen(mode, sd)
                x = sum(1 for t in g if t in fset)
                dis = len({t for t in g if t in fset})
                plant = 1.0 if any(t in g[:10] for t in fset) else 0.0
                div = len(set(g)) / len(g)
                mr = max((sum(1 for _ in grp) for _, grp in
                          itertools.groupby(g)), default=0)
                rp = rep4(g)
                nsep = sum(1 for t in g if t == native)
                txt = tok.decode(g)
                rows.append(dict(model=model_name, prompt=pname,
                                 plen=plen, mode=mode, seed=sd,
                                 plant=plant, xtgt=x, dis=dis,
                                 div=round(div, 3), maxrun=mr,
                                 rep4=round(rp, 3), nsep=nsep, text=txt))
                print(f"    {mode:>8} {sd:>4d} {plant:>6.1f} {x:>5d} "
                      f"{dis:>4d} {div:>5.2f} {mr:>6d} {rp:>5.2f}  "
                      f"{txt[:56]!r}", flush=True)
    return rows


def main():
    t0 = time.time()
    modes = ['baseline', 'walk3', 'walk5']
    all_rows = []
    all_rows += run_model(QUWEN, PROMPTS[QUWEN], SEEDS, modes)
    all_rows += run_model(GEMMA, PROMPTS[GEMMA], [0], ['baseline', 'walk5'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()