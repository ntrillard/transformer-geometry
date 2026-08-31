#!/usr/bin/env python3
"""eval_metasweep.py — META-SWEEP THE CONTROL VARIABLES (big-leap, 30s).

metamode2-6 meta-learned FEATURES against coherence proxies and real
single-seed labels; the honest finding was that cheap natural features
do not predict real free-run coherence (entropy proxy inverted, cosang
fails robustness). The real direction is to sweep the CONTROL VARIABLES
themselves with REAL multi-seed rollout coherence as the label.

Controls (the only levers we have at the readout):
  anti_window : how many steps the planted target token stays suppressed
                at lm_head after the step-0 graft
                (1 = shot_anti, 2 = two steps, 3 = three steps, 0 = never/
                control = per-step re-graft walk)
  alpha       : the law-budget steer amplitude at step 0

Label per (prompt, window, alpha): SEED-ROBUST coherence = strict-coherent
on >=2/3 independent seeds (strict: plant, div>=0.7, rep4==0, maxrun<=2,
no <eos>, no token >2x). The big-leap question: is there an anti_window
that robustly yields coherent prose ACROSS prompts (a regime, not a
single lucky prompt)? And does it beat the 1-step default everywhere?

Budget(30s): 3 prompts x 2 windows {1,3} x 3 seeds x 5 tok ~ 18 rollouts.
One model, no template.
Run: HF_TOKEN=<tok> timeout 28 python3 -u eval_metasweep.py
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
NTOK = 5
SEEDS = [0, 1, 2]
TARGET = 'city'
OUT = Path('../steering_geometry_results/metasweep.csv')

PROMPTS = [
    ('ask', 'If you ask me which European city is the most beautiful, I would say that'),  # known robust-coherent
    ('fr',  'The capital of France is'),                                                    # collapse
    ('jpn', 'The capital of Japan is'),                                                     # mixed
]
WINDOWS = [1, 3]      # anti suppression steps after the graft
# amp factor list (law budget * f)
AMPS = [1.0]


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

    def nat_vL(ids):
        cv = {}
        hk = model.model.norm.register_forward_hook(
            lambda m, i, o: cv.__setitem__('v', o[0, -1, :].float()))
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        hk.remove()
        return cv['v'], L

    def sample(L):
        p = torch.softmax(L.float(), 0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk) / (q * msk).sum()
        return int(torch.multinomial(qq, 1))

    def rollout(ids0, vp, tgt, window, sd):
        torch.manual_seed(sd)
        ids = ids0.clone()
        toks = []
        for step in range(NTOK):
            hooks = []
            if step == 0:
                def inj(m, i, o, p=vp):
                    out = o.clone()
                    out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                    device=out.device)
                    return out
                hooks.append(model.model.norm.register_forward_hook(inj))
            if step >= 1 and step <= window:
                def anti(m, i, o, tid=tgt):
                    out = o.clone()
                    out[0, -1, tid] = -30.0
                    return out
                hooks.append(model.lm_head.register_forward_hook(anti))
            with torch.no_grad():
                L = model(ids).logits[0, -1].float()
            for h in hooks:
                h.remove()
            nxt = sample(L)
            toks.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        return toks

    def strict_ok(toks, txt):
        if not any(t in toks for t in fset):
            return False
        if '<eos>' in txt:
            return False
        if len(set(toks)) / len(toks) < 0.7:
            return False
        if rep4(toks) != 0.0:
            return False
        mr = max((sum(1 for _ in grp) for _, grp in
                  itertools.groupby(toks)), default=0)
        if mr > 2:
            return False
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        if any(c > 2 for c in counts.values()):
            return False
        return True

    rows = []
    print(f"[{MODEL}] CONTROL sweep: anti_window x alpha, seed-robust "
          f"coherence ({len(SEEDS)} seeds, strict)")
    print(f"  {'prompt':<6}{'win':>4}{'amp':>5} | per-seed  robust  text"
          )
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        for fac in AMPS:
            alpha = fac * (2 * (gap / 97.0) + 0.02)
            tgt = closest(vf)
            vp = rot(vf, tgt, alpha)
            for window in WINDOWS:
                oks = []
                str_seeds = ''
                txts = []
                for sd in SEEDS:
                    toks = rollout(ids0, vp, tgt, window, sd)
                    txt = tok.decode(toks)
                    o = strict_ok(toks, txt)
                    oks.append(o)
                    str_seeds += 'Y' if o else 'n'
                    txts.append(txt)
                robust = 1 if sum(oks) >= 2 else 0
                rows.append(dict(prompt=pname, window=window, amp=fac,
                                 gap=round(gap, 3), alpha=round(alpha, 3),
                                 per_seed=str_seeds, n_coherent=sum(oks),
                                 robust=robust,
                                 text_w0=txts[0][:28],
                                 text_w1=txts[1][:28],
                                 text_w2=txts[2][:28]))
                print("  %-6s %4d %5.2f | %s  %d   %s / %s / %s"
                      % (pname[:6], window, fac, str_seeds, robust,
                         txts[0][:22].strip(), txts[1][:22].strip(),
                         txts[2][:22].strip()), flush=True)

    # aggregate: which window is robust across prompts?
    print("\n-- ROBUSTNESS TO SWEEP (window -> robust across prompts)")
    for window in WINDOWS:
        for fac in AMPS:
            sel = [r for r in rows if r['window'] == window and r['amp'] == fac]
            nr = sum(r['robust'] for r in sel)
            print(f"  window={window} amp={fac}: robust {nr}/{len(sel)} "
                  f"prompts   ({[r['prompt'] for r in sel if r['robust']]})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()