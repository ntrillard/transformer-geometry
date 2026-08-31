#!/usr/bin/env python3
"""eval_readout2.py — the last 2 steps: final norm (model.model.norm)
and the lm_head. Where does readout steering actually live?

Dataflow at the readout:
  layers ... -> residual h -> model.model.norm -> v (lm_head input)
  v -> model.lm_head (W^T) -> logits

All committed recipes rotate v AT the norm output. Because the norm is
post-all-layers, that rotation changes ONLY this step's logits - the
next step's state comes from the pre-norm residual stream, untouched.
So the same effect must be reproducible by hooking the HEAD instead and
adding the equivalent linear correction to the logits. Is it?

Modes (policy FIXED; only the hook point varies):
  norm_only : rotation at model.model.norm  (the committed recipe)
  head_full : NO norm inject; at lm_head output add
              delta = (v' - v) @ W^T   (v' = same rotated v, computed
              from the same captured state) -> mathematically identical
              logits -> must reproduce norm_only text-for-text
  head_single : add ONLY the target-token scalar logit bump delta_t
              (pure-logit control; NOT a state rotation - the
              non-geometric corner that the old 'trivial boost' probe
              showed loops).

Per step: capture natural v + logits (capture pass), gap = nat_probe
proxy, alpha = 2*(gap/97)+0.02, t = closest city member of v, v' =
rot(v, t, alpha); then GENERATE with the chosen hook. Seeds {0,7}.

Metrics: plant/xtgt/dis/div/maxrun/rep4/nsep + text. CSV persisted.
Run: HF_TOKEN=<tok> timeout 150 python3 -u eval_readout2.py
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
PROMPT = 'The capital of France is'
NTOK = 12
SEEDS = [0, 7]
TARGET = 'city'
OUT = Path('../steering_geometry_results/readout2.csv')


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def build_family(tok, cls='city'):
    fam = []
    for w in CLASSES[cls]:
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            fam.append(int(ids[0]))
    return fam


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach().float()          # [V, H]
    V, H = W.shape
    Wn = (W / W.norm(dim=1, keepdim=True)).float()
    fam = build_family(tok, TARGET)
    fset = set(fam)
    Wff = Wn[fam].float()
    ids0 = tok(PROMPT, add_special_tokens=False,
               return_tensors='pt').input_ids.to(DEV)
    root_ids = ids0.clone()
    cf = {}

    def capture(m, i, o):
        cf['v'] = o[0, -1, :].float()

    hcap = model.model.norm.register_forward_hook(capture)
    with torch.no_grad():
        L0 = model(root_ids).logits[0, -1].float()
    hcap.remove()
    vf0 = cf['v'].float()
    native = int(L0.argmax())
    print(f"[{MODEL}] {PROMPT!r} v={{norm:{float(vf0.norm()):.1f}}} "
          f"V={V} H={H} native={tok.decode([native])!r}")

    def closest(vv):
        u = vv / vv.norm()
        return fam[int((Wff @ u).argmax())]

    def rot(vv, tid, alpha):
        v1 = vv / vv.norm()
        Wb = Wn[tid].float()
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(alpha) + g * math.sin(alpha)) * vv.norm()

    def get_nat(ids):
        # capture pass: returns natural last-state v and logits
        cv = {}
        hk = model.model.norm.register_forward_hook(
            lambda m, i, o: cv.__setitem__('v', o[0, -1, :].float()))
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        hk.remove()
        return cv['v'], L

    def sample(L):
        p = torch.softmax(L.float(), dim=0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk) / (q * msk).sum()
        return int(torch.multinomial(qq, 1))

    def gen(mode, sd):
        torch.manual_seed(sd)
        ids = root_ids.clone()
        toks = []
        for step in range(NTOK):
            vf, Ln = get_nat(ids)
            nat_top = int(Ln.argmax())
            gap = float(Ln[nat_top] - Ln[fam].max())
            alpha = 2 * (gap / 97.0) + 0.02
            t = closest(vf)
            vp = rot(vf, t, alpha)

            def inj(m, i, o, p=None):
                out = o.clone()
                if p is not None:
                    out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                    device=out.device)
                return out

            def head(m, i, o, p=None):
                out = o.clone()
                if p is not None:
                    out[0, -1, :] = (out[0, -1, :].float() + p).to(out.dtype)
                return out

            h1 = h2 = None
            if mode == 'norm_only':
                h1 = model.model.norm.register_forward_hook(
                    lambda m, i, o, q=vp: inj(m, i, o, q))
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
            elif mode == 'head_full':
                delta = (vp - vf).float() @ W.t()          # [V]
                h2 = model.lm_head.register_forward_hook(
                    lambda m, i, o, q=delta: head(m, i, o, q))
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
            else:  # head_single
                delta_t = float((vp - vf).float() @ W.t()[t])
                h2 = model.lm_head.register_forward_hook(
                    lambda m, i, o, q=delta_t: head(m, i, o, q))
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
            h1 and h1.remove()
            h2 and h2.remove()
            nxt = sample(L)
            toks.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        return toks

    modes = ['norm_only', 'head_full', 'head_single']
    print(f"    {'mode':>11} {'seed':>4} {'plant':>6} {'xtgt':>5} "
          f"{'dis':>4} {'div':>5} {'maxrun':>6} {'rep4':>5}  text")
    rows = []
    for md in modes:
        for sd in SEEDS:
            g = gen(md, sd)
            x = sum(1 for t in g if t in fset)
            dis = len({t for t in g if t in fset})
            plant = 1.0 if any(t in g[:10] for t in fset) else 0.0
            div = len(set(g)) / len(g)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(g)), default=0)
            rp = rep4(g)
            txt = tok.decode(g)
            rows.append(dict(model=MODEL, prompt=PROMPT, mode=md, seed=sd,
                             plant=plant, xtgt=x, dis=dis,
                             div=round(div, 3), maxrun=mr,
                             rep4=round(rp, 3), text=txt))
            print(f"    {md:>11} {sd:>4d} {plant:>6.1f} {x:>5d} "
                  f"{dis:>4d} {div:>5.2f} {mr:>6d} {rp:>5.2f}  "
                  f"{txt[:56]!r}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()