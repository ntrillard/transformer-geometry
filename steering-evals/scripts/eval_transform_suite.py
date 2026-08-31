#!/usr/bin/env python3
"""eval_transform_suite.py — the residual-state rotation vs OTHER state
transformations, LONG generations to watch (20-30s, one model, no template).

Core idea kept: grab the natural final-norm residual state v, compute the
target direction (tangent g toward closest city row) + law-budget alpha =
2*(gap/97)+0.02. But instead of ONLY the plain rotation, apply DIFFERENT
transformations of the same steering intent and free-run long text so you
can SEE how each shapes generation.

Transformations (all equal-norm: input to the LM head stays ||v_final||):
  baseline : NO steer (native generation, reference)
  rot      : plain rotation v' = R(a){v} -> g   (the committed recipe)
  fourrier : rotate the state in the FREQUENCY domain: FFT(v), rotate the
             low-band components toward FFT(g), IFFT back, renormalize
  adddelta : additive subspace step v' = normalize(v + alpha*||v||*g)*||v||
             (no Schur/tangent projection - pure vector add)
  freqlow  : keep only the LOW-frequency half of the steering delta d=v'-v,
             add back, renormalize  (smooth/topic-scale edit)
  freqhigh : keep only the HIGH-frequency half of d  (sharp/token-scale edit)

Each runs a LONG free-run (NTOK tokens) -> you watch the text. Metrics
per transform: plant, xtgt (#city), div, rep4, maxrun. Text printed live.

Run: HF_TOKEN=<tok> timeout 30 python3 -u eval_transform_suite.py
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
NTOK = 60                      # LONG generation - watch
SEEDS = [0]
TARGET = 'city'
OUT = Path('../steering_geometry_results/transform_suite.csv')

PROMPTS = [
    ('ask', 'If you ask me which European city is the most beautiful, I would say that'),
    ('fr',  'The capital of France is'),
]
TRANSFORMS = ['baseline', 'rot', 'fourrier', 'adddelta', 'freqlow', 'freqhigh']


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

    def transform(v, g, alpha, mode):
        """map natural state v -> steered state (equal norm preserved)."""
        H = v.numel()
        vn = v / v.norm()
        if mode == 'rot':
            return (vn * math.cos(alpha) + g * math.sin(alpha)) * v.norm()
        if mode == 'fourrier':
            # rotate in frequency domain: rotate only a low band of the
            # state toward the low band of g, IFFT back, renormalize.
            b = max(1, int(H * 0.25))
            Vf = torch.fft.fft(v.detach().float())
            Gf = torch.fft.fft(g.detach().float())
            Vs = Vf.clone()
            band = torch.linspace(0, math.pi / 2, b, device=Vf.device)
            cosv = torch.cos(band)
            sinv = torch.sin(band)
            # rotate each low-freq complex component toward g's component
            Vs[:b] = Vf[:b] * cosv + Gf[:b] * sinv
            out = torch.fft.ifft(Vs).real.float()
            return out / out.norm() * v.norm()
        if mode == 'adddelta':
            out = v + alpha * v.norm() * g
            return out / out.norm() * v.norm()
        if mode in ('freqlow', 'freqhigh'):
            vr = transform(v, g, alpha, 'rot')
            d = (vr - v).detach().float()
            Df = torch.fft.fft(d)
            Hh = int(H / 2)
            if mode == 'freqlow':
                Df[Hh:] = 0
            else:
                Df[:Hh] = 0
            dl = torch.fft.ifft(Df).real.float()
            out = v + dl
            return out / out.norm() * v.norm()
        raise ValueError(mode)

    rows = []
    for pname, pr in PROMPTS:
        ids0 = tok(pr, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        vf, Ln = nat_vL(ids0)
        nat_top = int(Ln.argmax())
        gap = float(Ln[nat_top] - Ln[fam].max())
        alpha = 2 * (gap / 97.0) + 0.02
        tgt = closest(vf)
        tgt2 = closest(vf)
        v1 = vf / vf.norm()
        Wb = Wn[tgt].float()
        tau = Wb - (v1 @ Wb) * v1
        g = (tau / tau.norm()).float()
        print(f"\n==== {pname}: {pr!r}  gap={gap:.1f} alpha={alpha:.3f} "
              f"-> {tok.decode([tgt])!r}  ({NTOK}-tok free-run) ====")
        for mode in TRANSFORMS:
            torch.manual_seed(0)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                hooks = []
                if mode != 'baseline' and step == 0:
                    vs = transform(vf, g, alpha, mode)
                    def inj(m, i, o, p=vs):
                        out = o.clone()
                        out[0, -1, :] = torch.as_tensor(
                            p, dtype=out.dtype, device=out.device)
                        return out
                    hooks.append(model.model.norm.register_forward_hook(inj))
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
                for h in hooks:
                    h.remove()
                nxt = sample(L)
                toks.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)],
                                dim=1)
            x = sum(1 for t in toks if t in fset)
            dis = len({t for t in toks if t in fset})
            plant = 1.0 if any(t in toks[:10] for t in fset) else 0.0
            div = len(set(toks)) / len(toks)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(toks)), default=0)
            rp = rep4(toks)
            txt = tok.decode(toks)
            rows.append(dict(prompt=pname, transform=mode, gap=round(gap, 3),
                             alpha=round(alpha, 3), plant=plant, xtgt=x,
                             dis=dis, div=round(div, 3), maxrun=mr,
                             rep4=round(rp, 3), text=txt))
            print(f"\n--- [{mode:>9}] plant={plant:.0f} xtgt={x} "
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