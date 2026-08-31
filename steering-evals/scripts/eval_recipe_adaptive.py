#!/usr/bin/env python3
"""eval_recipe_adaptive.py — FAST: conditional soft-steering policy.

From eval_recipe_dual: L10-soft IMPROVES loose-native prompts (4rep 0.08,
0.25) but HURTS tight-native ones (4rep 0.65 vs 0.31). The adaptive policy:
measure the NEXT-STEP argmax confidence at the moment of steering; if loose
(entropy high / top-1 prob low) -> steer soft; if tight -> don't steer
(respect the native attractor).

Compare: native vs always-L10-soft vs adaptive L10-soft, on the 4 prompts.
Metric: plant (topic appears), 4rep (prose quality).

Run: timeout 120 python3 -u eval_recipe_adaptive.py google/gemma-3-1b-it
     timeout 90  python3 -u eval_recipe_adaptive.py Qwen/Qwen2-0.5B-Instruct
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = [
    ('dinner', 'For dinner I made'),
    ('foerst', 'There once was a chicken'),
    ('deer', 'I bought a chicken at the market'),
    ('store', 'I went to the store and bought'),
]
NTOK = 16
SEEDS = 3
TARGET = 'chicken'
ALPHA = 0.30
ENTROPY_TH = 2.0  # nat log; softmax entropy above this = 'loose'


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    NL = model.config.num_hidden_layers

    tid = tok(' ' + TARGET, add_special_tokens=False).input_ids
    if len(tid) != 1:
        print("target not single token; abort")
        return
    tid = tid[0]
    Wt = lm_w[tid].detach().float().cpu().numpy()
    capid = tok(' ' + TARGET.capitalize(), add_special_tokens=False).input_ids
    caplist = [int(c) for c in capid] if capid else [tid]

    def rot(out, alpha):
        v = out[:, -1, :].float().reshape(-1)
        vn = v / v.norm()
        t = Wt - (Wt @ vn.cpu().numpy()) * vn.cpu().numpy()
        t = t / (np.linalg.norm(t) + 1e-12)
        tg = torch.as_tensor(t, dtype=torch.float32, device=DEV)
        g = tg - (tg @ vn) * vn
        g = g / (g.norm() + 1e-8)
        v2 = vn * math.cos(alpha) + g * math.sin(alpha)
        out = out.clone()
        out[:, -1, :] = (v.norm() * v2).to(out.dtype)
        return out

    def gen(mode, seed=0, n=NTOK, top_p=0.9, temp=1.0):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        toks = []
        for step in range(n):
            # ---- steering hook ----
            hook = None
            if mode == 'soft':
                hook = model.model.layers[9].register_forward_hook(
                    lambda m, i, o: rot(o, ALPHA))
            elif mode == 'adaptive':
                # decide BEFORE the forward: look at the CURRENT prompt head
                with torch.no_grad():
                    Lpv = model(ids).logits[0, -1].float()
                pv = torch.softmax(Lpv, dim=0)
                ent = -torch.sum(pv * torch.log(pv + 1e-12)).item()
                if ent >= ENTROPY_TH:
                    hook = model.model.layers[9].register_forward_hook(
                        lambda m, i, o: rot(o, ALPHA))
                else:
                    L = Lpv
            try:
                with torch.no_grad():
                    if 'L' not in locals():
                        L = model(ids).logits[0, -1].float()
            finally:
                if hook is not None:
                    hook.remove()
            p = torch.softmax(L / temp, dim=0)
            q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
            keep = order[:int((cum <= top_p).sum()) + 1]
            m = torch.zeros_like(q); m[keep] = 1
            q = (q * m) / (q * m).sum()
            nxt = int(torch.multinomial(q, 1))
            toks.append(int(nxt))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return toks

    def rep4(toks):
        if len(toks) < 8:
            return 1.0
        n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])

    print(f"[{MODEL}] target={TARGET!r}  entropy-th {ENTROPY_TH}  "
          f"(each row: plant / 4rep, {SEEDS} seeds)")
    print(f"{'prompt':>7} {'native':>12} {'soft':>12} {'adaptive':>13}")
    for pname, PROMPT in PROMPTS:
        line = f"{pname:>7}"
        for mode in ('native', 'soft', 'adaptive'):
            plants, reps = [], []
            for sd in range(SEEDS):
                toks = gen(mode, seed=sd)
                head = toks[:10]
                plants.append(1.0 if (tid in head or any(c in head for c in caplist)) else 0.0)
                reps.append(rep4(toks))
            line += f"  {np.mean(plants):.2f}/{np.mean(reps):.2f}"
        print(line)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()
