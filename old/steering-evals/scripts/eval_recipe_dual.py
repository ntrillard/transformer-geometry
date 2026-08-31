#!/usr/bin/env python3
"""eval_recipe_dual.py — FAST: L10 soft steer (loop dissolve) + final steer
(plant) = topic AND natural prose on Gemma?

From eval_steer_depth: final alpha>=0.2 plants chicken but loops (4rep 0.9);
L10 alpha~0.2 dissolves the native loop (4rep 0.05) but doesn't plant.
Combine BOTH hooks in one generation: does L10 keep prose while final lands
the topic?

Modes:
  native     no steering
  final.2    final alpha=0.2              (plant, loop)
  l10.2      L10 alpha=0.2                (prose, no plant)
  comb       L10 alpha in {0.12,0.15,0.2} + final alpha in {0.15,0.2}

Run: timeout 120 python3 -u eval_recipe_dual.py google/gemma-3-1b-it
     timeout 90  python3 -u eval_recipe_dual.py Qwen/Qwen2-0.5B-Instruct
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'google/gemma-3-1b-it'
PROMPT = sys.argv[2] if len(sys.argv) > 2 else 'For dinner I made'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 16
SEEDS = 4
TARGET = sys.argv[3] if len(sys.argv) > 3 else 'chicken'


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

    def rot_hook(out, alpha):
        """generic rotation hook body toward Wt by alpha."""
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

    def gen(mode, l10_a, fin_a, seed=0, n=NTOK, top_p=0.9, temp=1.0):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        toks = []
        hooks = []
        # register the hooks (mode-dependent)
        if mode == 'final':
            hooks.append(model.model.norm.register_forward_hook(
                lambda m, i, o: rot_hook(o, fin_a)))
        elif mode == 'l10':
            hooks.append(model.model.layers[9].register_forward_hook(
                lambda m, i, o: rot_hook(o, l10_a)))
        elif mode == 'comb':
            hooks.append(model.model.layers[9].register_forward_hook(
                lambda m, i, o: rot_hook(o, l10_a)))
            hooks.append(model.model.norm.register_forward_hook(
                lambda m, i, o: rot_hook(o, fin_a)))
        try:
            for step in range(n):
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
                p = torch.softmax(L / temp, dim=0)
                q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
                keep = order[:int((cum <= top_p).sum()) + 1]
                m = torch.zeros_like(q); m[keep] = 1
                q = (q * m) / (q * m).sum()
                nxt = int(torch.multinomial(q, 1))
                toks.append(int(nxt))
                ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
                print(f"  [step {step}]", end='', flush=True)
            print()
        finally:
            for h in hooks:
                h.remove()
        return toks

    def rep4(toks):
        if len(toks) < 8:
            return 1.0
        n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])

    def run(mode, l10_a=0.0, fin_a=0.0):
        plants, reps, divs, samples = [], [], [], []
        for sd in range(SEEDS):
            toks = gen(mode, l10_a, fin_a, seed=sd)
            head = toks[:10]
            p = 1.0 if (tid in head or any(c in head for c in caplist)) else 0.0
            plants.append(p)
            reps.append(rep4(toks))
            divs.append(len(set(toks)) / len(toks))
            samples.append(tok.decode(toks)[:40])
        print(f"  {mode:>8}  plant {np.mean(plants):.2f}  4rep {np.mean(reps):.2f}  ",
              f"div {np.mean(divs):.2f}", flush=True)
        for i, s in enumerate(samples):
            print(f"           sd{i}: {s!r}", flush=True)

    print(f"[{MODEL}] prompt={PROMPT!r} target={TARGET!r}  (each ~{NTOK} tok)")
    run('native')
    l10a = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30
    run('l10', l10_a=l10a)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()