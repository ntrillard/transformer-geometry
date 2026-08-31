#!/usr/bin/env python3
"""eval_steer_depth.py — FAST: does steering depth survive to generation?

Target 'chicken' (Gemma's known escape basin). One-shot rotation toward
Wn[chicken] applied at DIFFERENT hook depths:
  'final'  the final norm output (proven recipe, baseline)
  layer k  the post-layerk residual state (k = 10/14/18/22 and NL for Gemma;
           10/14/18/22/24 for Qwen)
Then FREE-RUN 24 tokens. If rotations early survive the remaining stack,
plant should be near-final-level; if washed out, mid-depth fails.

Alpha sweep {0.1, 0.2, 0.3, 0.45} since the closed-from alpha* is only exact
at final depth. Metrics: plant (target token, case-insensitive, in first 10),
4-rep, diversity, sample.

Run: timeout 90 python3 -u eval_steer_depth.py google/gemma-3-1b-it
     timeout 90 python3 -u eval_steer_depth.py Qwen/Qwen2-0.5B-Instruct
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
ALPHAS = [0.12, 0.15, 0.2, 0.35]
NTOK = 12
SEEDS = 2
TARGET = 'chicken'


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    NL = model.config.num_hidden_layers
    depths = [f'L{k}' for k in (10,) if 1 <= k <= NL]

    # target row
    tid = tok(' ' + TARGET, add_special_tokens=False).input_ids
    if len(tid) != 1:
        print(f"target {TARGET!r} not single token; abort")
        return
    tid = tid[0]
    Wt = lm_w[tid].detach().float().cpu().numpy()
    capid = tok(' ' + TARGET.capitalize(), add_special_tokens=False).input_ids
    caplist = [int(c) for c in capid] if capid else [tid]

    def gen(depth_key, alpha, seed=0, n=NTOK, top_p=0.9, temp=1.0):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        toks = []
        hook = None
        if depth_key == 'final':
            def hook(mod, inp, out):
                out2 = out.clone()
                v = out2[:, -1, :].float().reshape(-1)
                vn = v / v.norm()
                t = Wt - (Wt @ vn.cpu().numpy()) * vn.cpu().numpy()
                t = t / (np.linalg.norm(t) + 1e-12)
                tg = torch.as_tensor(t, dtype=torch.float32, device=DEV)
                g = tg - (tg @ vn) * vn
                g = g / (g.norm() + 1e-8)
                v2 = vn * math.cos(alpha) + g * math.sin(alpha)
                out2[:, -1, :] = (v.norm() * v2).to(out.dtype)
                return out2
            hook = model.model.norm.register_forward_hook(hook)
        else:
            k = int(depth_key[1:])
            layer = model.model.layers[k - 1]
            def hook(mod, inp, out):
                out2 = out.clone()
                v = out2[:, -1, :].float().reshape(-1)
                vn = v / v.norm()
                t = Wt - (Wt @ vn.cpu().numpy()) * vn.cpu().numpy()
                t = t / (np.linalg.norm(t) + 1e-12)
                tg = torch.as_tensor(t, dtype=torch.float32, device=DEV)
                g = tg - (tg @ vn) * vn
                g = g / (g.norm() + 1e-8)
                v2 = vn * math.cos(alpha) + g * math.sin(alpha)
                out2[:, -1, :] = (v.norm() * v2).to(out.dtype)
                return out2
            hook = layer.register_forward_hook(hook)
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
        finally:
            hook.remove()
        return toks

    def rep4(toks):
        if len(toks) < 8:
            return 1.0
        n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])

    print(f"[{MODEL}] target={TARGET!r} prompt={PROMPT!r}  depths: {depths}")
    print(f"{'depth':>6} {'alpha':>5} {'plant':>5} {'4rep':>5} {'div':>5}  sample")
    for d in depths:
        for alpha in ALPHAS:
            plants, reps, divs = [], [], []
            samples = []
            for sd in range(SEEDS):
                toks = gen(d, alpha, seed=sd)
                head = toks[:10]
                plant = 1.0 if (tid in head or any(c in head for c in caplist)) else 0.0
                plants.append(plant)
                reps.append(rep4(toks))
                divs.append(len(set(toks)) / len(toks))
                samples.append(tok.decode(toks)[:44])
            print(f"{d:>6} {alpha:>5.2f} {np.mean(plants):>5.2f} "
                  f"{np.mean(reps):>5.2f} {np.mean(divs):>5.2f}  {samples[0]!r}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()