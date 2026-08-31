#!/usr/bin/env python3
"""eval_depth_register.py — FAST: depth x register map of soft steering.

Mechanism probe for the register finding (7eb2d33): L10-soft is a register
mixer - plants always, prose depends on the prompt's native looseness.
Hypothesis: depth of the soft hook controls soft->hard. Push far BEFORE
the sink region (gemma sink birth @L18) leaves layers to recalc the state
(soft, prose survives); push AT the sink leaves none (hard loop, prose
dies). Loose-native registers should survive later pushes than tight ones.

Matrix: depth x prompt, metrics plant/4rep (2 seeds).
Run: timeout 90 python3 -u eval_depth_register.py google/gemma-3-1b-it
     timeout 90 python3 -u eval_depth_register.py Qwen/Qwen2-0.5B-Instruct
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 16
SEEDS = 2
TARGET = 'chicken'
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30

PROMPTS = [
    ('loose', 'There once was a chicken'),
    ('tight', 'I bought a chicken at the market'),
]
if 'qwen' in MODEL.lower():
    DEPTHS = [2, 6, 8, 10, 12]  # 24 layers; riser ~ L8 (16 remaining)
else:
    DEPTHS = [6, 10, 14, 18]   # 26 layers; riser ~ L10


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight

    tid = tok(' ' + TARGET, add_special_tokens=False).input_ids
    if len(tid) != 1:
        print("target not single token; abort")
        return
    tid = int(tid[0])
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

    def gen(pidx, depth, seed=0, n=NTOK, top_p=0.9, temp=1.0):
        torch.manual_seed(seed)
        PROMPT = PROMPTS[pidx][1]
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        toks = []
        for step in range(n):
            hook = None
            if depth is not None:
                hook = model.model.layers[depth].register_forward_hook(
                    lambda m, i, o: rot(o, ALPHA))
            try:
                with torch.no_grad():
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

    def cell(depth, pidx):
        plants, reps = [], []
        for sd in range(SEEDS):
            toks = gen(pidx, depth, seed=sd)
            head = toks[:10]
            plants.append(1.0 if (tid in head or any(c in head for c in caplist)) else 0.0)
            reps.append(rep4(toks))
        return np.mean(plants), np.mean(reps)

    print(f"[{MODEL}] target={TARGET!r} alpha={ALPHA} "
          f"({SEEDS} seeds, depths {DEPTHS})")
    print(f"{'depth':>7} {'loose p/rep':>14} {'tight p/rep':>14}", flush=True)
    for pidx, (pname, _) in enumerate(PROMPTS):
        pass
    print(f"{'native':>7}  {cell(None, 0)[0]:.2f}/{cell(None, 0)[1]:.2f}"
          f"           {cell(None, 1)[0]:.2f}/{cell(None, 1)[1]:.2f}", flush=True)
    agg = {}
    for d in DEPTHS:
        lv = cell(d, 0)
        tv = cell(d, 1)
        agg[d] = (lv, tv)
        print(f"L{d:>3}    {lv[0]:.2f}/{lv[1]:.2f}           {tv[0]:.2f}/{tv[1]:.2f}",
              flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()