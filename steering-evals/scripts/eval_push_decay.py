#!/usr/bin/env python3
"""eval_push_decay.py — FAST: soft push HALF-LIFE.

Follows the L10 riser sweet spot (d0c6c52): sustained L10 a=0.30 on the
loose register gives plant+prose. Question: is the push's effect a
ONE-SHOT transient that decays (plant then drift back to natural
narration = ideal), or does it need sustained re-pushing every step?

Modes (L10, a=0.30): native / sustained / once@0 / once@4 / once@8.
Per-step: record target-in-top5 -> plant persistence across the run.
Metrics: plant (first 10), plant_tail (last 8 steps = did it decay?),
rep4 (tail prose), full sample string.

Run: timeout 90 python3 -u eval_push_decay.py google/gemma-3-1b-it
     timeout 90 python3 -u eval_push_decay.py Qwen/Qwen2-0.5B-Instruct
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 24
SEEDS = 4
TARGET = 'chicken'
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30
DEPTH = int(sys.argv[3]) - 1 if len(sys.argv) > 3 else 9  # 0-based

PROMPT = sys.argv[2] if len(sys.argv) > 2 else 'There once was a chicken'
MODES = ['native', 'sustained', 'pulse@2']


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

    def gen(mode, seed=0, n=NTOK, top_p=0.9, temp=1.0):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        toks, pres = [], []
        push_step = None
        if mode == 'sustained':
            push = lambda step: True
        elif mode.startswith('once@'):
            ps = int(mode.split('@')[1])
            push = lambda step: step == ps
        elif mode.startswith('pulse@'):
            pr = int(mode.split('@')[1])
            push = lambda step: step % pr == 0
        elif mode == 'native':
            push = lambda step: False
        else:
            raise ValueError(mode)
        for step in range(n):
            hook = None
            if push(step):
                hook = model.model.layers[DEPTH].register_forward_hook(
                    lambda m, i, o: rot(o, ALPHA))
            try:
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
            finally:
                if hook is not None:
                    hook.remove()
            top5 = torch.topk(L, 5).indices.tolist()
            pres.append(1.0 if (tid in top5 or any(c in top5 for c in caplist)) else 0.0)
            p = torch.softmax(L / temp, dim=0)
            q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
            keep = order[:int((cum <= top_p).sum()) + 1]
            m = torch.zeros_like(q); m[keep] = 1
            q = (q * m) / (q * m).sum()
            nxt = int(torch.multinomial(q, 1))
            toks.append(int(nxt))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return toks, pres

    def rep4(toks):
        if len(toks) < 8:
            return 1.0
        n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])

    print(f"[{MODEL}] target={TARGET!r} alpha={ALPHA} depth=L{DEPTH + 1} "
          f"n={NTOK} ({SEEDS} seeds)")
    print(f"{'mode':>10} {'plant':>6} {'tail':>6} {'rep4':>6}  sample", flush=True)
    for mode in MODES:
        plants, tails, reps, samples = [], [], [], []
        for sd in range(SEEDS):
            toks, pres = gen(mode, seed=sd)
            head = toks[:10]
            plants.append(1.0 if (tid in head or any(c in head for c in caplist)) else 0.0)
            tails.append(np.mean(pres[-8:]))      # target still in top5 late?
            reps.append(rep4(toks[12:]))          # prose of the tail
            samples.append(tok.decode(toks))
        ps = ' '.join(f"{s[12:]}" for s in samples)
        print(f"{mode:>10} {np.mean(plants):.2f}   {np.mean(tails):.2f}   "
              f"{np.mean(reps):.2f}   | {ps[:60]}", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()