#!/usr/bin/env python3
"""eval_scramble_rank.py — FAST: is scramble target-dependent via final rank?

Follows af71861 (scramble law, target-dependent). Test: behavioral a* at a
FIXED mid depth (L9) for food vs city targets -> scramble per target ->
correlate with final-layer rank (blocker pressure). If high-rank targets
scramble harder, rank (the learner's top feature) governs the target part
of the stack transfer.

Design (<=10s): capture proxy v at L9 (1 fwd) + final-rank (1 fwd) +
per-target behavioral slope via gap at alpha=0, 0.25 (2 fwd/target).
4 targets x 2 fwd = 8 fwd total.

Run: timeout 60 python3 -u eval_scramble_rank.py google/gemma-3-1b-it
     timeout 60 python3 -u eval_scramble_rank.py Qwen/Qwen2-0.5B-Instruct
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
TARGETS = ['chicken', 'pizza', 'paris', 'tokyo']
D = 9  # fixed mid depth (L10, 0-based)


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    with torch.no_grad():
        L0 = model(ids).logits[0, -1].float()
    native = int(L0.argmax())
    Wn = lm_w[native].detach().float().cpu().numpy()

    # proxy residual at L9 (1 fwd)
    cap = {}
    layer = model.model.layers[D]

    def cap_out(m, i, o):
        cap['v'] = o[0, -1, :].float()

    h = layer.register_forward_hook(cap_out)
    with torch.no_grad():
        model(ids)
    h.remove()
    v = cap['v'].cpu().numpy()
    vn = v / (np.linalg.norm(v) + 1e-12)

    print(f"[{MODEL}] native={native!r} depth=L{D + 1}")
    print(f"  {'target':>8} {'rank':>5} {'gap0':>7} {'slopeB':>7} "
          f"{'behov':>7} {'scramble':>9}", flush=True)
    outs = []
    for tn in TARGETS:
        tids = tok(' ' + tn, add_special_tokens=False).input_ids
        if len(tids) != 1:
            print(f"  {tn}: skip (multi-token)")
            continue
        tid = int(tids[0])
        Wt = lm_w[tid].detach().float().cpu().numpy()
        A = float(vn @ (Wt - Wn))
        tau = Wt - (vn @ Wt) * vn
        B = float(tau @ (Wt - Wn)) / (np.linalg.norm(tau) + 1e-12)
        proxy = math.atan2(-A, B)
        gap0 = float(L0[tid] - L0[native])
        rank = int((L0 > L0[tid]).sum().item())

        # behavioral slope via gap at alpha=0.25 (1 fwd)
        def rot(out, alpha):
            vv = out[:, -1, :].float().reshape(-1)
            vvn = vv / vv.norm()
            g0 = tau / (np.linalg.norm(tau) + 1e-12)
            g0t = torch.as_tensor(g0, dtype=torch.float32, device=DEV)
            gg = g0t - (g0t @ vvn) * vvn
            gg = gg / (gg.norm() + 1e-8)
            v2 = vvn * math.cos(alpha) + gg * math.sin(alpha)
            out = out.clone()
            out[:, -1, :] = (vv.norm() * v2).to(out.dtype)
            return out

        hook = layer.register_forward_hook(
            lambda m, i, o: rot(o, 0.25))
        try:
            with torch.no_grad():
                L25 = model(ids).logits[0, -1].float()
        finally:
            hook.remove()
        gap25 = float(L25[tid] - L25[native])
        slope = (gap25 - gap0) / 0.25
        behav = -gap0 / max(abs(slope), 1e-9) if slope > 0 else float('nan')
        scram = abs(behav / proxy) if proxy != 0 else float('nan')
        outs.append((tn, rank, gap0, slope, behav, scram))
        print(f"  {tn:>8} {rank:>5} {gap0:>+7.2f} {slope:>+7.1f} "
              f"{behav:>+7.2f} {scram:>9.1f}", flush=True)

    ranks = np.array([o[1] for o in outs])
    scrams = np.array([o[5] for o in outs], dtype=float)
    ok = ~np.isnan(scrams)
    if ok.sum() > 2:
        cc = np.corrcoef(ranks[ok], scrams[ok])[0, 1]
        print(f"  corr(rank, scramble) = {cc:+.3f}", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()