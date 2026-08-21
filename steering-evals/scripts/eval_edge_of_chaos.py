#!/usr/bin/env python3
"""Edge-of-chaos measurements + pit trajectory stability.

1. Per-layer Lyapunov exponent lambda_l: perturb input embeddings by eps along a
   random direction, propagate both trajectories, and record
   lambda_l = ln(||dh_{l+1}|| / ||dh_l||).  Report mean lambda over layers and
   lambda*L (L = num layers).

2. Pit trajectory stability: during a greedy pit loop, measure
     c_n   = cos(h_n, h_{n+1})            (direction lock-in)
     r_n   = ||h_{n+2}-h_{n+1}|| / ||h_{n+1}-h_n||   (local expansion factor)
   and compare against ordinary generation.  A fixed point has r_n -> 0,
   i.e. lambda -> -infinity locally; "edge of chaos" generation should sit near
   r_n ~ 1.

Run: python eval_edge_of_chaos.py --model Qwen/Qwen2-0.5B-Instruct [--pit-id 15]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import eval_defense as E
import steering_geometry_test as M

OUT = Path("../steering_geometry_results")


@torch.no_grad()
def lyapunov_profile(model, tok, prompts, eps=1e-2, n_seeds=3):
    """Mean per-layer lambda over prompts x seeds."""
    L = model.config.num_hidden_layers
    emb = model.get_input_embeddings()
    lam = np.zeros(L)
    count = 0
    for p in prompts:
        ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        base = emb(ids)
        for s in range(n_seeds):
            g = torch.Generator(device="cpu").manual_seed(42 + s)
            noise = torch.randn(base.shape[-1], generator=g).to(base.device)
            noise = (noise / noise.norm()).to(base.dtype)
            hp = base + eps * noise
            out_r = model(inputs_embeds=base, output_hidden_states=True)
            out_p = model(inputs_embeds=hp, output_hidden_states=True)
            d = [(out_p.hidden_states[l] - out_r.hidden_states[l])[0, -1].float().norm().item()
                 for l in range(L + 1)]
            for l in range(L):
                if d[l] > 0 and d[l + 1] > 0:
                    lam[l] += np.log(d[l + 1] / d[l])
            count += 1
    return lam / max(count, 1)


@torch.no_grad()
def trajectory_stats(model, head, seed_ids, pit, max_new=25):
    """Greedy generation recording last-layer states each step."""
    ids = list(seed_ids)
    hs = []
    for _ in range(max_new):
        h = E.state_after_tokens(model, ids)
        hs.append(h.clone())
        logits = h.float() @ head.T
        ids.append(int(logits.argmax()))
    return ids, torch.stack(hs)


def expand_ratio(hs):
    """r_n = ||h_{n+2}-h_{n+1}|| / ||h_{n+1}-h_n||, plus mean cos(h_n,h_{n+1})."""
    d = [float((hs[i + 1] - hs[i]).norm()) for i in range(len(hs) - 1)]
    r = [d[i + 1] / d[i] for i in range(len(d) - 1) if d[i] > 1e-8]
    cs = [float(torch.cosine_similarity(hs[i], hs[i + 1], dim=0))
          for i in range(len(hs) - 1)]
    return float(np.median(r)), float(np.mean(cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--pit-id", type=int, default=None)
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tok = M.load_model(args.model, dtype="fp16")
    head = torch.as_tensor(
        (lambda w: w)(model.lm_head.weight.detach().float().cpu().numpy()),
        device=model.device)
    if head.numel() == 0:
        head = torch.as_tensor(model.get_input_embeddings().weight.detach()
                               .float().cpu().numpy(), device=model.device)

    L = model.config.num_hidden_layers
    safe = args.model.replace("/", "--")

    # --- 1. Lyapunov profile ---
    lam = lyapunov_profile(model, tok, M.PROMPTS[:8])
    mid = slice(L // 4, 3 * L // 4)
    mean_lam = float(np.median(lam[mid]))
    print(f"\nLyapunov: median mid-layer lambda={mean_lam:.4f}  "
          f"lambda*L={mean_lam * L:.3f}  (min {lam.min():.3f} @l{int(lam.argmin())}, "
          f"max {lam.max():.3f})")
    pd.DataFrame({"layer": range(L), "lambda": lam}).to_csv(
        OUT / f"lyapunov__{safe}.csv", index=False)

    # --- 2. Trajectory stability: pit loop vs normal ---
    rows = []
    if args.pit_id is not None:
        pit = args.pit_id
        _, hs = trajectory_stats(model, head, [pit] * 5, pit)
        r, c = expand_ratio(hs)
        print(f"Pit loop ({tok.decode([pit])!r}): median expansion r={r:.4g}  "
              f"mean cos={c:.6f}")
        rows.append(dict(condition=f"pit-{pit}", expansion_r=r, mean_cos=c))

    for p in M.PROMPTS[:5]:
        seed = tok(p, add_special_tokens=False).input_ids[:5]
        _, hs = trajectory_stats(model, head, seed, None)
        r, c = expand_ratio(hs)
        rows.append(dict(condition=f"normal:{p[:18]}", expansion_r=r, mean_cos=c))
        print(f"Normal '{p[:24]}': median expansion r={r:.4g}  mean cos={c:.6f}")

    pd.DataFrame(rows).to_csv(OUT / f"trajectory__{safe}.csv", index=False)
    print(f"\nSaved -> {OUT}/lyapunov__{safe}.csv, trajectory__{safe}.csv")


if __name__ == "__main__":
    main()
