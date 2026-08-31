#!/usr/bin/env python3
"""Multi-goal steering battery: beyond rank-1.

The LM head is a frozen SOM-like prototype grid (GfG Kohonen-maps analogy):
rows W_j are prototypes, the argmax token is the competitive winner
(Voronoi cell on the sphere), and the tangent steer is a competitive-learning
update applied to the STATE instead of the prototype.  This battery measures
goals other than binary rank-1, following that analogy:

  G1  rank-1 endpoint rate                 (existing claim, recomputed)
  G2  top-k endpoint rate (k = 3,5,10)     (graded reach: among k nearest prototypes)
  G3  winner-hold window                   (arc-angle width where target is argmax)
  G4  endpoint margin                      (target logit - best competitor, raw + rel)
  G5  no-worse rank fraction               (topology/ordering preservation along arc)
  G6  Voronoi-vs-logit identity break      (does argmax_logit == argmax_metric? bias/norm effect)
  Controls: wrong-target and random tangents for G1/G2 (all near 0 expected).
  Crowding: k-NN mean distance of each target row to the FULL-vocab prototype
            set -> is reachability a SOM-local-density property?

Run:
  python eval_multi_goal_steering.py --model Qwen/Qwen2-0.5B-Instruct \
        --targets 64 --contexts 2 --layer-fracs 0.0,0.33,0.67,0.99
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import steering_geometry_test as M

OUT = Path("../steering_geometry_results")


@torch.no_grad()
def _rank1_window_batched(uL, tauL, tid_idx, max_angle):
    """Return per-target (entry_deg, exit_deg) of the rank-1 window along the arc.

    Target beats competitor j where (P_j)cos d + (Q_j)sin d > 0, P = uL[t]-uL[j],
    Q = tauL[t]-tauL[j].  Each competitor contributes one contiguous target-ahead
    interval; the intersection over all j is [max lo_j, min hi_j].  Same math as
    _rank1_analytic_batched but returns the full window (exit may exceed budget).
    """
    P = uL[tid_idx][:, None] - uL[None, :]
    Q = tauL.gather(1, tid_idx[:, None]) - tauL
    R = torch.hypot(P, Q)
    cos_ = torch.where(R > 0, P / R.clamp_min(1e-12), torch.zeros_like(P))
    sin_ = torch.where(R > 0, Q / R.clamp_min(1e-12), torch.zeros_like(P))
    th = torch.atan2(sin_, cos_)
    lo, hi = th - math.pi / 2, th + math.pi / 2
    twopi = 2 * math.pi
    K = len(tid_idx)
    entry = torch.full((K,), float("nan"), device=P.device)
    exit_ = torch.full((K,), float("nan"), device=P.device)
    for k in range(K):
        lo_j = torch.full((len(P[k]),), float("inf"), device=P.device)
        hi_j = torch.full((len(P[k]),), float("inf"), device=P.device)
        pos0 = P[k] > 1e-12
        eps = 1e-9
        # competitors ahead at d=0 (P<=0): target first wins at +pi/2 branch
        for j in range(0, 3):
            L = (lo[k] + twopi * j).clamp(min=0.0)
            H = (hi[k] + twopi * j).clamp(max=float("inf"))
            ok = L < H
            better = (L < lo_j) | torch.isinf(lo_j)
            lo_j = torch.where(better & ok, L, lo_j)
            hi_j = torch.where(better & ok & ~pos0, torch.minimum(hi_j, H), hi_j)
        # competitors behind at d=0 (pos0): target ahead from 0 until exit
        for j in range(-1, 1):
            L = (lo[k] + twopi * j).clamp(min=0.0)
            H = (hi[k] + twopi * j)
            ok = L < H
            tou = L <= eps
            sel = pos0 & tou & ok
            lo_j = torch.where(sel, L, lo_j)
            hi_j = torch.where(sel, H, hi_j)
        lo_all = lo_j.max()
        hi_all = hi_j.min()
        if torch.isfinite(lo_all) and lo_all <= hi_all:
            entry[k] = math.degrees(lo_all.item())
            exit_[k] = math.degrees(hi_all.item())
    return entry, exit_


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--targets", type=int, default=64)
    ap.add_argument("--contexts", type=int, default=2)
    ap.add_argument("--layer-fracs", default="0.0,0.33,0.67,0.99")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=float, default=17.0, help="angular budget in deg")
    ap.add_argument("--plain-prompts", action="store_true",
                    help="skip chat template; encode prompts as plain text (base-model contract)")
    args = ap.parse_args()

    layer_fracs = [float(x) for x in args.layer_fracs.split(",")]
    max_angle = math.radians(args.budget)
    rng = np.random.default_rng(args.seed)

    model, tok = M.load_model(args.model, dtype="fp16")
    N = model.config.num_hidden_layers
    layers = sorted({int(round(f * (N - 1))) for f in layer_fracs})
    d = model.config.hidden_size
    vocab = model.config.vocab_size
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- LM head = SOM prototype grid, with optional bias ----
    head = getattr(model, "lm_head", None) or getattr(model, "embed_out", None)
    W_np = head.weight.detach().cpu().float().numpy()[:vocab]
    bias_np = None
    if getattr(head, "bias", None) is not None:
        bias_np = head.bias.detach().cpu().float().numpy()[:vocab]
        print(f"[bias] lm_head bias present, norm {np.linalg.norm(bias_np):.3f}")
    else:
        print("[bias] lm_head has no bias")
    W = torch.as_tensor(W_np, device=dev)
    Wn = torch.as_tensor(W_np / np.linalg.norm(W_np, axis=1, keepdims=True), device=dev)
    WT = W.T

    # ---- sample target tokens (printable single-token) ----
    all_ids = list(range(vocab))
    sample_texts = {}
    for tid in rng.choice(all_ids, size=min(2000, vocab), replace=False):
        txt = tok.decode([tid], skip_special_tokens=True)
        if txt and txt.strip() and all(32 <= ord(c) < 127 for c in txt):
            sample_texts[tid] = txt
    target_ids = sorted(sample_texts.keys())[:args.targets]
    tid_idx = torch.tensor(target_ids, device=dev)
    print(f"targets: {len(target_ids)} across vocab {vocab}")

    # ---- crowding: k-NN mean distance of each target row to FULL vocab ----
    k = 5
    K = len(target_ids)
    S = Wn[tid_idx]                                   # (K,d)
    proj = S @ Wn.T                                   # (K,V) cosines
    proj.scatter_(1, tid_idx[:, None], float("-inf"))  # exclude self
    topk = proj.topk(k, dim=1).values                 # (K,k)
    nn_dist = (1 - topk).mean(1).sqrt_()              # RMSE-ish distance to k neibs
    nn_ang = torch.acos(topk.clamp(-1, 1)).mean(1)
    print(f"[crowding] mean k-NN angle over targets: {nn_ang.mean().item():.2f} deg "
          f"(std {nn_ang.std().item():.2f})")
    # ---- contexts & states ----
    contexts = rng.choice(M.PROMPTS, size=min(args.contexts, len(M.PROMPTS)),
                          replace=False)
    if args.plain_prompts:
        states = {}
        for prompt in contexts:
            inputs = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            outputs = model(inputs, output_hidden_states=True)
            states[prompt] = {l: outputs.hidden_states[l + 1][0, -1, :].cpu().float().numpy()
                              for l in layers}
        print("[prompt-contract] PLAIN (no chat template)")
    else:
        states = M.get_states(model, tok, contexts, layers)
        print("[prompt-contract] get_states (chat template if the model defines one)")

    records = []
    for ctx in contexts:
        for l in layers:
            u = torch.as_tensor(states[ctx][l], device=dev)
            un = M._bn(u)
            uL = un @ WT
            S = Wn[tid_idx]
            cosd, sind = math.cos(max_angle), math.sin(max_angle)

            # target tangents
            TAU = M._bn(S - (S @ un)[:, None] * un)
            tauL = TAU @ WT
            v0L = cosd * uL[None, :] + sind * tauL
            own0 = v0L.gather(1, tid_idx[:, None]).squeeze(1)

            # wrong-target + random controls
            kw = (np.arange(K) + rng.integers(1, K, size=K)) % K
            tau_w = M._bn(S[kw] - (S[kw] @ un)[:, None] * un)
            vwL = cosd * uL[None, :] + sind * (tau_w @ WT)
            RR = torch.randn(K, d, device=dev)
            tau_r = M._bn(RR - (RR @ un)[:, None] * un)
            vrL = cosd * uL[None, :] + sind * (tau_r @ WT)

            # ---- G1/G2 ranks (endpoint) ----
            r_tan = M._rank_batched(v0L, own0)
            own_w = vwL.gather(1, tid_idx[:, None]).squeeze(1)
            r_wrong = M._rank_batched(vwL, own_w)
            own_r = vrL.gather(1, tid_idx[:, None]).squeeze(1)
            r_rand = M._rank_batched(vrL, own_r)

            # ---- G4 margin ----
            others = v0L.clone()
            others.scatter_(1, tid_idx[:, None], float("-inf"))
            best_other = others.max(1).values
            margin = own0 - best_other
            rel_margin = margin / own0.abs().clamp_min(1e-6)

            # ---- G3 winner-hold window (analytic crossings) ----
            entry, exit_ = _rank1_window_batched(uL, tauL, tid_idx, max_angle)
            # window only counts angle spent as rank-1 WITHIN the budget
            entry_ok = torch.isfinite(entry) & (entry <= args.budget)
            exit_c = torch.minimum(exit_, torch.full_like(exit_, args.budget))
            width_eff = torch.where(entry_ok & torch.isfinite(exit_c),
                                    (exit_c - entry).clamp_min(0.0),
                                    torch.zeros_like(entry))

            # ---- G5 no-worse rank fraction along arc (101-pt scan, paper contract) ----
            # rank only changes at sinusoid crossings; sample a fixed grid instead
            grid = torch.linspace(0.0, args.budget, 101, device=dev)
            ranks_grid = []
            for g in grid:
                vg = math.cos(math.radians(g.item())) * uL[None, :] \
                     + math.sin(math.radians(g.item())) * tauL
                og = vg.gather(1, tid_idx[:, None]).squeeze(1)
                ranks_grid.append(M._rank_batched(vg, og))
            ranks_grid = torch.stack(ranks_grid)          # (101,K)
            no_worse = (ranks_grid <= ranks_grid[0]).float().mean(0)  # frac never worse than start

            # ---- G6 Voronoi-vs-logit identity at endpoint ----
            # metric Voronoi winner: argmin ||v - W_j||^2  = argmax(W_j.v - ||W_j||^2/2)
            Wsq = (W * W).sum(1)                          # (V,)
            metric = v0L - 0.5 * Wsq[None, :]             # same argmax as pure distance
            logit_argmax = v0L.argmax(1)
            metric_argmax = metric.argmax(1)
            id_break = (logit_argmax != metric_argmax)    # (K,)

            # bias-aware version when bias present
            bias_break = torch.zeros(K, dtype=torch.bool, device=dev)
            if bias_np is not None:
                b_t = torch.as_tensor(bias_np, device=dev)
                bias_logit = v0L + b_t[None, :]
                bias_break = (bias_logit.argmax(1) != logit_argmax)

            for kk, tid in enumerate(target_ids):
                records.append(dict(
                    model=args.model, context=ctx, layer=l, target_id=tid,
                    target_text=sample_texts.get(tid, ""),
                    rank1_target=bool(r_tan[kk] == 1),
                    rank1_wrong=bool(r_wrong[kk] == 1),
                    rank1_random=bool(r_rand[kk] == 1),
                    top3=bool(r_tan[kk] <= 3), top5=bool(r_tan[kk] <= 5),
                    top10=bool(r_tan[kk] <= 10),
                    rank=float(r_tan[kk]),
                    margin=float(margin[kk]), rel_margin=float(rel_margin[kk]),
                    win_window_deg=float(width_eff[kk]),
                    entry_deg=float(entry[kk]) if torch.isfinite(entry[kk]) else None,
                    no_worse_frac=float(no_worse[kk]),
                    voronoi_break=bool(id_break[kk]),
                    bias_break=bool(bias_break[kk]),
                    nn5_angle=float(nn_ang[kk]),
                ))

    df = pd.DataFrame(records)
    safe = args.model.replace("/", "--")
    tag = f"t{args.targets}c{args.contexts}_lf{'-'.join(f'{g:g}' for g in layer_fracs)}_b{args.budget:g}"
    out = OUT / f"multi_goal_steering__{safe}__{tag}.csv"
    df.to_csv(out, index=False)
    print(f"Saved -> {out}")

    # ---- summary ----
    print(f"\n=== Multi-goal summary: {args.model} ({len(df)} cases) ===")
    for col, lab in [("rank1_target", "G1 rank-1 target"),
                     ("rank1_wrong", "G1 rank-1 wrong-target"),
                     ("rank1_random", "G1 rank-1 random"),
                     ("top3", "G2 top-3"), ("top5", "G2 top-5"), ("top10", "G2 top-10"),
                     ("voronoi_break", "G6 logit/metric break"),
                     ("bias_break", "G6 bias break")]:
        print(f"  {lab}: {df[col].mean():.3%}")
    reach = df[df.rank1_target]
    print(f"  G3 winner-window (reachable only): "
          f"mean {reach.win_window_deg.mean():.2f} deg, median {reach.win_window_deg.median():.2f}")
    print(f"  G3 window>0 rate over all: {(df.win_window_deg > 0).mean():.3%}")
    print(f"  G4 margin: mean {df.margin.mean():.3f}, rel {df.rel_margin.abs().mean():.3f}")
    print(f"  G5 no-worse frac: mean {df.no_worse_frac.mean():.3f}")
    print(f"  crowding k-NN angle: mean {df.nn5_angle.mean():.2f} deg")
    corr = df.nn5_angle.corr(df.rel_margin.abs())
    print(f"  corr(k-NN angle, |rel margin|) all targets: {corr:.3f}")
    corr_w = df.nn5_angle.corr((df.win_window_deg > 0).astype(float))
    print(f"  corr(k-NN angle, window>0): {corr_w:.3f}")
    # crowding vs reach (point-biserial)
    pb = df.groupby("target_id").agg(reach=("rank1_target", "mean"),
                                     nn5=("nn5_angle", "first")).dropna()
    if len(pb) > 5:
        r_pb = np.corrcoef(pb.reach, pb.nn5)[0, 1]
        print(f"  point-biserial corr(crowding, reach) across {len(pb)} targets: {r_pb:.3f}")


if __name__ == "__main__":
    main()