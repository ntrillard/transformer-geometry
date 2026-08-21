#!/usr/bin/env python3
"""Fast cross-model steering geometry test.

Tests the target-tangent / decision-cone / competitor-geometry hypotheses
across model families.  For each (state, target) pair we measure:
  1. target-tangent endpoint rank
  2. wrong-target-tangent endpoint rank
  3. random-tangent endpoint rank (same angular budget)
  4. off-arc rotations holding target score fixed:
       random, toward strongest competitor, away from strongest competitor
  5. first rank-1 crossing angle along target tangent
  6. (optional, expensive) shortest verified decision-cell angle

Run:
    python steering_geometry_test.py --model Qwen/Qwen2-1.5B-Instruct
"""

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = Path("models")
OUT = Path("steering_geometry_results")
OUT.mkdir(parents=True, exist_ok=True)

DISK_FREE_FLOOR = 8e9      # refuse downloads when < 8 GB free
DISK_SAFETY_MARGIN = 6e9   # keep at least 6 GB free after any download


def _disk_guard(model_id):
    """Refuse to download a model if the disk can't hold it with a safety margin.
    Skips the size estimate when the model is already fully cached."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    free = shutil.disk_usage(cache_root).free
    cached = list(cache_root.glob(f"models--{model_id.replace('/', '--')}/snapshots/*"))
    if cached:
        return 0.0
    need = 0.0
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id, files_metadata=True)
        if info.siblings:
            need = sum((s.size or 0) for s in info.siblings)
    except Exception:
        pass  # gated/offline: fall back to the free-space floor only
    if free < DISK_FREE_FLOOR:
        raise SystemExit(f"[disk-guard] only {free/1e9:.1f}G free; refusing to download "
                         f"{model_id}. Free disk space first.")
    if need > free - DISK_SAFETY_MARGIN:
        raise SystemExit(f"[disk-guard] {model_id} needs ~{need/1e9:.1f}G but only "
                         f"{free/1e9:.1f}G free (safety margin {DISK_SAFETY_MARGIN/1e9:.1f}G). Aborting.")
    if need > 2e9:
        print(f"[disk-guard] {model_id}: ~{need/1e9:.1f}G to download, {free/1e9:.1f}G free.")
    return need

PROMPTS = [
    "The capital of France is",
    "In the year 3000, humans will",
    "To bake sourdough bread",
    "The theory of relativity",
    "Once upon a time",
    "The quantum computer",
    "The quick brown fox",
    "For dinner I made",
]

# Defaults (tunable via CLI --targets/--contexts/--layers/--steps).
N_TARGETS = 128
N_CONTEXTS = 4
LAYERS = [0, 8, 16, 23]
ANGULAR_BUDGET = math.radians(17)  # ~atan(0.3)


def load_model(model_id, dtype="fp16", quantize=None):
    print(f"\nLoading {model_id} on {DEVICE} (dtype={dtype}, quant={quantize}) ...")
    _disk_guard(model_id)   # refuse to fill the disk with a download it can't hold
    DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=False)
    kwargs = dict(trust_remote_code=True, local_files_only=False)
    if quantize in ("int8", "nf4"):
        from transformers import BitsAndBytesConfig
        if quantize == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DT)
        kwargs["torch_dtype"] = DT if DEVICE == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=DT if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else "cpu", **kwargs)
    model.eval()
    return model, tok


@torch.no_grad()
def get_states(model, tok, prompts, layers):
    """Returns dict[prompt] -> dict[layer] -> hidden state (last token)."""
    states = {}
    for prompt in prompts:
        try:                                  # chat models
            inputs = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                             return_tensors="pt", add_generation_prompt=True)
        except Exception:                     # base models (gpt2, pythia, ...)
            inputs = tok(prompt, return_tensors="pt")
        if hasattr(inputs, "input_ids"):      # BatchEncoding (transformers 5.x) -> tensor
            inputs = inputs.input_ids
        inputs = inputs.to(model.device)
        outputs = model(inputs, output_hidden_states=True)
        h = {l: outputs.hidden_states[l + 1][0, -1, :].cpu().float().numpy() for l in layers}
        states[prompt] = h
    return states


def normalize(x):
    return x / np.linalg.norm(x)


def tangent_direction(u, s):
    """Projected target tangent on unit sphere: s - (s·u)u."""
    g = s - (s @ u) * u
    return normalize(g)


def rotate_toward(u, tau, delta):
    """Rotate u by angle delta in the plane spanned by u and tau (both unit, tau⊥u)."""
    return math.cos(delta) * u + math.sin(delta) * tau


def logits(h, W):
    return h @ W.T


def rank_of(logits_vec, target_id):
    return int((logits_vec > logits_vec[target_id]).sum() + 1)


def first_rank1_angle(u, s, W, target_id, max_delta=math.radians(45),
                      n_steps=200, use_scan=False):
    """First angle along target tangent where target becomes rank 1.

    Fast path (default): ANALYTIC. v(d)=cos(d)u+sin(d)tau, so
        logit_j(d) = cos(d)A[j] + sin(d)B[j],  A=u@W.T, B=tau@W.T.
    Target beats competitor j when f_j(d) = (A[t]-A[j])cosd + (B[t]-B[j])sind > 0.
    Each f_j is a sinusoid -> positive on one contiguous arc; intersecting all j
    arcs with [0,max_delta] gives the exact first rank-1 angle in O(vocab), no scan.
    (Equivalence verified against the 200-step scan on 400 random configs.)

    Fallback (--use-scan): the original 200-step loop, behaviorally identical to
    the pre-optimization version.
    """
    tau = tangent_direction(u, s)
    if use_scan:
        for i in range(1, n_steps + 1):
            delta = max_delta * i / n_steps
            v = rotate_toward(u, tau, delta)
            if rank_of(logits(v, W), target_id) == 1:
                return math.degrees(delta)
        return None

    A = u @ W.T
    B = tau @ W.T
    P = A[target_id] - A
    Q = B[target_id] - B
    R = np.hypot(P, Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_ = np.divide(P, R, out=np.zeros_like(P), where=R > 0)
        sin_ = np.divide(Q, R, out=np.zeros_like(P), where=R > 0)
    th = np.arctan2(sin_, cos_)          # positive arc = (th - pi/2, th + pi/2) mod 2pi
    lo = th - math.pi / 2
    hi = th + math.pi / 2
    twopi = 2 * math.pi
    lo_j = np.full(len(P), np.inf)
    hi_j = np.full(len(P), np.inf)
    # For competitors already ahead at d=0 (P<=0) keep the first positive interval.
    for k in range(0, 2):
        L = np.clip(lo + twopi * k, 0, None)
        H = np.clip(hi + twopi * k, None, float(max_delta))
        ok = L < H
        better = (L < lo_j) | np.isinf(lo_j)
        lo_j = np.where(better & ok, L, lo_j)
        hi_j = np.where(better & ok, np.where(ok, H, np.inf), hi_j)
    # For competitors already behind at d=0 (P>0), target is ahead from d=0 up to exit.
    pos0 = P > 1e-12
    if pos0.any():
        for k in range(-1, 1):
            L = np.clip(lo + twopi * k, 0, None)
            H = np.clip(hi + twopi * k, None, float(max_delta))
            touched0 = L <= 1e-9
            sel = pos0 & touched0 & (L < H)
            lo_j = np.where(sel, L, lo_j)
            hi_j = np.where(sel, H, hi_j)
    # Intersection over all competitors: [max lo_j, min hi_j].
    lo_all, hi_all = lo_j.max(), hi_j.min()
    if np.isfinite(lo_all) and lo_all <= hi_all and lo_all <= float(max_delta):
        return math.degrees(lo_all)
    return None


def _random_in_nullspace(v0, s, seed):
    """One random unit vector orthogonal to both v0 and s (span{v0,s}^perp), O(d).
    Previously built a full d x (d-2) Gram-Schmidt/QR basis (O(d^3)) per call."""
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(len(v0))
    for _ in range(3):                     # two/three Householder projections
        b = b - (b @ v0) * v0
        b = b - (b @ s) * s
    nb = np.linalg.norm(b)
    return b / nb if nb > 1e-8 else None


def competitor_controls(u, s, W, target_id, delta):
    """Return target rank after off-arc rotations that keep target score fixed."""
    tau = tangent_direction(u, s)
    # endpoint on target tangent
    v0 = rotate_toward(u, tau, delta)
    target_score = v0 @ s

    # find strongest competitor at v0
    l = logits(v0, W)
    comp_id = int(np.argmax(np.delete(l, target_id)))
    if comp_id >= target_id:
        comp_id += 1
    comp_dir = normalize(W[comp_id])

    # residual space: directions orthogonal to both v0 and s (keep norm + target score)
    b_rand = _random_in_nullspace(v0, s, seed=0)
    if b_rand is None:
        raise RuntimeError("could not find a nullspace direction")

    # toward competitor: project (comp_dir - v0) onto tangent plane at v0, then remove s
    toward = comp_dir - (comp_dir @ v0) * v0
    toward = toward - (toward @ s) * s
    if np.linalg.norm(toward) > 1e-8:
        toward = normalize(toward)
    else:
        toward = b_rand

    # away from competitor
    away = -toward

    eps = math.radians(8)  # off-arc angle
    results = {}
    for name, b in [("random", b_rand), ("toward_blocker", toward), ("away_blocker", away)]:
        v = math.cos(eps) * v0 + math.sin(eps) * b
        v = normalize(v)
        results[name] = {
            "rank": rank_of(logits(v, W), target_id),
            "target_score": float(v @ s),
            "comp_score": float(v @ comp_dir),
        }
    return results


def _bn(x):
    """GPU batch-normalize rows: x/(||x||) with guard; zero rows returned zero."""
    n = x.norm(dim=-1, keepdim=True)
    return x / n.clamp_min(1e-9)


def _rank_batched(logits_rows, own):
    """logits_rows (K,V), own (K) -> ranks (K): count entries strictly > own +1."""
    return (logits_rows > own[:, None]).sum(1) + 1


@torch.no_grad()
def _rank1_analytic_batched(uL, tauL, tid_idx, max_angle):
    """Batched analytic first rank-1 angle. uL (V) = u@W.T; tauL (K,V) = tau@W.T.
    Returns (K,) angles in degrees (NaN where target never ranks 1 within budget)."""
    P = uL[tid_idx][:, None] - uL[None, :]           # (K,V): (u-target) - (u-other)
    Q = tauL.gather(1, tid_idx[:, None]) - tauL      # (K,V)
    R = torch.hypot(P, Q)
    cos_ = torch.where(R > 0, P / R.clamp_min(1e-12), torch.zeros_like(P))
    sin_ = torch.where(R > 0, Q / R.clamp_min(1e-12), torch.zeros_like(P))
    th = torch.atan2(sin_, cos_)
    lo, hi = th - math.pi / 2, th + math.pi / 2
    twopi = 2 * math.pi
    out = torch.full((len(P),), float("nan"), device=P.device)
    pos0 = P > 1e-12
    eps = 1e-9
    for k in range(len(P)):
        lo_j = torch.full((len(P[k]),), float("inf"), device=P.device)
        hi_j = torch.full((len(P[k]),), float("inf"), device=P.device)
        # competitors already behind at d=0: positive from 0 up to exit
        for j in range(-1, 1):
            L = (lo[k] + twopi * j).clamp(min=0.0)
            H = (hi[k] + twopi * j).clamp(max=float(max_angle))
            ok = L < H
            tou = L <= eps
            sel = pos0[k] & tou & ok
            lo_j = torch.where(sel, L, lo_j)
            hi_j = torch.where(sel, H, hi_j)
        # competitors ahead at d=0: keep first positive interval in range
        for j in range(0, 2):
            L = (lo[k] + twopi * j).clamp(min=0.0)
            H = (hi[k] + twopi * j).clamp(max=float(max_angle))
            ok = L < H
            better = (L < lo_j) | torch.isinf(lo_j)
            lo_j = torch.where(better & ok, L, lo_j)
            hi_j = torch.where(better & ok, H, hi_j)
        lo_all = lo_j.max()
        if torch.isfinite(lo_all) and lo_all <= hi_j.min() and lo_all <= float(max_angle):
            out[k] = math.degrees(lo_all.item())
    return out


@torch.no_grad()
def _batched_block(u, W_t, Wn_t, tid_idx, max_angle, seed):
    """Compute all per-target metrics for one (context,layer) state u, vectorized
    over K=n_targets on GPU. Returns dict of (K,) arrays + comp metadata."""
    rng = np.random.default_rng(seed)
    d = u.shape[0]
    WT = W_t.T                                   # (d,V) for matmuls; W_t stays (V,d) for row lookups
    un = _bn(u)
    uL = un @ WT                                 # (V)
    S = Wn_t[tid_idx]                            # (K,d) rows for targets
    cosd, sind = math.cos(max_angle), math.sin(max_angle)

    # 1/2. target + wrong-target tangents
    TAU = _bn(S - (S @ un)[:, None] * un)        # (K,d)
    tauL = TAU @ WT                              # (K,V)
    v0L = cosd * uL[None, :] + sind * tauL
    own0 = v0L.gather(1, tid_idx[:, None]).squeeze(1)
    r_tan = _rank_batched(v0L, own0)

    kw = rng.integers(0, len(tid_idx), size=len(tid_idx))
    tau_w = _bn(S[kw] - (S[kw] @ un)[:, None] * un)
    tauwL = tau_w @ WT
    vwL = cosd * uL[None, :] + sind * tauwL
    ownw = vwL.gather(1, tid_idx[:, None]).squeeze(1)
    r_wrong = _rank_batched(vwL, ownw)

    # 3. random tangent
    R = torch.randn(len(tid_idx), d, device=u.device)
    rr = R - (R @ un)[:, None] * un
    rn = _bn(rr)
    rL = rn @ WT
    vrL = cosd * uL[None, :] + sind * rL
    ownr = vrL.gather(1, tid_idx[:, None]).squeeze(1)
    r_rand = _rank_batched(vrL, ownr)

    # 4. competitor controls (batched)
    v0 = cosd * un[None, :] + sind * TAU            # (K,d)
    compL = v0L.clone()
    compL.scatter_(1, tid_idx[:, None], float("-inf"))
    comp_i = compL.argmax(1)                        # (K)
    comp_dir = W_t[comp_i]                          # (K,d)
    toward = comp_dir - (comp_dir * v0).sum(1, keepdim=True) * v0
    toward = toward - (toward * S).sum(1, keepdim=True) * S
    B0 = torch.randn(len(tid_idx), d, device=u.device)
    B0 = B0 - (B0 * v0).sum(1, keepdim=True) * v0 - (B0 * S).sum(1, keepdim=True) * S
    b_rand = _bn(B0)
    tn = _bn(toward)
    ok_t = toward.norm(dim=-1) > 1e-8
    toward = torch.where(ok_t[:, None], tn, b_rand)

    eps = math.radians(8)
    res = {}
    for name, b in [("random", b_rand), ("toward_blocker", toward), ("away_blocker", -toward)]:
        v = math.cos(eps) * v0 + math.sin(eps) * b
        vL = v @ WT
        own = vL.gather(1, tid_idx[:, None]).squeeze(1)
        res[name] = _rank_batched(vL, own)

    # 5. first rank-1 angle (analytic, batched)
    rank1 = _rank1_analytic_batched(uL, tauL, tid_idx, max_angle)

    return dict(r_tan=r_tan, r_wrong=r_wrong, r_rand=r_rand,
                r_off_random=res["random"], r_off_toward=res["toward_blocker"],
                r_off_away=res["away_blocker"], rank1=rank1)


def run_model(model_id, seed=42, n_targets=N_TARGETS, n_contexts=N_CONTEXTS,
              layers=None, layer_fracs=None, use_scan=False, n_steps=200,
              max_angle=None, dtype="fp16", quantize=None):
    max_angle = max_angle if max_angle is not None else ANGULAR_BUDGET
    rng = np.random.default_rng(seed)
    model, tok = load_model(model_id, dtype=dtype, quantize=quantize)
    if layer_fracs is not None:
        N = model.config.num_hidden_layers
        layers = sorted({int(round(f * (N - 1))) for f in layer_fracs})
        print(f"  depth-adaptive layers for {N}-layer stack ({len(layers)} points): {layers}")
    layers = layers or LAYERS
    d = model.config.hidden_size
    vocab = model.config.vocab_size

    # Use tokenizer's vocab ids directly, but restrict to printable tokens
    all_ids = list(range(vocab))
    # Filter to single-token strings that are printable and not special
    sample_texts = {}
    for tid in rng.choice(all_ids, size=min(2000, vocab), replace=False):
        txt = tok.decode([tid], skip_special_tokens=True)
        if txt and txt.strip() and all(32 <= ord(c) < 127 for c in txt):
            sample_texts[tid] = txt
    target_ids = sorted(sample_texts.keys())[:n_targets]
    if len(target_ids) < n_targets:
        target_ids = list(rng.choice(all_ids, size=n_targets, replace=False))

    # Get W (LM head)
    W = model.lm_head.weight.detach().cpu().float().numpy()
    W = W[:vocab]
    W_n = W / np.linalg.norm(W, axis=1, keepdims=True)

    # Get contexts
    contexts = rng.choice(PROMPTS, size=min(n_contexts, len(PROMPTS)), replace=False)
    states = get_states(model, tok, contexts, layers)

    records = []
    if use_scan:
        # ---- exact original per-pair path (slow; only with --use-scan) ----
        for ctx in contexts:
            for l in layers:
                h = states[ctx][l]
                u = normalize(h)
                for tid in target_ids:
                    s = W_n[tid]
                    tau = tangent_direction(u, s)
                    v_tan = rotate_toward(u, tau, max_angle)
                    r_tan = rank_of(logits(v_tan, W), tid)
                    wrong_tid = rng.choice(target_ids)
                    s_wrong = W_n[wrong_tid]
                    tau_wrong = tangent_direction(u, s_wrong)
                    v_wrong = rotate_toward(u, tau_wrong, max_angle)
                    r_wrong = rank_of(logits(v_wrong, W), tid)
                    rand_dir = normalize(rng.standard_normal(d))
                    rand_dir = rand_dir - (rand_dir @ u) * u
                    if np.linalg.norm(rand_dir) > 1e-8:
                        rand_dir = normalize(rand_dir)
                    else:
                        rand_dir = tau
                    v_rand = rotate_toward(u, rand_dir, max_angle)
                    r_rand = rank_of(logits(v_rand, W), tid)
                    controls = competitor_controls(u, s, W, tid, max_angle)
                    rank1_angle = first_rank1_angle(u, s, W, tid, max_delta=max_angle,
                                                    n_steps=n_steps, use_scan=use_scan)
                    records.append({
                        "model": model_id, "context": ctx, "layer": l,
                        "target_id": tid, "target_text": sample_texts.get(tid, ""),
                        "rank_target_tangent": r_tan, "rank_wrong_tangent": r_wrong,
                        "rank_random_tangent": r_rand, "rank_offarc_random": controls["random"]["rank"],
                        "rank_offarc_toward": controls["toward_blocker"]["rank"],
                        "rank_offarc_away": controls["away_blocker"]["rank"],
                        "first_rank1_angle": rank1_angle,
                    })
    else:
        # ---- batched GPU path: all targets at once per (context,layer) ----
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        W_t = torch.as_tensor(W, device=dev)          # (V,d) fp32
        Wn_t = torch.as_tensor(W_n, device=dev)       # (V,d) normalized
        tid_idx = torch.tensor(target_ids, device=dev)
        for ctx in contexts:
            for l in layers:
                h = states[ctx][l]
                u = torch.as_tensor(h, device=dev)
                m = _batched_block(u, W_t, Wn_t, tid_idx, max_angle, seed=seed)
                r_tan, r_wr = m["r_tan"], m["r_wrong"]
                r_rd, r1 = m["r_rand"], m["rank1"]
                for k, tid in enumerate(target_ids):
                    records.append({
                        "model": model_id, "context": ctx, "layer": l,
                        "target_id": tid, "target_text": sample_texts.get(tid, ""),
                        "rank_target_tangent": int(r_tan[k]),
                        "rank_wrong_tangent": int(r_wr[k]),
                        "rank_random_tangent": int(r_rd[k]),
                        "rank_offarc_random": int(m["r_off_random"][k]),
                        "rank_offarc_toward": int(m["r_off_toward"][k]),
                        "rank_offarc_away": int(m["r_off_away"][k]),
                        "first_rank1_angle": None if math.isnan(r1[k]) else float(r1[k]),
                    })

    df = pd.DataFrame(records)
    fracs_tag = ("lf" + "-".join(f"{f:g}" for f in layer_fracs)) if layer_fracs else \
                ("l" + "-".join(map(str, layers)))
    cfg_tag = f"t{n_targets}c{n_contexts}_{fracs_tag}_{dtype}"
    if quantize:
        cfg_tag += f"_{quantize}"
    safe_name = model_id.replace("/", "--") + "__" + cfg_tag
    df.to_csv(OUT / f"{safe_name}.csv", index=False)
    print(f"\nSaved -> {OUT / (safe_name + '.csv')}")

    # Summary
    print(f"\n=== Summary for {model_id} ===")
    print(f"Cases: {len(df)}")
    print(f"target tangent rank-1 rate: {(df.rank_target_tangent == 1).mean():.3%}")
    print(f"wrong tangent rank-1 rate:  {(df.rank_wrong_tangent == 1).mean():.3%}")
    print(f"random tangent rank-1 rate: {(df.rank_random_tangent == 1).mean():.3%}")
    print(f"off-arc random rank-1 rate: {(df.rank_offarc_random == 1).mean():.3%}")
    print(f"off-arc toward blocker:     {(df.rank_offarc_toward == 1).mean():.3%}")
    print(f"off-arc away blocker:       {(df.rank_offarc_away == 1).mean():.3%}")
    print(f"median first rank-1 angle:  {df.first_rank1_angle.median(skipna=True):.2f}°")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", type=int, default=N_TARGETS,
                        help="number of target tokens per (context,layer)")
    parser.add_argument("--contexts", type=int, default=N_CONTEXTS)
    parser.add_argument("--layers", default=",".join(map(str, LAYERS)),
                        help="comma-separated layer indices")
    parser.add_argument("--use-scan", action="store_true",
                        help="use the original 200-step scan instead of the analytic "
                             "first rank-1 angle")
    parser.add_argument("--steps", type=int, default=200,
                        help="scan steps (only used with --use-scan)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quant", default=None, choices=["none", "int8", "nf4"],
                        help="quantize weights (int8=bitsandbytes 8-bit, nf4=4-bit)")
    parser.add_argument("--layer-fracs", default=None,
                        help="comma-separated relative depths (0..1) for fair cross-model "
                             "comparison, e.g. '0.0,0.33,0.67,0.99'; overrides --layers")
    args = parser.parse_args()
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    layer_fracs = [float(x) for x in args.layer_fracs.split(",")] if args.layer_fracs else None
    run_model(args.model, args.seed, n_targets=args.targets, n_contexts=args.contexts,
              layers=layers, layer_fracs=layer_fracs, use_scan=args.use_scan,
              n_steps=args.steps, dtype=args.dtype,
              quantize=None if args.quant in (None, "none") else args.quant)


if __name__ == "__main__":
    main()
