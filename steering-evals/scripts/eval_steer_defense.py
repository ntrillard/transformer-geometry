#!/usr/bin/env python3
"""Steering-based pit defense: single-layer vs multi-layer projection.

Registers forward hooks projecting the hidden state away from the pit LM-head
direction at one or many layers, applied during generation. Measures:
  - greedy pit-loop length from a [pit]*5 trigger
  - interference on ordinary prompts (fraction of tokens changed vs no hook)

Run: python eval_steer_defense.py --model Qwen/Qwen2.5-7B-Instruct --quant nf4 \
        --pit-id 15 [--layers-last K] [--alpha 0.3]
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
def gen_with_hooks(model, head, prompt_ids, max_new=30):
    ids = list(prompt_ids)
    for _ in range(max_new):
        h = E.state_after_tokens(model, ids)
        ids.append(int((h.to(head.dtype) @ head.T).float().argmax()))
    return ids


@torch.no_grad()
def make_hook(head, pit, alpha):
    d = head[pit].float()
    d = d / d.norm()

    def hook(module, inp, out):
        out2 = out.clone()
        h = out2[0, -1, :].float()
        h2 = h - alpha * (h @ d) * d
        out2[0, -1, :] = h2.to(out.dtype)
        return out2
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--pit-id", type=int, default=15)
    ap.add_argument("--quant", default=None, choices=["int8", "nf4"])
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--layers-last", type=int, nargs="+", default=[1, 4],
                    help="test hooks on the last-K layers for each K in this list")
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--normal-prompts", type=int, default=5)
    args = ap.parse_args()

    print(f"Loading {args.model} (quant={args.quant}) ...")
    model, tok = M.load_model(args.model, dtype="fp16", quantize=args.quant)
    W = model.lm_head.weight.detach().cpu().float().numpy()
    if W.size == 0:
        W = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    torch.cuda.empty_cache()
    try:
        head = torch.as_tensor(W, device=model.device, dtype=torch.float16)
    except RuntimeError:
        head = torch.as_tensor(W, dtype=torch.float16)  # CPU fallback

    def logits_of(h):
        return (h.to(head.dtype) @ head.T).float()
    L = len(model.model.layers)

    layer_sets = {"single": [L - 1]}
    for k in args.layers_last:
        if k > 1:
            layer_sets[f"last-{k}"] = list(range(L - k, L))
    if args.all_layers:
        layer_sets["all"] = list(range(L))

    trig = [args.pit_id] * 5
    base_loop = E.count_run(gen_with_hooks(model, head, trig), args.pit_id)
    norms = [tok(p, add_special_tokens=False).input_ids[:8]
             for p in M.PROMPTS[:args.normal_prompts]]
    baselines = [gen_with_hooks(model, head, n) for n in norms]

    rows = []
    for name, layers in layer_sets.items():
        hooks = []
        handles = [model.model.layers[li].register_forward_hook(
            make_hook(head, args.pit_id, args.alpha)) for li in layers]
        try:
            loop = E.count_run(gen_with_hooks(model, head, trig), args.pit_id)
            changed = []
            for n, b in zip(norms, baselines):
                g = gen_with_hooks(model, head, n)
                m = min(len(g), len(b)) - 1
                diff = sum(int(g[i] != b[i]) for i in range(8, m)) / max(m - 8, 1)
                changed.append(diff)
        finally:
            for h_ in handles:
                h_.remove()
        rows.append(dict(config=name, n_layers=len(layers), alpha=args.alpha,
                         loop_length=loop, broken=int(loop < base_loop),
                         mean_token_change=float(np.mean(changed)),
                         max_token_change=float(np.max(changed))))
        print(f"{name:12s} ({len(layers):2d} layers, a={args.alpha}): "
              f"loop {base_loop}->{loop} | token-change {np.mean(changed):.1%}")

    safe = args.model.replace("/", "--")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"steer_defense__{safe}.csv", index=False)
    print(f"\nBaseline loop={base_loop}; saved -> steer_defense__{safe}.csv")


if __name__ == "__main__":
    main()
