#!/usr/bin/env python3
"""Sustained multi-token steering with phrase-level targets.

A target phrase (possibly multi-token) is mapped to a steering direction by
averaging its LM-head rows. A forward hook adds alpha * tangent(direction)
at the last position at every generation step ("persist") or only the first
step ("single"). Greedy decoding.

Metrics per run: occurrences of the target phrase in the continuation,
distinct-token ratio (degeneracy guard), mean top-1 probability.

Run: python eval_sustained_steering.py --model Qwen/Qwen2-0.5B-Instruct \
        --targets "apple" "dragon" --alphas 1 2 --new-tokens 48
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import steering_geometry_test as M

OUT = Path("../steering_geometry_results")


@torch.no_grad()
def make_tangent_hook(head_rows, alpha):
    """head_rows: (k, d) tensor of LM-head rows for the target phrase."""
    w = head_rows.mean(0)
    w = w / w.norm()

    def hook(module, inp, out):
        out2 = out.clone()
        h = out2[0, -1, :].float()
        hn = h / h.norm()
        g = w - (w @ hn) * hn
        g = g / max(g.norm().item(), 1e-8)
        out2[0, -1, :] = (h + alpha * h.norm() * g).to(out.dtype)
        return out2
    return hook


@torch.no_grad()
def generate(model, tok, prompt_ids, new_tokens=48):
    ids = prompt_ids.clone()
    conf = []
    for _ in range(new_tokens):
        logits = model(ids).logits[0, -1].float()
        p = torch.softmax(logits, dim=0)
        conf.append(float(p.max()))
        ids = torch.cat([ids, torch.tensor([[int(logits.argmax())]], device=ids.device)],
                        dim=1)
    return tok.decode(ids[0].tolist()), float(np.mean(conf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--targets", nargs="+", default=["apple", "dragon"])
    ap.add_argument("--prompts", nargs="+",
                    default=["Once upon a time", "Tell me something interesting:"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--depth-fracs", type=float, nargs="+", default=[0.99])
    ap.add_argument("--new-tokens", type=int, default=48)
    args = ap.parse_args()

    model, tok = M.load_model(args.model, dtype="fp16")
    W = model.lm_head.weight.detach().cpu().float().numpy()
    if W.size == 0:
        W = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    L = model.config.num_hidden_layers

    rows = []
    for prompt in args.prompts:
        pids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        base_text, base_conf = generate(model, tok, pids, args.new_tokens)
        print(f"\n### {prompt}\nBASE: {base_text[:150]}")
        for tgt in args.targets:
            tids_t = tok(" " + tgt.strip(), add_special_tokens=False).input_ids
            trow = W[tids_t]
            first_tok_str = tok.decode([tids_t[0]])
            occ_b = base_text.lower().count(" " + tgt.strip().lower())
            print(f"[target '{tgt}' -> {len(tids_t)} ids, first={first_tok_str!r}]")
            rows.append(dict(prompt=prompt, target=tgt, mode="base", alpha=0.0,
                             depth=0.0, occurrences=occ_b, conf=base_conf))
            for frac in args.depth_fracs:
                li = int(round(frac * (L - 1)))
                for a in args.alphas:
                    results = {}
                    for mode in ["single", "persist"]:
                        h = make_tangent_hook(
                            torch.as_tensor(trow, device=model.device), a)
                        hd = model.model.layers[li].register_forward_hook(h)
                        try:
                            if mode == "single":
                                # steer only the first step: hook once, then plain
                                ids = pids.clone()
                                logits = model(ids).logits[0, -1].float()
                                hd.remove()
                                ids = torch.cat([ids, torch.tensor(
                                    [[int(logits.argmax())]], device=ids.device)], dim=1)
                                text, conf = generate(model, tok, ids, args.new_tokens - 1)
                            else:
                                text, conf = generate(model, tok, pids, args.new_tokens)
                        finally:
                            hd.remove()
                        results[mode] = (text, conf)
                    for mode, (text, conf) in results.items():
                        occ = sum(text.lower().count(v) for v in
                                  {tgt.lower(), tgt.lower() + "s"})
                        toks = tok(text, add_special_tokens=False).input_ids
                        div = len(set(toks)) / max(len(toks), 1)
                        print(f"{mode.upper():8s} a={a} d={frac:.2f} '{tgt}' x{occ} "
                              f"conf={conf:.2f} div={div:.2f} | {text[:110]}")
                        rows.append(dict(prompt=prompt, target=tgt, mode=mode,
                                         alpha=a, depth=frac, occurrences=occ,
                                         conf=conf, diversity=div))

    safe = args.model.replace("/", "--")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"sustained_steering__{safe}.csv", index=False)
    print("\n=== mean occurrences / confidence by mode ===")
    print(df.groupby("mode")[["occurrences", "conf"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
