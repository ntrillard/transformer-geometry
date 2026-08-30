#!/usr/bin/env python3
"""Confidence-gated steering: re-decide uncertain tokens under a slight steer.

Per generated token:
  1. forward pass -> confidence = max softmax prob
  2. if conf >= threshold : commit normally (model runs free)
  3. if conf <  threshold : re-run the step with a tangent steer toward the
     target phrase (paper formula, renormalized), commit steered choice

Compare against baseline (never steer) and persistent (always steer).

Run: python eval_gated_steering.py --model Qwen/Qwen2-0.5B-Instruct \
        --target apple --thresholds 0.3 0.5 0.7 --alphas 0.2 0.3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import steering_geometry_test as M

OUT = Path("../steering_geometry_results")


def base_module(model):
    return getattr(model, "model", None) or getattr(model, "transformer", None)


@torch.no_grad()
def tangent_dir(w_row, h):
    w = w_row / w_row.norm()
    hn = h / h.norm()
    g = w - (w @ hn) * hn
    return g / max(g.norm().item(), 1e-8)


@torch.no_grad()
def gen_gated(model, tok, pids, w_row, new_tokens=48, threshold=None,
              alpha=0.3, temperature=None, top_p=None):
    """threshold=None -> never steer.  alpha applied only on gated steps."""
    ids = pids.clone()
    confs, gates = [], []
    for _ in range(new_tokens):
        h = base_module(model)(ids).last_hidden_state[0, -1].float()
        logits = h @ W_DEV.T
        p = torch.softmax(logits, dim=0)
        conf = float(p.max())
        confs.append(conf)
        steer_now = threshold is not None and conf < threshold
        gates.append(int(steer_now))
        if steer_now:
            g = tangent_dir(w_row, h)
            h2 = h + alpha * h.norm() * g
            h2 = h2 / h2.norm() * h.norm()
            logits = h2 @ W_DEV.T
            p = torch.softmax(logits, dim=0)
        if top_p:
            q = p.clone()
            order = q.argsort(descending=True)
            cum = q[order].cumsum(0)
            keep = order[:int((cum <= top_p).sum()) + 1]
            m = torch.zeros_like(q)
            m[keep] = 1
            q = (q * m) / (q * m).sum()
            nxt = int(torch.multinomial(q, 1))
        elif temperature:
            nxt = int(torch.multinomial(torch.softmax(logits / temperature, 0), 1))
        else:
            nxt = int(logits.argmax())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    text = tok.decode(ids[0].tolist())
    return text, float(np.mean(confs)), int(np.sum(gates))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--target", default="apple")
    ap.add_argument("--prompts", nargs="+",
                    default=["Once upon a time", "Tell me something interesting:"])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.3])
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--gate-percentile", type=float, default=None,
                    help="calibrate threshold as this percentile of the model's "
                         "own step-confidence distribution (replaces fixed ones)")
    args = ap.parse_args()

    model, tok = M.load_model(args.model, dtype="fp16")
    W = model.lm_head.weight.detach().cpu().float().numpy()
    if W.size == 0:
        W = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    DEV = model.device
    W_DEV = torch.as_tensor(W, device=DEV, dtype=torch.float32)

    tids_t = tok(" " + args.target.strip(), add_special_tokens=False).input_ids
    w_row = torch.as_tensor(W[tids_t[0]], device=DEV)

    # calibrate the model's own confidence distribution if requested
    gate_thr = None
    if args.gate_percentile is not None:
        allc = []
        for prompt in args.prompts:
            if args.chat:
                enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                              add_generation_prompt=True,
                                              return_tensors="pt")
                ids = (enc if torch.is_tensor(enc) else enc.input_ids).to(DEV)
            else:
                ids = tok(prompt, add_special_tokens=False,
                          return_tensors="pt").input_ids.to(DEV)
            for _ in range(4):
                cur = ids.clone()
                for _st in range(args.new_tokens):
                    h = base_module(model)(cur).last_hidden_state[0, -1].float()
                    p = torch.softmax(h @ W_DEV.T, dim=0)
                    allc.append(float(p.max()))
                    cur = torch.cat([cur, torch.tensor([[int(p.argmax())]],
                                  device=DEV)], dim=1)
        gate_thr = float(np.percentile(allc, args.gate_percentile))
        print(f"calibrated gate threshold: {gate_thr:.3f} "
              f"(p{args.gate_percentile} of {len(allc)} steps)")

    rows = []
    for prompt in args.prompts:
        if args.chat:
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True,
                                          return_tensors="pt")
            pids = (enc if torch.is_tensor(enc) else enc.input_ids).to(DEV)
        else:
            pids = tok(prompt, add_special_tokens=False,
                       return_tensors="pt").input_ids.to(DEV)
        print(f"\n### {prompt}")
        thr_list = [gate_thr] if gate_thr is not None else args.thresholds
        variants = [("base", None, None)] + [
            (f"gate{t:.2f}-a{a}", t, a) for t in thr_list for a in args.alphas]
        for name, thr, a in variants:
            text, conf, ngates = gen_gated(
                model, tok, pids, w_row, args.new_tokens,
                threshold=thr, alpha=a if a else None, top_p=0.9)
            w = args.target.lower()
            occ = sum(text.lower().count(v) for v in
                      {" " + w, " " + w + "s", " " + w.capitalize(),
                       " " + w.capitalize() + "s"})
            tk = tok(text, add_special_tokens=False).input_ids
            div = len(set(tk)) / max(len(tk), 1)
            print(f"{name:10s} occ={occ:3d} gates={ngates:3d} conf={conf:.2f} "
                  f"div={div:.2f} | {text[:100]}")
            rows.append(dict(prompt=prompt, mode=name, threshold=thr or 0,
                             alpha=a or 0, occurrences=occ, gates=ngates,
                             conf=conf, diversity=div, sample=text))

    safe = args.model.replace("/", "--")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"gated_steering__{safe}.csv", index=False)
    print(f"\nSaved -> gated_steering__{safe}.csv")
