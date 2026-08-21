#!/usr/bin/env python3
"""Evidence for reply point 2.1: cow-tipping / repetition-loop ROBUSTNESS matrix.

Ships the measurement behind the numbers in the post:
  - discovers a token whose terminal-repetition collapses into a loop under greedy;
  - measures the loop under fp16 vs bf16 vs int8 (bitsandbytes);
  - and under sampling (temp 0.8 / temp 1.0 + top-p 0.9) and a chat template.

Run:  python eval_pit_robustness.py --model Qwen/Qwen2-0.5B-Instruct --seeds 64
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import eval_defense as E
import steering_geometry_test as M

OUT = Path("steering_geometry_results")
MODEL = "google/gemma-3-1b-it"


@torch.no_grad()
def find_looper(m, tok, head, min_loop=8):
    cands = []
    for t in range(5000):
        if len(cands) >= 400:
            break
        s = tok.decode([t])
        if s.strip() and all(32 <= ord(c) < 128 for c in s) and len(s) <= 8:
            cands.append(t)
    for t in cands:
        n = E.count_run(E.gen_with_detector(m, tok, head, [t] * 5, t, max_new=25), t)
        if n >= min_loop:
            return n, t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seeds", type=int, default=64,
                    help="number of stochastic decoder trials")
    args = ap.parse_args()

    m, tok = M.load_model(args.model, dtype="fp16")
    headT = torch.as_tensor(m.lm_head.weight.detach().float().cpu().numpy(),
                            device=m.device)
    res = find_looper(m, tok, headT)
    if res is None:
        print("no greedy self-loop token found (try a larger model)")
        return
    n0, t = res
    print(f"looping token under greedy: {tok.decode([t])!r} (id {t}, trail {n0})")

    rows = []
    for dtype, quant in [("fp16", None), ("bf16", None), ("fp16", "int8")]:
        try:
            mm, tk = M.load_model(args.model, dtype=dtype, quantize=quant)
            ht = torch.as_tensor(mm.lm_head.weight.detach().float().cpu().numpy(),
                                 device=mm.device)
            run = E.count_run(E.gen_with_detector(mm, tk, ht, [t] * 5, t, max_new=25), t)
            print(f"  {dtype}~{quant or 'full'}: trailing-repeat = {run}")
            rows.append({"config": f"{dtype}~{quant or 'full'}", "trailing_repeat": run})
            del mm
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  {dtype}~{quant or 'full'}: OOM (skipped)")
            rows.append({"config": f"{dtype}~{quant or 'full'}", "trailing_repeat": None, "note": "OOM"})
            torch.cuda.empty_cache()

    decoder_cfgs = [
        ("greedy", {}),
        ("multinomial T=0.8", dict(temperature=0.8)),
        ("top-p 0.9 T=1.0 weighted", dict(temperature=1.0, top_p=0.9, top_p_mode="weighted")),
        ("top-p 0.9 T=0.8 weighted", dict(temperature=0.8, top_p=0.9, top_p_mode="weighted")),
        ("top-p 0.9 T=0.8 uniform", dict(temperature=0.8, top_p=0.9, top_p_mode="uniform")),
    ]
    for label, kw in decoder_cfgs:
        runs = [E.count_run(E.gen_with_detector(m, tok, headT, [t] * 5, t,
                                                max_new=25, **kw), t) for _ in range(args.seeds)]
        mean = float(np.mean(runs))
        mx = int(max(runs))
        print(f"  {label:26s} mean trailing-repeat = {mean:.2f}  max = {mx}")
        rows.append({"config": label, "trailing_repeat": mean, "max_repeat": mx,
                     "n": args.seeds})

    chat = tok.apply_chat_template([{"role": "user", "content": f"the balance is {tok.decode([t]*4)}"}],
                                   add_generation_prompt=True)
    if hasattr(chat, "input_ids"):
        chat = chat.input_ids
    chat_loop = E.count_run(E.gen_with_detector(m, tok, headT, chat, t, max_new=25), t)
    print(f"  chat-template-wrapped trigger: trailing-pit = {chat_loop}")
    rows.append({"config": "chat-template-wrapped", "trailing_repeat": chat_loop})

    safe = args.model.replace("/", "--")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"pit_robustness__{safe}.csv", index=False)
    print(f"\nSaved -> {OUT / f'pit_robustness__{safe}.csv'}")


if __name__ == "__main__":
    main()
