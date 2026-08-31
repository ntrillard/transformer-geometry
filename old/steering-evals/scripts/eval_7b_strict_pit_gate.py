#!/usr/bin/env python3
"""Small decoder gate for a known strict s(T) pit on Qwen2.5-7B.

Tests whether the token id 15 '0' pit survives:
  - greedy
  - multinomial T=0.8
  - standard top-p=0.9, T=1.0
  - standard top-p=0.9, T=0.8
  - uniform-over-nucleus top-p=0.9, T=0.8
  - chat-template wrapped trigger

Run: python eval_7b_strict_pit_gate.py [--quant nf4]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import eval_defense as E
import steering_geometry_test as M

OUT = Path("steering_geometry_results")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
PIT_ID = 15  # "0" token reported in paper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--pit-id", type=int, default=PIT_ID)
    ap.add_argument("--quant", default=None, choices=[None, "none", "nf4"],
                    help="quantization for the 7B run (nf4 if OOM)")
    ap.add_argument("--seeds", type=int, default=64)
    args = ap.parse_args()

    quant = None if args.quant in (None, "none") else args.quant
    print(f"Loading {args.model} (quant={quant or 'none'}) ...")
    m, tok = M.load_model(args.model, dtype="fp16", quantize=quant)
    head = torch.as_tensor(m.lm_head.weight.detach().float().cpu().numpy(),
                           device=m.device)
    t = args.pit_id
    print(f"Strict pit token: {tok.decode([t])!r} (id {t})")

    # greedy baseline
    greedy = E.count_run(E.gen_with_detector(m, tok, head, [t] * 5, t, max_new=25), t)
    print(f"  greedy: trailing-repeat = {greedy}")

    decoder_cfgs = [
        ("multinomial T=0.8", dict(temperature=0.8)),
        ("top-p 0.9 T=1.0 weighted", dict(temperature=1.0, top_p=0.9, top_p_mode="weighted")),
        ("top-p 0.9 T=0.8 weighted", dict(temperature=0.8, top_p=0.9, top_p_mode="weighted")),
        ("top-p 0.9 T=0.8 uniform", dict(temperature=0.8, top_p=0.9, top_p_mode="uniform")),
    ]
    rows = [{"config": "greedy", "trailing_repeat": greedy}]
    for label, kw in decoder_cfgs:
        runs = [E.count_run(E.gen_with_detector(m, tok, head, [t] * 5, t,
                                                max_new=25, **kw), t) for _ in range(args.seeds)]
        mean = float(np.mean(runs))
        mx = int(max(runs))
        print(f"  {label:26s} mean trailing-repeat = {mean:.2f}  max = {mx}")
        rows.append({"config": label, "trailing_repeat": mean, "max_repeat": mx, "n": args.seeds})

    # chat template wrapped trigger
    chat = tok.apply_chat_template([{"role": "user", "content": "the balance is 0000"}],
                                   add_generation_prompt=True)
    if hasattr(chat, "input_ids"):
        chat = chat.input_ids
    chat_loop = E.count_run(E.gen_with_detector(m, tok, head, chat, t, max_new=25), t)
    print(f"  chat-template-wrapped trigger: trailing-pit = {chat_loop}")
    rows.append({"config": "chat-template-wrapped", "trailing_repeat": chat_loop})

    safe = args.model.replace("/", "--")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"strict_pit_gate__{safe}.csv", index=False)
    print(f"\nSaved -> {OUT / f'strict_pit_gate__{safe}.csv'}")


if __name__ == "__main__":
    main()
