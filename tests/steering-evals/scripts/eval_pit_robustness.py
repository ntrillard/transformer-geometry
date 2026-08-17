#!/usr/bin/env python3
"""Evidence for reply point 2.1: cow-tipping / repetition-loop ROBUSTNESS matrix.

Ships the measurement behind the numbers in the post:
  - discovers a token whose terminal-repetition collapses into a loop under greedy;
  - measures the loop under fp16 vs bf16 vs int8 (bitsandbytes);
  - and under sampling (temp 0.8 / temp 1.0 + top-p 0.9) and a chat template.

Run:  python eval_pit_robustness.py        (Qwen2-0.5B cached; < ~1 min)
"""
import numpy as np
import torch

import eval_defense as E
import steering_geometry_test as M

MODEL = "Qwen/Qwen2-0.5B-Instruct"


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
    m, tok = M.load_model(MODEL, dtype="fp16")
    headT = torch.as_tensor(m.lm_head.weight.detach().float().cpu().numpy(),
                            device=m.device)
    res = find_looper(m, tok, headT)
    if res is None:
        print("no greedy self-loop token found at 0.5B scale (try a larger model)")
        return
    n0, t = res
    print(f"looping token under greedy: {tok.decode([t])!r} (id {t}, trail {n0})")

    for dtype, quant in [("fp16", None), ("bf16", None), ("fp16", "int8")]:
        mm, tk = M.load_model(MODEL, dtype=dtype, quantize=quant)
        ht = torch.as_tensor(mm.lm_head.weight.detach().float().cpu().numpy(),
                             device=mm.device)
        run = E.count_run(E.gen_with_detector(mm, tk, ht, [t] * 5, t, max_new=25), t)
        print(f"  {dtype}~{quant or 'full'}: trailing-repeat = {run}")
        del mm
        torch.cuda.empty_cache()

    for label, kw in [("greedy", {}), ("temp=0.8", dict(temperature=0.8)),
                      ("temp=1.0 top-p=0.9", dict(temperature=1.0, top_p=0.9))]:
        runs = [E.count_run(E.gen_with_detector(m, tok, headT, [t] * 5, t,
                                                max_new=25, **kw), t) for _ in range(10)]
        print(f"  {label:20s} mean trailing-repeat = {np.mean(runs):.1f}")

    chat = tok.apply_chat_template([{"role": "user", "content": "the balance is 0000"}],
                                   add_generation_prompt=True)
    if hasattr(chat, "input_ids"):
        chat = chat.input_ids
    zero = next((i for i in range(500) if tok.decode([i]).strip() == "0"),
                tok("0").input_ids[0])
    print(f"  chat-template-wrapped digit-0 trigger: trailing-0 = "
          f"{E.count_run(E.gen_with_detector(m, tok, headT, chat, zero, max_new=25), zero)}")


if __name__ == "__main__":
    main()