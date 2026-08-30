#!/usr/bin/env python3
"""Steering cost on a standardized benchmark (MMLU subsets, lm-eval harness).

Conditions: base / persist tangent-hook alpha in {0.1, 0.3} toward ' apple'
applied at the final layer during ALL forwards (scoring included).
Accuracy per subject via lm_eval.simple_evaluate, letter-logit scoring,
0-shot (speed; leaderboard convention is 5-shot - noted).

Run: python eval_mmlu_steered.py
"""
import argparse
from pathlib import Path
import pandas as pd
import torch

import steering_geometry_test as M
from eval_practical_steering import sphere_hook
import lm_eval
from lm_eval.models.huggingface import HFLM

TASKS = ["mmlu_abstract_algebra", "mmlu_anatomy", "mmlu_astronomy"]


@torch.no_grad()
def get_w_row(model, tok, word):
    W = model.lm_head.weight.detach().cpu().float().numpy()
    if W.size == 0:
        W = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    tid = tok(" " + word.strip(), add_special_tokens=False).input_ids[0]
    return torch.as_tensor(W[tid], device=model.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen2-0.5B-Instruct", "google/gemma-3-1b-it"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--word", default="apple")
    args = ap.parse_args()

    all_rows = []
    for mid in args.models:
        print(f"\n===== {mid} =====")
        model, tok = M.load_model(mid, dtype="fp16")
        layer = model.model.layers[-1]
        w_row = get_w_row(model, tok, args.word)

        for cond, a in [("base", None)] + [
                (f"persist-a{a}", a) for a in args.alphas]:
            handle = None
            if a is not None:
                handle = layer.register_forward_hook(sphere_hook(w_row, a))
            try:
                lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
                res = lm_eval.simple_evaluate(model=lm, tasks=TASKS,
                                              num_fewshot=0)
            finally:
                if handle is not None:
                    handle.remove()
            for t in TASKS:
                acc = res["results"][t].get("acc,none")
                accs = [v for k, v in res["results"][t].items() if k.startswith("acc")]
                acc = acc if acc is not None else accs[0]
                all_rows.append(dict(model=mid.split("/")[-1], condition=cond,
                                     task=t, acc=float(acc)))
                print(f"  {cond:12s} {t:26s} acc={float(acc):.3f}")

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    OUT = Path("../steering_geometry_results")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "mmlu_steered.csv", index=False)
    print("\n=== mean accuracy across subjects ===")
    print(df.groupby(["model", "condition"]).acc.mean().round(4).to_string())


if __name__ == "__main__":
    main()
