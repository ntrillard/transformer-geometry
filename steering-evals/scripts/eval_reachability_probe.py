#!/usr/bin/env python3
"""Reachability-as-knowledge probe: is the correct answer geometrically closer?

For each cloze fact, at the final-layer state:
  - theta_cell(correct answer token)  via full-vocab cone projection
  - theta_cell for same-category distractors
  - model's greedy answer (does it match?)

Analysis: does theta_cell separate known from unknown facts, and correct from
distractor answers?  A white-box "epistemic signal" would show smaller cone
angles for answers the model actually holds.

Run: python eval_reachability_probe.py --model Qwen/Qwen2-0.5B-Instruct
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import eval_boundary_instruments as B
import steering_geometry_test as M

FACTS = [
    ("The capital of France is", "Paris",
     ["London", "Berlin", "Madrid", "Rome", "Tokyo"]),
    ("The capital of Japan is", "Tokyo",
     ["Paris", "Seoul", "Beijing", "Bangkok", "Rome"]),
    ("The capital of Italy is", "Rome",
     ["Milan", "Naples", "Turin", "Madrid", "Athens"]),
    ("The capital of Germany is", "Berlin",
     ["Munich", "Hamburg", "Vienna", "Paris", "Warsaw"]),
    ("The capital of Canada is", "Ottawa",
     ["Toronto", "Montreal", "Vancouver", "Canberra", "London"]),
    ("The capital of Russia is", "Moscow",
     ["Kiev", "Minsk", "Berlin", "Warsaw", "Prague"]),
    ("A baby cat is called a", "kitten",
     ["puppy", "cub", "lamb", "calf", "chick"]),
    ("A baby dog is called a", "puppy",
     ["kitten", "cub", "foal", "chick", "joey"]),
    ("A baby sheep is called a", "lamb",
     ["goat", "calf", "kitten", "colt", "piglet"]),
    ("A baby cow is called a", "calf",
     ["lamb", "pony", "kid", "puppy", "duckling"]),
    ("The largest planet is", "Jupiter",
     ["Saturn", "Neptune", "Mars", "Venus", "Mercury"]),
    ("The fastest land animal is the", "cheetah",
     ["lion", "leopard", "horse", "tiger", "wolf"]),
    ("The closest star to Earth is the", "Sun",
     ["Moon", "Sirius", "Proxima", "Venus", "North"]),
    ("One plus one equals", "two",
     ["three", "four", "five", "ten", "zero"]),
    ("The opposite of hot is", "cold",
     ["warm", "wet", "soft", "dark", "fast"]),
    ("The color of grass is", "green",
     ["blue", "yellow", "red", "brown", "black"]),
    ("The color of the sky is", "blue",
     ["green", "red", "yellow", "white", "orange"]),
    ("Bees make", "honey",
     ["milk", "silk", "bread", "wool", "paper"]),
    ("Spiders spin", "webs",
     ["nests", "holes", "pots", "balls", "shells"]),
    ("The freezing point of water is", "freezing",
     ["boiling", "warm", "hot", "steam", "dry"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    args = ap.parse_args()

    safe = args.model.replace("/", "--")
    model, tok = M.load_model(args.model, dtype="fp16")
    V = model.config.vocab_size
    W = model.lm_head.weight.detach().cpu().float().numpy()
    if W.size == 0 or W.shape[0] != V:
        W = model.get_input_embeddings().weight.detach().cpu().float().numpy()[:V]
    W_dev = torch.as_tensor(W, device=model.device, dtype=torch.float32)
    Wn_dev = torch.as_tensor(W / np.linalg.norm(W, axis=1, keepdims=True),
                             device=model.device, dtype=torch.float32)
    L = model.config.num_hidden_layers

    def first_id(word):
        return tok(" " + word.strip(), add_special_tokens=False).input_ids[0]

    TEMPLATES = [lambda f: f, lambda f: f"Q: {f} A:", lambda f: f"Fact: {f}"]
    rows = []
    for prompt, correct, distract in FACTS:
        corr_id = first_id(correct)
        cand_words = [correct] + list(distract)
        cand_ids = []
        for wd in cand_words:
            tid = first_id(wd)
            if tid not in cand_ids:
                cand_ids.append(tid)
        for ti, tmpl in enumerate(TEMPLATES):
            p = tmpl(prompt)
            pids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                out = model(pids, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1].float()
            u = (h / h.norm()).cpu().numpy()

            angles, _ = B.cone_angles(u, cand_ids, W_dev)
            th = {wd: float(a) for wd, a in zip(cand_words, angles)}

            pred_id = int((h @ Wn_dev.T).argmax())
            ok = int(pred_id == corr_id)
            theta_pred = float(angles[cand_ids.index(pred_id)]) \
                if pred_id in cand_ids else float("nan")

            # sampling recovery (5 samples): does correct first-token appear?
            rec = 0
            if not ok:
                for s in range(5):
                    torch.manual_seed(100 + s)
                    lg = h @ Wn_dev.T
                    p = torch.softmax(lg, dim=0)
                    order = p.argsort(descending=True)
                    cum = p[order].cumsum(0)
                    keep = order[:int((cum <= 0.9).sum()) + 1]
                    m = torch.zeros_like(p); m[keep] = 1
                    p = (p * m) / (p * m).sum()
                    if int(torch.multinomial(p, 1)) == corr_id:
                        rec += 1

            d_corr = th[correct]
            d_dist = [th[d] for d in distract]
            rows.append(dict(model=safe, prompt=p, template=ti, correct=correct,
                             theta_correct=d_corr,
                             theta_distr_min=min(d_dist),
                             theta_distr_mean=float(np.mean(d_dist)),
                             margin=float(np.mean(d_dist)) - d_corr,
                             greedy_ok=ok, theta_own_pred=theta_pred,
                             recovery_of_5=rec))
        tag = "OK " if rows[-1]["greedy_ok"] else "MISS"
        print(f"{tag} {prompt[:40]:42s} ans={correct:9s} "
              f"th_corr={rows[-1]['theta_correct']:5.2f} "
              f"own_pred_th={rows[-1]['theta_own_pred']:5.2f} "
              f"rec={rows[-1]['recovery_of_5']}")

    df = pd.DataFrame(rows)
    safe = args.model.replace("/", "--")
    OUT = Path("../steering_geometry_results")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"reachability_probe__{safe}.csv", index=False)

    def auc(pos, neg):
        pos, neg = np.asarray(pos, float), np.asarray(neg, float)
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        gt = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
        return float(gt)

    known = df[df.greedy_ok == 1]
    unknown = df[df.greedy_ok == 0]
    print(f"\n=== {safe} : {len(df)} states, {df.greedy_ok.sum()} known ===")
    print(f"AUC theta_correct (known vs missed, smaller=better): "
          f"{auc(known.theta_correct, unknown.theta_correct):.3f}")
    print(f"mean theta_correct  known={known.theta_correct.mean():.2f}  "
          f"missed={unknown.theta_correct.mean():.2f}")
    print(f"theta(own greedy pred): known={known.theta_own_pred.mean():.2f}  "
          f"missed={unknown.theta_own_pred.mean():.2f}  "
          f"(confidence signal if known < missed)")
    diff = df.theta_distr_mean - df.theta_correct
    print(f"correct < distractor-mean: {(diff>0).sum()}/{len(df)}  "
          f"median margin +{diff.median():.2f} deg")
    print(f"correct smallest among candidates: "
          f"{(df.theta_correct <= df.theta_distr_min).sum()}/{len(df)}")
    missed = df[df.greedy_ok == 0]
    if len(missed):
        print(f"missed-but-recoverable-by-sampling (>=1/5): "
              f"{(missed.recovery_of_5 > 0).sum()}/{len(missed)}  "
              f"(their mean theta_correct={missed.theta_correct.mean():.2f})")


if __name__ == "__main__":
    main()
