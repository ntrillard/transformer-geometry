"""eval_target_trajectory.py — FAST: is the repetition loop a property of the
TARGET's native continuation, independent of steering?

Force the target token as the first generated token (NO steering — cursor
override), then free-run. Measure native 4-gram repetition and on-topic
drift. Prediction: 'bread' self-reinforces (loops) while 'apple'/'chicken'
drift naturally — the loop is a target-trajectory property, steering just
plants the seed.

Run: python3 eval_target_trajectory.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
     python3 eval_target_trajectory.py --model google/gemma-3-1b-it --tag gemma
"""
import argparse

import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGETS = ['chicken', 'apple', 'bread', 'soup', 'milk', 'rice']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2-0.5B-Instruct')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--prompt', default="For dinner I made")
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    model, tok = M.load_model(a.model, dtype='fp16')
    word2id = {}
    for w in CLASSES['food']:
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])

    def gen(forced_tid, seed=0, n=24, top_p=0.9):
        torch.manual_seed(seed)
        ids = tok(a.prompt, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        # force the target as the FIRST generated token (native continuation after)
        ids = torch.cat([ids, torch.tensor([[forced_tid]], device=ids.device)], dim=1)
        toks = [forced_tid]
        for _ in range(n - 1):
            with torch.no_grad():
                Ln = model(ids).logits[0, -1].float()
            p = torch.softmax(Ln, dim=0)
            q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
            keep = order[:int((cum <= top_p).sum()) + 1]
            m = torch.zeros_like(q); m[keep] = 1
            q = (q * m) / (q * m).sum()
            nxt = int(torch.multinomial(q, 1))
            toks.append(int(nxt))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return toks

    def rep4(toks):
        if len(toks) < 8:
            return 1.0
        reps = 0
        n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        for i in range(len(toks) - 3):
            if n4[i] in n4[i + 1:]:
                reps += 1
        return reps / (len(toks) - 3)

    def rep1(toks):
        """fraction of positions where the token repeats the PREVIOUS one."""
        return np.mean([toks[i] == toks[i - 1] for i in range(1, len(toks))])

    print(f"== [{a.tag}] native continuation after forced target ({a.prompt!r}) ==")
    print(f"{'target':>8} {'4rep':>6} {'prevrep':>7} {'diversity':>9}  sample")
    for tgt in TARGETS:
        if tgt not in word2id:
            continue
        ti = word2id[tgt]
        reps4, reps1, divs, samples = [], [], [], []
        for sd in range(a.seeds):
            toks = gen(ti, seed=sd, n=a.n)
            reps4.append(rep4(toks))
            reps1.append(rep1(toks))
            divs.append(len(set(toks)) / len(toks))
            samples.append(tok.decode(toks)[:56])
        print(f"{tgt:>8} {np.mean(reps4):>6.2f} {np.mean(reps1):>7.2f} "
              f"{np.mean(divs):>9.2f}  {samples[0]!r}")


if __name__ == "__main__":
    main()