#!/usr/bin/env python3
"""Show the TOKENS as we walk the topic path across sphere layers.

For the chosen start->target pair, at each sampled sphere layer, run the
chord walk (3 settle steps + 8 walk steps) and print the decoded top-1 token
+ its topic at every step.  The 'map' rendered as actual text.

Run:  python eval_walk_tokens.py [model] [start] [target]   (Qwen ~8s, fast)
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2-0.5B-Instruct'
START = sys.argv[2] if len(sys.argv) > 2 else 'food'
TARGET = sys.argv[3] if len(sys.argv) > 3 else 'city'
STEP_DEG = 4.0
MAX_WALK = 8
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()[:model.config.vocab_size]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    topics = {cls: np.array([word2id[w] for w in words if w in word2id])
              for cls, words in CLASSES.items()
              if sum(1 for w in words if w in word2id) >= 6}
    names = list(topics)
    rows = {c: Wn[i] for c, i in topics.items()}

    pid = tok('Once upon a time', add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
    hidden = [h[0].cpu().float().numpy() for h in hid.hidden_states]

    n_hidden = len(hidden) - 1
    spheres = np.linspace(0, n_hidden - 1, 6, dtype=int)

    def walk(layer):
        hs = hidden[layer + 1][-1, :]
        hs = hs / np.linalg.norm(hs)
        u = hidden[layer + 1][0, :]
        u = u / np.linalg.norm(u)

        def depole(v):
            v = v - (v @ u) * u
            nn = np.linalg.norm(v)
            return v / nn if nn > 1e-9 else v

        cent = {c: depole(Wn[i].mean(0)) for c, i in topics.items()}

        def toknow(h):
            t = int((h @ Wn.T).argmax())
            r = depole(Wn[t])
            top = min(names, key=lambda c: np.degrees(
                np.arccos(np.clip(r @ cent[c], -1, 1))))
            return t, tok.decode([t], skip_special_tokens=True), top

        h = hs.copy()
        # settle into START
        for _ in range(3):
            best = rows[START][int(np.argmax(rows[START] @ h))]
            h = M.rotate_toward(h, M.tangent_direction(h, best), math.radians(STEP_DEG))
        steps = [toknow(h)]
        for _ in range(MAX_WALK):
            best = rows[TARGET][int(np.argmax(rows[TARGET] @ h))]
            h = M.rotate_toward(h, M.tangent_direction(h, best), math.radians(STEP_DEG))
            steps.append(toknow(h))
        return steps

    print(f"[model] {MODEL}  |  {START} -> {TARGET}")
    print(f"{'sphere':>6} | tokens at each step (topic):")
    print("-" * 70)
    for l in spheres:
        steps = walk(int(l))
        toks = " | ".join(f"`{t}`:{c[:3]}" for _, t, c in steps)
        print(f"{l:6d} | {toks}")
    print(f"\n[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()