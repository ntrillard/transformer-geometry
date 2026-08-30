#!/usr/bin/env python3
"""Chord WALK on every sphere in the multiverse (Qwen2-0.5B, ~15 s).

The multiverse theorem: every layer's sphere S_L shares one coordinate system
(same BOS axis up to scale, same topic ring).  Test whether the chord walk
works from EVERY sphere's state - i.e. the topic-transition path exists at
every depth, not just the readout sphere.

W1  TRANSITIONS PER SPHERE: from each sphere's position-last state, run the
    4-deg re-aimed chord walk toward every other topic family (6 pairs x
    8 spheres).  Report reached? / crossing step / unreached per sphere.
W2  LAYER-CONSISTENCY: does the crossing step (and the direct-jump behavior)
    hold at every sphere, or does reachability degrade with depth?

Run:  python eval_layer_walk.py      (~15 s)
"""
import math
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = 'Qwen/Qwen2-0.5B-Instruct'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_STEPS = 12
STEP_DEG = 4.0
LAYERS = [0, 1, 3, 6, 9, 12, 15, 18, 21, 23]


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
    hidden = [h[0].cpu().float().numpy() for h in hid.hidden_states]   # [l+1] = layer l

    def depole(v, u):
        v = v - (v @ u) * u
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def ring_centers(u):
        return {c: depole(Wn[i].mean(0), u) for c, i in topics.items()}

    print(f"[model] {MODEL}  |  chord walk per sphere")
    print(f"{'sphere':>6} | {'unreached':>9} | {'mean step':>9} | pairs (step | -)")
    print("-" * 70)

    all_rows = []
    for l in LAYERS:
        hs = hidden[l + 1][-1, :]                 # position-last state on sphere L
        hs = hs / np.linalg.norm(hs)
        u = hidden[l + 1][0, :]                   # position-0 = BOS axis of sphere L
        u = u / np.linalg.norm(u)
        cent = ring_centers(u)

        def topic_of(h):
            r = depole(Wn[int((h @ Wn.T).argmax())], u)
            return min(names, key=lambda c: np.degrees(
                np.arccos(np.clip(r @ cent[c], -1, 1))))

        row = {}
        unreached = 0
        steps_all = []
        for s in names:
            for t in names:
                if t == s:
                    continue
                ht = hs.copy()
                for _ in range(3):                # settle into start topic
                    best = rows[s][int(np.argmax(rows[s] @ ht))]
                    ht = M.rotate_toward(ht, M.tangent_direction(ht, best),
                                         math.radians(STEP_DEG))
                crossed = None
                for k in range(1, MAX_STEPS + 1):
                    best = rows[t][int(np.argmax(rows[t] @ ht))]
                    ht = M.rotate_toward(ht, M.tangent_direction(ht, best),
                                         math.radians(STEP_DEG))
                    if topic_of(ht) == t:
                        crossed = k
                        break
                row[(s, t)] = crossed
                if crossed is None:
                    unreached += 1
                else:
                    steps_all.append(crossed)
        all_rows.append((l, row))
        mean_step = np.mean(steps_all) if steps_all else float('nan')
        cells = " ".join(
            (f"{row[(s, t)] if row[(s, t)] else '-'}" for s in names for t in names
             if t != s))
        print(f"{l:6d} | {unreached:9d} | {mean_step:9.1f} | {cells}")

    print("\nconsistency: crossing step same across spheres?  "
          "(row = sphere, column = ordered pair)")
    pairs = [(s, t) for s in names for t in names if t != s]
    head = "       " + "".join(f"{s[0]}{t[0]:>3s}" for s, t in pairs)
    print(head)
    for l, row in all_rows:
        print(f"{l:5d}  " + "".join(
            f"{str(row[p]) if row[p] else '-':>4s}" for p in pairs))
    print(f"\n[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()