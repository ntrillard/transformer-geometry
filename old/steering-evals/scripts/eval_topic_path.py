#!/usr/bin/env python3
"""Fast topic-path experiments on the LM-head map (~15 s on a GPU box).

T1  TOPIC RING: de-poled class centroids -> angular order around the equator.
    If topics sit at distinct azimuths, the shortest 'path' between two topics
    is their equatorial angular distance (the map 'between' topics).
T2  CHORD WALK: from a start topic, stepwise chord-inversion (4 deg re-aim at
    the target's best-positioned member) toward a target topic; log the top-1
    token + nearest topic at every step -> the literal path the geometry
    traces across topics.
T3  OPEN- vs CLOSED-LOOP: one 32 deg arc toward the initial best member
    (open loop) vs 8x4 deg re-aimed steps (closed loop) - do they land in the
    same topic cone?

Run:  python eval_topic_path.py [model]     (default Qwen2-0.5B)
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
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()[:model.config.vocab_size]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)

    # class member rows (single-token words only)
    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    topics = {cls: np.array([word2id[w] for w in words if w in word2id])
              for cls, words in CLASSES.items()
              if sum(1 for w in words if w in word2id) >= 6}
    print(f"[topic rows] { {c: len(i) for c, i in topics.items()} }")

    # BOS/latitude axis = position-0 final-layer state ('Once upon a time')
    pid = tok('Once upon a time', add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    li = model.config.num_hidden_layers - 1
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
        h0 = hid.hidden_states[li + 1][0, 0]
        hs = hid.hidden_states[li + 1][0, -1]
    u = (h0 / h0.norm()).cpu().float().numpy()

    def depole(v):
        v = v - (v @ u) * u
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    cent = {c: depole(Wn[i].mean(0)) for c, i in topics.items()}

    # ---- T1 topic ring ----
    names = list(topics)
    C = np.stack([cent[c] for c in names])
    C0 = C - C.mean(0)
    _, Vp = np.linalg.eigh(C0.T @ C0)               # PCA of de-poled centroids
    P2 = C0 @ Vp[:, -2:]
    angs = (np.degrees(np.arctan2(P2[:, 1], P2[:, 0])) + 360) % 360
    order = sorted(range(len(names)), key=lambda i: angs[i])
    print("\n[T1] topic ring (de-poled class centroids around the equator):")
    for i in order:
        print(f"     {names[i]:8s}  az {angs[i]:6.1f} deg")
    print("     circular order: " + " -> ".join(names[i] for i in order))
    D = np.degrees(np.arccos(np.clip(C @ C.T, -1, 1)))
    print("     equatorial distances (deg):")
    print("          " + " ".join(f"{c:>8s}" for c in names))
    for a, r in zip(names, D):
        print(f"     {a:8s} " + " ".join(f"{x:8.1f}" for x in r))

    # ---- T2 chord walk + T3 open vs closed loop ----
    start, target = START, TARGET
    h = (hs / hs.norm()).cpu().float().numpy()
    rows_tgt = Wn[topics[target]]
    rows_start = Wn[topics[start]]
    Lbl = lambda t: min(names, key=lambda c: np.degrees(
        np.arccos(np.clip(depole(Wn[t]) @ cent[c], -1, 1))))

    def logit_top1(h):
        L = h @ Wn.T
        t = int(L.argmax())
        return t, tok.decode([t], skip_special_tokens=True), L[t]

    print(f"\n[T2] chord walk  {start} -> {target}   (4 deg steps, re-aim at best member):")
    t, s, _ = logit_top1(h)
    print(f"     start          {s!r:>14s}  (topic {Lbl(t):>6s})")
    # 3 steps INTO the start topic first
    for _ in range(3):
        best = rows_start[int(np.argmax(rows_start @ h))]
        tau = M.tangent_direction(h, best)
        h = M.rotate_toward(h, tau, math.radians(4))
    t, s, _ = logit_top1(h)
    print(f"     after enter    {s!r:>14s}  (topic {Lbl(t):>6s})")
    trail = []
    for k in range(8):
        best = rows_tgt[int(np.argmax(rows_tgt @ h))]
        tau = M.tangent_direction(h, best)
        h = M.rotate_toward(h, tau, math.radians(4))
        t, s, _ = logit_top1(h)
        trail.append((k + 1, s, Lbl(t)))
        print(f"     step {k + 1:<2d}         {s!r:>14s}  (topic {Lbl(t):>6s})")

    # T3: open loop (single 32 deg arc, initial best member)
    h2 = (hs / hs.norm()).cpu().float().numpy()
    for _ in range(3):
        best = rows_start[int(np.argmax(rows_start @ h2))]
        h2 = M.rotate_toward(h2, M.tangent_direction(h2, best), math.radians(4))
    best0 = rows_tgt[int(np.argmax(rows_tgt @ h2))]
    h2 = M.rotate_toward(h2, M.tangent_direction(h2, best0), math.radians(32))
    t2, s2, _ = logit_top1(h2)
    t8, s8, _ = logit_top1(h)                      # closed-loop endpoint above
    print(f"\n[T3] open 32 deg one-arc:  {s2!r:>14s}  (topic {Lbl(t2):>6s})")
    print(f"     closed 8x4 deg walk:  {s8!r:>14s}  (topic {Lbl(t8):>6s})")
    print(f"     in target family: open {Lbl(t2)==target}   walk {Lbl(t8)==target}")

    print(f"\n[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()