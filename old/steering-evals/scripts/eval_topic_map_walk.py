#!/usr/bin/env python3
"""eval_topic_map_walk.py — FAST: closed-form topic map + Markov walk (inward).

The old chord-walk (4 deg re-aimed steps) is replaced by the closed-form
machinery: planting topic A = ONE rotation at 2*alpha*+0.02 (exact crossing
root vs native). From each planted state we read the affinity to ALL topic
centroids -> the DIRECTED topic-transition map of the model's semantics.

T1  TRANSITION MAP  A[a,b] = cos(depole(state after planting a), centroid b)
T2  GEOGRAPHY LAW   corr(A[a,b], ring_dist(a,b))  (does azimuth predict pull?)
T3  ASYMMETRY       mean |A[a,b]-A[b,a]|; strongest directional edges
T4  MARKOV WALK     p(b|a) = softmax(A[a,:]/tau); stationary distribution via
                    power iteration -> which topic basins are ABSORBING.
T5  GREEDY WALK     a_{k+1} = argmax A[a_k,:] (cycle / attractor detection).

Run: timeout 60 python3 -u eval_topic_map_walk.py Qwen/Qwen2-0.5B-Instruct
     timeout 90 python3 -u eval_topic_map_walk.py google/gemma-3-1b-it
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2-0.5B-Instruct'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TAU = 0.05


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight  # (V, dim) fp16 on GPU - never copied wholesale

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    topics = {cls: np.array([word2id[w] for w in words if w in word2id])
              for cls, words in CLASSES.items()
              if sum(1 for w in words if w in word2id) >= 6}
    names = list(topics)

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

    # tiny CPU row slices (keep RAM flat - never copy the vocab)
    rows = {}   # normalized per-topic member rows, CPU, small
    for c, ids in topics.items():
        r = lm_w[ids].detach().float().cpu().numpy()
        rows[c] = r * (1.0 / np.sqrt((r * r).sum(1, keepdims=True) + 1e-12))
    cent = {c: depole(rows[c].mean(0)) for c in names}

    # ring azimuths (PCA of de-poled centroids, same as old T1)
    C = np.stack([cent[c] for c in names])
    C0 = C - C.mean(0)
    _, Vp = np.linalg.eigh(C0.T @ C0)
    P2 = C0 @ Vp[:, -2:]
    az = {c: (np.degrees(np.arctan2(*P2[i][::-1])) + 360) % 360
          for i, c in enumerate(names)}

    def ring_dist(a, b):
        d = abs(az[a] - az[b]) % 360
        return min(d, 360 - d)

    # base state + native argmax ON GPU (no big CPU copy)
    h_base = (hs / hs.norm()).cpu().float().numpy()
    with torch.no_grad():
        L0g = (hs.float() @ lm_w.float().T)
    native = int(L0g.argmax().item())
    Wn_nat = lm_w[native].detach().float().cpu().numpy()
    Wn_nat = Wn_nat / np.linalg.norm(Wn_nat)

    states = {}
    for c in names:
        best = rows[c][int(np.argmax(rows[c] @ h_base))]
        tau = M.tangent_direction(h_base, best)
        A_ = float(h_base @ (Wn_nat - best))
        B_ = float(tau @ (best - Wn_nat))
        # rotate toward best member, past the crossing by margin
        alpha = math.atan2(A_, B_) if B_ > 1e-12 else 0.3
        alpha = min(2 * alpha + 0.02, 0.5)
        hc = M.rotate_toward(h_base, tau, alpha)
        states[c] = hc

    # T1: affinity transition map
    A = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            A[i, j] = float(depole(states[a]) @ cent[b])
    print(f"[model] {MODEL}   native={tok.decode([native]).strip()!r}")
    print(f"\n[T1] closed-form topic transition map A[a->b] = cos post-plant affinity")
    print("      " + "".join(f"{c:>7s}" for c in names))
    for i, a in enumerate(names):
        print(f"  {a:>6s}" + "".join(f"{A[i, j]:>7.2f}" for j in range(len(names))))

    # T2: geography law
    pairs = [(i, j) for i in range(len(names)) for j in range(len(names)) if i != j]
    rd = np.array([ring_dist(names[i], names[j]) for i, j in pairs])
    aff = np.array([A[i, j] for i, j in pairs])
    print(f"\n[T2] geography: corr(affinity, az-ring distance) = "
          f"{np.corrcoef(rd, aff)[0, 1]:+.3f}   "
          f"(affinity to ADJACENT topics: {aff[rd < 60].mean():.3f} vs "
          f"FAR topics: {aff[rd >= 60].mean():.3f})")

    # T3: asymmetry
    asym = np.abs(A - A.T)
    print(f"\n[T3] asymmetry: mean |A[a,b]-A[b,a]| = {asym.mean():.3f}")
    idx = np.argsort(asym, axis=None)[::-1][:4]
    for k in idx:
        i, j = np.unravel_index(k, asym.shape)
        if i < j:
            print(f"     {names[i]:>6s} -> {names[j]:<6s} {A[i, j]:.2f}  vs  "
                  f"{names[j]:>6s} -> {names[i]:<6s} {A[j, i]:.2f}")

    # T4: Markov walk stationary distribution
    P = np.exp(A / TAU)
    P /= P.sum(1, keepdims=True)
    pi = np.full(len(names), 1 / len(names))
    for _ in range(200):
        pi = pi @ P
    order = np.argsort(pi)[::-1]
    print(f"\n[T4] Markov stationary distribution (absorbing topic basins):")
    for i in order:
        print(f"     {names[i]:>8s}  {pi[i]:.3f}")
    print(f"     max/min = {pi.max() / pi.min():.1f}x")

    # T5: greedy drift excluding self, to reveal the model's associative pull
    print(f"\n[T5] greedy drift (follow strongest affinity to a DIFFERENT topic):")
    start = names[int(order[0])]
    walk, seen = [start], {start}
    cur = start
    while True:
        j = names.index(cur)
        Aj = A[j].copy()
        Aj[j] = -np.inf
        nn = names[int(np.argmax(Aj))]
        walk.append(nn)
        if nn in seen:
            break
        seen.add(nn)
        cur = nn
    print("     " + " -> ".join(walk))
    steps = [ring_dist(a, b) for a, b in zip(walk, walk[1:])]
    if steps:
        print(f"     mean |azimuth step| of greedy edges: {np.mean(steps):.0f} deg"
              f"  (chance ~{int(180 - 360 / len(names))} deg)")
    print(f"\n[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()