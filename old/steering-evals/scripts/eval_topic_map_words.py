#!/usr/bin/env python3
"""eval_topic_map_words.py — FAST: word-level topic map walk (60 nodes).

Same closed-form map as eval_topic_map_walk but at WORD resolution: each of
the 60 single-token food/animal/... words is a node. Plant each word via
2*alpha* (closed form), read affinity to every other word row -> 60x60 map.

Measures:
  W1 ring: PCA ring of the 60 centroids; corr(affinity, ring distance)
  W2 absorbing: Markov stationary over 60 nodes -> top absorbing words
  W3 greedy drift from native; the walk path in words
  W4 sink test: is the raw (unsteered) base state already leaning into the
     absorbing basin?  max base affinity vs stationary argmax

Run: timeout 90 python3 -u eval_topic_map_words.py Qwen/Qwen2-0.5B-Instruct
     timeout 90 python3 -u eval_topic_map_words.py google/gemma-3-1b-it
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
    lm_w = model.lm_head.weight

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    words = list(word2id)
    idxs = np.array([word2id[w] for w in words])
    N = len(words)

    # RAM-safe: only slice the needed rows
    rows = lm_w[idxs].detach().float().cpu().numpy()
    Wn = rows * (1.0 / np.sqrt(np.einsum('ij,ij->i', rows, rows)[:, None] + 1e-12))

    pid = tok('Once upon a time', add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    li = model.config.num_hidden_layers - 1
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
        h0 = hid.hidden_states[li + 1][0, 0]
        hs = hid.hidden_states[li + 1][0, -1]
        native = int((hs.float() @ lm_w.float().T).argmax().item())
    u = (h0 / h0.norm()).cpu().float().numpy()

    def depole(v):
        v = v - (v @ u) * u
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    # ring of the 60 centroids
    cents = np.stack([depole(Wn[i]) for i in range(N)])
    C0 = cents - cents.mean(0)
    _, Vp = np.linalg.eigh(C0.T @ C0)
    P2 = C0 @ Vp[:, -2:]
    az = (np.degrees(np.arctan2(P2[:, 1], P2[:, 0])) + 360) % 360

    def ring_dist(i, j):
        d = abs(az[i] - az[j]) % 360
        return min(d, 360 - d)

    h_base = (hs / hs.norm()).cpu().float().numpy()
    Wn_nat = lm_w[native].detach().float().cpu().numpy()
    Wn_nat = Wn_nat / np.linalg.norm(Wn_nat)
    # plant each word -> state
    states = np.zeros((N, h_base.shape[0]))
    for c in range(N):
        best = Wn[c]
        tau = M.tangent_direction(h_base, best)
        A_ = float(h_base @ (Wn_nat - best))
        B_ = float(tau @ (best - Wn_nat))
        alpha = math.atan2(A_, B_) if B_ > 1e-12 else 0.3
        alpha = min(2 * alpha + 0.02, 0.5)
        states[c] = M.rotate_toward(h_base, tau, alpha)

    # W1: affinity map
    A = np.zeros((N, N))
    for i in range(N):
        si = depole(states[i])
        A[i] = si @ Wn.T

    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    rd = np.array([ring_dist(i, j) for i, j in pairs])
    aff = np.array([A[i, j] for i, j in pairs])
    print(f"[{MODEL}] native={tok.decode([native]).strip()!r}  nodes={N}")
    print(f"[W1] word-ring: corr(affinity, ring-dist) = "
          f"{np.corrcoef(rd, aff)[0, 1]:+.3f}   "
          f"adjacent {aff[rd < 60].mean():.3f} vs far {aff[rd >= 60].mean():.3f}")

    # W2: absorbing words
    P = np.exp(A / TAU)
    P /= P.sum(1, keepdims=True)
    pi = np.full(N, 1 / N)
    for _ in range(300):
        pi = pi @ P
    order = np.argsort(pi)[::-1]
    print(f"[W2] absorbing words (top 8): "
          + ", ".join(f"{words[i]}({pi[i]:.2f})" for i in order[:8]))
    print(f"     bottom 3: " + ", ".join(f"{words[i]}({pi[i]:.2f})"
          for i in order[-3:]))

    nidx = int(np.where(idxs == native)[0][0]) if native in idxs else 0
    walk, seen = [nidx], {nidx}
    cur = nidx
    while True:
        Aj = A[cur].copy()
        Aj[cur] = -np.inf
        nn = int(np.argmax(Aj))
        walk.append(nn)
        if nn in seen:
            break
        seen.add(nn)
        cur = nn
    print(f"[W3] greedy drift: {' -> '.join(words[i] for i in walk)}")
    steps = [ring_dist(a, b) for a, b in zip(walk, walk[1:])]
    if steps:
        print(f"     mean |azimuth step|: {np.mean(steps):.0f} deg "
              f"(chance ~{int(180 - 360 / N)} deg)")

    # W4: is the sink native?  raw base-state affinity distribution
    base_aff = depole(h_base) @ Wn.T
    bmax = int(np.argmax(base_aff))
    print(f"[W4] unsteered base already closest to {words[bmax]!r} "
          f"(aff {base_aff[bmax]:.3f}); "
          f"stationary-argmax {words[order[0]]!r}")
    print(f"     base-affinity to absorbing argmax: {base_aff[order[0]]:.3f}")

    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()