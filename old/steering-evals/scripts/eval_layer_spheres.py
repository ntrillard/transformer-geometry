#!/usr/bin/env python3
"""The sphere multiverse, measured per layer (Qwen2-0.5B, ~10 s).

The multiverse thesis: a transformer is a STACK of spheres - RMSNorm at every
layer L rescales onto its own sphere S_L (its own BOS/latitude axis u_L, its
own raw-head decision partition, and - if the semantic map survives depth -
its own topic ring).  Models are different universes (Qwen/GPT-2 equatorial,
Pythia polar).  Steering operates within a single sphere's tangent space, but
the computation walks the tower.

L1  EQUATOR PER SPHERE: median angle of the token rows to each layer's BOS
    axis u_L.  ~90 deg at every layer => every sphere in the tower is
    equatorial (content on the longitude, context on the pole).
L2  RING PER SPHERE: recompute the topic-ring azimuths de-poled by each
    layer's OWN u_L.  Does the ring (order, distances) survive at mid-stack
    spheres, or is it a readout-sphere-only object?
L3  ACCESSIBILITY PER SPHERE: from the position-last state at layer L, does a
    17 deg chord-inversion arc toward a family make that family top-1 under
    the FINAL head (raw-head accessibility of the final sphere)?  This is the
    #17 diagnostic, resolved per sphere.

Run:  python eval_layer_spheres.py     (~10 s)
"""
import math
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = 'Qwen/Qwen2-0.5B-Instruct'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 23]       # Qwen2-0.5B has 24 layers


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
    cent_full = {c: Wn[i].mean(0) for c, i in topics.items()}

    pid = tok('Once upon a time', add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
    hidden = hid.hidden_states                    # [0] = embed, [l+1] = layer l

    def angle_to(u, mat):
        return np.degrees(np.arccos(np.clip(np.abs(mat @ u), 0, 1)))

    print(f"[model] {MODEL}  ({len(hidden)-1} hidden spheres)\n")
    print(f"{'layer':>5} | {'equator':>7} | {'ring order':<42} | {'ring-dist':>9} | acc")
    print("-" * 90)
    for l in LAYERS:
        h = hidden[l + 1][0, 0].cpu().float().numpy()       # position-0 (BOS axis)
        u = h / np.linalg.norm(h)
        eq = np.median(angle_to(u, Wn))
        # L2: ring de-poled by THIS sphere's axis
        def depole(v):
            v = v - (v @ u) * u
            n = np.linalg.norm(v)
            return v / n if n > 1e-9 else v
        cent = {c: depole(Wn[i].mean(0)) for c, i in topics.items()}
        C = np.stack([cent[c] for c in names])
        C0 = C - C.mean(0)
        _, Vp = np.linalg.eigh(C0.T @ C0)
        P2 = C0 @ Vp[:, -2:]
        az = (np.degrees(np.arctan2(P2[:, 1], P2[:, 0])) + 360) % 360
        order = [names[i] for i in np.argsort(az)]
        # ring distances: mean pairwise + min
        pair = []
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                d = abs(az[a] - az[b]) % 360
                pair.append(min(d, 360 - d))
        ring_mean = np.mean(pair)
        ring_min = np.min(pair)
        # L3: accessibility from this sphere's last state (17 deg, final head)
        hs = hidden[l + 1][0, -1].cpu().float().numpy()
        hs = hs / np.linalg.norm(hs)
        acc = 0
        for c in names:
            fam = rows[c]
            best = fam[int(np.argmax(fam @ hs))]
            v = M.rotate_toward(hs, M.tangent_direction(hs, best), math.radians(17))
            L = v @ Wn.T
            acc += int(int(L.argmax()) in topics[c])
        order_s = " ".join(o[:3] for o in order)
        print(f"{l:5d} | {eq:7.1f} | {order_s:<42} | {ring_mean:5.1f}({ring_min:4.1f}) | {acc}/6")

    # equator across ALL layers (denser), to see where it stops being ~90
    eqs = []
    for l in range(len(hidden) - 1):
        h = hidden[l + 1][0, 0].cpu().float().numpy()
        u = h / np.linalg.norm(h)
        eqs.append(np.median(angle_to(u, Wn)))
    print(f"\nequator by layer (dense): " +
          " ".join(f"{i}:{e:.0f}" for i, e in enumerate(eqs)))
    dev = np.std(eqs[1:]) if len(eqs) > 1 else 0.0
    print(f"  mean {np.mean(eqs):.1f} deg, std {dev:.1f}  "
          f"({'EQUATORIAL THROUGHOUT the tower' if np.mean(eqs) > 80 else 'not'})")
    print(f"\n[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()