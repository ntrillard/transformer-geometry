#!/usr/bin/env python3
"""Full topic-transition map via chord walk on a single model (~10 s, Qwen).

M1  TRANSITION MATRIX: for every ordered class pair (30 pairs), run the chord
    walk (3 settle steps + up to 12x4 deg re-aimed steps toward the target's
    best member).  Record reached?, crossing step, and topic labels along the
    path.
M2  INTERMEDIATES: do paths trace the ring (pass through intermediate topics)
    or jump directly from start to target?
M3  DISTANCE LAW: does the crossing step correlate with azimuthal ring
    distance more than with raw equatorial distance?

Run:  python eval_topic_map.py [model]     (default Qwen2-0.5B, ~10 s)
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
MAX_STEPS = 12
STEP_DEG = 4


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

    def logit_top1(h):
        t = int((h @ Wn.T).argmax())
        return t, tok.decode([t], skip_special_tokens=True)

    def topic_of(h):
        return min(names, key=lambda c: np.degrees(
            np.arccos(np.clip(depole(Wn[int((h @ Wn.T).argmax())]) @ cent[c], -1, 1))))

    # ring azimuths (same PCA as T1)
    C = np.stack([cent[c] for c in names])
    C0 = C - C.mean(0)
    _, Vp = np.linalg.eigh(C0.T @ C0)
    P2 = C0 @ Vp[:, -2:]
    az = {c: (np.degrees(np.arctan2(*P2[i][::-1])) + 360) % 360
          for i, c in enumerate(names)}

    def ring_dist(a, b):
        d = abs(az[a] - az[b]) % 360
        return min(d, 360 - d)

    h_base = (hs / hs.norm()).cpu().float().numpy()

    rows = {c: Wn[i] for c, i in topics.items()}

    print(f"[model] {MODEL}")
    print("\n[M1] chord-walk transition matrix (crossing step; '-' = not reached in 12 steps)")
    header = "      " + "".join(f"{c:>8s}" for c in names)
    print(header)
    mat = {}
    unreached = 0
    for si, s in enumerate(names):
        line = f"{s:>6s}"
        for t in names:
            if t == s:
                line += f"{'-':>8s}"
                continue
            ht = h_base.copy()
            # settle into start topic
            for _ in range(3):
                best = rows[s][int(np.argmax(rows[s] @ ht))]
                ht = M.rotate_toward(ht, M.tangent_direction(ht, best), math.radians(STEP_DEG))
            crossed = None
            for k in range(1, MAX_STEPS + 1):
                best = rows[t][int(np.argmax(rows[t] @ ht))]
                ht = M.rotate_toward(ht, M.tangent_direction(ht, best), math.radians(STEP_DEG))
                if topic_of(ht) == t:
                    crossed = k
                    break
            mat[(s, t)] = crossed
            if crossed is None:
                unreached += 1
            line += f"{crossed if crossed else '-':>8}"
        print(line)
    print(f"    unreached: {unreached}/30 ordered pairs")

    # M2: intermediates along trajectories
    print("\n[M2] path topology (topics visited on the way to target):")
    n_int = 0
    for si, s in enumerate(names):
        for t in names:
            if t == s:
                continue
            ht = h_base.copy()
            for _ in range(3):
                best = rows[s][int(np.argmax(rows[s] @ ht))]
                ht = M.rotate_toward(ht, M.tangent_direction(ht, best), math.radians(STEP_DEG))
            visited = set()
            for k in range(1, MAX_STEPS + 1):
                best = rows[t][int(np.argmax(rows[t] @ ht))]
                ht = M.rotate_toward(ht, M.tangent_direction(ht, best), math.radians(STEP_DEG))
                visited.add(topic_of(ht))
            interm = visited - {s, t}
            if interm:
                n_int += 1
                if len(visited.union({s, t})) <= 4:
                    print(f"    {s:>7s} -> {t:<7s} visits {sorted(visited):}")
    print(f"    paths with an intermediate topic: {n_int}/30")

    # M3: distance law
    print("\n[M3] crossing step vs distance:")
    steps, rd, eq = [], [], []
    for (s, t), crossed in mat.items():
        if crossed is None:
            continue
        steps.append(crossed)
        eq.append(np.degrees(np.arccos(np.clip(cent[s] @ cent[t], -1, 1))))
        rd.append(ring_dist(s, t))
    if len(steps) >= 4:
        crd = np.corrcoef(steps, rd)[0, 1]
        ceq = np.corrcoef(steps, eq)[0, 1]
        print(f"    corr(steps, az-ring dist)   = {crd:+.3f}")
        print(f"    corr(steps, equatorial dist)= {ceq:+.3f}")
        print(f"    mean crossing step: {np.mean(steps):.1f}  (adjacent-pair mean "
              f"{np.mean([s for (a,b),s in mat.items() if ring_dist(a,b) < 60 and s]):.1f}, "
              f"far-pair mean {np.mean([s for (a,b),s in mat.items() if ring_dist(a,b) >= 60 and s]):.1f})")
    print(f"\n[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()