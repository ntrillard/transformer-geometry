#!/usr/bin/env python3
"""eval_ring_dist.py — FAST: continuation DISTRIBUTION azimuth along the ring.

From each ring node w_i (context = BASE + w_i): read the top-15 tokens and
their probabilities. Map each onto the ring plane (project its de-poled row
onto the ring PCA basis) and take the probability-weighted mean azimuth.
If the ring is the model's semantic gradient, the distribution centroid
sits near the node's azimuth and ADVANCES forward even though the argmax
is usually a grammar word.

Metrics:
  dist  mean circular |dist-az - node-az|      (semantic locality)
  stay  P(dist within 30 deg)                  (distribution follows ring)
  adv   P(dist ahead of node)                  (ring follow-through)
  sem   fraction of nodes whose TOP CONTENT token is within 60 deg
        (argmax ignored; the most probable NON-function continuation)

Run: timeout 90 python3 -u eval_ring_dist.py Qwen/Qwen2-0.5B-Instruct
     timeout 90 python3 -u eval_ring_dist.py google/gemma-3-1b-it
"""
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2-0.5B-Instruct'
BASE = "Once upon a time"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP = 15


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight

    word2id = {}
    word_cls = {}
    for cls, words in CLASSES.items():
        for w in words:
            ids = tok(' ' + w, add_special_tokens=False).input_ids
            if len(ids) == 1:
                word2id[w] = int(ids[0])
                word_cls[w] = cls
    words = list(word2id)
    idxs = np.array([word2id[w] for w in words])
    N = len(words)

    rows = lm_w[idxs].detach().float().cpu().numpy()
    Wn = rows * (1.0 / np.sqrt(np.einsum('ij,ij->i', rows, rows)[:, None] + 1e-12))

    pid = tok(BASE, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
    li = model.config.num_hidden_layers - 1
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
        h0 = hid.hidden_states[li + 1][0, 0]
    u = (h0 / h0.norm()).cpu().float().numpy()

    def depole(v):
        v = v - (v @ u) * u
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    cents = np.stack([depole(Wn[i]) for i in range(N)])
    C0 = cents - cents.mean(0)
    _, Vp = np.linalg.eigh(C0.T @ C0)
    B = Vp[:, -2:]

    def az_of(v):
        return np.degrees(np.arctan2(v @ B[:, 1], v @ B[:, 0])) % 360

    azw = np.array([az_of(c) for c in cents])

    def cir_mean(azs, ws):
        """probability-weighted circular mean azimuth."""
        x = np.sum(ws * np.cos(np.radians(azs)))
        y = np.sum(ws * np.sin(np.radians(azs)))
        return np.degrees(np.arctan2(y, x)) % 360

    def cdist(a, b):
        return min((a - b) % 360, (b - a) % 360)

    dists, stays, advs, sems = [], [], [], []
    rows_out = []
    for i in range(N):
        ctx = tok(BASE + ' ' + words[i], add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        with torch.no_grad():
            logits = model(ctx).logits[0, -1].float()
            probs = torch.softmax(logits, dim=0)
            topv, topi = torch.topk(probs, TOP)
        topi_np = topi.cpu().numpy()
        topv_np = topv.cpu().numpy()
        # de-pole + project top tokens onto ring basis
        prog = lm_w[topi].detach().float().cpu().numpy()
        # unit rows
        prog = prog * (1.0 / np.sqrt(np.einsum('ij,ij->i', prog, prog)[:, None] + 1e-12))
        azs = np.array([az_of(depole(r)) for r in prog])
        daz = cir_mean(azs, topv_np)
        d = cdist(daz, azw[i])
        signed = ((daz - azw[i]) % 360 + 180) % 360 - 180
        dists.append(d)
        stays.append(1.0 if d <= 30 else 0.0)
        advs.append(1.0 if signed > 0 else 0.0)
        # semantic: most probable continuation among the 60 ring words
        inring = [j for j, t in enumerate(topi_np) if t in idxs]
        sem_ok = 0.0
        if inring:
            jj = inring[0]
            ta = azw[int(np.argmax(idxs == topi_np[jj]))]
            sem_ok = 1.0 if cdist(ta, azw[i]) <= 60 else 0.0
        sems.append(sem_ok)
        rows_out.append((words[i], azw[i], daz, topi_np, topv_np))

    print(f"[{MODEL}] nodes={N}")
    print(f"  dist |weighted-centroid - node|: {np.mean(dists):.0f} deg")
    print(f"  stay (<=30 deg):   {np.mean(stays):.2f}")
    print(f"  advance (forward): {np.mean(advs):.2f}")
    print(f"  semantic top-cont (<=60 deg): {np.mean(sems):.2f}")
    for i in range(N):
        w, a, daz, topi_np, topv_np = rows_out[i]
        mark = 'o' if cdist(daz, a) <= 30 else ' '
        conts = ', '.join(tok.decode([t]).strip() for t in topi_np[:4])
        print(f"    {mark} {w:>10} (az {a:5.0f}) -> dist-az {daz:5.0f}  "
              f"top: {conts}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()