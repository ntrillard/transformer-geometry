#!/usr/bin/env python3
"""eval_ring_next.py — FAST: the model's own next-token edges along the ring.

Context = base prompt + " " + ring word w_i. Read the model's committed
top-1 continuation (greedy, full vocab). The model chooses the next word:
if the ring is its semantic gradient, the continuation azimuth should stay
local to w_i (small deviation) and advance forward, tracing the ring.

Metrics (across the 60 ring nodes):
  dev   mean circular |cont-az - node-az|        (stay local?)
  stay  P(within 30 deg)                         (self-consistent ring)
  adv   P(advancing forward past the node)       (ring follow-through)
  step  mean signed azimuth step (chance ~0, ring gradient -> positive)
  stops the model's continuation words in ring order (its narrated tour)

Run: timeout 90 python3 -u eval_ring_next.py Qwen/Qwen2-0.5B-Instruct
     timeout 90 python3 -u eval_ring_next.py google/gemma-3-1b-it
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
    cls_of = np.array([word_cls[w] for w in words])
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

    def cdist(a, b):
        return min((a - b) % 360, (b - a) % 360)

    devs, stays, advs, steps = [], [], [], []
    row = []
    for i in range(N):
        ctx = tok(BASE + ' ' + words[i], add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        with torch.no_grad():
            t = int(model(ctx).logits[0, -1].argmax().item())
        ta = azw[int(np.argmax(idxs == t))] if t in idxs else None
        d = cdist(ta, azw[i]) if ta is not None else 180.0
        signed = ((ta - azw[i]) % 360 + 180) % 360 - 180 if ta is not None else 0.0
        devs.append(d)
        stays.append(1.0 if d <= 30 else 0.0)
        advs.append(1.0 if signed > 0 else 0.0)
        steps.append(signed)
        row.append((words[i], azw[i], t, ta))

    order = np.argsort(azw)
    print(f"[{MODEL}] nodes={N}")
    print(f"  mean |continuation - node| azimuth {np.mean(devs):.0f} deg")
    print(f"  stay (within 30 deg)   {np.mean(stays):.2f}")
    print(f"  advance (forward step) {np.mean(advs):.2f}")
    print(f"  mean signed step       {np.mean(steps):+.0f} deg")
    print(f"  continuations in ring order (node -> model's next):")
    for i in order:
        w, a, t, ta = row[i]
        mark = 'o' if ta is not None and cdist(ta, a) <= 30 else ' '
        cont = tok.decode([t]).strip() if t is not None else '?'
        print(f"    {mark} {w:>10} (az {a:5.0f}) -> {cont:>10} "
              f"(az {ta if ta is not None else float('nan'):5.0f})")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()