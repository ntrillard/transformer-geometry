#!/usr/bin/env python3
"""eval_ring_walk_gen.py — FAST: steer along the topic ring, watch the model
narrate its own semantic journey.

From the base state, rotate in the ring's azimuth-increasing direction
(the in-plane tangent ⊥ to the base's ring-plane radius). At each step the
top-1 token is read from the full vocab (GPU) and classified to a topic.
If the ring is real geography, the visited tokens' azimuths advance
monotonically and the decoded journey traces a coherent topic tour.

Run: timeout 90 python3 -u eval_ring_walk_gen.py Qwen/Qwen2-0.5B-Instruct
     timeout 90 python3 -u eval_ring_walk_gen.py google/gemma-3-1b-it
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2-0.5B-Instruct'
PROMPT = sys.argv[2] if len(sys.argv) > 2 else 'Once upon a time'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
STEPS = 36
STEP_DEG = 2.0


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

    pid = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
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

    cents = np.stack([depole(Wn[i]) for i in range(N)])
    C0 = cents - cents.mean(0)
    _, Vp = np.linalg.eigh(C0.T @ C0)
    B = Vp[:, -2:]                    # (dim, 2) ring-plane basis
    P2 = C0 @ B                       # word azimuths in plane

    def az_of(v):                     # azimuth of a de-poled unit vector
        return np.degrees(np.arctan2(v @ B[:, 1], v @ B[:, 0])) % 360

    azw = np.array([az_of(c) for c in cents])

    h_base = (hs / hs.norm()).cpu().float().numpy()
    hb_p = depole(h_base)             # base de-poled direction
    r = hb_p - C0.mean(0)             # base radius in full space
    rp = r @ B                        # in-plane radius, azimuth angle
    az0 = np.degrees(np.arctan2(rp[1], rp[0])) % 360
    # in-plane tangent that INCREASES azimuth (derivative of atan2(y,x))
    tang = np.array([-rp[1], rp[0]])
    tang = tang / np.linalg.norm(tang)          # azimuth-increasing dir in plane
    # lift to hidden space (already ⊥ u by construction of plane span)
    tau_full = B @ tang
    tau_full = tau_full / np.linalg.norm(tau_full)

    def top1(h):
        with torch.no_grad():
            t = int((torch.as_tensor(h, dtype=torch.float32, device=DEV)
                     @ lm_w.float().T).argmax().item())
        return t

    def topic_of_tok(t):
        if t in idxs:
            return cls_of[int(np.argmax(idxs == t))]
        return '?'

    print(f"[{MODEL}] prompt={PROMPT!r}  base az={az0:.0f}deg")
    print(f"{'step':>4} {'az':>6} {'tok-az':>6} {'token':>16} {'topic':>7}")
    prev_az = None
    monotone = []
    az_toks = []
    h = h_base
    for k in range(STEPS + 1):
        t = top1(h)
        tok_s = tok.decode([t]).strip()
        ta = None
        if t in idxs:
            ta = azw[int(np.argmax(idxs == t))]
            az_toks.append(ta)
        if k > 0 and ta is not None and prev_az is not None:
            monotone.append(((ta - prev_az) % 360) < 180)
        if ta is not None:
            prev_az = ta
        print(f"{k:>4} {(az0 + k * STEP_DEG) % 360:>6.0f} "
              f"{(ta if ta is not None else float('nan')):>6.0f} "
              f"{tok_s[:16]:>16} {topic_of_tok(t):>7}")
        h = M.rotate_toward(h, tau_full, math.radians(STEP_DEG))
    if monotone:
        print(f"\n  azimuth-advance monotone: {np.mean(monotone):.2f} "
              f"(1 = tokens always advance along ring)")
        if len(az_toks) >= 2:
            print(f"  visited azimuth range: {np.min(az_toks):.0f}-{np.max(az_toks):.0f} deg")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()