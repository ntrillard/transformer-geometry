#!/usr/bin/env python3
"""eval_ring_layers.py — FAST: the hidden state's journey around the topic
ring, layer by layer (logit-lens readout).

Final-layer topic ring (B basis) is computed once. Then for EVERY layer's
post-state we report:
  az      the state's azimuth on the ring plane
  ringE   fraction of the state's energy in the ring plane  (||h.B||/||h||)
  topic   nearest topic centroid (max affinity, name + affinity value)
  top1    the model's own top-1 token via logit-lens (h_l @ W^T, full vocab)

If the ring is real semantic topology, deeper layers should (a) increase
ring energy, (b) walk the azimuth (possibly circling), (c) converge on the
topic that ends up being generated.

Run: timeout 60 python3 -u eval_ring_layers.py Qwen/Qwen2-0.5B-Instruct
     timeout 90 python3 -u eval_ring_layers.py google/gemma-3-1b-it
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


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    NL = model.config.num_hidden_layers

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
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)

    # final-layer ring plane (fixed coordinate system for all layers)
    u = hid.hidden_states[NL][0, 0].float().cpu().numpy()
    u = u / np.linalg.norm(u)

    def depole(v):
        v = v - (v @ u) * u
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    cents = np.stack([depole(Wn[i]) for i in range(N)])
    C0 = cents - cents.mean(0)
    _, Vp = np.linalg.eigh(C0.T @ C0)
    B = Vp[:, -2:]                       # (dim, 2)

    def az_of(v):
        return np.degrees(np.arctan2(v @ B[:, 1], v @ B[:, 0])) % 360

    azw = np.array([az_of(c) for c in cents])

    # collect per-layer states (post-layer k, k=1..NL)
    hs = [hid.hidden_states[k][0, -1].float().cpu().numpy() for k in range(1, NL + 1)]

    # one batched logit-lens full-vocab argmax (GPU)
    stack = torch.stack([torch.as_tensor(h, dtype=torch.float32, device=DEV)
                         for h in hs])
    with torch.no_grad():
        top1s = (stack @ lm_w.float().T).argmax(dim=1).cpu().numpy()

    print(f"[{MODEL}] prompt={PROMPT!r}  layers 1..{NL}  "
          f"native={tok.decode([int(top1s[-1])]).strip()!r}")
    print(f"{'layer':>5} {'az':>6} {'ringE':>6} {'topic':>8} {'aff':>6}  top1")
    azs = []
    topics_seen = []
    for k in range(NL):
        h = hs[k]
        hn = h / np.linalg.norm(h)
        az = az_of(h)
        azs.append(az)
        ren = float(np.linalg.norm(h @ B)) / float(np.linalg.norm(h))
        affs = hn @ cents.T
        ti = int(affs.argmax())
        topics_seen.append(cls_of[ti])
        t1 = tok.decode([int(top1s[k])]).strip()
        print(f"{k + 1:>5} {az:>6.0f} {ren:>6.2f} {cls_of[ti]:>8} "
              f"{float(affs[ti]):>6.2f}  {t1[:16]!r}")

    # summary
    azs = np.array(azs)
    d = np.diff(azs)
    wrapped = int((np.abs(d) > 180).sum())
    tot = float(np.sum((d % 360 + 180) % 360 - 180))
    print(f"\n  azimuth: total signed advance {tot:+.0f} deg across {NL} layers, "
          f"{wrapped} wrap(s)")
    final_topic = topics_seen[-1]
    first_final = next((i + 1 for i, t in enumerate(topics_seen)
                        if t == final_topic), NL)
    print(f"  final topic {final_topic!r} reached first at layer {first_final}")
    # ring energy trend
    rens = np.array([float(np.linalg.norm(hs[k] @ B)) / float(np.linalg.norm(hs[k]))
                     for k in range(NL)])
    print(f"  ring energy: first{rens[0]:.2f} mid{rens[NL // 2]:.2f} "
          f"final{rens[-1]:.2f}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()