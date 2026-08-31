#!/usr/bin/env python3
"""eval_final_block_ablate.py — FAST: causal test that the final block makes
the ring-topology decision.

FULL:      normal forward -> top-1, topic, azimuth.
BYPASS:    replace the final block with identity: norm(pre-final state) fed
           straight to the LM head -> top-1, topic, azimuth.

Prediction (from eval_ring_layers): FULL lands on the jumped azimuth
(Qwen animal 246, Gemma number 81). BYPASS should revert toward the
mid-stack limbo azimuth (~0-30 Qwen, ~240-260 Gemma) and change the
dominant topic accordingly.

Run: timeout 60 python3 -u eval_final_block_ablate.py Qwen/Qwen2-0.5B-Instruct
     timeout 90 python3 -u eval_final_block_ablate.py google/gemma-3-1b-it
"""
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

    u = hid.hidden_states[NL][0, 0].float().cpu().numpy()
    u = u / np.linalg.norm(u)

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

    def report(tag, h_np):
        hn = h_np / np.linalg.norm(h_np)
        with torch.no_grad():
            t1 = int((torch.as_tensor(hn, dtype=torch.float32, device=DEV)
                      @ lm_w.float().T).argmax().item())
        affs = hn @ cents.T
        ti = int(affs.argmax())
        az = az_of(h_np)
        print(f"  {tag:>8}  top1={tok.decode([t1]).strip()!r:12}  "
              f"topic={cls_of[ti]:>7} (aff {float(affs[ti]):.2f})  az={az:5.0f}")

    # FULL
    h_full = hid.hidden_states[NL][0, -1].float().cpu().numpy()
    report('full', h_full)

    # BYPASS: pre-final state through the final norm
    h_pre = hid.hidden_states[NL - 1][0, -1].float()
    with torch.no_grad():
        h_norm = model.model.norm(h_pre[None])[0].cpu().numpy()
    report('bypass', h_norm)

    print(f"[{MODEL}] prompt={PROMPT!r}  (bypass = identity final block)")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()