#!/usr/bin/env python3
"""eval_sink_depth.py — FAST: where does the readout decision form in depth?

For every layer k: take the post-layer state h_k, apply the FINAL norm
(scale into readout space), and read the LM head argmax (through-norm
logit lens). Report the first layer whose through-norm argmax ALREADY
matches the final native token, and the layer-by-layer topic affinity.

This locates the 'birth depth' of the absorbing decision: Gemma's number
sink ('a'/number), Qwen's narration near-tie (',' vs 'there').

Run: timeout 60 python3 -u eval_sink_depth.py Qwen/Qwen2-0.5B-Instruct
     timeout 90 python3 -u eval_sink_depth.py google/gemma-3-1b-it
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
    dtype = torch.float16 if hasattr(lm_w, 'dtype') and lm_w.dtype == torch.float16 \
        else torch.float32

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

    # batched through-norm states: (NL, 1, dim) -> norm -> (NL, dim) @ head
    states = torch.stack([hid.hidden_states[k][:, -1, :] for k in range(1, NL + 1)])
    with torch.no_grad():
        states_n = model.model.norm(states.to(dtype))
        logits = torch.matmul(states_n.float(), lm_w.float().T)
        top1s = logits.argmax(dim=-1).reshape(-1).cpu().numpy()
    states_cpu = states.float().cpu().numpy()

    native_tok = tok.decode([int(top1s[-1])]).strip()
    print(f"[{MODEL}] native={native_tok!r}")
    print(f"{'layer':>5} {'tn-top1':>16} {'topic':>8} {'aff':>6} {'az':>5}")
    tns = []
    first_same = None
    for k in range(NL):
        top = tok.decode([int(top1s[k])]).strip()
        hn = states_cpu[k][0] / np.linalg.norm(states_cpu[k][0])
        affs = hn @ cents.T
        ti = int(affs.argmax())
        az = az_of(states_cpu[k][0])
        tns.append(top)
        if first_same is None and top == native_tok:
            first_same = k + 1
        print(f"{k + 1:>5} {top[:16]:>16} {cls_of[ti]:>8} "
              f"{float(affs[ti]):>6.2f} {az:>5.0f}")

    # quantiles / summary
    same = 1.0 if first_same else float('nan')
    print(f"\n  final native {native_tok!r} first present at layer "
          f"{first_same if first_same else 'NEVER'}")
    # how stable: fraction of layers >= first_same that STAY on native
    if first_same:
        stayed = np.mean([t == native_tok for t in tns[first_same - 1:]])
        print(f"  stability after birth: {stayed:.2f}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()