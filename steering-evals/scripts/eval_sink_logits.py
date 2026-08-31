#!/usr/bin/env python3
"""eval_sink_logits.py — FAST: does the absorbing basin manifest in LOGITS?

From each class's planted state (closed-form 2*alpha*), read the mean logit
on NUMBER tokens vs FOOD tokens (and the lift vs the unsteered base state).
If Gemma's number-sink is a real semantic pull, numbers should be preferred
(logit gap > 0) from EVERY basin, while Qwen should show no such systematic
bias and instead prefer its own scattered attractors.

Run: timeout 60 python3 -u eval_sink_logits.py Qwen/Qwen2-0.5B-Instruct
     timeout 90 python3 -u eval_sink_logits.py google/gemma-3-1b-it
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


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    topics = {cls: np.array([word2id[w] for w in words if w in word2id])
              for cls, words in CLASSES.items()
              if sum(1 for w in words if w in word2id) >= 6}

    # probe token sets
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five',
               'six', 'seven', 'eight', 'nine'] + list('0123456789 ')
    num_ids, food_ids = [], []
    for w in numbers:
        ids = tok(' ' + w.strip(), add_special_tokens=False).input_ids
        if len(ids) == 1:
            num_ids.append(ids[0])
    for w in CLASSES['food']:
        if w in word2id:
            food_ids.append(word2id[w])
    pid_ids = list(topics['city'])[:6]
    num_ids = np.array(num_ids)
    food_ids = np.array(food_ids)
    pid_ids = np.array(pid_ids)
    probe = np.unique(np.concatenate([num_ids, food_ids, pid_ids])).astype(np.int64)
    probe_t = torch.as_tensor(probe, dtype=torch.long, device=DEV)
    probe_cpu = lm_w[probe_t].detach().float().cpu().numpy()

    pid = tok('Once upon a time', add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    li = model.config.num_hidden_layers - 1
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
        hs = hid.hidden_states[li + 1][0, -1]
    h_base = (hs / hs.norm()).cpu().float().numpy()
    native = int((hs.float() @ lm_w.float().T).argmax().item())
    Wn_nat = lm_w[native].detach().float().cpu().numpy()
    Wn_nat = Wn_nat / np.linalg.norm(Wn_nat)

    rowmap = {int(t): j for j, t in enumerate(probe)}
    ni = np.array([rowmap[t] for t in num_ids if t in rowmap])
    fi = np.array([rowmap[t] for t in food_ids if t in rowmap])

    def lift_logits(h, subtitle):
        L = h @ probe_cpu.T
        lnum = L[ni].mean()
        lfood = L[fi].mean()
        return lnum - lfood, L

    # baseline: base state
    gap0, L0 = lift_logits(h_base, 'base')
    print(f"[{MODEL}] native={tok.decode([native]).strip()!r}")
    print(f"  mean logit gap (numbers - food): base state {gap0:+.3f}")
    print(f"  {'basin':>8} {'num-food gap':>12} {'numbers lift':>12} "
          f"{'food lift':>10}")
    for cls in topics:
        rows = lm_w[topics[cls]].detach().float().cpu().numpy()
        rows = rows * (1.0 / np.sqrt(np.einsum('ij,ij->i', rows, rows)[:, None] + 1e-12))
        best = rows[int(np.argmax(rows @ h_base))]
        tau = M.tangent_direction(h_base, best)
        A_ = float(h_base @ (Wn_nat - best))
        B_ = float(tau @ (best - Wn_nat))
        alpha = math.atan2(A_, B_) if B_ > 1e-12 else 0.3
        alpha = min(2 * alpha + 0.02, 0.5)
        hc = M.rotate_toward(h_base, tau, alpha)
        gap, L = lift_logits(hc, cls)
        L0n = L0[ni].mean(); L0f = L0[fi].mean()
        print(f"  {cls:>8} {gap:>+12.3f} {L[ni].mean() - L0n:>+12.3f} "
              f"{L[fi].mean() - L0f:>+10.3f}")

    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()