"""eval_cross_alpha.py — FAST: verify the closed-form crossing alpha.

For each food note: predict alpha* = margin / (||h|| * tau . d) where d =
(Wn[target] - Wn[native_argmax]) is the decision axis; then measure (binary
search over alpha) the actual alpha where the target beats the native argmax.
Report prediction vs measurement error.

Run: python3 eval_cross_alpha.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
     python3 eval_cross_alpha.py --model google/gemma-3-1b-it --tag gemma
"""
import argparse
import math

import numpy as np
import torch

import steering_geometry_test as M

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from multiverse_lab import chord_summary

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2-0.5B-Instruct')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--prompt', default="For dinner I made")
    a = ap.parse_args()

    model, tok = M.load_model(a.model, dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    V = model.config.vocab_size
    W = W[:V]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    NL = model.config.num_hidden_layers

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    food = np.array([word2id[w] for w in CLASSES['food'] if w in word2id])
    C, _, _ = chord_summary(food, Wn)

    pid = tok(a.prompt, add_special_tokens=False,
              return_tensors='pt').input_ids.to(model.device)
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
    h = hid.hidden_states[NL][0, -1].float().cpu().numpy()
    L0 = h @ W.T
    native = int(L0.argmax())
    hn = h / np.linalg.norm(h)
    tau = M.tangent_direction(hn, C)

    def crossing_measured(ti, lo=0.0, hi=2.0, iters=22):
        """Binary search: smallest alpha where target logit > native logit."""
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            v = M.rotate_toward(hn, tau, mid)
            Lm = v @ W.T
            if Lm[ti] > Lm[native]:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)
    def crossing_pred(ti):
        """Exact trig root: A cos a + B sin a = 0 over RAW rows (row norms matter)."""
        A = float(hn @ (W[ti] - W[native]))
        B = float(tau @ (W[ti] - W[native]))
        return math.atan2(-A, B) if B > 1e-12 else float('nan')

    print(f"== [{a.tag}] crossing alpha*: predicted vs measured (exact trig) ==")
    print(f"{'target':>8} {'native':>8} {'margin':>8} {'pred a*':>9} "
          f"{'meas a*':>9} {'err':>6}")
    errs = []
    for ti in food:
        nm = tok.decode([ti]).strip()
        margin = float(L0[ti] - L0[native])
        pred = crossing_pred(int(ti))
        meas = crossing_measured(int(ti))
        err = abs(pred - meas) / max(meas, 1e-6)
        errs.append(err)
        print(f"{nm:>8} {tok.decode([native]).strip()!r:>8} {margin:8.4f} "
              f"{pred:9.4f} {meas:9.4f} {err:6.2f}")
    print(f"  median rel-err: {np.median(errs):.3f}")


if __name__ == "__main__":
    main()