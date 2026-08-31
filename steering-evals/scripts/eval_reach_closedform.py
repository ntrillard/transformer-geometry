"""eval_reach_closedform.py — FAST: is rank-1 reach fully explained by alpha*?

For targets x contexts at the final layer: binary-search the MINIMAL rotation
angle where the target ranks #1 in the FULL vocab (measured). Compare to the
closed-form prediction alpha* = atan2(-A, B) vs the native argmax.

Prediction: alpha* (vs native) == measured minimal rank-1 angle (vs all),
so the paper's reach-vs-budget curves are exactly the alpha* CDF.

Run: python3 eval_reach_closedform.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
     python3 eval_reach_closedform.py --model google/gemma-3-1b-it --tag gemma
"""
import argparse
import math

import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = ["For dinner I made", "I went to the store and bought"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2-0.5B-Instruct')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--targets', type=int, default=60)
    a = ap.parse_args()

    model, tok = M.load_model(a.model, dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    V = model.config.vocab_size
    W = W[:V]
    NL = model.config.num_hidden_layers

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    targets = list(word2id.values())[:a.targets]

    # collect states per context
    states = {}
    for pr in PROMPTS:
        pid = tok(pr, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        with torch.no_grad():
            hid = model(pid, output_hidden_states=True)
        states[pr] = hid.hidden_states[NL][0, -1].float().cpu().numpy()

    def alpha_star(h, ti, native):
        hn = h / np.linalg.norm(h)
        t = W[ti] - (W[ti] @ hn) * hn
        nt = np.linalg.norm(t)
        tau = t / (nt + 1e-12)
        A = float(hn @ (W[ti] - W[native]))
        B = float(tau @ (W[ti] - W[native]))
        if B <= 1e-12:
            return float('nan')
        return math.atan2(-A, B)

    def measured_minangle(h, ti, lo=0.0, hi=1.2, iters=26):
        """smallest rotation angle where target ranks #1 vs the FULL vocab."""
        hn = h / np.linalg.norm(h)
        t = W[ti] - (W[ti] @ hn) * hn
        nt = np.linalg.norm(t)
        tau = t / (nt + 1e-12)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            v = M.rotate_toward(hn, tau, mid)
            Lm = v @ W.T
            if int(Lm.argmax()) == ti:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    preds, meas = [], []
    for pr, h in states.items():
        L0 = h @ W.T
        native = int(L0.argmax())
        for ti in targets:
            p = alpha_star(h, ti, native)
            if math.isnan(p):
                continue
            m = measured_minangle(h, ti)
            preds.append(p)
            meas.append(m)
    preds = np.array(preds)
    meas = np.array(meas)

    print(f"== [{a.tag}] alpha* vs measured minimal rank-1 angle "
          f"({len(preds)} targets x {len(states)} contexts, final layer) ==")
    print(f"  corr(alpha*, measured): {np.corrcoef(preds, meas)[0, 1]:.4f}")
    print(f"  median |alpha* - meas|: {np.median(np.abs(preds - meas)):.4f} rad "
          f"({np.median(np.abs(preds - meas)) * 57.3:.2f} deg)")
    print(f"  {'theta':>6} {'reach(alpha*<=t)':>16} {'reach(measured)':>16}")
    for deg in (5, 10, 17, 25, 45, 60):
        t = math.radians(deg)
        print(f"  {deg:>5}d {np.mean(preds <= t):>14.3f} {np.mean(meas <= t):>16.3f}")


if __name__ == "__main__":
    main()