#!/usr/bin/env python3
"""eval_collapse_map.py — BIG LEAP: the information-collapse map over a
MULTI-TOKEN GENERATION (single model, <=10s).

Captures each target's INFORMATION TRAJECTORY through the full stack over
3 generated tokens: delta_t(l) = <v_l, W_t> - <v_l, W_native>, the
target's 'population' vs the native attractor, layer by layer. Shows when
a direction is born, peaks, is collapsed by the native attractor BEFORE
the readout. Then META-LEARN: does the collapse trajectory predict final
steerability (alpha*@final) across registers? Held-one-prompt-out ridge.

Run: timeout 60 python3 -u eval_collapse_map.py  (GEMMA-3-1B, ~12 fwd)
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = [
    'For dinner I made',
    'I went to the store and bought',
    'The recipe calls for',
    'In my kitchen I have',
    'There once was a chicken',
    'My favorite meal is',
]
TARGETS = ('chicken pizza pasta bread steak paris tokyo london cheese '
           'butter soup salad beef cake rome ocean river apple milk wine').split()
NTOK = 3


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    NL = model.config.num_hidden_layers

    tid_l = {}
    for w in TARGETS:
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            tid_l[w] = int(ids[0])
    tids = list(tid_l.values())
    tnames = list(tid_l.keys())
    Wt = lm_w[tids].detach().float().cpu().numpy().astype(np.float32)
    T = len(tids)

    feat_rows = []
    for pidx, PROMPT in enumerate(PROMPTS):
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        full_deltas = []   # per generated token: (27, T)
        Wn = None
        for _ in range(NTOK):
            caps = {}

            def mk(li):
                def h(m, i, o):
                    caps[li] = o[0, -1, :].float()
                return h

            hooks = [model.model.layers[li].register_forward_hook(mk(li))
                     for li in range(NL)]
            hooks.append(model.model.norm.register_forward_hook(mk('f')))
            with torch.no_grad():
                L0 = model(ids).logits[0, -1].float()
            for h in hooks:
                h.remove()
            native = int(L0.argmax())
            Wn = lm_w[native].detach().float().cpu().numpy().astype(np.float32)
            V = np.stack([caps[li].cpu().numpy() for li in range(NL)] +
                         [caps['f'].cpu().numpy()])          # (27, d)
            lgt_ln = V @ Wn                                   # (27,)
            delta = (V @ Wt.T) - lgt_ln[:, None]              # (27, T)
            full_deltas.append(delta)
            nxt = int(L0.argmax())
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)],
                            dim=1)

        delta_agg = np.stack(full_deltas).mean(0)             # (27, T)
        for t in range(T):
            curve = delta_agg[:, t]
            peak_li = int(np.argmax(curve))
            a_curve = np.zeros(NL + 1)
            for li in range(NL + 1):
                v = V[li]
                vn = v / (np.linalg.norm(v) + 1e-12)
                A = float(vn @ (Wt[t] - Wn))
                tau = Wt[t] - (vn @ Wt[t]) * vn
                B = float(tau @ (Wt[t] - Wn)) / (np.linalg.norm(tau) + 1e-12)
                a_curve[li] = math.atan2(-A, B)
            nf = float(np.linalg.norm(V[-1]))
            feat_rows.append(dict(
                prompt=pidx, target=tnames[t],
                peak_delta=float(delta_agg[:, t].max()),
                peak_li=peak_li,
                birth_li=int(np.argmax(curve > 0)),
                final_delta=float(curve[-1]),
                early_delta=float(curve[4]),
                a_final=float(a_curve[-1]),
                a_min=float(np.min(np.abs(a_curve))),
                a_at_peak=float(a_curve[peak_li]),
                n_final=nf))

        md = delta_agg.mean(1)
        print(f"P{pidx} {PROMPT!r:38} native={tok.decode([native])!r:8} "
              f"mean_delta L[2,6,10,14,18,22,f]= "
              f"[{md[2]:+.1f},{md[6]:+.1f},{md[10]:+.1f},{md[14]:+.1f},"
              f"{md[18]:+.1f},{md[22]:+.1f},{md[-1]:+.1f}]", flush=True)

    feats = ['peak_delta', 'peak_li', 'birth_li', 'final_delta',
             'early_delta', 'a_min', 'a_at_peak', 'n_final']
    X = np.array([[r[f] for f in feats] for r in feat_rows])
    y = np.array([r['a_final'] for r in feat_rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    NP = len(PROMPTS)
    print("\n  META: held-out-prompt ridge, predict a_final from curve:")
    preds = np.zeros_like(y)
    for pi in range(NP):
        tr = np.array([i for i, r in enumerate(feat_rows)
                       if r['prompt'] != pi])
        te = np.array([i for i, r in enumerate(feat_rows)
                       if r['prompt'] == pi])
        A = Z[tr].T @ Z[tr] + np.eye(Z.shape[1]) * 1.0
        w = np.linalg.solve(A, Z[tr].T @ y[tr])
        preds[te] = Z[te] @ w
    r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
    mae = np.abs(y - preds).mean()
    bl = np.full_like(y, np.median(y))
    print(f"    {'ridge-loo-prompt':>14} R2={r2:>6.3f} MAE={mae:>6.4f}")
    print(f"    {'baseline-median':>14} R2="
          f"{1 - np.sum((y - bl) ** 2) / np.sum((y - y.mean()) ** 2):>6.3f} "
          f"MAE={np.abs(y - bl).mean():>6.4f}")
    print("\n  curve feature vs a_final ALONE (corr):")
    for f in feats:
        cc = np.corrcoef(X[:, feats.index(f)], y)[0, 1]
        print(f"    {f:>12}: corr={cc:+.3f}", flush=True)
    ff = feats.index('peak_delta')
    cc = np.corrcoef(X[:, ff], y)[0, 1]
    foods = [i for i, tn in enumerate(tnames) if tn in
             ('chicken', 'pizza', 'pasta', 'bread', 'steak', 'cheese',
              'butter', 'soup', 'salad', 'beef', 'cake', 'apple', 'milk',
              'wine')]
    cits = [i for i, tn in enumerate(tnames) if tn in
            ('paris', 'tokyo', 'london', 'rome', 'ocean', 'river')]
    fpk = np.array([feat_rows[i]['peak_delta'] for i in range(len(feat_rows))
                    if i % T in foods])
    cpk = np.array([feat_rows[i]['peak_delta'] for i in range(len(feat_rows))
                    if i % T in cits])
    print(f"  corr(a_final, peak_delta) = {cc:+.3f}")
    print(f"  peak_delta: food={fpk.mean():+.2f} city={cpk.mean():+.2f}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()