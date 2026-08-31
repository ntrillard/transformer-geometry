"""build_meta_cache.py — ONE-OFF dataset builder for the meta-budget experiment.

For EVERY (prompt, target) pair at the final layer: record cheap features and
the MEASURED minimal rank-1 angle (binary search vs full vocab, on GPU).
Caches to eval_meta_cache_<tag>.npz.

Features (fixed order):
  0 alpha_star   closed-form crossing root atan2(-A, B)
  1 margin       L0[t] - L0[native]            (negative)
  2 A            hhat.(W_t - W_native)
  3 B            tau.(W_t - W_native)
  4 state norm   ||h||
  5 ||W_native||
  6 ||W_t||
  7 rowcos       (W_t . W_native) / (||W_t|| ||W_native||)
  8 rank_t       (L0 > L0[t]).sum()   target's native rank (blocker pressure)
  9 spread12     L0[native] - L0[2nd]
 10 spread13     L0[native] - L0[3rd]
 11 junk_native  native decodes to a junk/filler token (1/0)
 12 has_cap      capitalized single-token variant exists (1/0)

Label: y = measured minimal rank-1 angle (rad).

Run: python3 build_meta_cache.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
     python3 build_meta_cache.py --model google/gemma-3-1b-it --tag gemma
"""
import argparse
import math

import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = [
    "For dinner I made", "I went to the store and bought", "The recipe calls for",
    "In my kitchen I have", "My favorite meal is", "I was cooking when",
    "The restaurant served", "For lunch I had", "Breakfast today was",
    "At the market I found",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2-0.5B-Instruct')
    ap.add_argument('--tag', default='qwen')
    ap.add_argument('--targets', type=int, default=60)
    a = ap.parse_args()

    model, tok = M.load_model(a.model, dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    V = model.config.vocab_size
    W = W[:V]
    NL = model.config.num_hidden_layers
    Wt_g = torch.as_tensor(W.T, dtype=torch.float32, device=DEV)

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    tarr = np.array(list(word2id.values())[:a.targets])
    T = len(tarr)
    cap_exists = np.zeros(T, dtype=bool)
    for i, w in enumerate(list(word2id.keys())[:a.targets]):
        ids = tok(' ' + w.capitalize(), add_special_tokens=False).input_ids
        cap_exists[i] = (len(ids) == 1)
    print(f"targets {T}, cap-variants {cap_exists.sum()}")

    # batched state extraction
    enc = tok(PROMPTS, add_special_tokens=False, padding='longest',
              return_tensors='pt').to(model.device)
    with torch.no_grad():
        hid = model(enc.input_ids, attention_mask=enc.attention_mask,
                    output_hidden_states=True)
    hs = hid.hidden_states[NL].float().cpu().numpy()
    lens = (enc.input_ids != tok.pad_token_id).sum(dim=1).cpu().numpy()
    P = len(PROMPTS)

    FEAT_N = 13
    X = np.zeros((P * T, FEAT_N), dtype=np.float32)
    y = np.zeros(P * T, dtype=np.float32)
    yd = np.zeros(P * T, dtype=np.float32)
    meta = np.zeros((P * T, 2), dtype=np.int64)  # prompt idx, target idx
    row = 0

    import time
    t0 = time.time()
    for p in range(P):
        h = hs[p, lens[p] - 1] if False else hs[p, lens[p] - 1]
        hn = h / np.linalg.norm(h)
        L0 = h @ W.T
        native = int(L0.argmax())
        srt = np.sort(L0)[::-1]
        spread12 = float(srt[0] - srt[1])
        spread13 = float(srt[0] - srt[2])
        ndec = tok.decode([native])
        junk = 1.0 if (len(ndec.strip()) == 0 or len(ndec) <= 1) else 0.0
        Wn = W[native]
        Wt_rows = W[tarr]

        proj = Wt_rows @ hn
        tvec = Wt_rows - proj[:, None] * hn[None, :]
        nt = np.linalg.norm(tvec, axis=1)
        tau = tvec / (nt[:, None] + 1e-12)
        drow = Wt_rows - Wn[None, :]
        A = drow @ hn
        B = np.sum(tau * drow, axis=1)
        with np.errstate(invalid='ignore'):
            astar = np.where(B > 1e-12, np.arctan2(-A, B), np.nan)
        margins = (L0[tarr] - L0[native])
        rank_t = (L0[None, :] > L0[tarr, None]).sum(axis=1).astype(np.float32)
        rowcos = (Wt_rows * Wn[None, :]).sum(axis=1) / (
            np.linalg.norm(Wt_rows, axis=1) * np.linalg.norm(Wn))

        for i in range(T):
            if math.isnan(astar[i]):
                continue
            # measured minimal rank-1 angle via GPU binary search
            lo, hi = 0.0, 1.2
            ti = int(tarr[i])
            for _ in range(15):
                mid = 0.5 * (lo + hi)
                v = torch.as_tensor(hn * math.cos(mid) + tau[i] * math.sin(mid),
                                    dtype=torch.float32, device=DEV)
                if (v @ Wt_g).argmax().item() == ti:
                    hi = mid
            y[row] = 0.5 * (lo + hi)
            # measured rank-1 angle WITH margin delta (reliable budget)
            lo2, hi2 = 0.0, 1.2
            for _ in range(14):
                mid = 0.5 * (lo2 + hi2)
                v = torch.as_tensor(hn * math.cos(mid) + tau[i] * math.sin(mid),
                                    dtype=torch.float32, device=DEV)
                lg = v @ Wt_g
                topv, topi = lg.topk(2)
                ok = (topi[0].item() == ti and (topv[0] - topv[1]).item() >= 0.1)
                if ok:
                    hi2 = mid
                else:
                    lo2 = mid
            yd[row] = 0.5 * (lo2 + hi2)
            f = [
                float(astar[i]), float(margins[i]), float(A[i]), float(B[i]),
                float(np.linalg.norm(h)), float(np.linalg.norm(Wn)),
                float(np.linalg.norm(Wt_rows[i])), float(rowcos[i]),
                float(rank_t[i]), spread12, spread13, junk,
                1.0 if cap_exists[i] else 0.0,
            ]
            X[row] = f
            meta[row] = (p, i)
            row += 1
        print(f"  prompt {p}: native={ndec.strip()!r:10}")
    X = X[:row]; y = y[:row]; yd = yd[:row]; meta = meta[:row]
    out = f"eval_meta_cache_{a.tag}.npz"
    np.savez(out, X=X, y=y, yd=yd, meta=meta, feats=[
        'alpha_star', 'margin', 'A', 'B', 'h_norm', 'Wn_norm', 'Wt_norm',
        'rowcos', 'rank_t', 'spread12', 'spread13', 'junk_native', 'has_cap'])
    print(f"cached {row} rows -> {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()