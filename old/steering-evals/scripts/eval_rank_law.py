"""eval_rank_law.py — FAST: can steering outcomes be predicted in closed form?

First-order lift model: under a rotation toward tangent tau, the first-order
change to every logit is  lift[t] = ||h|| * (tau . Wn[t]). Predicted logits at
steering alpha:  Lhat(alpha) = L0 + alpha*lift. PREDICTION: the measured rank
of any target (from the exact rotate_toward) equals its rank under Lhat, for
all alpha and all targets.

Also prints the junk-FILTERED flip schedule (skipping non-printable/replacement
tokens) to show clean monotone lambda -> rank behavior.

Run: python3 eval_rank_law.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
     python3 eval_rank_law.py --model google/gemma-3-1b-it --tag gemma
"""
import argparse

import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from multiverse_lab import chord_summary

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ALPHAS = (0.02, 0.04, 0.06, 0.1, 0.15, 0.2, 0.3)


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
    normh = float(np.linalg.norm(model.model.layers[NL - 1].input_layernorm.weight
                                 .detach().cpu().float().numpy()))

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    food = np.array([word2id[w] for w in CLASSES['food'] if w in word2id][:5])
    C, _, _ = chord_summary(food, Wn)

    pid = tok(a.prompt, add_special_tokens=False,
              return_tensors='pt').input_ids.to(model.device)
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
    h = hid.hidden_states[NL][0, -1].float().cpu().numpy()   # post-norm readout
    L0 = h @ W.T
    hn = h / np.linalg.norm(h)
    tau = M.tangent_direction(hn, C)
    lift = np.linalg.norm(h) * (tau @ Wn.T)                  # first-order logit lift

    # junk mask: skip replacement/control/whitespace-only/CJK-filler tokens
    def junk(id_):
        t = tok.decode([id_])
        return (not t.strip()) or ('\ufffd' in t) or (len(t) > 3 and not t.isascii())
    junk_ids = np.array([junk(i) for i in range(V)])

    print(f"== [{a.tag}] rank-law: predicted vs measured rank of food notes ==")
    print(f"{'target':>8} {'theta':>6} | " +
          " ".join(f"a{a:<4}" for a in ALPHAS))
    print(f"{'':>8} {'':>6} | " +
          " ".join(f"{'p/m':>6}" for _ in ALPHAS))
    for ti in food:
        nm = tok.decode([ti]).strip()
        theta = np.degrees(np.arccos(np.clip(hn @ Wn[ti], -1, 1)))
        cells = []
        for al in ALPHAS:
            v = M.rotate_toward(hn, tau, al)
            Lm = v @ W.T
            r_meas = int((Lm > Lm[ti]).sum() + 1)
            Lp = L0 + al * lift
            r_pred = int((Lp > Lp[ti]).sum() + 1)
            cells.append(f"{r_pred}/{r_meas}")
        print(f"{nm:>8} {theta:5.1f} | " + " ".join(f"{c:>6}" for c in cells))

    # ---- junk-filtered flip schedule (clean monotonicity) ----
    print(f"\n== [{a.tag}] junk-filtered flip schedule ==")
    native_top = int(L0.argmax())
    print(f"  native top-1: {tok.decode([native_top]).strip()!r} "
          f"(junk={junk(native_top)})")
    print("  top-5 (filtered, native): " + ", ".join(
        tok.decode([t]).strip() for t in np.argsort(-L0)[:20]
        if not junk(int(t)))[:60])
    prev = False
    for al in ALPHAS:
        Lp = L0 + al * lift
        top = np.argsort(-Lp)
        top_clean = next(t for t in top if not junk(int(t)))
        flip = top_clean != (native_top if not junk(native_top)
                             else int(next(t for t in np.argsort(-L0)
                                           if not junk(int(t)))))
        mon = ''
        if prev and not flip:
            mon = '  <- unfix (NaN?)'
        prev = flip
        print(f"  a {al:>4.2f}  clean-argmax {tok.decode([top_clean]).strip()!r:>10}  "
              f"flip={str(flip):>5}{mon}")


if __name__ == "__main__":
    main()