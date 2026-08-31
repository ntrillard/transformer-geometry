"""eval_reach_scale.py — FAST (<10s probe): closed-form alpha* at scale.

Design for speed: all prompts in ONE batched forward; per target the check
is ONE rotation at the predicted alpha* then argmax over the full vocab
(single batched matmul per prompt). This validates: (a) does alpha* rank the
target #1, (b) reach curves, (c) blocker incidence, at T targets x P prompts.

Run: timeout 60 python3 -u eval_reach_scale.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
     timeout 90 python3 -u eval_reach_scale.py --model google/gemma-3-1b-it --tag gemma
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
    T = len(targets)
    tarr = np.array(targets)

    # ONE batched forward for all prompts
    enc = tok(PROMPTS, add_special_tokens=False, padding='longest',
              return_tensors='pt').to(model.device)
    with torch.no_grad():
        hid = model(enc.input_ids, attention_mask=enc.attention_mask,
                   output_hidden_states=True)
    hs = hid.hidden_states[NL].float().cpu().numpy()
    lens = (enc.input_ids != tok.pad_token_id).sum(dim=1).cpu().numpy()
    states = np.stack([hs[i, lens[i] - 1] for i in range(len(PROMPTS))])  # (P, dim)
    Wt = W.T  # (dim, V)
    Wt_t = torch.as_tensor(W.T, dtype=torch.float32, device=DEV)  # GPU argmax

    print(f"== [{a.tag}] alpha* at scale: {T} targets x {len(PROMPTS)} prompts "
          f"(batched, final layer) ==")
    hit = np.zeros((len(PROMPTS), T), dtype=bool)
    astar = np.zeros((len(PROMPTS), T))
    for p, h in enumerate(states):
        native = int((h @ Wt).argmax())
        hn = h / np.linalg.norm(h)
        proj = W[tarr] @ hn
        tvec = W[tarr] - proj[:, None] * hn[None, :]
        nt = np.linalg.norm(tvec, axis=1)
        tau = tvec / (nt[:, None] + 1e-12)
        A = (W[tarr] - W[native]) @ hn
        B = np.sum(tau * (W[tarr] - W[native]), axis=1)
        with np.errstate(invalid='ignore'):
            astar[p] = np.where(B > 1e-12, np.arctan2(-A, B), np.nan)
        valid = ~np.isnan(astar[p])
        rotv = (hn[None, :] * np.cos(astar[p, valid])[:, None] +
                tau[valid] * np.sin(astar[p, valid])[:, None])
        valid = ~np.isnan(astar[p])
        sub = np.flatnonzero(valid)
        rotv_t = (torch.as_tensor(hn, device=DEV)[None, :]
                  * torch.cos(torch.as_tensor(astar[p, sub], device=DEV))[:, None]
                  + torch.as_tensor(tau[sub], device=DEV)
                  * torch.sin(torch.as_tensor(astar[p, sub], device=DEV))[:, None])
        argmax = (rotv_t.float() @ Wt_t).argmax(dim=1).cpu().numpy()  # fast on GPU
        hit[p, sub] = (argmax == tarr[sub])

    fin = hit[~np.isnan(astar)]
    ast = astar[~np.isnan(astar)]
    print(f"  rank-1 at predicted alpha*: {fin.mean():.3f} "
          f"({fin.sum()}/{fin.size})  <- blocker cases fail this check")
    print(f"  nan alpha* (no crossing): {np.isnan(astar).sum()}/{astar.size}")
    print(f"  {'theta':>6} {'reach(alpha*<=t)':>16} {'rank1@alpha*':>13}")
    for deg in (5, 10, 15, 20, 25, 30, 45, 60):
        t = math.radians(deg)
        print(f"  {deg:>5}d {np.mean(ast <= t):>14.3f} "
              f"{np.mean(fin[ast <= t]):>13.3f}")


if __name__ == "__main__":
    main()