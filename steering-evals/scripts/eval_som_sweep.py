#!/usr/bin/env python3
"""Assumption-free tests for the 'semantic topography' + 'chord' claims.

S1  TYING: is lm_head.weight a SEPARATE matrix or TIED to the input embedding?
    (If tied, the head rows ARE the embeddings => semantic structure is
     inherited, no head-specific mechanism needed.)
S2  LATTICE SWEEP: the 16x16 grid was arbitrary.  Sweep sides {4,8,16,24,32}
    (16..1024 neurons) + a 1D ring (256) control.  Metrics: quantization
    error, empty cells, max membership, prototype NN angle vs DATA 1-NN scale.
    Does the collapse persist at every size?
S3  AUTO-CLUSTER CHORDS: replace hand-picked classes with spherical k-means
    clusters (no assumed semantics).  Do label-free families still resolve
    under inversion steering, and does the spread->resolution law hold?
S4  INTERCHANGEABILITY (behavioral mechanism): for random pairs (t, NN(t)) vs
    (t, random), correlate their model logits across ~60 contexts.  If
    geometric neighbors behave alike, geometry => function WITHOUT labels.

Run: python eval_som_sweep.py
"""
import math
import time
from pathlib import Path

import numpy as np
import torch

import steering_geometry_test as M

MODEL = 'Qwen/Qwen2-0.5B-Instruct'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FIT = 20000


def s1_tying(model):
    if getattr(model, 'lm_head', None) is None:
        print("S1  (skip: no lm_head attr)")
        return
    try:
        tied = model.lm_head.weight.data_ptr() == \
               model.model.embed_tokens.weight.data_ptr()
        cfg = getattr(model.config, 'tie_word_embeddings', '?')
        print(f"S1  lm_head.weight is {'TIED to embed' if tied else 'SEPARATE'} "
              f"(tie_word_embeddings={cfg})")
        if tied:
            r = torch.cosine_similarity(model.lm_head.weight[0],
                                        model.model.embed_tokens.weight[0],
                                        dim=0)
            print(f"S1  (first-row cosine {r:.6f} confirms identical memory)")
    except Exception as e:
        print(f"S1  FAIL {type(e).__name__}: {e}")


def s2_sweep(Wn):
    rng = np.random.default_rng(7)
    V, d = Wn.shape
    X = Wn[rng.choice(V, N_FIT, replace=False)].astype(np.float32)
    Xt = torch.as_tensor(X, device=DEV)
    data_nn = None
    # data 1-NN scale (reference)
    idx = rng.choice(V, 400, replace=False)
    Sm = torch.as_tensor(Wn[idx].astype(np.float32), device=DEV)
    c = (Sm @ Sm.T).clamp(-1, 1)
    c.fill_diagonal_(-1)
    data_nn = float(np.degrees(torch.acos(c.max(1).values).cpu().mean()))
    print(f"S2  data 1-NN scale: {data_nn:.1f} deg  (the tiling scale a SOM must beat)")
    sizes = [4, 8, 16, 24, 32]          # side -> 16..1024 neurons
    print("    side   neurons   quant-err(deg)  empty%  max-memb%  proto-NN(deg)  vs-data")
    for side in sizes:
        P = side * side
        gy, gx = np.meshgrid(np.arange(side), np.arange(side))
        G = np.stack([gy.ravel(), gx.ravel()], 1).astype(np.float32)
        Gt = torch.as_tensor(G, device=DEV)
        Pn = torch.as_tensor(Wn[rng.choice(V, P, replace=False)].astype(np.float32),
                             device=DEV)
        for ep in range(6):
            sigma = max(3.0 * (1 - ep / 6) + 0.4, 0.4)
            Xt = Xt[torch.randperm(N_FIT, device=DEV)]
            for b0 in range(0, N_FIT, 512):
                Xb = Xt[b0:b0 + 512]
                C = Xb @ Pn.T
                bmu = C.argmax(1)
                d2 = ((Gt[:, None, :] - Gt[bmu][None, :, :]) ** 2).sum(-1)
                K = torch.exp(-d2 / (2 * sigma * sigma)).T.float()
                num = K.T @ Xb
                den = K.sum(0).clamp_min(1e-6)
                Pn = num / den[:, None]
                Pn = Pn / Pn.norm(dim=1, keepdim=True).clamp_min(1e-9)
        # metrics
        C = Xt @ Pn.T
        q = torch.acos(C.clamp(-1, 1).max(1).values)
        qe = float(np.degrees(q.mean().cpu()))
        bmu = C.argmax(1)
        uniq, counts = torch.unique(bmu, return_counts=True)
        empty = (P - len(uniq)) / P * 100
        maxm = float(counts.max()) / N_FIT * 100
        nn = (Pn @ Pn.T).clamp(-1, 1)
        nn.fill_diagonal_(-1)
        pnn = float(np.degrees(torch.acos(nn.max(1).values).mean().cpu()))
        print(f"    {side:4d}  {P:6d}     {qe:7.1f}      {empty:5.1f}   {maxm:6.1f}   "
              f"{pnn:8.1f}    {'OK' if pnn > data_nn else 'sub-data'}")
    # 1D ring control (256 units)
    P = 256
    ring = np.linspace(0, 2 * math.pi, P, endpoint=False)
    G = np.stack([np.cos(ring), np.sin(ring)], 1).astype(np.float32)
    Gt = torch.as_tensor(G, device=DEV)
    Pn = torch.as_tensor(Wn[rng.choice(V, P, replace=False)].astype(np.float32),
                         device=DEV)
    for ep in range(6):
        sigma = max(3.0 * (1 - ep / 6) + 0.4, 0.4)
        Xt = Xt[torch.randperm(N_FIT, device=DEV)]
        for b0 in range(0, N_FIT, 512):
            Xb = Xt[b0:b0 + 512]
            C = Xb @ Pn.T
            bmu = C.argmax(1)
            d2 = ((Gt[:, None, :] - Gt[bmu][None, :, :]) ** 2).sum(-1)
            K = torch.exp(-d2 / (2 * sigma * sigma)).T.float()
            num = K.T @ Xb
            den = K.sum(0).clamp_min(1e-6)
            Pn = num / den[:, None]
            Pn = Pn / Pn.norm(dim=1, keepdim=True).clamp_min(1e-9)
    C = Xt @ Pn.T
    qe = float(np.degrees(torch.acos(C.clamp(-1, 1).max(1).values).mean().cpu()))
    bmu = C.argmax(1)
    uniq, counts = torch.unique(bmu, return_counts=True)
    empty = (P - len(uniq)) / P * 100
    maxm = float(counts.max()) / N_FIT * 100
    print(f"    1D-ring 256   qe {qe:5.1f} deg  empty {empty:5.1f}%  max-memb {maxm:5.1f}%")


def s3_autocluster(Wn, model, tok):
    """Spherical k-means (label-free families) -> inversion resolution."""
    rng = np.random.default_rng(3)
    V, d = Wn.shape
    X = Wn[rng.choice(V, N_FIT, replace=False)].astype(np.float32)
    K = 30
    Pn = X[rng.choice(N_FIT, K, replace=False)]
    Xt = torch.as_tensor(X, device=DEV)
    Pt = torch.as_tensor(Pn, device=DEV)
    for it in range(20):
        C = Xt @ Pt.T
        bmu = C.argmax(1)
        for k in range(K):
            m = bmu == k
            if m.sum() > 0:
                Pt[k] = Xt[m].mean(0)
        Pt = Pt / Pt.norm(dim=1, keepdim=True).clamp_min(1e-9)
    bmu = (Xt @ Pt.T).argmax(1)
    sizes = torch.bincount(bmu, minlength=K)
    big = [k for k in range(K) if 5 <= sizes[k] <= 60]
    if len(big) < 3:
        big = [int(k) for k in torch.argsort(-sizes)[:5] if sizes[k] >= 4]
    print(f"S3  auto-clusters (spherical k-means, k=30): {len(big)} usable families "
          f"size {int(min(sizes[big]))}-{int(max(sizes[big]))}")
    # spread + inversion resolution per auto-family
    pids_list = ['The capital of France is', 'Once upon a time',
                 'Tell me something interesting:', 'To bake sourdough bread']
    li = model.config.num_hidden_layers - 1
    states = {}
    for p in pids_list:
        pid = tok(p, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        with torch.no_grad():
            outs = model(pid, output_hidden_states=True)
        states[p] = outs.hidden_states[li + 1][0, -1, :].cpu().float().numpy()
    Wn32 = torch.as_tensor(Wn.astype(np.float32), device=DEV)
    res_center, res_inv, spreads = [], [], []
    for k in big:
        mem = torch.where(bmu == k)[0].cpu().numpy()
        fam = mem[:min(10, len(mem))]          # use up to 10 members
        frows = Wn[fam]
        Cc = frows.mean(0); Cc = Cc / np.linalg.norm(Cc)
        spread = float(np.degrees(np.arccos(
            np.clip(frows @ Cc, -1, 1)).mean()))
        seeds = num_ids(fam, model, tok)
        rc, ri = 0, 0
        for p, h in states.items():
            u = h / np.linalg.norm(h)
            # center steering
            tau = M.tangent_direction(u, Cc)
            v = M.rotate_toward(u, tau, math.radians(17))
            L = v @ Wn.T
            if L[fam].max() > np.delete(L, fam).max():
                rc += 1
            # inversion: best-positioned member
            best = fam[int(np.argmax(np.clip(Wn[fam] @ u, -1, 1)))]
            tau = M.tangent_direction(u, Wn[best])
            v = M.rotate_toward(u, tau, math.radians(17))
            L = v @ Wn.T
            if L[fam].max() > np.delete(L, fam).max():
                ri += 1
        n = len(states)
        res_center.append(rc / n); res_inv.append(ri / n); spreads.append(spread)
    res_center, res_inv, spreads = map(np.array, (res_center, res_inv, spreads))
    for i in range(len(big)):
        print(f"    family {big[i]:3d} size {int(sizes[big[i]]):3d}  spread {spreads[i]:5.1f} deg  "
              f"center-res {res_center[i]*100:5.1f}%  inversion-res {res_inv[i]*100:5.1f}%")
    print(f"    AVG  center {res_center.mean()*100:5.1f}%  inversion {res_inv.mean()*100:5.1f}%  "
          f"corr(spread,center-res) {np.corrcoef(spreads, res_center)[0,1]:+.3f}  "
          f"corr(spread,inversion-res) {np.corrcoef(spreads, res_inv)[0,1]:+.3f}")


def num_ids(fam, model, tok):
    """Small hint: pick ~6 decodable member token ids + fallback rows."""
    # fam rows are token ids here (0..V-1)
    return fam


def s4_interchangeability(Wn, model, tok):
    """Do geometric NN pairs get correlated logits across contexts? (no labels)"""
    rng = np.random.default_rng(11)
    V, d = Wn.shape
    # 40 random tokens (printable single-token)
    cand = []
    for tid in rng.choice(V, 4000, replace=False):
        t = tok.decode([int(tid)], skip_special_tokens=True)
        if t and t.strip() and all(32 <= ord(c) < 127 for c in t):
            cand.append(int(tid))
        if len(cand) >= 40:
            break
    cand = np.array(cand)
    # 60 contexts
    prompts = ['The capital of France is', 'Once upon a time',
               'Tell me something interesting:', 'To bake sourdough bread',
               'In the year 3000, humans will', 'The quantum computer',
               'For dinner I made', 'The quick brown fox'] * 8
    li = model.config.num_hidden_layers - 1
    Hs = []
    for p in prompts:
        pid = tok(p, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        with torch.no_grad():
            outs = model(pid, output_hidden_states=True)
        Hs.append(outs.hidden_states[li + 1][0, -1, :].cpu().float().numpy())
    H = np.stack(Hs)                       # (64, d)
    # logits matrix (contexts x tokens)
    L = H @ Wn[cand].T                     # (64,40)
    # per token its geometric top neighbor (exclude self)
    Cmat = np.clip(Wn[cand] @ Wn[cand].T, -1, 1)
    np.fill_diagonal(Cmat, -1)
    nn_id = np.argmax(Cmat, axis=1)
    # correlation per (t, nn) across contexts
    corr_nn = []
    for i in range(len(cand)):
        corr_nn.append(np.corrcoef(L[:, i], L[:, nn_id[i]])[0, 1])
    corr_rand = []
    for i in range(len(cand)):
        j = rng.integers(len(cand))
        corr_rand.append(np.corrcoef(L[:, i], L[:, j])[0, 1])
    corr_nn, corr_rand = np.array(corr_nn), np.array(corr_rand)
    print(f"S4  logit correlation across 64 contexts, n={len(cand)} tokens:")
    print(f"    (t, geometric-NN(t)) pairs:  mean {corr_nn.mean():+.3f}  "
          f"n0-free {np.mean([c for c in corr_nn if abs(c) > 0.2]):+.3f}")
    print(f"    (t, random) pairs:           mean {corr_rand.mean():+.3f}  "
          f"med {np.median(corr_rand):+.3f}")
    print(f"    NN-beats-random: {np.mean(corr_nn > np.median(corr_rand)):.1%} of tokens")


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    s1_tying(model)
    print()
    if getattr(model, 'lm_head', None) is not None:
        W = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W = model.gpt_neox.embed_out.weight.detach().cpu().float().numpy()
    W = W[:model.config.vocab_size]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    print(f"(rows {Wn.shape}, {time.time()-t0:.0f}s)\n")
    s2_sweep(Wn)
    print()
    s3_autocluster(Wn, model, tok)
    print()
    s4_interchangeability(Wn, model, tok)
    print(f"\n[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()