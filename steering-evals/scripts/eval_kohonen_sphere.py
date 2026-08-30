#!/usr/bin/env python3
"""Spherical Kohonen (SOM) steering: a topographic prototype map on the state sphere.

Premise: the LM head already acts as a degenerate SOM -- rows W_t are
prototypes, argmax is winner-take-all, and tangent steering is the DUAL
competitive update (move the STATE toward a prototype; a SOM moves the
prototype toward the data). The paper's sustained-steering failure (persistent
row-steering -> pits) is exactly a map that collapses onto a single winner
cell. Two constructions are tested here:

  T1 global  : fit a 16x16 spherical batch-SOM to the head-row manifold.
               (Evidence: can a low-dim geometric lattice resolve this
               high-dim manifold? measured via quantization error + cell sizes)
  T1b local  : semantic coherence of each test word's row-neighborhood
               (are nearest head rows topographically meaningful?)
  T2         : steering geometry, ROW vs LOCAL-CENTROID ("concept") direction
               -- the centroid is the sigma->0 batch-SOM prototype of the
               target's neighborhood: the Kohonen limit for concept steering.
  T3         : persistent generation, row vs centroid: does concept steering
               dodge the pit (topical diversity) that row steering collapses into?

Run: python eval_kohonen_sphere.py
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import steering_geometry_test as M
from eval_practical_steering import gen

OUT = Path("../steering_geometry_results")

GRID = 16           # 16x16 neurons = 256 prototypes (global lattice)
EPOCHS = 30
N_FIT = 20_000
K_NEIGH = 30        # local-centroid neighborhood size
TEST_WORDS = ["apple", "Paris", "dog", "money", "red", "door", "king", "ocean"]
SEED = 0


@torch.no_grad()
def fit_som(Wn, n_neurons, epochs, seed=0):
    """Batch spherical SOM (soft Gaussian assignment). Returns unit prototypes P."""
    d = Wn.shape[1]
    side = int(math.sqrt(n_neurons))
    gy, gx = np.meshgrid(np.arange(side), np.arange(side))
    G = np.stack([gy.ravel(), gx.ravel()], 1).astype(np.float32)  # (P,2)
    P = int(n_neurons)
    rng = np.random.default_rng(seed)
    Pn = Wn[rng.choice(len(Wn), size=P, replace=False)].copy()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Pn = torch.as_tensor(Pn, device=dev)
    Gt = torch.as_tensor(G, device=dev)

    idx = rng.choice(len(Wn), size=N_FIT, replace=False)
    X = torch.as_tensor(Wn[idx].astype(np.float32), device=dev)
    S = len(X)
    for ep in range(epochs):
        sigma = max(4.0 * (1 - ep / epochs) + 0.3, 0.3)
        X = X[torch.randperm(S, device=dev)]
        for b0 in range(0, S, 512):
            Xb = X[b0:b0 + 512]
            C = Xb @ Pn.T                      # (B,P) cosines
            bmu = C.argmax(1)
            d2 = ((Gt[:, None, :] - Gt[bmu][None, :, :]) ** 2).sum(-1)  # (P,B)
            K = torch.exp(-d2 / (2 * sigma * sigma)).T.float()
            num = K.T @ Xb                     # (P,d)
            den = K.sum(0).clamp_min(1e-6)
            Pn = num / den[:, None]
            Pn = Pn / Pn.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return Pn.cpu().numpy()


@torch.no_grad()
def assign(Wn, Pn, chunk=8192):
    """BMU per row on GPU. Returns list of member id arrays per prototype."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Pt = torch.as_tensor(Pn, device=dev)
    bmu_all = []
    for b0 in range(0, len(Wn), chunk):
        Xb = torch.as_tensor(Wn[b0:b0 + chunk].astype(np.float32), device=dev)
        bmu_all.append((Xb @ Pt.T).argmax(1).cpu().numpy())
    bmu = np.concatenate(bmu_all)
    return [np.where(bmu == p)[0] for p in range(len(Pn))]


def knn_rows(Wn, s, k, exclude=None):
    c = np.clip(Wn @ s, -1, 1)
    if exclude is not None:
        c[exclude] = -1
    return np.argsort(-c)[:k]


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    model, tok = M.load_model("Qwen/Qwen2-0.5B-Instruct", dtype="fp16")
    W = model.lm_head.weight.detach().cpu().float().numpy()
    vocab = model.config.vocab_size
    W = W[:vocab]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    d = W.shape[1]
    print(f"head rows {W.shape}, fitted sample {N_FIT}")

    # ---------- T1: global spherical SOM ----------
    Pn = fit_som(Wn, GRID * GRID, EPOCHS, seed=SEED)
    members = assign(Wn, Pn)
    cell_sizes = np.array([len(m) for m in members])
    qerr = [np.clip(Wn[m] @ Pn[p], -1, 1).mean() for p, m in enumerate(members)
            if len(m)]
    Cpp = np.clip(Pn @ Pn.T, -1, 1); np.fill_diagonal(Cpp, -1)
    nn_ang = np.degrees(np.arccos(Cpp.max(1)))
    print(f"\nT1 global 16x16 SOM ({time.time()-t0:.0f}s):")
    print(f"   quantization error (mean cos member->proto): {np.mean(qerr):+.4f}")
    print(f"   prototype NN spacing: mean {nn_ang.mean():.1f} deg")
    print(f"   cell sizes: min {cell_sizes.min()} med {np.median(cell_sizes):.0f} "
          f"max {cell_sizes.max()}  (n_empty {int((cell_sizes==0).sum())})")

    # ---------- T1b: local neighborhood coherence ----------
    words = {}
    for w in TEST_WORDS:
        tids = tok(" " + w, add_special_tokens=False).input_ids
        if len(tids) != 1:
            print(f"skip {w!r}: {len(tids)} tokens"); continue
        tid = int(tids[0])
        words[w] = tid
    print(f"\nT1b local neighborhood (k={K_NEIGH}) coherence, " 
          f"K_NEIGH-NN angle + top decoded neighbors of each word's row:")
    nb = {}
    for w, tid in words.items():
        ids = knn_rows(Wn, Wn[tid], K_NEIGH, exclude=tid)
        ang = np.degrees(np.arccos(np.clip(Wn[ids] @ Wn[tid], -1, 1)))
        texts = [tok.decode([int(i)], skip_special_tokens=True) for i in ids[:6]]
        # exclude empty decode
        texts = [t if t.strip() else "«empty»" for t in texts]
        print(f"   {w:6s} tid={tid:6d} kNN-angle p50={np.median(ang):.2f}deg "
              f"top6={texts}")
        nb[w] = ids

    # ---------- T2: row vs local-centroid steering geometry ----------
    states = M.get_states(model, tok, ["The capital of France is",
                                       "Once upon a time"],
                          sorted({int(round(f * (model.config.num_hidden_layers - 1)))
                                  for f in (0.0, 0.33, 0.67, 0.99)}))
    centroids = {w: Wn[nb[w]].mean(0) for w in words}
    centroids = {w: c / np.linalg.norm(c) for w, c in centroids.items()}

    rows = []
    for w, tid in words.items():
        s = Wn[tid]; c = centroids[w]
        for ctx, llayer in states.items():
            for l, h in llayer.items():
                u = h / np.linalg.norm(h)
                for name, tgt in (("row", s), ("centroid", c)):
                    tau = M.tangent_direction(u, tgt)
                    for budget in (17, 45):
                        v = M.rotate_toward(u, tau, math.radians(budget))
                        L = v @ W.T
                        own = float(v @ tgt)
                        rank = int((L > own).sum() + 1)
                        deg = M.first_rank1_angle(u, tgt, W, tid,
                                                  max_delta=math.radians(45))
                        margin = own - float(np.delete(L, tid).max())
                        rows.append(dict(word=w, ctx=ctx, layer=l, dir=name,
                                         budget=budget, rank1=rank == 1,
                                         entry_deg=deg, margin=margin))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kohonen_steering_geometry.csv", index=False)
    print(f"\nT2 steering geometry (17-deg budget), row vs LOCAL-CENTROID:")
    d17 = df[df.budget == 17]
    for name, g in d17.groupby("dir"):
        print(f"   {name:8s} reach@17={g.rank1.mean()*100:5.1f}%  "
              f"entry(med)={g.entry_deg.median():5.1f}deg  "
              f"margin(mean)={g.margin.mean():+.3f}")

    # ---------- T3: persistent generation, row vs centroid ----------
    for w in ("apple", "Paris", "money"):
        if w not in words:
            continue
        pids = tok("Once upon a time", add_special_tokens=False,
                   return_tensors="pt").input_ids.to(model.device)
        li = model.config.num_hidden_layers - 1
        s = Wn[words[w]]; c = centroids[w]

        def hook_for(dirv):
            dn = dirv / np.linalg.norm(dirv)
            dn_t = torch.as_tensor(dn, device=model.device, dtype=model.dtype)

            def hook(module, inp, out):
                out2 = out.clone()
                h = out2[0, -1, :].float()
                hn = h / h.norm()
                g = dn_t.float() - (dn_t.float() @ hn) * hn
                g = g / max(g.norm().item(), 1e-8)
                h2 = h + 0.3 * h.norm() * g
                out2[0, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
                return out2
            return hook

        print(f"\nT3 persistent steer '{w}' (48 tok, top-p .9, 4 seeds):")
        for name, tgt in (("row", s), ("centroid", c)):
            occs, divs = [], []
            for sd in range(4):
                text, conf = gen(model, tok, pids, 48, hook_fn=hook_for(tgt),
                                 layer=li, duty=1, top_p=0.9, seed=sd)
                o = text.lower().count(w)
                toks = tok(text, add_special_tokens=False).input_ids
                div = len(set(toks)) / max(len(toks), 1)
                occs.append(o); divs.append(div)
                if sd == 0:
                    print(f"   {name:8s} seed0: {text[:85]!r}")
            print(f"   {name:8s} '{w}' x{np.mean(occs):.1f}+/-{np.std(occs):.1f} "
                  f"diversity {np.mean(divs):.2f}")

    print(f"\n total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()