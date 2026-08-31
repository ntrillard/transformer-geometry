#!/usr/bin/env python3
"""Diagnose WHY the global spherical SOM collapses on LM-head rows.

Measured earlier (eval_kohonen_sphere): a 16x16 spherical batch SOM over
Qwen2-0.5B head rows yields 214/256 empty cells, one cell holding ~86% of
151,936 rows, quantization error ~81 deg. This script dissects the cause:

  A) Row-manifold density profile: sorted NN-angle profile over the FULL vocab.
     Is the manifold smooth-dense, or twins+desert? At what angular scale
     must a SOM kernel operate?
  B) Collapse dynamics: per-epoch pairwise prototype angular spread, max cell
     membership share, prototype-vs-data-centroid cosine.
  C) Prior mismatch: correlation between GRID distance and ANGULAR distance
     over prototype pairs (does the 2-D lattice topology prior ever hold?).
  D) Control: re-key the update to DATA-space angular distance (neural-gas /
     spherical-SOM style). If the collapse disappears, the cause is the
     grid a-priori, not the sphere or the data.

Run: python eval_som_failure.py
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as st_load

CACHE = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct'


def load_head_rows():
    snaps = sorted((CACHE / 'snapshots').glob('*'))
    st = sorted(snaps[0].glob('*.safetensors'))[0]
    d = st_load(str(st))
    key = [k for k in d if 'weight' in k and d[k].ndim == 2][0]
    return d[key].float().numpy()


def main():
    t0 = time.time()
    W = load_head_rows()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    V, d = Wn.shape
    print(f"rows {Wn.shape}")
    rng = np.random.default_rng(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Wt = torch.as_tensor(Wn.astype(np.float32), device=dev)

    # row-norm / used-token structure (Qwen2 pads its vocab) ----
    row_norm = np.linalg.norm(W, axis=1)
    med = np.median(row_norm)
    print(f"row-norm: median {med:.3e}  p10 {np.percentile(row_norm,10):.2e}  ",
          f"share<1e-3 {np.mean(row_norm<1e-3)*100:.1f}%  share<0.05*med {np.mean(row_norm<0.05*med)*100:.1f}%")

    # ---- A) density profile over the full vocab, GPU ----
    toks = rng.choice(V, size=60, replace=False)
    St = Wt[toks]                                  # (60,d)
    C_full = Wt @ St.T                             # (V,60) cosines
    top = C_full.topk(101, dim=0).values           # (101,60), row0=self
    ang = torch.rad2deg(torch.acos(top.clamp(-1, 1))).cpu().numpy()
    print(f"\nA) row-manifold density profile (median over 60 random tokens, [Q25 Q75]); self at index 0:")
    for k in (1, 2, 5, 10, 30, 100):
        a = ang[k]
        print(f"   {k}-th NN: median {np.median(a):6.2f} deg  [{np.percentile(a,25):.2f}, {np.percentile(a,75):.2f}]")

    # ---- shared SOM machinery ----
    side = 16
    P = side * side
    init = Wn[rng.choice(V, size=P, replace=False)].copy()
    gy, gx = np.meshgrid(np.arange(side), np.arange(side))
    G = np.stack([gy.ravel(), gx.ravel()], 1).astype(np.float32)
    Gt = torch.as_tensor(G, device=dev)
    Xt = torch.as_tensor(Wn[rng.choice(V, size=20000, replace=False)].astype(np.float32),
                         device=dev)
    S = len(Xt)

    def report(ep, Pn, label, params):
        Pn = np.asarray(Pn)
        r = np.random.default_rng(ep)
        m = min(150, P)
        a = r.choice(P, size=m, replace=False)
        b = r.choice(P, size=m, replace=False)
        pair = np.degrees(np.arccos(np.clip((Pn[a] * Pn[b]).sum(1), -1, 1)))
        C = Xt @ torch.as_tensor(Pn, device=dev).T
        bmu = C.argmax(1)
        uniq, counts = torch.unique(bmu, return_counts=True)
        maxshare = float(counts.max()) / S
        cen = Xt.mean(0)
        cen = cen / cen.norm()
        cc = float((torch.as_tensor(Pn, device=dev) @ cen).mean())
        dg = np.sqrt(((G[a] - G[b]) ** 2).sum(1))
        corr = float(np.corrcoef(dg, pair)[0, 1])
        print(f"   [{label}] ep{ep} ({params})  pair-angle med {np.median(pair):5.1f} deg "
              f" max-membership {maxshare*100:5.1f}%  proto-centroid cos {cc:+.3f}  "
              f"corr(grid,angle) {corr:+.3f}")

    # ---- B+C) grid-kernel Kohonen batch-SOM ----
    print(f"\nB/C) grid-kernel SOM (16x16, 20k fit, 8 epochs):")
    Pnt = torch.as_tensor(init, device=dev)
    for ep in range(8):
        sigma = max(4.0 * (1 - ep / 8) + 0.3, 0.3)
        Xt = Xt[torch.randperm(S, device=dev)]
        for b0 in range(0, S, 512):
            Xb = Xt[b0:b0 + 512]
            C = Xb @ Pnt.T
            bmu = C.argmax(1)
            d2 = ((Gt[:, None, :] - Gt[bmu][None, :, :]) ** 2).sum(-1)   # (P,B)
            K = torch.exp(-d2 / (2 * sigma * sigma)).T.float()           # (B,P)
            num = K.T @ Xb
            den = K.sum(0).clamp_min(1e-6)
            Pnt = num / den[:, None]
            Pnt = Pnt / Pnt.norm(dim=1, keepdim=True).clamp_min(1e-9)
        report(ep, Pnt.cpu(), 'grid', f'sig={sigma:.1f}')
    qe_grid = float((torch.clamp(Xt @ Pnt.T, -1, 1).max(1).values).mean())

    # ---- D) data-space-kernel control (neural-gas style) ----
    print(f"\nD) data-space-kernel control (angular sigma, same init):")
    Pnt2 = torch.as_tensor(init, device=dev)
    for ep in range(8):
        sigma = max(50.0 * (1 - ep / 8) + 3.0, 3.0)     # angular sigma, degrees
        sig_rad = math.radians(sigma)
        Xt = Xt[torch.randperm(S, device=dev)]
        for b0 in range(0, S, 512):
            Xb = Xt[b0:b0 + 512]
            C = Xb @ Pnt2.T
            bmu = C.argmax(1)
            Pb = Pnt2[bmu]                                       # (B,d)
            ang_all = torch.acos((Pnt2[None, :, :] @ Pb[:, :, None]).squeeze(-1)
                                 .clamp(-1, 1))                  # (B,P) angle proto vs bmu-proto
            K = torch.exp(-(ang_all / sig_rad) ** 2)
            num = K.T @ Xb
            den = K.sum(0).clamp_min(1e-6)
            Pnt2 = num / den[:, None]
            Pnt2 = Pnt2 / Pnt2.norm(dim=1, keepdim=True).clamp_min(1e-9)
        report(ep, Pnt2.cpu(), 'data', f'sig={sigma:.0f}deg')
    qe_data = float((torch.clamp(Xt @ Pnt2.T, -1, 1).max(1).values).mean())

    print(f"\nquantization error (mean max-cos over fit sample): "
          f"grid-kernel {qe_grid:+.4f} ({(90-np.degrees(np.arccos(qe_grid))):.1f} deg) "
          f"| data-kernel {qe_data:+.4f} ({(90-np.degrees(np.arccos(qe_data))):.1f} deg)")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()