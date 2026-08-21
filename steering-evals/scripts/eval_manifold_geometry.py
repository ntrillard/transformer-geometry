#!/usr/bin/env python3
"""Sphere vs ellipsoid vs natural activation manifold geometry.

Collects hidden states from a small corpus of ordinary prompts, then asks:
  1. Are natural activations closer to a sphere or to an ellipsoid?
  2. Do steering-induced steps deviate from the natural manifold?
  3. Is the shortest route to a target a spherical geodesic or an ellipsoidal one?

Run: python eval_manifold_geometry.py --model Qwen/Qwen2-0.5B-Instruct --layer-idx 8
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import steering_geometry_test as M

OUT = Path("steering_geometry_results")
MODEL = "Qwen/Qwen2-0.5B-Instruct"
PROMPTS = [
    "The capital of France is", "In the year 3000, humans will", "To bake sourdough bread",
    "The theory of relativity", "Once upon a time", "The quantum computer",
    "The quick brown fox", "For dinner I made", "In machine learning, a transformer is",
    "The stock market closed higher today because", "A valid Python function to sort a list is",
    "The user asked for help with", "For a secure password, choose", "Quantum computers use",
    "In Canada, winter temperatures can reach", "The best way to learn Python is",
    "The history of Rome began with", "Artificial intelligence will change",
    "To make espresso, first", "The mission to Mars is",
    "Climate change affects", "The orchestra played a",
    "When debugging code, always", "The recipe calls for",
    "In the courtroom, the lawyer", "The satellite launched at",
    "The encrypted message read", "The athlete won the",
    "The compiler optimizes", "The museum houses a",
    "To negotiate a salary", "The vaccine prevents",
    "The router connects", "The algorithm sorts by",
    "The sensor detected", "The treaty was signed in",
    "The telescope observed", "The battery stores",
    "The firewall blocks", "The enzyme catalyzes",
    "The submarine dove", "The protest demanded",
    "The catalyst speeds up", "The hurricane made landfall in",
    "The blockchain records", "The microscope revealed",
    "The drone captured", "The invoice total was",
    "The glacier melted", "The conductor raised",
    "The archive contains", "The reactor generates",
]


def hidden_states(model, tok, prompts, layer_idx):
    """Return hidden states at layer `layer_idx` for the last token of each prompt."""
    states = []
    for p in prompts:
        ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            h = out.hidden_states[layer_idx + 1]  # embedding layer is index 0
            states.append(h[0, -1, :].float().cpu().numpy())
    return np.stack(states, axis=0)


def spherical_geodesic(a, b):
    """Angle between two vectors in radians (great-circle distance on sphere)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def ellipsoid_distance(a, b, P, inv_k):
    """Mahalanobis distance in top-k PCA subspace."""
    d = (a - b) @ P
    return float(np.sqrt(d @ inv_k @ d))


def fit_pca_ellipsoid(states, k=16):
    """Fit ellipsoid in the top-k PCA subspace and return projector + inverse metric."""
    centered = states - states.mean(axis=0, keepdims=True)
    n = centered.shape[0]
    k = min(k, n - 1)
    _, s, Vh = np.linalg.svd(centered, full_matrices=False)
    P = Vh[:k, :].T  # (d, k)
    z = centered @ P  # (n, k)
    cov_k = np.cov(z.T)
    inv_k = np.linalg.inv(cov_k + 1e-6 * np.eye(k))
    return P, inv_k, s


def steering_step(model, tok, head, layer_idx, prompt, target_id, alpha=1.0):
    """Return (h0, h_sphere, h_ellipsoid, h_actual) at layer_idx."""
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)

    # baseline hidden state
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        h0 = out.hidden_states[layer_idx + 1][0, -1, :].float().cpu().numpy()

    # target direction = LM-head row for target_id
    w_t = head[target_id].float().cpu().numpy()
    w_t = w_t / np.linalg.norm(w_t)

    # tangent step on sphere: h' = h0 + alpha * (w_t - (w_t·h0/||h0||^2) h0)
    h0n = h0 / np.linalg.norm(h0)
    tangent = w_t - (w_t @ h0n) * h0n
    tangent = tangent / np.linalg.norm(tangent) * alpha * np.linalg.norm(h0)
    h_sphere = h0 + tangent

    # ellipsoid step: scale tangent by inverse sqrt eigenvalues in PCA basis
    # simpler: move along tangent and reproject to ellipsoid surface
    # here we just return h_sphere renormalized to ellipsoid radius
    # (will be compared in distances)
    h_ellipsoid = h_sphere  # placeholder; refined below

    # actual steering step: add target token and observe hidden state
    with torch.no_grad():
        out2 = model(torch.cat([ids, torch.tensor([[target_id]], device=ids.device)], dim=1),
                     output_hidden_states=True)
        h_actual = out2.hidden_states[layer_idx + 1][0, -1, :].float().cpu().numpy()

    return h0, h_sphere, h_ellipsoid, h_actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layer-idx", type=int, default=8,
                    help="layer index at which to inspect the manifold")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="relative tangent step size")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tok = M.load_model(args.model, dtype="fp16")
    head = torch.as_tensor(model.lm_head.weight.detach().float().cpu().numpy(),
                           device="cpu")
    layer = args.layer_idx

    print(f"Collecting natural activations at layer {layer} ...")
    states = hidden_states(model, tok, PROMPTS, layer)
    norms = np.linalg.norm(states, axis=1)
    P, inv_k, s = fit_pca_ellipsoid(states, k=16)
    var_explained = np.cumsum(s**2) / np.sum(s**2)

    print(f"  mean norm: {norms.mean():.4f}  std: {norms.std():.4f}")
    print(f"  top-16 variance explained: {var_explained[min(15, len(var_explained)-1)]:.4f}")
    print(f"  top-3 singular-value ratios (to max): " + ", ".join(f"{s[-(i+1)]/s.max():.4f}" for i in range(3)))

    # Distances between consecutive natural activations
    sph_dists = [spherical_geodesic(states[i], states[i+1]) for i in range(len(states)-1)]
    ell_dists = [ellipsoid_distance(states[i], states[i+1], P, inv_k) for i in range(len(states)-1)]
    euc_dists = [float(np.linalg.norm(states[i] - states[i+1])) for i in range(len(states)-1)]

    print(f"\nNatural-activation distances:")
    print(f"  spherical geodesic (rad): median={np.median(sph_dists):.4f}  max={max(sph_dists):.4f}")
    print(f"  ellipsoid Mahalanobis:    median={np.median(ell_dists):.4f}  max={max(ell_dists):.4f}")
    print(f"  euclidean:                median={np.median(euc_dists):.4f}  max={max(euc_dists):.4f}")

    # Steering steps
    print(f"\nSteering-step geometry (alpha={args.alpha}):")
    rows = []
    for p in PROMPTS[:5]:
        # pick a plausible target token: first token of a continuation
        cont = "the"
        target_id = tok(cont, add_special_tokens=False).input_ids[0]
        h0, h_sphere, _, h_actual = steering_step(model, tok, head, layer, p, target_id, alpha=args.alpha)

        d_sphere = spherical_geodesic(h0, h_sphere)
        d_actual = spherical_geodesic(h0, h_actual)
        d_ell_sphere = ellipsoid_distance(h0, h_sphere, P, inv_k)
        d_ell_actual = ellipsoid_distance(h0, h_actual, P, inv_k)

        rows.append({
            "prompt_prefix": p[:40],
            "d_sphere_rad": d_sphere,
            "d_actual_rad": d_actual,
            "d_ellipsoid_sphere": d_ell_sphere,
            "d_ellipsoid_actual": d_ell_actual,
            "norm_h0": float(np.linalg.norm(h0)),
            "norm_actual": float(np.linalg.norm(h_actual)),
        })
        print(f"  '{p[:40]:40s}'  sphere step={np.degrees(d_sphere):.2f}°  actual step={np.degrees(d_actual):.2f}°")

    safe = args.model.replace("/", "--")
    OUT.mkdir(parents=True, exist_ok=True)

    df_natural = pd.DataFrame({
        "spherical_geodesic": sph_dists,
        "ellipsoid_mahalanobis": ell_dists,
        "euclidean": euc_dists,
    })
    df_natural.to_csv(OUT / f"manifold_natural__{safe}_L{layer}.csv", index=False)

    df_steering = pd.DataFrame(rows)
    df_steering.to_csv(OUT / f"manifold_steering__{safe}_L{layer}.csv", index=False)

    print(f"\nSaved -> {OUT / f'manifold_natural__{safe}_L{layer}.csv'}")
    print(f"Saved -> {OUT / f'manifold_steering__{safe}_L{layer}.csv'}")


if __name__ == "__main__":
    main()
