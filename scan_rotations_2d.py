#!/usr/bin/env python3
"""2D rotation-scan — sweeps two independent angles over a spherical patch.

For each token, the final hidden state h is rotated within the 3D subspace
spanned by {h, t₁, t₂} where:
  t₁ = self-tangent (toward token's own unembedding, ⟂ h)
  t₂ = target-tangent (toward a fixed "target" token's unembedding, ⟂ {h, t₁})

Two-angle parameterization of the sphere surface:
  x(θ, φ) = ‖h‖ · (cos φ · (cos θ · ĥ + sin θ · t₁) + sin φ · t₂)

  θ sweeps the original great circle (the {h, t₁} plane)
  φ sweeps "upward" into the third orthogonal direction t₂

Top-K predictions are recorded at each (θ, φ) → output shape (n, n_φ, n_θ, K).

FIX vs. earlier draft: chunks were being flattened into a single list and
concatenated along one axis regardless of which angle (θ or φ) they varied
over. That scrambles the (φ, θ) grid and, once batch sizes stop dividing
evenly into 360, raises a shape-mismatch error in torch.cat. This version
concatenates θ-batches first (inner), then φ-batches (outer), so uneven
batch sizes are handled correctly and the grid stays correctly ordered.

GPU: one hidden-state forward per chunk (tiny). All heavy matmuls on CPU.
Peak RAM bounded by (PHI_BATCH × THETA_BATCH × V × n).
"""
import sys, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2-1.5B-Instruct"
OUT = "/home/nicolas/model-harness/151k_states/chunks/rot2d"
KPER = 8
CHUNK = 128
THETA_BATCH = 45
PHI_BATCH = 30

THETA = np.arange(0, 360)
PHI = np.arange(0, 360)
n_theta = len(THETA)
n_phi = len(PHI)

ct = np.cos(np.deg2rad(THETA)).astype(np.float32)
st = np.sin(np.deg2rad(THETA)).astype(np.float32)
cp = np.cos(np.deg2rad(PHI)).astype(np.float32)
sp = np.sin(np.deg2rad(PHI)).astype(np.float32)

TARGET_TID = 220   # verify below


def main(start, end):
    os.makedirs(OUT, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        MODEL, cache_dir="models", local_files_only=True, trust_remote_code=True
    )
    print(f"TARGET_TID {TARGET_TID} decodes to: {tok.decode([TARGET_TID])!r}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, cache_dir="models", local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda",
    ).eval()

    H = model.lm_head.weight.detach().float().cpu().numpy()   # (V, d)
    V = H.shape[0]

    w_target = H[TARGET_TID]
    w_target_n = w_target / (np.linalg.norm(w_target) + 1e-12)

    ct_t = torch.from_numpy(ct).to(dtype=torch.bfloat16)
st_t = torch.from_numpy(st).to(dtype=torch.bfloat16)
cp_t = torch.from_numpy(cp).to(dtype=torch.bfloat16)
sp_t = torch.from_numpy(sp).to(dtype=torch.bfloat16)

    for s in range(start, end, CHUNK):
        e = min(s + CHUNK, end)
        tids = np.arange(s, e)
        n = len(tids)

        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([tids.tolist()], device="cuda"),
                output_hidden_states=True,
            )
        h = out.hidden_states[-1][0].float().cpu().numpy()    # (n, d)
        hnorm = np.linalg.norm(h, axis=1, keepdims=True) + 1e-9
        hn = h / hnorm

        selves = H[tids]
        selves_n = selves / (np.linalg.norm(selves, axis=1, keepdims=True) + 1e-12)
        t1 = selves_n - (selves_n * hn).sum(1, keepdims=True) * hn
        t1 = t1 / (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-9)

        t2 = w_target_n[None, :] - (w_target_n * hn).sum(1, keepdims=True) * hn
        t2 = t2 - (t2 * t1).sum(1, keepdims=True) * t1
        t2 = t2 / (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-9)

        G0 = H @ h.T                    # (V, n)
        G1 = H @ (t1 * hnorm).T         # (V, n)
        G2 = H @ (t2 * hnorm).T         # (V, n)

G0t = torch.from_numpy(G0).to(dtype=torch.bfloat16)
    G1t = torch.from_numpy(G1).to(dtype=torch.bfloat16)
    G2t = torch.from_numpy(G2).to(dtype=torch.bfloat16)

        # Nested concat: inner over θ, outer over φ
        phi_top_chunks, phi_log_chunks = [], []
        for p0 in range(0, n_phi, PHI_BATCH):
            p1 = min(p0 + PHI_BATCH, n_phi)
            cp_b = cp_t[p0:p1, None, None, None]   # (b_φ, 1, 1, 1)
            sp_b = sp_t[p0:p1, None, None, None]

            theta_top_chunks, theta_log_chunks = [], []
            for t0 in range(0, n_theta, THETA_BATCH):
                t1b = min(t0 + THETA_BATCH, n_theta)
                ct_b = ct_t[t0:t1b, None, None]     # (b_θ, 1, 1)
                st_b = st_t[t0:t1b, None, None]

                L = (cp_b * ct_b) * G0t[None, None] \
                  + (cp_b * st_b) * G1t[None, None] \
                  + (sp_b * 1.0)  * G2t[None, None]   # (b_φ, b_θ, V, n)

                tv, ti = torch.topk(L, KPER, dim=2)   # (b_φ, b_θ, K, n)
                theta_top_chunks.append(ti)
                theta_log_chunks.append(tv)

            # concat over θ axis (dim=1) → (b_φ, n_θ, K, n)
            ti_row = torch.cat(theta_top_chunks, dim=1)
            tv_row = torch.cat(theta_log_chunks, dim=1)
            phi_top_chunks.append(ti_row)
            phi_log_chunks.append(tv_row)

        # concat over φ axis (dim=0) → (n_φ, n_θ, K, n)
        top_grid = torch.cat(phi_top_chunks, dim=0)
        log_grid = torch.cat(phi_log_chunks, dim=0)

        # → (n, n_φ, n_θ, K)
        top_all = top_grid.permute(3, 0, 1, 2).numpy().astype(np.int32)
        log_all = log_grid.permute(3, 0, 1, 2).float().numpy()

        f = os.path.join(OUT, f"rot2d_scan_{s:07d}_{e:07d}.npz")
        np.savez(
            f,
            theta=THETA.astype(np.float32),
            phi=PHI.astype(np.float32),
            top=top_all,
            logits=log_all,
            tids=tids,
        )
        mb = (top_all.nbytes + log_all.nbytes) / 1e6
        print(f"  {os.path.basename(f)} ({n} tok, {n_theta}×{n_phi} ang, {mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 151643
    main(start, end)
    print("done")