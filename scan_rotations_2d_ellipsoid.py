#!/usr/bin/env python3
"""2D rotation-scan on the diagonal ellipsoid (gain-corrected sphere).

Instead of rotating the final hidden state h in the Euclidean {h, t} plane,
this version:
  1. Transforms h to gain-corrected space:  h' = h / γ
  2. Normalizes h' to the √d sphere
  3. Computes self-tangent t₁' and target-tangent t₂' in gain-corrected space
  4. Rotates on the sphere in gain-corrected space
  5. Transforms back:  x(θ, φ) = γ ⊙ x'(θ, φ)

This keeps the rotated state on the natural hidden-state ellipsoid.

Output shape: (n, n_φ, n_θ, K) — same convention as scan_rotations_2d.py.
"""
import sys, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2-1.5B-Instruct"
OUT = "/home/nicolas/model-harness/151k_states/chunks/rot2d_ellipse"
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

TARGET_TID = 220


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

    # Get the final RMSNorm gain vector γ
    # In Qwen2, the final norm is model.model.norm
    final_norm = model.model.norm
    gamma = final_norm.weight.detach().float().cpu().numpy()   # (d,)
    print(f"Final norm gamma: mean={gamma.mean():.3f} std={gamma.std():.3f} "
          f"||gamma||={np.linalg.norm(gamma):.1f}",
          flush=True)

    V = H.shape[0]

    # Precompute target unembedding row and convert to gain-corrected space
    w_target = H[TARGET_TID]                                      # (d,)
    w_target_gc = w_target / gamma                                # (d,)  gain-corrected
    w_target_gc_n = w_target_gc / (np.linalg.norm(w_target_gc) + 1e-12)

    ct_t = torch.from_numpy(ct).to(dtype=torch.bfloat16)
    st_t = torch.from_numpy(st).to(dtype=torch.bfloat16)
    cp_t = torch.from_numpy(cp).to(dtype=torch.bfloat16)
    sp_t = torch.from_numpy(sp).to(dtype=torch.bfloat16)

    d = H.shape[1]
    sphere_r = np.sqrt(d, dtype=np.float32)

    for s in range(start, end, CHUNK):
        e = min(s + CHUNK, end)
        tids = np.arange(s, e)
        n = len(tids)

        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([tids.tolist()], device="cuda"),
                output_hidden_states=True,
            )
        h = out.hidden_states[-1][0].float().cpu().numpy()       # (n, d)

        # Step 1: transform to gain-corrected space
        h_gc = h / gamma[None, :]                                 # (n, d)
        h_gc_norm = np.linalg.norm(h_gc, axis=1, keepdims=True) + 1e-9
        h_gc_hat = h_gc / h_gc_norm                               # unit vector in gc space
        # Scale to sphere radius
        h_gc_s = h_gc_hat * sphere_r

        # Step 2: self-tangent in gain-corrected space
        selves = H[tids]                                           # (n, d)
        selves_gc = selves / gamma[None, :]                        # (n, d)
        selves_gc_n = selves_gc / (np.linalg.norm(selves_gc, axis=1, keepdims=True) + 1e-12)
        t1_gc = selves_gc_n - (selves_gc_n * h_gc_hat).sum(1, keepdims=True) * h_gc_hat
        t1_gc = t1_gc / (np.linalg.norm(t1_gc, axis=1, keepdims=True) + 1e-9)

        # Step 3: target tangent in gain-corrected space
        t2_gc = w_target_gc_n[None, :] - (w_target_gc_n * h_gc_hat).sum(1, keepdims=True) * h_gc_hat
        t2_gc = t2_gc - (t2_gc * t1_gc).sum(1, keepdims=True) * t1_gc
        t2_gc = t2_gc / (np.linalg.norm(t2_gc, axis=1, keepdims=True) + 1e-9)

        # Step 4: three cached matmuls in gain-corrected space
        # logits = H @ x, where x = gamma * x_gc
        # H @ x = H @ (gamma * x_gc) = (H * gamma[None, :]) @ x_gc
        # So we can pre-multiply H by gamma and use x_gc directly
        H_gamma = H * gamma[None, :]                               # (V, d)
        G0 = H_gamma @ h_gc_s.T                                    # (V, n)  true logits via gc
        G1 = H_gamma @ (t1_gc * sphere_r).T                        # (V, n)  self-tangent via gc
        G2 = H_gamma @ (t2_gc * sphere_r).T                        # (V, n)  target-tangent via gc

        G0t = torch.from_numpy(G0).to(dtype=torch.bfloat16)
        G1t = torch.from_numpy(G1).to(dtype=torch.bfloat16)
        G2t = torch.from_numpy(G2).to(dtype=torch.bfloat16)

        # Step 5: two-angle logits (same formula, but in gc space)
        phi_top_chunks, phi_log_chunks = [], []
        for p0 in range(0, n_phi, PHI_BATCH):
            p1 = min(p0 + PHI_BATCH, n_phi)
            cp_b = cp_t[p0:p1, None, None, None]
            sp_b = sp_t[p0:p1, None, None, None]

            theta_top_chunks, theta_log_chunks = [], []
            for t0 in range(0, n_theta, THETA_BATCH):
                t1b = min(t0 + THETA_BATCH, n_theta)
                ct_b = ct_t[t0:t1b, None, None]
                st_b = st_t[t0:t1b, None, None]

                L = (cp_b * ct_b) * G0t[None, None] \
                  + (cp_b * st_b) * G1t[None, None] \
                  + (sp_b * 1.0)  * G2t[None, None]

                tv, ti = torch.topk(L, KPER, dim=2)
                theta_top_chunks.append(ti)
                theta_log_chunks.append(tv)

            ti_row = torch.cat(theta_top_chunks, dim=1)
            tv_row = torch.cat(theta_log_chunks, dim=1)
            phi_top_chunks.append(ti_row)
            phi_log_chunks.append(tv_row)

        top_grid = torch.cat(phi_top_chunks, dim=0)
        log_grid = torch.cat(phi_log_chunks, dim=0)

        top_all = top_grid.permute(3, 0, 1, 2).numpy().astype(np.int32)
        log_all = log_grid.permute(3, 0, 1, 2).float().numpy()

        f = os.path.join(OUT, f"rot2d_ellipse_scan_{s:07d}_{e:07d}.npz")
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