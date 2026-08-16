#!/usr/bin/env python3
"""Rotation-scan chunk extraction — VRAM-safe & RAM-safe revision.

What is rotated:
  For each token, the final-layer residual-stream hidden state h is rotated
  within the 2D plane spanned by {h, t} where t is the SELF-TANGENT —
  the component of that token's own LM-head (unembedding) row w_tid
  that is perpendicular to h (i.e. the "steer toward self" direction
  from the steering proof).

  At angle θ the steered hidden state is:
    x(θ) = cosθ·h + sinθ·(t·‖h‖)
  and the logits are:
    logits(θ) = H·x(θ) = cosθ·G0 + sinθ·G1
  where G0 = H·h (true logits) and G1 = H·(t·‖h‖).

    θ=0°    → true hidden state (real predictions)
    θ=90°   → pure self-tangent (maximizes own logit)
    θ=180°  → antipode −h
    θ=270°  → anti-tangent −t

  This is NOT a RoPE/positional rotation — no positional encoding is touched.
  The input is just a vocab-ordered token sequence; each position's h is
  rotated independently in its own {h, t} 2-plane.

GPU usage is reduced: only ONE batched hidden-state forward per chunk (tiny).
All heavy H·X matmuls run in CPU numpy BLAS. The (angle, vocab, token)
logit tensor is never materialized in full — it is built and reduced to
top-K in small angle batches to keep peak host RAM bounded.
"""
import sys, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2-1.5B-Instruct"
OUT = "/home/nicolas/model-harness/151k_states/chunks/rot"
KPER = 8
CHUNK = 128          # tokens processed per forward pass
ANG_BATCH = 16       # angles processed per top-k reduction (tune vs RAM)

ANGLES_DEG = np.arange(0, 360)
n_ang = len(ANGLES_DEG)
costh = np.cos(np.deg2rad(ANGLES_DEG)).astype(np.float32)
sinth = np.sin(np.deg2rad(ANGLES_DEG)).astype(np.float32)


def main(start, end):
    os.makedirs(OUT, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        MODEL, cache_dir="models", local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, cache_dir="models", local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda",
    ).eval()

    # Unembedding matrix, single CPU copy, shared across all chunks.
    H = model.lm_head.weight.detach().float().cpu().numpy()   # (V, d) fp32
    V = H.shape[0]
    c_t = torch.from_numpy(costh)   # (360,)
    s_t = torch.from_numpy(sinth)   # (360,)

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

        # Self-tangent: each token's own unembedding row, projected
        # perpendicular to its own hidden state, unit-normalized.
        selves = H[tids]                                      # (n, d)
        selves_n = selves / (np.linalg.norm(selves, axis=1, keepdims=True) + 1e-12)
        tn = selves_n - (selves_n * hn).sum(1, keepdims=True) * hn
        tn = tn / (np.linalg.norm(tn, axis=1, keepdims=True) + 1e-9)

        # HEAVY: two CPU BLAS matmuls, not on GPU.
        G0 = H @ h.T               # (V, n)  true logits
        G1 = H @ (tn * hnorm).T    # (V, n)  self-tangent logits

        G0t = torch.from_numpy(G0)
        G1t = torch.from_numpy(G1)

        # Reduce to top-K per angle in batches so we never hold the full
        # (n_ang, V, n) tensor in memory at once.
        top_chunks, log_chunks = [], []
        for a0 in range(0, n_ang, ANG_BATCH):
            a1 = min(a0 + ANG_BATCH, n_ang)
            c_b = c_t[a0:a1, None, None]                       # (b,1,1)
            s_b = s_t[a0:a1, None, None]
            L_b = c_b * G0t[None] + s_b * G1t[None]            # (b, V, n)
            tv, ti = torch.topk(L_b, KPER, dim=1)              # (b, KPER, n)
            top_chunks.append(ti.permute(2, 0, 1))             # (n, b, KPER)
            log_chunks.append(tv.permute(2, 0, 1))

        top_all = torch.cat(top_chunks, dim=1).numpy().astype(np.int32)     # (n, 360, KPER)
        log_all = torch.cat(log_chunks, dim=1).numpy().astype(np.float32)   # (n, 360, KPER)

        f = os.path.join(OUT, f"rot_scan_{s:07d}_{e:07d}.npz")
        np.savez(
            f,
            angles=ANGLES_DEG.astype(np.float32),
            top=top_all,
            logits=log_all,
            tids=tids,
        )
        print(f"  {os.path.basename(f)} ({n} tok, {n_ang} ang)", flush=True)


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 151643
    main(start, end)
    print("done")