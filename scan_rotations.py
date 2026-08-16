#!/usr/bin/env python3
"""Rotation-scan chunk extraction — VRAM-safe revision.

GPU usage is REDUCED to the minimum: only ONE batched hidden-state forward per
chunk (tiny). All the heavy H·X matmuls run in CPU numpy BLAS, so VRAM stays
~1.1GB and we never collide with the desktop compositor (the cause of the
earlier driver hang).

logits(θ) = cosθ·(H·v0) + sinθ·(H·tn)   (two matmuls once, then cheap mix)
"""
import sys, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2-1.5B-Instruct"
OUT = "/home/nicolas/model-harness/151k_states/chunks/rot"
KPER = 8
CHUNK = 128
ANGLES_DEG = np.arange(0, 360)   # full circle
n_ang = len(ANGLES_DEG)
costh = np.cos(np.deg2rad(ANGLES_DEG)).astype(np.float32)
sinth = np.sin(np.deg2rad(ANGLES_DEG)).astype(np.float32)

def main(start, end):
    os.makedirs(OUT, exist_ok=True)
    # load model for the forward only
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir="models",
                                        local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, cache_dir="models", local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda").eval()
    # head on CPU (single copy, shared)
    H = model.lm_head.weight.detach().float().cpu().numpy()      # (V,d) fp32
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)

    for s in range(start, end, CHUNK):
        e = min(s + CHUNK, end)
        tids = np.arange(s, e); n = len(tids)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([tids.tolist()], device="cuda"),
                        output_hidden_states=True)
        h = out.hidden_states[-1][0].float().cpu().numpy()       # (n,d)
        hnorm = np.linalg.norm(h, axis=1, keepdims=True) + 1e-9
        hn = h / hnorm
        hl = hn * hnorm

        selves = Hn[tids]
        tn = selves - (selves * hn).sum(1, keepdims=True) * hn
        tn = tn / (np.linalg.norm(tn, axis=1, keepdims=True) + 1e-9)

        # HEAVY: two numpy BLAS matmuls (CPU) — not on GPU
        G0 = H @ hl.T                       # (V,n)
        G1 = H @ (tn * hnorm).T            # (V,n)

        # vectorized: stack cos·G0+sin·G1 once, then one topk
        G0t = torch.from_numpy(G0)                       # (V,n)
        G1t = torch.from_numpy(G1)
        c = torch.from_numpy(costh); sn = torch.from_numpy(sinth)
        # L_all[ang,V,n]
        L_all = c[:, None, None]*G0t[None] + sn[:, None, None]*G1t[None]
        Lf = L_all.reshape(n_ang*V, n)                   # (360*V, n)
        tv, ti = torch.topk(Lf, KPER, dim=0)
        top_all = ti.T.reshape(n, n_ang, KPER).numpy().astype(np.int32)
        log_all = tv.T.reshape(n, n_ang, KPER).numpy().astype(np.float32)
        f = os.path.join(OUT, f"rot_scan_{s:07d}_{e:07d}.npz")
        np.savez(f, angles=ANGLES_DEG.astype(np.float32),
                 top=top_all, logits=log_all, tids=tids)
        print(f"  {os.path.basename(f)} ({n} tok, {n_ang} ang)", flush=True)

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end   = int(sys.argv[2]) if len(sys.argv) > 2 else 151643
    main(start, end)
    print("done")