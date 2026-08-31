"""multiverse_arch.py — ARCHITECTURE anatomy of Qwen2-0.5B's sphere.

Phase 1: per-layer anatomy — residual norm vs RMSNorm radius (paper's sphere
claim), angle of each layer's state to the food/color chord centers, linear
readout fidelity (how early the model has 'decided' the next token).

Phase 2: residual-stream decomposition — 'who writes the target token':
one forward with per-layer hooks splits h_l = h_emb + sum(attn_delta) +
sum(mlp_delta); project each component onto a chord direction to see whether
attention or MLP carries the topic, per layer.

FAST: one forward pass, numpy/GPU sweeps. Run: python3 multiverse_arch.py
"""
import math
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from multiverse_lab import chord_summary

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def norm(x):
    return x / np.linalg.norm(x)


def main():
    model, tok = M.load_model('Qwen/Qwen2-0.5B-Instruct', dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    V = model.config.vocab_size
    W = W[:V]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    Wt = torch.as_tensor(W, device=DEV, dtype=torch.float32)
    NL = model.config.num_hidden_layers
    print(f"[arch] Qwen2-0.5B: {NL} layers, dim {W.shape[1]}, vocab {V}")

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    avail = {c: [w for w in words if w in word2id] for c, words in CLASSES.items()}

    chords = {}
    for cls, words in avail.items():
        ids = np.array([word2id[w] for w in words[:5]])
        C, _, _ = chord_summary(ids, Wn)
        chords[cls] = (ids, C)

    prompt = "For dinner I made"
    print(f"[arch] prompt: {prompt!r}")

    # ---- one forward pass capturing per-layer hidden + component deltas ----
    deltas = {'attn': [], 'mlp': []}
    handles = []

    def make_hook(key):
        def hook(mod, inp, out):
            o = out[0] if (key == 'attn' and isinstance(out, tuple)) else out
            deltas[key].append(o.detach().float())
        return hook

    for l in range(NL):
        lay = model.model.layers[l]
        handles.append(lay.self_attn.register_forward_hook(make_hook('attn')))
        handles.append(lay.mlp.register_forward_hook(make_hook('mlp')))

    t0 = time.time()
    pid = tok(prompt, add_special_tokens=False,
              return_tensors='pt').input_ids.to(model.device)
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
    for h in handles:
        h.remove()
    hidden = [h[0, -1].float().cpu().numpy() for h in hid.hidden_states]   # emb + NL
    attn_d = torch.stack([d[0, -1] for d in deltas['attn']]).cpu().numpy()  # NL x dim
    mlp_d = torch.stack([d[0, -1] for d in deltas['mlp']]).cpu().numpy()
    print(f"[arch] forward + hooks in {time.time()-t0:.1f}s")

    # ---- sphere radius (RMSNorm gamma) per layer ----
    gammas = []
    for l in range(NL):
        g = model.model.layers[l].input_layernorm.weight.detach().cpu().float().numpy()
        gammas.append(float(np.linalg.norm(g)))
    gfin = model.model.norm.weight.detach().cpu().float().numpy()
    gammas[-1] = float(np.linalg.norm(gfin))

    # ---- readout fidelity + per-layer argmax ----
    U = np.stack([norm(h) for h in hidden[1:]])                     # NL x dim
    logits_all = (torch.as_tensor(U, device=DEV, dtype=torch.float32) @ Wt.T).cpu().numpy()
    final_arg = int(logits_all[-1].argmax())
    final_tok = tok.decode([final_arg], skip_special_tokens=True).strip()

    # ---- Phase 1 table ----
    print("\n== PHASE 1: per-layer anatomy ==")
    print(f"{'L':>3} {'||h||':>7} {'||g||':>7} {'h/g':>6} {'ang_food':>8} {'ang_col':>8} "
          f"{'argmax':>14} {'fid':>4}")
    for l in range(NL):
        h = hidden[l + 1]
        hn = norm(h)
        af = math.degrees(math.acos(np.clip(hn @ chords['food'][1], -1, 1)))
        ac = math.degrees(math.acos(np.clip(hn @ chords['color'][1], -1, 1)))
        tn = tok.decode([int(logits_all[l].argmax())], skip_special_tokens=True).strip()[:14]
        print(f"{l:>3} {float(np.linalg.norm(h)):7.1f} {gammas[l]:7.1f} "
              f"{float(np.linalg.norm(h)) / gammas[l]:6.2f} {af:8.1f} {ac:8.1f} "
              f"{tn:>14} {int(fidelity := (int(logits_all[l].argmax()) == final_arg)):>4}")
    print(f"final-layer argmax: {final_tok!r}  (reach@17 target food {avail['food'][:5]})")

    # ---- chord-reach@17 per layer ----
    print("\n== chord-reach@17 per layer (food5) ==")
    food_ids, foodC = chords['food']
    for l in range(NL):
        u = norm(hidden[l + 1])
        tau = M.tangent_direction(u, foodC)
        v = M.rotate_toward(u, tau, math.radians(17))
        L = v @ W.T
        fam = L[food_ids]
        outsider = float(np.delete(L, food_ids).max())
        top = tok.decode([food_ids[int(np.argmax(fam))]], skip_special_tokens=True).strip()
        print(f"  L{l:>2} reach={bool(fam.max() > outsider)}  "
              f"margin={float(fam.max() - outsider):+.3f}  top={top!r}")

    # ---- Phase 2: residual decomposition ----
    print("\n== PHASE 2: who writes the target? projections onto food-chord dir C ==")
    h_emb = hidden[0]
    C = chords['food'][1]
    h23 = hidden[NL]
    recon = h_emb + attn_d.sum(0) + mlp_d.sum(0)
    h_emb = hidden[0]
    C = chords['food'][1]
    h23 = hidden[NL]                    # POST-final-norm (what the head reads)
    h23pre = hidden[NL - 1] + attn_d[NL - 1] + mlp_d[NL - 1]   # pre-norm block-23 out
    recon = h_emb + attn_d.sum(0) + mlp_d.sum(0)
    print(f"   [check] (pre-norm space) ||h_23pre - (h_emb+sum_attn+sum_mlp)||/||h_23pre|| = ",
          f"{float(np.linalg.norm(h23pre - recon)) / float(np.linalg.norm(h23pre)):.3f}")
    cum_attn = np.zeros_like(h_emb)
    cum_mlp = np.zeros_like(h_emb)
    print(f"{'L':>3} {'attn·C':>10} {'mlp·C':>10} {'cum_attn':>9} {'cum_mlp':>9} {'h_l·C':>9}")
    for l in range(NL):
        cum_attn += attn_d[l]
        cum_mlp += mlp_d[l]
        print(f"{l:>3} {float(attn_d[l] @ C):10.3f} {float(mlp_d[l] @ C):10.3f} "
              f"{float(cum_attn @ C):9.3f} {float(cum_mlp @ C):9.3f} "
              f"{float(hidden[l + 1] @ C):9.3f}")
    print(f"emb·C = {float(h_emb @ C):.3f}   final h_23·C = {float(hidden[NL] @ C):.3f}")
    # NOTE: hidden[NL] is POST-final-RMSNorm (verified: == model.norm(block23_out),
    # rel-err 4e-4). The component decomposition above is exact in PRE-norm space;
    # the readout state the head sees is the norm-amplified image of it:
    pre23 = hidden[NL - 1] + attn_d[NL - 1] + mlp_d[NL - 1]
    print(f"   readout amplification: ||pre-norm h_23||={float(np.linalg.norm(pre23)):.1f} ",
          f"-> ||readout state||={float(np.linalg.norm(hidden[NL])):.1f}  ",
          f"(x{float(np.linalg.norm(hidden[NL]))/float(np.linalg.norm(pre23)):.2f})")
    print(f"   food projection: pre-norm {float(pre23 @ C):.2f} -> readout ",
          f"{float(hidden[NL] @ C):.2f}  ",
          f"(x{float(hidden[NL] @ C)/max(float(pre23 @ C),1e-9):.2f})")


if __name__ == "__main__":
    main()