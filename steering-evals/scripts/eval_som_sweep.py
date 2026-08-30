#!/usr/bin/env python3
"""S1-S4 basics: four questions, no semantic assumptions.

S1  Is the LM head tied to the input embedding?
    (if yes the semantic map is the embedding space, no head-specific claim)
S2  Does ANY lattice size rescue the global SOM?
    (quant-err pinned ~64 deg at 4x4..32x32 and a 1D ring = the data 1-NN scale;
     bigger grids only add empty neurons, one prototype keeps half the mass)
S3  Do label-free families resolve under inversion > center steering?
    (spherical k-means clusters; measured across the DIVERSE PROMPTS list below
     to check the law is not prompt-dependent)
S4  Do geometric NN token pairs behave alike?
    (logit correlation across all diverse prompts, NN vs random)
S5  Is the vocabulary EQUATORIAL to the BOS axis?
    (median token-row angle to the position-0 final-layer state; equatorial
      ~90 deg predicts cheap steering reach, polar ~20 deg predicts hard)

Run:  python eval_som_sweep.py [model]    (~15 s/model on the 3080)
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2-0.5B-Instruct'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N = 20000                    # head rows sampled for the SOM / k-means fits

PROMPTS = [
    # facts
    'The capital of France is', 'In physics, gravity is',
    'The chemical symbol for water is',
    # stories / openers
    'Once upon a time', 'In a land far away,', 'The old man opened the door and',
    # questions
    'Tell me something interesting:', 'What is the meaning of life?',
    'Can you explain quantum mechanics?',
    # instructions
    'Please summarize the following text:', 'Translate this sentence into French:',
    'Write a poem about',
    # code / structured
    'def fibonacci(n):', 'if x > 0: print(', 'import numpy as np',
    'SELECT * FROM users WHERE',
    # numbers
    'In the year 3000, humans will', 'The answer is 42 because',
    # other languages
    'La capitale de la France est', 'Hallo, wie geht es dir?',
    '北京的天气怎么样？', '今日はいい天気ですね',
    # register edges
    'For dinner I made', 'Dear Sir or Madam,', 'The quick brown fox jumps over',
    # degenerate 1-token contexts
    '.', ',', '?', ')',
]


def som_fit(X, P0, G):
    """Spherical batch-SOM, Gaussian grid neighborhood (same update as before)."""
    Xt = torch.as_tensor(X, device=DEV)
    Gt = torch.as_tensor(G, device=DEV)
    Pn = torch.as_tensor(P0, device=DEV)
    for ep in range(6):
        sigma = max(3.0 * (1 - ep / 6) + 0.4, 0.4)   # same schedule as the original fit
        Xt = Xt[torch.randperm(N, device=DEV)]
        for b0 in range(0, N, 512):
            Xb = Xt[b0:b0 + 512]
            bmu = (Xb @ Pn.T).argmax(1)
            d2 = ((Gt[:, None] - Gt[bmu][None]) ** 2).sum(-1)   # (P, B)
            K = torch.exp(-d2 / (2 * sigma ** 2)).T.float()
            Pn = (K.T @ Xb) / K.sum(0).clamp_min(1e-6)[:, None]
            Pn /= Pn.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return Pn


def kmeans_fit(X, K=30, iters=20):
    """Spherical k-means over the sampled rows (label-free families)."""
    Xt = torch.as_tensor(X, device=DEV)
    P = Xt[torch.randperm(N, device=DEV)[:K]].clone()
    for _ in range(iters):
        bmu = (Xt @ P.T).argmax(1)
        for k in range(K):
            m = bmu == k
            if m.sum():
                P[k] = Xt[m].mean(0)
        P /= P.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return P, bmu


def _embed(model):
    """Locate the input-embedding table across architectures."""
    for path in ('model.embed_tokens', 'transformer.wte',
                 'gpt_neox.embed_in'):
        obj = model
        ok = True
        for a in path.split('.'):
            obj = getattr(obj, a, None)
            if obj is None:
                ok = False; break
        if ok:
            return obj
    return None

def s1_tying(model):
    head = getattr(model, 'lm_head', None)
    embed = _embed(model)
    if head is None or embed is None:
        print("S1  (no lm_head / embed table found) - tied check n/a")
        return
    tied = head.weight.data_ptr() == embed.weight.data_ptr()
    cfg = getattr(model.config, 'tie_word_embeddings', '?')
    print(f"S1  lm_head is {'TIED to embed' if tied else 'SEPARATE'} ",
          f"(tie_word_embeddings={cfg})")
    if tied:
        r = torch.cosine_similarity(head.weight[0], embed.weight[0], dim=0)
        print(f"    (first-row cosine {r:.6f} = same memory)")


def s2_sweep(Wn):
    rng = np.random.default_rng(7)
    V = Wn.shape[0]
    X = Wn[rng.choice(V, N, replace=False)]
    Xt = torch.as_tensor(X, device=DEV)
    # data 1-NN scale: the intrinsic angular resolution no tiling can beat
    Sm = torch.as_tensor(Wn[rng.choice(V, 400, replace=False)], device=DEV)
    c = (Sm @ Sm.T).clamp(-1, 1)
    c.fill_diagonal_(-1)
    data_nn = math.degrees(torch.acos(c.max(1).values).mean().item())
    print(f"S2  data 1-NN scale = {data_nn:.1f} deg (quant-err can't beat it)")
    print("    side  neurons  quant-err(deg)  empty%  max-memb%")
    for side in (4, 8, 16, 24, 32):
        gy, gx = np.meshgrid(np.arange(side), np.arange(side))
        G = np.stack([gy.ravel(), gx.ravel()], 1).astype(np.float32)
        Pn = som_fit(X, Wn[rng.choice(V, side * side, replace=False)], G)
        C = Xt @ Pn.T
        qe = math.degrees(torch.acos(C.clamp(-1, 1).max(1).values).mean().item())
        bmu = C.argmax(1)
        empty = (side * side - len(bmu.unique())) / (side * side) * 100
        maxm = bmu.bincount().max().item() / N * 100
        print(f"    {side:4d}  {side*side:6d}     {qe:5.1f}     {empty:5.1f}  {maxm:6.1f}")
    # 1D ring: even spread at the SAME error => 2D topology is not the issue
    ring = np.linspace(0, 2 * math.pi, 256, endpoint=False)
    G = np.stack([np.cos(ring), np.sin(ring)], 1).astype(np.float32)
    Pn = som_fit(X, Wn[rng.choice(V, 256, replace=False)], G)
    C = Xt @ Pn.T
    qe = math.degrees(torch.acos(C.clamp(-1, 1).max(1).values).mean().item())
    maxm = C.argmax(1).bincount().max().item() / N * 100
    print(f"    1D-ring 256  qe {qe:5.1f} deg  empty 0.0  max-memb {maxm:5.1f}%")


def s3_autocluster(Wn, model, tok):
    """Label-free families: inversion steering vs center steering."""
    rng = np.random.default_rng(3)
    V = Wn.shape[0]
    X = Wn[rng.choice(V, N, replace=False)]
    _, bmu = kmeans_fit(X)
    bmu_c = bmu.cpu()
    sizes = torch.bincount(bmu_c, minlength=30)
    big = [int(k) for k in torch.argsort(-sizes)[:5] if sizes[k] >= 4]
    print(f"S3  label-free families (spherical k-means k=30): {len(big)} usable, "
          f"sizes {int(min(sizes[big]))}-{int(max(sizes[big]))}")
    prompts = ['The capital of France is', 'Once upon a time',
               'Tell me something interesting:', 'To bake sourdough bread']
    li = model.config.num_hidden_layers - 1
    states = []
    for p in PROMPTS:
        pid = tok(p, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        if pid.numel() == 0:
            continue
        with torch.no_grad():
            h = model(pid, output_hidden_states=True).hidden_states[li + 1][0, -1]
        states.append(h.cpu().float().numpy())
    n_prompts = len(states)
    tot_c = tot_i = 0
    per_prompt_c = np.zeros(n_prompts)
    per_prompt_i = np.zeros(n_prompts)
    fams = []
    for k in big:
        fam = torch.where(bmu_c == k)[0].numpy()[:10]
        rows = Wn[fam]
        center = rows.mean(0)
        center /= np.linalg.norm(center)
        spread = float(np.degrees(np.arccos(np.clip(rows @ center, -1, 1)).mean()))
        rc = ri = 0
        for j, h in enumerate(states):
            u = h / np.linalg.norm(h)
            tau = M.tangent_direction(u, center)
            L = M.rotate_toward(u, tau, math.radians(17)) @ Wn.T
            okc = L[fam].max() > np.delete(L, fam).max()
            best = fam[int(np.argmax(rows @ u))]
            tau = M.tangent_direction(u, Wn[best])
            L = M.rotate_toward(u, tau, math.radians(17)) @ Wn.T
            oki = L[fam].max() > np.delete(L, fam).max()
            rc += okc; ri += oki
            per_prompt_c[j] += okc; per_prompt_i[j] += oki
        fams.append((k, len(fam), spread, rc, ri))
        tot_c += rc; tot_i += ri
    n_cells = len(big) * n_prompts
    print(f"    {n_prompts} diverse prompts (facts/story/Q/code/CJK/register/1-token)")
    print(f"    overall cells (family x prompt):  center {tot_c/n_cells*100:5.1f}%   ",
          f"inversion {tot_i/n_cells*100:5.1f}%")
    print("    per-family (averaged over prompts):")
    for k, sz, sp, rc, ri in fams:
        print(f"      fam {k:3d} size {sz:3d}  spread {sp:5.1f} deg  ",
              f"center {rc/n_prompts*100:5.1f}%  inversion {ri/n_prompts*100:5.1f}%")
    worst = per_prompt_i.min()
    all_fams = int((per_prompt_i == len(big)).sum())
    beats = int((per_prompt_i >= per_prompt_c).sum())
    print(f"    across prompts: inversion resolves ALL {len(big)} families on {all_fams}/{n_prompts} ",
          f"prompts; worst prompt resolves {worst:.0f}/{len(big)}; ",
          f"inversion >= center on {beats}/{n_prompts} prompts")


def s4_interchangeability(Wn, model, tok):
    rng = np.random.default_rng(11)
    V = Wn.shape[0]
    cand = []
    for tid in rng.choice(V, 4000, replace=False):          # printable single tokens
        t = tok.decode([int(tid)], skip_special_tokens=True)
        if t and t.strip() and all(32 <= ord(c) < 127 for c in t):
            cand.append(int(tid))
        if len(cand) >= 40:
            break
    cand = np.array(cand)
    li = model.config.num_hidden_layers - 1
    H = []
    for p in PROMPTS:
        pid = tok(p, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        if pid.numel() == 0:
            continue
        with torch.no_grad():
            H.append(model(pid, output_hidden_states=True)
                     .hidden_states[li + 1][0, -1].cpu().float().numpy())
    H = np.stack(H)                                          # (n_prompts, d)
    Wc = Wn[cand].copy()
    # de-pole: project the candidate rows off the BOS axis (for polar models
    # the shared pole component inflates every correlation to ~1.0 otherwise)
    u = _bos_axis(model, tok)
    Wc -= (Wc @ u)[:, None] * u[None, :]
    L = H @ Wc.T                                             # logits (n_prompts, 40)
    Cmat = np.clip(Wn[cand] @ Wn[cand].T, -1, 1)
    np.fill_diagonal(Cmat, -1)
    nn_id = Cmat.argmax(1)
    corr_nn = np.array([np.corrcoef(L[:, i], L[:, nn_id[i]])[0, 1]
                        for i in range(len(cand))])
    corr_rand = np.array([np.corrcoef(L[:, i], L[:, rng.integers(len(cand))])[0, 1]
                          for i in range(len(cand))])
    print(f"S4  logit correlation across {len(PROMPTS)} diverse prompts, {len(cand)} tokens:")
    print(f"    (t, NN(t))  mean {corr_nn.mean():+.3f}   ",
          f"(t, random)  mean {corr_rand.mean():+.3f}  med {np.median(corr_rand):+.3f}")
    print(f"    NN beats random on {np.mean(corr_nn > np.median(corr_rand)):.1%} of tokens")
def _bos_axis(model, tok):
    """Normalized position-0 final-layer hidden state (the latitude/BOS axis)."""
    pid = tok('Once upon a time', add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    li = model.config.num_hidden_layers - 1
    with torch.no_grad():
        h = model(pid, output_hidden_states=True).hidden_states[li + 1][0, 0]
    return (h / h.norm()).cpu().float().numpy()


def s5_equator(Wn, model, tok):
    """Median token-row angle to the BOS/position-0 axis (the E1 law)."""
    u = _bos_axis(model, tok)
    angs = np.degrees(np.arccos(np.clip(Wn @ u, -1, 1)))
    print(f"S5  token-row angle to BOS axis:  med {np.median(angs):.1f} deg  ",
          f"(p10 {np.percentile(angs,10):.1f}, p90 {np.percentile(angs,90):.1f})")
    print(f"    -> {'EQUATORIAL (semantic map ~ longitude; steering cheap)' if np.median(angs) > 70
          else 'POLAR (semantics near the latitude axis; steering hard)'}")


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    s1_tying(model)
    if getattr(model, 'lm_head', None) is not None:
        W = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W = model.gpt_neox.embed_out.weight.detach().cpu().float().numpy()
    Wn = W[:model.config.vocab_size]
    Wn /= np.linalg.norm(Wn, axis=1, keepdims=True)
    print()
    s2_sweep(Wn)
    print()
    s3_autocluster(Wn, model, tok)
    print()
    s4_interchangeability(Wn, model, tok)
    print()
    s5_equator(Wn, model, tok)
    print(f"\n[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()