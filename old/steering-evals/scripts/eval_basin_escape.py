#!/usr/bin/env python3
"""eval_basin_escape.py — FAST: steer AWAY from the absorbing basin.

For each prompt: find the continuation basin azimuth (probability-weighted
azimuth of top-15 continuation in the ring plane) and the state's own
azimuth. Rotate the state in the ring plane AWAY from the basin (tangent t
with t.basin < 0) by alpha. Then FREE-RUN generation (no further steering).
If the basin is what traps Gemma's output ('I I I I', fairy-tale tokens),
escaping it should give natural content with lower 4-rep / higher diversity.

Sweep alpha in {0.1, 0.2, 0.3, 0.45}.

Run: timeout 90 python3 -u eval_basin_escape.py Qwen/Qwen2-0.5B-Instruct
     timeout 90 python3 -u eval_basin_escape.py google/gemma-3-1b-it
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2-0.5B-Instruct'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = ['Once upon a time', 'For dinner I made']
ALPHAS = [0.1, 0.2, 0.3, 0.45]
TOP = 15
NTOK = 24
SEEDS = 2


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight

    word2id = {}
    for cls, words in CLASSES.items():
        for w in words:
            ids = tok(' ' + w, add_special_tokens=False).input_ids
            if len(ids) == 1:
                word2id[w] = int(ids[0])
    words = list(word2id)
    idxs = np.array([word2id[w] for w in words])
    N = len(words)

    rows = lm_w[idxs].detach().float().cpu().numpy()
    Wn = rows * (1.0 / np.sqrt(np.einsum('ij,ij->i', rows, rows)[:, None] + 1e-12))

    # ring plane from word rows
    pid0 = tok('Once upon a time', add_special_tokens=False,
               return_tensors='pt').input_ids.to(DEV)
    li = model.config.num_hidden_layers - 1
    with torch.no_grad():
        hid = model(pid0, output_hidden_states=True)
        h0 = hid.hidden_states[li + 1][0, 0]
    u = (h0 / h0.norm()).cpu().float().numpy()

    def depole(v):
        v = v - (v @ u) * u
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    cents = np.stack([depole(Wn[i]) for i in range(N)])
    C0 = cents - cents.mean(0)
    _, Vp = np.linalg.eigh(C0.T @ C0)
    B = Vp[:, -2:]

    def az_of(v):
        return np.degrees(np.arctan2(v @ B[:, 1], v @ B[:, 0])) % 360

    def cir_mean(azs, ws):
        x = np.sum(ws * np.cos(np.radians(azs)))
        y = np.sum(ws * np.sin(np.radians(azs)))
        return np.degrees(np.arctan2(y, x)) % 360

    def basin_az(h):
        """probability-weighted azimuth of top-15 continuation from state h."""
        with torch.no_grad():
            lg = (torch.as_tensor(h, dtype=torch.float32, device=DEV)
                  @ lm_w.float().T)
            pr = torch.softmax(lg, dim=0)
            topv, topi = torch.topk(pr, TOP)
        proj = lm_w[topi].detach().float().cpu().numpy()
        proj = proj * (1.0 / np.sqrt(np.einsum('ij,ij->i', proj, proj)[:, None] + 1e-12))
        azs = np.array([az_of(depole(r)) for r in proj])
        return cir_mean(azs, topv.cpu().numpy())

    def state_az(h):
        return az_of(depole(h))

    def escape_dir(h, basin_a):
        """unit ring-plane tangent AWAY from basin, perpendicular to h."""
        sp = depole(h)
        rp = sp @ B
        rp = rp / (np.linalg.norm(rp) + 1e-12)
        tang = np.array([-rp[1], rp[0]])
        ba = np.deg2rad(basin_a)
        bb = np.array([np.cos(ba), np.sin(ba)])
        if tang @ bb > 0:
            tang = -tang
        tau = B @ tang
        return tau / (np.linalg.norm(tau) + 1e-12)

    def gen_from(alpha, tau=None, seed=0, n=NTOK, top_p=0.9, temp=1.0):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        toks = []
        for step in range(n):
            if step == 0 and alpha > 0:
                tauv = torch.as_tensor(tau, dtype=torch.float32, device=DEV)
                def hook(mod, inp, out):
                    out2 = out.clone()
                    v = out2[:, -1, :].float()
                    vn = v / v.norm()
                    g = tauv - (tauv @ vn.transpose(0, 1)) * vn
                    g = g / (g.norm() + 1e-8)
                    v2 = vn * math.cos(alpha) + g * math.sin(alpha)
                    out2[:, -1, :] = (v.norm() * v2).to(out.dtype)
                    return out2
                hd = model.model.norm.register_forward_hook(hook)
            else:
                hd = None
            try:
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
            finally:
                if hd is not None:
                    hd.remove()
            p = torch.softmax(L / temp, dim=0)
            q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
            keep = order[:int((cum <= top_p).sum()) + 1]
            m = torch.zeros_like(q); m[keep] = 1
            q = (q * m) / (q * m).sum()
            nxt = int(torch.multinomial(q, 1))
            toks.append(int(nxt))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return toks

    def rep4(toks):
        if len(toks) < 8:
            return 1.0
        n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])

    print(f"[{MODEL}]")
    for PROMPT in PROMPTS:
        pid = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        with torch.no_grad():
            hid = model(pid, output_hidden_states=True)
        h_base = hid.hidden_states[li + 1][0, -1].float().cpu().numpy()
        h_base = h_base / np.linalg.norm(h_base)
        basin = basin_az(h_base)
        saz = state_az(h_base)
        print(f"\n[prompt {PROMPT!r}] state az {saz:.0f}  basin az {basin:.0f} "
              f"(delta {min((saz - basin) % 360, (basin - saz) % 360):.0f} deg)")

        # native first
        reps, divs = [], []
        for sd in range(SEEDS):
            toks = gen_from(0.0, seed=sd, temp=1.3)
            reps.append(rep4(toks))
            divs.append(len(set(toks)) / len(toks))
        print(f"  {'native':>9}  4rep {np.mean(reps):.2f}  div {np.mean(divs):.2f}  "
              f"{tok.decode(gen_from(0.0, seed=0, temp=1.3))[:48]!r}")
        for alpha in ALPHAS:
            tau = escape_dir(h_base, basin)
            reps, divs, samples = [], [], []
            for sd in range(SEEDS):
                toks = gen_from(alpha, tau=tau, seed=sd, temp=1.3)
                reps.append(rep4(toks))
                divs.append(len(set(toks)) / len(toks))
                samples.append(tok.decode(toks)[:48])
            print(f"  away {alpha:>5.2f}  4rep {np.mean(reps):.2f}  "
                  f"div {np.mean(divs):.2f}  {samples[0]!r}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()