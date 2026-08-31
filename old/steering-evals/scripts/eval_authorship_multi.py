"""eval_authorship_multi.py — FAST: authorship across 4 prompts + "who wrote the
winner" decision decomposition.

Table 1: per-class chord authorship (plant attn / plant mlp / write mlp /
final-mlp spike), mean +- std over prompts.

Table 2: the actual decision — final margin (winner - runner-up) split into
phase contributions (emb / attn-plant / attn-write / mlp-plant / mlp-write),
exactly linear in PRE-norm space. Shows which phase creates the winning margin,
per prompt.

~12s.  Run: python3 eval_authorship_multi.py"""
import time
from collections import defaultdict

import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from multiverse_lab import chord_summary

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = ["For dinner I made", "The recipe calls for",
           "In the forest", "The capital of France is"]


def main():
    model, tok = M.load_model('Qwen/Qwen2-0.5B-Instruct', dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    W = W[:model.config.vocab_size]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    NL = model.config.num_hidden_layers
    t0 = time.time()

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    avail = {c: [w for w in words if w in word2id] for c, words in CLASSES.items()}
    Cs = {}
    for cls, words in avail.items():
        ids = np.array([word2id[w] for w in words[:5]])
        C, _, _ = chord_summary(ids, Wn)
        Cs[cls] = C

    def capture(prompt):
        deltas = {'attn': [], 'mlp': []}
        handles = []
        for l in range(NL):
            lay = model.model.layers[l]
            handles.append(lay.self_attn.register_forward_hook(
                lambda m, i, o, k='attn': deltas[k].append(
                    (o[0] if isinstance(o, tuple) else o).detach().float())))
            handles.append(lay.mlp.register_forward_hook(
                lambda m, i, o, k='mlp': deltas[k].append(o.detach().float())))
        pid = tok(prompt, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        with torch.no_grad():
            hid = model(pid, output_hidden_states=True)
        for h in handles:
            h.remove()
        hs = [h[0, -1].float().cpu() for h in hid.hidden_states]
        ad = torch.stack([d[0, -1] for d in deltas['attn']]).cpu().float().numpy()
        md = torch.stack([d[0, -1] for d in deltas['mlp']]).cpu().float().numpy()
        return hs, ad, md

    caps = {p: capture(p) for p in PROMPTS}
    print(f"[captured {len(caps)} prompts in {time.time()-t0:.1f}s]")

    # ---- Table 1: per-class authorship mean+-std ----
    print("\n== TABLE 1: chord authorship across " + f"{len(PROMPTS)} prompts ==")
    print(f"{'class':>7} {'plant-attn':>18} {'plant-mlp':>16} {'write-mlp':>16} {'finalMLP':>10}")
    agg = defaultdict(list)
    for p, (hs, ad, md) in caps.items():
        for name, C in Cs.items():
            pa = float((ad[5:17] @ C).sum())
            pm = float((md[5:17] @ C).sum())
            wm = float((md[17:24] @ C).sum())
            fm = float(md[23] @ C)
            agg[name].append((pa, pm, wm, fm))
    for name, v in agg.items():
        v = np.array(v)
        m, s = v.mean(0), v.std(0)
        print(f"{name:>7} attn {m[0]:+6.2f}+-{s[0]:.2f}  mlp {m[1]:+6.2f}+-{s[1]:.2f}  "
              f"mlp {m[2]:+6.2f}+-{s[2]:.2f}  {m[3]:+6.2f}+-{s[3]:.2f}")

    # ---- Table 2: who wrote the winner (margin decomposition) ----
    print("\n== TABLE 2: the actual decision — winner vs runner-up ==")
    print(f"{'prompt':>26} {'winner':>12} {'runner':>12} {'margin':>7}  "
          f"{'emb':>6} {'attPl':>6} {'attWr':>6} {'mlpPl':>6} {'mlpWr':>6} {'sum':>6}")
    for p, (hs, ad, md) in caps.items():
        h_read = hs[24]                       # post-norm readout: what the model picks
        L = h_read @ W.T
        w_idx = int(np.argmax(L))
        r_idx = int(np.argmax(np.delete(L, w_idx)))
        if r_idx >= w_idx:
            r_idx += 1
        winner = tok.decode([w_idx]).strip()
        runner = tok.decode([r_idx]).strip()
        d = Wn[w_idx] - Wn[r_idx]             # decision direction
        h_pre = hs[23] + ad[23] + md[23]
        margin = float(h_pre @ d)             # pre-norm margin
        early = float(hs[0] @ d) + float(ad[0:5].sum(0) @ d) + float(md[0:5].sum(0) @ d)
        att_pl = float(ad[5:17].sum(0) @ d)
        att_wr = float(ad[17:24].sum(0) @ d)
        mlp_pl = float(md[5:17].sum(0) @ d)
        mlp_wr = float(md[17:24].sum(0) @ d)
        tot = early + att_pl + att_wr + mlp_pl + mlp_wr
        rel = abs(tot - margin) / max(abs(margin), 1e-9)
        print(f"{p:>26} {winner:>12} {runner:>12} {margin:7.3f}  ",
              f"{early:6.3f} {att_pl:6.3f} {att_wr:6.3f} {mlp_pl:6.3f} ",
              f"{mlp_wr:6.3f} {tot:6.3f} (rel {rel:.2f})")
    print("margin = pre-norm winner-vs-runner advantage; decomposition exact ",
          "(early+L0-4 + attn-plant + attn-write + mlp-plant + mlp-write)")
    print("margin = pre-norm winner-vs-runner advantage; sum of phase contributions "
          "reconstructs it (emb+attPl+attWr+mlpPl+mlpWr)")


if __name__ == "__main__":
    main()