#!/usr/bin/env python3
"""H-Cone test: does decision-cell geometry predict steering success?

For each model: sample random printable targets x several prompt states
(final layer), and compute per pair:
  - theta_author: first rank-1 crossing along the target-tangent arc (analytic,
    17 deg budget)
  - theta_cell : shortest angle into the rank-1 decision cone (active-set NNLS)

Then correlate cross-family arc-reach (from cross_family_summary) against
median theta_cell and cell-reach rate.

Run: python eval_cone_geometry.py --models Qwen/Qwen2-0.5B-Instruct ...
"""
import argparse
import math

import numpy as np
import pandas as pd
import torch

import eval_boundary_instruments as B
import steering_geometry_test_offarc as X
import steering_geometry_test as M

OUT = Path("../steering_geometry_results") if False else None
from pathlib import Path
OUT = Path("../steering_geometry_results")

ARC_REACH = {  # from cross_family_summary.csv (t64 c2 seed 42)
    "Qwen/Qwen2-1.5B-Instruct": 99.2,
    "Qwen/Qwen2-0.5B-Instruct": 97.3,
    "openai-community/gpt2": 90.8,
    "HuggingFaceTB/SmolLM-135M-Instruct": 67.2,
    "EleutherAI/pythia-160m": 27.9,
}
BUDGET_DEG = 17.0


def printable_targets(tok, rng, vocab, n=64):
    ids = []
    for tid in rng.choice(np.arange(1000, min(vocab, 20000)), size=n * 6, replace=False):
        txt = tok.decode([int(tid)])
        if txt.strip() and all(32 <= ord(c) < 127 for c in txt) and len(txt) <= 6:
            ids.append(int(tid))
        if len(ids) >= n:
            break
    return sorted(set(ids))[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []; pair_rows = []
    for model_id, reach in ARC_REACH.items():
        print(f"\n=== {model_id} ===")
        model, tok = M.load_model(model_id, dtype="fp16")
        V = model.config.vocab_size
        W = model.lm_head.weight.detach().cpu().float().numpy()
        if W.size == 0 or W.shape[0] != V:
            W = model.get_input_embeddings().weight.detach().cpu().float().numpy()[:V]
        Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
        W_dev = torch.as_tensor(W, device=model.device, dtype=torch.float32)

        rng = np.random.default_rng(args.seed)
        tids = printable_targets(tok, rng, V, args.targets)
        L = model.config.num_hidden_layers
        states = X.get_states(model, tok, M.PROMPTS[:4], [L - 1])

        ta, tc, phis, both_a, both_c = [], [], [], [], []
        for p in M.PROMPTS[:4]:
            u = X.normalize(states[p][L - 1])
            cells, cellX = B.cone_angles(u, tids, W_dev)
            s_rows = Wn[tids]
            tau_all = s_rows - (s_rows @ u)[:, None] * u          # tangents
            tau_all /= np.linalg.norm(tau_all, axis=1, keepdims=True)
            d_all = cellX.cpu().numpy() - u[None, :]              # cell displacements
            dn = d_all / np.clip(np.linalg.norm(d_all, axis=1, keepdims=True), 1e-12, None)
            phi = np.degrees(np.clip((tau_all * dn).sum(1), -1, 1))
            for i, t in enumerate(tids):
                a = X.first_rank1_angle(u, Wn[t], W, t,
                                        max_delta=math.radians(BUDGET_DEG))
                ta.append(a); tc.append(float(cells[i])); phis.append(float(phi[i]))
                if a is not None and np.isfinite(cells[i]):
                    both_a.append(a); both_c.append(float(cells[i]))
        ta = np.array([x if x is not None else np.nan for x in ta], dtype=float)
        tc = np.array(tc)
        n = len(ta)
        rec = dict(
            model=model_id.split("/")[-1], n_pairs=n,
            arc_reach_known=reach,
            med_theta_cell=float(np.median(tc)),
            cell_reach=float(100 * (tc <= BUDGET_DEG).mean()),
            author_reach=float(100 * np.isfinite(ta).mean()),
            med_theta_author=float(np.nanmedian(ta)),
            corr_author_cell=float(pd.Series(both_a).corr(pd.Series(both_c))) if len(both_a) > 3 else np.nan,
            med_phi=float(np.median(phis)),
        )
        for j in range(len(tc)):
            pair_rows.append(dict(model=rec["model"], theta_cell=tc[j],
                                  theta_author=ta[j], phi=phis[j]))
        rows.append(rec)
        print(f"  pairs={n}  med_theta_cell={rec['med_theta_cell']:.2f}deg  "
              f"cell_reach<17deg={rec['cell_reach']:.1f}%  "
              f"author_reach={rec['author_reach']:.1f}%  "
              f"corr(a,c)={rec['corr_author_cell']}")
        del model
        torch.cuda.empty_cache()

    pd.DataFrame(pair_rows).to_csv(OUT / "cone_geometry_pairs.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cone_geometry_cross_family.csv", index=False)
    r_cell = np.corrcoef(df["med_theta_cell"], df["arc_reach_known"])[0, 1]
    r_rate = np.corrcoef(df["cell_reach"], df["arc_reach_known"])[0, 1]
    print("\n=== H-Cone cross-family correlation (n=5 models) ===")
    print(f"Pearson r(arc-reach, median theta_cell)  = {r_cell:+.3f}")
    print(f"Pearson r(arc-reach, cell-reach rate)    = {r_rate:+.3f}")
    print(df[["model", "arc_reach_known", "med_theta_cell", "cell_reach",
              "author_reach", "med_phi", "corr_author_cell"]].to_string(index=False))
    pp = pd.DataFrame(pair_rows)
    pp["reach"] = pp.theta_author.notna()
    import numpy as _np
    print("\nphi (route-cell misalignment): reachable vs unreachable")
    print(pp.groupby(["model","reach"])["phi"].median().unstack().round(2).to_string())


if __name__ == "__main__":
    main()
