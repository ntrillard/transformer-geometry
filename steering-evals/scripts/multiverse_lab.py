"""multiverse_lab.py — FAST chord-walking experiments on the sphere.

Every sphere layer is a "universe" with its own readout geometry. This lab
walks the hidden state toward chord centroids (5/7/10-note chords, semantic
vs random) and asks, per (layer, context, budget), the geometry questions in
milliseconds — then runs a few short REAL generations (single note vs chords)
to see what the walk *writes*. Each test completes in seconds by design, so it
can be re-run in a loop to recursively learn how the system behaves.

Run:  python3 multiverse_lab.py [--model Qwen/Qwen2-0.5B-Instruct]
                                   [--layers 0,8,16,23] [--chords 5,7,10]
                                   [--budgets 5,10,17,30] [--gen 1]
      --quick  = single context, chords only size 5, no generation (fastest)
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch

import steering_geometry_test as M
from steering_geometry_test_offarc import first_rank1_angle
from eval_chord_steering import CLASSES
from eval_practical_steering import gen

SEED = 0


def norm(x):
    return x / np.linalg.norm(x)


angle_cache = {}   # by note id: median first-rank1 angle over contexts (deepest layer)


def note_angle(ti, deepest, states, Wn, W):
    """Median analytic first-rank1 angle of a note across contexts at the
    deepest layer (cached: notes repeat across chord sizes)."""
    if ti in angle_cache:
        return angle_cache[ti]
    aa = [first_rank1_angle(norm(h), Wn[ti], W, int(ti))
          for h in (ll[deepest] for ll in states.values())]
    aa = [x for x in aa if x is not None]
    angle_cache[ti] = float(np.median(aa)) if aa else float('nan')
    return angle_cache[ti]


def chord_summary(ids, Wn):
    """Center (unit) of the chord's note rows + interference stats."""
    C = Wn[ids].mean(0)
    rho = float(np.linalg.norm(C))
    Cc = np.clip(Wn[ids] @ (C / rho), -1, 1)
    return C / rho, rho, float(np.degrees(np.arccos(Cc.mean())))

def chord_summary(ids, Wn):
    """Center (unit) of the chord's note rows + interference stats."""
    C = Wn[ids].mean(0)
    rho = float(np.linalg.norm(C))
    Cc = np.clip(Wn[ids] @ (C / rho), -1, 1)
    return C / rho, rho, float(np.degrees(np.arccos(Cc.mean())))


def hook_for(dn, alpha=0.3, layer=None):
    dn_t = torch.as_tensor(dn, device=DEV, dtype=torch.float32)

    def hook(module, inp, out):
        out2 = out.clone()
        h = out2[0, -1, :].float()
        hn = h / h.norm()
        g = dn_t - (dn_t @ hn) * hn
        g = g / max(g.norm().item(), 1e-8)
        h2 = h + alpha * h.norm() * g
        out2[0, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
        return out2

    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2-0.5B-Instruct')
    ap.add_argument('--layers', default='0,8,16,23')
    ap.add_argument('--chords', default='5,7,10')
    ap.add_argument('--budgets', default='5,10,17,30')
    ap.add_argument('--gen', type=int, default=1, help='generation cells (0|1)')
    ap.add_argument('--gen-tokens', type=int, default=16)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()

    global DEV
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
    LAYERS = [int(x) for x in a.layers.split(',')]
    SIZES = [int(x) for x in a.chords.split(',')]
    BUDGETS = [float(x) for x in a.budgets.split(',')]

    t0 = time.time()
    print(f"[multiverse] loading {a.model} on {DEV} ...")
    model, tok = M.load_model(a.model, dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    V = model.config.vocab_size
    W = W[:V]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    Wt = torch.as_tensor(W, device=DEV, dtype=torch.float32)
    print(f"[multiverse] loaded in {time.time()-t0:.1f}s (vocab {V})")

    # ---- note vocabulary -------------------------------------------------
    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    avail = {c: [w for w in words if w in word2id] for c, words in CLASSES.items()}
    print(f"[multiverse] note rows available: {len(word2id)}/60")

    rng = np.random.default_rng(SEED)
    chords = {}
    for size in SIZES:
        for cls, words in avail.items():
            ids = [word2id[w] for w in words[:size]]
            if len(ids) == size:
                chords[f'{cls}{size}'] = np.array(ids)
        for k in range(3):
            chords[f'rand{size}_{k}'] = rng.choice(V, size=size, replace=False)

    # ---- contexts (one forward pass each, ALL layers reused) --------------
    prompts = (["The capital of France is"] if a.quick else
               ["The capital of France is", "Once upon a time"])
    print(f"[multiverse] forward pass on {len(prompts)} contexts, "
          f"layers {LAYERS} ...")
    t1 = time.time()
    states = M.get_states(model, tok, prompts, LAYERS)
    print(f"[multiverse] states in {time.time()-t1:.1f}s")

    # ============ GEOMETRY SWEEP ============
    rows = []
    n_cells = 0
    t2 = time.time()
    for name, ids in chords.items():
        C, rho, spread = chord_summary(ids, Wn)
        is_real = not name.startswith('rand')
        # harmonic convergence: analytic (cached) first-rank1 angle per note
        angles = [note_angle(int(ti), LAYERS[-1], states, Wn, W) for ti in ids]
        # walk cells
        for ctx, llayer in states.items():
            for l, h in llayer.items():
                u = norm(h)
                tau = M.tangent_direction(u, C)
                for budget in BUDGETS:
                    if budget > 0:
                        v = M.rotate_toward(u, tau, math.radians(budget))
                        L = (torch.as_tensor(v, device=DEV, dtype=torch.float32) @ Wt.T)
                        L = L.cpu().float().numpy()
                    else:
                        L = u @ W.T
                    family = L[ids]
                    outsider = float(np.delete(L, ids).max())
                    chord_rank1 = bool(family.max() > outsider)
                    best_note = int(np.argmax(family))
                    margin = float(family.max() - outsider)
                    # tonic: is the winning note the most C-aligned one?
                    align = int(np.argmax(np.clip(Wn[ids] @ C, -1, 1)))
                    tonic = bool(best_note == align and chord_rank1)
                    rows.append(dict(chord=name, real=is_real, nsize=len(ids),
                                     ctx=ctx, layer=l, budget=budget,
                                     chord_rank1=chord_rank1, tonic=tonic,
                                     margin=margin,
                                     note_median_angle=float(np.nanmedian(angles))))
                    n_cells += 1
    print(f"[multiverse] geometry sweep: {n_cells} cells in {time.time()-t2:.1f}s")
    df = pd.DataFrame(rows)

    # ---- summaries --------------------------------------------------------
    for size in SIZES:
        d = df[(df.nsize == size) & (df.budget == 17)]
        rg = d[d.real]
        rr = d[~d.real]
        print(f"\n== chords of {size} notes @17deg ==")
        print(f"   semantic : chord-reach {rg.chord_rank1.mean()*100:5.1f}%  "
              f"tonic {rg.tonic.mean()*100:5.1f}%  margin {rg.margin.median():+.3f}")
        print(f"   random   : chord-reach {rr.chord_rank1.mean()*100:5.1f}%  "
              f"col margin {rr.margin.median():+.3f}")
        print(f"   note median first-rank1 angle: "
              f"{np.nanmedian([r for r in d[d.real].note_median_angle if np.isfinite(r)]):.1f} deg")

    # per-layer multiverse map @17
    print("\n== multiverse map: chord-reach@17 by layer ==")
    real = df[df.real & (df.budget == 17)]
    piv = real.pivot_table(index='layer', columns='chord', values='chord_rank1',
                           aggfunc='mean')
    print(piv.round(2).to_string())
    print("\n== random chords @17 by layer ==")
    rr = df[~df.real & (df.budget == 17)]
    print(rr.groupby('layer').chord_rank1.mean().round(2).to_string())

    # ============ GENERATION CELLS ============
    if a.gen:
        print(f"\n== generation cells ({a.gen_tokens} tokens, top_p .9, "
              f"alpha .3, seed0..1) ==")
        pids = tok("Once upon a time", add_special_tokens=False,
                   return_tensors="pt").input_ids.to(model.device)
        li = model.config.num_hidden_layers - 1
        food = {5: np.array([word2id[w] for w in avail['food'][:5]]),
                7: np.array([word2id[w] for w in avail['food'][:7]]),
                10: np.array([word2id[w] for w in avail['food'][:10]])}
        s_apple = Wn[word2id['apple']]
        fam_words = set(avail['food'])
        C5, _, _ = chord_summary(food[5], Wn)

        cells = [("note 'apple'", norm(s_apple), {'apple'}, 0.3, 1),

                 ("note 'apple' soft", norm(s_apple), {'apple'}, 0.15, 1),

                 ("chord food5", C5, fam_words, 0.3, 1),

                 ("chord food5 soft", C5, fam_words, 0.15, 1),

                 ("chord food5 duty2", C5, fam_words, 0.3, 2)]
        for name, tgt, family, alpha, duty in cells:
            occs, divs, distincts = [], [], []
            for sd in range(2):
                text, conf = gen(model, tok, pids, a.gen_tokens,

                                 hook_fn=hook_for(tgt, alpha=alpha, layer=li),

                                 layer=li, duty=duty, top_p=0.9, seed=sd)
                low = text.lower()
                occs.append(sum(low.count(w) for w in family))
                toks = tok(text, add_special_tokens=False).input_ids
                divs.append(len(set(toks)) / max(len(toks), 1))
                distincts.append(sum(1 for w in family if w in low))
                if sd == 0:
                    print(f"   {name:16s} seed0: {text[:72]!r}")
            print(f"   {name:16s} family-occ {np.mean(occs):4.1f}  "
                  f"distinct-family {np.mean(distincts):.1f}  "
                  f"diversity {np.mean(divs):.2f}")

    print(f"\n[multiverse] TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()