"""eval_chord_recipe_probe.py — FAST: does the soft+duty steering recipe
(alpha~0.15, duty~3) generalize across ALL semantic classes?

Sweeps 6 classes x {recipe, hard-control} x 2 seeds x 16 tokens toward each
class's 5-note chord centroid. Reports family-occurrence, distinct-family,
diversity. ~30s.  Run: python3 eval_chord_recipe_probe.py"""
import math
import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from eval_practical_steering import gen
from multiverse_lab import chord_summary, hook_for, norm


def main():
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
    import multiverse_lab
    multiverse_lab.DEV = DEV

    model, tok = M.load_model('Qwen/Qwen2-0.5B-Instruct', dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])

    pids = tok("Once upon a time", add_special_tokens=False,
               return_tensors="pt").input_ids.to(model.device)
    li = model.config.num_hidden_layers - 1

    classes = {c: [w for w in words if w in word2id] for c, words in CLASSES.items()}
    cells = [("recipe-hard", 0.3, 1), ("recipe-soft", 0.15, 3)]
    for cls, words in classes.items():
        ids = np.array([word2id[w] for w in words[:5]])
        C, _, _ = chord_summary(ids, Wn)
        fam = set(words)
        for name, alpha, duty in cells:
            occs, distincts, divs = [], [], []
            for sd in range(2):
                text, _ = gen(model, tok, pids, 16, hook_fn=hook_for(C, alpha=alpha, layer=li),
                              layer=li, duty=duty, top_p=0.9, seed=sd)
                low = text.lower()
                occs.append(sum(low.count(w.lower()) for w in fam))
                distincts.append(sum(1 for w in fam if w in low))
                toks = tok(text, add_special_tokens=False).input_ids
                divs.append(len(set(toks)) / max(len(toks), 1))
                if sd == 0:
                    print(f"  {cls:7s} {name:13s} seed0: {text[:60]!r}")
            print(f"  {cls:7s} {name:13s} occ {np.mean(occs):4.1f}  "
                  f"distinct {np.mean(distincts):.1f}  div {np.mean(divs):.2f}")
        print()


if __name__ == "__main__":
    main()