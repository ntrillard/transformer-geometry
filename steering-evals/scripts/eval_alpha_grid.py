"""eval_alpha_grid.py — FAST: per-class alpha x duty grid for the two classes
that ESCAPED the soft recipe (food, nature) plus color as a held control.
Finds where the native 'Hannah Spearman' attractor loses. 16 tokens, 2 seeds,
~20s.  Run: python3 eval_alpha_grid.py"""
import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from eval_practical_steering import gen
from multiverse_lab import chord_summary, hook_for


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
    for cls in ('food', 'nature', 'color'):
        ids = np.array([word2id[w] for w in classes[cls][:5]])
        C, _, _ = chord_summary(ids, Wn)
        fam = set(classes[cls])
        print(f"== {cls} ==")
        for alpha, duty in ((0.15, 3), (0.2, 3), (0.25, 2), (0.3, 2)):
            occs, distincts, divs = [], [], []
            for sd in range(2):
                text, _ = gen(model, tok, pids, 16, hook_fn=hook_for(C, alpha=alpha, layer=li),
                              layer=li, duty=duty, top_p=0.9, seed=sd)
                low = text.lower()
                occs.append(sum(low.count(w.lower()) for w in fam))
                distincts.append(sum(1 for w in fam if w.lower() in low))
                toks = tok(text, add_special_tokens=False).input_ids
                divs.append(len(set(toks)) / max(len(toks), 1))
                if sd == 0:
                    print(f"   a{alpha:.2f} d{duty} seed0: {text[:58]!r}")
            print(f"   a{alpha:.2f} d{duty}  occ {np.mean(occs):4.1f}  "
                  f"distinct {np.mean(distincts):.1f}  div {np.mean(divs):.2f}")
        print()


if __name__ == "__main__":
    main()