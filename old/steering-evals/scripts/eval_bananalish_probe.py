"""eval_bananalish_probe.py — FAST dose-response: how duty (fraction of
steered steps) shapes generation. Steer toward the food5 chord centroid at
alpha=0.3, top_p=0.9, 20 tokens, sweep duty 1..4 x 2 seeds. ~10s.
Run: python3 eval_bananalish_probe.py"""
import numpy as np
import torch
import steering_geometry_test as M
from eval_chord_steering import CLASSES
from eval_practical_steering import gen
from multiverse_lab import chord_summary, hook_for, norm

from multiverse_lab import chord_summary, hook_for, norm
import multiverse_lab

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
multiverse_lab.DEV = DEV


def main():
    model, tok = M.load_model('Qwen/Qwen2-0.5B-Instruct', dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    food = [word2id[w] for w in CLASSES['food'] if w in word2id][:5]
    fam = set(CLASSES['food'])
    C, _, _ = chord_summary(np.array(food), Wn)

    pids = tok("Once upon a time", add_special_tokens=False,
               return_tensors="pt").input_ids.to(model.device)
    li = model.config.num_hidden_layers - 1

    for duty in (1, 2, 3, 4):
        for sd in range(2):
            text, conf = gen(model, tok, pids, 20, hook_fn=hook_for(C, alpha=0.3, layer=li),
                             layer=li, duty=duty, top_p=0.9, seed=sd)
            low = text.lower()
            occ = sum(low.count(w) for w in fam)
            distinct = sum(1 for w in fam if w in low)
            toks = tok(text, add_special_tokens=False).input_ids
            div = len(set(toks)) / max(len(toks), 1)
            print(f"duty={duty} seed{sd}  occ={occ:2d} distinct={distinct} div={div:.2f}  {text[:58]!r}")
        print()


if __name__ == "__main__":
    main()