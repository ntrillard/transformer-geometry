"""eval_phase_steering.py — FAST: is the plant/write decoupling a USABLE lever?
Steer the food chord at DIFFERENT phases (plant L5-16 / write L17-23 / final
L23-only / all-layers cumulative) and measure topic-occurrence of the
generation. Compares against the plain chord-hook baseline (attractor escape).

Per-phase injection: add alpha * ||h|| * tangent(C) to the PRE-norm residual
each layer in the phase (recipe: alpha .15, duty 3). 20 tokens, 2 seeds,
2 prompts. ~35s.  Run: python3 eval_phase_steering.py"""
import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from eval_practical_steering import gen
from multiverse_lab import chord_summary

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    import multiverse_lab
    multiverse_lab.DEV = DEV
    model, tok = M.load_model('Qwen/Qwen2-0.5B-Instruct', dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    NL = model.config.num_hidden_layers

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    fam = set(CLASSES['food'])
    food = np.array([word2id[w] for w in CLASSES['food'] if w in word2id][:5])
    C, _, _ = chord_summary(food, Wn)

    # one forward hook per selected layer; gen's own hook_fn unused (layer=None)
    def run2(prompt, phases, alpha=0.15, duty=3, seeds=(0, 1), n=20):
        occs, distincts, divs, txts = [], [], [], []
        dn = torch.as_tensor(C, device=DEV, dtype=torch.float32)

        def make_hooks():
            hs = {}

            def lhook(l):
                def hook(mod, inp, out):
                    out2 = out.clone()
                    h = out2[:, -1, :].float()        # last-position residual is pre-norm
                    hn = h / h.norm()
                    dn2 = dn.unsqueeze(0)
                    g = dn2 - (dn2 @ hn.transpose(0, 1)) * hn
                    g = g / max(g.norm().item(), 1e-8)
                    h2 = h + alpha * h.norm() * g
                    out2[:, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
                    return out2
                return hook
            for l in phases:
                hs[l] = model.model.layers[l].register_forward_hook(lhook(l))
            return hs

        for sd in seeds:
            hs = make_hooks()
            try:
                text, _ = gen(model, tok,
                              tok(prompt, add_special_tokens=False,
                                  return_tensors='pt').input_ids.to(model.device),
                              n, layer=None, duty=duty, top_p=0.9, seed=sd)
            finally:
                for h in hs.values():
                    h.remove()
            txts.append(text)
            low = text.lower()
            occs.append(sum(low.count(w.lower()) for w in fam))
            distincts.append(sum(1 for w in fam if w.lower() in low))
            toks = tok(text, add_special_tokens=False).input_ids
            divs.append(len(set(toks)) / max(len(toks), 1))
        return txts, occs, distincts, divs

    recipes = {
        'plain-hook (cumulative)': list(range(NL)),
        'plant-only (5-16)': list(range(5, 17)),
        'write-only (17-23)': list(range(17, 24)),
        'final-only (23)': [23],
    }
    print("== phase steering of food5 chord (alpha .15, duty 3, 20 tok) ==")
    for name, phases in recipes.items():
        for prompt in ("For dinner I made", "Once upon a time"):
            txts, occs, distincts, divs = run2(prompt, phases)
            print(f"  {name:26s} {prompt:22s} occ {np.mean(occs):4.1f}  "
                  f"distinct {np.mean(distincts):.1f}  div {np.mean(divs):.2f}")
            print(f"      seed0: {txts[0][:56]!r}")
        print()


if __name__ == "__main__":
    main()