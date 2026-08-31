"""eval_final_only.py — FAST: does the FINAL-ONLY steering recipe generalize?
Chord injected ONLY at layer 23 (the writer) for all 6 classes, with a
topic-aligned prompt per class + 'Once upon a time' control; calibrated
against the plain (all-layers) baseline.

alpha .15, duty 3, 20 tokens, 2 seeds. ~50s.
Run: python3 eval_final_only.py"""
import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from eval_practical_steering import gen
from multiverse_lab import chord_summary

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ALIGNED = {'food': "For dinner I made", 'animal': "In the garden there was",
           'color': "The sky was painted", 'city': "The capital of France is",
           'nature': "In the forest", 'number': "The count went from"}


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
    avail = {c: [w for w in words if w in word2id] for c, words in CLASSES.items()}

    def steer_gen(prompt, C, fam, layers, alpha=0.15, duty=3, seeds=(0, 1), n=20):
        dn = torch.as_tensor(C, device=DEV, dtype=torch.float32)

        def lhook(l):
            def hook(mod, inp, out):
                out2 = out.clone()
                h = out2[:, -1, :].float()
                hn = h / h.norm()
                g = dn - (dn @ hn.transpose(0, 1)) * hn
                g = g / max(g.norm().item(), 1e-8)
                h2 = h + alpha * h.norm() * g
                out2[:, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
                return out2
            return hook

        occs, distincts, divs, txts = [], [], [], []
        for sd in seeds:
            hs = {l: model.model.layers[l].register_forward_hook(lhook(l)) for l in layers}
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

    print("== final-only (L23) vs plain (all layers): food5 chord, alpha .15, duty 3 ==")
    print(f"{'class':>7} {'recipe':>10} {'prompt':>26} {'occ':>5} {'dist':>5} {'div':>5}")
    for cls, words in avail.items():
        ids = np.array([word2id[w] for w in words[:5]])
        C, _, _ = chord_summary(ids, Wn)
        fam = set(words)
        for recipe, layers in (("final-only", [23]), ("plain", list(range(NL)))):
            for prompt in (ALIGNED[cls], "Once upon a time"):
                txts, occs, distincts, divs = steer_gen(prompt, C, fam, layers)
                print(f"{cls:>7} {recipe:>10} {prompt:>26} "
                      f"{np.mean(occs):5.1f} {np.mean(distincts):5.1f} {np.mean(divs):5.2f}")
            print(f"       seed0 ({recipe}, aligned): {txts[0][:52]!r}")
        print()


if __name__ == "__main__":
    main()