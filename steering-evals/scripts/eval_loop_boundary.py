"""eval_loop_boundary.py — FAST: two sharp tests of the difficulty->loop law.

Test 1 (loop boundary): food (hard class) vs color/number (easy classes),
final-only injection, sweep alpha 0.08..0.30. PREDICTION: easy classes loop at
every alpha (native == note); hard class has a non-loop window.

Test 2 (quantitative predictor): chord-reach specifically AT the steering site
(layer 23) for every class, correlated with the observed final-only div gain.
PREDICTION: reach@23 explains the div gain (lower reach@23 -> higher recovery).

aligned prompt per class, duty 3, 16 tokens, 2 seeds. ~70s.
Run: python3 eval_loop_boundary.py"""
import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from eval_practical_steering import gen
from multiverse_lab import chord_summary

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ALIGNED = {'food': "For dinner I made", 'color': "The sky was painted",
           'number': "The count went from"}


def main():
    import multiverse_lab
    multiverse_lab.DEV = DEV
    model, tok = M.load_model('Qwen/Qwen2-0.5B-Instruct', dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    Wt = torch.as_tensor(W, device=DEV, dtype=torch.float32)
    NL = model.config.num_hidden_layers

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    avail = {c: [w for w in words if w in word2id] for c, words in CLASSES.items()}

    # ---- reach@23 for every class: the quantitative predictor ----
    pid = tok("For dinner I made", add_special_tokens=False,
              return_tensors='pt').input_ids.to(model.device)
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
    u23 = hid.hidden_states[23][0, -1].float().cpu().numpy()
    u23 = u23 / np.linalg.norm(u23)
    print("== chord-reach AT layer 23 (the steering site), budget 17 ==")
    reach23 = {}
    for cls, words in avail.items():
        ids = np.array([word2id[w] for w in words[:5]])
        C, _, _ = chord_summary(ids, Wn)
        tau = M.tangent_direction(u23, C)
        v = M.rotate_toward(u23, tau, np.radians(17))
        L = (torch.as_tensor(v, device=DEV, dtype=torch.float32) @ Wt.T).cpu().numpy()
        rank1 = bool(L[ids].max() > np.delete(L, ids).max())
        margin = float(L[ids].max() - np.delete(L, ids).max())
        reach23[cls] = (rank1, margin)
        print(f"  {cls:>7} reach@23={str(rank1):>5}  margin={margin:+.4f}")

    # ---- Test 1: loop boundary ----
    def steer_gen(prompt, C, fam, alpha, n=16, seeds=(0, 1), duty=3):
        dn = torch.as_tensor(C, device=DEV, dtype=torch.float32)

        def hook(mod, inp, out):
            out2 = out.clone()
            h = out2[:, -1, :].float()
            hn = h / h.norm()
            g = dn - (dn @ hn.transpose(0, 1)) * hn
            g = g / max(g.norm().item(), 1e-8)
            h2 = h + alpha * h.norm() * g
            out2[:, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
            return out2

        occs, distincts, divs, txts = [], [], [], []
        for sd in seeds:
            hd = model.model.layers[23].register_forward_hook(hook)
            try:
                text, _ = gen(model, tok,
                              tok(prompt, add_special_tokens=False,
                                  return_tensors='pt').input_ids.to(model.device),
                              n, layer=None, duty=duty, top_p=0.9, seed=sd)
            finally:
                hd.remove()
            txts.append(text)
            low = text.lower()
            occs.append(sum(low.count(w.lower()) for w in fam))
            distincts.append(sum(1 for w in fam if w.lower() in low))
            toks = tok(text, add_special_tokens=False).input_ids
            divs.append(len(set(toks)) / max(len(toks), 1))
        return txts, np.mean(occs), np.mean(distincts), np.mean(divs)

    print("\n== Test 1: loop boundary (final-only, duty 3, aligned prompt) ==")
    print(f"{'class':>7} {'reach@23':>8} " +
          " ".join(f"a{a:<4}" for a in (0.08, 0.12, 0.15, 0.2, 0.25, 0.3)))
    for cls in ('food', 'color', 'number'):
        ids = np.array([word2id[w] for w in avail[cls][:5]])
        C, _, _ = chord_summary(ids, Wn)
        fam = set(avail[cls])
        row = []
        for a in (0.08, 0.12, 0.15, 0.2, 0.25, 0.3):
            txts, occ, dist, div = steer_gen(ALIGNED[cls], C, fam, a)
            loop = div < 0.45
            row.append(f"{div:.2f}{'L' if loop else ' '}")
        print(f"{cls:>7} {str(reach23[cls][0]):>8}  " + "  ".join(
            f"{a:<5}" for a in row))
    print("div < 0.45 marks a LOOP; 'L' suffix = looped at that alpha")


if __name__ == "__main__":
    main()