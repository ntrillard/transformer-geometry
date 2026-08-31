"""eval_alpha_law.py — FAST: test the margin->loop and flip-vs-alpha laws on any
model. Sweeps alpha (0.04..0.30) x seeds and reports diversity (loopness) + the
flip rate at each alpha, plus a same-state monotonicity test (the strongest
form of the law).

Run: python3 eval_alpha_law.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
     python3 eval_alpha_law.py --model google/gemma-3-1b-it --tag gemma
"""
import argparse
import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from multiverse_lab import chord_summary

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ALPHAS = (0.04, 0.06, 0.08, 0.12, 0.15, 0.2, 0.25, 0.3)


def _flip_clean(model, tok, ids, dn, NL_dev, alpha):
    def hook(mod, inp, out):
        out2 = out.clone()
        h = out2[:, -1, :].float()
        hn = h / h.norm()
        g = dn - (dn @ hn.transpose(0, 1)) * hn
        g = g / max(g.norm().item(), 1e-8)
        h2 = h + alpha * h.norm() * g
        out2[:, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
        return out2
    hd = model.model.layers[NL_dev - 1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            Ls = model(ids).logits[0, -1].float()
    finally:
        hd.remove()
    return int(Ls.argmax())


def clean_monotonicity(model, tok, ids, dn, NL_dev, tag):
    with torch.no_grad():
        Ln = model(ids).logits[0, -1].float()
    nt = int(Ln.argmax())
    prev = None
    print(f"   [{tag}] clean monotonicity (fixed state):")
    for alpha in (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.9, 1.2):
        st = _flip_clean(model, tok, ids, dn, NL_dev, alpha)
        flip = st != nt
        mon = ''
        if prev is not None:
            mon = '  (VIOLATES MONO!)' if flip and not prev else ''
        prev = flip
        print(f"      a {alpha:>5.2f}  argmax {tok.decode([st]).strip()!r:>8}  "
              f"flip={str(flip):>5}{mon}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2-0.5B-Instruct')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--prompt', default="For dinner I made")
    ap.add_argument('--clean', action='store_true')
    a = ap.parse_args()

    import multiverse_lab
    multiverse_lab.DEV = DEV
    model, tok = M.load_model(a.model, dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    W = W[:model.config.vocab_size]
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
    dn = torch.as_tensor(C, device=DEV, dtype=torch.float32)

    def gen_with(alpha, seed=0, n=16, duty=3):
        torch.manual_seed(seed)
        ids = tok(a.prompt, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        txt_toks = []
        for step in range(n):
            if step % duty == 0:
                def hook(mod, inp, out):
                    out2 = out.clone()
                    h = out2[:, -1, :].float()
                    hn = h / h.norm()
                    g = dn - (dn @ hn.transpose(0, 1)) * hn
                    g = g / max(g.norm().item(), 1e-8)
                    h2 = h + alpha * h.norm() * g
                    out2[:, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
                    return out2
                hd = model.model.layers[NL - 1].register_forward_hook(hook)
            else:
                hd = None
            try:
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
            finally:
                if hd is not None:
                    hd.remove()
            p = torch.softmax(L, dim=0)
            q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
            keep = order[:int((cum <= 0.9).sum()) + 1]
            m = torch.zeros_like(q); m[keep] = 1
            q = (q * m) / (q * m).sum()
            nxt = int(torch.multinomial(q, 1))
            txt_toks.append(int(nxt))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        text = tok.decode(txt_toks)
        return text

    print(f"== [{a.tag}] alpha sweep: food5 @ L{NL-1}, duty 3, {a.prompt!r} ==")
    print(f"{'alpha':>5} {'div':>5} {'occ':>5} {'dist':>5}  sample")
    for alpha in ALPHAS:
        divs, occs, dists, first = [], [], [], ''
        for sd in (0, 1):
            text = gen_with(alpha, seed=sd)
            low = text.lower()
            divs.append(len(set(text)) / max(len(text), 1))
            occs.append(sum(low.count(w.lower()) for w in fam))
            dists.append(sum(1 for w in fam if w.lower() in low))
            if sd == 0:
                first = text
        d = np.mean(divs)
        print(f"{alpha:>5.2f} {d:5.2f} {np.mean(occs):5.1f} {np.mean(dists):5.1f}  "
              f"{first[:44]!r}{'  LOOP' if d < 0.45 else ''}")

    # ---- same-state monotonicity: one state, alpha rising ----
    print(f"\n== [{a.tag}] same-state flip vs alpha (single 4-token state) ==")
    ids = tok(a.prompt, add_special_tokens=False,
              return_tensors='pt').input_ids.to(model.device)
    with torch.no_grad():
        Ln = model(ids).logits[0, -1].float()
    nt = int(Ln.argmax())
    prev = None
    for alpha in ALPHAS:
        def hook(mod, inp, out):
            out2 = out.clone()
            h = out2[:, -1, :].float()
            hn = h / h.norm()
            g = dn - (dn @ hn.transpose(0, 1)) * hn
            g = g / max(g.norm().item(), 1e-8)
            h2 = h + alpha * h.norm() * g
            out2[:, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
            return out2
        hd = model.model.layers[NL - 1].register_forward_hook(hook)
        try:
            with torch.no_grad():
                Ls = model(ids).logits[0, -1].float()
        finally:
            hd.remove()
        st = int(Ls.argmax())
        flip = st != nt
        mon = ''
        if prev is not None:
            mon = '  (violates mono!)' if flip and not prev else ''
        prev = flip
        print(f"  alpha {alpha:>5.2f}  argmax {tok.decode([st]).strip()!r:>8}  "
              f"flip={str(flip):>5}{mon}")
    if a.clean:
        clean_monotonicity(model, tok, ids, dn, NL, a.tag)


if __name__ == "__main__":
    main()
