"""multiverse_compare.py — run the core sphere battery on ANY model and print
the anatomy/reach/authorship/steered-path tables. Currently validated for
Qwen2-0.5B-Instruct and gemma-3-1b-it.

Run: python3 multiverse_compare.py --model google/gemma-3-1b-it --tag gemma
     python3 multiverse_compare.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
"""
import argparse
import math
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from multiverse_lab import chord_summary

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = ["For dinner I made", "The capital of France is"]


def norm(x):
    return x / np.linalg.norm(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2-0.5B-Instruct')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--prompt', default="For dinner I made")
    ap.add_argument('--reach', action='store_true', help='per-layer chord-reach table')
    ap.add_argument('--path', action='store_true', help='steered-path flip stats (slow)')
    a = ap.parse_args()

    t0 = time.time()
    model, tok = M.load_model(a.model, dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    V = model.config.vocab_size
    W = W[:V]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    Wt = torch.as_tensor(W, device=DEV, dtype=torch.float32)
    NL = model.config.num_hidden_layers
    D = W.shape[1]
    model_dtype = model.dtype
    print(f"[{a.tag}] {a.model}: {NL} layers, dim {D}, vocab {V} "
          f"(load {time.time()-t0:.0f}s)")

    # note vocabulary
    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    avail = {c: [w for w in words if w in word2id] for c, words in CLASSES.items()}
    print(f"[{a.tag}] note rows available: {len(word2id)}/60 "
          f"(classes {[c for c, w in avail.items() if len(w) >= 5]})")
    chords = {}
    for cls, words in avail.items():
        if len(words) >= 5:
            ids = np.array([word2id[w] for w in words[:5]])
            C, _, _ = chord_summary(ids, Wn)
            chords[cls] = (ids, C)

    # ---- one forward + residual hooks: anatomy + authorship ----
    deltas = {'attn': [], 'mlp': []}
    handles = []
    for l in range(NL):
        lay = model.model.layers[l]
        handles.append(lay.self_attn.register_forward_hook(
            lambda m, i, o, k='attn': deltas[k].append(
                (o[0] if isinstance(o, tuple) else o).detach().float())))
        handles.append(lay.mlp.register_forward_hook(
            lambda m, i, o, k='mlp': deltas[k].append(o.detach().float())))
    pid = tok(a.prompt, add_special_tokens=False,
              return_tensors='pt').input_ids.to(model.device)
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
    for h in handles:
        h.remove()
    hs = [h[0, -1].float().cpu().numpy() for h in hid.hidden_states]
    ad = torch.stack([d[0, -1] for d in deltas['attn']]).cpu().float().numpy()
    md = torch.stack([d[0, -1] for d in deltas['mlp']]).cpu().float().numpy()
    print(f"[{a.tag}] forward+hooks {time.time()-t0:.0f}s; delta identity check: ",
          end='')
    preN = hs[0] + ad.sum(0) + md.sum(0)
    print(f"rel-err {np.linalg.norm(preN - hs[NL])/np.linalg.norm(hs[NL]):.3f} "
          f"(hs[-1] is post-final-norm)" if NL else "")

    # gamma per layer (RMSNorm scale) - model-agnostic: input_layernorm + final
    gammas = []
    for l in range(NL):
        w = model.model.layers[l].input_layernorm.weight.detach().cpu().float().numpy()
        gammas.append(float(np.linalg.norm(w)))
    fn = getattr(model.model, 'norm', None) or getattr(model.model, 'final_layernorm', None)
    gammas[-1] = float(np.linalg.norm(fn.weight.detach().cpu().float().numpy()))

    # ---- anatomy table (compact every-other row) ----
    U = np.stack([norm(h) for h in hs[1:]])
    La = (torch.as_tensor(U, device=DEV, dtype=torch.float32) @ Wt.T).cpu().numpy()
    final_arg = int(La[-1].argmax())
    final_tok = tok.decode([final_arg]).strip()
    print(f"\n== [{a.tag}] anatomy (prompt {a.prompt!r}) final argmax {final_tok!r} ==")
    print(f"{'L':>3} {'||h||':>7} {'||g||':>7} {'h/g':>5} {'angFood':>7} "
          f"{'reach@17':>8} {'margin':>7}")
    food_ids, foodC = chords.get('food', (None, None))
    for l in range(NL):
        h, hn = hs[l + 1], norm(hs[l + 1])
        af = math.degrees(math.acos(np.clip(hn @ foodC, -1, 1))) if foodC is not None else float('nan')
        r, mg = '   ', float('nan')
        if a.reach and food_ids is not None:
            tau = M.tangent_direction(hn, foodC)
            v = M.rotate_toward(hn, tau, np.radians(17))
            Lv = (torch.as_tensor(v, device=DEV, dtype=torch.float32) @ Wt.T).cpu().numpy()
            fam = Lv[food_ids]
            mg = float(fam.max() - np.delete(Lv, food_ids).max())
            r = 'True ' if fam.max() > np.delete(Lv, food_ids).max() else 'False'
        print(f"{l:>3} {float(np.linalg.norm(h)):7.1f} {gammas[l]:7.1f} "
              f"{float(np.linalg.norm(h))/gammas[l]:5.2f} {af:7.1f} {r:>8} "
              f"{mg:+.4f}" if a.reach else
              f"{l:>3} {float(np.linalg.norm(h)):7.1f} {gammas[l]:7.1f} "
              f"{float(np.linalg.norm(h))/gammas[l]:5.2f} {af:7.1f}")

    # ---- authorship per class - proportional depth windows for fair cross-model view
    PE = int(NL * 0.66)
    print(f"\n== [{a.tag}] authorship ({a.prompt!r})  plant[5:{PE}] write[{PE}:{NL - 1}] =")

    print(f"{'class':>7} {'plantAttn':>10} {'plantMlp':>10} {'writeMlp':>10} {'finalMlp':>10}")
    for name, (ids, C) in chords.items():
        pa = float((ad[5:PE] @ C).sum())
        pm = float((md[5:PE] @ C).sum())
        wm = float((md[PE:NL - 1] @ C).sum())
        fm = float(md[NL - 1] @ C)
        print(f"{name:>7} {pa:+9.3f} {pm:+9.3f} {wm:+9.3f} {fm:+9.3f}")

    LAY = NL // 2   # plant/write split scales with depth
    # (for gemma 24L the 5-17 window is fine; keep as-is)

    # ---- steered path (optional) ----
    if a.path and food_ids is not None:
        C = chords['food'][1]
        dn = torch.as_tensor(C, device=DEV, dtype=torch.float32)
        fam = set(avail['food'])
        print(f"\n== [{a.tag}] steered path (food5 @ L{NL-1}, a.15 d3, 16 tok) ==")
        for prompt in PROMPTS:
            for sd in (0, 1):
                torch.manual_seed(sd)
                ids = tok(prompt, add_special_tokens=False,
                          return_tensors='pt').input_ids.to(model.device)
                flips, lo, hi = [], [], []
                path_toks = []
                for _ in range(16):
                    with torch.no_grad():
                        Ln = model(ids).logits[0, -1].float()
                    nt = int(Ln.argmax())
                    l2 = Ln.clone(); l2[nt] = -float('inf')
                    margin = float(Ln[nt] - l2.max())

                    def hook(mod, inp, out):
                        out2 = out.clone()
                        h = out2[:, -1, :].float()
                        hn = h / h.norm()
                        g = dn - (dn @ hn.transpose(0, 1)) * hn
                        g = g / max(g.norm().item(), 1e-8)
                        h2 = h + 0.15 * h.norm() * g
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
                    flips.append(flip)
                    if margin <= 0.1:
                        lo.append(flip)
                    else:
                        hi.append(flip)
                    nt_s = tok.decode([nt]).strip()[:6]
                    st_s = tok.decode([st]).strip()[:6]
                    path_toks.append(f"[{st_s}]" if flip else nt_s)
                    p = torch.softmax(Ls, dim=0)
                    q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
                    keep = order[:int((cum <= 0.9).sum()) + 1]
                    m = torch.zeros_like(q); m[keep] = 1
                    q = (q * m) / (q * m).sum()
                    nxt = int(torch.multinomial(q, 1))
                    ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
                f_lo = sum(lo); f_hi = sum(hi)
                print(f"  {prompt:>24} s{sd} flips {sum(flips):2d}/16 "
                      f"lo {f_lo}/{len(lo)} hi {f_hi}/{len(hi)}  "
                      f"hi-flip-rate {f_hi/max(len(hi),1):.2f}")
                print(f"    path: {' '.join(path_toks)[:92]}")

    print(f"[{a.tag}] TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()