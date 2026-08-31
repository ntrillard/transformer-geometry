#!/usr/bin/env python3
"""Practical steering battery: replication, suppression (anti-steering),
phrase targets, statistical sampling.  All hooks renormalize to ||h|| after
the tangent step (exact sphere contract).

Run: python eval_practical_steering.py --model Qwen/Qwen2-0.5B-Instruct
"""
import argparse
import numpy as np
import pandas as pd
import torch

import steering_geometry_test as M
from eval_sustained_steering import generate, OUT


@torch.no_grad()
def sphere_hook(head_rows, alpha):
    """Tangent step + renormalization back onto the state's shell (paper formula)."""
    w = head_rows if head_rows.dim() == 1 else head_rows.mean(0)
    w = w / w.norm()

    def hook(module, inp, out):
        out2 = out.clone()
        h = out2[0, -1, :].float()
        hn = h / h.norm()
        g = w - (w @ hn) * hn
        g = g / max(g.norm().item(), 1e-8)
        h2 = h + alpha * h.norm() * g
        out2[0, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
        return out2
    return hook


@torch.no_grad()
def gen(model, tok, pids, n=48, hook_fn=None, layer=None, duty=1,
        temperature=None, top_p=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    ids = pids.clone()
    conf = []
    for step in range(n):
        hd = None
        if hook_fn is not None and step % duty == 0:
            hd = model.model.layers[layer].register_forward_hook(hook_fn)
        try:
            logits = model(ids).logits[0, -1].float()
        finally:
            if hd is not None:
                hd.remove()
        p = torch.softmax(logits, dim=0)
        conf.append(float(p.max()))
        if top_p:
            q = p.clone()
            order = q.argsort(descending=True)
            cum = q[order].cumsum(0)
            keep = order[:int((cum <= top_p).sum()) + 1]
            m = torch.zeros_like(q)
            m[keep] = 1
            q = (q * m) / (q * m).sum()
            nxt = int(torch.multinomial(q, 1))
        elif temperature:
            nxt = int(torch.multinomial(torch.softmax(logits / temperature, 0), 1))
        else:
            nxt = int(logits.argmax())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return tok.decode(ids[0].tolist()), float(np.mean(conf))


def lemma_count(text, word):
    t = text.lower()
    w = word.lower()
    return sum(t.count(w + s) for s in ["", "s", "es"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()

    model, tok = M.load_model(args.model, dtype="fp16")
    W = model.lm_head.weight.detach().cpu().float().numpy()
    if W.size == 0:
        W = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    L = model.config.num_hidden_layers
    li = L - 1
    safe = args.model.replace("/", "--")

    def row(word):
        return torch.as_tensor(W[tok(" " + word.strip(),
                                     add_special_tokens=False).input_ids],
                               device=model.device)

    P_STORY = "Once upon a time"
    P_FACT = "Tell me something interesting:"
    rows = []

    def record(name, prompt, text, conf, word=None):
        occ = lemma_count(text, word) if word else 0
        div = len(set(tok(text, add_special_tokens=False).input_ids)) / max(
            len(tok(text, add_special_tokens=False).input_ids), 1)
        rows.append(dict(test=name, prompt=prompt, occurrences=occ, conf=conf,
                         diversity=div, sample=text))
        print(f"[{name}] occ={occ} conf={conf:.2f} div={div:.2f} | {text[:100]}")

    # --- T1: dose-response grid, sampled stats ---
    pids = tok(P_STORY, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    for a in [0.1, 0.2, 0.3]:
        for duty in [4, 6]:
            h = sphere_hook(row("apple"), a)
            occs, divs = [], []
            ex = ""
            for s in range(args.samples):
                text, conf = gen(model, tok, pids, args.new_tokens, hook_fn=h,
                                 layer=li, duty=duty, top_p=0.9, seed=s)
                o = lemma_count(text, "apple")
                occs.append(o)
                n_toks = len(tok(text, add_special_tokens=False).input_ids)
                divs.append(len(set(tok(text, add_special_tokens=False).input_ids)) / max(n_toks, 1))
                if s == 0:
                    ex = text
            rows.append(dict(test=f"T1-grid a={a} k={duty}", prompt=P_STORY,
                             occurrences=float(np.mean(occs)), conf=np.nan,
                             diversity=float(np.mean(divs)),
                             sample=f"x{np.mean(occs):.1f}+/-{np.std(occs):.1f} | {ex[:90]}"))
            print(f"T1 a={a} k={duty}: x{np.mean(occs):.1f} +/- {np.std(occs):.1f}, "
                  f"div {np.mean(divs):.2f} | {ex[:80]}")

    # --- T2: anti-steering (suppress a word the model over-produces) ---
    pids = tok(P_FACT, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)[:, :8]
    base_texts = [gen(model, tok, pids, args.new_tokens, top_p=0.9, seed=s)[0]
                  for s in range(args.samples)]
    base_occ = float(np.mean([lemma_count(t, "world") for t in base_texts]))
    h_neg = sphere_hook(row("world"), -0.5)
    steered = [gen(model, tok, pids, args.new_tokens, hook_fn=h_neg, layer=li,
                   duty=1, top_p=0.9, seed=s)[0] for s in range(args.samples)]
    st_occ = float(np.mean([lemma_count(t, "world") for t in steered]))
    print(f"\nT2-antisuppress 'world': baseline x{base_occ:.1f} -> steered x{st_occ:.1f}")
    record("T2-suppress-world", P_FACT, steered[0], np.nan, "world")
    rows.append(dict(test="T2-summary", prompt=P_FACT, occurrences=st_occ,
                     conf=np.nan, diversity=np.nan,
                     sample=f"baseline x{base_occ:.1f} -> steered x{st_occ:.1f}"))

    # --- T3: multi-token phrase target ---
    pids = tok(P_STORY, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    phrase_rows = W[tok(" chocolate cake", add_special_tokens=False).input_ids]
    h = sphere_hook(torch.as_tensor(phrase_rows, device=model.device), 0.3)
    text, conf = gen(model, tok, pids, args.new_tokens, hook_fn=h, layer=li, duty=6,
                     top_p=0.9, seed=0)
    print("\nT3-phrase 'chocolate cake':")
    record("T3-phrase-choc-cake", P_STORY, text, conf, "chocolate cake")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"practical_steering__{safe}.csv", index=False)
    print(f"\nSaved -> practical_steering__{safe}.csv")


if __name__ == "__main__":
    main()
