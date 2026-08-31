#!/usr/bin/env python3
"""Topic steering: develop the chord-inversion idea into a generation primitive.

From chord/inversion results (readout level): aim at the best-POSITIONED family
note from the current state, not the centroid. This script is the first
END-TO-END test of that recipe in actual multi-token generation.

Conditions (48 tokens, top-p=0.9, 3 seeds, prompts x topics):
  base        : no steer
  ss_note     : single-shot inversion -- steer once at step 0 toward the
                family note best-positioned from the PROMPT state, then free
  ss_center   : single-shot toward the centroid C (control)
  cad_k1/k3/k6: CADENCE -- every k-th step, RE-AIM toward whichever family note
                has the highest logit from the CURRENT state.  Tests whether
                re-aiming keeps the text on-topic without collapsing into a run
  per_note    : persistent single-note steer (the known pit control)
  per_center  : persistent centroid steer (pit control 2)

Metrics: family occurrences, distinct family words, diversity, max token run,
first-token family hit (target adherence).

Run: python eval_topic_steering.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import steering_geometry_test as M

OUT = Path("../steering_geometry_results")

CLASSES = {
    'food':   ['apple', 'banana', 'bread', 'cheese', 'chicken', 'grape',
               'honey', 'milk', 'rice', 'soup'],
    'animal': ['dog', 'cat', 'horse', 'lion', 'bird', 'wolf', 'tiger',
               'fish', 'snake', 'rabbit'],
    'color':  ['red', 'blue', 'green', 'black', 'white', 'yellow',
               'pink', 'purple', 'brown', 'gray'],
    'city':   ['Paris', 'London', 'Tokyo', 'Berlin', 'Rome', 'Moscow',
               'Cairo', 'Delhi', 'Seoul', 'Madrid'],
    'nature': ['ocean', 'tree', 'mountain', 'river', 'forest', 'flower',
               'stone', 'cloud', 'star', 'moon'],
    'number': ['one', 'two', 'three', 'four', 'five', 'six', 'seven',
               'eight', 'nine', 'ten'],
}
PROMPTS = ["Once upon a time", "Tell me something interesting:"]
ALPHA = 0.3
NEW_TOKENS = 48
SEEDS = (0, 1, 2)


@torch.no_grad()
def generate(model, tok, pids, layer, fam_rows, mode, k=0, alpha=ALPHA):
    """One continuation. fam_rows: (m,d) normalized head rows on device."""
    ids = pids.clone()
    li = layer
    steered = 0
    first_fam = None
    for step in range(NEW_TOKENS):
        # ---- decide steering direction for THIS step ----
        hd = None
        use = mode in ('cad', 'ss')
        if mode == 'ss' and step != 0:
            use = False
        if use:
            if mode == 'cad':
                do_steer = (step % k == 0)
            else:
                do_steer = True
            if do_steer:
                def make_hook():
                    # capture: at activation, compute best-positioned family note
                    def hook(module, inp, out):
                        nonlocal steered
                        out2 = out.clone()
                        h = out2[0, -1, :].float()
                        hn = h / h.norm()
                        # best family note from CURRENT state
                        fscore = hn @ fam_rows.T          # (m,)
                        best = int(fscore.argmax().item())
                        w = fam_rows[best]
                        g = w - (w @ hn) * hn
                        g = g / max(g.norm().item(), 1e-8)
                        h2 = h + alpha * h.norm() * g
                        out2[0, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
                        steered += 1
                        return out2
                    return hook
                hd = model.model.layers[li].register_forward_hook(make_hook())
        try:
            logits = model(ids).logits[0, -1].float()
        finally:
            if hd is not None:
                hd.remove()
        p = torch.softmax(logits, dim=0)
        order = p.argsort(descending=True)
        cum = p[order].cumsum(0)
        keep = order[:int((cum <= 0.9).sum()) + 1]
        msk = torch.zeros_like(p); msk[keep] = 1
        q = (p * msk) / (p * msk).sum()
        nxt = int(torch.multinomial(q, 1))
        if first_fam is None:
            first_fam = nxt
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return ids, steered, first_fam


def main():
    t0 = time.time()
    model, tok = M.load_model('Qwen/Qwen2-0.5B-Instruct', dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    vocab = model.config.vocab_size
    W = W[:vocab]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    li = model.config.num_hidden_layers - 1

    word2id = {}
    for w in sorted({x for cls in CLASSES.values() for x in cls}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])

    rows = []
    for topic, words in CLASSES.items():
        fam = [word2id[w] for w in words if w in word2id]
        if len(fam) < 6:
            continue
        fam_arr = np.array(fam, dtype=np.int64)
        fam_rows = torch.as_tensor(Wn[fam_arr], device=dev)
        # centroid (center control)
        C = Wn[fam_arr].mean(0); C = C / np.linalg.norm(C)
        C_n = torch.as_tensor(C, device=dev)

        for prompt in PROMPTS:
            pids = tok(prompt, add_special_tokens=False,
                       return_tensors="pt").input_ids.to(model.device)
            # one-shot target note chosen from the PROMPT state
            # -> compute hidden state at layer li for the prompt (1 forward)
            outs = model(pids, output_hidden_states=True)
            u = outs.hidden_states[li + 1][0, -1, :].float()
            un = u / u.norm()
            fscore = un @ fam_rows.T
            best0 = int(fscore.argmax().item())

            # conditions
            conds = []
            # base
            conds.append(('base', None, 0))
            # ss_note (steer once at step 0 toward pre-chosen best0 note)
            ss_note_dir = fam_rows[best0]
            conds.append(('ss_note', 'ss', ss_note_dir))
            conds.append(('ss_center', 'ss', C_n))
            for k in (1, 3, 6):
                conds.append((f'cad_k{k}', 'cad', k))
            conds.append(('per_note', 'pers', fam_rows[best0]))
            conds.append(('per_center', 'pers', C_n))

            for cname, mode, arg in conds:
                for seed in SEEDS:
                    torch.manual_seed(seed)
                    if mode is None:
                        ids, st, first = generate(model, tok, pids, li, fam_rows,
                                                  'base')
                    elif mode == 'ss':
                        # single-shot via re-aiming hook that only fires at step 0
                        # -> use cadence-style but force best==prechosen? simpler:
                        # we reuse generate with a mode that steers once using
                        # the pre-chosen direction.
                        ids, st, first = ss_once(model, tok, pids, li, fam_rows,
                                                 arg)
                    elif mode == 'cad':
                        ids, st, first = generate(model, tok, pids, li, fam_rows,
                                                  'cad', k=arg)
                    else:  # persistent single direction
                        ids, st, first = persist(model, tok, pids, li, fam_rows,
                                                 arg)
                    text = tok.decode(ids[0].tolist()[len(pids[0]):],
                                      skip_special_tokens=True)
                    low = text.lower()
                    occ = sum(low.count(w) for w in words)
                    distinct = sum(1 for w in words if w in low)
                    toks = ids[0, len(pids[0]):].tolist()
                    div = len(set(toks)) / max(len(toks), 1)
                    run, cur, last = 0, 0, None
                    for tid in toks:
                        if tid == last:
                            cur += 1
                        else:
                            cur = 1
                        run = max(run, cur)
                        last = tid
                    first_hit = int(first in fam_arr)
                    rows.append(dict(topic=topic, prompt=prompt, cond=cname,
                                     seed=seed, occ=occ, distinct=distinct,
                                     diversity=div, maxrun=run,
                                     first_hit=first_hit, text=text[:120]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "topic_steering.csv", index=False)
    print(f"[{time.time()-t0:.0f}s] saved topic_steering.csv\n")

    # summary by condition
    print("== By condition (mean over topics x prompts x seeds) ==")
    g = df.groupby('cond').agg(occ=('occ', 'mean'),
                               distinct=('distinct', 'mean'),
                               diversity=('diversity', 'mean'),
                               maxrun=('maxrun', 'mean'),
                               first_hit=('first_hit', 'mean'))
    print(g.round(2).to_string())
    print("\nfirst_hit = first token is a family word (adherence)")


@torch.no_grad()
def ss_once(model, tok, pids, li, fam_rows, dirv):
    """Steer step 0 toward direction dirv, then free."""
    ids = pids.clone()
    first = None
    fired = False
    for step in range(NEW_TOKENS):
        hd = None
        if step == 0 and dirv is not None:

            def hook(module, inp, out):
                nonlocal fired
                out2 = out.clone()
                h = out2[0, -1, :].float()
                hn = h / h.norm()
                w = dirv
                g = w - (w @ hn) * hn
                g = g / max(g.norm().item(), 1e-8)
                h2 = h + ALPHA * h.norm() * g
                out2[0, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
                fired = True
                return out2
            hd = model.model.layers[li].register_forward_hook(hook)
        try:
            logits = model(ids).logits[0, -1].float()
        finally:
            if hd is not None:
                hd.remove()
        p = torch.softmax(logits, dim=0)
        order = p.argsort(descending=True)
        cum = p[order].cumsum(0)
        keep = order[:int((cum <= 0.9).sum()) + 1]
        msk = torch.zeros_like(p); msk[keep] = 1
        q = (p * msk) / (p * msk).sum()
        nxt = int(torch.multinomial(q, 1))
        if first is None:
            first = nxt
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return ids, int(fired), first


@torch.no_grad()
def persist(model, tok, pids, li, fam_rows, dirv):
    """Persistent steer toward fixed dirv every step (pit control)."""
    ids = pids.clone()
    first = None
    for step in range(NEW_TOKENS):
        hd = None
        if dirv is not None:

            def hook(module, inp, out):
                out2 = out.clone()
                h = out2[0, -1, :].float()
                hn = h / h.norm()
                w = dirv
                g = w - (w @ hn) * hn
                g = g / max(g.norm().item(), 1e-8)
                h2 = h + ALPHA * h.norm() * g
                out2[0, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
                return out2
            hd = model.model.layers[li].register_forward_hook(hook)
        try:
            logits = model(ids).logits[0, -1].float()
        finally:
            if hd is not None:
                hd.remove()
        p = torch.softmax(logits, dim=0)
        order = p.argsort(descending=True)
        cum = p[order].cumsum(0)
        keep = order[:int((cum <= 0.9).sum()) + 1]
        msk = torch.zeros_like(p); msk[keep] = 1
        q = (p * msk) / (p * msk).sum()
        nxt = int(torch.multinomial(q, 1))
        if first is None:
            first = nxt
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return ids, NEW_TOKENS, first


if __name__ == "__main__":
    main()