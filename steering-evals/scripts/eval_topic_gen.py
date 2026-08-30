#!/usr/bin/env python3
"""Multi-token generation under chord steering: test the topic walks end-to-end.

G1  BASELINE: sample N tokens from the prompt (no steering) - topic drift.
G2  WALK: at every step, chord-invert toward the TARGET family's best member
      (k deg re-aim) then sample - does the generation actually land in/on
      the target topic, and when does the topic flip?
G3  SINGLE-SHOT: one 17 deg arc at step 0, then free sampling - does one push
      hold the topic through generation, or does it drift back?

Per-token topic = nearest class centroid (de-poled) to the generated token.

Run:  python eval_topic_gen.py [start] [target] [tokens]   (Qwen, ~10 s)
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = 'Qwen/Qwen2-0.5B-Instruct'
START = sys.argv[1] if len(sys.argv) > 1 else 'food'
TARGET = sys.argv[2] if len(sys.argv) > 2 else 'city'
N_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 12
PROMPT = 'Once upon a time, there was a'
SEED = 7
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
STEP_DEG = 3.0          # re-aim angle per step for the walk


def main():
    t0 = time.time()
    model, tok = M.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()[:model.config.vocab_size]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    topics = {cls: np.array([word2id[w] for w in words if w in word2id])
              for cls, words in CLASSES.items()
              if sum(1 for w in words if w in word2id) >= 6}
    names = list(topics)
    rows = {c: Wn[i] for c, i in topics.items()}

    pid = tok('Once upon a time', add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    li = model.config.num_hidden_layers - 1
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
        h0 = hid.hidden_states[li + 1][0, 0]
    u = (h0 / h0.norm()).cpu().float().numpy()

    def depole(v):
        v = v - (v @ u) * u
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    cent = {c: depole(Wn[i].mean(0)) for c, i in topics.items()}

    def topic_of(tid):
        r = depole(Wn[tid])
        return min(names, key=lambda c: np.degrees(
            np.arccos(np.clip(r @ cent[c], -1, 1))))

    def sample(h, top_k=10):
        L = h @ Wn.T
        v, i = torch.topk(torch.as_tensor(L).float(), top_k)
        p = torch.softmax(v, 0)
        return int(i[int(torch.multinomial(p, 1))])

    def step(tids, steer=None, steer_mag=0.0):
        """One autoregressive step; steer = ('member', topic-row-array) or None."""
        in_ids = torch.as_tensor([tids], device=DEV)
        with torch.no_grad():
            h = model(in_ids, output_hidden_states=True).hidden_states[li + 1][0, -1]
            h = (h / h.norm()).cpu().float().numpy()
        if steer is not None:
            fam = steer
            best = fam[int(np.argmax(fam @ h))]
            tau = M.tangent_direction(h, best)
            h = M.rotate_toward(h, tau, math.radians(steer_mag))
        return sample(h)

    pts = tok(PROMPT, add_special_tokens=False).input_ids

    def run(name, steer, mag):
        torch.manual_seed(SEED)
        tids = list(pts)
        if name != 'baseline':
            # settle 3 steps INTO the start topic (as in the map script)
            in_ids = torch.as_tensor([tids], device=DEV)
            with torch.no_grad():
                hh = model(in_ids, output_hidden_states=True).hidden_states[li + 1][0, -1]
            hs0 = (hh / hh.norm()).cpu().float().numpy()
            for _ in range(3):
                best = rows[START][int(np.argmax(rows[START] @ hs0))]
                hs0 = M.rotate_toward(hs0, M.tangent_direction(hs0, best), math.radians(STEP_DEG))
            tids.append(sample(hs0))
        seen_topics = []
        for i in range(N_TOK):
            pick = STEP_DEG if (i == 0 and name.startswith('single')) else mag
            nxt = step(tids, steer, pick)
            tids.append(nxt)
            seen_topics.append(topic_of(nxt))
        torch.manual_seed(SEED)
        tids = list(pts)
        seen_topics = []
        for i in range(N_TOK):
            pick = STEP_DEG if (i == 0 and name.startswith('single')) else mag
            nxt = step(tids, steer, pick)
            tids.append(nxt)
            seen_topics.append(topic_of(nxt))
        txt = tok.decode(tids[len(pts):], skip_special_tokens=True)
        counts = {c: seen_topics.count(c) for c in names}
        top = max(counts, key=counts.get)
        print(f"\n[{name}]  topics: {' '.join(f'{c[:3]}' for c in seen_topics)}")
        print(f"    target-family tokens: {seen_topics.count(TARGET)}/{N_TOK}  "
              f"(dominant {top})")
        print(f"    text: {txt!r}")
        return seen_topics, txt

    print(f"[walk map] {START} -> {TARGET}, {N_TOK} tokens, seed {SEED}\n")
    run('baseline', None, 0.0)
    run('walk', rows[TARGET], STEP_DEG)
    run('single-shot', rows[TARGET], 17.0)
    run('persistent', rows[TARGET], 8.0)
    print(f"\n[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()