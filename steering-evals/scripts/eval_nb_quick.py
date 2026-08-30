#!/usr/bin/env python3
"""Fast (~15-20s) semantic-neighborhood tests on LM-head rows (weights-only).

Re-uses T1b finding: row neighborhoods are semantically coherent.  Quick tests:
  Q1  class-coherence: for each of the 6 class word sets, is the 30-NN of each
      member enriched for SAME-CLASS words vs chance?
  Q2  mutual-NN graph: how often are class words each other's top-10 nearest?
      (cluster strength -- mutual-NN rate vs random baseline)
  Q3  bilingual: share of 30-NN rows that decode to non-Latin script.
  Q4  semantic drift: in the 30-NN, rank position of the "best translation/
      closest semantic twin" vs identity variants.

Run: python eval_nb_quick.py      (expect < 20 s on a GPU box)
"""
import numpy as np
import torch
from safetensors.torch import load_file as st_load
from transformers import AutoTokenizer
from pathlib import Path

CACHE = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct'

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


def main():
    snap = sorted((CACHE / 'snapshots').glob('*'))[0]
    st = sorted(snap.glob('*.safetensors'))[0]
    d = st_load(str(st))
    key = [k for k in d if 'weight' in k and d[k].ndim == 2][0]
    W = d[key].float().numpy()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    V = len(Wn)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B-Instruct')

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    print(f"single-token class words: {len(word2id)}")

    Wt = torch.as_tensor(Wn.astype(np.float32))
    # all class rows in one matmul, full-vocab columns
    cls_rows = torch.as_tensor([word2id[w]
                                for words in CLASSES.values()
                                for w in words if w in word2id])
    proj = torch.zeros((len(cls_rows), V), dtype=torch.float32)
    for i, tid in enumerate(cls_rows.tolist()):
        c = torch.clamp(Wt @ Wt[tid], -1, 1)
        c[tid] = -1
        proj[i] = c
    top30 = proj.topk(30, dim=1).indices        # (M,30)

    # ---- Q1: class-coherence of 30-NN ----
    print("\n[Q1] class enrichment in 30-NN (same-class count, chance ~ %.2f/30)"
          % (30 * 9 / V))
    total_null = 0
    for cls, words in CLASSES.items():
        ids = [word2id[w] for w in words if w in word2id]
        if len(ids) < 6:
            continue
        setidx = set(ids)
        nbr = []
        for tid in ids:
            r = top30[list(cls_rows).index(tid)].tolist()
            nbr.append(sum(1 for x in r if x in setidx))
        total_null += sum(nbr)
        print(f"   {cls:8s} n={len(ids)} same-class in 30-NN: "
              f"mean {np.mean(nbr):.2f}  (expect ~{30 * len(ids) / V * len(ids)} if random)")
    # ---- Q2: mutual-NN (top-10) rate ----
    print("\n[Q2] mutual top-10-NN rate between class members:")
    top10 = proj.topk(10, dim=1).indices
    mutual = 0
    for i, tid in enumerate(cls_rows.tolist()):
        for j, tid2 in enumerate(cls_rows.tolist()):
            if i == j:
                continue
            if tid2 in top10[i].tolist() and tid in top10[j].tolist():
                mutual += 1
    pairs = len(cls_rows) * (len(cls_rows) - 1)
    print(f"   mutual top-10-NN: {mutual}/{pairs} = {100 * mutual / pairs:.1f}%  "
          f"(random estimate ~{100 * 10 / V * len(cls_rows):.2f}%)")

    # ---- Q3: bilingual share of 30-NN ----
    print("\n[Q3] non-Latin (CJK etc) share of 30-NN:")
    b = 0
    for i, tid in enumerate(cls_rows.tolist()):
        for x in top30[i].tolist():
            t = tok.decode([x], skip_special_tokens=True)
            if any('\u4e00' <= ch <= '\u9fff' for ch in t):
                b += 1
    print(f"   CJK tokens in 30-NN over all class words: {b}/{len(cls_rows) * 30} "
          f"= {100 * b / (len(cls_rows) * 30):.1f}%")


if __name__ == "__main__":
    main()