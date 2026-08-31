#!/usr/bin/env python3
"""Systematic probe of the LM-head semantic topography (T1b deep dive).

T1b showed local neighborhoods of Qwen2-0.5B head rows are semantically
coherent ('apple' -> Apple/苹果/apples, 'Paris' -> 巴黎/France/London) at a
60-70 deg scale. Here we make that measurable:

  P1  Nested angular scales. Three axes in row space, measured as angular
      distributions:
        * identity: same-word surface variants (' apple'/'Apple'/'APPLE'/...)
        * language : cross-lingual translations (apple <-> 苹果)
        * semantic : same-class words (dog/horse/lion)
        baseline  : unrelated (52 deg 5-NN)
  P2  Neighborhood purity. For seed words across classes: among the 30-NN rows,
      classify each decoded neighbor as same-word-variant / CJK-translation /
      other; report the split.  Does the head organize by form + meaning?
  P3  Class layout on the sphere. Intra-class vs inter-class angular distance
      (do semantic classes form separated caps?) + class-crossing rate in
      neighborhoods.
  P4  Cross-model. Intra/inter separability ratio on Qwen, GPT-2, Pythia,
      Gemma (whatever words are single tokens).

Run: python eval_semantic_map.py
"""
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as st_load
from transformers import AutoTokenizer

CACHE = Path.home() / '.cache/huggingface' / 'hub'

CLASSES = {
    'food':    ['apple', 'banana', 'bread', 'cheese', 'chicken', 'grape', 'honey', 'milk', 'rice', 'soup'],
    'animal':  ['dog', 'cat', 'horse', 'lion', 'bird', 'wolf', 'tiger', 'fish', 'snake', 'rabbit'],
    'color':   ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink', 'purple', 'brown', 'gray'],
    'city':    ['Paris', 'London', 'Tokyo', 'Berlin', 'Rome', 'Moscow', 'Cairo', 'Delhi', 'Seoul', 'Madrid'],
    'number':  ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'],
    'nature':  ['ocean', 'tree', 'mountain', 'river', 'forest', 'flower', 'stone', 'cloud', 'star', 'moon'],
}
# bilingual pairs we hope exist as single tokens in Qwen's vocab
BILINGUAL = [('apple', '\u82f9\u679c'), ('dog', '\u72d7'), ('cat', '\u732b'),
             ('red', '\u7ea2'), ('king', '\u56fd\u738b'), ('ocean', '\u6d77\u6d0b'),
             ('water', '\u6c34'), ('hand', '\u624b'), ('one', '\u4e00'), ('tree', '\u6811')]

MODELS = [
    ('Qwen/Qwen2-0.5B-Instruct', 'models--Qwen--Qwen2-0.5B-Instruct'),
    ('openai-community/gpt2', 'models--openai-community--gpt2'),
    ('EleutherAI/pythia-160m', 'models--EleutherAI--pythia-160m'),
    ('google/gemma-3-1b-it', 'models--google--gemma-3-1b-it'),
]


def load_rows(model_dir, n=None):
    snap = sorted((CACHE / model_dir / 'snapshots').glob('*'))[0]
    st = sorted(snap.glob('*.safetensors'))[0]
    d = st_load(str(st))
    # pick the LM-head-like weight: the 2D weight with the LARGEST first dim
    cands = [(k, int(d[k].shape[0])) for k in d if 'weight' in k and d[k].ndim == 2]
    key = max(cands, key=lambda kv: kv[1])[0]
    W = d[key].float().numpy()
    if n:
        W = W[:n]
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def decode_map(tok, ids):
    return {int(i): tok.decode([int(i)], skip_special_tokens=True) for i in ids}


def main():
    rng = np.random.default_rng(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ============ P1/P2/P3 on Qwen (detailed) ============
    print("=" * 78)
    print("P1/P2/P3: Qwen2-0.5B head-row semantic topography")
    print("=" * 78)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B-Instruct')
    Wn = load_rows('models--Qwen--Qwen2-0.5B-Instruct')
    V = len(Wn)
    Wt = torch.as_tensor(Wn.astype(np.float32), device=dev)

    # ---- word -> token id, verify single-token ----
    word2id = {}
    for w in sorted({w for cls in CLASSES.values() for w in cls}) + \
              sum((list(p) for p in BILINGUAL), []):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    print(f"single-token words available: {len(word2id)}/{sum(len(c) for c in CLASSES.values()) + 2 * len(BILINGUAL)}")

    # ---- P1: nested scales ----
    def ang(a, b):
        return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))

    # identity variants: find all case/prefix variants of a word in vocab
    def variants_of(word):
        base = word.strip().lower()
        out = {}
        for tid in range(min(V, 200_000)):
            t = tok.decode([tid], skip_special_tokens=True)
            if not t or not t.strip():
                continue
            if t.strip().lower() == base and len(t) <= len(base) + 4:
                out[t] = tid
            if len(out) >= 8:
                break
        return out

    print("\n[P1] nested angular scales (Qwen):")


    def stats(arr):
        a = np.asarray(arr)
        if len(a) == 0:
            return None
        return (len(a), float(np.median(a)), float(np.percentile(a, 25)),
                float(np.percentile(a, 75)))


    # identity
    id_angles = []
    for w in ['apple', 'dog', 'red', 'king', 'money', 'door']:
        vs = variants_of(w)
        if len(vs) >= 2:
            ids = list(vs.values())
            for i in range(min(4, len(ids))):
                for j in range(i + 1, min(4, len(ids))):
                    id_angles.append(ang(Wn[ids[i]], Wn[ids[j]]))
    s = stats(id_angles)
    if s:
        print(f"   identity (same-word variants):  n={s[0]:3d} med {s[1]:5.1f} deg",
              f"  [Q25 {s[2]:.1f}, Q75 {s[3]:.1f}]")
    else:
        print("   identity: none")
    # bilingual
    bi_angles = []
    resolved = []
    for en, zh in BILINGUAL:
        if en in word2id and zh in word2id:
            bi = (en, zh)
            resolved.append((en, zh))
            bi_angles.append(ang(Wn[word2id[en]], Wn[word2id[zh]]))
    s = stats(bi_angles)
    if s:
        print(f"   language   (EN<->CJK pairs):    n={s[0]:3d} med {s[1]:5.1f} deg",
              f"  [Q25 {s[2]:.1f}, Q75 {s[3]:.1f}]  pairs: {resolved}")
    else:
        print("   language: no single-token EN<->CJK pairs resolved")
    # intra-class
    intra = []
    for cls, words in CLASSES.items():
        ids = [word2id[w] for w in words if w in word2id]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                intra.append(ang(Wn[ids[i]], Wn[ids[j]]))
    s = stats(intra)
    if s:
        print(f"   semantic   (same-class pairs):  n={s[0]:4d} med {s[1]:5.1f} deg",
              f"  [Q25 {s[2]:.1f}, Q75 {s[3]:.1f}]")
    else:
        print("   semantic: none")
    # baseline unrelated
    rids = rng.choice(V, 400, replace=False)
    Sm = Wt[torch.as_tensor(rids)]
    c5 = (Sm @ Sm.T).clamp(-1, 1)
    mask = torch.triu(torch.ones(len(rids), len(rids)), 1).bool()
    c5 = c5[mask]
    base = np.degrees(np.arccos(c5.cpu().numpy()))
    print(f"   unrelated  (random row pairs):  n={len(base):5d} med {np.median(base):5.1f} deg"
          f"  [Q25 {np.percentile(base,25):.1f}, Q75 {np.percentile(base,75):.1f}]")

    # ---- P2: neighborhood purity ----
    print("\n[P2] neighborhood purity (30-NN of seed words, decoded):")
    seeds = ['apple', 'Paris', 'dog', 'red', 'king', 'money', 'ocean', 'one']
    Wt64 = Wt[: V]  # full-vocab ranking needs all rows
    purity = []
    for w in seeds:
        if w not in word2id:
            continue
        tid = word2id[w]
        c = torch.clamp(Wt64 @ Wt[tid], -1, 1)
        c[tid] = -1
        nbrs = c.topk(30).indices.cpu().numpy()
        dec = [tok.decode([int(i)], skip_special_tokens=True) for i in nbrs]
        variant = sum(1 for t in dec if t.strip().lower() == w.lower())
        cjk = sum(1 for t in dec if any('\u4e00' <= ch <= '\u9fff' for ch in t))
        purity.append((w, variant, cjk))
        top = dec[:8]
        print(f"   {w:6s} tid={tid:6d}  var={variant}/30  cjk={cjk}/30 | "
              f"{[t if t.strip() else '«»' for t in top]}")
    if purity:
        v = np.mean([p[1] for p in purity]); cj = np.mean([p[2] for p in purity])
        print(f"   AVG: same-word-variant {v:.1f}/30, CJK-translation {cj:.1f}/30 "
              f"-> {v + cj:.1f}/30 ({100 * (v + cj) / 30:.0f}%) form+translation")

    # ---- P3: class layout ----
    print("\n[P3] class separability (Qwen):")
    inter = []
    cls_ids = {}
    for cls, words in CLASSES.items():
        ids = [word2id[w] for w in words if w in word2id]
        if len(ids) >= 5:
            cls_ids[cls] = ids
    keys = list(cls_ids.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for a in cls_ids[keys[i]]:
                for b in cls_ids[keys[j]]:
                    inter.append(ang(Wn[a], Wn[b]))
    intra_arr = np.array(intra); inter_arr = np.array(inter)
    if len(intra_arr) and len(inter_arr):
        ratio = np.median(intra_arr) / np.median(inter_arr)
        print(f"   intra med {np.median(intra_arr):5.1f} deg  |  inter med {np.median(inter_arr):5.1f} deg",
              f"  |  ratio {ratio:.3f}")
        print(f"   -> classes {'ARE' if ratio < 0.9 else 'are NOT'} separated caps ",
              f"({'yes' if ratio < 0.9 else 'no'}, medians differ by {np.median(inter_arr) - np.median(intra_arr):.1f} deg)")
    else:
        print("   (no class data)")
    # neighborhood class-crossing: does 'dog' 30-NN contain other animal class words?
    print("   class-crossing in 30-NN:")
    for cls, words in CLASSES.items():
        for w, wid in list(word2id.items()):
            if w in words and w in ('dog', 'apple', 'red', 'Paris', 'one'):
                c = torch.clamp(Wt64 @ Wt[wid], -1, 1); c[wid] = -1
                nbrs = c.topk(30).indices.cpu().numpy()
                dec = {tok.decode([int(i)], skip_special_tokens=True).strip().lower()
                       for i in nbrs}
                own = sum(1 for o in words if o.lower() in dec)
                print(f"   {w:6s}: {own}/30 of its class's neighbors present in 30-NN")

    # ============ P4: cross-model ============
    print("\n" + "=" * 78)
    print("P4: cross-model intra/inter separability ratio")
    print("=" * 78)
    for name, mdir in MODELS:
        try:
            tk = AutoTokenizer.from_pretrained(name)
        except Exception as e:
            print(f"{name:24s} tok-fail {e}"); continue
        try:
            Wn_m = load_rows(mdir)
        except Exception as e:
            print(f"{name:24s} load-fail {e}"); continue
        w2 = {}
        for cls, words in CLASSES.items():
            for w in words:
                ids = tk(' ' + w, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    w2[w] = int(ids[0])
        intra_m, inter_m = [], []
        cids = {}
        for cls, words in CLASSES.items():
            ids = [w2[w] for w in words if w in w2]
            if len(ids) >= 3:
                cids[cls] = ids
        keys_m = list(cids.keys())
        for cls, ids in cids.items():
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    intra_m.append(ang(Wn_m[ids[i]], Wn_m[ids[j]]))
        for i in range(len(keys_m)):
            for j in range(i + 1, len(keys_m)):
                for a in cids[keys_m[i]]:
                    for b in cids[keys_m[j]]:
                        inter_m.append(ang(Wn_m[a], Wn_m[b]))
        if intra_m and inter_m:
            r = np.median(intra_m) / np.median(inter_m)
            print(f"   {name:26s} intra med {np.median(intra_m):5.1f}  inter med "
                  f"{np.median(inter_m):5.1f}  ratio {r:.3f}  (classes {len(cids)}/6)")


if __name__ == '__main__':
    main()