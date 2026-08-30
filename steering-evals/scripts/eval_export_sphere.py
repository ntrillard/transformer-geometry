#!/usr/bin/env python3
"""Export the sphere-multiverse geometry to JSON for the browser 3js walker.

Per layer sphere (Qwen): BOS axis u, ring PCA frame (r1,r2), start state h0,
topic centroids + azimuths, and a ring-azimuth -> top-token sample table
(every 10 deg along the ring longitude: the vocab token whose row is closest
to that direction on the readout sphere).  Writes web/sphere_data.json.

Run:  python eval_export_sphere.py     (Qwen, ~15 s, no GPU-heavy work)
"""
import json
import math
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = 'Qwen/Qwen2-0.5B-Instruct'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYERS = [0, 4, 9, 13, 18, 23]
OUT = 'web/sphere_data.json'


def norm(v):
    return v / np.linalg.norm(v)


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
    member_words = {cls: [w for w in words if w in word2id]
                    for cls, words in CLASSES.items()}

    pid = tok('Once upon a time', add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
    hidden = [h[0].cpu().float().numpy() for h in hid.hidden_states]

    data = {'model': MODEL, 'layers': []}
    for l in LAYERS:
        u = norm(hidden[l + 1][0, :])
        hs = hidden[l + 1][-1, :]
        h0 = norm(hs)
        cent = {c: norm(Wn[i].mean(0) - (Wn[i].mean(0) @ u) * u)
                for c, i in topics.items()}
        C0 = np.stack(list(cent.values())) - np.stack(list(cent.values())).mean(0)
        _, eig = np.linalg.eigh(C0 @ C0.T)
        Vp = C0.T @ eig[:, -2:]
        Vp = Vp / np.linalg.norm(Vp, axis=0, keepdims=True)
        r1, r2 = Vp[:, 0], Vp[:, 1]
        az = {}
        for c in names:
            p = np.array([cent[c] @ r1, cent[c] @ r2])
            az[c] = (math.degrees(math.atan2(p[1], p[0])) + 360) % 360
        # ring token samples (every 10 deg azimuth, closest vocab row)
        ring = []
        for a in range(0, 360, 10):
            d = math.cos(math.radians(a)) * r1 + math.sin(math.radians(a)) * r2
            tid = int((Wn @ d).argmax())
            ring.append({'az': a, 'token': tok.decode([tid], skip_special_tokens=True).strip()})
        data['layers'].append({
            'layer': l,
            'u': u.tolist(), 'r1': r1.tolist(), 'r2': r2.tolist(), 'h0': h0.tolist(),
            'centroids': {c: cent[c].tolist() for c in names},
            'azimuths': {c: round(az[c], 1) for c in names},
            'members': {c: member_words[c] for c in names},
            'ring': ring,
        })
        print(f"  layer {l:2d}: az { {c: round(az[c]) for c in names} }")
    import os
    os.makedirs('web', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(data, f)
    print(f"wrote {OUT}  ({os.path.getsize(OUT)//1024} KB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()