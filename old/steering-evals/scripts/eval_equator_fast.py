#!/usr/bin/env python3
"""Fast equator battery (~5s each).  Tests the equatorial-fibering idea:

  E1 equator universality : median token-row angle to the BOS axis on Qwen,
      Pythia, GPT-2 (loads head rows + [bos]-state, ~5-8s each model).
  E2 chord polar tilt      : BOS-component (in deg) of each semantic-class
      centroid vs a single token row -- are wide chords polar-tilted?
  E3 latitude conservation : does a 17-deg steering arc move the state's
      latitude (angle to BOS axis)?  Inversion steering should keep it ~fixed.

Run: python eval_equator_fast.py  (each test prints ~5s)
"""
import math
import time
from pathlib import Path

import numpy as np
import torch

import steering_geometry_test as M
from safetensors.torch import load_file as st_load

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
MODELS = ['Qwen/Qwen2-0.5B-Instruct', 'EleutherAI/pythia-160m',
          'openai-community/gpt2']
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def ang_deg(x, y):
    return float(np.degrees(np.arccos(np.clip(np.dot(x, y), -1, 1))))


@torch.no_grad()
def bos_axis(model_id):
    """Return (BOS-axis unit vector, vocab-normalized Wn) using position-0 of
    a short prompt's final-layer hidden state as the latitude axis."""
    model, tok = M.load_model(model_id, dtype='fp16')
    if getattr(model, 'lm_head', None) is not None:
        W = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W = model.embed_out.weight.detach().cpu().float().numpy()
    vocab = model.config.vocab_size
    W = W[:vocab]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    prompt = "The"
    pids = tok(prompt, add_special_tokens=False,
               return_tensors='pt').input_ids.to(model.device)
    outs = model(pids, output_hidden_states=True)
    li = model.config.num_hidden_layers - 1
    h0 = outs.hidden_states[li + 1][0, 0, :].cpu().float().numpy()
    return h0 / np.linalg.norm(h0), Wn, model, tok


def e1():
    t0 = time.time()
    for m in MODELS:
        try:
            z, Wn, _, _ = bos_axis(m)
            lat = np.degrees(np.arccos(np.clip(Wn @ z, -1, 1)))
            print(f"E1 {m:28s} med {np.median(lat):5.1f}  "
                  f"p10 {np.percentile(lat,10):5.1f}  p90 {np.percentile(lat,90):5.1f}  "
                  f"eq {np.median(lat)>80 and np.median(lat)<100}")
        except Exception as ex:
            print(f"E1 {m:28s} FAIL {type(ex).__name__}: {ex}")
        print(f"   [{time.time()-t0:.0f}s]")


def e2():
    t0 = time.time()
    z, Wn, model, tok = bos_axis(MODELS[0])          # Qwen only, ~3-4s
    w2 = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            w2[w] = int(ids[0])
    print(f"E2 chord polar tilt (Qwen, {time.time()-t0:.0f}s):")
    for cls, words in CLASSES.items():
        ids = [w2[w] for w in words if w in w2]
        if len(ids) < 6:
            continue
        rows = Wn[ids]
        row_tilt = np.mean([ang_deg(r, z) for r in rows])
        C = rows.mean(0); C = C / np.linalg.norm(C)
        print(f"   {cls:8s} row-tilt med {row_tilt:5.1f} deg   "
              f"centroid-tilt {ang_deg(C, z):5.1f} deg   "
              f"tilt-shift {ang_deg(C, z) - row_tilt:+5.1f}")


def e3():
    t0 = time.time()
    z, Wn, model, tok = bos_axis(MODELS[0])
    w2 = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            w2[w] = int(ids[0])
    fam = np.array([w2['apple'], w2['bread'], w2['milk'], w2['rice']])
    # states: 4 prompts at final layer
    pids_list = ['The capital of France is', 'Once upon a time',
                 'Tell me something interesting:', 'To bake sourdough bread']
    lat0, lat1 = [], []
    li = model.config.num_hidden_layers - 1
    for p in pids_list:
        pid = tok(p, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        with torch.no_grad():
            outs = model(pid, output_hidden_states=True)
        u = outs.hidden_states[li + 1][0, -1, :].cpu().float().numpy()
        u = u / np.linalg.norm(u)
        # pick best-positioned family note for inversion
        fscore = np.clip(Wn[fam] @ u, -1, 1)
        best = Wn[fam[int(np.argmax(fscore))]]
        tau = M.tangent_direction(u, best)
        v = M.rotate_toward(u, tau, math.radians(17))
        lat0.append(ang_deg(u, z)); lat1.append(ang_deg(v, z))
    lat0, lat1 = np.array(lat0), np.array(lat1)
    print(f"E3 latitude conservation along 17-deg inversion arc "
          f"(Qwen, {time.time()-t0:.0f}s):")
    print(f"   state latitude before {np.median(lat0):5.1f} deg -> after "
          f"{np.median(lat1):5.1f} deg   (max delta {np.abs(lat1-lat0).max():.2f} deg)")


if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'e1'
    {'e1': e1, 'e2': e2, 'e3': e3}[which]()