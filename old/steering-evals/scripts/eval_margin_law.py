#!/usr/bin/env python3
"""eval_margin_law.py — BIG LEAP: an LLM's confidence = its OWN steering
margin (generation is self-steering).

Gemma-3-1B only, 1 forward x 5 prompts, <=10s.

The unification claim: every token generation IS a steering decision. The
model sits at state vf; between its native winner n and the 2nd choice s
there is a crossing angle alpha*(s) = atan2(-A_s, B_s) (THE SAME closed
form as our steering, fbb2055). If 'confidence' (top-1 softmax prob) is
just the model's own steering margin (distance from the n-s crossing),
then:
  - high prob <=> far from the s-crossing (large alpha*(s))
  - low prob (uncertain) <=> sitting ON the crossing (small alpha*(s))
and generation = self-steering toward the highest-margin option.

Tests (5 prompts -> 5 (prob, alpha*) points):
  Q1  corr(softmax_1, alpha*_2nd)  -> the margin law
  Q2  spectral version: same corr using logit gap / slope (the linear
      margin = -gap0/slope), incl. the top-3 choices (3 points/prompt)
  Q3  'unsure = steerable' corollary: rank_t(target) of an OUTSIDE target
      vs its alpha*: blocking distance (rank) IS a steering-margin too.
  Q4  the 2nd-choice state: does the s-vs-n difference = rotate toward
      W_s by ~alpha*(s)? (sample: regenerate with forced s, compare the
      'chosen' trajectory to our predicted steer vector) [2 fwd]

Run: timeout 60 python3 -u eval_margin_law.py  # GEMMA-3-1B
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = ['For dinner I made', 'I went to the store and bought',
           'The recipe calls for', 'In my kitchen I have',
           'There once was a chicken']


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    NL = model.config.num_hidden_layers

    datas = []   # (prob1, a2nd, gap2nd, slope2nd, volatile, pname)
    for pidx, PROMPT in enumerate(PROMPTS):
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        caps = {}

        def mk(li):
            def h(m, i, o):
                caps[li] = o[0, -1, :].float()
            return h

        hooks = [model.model.layers[li].register_forward_hook(mk(li))
                 for li in range(NL)]
        hooks.append(model.model.norm.register_forward_hook(mk(NL)))
        with torch.no_grad():
            L0 = model(ids).logits[0, -1].float()
        for h in hooks:
            h.remove()
        vf = caps[NL].cpu().numpy().astype(np.float64)
        vfn = vf / (np.linalg.norm(vf) + 1e-12)
        p = torch.softmax(L0, dim=0).cpu().numpy()
        prob1 = float(p.max())
        order = np.argsort(-L0.cpu().numpy())
        n1, n2, n3 = int(order[0]), int(order[1]), int(order[2])
        gap12 = float(L0[n1] - L0[n2])
        gap13 = float(L0[n1] - L0[n3])

        def astar(tid):
            Wt = lm_w[tid].detach().float().cpu().numpy().astype(np.float64)
            Wn = lm_w[n1].detach().float().cpu().numpy().astype(np.float64)
            A = float(vfn @ (Wt - Wn))
            tau = Wt - (vfn @ Wt) * vfn
            B = float(tau @ (Wt - Wn)) / (np.linalg.norm(tau) + 1e-12)
            return math.atan2(-A, B)

        a2 = astar(n2)
        a3 = astar(n3)
        # slope proxies: gap / (a2 - 0) approx (a2 = crossing from gap0~gap12)
        slope2 = -gap12 / max(a2, 1e-6)
        slope3 = -gap13 / max(a3, 1e-6)
        # Q4: forced-2nd forward -> trajectory of choosing s
        ids2 = torch.cat([ids, torch.tensor([[n2]], device=ids.device)], dim=1)
        caps2 = {}

        def mk2(li):
            def h(m, i, o):
                caps2[li] = o[0, -1, :].float()
            return h

        hooks2 = [model.model.layers[li].register_forward_hook(mk2(li))
                  for li in range(NL)]
        hooks2.append(model.model.norm.register_forward_hook(mk2(NL)))
        with torch.no_grad():
            model(ids2)
        for h in hooks2:
            h.remove()
        vf2 = caps2[NL].cpu().numpy().astype(np.float64)  # after ' s'
        # the model's own move toward s (next-position state):
        d = vf2 - vf
        # our predicted steer direction toward W_s
        Wt2 = lm_w[n2].detach().float().cpu().numpy().astype(np.float64)
        tdir = Wt2 - (vfn @ Wt2) * vfn
        tdir = tdir / (np.linalg.norm(tdir) + 1e-12)
        align = float(np.dot(d, tdir) / (np.linalg.norm(d) + 1e-12))
        datas.append((prob1, a2, gap12, slope2, a3, align, PROMPT))
        print(f"P{pidx} {PROMPT!r:34} p1={prob1:.3f} "
              f"n1={tok.decode([n1])!r:6} n2={tok.decode([n2])!r:5} "
              f"a*(2nd)={a2:+.3f} gap12={gap12:+.2f} slope={slope2:+.1f} "
              f"align(forced2nd)={align:+.3f}", flush=True)

    datas_full = list(datas)
    datas = np.array([d[:5] for d in datas], dtype=float)
    probs, a2s, gaps, slops = datas[:, 0], datas[:, 1], datas[:, 2], datas[:, 3]
    print(f"\n  MARGIN LAW (corr across {len(PROMPTS)} prompts):")
    cc_pa = float(np.corrcoef(probs, np.abs(a2s))[0, 1])
    cc_pg = float(np.corrcoef(probs, gaps)[0, 1])
    print(f"    corr(prob1, |a*(2nd)|) = {cc_pa:+.3f}  "
          f"(the steering-margin law)")
    print(f"    corr(prob1, gap12)     = {cc_pg:+.3f}  "
          f"(linear margin, order baseline)")
    print(f"    + unsiggned: higher |a*2nd| = MORE certain (positive = "
          f"'confidence = geometric steering margin')")
    aligns = [d_[5] for d_ in datas_full if np.isfinite(d_[5])]
    if aligns:
        print(f"  Q4 forced-2nd alignment with our steer dir: "
              f"mean={np.mean(aligns):+.3f}  (1 = the model steers toward "
              f"its 2nd choice EXACTLY along our tangent)")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()