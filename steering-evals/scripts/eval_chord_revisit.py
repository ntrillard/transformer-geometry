#!/usr/bin/env python3
"""eval_chord_revisit.py — HEAD-TO-HEAD: old chord technique vs current
closed-form recipe, same model (gemma-3-1b-it), same harness, <=10s.

The old chord technique (eval_chord_steering.py, Qwen era) steered
PERSISTENTLY toward the semantic-class centroid:
    chord C = normalize(mean of note rows W_ti over the class)
    each step: h += 0.3*||h||*g(C)  (additive tangent, duty=1)
The current recipe steers ONCE with the law budget + anti-last loop-break.

Modes (food class, prompt 'For dinner I made'):
  chord-pers   old way: persistent centroid steer (duty 1, a=0.3)
  note-pers    old way: persistent single-note steer ('chicken')
  chord-once   chord centroid, ONCE at step 0 (law budget 2*gap_C/97+0.02)
  chord-once+anti  chord once + anti-last 0.15 every step
  note-once+anti   SINGLE note once + anti-last  (current champion)

Metrics (old T4's family view + our modern view):
  x      = mean total food-word occurrences in the text
  dis    = mean distinct food words seen /10
  div    = unique/tokens
  maxrun = longest run of a single token
  rep4   = 4-gram repetition rate
  plant  = any food word in first 10 tokens

Run: timeout 60 python3 -u eval_chord_revisit.py  # GEMMA-3-1B
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
NTOK = 12
SEEDS = 1
A_REP = 0.15
A_PERS = 0.3       # old persistent additive budget
FOOD = ['apple', 'banana', 'bread', 'cheese', 'chicken', 'grape',
        'honey', 'milk', 'rice', 'soup']
NOTE = 'chicken'   # single-note champion


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()

    # note rows for the food class (single-token members)
    word2id = {}
    for w in FOOD + [NOTE]:
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    food_ids = [word2id[w] for w in FOOD if w in word2id]
    tid_note = word2id[NOTE]

    ids0 = tok(PROMPT, add_special_tokens=False,
               return_tensors='pt').input_ids.to(DEV)
    cf = {}

    def hook_c(m, i, o):
        cf['v'] = o[0, -1, :].float()

    h = model.model.norm.register_forward_hook(hook_c)
    with torch.no_grad():
        L0 = model(ids0).logits[0, -1].float()
    h.remove()
    native = int(L0.argmax())
    vf = cf['v'].float()
    vfn = vf / vf.norm()
    Wn = W / W.norm(dim=1, keepdim=True)
    # chord centroid (unit) -- the old chord direction
    Wf = Wn[food_ids].float()
    C = Wf.mean(0)
    C = C / C.norm()
    # law budgets: chord gap = native logit - max food-family logit?
    gap_C = float((L0[native] - L0[food_ids]).min())   # nearest food note
    A_C = 2 * gap_C / 97.0 + 0.02
    # single-note budget (the champion)
    gap_n = float(L0[native] - L0[tid_note])
    A_N = 2 * gap_n / 97.0 + 0.02
    # tangent toward chord centroid from the natural final state
    tau_C = C - (vfn @ C) * vfn
    g_C = tau_C / tau_C.norm()
    # tangent toward the note
    Wt = W[tid_note].float()
    tau_n = Wt - (vfn @ Wt) * vfn
    g_n = tau_n / tau_n.norm()
    nname = tok.decode([native])
    print(f"[{MODEL}] {PROMPT!r} native={nname!r} food-n={len(food_ids)} "
          f"gap_C={gap_C:.2f} a_C={A_C:.3f} gap_n={gap_n:.2f} "
          f"a_N={A_N:.3f}")
    print(f"  chord C = mean of {[tok.decode([i]) for i in food_ids]}")

    def anti(vv, tid, amt=A_REP):
        vv1 = vv / vv.norm()
        Wb = W[tid].float()
        tauv = Wb - (vv1 @ Wb) * vv1
        gl = -tauv / tauv.norm()
        return (vv1 * math.cos(amt) + gl * math.sin(amt)) * vv.norm()

    def gen(mode):
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                if mode in ('chord-pers', 'note-pers'):
                    # OLD WAY: re-aim from the LIVE state every step
                    dn = C if mode == 'chord-pers' else Wn[tid_note]
                    dn_t = dn.float()

                    def inj(m, i, o, q=dn_t):
                        out = o.clone()
                        hh = out[0, -1, :].float()
                        hn = hh / hh.norm()
                        g = q - (q @ hn) * hn
                        g = g / max(g.norm().item(), 1e-8)
                        h2 = hh + A_PERS * hh.norm() * g
                        out[0, -1, :] = (hh.norm() * h2 / h2.norm()).to(
                            out.dtype)
                        return out

                    hi = model.model.norm.register_forward_hook(inj)
                    try:
                        with torch.no_grad():
                            L = model(ids).logits[0, -1].float()
                    finally:
                        hi.remove()
                else:
                    # MODERN: precomputed inject
                    vv = vf
                    if step == 0:
                        if mode.startswith('chord'):
                            vv = (vfn * math.cos(A_C) +
                                  g_C * math.sin(A_C)) * vf.norm()
                        else:
                            vv = (vfn * math.cos(A_N) +
                                  g_n * math.sin(A_N)) * vf.norm()
                    if mode.endswith('+anti') and toks:
                        vv = anti(vv, toks[-1])

                    def inj(m, i, o, p=vv):
                        out = o.clone()
                        out[0, -1, :] = torch.as_tensor(
                            p, dtype=out.dtype, device=out.device)
                        return out

                    hi = model.model.norm.register_forward_hook(inj)
                    try:
                        with torch.no_grad():
                            L = model(ids).logits[0, -1].float()
                    finally:
                        hi.remove()
                p = torch.softmax(L.float(), dim=0)
                q = p.clone(); order = q.argsort(descending=True)
                k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
                msk = torch.zeros_like(q); msk[order[:k]] = 1
                qq = (q * msk) / (q * msk).sum()
                nxt = int(torch.multinomial(qq, 1))
                toks.append(int(nxt))
                ids = torch.cat([ids,
                                 torch.tensor([[nxt]], device=ids.device)],
                                dim=1)
            allres.append(toks)
        return allres

    print(f"  {'mode':>15} {'plant':>6} {'x':>5} {'dis':>5} {'div':>6} "
          f"{'maxrun':>6} {'rep4':>6}  samples")
    for mode in ('chord-pers', 'chord-once',
                 'chord-once+anti', 'note-once+anti'):
        gs = gen(mode)
        food_tids = set(food_ids)
        x = np.mean([sum(1 for t in g if t in food_tids) for g in gs])
        dis = np.mean([len({t for t in g if t in food_tids}) for g in gs])
        plant = np.mean([1.0 if any(t in g[:10] for t in food_tids)
                         else 0.0 for g in gs])
        div = np.mean([len(set(g)) / len(g) for g in gs])
        maxrun = np.mean([max((sum(1 for _ in grp) for _, grp in
                              __import__('itertools').groupby(g)),
                              default=0) for g in gs])
        rp = np.mean([rep4(g) for g in gs])
        print(f"  {mode:>15} {plant:>6.2f} {x:>5.1f} {dis:>5.1f} "
              f"{div:>6.2f} {maxrun:>6.1f} {rp:>6.2f}  "
              f"{[tok.decode(g)[:40] for g in gs]}", flush=True)
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()