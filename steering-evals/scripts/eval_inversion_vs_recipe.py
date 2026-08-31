#!/usr/bin/env python3
"""eval_inversion_vs_recipe.py — FAIR head-to-head, CSV persisted.

Corrects eval_chord_revisit (aa423aa), which benchmarked only the chord
CENTER and wrongly declared the chord technique dead. The old arc's own
data (steering_geometry_results/chord_inversion.csv, Qwen-0.5B) shows
the real champions were 'root' (88.5%) and 'best_note' (94.8%) vs
'center' (34.4%). This reruns the comparison on gemma-3-1b-it with the
same single-forward resolution metric + a generative comparison, and
writes a CSV.

Part A: 5 targets, one rotation at the law budget (2*gap/97+0.02),
        single forward. Resolution = any of the 10 food words beats the
        full vocabulary in logits.
          center    : chord centroid       (the old loser, for the record)
          root      : most C-aligned note  (old champion)
          best_note : best-LOGIT food note (old champion variant)
          highest   : best-logit/97 pick
          chicken   : the current recipe's single note
Part B: 4 generative modes x 12 tok x 1 seed, top_p 0.9:
          inversion-once : best_note once + anti-last
          recipe-once    : chicken once + anti-last
          centroid-once  : center once + anti-last (the old loser)
          recipe-pers    : chicken persistent (old-style), for the record
        metrics: plant (any food word in first 10), x (# food words),
        dis (distinct food words), div, maxrun, rep4, sample text.

Writes ../steering_geometry_results/chord_vs_recipe_gemma.csv
"""
import csv
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
NTOK = 12
SEEDS = 1
A_REP = 0.15
A_PERS = 0.3
FOOD = ['apple', 'banana', 'bread', 'cheese', 'chicken', 'grape',
        'honey', 'milk', 'rice', 'soup']
OUT = Path('../steering_geometry_results/chord_vs_recipe_gemma.csv')


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()

    word2id = {}
    for w in FOOD:
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    food_ids = sorted(word2id.values())

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

    # ---- chord machinery (same as the old arc) ----
    Wf = Wn[food_ids].float()
    C = Wf.mean(0)
    C = C / C.norm()
    align = Wf @ C
    root_id = food_ids[int(align.argmax())]
    flog = L0[food_ids]
    best_id = food_ids[int(flog.argmax())]
    highest_id = food_ids[int((flog / 97).argmax())]
    fset = set(food_ids)

    residents = {
        'center': C,
        'root': Wn[root_id],
        'best_note': Wn[best_id],
        'highest': Wn[highest_id],
        'chicken': Wn[word2id['chicken']],
    }

    # law budget from the nearest food gap (family approach)
    A_LAW = 2 * float(L0[native] - L0[food_ids].max()) / 97.0 + 0.02

    def resolve(tdir):
        """one rotation toward tdir at the law budget; returns
        (resolved, argmax_food_word, fam_max, outsider_max)."""
        tau = tdir - (vfn @ tdir) * vfn
        g = tau / tau.norm()
        vv = (vfn * math.cos(A_LAW) + g * math.sin(A_LAW)) * vf.norm()

        def inj(m, i, o, p=vv):
            out = o.clone()
            out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                            device=out.device)
            return out

        hi = model.model.norm.register_forward_hook(inj)
        try:
            with torch.no_grad():
                L = model(ids0).logits[0, -1].float()
        finally:
            hi.remove()
        fam = L[food_ids]
        fammax = float(fam.max())
        argmax_id = food_ids[int(fam.argmax())]
        mask = torch.ones_like(L, dtype=torch.bool)
        mask[food_ids] = False
        outsider = float(L[mask].max())
        return (fammax > outsider, argmax_id, fammax, outsider)

    print(f"[{MODEL}] {PROMPT!r} food n={len(food_ids)} a_law={A_LAW:.3f} "
          f"root={tok.decode([root_id])!r} "
          f"best={tok.decode([best_id])!r} "
          f"highest={tok.decode([highest_id])!r}")
    rows = []
    print("  Part A  single-forward family resolution at the law budget:")
    print(f"    {'target':>10} {'resolved':>8} {'argmax':>10} "
          f"{'fam':>7} {'out':>7}")
    for tname, tdir in residents.items():
        resolved, aid, fammax, outsider = resolve(tdir)
        rows.append(dict(model=MODEL, prompt=PROMPT, target=tname,
                         budget=round(A_LAW, 4), resolved=resolved,
                         argmax_word=tok.decode([aid]),
                         family_max=round(fammax, 2),
                         outsider_max=round(outsider, 2)))
        print(f"    {tname:>10} {str(resolved):>8} "
              f"{tok.decode([aid])!r:>10} {fammax:+7.2f} "
              f"{outsider:+7.2f}", flush=True)

    def anti(vv, tid, amt=A_REP):
        vv1 = vv / vv.norm()
        Wb = W[tid].float()
        tauv = Wb - (vv1 @ Wb) * vv1
        gl = -tauv / tauv.norm()
        return (vv1 * math.cos(amt) + gl * math.sin(amt)) * vv.norm()

    def gen(tdir, once, with_anti):
        """steer toward tdir once at step 0 (or persistent re-aim),
        optionally anti-last the rest of the steps."""
        allres = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            ids = ids0.clone()
            toks = []
            for step in range(NTOK):
                if step == 0 or not once:
                    vv = vf
                    tgt = tdir - ((vv / vv.norm()) @ tdir) * (vv / vv.norm())
                    gg = tgt / tgt.norm()
                    ang = A_LAW if (once and step == 0) else A_PERS
                    vv = ((vv / vv.norm()) * math.cos(ang) +
                          gg * math.sin(ang)) * vv.norm()
                else:
                    vv = vf
                if with_anti and toks:
                    vv = anti(vv, toks[-1])

                def inj(m, i, o, p=vv):
                    out = o.clone()
                    out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                    device=out.device)
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

    modes = {
        'inversion-once': (residents['best_note'], True, True),
        'recipe-once': (residents['chicken'], True, True),
        'centroid-once': (residents['center'], True, True),
        'recipe-pers': (residents['chicken'], False, True),
    }
    print(f"  Part B  generative ({SEEDS} seed x {NTOK} tok, top_p 0.9):")
    print(f"    {'mode':>15} {'plant':>6} {'x':>4} {'dis':>4} "
          f"{'div':>5} {'maxrun':>6} {'rep4':>5}  sample")
    for mname, (td, once, wa) in modes.items():
        gs = gen(td, once, wa)
        for g in gs:
            x = sum(1 for t in g if t in fset)
            dis = len({t for t in g if t in fset})
            plant = 1.0 if any(t in g[:10] for t in fset) else 0.0
            div = len(set(g)) / len(g)
            mr = max((sum(1 for _ in grp) for _, grp in
                      itertools.groupby(g)), default=0)
            rp = rep4(g)
            print(f"    {mname:>15} {plant:>6.1f} {x:>4d} {dis:>4d} "
                  f"{div:>5.2f} {mr:>6d} {rp:>5.2f}  "
                  f"{tok.decode(g)[:38]!r}", flush=True)

    # ---- persist Part A ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()