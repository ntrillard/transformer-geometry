#!/usr/bin/env python3
"""eval_switch_big.py — sequential topic switching on LARGER models.
argv[1]=model  argv[2]=prompt   env: SW_FREE=1 (baseline), SW_NO_SUB=1

Controller: readout graft (rotate 10deg toward topic member's word
direction) + anti of the planted member + rep-penalty nucleus decode.
v2 HONESTY FIXES: (1) stop at <eos> (never sample garbage past it);
(2) FREE baseline arm - run with NO hooks to compare steered vs the
model's own free continuation (so we see if steering hurts); (3) real
per-seed text-quality metric (rep4, run, word-dup, eos) alongside the
topic-hit count; (4) MULTI-WORD family members written with SPACES
(e.g. 'new york' -> 2 tokens) - phrase direction = sum of member token
directions, anti blocks the planted member; (5) SUBSTRING anti: also
suppress any sampled token whose text CONTAINS the planted member's
words (catches fused tokens like 'mind-wandered-to-thoughts-of-paris',
which token-id anti cannot touch).

One model, no quant. Run: HF_TOKEN=<tok> python3 -u eval_switch_big.py \
    google/gemma-3-4b-pt "The room was quiet, and my mind wandered to thoughts of"
"""
import csv
import itertools
import math
import os
import sys
import time
from pathlib import Path

import torch
import transformers

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'google/gemma-3-1b-pt'
PROMPT = (sys.argv[2] if len(sys.argv) > 2
          else 'The whole group sat down and began to discuss')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 64
SEEDS = [0, 1]
ANGLE = 10.0

# optional per-step calibrated angles (order: sorted SWITCHES steps)

# e.g. SW_ANGLES=4,8,10,12 for Qwen2-1.5B on the discuss prompt

_step_angles = [float(x) for x in

                os.environ.get('SW_ANGLES', '').split(',') if x]

ONLINE_CALIB = os.environ.get('SW_ONLINE') == '1'

CALIB_SWEEP = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]

CALIB_MARGIN = 2.0


PEN = 0.5

SUSTAIN = True                 # keep planted member blocked all segment

SW_FREE = os.environ.get('SW_FREE') == '1'     # baseline arm: no hooks

SW_NO_SUB = os.environ.get('SW_NO_SUB') == '1' # disable substring anti
_stop = ''.join(c for c in PROMPT[:20] if c.isalnum())
OUT = Path(f'../steering_geometry_results/switch_big_'
           f'{MODEL.split("/")[-1]}_{_stop}.csv')
SWITCHES = {0: 'city', 16: 'animal', 32: 'food', 48: 'nature'}
SEG_N = 16
# family members: single words and MULTI-WORD phrases (space-separated)
FAMILIES = {
    'city':   ['paris', 'london', 'berlin', 'madrid', 'tokyo', 'new york'],
    'animal': ['cat', 'dog', 'bird', 'bear', 'horse', 'polar bear'],
    'food':   ['pizza', 'sushi', 'pasta', 'burger', 'sushi bar'],
    'nature': ['forest', 'rice', 'water', 'sun', 'tree'],
}


def _lm_head_fp16(model):
    w = model.lm_head.weight.detach()
    if hasattr(w, 'quant_state') and w.quant_state is not None:
        import bitsandbytes as bnb
        try:
            qs = w.quant_state.cpu()
        except Exception:
            qs = w.quant_state
        return bnb.functional.dequantize_4bit(w.data.cpu(), qs).float().cpu()
    return w.cpu().float()


def readout_model(model):
    if hasattr(model.model, 'norm'):
        return model.model
    lm = getattr(model.model, 'language_model', None)
    if lm is not None and hasattr(lm, 'norm'):
        return lm
    raise RuntimeError(f'{MODEL}: no readout norm found '
                       f'(model.model.norm / language_model.norm)')


def rep4(toks):
    if len(toks) < 4:
        return 0.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return sum(1 for i in range(len(toks) - 3) if n4[i] in n4[i + 1:]) \
        / (len(toks) - 3)


def quality(toks, txt):
    """real text metric - dict of scores."""
    words = [w.lower() for w in txt.split() if w]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    mr = max((sum(1 for _ in grp) for _, grp in
              itertools.groupby(toks)), default=0)
    return dict(
        eos='<eos>' in txt,
        rep4=round(rep4(toks), 3),
        max_run=mr,
        max_wfreq=max(freq.values()) if freq else 0,
        n_words=len(words),
        good=(rep4(toks) == 0.0 and mr <= 2
              and (max(freq.values()) if freq else 0) <= 2),
    )


def main():
    t0 = time.time()
    print(f'\nLoading {MODEL} (bf16, no quant) ...')
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    eos_id = int(tok.eos_token_id)
    RO_norm = readout_model(model).norm
    W = _lm_head_fp16(model)
    Wn = (W / W.norm(dim=1, keepdim=True)).float()

    # members: name -> (ids, dir)   (multi-word phrases supported)
    members = {}
    famids = {}
    for fam, words in FAMILIES.items():
        mem = []
        for w in words:
            ids = tok(' ' + w, add_special_tokens=False).input_ids
            ids = [int(i) for i in ids]
            d = Wn[ids].float().sum(0)
            d = d / d.norm()
            mem.append((w, ids, d))
        members[fam] = mem
        famids[fam] = [i for _, ids, _ in mem for i in ids]
        print(f"  family {fam:>6}: "
              + ', '.join(f'{w}({len(ids)})' for w, ids, _ in mem))

    def closest_member(vv, fam):
        u = vv / vv.norm()
        best = None
        for w, ids, d in members[fam]:
            s = float(d.to(DEV) @ u)
            if best is None or s > best[0]:
                best = (s, w, ids)
        _, w, ids = best
        return w, ids, ids[0]            # name, all ids, graft target

    def rot_to_angle(vv, tid, theta):
        a = math.radians(theta)
        v1 = vv / vv.norm()
        Wb = Wn[tid].float().to(DEV)
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(a) + g * math.sin(a)) * vv.norm()

    def sample(L, prefix, block_words=None):
        L = torch.nan_to_num(L.float(), nan=-50.0).clamp(-50.0, 50.0)
        p = torch.softmax(L, 0)
        q = p.clone()
        order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q)
        msk[order[:k]] = 1
        qq = (q * msk)
        # SUBSTRING anti: drop top candidates whose text contains a
        # planted member's word (cracks fused tokens, multiword phrases)
        if block_words:
            top = order[:200].tolist()
            dec = tok.batch_decode([[i] for i in top])
            drop = [i for i, s in zip(top, dec)
                    if any(w in s.lower() for w in block_words)]
            for i in drop:
                qq[i] = 0.0
        for t in set(prefix):
            c = prefix.count(t)
            if c:
                qq[t] = qq[t] * (PEN ** c)
        tot = qq.sum()
        if tot <= 0 or not torch.isfinite(tot):
            qq = torch.ones_like(qq)
        qq = qq / qq.sum()
        return int(torch.multinomial(qq, 1))

    def forward(ids, inj_p=None, anti_ids=None):
        hs = []
        try:
            if inj_p is not None:
                def inj(m, i, o, p=inj_p):
                    o[0, -1, :] = torch.as_tensor(p, dtype=o.dtype,
                                                  device=o.device)
                hs.append(RO_norm.register_forward_hook(inj))
            if anti_ids:
                def anti(m, i, o, aids=anti_ids):
                    o[0, -1, aids] = -30.0
                hs.append(model.lm_head.register_forward_hook(anti))
            with torch.no_grad():
                return model(ids).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()

    def capture_v(ids):
        vc = {}
        hk = RO_norm.register_forward_hook(
            lambda m, i, o: vc.__setitem__('v', o[0, -1, :].float()))
        try:
            with torch.no_grad():
                model(ids).logits[0, -1].float()
        finally:
            hk.remove()
        return vc['v']



    def best_angle(ids, vv, tid):

        """in-situ recalibration: min angle making tid rank-1, +margin."""

        for th in CALIB_SWEEP:

            vp = rot_to_angle(vv, tid, th)

            L = forward(ids, inj_p=vp)

            if (int(L.argmax()) == tid

                    and torch.isfinite(L[tid])

                    and float(L[tid]) > float(L.max()) - 0.001):

                return th + CALIB_MARGIN

        return CALIB_SWEEP[-1]


    def run_schedule(sd, free=False):
        torch.manual_seed(sd)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        sampled = []
        hits = {}
        last_switch = -10
        last_name = None
        last_ids = None
        for step in range(NTOK):
            inj_p = None
            anti_ids = None
            block_words = None
            if not free:

                if step in SWITCHES:

                    fam = SWITCHES[step]

                    v = capture_v(ids)

                    name, ids_m, tgt = closest_member(v, fam)

                    if ONLINE_CALIB:

                        th = best_angle(ids, v, tgt)   # recalibrate NOW

                    elif _step_angles:

                        th = _step_angles[list(SWITCHES).index(step)]

                    else:

                        th = ANGLE

                    vp = rot_to_angle(v, tgt, th)

                    inj_p = vp

                    last_switch = step

                    last_name = name

                    last_ids = ids_m

                    # NOTE: block_words intentionally LEFT OFF here -

                    # substring-anti would zero the token we just planted
                elif last_ids is not None:
                    since = step - last_switch
                    if 1 <= since <= 2:
                        anti_ids = last_ids          # short window
                    elif SUSTAIN and since > 2:
                        anti_ids = [last_ids[0]]     # sustain: block first
                        if not SW_NO_SUB:
                            block_words = set(last_name.lower().split())
            L = forward(ids, inj_p=inj_p, anti_ids=anti_ids)
            nxt = sample(L, sampled, block_words=block_words)
            if nxt == eos_id:
                sampled.append(nxt)
                break
            sampled.append(nxt)
            if (not free) and step in SWITCHES:
                fam = SWITCHES[step]
                hits[step] = (fam, last_name, nxt in famids[fam])
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        return sampled, hits

    steps = sorted(SWITCHES)
    ltr = {st: chr(ord('A') + i) for i, st in enumerate(steps)}
    print(f"\n[{MODEL}] SWITCH v2 {PROMPT!r}  NTOK={NTOK} "
          f"PEN={PEN} SUSTAIN={SUSTAIN} substring_anti={not SW_NO_SUB} "
          f"free_arm={SW_FREE}")
    rows = []
    agg = {}
    for sd in SEEDS:
        toks_s, hits = run_schedule(sd, free=False)
        txt_s = tok.decode(toks_s)
        qs = quality(toks_s, txt_s)
        segs = [tok.decode(toks_s[i:i + SEG_N]) for i in
                range(0, NTOK, SEG_N)]
        print(f"\n  seed {sd} STEERED  q={qs['good']}  "
              f"(rep4={qs['rep4']} run={qs['max_run']} "
              f"wfreq={qs['max_wfreq']} eos={qs['eos']})")
        for i, st in enumerate(steps):
            if st in hits:
                fam, w, hit = hits[st]
                print(f"    switch@{st:>2} {fam:>6} -> {w:<12} "
                      f"{'HIT' if hit else 'miss'}  | {segs[i].strip()[:70]}")
        # FREE baseline
        toks_f, _ = run_schedule(sd, free=True)
        txt_f = tok.decode(toks_f)
        qf = quality(toks_f, txt_f)
        print(f"  seed {sd} FREE     q={qf['good']}  "
              f"(rep4={qf['rep4']} run={qf['max_run']} "
              f"wfreq={qf['max_wfreq']} eos={qf['eos']})")
        print(f"    free: {PROMPT} {txt_f[:100]}")
        print(f"    full: {PROMPT} {txt_s[:200]}")
        # honesty: did steering hurt the free run's quality?
        for k in ('rep4', 'max_run', 'max_wfreq'):
            agg.setdefault(k, 0)
        qd = sum(qs[k] for k in ('rep4', 'max_run', 'max_wfreq')) \
             - sum(qf[k] for k in ('rep4', 'max_run', 'max_wfreq'))
        print(f"    steer-vs-free delta (lower=steer worse, "
              f"higher=steer better): {qd:+.2f}")
        rows.append(dict(seed=sd, full=txt_s, free_full=txt_f,

                         steered_good=int(qs['good']),

                         free_good=int(qf['good']),

                         **{f'{ltr[st]}_hit': (st in hits and hits[st][2])

                            for st in steps}))



    # forced MULTI-WORD probes: graft straight toward each multiword

    # member so it actually gets exercised (closest may never pick it)

    print("\n  MULTI-WORD probes (graft toward phrase):")

    for fam, mw in [('city', 'new york'), ('animal', 'polar bear'),

                    ('food', 'sushi bar')]:

        mem = next((m for m in members[fam] if m[0] == mw), None)

        if mem is None:

            continue

        _, ids_m, _ = mem

        tgt = ids_m[0]

        ids0 = tok(PROMPT, add_special_tokens=False,

                   return_tensors='pt').input_ids.to(DEV)

        v = capture_v(ids0)

        vp = rot_to_angle(v, tgt, ANGLE)

        torch.manual_seed(0)

        toks = []

        ids = ids0.clone()

        for step in range(10):

            inj_p = vp if step == 0 else None

            anti_ids = [tgt] if step >= 1 else None

            bw = (set(mw.split()) if not SW_NO_SUB and step >= 1 else None)

            L = forward(ids, inj_p=inj_p, anti_ids=anti_ids)

            nxt = sample(L, toks, block_words=bw)

            if nxt == eos_id:

                toks.append(nxt)

                break

            toks.append(nxt)

            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)],

                            dim=1)

        txt = tok.decode(toks)

        print(f"    {fam:>6} -> {mw:<10} | {txt.strip()[:70]}")
    for i, st in enumerate(steps):
        n_hit = sum(1 for r in rows if r[f'{ltr[st]}_hit'])
        print(f"\n  switch@{st} ({SWITCHES[st]}): planted-any-family "
              f"hit={n_hit}/{len(SEEDS)}")
    ng = sum(1 for r in rows if r['steered_good'])
    nfg = sum(1 for r in rows if r['free_good'])
    print(f"\n  quality-good: STEERED {ng}/{len(SEEDS)}  FREE {nfg}/{len(SEEDS)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()