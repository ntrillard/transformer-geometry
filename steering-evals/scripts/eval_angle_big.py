#!/usr/bin/env python3
"""eval_angle_big.py — recalibrate steer angle for the 4B at each switch
step. At the 4B scale, 10deg leaves the planted topic at rank-2, so we
sweep rotation angle at the REAL switch-step contexts and find the min
angle forcing rank-1 per step, then re-run the switch schedule with
per-step angles.

Flow:
  1) reference trajectory (theta=10) -> save context ids at switch steps
  2) at each context: for theta in sweep, inject, forward, record
     rank(target) + top1
  3) per-step theta* = min theta with rank==1
  4) re-run full switch schedule with per-step thetas

One model, no quant. Run: HF_TOKEN=<tok> timeout 300 python3 -u
eval_angle_big.py google/gemma-3-4b-pt
"""
import csv
import itertools
import math
import sys
import time
from pathlib import Path

import torch
import transformers

import steering_geometry_test as SGT

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'google/gemma-3-4b-pt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 64
SEEDS = [0, 1]
PEN = 0.5
SHORT = ['google/gemma-3-4b-pt', 'google/gemma-3-4b-it']
OUT = Path('../steering_geometry_results/angle_big_' +
           MODEL.split('/')[-1] + '.csv')
PROMPT = 'The whole group sat down and began to discuss'
SWITCHES = {0: 'city', 16: 'animal', 32: 'food', 48: 'nature'}
FAMILIES = {
    'city':   ['paris', 'london', 'berlin', 'madrid', 'tokyo'],
    'animal': ['cat', 'dog', 'bird', 'bear', 'horse'],
    'food':   ['pizza', 'sushi', 'pasta', 'burger'],
    'nature': ['forest', 'rice', 'water', 'sun', 'tree'],
}
SWEEP = [5, 8, 10, 12, 15, 20, 25, 30]


def rep4(toks):
    if len(toks) < 4:
        return 0.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return sum(1 for i in range(len(toks) - 3) if n4[i] in n4[i + 1:]) \
        / (len(toks) - 3)


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
    raise RuntimeError(f'{MODEL}: no readout norm '
                       f'(model.model.norm / language_model.norm)')


def main():
    t0 = time.time()
    print(f'\nLoading {MODEL} (fp16, low_cpu_mem_usage, no quant) ...')
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    RO_norm = readout_model(model).norm
    Wn = (_lm_head_fp16(model) / _lm_head_fp16(model).norm(
        dim=1, keepdim=True)).float()

    famids = {}
    names = {}
    for fam, words in FAMILIES.items():
        ids = []
        for w in words:
            ids1 = tok(' ' + w, add_special_tokens=False).input_ids
            if len(ids1) == 1:
                ids.append(int(ids1[0]))
                names[int(ids1[0])] = w
        assert ids, f'family {fam} all multi-token'
        famids[fam] = ids
        print(f"  family {fam:>6}: {[names[i] for i in ids]}")

    def closest_to_fam(vv, fam):
        u = vv / vv.norm()
        s = Wn[famids[fam]].float().to(DEV) @ u
        return famids[fam][int(s.argmax())]

    def rot_to_angle(vv, tid, theta):
        a = math.radians(theta)
        v1 = vv / vv.norm()
        Wb = Wn[tid].float().to(DEV)
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(a) + g * math.sin(a)) * vv.norm()

    def forward(ids, inj_p=None, anti_t=None):
        hs = []
        try:
            if inj_p is not None:
                def inj(m, i, o, p=inj_p):
                    o[0, -1, :] = torch.as_tensor(p, dtype=o.dtype,
                                                  device=o.device)
                hs.append(RO_norm.register_forward_hook(inj))
            if anti_t is not None:
                def anti(m, i, o, tid=anti_t):
                    o[0, -1, tid] = -30.0
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
                model(ids)
        finally:
            hk.remove()
        return vc['v']

    def sample(L, prefix):
        L = torch.nan_to_num(L.float(), nan=-50.0).clamp(-50.0, 50.0)
        p = torch.softmax(L, 0)
        q = p.clone(); order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q); msk[order[:k]] = 1
        qq = (q * msk)
        for t in set(prefix):
            c = prefix.count(t)
            if c:
                qq[t] = qq[t] * (PEN ** c)
        tot = qq.sum()
        if tot <= 0 or not torch.isfinite(tot):
            qq = torch.ones_like(qq)
        qq = qq / qq.sum()
        return int(torch.multinomial(qq, 1))

    def run_schedule(sd, theta_map, collect=False):
        """theta_map: step -> angle; collect: return (ids_at_switch_steps)."""
        torch.manual_seed(sd)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        sampled = []
        hits = {}
        ctx = {}
        last_switch = -10
        last_tgt = None
        for step in range(NTOK):
            inj_p = None
            anti_t = None
            if step in SWITCHES:
                fam = SWITCHES[step]
                v = capture_v(ids)
                tgt = closest_to_fam(v, fam)
                th = theta_map.get(step, theta_map.get('default', 10.0))
                inj_p = rot_to_angle(v, tgt, th)
                last_switch = step
                last_tgt = tgt
            elif 1 <= step - last_switch <= 2 and last_tgt is not None:
                anti_t = last_tgt
            if collect and step in SWITCHES:
                ctx[step] = ids.clone()
            L = forward(ids, inj_p=inj_p, anti_t=anti_t)
            nxt = sample(L, sampled)
            if step in SWITCHES:
                fam = SWITCHES[step]
                hits[step] = (fam, names[last_tgt], nxt in famids[fam])
            sampled.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        if collect:
            return sampled, hits, ctx
        return sampled, hits

    print(f"\n[{MODEL}] ANGLE-BIG: sweep rank vs theta at switch contexts")
    steps = sorted(SWITCHES)
    thetas = {'default': 10.0}
    calib_rows = []
    # reference run to get the real contexts
    _, _, ctx = run_schedule(SEEDS[0], thetas, collect=True)
    for step in steps:
        fam = SWITCHES[step]
        ids = ctx[step]
        v = capture_v(ids)
        tgt = closest_to_fam(v, fam)
        row = []
        for th in SWEEP:
            vp = rot_to_angle(v, tgt, th)
            L = forward(ids, inj_p=vp)
            rk = int((L > L[tgt]).sum()) + 1
            top1 = tok.decode([int(L.argmax())]).strip()
            row.append((th, rk, top1))
        best = min((th for th, rk, _ in row if rk == 1), default=None)
        thetas[step] = best if best is not None else SWEEP[-1]
        print(f"  step{step:>2} {fam:>6}: rank/theta  " +
              '  '.join(f'{th}->{rk}({t1[:8]})' for th, rk, t1 in row))
        print(f"      theta*={thetas[step]:.0f}")
        calib_rows.append(dict(step=step, fam=fam, theta_star=thetas[step],
                               sweep=[(th, rk) for th, rk, _ in row]))

    print(f"\n  per-step angles: "
          + '  '.join(f'{SWITCHES[s]}@{s}={thetas[s]:.0f}' for s in steps))

    # re-run with per-step angles, both seeds
    rows = []
    for sd in SEEDS:
        toks, hits = run_schedule(sd, thetas)
        txt = tok.decode(toks)
        print(f"\n  seed {sd}:")
        for i, st in enumerate(steps):
            fam, word, hit = hits[st]
            seg = tok.decode(toks[i * 16:(i + 1) * 16])
            print(f"    switch@{st:>2} {fam:>6} -> {word:<6} "
                  f"{'HIT' if hit else 'miss'}  | {seg.strip()[:72]}")
        rows.append(dict(seed=sd, full=txt,
                         **{f'A{i}_hit': hits[s][2] for i, s in enumerate(steps)}))
        print(f"    FULL: {PROMPT} {txt[:150]}")
    for i, st in enumerate(steps):
        ok = sum(1 for r in rows if r[f'A{i}_hit'])
        print(f"\n  switch@{st} ({SWITCHES[st]}): hit={ok}/{len(SEEDS)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()