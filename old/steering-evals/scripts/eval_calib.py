#!/usr/bin/env python3
"""eval_calib.py — per-model angle recalibration for the switch controller.
For a given model+prompt, at the REAL switch contexts, sweep steer angle
and measure the rank of the ACTUAL graft-target token (the closest family
member's first token, incl. multi-word phrase directions). Outputs the
per-step theta* (min angle that makes the target rank 1).

The v2 runs used the Gemma-1B-tuned 10deg on every model — but Qwen
models and multi-word phrase grafts have different rank-1 reach, so the
graft fired yet the topic token never landed ('Lon' won over 'new').
This measures the truth per model.

Run: HF_TOKEN=<tok> python3 -u eval_calib.py Qwen/Qwen2-1.5B \
        "The whole group sat down and began to discuss"
"""
import math
import sys
import time

import torch
import transformers

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2-1.5B'
PROMPT = (sys.argv[2] if len(sys.argv) > 2
          else 'The whole group sat down and began to discuss')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SWEEP = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
SWITCHES = {0: 'city', 16: 'animal', 32: 'food', 48: 'nature'}
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


def main():
    t0 = time.time()
    print(f'\nLoading {MODEL} (bf16, no quant) ...')
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    RO = model.model
    RO_norm = RO.norm
    W = _lm_head_fp16(model)
    Wn = (W / W.norm(dim=1, keepdim=True)).float()

    members = {}
    famids = {}
    for fam, words in FAMILIES.items():
        mem = []
        for w in words:
            ids = [int(i) for i in
                   tok(' ' + w, add_special_tokens=False).input_ids]
            d = Wn[ids].float().sum(0)
            d = d / d.norm()
            mem.append((w, ids, d))
        members[fam] = mem
        famids[fam] = [i for _, ids, _ in mem for i in ids]

    def closest_member(vv, fam):
        u = vv / vv.norm()
        best = None
        for w, ids, d in members[fam]:
            s = float(d.to(DEV) @ u)
            if best is None or s > best[0]:
                best = (s, w, ids)
        return best[1], best[2]        # name, ids

    def rot_to_angle(vv, tid, theta):
        a = math.radians(theta)
        v1 = vv / vv.norm()
        Wb = Wn[tid].float().to(DEV)
        tau = Wb - (v1 @ Wb) * v1
        g = tau / tau.norm()
        return (v1 * math.cos(a) + g * math.sin(a)) * vv.norm()

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

    # reference trajectory (plain free run, no hooks) to get real contexts
    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    ctx = {}
    n = ids.shape[1]
    for step in sorted(SWITCHES):
        while ids.shape[1] < n + step:
            with torch.no_grad():
                nxt = int(model(ids).logits[0, -1].argmax())
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        ctx[step] = ids.clone()

    print(f'\n[{MODEL}] CALIB: rank-1 reach vs angle at switch contexts')
    print(f'  prompt: {PROMPT!r}')
    results = {}
    for step in sorted(SWITCHES):
        fam = SWITCHES[step]
        ids_c = ctx[step]
        v = capture_v(ids_c)
        w_name, w_ids = closest_member(v, fam)
        tgt = w_ids[0]
        print(f'\n  step{step:>2} {fam:>6}: closest member {w_name!r} '
              f'(ids={w_ids}) target={tok.decode([tgt])!r}')
        row = []
        best = None
        for th in SWEEP:
            vp = rot_to_angle(v, tgt, th)
            hs = [RO_norm.register_forward_hook(
                lambda m, i, o, p=vp: o.__setitem__(
                    (0, -1, slice(None)),
                    torch.as_tensor(p, dtype=o.dtype, device=o.device)))]
            try:
                with torch.no_grad():
                    L = model(ids_c).logits[0, -1].float()
            finally:
                for h in hs:
                    h.remove()
            rk = int((L > L[tgt]).sum()) + 1
            top1 = tok.decode([int(L.argmax())]).strip()
            row.append((th, rk, top1))
            if rk == 1 and best is None:
                best = th
        print(f"    " + '  '.join(f'{th}°->rk{rk}({t1[:6]})'
                                  for th, rk, t1 in row))
        theta_star = best if best is not None else '>30'

        results[step] = (w_name, theta_star)

        safe = (best + 2 if best is not None else None)

        print(f"    theta* = {theta_star}")

        if safe is not None:

            print(f"    safe (theta*+2) = {safe}")

    print('\n  calibration summary (step: member, theta*):')
    for s in sorted(results):
        print(f"    @{s:>2} {SWITCHES[s]:>6} -> {results[s][0]!r} "
              f"theta*={results[s][1]}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()