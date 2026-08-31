#!/usr/bin/env python3
"""gen_blendtraj.py - two-series blend with planted words + settling window.

The problem it solves: force-inserting a word at the readout makes the model
'snap back' on the next step because the word was never part of its trajectory.
Fix: plant each target word as a REAL TOKEN in the shared context, then for a
SETTLE-step window after each plant run TWO series and blend them at the
readout:

   natural : plain forward (no injection)
   steered : forward with a small rotation (HOLD_ANGLE) toward the planted
             word's row - keeps the story weaving the word

   L = (1-lam)*L_nat + lam*L_steer   -> sample once -> append to context

After the window, the word(s) are genuinely in context, so the model continues
naturally with the context already bent around the inserts.

Env:  LAM=<0..1>          (default 0.6 - blend of the two series)
      SETTLE=<steps>      (default 8 - settling window after each plant)
      HOLD_ANGLE=<deg>    (default 8 - steer rotation during settling)
      TRACE=1

Run:  HF_TOKEN=<tok> python3 gen_blendtraj.py [model] [prompt] [w1,w2,..]
"""
import math
import os
import sys
import time

import torch
import transformers

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2-1.5B'
PROMPT = (sys.argv[2] if len(sys.argv) > 2
          else 'It was a warm morning in a small kitchen')
WORDS = ([w.strip() for w in sys.argv[3].split(',') if w.strip()]
         if len(sys.argv) > 3 else
         ['diamond', 'camel', 'volcano'])
NTOK = 120
SEED = int(os.environ.get('SEED', '0'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

NUCLEUS = 0.9
ANTI = 5                        # de-repeat steps after a plant
LAM = float(os.environ.get('LAM', '0.6'))
SETTLE = int(os.environ.get('SETTLE', '8'))
HOLD_ANGLE = float(os.environ.get('HOLD_ANGLE', '8'))
PLANT0 = int(os.environ.get('PLANT0', '20'))
TRACE = os.environ.get('TRACE') == '1'


def main():
    t0 = time.time()
    print(f'\nBlend-trajectory | {MODEL} | prompt={PROMPT!r} | words={WORDS} '
          f'| LAM={LAM} SETTLE={SETTLE} HOLD_ANGLE={HOLD_ANGLE} ntok={NTOK}')
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    eos_id = int(tok.eos_token_id)
    norm = model.model.norm if hasattr(model.model, 'norm') \
        else model.model.language_model.norm
    W = model.lm_head.weight.detach().float()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()

    word_ids = {}
    for w in WORDS:
        # prefer the SPACE-PREFIXED single token when planting mid-text
        # (a bare token fuses into the prior word: 'anddiamond')
        sp = tok(' ' + w, add_special_tokens=False).input_ids
        bare = tok(w, add_special_tokens=False).input_ids
        ids = sp if len(sp) == 1 else (bare if len(bare) == 1 else sp)
        word_ids[w] = int(ids[0])
        print(f'  {w:12} -> token {ids} {[tok.decode([i]) for i in ids]}')

    n_sw = len(WORDS)
    switch_at = {PLANT0 + i * (NTOK // (n_sw + 1)): w
                 for i, w in enumerate(WORDS)}
    plant_tid_at = {s: word_ids[w] for s, w in switch_at.items()}

    def forward(ids, inj_p=None, anti_ids=None):
        hs = []
        try:
            if inj_p is not None:
                def inj(m, i, o, p=inj_p):
                    o[0, -1, :] = torch.as_tensor(p, dtype=o.dtype,
                                                  device=o.device)
                hs.append(norm.register_forward_hook(inj))
            if anti_ids:
                def anti(m, i, o, aids=anti_ids):
                    o[0, -1, aids] = -30.0
                hs.append(model.lm_head.register_forward_hook(anti))
            with torch.no_grad():
                return model(ids).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()

    def forward_v(ids):
        vc = {}
        hk = norm.register_forward_hook(
            lambda m, i, o: vc.__setitem__('v', o[0, -1, :].float()))
        try:
            with torch.no_grad():
                L = model(ids).logits[0, -1].float()
        finally:
            hk.remove()
        return L, vc['v']

    def rot_to_angle(vv, tid, theta):
        a = math.radians(theta)
        v1 = vv / vv.norm()
        Wb = Wn[tid].to(vv.device)
        g0 = Wb - (v1 @ Wb) * v1
        gn = g0 / (g0.norm() + 1e-12)
        return (v1 * math.cos(a) + gn * math.sin(a)) * vv.norm()

    def sample(L, block_words=None):
        L = torch.nan_to_num(L.float(), nan=-50.0).clamp(-50.0, 50.0)
        p = torch.softmax(L, 0)
        q = p.clone()
        order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= NUCLEUS).sum()) + 1
        msk = torch.zeros_like(q)
        msk[order[:k]] = 1
        qq = q * msk
        if block_words:
            top = order[:200].tolist()
            dec = tok.batch_decode([[i] for i in top])
            drop = [i for i, s in zip(top, dec)
                    if any(w in s.lower() for w in block_words)]
            for i in drop:
                qq[i] = 0.0
        tot = qq.sum()
        if tot <= 0 or not torch.isfinite(tot):
            qq = torch.ones_like(qq)
        qq = qq / qq.sum()
        return int(torch.multinomial(qq, 1))

    # ---- generation ----
    torch.manual_seed(SEED)
    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    sampled = []
    settle_until = -1               # steps inside the settling window
    last_plant_tid = None
    settle_word = None

    for step in range(NTOK):
        in_settle = step < settle_until and settle_word is not None
        anti_a = last_plant_tid if (step <= settle_until
                                    and last_plant_tid is not None) else None
        bw = ({settle_word} if anti_a is not None else None)

        if step in switch_at:
            # plant the word as a REAL TOKEN in the shared context
            wid = plant_tid_at[step]
            nxt = wid
            settle_until = step + 1 + SETTLE
            last_plant_tid = wid
            settle_word = switch_at[step]
            if TRACE:
                print(f'      plant@{step} -> {tok.decode([wid])!r} '
                      f'settle until {settle_until}')
        elif in_settle:
            # TWO SERIES, blended at the readout
            L_nat, v = forward_v(ids)                 # natural series
            L_steer = forward(                       # steered series (hold)
                ids, inj_p=rot_to_angle(v, word_ids[settle_word],
                                        HOLD_ANGLE))
            L = (1 - LAM) * L_nat + LAM * L_steer
            nxt = sample(L, block_words=bw)
        elif anti_a is not None:
            L = forward(ids, anti_ids=[anti_a])
            nxt = sample(L, block_words=bw)
        else:
            L, _ = forward_v(ids)
            nxt = sample(L)

        if nxt == eos_id:
            sampled.append(nxt)
            break
        sampled.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)

    txt = tok.decode(sampled)
    print(f'\n===== BLEND-TRAJ lam={LAM} settle={SETTLE} '
          f'hold={HOLD_ANGLE:.0f} ({", ".join(WORDS)}) =====')
    print(f'{PROMPT} {txt}')
    hits = {w: (w in txt) for w in WORDS}
    print(f'\nwords present: {hits}')
    print(f'[{time.time() - t0:.0f}s] {len(sampled)} tokens, '
          f'eos={"YES" if sampled and sampled[-1] == eos_id else "NO"}')


if __name__ == "__main__":
    main()