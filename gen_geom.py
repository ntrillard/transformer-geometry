#!/usr/bin/env python3
"""gen_geom.py - PURE geometric steering (rotation-only, NO token planting).

Contrast with gen_blendtraj.py: that script edits the INPUT by planting each
target word as a real token in context, then blends two readout series to
weave the word back into the narrative. This script does none of that.

At every step inside a per-word window it simply rotates the residual hidden
state at the readout toward the target word's embedding row by a fixed angle:

    v' = rotate(v, Wn[target], G_ANGLE)
    L  = (1 - G_LAN) * L_nat + G_LAN * L_steer      # L_steer from v'

Nothing is inserted into context, no anti-repeat, no settle logic. The word
may therefore honestly MISS: if the natural trajectory points far away, a
fixed rotation never reaches the word's rank-1 region.

This is the capability test: does pure geometric bias alone put an
out-of-place word into the narrative, and how much of blendtraj's success is
really the planted token vs the geometry? Compare words present here vs
gen_blendtraj.py on the same prompts/seeds.

Env:  G_ANGLE=<deg>    rotation toward target per step (default 6)
      G_LAN=<0..1>     share of the steered logits in the readout blend
                       (default 0.5)
      WINDOW=<steps>   window length per word (default 12)
      SW0=<steps>      first window start (default 20)
      SEED=<int>       (default 0)
      BLEND_STEPS=<n>  ramp levels inside the window (default 1)

Run:  HF_TOKEN=<tok> python3 gen_geom.py [model] [prompt] [w1,w2,..]
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
G_ANGLE = float(os.environ.get('G_ANGLE', '6'))
G_LAN = float(os.environ.get('G_LAN', '0.5'))
WINDOW = int(os.environ.get('WINDOW', '12'))
SW0 = int(os.environ.get('SW0', '20'))
BLEND_STEPS = max(1, int(os.environ.get('BLEND_STEPS', '1')))
TRACE = os.environ.get('TRACE') == '1'


def main():
    t0 = time.time()
    print(f'\nPure-geometry | {MODEL} | prompt={PROMPT!r} | words={WORDS} '
          f'| G_ANGLE={G_ANGLE} G_LAN={G_LAN} WINDOW={WINDOW} ntok={NTOK}')
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
        # used ONLY for the embedding row to rotate toward - never planted
        sp = tok(' ' + w, add_special_tokens=False).input_ids
        bare = tok(w, add_special_tokens=False).input_ids
        ids = sp if len(sp) == 1 else (bare if len(bare) == 1 else sp)
        word_ids[w] = int(ids[0])
        print(f'  {w:12} -> token {ids} {[tok.decode([i]) for i in ids]}')

    n_sw = len(WORDS)
    win_at = {SW0 + i * (NTOK // (n_sw + 1)): w for i, w in enumerate(WORDS)}

    def forward(ids, inj_p=None):
        hs = []
        try:
            if inj_p is not None:
                def inj(m, i, o, p=inj_p):
                    o[0, -1, :] = torch.as_tensor(p, dtype=o.dtype,
                                                  device=o.device)
                hs.append(norm.register_forward_hook(inj))
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

    def sample(L):
        L = torch.nan_to_num(L.float(), nan=-50.0).clamp(-50.0, 50.0)
        p = torch.softmax(L, 0)
        q = p.clone()
        order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= NUCLEUS).sum()) + 1
        msk = torch.zeros_like(q)
        msk[order[:k]] = 1
        qq = q * msk
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

    for step in range(NTOK):
        # any window covering this step
        w_active = next((w for s, w in win_at.items()
                         if s <= step < s + WINDOW), None)
        w_active = next((w for s, w in win_at.items()
                         if s <= step < s + WINDOW), None)
        if w_active is not None:
            sw = next(s for s, w in win_at.items() if w == w_active)
            done = step - sw
            level = min(BLEND_STEPS - 1,
                        int(done * BLEND_STEPS / WINDOW)) if BLEND_STEPS > 1 \
                else 0
            lam_k = G_LAN * (level + 1) / BLEND_STEPS
            th_k = G_ANGLE * (level + 1) / BLEND_STEPS
            L_nat, v = forward_v(ids)
            L_steer = forward(
                ids, inj_p=rot_to_angle(v, word_ids[w_active], th_k))
            L = (1 - lam_k) * L_nat + lam_k * L_steer
            if TRACE:
                print(f'      window[{step}] {w_active} level {level + 1}/'
                      f'{BLEND_STEPS} lam={lam_k:.2f} th={th_k:.1f}')
            nxt = sample(L)
        else:
            L, _ = forward_v(ids)
            nxt = sample(L)

        if nxt == eos_id:
            sampled.append(nxt)
            break
        sampled.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)

    txt = tok.decode(sampled)
    print(f'\n===== PURE-GEOMETRY angle={G_ANGLE:.0f} lam={G_LAN} '
          f'({", ".join(WORDS)}) =====')
    print(f'{PROMPT} {txt}')
    hits = {w: (w in txt) for w in WORDS}
    print(f'\nwords present: {hits}')
    print(f'[{time.time() - t0:.0f}s] {len(sampled)} tokens, '
          f'eos={"YES" if sampled and sampled[-1] == eos_id else "NO"}')


if __name__ == "__main__":
    main()
