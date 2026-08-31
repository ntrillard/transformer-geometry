#!/usr/bin/env python3
"""gen_geom.py - PURE geometric steering (rotation-only, NO token planting).

Contrast with gen_blendtraj.py: that script edits the INPUT by planting each
target word as a real token in context, then blends two readout series to
weave the word back into the narrative. This script does none of that.

MODES
-----
  MODE=hold (default)  nudge the readout toward the target row for the whole
                       WINDOW. If the angle sits at the rank-1 threshold the
                       model emits the forced token every step -> a bounded
                       but spammy loop (windowing keeps it finite; the model
                       recovers between windows).
  MODE=emit            nudge until the word is emitted ONCE, then drop the
                       forcing immediately with NO suppression - no logit
                       edit of any kind. The word is now genuinely in the
                       context the model wrote, so the narrative can
                       continue around it - the "sheep -> sheepishly"
                       integration without ever planting.

HIT_STOP is implicit in MODE=emit: once a window's word has appeared, that
window stops steering (honest miss if the window ends without the word).

Env:  MODE=<hold|emit> (default emit)
      G_ANGLE=<deg>    rotation toward target per step (default 7)
      G_LAN=<0..1>     share of steered logits in the readout blend (0.6)
      WINDOW=<steps>   window length per word (default 12)
      SW0=<steps>      first window start (default 20)
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
MODE = os.environ.get('MODE', 'emit')
G_ANGLE = float(os.environ.get('G_ANGLE', '7'))
G_LAN = float(os.environ.get('G_LAN', '0.6'))
WINDOW = int(os.environ.get('WINDOW', '12'))
SW0 = int(os.environ.get('SW0', '20'))
BLEND_STEPS = max(1, int(os.environ.get('BLEND_STEPS', '1')))
TRACE = os.environ.get('TRACE') == '1'


def main():
    t0 = time.time()
    print(f'\nPure-geometry[{MODE}] | {MODEL} | prompt={PROMPT!r} '
          f'| words={WORDS} | G_ANGLE={G_ANGLE} G_LAN={G_LAN} '
          f'WINDOW={WINDOW} ntok={NTOK}')
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

    def sample(L, block_a=None):
        L = torch.nan_to_num(L.float(), nan=-50.0).clamp(-50.0, 50.0)
        p = torch.softmax(L, 0)
        q = p.clone()
        order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= NUCLEUS).sum()) + 1
        msk = torch.zeros_like(q)
        msk[order[:k]] = 1
        qq = q * msk
        return int(torch.multinomial(qq, 1))

    # ---- generation ----
    torch.manual_seed(SEED)
    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    sampled = []
    emitted = {}                 # word -> step it first appeared this generation
    for step in range(NTOK):
        w_active = next((w for s, w in win_at.items()
                         if s <= step < s + WINDOW), None)

        if (w_active is not None and MODE == 'hold') or \
           (w_active is not None and MODE == 'emit'
                and w_active not in emitted):
            # --- steer this step toward w_active ---
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
                print(f'      [{step}] steer {w_active} lvl {level + 1}/'
                      f'{BLEND_STEPS} lam={lam_k:.2f} th={th_k:.1f}')
            nxt = sample(L)
            if nxt == word_ids[w_active]:
                emitted[w_active] = step
                if TRACE:
                    print(f'      [{step}] WORD {w_active} EMITTED')
        else:
            # --- free run (window over, or word already landed) ---
            L, _ = forward_v(ids)
            nxt = sample(L)

        if nxt == eos_id:
            sampled.append(nxt)
            break
        sampled.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)

    txt = tok.decode(sampled)
    print(f'\n===== PURE-GEOMETRY[{MODE}] angle={G_ANGLE:.0f} lam={G_LAN} '
          f'({', '
.join(WORDS)}) =====')
    print(f'{PROMPT} {txt}')
    hits = {w: (w in txt) for w in WORDS}
    counts = {w: txt.count(w) for w in WORDS}
    print(f'\nwords present: {hits}')
    print(f'word counts  : {counts}')
    print(f'emitted at step: {emitted}')
    print(f'[{time.time() - t0:.0f}s] {len(sampled)} tokens, '
          f'eos={"YES" if sampled and sampled[-1] == eos_id else "NO"}')


if __name__ == "__main__":
    main()
