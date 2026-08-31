#!/usr/bin/env python3
"""gen_blend3.py - backward/forward trajectory blend at a branch point.

The mechanism (user spec):
  1. ONE generator drives BOTH branches, so the main and pure branches are
     identical up to the first plant - at the branch point the pure branch
     IS the story's counterfactual future.
  2. At the branch point (plant step P) we capture L_pre - the state where
     the trajectory branched (the "backwards" anchor).
  3. After P the branches separate:
       - main : the planted word is a REAL token; samples from the blend.
       - pure : continues naturally, never sees the plant (the would-have-been
         future - the parallel non-steered trajectory).
  4. Every settle step samples from a temporally-oriented blend:
       L_pre   : branch-point state  - decays from full right after P
       L_back  : pure branch readout - the backward (would-have-been) story,
                 decays from P
       L_fwd   : main + small steer  - the forward (steered) trajectory,
                 humps mid-window then fades
       L_nat   : main natural readout - remainder; takes over after window
  5. After the window: pure natural - the planted context alone carries
     the new trajectory.

Env: W_PRE=0.35  W_BACK_MAX=0.35  W_FWD_MAX=0.25  HOLD_ANGLE=4
     SETTLE=8  PLANT0=20  SEED=0  TRACE=1
Run: HF_TOKEN=<tok> python3 gen_blend3.py [model] [prompt] [w1,w2,..]
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
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

NUCLEUS = 0.9
ANTI = 5                        # de-repeat steps after a plant
SEED = int(os.environ.get('SEED', '0'))
SETTLE = int(os.environ.get('SETTLE', '8'))
PLANT0 = int(os.environ.get('PLANT0', '20'))
HOLD_ANGLE = float(os.environ.get('HOLD_ANGLE', '4'))
W_PRE = float(os.environ.get('W_PRE', '0.15'))
W_BACK_MAX = float(os.environ.get('W_BACK_MAX', '0.15'))
W_FWD_MAX = float(os.environ.get('W_FWD_MAX', '0.25'))
TRACE = os.environ.get('TRACE') == '1'


def main():
    t0 = time.time()
    print(f'\nBlend3 bw/fwd | {MODEL} | prompt={PROMPT!r} | words={WORDS} | '
          f'W_PRE={W_PRE} W_BACK_MAX={W_BACK_MAX} W_FWD_MAX={W_FWD_MAX} '
          f'HOLD={HOLD_ANGLE} ntok={NTOK}')
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

    # target word -> space-prefixed single token when possible
    word_ids = {}
    for w in WORDS:
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

    def sample(L, gen, block_words=None):
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
        return int(torch.multinomial(qq, 1, generator=gen))

    # ---- generation: two branches sharing ONE generator ----
    gen = torch.Generator(DEV).manual_seed(SEED)

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    ids_p = ids.clone()                     # pure branch (counterfactual)
    sampled = []
    sampled_p = []
    settle_until = -1
    last_plant_tid = None
    settle_word = None
    pre_logits = None
    pure_done = False

    for step in range(NTOK):
        in_settle = step < settle_until and settle_word is not None
        anti_a = last_plant_tid if (step <= settle_until
                                    and last_plant_tid is not None) else None
        bw = ({settle_word} if anti_a is not None else None)

        if step in switch_at:
            # capture the branch-point state (where the trajectory branched)
            L_pre, _ = forward_v(ids)
            pre_logits = L_pre
            wid = plant_tid_at[step]
            nxt = wid
            settle_until = step + 1 + SETTLE
            last_plant_tid = wid
            settle_word = switch_at[step]
            if TRACE:
                print(f'      branch@{step} -> {tok.decode([wid])!r} '
                      f'until {settle_until}')
        elif in_settle:
            t0_ = min(1.0, (settle_until - step - 1) / max(1, SETTLE - 1))
            t = 1.0 - t0_            # 0 at branch, 1 at window end
            w_pre = W_PRE * (1 - t)                # backward: decays from P
            w_back = W_BACK_MAX * (1 - t)          # would-have-been, decays
            w_fwd = W_FWD_MAX * math.sin(math.pi * t)  # forward steer, hump
            w_nat = max(0.0, 1 - w_pre - w_back - w_fwd)

            L_nat, v = forward_v(ids)
            L_fwd = forward(ids, inj_p=rot_to_angle(
                v, word_ids[settle_word], HOLD_ANGLE))
            L_back = forward_v(ids_p)[0] if not pure_done else L_nat
            L = w_nat * L_nat + w_fwd * L_fwd + w_back * L_back
            if pre_logits is not None:
                L = L + w_pre * pre_logits
            if TRACE:
                print(f'      settle[{step}] nat={w_nat:.2f} fwd={w_fwd:.2f} '
                      f'back={w_back:.2f} pre={w_pre:.2f}')
            nxt = sample(L, gen, block_words=bw)
        elif anti_a is not None:
            L = forward(ids, anti_ids=[anti_a])
            nxt = sample(L, gen, block_words=bw)
        else:
            L, _ = forward_v(ids)
            nxt = sample(L, gen)

        # append to main branch
        if nxt == eos_id:
            sampled.append(nxt)
            break
        sampled.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)

        # pure branch: counterfactual continuation (never sees plants)
        if not pure_done:
            L_p, _ = forward_v(ids_p)
            nxt_p = sample(L_p, gen)
            if nxt_p == eos_id:
                pure_done = True
            else:
                sampled_p.append(nxt_p)
                ids_p = torch.cat(
                    [ids_p, torch.tensor([[nxt_p]], device=DEV)], dim=1)

    txt = tok.decode(sampled)
    txt_p = tok.decode(sampled_p)
    print(f'\n===== BLEND3 bw/fwd ({" ".join(WORDS)}) =====')
    print(f'{PROMPT} {txt}')
    print(f'\n== PURE BRANCH (simultaneous counterfactual) ==')
    print(f'{PROMPT} {txt_p}')
    hits = {w: (w in txt) for w in WORDS}
    print(f'\nwords present: {hits}')
    print(f'[{time.time() - t0:.0f}s] {len(sampled)} tokens, '
          f'eos={"YES" if sampled and sampled[-1] == eos_id else "NO"}')


if __name__ == "__main__":
    main()