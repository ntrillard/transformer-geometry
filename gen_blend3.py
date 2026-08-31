#!/usr/bin/env python3
"""gen_blend3.py - LARGER temporal forward/backward branch-point blend.

Extends the branch-point idea but captures MORE context around each plant:

  * SETTLE  - longer settling window (default 14).
  * PRE_CONTEXT - rolling memory of the last N natural readouts BEFORE
    each plant ("a few states before"), blended recency-weighted.
  * MEM_N   - rolling memory of the last N settle readouts AFTER each
    plant ("a few states after"), blended small.
  * The branch itself: ONE generator drives both branches, so the pure
    branch is the story's counterfactual future (identical until plant).

Forward/backward temporal schedule in the settle window:
    pre  (branch + pre-context states) : decays from the branch point
    back (pure-branch readout)         : decays from the branch point
    fwd  (main + small steer)          : humps mid-window then fades
    mem  (post-insert states)          : constant small
    nat  (main natural readout)        : remainder; takes over after

MODE=plain disables ALL blending: the word is still planted as a real
token, but the settle window is pure natural sampling (control).

Env:  MODE=blend|plain  SETTLE=14  PRE_CONTEXT=5  MEM_N=5
      W_PRE=0.15  W_BACK_MAX=0.15  W_FWD_MAX=0.25  W_MEM=0.1
      HOLD_ANGLE=4  PLANT0=20  SEED=0  TRACE=1
Run:  HF_TOKEN=<tok> python3 gen_blend3.py [model] [prompt] [w1,w2,..]
"""
import collections
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
MODE = os.environ.get('MODE', 'blend')          # blend | plain
SETTLE = int(os.environ.get('SETTLE', '14'))
PRE_CONTEXT = int(os.environ.get('PRE_CONTEXT', '5'))
MEM_N = int(os.environ.get('MEM_N', '5'))
PLANT0 = int(os.environ.get('PLANT0', '20'))
HOLD_ANGLE = float(os.environ.get('HOLD_ANGLE', '4'))
W_PRE = float(os.environ.get('W_PRE', '0.15'))
W_BACK_MAX = float(os.environ.get('W_BACK_MAX', '0.15'))
W_FWD_MAX = float(os.environ.get('W_FWD_MAX', '0.25'))
W_MEM = float(os.environ.get('W_MEM', '0.1'))
TRACE = os.environ.get('TRACE') == '1'


def main():
    t0 = time.time()
    print(f'\nBlend3 {MODE} | {MODEL} | prompt={PROMPT!r} | words={WORDS} | '
          f'SETTLE={SETTLE} PRE={PRE_CONTEXT} MEM={MEM_N} '
          f'W_PRE={W_PRE} W_BACK={W_BACK_MAX} W_FWD={W_FWD_MAX} '
          f'W_MEM={W_MEM} HOLD={HOLD_ANGLE} ntok={NTOK}')
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
    pre_mem = collections.deque(maxlen=PRE_CONTEXT)
    mem = collections.deque(maxlen=MEM_N)
    pure_done = False

    for step in range(NTOK):
        in_settle = step < settle_until and settle_word is not None
        anti_a = last_plant_tid if (step <= settle_until
                                    and last_plant_tid is not None) else None
        bw = ({settle_word} if anti_a is not None else None)

        if step in switch_at:
            # capture the branch point + push pre-insert context
            L_pre, _ = forward_v(ids)
            pre_mem.append(L_pre)
            wid = plant_tid_at[step]
            nxt = wid
            settle_until = step + 1 + SETTLE
            last_plant_tid = wid
            settle_word = switch_at[step]
            if TRACE:
                print(f'      branch@{step} -> {tok.decode([wid])!r} '
                      f'until {settle_until} (pre-context={len(pre_mem)})')
        elif in_settle:
            L_nat, v = forward_v(ids)
            if MODE == 'plain':
                # NO BLENDING - plant alone carries the steering
                nxt = sample(L_nat, gen, block_words=bw)
            else:
                t0_ = min(1.0, (settle_until - step - 1) / max(1, SETTLE - 1))
                t = 1.0 - t0_                # 0 at branch, 1 at window end
                w_pre = W_PRE * (1 - t)      # pre-context decays from branch
                w_back = W_BACK_MAX * (1 - t)  # counterfactual decays
                w_fwd = W_FWD_MAX * math.sin(math.pi * t)  # fwd hump+fade
                w_mem = W_MEM if mem else 0.0
                w_nat = max(0.0, 1 - w_pre - w_back - w_fwd - w_mem)

                L_fwd = forward(ids, inj_p=rot_to_angle(
                    v, word_ids[settle_word], HOLD_ANGLE))
                L_back = forward_v(ids_p)[0] if not pure_done else L_nat
                L = w_nat * L_nat + w_fwd * L_fwd + w_back * L_back
                # pre-insert multi-state memory (recency-weighted)
                if pre_mem:
                    pts = list(pre_mem)
                    tot = sum(i + 1 for i in range(len(pts)))
                    for i, x in enumerate(pts):
                        L = L + w_pre * (i + 1) / tot * x
                # post-insert memory (a few states after)
                if mem:
                    L = L + (w_mem / len(mem)) * sum(mem)
                if TRACE:
                    print(f'      settle[{step}] nat={w_nat:.2f} fwd={w_fwd:.2f} '
                          f'back={w_back:.2f} pre={w_pre:.2f} mem={w_mem:.2f}')
                nxt = sample(L, gen, block_words=bw)
            mem.append(L_nat)
        elif anti_a is not None:
            L = forward(ids, anti_ids=[anti_a])
            nxt = sample(L, gen, block_words=bw)
            L_nat, _ = forward_v(ids)
            pre_mem.append(L_nat)
        else:
            L, _ = forward_v(ids)
            nxt = sample(L, gen)
            pre_mem.append(L)

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
    print(f'\n===== BLEND3 {MODE} ({" ".join(WORDS)}) =====')
    print(f'{PROMPT} {txt}')
    print(f'\n== PURE BRANCH (simultaneous counterfactual) ==')
    print(f'{PROMPT} {txt_p}')
    hits = {w: (w in txt) for w in WORDS}
    print(f'\nwords present: {hits}')
    print(f'[{time.time() - t0:.0f}s] {len(sampled)} tokens, '
          f'eos={"YES" if sampled and sampled[-1] == eos_id else "NO"}')


if __name__ == "__main__":
    main()