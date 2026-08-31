#!/usr/bin/env python3
"""gen_blend3.py - THREE-SERIES + STATE-MEMORY blend at the readout.

Extends gen_blendtraj.py's plant+settle idea:

  * planted words are REAL tokens in context (space-prefixed single token).
  * settling window: at each step run THREE series and blend them (simplex):
      L_nat  : natural forward                        (no injection)
      L_mild : forward with a mild rotation           (MILD_ANGLE)
      L_full : forward with the calibrated rank-1 rotation (best_angle)
  * STATE MEMORY (states around the insert):
      - L_pre : captured ONCE, right before the word is appended (the model's
        prediction on the pre-insert context), blended with decaying weight
        so the story doesn't fully snap to the word.
      - L_mem : rolling window of the MEM_N most recent settled readouts,
        blended with a small weight ("a few states after").
  * weights ramp across the window; natural = remainder.

Env: MILD_ANGLE=3  W_FULL_MAX=0.4  W_MILD_MAX=0.2  W_PRE=0.15  W_MEM=0.1
     MEM_N=3  SETTLE=8  PLANT0=20  BLEND_STEPS=1  TRACE=1  SEED=0

Run: HF_TOKEN=<tok> python3 gen_blend3.py [model] [prompt] [w1,w2,..]
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
ANTI = 5
SEED = int(os.environ.get('SEED', '0'))
SETTLE = int(os.environ.get('SETTLE', '8'))
PLANT0 = int(os.environ.get('PLANT0', '20'))
MILD_ANGLE = float(os.environ.get('MILD_ANGLE', '3'))
W_FULL_MAX = float(os.environ.get('W_FULL_MAX', '0.4'))
W_MILD_MAX = float(os.environ.get('W_MILD_MAX', '0.2'))
W_PRE = float(os.environ.get('W_PRE', '0.15'))
W_MEM = float(os.environ.get('W_MEM', '0.1'))
MEM_N = int(os.environ.get('MEM_N', '3'))
TRACE = os.environ.get('TRACE') == '1'


def main():
    t0 = time.time()
    print(f'\nBlend3 | {MODEL} | prompt={PROMPT!r} | words={WORDS} | '
          f'SETTLE={SETTLE} MILD={MILD_ANGLE} W_FULL_MAX={W_FULL_MAX} '
          f'W_PRE={W_PRE} W_MEM={W_MEM} MEM_N={MEM_N} ntok={NTOK}')
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

    def best_angle(ids, vv, tid):
        """min angle making tid rank-1 at this context, +margin."""
        for th in [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]:
            L = forward(ids, inj_p=rot_to_angle(vv, tid, th))
            if (int(L.argmax()) == tid
                    and torch.isfinite(L[tid])
                    and float(L[tid]) > float(L.max()) - 0.001):
                return th + 2.0
        return 30.0

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
    settle_until = -1
    settle_word = None
    last_plant_tid = None
    pre_logits = None              # captured before each insert
    mem = collections.deque(maxlen=MEM_N)   # recent settle readouts

    for step in range(NTOK):
        in_settle = step < settle_until and settle_word is not None
        anti_a = last_plant_tid if (step <= settle_until
                                    and last_plant_tid is not None) else None
        bw = ({settle_word} if anti_a is not None else None)

        if step in switch_at:
            # capture the PRE-insert prediction
            L_pre, _ = forward_v(ids)
            pre_logits = L_pre
            wid = plant_tid_at[step]
            nxt = wid
            settle_until = step + 1 + SETTLE
            last_plant_tid = wid
            settle_word = switch_at[step]
            if TRACE:
                print(f'      plant@{step} -> {tok.decode([wid])!r} '
                      f'settle until {settle_until}')
        elif in_settle:
            frac = min(1.0, (settle_until - step - 1) / max(1, SETTLE - 1))
            w_full = W_FULL_MAX * frac
            w_mild = W_MILD_MAX * frac
            w_mem = W_MEM if mem else 0.0
            w_pre = W_PRE * (1 - frac)          # decay away from pre-insert
            w_nat = max(0.0, 1 - w_full - w_mild - w_mem - w_pre)

            L_nat, v = forward_v(ids)
            L_mild = forward(ids, inj_p=rot_to_angle(
                v, word_ids[settle_word], MILD_ANGLE))
            L_full = forward(ids, inj_p=rot_to_angle(
                v, word_ids[settle_word],
                best_angle(ids, v, word_ids[settle_word])))
            L = (w_nat * L_nat + w_mild * L_mild + w_full * L_full)
            if pre_logits is not None:
                L = L + w_pre * pre_logits
            for lm in mem:
                L = L + w_mem / max(1, len(mem)) * lm
            if TRACE:
                print(f'      settle[{step}] nat={w_nat:.2f} mild={w_mild:.2f} '
                      f'full={w_full:.2f} pre={w_pre:.2f} mem={w_mem:.2f}')
            nxt = sample(L, block_words=bw)
            mem.append(L_nat)
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
    print(f'\n===== BLEND3 ({" ".join(WORDS)}) =====')
    print(f'{PROMPT} {txt}')
    hits = {w: (w in txt) for w in WORDS}
    print(f'\nwords present: {hits}')
    print(f'[{time.time() - t0:.0f}s] {len(sampled)} tokens, '
          f'eos={"YES" if sampled and sampled[-1] == eos_id else "NO"}')


if __name__ == "__main__":
    main()