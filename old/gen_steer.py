#!/usr/bin/env python3
"""gen_steer.py - single-word steering, minimal. Built directly from
gen_pure.py (pure unsteered generation) + ONLY the core steering
techniques needed for one-word control:

  1. CALIBRATED GRAFT (STEER_MODE=graft, default): at a switch step,
     find the smallest readout rotation that makes the target word
     rank-1 at the live context (+margin), then rotate the hidden
     state by that arc so the word is sampled.
  2. HERDING (STEER_MODE=herd): instead of one hard graft, ramp a
     calibrated rotation toward each target word from 0 up to its
     rank-1 angle over HERD_WINDOW steps (scaled by HERD_GAIN), so
     the model picks the word up when the sentence is ready, not
     forced.
  3. DE-REPEAT WINDOW (both modes): for a few steps after the word
     is planted/lands, anti-block the planted token (+ substring
     forms) so the model WRITES about the word instead of parroting
     it.
  Everything else is the pure sampler from gen_pure.py: no
  families, no centroids, no meta-escape-zeroing, no repetition
  penalty.

Env: STEER_MODE=graft|herd  HERD_GAIN=<mult>  HERD_WINDOW=<steps>
     TRACE=1
     TRACE=1

Run: HF_TOKEN=<tok> python3 gen_steer.py [model] [prompt] [word1,word2,..]
Switch positions are spread evenly (every NTOK//(n+1) tokens).
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
SEED = 0
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

SWEEP = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
MARGIN = 2.0
ANTI = 4                       # de-repeat window length after a plant
NUCLEUS = 0.9                  # top-p keep
STEER_MODE = os.environ.get('STEER_MODE', 'graft')   # graft | herd
HERD_GAIN = float(os.environ.get('HERD_GAIN', '1.0'))
HERD_WINDOW = int(os.environ.get('HERD_WINDOW', '10'))
TRACE = os.environ.get('TRACE') == '1'


def main():
    t0 = time.time()
    print(f'\nSteered generation | {MODEL} | prompt={PROMPT!r} | '
          f'words={WORDS} | ntok={NTOK}')
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    eos_id = int(tok.eos_token_id)
    norm = model.model.norm if hasattr(model.model, 'norm') \
        else model.model.language_model.norm
    W = model.lm_head.weight.detach().cpu().float()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()

    # target word -> single clean token id (bare, else leading-space)
    word_ids = {}
    for w in WORDS:
        bare = tok(w, add_special_tokens=False).input_ids
        ids = bare if len(bare) == 1 else \
            tok(' ' + w, add_special_tokens=False).input_ids
        word_ids[w] = int(ids[0])
        print(f'  {w:12} -> token {ids} {[tok.decode([i]) for i in ids]}')

    # even spread of switch steps
    n_sw = len(WORDS)
    switch_at = {i * (NTOK // (n_sw + 1)): w for i, w in enumerate(WORDS)}
    steps = sorted(switch_at)

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
        for th in SWEEP:
            L = forward(ids, inj_p=rot_to_angle(vv, tid, th))
            if (int(L.argmax()) == tid
                    and torch.isfinite(L[tid])
                    and float(L[tid]) > float(L.max()) - 0.001):
                return th + MARGIN
        return SWEEP[-1]

    def sample(L, prefix, block_words=None):
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
    plant_until = -1
    plant_word = None
    plant_tid = None
    landed = set()                    # herd mode: words already written
    herd_cal = {}                     # herd mode: calibrated rank-1 angles

    for step in range(NTOK):
        L, v = forward_v(ids)
        anti_a = plant_tid if (step < plant_until and plant_tid is not None) \
            else None
        bw = ({plant_word} if anti_a is not None else None)

        if STEER_MODE == 'herd':
            # ramp a calibrated rotation 0..rank-1-angle over the window
            for s, w in switch_at.items():
                if step == s:
                    herd_cal[w] = best_angle(ids, v, word_ids[w])
                    if TRACE:
                        print(f'      herd@{s} -> {w}:{tok.decode([word_ids[w]])!r} '
                              f'cal={herd_cal[w]:.0f}deg')
                if s <= step < s + HERD_WINDOW and w not in landed:
                    frac = min(1.0, (step - s + 1) / HERD_WINDOW)
                    th = herd_cal[w] * frac * HERD_GAIN
                    L = forward(ids, inj_p=rot_to_angle(v, word_ids[w], th))
        elif step in switch_at:
            w = switch_at[step]
            th = best_angle(ids, v, word_ids[w])
            L = forward(ids, inj_p=rot_to_angle(v, word_ids[w], th))
            plant_word = w
            plant_tid = word_ids[w]
            plant_until = step + 1 + ANTI
            if TRACE:
                print(f'      switch@{step} -> {w}:{tok.decode([word_ids[w]])!r} '
                      f'th={th:.0f}')
        elif anti_a is not None:
            L = forward(ids, anti_ids=[anti_a])

        nxt = sample(L, sampled, block_words=bw)
        if nxt == eos_id:
            sampled.append(nxt)
            break
        sampled.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)

        # herd mode: once a herded word is sampled, mark it landed + de-repeat
        if STEER_MODE == 'herd':
            for w, tid in word_ids.items():
                if nxt == tid and w not in landed:
                    landed.add(w)
                    plant_word = w
                    plant_tid = tid
                    plant_until = step + 1 + ANTI
                    break

    txt = tok.decode(sampled)
    print(f'\n===== STEERED {STEER_MODE} ({', '.join(WORDS)}) =====')
    print(f'{PROMPT} {txt}')
    hits = {w: (w in txt) for w in WORDS}
    print(f'\nwords present: {hits}')
    print(f'[{time.time() - t0:.0f}s] {len(sampled)} tokens, '
          f'eos={"YES" if sampled and sampled[-1] == eos_id else "NO"}')


if __name__ == "__main__":
    main()