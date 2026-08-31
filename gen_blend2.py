#!/usr/bin/env python3
"""gen_blend2.py - blend of TWO FULL STATES at the LM head.

At each switch step we run TWO COMPLETE forward passes over the SAME
context ids:

  (A) NATURAL branch : no injection anywhere.
                        -> full final state v_nat   (at final RMSNorm)
  (B) STEERED branch : steering injected INTO layer INJ_LAYER - the
      last-token hidden state at that layer is rotated toward the
      target token's row by THETA_INJ degrees - then the REMAINING
      layers process it all the way to the final norm.
                        -> full final state v_steer

The LM head reads the HIDDEN-STATE BLEND of the two FULL states

      h = normalize((1-lam)*v_nat + lam*v_steer) * ||v_nat||

and we sample from head(h). lam = BLEND_PCT (0..1): lam=0 -> pure
natural, lam=1 -> pure steered (in-layer). So the steered side is a
genuinely different FULL forward pass whose steering signal propagated
through the stack - not a readout-time rotation of the natural state.

Env:  BLEND_PCT=<0..1> (default 0.5)
      INJ_LAYER=<0..27> (default 14 - mid-stack on Qwen2-1.5B, 28 layers)
      THETA_INJ=<deg>   (default 20 - arc of the in-layer rotation)
      TRACE=1           (print per-switch: eff readout angle, hits)

Run:  HF_TOKEN=<tok> python3 gen_blend2.py [model] [prompt] [w1,w2,..]
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

ANTI = 4                       # de-repeat window length after a plant
NUCLEUS = 0.9                  # top-p keep
BLEND_PCT = float(os.environ.get('BLEND_PCT', '0.5'))
INJ_LAYER = int(os.environ.get('INJ_LAYER', '14'))
THETA_INJ = float(os.environ.get('THETA_INJ', '20'))
TRACE = os.environ.get('TRACE') == '1'


def main():
    t0 = time.time()
    print(f'\nBlend2 generation | {MODEL} | prompt={PROMPT!r} | words={WORDS} '
          f'| BLEND_PCT={BLEND_PCT} INJ_LAYER={INJ_LAYER} '
          f'THETA_INJ={THETA_INJ} | ntok={NTOK}')
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    eos_id = int(tok.eos_token_id)
    norm = model.model.norm if hasattr(model.model, 'norm') \
        else model.model.language_model.norm
    n_layers = len(model.model.layers)
    if not (0 <= INJ_LAYER < n_layers):
        raise SystemExit(f'INJ_LAYER {INJ_LAYER} out of range 0..{n_layers-1}')
    W = model.lm_head.weight.detach().float()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()
    bias = model.lm_head.bias
    bias = bias.detach().float() if bias is not None else None

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

    # ---- hooks ----
    def hook_norm(fn):
        return norm.register_forward_hook(fn)

    def forward_v(ids):
        vc = {}
        hk = hook_norm(
            lambda m, i, o: vc.__setitem__('v', o[0, -1, :].float()))
        try:
            with torch.no_grad():
                L = model(ids)
        finally:
            hk.remove()
        return L.logits[0, -1].float(), vc['v']

    def forward(ids, inj_p=None, anti_ids=None):
        hs = []
        try:
            if inj_p is not None:
                def inj(m, i, o, p=inj_p):
                    o[0, -1, :] = torch.as_tensor(p, dtype=o.dtype,
                                                  device=o.device)
                hs.append(hook_norm(inj))
            if anti_ids:
                def anti(m, i, o, aids=anti_ids):
                    o[0, -1, aids] = -30.0
                hs.append(model.lm_head.register_forward_hook(anti))
            with torch.no_grad():
                L = model(ids)
            return L.logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()

    def rot_to_angle(vv, tid, theta):
        a = math.radians(theta)
        v1 = vv / vv.norm()
        Wb = Wn[tid].to(vv.device)
        g0 = Wb - (v1 @ Wb) * v1
        gn = g0 / (g0.norm() + 1e-12)
        return (v1 * math.cos(a) + gn * math.sin(a)) * vv.norm()

    def natural_pass(ids):
        """FULL forward, no injection. Returns (logits, final state v_nat,
        mid-state at INJ_LAYER)."""
        params = {}
        hk = hook_norm(
            lambda m, i, o: params.__setitem__('v', o[0, -1, :].float()))
        hm = model.model.layers[INJ_LAYER].input_layernorm.register_forward_hook(
            lambda m, i, o: params.__setitem__(
                'mid', o[0, -1, :].float()))
        try:
            with torch.no_grad():
                L = model(ids)
        finally:
            hk.remove()
            hm.remove()
        return L.logits[0, -1].float(), params['v'], params['mid']

    def steered_pass(ids, mid, tid, theta):
        """FULL forward with steering injected INTO INJ_LAYER: rotates the
        mid-state toward the target row, then the rest of the stack runs.
        Returns (logits, full final state v_steer)."""
        steered_mid = rot_to_angle(mid.float(), tid, theta).to(mid.dtype)
        vc = {}
        # Return a NEW tensor (not in-place copy_) to avoid CUDA aliasing races
        def inj(m, i, o):
            new_o = o.clone()
            new_o[0, -1, :] = torch.as_tensor(steered_mid, dtype=o.dtype,
                                              device=o.device)
            return new_o
        hm = model.model.layers[INJ_LAYER].input_layernorm.register_forward_hook(
            inj)
        hk = hook_norm(
            lambda m, i, o: vc.__setitem__('v', o[0, -1, :].float()))
        try:
            with torch.no_grad():
                L = model(ids)
        finally:
            hm.remove()
            hk.remove()
        return L.logits[0, -1].float(), vc['v']

    def blend_states(v_nat, v_steer, lam):
        b = (1 - lam) * v_nat + lam * v_steer
        n = v_nat.norm()
        return b / b.norm() * n

    def head_logits(h):
        """read the LM head of a blended hidden state (fp32 matmul)."""
        log = (h.float() @ W.t())
        if bias is not None:
            log = log + bias
        return log

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
    plant_until = -1
    plant_word = None
    plant_tid = None

    for step in range(NTOK):
        L, v_nat = forward_v(ids)
        anti_a = plant_tid if (step < plant_until and plant_tid is not None) \
            else None
        bw = ({plant_word} if anti_a is not None else None)

        if step in switch_at:
            w = switch_at[step]
            # natural full pass (captures mid state too)
            L, v_nat_2, mid = natural_pass(ids)
            # steered full pass (in-layer injection, rest of stack runs)
            L_s, v_steer = steered_pass(ids, mid, word_ids[w], THETA_INJ)
            # hmm: L from natural_pass; blend final states at the head
            b = blend_states(v_nat_2, v_steer, BLEND_PCT)
            L = head_logits(b)
            plant_word = w
            plant_tid = word_ids[w]
            plant_until = step + 1 + ANTI
            if TRACE:
                ang = math.degrees(math.acos(
                    float((v_nat_2 / v_nat_2.norm())
                          @ (v_steer / v_steer.norm()))))
                print(f'      switch@{step} -> {w}:{tok.decode([word_ids[w]])!r} '
                      f'lam={BLEND_PCT} readout-angle_nat2steer={ang:.1f}deg '
                      f'head-top={tok.decode([int(L.argmax())])!r}')
        elif anti_a is not None:
            L = forward(ids, anti_ids=[anti_a])

        nxt = sample(L, block_words=bw)
        if nxt == eos_id:
            sampled.append(nxt)
            break
        sampled.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)

    txt = tok.decode(sampled)
    print(f'\n===== STEERED blend2 lam={BLEND_PCT} lay={INJ_LAYER} '
          f'th={THETA_INJ:.0f} ({", ".join(WORDS)}) =====')
    print(f'{PROMPT} {txt}')
    hits = {w: (w in txt) for w in WORDS}
    print(f'\nwords present: {hits}')
    print(f'[{time.time() - t0:.0f}s] {len(sampled)} tokens, '
          f'eos={"YES" if sampled and sampled[-1] == eos_id else "NO"}')


if __name__ == "__main__":
    main()