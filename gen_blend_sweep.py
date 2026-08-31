#!/usr/bin/env python3
"""gen_blend_sweep.py - sweep BLEND FRACTION x BLEND TECHNIQUE at the readout.

At each switch step we take the PURE final hidden state v (natural) and the
STEERED final state s (the same state rotated on the sphere to the target's
rank-1 angle). We blend them in STATE space and inject the blend right before
the LM head reads (hook on the final RMSNorm):

    linear:  h = normalize((1-lam)*v + lam*s) * ||v||
    slerp :  h = ( sin((1-lam)*th)*vhat + sin(lam*th)*shat ) / sin(th) * ||v||
               where th = ang(v, s)

Sweeps all (mode, lam) combinations in-process (model loaded once).

Env:  MODES=linear,slerp   (comma list; default linear,slerp)
      LAMS=0.5,0.7,0.9     (comma list; default 0.5,0.7,0.9)
      TRACE=1              (print per-switch effective angles)

Run:  HF_TOKEN=<tok> python3 gen_blend_sweep.py [model] [prompt] [w1,w2,..]
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
ANTI = 4                        # de-repeat window
NUCLEUS = 0.9                   # top-p keep
MODES = [m for m in os.environ.get('MODES', 'linear,slerp').split(',') if m]
LAMS = [float(x) for x in os.environ.get('LAMS', '0.5,0.7,0.9').split(',')]
TRACE = os.environ.get('TRACE') == '1'


def blend_state(v, s, lam, mode):
    """Blend pure state v and steered state s by fraction lam.
    Always returns a unit-ish state with norm == ||v||."""
    vn = v.norm()
    vh = v / vn
    sh = s / s.norm()
    if mode == 'linear':
        b = (1 - lam) * v + lam * s
        return b / b.norm() * vn
    if mode == 'slerp':
        th = math.acos(float((vh @ sh).clamp(-1.0, 1.0)))
        if th < 1e-6:
            return v.clone()
        num = math.sin((1 - lam) * th) * vh + math.sin(lam * th) * sh
        return num / num.norm() * vn
    raise ValueError(mode)


def main():
    t0 = time.time()
    print(f'\nBlend sweep | {MODEL} | prompt={PROMPT!r} | words={WORDS} | '
          f'modes={MODES} | lams={LAMS} | ntok={NTOK}')
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

    def run(config):
        mode, lam = config
        torch.manual_seed(SEED)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        sampled = []
        plant_until = -1
        plant_word = None
        plant_tid = None
        eff_angles = []
        for step in range(NTOK):
            L, v = forward_v(ids)
            anti_a = plant_tid if (step < plant_until
                                   and plant_tid is not None) else None
            bw = ({plant_word} if anti_a is not None else None)
            if step in switch_at:
                w = switch_at[step]
                th_full = best_angle(ids, v, word_ids[w])
                s = rot_to_angle(v, word_ids[w], th_full)   # steered state
                h = blend_state(v, s, lam, mode)            # blended state
                L = forward(ids, inj_p=h)
                plant_word = w
                plant_tid = word_ids[w]
                plant_until = step + 1 + ANTI
                ang = math.degrees(math.acos(
                    float((v / v.norm()) @ (h / h.norm()))))
                eff_angles.append(round(ang, 1))
            elif anti_a is not None:
                L = forward(ids, anti_ids=[anti_a])
            nxt = sample(L, block_words=bw)
            if nxt == eos_id:
                sampled.append(nxt)
                break
            sampled.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        txt = tok.decode(sampled)
        hits = {w: (w in txt) for w in WORDS}
        return txt, hits, eff_angles

    # ---- sweep ----
    for lam in LAMS:
        for mode in MODES:
            txt, hits, eff = run((mode, lam))
            print(f'\n{"=" * 70}')
            print(f'=== BLEND {mode}  lam={lam}  '
                  f'eff-angles={eff}  ({", ".join(WORDS)}) ===')
            print(f'{PROMPT} {txt}')
            print(f'words present: {hits}')

    print(f'\n[{time.time() - t0:.0f}s] sweep complete: '
          f'{len(MODES) * len(LAMS)} configs x {len(WORDS)} words')


if __name__ == "__main__":
    main()