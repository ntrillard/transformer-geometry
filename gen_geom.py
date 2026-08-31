#!/usr/bin/env python3
"""gen_geom.py - PURE geometric steering (rotation-only, NO token planting).

Contrast with gen_blendtraj.py: that script edits the INPUT by planting each
target word as a real token in context. This script does none of that - it
only rotates the residual readout toward the target row and blends logits.

MODES
-----
Both modes use the SAME positive edit (rotate the readout toward the target
word) and NO suppression at all - nothing is ever pushed down, no logit is
ever decreased. The only difference is WHEN the edit stops:

  MODE=hold   steer for the entire WINDOW, every step. The target is forced to
              rank-1 continuously, so it is re-emitted forever (degenerate
              loop). The window keeps it finite; the model recovers between.

  MODE=emit   steer only UNTIL the target token is sampled once, then stop
              steering (the positive force is removed - the window goes
              passive). "Stop pushing up" (emit) is NOT suppression: it never
              lowers any probability, it just decides when the nudge ends.
              The word is now in the model's own context, so the model
              continues around it on its own. Anti-repeat blocks were tried
              and REMOVED - output was byte-identical, they never mattered.

TIMING CONTROLS
---------------
G_LAN is the share of STEERED logits in the readout blend; the rest is the
natural forward logits. G_LAN=0.8 keeps 20% natural logits alive in every
steered step - the natural trajectory is never fully silenced, which keeps
the model's own voice (and avoids template collapse at high G_LAN).

  ADAPT=<0|1>   Adaptive timing: instead of a fixed schedule, open the steer
                window exactly when the live sentence context is most
                amenable - when a pending word's NATURAL probability crosses
                ADAPT_THRESH or reaches the top-ADAPT_TOP of the live
                softmax. Steer right where the model is already closest to
                saying the word; force only when the context is ready.

  PRE_STEPS=<n> Multi-token pre-steer (only with fixed schedule): for the n
                steps BEFORE a window starts, rotate the readout at a ramped
                low angle (PRE_FRAC * G_ANGLE), so several tokens are bent
                toward the word before the insert point arrives. The sentence
                is already leading into the word when the window opens.

HIT_STOP is implicit in MODE=emit: once a window's word has appeared, that
window stops steering (honest miss if the window ends without the word).
Words that appear NATURALLY in free generation are also marked emitted, so a
scheduled window never re-forces an already-present word.

Env:  MODE=<hold|emit>      (default emit)
      G_ANGLE=<deg>         rotation toward target per step (default 9)
      G_LAN=<0..1>          share of steered logits (default 0.8 = 20% natural)
      WINDOW=<steps>        active window length per word (default 12)
      SW0=<steps>           first fixed window start (default 20)
      ADAPT=<0|1>           adaptive live-context timing (default 0)
      ADAPT_THRESH=<p>      natural-prob threshold to activate (default 0.01)
      ADAPT_TOP=<n>         also activate if word is top-n in live softmax (100)
      PRE_STEPS=<n>         ramped pre-steer before each fixed window (default 0)
      PRE_FRAC=<0..1>       max pre-steer angle as fraction of G_ANGLE (0.5)
      SENT=<0|1>            align fixed windows to fresh sentence boundaries (0)
      TARGET_TYPE=<token|dir>  steering target (token rows | a direction)
      TARGET_SENT=<str>     dir target: sentence whose hidden-state direction
                            becomes u_target (full-sentence steering)
      CONCEPT=<words>       dir target: space-separated words whose mean row
                            becomes u_target (concept centroid, no token needed)
      CONTRAST_TARGET=<s|s|..>  dir target via CONTRAST DIRECTION: mean hidden
                            state over these pipe-separated TARGET sentences
      CONTRAST_NEUTRAL=<s|s|..> minus mean state over these NEUTRAL sentences
                            gives u_dir (the steering-vector recipe).
      CONTRAST_MODE=<state|logit>  state: rotation toward state-diff (weak for
                            far directions). logit: additive dL = mean next-token
                            logits(target sents) - mean next-token logits(neutral
                            sents); readout = L_nat + ALPHA*dL - directly raises
                            the target-vocab logits (the strong construction).
      When CONTRAST_TARGET is set, every step >= SW0 gets a gentle constant
      steer toward that contrast direction (no window/emit - a sustained bias,
      for topic/sentence steering).
      BLEND_STEPS=<n>       ramp levels inside the window (default 1)

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
G_ANGLE = float(os.environ.get('G_ANGLE', '9'))
G_LAN = float(os.environ.get('G_LAN', '0.8'))
WINDOW = int(os.environ.get('WINDOW', '12'))
SW0 = int(os.environ.get('SW0', '20'))
ADAPT = os.environ.get('ADAPT', '0') == '1'
ADAPT_THRESH = float(os.environ.get('ADAPT_THRESH', '0.01'))
ADAPT_TOP = int(os.environ.get('ADAPT_TOP', '100'))
ADAPT_MIN = int(os.environ.get('ADAPT_MIN', '15'))
PRE_STEPS = int(os.environ.get('PRE_STEPS', '0'))
SENT = os.environ.get('SENT', '1') == '1'
PRE_FRAC = float(os.environ.get('PRE_FRAC', '0.5'))
TARGET_TYPE = os.environ.get('TARGET_TYPE', 'token')
TARGET_SENT = os.environ.get('TARGET_SENT', '')
CONCEPT = os.environ.get('CONCEPT', '')
BLOCK_REGION = os.environ.get('BLOCK_REGION', '0') == '1'
ANTI = int(os.environ.get('ANTI', '4'))  # region-block steps (dir only)
CONTRAST_TARGET = os.environ.get('CONTRAST_TARGET', '')
CONTRAST_NEUTRAL = os.environ.get('CONTRAST_NEUTRAL', '')
CONTRAST_MODE = os.environ.get('CONTRAST_MODE', 'state')
ALPHA = float(os.environ.get('ALPHA', '3.0'))
DL_TOP = int(os.environ.get('DL_TOP', '200'))
DL_DROP = int(os.environ.get('DL_DROP', '0'))
BLEND_STEPS = max(1, int(os.environ.get('BLEND_STEPS', '1')))
TRACE = os.environ.get('TRACE') == '1'


def main():
    t0 = time.time()
    print(f'\nPure-geometry[{MODE}] | {MODEL} | prompt={PROMPT!r} '
          f'| words={WORDS} | G_ANGLE={G_ANGLE} G_LAN={G_LAN} '
          f'WINDOW={WINDOW} ADAPT={"1" if ADAPT else "0"} SENT={"1" if SENT else "0"} '
          f'PRE={PRE_STEPS} ntok={NTOK}')
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

    # dir-target direction (sentence state or concept centroid)
    u_dir = None
    if TARGET_TYPE == 'dir':
        if TARGET_SENT:
            tids = tok(TARGET_SENT, add_special_tokens=False,
                       return_tensors='pt').input_ids.to(DEV)
            vc = {}
            hk = norm.register_forward_hook(
                lambda m, i, o: vc.__setitem__('states',
                                               o[0].float().clone()))
            with torch.no_grad():
                model(tids)
            hk.remove()
            u_dir = vc['states'].mean(0)     # sentence centroid over all tokens
            print(f'  TARGET_SENT -> {TARGET_SENT!r} '
                  f'(mean state over {tids.shape[1]} tokens)')
        elif CONTRAST_TARGET:
            tgt = [x.strip() for x in CONTRAST_TARGET.split('|') if x.strip()]
            neu = [x.strip() for x in CONTRAST_NEUTRAL.split('|') if x.strip()]
            if CONTRAST_MODE == 'logit':
                dL = None
                with torch.no_grad():
                    for s in tgt + neu:
                        t = tok(s, add_special_tokens=False,
                                return_tensors='pt').input_ids.to(DEV)
                        Ls = model(t).logits[0, -1].float().cpu()  # next-token logits
                        if dL is None:
                            dL = torch.zeros_like(Ls)
                        dL += Ls if s in tgt else -Ls
                dL /= max(1, len(tgt) + len(neu))
                # z-score, optionally drop the top few extreme tokens (they
                # are single-token loop latchers), then top-k positive mask.
                # Diffuse z-magnitudes over a clean vocab = topic transport
                # (beach->fantasy a=2); extreme spikes = degenerate loops.
                dL = (dL - dL.mean()) / (dL.std() + 1e-6)
                drop = int(os.environ.get('DL_DROP', '0'))
                if drop > 0:
                    for _ in range(drop):
                        dL[dL.argmax()] = 0.0
                k = int(os.environ.get('DL_TOP', '200'))
                topk = dL.argsort(descending=True)[:k]
                m = torch.zeros_like(dL)
                m[topk] = 1.0
                dL = (dL * m).to(DEV)
                print(f'  CONTRAST[logit]: {len(tgt)} tgt - {len(neu)} neu, '
                      f'|dL_z|={dL.norm().item():.2f}, top-{k} mask')
                print(f'    top dL tokens: {[tok.decode([int(i)]) for i in dL.argsort(descending=True)[:6]]}')
            else:
                def _sent_mean(s):
                    t = tok(s, add_special_tokens=False,
                            return_tensors='pt').input_ids.to(DEV)
                    vc = {}
                    hk = norm.register_forward_hook(
                        lambda m, i, o: vc.__setitem__('states', o[0].float().clone()))
                    with torch.no_grad():
                        model(t)
                    hk.remove()
                    return vc['states'].mean(0).cpu()
                tmean = torch.stack([_sent_mean(s) for s in tgt]).mean(0)
                nmean = torch.stack([_sent_mean(s) for s in neu]).mean(0)
                u_dir = (tmean - nmean).float()
                print(f'  CONTRAST[state]: {len(tgt)} tgt - {len(neu)} neu, |u|={u_dir.norm().item():.3f}')
        elif CONCEPT:
            ws = [w.strip() for w in CONCEPT.split() if w.strip()]
            rows = []
            for w in ws:
                sp = tok(' ' + w, add_special_tokens=False).input_ids
                if len(sp) == 1:
                    rows.append(Wn[int(sp[0])])
            if rows:
                u_dir = torch.stack(rows).mean(0)
                print(f'  CONCEPT {ws} -> centroid of {len(rows)} rows')
        if u_dir is not None:
            u_dir = (u_dir / u_dir.norm()).to(DEV)
            print(f'  dir-target: u_dir norm=1, first dims={u_dir[:3].tolist()}')

    region_ids = set()      # single-token CONCEPT rows, for region-emit
    if TARGET_TYPE == 'dir' and CONCEPT:
        for w in CONCEPT.split():
            sp = tok(' ' + w, add_special_tokens=False).input_ids
            if len(sp) == 1:
                region_ids.add(int(sp[0]))

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

    def rot_to_vec(vv, u, theta):
        # rotate vv toward an ARBITRARY unit direction u (not a token row)
        a = math.radians(theta)
        v1 = vv / vv.norm()
        u = u.to(vv.device)
        g0 = u - (v1 @ u) * v1
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
    emitted = {}            # word -> step emitted (steered or natural)
    how = {}                # word -> 'steered' / 'natural'
    pending = list(WORDS)
    sent_win = {}             # scheduled step -> aligned start
    sent_boundary = -1       # first step AFTER a sentence-final token
    adapt_active = None
    adapt_start = -1
    adapt_until = -1
    dir_block_until = -1
    dir_block_id = None

    for step in range(NTOK):
        w_active = None
        pre_frac = 0.0
        is_pre = False
        contrast_steer = (CONTRAST_TARGET and step >= SW0)
        if contrast_steer:
            L_nat, v = forward_v(ids)
            if CONTRAST_MODE == 'logit':
                # additive logit contrast: directly raise target-vocab logits
                L = L_nat + ALPHA * dL
                if TRACE:
                    print(f'      [{step}] contrast[logit] alpha={ALPHA}')
            else:
                # state-space rotation toward the contrast direction
                L_steer = forward(ids, inj_p=rot_to_vec(v, u_dir, G_ANGLE))
                L = (1 - G_LAN) * L_nat + G_LAN * L_steer
                if TRACE:
                    sim = (v / v.norm()) @ u_dir
                    print(f'      [{step}] contrast cos(v,u)={sim.item():+.3f}')
            nxt = sample(L)
            if nxt == eos_id:
                sampled.append(nxt)
                break
            sampled.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
            continue

        if ADAPT:
            if (adapt_active is not None
                    and adapt_active not in emitted
                    and step < adapt_until):
                w_active = adapt_active
        else:
            # fixed schedule (optionally sentence-aligned)
            for s, w in win_at.items():
                if SENT:
                    # wait for the next fresh sentence boundary at/after s
                    if step == s and s not in sent_win:
                        sent_win[s] = None
                    if s in sent_win and sent_win[s] is None and step <= s + (NTOK // (n_sw + 1)):
                        if step == sent_boundary:
                            sent_win[s] = step
                    sw = sent_win.get(s)
                    if sw is not None and sw <= step < sw + WINDOW:
                        w_active = w
                        break
                else:
                    if s <= step < s + WINDOW:
                        w_active = w
                        break
            if w_active is None and PRE_STEPS > 0:
                for s, w in win_at.items():
                    if s - PRE_STEPS <= step < s:
                        w_active = w
                        is_pre = True
                        pre_frac = PRE_FRAC * (1.0 - (s - step) / (PRE_STEPS + 1))
                        break

        # MODE=hold: steer the whole window. MODE=emit: only until the word
        # has been sampled once - then the window goes passive (the positive
        # force stops; nothing is ever suppressed). No logit is decreased in
        # either mode. dir targets: hold the whole window (region targets
        # cannot degenerate-loop, so continuous holding is safe).
        if TARGET_TYPE == 'dir':
            steer_this = (w_active is not None
                          and (MODE == 'hold' or w_active not in emitted))
        else:
            steer_this = (w_active is not None and MODE == 'hold') or \
                         (w_active is not None and MODE == 'emit'
                             and w_active not in emitted)

        if steer_this:
            if is_pre:
                th = G_ANGLE * pre_frac
                lam = G_LAN
            else:
                if ADAPT:
                    sw = adapt_start
                elif SENT:
                    sw = sent_win.get(next(s for s, w in win_at.items()
                                           if w == w_active), 0)
                else:
                    sw = next(s for s, w in win_at.items() if w == w_active)
                done = step - sw
                level = min(BLEND_STEPS - 1,
                            int(done * BLEND_STEPS / WINDOW)) if BLEND_STEPS > 1 \
                    else 0
                lam_k = G_LAN * (level + 1) / BLEND_STEPS
                th_k = G_ANGLE * (level + 1) / BLEND_STEPS
                th = th_k
                lam = lam_k
            L_nat, v = forward_v(ids)
            if TARGET_TYPE == 'dir':
                L_steer = forward(ids, inj_p=rot_to_vec(v, u_dir, th))
                if TRACE:
                    sim = (v / v.norm()) @ u_dir
                    print(f'      [{step}] cos(v,u_dir)={sim.item():+.3f}')
            else:
                L_steer = forward(ids,
                                  inj_p=rot_to_angle(v, word_ids[w_active], th))
            L = (1 - lam) * L_nat + lam * L_steer
            if BLOCK_REGION and step < dir_block_until and dir_block_id is not None:
                L[dir_block_id] = -30.0
            if TRACE:
                print(f'      [{step}] steer {w_active} '
                      f'{"PRE" if is_pre else "win"} lam={lam:.2f} th={th:.1f}')
            nxt = sample(L)
            if TARGET_TYPE == 'dir' and region_ids:
                # region-emit: any region token sampled ends this window
                if nxt in region_ids:
                    emitted[w_active] = step
                    how[w_active] = 'region'
                    if w_active in pending:
                        pending.remove(w_active)
                    if BLOCK_REGION:
                        dir_block_until = step + 1 + ANTI
                        dir_block_id = nxt
                    if TRACE:
                        print(f'      [{step}] REGION {tok.decode([nxt])!r} EMITTED')
            elif nxt == word_ids[w_active]:
                emitted[w_active] = step
                how[w_active] = 'steered'
                if w_active in pending:
                    pending.remove(w_active)
                if ADAPT:
                    adapt_active = None
                if TRACE:
                    print(f'      [{step}] WORD {w_active} EMITTED (steered)')
        else:
            # --- free run ---
            L, _ = forward_v(ids)
            if BLOCK_REGION and step < dir_block_until and dir_block_id is not None:
                L = L.clone()
                L[dir_block_id] = -30.0
            nxt = sample(L)
            # defensive: word appeared naturally while pending
            for w in pending:
                if nxt == word_ids[w]:
                    emitted[w] = step
                    how[w] = 'natural'
                    pending.remove(w)
                    if ADAPT:
                        adapt_active = None
                    if TRACE:
                        print(f'      [{step}] WORD {w} EMITTED (natural)')
                    break
            if ADAPT:
                # read the LIVE context: activate the pending word whose
                # natural probability is already high / near the top.
                if adapt_active is None or adapt_active in emitted:
                    p = torch.softmax(L, 0)
                    for w in pending:
                        if step < ADAPT_MIN:
                            break
                        tid = word_ids[w]
                        pr = p[tid].item()
                        rank = int((p > p[tid]).sum())
                        if pr >= ADAPT_THRESH or rank <= ADAPT_TOP:
                            adapt_active = w
                            adapt_start = step
                            adapt_until = step + WINDOW
                            if TRACE:
                                print(f'      [{step}] ADAPT activate {w} '
                                      f'p={pr:.4f} rank={rank}')
                            break
                elif step >= adapt_until:
                    # window over without landing: honest miss
                    if TRACE:
                        print(f'      [{step}] ADAPT MISS {adapt_active}')
                    if adapt_active in pending:
                        pending.remove(adapt_active)
                    adapt_active = None

        if nxt == eos_id:
            sampled.append(nxt)
            break
        sampled.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        if SENT:
            d = tok.decode([nxt]).strip()
            if d in ('.', '!', '?', '.\n', '\n') or d.endswith('.'):
                sent_boundary = step + 1

    txt = tok.decode(sampled)
    print(f'\n===== PURE-GEOMETRY[{MODE}] angle={G_ANGLE:.0f} lam={G_LAN} '
          f'adapt={"1" if ADAPT else "0"} pre={PRE_STEPS} '
          f'tgt={TARGET_TYPE} ({", ".join(WORDS)}) =====')
    print(f'{PROMPT} {txt}')
    hits = {w: (w in txt) for w in WORDS}
    counts = {w: txt.count(w) for w in WORDS}
    print(f'\nwords present: {hits}')
    print(f'word counts  : {counts}')
    print(f'emitted: {how} @ {emitted}')
    print(f'[{time.time() - t0:.0f}s] {len(sampled)} tokens, '
          f'eos={"YES" if sampled and sampled[-1] == eos_id else "NO"}')


if __name__ == "__main__":
    main()
