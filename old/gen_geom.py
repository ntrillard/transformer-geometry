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

De-latch controls (topic steering, logit contrast):
      REP_PEN=<logits>      anti-priming: reduce logits of tokens sampled in the
                            last REP_WINDOW steps by REP_PEN each (default 0).
                            REP_COUNT=1 makes it per-occurrence (count-scaled) so
                            a dense near-synonym cluster collapses instead of
                            cycling through its members.
      CONTRAST_WINDOW=<n>   apply the logit contrast for only n steps after SW0,
                            then free-run. Bounds the dose so sustained re-tilting
                            cannot convert a tilt into a repetition latch.
      DOSE_*               (experimental, refuted in Appendix E) DOSE_C absolute
                            junk pool, DOSE_FLAT cap family, DOSE_CJK synthetic
                            CJK variance pump, CJK_EXCLUDE sampler exclusion -
                            kept for reproduction of the failures.

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
NTOK = int(os.environ.get('NTOK', '120'))
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
CONTRAST_BLOCK = os.environ.get('CONTRAST_BLOCK', '0') == '1'
CONTRAST_ANTI = int(os.environ.get('CONTRAST_ANTI', '4'))
DOSE_C = float(os.environ.get('DOSE_C', '0'))
DOSE_POOL = int(os.environ.get('DOSE_POOL', '3000'))
DOSE_FLAT = os.environ.get('DOSE_FLAT', '0') == '1'
DOSE_MAX = float(os.environ.get('DOSE_MAX', '0.6'))
DOSE_CJK = os.environ.get('DOSE_CJK', '0') == '1'
CJK_COUNT = int(os.environ.get('CJK_COUNT', '3000'))
CJK_BOOST = float(os.environ.get('CJK_BOOST', '30.0'))
REP_PEN = float(os.environ.get('REP_PEN', '0'))
REP_WINDOW = int(os.environ.get('REP_WINDOW', '50'))
REP_COUNT = os.environ.get('REP_COUNT', '0') == '1'
CJK_EXCLUDE = os.environ.get('CJK_EXCLUDE', '0') == '1'
EXAM_EXCLUDE = os.environ.get('EXAM_EXCLUDE', '0') == '1'
CONTRAST_WINDOW = int(os.environ.get('CONTRAST_WINDOW', '0'))
CONTRAST_TARGET = os.environ.get('CONTRAST_TARGET', '')
CONTRAST_NEUTRAL = os.environ.get('CONTRAST_NEUTRAL', '')
CONTRAST_MODE = os.environ.get('CONTRAST_MODE', 'state')
ALPHA = float(os.environ.get('ALPHA', '3.0'))
DL_TOP = int(os.environ.get('DL_TOP', '200'))
DL_DROP = int(os.environ.get('DL_DROP', '0'))
DL_CLEAN = int(os.environ.get('DL_CLEAN', '0'))
CONTRAST_EMIT = os.environ.get('CONTRAST_EMIT', '0') == '1'
BLEND_STEPS = max(1, int(os.environ.get('BLEND_STEPS', '1')))
# discriminant steering: target word-cluster minus distract word-cluster
DISC_TARGET = os.environ.get('DISC_TARGET', '')
DISC_DISTRACT = os.environ.get('DISC_DISTRACT', '')
TRACE = os.environ.get('TRACE') == '1'
# experimental no-insertion proof: forbid the steering TARGETS themselves from
# ever being sampled (CONCEPT region words / boosted contrast tokens). If a
# run still carries the theme with ZERO of its target tokens in the output,
# the transport is synthesis, not insertion.
CONCEPT_BLOCK = os.environ.get('CONCEPT_BLOCK', '0') == '1'
BLOCK_WORDS = os.environ.get('BLOCK_WORDS', '')   # strict-block extra word set (held-out test)
BLOCK_BOOSTED = os.environ.get('BLOCK_BOOSTED', '0') == '1'


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
        cjk_ids = []
        cjk_mode = False
        rep_hist = []
        cont_emit_done = False
        if CJK_EXCLUDE:
            # build the CJK pool up-front (has to exist before first sample)
            cache = '/tmp/cjk_pool.txt'
            if os.path.exists(cache):
                cjk_ids = [int(x) for x in open(cache).read().split()]
            else:
                vs = tok.vocab_size
                pids = []
                for i in range(vs):
                    d = tok.decode([i]).strip()
                    if any('\u4e00' <= ch <= '\u9fff' for ch in d):
                        pids.append(i)
                cjk_ids = pids
                with open(cache, 'w') as f:
                    f.write('\n'.join(map(str, cjk_ids)))
            print(f'  CJK_EXCLUDE: len={len(cjk_ids)} script tokens excluded from sampling')
        exam_ids = []
        if EXAM_EXCLUDE:
            # the exam-template lock: the model's English-exam prior (this model
            # was trained on massive CN English-exam corpora) makes a short
            # fragment like 'We walked in the park and' collapse into a cloze
            # template ('____ A. golf B. football'). These marker tokens are
            # structurally unreachable in narrative prose but strongly primed by
            # that prior - same class as CJK eruption. Bar them from sampling so
            # template-bound seeds must continue narratively.
            _m = ['____', '___', '__', '_', '________', '______', '[', ']',
                  '．', '答案', '答案:', '　', '（）', '[ ]', '[  ]']
            _blocked = {2130, 5973, 563, 62, 3979, 58, 60, 2279, 58883,
                        9909, 7552, 102349, 25, 22441,}
            _seen = set()
            for _s in _m:
                for _i in tok(_s, add_special_tokens=False).input_ids:
                    if _i in _blocked and _i not in _seen:
                        _seen.add(_i); exam_ids.append(_i)
            # content-wise: ANY token whose decoded content is an underscore run
            # (len>=2, optional surrounding space/newline) is an exam cloze marker
            # - never legitimate narrative. Catches fused/space-prefixed variants
            # that exact-id blocking misses (' ____'=30743, '______'=32671, etc).
            for _i in range(tok.vocab_size):
                _d = tok.decode([_i]).strip()
                if len(_d) >= 2 and set(_d) == {'_'} and _i not in _seen:
                    _seen.add(_i); exam_ids.append(_i)
                # singleton bracket tokens (fused with a space) are the OTHER
                # fusion hole: ' [' is id 508, a different id than bare '['=58.
                # Block any token whose stripped content is exactly '[' or ']',
                # plus full-width parens (also fusion-prone: ' （' is 42344,
                # not bare 9909). Never legitimate narrative content alone.
                elif _d in ('[', ']', '（', '）') and _i not in _seen:
                    _seen.add(_i); exam_ids.append(_i)
            print(f'  EXAM_EXCLUDE: {len(exam_ids)} exam-marker tokens excluded: {exam_ids}')
        if CONTRAST_TARGET:
            tgt = [x.strip() for x in CONTRAST_TARGET.split('|') if x.strip()]
            neu = [x.strip() for x in CONTRAST_NEUTRAL.split('|') if x.strip()]
            if CONTRAST_MODE == 'logit':
                ns = None
                with torch.no_grad():
                    for s in neu:
                        t = tok(s, add_special_tokens=False,
                                return_tensors='pt').input_ids.to(DEV)
                        Ls = model(t).logits[0, -1].float().cpu()
                        ns = Ls if ns is None else ns + Ls
                nm = ns / max(1, len(neu))
                # per-sentence contributions: each target sentence's diff vs the
                # neutral mean is z-scored SEPARATELY, so a sentence in any
                # language contributes by SIGNAL, not by raw logit magnitude
                # (this fixes the EN+ZH mixing failure where Chinese swamped
                # the mean). Language channels are inherently balanced.
                dL = None
                with torch.no_grad():
                    for s in tgt:
                        t = tok(s, add_special_tokens=False,
                                return_tensors='pt').input_ids.to(DEV)
                        Ls = model(t).logits[0, -1].float().cpu()
                        c = Ls - nm
                        c = (c - c.mean()) / (c.std() + 1e-6)
                        dL = c if dL is None else dL + c
                dL = dL / max(1, len(tgt))
                # SIGNAL: top-k positive mask of the per-sentence-z-summed diff.
                k = int(os.environ.get('DL_TOP', '200'))
                sig = dL.clone()
                topk = sig.argsort(descending=True)[:k]
                m = torch.zeros_like(sig)
                m[topk] = 1.0
                sig = sig * m
                if DL_CLEAN:
                    # The raw top-K boost is polluted by (a) CJK mass and (b) BPE
                    # sub-word fragments (aurus/beros/imension) that decode into
                    # vocabulary-salad when sampled. Filter to CLEAN WORD-BOUNDARY
                    # tokens only: raw token string starts with the space marker
                    # (a fresh word, never a mid-word piece) AND decodes to a pure
                    # ASCII word (excludes CJK, numbers, symbols). Then re-take the
                    # top-k over the surviving set so the boost is pure English
                    # vocabulary with no junk tail.
                    vt = tok.convert_ids_to_tokens(list(range(tok.vocab_size)))
                    keep = []
                    for i in topk.tolist():
                        rt = vt[i] or ''
                        dt = tok.decode([i]).strip()
                        if (rt.startswith('\u0120') and dt and
                                all(('a' <= ch <= 'z') or ('A' <= ch <= 'Z')
                                    or ch in ("'", '-') for ch in dt) and
                                len(dt) >= 3):
                            keep.append(i)
                    m2 = torch.zeros_like(sig)
                    m2[keep] = 1.0
                    sig = sig * m2
                    k2 = min(k, len(keep))
                    topk2 = sig.argsort(descending=True)[:k2]
                    topk = topk2
                    print(f'  DL_CLEAN: {len(keep)} clean word-boundary tokens '
                          f'({k2} in final mask)')
                if DOSE_FLAT:
                    # DOSE_FLAT = the DETERMINISTIC replication of the emergent
                    # EN+ZH mass-sink, with NO foreign language, NO junk pool,
                    # NO suppression. The legacy z machinery (full-vocab z,
                    # then top-K mask) naturally produces the peaked-but-ordered
                    # tail that the bilingual case also produced - that SHAPE is
                    # the emergent property, not the flatness: strong ordered
                    # peaks among the boosted set (peak:mean ~2-3), only their
                    # absolute magnitude was compressed by the foreign mass.
                    # We reproduce the exact legacy shape (full-vocab z -> top-K
                    # mask) and scale only the PEAK to DOSE_MAX, so the knob is:
                    # applied peak = ALPHA * DOSE_MAX. At DOSE_MAX = raw_peak
                    # this is byte-identical to legacy; at lower sizes it is
                    # the same shape with a tamer (anti-latch) dose.
                    with torch.no_grad():
                        dz = (dL - dL.mean()) / (dL.std() + 1e-6)
                        t2 = dz.argsort(descending=True)[:k]
                        mz = torch.zeros_like(dz)
                        mz[t2] = 1.0
                        dz = dz * mz
                        rp = dz.abs().max()
                        if rp > 1e-9:
                            dz = dz / rp * DOSE_MAX
                        dL = dz.to(DEV)
                        boosted_set = set(t2.tolist())
                        nz = dz[dz != 0]
                        print(f'  DOSE_FLAT: peak={DOSE_MAX} raw_peak={rp.item():.1f} '
                              f'mean={nz.abs().mean().item():.2f} '
                              f'||dL||={dz.norm().item():.1f}')
                elif DOSE_CJK:
                    # DOSE_CJK = the DETERMINISTIC, SCRIPT-BASED replication of the
                    # emergent EN+ZH dilution. The bilingual case worked because the
                    # target's continuation carried a majority-foreign mass which
                    # dominated the z-variance, re-scaling every English token's
                    # applied boost to a gentle tilt (~0.4-0.8 sigma - strong enough
                    # to bend a narrative, too weak for any token to latch). We
                    # synthesize that SAME shape for ANY topic: scan the vocab for
                    # CJK-script tokens (never a grammatical continuation of English
                    # prose), pump a flat boost into a pool of them, z-score the
                    # full vector (pool dominates variance -> English signal gentle),
                    # keep the top-K signal masked. The pool is excluded from the
                    # sampler, so it is a pure variance pump: no Chinese eruption
                    # (T5), no rare-token leak (DOSE_C), no reliance on the topic
                    # happening to be Chinese-leaning.
                    with torch.no_grad():
                        if not cjk_ids:
                            # Qwen's tokenizer is byte-level BPE: convert_ids_to_tokens
                            # returns byte-encoded strings, so detect CJK by DECODING
                            # each id (tok.decode reverses the byte encoding). Cache
                            # the pool so a 4-8 run battery doesn't re-decode 151k ids.
                            cache = '/tmp/cjk_pool.txt'
                            if os.path.exists(cache):
                                cjk_ids = [int(x) for x in open(cache).read().split()]
                            else:
                                vs = tok.vocab_size
                                pids = []
                                for i in range(vs):
                                    d = tok.decode([i]).strip()
                                    if any('\u4e00' <= ch <= '\u9fff' for ch in d):
                                        pids.append(i)
                                cjk_ids = pids
                                with open(cache, 'w') as f:
                                    f.write('\n'.join(map(str, cjk_ids)))
                            print(f'  CJK pool: {len(cjk_ids)} script tokens')
                        pump = cjk_ids[:CJK_COUNT]
                        total = sig.clone()
                        total[pump] = CJK_BOOST
                        mean = total.mean()
                        std = total.std() + 1e-6
                        dL = (total - mean) / std
                        mask = torch.zeros_like(dL)
                        mask[topk] = 1.0
                        mask[pump] = 1.0
                        dL = (dL * mask).to(DEV)
                        boosted_set = set(topk.tolist())
                        cjk_mode = True
                        sig_z = ((sig - mean) / std)[topk].abs().mean().item()
                        print(f'  DOSE_CJK: pump {len(pump)} of {len(cjk_ids)} CJK @ {CJK_BOOST}; '
                              f'signal mean|z|={sig_z:.2f}')
                elif DOSE_C > 0:
                    # DOSE = replicate the emergent EN+ZH mass-sink WITHOUT any
                    # foreign language. The bilingual dL worked because the
                    # unreachable foreign block dominated the z-variance
                    # (tokens at 5-10 sigma), which re-scaled the reachable
                    # English signal down to a gentle tilt (strong enough to
                    # steer, too weak to latch). Here we inject an ABSOLUTE
                    # boost DOSE_C into a flat pool of unreachable tokens
                    # (natural prob < 1e-9), so the z-score re-scales the
                    # reachable signal exactly as the Chinese mass did -
                    # deterministic, any language.
                    pnv = torch.softmax(nm.float(), 0)
                    reach = (pnv < 1e-9).nonzero().flatten().tolist()
                    n_pool = min(DOSE_POOL, len(reach))
                    pool = reach[:n_pool]
                    d = torch.zeros_like(sig)
                    d[pool] = DOSE_C
                    total = (sig + d)
                    mean = total.mean()
                    std = total.std() + 1e-6
                    dL = (total - mean) / std
                    mask = torch.zeros_like(dL)
                    mask[topk] = 1.0
                    mask[pool] = 1.0
                    dL = (dL * mask).to(DEV)
                    boosted_set = set(topk.tolist()) | set(pool)
                    sig_z = ((sig.cpu() - mean) / std)[topk].abs().mean().item()
                    pool_z = ((d[pool].cpu() - mean) / std).abs().mean().item()
                    print(f'  DOSE: C={DOSE_C} into {n_pool} unreachable tokens; '
                          f'mean|z| signal={sig_z:.2f} pool={pool_z:.2f}')
                else:
                    # legacy path
                    if DL_CLEAN:
                        # DL_CLEAN built the filtered mask over sig; z-score the
                        # surviving set and mask to it - same shape as legacy, but
                        # restricted to clean word-boundary tokens.
                        mz = torch.zeros_like(sig)
                        mz[topk] = 1.0
                        dL = ((sig - sig.mean()) / (sig.std() + 1e-6) * mz).to(DEV)
                        boosted_set = set(topk.tolist())
                    else:
                        dL = (dL - dL.mean()) / (dL.std() + 1e-6)
                        if os.environ.get('DL_FULL') == '1':
                            # use the ENTIRE 152k-dim contrast (the diffuse 98%):
                            # the meaningful signal is a global tilt, not the top-k
                            # spike (measured: top-200 = 2.1% of ||z||^2, mean|z|
                            # = 0.78 across all dims). Apply the full z unfiltered.
                            dL = dL.to(DEV)
                            boosted_set = set()
                        else:
                            drop = int(os.environ.get('DL_DROP', '0'))
                            if drop > 0:
                                for _ in range(drop):
                                    dL[dL.argmax()] = 0.0
                            topk2 = dL.argsort(descending=True)[:k]
                            m2 = torch.zeros_like(dL)
                            m2[topk2] = 1.0
                            dL = (dL * m2).to(DEV)
                            boosted_set = set(topk2.tolist())
                boost_block = boosted_set.copy()   # survives the gen-loop wipe (bug 637)
                print(f'  CONTRAST[logit]: {len(tgt)} tgt - {len(neu)} neu, '
                      f'|dL_z|={dL.norm().item():.2f}, top-{k} mask')
                dL_c = dL.cpu()
                pn = torch.softmax(nm, 0)
                bidx = dL_c.argsort(descending=True)[:k]
                maxn = pn[bidx].max().item()
                eng = 0
                for i in bidx[:k].tolist():
                    s = tok.decode([i]).strip()
                    if s and all(ord(c) < 128 for c in s) and sum(c.isalpha() for c in s) >= 3:
                        eng += 1
                print(f'    boost max natural prob={maxn:.5f}  eng-frac={eng / k:.2f}')
                if os.environ.get('CHECKONLY') == '1':
                    engf = eng / k
                    if engf >= 0.6:
                        zone = 'LATCH-RISK (>=0.6): selectable EN tokens will self-latch'
                    elif engf <= 0.35:
                        zone = 'INVISIBLE (<=0.35): foreign-only boost, no entry into EN prose'
                    else:
                        zone = 'TRANSPORT-ZONE (0.35-0.6): bilingual diffusion, clean bend expected'
                    print(f'  SCORECARD: eng-frac={engf:.2f} -> {zone}')
                    print('  CHECKONLY: skipping generation')
                    return
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
        if DISC_TARGET:
            # discriminant steering: target-cluster centroid MINUS distract-cluster
            # centroid. Cancels the shared (e.g. beach-like) component so the
            # direction points at what is UNIQUE to the target cluster - reaches
            # semantic neighbors (funeral/cemetery/葬) no single anchor contains.
            dt_ws = [w.strip() for w in DISC_TARGET.split() if w.strip()]
            dd_ws = [w.strip() for w in DISC_DISTRACT.split() if w.strip()]
            def _row(w):
                sp = tok(' ' + w, add_special_tokens=False).input_ids
                return Wn[int(sp[0])] if len(sp) == 1 else None
            tr = [_row(w) for w in dt_ws]; tr = [r for r in tr if r is not None]
            dr = [_row(w) for w in dd_ws]; dr = [r for r in dr if r is not None]
            if tr and dr:
                u_disc = (torch.stack(tr).mean(0) - torch.stack(dr).mean(0))
                u_dir = u_disc
                print(f'  DISC: {dt_ws} - {dd_ws} -> |u|={u_dir.norm().item():.3f}',
                      f' (tr={len(tr)} dr={len(dr)})')
        if u_dir is not None:
            u_dir = (u_dir / u_dir.norm()).to(DEV)
            print(f'  dir-target: u_dir norm=1, first dims={u_dir[:3].tolist()}')

    region_ids = set()      # single-token CONCEPT rows, for region-emit
    strict_block_ids = set()   # CONCEPT_BLOCK: exact ids + case/plural/inflection variants
    if TARGET_TYPE == 'dir':
        _roots = []
        if CONCEPT:
            for w in CONCEPT.split():
                sp = tok(' ' + w, add_special_tokens=False).input_ids
                if len(sp) == 1:
                    region_ids.add(int(sp[0]))
            if os.environ.get('CONCEPT_BLOCK') == '1':
                # strict no-insertion: block the concept words in EVERY surface form
                # (case, plural, -ing/-ed, fused). A concept token leaking through
                # as 'Ritual' or 'rituals' would make the 'no insertion' test a lie.
                for w in CONCEPT.split():
                    d = tok.decode(tok(' ' + w, add_special_tokens=False).input_ids[0]).strip().lower()
                    if d and d.isalpha() and len(d) >= 3 and d not in _roots:
                        _roots.append(d)
        if DISC_TARGET and os.environ.get('CONCEPT_BLOCK') == '1':
            # no-insertion for discriminant steering: also block the DISC_TARGET
            # words (every surface form) so the theme in the output is NOT any
            # steering word - it must be the discriminant's semantic reach.
            for w in DISC_TARGET.split():
                sp = tok(' ' + w, add_special_tokens=False).input_ids
                if len(sp) == 1 and CONCEPT_BLOCK:  # not a region for emit (blocked anyway)
                    pass
                d = tok.decode(sp[0]).strip().lower() if sp else ''
                if d and d.isalpha() and len(d) >= 3 and d not in _roots:
                    _roots.append(d)
        if _roots and os.environ.get('CONCEPT_BLOCK') == '1':
            for _w in _roots:
                for _v in [_w, _w + 's', _w + 'es', _w + 'ed', _w + 'ing',
                           _w.capitalize(), _w.capitalize() + 's', _w + 's'.capitalize()]:
                    for _vs in [_v, ' ' + _v]:
                        for _i in tok(_vs, add_special_tokens=False).input_ids:
                            strict_block_ids.add(int(_i))
            strict_block_ids |= region_ids
            print(f'  CONCEPT_BLOCK strict: {len(strict_block_ids)} blocked ids',
                  f' roots={_roots}')
        if BLOCK_WORDS:
            # held-out vocabulary test: strictly block these words (all surface
            # forms) regardless of steering mode, so any theme vocabulary in the
            # output is NOT the steering set itself.
            for w in BLOCK_WORDS.split():
                sp = tok(' ' + w, add_special_tokens=False).input_ids
                d = tok.decode(sp[0]).strip().lower() if sp else ''
                if d and d.isalpha() and len(d) >= 3:
                    for _v in [d, d + 's', d + 'es', d + 'ed', d + 'ing',
                               d.capitalize(), d.capitalize() + 's', d + 's'.capitalize()]:
                        for _vs in [_v, ' ' + _v]:
                            for _i in tok(_vs, add_special_tokens=False).input_ids:
                                strict_block_ids.add(int(_i))
            print(f'  BLOCK_WORDS strict: {len(strict_block_ids)} blocked ids total')

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
        # anti-priming: penalize RECENTLY EMITTED tokens (the real latch driver is
        # repetition priming at natural continuation points, not boosted-token
        # re-sampling - CONTRAST_BLOCK proved that). Subtract REP_PEN from the
        # last REP_WINDOW sampled ids; this kills "ship captain ship captain"
        # without touching the narrative tilt.
        if REP_PEN > 0 and rep_hist:
            L = L.clone()
            recent = rep_hist[-REP_WINDOW:]
            if REP_COUNT:
                # count-scaled anti-priming: a token seen k times inside the
                # window loses k*REP_PEN. Farm's latch is a SMALL CLUSTER of
                # distinct near-synonyms (fence/fences/gate/barn/yard) - a flat
                # per-token penalty suppresses each only once and the cluster
                # keeps reselecting. Accumulating penalty on repeat offenders
                # collapses the whole cluster: the loop driver (repetition
                # priming) pays more each time it recurs.
                rt = torch.tensor(recent, device=L.device)
                uniq, cnts = torch.unique(rt, return_counts=True)
                pen = torch.zeros_like(L)
                pen[uniq] = REP_PEN * cnts.float()
            else:
                pen = torch.zeros_like(L)
                pen[recent] = REP_PEN
            L = L - pen
        if cjk_mode or CJK_EXCLUDE:
            # the CJK pool is a pure variance pump, structurally unreachable in
            # English prose: exclude it from sampling so it can never erupt
            # (T5 failure mode) nor be selected. Synthetic pool, not narrative.
            L = L.clone()
            L[cjk_ids] = -50.0
        if EXAM_EXCLUDE and exam_ids:
            # bar the exam-template marker tokens (see list at build time).
            # exam_ids is a main() local captured by this closure; always bound.
            L = L.clone()
            L[exam_ids] = -50.0
        if strict_block_ids:
            # strict no-insertion proof: the CONCEPT/BLOCK_WORDS words (every
            # surface form: case, plural, inflected, fused) are forbidden.
            # The readout is STILL rotated toward their centroid - only the
            # words themselves can never be sampled.
            # strict no-insertion proof: the CONCEPT words (every surface form: case,
            # plural, inflected, fused) are forbidden. The readout is STILL rotated
            # toward their centroid - only the words themselves can never be sampled.
            L = L.clone()
            L[list(strict_block_ids)] = -50.0
        if BLOCK_BOOSTED and boost_block:
            # no-insertion proof for CONTRAST mode: the top-k boosted tokens
            # (the literal steering target list) can never be sampled. Any
            # theme that still lands is carried by the aggregate direction.
            # (boost_block survives the boosted_set wipe - see CONTRAST build.)
            L = L.clone()
            L[list(boost_block)] = -50.0
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
    rep_hist = []
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
    cb_until = -1
    cb_id = None
    boosted_set = set()
    boost_block = set()   # survives for no-insertion proof (not wiped below)

    for step in range(NTOK):
        w_active = None
        pre_frac = 0.0
        is_pre = False
        if CONTRAST_EMIT and cont_emit_done:
            contrast_steer = False
        else:
            contrast_steer = (CONTRAST_TARGET and step >= SW0 and
                             (CONTRAST_WINDOW == 0 or step < SW0 + CONTRAST_WINDOW))
        if contrast_steer:
            L_nat, v = forward_v(ids)
            if CONTRAST_MODE == 'logit':
                # Additive logit contrast. G_BLEND in (0,1] scales the contrast
                # contribution down (keeps full natural logits); G_RAMP applies
                # a cosine intensity envelope across [SW0, SW0+CW) so the splice
                # eases in and out instead of slamming (the dense-cluster
                # enumeration failure).
                gb = float(os.environ.get('G_BLEND', '0.0'))
                ramp = float(os.environ.get('G_RAMP', '0.0'))
                amp = 1.0
                if ramp > 0 and CONTRAST_WINDOW > 0:
                    tprog = (step - SW0) / CONTRAST_WINDOW
                    amp = 0.5 - 0.5 * math.cos(min(1.0, max(0.0, tprog)) * math.pi)
                if gb <= 0:
                    L = L_nat + ALPHA * amp * dL
                else:
                    L = L_nat + ALPHA * amp * (1 - gb) * dL
                if CONTRAST_BLOCK and step < cb_until and cb_id is not None:
                    L = L.clone()
                    L[cb_id] = -30.0       # break boosted-token latch loops
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
            rep_hist.append(nxt)
            if CONTRAST_EMIT and nxt in boosted_set:
                cont_emit_done = True
                if TRACE:
                    print(f'      [{step}] CONTRAST_EMIT: {tok.decode([nxt])!r} '
                          f'sampled, contrast off')
            if CONTRAST_BLOCK and CONTRAST_MODE == 'logit' and nxt in boosted_set:
                cb_until = step + 1 + CONTRAST_ANTI
                cb_id = nxt
                if TRACE:
                    print(f'      [{step}] BLOCK boosted {tok.decode([nxt])!r} '
                          f'for {CONTRAST_ANTI} steps')
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
            steer_this = (w_active is not None and u_dir is not None
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
            rep_hist.append(nxt)
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
            rep_hist.append(nxt)
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
