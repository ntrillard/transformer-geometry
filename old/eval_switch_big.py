#!/usr/bin/env python3
"""eval_switch_big.py - sequential topic-switching generation.

Controller (SOFT): at each switch step a calibrated graft rotates the
readout toward the closest family member's word so that word lands; between
switches, if the model drifts off-topic (low family support / low alignment)
we accumulate a small rotation toward the family centroid and release once
the model emits topic on its own. A short post-plant de-repeat window blocks
the family so the model writes ABOUT the topic instead of parroting the
planted word. Template/refusal escape-valve tokens are dropped in the first
steps after a switch (SW_META). Every run is paired with a FREE baseline
(no hooks) to compare steering against the model's own continuation.

Run: HF_TOKEN=<tok> SW_SOFT=1 SW_META=1 python3 eval_switch_big.py \
     Qwen/Qwen2-1.5B "We were in the city..."
"""
import csv
import itertools
import math
import os
import sys
import time
from pathlib import Path

import torch
import transformers

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2-1.5B'
PROMPT = (sys.argv[2] if len(sys.argv) > 2
          else 'We were in the city...')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NTOK = 72
SEEDS = [0, 1]
ANGLE = 10.0

ONLINE_CALIB = os.environ.get('SW_ONLINE') == '1'
CALIB_SWEEP = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
CALIB_MARGIN = 2.0

WRAP = os.environ.get('WRAP', 'before').lower()
PEN = 0.3                      # repetition penalty at decode
SW_META = os.environ.get('SW_META') == '1'
TRACE = os.environ.get('TRACE') == '1'

# soft-herding knobs
SOFT_ACC_START = float(os.environ.get('SOFT_ACC_START', '6.0'))
# NEIGHBORHOOD steering: drift herding aims at the CURRENTLY CLOSEST
# family member (its word direction), not the family centroid. This keeps
# the story inside one word's local neighborhood and gives the graft a
# concrete lexical anchor instead of a semantic blur.
SOFT_NEIGHBOR = os.environ.get('SOFT_NEIGHBOR') == '1'
SOFT_ACC_STEP = float(os.environ.get('SOFT_ACC_STEP', '2.0'))
SOFT_ACC_MAX = float(os.environ.get('SOFT_ACC_MAX', '20.0'))
SOFT_TARGET = float(os.environ.get('SOFT_TARGET', '0.02'))       # support release
SOFT_TARGET_ALIGN = float(os.environ.get('SOFT_TARGET_ALIGN', '0.06'))
SOFT_HYST = float(os.environ.get('SOFT_HYST', '0.01'))
SOFT_PLANT_ANTI = int(float(os.environ.get('SOFT_PLANT_ANTI', '5')))
SOFT_ALIGN_SKIP = float(os.environ.get('SOFT_ALIGN_SKIP', '0.12'))
SOFT_SUPPORT_SKIP = float(os.environ.get('SOFT_SUPPORT_SKIP', '0.04'))

# template/refusal/QA escape valves dropped in the first steps after a switch
HIJACK_SUBSTRINGS = (
    '答案', 'a.', 'b.', 'c.', 'd.', 'option', 'which', 'does it',
    'choose', 'select', 'true or false', 'fill in', 'i am not',
    'i do not', 'not capable', 'preference', 'your audience',
    'explain', 'describe', 'question', 'answer', 'survey',
    'verify', 'investigate', 'note', 'summary', 'follow',
    'blank', '____', 'correct', 'wrong',
)

import json as _json
_FAM_ENV = os.environ.get('SW_FAMILIES')
FAMILIES = (_json.loads(_FAM_ENV) if _FAM_ENV else {
    'city':   ['paris', 'london', 'berlin', 'madrid', 'oslo'],
    'animal': ['cat', 'dog', 'bird', 'bear', 'horse', 'polar bear'],
    'food':   ['pizza', 'sushi', 'pasta', 'burger', 'sushi bar'],
    'nature': ['forest', 'rice', 'water', 'sun', 'tree'],
})
_ORD_ENV = os.environ.get('SW_ORDER')
SWITCHES = ({i * 16: f for i, f in enumerate(_ORD_ENV.split(','))}
            if _ORD_ENV else
            {0: 'city', 16: 'animal', 32: 'food', 48: 'nature'})
SEG_N = 16
_stop = ''.join(c for c in PROMPT[:20] if c.isalnum())
OUT = Path('steering-evals/steering_geometry_results/switch_big_'
           f'{MODEL.split("/")[-1]}_{_stop}.csv')


def _lm_head_fp16(model):
    w = model.lm_head.weight.detach()
    if hasattr(w, 'quant_state') and w.quant_state is not None:
        import bitsandbytes as bnb
        try:
            qs = w.quant_state.cpu()
        except Exception:
            qs = w.quant_state
        return bnb.functional.dequantize_4bit(w.data.cpu(), qs).float().cpu()
    return w.cpu().float()


def readout_model(model):
    if hasattr(model.model, 'norm'):
        return model.model
    lm = getattr(model.model, 'language_model', None)
    if lm is not None and hasattr(lm, 'norm'):
        return lm
    raise RuntimeError(f'{MODEL}: no readout norm found '
                       f'(model.model.norm / language_model.norm)')


def rep4(toks):
    if len(toks) < 4:
        return 0.0
    n4 = [tuple(toks[i:i+4]) for i in range(len(toks)-3)]
    return sum(1 for i in range(len(toks)-3) if n4[i] in n4[i+1:]) / (len(toks)-3)


def quality(toks, txt):
    """text metrics - dict of scores."""
    words = [w.lower() for w in txt.split() if w]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    mr = max((sum(1 for _ in grp) for _, grp in
              itertools.groupby(toks)), default=0)
    return dict(
        eos='<eos>' in txt,
        rep4=round(rep4(toks), 3),
        max_run=mr,
        max_wfreq=max(freq.values()) if freq else 0,
        n_words=len(words),
        good=(rep4(toks) == 0.0 and mr <= 2
              and (max(freq.values()) if freq else 0) <= 2),
    )


def main():
    t0 = time.time()
    print(f'\nLoading {MODEL} (bf16, no quant) ...')
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    eos_id = int(tok.eos_token_id)
    RO_norm = readout_model(model).norm
    W = _lm_head_fp16(model)
    Wn = (W / W.norm(dim=1, keepdim=True)).float()

    # members: name -> (ids, dir); family ids + centroid direction
    members = {}
    famids = {}
    for fam, words in FAMILIES.items():
        mem = []
        for w in words:
            # steep word must stay ONE token: bare token when it is single,
            # else leading-space single token (sushi -> ' sushi', cheese -> ' o" cheese')
            wrap = {'none': '{}', 'before': ' {}', 'after': '{} ',
                    'both': ' {} '}[WRAP]
            if WRAP == 'none':
                ids = tok(w, add_special_tokens=False).input_ids
                if len(ids) != 1:
                    ids = tok(' ' + w, add_special_tokens=False).input_ids
            else:
                ids = tok(wrap.format(w), add_special_tokens=False).input_ids
            ids = [int(i) for i in ids]
            d = Wn[ids].float().sum(0)
            d = d / d.norm()
            mem.append((w, ids, d))
        members[fam] = mem
        famids[fam] = [i for _, ids, _ in mem for i in ids]
        print(f"  family {fam:>6}: "
              + ', '.join(f'{w}({len(ids)})' for w, ids, _ in mem))
    fam_dir = {f: (torch.stack([m[2] for m in memb]).sum(0)
                   / torch.stack([m[2] for m in memb]).sum(0).norm()).to(DEV)
               for f, memb in members.items()}

    def closest_member(vv, fam):
        u = vv / vv.norm()
        best = None
        for w, ids, d in members[fam]:
            s = float(d.to(DEV) @ u)
            if best is None or s > best[0]:
                best = (s, w, ids)
        return best[1], best[2], best[2][0]

    def rot_toward(vv, goal, theta):
        goal = goal.to(vv.device)
        a = math.radians(theta)
        v1 = vv / vv.norm()
        g0 = goal - (v1 @ goal) * v1
        gn = g0 / (g0.norm() + 1e-12)
        return (v1 * math.cos(a) + gn * math.sin(a)) * vv.norm()

    def rot_to_angle(vv, tid, theta):
        return rot_toward(vv, Wn[tid].float().to(DEV), theta)

    def sample(L, prefix, block_words=None, extra_zero=None):
        L = torch.nan_to_num(L.float(), nan=-50.0).clamp(-50.0, 50.0)
        p = torch.softmax(L, 0)
        q = p.clone()
        order = q.argsort(descending=True)
        k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
        msk = torch.zeros_like(q)
        msk[order[:k]] = 1
        qq = q * msk
        # substring anti: drop top candidates containing planted word forms
        if block_words:
            top = order[:200].tolist()
            dec = tok.batch_decode([[i] for i in top])
            drop = [i for i, s in zip(top, dec)
                    if any(w in s.lower() for w in block_words)]
            for i in drop:
                qq[i] = 0.0
        # meta hijack-zero: drop template/refusal/QA escape valves
        if extra_zero:
            for i in extra_zero:
                qq[i] = 0.0
        for t in set(prefix):
            c = prefix.count(t)
            if c:
                qq[t] = qq[t] * (PEN ** c)
        tot = qq.sum()
        if tot <= 0 or not torch.isfinite(tot):
            qq = torch.ones_like(qq)
        qq = qq / qq.sum()
        return int(torch.multinomial(qq, 1))

    def forward(ids, inj_p=None, anti_ids=None):
        hs = []
        try:
            if inj_p is not None:
                def inj(m, i, o, p=inj_p):
                    o[0, -1, :] = torch.as_tensor(p, dtype=o.dtype,
                                                  device=o.device)
                hs.append(RO_norm.register_forward_hook(inj))
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
        """logits + hidden state at the last position, one forward pass."""
        vc = {}
        hk = RO_norm.register_forward_hook(
            lambda m, i, o: vc.__setitem__('v', o[0, -1, :].float()))
        try:
            with torch.no_grad():
                L = model(ids).logits[0, -1].float()
        finally:
            hk.remove()
        return L, vc['v']

    def best_angle(ids, vv, tid):
        """in-situ recalibration: min angle making tid rank-1, +margin."""
        for th in CALIB_SWEEP:
            vp = rot_to_angle(vv, tid, th)
            L = forward(ids, inj_p=vp)
            if (int(L.argmax()) == tid
                    and torch.isfinite(L[tid])
                    and float(L[tid]) > float(L.max()) - 0.001):
                return th + CALIB_MARGIN
        return CALIB_SWEEP[-1]

    def hijack_ids(L):
        """live meta-probe: template/refusal/QA tokens in top-64."""
        if not SW_META:
            return []
        top = L.argsort(descending=True)[:64].tolist()
        dec = tok.batch_decode([[i] for i in top])
        hit = [i for i, s in zip(top, dec)
               if any(sub in s.lower() for sub in HIJACK_SUBSTRINGS)]
        return hit[:8]

    def run_schedule(sd, free=False):
        torch.manual_seed(sd)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(DEV)
        sampled = []
        cur_fam = None
        acc = 0.0
        recent = []
        plant_until = -1
        plant_fam = None
        plant_word = None
        align_hist = []
        corr = 0
        n_seg_tok = 0
        hj = None

        for step in range(NTOK):
            if free:
                nxt = sample(forward(ids), sampled)
                if nxt == eos_id:
                    sampled.append(nxt); break
                sampled.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
                continue

            L, v = forward_v(ids)
            anti_a = (famids[plant_fam] if (step < plant_until and plant_fam)
                      else None)
            bw = ({w for w in [plant_word]} if anti_a else None)
            if anti_a:
                L = forward(ids, anti_ids=anti_a)
            if step in SWITCHES:
                cur_fam = SWITCHES[step]
                hj = hijack_ids(L)
                name, ids_m, tgt = closest_member(v, cur_fam)
                u = v / v.norm()
                align = float(u @ fam_dir[cur_fam])
                ps = torch.softmax(L, 0)
                support = float(ps[famids[cur_fam]].sum())
                if (support >= SOFT_SUPPORT_SKIP or align >= SOFT_ALIGN_SKIP):
                    acc = 0.0
                    if TRACE:
                        print(f"      skip@{step} {cur_fam} ALREADY-ON")
                else:
                    th = best_angle(ids, v, tgt) if ONLINE_CALIB else ANGLE
                    gain = max(0.15,
                               1.0 - max(support / SOFT_SUPPORT_SKIP,
                                         abs(align) / SOFT_ALIGN_SKIP))
                    th = max(th * gain, 0.5)
                    L = forward(ids, inj_p=rot_to_angle(v, tgt, th))
                    corr += 1
                    acc = SOFT_ACC_START
                    plant_fam = cur_fam
                    plant_word = name
                    plant_until = step + 1 + SOFT_PLANT_ANTI
                    if TRACE:
                        print(f"      switch@{step} {cur_fam}->{name} "
                              f"supp={support:.4f} align={align:.3f} th={th:.0f}")
                recent = []
                n_seg_tok = 0
            else:
                p = torch.softmax(L, 0)
                support = float(p[famids[cur_fam]].sum())
                hj = hijack_ids(L)
                u = v / v.norm()
                align = float(u @ fam_dir[cur_fam])
                align_hist.append(align)
                n_seg_tok += 1
                recent.append(nxt)
                if len(recent) > 5:
                    recent.pop(0)
                rolling_topic = sum(1 for t in recent if t in famids[cur_fam])
                if (support >= SOFT_TARGET
                        or (n_seg_tok > 3 and rolling_topic >= 2)):
                    acc = 0.0
                    if TRACE:
                        print(f"      off@{step} {cur_fam} supp={support:.4f} RELEASE")
                elif align < SOFT_TARGET_ALIGN - SOFT_HYST:
                    acc = min(acc + SOFT_ACC_STEP, SOFT_ACC_MAX)
                    if SOFT_NEIGHBOR:
                        _, nids, _ = closest_member(v, cur_fam)
                        goal = Wn[nids[0]].float().to(DEV)
                    else:
                        goal = fam_dir[cur_fam]
                    L = forward(ids, inj_p=rot_toward(v, goal, acc),
                                anti_ids=anti_a if anti_a else None)
                    corr += 1
                    if TRACE:
                        print(f"      drift@{step} {cur_fam} supp={support:.4f} "
                              f"align={align:.3f} acc={acc:.0f}")

            nxt = sample(L, sampled, block_words=bw, extra_zero=hj)
            if nxt == eos_id:
                sampled.append(nxt); break
            sampled.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)

        mean_align = (float(sum(align_hist) / max(len(align_hist), 1))
                      if align_hist else float('nan'))
        return sampled, mean_align, corr

    steps = sorted(SWITCHES)
    print(f"\n[{MODEL}] SWITCH {PROMPT!r} NTOK={NTOK} PEN={PEN} "
          f"WRAP={WRAP} META={SW_META}\n")

    # prompt hidden-state alignment with each family
    ids0 = tok(PROMPT, add_special_tokens=False,
               return_tensors='pt').input_ids.to(DEV)
    _, v0 = forward_v(ids0)
    u0 = v0 / v0.norm()
    al = ', '.join(f"{f}:{float(u0 @ fam_dir[f]):.2f}" for f in FAMILIES)
    print(f"  prompt alignments: {al}")

    rows = []
    for sd in SEEDS:
        toks_s, mean_align, corr = run_schedule(sd, free=False)
        txt_s = tok.decode(toks_s)
        qs = quality(toks_s, txt_s)
        print(f"\n  seed {sd} STEERED  q={qs['good']}  "
              f"(rep4={qs['rep4']} run={qs['max_run']} "
              f"wfreq={qs['max_wfreq']} eos={qs['eos']})")
        print(f"    SOFT: mean family-align={mean_align:.3f} "
              f"correction-steps={corr}")

        # segment-level: hijack-clean + topic-follow
        seg_hijack = []
        seg_follow = []
        for i, st in enumerate(steps):
            if i * SEG_N >= len(toks_s):
                break
            fam = SWITCHES[st]
            seg = toks_s[i * SEG_N + 1:(i + 1) * SEG_N]
            seg_txt = tok.decode(seg).lower()
            seg_hijack.append(any(s in seg_txt for s in HIJACK_SUBSTRINGS))
            seg_follow.append(any(t in seg for t in famids[fam]))
        print(f"    seg hijack={sum(seg_hijack)}/{len(seg_hijack)}  "
              f"topic-follow={sum(seg_follow)}/{len(seg_follow)}")

        toks_f, _, _ = run_schedule(sd, free=True)
        txt_f = tok.decode(toks_f)
        qf = quality(toks_f, txt_f)
        print(f"  seed {sd} FREE     q={qf['good']}  "
              f"(rep4={qf['rep4']} run={qf['max_run']} "
              f"wfreq={qf['max_wfreq']} eos={qf['eos']})")
        qd = sum(qs[k] for k in ('rep4', 'max_run', 'max_wfreq')) \
             - sum(qf[k] for k in ('rep4', 'max_run', 'max_wfreq'))
        print(f"    steer-vs-free delta: {qd:+.2f}")
        print(f"    FULL: {PROMPT} {txt_s}")
        print(f"    FREE: {PROMPT} {txt_f}")
        rows.append(dict(seed=sd, full=txt_s, free_full=txt_f,
                         steered_good=int(qs['good']),
                         free_good=int(qf['good'])))

    ng = sum(1 for r in rows if r['steered_good'])
    nfg = sum(1 for r in rows if r['free_good'])
    print(f"\n  quality-good: STEERED {ng}/{len(SEEDS)}  FREE {nfg}/{len(SEEDS)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {OUT}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()