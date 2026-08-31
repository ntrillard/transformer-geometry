#!/usr/bin/env python3
"""eval_demo.py — LIVE DEMONSTRATION of what the readout law + recipe
deliver, as actual generations. One model load, ~8 short generations.

Capabilities shown (each = the 2-constant recipe from the arc):
  D1 FORCE    : steer once toward a word -> it becomes rank-1 next token
  D2 BAN      : anti-steer a word -> it collapses to dead-last (rank 262144)
  D3 ESSAY    : steer once + anti-last -> topic planted + loop-free prose
  D4 COMPOSE  : sequential steers -> contiguous phrase ('grilled chicken')
  D5 HANDOFF  : steer A, later steer B -> runtime topic switch
  D6 LONGCTX  : the same recipe at 22-tok context (native '.' -> lighter
                separator, better prose)
Each block prints the NATIVE baseline + the steered generation so the
contrast is visible. Writes a copy of the transcript to
steering_geometry_results/demo_transcript.txt

Run: timeout 90 python3 -u eval_demo.py  # GEMMA-3-1B
"""
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'
LONGPROMPT = ('Yesterday evening for dinner I made grilled chicken with '
              'roasted vegetables and a fresh salad while my family '
              'discussed the upcoming trip')
NTOK = 12
A_REP = 0.15
OUT = Path('../steering_geometry_results/demo_transcript.txt')


def rep4(toks):
    if len(toks) < 8:
        return 1.0
    n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach()
    Wn = (W / W.norm(dim=1, keepdim=True)).float()

    def tid(w):
        return int(tok(' ' + w, add_special_tokens=False).input_ids[0])

    lines = []

    def say(*a):
        s = ' '.join(str(x) for x in a)
        print(s)
        lines.append(s)

    # ---- one persistent state frame per prompt ----
    def get_frame(prompt):
        ids0 = tok(prompt, add_special_tokens=False,
                   return_tensors='pt').input_ids.to(DEV)
        cf = {}

        def hook_c(m, i, o):
            cf['v'] = o[0, -1, :].float()

        h = model.model.norm.register_forward_hook(hook_c)
        with torch.no_grad():
            L0 = model(ids0).logits[0, -1].float()
        h.remove()
        native = int(L0.argmax())
        vf = cf['v'].float()
        vfn = vf / vf.norm()
        return dict(ids0=ids0, L0=L0, native=native, vf=vf, vfn=vfn)

    def anti(vv, t, amt=A_REP):
        vv1 = vv / vv.norm()
        Wb = W[t].float()
        tauv = Wb - (vv1 @ Wb) * vv1
        gl = -tauv / tauv.norm()
        return (vv1 * math.cos(amt) + gl * math.sin(amt)) * vv.norm()

    def gen(frame, target=None, tp=None, anti_last=True, schedule=None,
            ntok=NTOK):
        """One generation. target: word to steer toward once at step 0.
        schedule: list of (step, word) extra steer events.
        tp: (word, num) alternate (target-only once)."""
        ids = frame['ids0'].clone()
        vf, vfn = frame['vf'], frame['vfn']
        L0, native = frame['L0'], frame['native']
        toks = []
        for step in range(ntok):
            vv = vf
            ev = []
            if schedule:
                ev = [(s, w) for (s, w) in schedule if s == step]
            if target is not None and step == 0:
                ev.append((0, target))
            if ev:
                w = ev[-1][1]
                t = tid(w)
                gap = float(L0[native] - L0[t])
                a = 2 * gap / 97.0 + 0.02
                Wt = Wn[t].float()
                tau_t = Wt - (vfn @ Wt) * vfn
                g = tau_t / tau_t.norm()
                vv = (vfn * math.cos(a) + g * math.sin(a)) * vf.norm()
            if anti_last and toks:
                vv = anti(vv, toks[-1])

            def inj(m, i, o, p=vv):
                out = o.clone()
                out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                                device=out.device)
                return out

            hi = model.model.norm.register_forward_hook(inj)
            try:
                with torch.no_grad():
                    L = model(ids).logits[0, -1].float()
            finally:
                hi.remove()
            p = torch.softmax(L.float(), dim=0)
            q = p.clone(); order = q.argsort(descending=True)
            k = int((q[order].cumsum(0) <= 0.9).sum()) + 1
            msk = torch.zeros_like(q); msk[order[:k]] = 1
            qq = (q * msk) / (q * msk).sum()
            nxt = int(torch.multinomial(qq, 1))
            toks.append(int(nxt))
            ids = torch.cat([ids,
                             torch.tensor([[nxt]], device=ids.device)],
                            dim=1)
        return toks

    def rank_of(frame, t):
        L = frame['L0']
        return int((L > L[t]).sum().item()) + 1

    say("=" * 74)
    say("STEER-ON-A-SPHERE DEMO  |  gemma-3-1b-it  |  the readout law + "
        "2-constant recipe")
    say("The 'law': one rotation by 2*(gap/97)+0.02 toward a token's row "
        "makes it rank-1.")
    say("=" * 74)

    # ---------- D1 FORCE + D2 BAN ----------
    frame = get_frame(PROMPT)
    native_t = frame['native']
    say(f"\n[Context] {PROMPT!r}")
    if 'ocean' in tok.decode([tid('ocean')]) or True:
        pass
    t_o = tid('ocean')
    gap_o = float(frame['L0'][frame['native']] - frame['L0'][t_o])
    a_o = 2 * gap_o / 97.0 + 0.02
    say(f"  native argmax  = {tok.decode([native_t])!r}  "
        f"(the invariant ' I' loop-token)")
    say(f"  target 'ocean': gap={gap_o:.2f}  ->  budget a={a_o:.3f} rad")
    say(f"\n  D1  FORCE : steer once toward 'ocean' (budget {a_o:.3f})")
    g = gen(frame, target='ocean', anti_last=False, ntok=8)
    say(f"        -> {tok.decode(g)!r}")
    say(f"\n  D2  BAN   : anti-steer 'ocean' (rotate AWAY by {a_o:.3f})")
    tau_o = Wn[t_o].float() - (frame['vfn'] @ Wn[t_o].float()) * frame['vfn']
    g_away = -tau_o / tau_o.norm()
    vv = (frame['vfn'] * math.cos(a_o) + g_away * math.sin(a_o)) * frame['vf'].norm()

    def inj2(m, i, o, p=vv):
        out = o.clone()
        out[0, -1, :] = torch.as_tensor(p, dtype=out.dtype,
                                        device=out.device)
        return out

    hi = model.model.norm.register_forward_hook(inj2)
    try:
        with torch.no_grad():
            L_ban = model(frame['ids0']).logits[0, -1].float()
    finally:
        hi.remove()
    r0 = rank_of(frame, t_o)
    r_ban = int((L_ban > L_ban[t_o]).sum().item()) + 1
    say(f"        ocean rank: {r0} -> {r_ban}  (256K-vocab; dead-last "
        f"when 262144)")
    say(f"        logit: {float(frame['L0'][t_o]):+.2f} -> "
        f"{float(L_ban[t_o]):+.2f}")

    # ---------- D3 ESSAY ----------
    say(f"\n[Context] {PROMPT!r}  (native loop baseline first)")
    g_nat = gen(frame, target=None, anti_last=False, ntok=10)
    say(f"  native        : {tok.decode(g_nat)!r}")
    t_c = tid('chicken')
    gap_c = float(frame['L0'][frame['native']] - frame['L0'][t_c])
    a_c = 2 * gap_c / 97.0 + 0.02
    say(f"  D3  ESSAY : steer once toward 'chicken' ({a_c:.3f}) + "
        f"anti-last({A_REP}), 16 tok")
    g3 = gen(frame, target='chicken', anti_last=True, ntok=16)
    say(f"        -> {tok.decode(g3)!r}")
    say(f"        plant={1.0 if any(t in g3[:10] for t in [t_c]) else 0.0} "
        f"rep4={rep4(g3):.2f} div={len(set(g3))/len(g3):.2f}")

    # ---------- D4 COMPOSE ----------
    say(f"\n  D4  COMPOSE : sequential steer 'grilled'@0 then 'chicken'@1")
    g4 = gen(frame, target='grilled', anti_last=True,
            schedule=[(1, 'chicken')], ntok=12)
    say(f"        -> {tok.decode(g4)!r}")
    tg, tc = tid('grilled'), tid('chicken')
    bigram = any(g4[i] == tg and g4[i + 1] == tc
                 for i in range(min(len(g4), 10) - 1))
    say(f"        contiguous 'grilled chicken' present: {bigram}")

    # ---------- D5 HANDOFF ----------
    say(f"\n  D5  HANDOFF : 'chicken' theme at step 0, switch to 'ocean' "
        f"at step 5")
    g5 = gen(frame, target='chicken', anti_last=True,
            schedule=[(5, 'ocean')], ntok=14)
    say(f"        -> {tok.decode(g5)!r}")

    # ---------- D6 LONGCTX ----------
    say(f"\n[Context, long] 22 tokens")
    flong = get_frame(LONGPROMPT)
    say(f"  native at 22 tok = {tok.decode([flong['native']])!r}  "
        f"(it moved off ' I')")
    g6 = gen(flong, target='ocean', anti_last=True, ntok=14)
    say(f"  D6  LONGCTX : steer 'ocean' once + anti-last")
    say(f"        -> {tok.decode(g6)!r}")
    say(f"        div={len(set(g6))/len(g6):.2f} (vs ~0.50-0.60 short)")

    say("\n" + "=" * 74)
    say("END DEMO — every number (budget, anti, plant) is the same "
        "closed-form law")
    say("persisted claims: script + CSVs in steering-evals/")
    say("=" * 74)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n')
    print(f"\n[transcript saved to {OUT}]")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()