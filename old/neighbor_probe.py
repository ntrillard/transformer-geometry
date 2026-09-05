#!/usr/bin/env python3
"""neighbor_probe.py — Semantic vs LEXICAL transport.

The core confound the reviewer flagged: our "transport" held-out words may sit
INSIDE the top-200 coordinate set, so they are directly lexically boosted and
their appearance proves nothing about semantic generalization.

This probe removes that confound:

  - Build the SAME working intervention dL = ALPHA * perz * top200 (the
    mechanism_matrix working vector), norm N_REF.
  - Define three probe classes of SINGLE-TOKEN words, each FILTERED so that the
    token is *NOT* among the boosted top-200 coordinates (direct additive boost
    at that coordinate == 0):

      SEM:  semantic neighbors / generalizations of the target concept whose
            tokens are outside top-200  (e.g. fiend, goblin, ghoul, hag, ...)
      UNR:  frequency-matched, semantically unrelated single-token words
            outside top-200 (control for generic coordinated-lift / normalization).
      LEX:  words explicitly inside the top-200 (positive control: these ARE
            boosted; used only to sanity-check the harness).

  Measure, by class, under the STEERED run (add dd every step after SW0) vs.
  an UNSTEERED control run (no dd) at matched seed:

      rankOf = best (min) rank the class achieves during generation
      emit   = fraction of seeds where >=1 class token appears in output
      dminR  = rank baseline, so we can separate generalization from pure
               lexical forcing.

  KEY INTERPRETATION RULE:
    If SEM tokens (zero direct boost) improve rank / emit substantially more
    than UNR tokens (also zero direct boost), the intervention generalizes
    semantically. If SEM ~ UNR ~ baseline, the intervention is lexical forcing
    of exactly the boosted coordinates.

Run:
  SEEDS=30 NTOK=120 python3 neighbor_probe.py [seed-base]
"""
import os, sys, time, re
from collections import Counter
import torch, transformers

SEEDBASE = int(os.environ.get('SEEDBASE', '0'))
SEEDS   = int(os.environ.get('SEEDS', '30'))
NTOK    = int(os.environ.get('NTOK', '120'))
NUCLEUS = float(os.environ.get('NUCLEUS', '0.9'))
ALPHA   = float(os.environ.get('ALPHA', '2.0'))
SW0     = int(os.environ.get('SW0', '20'))
MODEL   = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
K       = int(os.environ.get('K', '200'))
PROMPT  = os.environ.get('PROMPT', 'The waves crashed gently on the beach')

# Target concept (same fantasy setup as mechanism_matrix default).
TGT = ('A dragon circled the ruined towers of the ancient kingdom|'
       'A knight drew his sword against the fire-breathing beast|'
       "The wizard's spell shattered the castle gates")
NEU = ('The waves crashed gently on the beach|'
       'The sand was cool to the touch|The sun was warm over the water')

# Probe classes. SEM = semantic-adjacent dark-fantasy vocabulary whose tokens we
# expect to generalize (may or may not be boosted); UNR = frequency-matched
# unrelated words. We FILTER later to keep only tokens NOT in the top-K boost.
SEM = ('fiend goblin ghoul hag wraith apparition chimera basilisk '
       'sorcery enchantment incantation hex curse '
       'armored blade hilt lance shield '
       'tyrant dominion tyranny desolation')
UNR = ('table pencil window cereal kitchen bicycle morning winter '
       'door chair river stone leaf bird water bread '
       'walk talk sleep eat read book house garden')

# LEX positive control: words that ARE in top-K (filled programmatically).
LEX_SRC = 'dragon knight wizard beast towers castle sword spell kingdom'

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    norm = model.model.norm
    V = model.lm_head.weight.shape[0]

    def logits_of(s):
        t = tok(s, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        with torch.no_grad():
            return model(t).logits[0, -1, :].float().cpu()

    tgt = [x.strip() for x in TGT.split('|') if x.strip()]
    neu = [x.strip() for x in NEU.split('|') if x.strip()]

    nm = torch.stack([logits_of(s) for s in neu]).mean(0)
    perz_sum = None
    for s in tgt:
        Ls = logits_of(s)
        c = Ls - nm
        c = (c - c.mean()) / (c.std() + 1e-6)
        perz_sum = c if perz_sum is None else perz_sum + c
    perz = perz_sum / max(1, len(tgt))
    perz = (perz - perz.mean()) / (perz.std() + 1e-6)

    mK = torch.zeros(V); mK[perz.argsort(descending=True)[:K]] = 1.0
    dL = ALPHA * perz * mK
    N_REF = dL.norm()
    print(f'  K={K}  N_REF={N_REF:.3f}  toggle_count={int(mK.sum())}', flush=True)

    idxK = set(perz.argsort(descending=True)[:K].tolist())

    def single_token_ids(words):
        ids = {}
        for w in words.split():
            sp = tok(' ' + w, add_special_tokens=False).input_ids
            if len(sp) == 1:
                ids[int(sp[0])] = w
        return ids

    # Positive control = words whose token IS one of the boosted top-K coords.
    # Build from the ACTUAL top-K ids (decode each; keep clean alpha tokens).
    top_ids = perz.argsort(descending=True)[:K].tolist()
    lex = {}
    for tid in top_ids:
        w = tok.decode([int(tid)]).strip()
        if w and re.fullmatch(r'[A-Za-z]+', w):
            lex[int(tid)] = w
    sem = {tid: w for tid, w in single_token_ids(SEM).items() if tid not in idxK}
    unr = {tid: w for tid, w in single_token_ids(UNR).items() if tid not in idxK}

    print(f'  probe counts (filtered to NOT-in-top-{K}):')
    print(f'    LEX (in top-K, +control) : {len(lex)}  {sorted(lex.values())}')
    print(f'    SEM (not boosted, semantic): {len(sem)}  {sorted(sem.values())}')
    print(f'    UNR (not boosted, unrelated): {len(unr)}  {sorted(unr.values())}', flush=True)

    def run(add_dd, seed):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        past = None; out_ids = []
        best = {'lex': 10**9, 'sem': 10**9, 'unr': 10**9}
        dd = add_dd.to(DEV) if add_dd is not None else None
        with torch.no_grad():
            for step in range(NTOK):
                vc = {}
                hk = norm.register_forward_hook(lambda m, i, o: vc.__setitem__('o', o[0, -1, :].float().clone()))
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                hk.remove()
                if past is None: past = out.past_key_values
                L1 = out.logits[0, -1, :].float()
                if dd is not None and step >= SW0:
                    L1 = L1 + dd
                order1 = L1.argsort(descending=True).tolist()
                pos1 = {tid: k for k, tid in enumerate(order1)}
                if lex: best['lex'] = min(best['lex'], min(pos1[i] for i in lex))
                if sem: best['sem'] = min(best['sem'], min(pos1[i] for i in sem))
                if unr: best['unr'] = min(best['unr'], min(pos1[i] for i in unr))
                p = torch.softmax(L1, 0)
                q = p.clone(); ooo = q.argsort(descending=True)
                kk = int((q[ooo].cumsum(0) <= NUCLEUS).sum()) + 1
                msk = torch.zeros_like(q); msk[ooo[:kk]] = 1
                qq = (q * msk); qq = qq / qq.sum()
                nxt = int(torch.multinomial(qq, 1))
                if nxt == int(tok.eos_token_id): break
                out_ids.append(nxt)
                ids = torch.tensor([[nxt]], device=DEV)
        txt = tok.decode(out_ids)
        low = txt.lower()
        def emitted(d):
            return sum(1 for tid, w in d.items() if re.search(r'\b' + re.escape(w) + r'\w*', low))
        return dict(txt=txt, best=best,
                    emit_lex=emitted(lex), emit_sem=emitted(sem), emit_unr=emitted(unr))

    for label, dd in [('UNSTEERED (no dd)', None), ('STEERED (working dL)', dL)]:
        R = {'lex': 0, 'sem': 0, 'unr': 0}
        mbest = {'lex': [], 'sem': [], 'unr': []}
        text_emitters = 0
        for s in range(SEEDS):
            m = run(dd, SEEDBASE + s)
            for c in ('lex', 'sem', 'unr'):
                if m['emit_' + c] > 0: R[c] += 1
                mbest[c].append(m['best'][c])
            # semantic-emission coherence: count seeds that emit a SEM token
            if m['emit_sem'] > 0: text_emitters += 1
        print(f'\n  == {label} (SEEDS={SEEDS}) ==', flush=True)
        print(f'  {"class":>5}  {"seeds_emit":>11}  {"median_bestRank":>15}')
        for c in ('lex', 'sem', 'unr'):
            med = sorted(mbest[c])[SEEDS // 2] if mbest[c] else -1
            print(f'  {c:>5}  {R[c]:3d}/{SEEDS:<6}  {med:16d}', flush=True)
    print(f'\n[{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
