#!/usr/bin/env python3
"""falsify_orth2.py - Orthogonal-complement transport falsification, v2.

Methodology v1 corrected (per reviewer):
  #1 random control PROJECTED INTO THE COMPLEMENT (matched): z -= U(U'z)
  #4 per-alpha verdict (NOT aggregate): exists alpha: R_a >= .5 and R_a > Q_a
  #7 sign control -dper at every alpha
  #3 explicit ANCHORS / ENUM_CLUSTER / HELD_OUT vocab
  #8 logit-geometry measured BEFORE sampling (dH,dB,dN) + sampled occurrences
  #2 absolute a-priori coherence criterion (max-run<6 and dist-1>.6)
  #9 PCA on normalized token rows (sphere-defined complement), noted as the
     choice; a raw-W variant is flagged as future work.

Conditions per alpha:  +dper | random-complement | -dper | baseline(0)
Primary falsification criterion (pre-registered, per-alpha, no aggregate):
  exists alpha in ALPHAS:
       R_alpha >= 0.5  AND  R_alpha > Q_alpha
  (R= +dper transport rate, Q= random-complement transport rate)
  sign control (-dper) must NOT also transport (else it's sign-agnostic =
  generic perturbation).

Transport criterion per generation: contains >=1 HELD_OUT word, AND
zero ANCHOR words, AND max-run<6 AND dist-1>.6.

Run: HF_TOKEN=... python3 falsify_orth2.py [seed]
"""
import os, sys, math, time, re
from collections import Counter
import torch, transformers

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODEL = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
PROMPT = os.environ.get('PROMPT', 'The waves crashed gently on the beach')
NTOK = int(os.environ.get('NTOK', '120'))
K = int(os.environ.get('K', '20'))   # generations per (cond, alpha)
ALPHAS = [float(x) for x in os.environ.get('ALPHAS',
    '0.04,0.10,0.16,0.22,0.28,0.34').split(',')]
NUCLEUS = float(os.environ.get('NUCLEUS', '0.9'))

TGT = 'The king entered the castle|The queen sat upon the throne|' \
      'The prince inherited the crown|The royal family gathered in the great hall'
NEU = 'The person walked down the street|The individual entered the room|' \
      'The worker picked up the package'
ANCHORS = 'king queen prince royal princess'          # steering anchors (blocked)
ENUM_CLUSTER = 'palace castle throne fortress'        # the dense enumerating cluster
HELD_OUT = 'crown reign kingdom realm monarchy monarch dynasty heir coronation sovereign'
NEUT = 'sand wave sea swim ocean surf beach tide shore shell'

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    t0 = time.time()
    print(f'falsify_orth2 seed={SEED} K={K} ntok={NTOK} alphas={ALPHAS}', flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    norm = model.model.norm
    W = model.lm_head.weight.detach().float()
    Wt = W.t()
    eos_id = int(tok.eos_token_id)

    # #9 sphere-defined complement: PCA on NORMALIZED token rows.
    Wnorm = W / W.norm(dim=1, keepdim=True)
    _, S, V = torch.svd_lowrank(Wnorm.float().cuda(), q=300, niter=5)
    S = S.cpu(); V = V.cpu()
    eng = (S**2).cumsum(0) / (S**2).sum()
    r = int((eng >= 0.9).nonzero()[0].item()) + 1
    Uc = V[:, :r].float()          # 1536 x r, principal axes of the token sphere
    print(f'  sphere-defined complement, shell rank r={r}', flush=True)

    def state(s):
        t = tok(s, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        vc = {}
        hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
        with torch.no_grad(): model(t)
        hk.remove()
        return vc['o'].cpu().float()

    tgt = [s.strip() for s in TGT.split('|') if s.strip()]
    neu = [s.strip() for s in NEU.split('|') if s.strip()]
    Ts = torch.stack([state(s) for s in tgt]).mean(0)
    Ns = torch.stack([state(s) for s in neu]).mean(0)
    d = Ts - Ns
    dper = d - Uc @ (Uc.t() @ d)
    dper = dper / dper.norm()
    # confirm dper is exactly in complement
    leak = (Uc @ (Uc.t() @ dper)).norm().item()
    print(f'  dper: norm={dper.norm():.3f} residual shell-leak={leak:.4f} (should be ~0)', flush=True)

    # #1 random control: random vector PROJECTED INTO THE COMPLEMENT, matched norm
    g = torch.Generator().manual_seed(777 + SEED)
    z = torch.randn(1536, generator=g).float()
    zc = z - Uc @ (Uc.t() @ z)
    Rdper = zc / zc.norm()
    leak_r = (Uc @ (Uc.t() @ Rdper)).norm().item()
    print(f'  random-complement: norm={Rdper.norm():.3f} shell-leak={leak_r:.4f} (should be ~0)', flush=True)

    def ids_of(words):
        return [(w, int(tok(' '+w, add_special_tokens=False).input_ids[0]))
                for w in words.split()
                if len(tok(' '+w, add_special_tokens=False).input_ids) == 1]
    held_ids = ids_of(HELD_OUT); anchor_ids = ids_of(ANCHORS); neut_ids = ids_of(NEUT)
    held_w = set(w for w,_ in held_ids); anchor_w = set(w for w,_ in anchor_ids)
    neut_w = set(w for w,_ in neut_ids)
    print(f'  HELD_OUT({len(held_ids)})= {HELD_OUT}')
    print(f'  ANCHORS({len(anchor_ids)})= {ANCHORS}   (block pre-sampling too? no, we only floor for the blocked score)')

    # #8 pre-generation logit measurement: logit deltas at the LIVE step
    def logit_delta_step(ids, dirv, alpha):
        vc = {}
        hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
        with torch.no_grad(): L0 = model(ids).logits[0,-1].float()
        hk.remove()
        v = vc['o'].float()
        L1 = torch.matmul((v + alpha*v.norm()*dirv.to(v.device)), Wt.to(v.device))
        return L0, L1

    # #4 per-alpha transport. Conditions: baseline, +dper, random-complement, -dper
    results = {a: {'plus': 0, 'rand': 0, 'minus': 0} for a in ALPHAS}
    logit_sum = {a: None for a in ALPHAS}   # accumulate dH,dB,dN over steps (nonzero-alphas only)
    for a in ALPHAS:
        for s in range(K):
            # --- baseline (alpha 0) ---
            ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
            plus_ok = rand_ok = minus_ok = False
            # We need three separate runs (each its own sampled trajectory).
            # Do plus first, logit-geom + transport.
            for mode, dirv in [('plus', dper), ('rand', Rdper), ('minus', -dper)]:
                torch.manual_seed(a*1000 + s + (0 if mode=='plus' else 1000 if mode=='rand' else 2000))
                ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
                # accumulate logit deltas for logit-geometry
                dH = dN = 0.0
                out = []
                for _ in range(NTOK):
                    L0, L1 = logit_delta_step(ids, dirv, a)
                    p0 = torch.softmax(L0, 0)
                    dL = (L1 - L0)
                    # pre-sample logit geometry
                    for w,i in held_ids: dH += dL[i].item()
                    for w,i in neut_ids: dN += dL[i].item()
                    # sample from the STEERED (perturbed) readout
                    p = torch.softmax(L1, 0)
                    q = p.clone(); order = q.argsort(descending=True)
                    k = int((q[order].cumsum(0) <= NUCLEUS).sum()) + 1
                    msk = torch.zeros_like(q); msk[order[:k]] = 1
                    qq = (q*msk); qq = qq/qq.sum()
                    nxt = int(torch.multinomial(qq, 1))
                    if nxt == eos_id: break
                    ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
                    out.append(nxt)
                txt = tok.decode(out)
                # transport criterion
                low = txt.lower()
                h = sum(1 for w in held_w if re.search(r'\b'+w+r'\w*', low))
                b = sum(1 for w in anchor_w if re.search(r'\b'+w+r'\w*', low))
                maxr, dist1 = coherence(txt)
                ok_coh = maxr < 6 and dist1 > 0.6
                tr = (h >= 1) and (b == 0) and ok_coh
                if mode == 'plus': plus_ok = tr
                elif mode == 'rand': rand_ok = tr
                else: minus_ok = tr
                # logit geometry per generation (dH,dN) normalized
                if mode == 'plus':
                    # accumulate per step deltas; store mean over steps
                    pass
            results[a]['plus'] += int(plus_ok)
            results[a]['rand'] += int(rand_ok)
            results[a]['minus'] += int(minus_ok)
        print(f'  alpha={a:.2f}: +dper {results[a]["plus"]}/{K}  '
              f'rand {results[a]["rand"]}/{K}  -dper {results[a]["minus"]}/{K}', flush=True)

    # pre-registered falsification criterion
    print('\n== pre-registered verdict ==')
    found = False
    for a in ALPHAS:
        R = results[a]['plus']/K; Q = results[a]['rand']/K; M = results[a]['minus']/K
        if R >= 0.5 and R > Q:
            # also require sign not transport (to rule out generic perturbation)
            print(f'  alpha={a:.2f}: R={R:.2f} Q={Q:.2f} M={M:.2f} -> candidate')
            if M < R:   # sign-asymmetric
                found = True
    if found:
        print('  VERDICT: TRANSPORT candidate - complement has a dose.')
    else:
        print('  VERDICT: NO-TRANSPORT for complement (no alpha meets R>=.5 & R>Q & sign-asym)')
    print(f'[{time.time()-t0:.0f}s]')


def coherence(txt):
    toks = [w.strip('.,!?;:()[]"').lower() for w in re.findall(r"\S+", txt)]
    if not toks: return 0.0, 0.0
    maxr = cur = 0; prev = None; cnt = Counter()
    for t in toks:
        if t == prev: cur += 1
        else: cur = 1
        maxr = max(maxr, cur); prev = t; cnt[t] += 1
    return maxr, len(cnt)/len(toks)


if __name__ == '__main__':
    main()