#!/usr/bin/env python3
"""falsify_orth3.py - Orthogonal-complement transport falsification, v3.

FAST version with incremental (KV-cache) generation + all reviewer fixes:
  - random control PROJECTED INTO THE COMPLEMENT, matched norm
  - per-alpha primary verdict (exists alpha: R>=.5 & R>Q & sign-asymmetric)
  - sign control -dper at every alpha
  - explicit ANCHORS / ENUM_CLUSTER / HELD_OUT vocab
  - HELD_OUT measured at STRING level in generated text (catches coronation,
    which is 2 tokens); token-IDs used ONLY for the pre-sample logit diag
  - retained per-alpha Delta-logit columns (Type-A vs Type-B disambiguation)
  - effect size reported alongside the pre-registered verdict
  - absolute a-priori coherence criterion (max-run<6 and dist-1>.6)
  - sphere-defined complement (PCA of NORMALIZED token rows); raw-W variant is
    flagged future work.

Primary falsification verdict (pre-registered):
  exists alpha: R_a>=.5 AND R_a>Q_a AND M_a<R_a   (R=+dper, Q=random, M=-dper)

Run: HF_TOKEN=... python3 falsify_orth3.py [seed]
"""
import os, sys, math, time, re, json
from collections import Counter
import torch, transformers

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODEL = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
PROMPT = os.environ.get('PROMPT', 'The waves crashed gently on the beach')
NTOK = int(os.environ.get('NTOK', '120'))
K = int(os.environ.get('K', '20'))
ALPHAS = [float(x) for x in os.environ.get('ALPHAS',
    '0.04,0.10,0.16,0.22,0.28,0.34').split(',')]
NUCLEUS = float(os.environ.get('NUCLEUS', '0.9'))

TGT = 'The king entered the castle|The queen sat upon the throne|' \
      'The prince inherited the crown|The royal family gathered in the great hall'
NEU = 'The person walked down the street|The individual entered the room|' \
      'The worker picked up the package'
ANCHORS = 'king queen prince royal princess'
ENUM_CLUSTER = 'palace castle throne fortress'
HELD_OUT = 'crown reign kingdom realm monarchy monarch dynasty heir coronation sovereign'
NEUT = 'sand wave sea swim ocean surf beach tide shore shell'

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_gen(seed):
    torch.manual_seed(seed)
    return seed


def main():
    t0 = time.time()
    print(f'falsify_orth3 seed={SEED} K={K} ntok={NTOK} alphas={ALPHAS}', flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    norm = model.model.norm
    W = model.lm_head.weight.detach().float()
    Wt = W.t()
    eos_id = int(tok.eos_token_id)

    # #9 sphere-defined complement: PCA of NORMALIZED token rows
    Wnorm = W / W.norm(dim=1, keepdim=True)
    _, S, V = torch.svd_lowrank(Wnorm.float().cuda(), q=300, niter=5)
    S = S.cpu(); V = V.cpu()
    eng = (S**2).cumsum(0) / (S**2).sum()
    r = int((eng >= 0.9).nonzero()[0].item()) + 1
    Uc = V[:, :r].float()
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
    dper = d - Uc @ (Uc.t() @ d); dper = dper / dper.norm()
    print(f'  dper norm={dper.norm():.3f} shell-leak={(Uc@(Uc.t()@dper)).norm():.4f}', flush=True)

    # #1 random control projected into complement
    g = torch.Generator().manual_seed(777 + SEED)
    z = torch.randn(1536, generator=g).float()
    zc = z - Uc @ (Uc.t() @ z); Rdper = zc / zc.norm()
    print(f'  random-complement norm={Rdper.norm():.3f} shell-leak={(Uc@(Uc.t()@Rdper)).norm():.4f}', flush=True)

    # vocab: string-level (generated text) + token-id (logit diag)
    def sids(words):
        ids = {}
        for w in words.split():
            sp = tok(' '+w, add_special_tokens=False).input_ids
            if len(sp) == 1:
                ids[w] = int(sp[0])
        return ids, [w for w in words.split() if w not in ids]
    held_id, held_multi = sids(HELD_OUT)
    anchor_id, _ = sids(ANCHORS)
    neut_id, _ = sids(NEUT)
    held_strs = [w for w in HELD_OUT.split()]
    anchor_strs = [w for w in ANCHORS.split()]
    print(f'  HELD_OUT: single-token ids={list(held_id.keys())} multi(dropped-for-logit-only)={held_multi}')
    print(f'  ANCHORS: {anchor_strs}  ENUM_CLUSTER: {ENUM_CLUSTER}')

    # KV-cache forward WITH hidden capture in the same call
    def generate2(dirv, alpha, seed, suppress_anchor=False):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        past = None
        from collections import defaultdict
        acc = defaultdict(float)
        out_ids = []
        min_rank = 10**9          # min over steps of the best (min) held-out rank
        top_p_held_steps = 0      # steps where any held-out word enters top-p set
        held_ranks = []           # per-step min held rank
        with torch.no_grad():
            for _ in range(NTOK):
                vc = {}
                hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                hk.remove()
                if past is None:
                    past = out.past_key_values
                v = vc['o'].float()                      # last hidden state (1536)
                L0 = out.logits[0, -1, :].float()        # natural logits on h
                if dirv is not None:
                    vp = v + alpha * v.norm() * dirv.to(v.device)
                    L1base = torch.matmul(vp, Wt.to(v.device)).float()
                    dL = L1base - L0
                    for w, i in held_id.items(): acc['H'] += dL[i].item()
                    for w, i in anchor_id.items(): acc['B'] += dL[i].item()
                    for w, i in neut_id.items(): acc['N'] += dL[i].item()
                    L1 = L1base
                else:
                    L1 = L0
                if suppress_anchor:
                    L1 = L1.clone(); L1[list(anchor_id.values())] = -50.0
                # rank tracking of held-out words in the SAMPLING logits
                order = L1.argsort(descending=True).tolist()
                pos = {tok_id: rank for rank, tok_id in enumerate(order)}
                mr = min(pos[i] for i in held_id.values())
                min_rank = min(min_rank, mr)
                held_ranks.append(mr)
                p = torch.softmax(L1, 0)
                q = p.clone(); ooo = q.argsort(descending=True)
                kk = int((q[ooo].cumsum(0) <= NUCLEUS).sum()) + 1
                top_p = set(ooo.tolist() for ooo in [ooo[:kk]][0])
                if any(i in top_p for i in held_id.values()):
                    top_p_held_steps += 1
                msk = torch.zeros_like(q); msk[ooo[:kk]] = 1
                qq = (q*msk); qq = qq/qq.sum()
                nxt = int(torch.multinomial(qq, 1))
                if nxt == eos_id: break
                out_ids.append(nxt)
                ids = torch.tensor([[nxt]], device=DEV)
        txt = tok.decode(out_ids)
        info = {'min_held_rank': min_rank, 'top_p_held_steps': top_p_held_steps,
                'steps': len(held_ranks), 'avg_min_held_rank': sum(held_ranks)/max(1,len(held_ranks))}
        return txt, acc, info

    # a-priori coherence criterion is absolute: maxrun<6 & dist1>.6 (in score)
    def score(txt):
        toks = [w.strip('.,!?;:()[]"').lower() for w in re.findall(r"\S+", txt)]
        if not toks: return 0, 0.0, 0, 0, 0
        maxr = cur = 0; prev = None; cnt = Counter()
        for t in toks:
            if t == prev: cur += 1
            else: cur = 1
            maxr = max(maxr, cur); prev = t; cnt[t] += 1
        dist1 = len(cnt)/len(toks)
        low = txt.lower()
        h = sum(1 for w in held_strs if re.search(r'\b'+w+r'\w*', low))
        b = sum(1 for w in anchor_strs if re.search(r'\b'+w+r'\w*', low))
        ok_coh = maxr < 6 and dist1 > 0.6
        transport = (h >= 1) and (b == 0) and ok_coh
        return h, b, maxr, dist1, transport

    # --- run the table ---
    rows = []
    for a in ALPHAS:
        R = Q = M = SA = 0
        min_rank_plus = 10**9
        tentry_plus = 0; tsteps_plus = 0
        # logit accumulations averaged over the K nonzero runs
        lH = {'plus': 0.0, 'rand': 0.0, 'minus': 0.0}
        lB = {'plus': 0.0, 'rand': 0.0, 'minus': 0.0}
        lN = {'plus': 0.0, 'rand': 0.0, 'minus': 0.0}
        for s in range(K):
            tp, accp, infop = generate2(dper, a, 3000+s)
            tr, accr, infor = generate2(Rdper, a, 4000+s)
            tm, accm, infom = generate2(-dper, a, 5000+s)
            # anchor-suppressed version of +dper (tests transport past anchors)
            ta, acca, infoa = generate2(dper, a, 6000+s, suppress_anchor=True)
            _,_,_,_,tp_ok = score(tp); R += int(tp_ok)
            _,_,_,_,tr_ok = score(tr); Q += int(tr_ok)
            _,_,_,_,tm_ok = score(tm); M += int(tm_ok)
            _,_,_,_,ta_ok = score(ta); SA += int(ta_ok)
            min_rank_plus = min(min_rank_plus, infop['min_held_rank'])
            tentry_plus += infop['top_p_held_steps']
            tsteps_plus += infop['steps']
            for mode, acc in [('plus',accp),('rand',accr),('minus',accm)]:
                lH[mode]+=acc.get('H',0.0); lB[mode]+=acc.get('B',0.0); lN[mode]+=acc.get('N',0.0)
        # normalize logit deltas by #steps * #words * K (avg per word per step)
        nH = max(1, len(held_id)); nB = max(1, len(anchor_id)); nN = max(1, len(neut_id))
        rows.append((a, R, Q, M, SA,
                     lH['plus']/(K*nH), lB['plus']/(K*nB), lN['plus']/(K*nN),
                     lH['rand']/(K*nH), lB['rand']/(K*nB), lN['rand']/(K*nN),
                     min_rank_plus, tentry_plus/max(1,tsteps_plus)))
        print(f"  a={a:.2f}  +dper {R:2d}/{K}  rand {Q:2d}/{K}  -dper {M:2d}/{K}  "
              f"supA {SA:2d}/{K}",
              f"| dLogitH+={lH['plus']/(K*nH):+.2f}  B+={lB['plus']/(K*nB):+.2f}  N+={lN['plus']/(K*nN):+.2f}  "
              f"H+rand={lH['rand']/(K*nH):+.2f}",
              f"| minHrank={min_rank_plus}  topPentry={tentry_plus/max(1,tsteps_plus):.2f}", flush=True)

    print('\n== table ==')
    print(' a     +dper  supA  rand  -dper  dLogitH+  dLogitB+  dLogitN+  H-rand  minHrank topPentry')
    for (a,R,Q,M,SA,hp,bp,np,hr,br,nr,mr,te) in rows:
        print(f'{a:.2f}  {R:2d}   {SA:2d}  {Q:2d}   {M:2d}   {hp:+6.2f}  {bp:+6.2f}  {np:+6.2f}  {hp-hr:+6.2f}  {mr:4d}  {te:.2f}')

    # pre-registered verdict
    print('\n== pre-registered verdict ==')
    found = False
    for (a,R,Q,M,SA,hp,bp,np,hr,br,nr,mr,te) in rows:
        if R/K >= 0.5 and R > Q and M < R:
            found = True
            print(f'  TRANSPORT-CANDIDATE at a={a:.2f}: R={R}/{K}={R/K:.2f} Q={Q/K:.2f} M={M/K:.2f}')
    if not found:
        print('  NO-TRANSPORT (no alpha satisfies R>=.5 & R>Q & sign-asym)')
        any_logit = any(hp > 2.0 for (a,R,Q,M,SA,hp,bp,np,hr,br,nr,mr,te) in rows)
        print('  logit-geometry: ' + ('Type B (held-out logits rise but do not sample)'
              if any_logit else 'Type A (no held-out logit advantage)'))
    print(f'[{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()