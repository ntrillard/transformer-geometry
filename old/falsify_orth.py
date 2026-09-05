#!/usr/bin/env python3
"""falsify_orth.py - Bounded falsification of the orthogonal-complement
transport hypothesis.

Single purpose: is there a narrow alpha regime where +alpha*d_per (embedding
the true contextual-royal-minus-neutral state contrast, orthogonal to the
token shell) transports HELD-OUT royal vocabulary into *coherent sampled
narrative* -- strictly better than a random rotation of d_per at matched dose?

Precommitted transport criterion (no alpha-shopping):
  Given K=8 generations per (cond, alpha):
    transport_generations(g) = 1 if
       (a) >= 1 held-out theme word present in the text, AND
       (b) zero blocked-anchor words present, AND
       (c) coherence(g) >= coherence baseline (rep1d < 6 AND dist-1 > 0.6)
  condWins = argmax over cond of mean(transport_generations)
  verdict = TRANSPORT if realWins > randomWins AND realWins >= 0.5
            else NO-TRANSPORT
  Random control at every alpha. Multi-sample per (cond,alpha).

Run: HF_TOKEN=... python3 falsify_orth.py [seed]
"""
import os, sys, math, time, re
from collections import Counter
import torch, transformers

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODEL = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
PROMPT = os.environ.get('PROMPT', 'The waves crashed gently on the beach')
NTOK = int(os.environ.get('NTOK', '120'))
K = int(os.environ.get('K', '8'))                 # generations per (cond, alpha)
ALPHAS = [float(x) for x in os.environ.get('ALPHAS',
    '0.04,0.08,0.12,0.16,0.20,0.25,0.30,0.35').split(',')]
NUCLEUS = float(os.environ.get('NUCLEUS', '0.9'))

TGT = os.environ.get('TGT',
  'The king entered the castle|The queen sat upon the throne|'
  'The prince inherited the crown|The royal family gathered in the great hall')
NEU = os.environ.get('NEU',
  'The person walked down the street|The individual entered the room|'
  'The worker picked up the package')
HELD = 'crown reign kingdom realm monarchy crowned monarch court dynasty heir throne castle palace'
BLOCK = 'king queen prince royal princess'
NEUT = 'sand wave sea swim ocean surf beach tide shore shell'

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    t0 = time.time()
    print(f'falsify_orth seed={SEED} K={K} ntok={NTOK} alphas={ALPHAS}', flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    norm = model.model.norm
    W = model.lm_head.weight.detach().float()
    Wt = W.t()
    eos_id = int(tok.eos_token_id)

    # --- token-shell PCA basis (90% energy) ---
    _, S, V = torch.svd_lowrank((W/W.norm(dim=1,keepdim=True)).float().cuda(), q=300, niter=5)
    S = S.cpu(); V = V.cpu()
    eng = (S**2).cumsum(0) / (S**2).sum()
    r = int((eng >= 0.9).nonzero()[0].item()) + 1
    Uc = V[:, :r].float()
    print(f'  token-shell rank r={r}', flush=True)

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
    dpar = Uc @ (Uc.t() @ d)
    dper = d - dpar
    dper = dper / dper.norm()
    print(f'  dperp norm=1, frac=|dperp|^2/|d|^2=', flush=True)

    # random rotation of the complement (Haar-random orthogonal, fixed seed per seed)
    g = torch.Generator().manual_seed(777 + SEED)
    Q = torch.linalg.qr(torch.randn(1536, 1536, generator=g), 'reduced')[0].float()
    Rdper = (Q @ dper); Rdper = Rdper / Rdper.norm()

    def ids_of(words):
        out = []
        for w in words.split():
            sp = tok(' '+w, add_special_tokens=False).input_ids
            if len(sp) == 1: out.append((w, int(sp[0])))
        return out
    held_ids = ids_of(HELD); block_ids = ids_of(BLOCK); neut_ids = ids_of(NEUT)
    held_w = set(w for w,_ in held_ids); block_w = set(w for w,_ in block_ids)

    def generate(dirv, alpha, seed):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        out = []
        for _ in range(NTOK):
            vc = {}
            hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
            with torch.no_grad(): L = model(ids).logits[0,-1].float()
            hk.remove()
            v = vc['o'].float()
            if dirv is not None:
                v = v + alpha * v.norm() * dirv.to(v.device)
                L = torch.matmul(v, Wt.to(v.device))
            p = torch.softmax(L, 0)
            q = p.clone(); order = q.argsort(descending=True)
            k = int((q[order].cumsum(0) <= NUCLEUS).sum()) + 1
            msk = torch.zeros_like(q); msk[order[:k]] = 1
            qq = (q*msk); qq = qq/qq.sum()
            nxt = int(torch.multinomial(qq, 1))
            if nxt == eos_id: break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
            out.append(nxt)
        return tok.decode(out)

    def coherence(txt):
        toks = [w.strip(' .,!?;:()[]\"\'').lower() for w in re.findall(r"\S+", txt)]
        if not toks: return 0.0, 0.0
        maxr = cur = 0; prev = None; cnt = Counter()
        for t in toks:
            if t == prev: cur += 1
            else: cur = 1
            maxr = max(maxr, cur); prev = t; cnt[t] += 1
        dist1 = len(cnt) / len(toks)
        return maxr, dist1

    # baseline coherence (seed batch, no steering)
    base_maxr, base_dist = [], []
    for s in range(5):
        t = generate(None, 0.0, 1000 + s)
        m, d1 = coherence(t)
        base_maxr.append(m); base_dist.append(d1)
    base_coh = (sum(base_maxr)/len(base_maxr), sum(base_dist)/len(base_dist))
    print(f'  baseline coherence: max_run={base_coh[0]:.1f} dist1={base_coh[1]:.2f}', flush=True)

    def score_generation(txt):
        low = txt.lower()
        h = sum(1 for w in held_w if re.search(r'\b' + w + r'\w*', low))
        b = sum(1 for w in block_w if re.search(r'\b' + w + r'\w*', low))
        m, d1 = coherence(txt)
        ok_coh = m < 6 and d1 > 0.6
        transport = (h >= 1) and (b == 0) and ok_coh
        return h, b, m, d1, transport

    # run baseline transport (should be ~0)
    base_transport = 0
    for s in range(K):
        t = generate(None, 0.0, 2000 + s)
        h, b, m, d1, tr = score_generation(t)
        base_transport += int(tr)
    print(f'  BASELINE transport@{K} = {base_transport} (of {K})', flush=True)

    print(f'\n{"alpha":>6} | respons  wins  rate | rand  wins  rate | real-wins-rand')
    verdict_rows = []
    real_total = rand_total = 0
    for a in ALPHAS:
        rw = 0; rn = 0
        for s in range(K):
            t_real = generate(dper, a, 3000 + s)
            t_rand = generate(Rdper, a, 4000 + s)
            h,b,m,d1,tr = score_generation(t_real); rw += int(tr)
            h,b,m,d1,tr = score_generation(t_rand); rn += int(tr)
        real_total += rw; rand_total += rn
        verdict_rows.append((a, rw, rn))
        print(f'{a:6.3f} | real {rw:2d}/{K} ({rw/K:.2f}) | rand {rn:2d}/{K} ({rn/K:.2f}) | {rw-rn:+d}', flush=True)

    real_rate = real_total / (len(ALPHAS)*K)
    rand_rate = rand_total / (len(ALPHAS)*K)
    print(f'\naggregate: real={real_total} ({real_rate:.2f})  rand={rand_total} ({rand_rate:.2f})')
    if real_rate > rand_rate and real_rate >= 0.5:
        print('VERDICT: TRANSPORT (real complement beats random at matched dose)')
    else:
        print('VERDICT: NO-TRANSPORT for complement under bounded scan')
    print(f'[{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()