#!/usr/bin/env python3
"""logit_scan.py - Cheap per-step logit scan. The mechanistic first test:
does d_per produce a SELECTIVE held-out logit signal from
h -> h + alpha*|h|*d_per ?

No sampling. For a batch of fixed prefixes, at each alpha, measure the mean
logit delta on HELD_OUT vs ANCHORS vs NEUTRAL for:
  +d_per, +random-complement, -d_per
vs baseline (alpha 0).

This decides whether a 22-min narrative sweep is worth running at all:
  - if no alpha beats random on held-out selectivity -> stop, Type A/dead
  - if a dose shows selective held-out signal -> go sample ONLY there.

Run: HF_TOKEN=... python3 logit_scan.py [seed]
"""
import os, sys, math, time
import torch, transformers

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODEL = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
PROMPT = os.environ.get('PROMPT', 'The waves crashed gently on the beach')
NPREFIX = int(os.environ.get('NPREFIX', '12'))   # fixed prefixes
ALPHAS = [float(x) for x in os.environ.get('ALPHAS',
    '0.04,0.10,0.16,0.22,0.28,0.34,0.40').split(',')]

TGT = 'The king entered the castle|The queen sat upon the throne|' \
      'The prince inherited the crown|The royal family gathered in the great hall'
NEU = 'The person walked down the street|The individual entered the room|' \
      'The worker picked up the package'
ANCHORS = 'king queen prince royal princess'
HELD_OUT = 'crown reign kingdom realm monarchy monarch dynasty heir sovereign'
NEUT = 'sand wave sea swim ocean surf beach tide shore shell'

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    t0 = time.time()
    print(f'logit_scan seed={SEED} npfx={NPREFIX} alphas={ALPHAS}', flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    norm = model.model.norm
    W = model.lm_head.weight.detach().float(); Wt = W.t()

    Wnorm = W / W.norm(dim=1, keepdim=True)
    _, S, V = torch.svd_lowrank(Wnorm.float().cuda(), q=300, niter=5)
    S = S.cpu(); V = V.cpu()
    eng = (S**2).cumsum(0) / (S**2).sum()
    r = int((eng >= 0.9).nonzero()[0].item()) + 1
    Uc = V[:, :r].float()
    print(f'  shell rank r={r}', flush=True)

    def state(s):
        t = tok(s, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        vc = {}
        hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
        with torch.no_grad(): model(t)
        hk.remove()
        return vc['o'].cpu().float()

    tgt = [s.strip() for s in TGT.split('|') if s.strip()]
    neu = [s.strip() for s in NEU.split('|') if s.strip()]
    d = torch.stack([state(s) for s in tgt]).mean(0) - torch.stack([state(s) for s in neu]).mean(0)
    dper = d - Uc @ (Uc.t() @ d); dper = dper/dper.norm()
    g = torch.Generator().manual_seed(777+SEED)
    z = torch.randn(1536, generator=g).float()
    zc = z - Uc @ (Uc.t() @ z); Rdper = zc/zc.norm()
    print(f'  dper leak={(Uc@(Uc.t()@dper)).norm():.4f}  random leak={(Uc@(Uc.t()@Rdper)).norm():.4f}', flush=True)

    def ids(words):
        return [(w, int(tok(' '+w, add_special_tokens=False).input_ids[0]))
                for w in words.split() if len(tok(' '+w, add_special_tokens=False).input_ids)==1]
    held = ids(HELD_OUT); anch = ids(ANCHORS); neut = ids(NEUT)
    nH=len(held); nB=len(anch); nN=len(neut)

    # fixed prefixes: start from PROMPT, walk out NPREFIX steps with baseline sampling
    torch.manual_seed(111+SEED)
    ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
    prefixes = []
    with torch.no_grad():
        for _ in range(NPREFIX):
            out = model(ids)
            L = out.logits[0,-1].float()
            p = torch.softmax(L,0); q=p.clone(); oo=q.argsort(descending=True); k=int((q[oo].cumsum(0)<=0.9).sum())+1
            m=torch.zeros_like(q); m[oo[:k]]=1; qq=(q*m); qq=qq/qq.sum()
            nxt=int(torch.multinomial(qq,1))
            prefixes.append(ids.clone())
            ids = torch.cat([ids, torch.tensor([[nxt]],device=DEV)],dim=1)

    # for each prefix & alpha & direction, compute mean dL over the vocab groups
    def step_dL(prefix_ids, dirv, alpha):
        vc={}
        hk=norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
        with torch.no_grad(): L0 = model(prefix_ids).logits[0,-1].float()
        hk.remove()
        v=vc['o'].float()
        vp = v + alpha*v.norm()*dirv.to(v.device)
        L1 = torch.matmul(vp, Wt.to(v.device)).float()
        dL = L1 - L0
        return (sum(dL[i].item() for _,i in held)/nH,
                sum(dL[i].item() for _,i in anch)/nB,
                sum(dL[i].item() for _,i in neut)/nN)

    dirs = {'+dper': dper, 'rand': Rdper, '-dper': -dper}
    print('\n alpha  cond     dL held     dL anchor   dL neutral   held-minus-neutral')
    for a in ALPHAS:
        for cond, dv in dirs.items():
            hsum=bsum=nsum=0.0
            for pfx in prefixes:
                h,b,n = step_dL(pfx, dv, a)
                hsum+=h; bsum+=b; nsum+=n
            hh=hsum/len(prefixes); bb=bsum/len(prefixes); nn=nsum/len(prefixes)
            print(f'{a:.2f}  {cond:6}  {hh:+9.3f}   {bb:+9.3f}   {nn:+9.3f}   {hh-nn:+9.3f}')
        print()
    print(f'[{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()