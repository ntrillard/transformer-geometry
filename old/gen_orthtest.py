#!/usr/bin/env python3
"""gen_orthtest.py - Does contextual MEANING live in the token-shell COMPLEMENT?

Decomposition experiment (Qwen2-1.5B):
  d     = mean_state(target royal sents) - mean_state(neutral sents)
  d_par = U U^T d     (projection onto the rank-r token-shell PCA basis)
  d_per = d - d_par   (orthogonal complement)

For each candidate direction we steer the READOUT (h -> h + alpha*dir, equal
norm of d) and measure whether the RELATIVE distribution of ROYAL vocabulary
rises (delta_theme - delta_neutral) while blocked anchors stay suppressed.

Conditions (matched dose, normalized dirs):
  baseline            no steering
  shell    +a d_par   (token-shell part)
  comp     +a d_per   (orthogonal complement)
  full     +a d       (both)
  rand     +a R d_per (random rotation of complement - control)
  sign-    -a d_per   (sign flip)

Scoring per condition (all over the SAME sampled-token distribution):
  L_H, L_B, L_N   = mean logP over held-out / blocked / neutral words
  delta_theme   = L_H(steered) - L_H(baseline)
  delta_blocked = L_B(steered) - L_B(baseline)
  delta_neutral = L_N(steered) - L_N(baseline)
  net_theme     = delta_theme - delta_neutral   (removes generic perturbation)
  net_block     = delta_blocked - delta_neutral (should be ~0 when blocked works)

Run: HF_TOKEN=... python3 gen_orthtest.py [seed]
"""
import os, sys, math, time
import torch, transformers

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODEL = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
PROMPT = os.environ.get('PROMPT', 'The waves crashed gently on the beach')
NTOK = int(os.environ.get('NTOK', '90'))
ALPHA = float(os.environ.get('ALPHA', '0.35'))   # fraction of state norm
NUCLEUS = float(os.environ.get('NUCLEUS', '0.9'))

TGT = os.environ.get('TGT',
  'The king entered the castle|The queen sat upon the throne|'
  'The prince inherited the crown|The royal family gathered in the great hall')
NEU = os.environ.get('NEU',
  'The person walked down the street|The individual entered the room|'
  'The worker picked up the package')
HELD = os.environ.get('HELD', 'crown reign kingdom realm monarchy crowned '
                      'monarch court dynasty heir throne castle palace')
BLOCK = os.environ.get('BLOCK', 'king queen prince royal princess')
NEUT = os.environ.get('NEUT', 'sand wave sea swim ocean surf beach tide shore shell')

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    norm = model.model.norm
    W = model.lm_head.weight.detach().float()
    Wt = W.t()                       # (1536, 152k) for readout
    eos_id = int(tok.eos_token_id)

    # ---- token-shell PCA basis (columns V[:, :r]) ----
    print('building token-shell PCA...', flush=True)
    _, S, V = torch.svd_lowrank(Wn().cuda() if False else (W/W.norm(dim=1,keepdim=True)).float().cuda(), q=300, niter=5)
    S = S.cpu(); V = V.cpu()
    eng = (S**2).cumsum(0) / (S**2).sum()
    r = int((eng >= 0.9).nonzero()[0].item()) + 1
    Uc = V[:, :r].float()            # 1536 x r
    print(f'  token-shell rank r={r} (90% energy)', flush=True)

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
    dpar = dpar / dpar.norm(); dper = dper / dper.norm(); d = d / d.norm()
    comp_frac = ((d - dpar) if False else dper).norm()**2   # =1 since dpar norm 1
    print(f'  |dpar|={dpar.norm():.3f} |dper|={dper.norm():.3f} '
          f'orthogonal-complement fraction={dper.norm()**2:.3f}', flush=True)
    # random rotation of the complement (Haar-random orthogonal in 1536-D)
    g = torch.Generator().manual_seed(999 + SEED)
    Q = torch.linalg.qr(torch.randn(1536, 1536, generator=g), 'reduced')[0].float()
    Rdper = (Q @ dper); Rdper = Rdper / Rdper.norm()

    dirs = {'shell': dpar, 'comp': dper, 'full': d, 'rand': Rdper, 'sign-': -dper}

    # ---- word id lists ----
    def ids_of(words):
        out = []
        for w in words.split():
            sp = tok(' '+w, add_special_tokens=False).input_ids
            if len(sp) == 1:
                out.append((w, int(sp[0])))
        return out
    held_ids = ids_of(HELD); block_ids = ids_of(BLOCK); neut_ids = ids_of(NEUT)
    nH = max(1, len(held_ids)); nB = max(1, len(block_ids)); nN = max(1, len(neut_ids))

    def generate(dirv=None, suppress_blocked=False):
        torch.manual_seed(SEED)
        ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        accH = {w: 0.0 for w,_ in held_ids}
        accB = {w: 0.0 for w,_ in block_ids}
        accN = {w: 0.0 for w,_ in neut_ids}
        txt = []
        for _ in range(NTOK):
            vc={}
            hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
            with torch.no_grad(): L = model(ids).logits[0,-1].float()
            hk.remove()
            v = vc['o'].float()
            if dirv is not None:
                v = v + ALPHA * v.norm() * dirv.to(v.device)
                L = torch.matmul(v, Wt.to(v.device))
            if suppress_blocked:
                L = L.clone(); L[[i for _,i in block_ids]] = -50.0
            p = torch.softmax(L, 0)
            for w,i in held_ids: accH[w] += math.log(max(p[i].item(),1e-12))
            for w,i in block_ids: accB[w] += math.log(max(p[i].item(),1e-12))
            for w,i in neut_ids: accN[w] += math.log(max(p[i].item(),1e-12))
            q = p.clone(); order=q.argsort(descending=True); k=int((q[order].cumsum(0)<=NUCLEUS).sum())+1
            msk=torch.zeros_like(q); msk[order[:k]]=1; qq=(q*msk); qq=qq/(qq.sum())
            nxt = int(torch.multinomial(qq,1))
            if nxt==eos_id: break
            ids = torch.cat([ids, torch.tensor([[nxt]],device=DEV)],dim=1)
            txt.append(tok.decode([nxt]))
        return accH, accB, accN, ''.join(txt).replace('▁',' ')

    # baseline
    lbH, lbB, lbN, _ = generate(None)
    baseH = sum(lbH.values())/nH; baseB = sum(lbB.values())/nB; baseN = sum(lbN.values())/nN
    print(f'\nBASELINE  <logP held>={baseH:+.2f} <logP blocked>={baseB:+.2f} <logP neutral>={baseN:+.2f}', flush=True)

    print(f'\n{"cond":10} {"dH":>8} {"dB":>8} {"dN":>8} {"netTheme":>9} {"netBlock":>9}')
    for nm, dv in dirs.items():
        sH, sB, sN, txt = generate(dv)
        dH = sum(sH.values())/nH - baseH
        dB = sum(sB.values())/nB - baseB
        dN = sum(sN.values())/nN - baseN
        print(f'{nm:10} {dH:+8.3f} {dB:+8.3f} {dN:+8.3f} {dH-dN:+9.3f} {dB-dN:+9.3f}', flush=True)
        # show first 100 chars of the generation for qualitative read
        print(f'    ...{txt[:120]}', flush=True)

    # comp + active suppression: ensure blocked words are truly suppressed in the
    # POST-suppression distribution (their logp floor). Use comp direction.
    sH, sB, sN, txt = generate(dirs['comp'], suppress_blocked=True)
    dH = sum(sH.values())/nH - baseH
    dB = sum(sB.values())/nB - baseB
    dN = sum(sN.values())/nN - baseN
    print(f'comp+blk   {dH:+8.3f} {dB:+8.3f} {dN:+8.3f} {dH-dN:+9.3f} {dB-dN:+9.3f}', flush=True)
    print(f'    ...{txt[:120]}', flush=True)

    print(f'\n[{time.time()-t0:.0f}s]')


def Wn():
    raise SystemExit("unused")


if __name__ == '__main__':
    main()