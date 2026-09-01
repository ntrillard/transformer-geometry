#!/usr/bin/env python3
"""pivot_contrast.py — Compare logit-space contrast (the WORKING mechanism
from gen_geom.py CONTRAST_MODE=logit) against hidden-state d_per injection
mapped into logit space (W @ d_per), at the SAME continuation state, along
the reviewer's four dimensions:

  1. Anchor selectivity        ΔL_H / ΔL_A
  2. Distributional concentration  how many vocab entries actually receive
                                 probability mass (vs a diffuse W@d_per)
  3. Rank movement             do HELD_OUT words enter the competitive region
                               WITHOUT simultaneously elevating the anchors?
  4. Context adaptivity        does the logit-contrast effect change as the
                               generated context changes? (test static vs
                               per-context dL) vs a fixed global d_per

Mechanistic, NO sampling: for a batch of continuation states (diverse
contexts), apply each intervention to the SAME base logits and measure.

Interventions (all on the SAME base logits L0 at the SAME state v):
  dL_static : the working logit contrast, computed ONCE (top-200 masked),
              added at ALPHA (the gen_geom default ALPHA=2)
  dL_adapt  : the logit contrast RECOMPUTED at each continuation state
  Wdper     : hidden d_per injection viewed in logit space = ALPHA*norm(v)*W@d_per

Run: HF_TOKEN=... python3 pivot_contrast.py [seed]
"""
import os, sys, time
import torch, transformers

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODEL = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
NCTX = int(os.environ.get('NCTX', '12'))          # continuation states (contexts)
ALPHA = float(os.environ.get('ALPHA', '2.0'))     # working logit-contrast dose
NTOPK = int(os.environ.get('NTOPK', '200'))       # logit-contrast top-k mask

TGT = 'The king entered the castle|The queen sat upon the throne|' \
      'The prince inherited the crown|The royal family gathered in the great hall'
NEU = 'The person walked down the street|The individual entered the room|' \
      'The worker picked up the package'
HELD = 'crown reign kingdom realm monarchy monarch dynasty heir sovereign'
ANCH = 'king queen prince royal princess'

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def ids_of(tok, words):
    out = {}
    for w in words.split():
        sp = tok(' ' + w, add_special_tokens=False).input_ids
        if len(sp) == 1:
            out[w] = int(sp[0])
    return out


def main():
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    norm = model.model.norm
    W = model.lm_head.weight.detach().float(); Wt = W.t()

    # shell (for d_per)
    Wn = W / W.norm(dim=1, keepdim=True)
    _, S, V = torch.svd_lowrank(Wn.float().cuda(), q=300, niter=5)
    S = S.cpu(); V = V.cpu()
    eng = (S**2).cumsum(0) / (S**2).sum()
    r = int((eng >= 0.9).nonzero()[0].item()) + 1
    Uc = V[:, :r].float()

    def fwd(sents):
        """mean next-token logits + final hidden state over the sentence set."""
        Ls, Hs = [], []
        for s in sents:
            ids = tok(s, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
            vc = {}
            hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
            with torch.no_grad():
                L = model(ids).logits[0,-1,:].float()
            hk.remove()
            Ls.append(L); Hs.append(vc['o'].float())
        return torch.stack(Ls).mean(0), torch.stack(Hs).mean(0)

    tgt = [x.strip() for x in TGT.split('|') if x.strip()]
    neu = [x.strip() for x in NEU.split('|') if x.strip()]
    L_tgt, h_tgt = fwd(tgt)
    L_neu, h_neu = fwd(neu)

    # hidden d_per in the complement (0 leak)
    d = h_tgt - h_neu
    Ucd = Uc.to(d.device)
    dper = d - Ucd @ (Ucd.t() @ d); dper = dper / dper.norm()

    # static logit contrast (the WORKING gen_geom mechanism), top-k masked
    dl_raw = L_tgt - L_neu
    dz = (dl_raw - dl_raw.mean()) / (dl_raw.std() + 1e-6)
    order = dz.argsort(descending=True)
    mask = torch.zeros_like(dz); mask[order[:NTOPK]] = 1
    dL_static = (dz * mask)

    held = ids_of(tok, HELD); anch = ids_of(tok, ANCH)
    nH = len(held); nA = len(anch)
    print(f'pivot seed={SEED} nctx={NCTX} ALPHA={ALPHA} topk={NTOPK}', flush=True)
    print(f'  HELD_OUT n={nH} {list(held)}   ANCHOR n={nA}', flush=True)
    print(f'  d_per shell-leak={(Ucd@(Ucd.t()@dper)).norm():.4f}', flush=True)

    # diverse continuation states: walk baseline sampling from a seed prompt
    base = tok('The', add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
    torch.manual_seed(777 + SEED)
    pfx = []
    with torch.no_grad():
        ids = base
        for _ in range(NCTX):
            pfx.append(ids.clone())
            out = model(ids)
            p = torch.softmax(out.logits[0,-1,:].float(), 0)
            nxt = int(torch.multinomial(p, 1))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)

    # ---- per-state measurement ----
    def measure(v, L0, delta, name):
        """delta is a LOGIT-space increment applied at this state."""
        L1 = L0 + delta.to(L0.device)
        dH = sum((L1[i]-L0[i]).item() for i in held.values())/nH
        dA = sum((L1[i]-L0[i]).item() for i in anch.values())/nA
        # 2. concentration: how many entries moved by >0.1 logit
        moved = int((delta.abs() > 0.1).sum().item())
        # 3. rank movement (in the INTERVENED distribution)
        # 3. rank movement (in the INTERVENED distribution)
        order = L1.argsort(descending=True).tolist()
        pos = {tid: k for k, tid in enumerate(order)}
        mr = min(pos[i] for i in held.values())
        ma = min(pos[i] for i in anch.values())
        return dict(dH=dH, dA=dA, sel=(dH/dA if abs(dA) > 1e-6 else float('nan')),
                    moved=moved, minH=mr, minA=ma, name=name)

    # delta per intervention at THIS state
    # dL_adapt: recompute logit contrast at this state
    def dL_at(v):
        # the next-token readout surface implied at this state; use the W@v
        # surface as the "continuation distribution" estimate.
        surf = (v @ Wt.to(v.device))
        dl = (surf - L_neu.to(v.device))
        dz = (dl - dl.mean()) / (dl.std() + 1e-6)
        o = dz.argsort(descending=True); m = torch.zeros_like(dz); m[o[:NTOPK]] = 1
        return dz * m

    agg = {'dL_static': [], 'dL_adapt': [], 'Wdper': []}
    for j, pj in enumerate(pfx):
        vc = {}
        hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
        with torch.no_grad():
            L0 = model(pj).logits[0,-1,:].float()
        hk.remove()
        v = vc['o'].float()
        dH = v.norm().item() if False else v.norm().item()

        deltas = {
            'dL_static': (ALPHA * dL_static).to(DEV),
            'dL_adapt':  ALPHA * dL_at(v),
            'Wdper':     ALPHA * dH * (dper.to(DEV) @ Wt.to(DEV)),
        }
        for k, dl in deltas.items():
            agg[k].append(measure(v, L0, dl, k))

    print('\n  dim-2 concentration (mean abs-delta>0.1 entries) + dim-1 selectivity per state')
    print('  ctx   method      dH       dA    sel=dH/dA  moved  minH  minA')
    for j in range(NCTX):
        for k in ('dL_static', 'dL_adapt', 'Wdper'):
            m = agg[k][j]
            print(f'  {j:>3}  {k:9}  {m["dH"]:+7.2f}  {m["dA"]:+7.2f}  '
                  f'{m["sel"]:7.2f}  {m["moved"]:5d}  {m["minH"]:4d}  {m["minA"]:4d}')
        print()

    # aggregate across contexts
    print('\n  == aggregate over contexts (dim-4: context stability) ==')
    print(f'  {"method":9} {"<dH>":>8} {"<dA>":>8} {"<sel>":>7} {"<moved>":>7} '
          f'{"minH(min)":>10} {"minH(max)":>9}  dH_std')
    for k in ('dL_static', 'dL_adapt', 'Wdper'):
        rs = agg[k]
        dHm = sum(m['dH'] for m in rs)/len(rs); dAm = sum(m['dA'] for m in rs)/len(rs)
        selm = sum(m['sel'] for m in rs)/len(rs); mov = sum(m['moved'] for m in rs)/len(rs)
        mh = min(m['minH'] for m in rs); Mh = max(m['minH'] for m in rs)
        std = (sum((m['dH']-dHm)**2 for m in rs)/len(rs))**0.5
        print(f'  {k:9} {dHm:+8.2f} {dAm:+8.2f} {selm:7.2f} {mov:7.1f} '
              f'{mh:10d} {Mh:9d}  {std:.2f}')
    print(f'[{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()