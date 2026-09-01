#!/usr/bin/env python3
"""decomp_gen.py — Causal test of the dL decomposition.

From vec_compare: dL_static = dL_parallel + dL_perp, where
  dL_parallel = proj_{Wd_per}(dL_static)   (hidden-representable component)
  dL_perp     = dL_static - dL_parallel    (readout residual, ~99% of energy)

Run generation under each component independently (all matched-norm), to ask:
  which component carries the transport?

  A : L + alpha*dL_parallel        expect ~0      (hidden-representable, no transport)
  B : L + alpha*dL_perp            expect HIGH    (readout residual carries it)
  C : L + alpha*dL_static          expect HIGH    (reference / full)
  D : random vector, norm-matched to dL_perp       expect ~0  (not just norm)

Run: HF_TOKEN=... python3 decomp_gen.py [seed]  (default fantasy task)
     TGT/NEU/ALPHA/K/NTOK/SW0 env-overridable; ROYAL=1 for the royal task.
"""
import os, sys, time, re
from collections import Counter
import torch, transformers

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODEL = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
NTOK = int(os.environ.get('NTOK', '120'))
K = int(os.environ.get('K', '6'))
NUCLEUS = float(os.environ.get('NUCLEUS', '0.9'))
ALPHA = float(os.environ.get('ALPHA', '2.0'))
NTOPK = int(os.environ.get('NTOPK', '200'))
SW0 = int(os.environ.get('SW0', '20'))
SEEDBASE = int(os.environ.get('SEEDBASE', '3000'))
REP_PEN = float(os.environ.get('REP_PEN', '0.0'))
PROMPT = os.environ.get('PROMPT', 'The waves crashed gently on the beach')

if os.environ.get('ROYAL') == '1':
    TGT = 'The king entered the castle|The queen sat upon the throne|' \
          'The prince inherited the crown|The royal family gathered in the great hall'
    NEU = 'The person walked down the street|The individual entered the room|' \
          'The worker picked up the package'
    HELD_OUT = 'crown reign kingdom realm monarchy monarch dynasty heir sovereign'
    ANCHORS = 'king queen prince royal princess'
else:
    TGT = ('A dragon circled the ruined towers of the ancient kingdom|'
           'A knight drew his sword against the fire-breathing beast|'
           "The wizard's spell shattered the castle gates")
    NEU = ('The waves crashed gently on the beach|'
           'The sand was cool to the touch|The sun was warm over the water')
    HELD_OUT = 'creature creatures evil monsters beast horror lurking nightmare demons'
    ANCHORS = ''

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def score(txt):
    toks = [w.strip('.,!?;:()[]"\'').lower() for w in re.findall(r"\S+", txt)]
    if not toks:
        return 0, 0.0, 0, 0, 0
    maxr = cur = 0; prev = None; cnt = Counter()
    for t in toks:
        cnt[t] += 1
        if t == prev:
            cur += 1; maxr = max(maxr, cur)
        else:
            cur = 1; prev = t
    dist1 = len(cnt) / len(toks)
    low = txt.lower()
    h = sum(1 for w in HELD_OUT.split() if re.search(r'\b'+w+r'\w*', low))
    anc = ANCHORS.split()
    b = sum(1 for w in anc if re.search(r'\b'+w+r'\w*', low)) if anc else 0
    ok_coh = maxr < 6 and dist1 > 0.6
    transport = (h >= 1) and (b == 0) and ok_coh
    return h, b, maxr, dist1, transport


def main():
    t0 = time.time()
    print(f'decomp_gen seed={SEED} K={K} ALPHA={ALPHA} '
          f'{"ROYAL" if os.environ.get("ROYAL")=="1" else "FANTASY"}', flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    norm = model.model.norm
    W = model.lm_head.weight.detach().float(); Wt = W.t()
    eos_id = int(tok.eos_token_id)

    Wnorm = W / W.norm(dim=1, keepdim=True)
    _, S, V = torch.svd_lowrank(Wnorm.float().cuda(), q=300, niter=5)
    S = S.cpu(); V = V.cpu()
    eng = (S**2).cumsum(0) / (S**2).sum()
    r = int((eng >= 0.9).nonzero()[0].item()) + 1
    Uc = V[:, :r].float()

    def logits_of(s):
        t = tok(s, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        with torch.no_grad():
            return model(t).logits[0,-1,:].float().cpu()

    def state_of(s):
        t = tok(s, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        vc = {}
        hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
        with torch.no_grad(): model(t)
        hk.remove()
        return vc['o'].float().cpu()

    tgt = [x.strip() for x in TGT.split('|') if x.strip()]
    neu = [x.strip() for x in NEU.split('|') if x.strip()]

    # --- Wd_per in logit space ---
    Ts = torch.stack([state_of(s) for s in tgt]).mean(0)
    Ns = torch.stack([state_of(s) for s in neu]).mean(0)
    Ucd = Uc.to(Ts.device)
    dper = (Ts - Ns) - Ucd @ (Ucd.t() @ (Ts - Ns)); dper = dper / dper.norm()
    Wd_per = (dper.cpu() @ Wt.cpu())

    # --- dL_static (faithful) ---
    nm = torch.stack([logits_of(s) for s in neu]).mean(0)
    dL_raw = None
    for s in tgt:
        Ls = logits_of(s)
        c = Ls - nm
        c = (c - c.mean()) / (c.std() + 1e-6)
        dL_raw = c if dL_raw is None else dL_raw + c
    dL_raw = dL_raw / max(1, len(tgt))
    dL_z = (dL_raw - dL_raw.mean()) / (dL_raw.std() + 1e-6)
    m = torch.zeros_like(dL_z); m[dL_z.argsort(descending=True)[:NTOPK]] = 1.0
    # APPLY ALPHA here (gen_geom: L = L_nat + ALPHA*dL). Effective offset ~102.
    dL_static = ALPHA * dL_z * m

    # --- decompose (scale-invariant, but carry the ALPHA magnitude) ---
    w = Wd_per / Wd_per.norm()
    dL_parallel = (dL_static @ w) * w
    dL_perp = dL_static - dL_parallel
    # random control matched to dL_perp's norm
    g = torch.Generator().manual_seed(555 + SEED)
    dL_rand = torch.randn(dL_static.numel(), generator=g)
    dL_rand = dL_rand * (dL_perp.norm() / dL_rand.norm())

    print(f'  ||dL_parallel||={dL_parallel.norm():.3f}  ||dL_perp||={dL_perp.norm():.3f}  '
          f'||dL_full||={dL_static.norm():.3f}', flush=True)

    deltas = {
        'dL_parallel': dL_parallel,
        'dL_perp':     dL_perp,
        'dL_full':     dL_static,
        'rand_perpN':  dL_rand,
    }

    def generate(delta, seed):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        past = None; out_ids = []; rep_hist = []; dd = delta.to(DEV)
        h_rank = 10**9; topP = 0; steps = 0; held_pos = None; held_set = set()
        h_ids = set();
        for w in HELD_OUT.split():
            sp = tok(' '+w, add_special_tokens=False).input_ids
            if len(sp) == 1: h_ids.add(int(sp[0]))
        with torch.no_grad():
            for step in range(NTOK):
                steps += 1
                vc = {}
                hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                hk.remove()
                if past is None: past = out.past_key_values
                L0 = out.logits[0,-1,:].float()
                on = (step >= SW0)
                L1 = L0 + (dd if on else 0.0)
                if REP_PEN > 0 and rep_hist:
                    L1 = L1.clone(); L1[rep_hist[-30:]] -= REP_PEN
                order = L1.argsort(descending=True).tolist()
                pos = {tid: kk for kk, tid in enumerate(order)}
                mr = min(pos[i] for i in h_ids) if h_ids else 10**9
                h_rank = min(h_rank, mr)
                p = torch.softmax(L1, 0)
                q = p.clone(); ooo = q.argsort(descending=True)
                kk = int((q[ooo].cumsum(0) <= NUCLEUS).sum()) + 1
                top_p = set(ooo.tolist()[:kk])
                if h_ids & top_p: topP += 1
                msk = torch.zeros_like(q); msk[ooo[:kk]] = 1
                qq = (q*msk); qq = qq/qq.sum()
                nxt = int(torch.multinomial(qq, 1))
                rep_hist.append(int(nxt))
                if nxt in h_ids:
                    held_set.add(tok.decode([nxt]).strip())
                    held_pos = held_pos if held_pos is not None else steps
                if nxt == eos_id: break
                out_ids.append(nxt)
                ids = torch.tensor([[nxt]], device=DEV)
        txt = tok.decode(out_ids)
        h,b,maxr,dist1,ok = score(txt)
        return dict(txt=txt, h=h, ok=ok, minHrank=h_rank, topPrate=topP/max(1,steps),
                    held_pos=held_pos, held_vocab=sorted(held_set),
                    maxrun=maxr, dist1=dist1, anchors=b)
    print(f'\n  {"cond":>12} {"transport":>9} {"minHrank":>8} {"topP":>5} {"hpos":>5} {"maxrun":>6} {"dist1":>6}  held-vocab')
    for name, delta in deltas.items():
        R = 0; agg = {k: 0 for k in ('minHrank','topPrate','maxrun','dist1')}
        held_all = set()
        for s in range(K):
            m = generate(delta, SEEDBASE + s)
            R += int(m['ok'])
            agg['minHrank'] = min(agg['minHrank'], m['minHrank']) if s else m['minHrank']
            agg['topPrate'] += m['topPrate']; agg['maxrun'] += m['maxrun']; agg['dist1'] += m['dist1']
            if m['held_pos'] is not None: held_all.add(f"@{m['held_pos']}")
        for k in ('topPrate','maxrun','dist1'): agg[k] /= max(1,K)
        hv = ', '.join(str(x) for x in list(held_all)[:6])
        print(f'  {name:>12} {R:3d}/{K:<4} {agg["minHrank"]:8d} {agg["topPrate"]:5.2f} '
              f'{hv[:14]:>14} {agg["maxrun"]:6.1f} {agg["dist1"]:6.2f}', flush=True)
    print(f'[{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()