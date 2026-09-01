#!/usr/bin/env python3
"""mechanism_matrix.py — Causal factorial: WHICH nonlinear operation turns a
row(W)-representable contrast into a generatively effective control signal?

Normalization axis (the per-sentence z construction is the working reference):
  raw   : dL = mean_tgt(L_s) - mean_neu(L_s)                  (no centering/z)
  cent  : raw, then centered x - mean(x)
  zs    : raw, then z-scored over all 152k dims
  perz  : mean_tgt( zscore(L_s - mean_neu) )      (gen_geom's working recipe)

Mask axis (positive top-k selection):
  none / top25 / top50 / top200

Controls (at top200 on the perz base, all matched-norm):
  rand200      : 200 uniform-random coords, their own (small) values
  magmatch200  : 200 random coords, assigned the SORTED top-200 magnitudes
  shuffle200   : the real top-200 coords, values PERMUTED among them
  equal200     : the real top-200 coords, all = +const (norm-matched)
  rowW_proj    : full rank-1536 projection of perz@top200 onto row(W)

Dose is MATCHED: every condition is rescaled to ||ALPHA * perz * top200||
(the known-working effective norm), so any difference is the mechanism, not
dose. Every cell reports:
  transport/N, held|anchor|unrel dLogP, maxrun, dist1,
  row-space fraction R = ||P_rowW(delta)||^2/||delta||^2.

Phase A (pure vector math, no generation): row-space fraction vs K for perz
and zs. Phase B (generation): the full matrix.

Run:
  python3 mechanism_matrix.py [seed]   # Phase A + Phase B (SEEDS=6 default)
  SEEDS=24 python3 mechanism_matrix.py # paper-quality numeric
  PHASEA=1 python3 mechanism_matrix.py # Phase A only
"""
import os, sys, time, re
from collections import Counter
import torch, transformers

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PHASEA = os.environ.get('PHASEA', '0') == '1'
MODEL = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
NTOK = int(os.environ.get('NTOK', '120'))
SEEDS = int(os.environ.get('SEEDS', '6'))
NUCLEUS = float(os.environ.get('NUCLEUS', '0.9'))
ALPHA = float(os.environ.get('ALPHA', '2.0'))
SW0 = int(os.environ.get('SW0', '20'))
SEEDBASE = int(os.environ.get('SEEDBASE', '0'))
KSWEEP = [int(x) for x in os.environ.get(
    'KSWEEP', '1,2,5,10,25,50,100,200,500,1000,5000,0').split(',')]  # 0=V
MASKS = [int(x) for x in os.environ.get('MASKS', '0,25,50,200').split(',')]
PROMPT = os.environ.get('PROMPT', 'The waves crashed gently on the beach')

if os.environ.get('ROYAL') == '1':
    TGT = 'The king entered the castle|The queen sat upon the throne|' \
          'The prince inherited the crown|The royal family gathered in the great hall'
    NEU = 'The person walked down the street|The individual entered the room|' \
          'The worker picked up the package'
    HELD_OUT = 'crown reign kingdom realm monarchy monarch dynasty heir sovereign'
    ANCHORS = 'king queen prince royal princess'
    UNREL = 'sand wave sea swim ocean surf beach tide shore shell'
else:
    TGT = ('A dragon circled the ruined towers of the ancient kingdom|'
           'A knight drew his sword against the fire-breathing beast|'
           "The wizard's spell shattered the castle gates")
    NEU = ('The waves crashed gently on the beach|'
           'The sand was cool to the touch|The sun was warm over the water')
    HELD_OUT = 'creature creatures evil monsters beast horror lurking nightmare demons'
    ANCHORS = ''
    UNREL = 'sand wave sea swim ocean surf beach tide shore shell'

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
    print(f'mechanism_matrix seed={SEED} SEEDS={SEEDS} '
          f'{"ROYAL" if os.environ.get("ROYAL")=="1" else "FANTASY"} '
          f'SW0={SW0} ALPHA={ALPHA}', flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    norm = model.model.norm
    W = model.lm_head.weight.detach().float().cpu(); Wt = W.t()
    eos_id = int(tok.eos_token_id)
    V = W.shape[0]

    def logits_of(s):
        t = tok(s, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        with torch.no_grad():
            return model(t).logits[0,-1,:].float().cpu()

    tgt = [x.strip() for x in TGT.split('|') if x.strip()]
    neu = [x.strip() for x in NEU.split('|') if x.strip()]

    nm = torch.stack([logits_of(s) for s in neu]).mean(0)

    # normalization variants of the contrast (before top-k)
    raw = None
    for s in tgt:
        Ls = logits_of(s)
        raw = Ls if raw is None else raw + Ls
    raw = raw / max(1, len(tgt)) - nm
    cent = raw - raw.mean()
    zs = (raw - raw.mean()) / (raw.std() + 1e-6)
    perz_sum = None
    for s in tgt:
        Ls = logits_of(s)
        c = Ls - nm
        c = (c - c.mean()) / (c.std() + 1e-6)
        perz_sum = c if perz_sum is None else perz_sum + c
    perz = perz_sum / max(1, len(tgt))
    # gen_geom re-z-scores the per-sentence z-SUM before masking (line 450)
    perz = (perz - perz.mean()) / (perz.std() + 1e-6)

    # reference: the known-working vector (norm-matched dose for ALL)
    m200 = torch.zeros(V); m200[perz.argsort(descending=True)[:200]] = 1.0
    dL_ref = ALPHA * perz * m200
    N_REF = dL_ref.norm()
    print(f'  N_REF (working effective norm) = {N_REF:.3f}', flush=True)

    # row-space projection operator (P_rowW = W (W^T W)^-1 W^T)
    WtW = W.t() @ W
    def proj_rowW(v):
        x = torch.linalg.solve(WtW, W.t() @ v)
        return W @ x
    def row_frac(v):
        if v.norm() < 1e-9:
            return float('nan')
        p = proj_rowW(v)
        return (p.norm()**2 / v.norm()**2).item()

    # ---------- PHASE A: row-space fraction vs K ----------
    print('\n  == PHASE A: row-space fraction R_K = ||P_rowW M_K(dL)||^2/||M_K||^2 ==')
    print(f'  {"K":>6}  {"perz":>8} {"zs":>8}   ("residual" = 1 - R_K)')
    for kk in KSWEEP:
        def maskof(v, k):
            if k <= 0 or k >= V:
                return v.clone()
            m = torch.zeros(V); m[v.argsort(descending=True)[:k]] = 1.0
            return v * m
        mk = kk if kk > 0 else V
        rp = row_frac(maskof(perz, kk)); rz = row_frac(maskof(zs, kk))
        tag = f'{kk if kk>0 else "V"}'
        print(f'  {tag:>6}  {rp:8.4f} {rz:8.4f}   (resid_perz={1-rp:.3f} resid_zs={1-rz:.3f})')
    if PHASEA:
        print(f'[{time.time()-t0:.0f}s]')
        return

    # ---------- PHASE B: build the delta bank (all rescaled to N_REF) ----------
    def rescale(v):
        n = v.norm()
        return v * (N_REF / n) if n > 1e-9 else v
    def topk_pos(v, k):
        if k <= 0 or k >= V:
            return v.clone()
        m = torch.zeros(V); m[v.argsort(descending=True)[:k]] = 1.0
        return v * m

    bank = {}
    norms = {'raw': raw, 'cent': cent, 'zs': zs, 'perz': perz}
    for nn_, vv in norms.items():
        for kk in MASKS:
            bank[f'{nn_}{"" if kk==0 else f"_t{kk}"}'] = rescale(topk_pos(vv, kk))
    # controls at top200 on the perz base
    idx200 = perz.argsort(descending=True)[:200]
    vals200 = perz[idx200]
    # rand200: 200 random coords, their own values
    gi = torch.Generator().manual_seed(1234 + SEED)
    ridx = torch.randperm(V, generator=gi)[:200]
    v_r = torch.zeros(V); v_r[ridx] = perz[ridx]
    bank['rand200'] = rescale(v_r)
    # magmatch200: 200 random coords, sorted top-200 magnitudes
    v_m = torch.zeros(V); v_m[ridx] = torch.sort(vals200.abs(), descending=True).values
    bank['magmatch200'] = rescale(v_m)
    # shuffle200: real coords, values permuted among them
    perm = torch.randperm(200, generator=torch.Generator().manual_seed(999 + SEED))
    v_s = torch.zeros(V); v_s[idx200] = vals200[perm]
    bank['shuffle200'] = rescale(v_s)
    # equal200: real coords, all = +const
    v_e = torch.zeros(V); v_e[idx200] = 1.0
    bank['equal200'] = rescale(v_e)
    # rowW_proj: full projection of the working vector
    bank['rowW_proj'] = rescale(proj_rowW(dL_ref))

    print(f'\n  == PHASE B: {len(bank)} conditions x {SEEDS} seeds x {NTOK} tok ==')
    print(f'  {"cond":>12} {"transport":>9} {"dLogP_H":>8} {"dLogP_A":>8} '
          f'{"dLogP_U":>8} {"maxrun":>6} {"dist1":>6} {"R_row":>6}')

    h_ids = set()
    for w in HELD_OUT.split():
        sp = tok(' '+w, add_special_tokens=False).input_ids
        if len(sp) == 1: h_ids.add(int(sp[0]))
    a_ids = set()
    for w in ANCHORS.split():
        sp = tok(' '+w, add_special_tokens=False).input_ids
        if len(sp) == 1: a_ids.add(int(sp[0]))
    u_ids = set()
    for w in UNREL.split():
        sp = tok(' '+w, add_special_tokens=False).input_ids
        if len(sp) == 1: u_ids.add(int(sp[0]))

    def generate(delta, seed):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        past = None; out_ids = []; dd = delta.to(DEV)
        steps = 0; dH = 0.0; dA = 0.0; dU = 0.0; nH = max(1, len(h_ids)); nA = max(1, len(a_ids)); nU = max(1, len(u_ids))
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
                if on:
                    p0 = torch.softmax(L0, 0); p1 = torch.softmax(L1, 0)
                    lp1 = p1.clamp_min(1e-12).log(); lp0 = p0.clamp_min(1e-12).log()
                    ddH = lp1 - lp0
                    if h_ids: dH += sum(ddH[i].item() for i in h_ids)/nH
                    if a_ids: dA += sum(ddH[i].item() for i in a_ids)/nA
                    dU += sum(ddH[i].item() for i in u_ids)/nU
                p = torch.softmax(L1, 0)
                q = p.clone(); ooo = q.argsort(descending=True)
                kk = int((q[ooo].cumsum(0) <= NUCLEUS).sum()) + 1
                msk = torch.zeros_like(q); msk[ooo[:kk]] = 1
                qq = (q*msk); qq = qq/qq.sum()
                nxt = int(torch.multinomial(qq, 1))
                if nxt == eos_id: break
                out_ids.append(nxt)
                ids = torch.tensor([[nxt]], device=DEV)
        txt = tok.decode(out_ids)
        h,b,maxr,dist1,ok = score(txt)
        return dict(txt=txt, ok=ok, dH=dH/max(1,steps), dA=dA/max(1,steps),
                    dU=dU/max(1,steps), maxr=maxr, dist1=dist1)

    for name, delta in bank.items():
        R = 0; agg = {'dH':0., 'dA':0., 'dU':0., 'maxr':0., 'dist1':0.}
        for s in range(SEEDS):
            m = generate(delta, SEEDBASE + s)
            R += int(m['ok'])
            agg['dH'] += m['dH']; agg['dA'] += m['dA']; agg['dU'] += m['dU']
            agg['maxr'] += m['maxr']; agg['dist1'] += m['dist1']
        for k in agg: agg[k] /= max(1, SEEDS)
        rf = row_frac(delta)
        print(f'  {name:>12} {R:3d}/{SEEDS:<4} {agg["dH"]:+8.2f} {agg["dA"]:+8.2f} '
              f'{agg["dU"]:+8.2f} {agg["maxr"]:6.1f} {agg["dist1"]:6.2f} {rf:6.3f}',
              flush=True)
    print(f'[{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()