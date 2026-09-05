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
REPL = os.environ.get('REPL', '0') == '1'     # 6-cell replication mode
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

    # helpers used by both SWEEP and PHASE B
    def rescale(v):
        n = v.norm()
        return v * (N_REF / n) if n > 1e-9 else v
    def topk_pos(v, k):
        if k <= 0 or k >= V:
            return v.clone()
        m = torch.zeros(V); m[v.argsort(descending=True)[:k]] = 1.0
        return v * m

    if os.environ.get('SWEEP') == '1':
        # ------- continuous-K transport sweep (generation) -------
        print('\n  == CONTINUOUS-K TRANSPORT SWEEP (perz base) ==')
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
        nH = max(1,len(h_ids)); nA = max(1,len(a_ids)); nU = max(1,len(u_ids))

        def genK(kk, seed):
            torch.manual_seed(seed)
            ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
            past = None; out_ids = []; dd = rescale(topk_pos(perz, kk)).to(DEV)
            minr = 10**9
            with torch.no_grad():
                for step in range(NTOK):
                    vc = {}
                    hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
                    out = model(input_ids=ids, past_key_values=past, use_cache=True)
                    hk.remove()
                    if past is None: past = out.past_key_values
                    L0 = out.logits[0,-1,:].float()
                    on = (step >= SW0)
                    L1 = L0 + (dd if on else 0.0)
                    order1 = L1.argsort(descending=True).tolist()
                    pos1 = {tid: k for k, tid in enumerate(order1)}
                    if h_ids: minr = min(minr, min(pos1[i] for i in h_ids))
                    p = torch.softmax(L1, 0)
                    q = p.clone(); ooo = q.argsort(descending=True)
                    kk2 = int((q[ooo].cumsum(0) <= NUCLEUS).sum()) + 1
                    msk = torch.zeros_like(q); msk[ooo[:kk2]] = 1
                    qq = (q*msk); qq = qq/qq.sum()
                    nxt = int(torch.multinomial(qq, 1))
                    if nxt == eos_id: break
                    out_ids.append(nxt)
                    ids = torch.tensor([[nxt]], device=DEV)
            txt = tok.decode(out_ids)
            _,_,_,_,ok = score(txt)
            return ok, minr

        print(f'  {"K":>6} {"transport":>9} {"medMinR":>8} {"R_row":>7}  cos->dLfull')
        for kk in [10,25,50,100,200,500,1000,2000,0]:
            R = 0; mrs = []
            for s in range(SEEDS):
                ok, mr = genK(kk, SEEDBASE + s)
                R += int(ok); mrs.append(mr)
            med = sorted(mrs)[SEEDS//2]
            dk = rescale(topk_pos(perz, kk))
            rf = row_frac(dk)
            cs = (dk @ dL_ref).item()/(dk.norm()*N_REF) if dk.norm() > 1e-9 else float('nan')
            print(f'  {kk if kk>0 else "V":>6} {R:3d}/{SEEDS:<4} {med:8d} {rf:7.3f}  {cs:+.3f}', flush=True)
        print(f'[{time.time()-t0:.0f}s]')
        return

    if os.environ.get('LAMBDA') == '1':
        # ------- lambda interpolation: rowW-proj (lam=0) -> residual (lam=1) ----
        print('\n  == LAMBDA INTERPOLATION (constant norm) ==')
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
        # dL_full (the working top-200). Compute its row-space projection and
        # its orthogonal residual.
        p_roww = proj_rowW(dL_ref)
        resid = dL_ref - p_roww
        print(f'  ||P_rowW(dL)||={p_roww.norm():.3f}  ||residual||={resid.norm():.3f}  '
              f'||dL||={dL_ref.norm():.3f}')

        def genVec(vec, seed):
            torch.manual_seed(seed)
            dd = rescale(vec).to(DEV)
            ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
            past = None; out_ids = []; minr = 10**9
            with torch.no_grad():
                for step in range(NTOK):
                    vc = {}
                    hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
                    out = model(input_ids=ids, past_key_values=past, use_cache=True)
                    hk.remove()
                    if past is None: past = out.past_key_values
                    L0 = out.logits[0,-1,:].float()
                    on = (step >= SW0)
                    L1 = L0 + (dd if on else 0.0)
                    order1 = L1.argsort(descending=True).tolist()
                    pos1 = {tid: k for k, tid in enumerate(order1)}
                    if h_ids: minr = min(minr, min(pos1[i] for i in h_ids))
                    p = torch.softmax(L1, 0)
                    q = p.clone(); ooo = q.argsort(descending=True)
                    ktt = int((q[ooo].cumsum(0) <= NUCLEUS).sum()) + 1
                    msk = torch.zeros_like(q); msk[ooo[:ktt]] = 1
                    qq = (q*msk); qq = qq/qq.sum()
                    nxt = int(torch.multinomial(qq, 1))
                    if nxt == eos_id: break
                    out_ids.append(nxt)
                    ids = torch.tensor([[nxt]], device=DEV)
            txt = tok.decode(out_ids)
            _,_,_,_,ok = score(txt)
            return ok, minr

        # ---- K x lambda causal surface ----
        # For each K, d_K = topk_pos(perz, K) defines the chosen sparse coord
        # set. Decompose d_K = proj_K + resid_K (onto/out of row(W)), then
        #   vec(K, lam) = (1-lam)*resid_K + lam*proj_K, NORM-MATCHED to N_REF
        # (= dL_ref) so dose is held fixed across the whole surface.
        # lam=0 -> pure residual (out-of-row); lam=1 -> pure rowW projection.
        klist = [int(x) for x in os.environ.get('KLIST', '100,150,200,250,300').split(',')]
        llist = [float(x) for x in os.environ.get('LLIST', '0,0.5,1').split(',')]
        print(f'\n  == K x lambda causal surface  (K={klist}, lam={llist}) ==')
        print(f'  {"K":>5} {"lam":>4} {"transport":>9} {"medMinR":>8} {"R_row":>7} {"cos_ref":>7}')
        for kk in klist:
            for lam in llist:
                dk = topk_pos(perz, kk)
                pk = proj_rowW(dk)
                rk = dk - pk
                vec = (1-lam)*rk + lam*pk
                vec = rescale(vec)          # fixed dose across all cells
                R = 0; mrs = []
                for s in range(SEEDS):
                    ok, mr = genVec(vec, SEEDBASE + s)
                    R += int(ok); mrs.append(mr)
                med = sorted(mrs)[SEEDS//2] if mrs else -1
                rf = row_frac(vec)
                cs = (vec @ dL_ref).item()/(vec.norm()*N_REF) if vec.norm()>1e-9 else float('nan')
                print(f'  {kk:5d} {lam:4.2f} {R:3d}/{SEEDS:<4} {med:8d} {rf:7.3f} {cs:+7.3f}', flush=True)
        print(f'[{time.time()-t0:.0f}s]')
        return

    # ---------- PHASE B: build the delta bank (all rescaled to N_REF) ----------

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
        min_rank = 10**9; min_rank_pct = 1.0; top10 = top50 = top100 = 0
        first_entry = None; cum_logp_H = 0.0
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
                    if h_ids: cum_logp_H += sum(lp1[i].item() for i in h_ids)/nH
                order1 = L1.argsort(descending=True).tolist()
                pos1 = {tid: k for k, tid in enumerate(order1)}
                if h_ids:
                    mr = min(pos1[i] for i in h_ids)
                    min_rank = min(min_rank, mr)
                    min_rank_pct = min(min_rank_pct, mr/len(order1))
                    top10 += sum(1 for i in h_ids if pos1[i] < 10)
                    top50 += sum(1 for i in h_ids if pos1[i] < 50)
                    top100 += sum(1 for i in h_ids if pos1[i] < 100)
                    if first_entry is None and mr < 100:
                        first_entry = steps
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
                    dU=dU/max(1,steps), maxr=maxr, dist1=dist1,
                    min_rank=min_rank, min_rank_pct=min_rank_pct,
                    top10=top10, top50=top50, top100=top100,
                    first_entry=first_entry, cum_logp_H=cum_logp_H/max(1,steps))

    keys = ['raw_t200', 'perz_t200', 'shuffle200', 'equal200', 'rowW_proj', 'rand200'] if REPL else list(bank.keys())
    print(f'\n  {"cond":>12} {"transport":>9} {"medMinR":>8} {"med%":>6} {"t10":>4} {"t50":>4} {"t100":>5} {"fstE":>5} {"cumdH":>7} {"maxrun":>6} {"dist1":>6} {"R_row":>6}')
    for name in keys:
        delta = bank[name]
        R = 0; agg = {'dH':0., 'dA':0., 'dU':0., 'maxr':0., 'dist1':0.,
                      'minr':[], 'pct':[], 't10':0, 't50':0, 't100':0,
                      'fst':[], 'cum':0.}
        for s in range(SEEDS):
            m = generate(delta, SEEDBASE + s)
            R += int(m['ok'])
            agg['dH'] += m['dH']; agg['dA'] += m['dA']; agg['dU'] += m['dU']
            agg['maxr'] += m['maxr']; agg['dist1'] += m['dist1']
            agg['minr'].append(m['min_rank']); agg['pct'].append(m['min_rank_pct'])
            agg['t10'] += m['top10']; agg['t50'] += m['top50']; agg['t100'] += m['top100']
            if m['first_entry'] is not None: agg['fst'].append(m['first_entry'])
            agg['cum'] += m['cum_logp_H']
        for k in ('dH','dA','dU','maxr','dist1','cum'): agg[k] /= max(1, SEEDS)
        mr_med = sorted(agg['minr'])[SEEDS//2] if agg['minr'] else -1
        pct_med = sorted(agg['pct'])[SEEDS//2] if agg['pct'] else -1
        fst_med = sorted(agg['fst'])[len(agg['fst'])//2] if agg['fst'] else -1
        rf = row_frac(delta)
        print(f'  {name:>12} {R:3d}/{SEEDS:<4} {mr_med:8d} {pct_med:6.3f} {agg["t10"]:4d} {agg["t50"]:4d} {agg["t100"]:5d} {fst_med:5d} {agg["cum"]:+7.2f} {agg["maxr"]:6.1f} {agg["dist1"]:6.2f} {rf:6.3f}',
              flush=True)
    print(f'[{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()