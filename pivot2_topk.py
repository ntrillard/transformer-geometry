#!/usr/bin/env python3
"""pivot2_topk.py — Decisive test of the sparsity hypothesis.

Hypothesis:
    failure of W d_per  ~  diffuse readout (energy spread across ~147k rows)
    success of logit contrast ~ sparse continuation-space intervention (top-k)

Decisive experiment: take the SAME d_per, compute W d_per, keep only its
top-k logit coordinates, add to live logits — does transport emerge where
the diffuse W d_per failed?

Magnitude is MATCHED: every condition's logit delta is re-scaled to the
working dL_static's effective logit norm (||ALPHA * dL_static||_2), so a
positive masked result cannot be a dose artifact.

Two phases:
  Phase A (cheap, no generation): K-sweep + sign test over
     K in {10,25,50,100,200,500,1000,V}
     positive-top-k vs absolute-top-k
     report held-out rank, anchor rank, dL_H, dL_A, #affected logits.
  Phase B (generation): matched-norm conditions (control GEN=1)
     dL_static (reference, expected transport)
     raw Wd_per          (expected NO transport)
     top200p Wd_per      (positive top-k; prediction: transport)
     rand top200 Wd_per  (random-complement -> W@ -> top-k; expected no/weak)
     perm top200 Wd_per  (real top-k coords, values permuted; expected no)
     (optional abs-top-k via ABS=1)

Run: HF_TOKEN=... python3 pivot2_topk.py [seed]      # phase A
     HF_TOKEN=... GEN=1 python3 pivot2_topk.py [seed] # phase B
"""
import os, sys, time, re
from collections import Counter
import torch, transformers

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
GEN = os.environ.get('GEN', '0') == '1'
MODEL = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
PROMPT = os.environ.get('PROMPT', 'The waves crashed gently on the beach')
NTOK = int(os.environ.get('NTOK', '110'))
K = int(os.environ.get('K', '4'))                 # generation samples per cond
NUCLEUS = float(os.environ.get('NUCLEUS', '0.9'))
ALPHA = float(os.environ.get('ALPHA', '2.0'))
NTOPK = int(os.environ.get('NTOPK', '200'))       # default top-k for phase B
ABS = os.environ.get('ABS', '0') == '1'           # abs-top-k vs positive-top-k
KSWEEP = [int(x) for x in os.environ.get(
    'KSWEEP', '10,25,50,100,200,500,1000,0').split(',')]  # 0 => full V
REP_PEN = float(os.environ.get('REP_PEN', '0.0')) # gen_geom default: 0
REP_WINDOW = int(os.environ.get('REP_WINDOW', '30'))
REP_COUNT = os.environ.get('REP_COUNT', '1') == '1'  # count-scaled penalty
SW0 = int(os.environ.get('SW0', '20'))      # gen_geom: steer from step 20
CW = int(os.environ.get('CW', '0'))          # CONTRAST_WINDOW: 0 = every step from SW0
SEEDBASE = int(os.environ.get('SEEDBASE', '3000'))  # seed base for generation

TGT = os.environ.get('TGT', 'The king entered the castle|The queen sat upon the throne|' 
      'The prince inherited the crown|The royal family gathered in the great hall')
NEU = os.environ.get('NEU', 'The person walked down the street|The individual entered the room|' 
      'The worker picked up the package')
ANCHORS = os.environ.get('ANCHORS', 'king queen prince royal princess')
HELD_OUT = os.environ.get('HELD_OUT', 'crown reign kingdom realm monarchy monarch dynasty heir coronation sovereign')
NEUT = os.environ.get('NEUT', 'sand wave sea swim ocean surf beach tide shore shell')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def sids(tok, words):
    ids, multi = {}, []
    for w in words.split():
        sp = tok(' ' + w, add_special_tokens=False).input_ids
        if len(sp) == 1:
            ids[w] = int(sp[0])
        else:
            multi.append(w)
    return ids, multi


def topk_mask(z, kk, use_abs):
    """sparse mask over logit coords: kk entries (0 => full)."""
    if kk <= 0 or kk >= z.numel():
        return torch.zeros_like(z)  # full => no mask (handled by caller)
    if use_abs:
        order = z.abs().argsort(descending=True)
    else:
        order = z.argsort(descending=True)
    m = torch.zeros_like(z); m[order[:kk]] = 1
    return m


def score(txt):
    toks = [w.strip('.,!?;:()[]"').lower() for w in re.findall(r"\S+", txt)]
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
    print(f'pivot2_topk seed={SEED} GEN={GEN} ALPHA={ALPHA} K={K}', flush=True)
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

    def state(s):
        t = tok(s, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        vc = {}
        hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
        with torch.no_grad(): model(t)
        hk.remove()
        return vc['o'].cpu().float()

    def logits_of(s):
        t = tok(s, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        with torch.no_grad():
            L = model(t).logits[0,-1,:].float()
        return L.cpu().float()

    tgt = [s.strip() for s in TGT.split('|') if s.strip()]
    neu = [s.strip() for s in NEU.split('|') if s.strip()]
    Ts = torch.stack([state(s) for s in tgt]).mean(0)
    Ns = torch.stack([state(s) for s in neu]).mean(0)
    d = Ts - Ns
    Ucd = Uc.to(d.device)
    dper = d - Ucd @ (Ucd.t() @ d); dper = dper / dper.norm()
    print(f'  dper shell-leak={(Ucd@(Ucd.t()@dper)).norm():.4f}', flush=True)

    g = torch.Generator().manual_seed(777 + SEED)
    z = torch.randn(1536, generator=g).float()
    zc = z - Ucd @ (Ucd.t() @ z); Rdper = zc / zc.norm()
    print(f'  random-complement shell-leak={(Ucd@(Ucd.t()@Rdper)).norm():.4f}', flush=True)

    held_id, held_multi = sids(tok, HELD_OUT)
    anchor_id, _ = sids(tok, ANCHORS)
    neut_id, _ = sids(tok, NEUT)
    nH = len(held_id); nA = len(anchor_id)
    print(f'  HELD_OUT={list(held_id)}  ANCHORS={list(anchor_id)}', flush=True)

    # --- build the interventions in LOGIT space (all on CPU float) ---
    # working logit contrast, FAITHFUL port of gen_geom's legacy path:
    #   1. nm = mean next-token logits over the neutral sentence set
    #   2. per-sentence z-sum: for each target sentence, c = zscore(Ls - nm);
    #      dL = mean over target sentences of c
    #   3. z-score the sum, mask to top-200 most-positive, scale by ALPHA.
    #      (gen_geom lines 280-302 + 450 + 463-466)
    nm = torch.stack([logits_of(s) for s in neu]).mean(0)
    dL_raw = None
    for s in tgt:
        Ls = logits_of(s)
        c = Ls - nm
        c = (c - c.mean()) / (c.std() + 1e-6)
        dL_raw = c if dL_raw is None else dL_raw + c
    dL_raw = dL_raw / max(1, len(tgt))
    dz = (dL_raw - dL_raw.mean()) / (dL_raw.std() + 1e-6)
    m = topk_mask(dz, NTOPK, use_abs=False)
    dL_static_raw = ALPHA * dz * m
    N_target = dL_static_raw.norm()          # == effective logit norm of working dL
    pre = dL_static_raw.norm() / ALPHA
    top5 = [tok.decode([int(i)]) for i in dL_static_raw.argsort(descending=True)[:5]]
    print(f'  |dL_z|(pre-ALPHA)={pre:.2f}  top5={top5}', flush=True)
    print(f'  working dL_static logit-norm = {N_target:.3f} (matched reference)', flush=True)

    # W @ d_per in logit space (diffuse)
    Wdper_full = (dper.cpu() @ Wt.cpu())
    # random-complement -> W@
    Wrdper_full = (Rdper.cpu() @ Wt.cpu())

    def rescale(vec):
        n = vec.norm()
        return vec * (N_target / n) if n > 0 else vec

    # ---- PHASE A: cheap K sweep + sign test (no generation) ----
    if not GEN:
        print('\n  == PHASE A: K sweep + sign test (held-out rank in intervened dist) ==')
        print(f'  {"K":>6} {"sign":>4} {"dL_H":>8} {"dL_A":>8} {"sel":>6} '
              f'{"minH":>5} {"minA":>5} {"#aff":>7}')
        for kk in KSWEEP:
            for use_abs in ([False, True] if ABS else [False]):
                mm = topk_mask(Wdper_full, kk, use_abs)
                delta = rescale(Wdper_full * mm)
                # cheap eval on ONE continuation state (baseline "The")
                ids0 = tok('The', add_special_tokens=False,
                           return_tensors='pt').input_ids.to(DEV)
                vc = {}
                hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
                with torch.no_grad(): L0 = model(ids0).logits[0,-1,:].float()
                hk.remove()
                L1 = L0 + delta.to(L0.device)
                dH = sum((L1[i]-L0[i]).item() for i in held_id.values())/nH
                dA = sum((L1[i]-L0[i]).item() for i in anchor_id.values())/nA
                sel = dH/dA if abs(dA) > 1e-6 else float('nan')
                order = L1.argsort(descending=True).tolist()
                pos = {tid: k for k, tid in enumerate(order)}
                mH = min(pos[i] for i in held_id.values())
                mA = min(pos[i] for i in anchor_id.values())
                naffected = int((delta.abs() > 0.1).sum().item())
                print(f'  {kk if kk>0 else "V":>6} {"abs" if use_abs else "pos":>4} '
                      f'{dH:+8.2f} {dA:+8.2f} {sel:6.2f} {mH:5d} {mA:5d} {naffected:7d}')
        print(f'  [REF] dL_static(work): dH/A see phase B')
        print(f'[{time.time()-t0:.0f}s]')
        return

    # ---- PHASE B: generation, matched norm ----
    conds = {}
    conds['dL_static']   = dL_static_raw                                   # reference (work)
    conds['raw_Wdper']   = rescale(Wdper_full)                             # diffuse
    conds[f'top{NTOPK}p'] = rescale(Wdper_full * topk_mask(Wdper_full, NTOPK, use_abs=False))
    conds['rand_top']    = rescale(Wrdper_full * topk_mask(Wrdper_full, NTOPK, use_abs=False))
    # permutation control: real top-k coords, values shuffled among them
    mm = topk_mask(Wdper_full, NTOPK, use_abs=False)
    idx = mm.nonzero().flatten()
    vals = Wdper_full[idx].clone()
    perm = torch.randperm(vals.numel(), generator=torch.Generator().manual_seed(999+SEED))
    Wperm = Wdper_full.clone(); Wperm[idx] = vals[perm]
    conds['perm_top']    = rescale(Wperm * mm)
    # sparsity sweep + sign controls for the secondary (supA) run
    conds['top25p'] = rescale(Wdper_full * topk_mask(Wdper_full, 25, use_abs=False))
    conds['top50p'] = rescale(Wdper_full * topk_mask(Wdper_full, 50, use_abs=False))
    conds['rand25p'] = rescale(Wrdper_full * topk_mask(Wrdper_full, 25, use_abs=False))
    conds['m25p']   = rescale((-Wdper_full) * topk_mask(-Wdper_full, 25, use_abs=False))

    def generate_delta(delta, seed, supA=False):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        past = None
        out_ids = []
        rep_hist = []
        dd = delta.to(DEV)
        a_ids = list(anchor_id.values())
        h_ids = list(held_id.values())
        top_p_held_steps = 0; held_sampled = 0; steps = 0
        with torch.no_grad():
            for _ in range(NTOK):
                steps += 1
                vc = {}
                hk = norm.register_forward_hook(lambda m,i,o: vc.__setitem__('o', o[0,-1,:].float().clone()))
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                hk.remove()
                if past is None: past = out.past_key_values
                L0 = out.logits[0,-1,:].float()
                on = (steps >= SW0) and (CW == 0 or steps < SW0 + CW)
                L1 = L0 + (dd if on else 0.0)
                # IDENTICAL post-intervention suppression operator for every
                # condition (secondary test): same norm preserved, same -inf mask.
                if supA:
                    L1 = L1.clone(); L1[a_ids] = -1e9
                # IDENTICAL post-intervention suppression operator for every
                # condition (secondary test): same norm preserved, same -inf mask.
                if supA:
                    L1 = L1.clone(); L1[a_ids] = -1e9
                # anti-priming de-latch (faithful port of gen_geom REP_PEN)
                if REP_PEN > 0 and rep_hist:
                    L1 = L1.clone()
                    recent = rep_hist[-REP_WINDOW:]
                    if REP_COUNT:
                        rt = torch.tensor(recent, device=L1.device)
                        uniq, cnts = torch.unique(rt, return_counts=True)
                        pen = torch.zeros_like(L1); pen[uniq] = REP_PEN * cnts.float()
                    else:
                        pen = torch.zeros_like(L1); pen[recent] = REP_PEN
                    L1 = L1 - pen
                p = torch.softmax(L1, 0)
                q = p.clone(); ooo = q.argsort(descending=True)
                kk = int((q[ooo].cumsum(0) <= NUCLEUS).sum()) + 1
                top_p = set(ooo.tolist()[:kk])
                if any(i in top_p for i in h_ids):
                    top_p_held_steps += 1
                msk = torch.zeros_like(q); msk[ooo[:kk]] = 1
                qq = (q*msk); qq = qq/qq.sum()
                nxt = int(torch.multinomial(qq, 1))
                rep_hist.append(int(nxt))
                if nxt in h_ids: held_sampled += 1
                if nxt == eos_id: break
                out_ids.append(nxt)
                ids = torch.tensor([[nxt]], device=DEV)
        m = dict(top_p_held_rate=top_p_held_steps/max(1,steps),
                 held_sampled=held_sampled, steps=steps)
        return tok.decode(out_ids), m

    def run_table(label, supA, names):
        print(f'\n  == PHASE B [{label}]{" (anchor-suppressed)" if supA else ""} ==')
        print(f'  {"cond":>12} {"transport":>9} {"topPheld":>9} {"heldSmp":>7}  sample(trunc)')
        for name in names:
            R = 0; tp_rates = []; hS = 0; samples = []
            for s in range(K):
                txt, m = generate_delta(conds[name], SEEDBASE + s, supA=supA)
                _,_,_,_,ok = score(txt)
                R += int(ok)
                tp_rates.append(m['top_p_held_rate'])
                hS += m['held_sampled']
                samples.append(txt[:55].replace('\n', ' '))
            rt = sum(tp_rates)/max(1,len(tp_rates))
            print(f'  {name:>12} {R:3d}/{K:<5} {rt:9.2f} {hS:7d}  {samples[0]!r}', flush=True)

    primary = ['dL_static', 'raw_Wdper', f'top{NTOPK}p', 'rand_top', 'perm_top']
    run_table('PRIMARY (no suppression)', False, primary)
    secondary = ['dL_static', 'raw_Wdper', 'top25p', 'top50p', f'top{NTOPK}p',
                 'rand25p', 'rand_top', 'm25p', 'perm_top']
    run_table('SECONDARY', True, secondary)
    print(f'[{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()