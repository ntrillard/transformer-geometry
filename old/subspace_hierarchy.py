#!/usr/bin/env python3
"""subspace_hierarchy.py — Where does the transport-carrying logit signal live?

From decomp_gen (causal): dL_parallel (1-dim proj onto Wd_per) does NOT
transport; dL_perp (orthogonal residual, 99% of energy) DOES. Question now:
is the boundary a DIRECTION-selection issue or a REPRESENTABILITY issue?

Hierarchy of hidden-reachable subspaces of the 152k logit space (all images
of the linear map v -> W v, so all "what a hidden-state direction can say"):

  L0 = span(W d_per)                     (1-dim, tested: no transport)
  L1 = span(W @ shellPCA)                (rank-231 token shell)
  L2 = rowspace(W)                       (rank-1536: ALL hidden-reachable)
  L3 = full logit space                  (= dL_full, the reference)

At each level: project dL_static onto the subspace, RESCALE to the fixed
reference norm ||dL_full|| (so dose is identical), generate K=6 seed 0-5.

Interpretation:
  - L1 transports  -> the signal lives in the token shell (the complement
                      hypothesis flipped: shell, not complement)
  - L2 transports  -> signal IS hidden-representable; the d_per DIRECTION
                      was simply insufficient (direction-selection issue)
  - only L3        -> the masked spikes genuinely need out-of-rowspace
                      geometry -> representability boundary

Run: HF_TOKEN=... python3 subspace_hierarchy.py    (default fantasy)
     ROYAL=1 for royalty task.
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
SEEDBASE = int(os.environ.get('SEEDBASE', '0'))
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
    print(f'subspace_hierarchy seed={SEED} K={K} SW0={SW0} '
          f'{"ROYAL" if os.environ.get("ROYAL")=="1" else "FANTASY"}', flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    norm = model.model.norm
    W = model.lm_head.weight.detach().float().cpu(); Wt = W.t()
    eos_id = int(tok.eos_token_id)

    Wnorm = W / W.norm(dim=1, keepdim=True)

    # shell PCA (from normalized rows) -- rank-231
    _, Sn, Vn = torch.svd_lowrank(Wnorm.float().cuda(), q=300, niter=5)
    Sn = Sn.cpu(); Vn = Vn.cpu()
    eng = (Sn**2).cumsum(0) / (Sn**2).sum()
    r = int((eng >= 0.9).nonzero()[0].item()) + 1
    Uc = Vn[:, :r].float()               # 1536 x r, basis of the shell
    print(f'  shell rank r={r}', flush=True)

    # full rowspace of W (rank-1536, hidden-representable). Orthogonal
    # projection onto span(columns of W) via the Gram matrix (W^T W is
    # 1536x1536; W has full column rank so it is invertible):
    #   P = W (W^T W)^-1 W^T, applied to dL_static.
    # The shell-based rank-231 image follows by the same Gram solve on the
    # (W @ Uc) image (r x r Gram).
    # W^T W Gram (computed once here; projections need dL_static later)
    WtW = (W.t() @ W).float()

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

    # --- Wd_per (1-dim) ---
    Ts = torch.stack([state_of(s) for s in tgt]).mean(0)
    Ns = torch.stack([state_of(s) for s in neu]).mean(0)
    Ucd = Uc.to(Ts.device)
    dper = (Ts - Ns) - Ucd @ (Ucd.t() @ (Ts - Ns)); dper = dper / dper.norm()
    Wd_per = (dper.cpu() @ Wt.cpu())
    print(f'  d_per shell-leak={(Ucd@(Ucd.t()@dper)).norm():.4f}', flush=True)

    # --- dL_static (faithful, ALPHA-scaled, top-k masked) ---
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
    dL_static = ALPHA * dL_z * m
    N_REF = dL_static.norm()
    print(f'  ||dL_full||={N_REF:.3f} (reference norm; all conditions rescaled to it)',
          flush=True)

    def proj1(vec):
        """orthogonal projection onto span(vec)."""
        v = vec / vec.norm()
        return (vec @ v) * v

    # --- Gram projections of dL_static onto hidden-reachable subspaces ---
    print('  building logit-space projections (Gram solves)...', flush=True)
    WWdL = (W.t() @ dL_static).float()
    # 1) rank-1536 rowspace projection
    x_full = torch.linalg.solve(WtW, WWdL)
    full_proj = (W @ x_full).float()
    # 2) rank-r shell image projection onto span(W @ Uc)
    Uc_t_WtW = (Uc.t() @ WtW).float()
    Gshell = (Uc_t_WtW @ Uc).float()
    rhs_shell = (Uc.t() @ WWdL).float()
    xs = torch.linalg.solve(Gshell, rhs_shell)
    shell_proj = (W @ (Uc @ xs)).float()
    # 3) 1-dim Wd_per projection
    w0 = Wd_per / Wd_per.norm()
    dL_par1 = (dL_static @ w0) * w0
    for nm, v in [('par_1d(Wdper)', dL_par1),
                  ('shell_r231', shell_proj),
                  ('rowspace_1536', full_proj)]:
        c = (v @ dL_static).item()/(v.norm()*N_REF) if v.norm()>1e-9 else float('nan')
        print(f'    {nm:>14}: ||proj||={v.norm():9.3f}  cos(proj,dLfull)={c:+.3f}')

    # --- build the projected conditions (all rescaled to N_REF) ---
    def rescale(v):
        n = v.norm()
        return v * (N_REF / n) if n > 1e-9 else v

    conds = {}
    # L0: 1-dim Wd_per (projection of dL_static onto it, then rescale)
    conds['L0_Wdper(1d)']   = rescale(dL_par1)
    # L1: shell image (rank-231)
    conds['L1_shell(r231)'] = rescale(shell_proj)
    # L2: full rowspace (rank-1536)
    conds['L2_rowspace']    = rescale(full_proj)
    # L3: full logit space (= reference)
    conds['L3_dLfull']      = dL_static
    conds['L3b_dLz_full']   = rescale(ALPHA * dL_z)

    for k, v in conds.items():
        print(f'  {k:>16}: ||proj||={v.norm():7.3f}  '
              f'cos(proj,dLfull)={(v @ dL_static).item()/(v.norm()*N_REF):+.3f}')

    # --- generation ---
    h_ids = set()
    for w in HELD_OUT.split():
        sp = tok(' '+w, add_special_tokens=False).input_ids
        if len(sp) == 1: h_ids.add(int(sp[0]))

    def generate(delta, seed):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
        past = None; out_ids = []; dd = delta.to(DEV)
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
                p = torch.softmax(L1, 0)
                q = p.clone(); ooo = q.argsort(descending=True)
                kk = int((q[ooo].cumsum(0) <= NUCLEUS).sum()) + 1
                msk = torch.zeros_like(q); msk[ooo[:kk]] = 1
                qq = (q*msk); qq = qq/qq.sum()
                nxt = int(torch.multinomial(qq, 1))
                if nxt == eos_id: break
                out_ids.append(nxt)
                ids = torch.tensor([[nxt]], device=DEV)
        return tok.decode(out_ids)

    print(f'\n  {"cond":>16} {"transport":>10}  {"minHr":>5} {"topP":>5}  sample')
    for name, delta in conds.items():
        R = 0; minH = 10**9; topP = 0; steps = 0
        samples = []
        for s in range(K):
            txt = generate(delta, SEEDBASE + s)
            _,_,_,_,ok = score(txt)
            R += int(ok)
            samples.append(txt[:60].replace('\n',' '))
        print(f'  {name:>16} {R:3d}/{K:<5} {minH:5d} {topP:5.2f}  {samples[0]!r}',
              flush=True)
    print(f'[{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()