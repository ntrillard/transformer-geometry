#!/usr/bin/env python3
"""vec_compare.py — Pure vector comparison of the two logit-space contrasts.

Question (from the falsified sparsity line): what actually differs between
  dL_static = z(L_tgt) - z(L_neu)  (per-sentence z-sum, top-k masked)  -> TRANSPORTS
  Wd_per    = W @ (T_s - N_s)_perp  (hidden contrast through output)   -> NO TRANSPORT

Measure, on the SAME concept (switch via TGT/NEU env):
  1. cosine(dL_static, Wd_per)
  2. top-k overlap for K in {10,25,50,100,200}
  3. Spearman rank correlation over the full 152k logit vector
  4. sign agreement over the full vector
  5. directional decomposition:
        dL = proj_Wd(dL) + dL_perp
     report norms of both components, and their OWN cosines (whether the
     shared component is even aligned with the target).

No generation. Run: HF_TOKEN=... python3 vec_compare.py
"""
import os, sys, time
import torch, transformers

MODEL = os.environ.get('MODEL', 'Qwen/Qwen2-1.5B')
NTOPK = int(os.environ.get('NTOPK', '200'))

TGT = os.environ.get('TGT', 'A dragon circled the ruined towers of the ancient kingdom|'
      'A knight drew his sword against the fire-breathing beast|'
      "The wizard's spell shattered the castle gates")
NEU = os.environ.get('NEU', 'The waves crashed gently on the beach|'
      'The sand was cool to the touch|The sun was warm over the water')

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def spearman(a, b, step=997):
    # subsampled rank correlation for 152k vectors (exact sort too heavy)
    n = a.numel()
    idx = torch.arange(0, n, step)
    ra = a[idx].argsort().argsort().float()
    rb = b[idx].argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm())
    return (ra @ rb / denom).item() if denom > 0 else float('nan')


def main():
    t0 = time.time()
    print(f'vec_compare NTOPK={NTOPK}', flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    norm = model.model.norm
    W = model.lm_head.weight.detach().float(); Wt = W.t()

    # shell (for d_per)
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
    d = Ts - Ns
    Ucd = Uc.to(d.device)
    dper = d - Ucd @ (Ucd.t() @ d); dper = dper / dper.norm()
    Wd_per = (dper.cpu() @ Wt.cpu())
    print(f'  d_per shell-leak={(Ucd@(Ucd.t()@dper)).norm():.4f}', flush=True)

    # --- dL_static (faithful per-sentence z-sum, top-k masked) ---
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
    dL_static = dL_z * m

    # --- 1. cosine ---
    cos = (dL_static @ Wd_per).item() / (dL_static.norm() * Wd_per.norm())
    print(f'\n  cosine(dL_static, Wd_per)      = {cos:+.4f}')
    print(f'  cosine(dL_z_full, Wd_per)      = {(dL_z @ Wd_per).item()/(dL_z.norm()*Wd_per.norm()):+.4f}')

    # --- 2. top-k overlap ---
    print(f'\n  {"K":>6} {"overlap":>10} {"Jaccard":>9}')
    dL_ids = set(dL_static.argsort(descending=True)[:NTOPK].tolist())
    for k in (10, 25, 50, 100, 200):
        wk = set(Wd_per.argsort(descending=True)[:k].tolist())
        inter = len(dL_ids & wk)
        jac = inter / (len(dL_ids | wk))
        print(f'  {k:>6} {inter:>10} {jac:>9.3f}')

    # --- 3. Spearman ---
    print(f'\n  spearman(dL_z, Wd_per)  = {spearman(dL_z, Wd_per):+.4f}')

    # --- 4. sign agreement ---
    sa = (dL_z.sign() == Wd_per.sign()).float().mean().item()
    print(f'  sign-agreement(dL_z, Wd_per) = {sa*100:.1f}%')

    # --- 5. directional decomposition ---
    # dL_static = proj + perp  (project onto Wd_per)
    w = Wd_per / Wd_per.norm()
    proj = (dL_static @ w) * w
    perp = dL_static - proj
    print(f'\n  == decomposition of dL_static along Wd_per ==')
    print(f'  ||dL_static|| = {dL_static.norm():.4f}')
    print(f'  ||proj||      = {proj.norm():.4f}  ({100*proj.norm()/dL_static.norm():.1f}% of dL)')
    print(f'  ||perp||      = {perp.norm():.4f}  ({100*perp.norm()/dL_static.norm():.1f}% of dL)')
    print(f'  cosine(proj, dL)  = {(proj @ dL_static).item()/(proj.norm()*dL_static.norm()):+.4f}')
    print(f'  cosine(perp, dL)  = {(perp @ dL_static).item()/(perp.norm()*dL_static.norm()):+.4f}')
    # where do the top-k entries lie?
    topk_ids = torch.tensor(dL_static.argsort(descending=True)[:NTOPK].tolist())
    pk = proj[topk_ids].norm(); ek = perp[topk_ids].norm()
    print(f'  top-{NTOPK} entries: |proj|={pk:.3f} |perp|={ek:.3f}  '
          f'(perp dominates: {100*ek/max(pk+ek,1e-9):.0f}%)')
    print(f'[{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()