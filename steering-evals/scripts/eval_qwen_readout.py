import math, sys, time
import numpy as np, torch
sys.path.insert(0, '/home/nicolas/transformer-geometry-final/steering-evals/scripts')
import steering_geometry_test as M
MODEL='Qwen/Qwen2-0.5B-Instruct'; DEV='cuda'
model, tok = M.load_model(MODEL, dtype='fp16')
lm_w = model.lm_head.weight
tid = int(tok(' chicken', add_special_tokens=False).input_ids[0])
Wt = lm_w[tid].detach().float().cpu().numpy()
import argparse
A = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
a = A
def rot(out):
    v = out[:, -1, :].float().reshape(-1); vn = v / v.norm()
    t = Wt - (Wt @ vn.cpu().numpy()) * vn.cpu().numpy()
    t = t / (np.linalg.norm(t) + 1e-12)
    tg = torch.as_tensor(t, dtype=torch.float32, device=DEV)
    g = tg - (tg @ vn) * vn; g = g / (g.norm() + 1e-8)
    v2 = vn * math.cos(a) + g * math.sin(a)
    out = out.clone(); out[:, -1, :] = (v.norm() * v2).to(out.dtype)
    return out
def rep4(toks):
    if len(toks) < 8: return 1.0
    n4 = [tuple(toks[i:i+4]) for i in range(len(toks)-3)]
    return np.mean([n4[i] in n4[i+1:] for i in range(len(toks)-3)])
PULSE = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = sustained
for seed in range(4):
    torch.manual_seed(seed)
    ids = tok('For dinner I made', add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)
    toks = []
    for step in range(16):
        h = None
        if PULSE == 0 or step % PULSE == 0:
            h = model.model.norm.register_forward_hook(lambda m,i,o: rot(o))
        try:
            with torch.no_grad(): L = model(ids).logits[0,-1].float()
        finally:
            if h is not None: h.remove()
        p = torch.softmax(L, dim=0); q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
        keep = order[:int((cum <= 0.9).sum())+1]; m = torch.zeros_like(q); m[keep] = 1
        q = (q*m)/(q*m).sum()
        nxt = int(torch.multinomial(q, 1)); toks.append(int(nxt))
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
    head = toks[:10]
    plant = 1.0 if tid in head else 0.0
    print(f"seed {seed}: plant={plant} rep4={rep4(toks):.2f}  {tok.decode(toks)!r}")
