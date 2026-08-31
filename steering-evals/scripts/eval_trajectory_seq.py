"""eval_trajectory_seq.py — FAST: chained one-shot steering (waypoint sequence).

Each waypoint (step, target): apply ONE rotation at 2x alpha* (closed form),
then free-run until the next waypoint. Does a sequence of analytic pushes
compose into clean multi-destination trajectory control (no loops, each
target appears after its waypoint)?

Run: python3 eval_trajectory_seq.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
     python3 eval_trajectory_seq.py --model google/gemma-3-1b-it --tag gemma
"""
import argparse
import math

import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = "For dinner I made"
# waypoints: (start_step, target)
SEQ = [(0, 'chicken'), (14, 'apple'), (28, 'bread')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2-0.5B-Instruct')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--n', type=int, default=42)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--mult', type=float, default=2.0)
    ap.add_argument('--temp', type=float, default=1.0)
    a = ap.parse_args()

    model, tok = M.load_model(a.model, dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    V = model.config.vocab_size
    W = W[:V]
    NL = model.config.num_hidden_layers

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    tid = {wd: word2id[wd] for _, wd in SEQ}
    capid = {}
    for wd in tid:
        ids = tok(' ' + wd.capitalize(), add_special_tokens=False).input_ids
        if len(ids) == 1:
            capid[wd] = int(ids[0])

    def crossings(target_id, h, native):
        hn = h / np.linalg.norm(h)
        t = W[target_id] - (W[target_id] @ hn) * hn
        nt = np.linalg.norm(t)
        tau = t / (nt + 1e-12)
        A = float(hn @ (W[target_id] - W[native]))
        B = float(tau @ (W[target_id] - W[native]))
        alpha = math.atan2(-A, B) if B > 1e-12 else 0.3
        return max(0.0, min(a.mult * alpha, 0.5)), tau

    def gen(seed=0, n=42, top_p=0.9):
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        toks = []
        hits = {wd: [] for _, wd in SEQ}
        for step in range(n):
            with torch.no_grad():
                hid = model(ids, output_hidden_states=True)
                Ln = hid.logits[0, -1].float().cpu().numpy()
            native = int(Ln.argmax())
            h = hid.hidden_states[NL][0, -1].float().cpu().numpy()

            # waypoint at this step?
            wp = [w for s, w in SEQ if s == step]
            if wp:
                w = wp[0]
                alpha, tau = crossings(tid[w], h, native)
                def hook(mod, inp, out):
                    out2 = out.clone()
                    hh = out2[:, -1, :].float().reshape(-1)
                    hhn = hh / hh.norm()
                    v = M.rotate_toward(hhn.cpu().numpy(), tau, alpha)
                    v = v * hh.norm().item()
                    out2[:, -1, :] = torch.as_tensor(v, device=out.device, dtype=out.dtype)
                    return out2
                hd = model.model.norm.register_forward_hook(hook)
            else:
                hd = None
            if hd is not None:
                try:
                    with torch.no_grad():
                        Ls = model(ids).logits[0, -1].float()
                finally:
                    hd.remove()
                p = torch.softmax(Ls / a.temp, dim=0)
            else:
                p = torch.softmax(torch.as_tensor(Ln, device=DEV) / a.temp, dim=0)
            q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
            keep = order[:int((cum <= top_p).sum()) + 1]
            m = torch.zeros_like(q); m[keep] = 1
            q = (q * m) / (q * m).sum()
            nxt = int(torch.multinomial(q, 1))
            toks.append(int(nxt))
            # record waypoint targets appearing within 6 steps after their step
            for s, wd in SEQ:
                ok = (nxt == tid[wd] or (wd in capid and nxt == capid[wd]))
                if s <= step < s + 6 and ok:
                    hits[wd].append(1.0)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return toks, hits

    def rep4(toks):
        if len(toks) < 8:
            return 1.0
        n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])

    print(f"== [{a.tag}] chained one-shot waypoints {[(s, w) for s, w in SEQ]} "
          f"(mult={a.mult}, n={a.n}) ==")
    print(f"{'seed':>5} {'appear chicken':>14} {'appear apple':>12} "
          f"{'appear bread':>12} {'4rep':>6}")
    agg = {w: [] for w in tid}
    reps = []
    for sd in range(a.seeds):
        toks, hits = gen(seed=sd, n=a.n)
        dec = tok.decode(toks)
        for w in tid:
            agg[w].append(1.0 if hits[w] else 0.0)
        reps.append(rep4(toks))
        short = ' '.join(dec.split())[:88]
        print(f"{sd:>5} {agg['chicken'][-1]:>14.0f} {agg['apple'][-1]:>12.0f} "
              f"{agg['bread'][-1]:>12.0f} {reps[-1]:>6.2f}  {short!r}")
    print(f"  mean: chicken {np.mean(agg['chicken']):.2f}  "
          f"apple {np.mean(agg['apple']):.2f}  bread {np.mean(agg['bread']):.2f}  "
          f"4rep {np.mean(reps):.2f}")


if __name__ == "__main__":
    main()