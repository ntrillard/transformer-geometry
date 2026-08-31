"""eval_waypoint_spacing.py — FAST: minimum spacing between chained one-shots.

Waypoints chicken@0, apple@k, bread@2k. Vary spacing k. For each waypoint,
measure: did the target land within window [wp, wp+6], and landing latency
(first step >= wp where the target appears).

The spacing floor is where the SECOND push fires on a state that hasn't
settled from the FIRST push -> late/missing landings or loops.

Run: python3 eval_waypoint_spacing.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
     python3 eval_waypoint_spacing.py --model google/gemma-3-1b-it --tag gemma --temp 1.3
"""
import argparse
import math

import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = "For dinner I made"
SPACINGS = [1, 2, 3, 4, 6, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2-0.5B-Instruct')
    ap.add_argument('--tag', default='model')
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
    tid = {w: word2id[w] for w in ('chicken', 'apple', 'bread')}
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

    def gen(seq, spine, seed=0, top_p=0.9):
        """seq: list of (step, target_word). Returns toks and first-appear map."""
        torch.manual_seed(seed)
        ids = tok(PROMPT, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        toks = []
        first = {}
        n = spine + 8
        for step in range(n):
            with torch.no_grad():
                hid = model(ids, output_hidden_states=True)
                Ln = hid.logits[0, -1].float().cpu().numpy()
            native = int(Ln.argmax())
            h = hid.hidden_states[NL][0, -1].float().cpu().numpy()

            wp = [w for s, w in seq if s == step]
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
            for wd in ('chicken', 'apple', 'bread'):
                if wd not in first and (nxt == tid[wd] or
                                        (wd in capid and nxt == capid[wd])):
                    first[wd] = step
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return toks, first

    def rep4(toks):
        if len(toks) < 8:
            return 1.0
        n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])

    print(f"== [{a.tag}] waypoint spacing sweep (chicken@0, apple@k, bread@2k) ==")
    print(f"{'k':>3} {'hit chicken':>11} {'hit apple':>10} {'hit bread':>10} "
          f"{'lat(apple)':>10} {'lat(bread)':>10} {'4rep':>6}")
    for k in SPACINGS:
        seq = [(0, 'chicken'), (k, 'apple'), (2 * k, 'bread')]
        spine = 2 * k
        hc, ha, hb, la, lb, reps = [], [], [], [], [], []
        for sd in range(a.seeds):
            toks, first = gen(seq, spine, seed=sd)
            hc.append(1.0 if ('chicken' in first and first['chicken'] <= 6) else 0.0)
            ha.append(1.0 if ('apple' in first and first['apple'] <= k + 6) else 0.0)
            hb.append(1.0 if ('bread' in first and first['bread'] <= 2 * k + 6) else 0.0)
            la.append(first.get('apple', float('nan')))
            lb.append(first.get('bread', float('nan')))
            reps.append(rep4(toks))
        print(f"{k:>3} {np.mean(hc):>11.2f} {np.mean(ha):>10.2f} {np.mean(hb):>10.2f} "
              f"{np.nanmean(la):>10.1f} {np.nanmean(lb):>10.1f} {np.mean(reps):>6.2f}")
    print("  latency = mean first-appear step (nan-masked); window = wp+6")


if __name__ == "__main__":
    main()