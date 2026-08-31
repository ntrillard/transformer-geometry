"""eval_surgical.py — FAST: closed-form MINIMAL-budget steering vs overdrive.

Per generated step: recompute the exact alpha* needed to make the target token
(e.g. 'chicken') beat the native argmax, apply alpha*+eps at the readout layer.
Compare against fixed alpha=0.3 overdrive and no steering.

Prediction: minimal-budget holds the target rank-1 with LESS loop than the
overdrive, because it never pushes further than needed.

Run: python3 eval_surgical.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
     python3 eval_surgical.py --model google/gemma-3-1b-it --tag gemma
"""
import argparse
import math

import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2-0.5B-Instruct')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--prompt', default="For dinner I made")
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
    target_id = word2id['chicken']

    def gen(mode, n=16, seed=0, top_p=0.9, duty=1):
        torch.manual_seed(seed)
        ids = tok(a.prompt, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        toks = []
        for step in range(n):
            with torch.no_grad():
                hid = model(ids, output_hidden_states=True)
                Ln = hid.logits[0, -1].float().cpu().numpy()
            h = hid.hidden_states[NL][0, -1].float().cpu().numpy()
            hn = h / np.linalg.norm(h)
            native = int(Ln.argmax())

            alpha = 0.0
            if mode == 'fixed':
                alpha = 0.3
            t = W[target_id] - (W[target_id] @ hn) * hn
            nt = np.linalg.norm(t)
            tau = t / (nt + 1e-12)
            if mode == 'fixed':
                alpha = 0.3
            elif mode == 'minimal':
                A = float(hn @ (W[target_id] - W[native]))
                B = float(tau @ (W[target_id] - W[native]))
                alpha = math.atan2(-A, B) + 0.02 if B > 1e-12 else 0.3
                alpha = max(0.0, min(alpha, 0.5))

            if alpha > 0:
                def hook(mod, inp, out):
                    # rotate the POST-norm state (the real readout feed) on the sphere
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
            try:
                with torch.no_grad():
                    Ls = model(ids).logits[0, -1].float()
            finally:
                if hd is not None:
                    hd.remove()
            p = torch.softmax(Ls, dim=0)
            q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
            keep = order[:int((cum <= top_p).sum()) + 1]
            m = torch.zeros_like(q); m[keep] = 1
            q = (q * m) / (q * m).sum()
            nxt = int(torch.multinomial(q, 1))
            toks.append(int(nxt))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return toks

    from collections import Counter
    print(f"== [{a.tag}] surgical vs overdrive vs native ('chicken' target, {a.prompt!r}) ==")
    print(f"{'mode':>9} {'chick#':>6} {'top-k':>6} {'diversity':>9}  sample")
    for mode in ('native', 'fixed', 'minimal'):
        agg_tgt, agg_div, samples = [], [], []
        for sd in (0, 1):
            toks = gen(mode, seed=sd)
            agg_tgt.append(toks.count(target_id))
            div = len(set(toks)) / max(len(toks), 1)
            agg_div.append(div)
            samples.append(tok.decode(toks)[:56])
        topk = f"top-{np.median(agg_tgt):.0f}"
        print(f"{mode:>9} {int(np.mean(agg_tgt)):>6} {topk:>6} "
              f"{np.mean(agg_div):>9.2f}  {samples[0]!r}")


if __name__ == "__main__":
    main()