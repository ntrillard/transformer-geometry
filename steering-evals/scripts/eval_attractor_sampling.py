"""eval_attractor_sampling.py — FAST: can sampling config free Gemma's 'I' loop?

Native (unsteered) generation sweep: temperature x top_p x repetition-penalty
on both models, prompt 'For dinner I made'. Find a config where the native
continuation is natural (low 4-rep, high diversity). If Gemma's 'I I I I'
loop is a sampling artifact, some config escapes it.

Run: python3 eval_attractor_sampling.py --model google/gemma-3-1b-it --tag gemma
     python3 eval_attractor_sampling.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
"""
import argparse

import numpy as np
import torch

import steering_geometry_test as M

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='google/gemma-3-1b-it')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--prompt', default="For dinner I made")
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--seeds', type=int, default=2)
    a = ap.parse_args()

    model, tok = M.load_model(a.model, dtype='fp16')

    def gen(seed=0, temp=1.0, top_p=0.9, rep_p=0.0, n=24):
        torch.manual_seed(seed)
        ids = tok(a.prompt, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        toks = []
        for _ in range(n):
            with torch.no_grad():
                Ln = model(ids).logits[0, -1].float()
            if rep_p > 0:
                for i, t in enumerate(toks[::-1]):
                    Ln[t] -= rep_p * np.power(0.5, i)
            p = torch.softmax(Ln / temp, dim=0)
            q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
            keep = order[:int((cum <= top_p).sum()) + 1]
            m = torch.zeros_like(q); m[keep] = 1
            q = (q * m) / (q * m).sum()
            nxt = int(torch.multinomial(q, 1))
            toks.append(int(nxt))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return toks

    def rep4(toks):
        if len(toks) < 8:
            return 1.0
        n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        return np.mean([n4[i] in n4[i + 1:] for i in range(len(toks) - 3)])

    print(f"== [{a.tag}] native continuation vs sampling config ({a.prompt!r}) ==")
    print(f"{'config':>22} {'4rep':>6} {'div':>6}  sample")
    configs = [
        ('temp1 top0.9', 1.0, 0.9, 0.0),
        ('temp0.7 top0.9', 0.7, 0.9, 0.0),
        ('temp1.3 top0.9', 1.3, 0.9, 0.0),
        ('temp1.0 top1.0', 1.0, 1.0, 0.0),
        ('temp1.0 top0.5', 1.0, 0.5, 0.0),
        ('rep-p 0.5', 1.0, 0.9, 0.5),
        ('rep-p 1.0', 1.0, 0.9, 1.0),
        ('rep-p 2.0', 1.0, 0.9, 2.0),
    ]
    for name, temp, top_p, rep in configs:
        reps, divs, samples = [], [], []
        for sd in range(a.seeds):
            toks = gen(seed=sd, temp=temp, top_p=top_p, rep_p=rep, n=a.n)
            reps.append(rep4(toks))
            divs.append(len(set(toks)) / len(toks))
            samples.append(tok.decode(toks)[:60])
        print(f"{name:>22} {np.mean(reps):>6.2f} {np.mean(divs):>6.2f}  {samples[0]!r}")


if __name__ == "__main__":
    main()