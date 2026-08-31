"""eval_oneshot.py — FAST: one-shot minimal steering vs duty-cycle vs overdrive.

Modes:
  native   no steering
  oneshot  apply alpha* (closed form) once at step 0, then run free
  duty3    final-layer chord at alpha=0.15 every 3rd step (known-good recipe)
  fixed    alpha=0.3 every step (overdrive -> loops)

Question: does a single minimal nudge plant the topic and let the model
continue naturally (no repetition loop)?

Run: python3 eval_oneshot.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
     python3 eval_oneshot.py --model google/gemma-3-1b-it --tag gemma
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
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--seeds', type=int, default=3)
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

    def crossings(target_id, h, native):
        """alpha* to beat native for the target, given post-norm state h."""
        hn = h / np.linalg.norm(h)
        t = W[target_id] - (W[target_id] @ hn) * hn
        nt = np.linalg.norm(t)
        tau = t / (nt + 1e-12)
        A = float(hn @ (W[target_id] - W[native]))
        B = float(tau @ (W[target_id] - W[native]))
        alpha = math.atan2(-A, B) + 0.03 if B > 1e-12 else 0.3
        return max(0.0, min(alpha, 0.5)), tau

    def gen(mode, target_id, seed=0, n=24, top_p=0.9):
        torch.manual_seed(seed)
        ids = tok(a.prompt, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        toks = []
        hd = None
        tau_fn = None
        for step in range(n):
            with torch.no_grad():
                hid = model(ids, output_hidden_states=True)
                Ln = hid.logits[0, -1].float().cpu().numpy()
            native = int(Ln.argmax())

            if mode == 'oneshot' and step == 0:
                h = hid.hidden_states[NL][0, -1].float().cpu().numpy()
                alpha, tau_fn = crossings(target_id, h, native)
                def hook(mod, inp, out):
                    out2 = out.clone()
                    hh = out2[:, -1, :].float().reshape(-1)
                    hhn = hh / hh.norm()
                    v = M.rotate_toward(hhn.cpu().numpy(), tau_fn, alpha)
                    v = v * hh.norm().item()
                    out2[:, -1, :] = torch.as_tensor(v, device=out.device, dtype=out.dtype)
                    return out2
                hd = model.model.norm.register_forward_hook(hook)
            elif mode == 'duty3' and step % 3 == 0:
                h = hid.hidden_states[NL][0, -1].float().cpu().numpy()
                alpha, tau_fn = crossings(target_id, h, native)
                def hook2(mod, inp, out):
                    out2 = out.clone()
                    hh = out2[:, -1, :].float().reshape(-1)
                    hhn = hh / hh.norm()
                    v = M.rotate_toward(hhn.cpu().numpy(), tau_fn, 0.15)
                    v = v * hh.norm().item()
                    out2[:, -1, :] = torch.as_tensor(v, device=out.device, dtype=out.dtype)
                    return out2
                hd = model.model.norm.register_forward_hook(hook2)
            elif mode == 'fixed':
                h = hid.hidden_states[NL][0, -1].float().cpu().numpy()
                alpha, tau_fn = crossings(target_id, h, native)
                def hook3(mod, inp, out):
                    out2 = out.clone()
                    hh = out2[:, -1, :].float().reshape(-1)
                    hhn = hh / hh.norm()
                    v = M.rotate_toward(hhn.cpu().numpy(), tau_fn, 0.3)
                    v = v * hh.norm().item()
                    out2[:, -1, :] = torch.as_tensor(v, device=out.device, dtype=out.dtype)
                    return out2
                hd = model.model.norm.register_forward_hook(hook3)
            else:
                hd = None

            if hd is not None:
                try:
                    with torch.no_grad():
                        Ls = model(ids).logits[0, -1].float()
                finally:
                    hd.remove()
                    hd = None
            else:
                Ls = torch.as_tensor(Ln, device=DEV)

            p = torch.softmax(Ls, dim=0)
            q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
            keep = order[:int((cum <= top_p).sum()) + 1]
            m = torch.zeros_like(q); m[keep] = 1
            q = (q * m) / (q * m).sum()
            nxt = int(torch.multinomial(q, 1))
            toks.append(int(nxt))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return toks

    def rep_score(toks):
        """4-gram repetition: fraction of positions inside a repeated 4-gram."""
        if len(toks) < 8:
            return 1.0
        reps = 0
        for i in range(len(toks) - 3):
            if toks[i:i + 4] in [toks[j:j + 4] for j in range(i + 1, len(toks) - 3)]:
                reps += 1
        return reps / (len(toks) - 3)

    print(f"== [{a.tag}] one-shot vs duty3 vs overdrive (target 'chicken', {a.prompt!r}) ==")
    print(f"{'mode':>8} {'tgt#':>5} {'div':>6} {'4rep':>6}  sample")
    for mode in ('native', 'oneshot', 'duty3', 'fixed'):
        tgt, divs, reps, samples = [], [], [], []
        ti = word2id['chicken']
        for sd in range(a.seeds):
            toks = gen(mode, ti, seed=sd, n=a.n)
            tgt.append(toks.count(ti))
            divs.append(len(set(toks)) / len(toks))
            reps.append(rep_score(toks))
            samples.append(tok.decode(toks)[:64])
        print(f"{mode:>8} {int(np.mean(tgt)):>5} {np.mean(divs):>6.2f} {np.mean(reps):>6.2f}"
              f"  {samples[0]!r}")


if __name__ == "__main__":
    main()