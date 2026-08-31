"""eval_oneshot_scale.py — FAST: one-shot steering robustness at scale.

For targets x prompts x seeds: one minimal-alpha nudge at step 0, then free run.
Metrics:
  plant   target word (any capitalization) appears in first 10 tokens
  topic   any food-class word appears anywhere in the generation
  on-topic fraction of generated tokens that are food-class words
  4rep    4-gram repetition score

Also --nudge N to apply the minimal nudge for the first N steps (for Gemma's
'I' attractor).

Run: python3 eval_oneshot_scale.py --model Qwen/Qwen2-0.5B-Instruct --tag qwen
     python3 eval_oneshot_scale.py --model google/gemma-3-1b-it --tag gemma [--nudge 3]
"""
import argparse
import math

import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = ["For dinner I made", "I went to the store and bought"]
TARGETS = ['chicken', 'apple', 'bread', 'soup']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2-0.5B-Instruct')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--nudge', type=int, default=1,
                    help='apply minimal nudge for first N steps')
    ap.add_argument('--show', default=None,
                    help='print raw samples for this target (oneshot mode)')
    ap.add_argument('--mult', type=float, default=1.0,
                    help='multiply alpha* (one-shot push strength)')
    a = ap.parse_args()

    model, tok = M.load_model(a.model, dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    V = model.config.vocab_size
    W = W[:V]
    NL = model.config.num_hidden_layers

    food_ids = set()
    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        for pre in (' ', ' C') if w[0].isupper() else (' ',):
            pass
    # collect single-token food words (lowercase normal form)
    for w in CLASSES['food']:
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    food_words = list(word2id.keys())
    # also all-capitalized variants for plant detection
    cap_ids = {}
    for w in food_words:
        ids = tok(' ' + w.capitalize(), add_special_tokens=False).input_ids
        if len(ids) == 1:
            cap_ids[w] = int(ids[0])

    def crossings(target_id, h, native):
        hn = h / np.linalg.norm(h)
        t = W[target_id] - (W[target_id] @ hn) * hn
        nt = np.linalg.norm(t)
        tau = t / (nt + 1e-12)
        A = float(hn @ (W[target_id] - W[native]))
        B = float(tau @ (W[target_id] - W[native]))
        alpha = math.atan2(-A, B) + 0.03 if B > 1e-12 else 0.3
        return max(0.0, min(alpha, 0.5)), tau

    def gen(mode, target_id, prompt, seed=0, n=24, top_p=0.9):
        torch.manual_seed(seed)
        ids = tok(prompt, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        toks = []
        for step in range(n):
            with torch.no_grad():
                hid = model(ids, output_hidden_states=True)
                Ln = hid.logits[0, -1].float().cpu().numpy()
            native = int(Ln.argmax())
            h = hid.hidden_states[NL][0, -1].float().cpu().numpy()
            if mode != 'native' and step < a.nudge:
                alpha, tau = crossings(target_id, h, native)
                def hook(mod, inp, out):
                    out2 = out.clone()
                    hh = out2[:, -1, :].float().reshape(-1)
                    hhn = hh / hh.norm()
                    v = M.rotate_toward(hhn.cpu().numpy(), tau, alpha * a.mult)
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
                p = torch.softmax(Ls, dim=0)
            else:
                p = torch.softmax(torch.as_tensor(Ln, device=DEV), dim=0)
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
        reps = 0
        n4 = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        for i in range(len(toks) - 3):
            if n4[i] in n4[i + 1:]:
                reps += 1
        return reps / (len(toks) - 3)

    print(f"== [{a.tag}] one-shot robustness (nudge={a.nudge}, {a.n} tokens, "
          f"{a.seeds} seeds) ==")
    print(f"{'target':>8} {'mode':>7} {'plant':>6} {'topic':>6} {'ontopic':>7} {'4rep':>6}")
    agg = {m: [] for m in ('native', 'oneshot')}
    for tgt in TARGETS:
        ti = word2id[tgt]
        for mode in ('native', 'oneshot'):
            plants, topics, onts, reps = [], [], [], []
            for pr in PROMPTS:
                for sd in range(a.seeds):
                    toks = gen(mode, ti, pr, seed=sd, n=a.n)
                    dec = tok.decode(toks)
                    plants.append(1.0 if (ti in toks[:10] or
                                          (tgt in cap_ids and cap_ids[tgt] in toks[:10]))
                                  else 0.0)
                    intoks = set(toks)
                    topics.append(1.0 if any(fid in intoks for fid in
                                             list(word2id.values()) + list(cap_ids.values()))
                                  else 0.0)
                    fi = [t for t in toks if t in set(word2id.values()) |
                          set(cap_ids.values())]
                    onts.append(len(fi) / len(toks))
                    reps.append(rep4(toks))
            print(f"{tgt:>8} {mode:>7} {np.mean(plants):>6.2f} {np.mean(topics):>6.2f} "
                  f"{np.mean(onts):>7.3f} {np.mean(reps):>6.2f}")
            if a.show and tgt == a.show and mode == 'oneshot':
                for pr in PROMPTS:
                    for sd in range(a.seeds):
                        print(f"  [{pr!r} sd={sd}] {tok.decode(gen(mode, ti, pr, seed=sd, n=a.n))[:96]!r}")


if __name__ == "__main__":
    main()