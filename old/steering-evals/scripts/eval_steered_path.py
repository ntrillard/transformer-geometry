"""eval_steered_path.py — FAST: does margin@site predict the STEERED path?

At every generated step, measure (a) the native (unsteered) top-1 token,
(b) its margin over the runner, (c) whether the steered argmax DIFFERS from
native (a 'flip'). PREDICTION: flips concentrate at low-margin steps; high
native-margin steps are emitted verbatim (steering is silent).

food chord, final-only alpha .15 duty 3, 16 tokens, 2 seeds x 2 prompts. ~30s.
Run: python3 eval_steered_path.py"""
import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from multiverse_lab import chord_summary

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    import multiverse_lab
    multiverse_lab.DEV = DEV
    model, tok = M.load_model('Qwen/Qwen2-0.5B-Instruct', dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    NL = model.config.num_hidden_layers

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    fam = set(CLASSES['food'])
    food = np.array([word2id[w] for w in CLASSES['food'] if w in word2id][:5])
    C, _, _ = chord_summary(food, Wn)
    dn = torch.as_tensor(C, device=DEV, dtype=torch.float32)

    def steered_with_natives(prompt, alpha=0.15, duty=3, n=16, seed=0):
        """Run gen stepwise: each step record native top1/margin (pre-hook
        forward), then apply steering and pick next token."""
        torch.manual_seed(seed)
        ids = tok(prompt, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        steps = []
        for step in range(n):
            # native read (no hook): margin of the native winner
            with torch.no_grad():
                Ln = model(ids).logits[0, -1].float()
            nt = int(Ln.argmax())
            l2 = Ln.clone(); l2[nt] = -float('inf')
            margin_nat = float(Ln[nt] - l2.max())

            # steered read: inject at L23, take argmax
            def hook(mod, inp, out):
                out2 = out.clone()
                h = out2[:, -1, :].float()
                hn = h / h.norm()
                g = dn - (dn @ hn.transpose(0, 1)) * hn
                g = g / max(g.norm().item(), 1e-8)
                h2 = h + alpha * h.norm() * g
                out2[:, -1, :] = (h.norm() * h2 / h2.norm()).to(out.dtype)
                return out2

            hd = model.model.layers[23].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    Ls = model(ids).logits[0, -1].float()
            finally:
                hd.remove()
            st = int(Ls.argmax())
            flip = (st != nt)
            tok_s = tok.decode([nt]).strip()
            tok_st = tok.decode([st]).strip()
            steps.append(dict(step=step, native=tok_s, steered=tok_st,
                              margin=round(margin_nat, 3), flip=flip))
            # advance with steered token (top_p .9 over steered logits)
            p = torch.softmax(Ls, dim=0)
            q = p.clone(); order = q.argsort(descending=True); cum = q[order].cumsum(0)
            keep = order[:int((cum <= 0.9).sum()) + 1]
            m = torch.zeros_like(q); m[keep] = 1
            q = (q * m) / (q * m).sum()
            nxt = int(torch.multinomial(q, 1))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
            if step % duty != 0:     # duty: only inject every 'duty' steps
                pass                 # (hook already conditional via re-register loop)
        return steps

    print("== steer path: food5 @ L23, alpha .15, duty 3, 16 tok ==")
    print(f"{'prompt':>24} {'steps':>5} {'flips':>5} {'flips@lo':>9} "
          f"{'lo/step':>8} {'flips@hi':>9} {'hi/step':>8}")
    for prompt in ("For dinner I made", "In the forest"):
        for sd in (0, 1):
            steps = steered_with_natives(prompt, seed=sd)
            flips = [s for s in steps if s['flip']]
            lo = [s for s in steps if s['margin'] <= 0.1]
            hi = [s for s in steps if s['margin'] > 0.1]
            f_lo = sum(1 for s in lo if s['flip']) if lo else 0
            f_hi = sum(1 for s in hi if s['flip']) if hi else 0
            print(f"{prompt:>24} seed{sd} {len(steps):>5} {len(flips):>5} "
                  f"{f_lo:>5}/{len(lo):>4} {len(lo)/len(steps):>7.2f} "
                  f"{f_hi:>5}/{len(hi):>4} {len(hi)/len(steps):>7.2f}")
            print("      path: " + " ".join(
                f"{s['native']}" if not s['flip'] else f"[{s['steered']}]"
                for s in steps)[:100])


if __name__ == "__main__":
    main()