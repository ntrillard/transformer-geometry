#!/usr/bin/env python3
"""Sustained multi-token steering with phrase-level targets.

A target phrase (possibly multi-token) is mapped to a steering direction by
averaging its LM-head rows. A forward hook adds alpha * tangent(direction)
at the last position at every generation step ("persist") or only the first
step ("single"). Greedy decoding.

Metrics per run: occurrences of the target phrase in the continuation,
distinct-token ratio (degeneracy guard), mean top-1 probability.

Run: python eval_sustained_steering.py --model Qwen/Qwen2-0.5B-Instruct \
        --targets "apple" "dragon" --alphas 1 2 --new-tokens 48
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import steering_geometry_test as M

OUT = Path("../steering_geometry_results")


@torch.no_grad()
def make_tangent_hook(head_rows, alpha):
    """head_rows: (k, d) tensor of LM-head rows for the target phrase."""
    w = head_rows.mean(0)
    w = w / w.norm()

    def hook(module, inp, out):
        out2 = out.clone()
        h = out2[0, -1, :].float()
        hn = h / h.norm()
        g = w - (w @ hn) * hn
        g = g / max(g.norm().item(), 1e-8)
        out2[0, -1, :] = (h + alpha * h.norm() * g).to(out.dtype)
        return out2
    return hook


@torch.no_grad()
def generate(model, tok, prompt_ids, new_tokens=48, temperature=None, top_p=None,
             hook_fn=None, layer=None, duty=1):
    ids = prompt_ids.clone()
    conf = []
    hd = None
    if hook_fn is not None and layer is not None and duty > 0:
        pass  # registered per-step below via permanent handle when duty==1;
          # for duty>1 we toggle by re-registering (cheap for tiny models)
    for step in range(new_tokens):
        use_hook = hook_fn is not None and layer is not None and (step % max(duty,1) == 0)
        hd = model.model.layers[layer].register_forward_hook(hook_fn) if use_hook else None
        try:
            logits = model(ids).logits[0, -1].float()
        finally:
            if hd is not None:
                hd.remove()
        p = torch.softmax(logits, dim=0)
        conf.append(float(p.max()))
        if temperature or top_p:
            q = p.clone()
            if temperature:
                pass  # already softmaxed at T=1; approximate: keep p
            if top_p:
                order = q.argsort(descending=True)
                cum = q[order].cumsum(0)
                keep = order[:int((cum <= top_p).sum()) + 1]
                mask = torch.zeros_like(q); mask[keep] = 1
                q = q * mask; q = q / q.sum()
            nxt = int(torch.multinomial(q, 1))
        else:
            nxt = int(logits.argmax())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return tok.decode(ids[0].tolist()), float(np.mean(conf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--targets", nargs="+", default=["apple", "dragon"])
    ap.add_argument("--prompts", nargs="+",
                    default=["Once upon a time", "Tell me something interesting:"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--depth-fracs", type=float, nargs="+", default=[0.99])
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--duty-cycles", type=int, nargs="+", default=[1],
                    help="apply the hook every K-th step")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--control-random", type=int, default=0,
                    help="contrastive: subtract mean of N random head rows")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    model, tok = M.load_model(args.model, dtype="fp16")
    W = model.lm_head.weight.detach().cpu().float().numpy()
    if W.size == 0:
        W = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    L = model.config.num_hidden_layers

    rows = []
    for prompt in args.prompts:
        pids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        base_text, base_conf = generate(model, tok, pids, args.new_tokens)
        print(f"\n### {prompt}\nBASE: {base_text[:150]}")
        for tgt in args.targets:
            tids_t = tok(" " + tgt.strip(), add_special_tokens=False).input_ids
            trow = W[tids_t]
            if args.control_random:
                rng_c = np.random.default_rng(7)
                ctrl = W[rng_c.choice(np.arange(len(W)), size=args.control_random,
                                      replace=False)].mean(0)
                trow = np.stack([trow.mean(0) - ctrl])
            first_tok_str = tok.decode([tids_t[0]])
            occ_b = base_text.lower().count(" " + tgt.strip().lower())
            print(f"[target '{tgt}' -> {len(tids_t)} ids, first={first_tok_str!r}]")
            rows.append(dict(prompt=prompt, target=tgt, mode="base", alpha=0.0,
                             depth=0.0, occurrences=occ_b, conf=base_conf))
            for frac in args.depth_fracs:
                li = int(round(frac * (L - 1)))
                for a in args.alphas:
                    results = {}
                    results = {}
                    h = make_tangent_hook(
                        torch.as_tensor(trow, device=model.device), a)
                    for mode, duty in [("single", 10**9)] + [
                            (f"duty{k}", k) for k in args.duty_cycles]:
                        if mode == "single":
                            hd = model.model.layers[li].register_forward_hook(h)
                            ids = pids.clone()
                            logits = model(ids).logits[0, -1].float()
                            hd.remove()
                            ids = torch.cat([ids, torch.tensor(
                                [[int(logits.argmax())]], device=ids.device)], dim=1)
                            text, conf = generate(model, tok, ids,
                                                  args.new_tokens - 1,
                                                  temperature=args.temperature,
                                                  top_p=args.top_p)
                        else:
                            text, conf = generate(model, tok, pids,
                                                  args.new_tokens,
                                                  temperature=args.temperature,
                                                  top_p=args.top_p,
                                                  hook_fn=h, layer=li, duty=duty)
                        results[mode] = (text, conf)
                    for mode, (text, conf) in results.items():
                        occ = sum(text.lower().count(v) for v in
                                  {tgt.lower(), tgt.lower() + "s"})
                        toks = tok(text, add_special_tokens=False).input_ids
                        div = len(set(toks)) / max(len(toks), 1)
                        print(f"{mode.upper():8s} a={a} d={frac:.2f} '{tgt}' x{occ} "
                              f"conf={conf:.2f} div={div:.2f} | {text[:110]}")
                        rows.append(dict(prompt=prompt, target=tgt, mode=mode,
                                         tag=args.tag,
                                         alpha=a, depth=frac, occurrences=occ,
                                         conf=conf, diversity=div))

    safe = args.model.replace("/", "--")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"sustained_steering__{safe}.csv", index=False)
    print("\n=== mean occurrences / confidence by mode ===")
    print(df.groupby("mode")[["occurrences", "conf"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
