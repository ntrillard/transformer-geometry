#!/usr/bin/env python3
"""Cheap cow-tipping robustness + defensive-encoding mitigation eval.

A) Find self-consistent (pit) tokens on Qwen2-0.5B via a small scan, then test
   fixed-point robustness across:
     - precision: fp16 / bf16 / int8 (bitsandbytes)
     - sampling: greedy vs temperature 1.0/top-p 0.9 vs temperature 0.8
     - chat template: trigger inside template vs plain continuation
B) Defensive-encoding mitigation:
     - baseline loop length when a chunk terminates in a pit trigger
     - control-character sanitisation (strip non-printable) effect
     - repetition-loop detection (halt after >=4 identical pit tokens), plus a
       normal-generation harm check (does the detector truncate ordinary text?)

Run: python eval_defense.py     (Qwen2-0.5B cached; < ~90 s)
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import steering_geometry_test as M

OUT = Path("steering_geometry_results")
MODEL = "google/gemma-3-1b-it"
SCAN = 600
MAX_NEW = 30
DETECT_RUN = 4
NORMAL_PROMPTS = 10


@torch.no_grad()
def state_after_tokens(model, ids, layer=-1):
    out = model(torch.tensor([ids], device=model.device), output_hidden_states=True)
    return out.hidden_states[layer][0, -1].float()


@torch.no_grad()
def self_score(model, tok, t, head):
    h = state_after_tokens(model, [t, t, t])
    logits = h.float() @ head.T
    p = F.softmax(logits, dim=0)[t].item()
    return p, logits


@torch.no_grad()
def permanence(model, tok, t, head, steps=8):
    ids = [t, t, t]
    n_keep = 0
    cos_last = None
    h_prev = None
    for _ in range(steps):
        h = state_after_tokens(model, ids)
        logits = h.float() @ head.T
        nxt = int(logits.argmax().item())
        if cos_last is None:
            h_prev = h
        else:
            cos_last = float(F.cosine_similarity(h_prev[None], h[None]).item())
            h_prev = h
        if nxt == t:
            n_keep += 1
            ids.append(t)
        else:
            ids.append(nxt)
            break
    return n_keep / steps, cos_last, ids


def pit_scan(model, tok, head, cand):
    ids = [t for t in cand]
    tallies = []
    for i in range(0, len(ids), 256):
        batch = ids[i:i + 256]
        inp = torch.tensor([batch], device=model.device)          # (1,B)
        out = model(inp, output_hidden_states=True)
        h = out.hidden_states[-1][0]                               # (B,d) last tokens
        logits = h.float() @ head.T                                        # (B,V)
        p = F.softmax(logits, dim=1)
        for j, t in enumerate(batch):
            tallies.append((float(p[j, t].item()), t))
    tallies.sort(reverse=True)
    return tallies[:8]


def gen_with_detector(model, tok, head, prompt_ids, pit, max_new=MAX_NEW,
                      temperature=None, top_p=None, top_p_mode="weighted",
                      stop_on_repeat=None):
    """Autoregressive loop with optional temperature / top-p and repeat detector.

    top_p_mode:
        "weighted" -> standard top-p: softmax, retain nucleus, renormalize,
                      then multinomial sample (respects in-nucleus ratios).
        "uniform"  -> retain nucleus, then uniform sample over retained tokens
                      (flattens the model's in-nucleus concentration).
    """
    ids = list(prompt_ids)
    consec = 0
    for _ in range(max_new):
        h = state_after_tokens(model, ids)
        logits = h.float() @ head.T
        if temperature:
            logits = logits / temperature
        probs = F.softmax(logits, dim=0)
        if top_p:
            order = probs.argsort(descending=True)
            cum = probs[order].cumsum(0)
            k = int((cum <= top_p).sum().item()) + 1
            k = min(k, len(probs))
            retained = order[:k]
            if top_p_mode == "uniform":
                chosen = retained[torch.randint(0, len(retained), (1,)).item()]
            else:  # probability-weighted, renormalized nucleus
                p_ret = probs[retained]
                p_ret = p_ret / p_ret.sum()
                idx = torch.multinomial(p_ret, 1).item()
                chosen = retained[idx]
            top = int(chosen.item())
        elif temperature:
            top = int(torch.multinomial(probs, 1).item())
        else:
            top = int(logits.argmax().item())
        ids.append(top)
        consec = consec + 1 if stop_on_repeat is not None and top == pit else 0
        if stop_on_repeat is not None and consec >= stop_on_repeat:
            break
    return ids


def count_run(ids, pit):
    """Length of the longest trailing run of the pit token."""
    n = 0
    for x in reversed(ids):
        if x == pit:
            n += 1
        else:
            break
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seeds", type=int, default=8,
                    help="number of stochastic decoder trials")
    args = ap.parse_args()

    t0 = time.perf_counter()
    results = {}

    def load(dtype="fp16", quant=None):
        return M.load_model(args.model, dtype=dtype, quantize=quant)

    model, tok = load("fp16")
    head = torch.as_tensor(model.lm_head.weight.detach().float().cpu().numpy(),
                          device=model.device)
    V = head.shape[0]

    # ---- A1: scan for pit tokens ----
    base = sorted({t for pr in M.PROMPTS for t in tok(pr).input_ids})
    cand = set()
    for t in range(V):
        txt = tok.decode([t])
        if txt and txt.strip() and all(32 <= ord(c) < 127 for c in txt) and len(txt) <= 6:
            cand.add(t)
        if len(cand) >= SCAN:
            break
    cand = list(cand)[:SCAN]
    pits = pit_scan(model, tok, head, cand)
    ctl = cand[len(cand) // 2]
    print("\n--- A1 pit scan (top self-consistent tokens) ---")
    for p, t in pits[:5]:
        print(f"  s={p:.3f}  id={t:<6} {tok.decode([t])!r}")

    # pick strongest + a control (random mid token)
    STRONG = [t for p,t in pits if p >= 0.4]
    print("  strong pits found:", [tok.decode([t]) for t in STRONG] or "NONE (no self-consistent token at this scale)")
    pit = STRONG[0] if STRONG else ctl
    ctl = cand[len(cand) // 2]

    # ---- A3: temperature / sampling + chat template (greedy loop rate) ----
    print("\n--- A3 sampling & chat-template robustness (pit token) ---")
    m, tk = model, tok  # use the already-loaded fp16 model
    head_t = head
    trig = [pit, pit, pit, pit, pit]
    decoder_cfgs = [
        ("greedy", dict()),
        ("multinomial T=0.8", dict(temperature=0.8)),
        ("top-p 0.9 T=1.0 weighted", dict(temperature=1.0, top_p=0.9, top_p_mode="weighted")),
        ("top-p 0.9 T=0.8 weighted", dict(temperature=0.8, top_p=0.9, top_p_mode="weighted")),
        ("top-p 0.9 T=0.8 uniform", dict(temperature=0.8, top_p=0.9, top_p_mode="uniform")),
    ]
    for label, kw in decoder_cfgs:
        runs = [count_run(gen_with_detector(m, tk, head_t, trig, pit, max_new=20, **kw), pit)
                for _ in range(args.seeds)]
        results[label] = float(np.mean(runs))
        print(f"  {label:26s} mean trailing-pit tokens/run = {np.mean(runs):.1f}")
    # template
    chat_ids = tok.apply_chat_template(
        [{"role": "user", "content": tok.decode(trig)}], add_generation_prompt=True)
    if hasattr(chat_ids, "input_ids"):
        chat_ids = chat_ids.input_ids
    templ_len = count_run(gen_with_detector(m, tk, head_t, chat_ids, pit, max_new=20), pit)
    plain_len = count_run(gen_with_detector(m, tk, head_t, trig, pit, max_new=20), pit)
    print(f"  chat-template wrapped trigger: trailing-pit = {templ_len}  (plain: {plain_len})")
    results["template"] = templ_len

    # ---- B: defensive-encoding mitigation ----
    print("\n--- B mitigation eval (defensive-encoding) ---")
    bert = tok("The final balance is ")["input_ids"]
    base_chunk = bert + trig
    base_loop = count_run(gen_with_detector(m, tk, head_t, base_chunk, pit, max_new=30), pit)
    print(f"  baseline: chunk ending in pit trigger -> trailing pit tokens = {base_loop}")
    # sanitisation: strip non-printable/control characters from the chunk
    txt = tok.decode(base_chunk)
    sanitized = "".join(ch for ch in txt if ch.isprintable())
    san_ids = tok(sanitized)["input_ids"]
    san_loop = count_run(gen_with_detector(m, tk, head_t, san_ids, pit, max_new=30), pit)
    print(f"  after control-char sanitisation: trailing pit tokens = {san_loop}  "
          f"(trigger {'removed' if san_loop < base_loop else 'survives'})")
    # repetition-loop detection: halt after DETECT_RUN identical pit tokens
    det_ids = gen_with_detector(m, tk, head_t, base_chunk, pit, max_new=30,
                                stop_on_repeat=DETECT_RUN)
    det_loop = count_run(det_ids, pit)
    emitted = len(det_ids) - len(base_chunk)
    print(f"  repetition detector (>= {DETECT_RUN} same pit): emitted {emitted} tokens, "
          f"trailing pit = {det_loop}")
    # normal-generation harm: does the detector truncate ordinary prompts?
    norms = [tok(p)["input_ids"][: 8] for p in M.PROMPTS[:NORMAL_PROMPTS]]
    harm = []
    for nids in norms:
        full = len(gen_with_detector(m, tk, head_t, nids, pit, max_new=30)) - len(nids)
        det = len(gen_with_detector(m, tk, head_t, nids, pit, max_new=30,
                                    stop_on_repeat=DETECT_RUN)) - len(nids)
        harm.append(full - det)
    print(f"  normal-generation harm: median truncation = {np.median(harm):.0f} tokens "
          f"(detector vs no-detector over {len(harm)} prompts)")
    results["sanitized_loop"] = san_loop
    results["detector_loop"] = det_loop
    results["emitted_before_halt"] = emitted
    results["normal_harm_median"] = float(np.median(harm))

    # Free the fp16 model before loading other precisions to avoid OOM.
    del model, tok, head, m, tk, head_t
    torch.cuda.empty_cache()

    # ---- A2: robustness across precision/quantization ----
    print("\n--- A2 fixed-point robustness (precision/quantization) ---")
    rows = []
    for cfg in [("fp16", None), ("bf16", None), ("fp16", "int8")]:
        dtype, quant = cfg
        mm, tk = load(dtype, quant)
        head_t = torch.as_tensor(mm.lm_head.weight.detach().float().cpu().numpy(),
                                device=mm.device)
        for label, t in [("pit", pit), ("control", ctl)]:
            s, _ = self_score(mm, tk, t, head_t)
            keep, cosl, _ = permanence(mm, tk, t, head_t)
            rows.append(dict(config=f"{dtype}~{quant or 'full'}", token=label,
                             s=round(s, 3), permanence=round(keep, 2), cos=cosl))
        del mm
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    safe = args.model.replace("/", "--")
    pd.DataFrame([results]).to_csv(OUT / f"defense_mitigation__{safe}.csv", index=False)
    df.to_csv(OUT / f"cowtip_robustness__{safe}.csv", index=False)
    print(f"\nSaved -> {OUT}/cowtip_robustness__{safe}.csv, "
          f"{OUT}/defense_mitigation__{safe}.csv   ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()