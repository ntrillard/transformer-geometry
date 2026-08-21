#!/usr/bin/env python3
"""Threat-model and mitigation evaluation for cow-tipping / repetition pits.

For a known or discovered pit token, measures:
  - baseline loop length under greedy decoding
  - loop-break rate of input/output mitigations
  - false-positive rate of each mitigation on ordinary prompts

Run: python eval_threat_model.py --model Qwen/Qwen2-0.5B-Instruct
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import eval_defense as E
import steering_geometry_test as M

OUT = Path("steering_geometry_results")
MODEL = "Qwen/Qwen2-0.5B-Instruct"
MAX_NEW = 30
NORMAL_PROMPTS = 20


def entropy_of_logits(logits):
    probs = F.softmax(logits, dim=0)
    log_probs = F.log_softmax(logits, dim=0)
    return -(probs * log_probs).sum().item()


@torch.no_grad()
def gen_with_mitigations(model, tok, head, prompt_ids, pit, max_new=MAX_NEW,
                         stop_on_repeat=None, entropy_floor=None):
    """Autoregressive loop with optional repetition and entropy mitigations."""
    ids = list(prompt_ids)
    consec = 0
    alerts = 0
    for _ in range(max_new):
        h = E.state_after_tokens(model, ids)
        logits = h.float() @ head.T
        top = int(logits.argmax().item())
        # entropy monitor
        if entropy_floor is not None and entropy_of_logits(logits) < entropy_floor:
            alerts += 1
            # mitigation: force a non-pit token (the second-highest logit)
            top = int(torch.topk(logits, k=2).indices[1].item())
        ids.append(top)
        consec = consec + 1 if stop_on_repeat is not None and top == pit else 0
        if stop_on_repeat is not None and consec >= stop_on_repeat:
            break
    return ids, alerts


def sanitize_repeated_tokens(tok, ids, pit, max_repeat=3):
    """Collapse runs of the pit token longer than max_repeat in the input."""
    out = []
    run = 0
    for t in ids:
        if t == pit:
            run += 1
            if run <= max_repeat:
                out.append(t)
        else:
            run = 0
            out.append(t)
    return out


def sanitize_control_chars(tok, ids):
    """Remove input ids that decode to non-printable text."""
    txt = tok.decode(ids)
    cleaned = "".join(ch for ch in txt if ch.isprintable())
    return tok(cleaned, add_special_tokens=False).input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--pit-id", type=int, default=None,
                    help="optional known pit token id; if omitted, scan for one")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tok = M.load_model(args.model, dtype="fp16")
    head = torch.as_tensor(model.lm_head.weight.detach().float().cpu().numpy(),
                           device=model.device)

    # Discover or use known pit
    if args.pit_id is not None:
        pit = args.pit_id
        print(f"Using provided pit token: {tok.decode([pit])!r} (id {pit})")
    else:
        pit = None
        for t in range(5000):
            s = tok.decode([t])
            if s.strip() and all(32 <= ord(c) < 128 for c in s) and len(s) <= 8:
                n = E.count_run(E.gen_with_detector(model, tok, head, [t] * 5, t, max_new=25), t)
                if n >= 8:
                    pit = t
                    break
        if pit is None:
            print("No greedy self-loop token found.")
            return
        print(f"Discovered pit token: {tok.decode([pit])!r} (id {pit})")

    trig = [pit] * 5
    baseline = E.count_run(E.gen_with_detector(model, tok, head, trig, pit, max_new=MAX_NEW), pit)
    print(f"\nBaseline greedy loop length: {baseline}")

    rows = []

    # 1. Repetition detector at several thresholds
    for thresh in [3, 4, 5]:
        ids, _ = gen_with_mitigations(model, tok, head, trig, pit, stop_on_repeat=thresh)
        loop = E.count_run(ids, pit)
        rows.append({
            "mitigation": f"repetition-detector-thresh-{thresh}",
            "loop_length": loop,
            "loop_broken": int(loop < baseline),
        })
        print(f"  repetition detector (>= {thresh} same pit): loop = {loop}")

    # 2. Entropy monitor
    for floor in [0.5, 1.0, 1.5]:
        ids, alerts = gen_with_mitigations(model, tok, head, trig, pit, entropy_floor=floor)
        loop = E.count_run(ids, pit)
        rows.append({
            "mitigation": f"entropy-floor-{floor}",
            "loop_length": loop,
            "loop_broken": int(loop < baseline),
            "alerts": alerts,
        })
        print(f"  entropy floor {floor}: loop = {loop}, alerts = {alerts}")

    # 3. Input sanitization
    sanitized = sanitize_repeated_tokens(tok, trig, pit, max_repeat=2)
    loop = E.count_run(E.gen_with_detector(model, tok, head, sanitized, pit, max_new=MAX_NEW), pit)
    rows.append({"mitigation": "input-sanitize-repeated-pit", "loop_length": loop, "loop_broken": int(loop < baseline)})
    print(f"  input sanitize repeated pit: loop = {loop}")

    sanitized = sanitize_control_chars(tok, trig)
    loop = E.count_run(E.gen_with_detector(model, tok, head, sanitized, pit, max_new=MAX_NEW), pit)
    rows.append({"mitigation": "input-sanitize-control-chars", "loop_length": loop, "loop_broken": int(loop < baseline)})
    print(f"  input sanitize control chars: loop = {loop}")

    # 4. Decoder change
    for label, kw in [("multinomial T=1.0", dict(temperature=1.0)),
                      ("top-p 0.9 T=1.0 weighted", dict(temperature=1.0, top_p=0.9, top_p_mode="weighted"))]:
        runs = [E.count_run(E.gen_with_detector(model, tok, head, trig, pit, max_new=MAX_NEW, **kw), pit)
                for _ in range(16)]
        rows.append({
            "mitigation": label,
            "loop_length": float(np.mean(runs)),
            "loop_broken": int(np.mean(runs) < baseline),
            "max_repeat": int(max(runs)),
        })
        print(f"  {label}: mean loop = {np.mean(runs):.2f}, max = {max(runs)}")

    # False-positive evaluation on normal prompts
    print(f"\nFalse-positive evaluation on {NORMAL_PROMPTS} normal prompts:")
    norms = [tok(p, add_special_tokens=False).input_ids[:10] for p in M.PROMPTS[:NORMAL_PROMPTS]]
    fp_rows = []

    for thresh in [3, 4, 5]:
        truncations = []
        for nids in norms:
            full_len = len(E.gen_with_detector(model, tok, head, nids, pit, max_new=MAX_NEW))
            det_len = len(gen_with_mitigations(model, tok, head, nids, pit, stop_on_repeat=thresh)[0])
            truncations.append(full_len - det_len)
        fp_rows.append({
            "mitigation": f"repetition-detector-thresh-{thresh}",
            "median_truncation_tokens": float(np.median(truncations)),
            "max_truncation_tokens": int(max(truncations)),
        })
        print(f"  repetition detector (>= {thresh}): median truncation = {np.median(truncations):.0f} tokens")

    for floor in [0.5, 1.0, 1.5]:
        truncations = []
        for nids in norms:
            full_len = len(E.gen_with_detector(model, tok, head, nids, pit, max_new=MAX_NEW))
            det_len = len(gen_with_mitigations(model, tok, head, nids, pit, entropy_floor=floor)[0])
            truncations.append(full_len - det_len)
        fp_rows.append({
            "mitigation": f"entropy-floor-{floor}",
            "median_truncation_tokens": float(np.median(truncations)),
            "max_truncation_tokens": int(max(truncations)),
        })
        print(f"  entropy floor {floor}: median truncation = {np.median(truncations):.0f} tokens")

    safe = args.model.replace("/", "--")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"threat_model_mitigations__{safe}.csv", index=False)
    pd.DataFrame(fp_rows).to_csv(OUT / f"threat_model_false_positives__{safe}.csv", index=False)
    print(f"\nSaved -> {OUT / f'threat_model_mitigations__{safe}.csv'}")
    print(f"Saved -> {OUT / f'threat_model_false_positives__{safe}.csv'}")


if __name__ == "__main__":
    main()
