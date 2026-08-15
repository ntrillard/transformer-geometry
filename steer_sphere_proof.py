#!/usr/bin/env python3
"""Sphere steering: reproduce Proof 5 with the paper's exact method, run efficiently.

Method (from paper_final.tex Proof 5):
  h <- final hidden state at last prompt position
  g_t = W_t - (W_t . h_hat) * h_hat          (tangent to sphere for token t)
  h <- h + alpha * g_t / ||g_t|| * sqrt(d)    (alpha = 0.3)
  h <- h / ||h|| * sqrt(d)                    (renormalize to sphere)
  generate normally; ONLY the first token is steered.
"""
import torch, math, json, re, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEVICE = "cuda"
DTYPE = torch.bfloat16
OUTPUT = {}

def extract_gsm8k_answer(text):
    """Extract final numeric answer from GSM8K CoT output."""
    m = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if m: return float(m.group(1).replace(",", ""))
    m = re.search(r"answer\s*(?:is|:|=)\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m: return float(m.group(1))
    m = re.search(r"\boxed\{(-?\d+(?:,\d+)*(?:\.\d+)?)\}", text)
    if m: return float(m.group(1).replace(",", ""))
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if nums: return float(nums[-1])
    return None

def sphere_steer(model, tokenizer, prompt, target_token, alpha=0.3, max_new=200):
    """Steer the first generated token toward target_token, then generate normally."""
    d = model.config.hidden_size
    sphere_r = math.sqrt(d)
    lm_head_w = model.lm_head.weight.detach().float().cpu()

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1, :].float().cpu()  # pre-norm final hidden state

    # Tangent direction: average over target sub-tokens
    tids = tokenizer.encode(target_token)
    w = sum(lm_head_w[tid] for tid in tids) / len(tids)
    h_hat = h / h.norm()
    g = w - (w @ h_hat) * h_hat
    if g.norm() > 0:
        g = g / g.norm() * sphere_r

    # Steer + renormalize onto the sphere
    h_s = h + alpha * g
    h_s = h_s / h_s.norm() * sphere_r

    # Hook the final norm to replace the first-step hidden state only
    norm = model.model.norm
    state = {"steer": True, "h_s": h_s.to(DEVICE, dtype=DTYPE)}
    def hook(module, inp, o):
        if not state["steer"]:
            return o
        o2 = o.clone()
        o2[0, -1, :] = state["h_s"]
        state["steer"] = False
        return o2
    handle = norm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id,
                                 use_cache=True)
    finally:
        handle.remove()
    return tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def baseline_generate(model, tokenizer, prompt, max_new=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# ================ MAIN ================
print("=" * 60)
print("SPHERE STEERING - Proof 5 Verification")
print("=" * 60)

model_name = "Qwen/Qwen2-1.5B-Instruct"
print(f"\nLoading {model_name}...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE, device_map=DEVICE)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
d = model.config.hidden_size
print(f"Loaded in {time.time()-t0:.1f}s - {d}d, {len(model.model.layers)}L, sphere r=sqrt({d})={math.sqrt(d):.2f}")

ds = load_dataset("openai/gsm8k", "main", split="test")
N = 50
print(f"\nTesting on {N} GSM8K problems (first {N} of test set)")

problems = []
for i in range(N):
    item = ds[i]
    gt_m = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", item["answer"])
    if not gt_m:
        continue
    gt = float(gt_m.group(1).replace(",", ""))
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": item["question"]}],
        tokenize=False, add_generation_prompt=True)
    problems.append((prompt, gt, item["question"]))
print(f"  {len(problems)} problems with ground truth")

# ==================== BASELINE ====================
print(f"\n--- BASELINE (no steering) ---")
baseline_results = []
ex_base = []
for i, (prompt, gt, q) in enumerate(problems):
    out = baseline_generate(model, tokenizer, prompt)
    pred = extract_gsm8k_answer(out)
    correct = pred is not None and abs(pred - gt) < 1e-6
    baseline_results.append(correct)
    if i < 3:
        ex_base.append(out[:90].replace("\n", " "))
        print(f"  [{i}] {out[:90].replace(chr(10),' ')}... -> {pred} vs {gt} {'OK' if correct else 'X'}")
bl_acc = sum(baseline_results) / len(baseline_results) * 100
OUTPUT["baseline"] = {"accuracy": round(bl_acc, 1), "n": len(baseline_results), "examples": ex_base}
print(f"  Baseline: {bl_acc:.1f}% (n={len(baseline_results)})")

# ==================== STEERING ====================
targets = ["answer", "solve", "result", "75"]
for target in targets:
    print(f"\n--- STEER toward '{target}' (alpha=0.3) ---")
    steer_results = []
    ex = []
    for i, (prompt, gt, q) in enumerate(problems):
        out = sphere_steer(model, tokenizer, prompt, target, alpha=0.3)
        pred = extract_gsm8k_answer(out)
        correct = pred is not None and abs(pred - gt) < 1e-6
        steer_results.append(correct)
        if i < 3:
            ex.append(out[:90].replace("\n", " "))
            print(f"  [{i}] {out[:90].replace(chr(10),' ')}... -> {pred} vs {gt} {'OK' if correct else 'X'}")
    acc = sum(steer_results) / len(steer_results) * 100
    OUTPUT[target] = {"accuracy": round(acc, 1), "n": len(steer_results), "examples": ex}
    print(f"  '{target}': {acc:.1f}%")

# ==================== ALPHA SWEEP (top target only) ====================
best = max(targets, key=lambda t: OUTPUT[t]["accuracy"])
print(f"\n--- ALPHA SWEEP for '{best}' (alpha in [0,0.1,0.3,0.5,1.0]) ---")
alpha_res = {}
for alpha in [0.0, 0.1, 0.3, 0.5, 1.0]:
    res = []
    for i, (prompt, gt, q) in enumerate(problems):
        out = sphere_steer(model, tokenizer, prompt, best, alpha=alpha)
        pred = extract_gsm8k_answer(out)
        res.append(pred is not None and abs(pred - gt) < 1e-6)
    acc = sum(res) / len(res) * 100
    alpha_res[str(alpha)] = round(acc, 1)
    print(f"  alpha={alpha}: {acc:.1f}%")
OUTPUT["alpha_sweep"] = {"target": best, "results": alpha_res}

# ==================== SUMMARY ====================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Baseline: {OUTPUT['baseline']['accuracy']}%")
for target in targets:
    r = OUTPUT[target]
    print(f"  '{target}': {r['accuracy']}% (delta={r['accuracy']-OUTPUT['baseline']['accuracy']:+.1f}pp)")

with open("steer_proof5_results.json", "w") as f:
    json.dump(OUTPUT, f, indent=2)
print(f"\nSaved to steer_proof5_results.json")
