#!/usr/bin/env python3
"""Fast sphere test suite: Proofs 1,2,3,5 on all cached models.

Tests per model:
  - Sphere norm per layer (Proof 1)        28 tests
  - Attention contraction per layer (Proof 2)  28 tests
  - Lyapunov λ (Proof 3)                     1 test
  - Per-layer λ (zone profile)              28 tests
  - Steering: 20 targets × 5 alphas          100 tests
  Total: ~185 tests per model × 7 models ≈ 1300 tests
  Each test = 1 forward pass or vector op, under 5min total.
"""
import torch, math, json, time, re
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
DTYPE = torch.bfloat16
RESULTS = {}

def measure_norms(model, tok, text, device=DEVICE):
    """Proof 1: hidden state norms after each LN."""
    templ = tok.apply_chat_template([{"role":"user","content":text}], tokenize=False, add_generation_prompt=True)
    inp = tok(templ, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    d = model.config.hidden_size
    sphere_r = math.sqrt(d)
    norms = {}
    for l, hs in enumerate(out.hidden_states):
        n = hs[0, -1, :].float().norm().item()  # last position
        norms[f"L{l}"] = {"norm": round(n, 2), "sphere_ratio": round(n / sphere_r, 3)}
    return norms, sphere_r

def measure_attn_contraction(model, tok, text, device=DEVICE):
    """Proof 2: attention output norm per layer."""
    d = model.config.hidden_size
    sphere_r = math.sqrt(d)
    norms = {}
    attn = {}
    for l in range(len(model.model.layers)):
        def make_hook(ll):
            def hook(m, i, o):
                attn[ll] = o[0][0, -1, :].float().norm().item()
            return hook
        h = model.model.layers[l].self_attn.register_forward_hook(make_hook(l))
        with torch.no_grad():
            templ = tok.apply_chat_template([{"role":"user","content":text}], tokenize=False, add_generation_prompt=True)
            inp = tok(templ, return_tensors="pt").to(device)
            model(**inp)
        h.remove()
    return {f"L{l}": {"h_attn_norm": round(n, 2),
                       "sphere_ratio": round(n / sphere_r, 3),
                       "contraction_pct": round((1 - n / sphere_r) * 100, 1)}
            for l, n in sorted(attn.items())}

def measure_lyapunov(model, tok, text, device=DEVICE, trials=3):
    """Proof 3: Lyapunov exponent λ over computation zone (L/3 to 2L/3)."""
    nL = len(model.model.layers)
    embed = model.model.embed_tokens
    templ = tok.apply_chat_template([{"role":"user","content":text}], tokenize=False, add_generation_prompt=True)
    inp = tok(templ, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ref = model(**inp, output_hidden_states=True)
    embeds = embed(inp.input_ids)
    l_sum = 0
    for _ in range(trials):
        noise = torch.randn_like(embeds) * 1e-4
        with torch.no_grad():
            outp = model(inputs_embeds=embeds + noise, output_hidden_states=True)
        lyaps = []
        for l in range(nL // 3, 2 * nL // 3):
            di = (outp.hidden_states[l][0] - out_ref.hidden_states[l][0]).norm(dim=1).mean().item()
            do = (outp.hidden_states[l+1][0] - out_ref.hidden_states[l+1][0]).norm(dim=1).mean().item()
            if di > 0:
                lyaps.append(math.log(do / di))
        l_sum += sum(lyaps) / len(lyaps) if lyaps else 0
    lam = l_sum / trials
    L_comp = nL // 3
    return {"λ": round(lam, 4), "λ·L/3": round(lam * L_comp, 3), "L": nL, "d": model.config.hidden_size}

def measure_per_layer_lyapunov(model, tok, text, device=DEVICE, trials=2):
    """Per-layer λ for zone profile."""
    nL = len(model.model.layers)
    embed = model.model.embed_tokens
    templ = tok.apply_chat_template([{"role":"user","content":text}], tokenize=False, add_generation_prompt=True)
    inp = tok(templ, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ref = model(**inp, output_hidden_states=True)
    embeds = embed(inp.input_ids)
    per_layer = {}
    for l in range(nL - 1):
        lyaps = []
        for _ in range(trials):
            noise = torch.randn_like(embeds) * 1e-4
            with torch.no_grad():
                outp = model(inputs_embeds=embeds + noise, output_hidden_states=True)
            di = (outp.hidden_states[l][0] - out_ref.hidden_states[l][0]).norm(dim=1).mean().item()
            do = (outp.hidden_states[l+1][0] - out_ref.hidden_states[l+1][0]).norm(dim=1).mean().item()
            if di > 0:
                lyaps.append(math.log(do / di))
        per_layer[f"L{l}"] = round(sum(lyaps) / len(lyaps), 4) if lyaps else 0
    return per_layer

def measure_steering_logit(model, tok, prompt, target_tokens, alphas=[0.0, 0.1, 0.3, 0.5, 1.0], device=DEVICE):
    """Proof 5: measure logit of target tokens before/after sphere steering.

    Fast: no generation, just compute logits at first position.
    """
    d = model.config.hidden_size
    sphere_r = math.sqrt(d)
    lm_head = model.lm_head.weight.detach().float().cpu()

    templ = tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
    inp = tok(templ, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1, :].float().cpu()
    base_logits = lm_head @ h

    results = {}
    for target in target_tokens:
        tids = tok.encode(target)
        # Average embedding over sub-tokens
        w = sum(lm_head[tid] for tid in tids) / len(tids)
        h_hat = h / h.norm()
        g = w - (w @ h_hat) * h_hat
        if g.norm() > 0:
            g = g / g.norm() * sphere_r
        t_results = {}
        for alpha in alphas:
            h_s = h + alpha * g
            h_s = h_s / h_s.norm() * sphere_r
            logits = lm_head @ h_s
            # Logit of each sub-token of target
            target_logit = sum(logits[tid].item() for tid in tids) / len(tids)
            # Rank of first sub-token among all tokens
            rank = (logits > logits[tids[0]]).sum().item()
            # Top-5 tokens
            top5 = [tok.decode([x]) for x in logits.topk(5).indices.tolist()]
            # Cosine similarity between steered and base hidden
            cos_sim = (h_s @ h) / (h_s.norm() * h.norm())
            t_results[f"α={alpha}"] = {
                "target_logit": round(target_logit, 3),
                "rank": rank,
                "top5": top5,
                "cos_sim": round(cos_sim.item(), 4)
            }
        results[target] = t_results
    return {"base_logits_top5": [tok.decode([x]) for x in base_logits.topk(5).indices.tolist()],
            "steering": results}

# ================ Models ================
MODELS = [
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "HuggingFaceTB/SmolLM-360M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/phi-1_5",
]
PROMPT = "A store has 120 apples, sells 45. How many left?"
TARGETS = ["answer","solve","result","the","yes","no","first","then","calculate","therefore",
           "number","value","total","left","remain","sum","difference","product","equation","solution"]

print("=" * 70)
print("SPHERE TEST SUITE — Proofs 1, 2, 3, 5 on all cached models")
print("=" * 70)

for m_name in MODELS:
    short = m_name.split("/")[-1]
    print(f"\n{'─'*70}")
    print(f"Model: {m_name}")
    print(f"{'─'*70}")
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(m_name, dtype=DTYPE, device_map=DEVICE, local_files_only=False)
        model.eval()
        tok = AutoTokenizer.from_pretrained(m_name, local_files_only=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print(f"  Loaded in {time.time()-t0:.1f}s | d={model.config.hidden_size} L={len(model.model.layers)}")
    except Exception as e:
        print(f"  FAILED to load: {e}")
        continue

    mres = {"config": {"d": model.config.hidden_size, "L": len(model.model.layers), "name": m_name}}

    # Proof 1: Sphere norms
    t1 = time.time()
    norms, sphere_r = measure_norms(model, tok, PROMPT)
    mres["proof1_norms"] = norms
    norm_ratio = [v["sphere_ratio"] for v in norms.values()]
    print(f"  Proof1: mean||h||/√d = {sum(norm_ratio)/len(norm_ratio):.2f} (range {min(norm_ratio):.2f}-{max(norm_ratio):.2f}) [{time.time()-t1:.1f}s]")

    # Proof 2: Attention contraction
    t1 = time.time()
    attn = measure_attn_contraction(model, tok, PROMPT)
    mres["proof2_attn_contraction"] = attn
    avg_contr = sum(v["contraction_pct"] for v in attn.values()) / len(attn)
    print(f"  Proof2: mean contraction = {avg_contr:.1f}% [{time.time()-t1:.1f}s]")

    # Proof 3: Lyapunov λ
    t1 = time.time()
    lyap = measure_lyapunov(model, tok, PROMPT, trials=3)
    mres["proof3_lyapunov"] = lyap
    print(f"  Proof3: λ={lyap['λ']}  λ·L/3={lyap['λ·L/3']}  [{time.time()-t1:.1f}s]")

    # Per-layer λ
    t1 = time.time()
    per_lyap = measure_per_layer_lyapunov(model, tok, PROMPT, trials=2)
    mres["proof3_per_layer_λ"] = per_lyap
    print(f"  Per-layer λ: min={min(per_lyap.values()):.4f} max={max(per_lyap.values()):.4f} [{time.time()-t1:.1f}s]")

    # Proof 5: Steering
    t1 = time.time()
    steer = measure_steering_logit(model, tok, PROMPT, TARGETS)
    mres["proof5_steering"] = steer
    # Summary: for each target, the best α's rank
    for t in TARGETS[:5]:
        ranks = [v["rank"] for k, v in steer["steering"][t].items()]
        best_rank = min(ranks)
        best_alpha = list(steer["steering"][t].keys())[ranks.index(best_rank)]
        print(f"  Steering '{t}': best rank={best_rank} at {best_alpha}")

    # Cleanup
    del model, tok
    torch.cuda.empty_cache()
    RESULTS[m_name] = mres

# ================ Summary table ================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"{'Model':<30} {'d':>5} {'L':>4} {'λ':>8} {'λ·L/3':>8} {'√d':>5} {'avg||h||/√d':>12} {'avg_contr':>10}")
print("-" * 82)
for m_name, mres in RESULTS.items():
    short = m_name.split("/")[-1]
    c = mres["config"]
    lyap = mres["proof3_lyapunov"]
    norms = mres["proof1_norms"]
    attn = mres["proof2_attn_contraction"]
    avg_ratio = sum(v["sphere_ratio"] for v in norms.values()) / len(norms)
    avg_contr = sum(v["contraction_pct"] for v in attn.values()) / len(attn)
    print(f"{short:<30} {c['d']:>5} {c['L']:>4} {lyap['λ']:>8} {lyap['λ·L/3']:>8.3f} {math.sqrt(c['d']):>5.1f} {avg_ratio:>12.2f} {avg_contr:>9.1f}%")

with open("sphere_test_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\nSaved to sphere_test_results.json")