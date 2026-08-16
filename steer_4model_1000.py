import torch, gc, math
from transformers import AutoModelForCausalLM, AutoTokenizer

def test(model_name, label, n=1000):
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
    tok = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    d = model.config.hidden_size; sr = math.sqrt(d); V = model.lm_head.weight.shape[0]
    lm_w = model.lm_head.weight.detach().float().cpu()
    # single hidden state
    inp = tok("Once upon a time", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[-1][0,-1,:].float().cpu()
    hn = h / h.norm() * sr
    hn_n = hn / sr
    # test n random tokens
    torch.manual_seed(0)
    targets = list(set(torch.randint(0, V, (n+200,)).tolist()))[:n]
    r1 = 0; improved = 0
    alpha = 0.3
    for tid in targets:
        w = lm_w[tid] / lm_w[tid].norm()
        baseline_logits = model.lm_head(hn.unsqueeze(0).to("cuda", torch.bfloat16)).float().cpu()[0]
        bl_rank = (baseline_logits > baseline_logits[tid]).sum().item() + 1
        g = w - (w * hn_n).sum().item() * hn_n
        hs = hn + alpha * g * sr
        hs = hs / hs.norm() * sr
        logits = model.lm_head(hs.unsqueeze(0).to("cuda", torch.bfloat16)).float().cpu()[0]
        st_rank = (logits > logits[tid]).sum().item() + 1
        if st_rank == 1: r1 += 1
        if st_rank < bl_rank: improved += 1
    print(f"{label:20s} rank1={100*r1/n:.1f}% improved={100*improved/n:.1f}%", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()

test("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B")
test("deepseek-ai/deepseek-llm-7b-chat", "DeepSeek-7B")
test("Qwen/Qwen2-1.5B-Instruct", "Qwen2-1.5B")
test("mistralai/Mistral-7B-v0.1", "Mistral-7B")
print("DONE", flush=True)
