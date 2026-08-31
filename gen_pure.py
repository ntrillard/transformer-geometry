#!/usr/bin/env python3
"""gen_pure.py - PURE unsteered generation. No hooks, no injection, no
anti, no families, no steering of any kind. Just the model + multinomial
sampling from its own logits, exactly as the base model would write.

Run: HF_TOKEN=<tok> python3 gen_pure.py [model] [prompt] [ntok] [seed]
"""
import sys
import time

import torch
import transformers

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2-1.5B'
PROMPT = (sys.argv[2] if len(sys.argv) > 2
          else 'Tell a story')
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 120
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    t0 = time.time()
    print(f'\nPure generation | {MODEL} | prompt={PROMPT!r} | '
          f'ntok={NTOK} | seed={SEED}')
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True).eval()
    eos_id = int(tok.eos_token_id)

    torch.manual_seed(SEED)
    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    sampled = []
    for _ in range(NTOK):
        with torch.no_grad():
            L = model(ids).logits[0, -1].float()
        L = torch.nan_to_num(L, nan=-50.0).clamp(-50.0, 50.0)
        p = torch.softmax(L, 0)
        nxt = int(torch.multinomial(p, 1))
        if nxt == eos_id:
            sampled.append(nxt)
            break
        sampled.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)

    txt = tok.decode(sampled)
    print(f'\n===== PURE / NO STEERING =====')
    print(f'{PROMPT} {txt}')
    print(f'\n[{time.time() - t0:.0f}s] {len(sampled)} tokens, '
          f'eos={"YES" if sampled and sampled[-1] == eos_id else "NO"}')


if __name__ == "__main__":
    main()