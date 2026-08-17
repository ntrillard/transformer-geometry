# Cross-family steering geometry (t64 c2, depth-adaptive layers 0/0.33/0.67/0.99, fp16)

| model | cases | target-tan | wrong-tan | random-tan | offarc-rand | toward-blk | away-blk | med rank1 angle |
|---|---|---|---|---|---|---|---|---|
| Qwen--Qwen2-1.5B-Instruct | 512 | 99.2% | 1.6% | 0.0% | 99.2% | 34.8% | 100.0% | 8.00° |
| Qwen--Qwen2-0.5B-Instruct | 512 | 97.3% | 1.6% | 0.0% | 97.7% | 13.7% | 100.0% | 10.60° |
| openai-community--gpt2 | 512 | 90.8% | 1.0% | 0.0% | 90.6% | 5.9% | 99.2% | 11.60° |
| HuggingFaceTB--SmolLM-135M-Instruct | 512 | 67.2% | 0.4% | 0.0% | 66.8% | 2.5% | 96.7% | 9.84° |
| EleutherAI--pythia-160m | 512 | 27.9% | 0.4% | 0.0% | 27.9% | 9.0% | 70.1% | 2.64° |
