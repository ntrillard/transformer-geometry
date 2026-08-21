# Cross-family steering geometry (t64 c2, depth-adaptive layers 0/0.33/0.67/0.99, fp16)

| model | cases | arc-reach | target-tan | wrong-tan | random-tan | offarc-rand | toward | away | med ° | q25° | q75° | p90° |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen--Qwen2-1.5B-Instruct | 512 | 99.2% | 99.2% | 1.6% | 0.0% | 99.2% | 34.8% | 100.0% | 8.00 | 6.4 | 9.9 | 11.3 |
| Qwen--Qwen2-0.5B-Instruct | 512 | 97.3% | 97.3% | 1.6% | 0.0% | 97.7% | 13.7% | 100.0% | 10.60 | 9.0 | 12.2 | 14.1 |
| openai-community--gpt2 | 512 | 90.8% | 90.8% | 1.0% | 0.0% | 90.6% | 5.9% | 99.2% | 11.60 | 1.3 | 13.8 | 15.4 |
| HuggingFaceTB--SmolLM-135M-Instruct | 512 | 67.2% | 67.2% | 0.4% | 0.0% | 66.8% | 2.5% | 96.7% | 9.84 | 6.2 | 13.4 | 15.3 |
| EleutherAI--pythia-160m | 512 | 27.9% | 27.9% | 0.4% | 0.0% | 27.9% | 9.0% | 70.1% | 2.64 | 2.0 | 3.7 | 9.3 |

Note: median/q25/q75/p90 are over **reachable** cases (first-rank-1 angle defined); 'reachable' = target-row arc reaches rank-1 within the 17° budget. Verified property: in all 2,560 cases a target that gains rank-1 along the arc is still rank-1 at the arc endpoint (no mid-arc loss; also confirmed at a 45° budget).