# Cross-family steering geometry (t64 c2, depth-adaptive layers 0/0.33/0.67/0.99, fp16, seed 42)

| model | cases | arc-reach | target-tan | wrong-tan | random-tan | offarc-rand | toward | away | med ° | q25° | q75° | p90° |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen--Qwen2-1.5B-Instruct | 512 | 99.2% | 99.2% | 0.0% | 0.0% | 99.2% | 34.8% | 100.0% | 8.0 | 6.42 | 9.93 | 11.33 |
| Qwen--Qwen2-0.5B-Instruct | 512 | 97.3% | 97.3% | 0.0% | 0.0% | 97.5% | 13.7% | 100.0% | 10.6 | 8.96 | 12.25 | 14.11 |
| openai-community--gpt2 | 512 | 90.8% | 90.8% | 0.0% | 0.0% | 90.4% | 5.9% | 99.2% | 11.6 | 1.3 | 13.84 | 15.42 |
| HuggingFaceTB--SmolLM-135M-Instruct | 512 | 67.2% | 67.2% | 0.0% | 0.0% | 67.0% | 2.5% | 96.7% | 9.84 | 6.15 | 13.36 | 15.3 |
| EleutherAI--pythia-160m | 512 | 27.9% | 27.9% | 0.0% | 0.0% | 28.1% | 9.0% | 70.1% | 2.64 | 2.0 | 3.67 | 9.27 |

Note: wrong-target control is strictly self-excluding (fixed 2026-08-21; prior version allowed 1/64 self-draws, inflating this column to ~1.6%).
median/q25/q75/p90 are over **reachable** cases (first-rank-1 angle defined). Verified property: in all 2,560 cases a target that gains rank-1 along the arc is still rank-1 at the arc endpoint (no mid-arc loss within the 17° budget).
