**Re #13 (John) and #14 (AdrienneNoctis): both answered with code and numbers.**

John — your blocker-branch finding is correct and I've now confirmed it inside the exact code path that produced the cross-family table. AdrienneNoctis — your three objections each deserve a measurement rather than a metaphor, so here are the measurements. Every number below is reproducible **without a GPU** in well under a minute; the three model-free verifiers are committed to the repo.

**Re #13 — the fixed-score blocker contract: confirmed**

In [steering_geometry_test.py](https://github.com/ntrillard/transformer-geometry/blob/404b0a4/steering-evals/scripts/steering_geometry_test.py) the toward-blocker direction is built as `q = W_b − (W_b·s)s`, normalized — it removes the `s` component but not the residual-axis component `r`, so `q·r ≠ 0` in general, and renormalizing the endpoint rescales `s` and drifts the target score. The random-residual branch already orthogonalizes against both `s` and `r`; the blocker branch was the one with the weaker contract.

Verified in [verify_offarc_contract.py](https://github.com/ntrillard/transformer-geometry/blob/c486e28/steering-evals/scripts/verify_offarc_contract.py) on rows with realistic low-rank structure:

- committed construction: max `|q·r| = 0.496`, max target-score drift `0.021` — matching your 0.58 / 0.02–0.03 on real rows;
- exact construction `q ∝ W_b − (W_b·s)s − (W_b·r)r`: both invariants at machine precision (2e-16);
- the toward/away **rank conclusions are unchanged** by the fix: toward-blocker destroys target rank-1 in 163/2933 cases under the committed construction vs 159/2933 under the exact one.

So the violation is real; the competitor-orientation effect survives the correction; the Pythia drift is a second, separate provenance signal.

The fix is being applied to the same file as your construction, with invariant asserts on `|q·s|`, `|q·r|`, norm, target-score and raw-logit errors so it fails loudly next time. The frozen-row rerun is queued — it needs a GPU with a model download, and this machine is currently out of disk (4 GB free against the harness's own 8 GB download floor).

Your 2-D separator example is the best one-liner in the thread, and it verifies exactly as written: `u=(1,0), Wt=(1,1), Wj=(0.9,10)` — target logit rises monotonically along the arc while the competitor overtakes at **0.638°**. That is now the canonical statement of the split: **logit ascent along the target tangent is guaranteed; rank ascent is empirical.** (The committed rows agree with that split at scale — see the audit below.)

On the rest of your harness list: wrong-target exclusion is already structural in the shipped code (sampled from the other K−1, no self-draws); the analytic first-rank-1 angle agrees with the 200-step scan to within scan resolution ([verify_harness_units.py](https://github.com/ntrillard/transformer-geometry/blob/c486e28/steering-evals/scripts/verify_harness_units.py)); the duplicate off-arc implementation is being folded into one canonical helper, and torch is being seeded alongside numpy (explicit generator into the batched block) so `--seed 42` is bit-reproducible.

**Re #14 — non-linearity, silicon, training: all three, measured**

*Non-linearity.* Conceded — that is exactly what the cross-family spread *is*. The logit guarantee is a property of the LM-head readout at a fixed state (a linear map by definition), not of the 12–64 attention/FFN layers in between. Those layers move states wherever they want; the question we measure is how often the target-tangent arc still reaches rank 1 from where they land. It does not do so 100% of the time (Pythia-160M: 27.9%; Qwen2-1.5B: 99.2%). It does so a reproducible fraction, with wrong-target and random-tangent controls at **exactly 0.0% on all five models, n = 2,560**, and with the toward/away ordering `toward < arc ≤ away` holding on every model ([verify_csv_audit.py](https://github.com/ntrillard/transformer-geometry/blob/c486e28/steering-evals/scripts/verify_csv_audit.py)). One scope note, so nobody over-reads the table: these are *readout-level* numbers — the final LM head applied directly to hidden states. Steering *through* the network (intervening at a layer and letting the remaining layers transform the state) is a separate validation branch, covered by the gated/practical steering battery.

*Silicon.* The identity the whole method leans on — tangent step plus renormalization is a same-target great-circle rotation — is arithmetic on the model's own weights, and it verifies to `4.2e-17` across 200 random (u, s) pairs in the already-committed [verify_identity.py](https://github.com/ntrillard/transformer-geometry/blob/404b0a4/steering-evals/scripts/verify_identity.py). There is no chip-dependent step in that statement; it holds wherever IEEE floats do. On the deployment side the pit branch already measures under 4-bit nf4 quantization on the 7B, and the gated/practical steering battery produces the effect in *generated text* under stochastic decoders (top-p, temperature), not just logit assays. As for the "±3° from thermal noise" — that number is cited nowhere in this thread, and I'd genuinely like to see the noise model. Point me at one and I'll test exactly that.

*Training.* This is the one I'd push back on hardest, because it's the sharpest control we have — run **today**, on the shipped code path, on rows nobody ever trained. Executing `_batched_block` on *random, untrained matrices* reproduces the same qualitative control ordering ([verify_harness_units.py](https://github.com/ntrillard/transformer-geometry/blob/c486e28/steering-evals/scripts/verify_harness_units.py)):

```
target tangent       89.1%  rank-1
wrong-target tangent  0.0%
random tangent        0.0%
toward blocker       21.9%   (collapse)
away blocker         95.3%   (restored)
```

Nothing in those rows was learned. That is the LM-head decision partition — it exists in any linear head, trained or not. What RLHF/training actually changes is the *rates* (27.9% → 99.2% reach) and the *fluency trade-offs*, both of which the repo reports as measured facts, not as consequences of geometry. The dog runs along the leash, yes — but the leash is bolted to the partition.

---

Both posts earned their reply. The code and the numbers are the argument; the verifiers are in the repo so anyone can rerun them without a GPU.