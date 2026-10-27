POST-HOC CHECK (non-gating; no delivery; no new arm or configuration). Run on 27 October 2026 during the self-review, before the owner asked for review rather than verification. No further checks follow.

# V164: is COLA's fixed ε negligible?

**Question** (Codex synthesis, memo 02 §9): V164 computes COLA's errors from the batch-mean loss, so the per-unit statistics are small. Is the fixed ε = 1e-8 in α = S/(Q + ε) and μ = S_AB/(S_A2 + ε) negligible next to them?

**Method:** `cola_eps_check.py` runs V164's COLA code, unchanged.
* Part 1: the first 5 batches from V164's initialisation.
* Part 2: re-executes V164's committed C2 COLA training. It reproduces V164's final α and μ quantiles exactly before anything else is read.

The values "without ε" are computed from the same running averages. Output: `cola_eps_check_output.txt`.

**Result: ε is not negligible.** At the end of training on S-COPY:

| Statistic | With ε | Without ε |
|---|---:|---:|
| Median Q | 1.3e-9 | – |
| Median \|α\| | 0.003 | 0.028 |
| Median μ | 0.09 | 1.39 |
| Unit-steps at the μφ′ cap | 0% | 31% |

Text is affected less: median μ is 0.85 with ε and 1.64 without.

**Reading:**
* V164 tested a COLA whose statistics were dominated by ε. That made it numerically close to one-step truncation (TBPTT-1).
* The structural argument still stands. COLA's teaching signal δ_t = e_t/(1 − μφ′_t) is a per-unit rescaling of the current error, so it cannot carry later errors back to earlier steps.
* What is withdrawn is the report's mechanistic sentence. α ≈ 0 and the inactive closure were partly numerical, not purely a property of the task statistics.
