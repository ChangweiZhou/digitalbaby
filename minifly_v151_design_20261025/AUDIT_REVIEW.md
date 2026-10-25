# Review of the attached V150 mechanism audit

The recommendation is sensible: identify native teaching response before another large run. Local source/result checks support its central observations. This review is static; no native model was run or updated.

| Audit point | Assessment and V151 refinement |
|---|---|
| B/C had about half A's coefficient amplitude | Supported. B/A mean squared coefficient ratios are about 0.2591/0.2581. Equal numerical eta did not imply equal delivered strength. This does not prove the unique cause of failure. |
| Native plasticity is not direct gradient descent in v | Correct. Identify a coefficient-to-value response on matched cloned states, then test the full learning direction and actual loss. |
| The 64 sign checks were not 64 independent learned states | Correct. They covered two newborn input conditions, signs and compartments. V151 includes trained final states; independence is not inflated. |
| Code learning depends on u | Correct for the declared one-step outer-product update. Net drift is not accumulated learning and cannot identify cancellation or a unique cold-start cause. Use code-only/native-only/joint counterfactuals. |
| Span≤8r and the concentration bound | Correct per position when code rows have norm≤4. Do not use mean RMS/span as a bound on all positions or all-state capacity. |
| Existing newborn/intermediate/final assets can all be reused | Requires correction. Six final native checkpoints exist; no native intermediate or birth checkpoint exists. Reconstruct a canonical newborn without feedback. No intermediate replay is permitted. |
| Match native displacement before comparing A and B | Good diagnostic, with one bounded scalar from separate calibration contexts. Freeze it before scored outcomes; no per-item loss tuning, clipping, inverse Jacobian or implicit oracle controller. |
| Improvement after matching explains the V150 run | Too strong as a causal conclusion. A one-step intervention can support a limited coupling hypothesis, not determine the main cause of an entire training history. |
| Missing EOS proves a broken stop mechanism | The audit correctly rejects this inference. Greedy text continuation is descriptive here; V151's one-symbol EOS probe is not a full-sequence stopping test. |

Native timing is central. A query after `_event(y,y)` has already consumed y. V151 does not feed the target in the intervention. A matched zero-coefficient event is the principal counterfactual; an additional ordinary passive event checks that the two nominally nonplastic paths agree. The first-order loss gradient belongs at that post-control query state.

For C-changing branches, the fixed-code formula is incomplete: the first-order effect includes both g_v·Δv and the code-gradient inner product. Measured code increments, projection and interaction residuals must be reported. The response operator and gain calibration are evaluator tools, not architecture components supplied to MiniFly.

V150 also found BIAS slightly better than learned C on the conditional fixture. This remains evidence against claiming learned-code necessity. V151 freezes C in the primary test and makes no new capacity mechanism, language capability or addition-generalization claim.

The static site check also exposes an informative edge case: Buster's first record has one byte, so its first calibration prefix is empty. The unchanged frontend/encoder yields no active KC for an all-zero receptor trace. V151 retains this structural null, records its actual response, and does not reinterpret it as a sign failure or silently replace the site. This is not yet a quantified explanation of V150's aggregate loss.

The resulting design is V151: one bounded response assay, with the original gradual warmup, local rules, native lineage and own-byte/EOS output competition intact. Addition remains the goal. Implementation and testing require a subsequent user instruction.
