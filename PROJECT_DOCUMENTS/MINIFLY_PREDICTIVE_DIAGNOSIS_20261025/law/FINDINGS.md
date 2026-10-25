# V147 one-update law: derivation and bounded verification

**Result: the FIXED one-update law and the LOCAL two-term decomposition match the actual V147 `Model.observe()` implementation.** Maximum absolute margin discrepancy across 12 disposable-copy fixtures and four taught query contexts each was **1.12 × 10⁻¹⁴**. This is an **E0 algebra and interface check**, not a new learning, retention or generalization result.

## Exact correspondence to the code

Let \(\phi(x)\) be the binary active KC vector divided by \(\sqrt{k_x}\). Thus \(\|\phi(x)\|=1\), and the nine output-bit margins are

\[
 m(x)=2\theta\phi(x),\qquad
 K(q,x)=\phi(q)^\top\phi(x).
\]

For an observed target symbol \(y\in\{0,\ldots,256\}\), let \(t_j(y)\in\{-1,+1\}\) be its nine signed bits. In V147 `Model.observe`, before any parameter change,

\[
 e_j(x)=\tfrac12[\tanh(m_j(x)/2)-t_j(y)],\quad
 s(n)=\tfrac1{16}+\tfrac{15}{16}\min(1,n/1024),\quad
 \theta'=\theta-\alpha s(n)e(x)\phi(x)^\top.
\]

Here `n` is `records`, which does not advance inside a single `observe` call. `teach()` advances it after a whole observed response and EOS; the present fixtures never call `teach()`.

For FIXED, \(W,b,V\) and hence \(\phi\) remain unchanged. The exact finite update is

\[
 \boxed{\Delta m_j(q)=-2\alpha s(n)e_j(x)K(q,x)
 =\alpha s(n)[t_j(y)-\tanh(m_j(x)/2)]K(q,x).}
\]

This is an exact one-step prediction from the pre-update state for a fixed query context. It is not a ridge fit, an asymptotic result, a derivative approximation or an added learning algorithm. On the same taught context, multiplying this margin change by its own signed target gives a nonnegative one-step improvement in every target bit. On another context with a different target, some signed bit changes can be negative. Unsigned overlap alone therefore cannot distinguish useful transfer from harmful interference.

LOCAL performs the same \(\theta\) update using old codes and then changes \(W,b\) through the inherited local feedback rule. With \(\phi_{old}\) and \(\phi_{new}\) denoting those codes, the exact finite decomposition is

\[
 \boxed{\Delta m_j(q)=-2\alpha s(n)e_j(x)K_{old}(q,x)
 +2\theta'_j[\phi_{new}(q)-\phi_{old}(q)].}
\]

The second term includes the interaction with the just-updated naming weights because it uses \(\theta'\), not \(\theta\). Top-K code changes are discrete; no smoothness assumption is needed. This decomposition is an identity after recomputing the changed code. **It is not an independently validated forecast of future representation drift:** using the observed post-update code to calculate the second term would reuse the outcome being explained.

## What was checked

The script reads the four saved FIXED calibration states (D1/D2 × α .003/.03) and the two higher-rate LOCAL states. Each fit supplies exactly two independent single-target fixtures: the first already cached taught prompt at its empty prefix and at its observed one-byte prefix. Every fixture starts from a fresh disposable copy of that same saved state. There is no accumulated teaching sequence or resume.

The query set is four already cached teacher contexts from the first two taught prompts. Their membership in saved TRAIN pairs is verified; no held prompt is constructed or scored. The target byte is recovered from the original cache insertion order, which preserves `teach()`'s empty-prefix/observed-prefix ordering. All six cache sizes are below the 4,096 eviction threshold, so that ordering has not been lost. No task/world generator is imported and no new world is generated.

| Saved fit | Two fixture margin-law errors, maximum | Changed query codes across its two fixtures | Largest moving-code term |
| --- | ---: | ---: | ---: |
| D1 FIXED .003 | 2.30e-15 | 0 / 0 | 0 |
| D1 FIXED .03 | 5.02e-15 | 0 / 0 | 0 |
| D2 FIXED .003 | 1.42e-15 | 0 / 0 | 0 |
| D2 FIXED .03 | 4.34e-15 | 0 / 0 | 0 |
| D1 LOCAL .03 | 4.02e-15 | 1 / 1 | 0.18165 |
| D2 LOCAL .03 | 1.12e-14 | 1 / 0 | 0.35152 |

The largest direct fixed-code terms in the D1/D2 LOCAL fixtures were respectively 0.02562 and 0.01916. The moving-code term is therefore large in some of these hand-limited examples. This is a descriptive illustration, **not** an estimate of its population frequency, sign, cumulative harm or causal share of V147's failed acquisition. One LOCAL fixture changed none of the four query codes. FIXED still failed its historical acquisition gate without any moving-code term.

All FIXED W/b arrays and codes stayed exactly invariant. The analytic \(\theta\) update matched within 3.47e-18. Probes preserved persistent state; the saved original models, trace caches, input PKLs and imported project source hashes were unchanged. Full per-query, per-bit terms and the input/source hashes are in [RESULTS.json](RESULTS.json); the exact event counts are also in [FIXTURE_LEDGER.json](FIXTURE_LEDGER.json).

## Exposure, compute and evidence boundary

- External record applications: **0**. Replayed trajectories/record applications: **0**. New worlds: **0**. Held queries: **0**.
- Disposable-copy synthetic target-update events: **12 attempted, 12 completed, 0 failed**. Each is one `observe()` invocation using a previously observed payload/EOS target; these are counted separately from external whole records and are not new scientific learning evidence.
- Script CPU including imports after the standard-library timer: **0.960504 seconds**; saved in RESULTS.json. A prior read-only cache inspection took 2.9 tool wall seconds; its CPU was not separately measured. Tool execution, authoring and report preparation are outside the measured CPU subtotal, so it is not an all-in project cost.
- All saved states are past warmup, with \(s=1\). The ramp formula is established by direct correspondence to the code; this fixture set does not sample its early numerical behavior.
- Full autonomous byte/EOS generation is not assessed here. The query prefix remains fixed, whereas a generated wrong byte changes future contexts.

Exact-key or canonicalized-key lookup remains a sufficient countermodel for the original taught associations; this E0 identity check supplies no counterevidence and no E3 claim. Native Full151 storage dynamics are not covered by this surrogate's equations. Multi-event sums are exact for FIXED only when the residual of **each evolving state** is used; summing birth-state effects cannot generally predict a trajectory because residuals change. LOCAL adds state-dependent, discrete address changes at each step.

## Consequence for the next analysis

Use signed changes to distinguish teaching benefit, held transfer and old-association damage. For FIXED, the pre-update residual and kernel exactly determine one update at a fixed context. To test a practical forecast of retention or generalization, freeze a predictor at an earlier checkpoint, compare it with current margin/accuracy plus later-write-count baselines, and evaluate later behavior without using future residuals or future codes as predictors. The present algebra supports that test design; it does not establish that the forecast will be useful or improve unseen addition.
