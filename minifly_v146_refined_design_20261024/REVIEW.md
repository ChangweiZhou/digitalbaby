# V146 review — retain the hypothesis, separate the causes

24 October 2026. This is a design review, not a new experiment. Claude's original is preserved in [CLAUDE_V146_ORIGINAL.md](CLAUDE_V146_ORIGINAL.md); the refinement is [DESIGN.md](DESIGN.md). Source: `c243b5ae8b0e3c0d944e014cf76a44d4e32c477b`. Published V145 evidence: `5f9eaf159292e4e0b3f6dc013f5ae891d90fce7e`.

**Judgment:** preserving byte identity while learning an additional generic representation is worth a bounded test. V145 did not isolate low-dimensional substitution as the sole bottleneck, however, and V146's “number concept + downscaling solves grokking” account is ahead of the evidence. Keep the goal and test the two interventions separately.

## Material changes

| Issue in Claude's memo | Why it matters | Refinement |
| --- | --- | --- |
| “Below the critical data size” inferred from p=10/23/47 | One seed, different task sizes, class counts and total exposures do not isolate a critical size; modular addition is not ordinary ASCII addition. | Treat the result as background motivation, not a theorem or reason to change the target. |
| Ordered-pair holdouts in background grokking analysis | Many held pairs have their reversed counterpart taught. This is a changed-presentation case for those relations, not V146's globally withheld orbit test. | Preserve orbit-wide exclusion; report the background analysis under its actual scope. |
| V145 held/taught ratio treated as evidence of a cheap rule | Taught and held items differ, and oracle/learned-code trials have different doses. A high ratio with poor acquisition does not isolate why it failed. | Keep absolute acquisition and held metrics; run actual-interface readiness before map learning. |
| A single permutation of 88+M channels | This would mix the identity and map blocks across time, changing the supposedly unchanged identity path. | Independent blocks; identity uses the exact parent event API. No joint normalization silently rescales it. |
| Unit-mass bumps with birth-fixed gain | Narrower bumps can have much larger L2 amplitude than broad bumps; gain may masquerade as geometry. | Normalize the receptor contribution to unit L2 and fix gain from all-byte identity birth statistics. |
| “The map orders run length” | FE-PERM changes direction with each repetition and normalization can suppress magnitude. SOM topology alone does not guarantee semantic order. | Define pre-byte context updates and measure learned organization; do not install a counter or treat the desired geometry as a fact. |
| Unspecified SOM/recall/combination/naming equations | The proposal is not yet an executable local learner or a source-pinned native route. | Write explicit updates, initialization, boundary timing and warmup. Label delta naming as a surrogate and V143 fixed-feedback teaching as an adaptation. |
| MAP-PERM pairs d with π(d) stars but also counts in π order | These operations need not produce a consistently relabeled world. | Apply one byte permutation to every digit in both Q and C, leaving run lengths intact. |
| “Equal frequency” suggests all input statistics are controlled | Equal digit counts coexist with 2,880 marker bytes in the proposed bounded dose. Every-event SOM updates may be dominated by the marker. | State exact per-phase byte/EOS counts; keep all-event learning and measure Q-to-C interference. |
| Shrinkage interpreted as a simple-solution mechanism | ReLU plus binary top-k has scale invariances; uniform shrink can preserve gates/KC membership and all immediate answer signs. | Specify which arrays shrink; test the invariant and measure later residual/learning effects. Lower norms alone do not demonstrate rule extraction. |
| FULL compared only with SYMBOL under shrink | A delayed gain might also occur without shrink or with identity-only continued training. | Add SYMBOL-NO-SHRINK and use the complete map × shrink factorial; report its interaction. |
| “FULL passes without signature ⇒ shrink not needed” | Missing the chosen signature is not evidence of no shrink benefit. The original primary gate did not require a FULL vs NO-DOWNSCALE improvement. | Separate map readiness from a causal shrinkage claim; use direct ON/OFF contrasts at fixed checkpoints. |
| “Resume replays one completed unit” | Replaying a committed training record changes exposure and state. | Resume from the next uncommitted unit. Replay is allowed only in a paid disposable equality fixture. |
| ORACLE-MAP called structure-free ceiling | Digit positions in that fixture are supplied arithmetic structure. | Quarantine it as an explicitly arithmetic-prior diagnostic, never a compliant candidate or guaranteed upper bound. |
| Nine arms counted as complete causality/efficiency roster | An OFF identity control and OFF budget comparator were missing. | Ten main arms / forty histories. Original 36-history count is superseded prospectively; six-CPU-hour cap does not expand. |

The V146 memo also groups some earlier findings as exact-gradient results. V143's native learned-bank improvement used fixed-feedback local feature updates, and V145 includes both BP and DFA reference trials. These routes must not be collapsed into one learning-rule lineage.

## Background analysis arithmetic, without rerunning training

Using the saved script's seed-0 ordered-pair split:

| p | Training fraction | Held pairs with reverse taught / held pairs | Presentations at 20,000 full-batch steps |
| --- | ---: | ---: | ---: |
| 10 | 0.50 | 24 / 50 | 1,000,000 |
| 10 | 0.75 | 16 / 25 | 1,500,000 |
| 23 | 0.50 | 114 / 265 | 5,280,000 |
| 23 | 0.75 | 92 / 133 | 7,920,000 |
| 47 | 0.50 | 539 / 1,105 | 22,080,000 |
| 47 | 0.75 | 377 / 553 | 33,120,000 |

This does not negate the saved high-p results. It means they are not a matched-exposure orbit-held experiment, and cannot establish why the present MiniFly must fail or succeed. The review recomputed only split membership and exposure arithmetic, not any model trajectory.

## What remains a hypothesis

- Quantity/counting records are additional structured teaching, even though their information is in the data rather than the architecture. The learner receives no k value, numeric feature or arithmetic loss.
- A generic one-dimensional cortical-map prior may help, hurt or organize the wrong context. Human-neuroscience analogies motivate the hypothesis; they do not validate the executable model.
- Preserving identity can improve taught acquisition without improving abstraction.
- A local surrogate can succeed while native Full151 fails; matching a rate does not establish integration.
- More practice can improve behavior while worsening total sample efficiency. Compare against spending the same tokens/CPU on legal addition training.

## Remaining execution boundary

No V146 candidate, oracle trial, three-cycle run, native port or main result is delivered here. The prospective starting map, finite readiness menu, exact roster/cost manifest, combined checkpoint qualification and source/asset restoration still require implementation and testing. A shorter fixture must not be called the full ten-arm D4 rehearsal. Failing readiness stops map training without proclaiming the map hypothesis impossible.

The goal is unchanged: complete own-byte/EOS unseen addition from feedback with lower total sample and compute cost. Backpropagation is excluded in this V146 proposal; attention, Transformers, ELM and supplied arithmetic candidate structure remain excluded. No noise-optimization or new-task detour was introduced.
