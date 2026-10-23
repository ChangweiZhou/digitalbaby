# V6-R1 revision after three actual development cycles

The exercise is complete. The bounded addition verdict is **NO_DEMONSTRATED_UNSEEN_ADDITION_GAIN**. No confirmation, adoption or rescue training is launched. The arithmetic goal, gradual schedules, native-only payload/EOS path and architecture exclusions are preserved.

## What was learned from the run

All full histories and native outputs are preserved in the package. E0 interface/causal-write checks passed within the tested bounded domain. Exact taught association and the strict held-out causal screen are distinct; lower next-byte loss does not satisfy either by itself. The favorable development gate did not pass when whole-answer exactness tied the controls.

The recurring output pattern is localized using saved outputs and observed bit priors, not another decoder fit:

| World | Arm | Mode predicted from observed bit priors | Final outputs identical to repeated mode + truncation |
|---|---|---|---:|
| 202610236101 | LEARN | 0x31 | 99.926% |
| 202610236101 | FROZEN | 0x31 | 99.926% |
| 202610236101 | NO_OLD | 0x31 | 99.107% |
| 202610236101 | SH_OLD | 0x31 | 99.777% |
| 202610236102 | LEARN | 0x31 | 98.586% |
| 202610236102 | FROZEN | 0x31 | 98.810% |
| 202610236102 | NO_OLD | 0x31 | 99.033% |
| 202610236102 | SH_OLD | 0x31 | 98.958% |

This supports a marginal-byte explanation for the apparent loss improvement. The countermodel ignores order and native physiology, so agreement is evidence about failure behavior, not a proof of the internal algorithm. It is posthoc and never replaces the locked endpoint.

Even the generous, offline target-length prefix diagnostic is separate from own-output success:

| World | Arm | OLD payload-prefix upper bound | NEW payload-prefix upper bound | Held payload-prefix upper bound |
|---|---|---:|---:|---:|
| 202610236101 | FROZEN | 0.000% | 0.000% | 3.125% |
| 202610236101 | LEARN | 0.000% | 0.000% | 3.125% |
| 202610236101 | NO_OLD | 0.000% | 0.000% | 3.125% |
| 202610236101 | SH_OLD | 0.000% | 0.000% | 3.125% |
| 202610236102 | FROZEN | 0.000% | 3.125% | 3.125% |
| 202610236102 | LEARN | 0.000% | 3.125% | 3.125% |
| 202610236102 | NO_OLD | 0.000% | 3.125% | 3.125% |
| 202610236102 | SH_OLD | 0.000% | 3.125% | 3.125% |

True length is used only in this offline upper bound. Margins after a wrong generated prefix cannot establish teacher-forced EOS recognition. Do not call the observed failure an EOS-only defect.

## Sensory signal versus random state

The controller/address diagnostic uses eight OLD one-digit-pair prompts per world, equal-length inputs, three fixed tapes, no labels, no held prompts, no native events and no learning. Its scope is E0 localization:

| World | State | Same prompt / different tapes, h distance | Different prompts / same tape, h distance | Ratio | Same-prompt different-tape KC Jaccard | Different-prompt same-tape KC Jaccard |
|---|---|---:|---:|---:|---:|---:|
| 202610236101 | BIRTH | 0.976728 | 0.013697 | 0.014023 | 0.01367 | 0.97819 |
| 202610236101 | FINAL | 0.976846 | 0.013812 | 0.014140 | 0.01706 | 0.98162 |
| 202610236102 | BIRTH | 0.954791 | 0.010623 | 0.011126 | 0.01363 | 0.98960 |
| 202610236102 | FINAL | 0.958305 | 0.011040 | 0.011521 | 0.01137 | 0.99153 |

The birth formula supplies only one active symbol column with SD0.05/sqrt348≈0.00268, while each open content proposal has noise SD0.1. The measured tape-versus-prompt contrast should therefore be examined before claiming learned abstraction or changing native capacity. A code can be mathematically order-sensitive yet vary much more with random innovations than with byte identity.

Working hypothesis, not a causal proof: weak cue signal produces unstable conditional addresses; native writes learn pooled bit frequencies; those near-unconditional values give little useful reward contrast to the representation learner. Correct local score derivatives do not by themselves break this joint-learning bottleneck. This explains why stable old addresses could support recitation while a new stochastic front end can fail even taught exact answers, without concluding that native learning or recurrence is impossible. Native capacity/interference, credit variance and coupled readout conditioning remain alternative contributors.

## Concrete revision

1. Keep the executed main kernel immutable. A separate `core_boundary_revision.py` enforces input/generation caps and rejects invalid payloads before mutation. Its zero-native-event tests pass; it changes no valid bounded arithmetic computation and does not repair the negative performance.
2. Preserve R1 as the comparator. `NEXT_DESIGN.md` proposes one generic active-fan-in initialization correction, with native values, stochastic rule, sigma, warmup, observed data and endpoint retained. All byte columns stay exchangeable and learned. It does not supply arithmetic roles, rules or a separate answer network.
3. Compare that revision against original R1, fixed-controller, no-prior-teaching and shuffled-label controls on fresh locked engineering worlds. Keep acquisition and held-out retention/sample/compute curves together. Do not launch confirmation of R1, increase n after failure, add a decoder, replay or population expansion to rescue this run.

Block rescaling also changes the native operating distribution and gate statistics; a favorable result would establish an integrated initialization intervention, not a pure mechanistic proof that noise alone caused the original failure. Further attribution would need a prospective control.

## Audit limitations retained, not erased

- Cycle 1 failed on an evaluator import-name collision after two synthetic records. Source and error are preserved; cycle 2 fixed module isolation and passed the bounded suite. No main trajectory was retried.
- The input-overflow check was missing from the frozen query API, although every actual addition prompt was within its cap. The separate wrapper is the tested fix.
- The original source lock covered identical vendor asset copies rather than some actual upstream birth-asset locations. Their hashes match; seven dormant configuration differences are document-path metadata. `ASSET_LOCATION_AUDIT.json` records the omission and equivalence, and each actual birth validates canonical anatomy.
- The independent artifact reducer was written during execution after baseline probes. Main scoring and decision rules were frozen beforehand. State, saved-output and frequency localization are supplemental; their posthoc timing is explicit.
- FROZEN also leaves its unused reward baseline fixed; because its weights never update, that baseline cannot affect actions. This is an operational deviation in ignored diagnostic state, not a causal treatment difference.
- Technical private-generation calls were timed but not individually counted: two runs have a conservative bound of 300 private native event applications (2×[2×(5+17)+2×3×17+1+3]). There are exactly20 observed full-record events and four separate partial fixture events. Even charging the full bound gives at most324 technical native event applications, below512. Main observed and private generation counts are exact.
- Hardware memory traffic and native clipping frequency were not instrumented. Dense scans, actual CPU, arrays, projection edges, output margins and sampled RSS are reported. Early first-suite import CPU was not timestamped; an explicit overhead reserve is charged in the final package.

The full bounded result is a failed addition-learning candidate under these conditions, not an arithmetic impossibility result and not an exclusion of every mechanism in the 254-entry register. No new task is substituted for addition.
