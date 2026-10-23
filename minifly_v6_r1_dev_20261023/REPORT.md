# V6-R1 three-cycle development result

**Verdict: NO_DEMONSTRATED_UNSEEN_ADDITION_GAIN.** This is an engineering comparison, not formal confirmation or adoption. The goal and native-only answer route remain unchanged.

Completed 8 arm histories in two prespecified worlds: **34,816 external record applications and 393,472 observed native byte/EOS events**. Four full synthetic qualification records (20 observed events) and four partial native fixture events are separate from addition science. Three frozen random tapes and both orientations are correlated readouts, not independent lives.

## Own-output addition results

| World | Arm | Final OLD exact | Final NEW exact | Final held exact | Held own EOS | Held truncated |
|---|---|---:|---:|---:|---:|---:|
| 202610236101 | FROZEN | 0.000% | 0.000% | 0.000% | 0.000% | 100.000% |
| 202610236101 | LEARN | 0.000% | 0.000% | 0.000% | 0.000% | 100.000% |
| 202610236101 | NO_OLD | 0.000% | 0.000% | 0.000% | 0.000% | 100.000% |
| 202610236101 | SH_OLD | 0.000% | 0.000% | 0.000% | 0.000% | 100.000% |
| 202610236102 | FROZEN | 0.000% | 0.000% | 0.000% | 0.000% | 100.000% |
| 202610236102 | LEARN | 0.000% | 0.000% | 0.000% | 0.000% | 100.000% |
| 202610236102 | NO_OLD | 0.000% | 0.000% | 0.000% | 0.000% | 100.000% |
| 202610236102 | SH_OLD | 0.000% | 0.000% | 0.000% | 0.000% | 100.000% |

Exact means every own-generated payload byte plus own EOS, with no supplied answer prefix or length. Native margins for every saved output decision were independently decoded again. `results/METRICS.json` includes payload exactness, own-prefix byte accuracy, empty answers and all checkpoints.

## Fixed gate and exposure audit

- World 202610236101: NEW_acquisition_guard=False, OLD_end_causal=False, OLD_retention_guard=False, final_causal=False, held_beats_all_arms=False, held_beats_all_countermodels=False, pass=False.
- World 202610236102: NEW_acquisition_guard=False, OLD_end_causal=False, OLD_retention_guard=False, final_causal=False, held_beats_all_arms=False, held_beats_all_countermodels=False, pass=False.

The 64 held unordered orbits per world, in both orientations, never receive feedback. OLD has 128 taught relations repeated 32 times; NEW has 32 disjoint relations repeated eight times. All prompt bytes and expected reply/EOS feedback count. No separate answer-bearing pretraining is free. SH_OLD preserves answer/length multisets; one answer is coincidentally unchanged in each world, with no singleton buckets. NO_OLD disables both native and recurrent/baseline plasticity during OLD, while its warmup clock advances. FROZEN retains its initial recurrent weights.

| World | Best final held exposure countermodel | Raw/canonical lookup |
|---|---:|---:|
| 202610236101 | 0.000% | 0.000% / 0.000% |
| 202610236102 | 0.000% | 0.000% / 0.000% |

Countermodels are independently refitted only from each arm’s permitted observed prefix: raw and canonical lookup, frequency, prompt-length frequency and suffixes 0–4. These evaluator stores never enter the fly. The analytic adder is a label/scorer check, not a learner. Teacher-forced loss reduction and taught recall cannot substitute for a passed held-out causal gate. No E3 addition improvement is claimed when the gate fails.

## Learning and address diagnostics

| World | Arm | Mean per-record observed loss | Gradient norm max | Clip count / updates | Final weight displacement | Initial→final KC support Jaccard | CPU per history |
|---|---|---:|---:|---:|---:|---:|---:|
| 202610236101 | FROZEN | 0.563779 | 4.553 | 0/0 | 0.000000 | 1.00000 | 386.12s |
| 202610236101 | LEARN | 0.563384 | 4.315 | 1042/4352 | 0.040821 | 0.94980 | 383.50s |
| 202610236101 | NO_OLD | 0.689529 | 4.896 | 256/256 | 0.015933 | 0.97457 | 373.90s |
| 202610236101 | SH_OLD | 0.563433 | 4.315 | 1049/4352 | 0.041165 | 0.94852 | 397.45s |
| 202610236102 | FROZEN | 0.564448 | 4.598 | 0/0 | 0.000000 | 1.00000 | 388.76s |
| 202610236102 | LEARN | 0.564306 | 4.567 | 1047/4352 | 0.035794 | 0.94976 | 385.13s |
| 202610236102 | NO_OLD | 0.689504 | 4.807 | 256/256 | 0.015675 | 0.96943 | 383.47s |
| 202610236102 | SH_OLD | 0.564328 | 4.568 | 1045/4352 | 0.035854 | 0.95101 | 369.80s |

Address Jaccard uses fixed tapes and eight OLD prompts only, before versus after controller learning. It measures learned address drift, not held-out performance or trial-to-trial noise. The local score identities and an enumerable delayed-credit circuit pass numerical checks; that verifies the estimator implementation in those fixtures, not its usefulness in this changing native learner.

## Cost and limitations

The main runner recorded 2,914,345 extra private native events for online and scored generation, in addition to observed training events. Scored query count is 48,000. All tapes, clones, projection, sorting, dense state scans, checkpoints and reductions are charged by runtime. See the cumulative `results/RESOURCE_LEDGER.json` and first-history `PROFILE_ADMISSION.json`. Caffeinate followed the active process.

The native mutable state uses 5,684,346 bytes; recurrent weights/traces/gradients use another 1,469,952 bytes. There are 27,572 PN–KC edges. The original dense scans and global competition remain; there is no sparse-compute or measured bandwidth claim. Actual hardware bytes moved were not instrumented. Stage-1 import startup CPU was not timestamped before operations initialization; later process CPU includes imports. Main curves store cumulative process CPU. Baseline-subtracted per-arm curves exclude baseline/birth startup; full-history CPU includes those costs. No cost-efficiency gain is inferred from a failed behavioral gate.

The independent artifact reducer was implemented while the pilot ran, after baseline queries, before probe scores were inspected. Main scoring, manifests, gates and learning code were frozen before feedback. This timing deviation from the strongest “all reduction code before probes” contract is disclosed in `AUDIT_SOURCE_LOCK.json`; no scores or endpoints were changed. The supplemental reducer does not train, rerun science or revalidate trajectories.

Artifact audit: 279 source/assets hashes, 48,000 native decision rows and 200 metric cells checked. Source mismatches and held-feedback overlap are zero. Immutable full histories have matching observed counts. Readonly and exact restore were tested in qualification; every scored probe also checks the persistent-state digest.

## Three cycles and revision

1. Implemented the declared generic recurrence and native route. Audit corrected tape count and redundant copying. Technical test failed on a bootstrap-induced evaluator import collision after two synthetic records; failure and source snapshot preserved.
2. Revised only module isolation/operations, re-tested causal score credit, generic IO, readonly/restore, no-write controls, manifests and native self influence. Passed; four technical full records total across both attempts.
3. Ran the unchanged bounded addition roster, independently reduced saved outputs, and revised the next-step document from actual evidence. No trajectory was rerun, gate relaxed, arithmetic rule added or new core adopted.

Read `REVISION.md` for the evidence-based next design boundary. The historical arithmetic execution pointer and completed memory pointer remain unchanged. GitHub publication was not requested in this turn.

## Additional review findings

See `REVISION.md` for the measured sensory/tape contrast and saved-output/bit-prior diagnostics. These are posthoc localization, not new primary endpoints. The original query API missed the input cap; a separate zero-event boundary revision passes, while the frozen arithmetic path remains unchanged. The source-lock location omission, unused FROZEN baseline difference and technical private-query count bound are explicitly retained there. The bounded results remain valid within the manifest; they do not amount to full protocol conformance for an unrestricted API.
