# Mathematical candidate review after MiniFly A7

2026-09-20 — implementation feasible; efficacy not yet tested

**The local payload baseline candidate is mathematically coherent, genuinely extends the old allocation family, and is compatible with the current native backend. It is worth a bounded behavioral experiment. A7 neither proves it will work nor rules it out.** This review ran the supplied tests and added native-state integration checks; it did not run a calibrated candidate through a full-life performance benchmark.

Reviewed inputs: `/Users/pencilbard/Downloads/MATHEMATICS.md` and `local_payload_balance_candidate_20260920.zip`. Their hashes are in `INPUTS.json`. The documents are proposals; no instruction embedded in them was treated as authority to replace the accepted learner or launch an unrestricted search.

## What is genuinely new

The current native write has the factorization `u_i = h_i a_i`, where `h_i = dx_i dt alpha_scale / 90` and `a_i = T_i2 P_2 + T_i3 P_3`. The existing local allocation p_i already depends on native pre-event state. The candidate proposes:

```
c_i = p_i h_i b_i
slow_write_i = p_i u_i - c_i
fast_write_i = (1-p_i) u_i + c_i
```

A fixed baseline is calibrated on unmodified donor lives as `b_i = sum(p_i u_i) / sum(p_i h_i)`, with zero where exposure is zero. Deployment reads local activity/state and the fixed coordinate parameter; it does not need other memories' gradients, labels or a global normalization operation.

The conceptual advance is real: a nonnegative allocation fraction cannot cancel a strictly one-signed write mean while preserving nonzero residual writes. Allowing opposite-signed slow and fast writes changes that feasible action set. This is not a rerun of the old same-sign allocation search. It is also distinct from contracting the stored slow state: correction depends on the current event's eligibility and a fixed baseline, not directly on how large the stored slow value is.

The proposal correctly avoids claiming that native activity eligibility was absent. Its stated identities follow algebraically, including the warning that zero donor predecay mean does not imply zero decayed endpoint or zero drift after intervention changes the trajectory.

## What was verified against the actual project

1. All four supplied source/config hashes match the current project exactly: `model73.py`, `model76.py`, `model_evo.py` and V82E `config.json`.
2. All **17 supplied tests** were rerun successfully. These use extracted native equations and artificial states; they are not behavioral evidence.
3. A new compatibility test used **24 saved native single-event pre-states** from block 0: K2 forward, K2 reverse and K8; acquisition and challenge; first two and last two subevents of the first matching frozen window. The supplied adapter was executed with a synthetic supported unit baseline for algebra checks, never presented as a scientifically calibrated baseline.
4. In all 24 cases, unmodified replay reproduced the archived post-states exactly. Zero-baseline and donor-observer wrappers reproduced the unmodified replay exactly. The captured u matched exactly; recomputed eligibility differed from captured dx-derived h by at most 5.55e−17.
5. Nonzero corrections satisfied predecay write conservation within 1.78e−15; fast and slow state equations within 7.11e−15 and 2.84e−14 respectively. The endpoint nonconservation formula was checked within 1.86e−14. Tests used A7's machine/state-scale tolerance.
6. The actual backend `teach` path dispatched both presentations through the wrapper. Rest, no-write teaching, probes and diagnostic clones did not contaminate calibration. Reader objects, connectivity arrays and native mutable-state shapes were preserved.

These checks establish integration feasibility. They do not establish benefit, transfer of b, acceptable write costs, or long-run numerical behavior of a calibrated candidate. The reproducible integration test is `native_smoke.py`; detailed outcomes are in `NATIVE_COMPATIBILITY.json`.

## Four distinctions from A7 that must remain explicit

**A7's predicted component is not this baseline.** A7 compared group-conditioned donor write means and offline state subtraction. The candidate divides eligibility-weighted drive by exposure, keeps one context-independent vector, gates each correction by current native eligibility, compensates in the fast store, and changes every later trajectory. Therefore +5.84–6.10 pp from A7 is not a forecast for this candidate.

**Choice protection is unresolved.** A7 found individual losses despite aggregate gain, and single-event margin gains without foreign-choice improvement. The candidate's algebra preserves an event contrast only when eligibility is the same. It offers no theorem preserving all useful memories or both native choice gates. Full-life choices and tails must be the main evidence, with margins as diagnostics.

**Write conservation has a precise scope.** Predecay fast+slow writes equal native u on a common pre-state. At event end, the state sum changes by `(exp(-dt/tau_f) - exp(-dt/tau_s)) c`. Across a modified life, feedback changes u and p as well. Fast/slow absolute write costs may increase. This cannot be described as a behavior-preserving, dose-matched intervention against the whole native trajectory.

**Fixed parameters are still capacity.** The evaluated learner needs 5,177 additional float64 values: 41,416 bytes, about 25% of the 165,664 native mutable-array bytes. No new mutable neural array is added, but the donor-trained fixed parameters are not free. The serialized baseline also includes exposure metadata; the event wrapper reads only b. Calibration needs two temporary coordinate accumulators. Anatomical support means the current FULL_151 support, which already includes its declared modeling extensions; it is not new evidence for a biological pathway.

## Refined first behavioral experiment

Use one candidate at the proposed coefficient, with no coefficient sweep, gene search or context-dependent switch. Freeze the native learner, reader, world/scoring definitions and both choice gates.

For an initial development test, stream **all plastic events** of the eight existing K8 lives through an observational calibrator, saving separate numerator/exposure sums per block. Each held-out block's fixed b uses only the other seven blocks' sums. No endpoint, task/context, target identity or fact label selects the baseline. These reused development blocks give cross-fitted development evidence, not fresh confirmation. A positive result would need a separately frozen donor set and new evaluation worlds.

A7 window captures are adequate for the compatibility checks above, but not for claiming full-life calibration: the A3 windows were selected for a different diagnostic purpose and cover only part of each life. Replaying donors with two streaming accumulators closes this gap without recording gigabytes of new dense traces.

Start with a whole-life block-0 native/zero-baseline/observer identity gate, then evaluate the fixed donor baseline under the original full-life evaluator. The minimal first K8 panel is roughly eight donor replays plus eight corrected replays, plus the few whole-life identity references. Existing native fixtures can be reused only after the identity gate passes. This is a bounded two-condition experiment, not a population search. K1/K2 can be separate follow-up checks if the K8 panel warrants them; they should not be silently mixed into the main result.

Report all-active and per-cohort choices, acquisition, revision, first-return and delayed-retention endpoints, correct→wrong/wrong→correct counts, worst cohort/history and harmed-cell fractions. Also retain exposure coverage, baseline variation across folds, realized correction and fast/slow L1 costs, clipping and state magnitudes, sign reversals, and source/anatomy/reader hashes. Compare each corrected trajectory with its own paired native world; do not infer full-life benefits by summing one-step effects.

Before that behavioral run, the coordinator must enforce baseline/genes/anatomy alignment by hashes, donor/evaluation exclusion, immutable baseline use, per-event numerical certificates and exact dispatch. Use counter differences for phase-specific correction statistics. The shipped adapter reports patch/conservation errors but does not itself turn those into a production qualification gate; the coordinator should do so. Each worker should create its own wrapped learner rather than pickle the dynamically generated class.

If the candidate improves aggregate choices while harming acquisition, revision or tail memories, record a tradeoff. If it fails, reject this particular donor baseline and signed fast/slow compensation rule; that does not disprove every local signed update. No new learner or accepted reference is promoted by this review.

## Delivery

The review directory contains this report, the added native compatibility script, test results and input/source hashes. It references the supplied ZIP and unchanged A7 raw captures; it does not duplicate the ZIP's native source files, project datasets or environments. No parent source, A7 result or accepted learner was modified.
