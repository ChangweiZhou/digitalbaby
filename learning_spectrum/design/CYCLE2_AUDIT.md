# Cycle 2 independent pre-test audit

Verdict: PASS for the source/history-only finite-horizon and geometry tests, with the precise implementation clarifications below. No cycle 2 outcomes were inspected. Cycle 1's failed necessary spectrum gate remains binding: this audit does not authorize a fresh/full run, irrespective of cycle 2 success.

## Accepted design

The 32 item-basis banks are a valid evaluator-side construction of the exact linear teacher-to-readout operator when features are independent of labels and clipping is absent. Evolving EMA features, normalized residual updates, elapsed-time decay and chronological interference are represented directly. Min-eigenvalue of the symmetric H-to-H compression is explicitly a uniform signed quadratic-gain condition, not an eigenvalue shortcut for task learning. Singular values, target-aligned responses and cross-task interference are jointly informative.

The exact-address exclusion has been repaired by removing the centered isotropic component, requiring nonzero residual, alignment, and a control-relative residual-strength floor. The fixed six-symbol directed panel and actual immutable FE0 sparse KC encoder are appropriate for the narrow geometry comparison. Distinct source/native versus candidate-history states are disclosed. Neither these operational criteria nor the name “relation panel” establish learned relational generalization.

## Required implementation clarifications

1. At each probe use the literal query read time and multiply by bank decay since the most recent teacher. Teacher-time decay alone omits the one-day rests at old_day/final and is not the specified finite-horizon operator. Include the cue-processing offset at old_end and new_end as well. The direct clipped bank check must implement the same read-time convention independently.
2. The basis bank injects e_item; map it to the actual four-channel +/-1 target table only at evaluation. Check the zero-state linear prediction against direct four-channel recurrence before and after both cohort boundaries and rests. Record maximum error at each checkpoint and a fixed numerical acceptance tolerance, not merely average agreement.
3. Restricting Q_int^T L_old Q_int measures H-to-H formation. Real target tables also contain lower-order components. Publish full-target signed gain plus B/F/G/H input-output block norms/projections so an H compression cannot hide low-order corruption or cancellation. Keep old-target contribution separate from new-target interference.
4. To attribute an old-learning effect, compare W and N_old or equivalently the old-teacher Jacobian block with all later updates still present. Do not equate the entire W bank with old learned content after new teaching. Decay, subsequent attenuation and new additive contribution have different meanings.
5. Kernel trace, residual norm and cosine denominators must be positive/defined at prespecified tolerances; undefined values fail gates. Fix symbol ordering and exclude self-pairs exactly. Native sensor initialization, cue bytes, DT and query read time must match the feature protocol. Report incidence rank and use projectors, not orientation-dependent basis vectors.
6. Verify the no-clipping certificate in the actual +/-1 four-channel trajectory, not in arbitrary basis trajectories alone. If actual clipping occurs, this linear forecast is invalid for that trajectory and may not be relabeled an exact prediction.
7. The accepted prediction is an added-bank conditional algebraic forecast. A comparison to the same recurrence mainly validates implementation. It does not satisfy the independent prospective total-native-score/margin forecast gate. Observed final native scores remain development diagnostics only.

## Final launch decision

The current design correctly retains NO QUANTITATIVE THEORY PREDICTION for future total accuracy/margins absent an independently derived native forecast and tolerance. Combined with cycle 1's failed necessary geometry gate, a full run is blocked. Remaining cycles may validate the fixed mechanism and diagnostics without searching for a rescue. Resource caps and all negative/undefined outcomes must be retained.
