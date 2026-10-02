# Cycle 3 independent pre-test audit

Verdict: CONDITIONAL PASS for the historical calibration and fail-closed validation test. Resolve the two specification clarifications below before the test; neither changes candidate A or rescues any launch gate. No cycle 3 outcomes were read for this audit.

## Clarifications required before testing

1. The first-16 anchor must itself be causal. Filtering the weighted entries to t_i<t does not prevent leakage if b0 was precomputed from 16 entries including future observations. Either define b0(t) as the average of the first min(16,n_past) eligible entries with a fixed empty-history convention, or declare the predictor unavailable until all first 16 entries are strictly past and reject earlier queries. Fix one convention before running. Test future-suffix mutation at an early prefix as well as after entry 16. Final-panel development evaluation occurs later and therefore does not depend on which legitimate early-history convention is chosen.
2. Predeclare baseline MSE in decision-relevant channel-centered space: compare Bhat−mean_channels(Bhat) and the stored H baseline minus its channel mean against the same raw W evaluator panel-mean vector minus its channel mean. Report uncentered MSE as a secondary diagnostic if desired. A shared scalar shift changes MSE but no argmax or pairwise margin, so an unqualified uncentered MSE advancement criterion could select irrelevant bias removal. If preserving the existing uncentered criterion instead, explicitly label its limited meaning and require the centered comparison additionally; do not select between definitions after results.

## Accepted interpretation and safeguards

The new B surrogate is frozen before its test, explicitly approximate, separate from A, and built only from pre-teacher outputs, cue embeddings and clocks. Its one-day extrapolation is not a mechanistic claim about native FE0 dynamics. The mixed-history first-16 anchor is not an untrained-state estimate. Development outcomes from the completed response histories may evaluate it but cannot retroactively make its assumptions theoretical predictions.

The kernel uses a uniform byte-average embedding and may collapse reversed pairs; that is a declared limited distribution-weighting heuristic, not task-directed feature extraction. Record denominator/fallback counts and the effective similarity distribution so improvements cannot be attributed to an unmeasured query-distribution mechanism.

Compute repairs and breaks against identical raw predictions on identical old queries, and check corrected_count = raw_count + repairs − breaks. Report ties and keep the historical argmax tie rule unchanged. Record per-item signed margins to every competing channel and the best-other margin independently; subtracting Bhat changes which rival is best, so mean best-other shifts cannot be inferred by holding the original rival fixed.

The paired-world confidence interval is a descriptive development screen, not independent confirmation after prior exposure to these histories and selection of this diagnostic. Preserve all per-world outcomes including harmful worlds. Compare H, raw, oracle and B on the same cohort, checkpoints and state convention. The oracle remains evaluator-only.

## Launch enforcement

A remains rejected because its frozen static and ordered finite-horizon requirements failed. B cannot substitute for that mechanism or start fresh science without a separate prospective forecast contract. Missing total-native-score forecasts are a second independent blocker. The future falsification sketch is not a currently satisfied numerical prediction contract and must be labeled as such.

The fail-closed manifest tests should mutate each gate independently, remove required fields, alter source hashes, alter numerical thresholds, and remove the quantitative forecast artifact. A boolean pass flag alone must not override underlying failed values. If no launch-capable executable exists, report a launch eligibility validator rather than implying that a scheduler was exercised. No Package A outcome needs to be read for any validation, and source hashes must show no upstream changes.

Maintain the stated caps, publish negative diagnostics and all three pre-test audits, and issue a clear blocked-full-run decision. This is a completed preflight rejection, not an unfinished scientific run and not prospective falsification by fresh-world data.
