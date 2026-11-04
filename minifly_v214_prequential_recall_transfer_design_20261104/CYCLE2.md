# Cycle 2 · Metrics and checkpoint forks

Refine. Put loss accounting in a driver that calls prediction with no target argument. Use a small byte-count test double to exercise actual predict-then-observe, serialization and branch isolation. Keep common-byte efficiency separate from the learner-coded history ratio.

Audit. Compare predictions under a changed unseen suffix, verify checkpoint resume against an uninterrupted replay, and check the six-word reference against its analytic code length. Exercise positive recall, tied sham, positive marginal with a falling ratio, and negative late marginal.

Result. All checks passed with the test double. The uniform six-word reference cost log2(6) bits per completed span. The reducer distinguished an increasing gain with a falling ratio from a late decline. It rejected a tied sham, an unmatched target, and eligibility for Stage B when the prose guard failed. These fixtures validate scoring and control flow.
