# Read-only audit of delivered RAW-RULE V2 draft

13 October 2026. Audited arithmetic_feedback_rule_discovery_design_v2_20261013/EXPERIMENT.md. No design/source edits or training occurred.

## Verdict

The fixed goal, source lineage, eight-arm roster, reader distinction, feedback losses, evidence boundaries and execution counts are internally consistent. One statistical fallback should be repaired before delivery; it does not require new data or changing the task, budgets or behavioral endpoints.

## Actionable issue: exact sign fallback is not a paired-mean test

The draft describes paired Student-t comparisons of world means but assigns p=2^-n to a constant positive sample as an exact sign rule. An exact binomial sign test addresses the probability of a positive difference (or a median hypothesis), not the population mean tested by the other contrasts. Frequent tiny gains and rare large losses can yield mostly positive worlds but a nonpositive mean. Calling this fallback an exact test of mean superiority is therefore unsupported.

A prospective repair is to retain the existing nondegenerate paired t rule and use a conservative distribution-free bounded-mean fallback for constant positive differences. Every declared accuracy, J and normalized G difference lies in [-1,1]. For sample mean d>0, Hoeffding gives a one-sided upper-bound p=min(1,exp(-n*d*d/2)); zero/negative gives p=1. This belongs to the same Holm family and requires no new accuracy threshold. Alternatively mark a degenerate positive mean test unresolved. State the selected method before any V2 data. Do not relabel sign-test evidence as a mean confidence interval.

## Counts verified

Eight DEV worlds by eight arms =64 jobs, 135,168 records. Conditional 48 confirmation worlds by six arms =288 jobs, 608,256 records. Maximum352 jobs, 743,424 records, 1,056 root histories plus6,336 inherited forks =7,392 correlated trajectories, across56 independent world identities. The confirmation family's eight causal operation contrasts plus four shortcut-envelope contrasts plus four J/G reference contrasts equals16.

G has19,696 reader parameters plus16,705 head parameters =36,401. L has4,144 embedding,32,768 input matrix and64 bias parameters plus16,705 head parameters =53,681. Parameter differences are disclosed; the design does not claim a matched-parameter isolated topology effect.

## Implementation clarifications to freeze during preparation

- Declare whether the lag buffer is right-aligned with leading absent slots; chronological oldest-to-newest alone does not uniquely specify unfilled-slot placement. Both are lawful generic conventions, but they produce different models. Freeze one before qualification.
- The shortcut envelope deliberately receives only old arithmetic feedback and does not forget during the cohort. Explicitly identify it as a protected old-feedback comparator, rather than a matched full-history learner. This is conservative for retention and should remain fixed.
- The complete-record finite-difference test applies exactly to the short assay. For long G records, the independent reference must detach the historical prefix and exclude evicted loss cotangents while retaining the declared denominator; it cannot compare against untruncated full-record derivatives.
- Frozen CE readers freeze E as well as reader weights; all optimizer and gradient-array costs should reflect actual trainable parameters. Padding is not NUL and is never supplied as feedback.

## Goal and attribution review

The saved-state diagnosis supports testing early-context access and categorical content discrimination. It does not justify an EOS fix, arithmetic decoder, frequency-only claim or more retention machinery as the default next repair. The new lag reader preserves chronological access without supplying arithmetic roles. Teacher-forced scores remain auxiliary; exact own-output retained held answers remain primary.

The sixteen-comparison family prospectively closes a known shortcut-attribution gap. It does not prove absence of all imaginable shortcuts or unrestricted arithmetic-law discovery; the draft appropriately retains bounded E3 and a separate fully correct grid claim. G/L remain different complete reader recipes, SQ/CE have different gradient scales, and backprop remains an engineered operation with historical locality restrictions preserved.

No further arm expansion, training-budget increase, new technical run or upload is necessary for this document audit.
