# Registered ERROR versus R_center comparison

This is a public scientific projection of the accepted fixed experiment. The underlying scientific registration was frozen before official execution. This public projection was prepared after launch; it is not a contemporaneously public preregistration and contains no official efficacy outcomes. This initial release contains source and qualification evidence, not an official efficacy result.

## Question and fixed mechanism

Does changing only the private teaching coefficient from R_center (0.25 minus the arriving one-hot outcome) to ERROR (the cached pre-outcome softmax of private scores divided by 1.3452365735750882, minus that outcome) repair taught-dependent held-out table completion while preserving old and new learning? ERROR softmax temperature is 1. The shared native update is unchanged. Both rules retain four shared and four private stores, native dynamics, encodings and clocks, and scales (1.4911274663291492, 1.3452365735750882).

The sole primary output is the argmax of the sum of the scaled shared and private four-channel scores; exact ties choose the lowest ASCII byte. Private/shared readouts are diagnostics. The arriving outcome byte supplies the teaching label only after prediction. An emitted choice is never a teaching target.

## Fixture, schedule and controls

The evaluator constructs relabeled order-four Latin tables from the unchanged fixture generator. Twelve old cue/outcome associations are each taught sixteen times (192 records); four balanced old associations are held out and never taught. Sixteen new associations are each taught twelve times (192 records). Each cue is twelve bytes. A record lasts 165 seconds, with byte spacing 30/14 seconds, a prediction then the actual outcome at slot twelve, followed by a newline. One 86,400-second delay separates old and new teaching and another precedes the final endpoint. Native final time is 236,160 seconds.

Each world contains six matched lives: R_center and ERROR, each under W, N_old_relation and N_new. W permits teaching. N_old_relation suppresses shared and private value writes during old teaching; N_new suppresses them during new teaching. Exposure and nonplastic native time continue under both clamps. Corresponding rule pairs must have exactly equal full operative shared state and shared clocks after every event and registered delay. Divergence is an implementation failure.

Old-end, new-end and final probes run on disposable clones without feedback or changes to the continuing state. Their evaluator-only targets are used only after prediction. The inference uses final-time combined accuracy: twelve old, sixteen new and four held-out cues per life. Earlier probe times are descriptive only.

## Fixed worlds and inference

The inferential unit is a world. The entire roster is 320001 through 320064 inclusive, in fixed missing-only order. Development worlds are 320000 and 320101; qualification uses 320200 and its one identical fresh-process replay. There is no world replacement, optional subset, adaptive sample extension or performance-selected retry.

All sixteen contrasts share two-sided Bonferroni-adjusted simultaneous 95% Student-t intervals, n=64 and df=63 (critical value 3.0734813460412793). For each rule the six contrasts are held-out W minus N_old_relation (E3); held-out W minus 0.25; old W minus N_old_relation (E1); new W minus N_new; old W minus 0.25; and new W minus 0.25. The four direct contrasts are ERROR minus R_center for E3, raw held-out W, old W and new W. The last two are the noninferiority endpoints.

Raw chance-adjusted accuracies have support [-0.25,0.75]; E3 difference-of-differences has support [-2,2]; all other contrasts have support [-1,1]. Nonconstant samples use the registered Student-t interval, clipped to support. Constant samples use the conservative Hoeffding radius: support width times sqrt(log(2*16/0.05)/(2*64)). Coverage of Student-t intervals is approximate. No cue-level pseudoreplication is used.

## Joint success and terminal labels

ERROR must have lower bounds above zero for E3, held-out chance-adjusted W, old E1, new causal gain, and both old/new chance-adjusted W accuracies. Its raw old and new W means must each be at least 90%. Both direct old/new W lower bounds must be strictly greater than -0.05. These are joint completion and preservation conditions. Direct held-out raw/causal superiority is reported separately and is not an additional success conjunct.

The exact order of classification in source/runtime/analyze.py is:

1. COMPLETION_JOINT_CRITERIA_MET if all joint conditions pass.
2. Otherwise REDUCTION_IN_HARM_WITHOUT_COMPLETION if direct raw held-out superiority passes but the ERROR held-out above-chance or E3 upper bound is at most zero.
3. Otherwise JOINT_REPAIR_EXCLUDED_BY_REGISTERED_BOUND if any required ERROR positive endpoint has upper bound at most zero, or either preservation upper bound is at most -0.05.
4. Otherwise INCONCLUSIVE_JOINT_REPAIR_NOT_ESTABLISHED.

All estimates, intervals and failed gates remain reported. Closing this finite experiment does not convert an inconclusive result into impossibility. No automatic temperature/alpha search, alternative mechanism, new bank, parser, label inversion or follow-on experiment is authorized by these results.

## Qualification and sensitivity

Three independent design/source review, execution and post-test acceptance cycles preceded launch. Cycle 1 used exactly twelve teaching calls and six disposable probes. Cycle 2 used twenty-four teaching calls and forty-eight probes. Cycle 3 used a full six-life world and one fresh-process replay: 4,608 teaching calls and 960 probes in total. Public numeric receipts and an audit summary accompany this protocol. Qualification outcomes do not estimate official efficacy.

Before new efficacy data, the fixed sixteen-contrast sensitivity calculation used world-level SDs 0.10, 0.16 and 0.25. The respective 80% marginal detectable positive effects were 4.917, 7.867 and 12.293 percentage points. Five-point noninferiority at true zero loss had approximate marginal powers 81.7%, 29.4% and 7.9%. Joint power is not promised. See the unchanged INDEPENDENT_POWER_SENSITIVITY.json for assumptions and additional effect sizes.

The registered execution bounds are one numerical worker, 768 MiB peak RSS, 900 seconds per paired world, 12 hours cumulative worker time and 18 hours active execution time. A genuine infrastructure failure preserves the same world, fixed scientific settings and conservative cost accounting; a failed bound cannot silently change the roster. Public files exclude deployment-specific execution state. The scientific endpoint remains the complete fixed roster and its one terminal analysis.
