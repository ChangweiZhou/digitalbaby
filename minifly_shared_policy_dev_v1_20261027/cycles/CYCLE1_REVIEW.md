# Cycle 1 — design audit and implementation amendments

The fixed conditioning protocol is retained. This is E1 reference fitting; no local candidate or E3 stage is launched.

1. CONST=s*ones has center(v)=0, so its recurrent r is identically zero. It removes content AND native-feedback signal. Compare with the structural R0 arm; do not call CONST alone a pure content lesion. Its independent positive-gate output box still supports every balanced codeword.
2. Fresh controller exposure clock starts at zero. Parent native state/clock is inherited and frozen. Adam warmup multiplies the actual step after preconditioning; original SGD uses its literal weighted-target gradient rule. Packages differ in realized step scale as declared.
3. Use unique identity SP-COND-V1, the exact name the user authorized. No global numeric version is guessed; V172 remains reserved to Claude. This resolves the design note requesting a number without creating a collision.
4. Reserve five opaque technical fitted records (two targets each), at most four technical own/TF pairs, within the same resource caps. They are additional charged qualification exposures, not science or hidden zero-cost references. No fitting world is consumed to repair a technical failure.
5. Save full event traces, including the end-of-prompt refresh, and recompute recurrent state after every update. Training cannot read old h. Own evaluation uses live frozen-native forks, including off-correct-prefix continuations.
6. The original source, design and historical negative results are not edited. This execution amendment precedes any model query or fitting.
