# MiniFly casebook: why the causal experiment loop matters

**Context:** A distilled workflow example grounded in the public repository `ChangweiZhou/digitalbaby`, consolidated branch `codex/minifly-consolidated-20261023`, checked through commit `55ad552930e8` (2026-11-09 EST). This file is **historical evidence**, not a claim about the current head at execution time. Refresh the repo before invoking the skill on a live project.

## Case A — 220+ versions versus representation diagnosis

Earlier variants repeatedly changed downstream learning rules while the fixed sparse input encoding destroyed much of the positional/similarity structure needed for arithmetic. The November 6 review highlights 4.58 -> 3.20 held bits from a sensor time-decay change in an earlier task, compared with individual plasticity tweaks usually <0.05 bits. **Procedure lesson:** check sensor coding and reference expressivity before searching updates. **Caveat:** these are different earlier harnesses; do not extrapolate absolute numbers to E16.

Source: https://github.com/ChangweiZhou/digitalbaby/blob/55ad552930e8/MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md

## Case B — raw observed replay replaces a task-aware parser

E16-R3 and E16-F3 show that replaying the observed raw prompt and answer bytes from a single label-free reservoir can meet the preregistered family-mean gate without task parsing or re-computed answers. E16-F3 fresh seeds: held composition 81.8/79.5/82.5%; the 5-point parsed-vs-raw band missed one seed **in the favorable direction** (+5.5). A claim about lower CPU cost on a new machine did not reproduce. **Lesson:** remove scaffolds one factor at a time and retain literal misses.

Sources: 
- https://github.com/ChangweiZhou/digitalbaby/blob/55ad552930e8/MOD97_LOCAL_RULE_20261106/E16F3_REPORT.md
- https://github.com/ChangweiZhou/digitalbaby/blob/55ad552930e8/MOD97_LOCAL_RULE_20261106/E45_REPORT.md

## Case C — E16-M2 separates learning rule from data

Two failing seeds scored ~64.7-64.9% on held compositions with the sign rule. BP on the learner's **identical** raw reservoir scored 88.7-91.3%, or 0.95-0.97 of a richer twin. Sign rule with an oracle/richer replay stream recovered to 78.8/84.5%, still below BP. **Lesson:** matched reference on identical observations is the causal lever; a BP joint-stream twin is a useful ceiling but not a same-stream comparator. Diagnosis: rule-by-sampling interaction plus rule weakness; do not say either cause alone was proven sufficient.

Source: https://github.com/ChangweiZhou/digitalbaby/blob/55ad552930e8/MOD97_LOCAL_RULE_20261106/E16M2_REPORT.md

## Case D — Fable 5.2 catches the gradient-unreachable / weight-decay defect

E17 Phase 1-1c local learners seemed to silence two banks under training. Weight inspection found that many prompt-character embedding rows got zero gradient in lifetime local updates and were nevertheless decayed to 9.94e-5 of their birth values. These phases' **causal interpretations were withdrawn**, experiments halted, and a no-decay policy plus per-token gradient audit introduced. **Lesson:** check learning-signal reachability, optimizer groups and real weight deltas before attributing a failure to a biological mechanism. Preventing decay is not the same as learning new symbol embeddings.

Sources:
- https://github.com/ChangweiZhou/digitalbaby/blob/55ad552930e8/PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/MEMO_TO_FABLE_WORKER_20261109.md
- https://github.com/ChangweiZhou/digitalbaby/blob/55ad552930e8/MOD97_LOCAL_RULE_20261106/E45_REPORT.md

## Case E — family averages conceal hard subgroup failures

Of 21 raw-replay lives, 18 passed the *pre-registered family-mean* test but only 4 passed the `+-`/`-+` hard-pair-only criterion. Some max/min-related compositions permitted a ~69% shortcut when ignoring a step; intermediate components partially overlapped taught histories. The evaluator initially had a held-panel key collision (later fixed in the E16 harness). **Lesson:** preserve the original gate and add explicit hard-subset, shortcut and history-level contamination reporting; do not rewrite the primary gate after viewing scores.

Source: https://github.com/ChangweiZhou/digitalbaby/blob/55ad552930e8/PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/MEMO_TO_FABLE_WORKER_20261109.md

## Case F — E17 Phase 2 isolates but does not fully solve a mechanism

With embedding decay disabled, fully unfrozen sign-feedback still yielded 0.7-1.2% F1 accuracy after six blocks; a frozen first bank with sign feedback yielded 59.4-62.4%; frozen first bank plus exact feedback *and* upstream extended trace yielded 87.5-91.9%, near the frozen BP twin's 90.0-91.5%. All arms, including BP, remained poor on hard two-step composition. **Lesson:** a multi-change rescue establishes a feasible subcase, not which modification caused it. Preregister a dismantling factorial: exact feedback alone; trace alone; freeze only embedding. No assertion that exact feedback meets the user's lifetime-locality constraints.

Sources:
- https://github.com/ChangweiZhou/digitalbaby/blob/55ad552930e8/MOD97_LOCAL_RULE_20261106/E45_REPORT.md
- https://github.com/ChangweiZhou/digitalbaby/blob/55ad552930e8/MOD97_LOCAL_RULE_20261106/E45_PREDICTIONS.md

## Case G — computational repetition is not independent statistical evidence

The E16-F6 job was run in two separate cloud containers and matched bit-for-bit in the resulting receipts and birth artifacts. This improved provenance/reproducibility confidence but **did not add six independent random seeds**. The early-signature prediction failed (Spearman ~0.22 vs pre-specified >=0.5) and the failure was retained. **Lesson:** distinguish independent seeds from identical replay and do not retrofit a post-hoc signature into a pre-registered one.

Source: https://github.com/ChangweiZhou/digitalbaby/blob/55ad552930e8/MOD97_LOCAL_RULE_20261106/E16F6_REPORT.md

## What to borrow (not what to copy)

- Borrow: fixed scientific target, explicit causal contrasts, BP/strong ceiling, preregistered gates, raw receipts, per-parameter gradient checks, one machine owner, hard-pair slicing, literal failures, correction without erasure.
- Do **not** copy blindly: modulo-97 arithmetic, E16 model shape, two-bank cell, 65,536 reservoir slots, 0.80 of BP as universal gate, or signed full-depth feedback as universally compliant with local synaptic learning.
