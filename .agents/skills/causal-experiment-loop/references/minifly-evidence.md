# MiniFly casebook: why the causal experiment loop matters

**Context:** A distilled workflow example grounded in the public repository `ChangweiZhou/digitalbaby`, consolidated branch `codex/minifly-consolidated-20261023`, checked through commit `0ea406286201` (2026-11-10 EST). This file is **historical evidence**, not a claim about the current head at execution time. Refresh the repo before invoking the skill on a live project.

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

## Case H — E17 Phase 3: temporal credit, not exact feedback alone, explains most of the rescue

Dismantling the Phase 2 multi-change rescue on a frozen first bank found F1 mod-97 family means of **67.9/72.9/72.8** with exact feedback alone versus **80.1/84.9/84.9** with the upstream extended trace alone; the combined arm scored **87.5/90.1/91.9**, and the frozen BP twin **90.2/90.0/91.5**. Freezing the embeddings alone while letting both banks update still collapsed. At birth, the ordinary sign rule's first-bank write projection had cosine **0.09/-0.06/0.05** with a BPTT reference; carrying sign-feedback credit through time to every layer increased per-layer cosine to **0.45-0.81**. **Lesson:** compare learning-signal *reachability and alignment over depth and time*, not only sign-vs-exact weight feedback; a whole-layer freeze is a diagnostic and not the lifelong learner goal.

Sources:
- https://github.com/ChangweiZhou/digitalbaby/blob/0ea406286201/MOD97_LOCAL_RULE_20261106/E45_REPORT.md
- https://github.com/ChangweiZhou/digitalbaby/blob/0ea406286201/MOD97_LOCAL_RULE_20261106/exp45/E17_ALIGN_AT_BIRTH.md

## Case I — E16-M3: a preregistered dose-direction prediction failed

The hypothesis predicted more replay would injure the sign rule's hard pairs while backprop stayed flat. Over replay shares **0.25/0.50/0.75**, hard-pair results instead increased: seed 400009 **43.7/46.3/73.2**, seed 600013 **23.3/44.8/60.2**, seed 300007 **59.2/67.7/75.7**. Backprop improved in the same direction but by less. Share zero left almost no old hard-pair retention. The improvement was mostly final retention rather than peak acquisition, and a larger share reduces the slots available for novel observations. **Lesson:** report a dose-response in actual replay fractions, novel presentations and CPU-seconds; an inverted prediction is valuable and must not be silently reclassified as predicted. Higher replay is *not* intrinsically a more efficient lifelong learner.

Source: https://github.com/ChangweiZhou/digitalbaby/blob/0ea406286201/MOD97_LOCAL_RULE_20261106/E16M3_REPORT.md

## Case J — E16-F7: an in-sample early warning did not generalize

Six new drifting-stream seeds all passed the predeclared family means, making **16/18** total drift seeds passing, but only **4/18** passed the hard plus-minus pair threshold. The early signature S was selected after analyzing the prior 12 seeds; prospective confirmation on six additional seeds made the rank correlation on all eighteen only **0.109**, below the preregistered **0.35**. No new seed tested S's low-threshold clause. Some passing seeds lost the hard pair only after the signature's observation window. **Lesson:** split discovery from genuinely new confirmation samples; report the primary family gate and hard mechanism-relevant subgroup gates together, and do not call a post-hoc separator a validated predictor.

Source: https://github.com/ChangweiZhou/digitalbaby/blob/0ea406286201/MOD97_LOCAL_RULE_20261106/E16F7_REPORT.md

## Case K — E17 Phase 4 and E18: pending contrasts, not results

E17 Phase 4 launched nine lives comparing full-depth **through-time signed credit** against projection-only trace and BP on identical learner data, with no layers frozen; **no completed result** was available at the cited head. E18 separately preregistered four development-stage arms that test a next-byte head at each memory bank with or without cross-bank relayed teaching signals. Its dry-run and gradient-leak checks are **technical qualifications**, not proof that local heads can learn or compose. **Lesson:** always label *prepared / dry run / launched / completed* and establish no-gradient leakage before interpreting a new local-objective experiment.

Sources:
- https://github.com/ChangweiZhou/digitalbaby/blob/0ea406286201/MOD97_LOCAL_RULE_20261106/E45_PREDICTIONS.md
- https://github.com/ChangweiZhou/digitalbaby/blob/0ea406286201/MOD97_LOCAL_RULE_20261106/E18_PREDICTIONS.md

## What to borrow (not what to copy)

- Borrow: fixed scientific target, explicit causal contrasts, BP/strong ceiling, preregistered gates, truly fresh confirmation seeds, raw receipts, depth-by-time gradient checks, one machine owner, hard-pair slicing, replay dose/cost curves, literal failures, correction without erasure.
- Do **not** copy blindly: modulo-97 arithmetic, E16 model shape, two-bank cell, 65,536 reservoir slots, 0.80 of BP as universal gate, or signed full-depth feedback as universally compliant with local synaptic learning.
