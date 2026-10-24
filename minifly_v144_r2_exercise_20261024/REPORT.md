# V144-R2 three-cycle preparation exercise

24 October 2026. **Three bounded development cycles completed. The full V144 Stage C capacity screen has not run, and no native addition or persistent-core result is claimed.** This exercise implements and tests the integrity/controls portion of [V144-R2](../minifly_v144_dev_20261024/NEXT_EXPERIMENT_DESIGN_R2.md) against unchanged V144 and V143 learner sources.

## Result and interpretation

The old V144 validity check could accept a nonempty but scientifically empty metrics object, and its reducer could silently summarize whatever files existed. The R2 exercise adds a typed reference-result validator and exact-roster reducer. Ten deliberately corrupted result fixtures are rejected, including missing nested rows, NaN/infinity, out-of-range accuracy, wrong identities/checkpoints/exposure and a falsely scored answer. Missing, extra and duplicate roster cells are also rejected. The local restoration check verifies 58 source/asset files by SHA256.

The tiny reference run used one fresh **development** world, 40 training orbits, H=128, rate 0.03, decay 0.001, and four passes per mode. Each mode received 292 record presentations and 708 observed answer/EOS tokens. The learner generated its own byte prefix and EOS. Held results were:

| Mode / control | Held complete accuracy, orbit weighted | Taught complete accuracy, orbit weighted |
| --- | ---: | ---: |
| BP | 6.67% | 2.50% |
| DFA | 6.67% | 5.00% |
| Frozen features | 3.33% | 2.50% |
| Literal COPY_A | 13.33% | evaluator-only |
| Literal COPY_B | 13.33% | evaluator-only |
| FE-PERM nearest training answer | 0% | evaluator-only |
| Raw/unordered lookup; frequency | 0% | evaluator-only |

Payload-only and complete held scores were recomputed with the **same orbit denominator**. At four passes they happen to be equal for each learned mode; this says nothing about the longer-run EOS failure rate. The two literal-copy controls beat every learned mode here. This strengthens the warning that zero-scoring held-key lookups are weak hurdles. Four passes leave taught acquisition very poor, so the fixture cannot diagnose generalization capacity or choose a learning mechanism.

## What each cycle established

**Cycle 1 — source and validity.** Pinned the local V144/V143 runtime closure and owner assets, matched the inherited FE-PERM event trace on representative byte prompts, checked all 257 legal byte/EOS codes, verified the development split constraints and 54-cell first-stage BP roster, and exercised the result-schema mutations and CPU-admission boundary. The source lock points to Git objects where available and otherwise names an owner asset or source file that must accompany the compact bundle. Its verifier refuses a missing or mismatched asset; it does not substitute data.

**Cycle 2 — development-only test.** Ran BP, DFA and FROZEN on `V144-R2-EXERCISE-C2` for four passes. Exact own-output row scoring, split/source IDs, checkpoint counts and exposure totals passed the stricter validator. Added literal operand-copy and FE-PERM nearest-neighbor controls; these remain evaluator-only. Saved a BP checkpoint after pass 2 for the resume rehearsal. This development world is not CAP-D1/D2 or CAP-H1/H2.

**Cycle 3 — audit and resume.** The exact three-cell roster reduced as `COMPLETE`; deleting a declared cell in the manifest, adding an undeclared cell, or declaring a duplicate produced `INCOMPLETE` with no final rows. A corrupted source hash was rejected. BP resumed from the pass-2 checkpoint to the same final state hash as a separate uninterrupted four-pass run.

The first Cycle 3 attempt failed **before reduction** because the checkpoint loader had not imported the parent's `fastmem` module. That attempt, its source lock and its three development fits were preserved under `attempts/attempt1_resume_import_failure/`. After the import-order repair, all three cycles and the development fits were rerun under a fresh source lock. No result from the failed attempt was reused as final evidence.

## Accounting and remaining boundary

The successful attempt applied 1,314 development records and 3,186 observed target tokens: three 4-pass fits, one additional uninterrupted 4-pass BP fit, and a 2-pass resumed BP continuation. The preserved failed attempt applied 876 records and 2,124 target tokens before its technical Cycle 3 failure. Thus this exercise incurred **at least 2,190 development record applications and 5,310 observed target tokens**, not merely the 876 retained final-mode records. The five completed cycle scripts record at least 4.50 summed process CPU-seconds; the failed Cycle 3 startup and some setup overhead are not included, so this is a lower bound, not an all-in cost. Technical trace and scoring queries are not counted as record presentations.

The **full Stage C dispatcher is still not qualified**: only its first-stage 54-cell roster, one-per-H shortlist rule and arithmetic CPU-admission predicate were rehearsed. There was no 27-configuration D1/D2 run, no 4,096-pass continuation, no 18-cell nested-dose holdback roster execution, and no CAP-H1/H2 split generation or evaluation. In particular, a pure admission predicate is not yet an interruption-safe six-hour scheduler. A subsequent implementation must add finite work-item dispatch, actual attempt/cost ledger, checkpoint/resume across 4,096 passes, dependency gating and final lock before opening H1/H2. If that cannot fit the ceiling, it must report `INCOMPLETE_CAPACITY_SCREEN` rather than a scientific failure.

The next scientific comparison remains the bounded complete-addition capacity screen. These exercises justify its validation and controls; they do not establish that the MiniFly can learn complete addition, retain it through later learning, or generalize beyond the tested relations.

## Compact deliverables

`SOURCE_LOCK.json` and `restore_verify.py` verify the dependency/asset closure without copying parent datasets. `MANIFEST.json`, `CYCLE1_CHECKS.json`, `CYCLE2_RESULTS.json`, `CYCLE3_AUDIT.json`, `FINAL_METRICS.csv`, `ACCEPTANCE.json` and the three small R2 source files form the reviewable result. Local per-prompt result JSON and the pass-2 pickle remain outside the compact bundle. The original V144 results and reports were not changed.
