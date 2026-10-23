# Independent Cycle 2 source/design review

## Decision

**NOT ACCEPTED FOR NATIVE EXECUTION.** The frozen scientific core, fixed fixture and narrow causal estimand are source-supported, but the proposed Cycle 2 certificate does not yet test the full state, clock, clone and target-label invariants it claims. The defects below are instrumentation/test defects, not evidence of a scientific learner failure. Correct them without changing the learner, task, scales, teaching quota, seeds or inferential rules, preserve this revision, and obtain re-review before the reserved test.

Reviewed design: `protocol/ROUND1_DESIGN_CYCLE2.md`, SHA-256 `f26726dd3c6347f8c38266b5ccfd10e59b7ebeb0795635925366e8f22ec169c8`. The accompanying JSON binds the complete nonempty map returned by the reviewed `integrity.hashes()`, plus the exact report hash. This rejection authorizes no native outcomes, pilot, official world, publication or later-cycle work.

The reviewer executed no native learner import, birth, prediction, teaching, or reserved test. Checks were file/source reading, AST parsing, hashes against both the supplied projection manifest and baseline public mapping, pure fixture generation, and an isolated pure helper exercised on synthetic namespaces. The check script and evidence are `audits/CYCLE2_SOURCE_PURE_CHECK.py/.json`.

## Blocking corrections

### C2-1: The claimed complete state certificate omits operative state

`tests/cycle2_adversarial.py:33–39` hashes native snapshot/state, fixed maps, visible bytes and the fly object, but not all operative brain/front-end/wrapper configuration. In particular, FE `tau` changes future sensory decay (`bytecore.py:704–712`) yet is absent from F151 snapshot and `full_state`; wrapper `arm` and `scales` control teaching/readout and are absent from `assay.state_identity` (`assay.py:26–29`). Brain `n_native_kc` is likewise not expressly certified. A comparison can report equality despite these operative differences.

Required fix: use one audited complete operative-state representation covering all native model/reader fields, brain parameters and feature maps, FE class/configuration/buffers, wrapper arm/scales/cache/time/count, plus the existing audit values. Exclude only explicitly identified nonoperative instrumentation and the intentionally different output-policy method binding. Fail on unclassified fields rather than silently omitting them. Add pure or disposable-clone mutation checks establishing that FE tau, wrapper arm/scales and representative fixed reader/model parameters change the certificate. Preserve unchanged frozen source.

### C2-2: Clone and probe isolation are overstated and incompletely guarded

The lone isolation test (`tests/cycle2_adversarial.py:67`) mutates one private fast entry at newborn state. It does not certify slow/adaptation, FE buffers and CONTENT visible state, predictor weights/bias, pending codes, cached wrapper values or clocks. Native `EvoLearner.clone()` shares fixed matrices and readers (`model_evo.py:239–250`); the brain fallback clone shares fixed pooling/hash maps (`common_platform.py:34–42`). Several shared nominally fixed objects are writable: phenotype matrices are produced by array arithmetic (`model_evo.py:67–83`), canonical CSR buffers by `portable_birth.py:86–93`, reference-reader Q by `bytecore.py:237–243`, and brain maps by `brain_byte.py:97–108`. This is intentional fixed sharing in the frozen core, not evidence that ordinary reads mutate it, but it defeats a blanket deep-isolation claim.

Moreover, `assay.probe:38–48` guards the continuing branch only with the incomplete `state_identity`, which omits fixed reader/native matrices and FE configuration. Such fields can change through a shared clone alias without the present guard detecting it. Equality among all four policy branches also cannot expose common-mode drift in a shared fixed object.

Required fix: state the precise mutable-state-isolation contract and intentional fixed sharing. Test all mutable operating categories using allowed disposable clones, including a pending-prediction state, with no extra valid outcome. Anchor fixed parameters/connectome/reader/predictor identity to the approved newborn fingerprint throughout phase/probe checks, not merely to other branches. Use the complete certificate for continuing-state guards around every probe and label intervention. Do not alter the native clone or learning rule to manufacture a pass. Shared audit records are external append-only instrumentation, not a deployable mutable-state guarantee.

### C2-3: Clock checks and nonplastic comparisons are incomplete

`check_clock` (`tests/cycle2_adversarial.py:47–52`) validates brain time, elapsed time and cleared pending state, but ignores FE time and last-byte consistency. The independent pure checker demonstrates that a synthetic store with `brain_t=elapsed=0` and `fe.t=12345` passes. The sole desynchronization adversary changes brain time, so it cannot catch this hole. Old-end and new-start checks discard their vectors (`:83–86`), leaving no receipt for those locked checkpoints. Invalid-byte, backward-prediction and backward-flush cases are not exercised by the existing invalid-call block.

The nonplastic digest omits pending codes, native event/presentation counts and other sensory configuration, and is compared across causal branches only after newline/flush (`:97–103`), when CONTENT visible has already reset. That cannot certify equality of the transient pre-outcome code or other sensory state used by teaching.

Required fix: define clock invariants by phase, validate FE/brain/elapsed consistency and ordered last-byte/pending timestamps, retain all four locked checkpoint vectors, and test independent FE/elapsed/brain corruption plus invalid byte and backward predict/flush using allowed invalid disposable clones. Compare the full source-justified nonplastic sensory/adaptation/pending/clock projection across W/N_old/N_new after each prediction, outcome, newline and flush. The plastic fast/slow values and causal audit write totals are expected to differ and must not be included in that cross-causal equality projection.

### C2-4: Target-label adversary does not exercise the evaluator under changed targets

`tests/cycle2_adversarial.py:111–115` calls the same target-free `cue` function twice, then constructs two target values and writes `prediction_unchanged=True`. Neither target was supplied to either actual evaluator/probe call, no scored rows are compared, and the continuing state is not guarded around this intervention. The static target boundary in `assay.probe` is sound, but this code is not the advertised adversarial test of that boundary.

Required fix: run the actual target-bearing evaluator/probe path on the same single held-out query under two different evaluator labels, with the same source state and cue/time. Compare raw prediction/policies excluding target/correctness fields; establish that correctness is calculated from each supplied label; compare complete continuing-state before/after both calls. Keep exactly the already allowed two extra disposable predictions, with no outcome supplied.

## Source-supported findings

All 94 baseline learner/bootstrap/vendor files in the current runtime closure match the projection manifest and the corresponding baseline public-provenance mapping. The frozen learner hash is `d69b5c1561048aaa37b432a22a107202bf2fbba666a161b392a7e7ab7c245973`. All 73 Python files in the closure parsed. The actual immutable source was read independently, including learner/bootstrap; stores/birth/import paths; native byte, CONTENT, clone, readout and snapshot methods; EvoLearner event/clone; and `advance73` adaptation/plastic-write equations.

Eight independently born canonical stores, shared FE0/private CONTENT, fixed scales, native shared teaching and centered signed private teaching are preserved. `observe_outcome` creates y only from the received environmental outcome; shared teaching uses `int(ALPHABET[j] != byte)` and R_center coefficients are c=0, s=1/4−y. Returned choice/combined scores do not enter teaching. Native writes still depend on their own state and native feedback; this is not an outcome-only dynamical system.

External clamps force only `write=False` on both store teach interfaces. Native no-write events retain adaptation and decay/clock advancement: adaptation in `advance73` depends on x, dt and preceding adaptation, not punishment or plastic. W−N_old_relation therefore measures the total effect of old value teaching, including its consequences during later new teaching. It does not remove a separate relation store while preserving old exact facts. W−N_new is the matched new-teaching control.

The engine imports byte-identical source behind new governance and asserts the new integrity module path. No world/stage/target/identity is passed into the Learner. The inherited fixture import is dormant in the inspected learner call graph; poison tests cover make_world aliases, although `common_platform.fact_variants` is a separately imported alias and should also be poisoned if the test claims comprehensive inherited-generator poisoning. Static inspection does not find a call to it in the active learner path.

The four output policies preserve teacher inputs by construction in this action-independent stream. Full operative-state equality remains unobserved and, before the above fixes, insufficiently certified. Alpha=.5 remains offline assessment until the corrected fixed-stream equivalence test passes. There is no closed-loop or environmental-action inference. CORR2 remains NOT_INSTANTIATED; no independent identity-corroboration metric appears in the implementation.

The pure fixture check generated all 68 registered worlds (64 official fixtures and four development/pilot fixtures), without learner execution. Each has 384 events, 192 old then 192 new, exactly 12 old/4 held-out/16 new cues, zero held-out teaching, balanced outcomes, the declared repetition counts and locked four times. The generator enumerates all 576 Latin squares and each tested partial table has one completion. The estimand remains taught-dependent held-out table completion; the acknowledged row-only missing-symbol shortcut prevents stronger reusable-operator claims. No official learner outcomes were exposed.

## Quota, admission and later gates

The reserved script statically contains one R_center base newborn, six matched writing/clamp policy clones, and exactly eight taught events per branch (first four old and first four new on 310101), totalling 48 taught records. Additional current invalid-call clones execute no valid teaching. Final causal probes total 96 first pre-feedback predictions; the two extra held-out predictions are within the stated label-test allowance, once repaired. No adaptive seed/task selection or extra mechanism is introduced.

Technical admission binds the accepted design, nonempty exact source closure, report hash, supervised child PID/source digest, nonoptimized Python and pinned runtime. The supervisor provides an exclusive lock, immutable-receipt existence gate, runtime/resource checks and bounded infrastructure-only retry admission. The proposed source/report tamper test is structurally sound and restores bytes in finally; include a frozen-learner-path mutation as well if claiming specifically demonstrated frozen-core tamper rejection rather than relying on the same source-map logic.

Cycle 1 remains accepted solely for its already completed two-record smoke. Cycle 3 recovery/replay, conservative crash accounting, durable ledger/journal, bounded artifacts, off-host restore, actual GitHub receipt roundtrip, publication scope/licensing, full-pilot resources and final launch/analysis approval remain separate gates. The current checkpoint builder still gathers the entire local source tree including unused baseline results; the eventual public exporter must implement the narrowed public projection rather than blindly publishing that private archive. This source review does not approve that pipeline.

After the four blocking corrections, re-review may authorize only the fixed Cycle 2 48-teach-event development test and its stated invalid clones/probes. Successful execution will still require independent receipt/ledger completion review. Correct the misleading Cycle-1 heading at the top of the Cycle-2 design when preserving the amended design revision.
