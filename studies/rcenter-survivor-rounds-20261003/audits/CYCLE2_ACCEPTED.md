# Independent Cycle 2 completion review

## Decision

**CYCLE 2 ACCEPTED AND CLOSED FOR ITS FIXED TECHNICAL SCOPE.** The source-reviewed 48-teach-record development test passed, and its receipt, source restoration, resource accounting and recorded invariants reconcile independently. This does not qualify a full 384-event life, a full pilot, recovery/publication, scientific utility, or official launch. No additional native execution is authorized by this completion report.

Accepted design SHA-256: `1541cfbcd7ea6e7a38113f9d3358db2ef313ea5a3822074ff65f76023c8f57b1`.

Accepted receipt: `receipts/cycle2/smoke.json`, SHA-256 `661ca626edd7aee7427b85a64c1f66034d2b4fc06dd404bcf20becdd6381ccef` (404,427 bytes).

Audited executable source-map digest: `d4465d61a48e2bafd4d1671b202d0b23c33c8b3cee135bdeece4fab73f36d937`, 108 files.

The independent reviewer did not import/run the native learner or duplicate the reserved test. Receipt validation was pure Python: JSON validation, source hashing, fixture reconstruction, digest/group comparisons, clock arithmetic and independent scalar recomputation of all policy scores, ties, choices and correctness. Reproducible validation is in `CYCLE2_RECEIPT_CHECK.py/.json`; `CYCLE2_LEDGER_SNAPSHOT.json` preserves the ledger as reviewed.

## Verified execution and accounting

The ledger records exactly one completed Cycle-2 attempt and no Cycle-2 failed/interrupted attempt. Its exact pinned runtime is Python 3.11.15, numpy 2.2.6, scipy 1.14.1 and numba 0.61.2. The supervisor charged 46.049941589997616 seconds and observed peak RSS 367,026,176 bytes (about 350.02 MiB), below the 900-second/512-MiB caps. The child reported 43.7514891259998 seconds and 366,878,720 bytes, both within the supervisor measurements. The single log record agrees with the receipt's pass/check/resource fields.

Cumulative recorded programme worker and active wall time is 52.05447847699543 seconds, exactly the sum of Cycle 1 and Cycle 2. Only Cycle-1 and Cycle-2 smoke receipts exist at review. These observations are not a feasibility estimate for 64 full official worlds.

All receipt source hashes equal the accepted source-review map and the current `integrity.hashes()` result. The design and source-review report hashes also match. In particular, the deliberately tampered evaluator/frozen-learner/report bytes were restored exactly. Active-job mode/source identity and the supervisor receipt hash reconcile. The audit preserves immutable input hashes rather than relying on a mutable future ledger.

## Verified causal and policy certificates

- Exactly 48 unique record/branch rows: first four old and first four new events of seed 310101 across W, N_old_relation, N_new, half, shared and forced. Stage and blocked flags match the frozen intervention schedule.
- Exactly 192 phase rows: 8 records × prediction/outcome/newline/flush × 6 branches. Every phase has one full operative digest across the four W-writing output policies, one nonplastic digest across all six branches, and the same fixed newborn fingerprint.
- The forced emitted answer differs from W on all eight records, so the output intervention was nonvacuous. W and N_new certificates agree through the old block, before new-write intervention. Old and new controls have differing full certificates after their respective intervention periods. Full certificate inequality includes instrumentation; it is not by itself a measurement of a particular plastic array. All 32 final raw readout pairs also differ between W and each causal control, confirming a technical observable distinction without implying useful accuracy.
- All 48 record flush errors are exactly zero. There are exactly 24 branch/checkpoint vectors (192 store rows) for old_end=31,680, new_start=118,080, new_end=149,760 and final=236,160. Every store has brain_t=FE_t=elapsed−elapsed_base at each checkpoint, no pending time, exact last-byte time, and the expected counters: 56 bytes/4 teaches after the four old records and 112 bytes/8 teaches after both sampled blocks. The empty periods are clock advancement only, not unrecorded teaching.
- The 13 named source-reviewed checks all passed, including invalid calls, independent brain/elapsed/FE corruption, operative-mutable clone isolation, full/fixed fingerprint sensitivity, inherited generator poisoning, write clamps and source/report tampering. Their execution is supported by the hash-pinned test and completed supervised process; the auditor did not rerun them.

The full/fixed certificate is the corrected, source-audited representation. Native clones still share nominally fixed objects; the accepted result is mutable-state isolation plus invariant-guarded fixed sharing, not a general deep-copy or serialization guarantee. Fly statistics and birth provenance remain outside the operative learned-state claim.

## Verified probe boundary

The receipt contains exactly 96 final causal probe rows, comprising 12 taught-old, 4 held-out and 16 new queries for each of W/N_old_relation/N_new, in exact fixture order. Every row is first pre-feedback, target-free on the learner side, and has matching complete continuing-state before/after hashes and the common fixed fingerprint.

The two additional actual target-bearing evaluator calls use the same held-out cue and continuing state with two distinct labels. Their raw vectors and all four policy scores/choices/ties are identical. Correctness is independently recomputed against each label, and complete continuing state is unchanged. In this particular pair both chosen labels differ from the emitted answer, so correctness remains zero; the passed assertion is label-independent prediction with correct label-dependent scoring logic, not a demonstrated flip in correctness.

All **392 policy rows** (98 probes × 4 policies) were independently recomputed from raw shared/private vectors using the fixed scales. Alpha-1 combined vectors, alpha-half, alpha-0 and private-only values match exactly; all tie sets, lowest-ASCII tie decisions, emissions and correctness agree. No probe target enters native teaching. No full-life scientific accuracy estimate or inferential decision is made from these technical rows.

## Interpretation and remaining gates

Cycle 2 establishes exact operative-state equality for alpha=1/.5/0 and adversarial returned emissions on this tested short, fixed, open-loop observed-outcome stream, together with the source-backed no-returned-output feedback argument. It supports using the prespecified alpha alternatives as output-only policies under these fixed task semantics. It does not empirically establish equality over the unrun full 384-event life or under action-dependent environments. Cycle 3 must qualify the complete life and execution/recovery path before official inference.

The endpoint remains taught-dependent held-out table completion, with the acknowledged row-completion shortcut. There is no abstract reusable-operator claim, formation diagnosis, useful accuracy claim or mechanism promotion. CORR2 remains NOT_INSTANTIATED.

Still unapproved: Cycle-3 design/audit/test; the full fixed pilot; full-life resource feasibility; replay/interrupted-resume and missing-only recovery; durable journal/backup/restore; public-source licensing/projection and actual GitHub receipt roundtrip; official analysis implementation; final source/runtime lock; programme budget/launch approval. Historical results and either completed technical cycle cannot substitute for these gates.

**Final disposition:** the second distinct design→independent audit→bounded test cycle is accepted. Preserve all evidence. No new native runs, publication, later round or official launch follows automatically.
