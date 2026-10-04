# Cycle 3 bounded post-test review

## Decision

The fixed technical tests are accepted: the completed pure operations attempt 2, full world 310200 pilot, and independently executed exact replay have valid immutable receipts under the accepted source and the separately accepted private-only pilot/replay policy. Receipt source/runtime, clocks, rollback, provenance, score formulas and replay identity passed independent pure validation. Neither execution is an official sample or a scientific efficacy conclusion.

The verified private v11 archive also passes a bounded isolated restoration/bootstrap test. Full Cycle 3 operational acceptance and official launch remain withheld. No CYCLE3_ACCEPTED or LAUNCH_ACCEPTED artifact is issued by this review.

## Frozen identities and actual executions

- Base closure: 132 files; digest `5546bbb941d64a4490b1e4469b24461f052d044ae586d62bb4ba9d70cb290077`.
- Separate five-file policy digest: `ea4bb446aec7185cc120deadeb0599369d04b2a8c81544b1d1a7e7412f093621`.
- Completed operations receipt: `8850d7dbd61a6707ac97809d633379c7a7c4eae5cd43419023e39d69d7aa18c9`; 72 checks, zero native teaching events. Its preserved failed first attempt, exact old-to-new source transition, log, charge and accepted historical Cycles 1/2 all pass the production validator.
- Pilot manifest: `f54530a9f6815fa5ff7d8a916707dcef535ad0adc3e47306f68a9123b31ffdc3`.
- Replay manifest: `6a18d5a414dcfd23ead491c2c631eae547fa9ff80af232afe2bd4d8508a9627f`.
- Both scientific digests: `ce9c1e28995f2f4901976db1f0734394692c46224bfdb59f11f6776bf7bf9f00`.

The pilot and replay have distinct supervisor/child process birth identities and host tokens. Each has exactly 49 parts, 384 events per branch, three branches, 240 first-pre-feedback probe rows and 3,132,834 raw bytes including its manifest. Every manifest part length and hash was recomputed. Direct part comparison found only end.json differed, and removing its resources object made that part exactly equal; no other scientific data were excluded.

The production semantic validator rederived fixture targets, source/birth/runtime identities, centered coefficients, clamped interventions, native readout combination, lowest-ASCII ties, exact event order, event/boundary/final store clocks, delayed times, probe scores and final state links. Both clock errors are 0.0. Every probe preserved its continuing-state fingerprint. There were no official world receipts or official ledger entries.

## Resource and private persistence evidence

Observed supervisor charges are 134.23481855000136 seconds for the pilot and 134.44827190500655 for replay; peak RSS is 301,379,584 and 301,666,304 bytes. Body-only durations are 131.17365727599827 and 131.06919465099782 seconds. All lie within the unchanged 900-second/768-MiB job caps. Cumulative charged worker time is 336.81436449699686 seconds, including the preserved failed operations attempt.

Each prebirth archive contains its exact reserved ledger entry charging 900 seconds, original policy attestation, policy acceptance/report and all five adapter files. Pilot prebirth was privately read back as version 7; replay prebirth as version 10. Current private completion barriers verify the actual archived receipt, every part and the applicable policy evidence. The private queue is empty. Hash-chained attempt, host and transport journals validate.

Both exact host tokens have host_closed journal events; measured host sessions were 168.12293302599574 and 173.12419269200473 seconds. The final host record is inactive, with cumulative active-wall accounting 2399.152572304083 seconds. This review does not infer live process state merely from recorded PID numbers.

The v11 original and materialized readback independently hash to `ee06cdf84c3e1ab74a0b3ded4d2c7ec6b9d2600c75a084cca6f2f6b7a2523ca3`. Its readback version xattr is 11. All 1,020 archive payload members passed streaming hash verification; its ledger is the exact current completed ledger. This uses the existing actual Library readback and journal, not a new remote-current-version query.

## Isolated restoration/bootstrap

The exact verified v11 payload was extracted to a fresh /tmp root with path checks and every extracted file rehashed. Its archived CHECKPOINT_STATE correctly still referred to v10, reflecting capture before v11 acknowledgement. Only in that isolated root, checkpoint metadata was refreshed from the actual verified v11 original/readback; both archive copies were made consumer-local and rehashed.

A fresh process used the restored source and accepted adapter with the pinned existing Python environment. It genuinely re-ACKed operations, pilot and replay by verifying each receipt and policy dependency in v11. Under its isolated exclusive lock, the restored production planner validated the historical cycles, source-transition evidence, WAL, completed receipts and private barriers, and returned completed qualification, missing=[], pending_barriers=[], blocked=[], next_key=null. The ledger hash and 336.81436449699686-second charge were unchanged. The preserved public queue bytes were unchanged. No learner/engine module was imported and no worker was launched. The child used approximately 108.9 MB peak RSS and 1.72 seconds.

Two subsequent unchanged calls to the restored production checkpoint builder both returned changed=false with the same v11 identity and archive; the archive count stayed one. These independently establish local no-op behavior. The parent reports actual private-controller no-op success; no standalone persisted return-value artifact for those external-host calls was available at report preparation.

This proves this exact completed snapshot's local portability and private missing-only planning. It does not package the Python installation, re-run the learner, test a new remote login/materialization, or authorize treating stale checkpoint metadata as the remote current version. The archived host was active at capture; any future real host startup must still perform its ordinary conservative recovery accounting. The unmodified legacy public-planner restore action was not substituted for the separately reviewed private-policy bootstrap.

## Remaining operational gates

1. **Retention forecast fails the current 2-GiB local-backup rule.** The measured v11 ZIP is 7,383,162 bytes; one replay world contributes 540,961 compressed payload bytes. Retaining one prejob and postjob original per official world already forecasts 128 × 7,383,162 + 4,096 × 540,961 = **3,160,820,992 bytes**, exceeding 2,147,483,648 before extra ZIP headers, ledger growth, readbacks, restored copies and staging. A bounded, independently reviewed retention implementation and updated complete forecast are required before launch. No original archive was deleted by this reviewer.
2. The registered worker projection does fit: cumulative 336.814364497 + 1.5 × 64 × 134.448271905 + 900 final-analysis reserve + two 900-second retry reserves = **15,943.848467378 seconds**, below 43,200. Launch must separately validate the 18-hour active-wall forecast including validation, growing checkpoint transfer/readback and restoration overhead, 1-GiB outputs, 48-MiB checkpoint cap, two newest readbacks, temporary/restore space and 3-GiB free reserve. Passing worker arithmetic alone is insufficient.
3. The accepted private-policy scope is only the fixed Cycle 3 pilot/replay and pure postvalidation. It does not admit official science or terminal official analysis. Any continuing private-only launch path must receive explicit narrow governance/source-policy review and the parent launch decision, with current frozen identities, the official lock and required acceptances; it must preserve the original accounting, immutable receipt and private prebirth/completion checks.
4. GitHub sync remains deferred and unqualified. The denied public queue remains exactly `761a6e2aaf6fc58b7bf0e981e62f3a384ed3259d6f1159b6c82fb18f97159cf1`. No private ACK is a public ACK. The owner's waiver can be represented explicitly in a reviewed launch policy, but this review neither clears that queue nor invents a remote tree/commit verification.

## Review boundaries and evidence

All validation was pure. Original source, five-file policy, study ledger, receipts, publication queue and external services were not changed. Writes were confined to isolated /tmp restoration/check scripts and these new audit outputs. The machine-readable review binds the full source/policy maps and receipt inputs. Detailed private operational identities and restoration proof are preserved separately in CYCLE3_POSTTEST_PRIVATE.json.
