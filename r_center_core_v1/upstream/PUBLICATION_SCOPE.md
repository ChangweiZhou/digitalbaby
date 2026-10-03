# Publication scope and intentional omissions

This is the final scientific publication, with narrowly disclosed privacy redactions. The original audited files were not modified.

## Preserved original evidence

All 192 compressed scientific receipts, all three technical receipts, the unit qualification receipt, qualification/admission records, final ANALYSIS.json, RUN_LEDGER.json, SCIENCE_MANIFEST.json, scientific report, final independent review and RESULT_ACCEPTED.json are copied byte-for-byte. The complete 113-file historical source/input lock roster is represented. All src/ and tests/ files are exact original bytes; 105 of the 113 lock entries remain byte-identical.

## Eight disclosed vendor exceptions

Nine personal local documentation paths in eight inherited vendor files are replaced with [REDACTED_LOCAL_DOCUMENT_PATH]. Seven config JSON files contain provenance-only proposal/source_proposal/literature_memo strings; one inherited helper source contains a handoff-document path. The replacements remove a personal account path and private attachment identifiers. No numeric configuration, model bytes, result, or receipt was edited. These are public copies, not newly executed scientific source.

**The original LOCK.json is preserved and is not rewritten to match the redactions. Frozen integrity entrypoints reject the changed vendor bytes. This publication does not provide complete byte-identical source closure or an immediately executable authenticated replay of the original locked study.** The publication manifest records each original execution hash and distinct published hash. It does not assert that redacted source produced the receipts. A reader can verify published-file integrity and trace the original execution hashes through the original lock and receipts, but cannot independently recover the omitted path strings from this publication.

Affected files:
- vendor/package/REFERENCE_SOURCE/minifly/iteration15/config.json: proposal
- vendor/package/REFERENCE_SOURCE/minifly/iteration18/config.json: proposal
- vendor/package/REFERENCE_SOURCE/minifly/iteration19/config.json: source_proposal
- vendor/package/REFERENCE_SOURCE/minifly/iteration20/config.json: proposal
- vendor/package/REFERENCE_SOURCE/minifly/iteration22/config.json: proposal
- vendor/package/REFERENCE_SOURCE/minifly/iteration23/config.json: proposal, literature_memo
- vendor/package/REFERENCE_SOURCE/minifly/iteration24/config.json: proposal
- vendor/package/REFERENCE_SOURCE/minifly/iteration5/src/prepare_interface.py: handoff documentation path

## Intentionally excluded categories

- Private administrative and checkpoint bookkeeping, including backup status, archive identities and storage/Library references
- Operational helper scripts and recovery correspondence/plans/test logs, which mix execution history with private administrative details
- Raw operational logs, caches, bytecode, local Git history, scratch files and supervisor lockfiles
- Original independent audit implementation/recomputation payload and aborted-audit diagnostics, which bind private archive/recovery administration and local executor details; their original hashes remain in RESULT_ACCEPTED.json
- Unlocked inherited narrative/vendor documentation, historical source-location provenance and unrelated historical saved-audit summaries that are outside this experiment's executed hash closure
- Earlier running-status checkpoint README/manifest, superseded by this final publication

## Authentication boundary

RUN_LEDGER.json keeps the three original generic relative recovery-acceptance references. They are scientific provenance pointers without private identifiers. The referenced administrative evidence is intentionally not published. Removing or rewriting these ledger fields would break its original hash in SCIENCE_MANIFEST.json and the accepted result, so the ledger is unchanged.

RESULT_ACCEPTED.json and RESULT_REVIEW.md are historical originals. They bind private evidence that is not included here, and their original conditions concerning archive transfers/publication remain as recorded at review time. This publication does not independently re-certify historical checkpoint persistence, omitted recovery evidence or every dependency of that private acceptance. It preserves the original scientific result and reports these verification limits explicitly.

publication/verify_public.py is a standard-library hash/size inventory checker only. It is publication tooling, not part of the historical 113-file science lock or an alternate scientific analysis. No new behavioral runs or analysis-method changes were made for publication.
