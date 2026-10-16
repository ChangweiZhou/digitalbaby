# Cycle 3 — complete real-job and failure-boundary audit

Final design remains the fixed HELD/global-sample comparison. Scope is still technical qualification only, with no training on any of the eight arithmetic worlds or two engineering maps.

This test uses the actual job/data/probe/audit/reducer code, not a substitute learner. Planned actual applications: 24 for an uninterrupted versus explicitly paused/restored INTERLEAVED three-history tiny job, plus 16 for both native variants' W/NO_WRITE tiny jobs with safe resume. Together with cycles 1–2 this is 89, under the shared 96 cap.

Resume contract: a write-ahead START and COMMIT pair per record; acknowledged PAUSED checkpoint cursor must exactly equal the journal cursor. An ambiguous started record, stale checkpoint, completed job or changed manifest is rejected. There is no auto replay or retry.

Additional refinements:

- Checkpoints bind operative source identity and complete job output, not just one learner's weights.
- Checkpoint and process-segment costs have append-only external records, so a pause does not lose the final serialization charge.
- Functional native geometry uses the retained floating-time subtraction exactly, rather than substituting an analytically equal but numerically different constant.
- Actual full-job auditors recompute shortcut predictions, exposure flags, own-output grades, target traces, update counts and the checkpoint grid; technical short truncation caps are explicit.
- The science reducer refuses technical receipts, incomplete paired rosters and a changed primary grid. Only fixed global HELD rows form the primary AUC.

Acceptance here means implementation/control qualification. It cannot establish addition E3 or the full-dose 16/16 native repair result.
