# Package A V3-CLAUDE — launch and persistence contract

Supersedes the execution wording in `SPEC_LOCK.md` ("batches of 16", restart rules). `SPEC_LOCK.md` itself is part of
the learner closure recorded in the final technical receipts and is therefore left unchanged. Implemented by
`src/drive_science.py`; acceptance-tested by `tests/test_launcher.py`; recorded in
`results/technical_final/LAUNCH_QUALIFICATION.json`. Basis: launch review `CLAUDE_A_V3_LAUNCH_REVIEW_20260930`.

1. **Preconditions.** `SOURCE_LOCK.json` verifies; `RESOURCE_BUDGET.json` carries an explicit approval; the lock and
   budget commit is acknowledged by the remote (`git ls-remote` head = local head) before the first job.
   `lock.py create` itself refuses without the approval, a passing final technical audit, unchanged final technical
   receipts and a passing launch qualification whose code hashes equal the current tree.
2. **Accepting receipts.** A receipt counts only after validation: schema, arm, world, `kind = science`, lock digest,
   executed-closure provenance (`source_sha256` = the locked closure), and the independent audit (`audit_a.py`)
   including yoked dependencies (R1_rand needs a validated R1; Z2_rand a validated Z2). Existing receipts are
   re-validated at every start. An invalid receipt is an integrity stop; it is never overwritten or replaced.
3. **Budget (enforced live every poll, cumulative across restarts in `results/science/RUN_LEDGER.json`).** Active
   wall hours ≤ 90; core-hours enforced as **worker-hours** (Σ over world-arm jobs of dispatch-to-exit wall seconds,
   killed jobs included) ≤ 360; per world-arm wall ≤ 2,400 s and resident memory ≤ 800 MB (job killed on breach);
   results disk ≤ 1 GB. A killed job leaves no receipt (receipts are linked into place only when complete).
4. **Restart rules.** Interruption (no failure recorded): resume only uncommitted world-arms, same lock.
   Infrastructure (persistence) stop: resume only after persistence is re-verified. Integrity stop or per-job budget
   stop: resume only with `results/science/CLEAR_AUTHORIZATION.json` naming the failure id, who authorised it and why
   (recorded in the ledger). Cumulative budget exhaustion: final.
5. **Persistence.** Every 4 validated receipts, and at stop: stage, commit and push with bounded retries, then verify
   the remote head equals the local head. Any failure stops dispatch and is recorded durably; committed trajectories
   are never rerun because of a push failure.
6. **Host.** The cloud VM is observed to compute only while a tool call is executing, and the container can be
   reclaimed at any time. The run is kept alive by continuous blocking waits; this is not a durable unattended
   background runner. If the host cannot sustain the contract, the run stops under rule 4 and is reported, not
   restarted blindly.
