# Held-out canary protocol (the checker's analogue of E3)

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`. Normative text: `tools/minifly_check/SPEC.md` section
12.3. Checks: HELD-01..03 (`heldout` command bundle). Ledger: `PROJECT_DOCUMENTS/KB/CANARY_COMMITMENTS.jsonl`
(fields in `tools/minifly_check/SCHEMAS.md` section 18).

**If you are the batch author, stop reading here.** The protocol allows you only `PROJECT_DOCUMENTS/KB/FAILURE_CLASSES.json`,
`tools/minifly_check/SCHEMAS.md` (its section 19.5 gives the fixture format) and Section 8 of the spec. Whoever starts
your session gives you the batch requirements of step 2. This file names checks and fixtures, so it would
contaminate the batch.

Why this exists: the visible corpus and the canary operators are public, so passing them shows regression safety, not
generalization. The checker's score is its held-out recall per checker version. A rise in corpus recall without a rise
in held-out recall counts as overfitting.

## 1. The protocol (spec 12.3, steps 1-8)

1. **Author.** The owner, or a fresh session that has read only FAILURE_CLASSES.json, SCHEMAS.md and the spec's
   Section 8, and has not opened `tools/minifly_check/*.py`. Record the author's identity. HELD-03 requires the batch
   author to differ from the authors of the `tools/minifly_check` commits in the interval.
2. **Batch.** 10 mini-deliveries in corpus fixture format, at least 3 of them clean. At least half must not be
   isomorphic to any visible fixture: a different FM instance shape, different names and a different layout. Each
   carries a key {fixture: expected fired classes}.
3. **Seal.**

   ```
   MFC heldout seal --batch B --in DIR --key-file OWNER_KEY
   ```

   - The command encrypts with an HMAC-SHA256 counter-mode keystream XOR and writes `corpus/heldout/sealed/B.bin`
     (ciphertext) under `tools/minifly_check/`.
   - It produces the CANARY_COMMITMENTS row {bundle_sha256, key_sha256, n, n_clean, author, utc} (the ledger field is
     author_identity). Confirm that exactly one row was added to the ledger. If the command printed the row instead of
     appending it, append it with `MFC ledger append --ledger CANARY_COMMITMENTS --row ROW.json`.
   - Commit the row, and the ciphertext with it, before the checker version under test merges (HELD-01). The
     ciphertext lies under a protected path, so commit it as a governance range and run `MFC governance` on it.
   - The owner holds the key, outside the repo. The plaintext batch never enters the repo.
4. **Reveal and score.** At census, or before a governance merge:

   ```
   MFC heldout reveal ...
   MFC heldout score --checker-sha S ...
   ```

   - Reveal checks that the bundle and key hashes match the commitment (HELD-02).
   - The score runs the frozen checker at S (extracted with git archive) and reports recall per FM class and the false
     alarm rate on the clean canaries, written to `PROJECT_DOCUMENTS/KB/RETRO/HELDOUT_<date>.json`.
   - Post-batch checks (spec 20.18): each canary's version is the checker_version in its own SOURCE_LOCK (the oldest
     one in its tree). A fired id whose RULES introduced_in or changed_in is later is listed under `post_batch`, per
     fixture and in the report, and never counts: it neither catches a dirty canary nor makes a clean one a false alarm.
     A canary without a SOURCE_LOCK has no version, and nothing is excluded. Authors set the version they wrote
     against (corpus builder option `source_lock_patch`); a batch written against 1.0.1 is scored on the round-1
     checks. The pilot's raw recall 3/9 with false alarms 3/3 was exactly this drift (RUN-03).
   - The report always carries the raw numbers next to the excluded ones (`recall_fixtures_raw`,
     `false_alarm_rate_clean_raw`, and per fixture `caught_raw` / `false_alarm_raw`), and `heldout score` prints both.
     Quote both: a batch built with the builder's default SOURCE_LOCK (checker_version 1.0.0) excludes every 1.0.1
     detection and every 1.0.1 false alarm, so its excluded numbers alone say nothing about round 1.
5. **Burn.** A scored batch moves into the visible corpus through governance and is never held-out again.
6. **Fixes after a miss must generalize.** A reviewer rejects any fix that matches canary literals (names, layouts,
   magic values), and the next sealed batch tests it.
7. **Held-out incident recall.** See section 2.
8. **Archive.** The pre-Full151 archive digest (`PROJECT_DOCUMENTS/KB/ANALYSIS_20261024/ARCHIVE_DIGEST.md`) is training
   data. It was already digested in the design session, and its FC01 is the same defect as corpus fixtures C01/C02a.
   Archive-derived fixtures (an A-series) may extend the visible corpus but never count as held-out. The F-ARCH-* seed
   findings are training data too.

## 2. Held-out incident recall (step 7), and how to record it

Held-out incident recall counts only incidents first recorded after the v1 freeze (UD-0001) whose row says
heldout_eligible true. The order of events is what makes an incident count:

1. Someone finds a defect and appends an INCIDENTS report row: fm_class, delivery, symptom, evidence `path:line@sha`,
   found_by_agent, found_utc.
2. The frozen checker is run on the defect as it stands, before anyone writes a check: run the gate or command that
   should have caught it. Record its result as caught_at_discovery and its checker_sha256 (from the GATE report) as
   checker_sha_at_discovery.
3. Someone other than the future check author writes the reproducing fixture and commits it by hash first: a
   CANARY_COMMITMENTS row with n = 1 (bundle_sha256 of the fixture, author_identity). Only then may anyone write a
   check.
4. Set heldout_eligible true only when 1-3 happened in that order. If the check author wrote the fixture, or a check
   came first, the incident still goes through the incident-to-check loop but does not count as held-out.
5. Recall for a checker version = the eligible incidents whose checker_sha_at_discovery is that version and whose
   caught_at_discovery is true, divided by all eligible incidents with that checker_sha_at_discovery. Report it beside
   the sealed-batch recall at census.

The seed incidents (INC-SEED-*) predate the freeze and are never eligible.

## Open limits

- The author's identity and 'has not read the code' are recorded, not proven. HELD-03 checks commit identities, which
  are tamper-evident only.
- Between sealed batches, a loosening that looks monotone on the visible corpus (regex narrowing, for example) can go
  unnoticed until the next batch is revealed (spec 15: partly open). Keep batches regular.
