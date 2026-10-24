# Held-out canary batches (SPEC 12.3)

The visible corpus measures regression. Held-out batches measure whether the checker generalizes to
failure shapes its authors never saw: the checker's analogue of E3. No held-out fixture is created now;
this directory only ever holds ciphertext (`sealed/<batch>.bin`) until a batch is revealed and burned.

## Protocol

1. **Author.** The owner, or a fresh session that has read only `PROJECT_DOCUMENTS/KB/FAILURE_CLASSES.json`,
   `tools/minifly_check/SCHEMAS.md` and SPEC Section 8, and has never opened `tools/minifly_check/*.py`.
   The author's identity is recorded. Authors of tools/minifly_check commits in the interval may not
   author a batch (HELD-03).
2. **Batch.** 10 mini-deliveries in corpus fixture format (TimelineGit: timeline.json, repo/, history/),
   at least 3 of them clean. At least half must not be isomorphic to any visible fixture: a different FM
   instance shape, different names and layout. Each batch carries a key {fixture: expected fired FM
   classes}.
3. **Seal.** `MFC heldout seal --batch B --in DIR --key-file OWNER_KEY` encrypts the batch with an
   HMAC-SHA256 counter-mode keystream XOR into `corpus/heldout/sealed/B.bin` and appends a
   CANARY_COMMITMENTS row {batch_id, author_identity, utc, bundle_sha256, key_sha256, n, n_clean,
   fm_coverage, checker_sha_under_test}. The row is committed before the checker version under test
   merges (HELD-01). The key stays with the owner, outside the repository.
4. **Reveal and score.** At a census, or before a governance merge: `MFC heldout reveal` (bundle and key
   hashes must match the commitment, HELD-02), then `MFC heldout score --checker-sha S`, which runs the
   frozen checker extracted with git archive at S and writes recall per FM and the false-alarm rate on
   the clean canaries to `PROJECT_DOCUMENTS/KB/RETRO/HELDOUT_<date>.json`.
5. **Burn.** A scored batch moves into the visible corpus through governance (like any new fixture, see
   `corpus/README.md`) and is never held-out again.
6. **Fixes after a miss** must generalize: a reviewer rejects any fix that string-matches canary
   literals, and the next sealed batch tests it.
7. **Held-out incident recall** counts only incidents first recorded after the v1 freeze with
   heldout_eligible true. Their fixture is written by someone other than the check author and committed
   by hash (a CANARY_COMMITMENTS row with n = 1) before any check is written; the frozen checker's
   result is recorded as caught_at_discovery.
8. **Archive.** The pre-Full151 archive digest is training data (its FC01 is the defect of C01/C02a).
   Archive-derived A-series fixtures may extend the visible corpus but never count as held-out.

The checker's score is held-out recall per checker version. A rise in corpus recall without a rise in
held-out recall counts as overfitting. Progress of the program is never measured by recall, gates or
flags (I11).
