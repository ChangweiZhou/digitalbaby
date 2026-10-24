# B2 (clean): honest FAIL by the OLD drop guard, renamed worlds, 9 MAIN worlds

**Kind:** clean. A correct checker must accept every command (exit 0) and fire nothing.

## What differs from the base (all legitimate)
- Worlds are renamed. Dev worlds are `w_tune_1`, `w_tune_2`; the rehearsal world is `w_fresh_1`; MAIN worlds are
  `w_main_1..w_main_9` (listed, n = 9, min_worlds 8). The lock, TAUGHT manifests, receipts, rows, ATTEMPTS argv
  and produced_paths are all generated with these names.
- The owner grant is reworded. Its caps are unchanged (20000 records, 4 CPU-hours, 3 cycles, each literally in
  the verbatim), and it says "one official MAIN run on nine listed worlds".
- W's held errors per world differ from the base: 1-2 S2 and 1-2 S3 items. Every world still has at least one
  wrong item in each class, so FIXED_1/FIXED_2 stay below DIGIT_NO_CARRY (no ladder tie). W is strictly above NPT
  (0.5625) and every shortcut in every world.
- Between old_end and final, W forgets one OLD fact on every MAIN world, and a second fact on w_main_4 and
  w_main_8. Mean OLD drop is 0.1528 (> max_drop 0.10), with t-based upper bound 0.1869. The old_drop guard FAILS
  honestly. The primary passes (LCB 0.2723 > 0.05) and old_acq passes (0.5).
- REPORT.md first line is the generated headline (reference calculator, corpus/_build/scores_doc.py):

  `FAIL | earned NONE (attempted E3) | held exact own-EOS 85.4% [S2+ determined, n=144] vs best shortcut DIGIT_NO_CARRY 50.0% | G 29.2pp | OLD final 84.7% (drop 15.3pp) | 9 MAIN worlds | best of k=1`

  The FAIL earns NONE, as in the corpus README interpretation 11 (C00b). No suffix applies: no arm or shortcut
  reaches the candidate, there is no amendment, prior or dev shadow, and the ladder has no WEAK_COMPARATOR.
- The committed CLAIM.json carries honest prediction scores: P2 is a miss, observed 0.1528 outside [0, 0.08]. Its
  findings_none_reason is filled. `MFC claim` regenerates CLAIM.json anyway.

The process is the base's: lock, cycle 0, learner, technical, cross-family review before the roster, dev,
SOURCE_LOCK ack, a fresh rehearsal, one official attempt, REPORT. There are 3 test cycles, within the cap.

## Why it must be accepted
An honest FAIL is first-class (I10, SPEC 3.3: "Report it."). The guard is the locked guard (SPEC 9.5
old_drop). World names are free ids, provided the roles are disjoint and the rehearsal world is fresh
(CYC-03). min_worlds is met (9 >= 8). Records total about 10400, within the 20000 grant and the lock budget.

## Commands (repo root `repo/`, timeline git)
- `gate --stage lock minifly_corpus_base_dev_v1_20261101`
- `gate --stage roster minifly_corpus_base_dev_v1_20261101`
- `gate --stage official minifly_corpus_base_dev_v1_20261101`
- `score minifly_corpus_base_dev_v1_20261101 --role official`
- `claim minifly_corpus_base_dev_v1_20261101`
- `gate --stage publish minifly_corpus_base_dev_v1_20261101`

## Author's uncertainty
- I could not read C00a's actual command list (EXPECTED.json is forbidden). I used lock, roster and official
  gates, then score official, claim and publish. I left out `canary --seed c0ffee` because I cannot predict its
  operator set on a FAIL delivery from the allowed sources.
- The headline is computed by the corpus reference calculator, not by the checker. If the checker words a
  guard-only FAIL differently (for example `earned` other than NONE), CLM-05 would fire. That would be a
  disagreement on headline wording, not a defect.
