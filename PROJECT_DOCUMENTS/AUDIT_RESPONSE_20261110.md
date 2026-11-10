# Response to the E16–E19 source audit of 10 November 2026

Audit: `MINIFLY_E16_E19_HOSTILE_SOURCE_AUDIT_20261110.md` (Codex, c5e2715; source and receipts, nothing run). Response by the coordinating session (Claude), which ran each claim. Review section 9.75 carries the errata. Verdict words: **confirmed** (reproduced), **confirmed, small** (reproduced, measured effect small), **accepted** (true by reading; no measurement needed).

| # | Audit finding | Verdict | Evidence | Action | Status |
|---|---|---|---|---|---|
| 1 | E18 B3 cannot learn its prompt-only embeddings | confirmed | `exp45/AUDIT_RESPONSE_CHECKS.json`: gradient reaches 11 of 45 symbol rows under B3, 45 of 45 under B0–B2 | census by symbol role in every birth and life receipt; `E16_FIXEDCODES=on` control; P18.6 re-read (errata 7) | fixed in harness; control not run |
| 2 | E16-M2 staged 600013 compares different panels | confirmed | receipts: checkpoint zero 25.21 (learner, reference) against 24.58 (backprop on pipeline, oracle); no saved weights for the first two | one-factor reading withdrawn for that seed; repeat of three arms on one panel prepared | needs the go |
| 3 | F7 tally mixes panel versions | confirmed | `exp45/E16_AUDIT_TABLES.md`: corrected 13/15, 2/15, 9/15; original 3/3, 2/3, 3/3 | strata published; repeat of the three original-panel seeds prepared | table done; repeat needs the go |
| 4 | Reversed pairs and taught sub-steps reach the held panels | confirmed, small | 25% reversal overlap (49–52% at birth); 71–73% sub-step exposure; strict re-score moves ratios by at most 0.025 | strict panels scored in every receipt; E2 wording amended | fixed |
| 5 | Suffix metric against whole worked answer | accepted | both are in every receipt | worked-answer gate shown beside the others | done |
| 6 | Drift clock: checkpoint 4 is not the end of composition teaching | confirmed | `drift_weights`: equal weights at two thirds, fresh compositions to three quarters | errata 2; P16M4.3 to be read as net change | done |
| 7 | Replay share is an allocation policy; rows are not tokens | confirmed | admissions 1,769,488 / 1,179,680 / 589,872 at shares 0.25 / 0.5 / 0.75 | errata 3; rows and supervised tokens by origin and family logged per stage | fixed |
| 8 | Phase 4 arms differ in feedback map | confirmed, small | cosine 0.991–0.999 between the two maps' gradients at birth; same alignment with truth | `E16_PROJFB=scaled`; errata 5c | knob done; not run over a life |
| 9 | Phase 4 claims too broad (birth skill, long gaps, compositions) | confirmed | receipts: birth operations 47–60 after block 1; k = 40 at 9–28 | errata 5a, 5b | done |
| 10 | Phase 4 equivalence check is dispatch parity | confirmed | — | hand-written forward and backward, no autograd: 2e-15 | fixed |
| 11 | E18 birth name omits the loss mode; checkpoint skipped on existence | confirmed | source | births refuse the every-byte loss; an existing checkpoint must match its receipt's configuration and hash | fixed |
| 12 | E18 isolation check gates on exit code | confirmed | source | numeric gate, and a relay-kept fixture that must leak (0.148 and 0.095 detected) | fixed |
| 13 | E18 arms differ in several factors | accepted | brief | `E16_HEADS=aux` one-factor control; errata 7 | knob done; not run |
| 14 | E19 rate buffers against module constants | accepted | worker's file, mid-run | assertion to be added at merge; in the addendum | open |
| 15 | E19 gradient equality is narrower than a streaming learner | accepted | — | streaming inference and reset check requested in the addendum | open |
| 16 | E19 decay differs between Stage A and the rule arms | confirmed | my brief | decay-matched backprop birth requested in the addendum; P19.3 marked confounded | open |
| 17 | Receipts expose scores before weights; identity by file name | confirmed | source | weights saved first by rename; receipt records their hash and completion; panel hash and effective configuration recorded | fixed for E16–E18; E19 at merge |
| 18 | Lock overwritten by substring; child exit codes unchecked | confirmed | source | `machine_lock.sh`: exclusive creation, unique token, per-job exit codes; self-tested | fixed |
| 19 | Replay tag rounding; comparator routing by substring | confirmed | source | ambiguous shares refused; rows per batch recorded; summarizer joins on recorded fields and lists mismatches | fixed |

**Also found while checking.** The embedding's start-of-sequence row never receives a gradient in the E17 architectures (it is the table's padding row).

**What the audit did not get wrong.** Nothing I tested. Where the measured effect is small (4, 8) the defect is still real and is fixed.

**Waiting for a decision.** Twelve repeats on the corrected panels (`audit_rerun_lives.sh`, about 2 h 45 min on the Mac): the three original-panel drift seeds, learner and reference; and seeds 400009 and 600013 staged, learner, reference and backprop on the pipeline. Predictions are in `E45_PREDICTIONS.md`.
