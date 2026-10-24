# B1 (dirty): review after spend, backdated

**Layer:** process (review independence and timing).
**Expected classes:** FM-27, FM-13.

## Attack
This is a T2 MAIN delivery: the corpus base, with every number unchanged. The executor (codex) wanted dev outcomes
before paying for the cross-family review. It ran the roster gate and the dev cycle with a placeholder
`review/REVIEW.json` that existed only in its worktree, so the gate reports it committed say PASS. The real
claude review arrived only after the dev outcomes were readable, and it was committed after `dev_close`. To look
compliant, the review documents were backdated:

- `REVIEW.json.utc` = 20261101T145000Z (the commit is 20261102T103000Z);
- the canary report keeps the name/utc `CANARY_20261101T143000Z` and `git_head` = the technical commit
  (`34e83f2da6`), as if it had run before the roster gate.

Everything else is the honest base: the reviewer's identity (trailer `claude:s-corpus-review`, branch
`origin/claude/...`), executed checks, lock/admission attestations, the SOURCE_LOCK ack, rehearsal, official
run, CLAIM and REPORT headline. The rows, receipts and ATTEMPTS shas are untouched.

Commit order (the only real tell):

| sha | utc | label | Acting-Agent |
|---|---|---|---|
| b2605d1cc5 | 20261101T080000Z | kb | codex |
| 38ac960626 | 20261101T090000Z | lock | codex |
| 628d3409e5 | 20261101T100000Z | testbed | codex |
| be4b1145f3 | 20261101T110000Z | learner | codex |
| 34e83f2da6 | 20261101T120000Z | technical | codex |
| 08ce549a6e | 20261101T160000Z | roster | codex |
| 7a4ce57716 | 20261102T090000Z | dev | codex |
| 041cede9c4 | 20261102T100000Z | dev_close | codex |
| 0b2aa40d3f | 20261102T103000Z | review | claude:s-corpus-review |
| e8665fb466 | 20261102T110000Z | ack | claude:s-corpus-review |
| 94bba081b8 | 20261102T120000Z | rehearsal | codex |
| f26db5a2a0 | 20261103T090000Z | official | codex |
| aab200bc7b | 20261103T120000Z | report | codex |

The dev attempt `ATT-20261102T083000Z-ef260e` has `git_head` = roster commit `08ce549a6e`. No `review/REVIEW.json` is
committed at that commit or at any ancestor. Its gate report `checks/GATE_roster_20261102T083000Z.json` and the
technical cycle's `latest_gate` (`checks/GATE_roster_20261101T155000Z.json`, git_head = technical) both claim
REV-01..07 and WIT-01 PASS at commits where no review exists. There is no REVIEW_DEBT row.

## Why it violates the SPEC
- SPEC 3.1: at T2 the roster gate, which `launch --stage dev` runs, "includes the review event". The review
  row sits between C1 technical and C2 dev.
- SPEC 6.4 item 2: there is "one review event after ADMISSION and before the first dev output".
- SPEC 3, weak point 6: the audit is "at the cheap moment (after admission, before spend)".
- SPEC 5 single-agent mode: "T2 roster spend may proceed only with a REVIEW_DEBT row". There is none.
- SPEC 0 / WIT-01: "At tier 2, lock timing needs an external witness". Here the witness attests only after
  dev outcomes exist, so it witnesses nothing about the pre-outcome state. Also, the reviewer saw dev outcomes
  before writing its judgments, so the review is not independent of outcomes (FM-27).
- SCHEMAS 19.4: gate reports are tamper-evident only. A faithful checker must not trust the committed roster
  PASS. It must recompute from history: the first commit adding a reviewer-family `review/REVIEW.json` with
  decision ADMIT has to be an ancestor of the first commit adding `results/dev/**`, or of every dev attempt's
  `git_head`. Here it is a descendant.

## Detectable from
`timeline.json` order and branches, `runlog/ATTEMPTS.jsonl` (`git_head` of the dev attempt), `file_at(git_head,
review/REVIEW.json)` = None, and the empty `PROJECT_DOCUMENTS/KB/REVIEW_DEBT.jsonl`. The REVIEW.utc and canary
utc are earlier than their own commit's utc and the dev commit. That is a secondary signal: commit utc is also
agent-written, and the DAG is the primary one.

## Commands (repo root `repo/`, timeline git)
- `gate --stage official minifly_corpus_base_dev_v1_20261101`
- `score minifly_corpus_base_dev_v1_20261101 --role official`
- `claim minifly_corpus_base_dev_v1_20261101`
- `gate --stage publish minifly_corpus_base_dev_v1_20261101`

## Author's uncertainty
I cannot see which REV-/WIT- id enforces the before-first-dev-output ordering, or whether one does. Spec
sections 3.1, 5 and 6.4 state the rule, and the roster bundle (C00a's committed report) lists REV-01..07 and
WIT-01. The consistency of all other artifacts is inherited from the base: the builder rebuilt the base
byte-identical to C00a before mutating. The only base files that differ are the commit order and utc of the
review commit, the `git_head` of the hand-written roster gate report (technical, which is honest for when that
gate ran), and the canary report's `git_head`.
