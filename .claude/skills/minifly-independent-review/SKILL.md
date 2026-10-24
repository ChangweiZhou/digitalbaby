---
name: minifly-independent-review
description: "Use when you review a MiniFly delivery you did not author (required once per tier-2 delivery, after ADMISSION.json exists and before roster spend, plus a lock acknowledgement before the official run, seal duties and a post-run replay for sealed confirmations and adoptions); when you receive a review and must reconcile it; when you find or suspect a defect, shortcut or reward-hacking route in any delivery, old or new; and when the other agent is unavailable and review debt must be recorded. Covers executed checks (rerun admit.py, feed an input that must be rejected, trace a headline number, cheapest-shortcut attempt, replay), the reviewer-seeded canary run, REVIEW.json, the seal nonce, reconciliation statuses with the V51 legacy map, counter-signing refuted blockers, REVIEW_DEBT, and why a same-family subagent review never counts as independent."
---

> **If you only read this.** You are a reviewer only if you belong to the other agent family, shown by your `Acting-Agent:` trailer and your own branch namespace. A same-family subagent review is review debt.
> Execute rather than read: rerun admission, feed an input that must be rejected, trace a number, try the cheapest shortcut, and run `MFC canary` with a seed you choose after reading the lock.
> A finding without an executed check is UNVERIFIED and carries no weight, but an UNVERIFIED BLOCKER still pauses the stage until you execute it.
> A shortcut that passes the locked rule forces decision BLOCK. A REFUTED BLOCKER stays open until you countersign it or the owner rules.
> With no other family available there is no review, only REVIEW_DEBT. Debt blocks sealing, findings activation, adoption and pointers.

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`, run from the repository root. Proceed only on exit 0.

- Normative spec: `tools/minifly_check/SPEC.md` (sections 5, 6.4, 7.4 REV/WIT/DEBT/CAN, 10 and 12). Field-level schema
  of REVIEW.json and the seal files: `tools/minifly_check/SCHEMAS.md` section 13. Template:
  `tools/minifly_check/templates/REVIEW.template.json`.
- References: [references/REVIEW_PROTOCOL.md](references/REVIEW_PROTOCOL.md) (REVIEW.json walk-through, recipes for each
  executed-check kind, canary seed, REVIEW_DEBT) and [references/SEAL_PROTOCOL.md](references/SEAL_PROTOCOL.md) (the
  seal steps and their failure handling).
- Failure classes and their judgment questions: `PROJECT_DOCUMENTS/KB/FAILURE_CLASSES.json`.

## 1. Preconditions

- You belong to the other family from the delivery's author (Codex vs Claude). REV-01 checks lock.author_agent and the
  trailers, then requires that the commit introducing REVIEW.json carries an `Acting-Agent:` trailer of your family and
  sits in a remote branch of your namespace (`origin/codex/*` or `origin/claude/*`). Identity never comes from the git
  author: Codex commits as 'Pencilbard'.
- Check before you start:

  ```
  git log --format='%H %(trailers:key=Acting-Agent,valueonly,separator=%x2C)' -- <dir>
  python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["author_agent"])' <dir>/ENDPOINT_LOCK.json
  ```

- One function derives every identity (spec 20.15, checker 1.0.1): `common.commit_family` reads the commit's
  `Acting-Agent:` trailer AND the origin/<family>/* branch that contains it. Trailer and namespace must agree: a
  claude trailer on a commit that only origin/codex/* contains (or only a branch of another remote such as
  upstream/claude/*) is identity_mismatch. REV-01 (your review commit and the author's lock commit), WIT-01 (your
  attestation and SOURCE_LOCK ack commits), SEAL-02 and `seal reveal` (your SEAL_COMMIT and SEAL_REVEAL commits), KB-03 (activations),
  GOV-REV and DEBT-02 (a repayment counts only when the repaying REVIEW.json was introduced by your commit) all use it.
  So commit with your trailer and push to your namespace before anyone runs a gate on it.
- Never fork or merge an author tip that carries a commit with YOUR family's trailer that you did not write. Once any
  origin/<your family>/* branch contains such a commit, the namespaces can no longer tell it apart from yours: a fork
  is enough, not only a merge (verifier probe C6, spec 20.15 residual). Before you branch from the author, list the
  range with the command above and stop if a commit claims your family; record it as an incident instead.
- A review by a same-family session or subagent is not a review. Record it as REVIEW_DEBT (section 9).
- Work from what was handed to you, `<commit sha> <path>`, and read the artifacts at that sha. A message or summary from
  the author is an input to check, not evidence.
- Commit only on your own branch, with `Acting-Agent: codex` or `Acting-Agent: claude:<session>`, and push only to your
  namespace.

## 2. Timing

| Event | When | What you commit | Checked by |
|---|---|---|---|
| Review event (one per T2 delivery) | after ADMISSION.json is committed and before the first dev output | `review/REVIEW.json`, review scripts and outputs, the CANARY report | REV-01..07, WIT-01 |
| SOURCE_LOCK ack | before the official run, after the final `lock build` | an entry in source_lock_acks, after a short delta review of the closure diff | WIT-01 (official) |
| Seal (SEALED_CONFIRM only) | after the final locks, around the owner's GO | `review/SEAL_COMMIT.json`, then `review/SEAL_REVEAL.json` | SEAL-01..05 |
| Post-run replay (SEALED_CONFIRM or adoption) | after the official outputs exist | postrun.replay digests | REV-08 |

- WIT-01: reviewed_endpoint_lock_sha256 equals the current ENDPOINT_LOCK. The commit that introduces your attestation
  must be in your namespace and be an ancestor of the first dev output commit. For official, source_lock_acks must
  hold the final SOURCE_LOCK sha in a commit of yours that is an ancestor of the first official output commit.
- The executor must merge your commit, not cherry-pick it: a cherry-pick creates a new sha outside your namespace, which
  fails REV-01 and WIT-01. Say so in the hand-off.
- Delta review for the ack: compare the closure maps of the reviewed and the final SOURCE_LOCK, then read every
  changed native or runner file (closure files may lie outside `<dir>`, in tools/ or files.shared).

  ```
  git diff <reviewed-sha> <final-sha> -- <dir>/SOURCE_LOCK.json      # which closure paths changed
  git diff <reviewed-sha> <final-sha> -- <changed closure paths>
  ```

## 3. Minimum executed checks

Recipes, caps and file placement: REVIEW_PROTOCOL.md section 2. REV-02 requires the kinds rerun_admission,
reject_input, trace_number and shortcut_attempt, plus oracle_upstream when lock.stage_chain is true. Each needs a
committed script under `<dir>/review/`, its cmd, its output, matching script and output sha256 values and a cpu_cap_s.

- rerun_admission: rerun at least one technical world and compare digests. A mismatch needs a finding (REV-03).
- reject_input: feed a gate or a fixture an input it must reject. Record a rejection only when the intended check
  fails for the intended reason.
- trace_number: trace one headline or admission number to its producing line and coefficients, recomputed by your own
  code.
- shortcut_attempt: build the cheapest non-learning policy that could pass the locked rule. If it passes, decision is
  BLOCK (REV-04).
- oracle_upstream: when a stage chain exists, give the next stage a perfect upstream and check that it can pass.
- Write nothing under `<dir>/results/` and change no delivery file: run in a scratch copy (RUN-01 and AMD-05 would
  flag it). Copy no output from another review: byte-identical outputs across deliveries fail REV-07. The padding it
  targets looks like `addition_multimechanism_dev_v1_20261011/cycles/cycle{1,2,3}/AUDIT.md`, three byte-identical
  audits (corpus C27).

## 4. Canary

```
MFC canary <dir> --seed <hex chosen after reading the lock>
```

- Choose the seed yourself, after reading the committed ENDPOINT_LOCK (REVIEW_PROTOCOL.md section 3). The CANARY
  report commit must follow the ENDPOINT_LOCK commit, and canary.detected must equal canary.n (REV-05).
- All applicable mutants must be caught (CAN-02). The unmutated copy must reproduce the original statuses (CAN-01),
  and at least half the operators must apply (CAN-03). Fewer applicable operators point at minimized declarations;
  say so in a finding.
- A surviving mutant is a checker miss. File an INCIDENTS row (`MFC ledger append --ledger INCIDENTS --row ROW.json`)
  and leave the fix to governance. REV-05 keeps the roster blocked meanwhile.
- The operators are public. They test checker regression and declaration minimization, not held-out generalization.

## 5. Judgment questions

Write one line per FM class the design triggers in fm_judgments, answering that class's judgment_question in
`FAILURE_CLASSES.json`. Typical triggers: arms with zero_channels (FM-02, FM-03); exposure streams and held sets
(FM-12); a lineage parent or priors_added (FM-23); a stage chain or a new task (FM-05); custom gates (FM-04). Always
answer these:

- Does the write counter sit at the mutation site? Open the function named by write_counter_site and check that the
  counter increments where the state changes, not in a wrapper that can skip it.
- Does the control cut only its declared path? Compare its zero_channels and declared_changed with the code path.
- Are any priors task-shaped (FM-12, FM-23)?
- Are there supplied-arithmetic hits (LOCK-07 REVIEW_REQUIRED in the roster GATE report)? At T2 each hit must be
  addressed by a REVIEW finding, with an executed check showing whether it reaches the native answer path.

## 6. Findings and reconciliation

- Severity: BLOCKER, MAJOR, MINOR or NOTE.
- A finding without an executed check is UNVERIFIED and carries no weight. An UNVERIFIED BLOCKER still pauses the
  stage until it is executed. Set executed only when a committed script and output support the finding.
- Status, set when the author reconciles: CONFIRMED, CONFIRMED_WITH_CORRECTION, REFUTED, NEW_DEFECT or SCOPE_LIMITED.
- V51 legacy map:
  - CONFIRMED_WITH_QUANTITATIVE_CORRECTION, RISK_CONFIRMED_UNDER_ACTUAL_NATIVE_MASKS and MATCHED_EVENT_SENSITIVITY_SUPPORTS_CALIBRATION_FRAME_DEFECT become CONFIRMED_WITH_CORRECTION.
  - COUNTERFACTUAL_FAILURE_REPRODUCED becomes CONFIRMED.
  - SPECIFICATION_IMPLEMENTATION_MISMATCH and LATENT_INTEGRATION_DEFECT become NEW_DEFECT.
  - CLAUDE_HISTORY_STATEMENT_CORRECTED becomes REFUTED.

  The source statuses are those of `PROJECT_DOCUMENTS/V51_CLAUDE_AUDIT_RECONCILIATION_20261020/AUDIT.json`
  (findings A1-A7). Apply the map when citing them; never rewrite that file.
- The author reconciles every finding before launch: reconciled_by_author {utc, note, evidence} plus a status. No
  BLOCKER or MAJOR may remain open, and decision must be ADMIT (REV-06).
- A REFUTED BLOCKER needs your countersign (countersigned_by_reviewer, set only after you have checked the refuting
  evidence yourself) or an owner ruling (ruling_user_decision). The author cannot clear it alone.

## 7. Seal duties

For SEALED_CONFIRM you hold the entropy. Follow SEAL_PROTOCOL.md step by step.

- Attest both lock shas, then run `MFC seal commit-nonce <dir>`. The nonce goes to a private file outside the repo,
  and only its hash is committed in `review/SEAL_COMMIT.json` on your branch.
- After the owner's GO is committed, run `MFC seal reveal <dir>` and commit `review/SEAL_REVEAL.json`. Never reveal
  before the GO commit, and never show the nonce to anyone before reveal.
- If the nonce is lost, repeat steps 2-4 with a fresh one. No run data is lost.
- Single-agent mode cannot seal. A same-family session cannot hold the nonce; state that, never claim otherwise.

## 8. Post-run replay (SEALED_CONFIRM or adoption)

```
MFC review replay-plan <dir> --seed HEX --n 4
```

- The plan picks (world, arm) histories deterministically from the seed. For SEALED_CONFIRM use the seal-derived seed
  for purpose 'replay' (`MFC seal derive <dir>`). For an adoption without a seal, choose the seed yourself after the
  official outputs are committed.
- Replay each chosen history with the locked command (the runner's `--only-world/--only-arm` flags) in a scratch copy,
  under a CPU cap of at most 30 minutes in total. Record {world, arm, expected_digest, observed_digest} in
  postrun.replay. REV-08 fails on any digest difference.
- Replay closes output fabrication only for SEALED_CONFIRM and adoption. MAIN results without replay stay open to it.
- A resumed official run (`launch --resume`, ATTEMPTS resume_of) carries the executor's own determinism replay of one
  launch-chosen unit in `results/official_replay/` (RUN-04). It shows that the identical lock reproduces a completed
  history; it is executor-run, so it never replaces REV-08. When the run was resumed, include in your plan at least one
  unit from each side of the interruption (from attempt and resume_of in runlog/ATTEMPTS.jsonl), and check the
  runtime closures (`runlog/RUNTIME_CLOSURE_<attempt_id>.json`, RUN-03) against the closure you acked.

## 9. Review debt

- Allowed only for T2 roster spend in single-agent mode, labelled DEV or PILOT (DEBT-01). Record it with
  `MFC ledger append --ledger REVIEW_DEBT --row ROW.json` (REVIEW_PROTOCOL.md section 4).
- A delivery with open debt cannot be SEALED_CONFIRM, cannot have findings activated and fails the pointer stage
  (DEBT-02). It cannot be adopted.
- At 3 or more open debts program-wide (CONSTRAINTS.review_debt_max_open), no new T2 roster may start.
- When the other family returns, repaying debt comes before any new review: a full review event on that delivery,
  then a repayment row with the same id and repaid_by set.

## 10. Reviewing history

- Legacy deliveries get only RETRO annotations and INCIDENTS rows. They are never edited, and no file is written
  inside them.

  ```
  MFC retro <legacy-dir> --out <scratch-dir-outside-every-delivery>
  git show f33d8dee:minifly_learned_bank_v1_20261024/summarize.py      # read other branches; never check them out
  ```

- An executing review of a legacy delivery (such as the Codex review LEARNED-BANK needs before its verdict is cited)
  runs in a scratch copy, and its record goes to `PROJECT_DOCUMENTS/KB/RETRO/`.
- File every defect as an INCIDENTS row with evidence `path:line@sha` and caught_at_discovery. Seed incidents
  predate the freeze and are never held-out.

## 11. The delivering agent's lessons are inputs

Read the author's lessons, then list what they omit. LEARNED-BANK's notes
(origin/charming-head `minifly_learned_bank_v1_20261024/NOTES_FOR_SKILLS_SESSION.md:40`) call `summarize.py` "a
mechanical decision rule". They do not mention:
- that the rule is fail-open: a gate is treated as passing when it `is not False` (`summarize.py:92-93`), and 3 of 4
  worlds count as 'partial' through `len(verdict)` (`summarize.py:99-100`);
- the lock-closure gap: `make_lock.py:12` hand-lists the hashed files (corpus C13b);
- the task's distance from E3: the final task T-PAR is parity, not unseen addition.

The same applies to any summary from the author, including one written for you. A second reader exists because
summaries drift toward the convenient story; LEARNED-BANK's notes say this about themselves.

## Open limits (spec 15)

- In-repo identity is tamper-evident, not tamper-proof: any agent can write any trailer or branch name. The owner's
  spot-check and an optional push ruleset are the only anchors.
- Executor-side instruments stay partly open. Write counters, answer_path labels and floor learners are executor code.
  You read the write site, rerun one floor config and replay histories; that mitigates the risk without removing it.
- Tuning outside `launch` is invisible for non-confirm work. Only SEALED_CONFIRM is immune.
- Witness-aware supplied arithmetic stays open pending the owner's decision on sealed digit relabelling. Your LOCK-07
  findings are the defence meanwhile.
- A causal control that diverges after the first record is a matter of judgment; ADM-CTRL-ISO checks only post_first.
- At T1, commit order is local only (DEV-stamped). WIT-01 and the seal give external order at T2.

## Not required

- No review at T0 or T1, and no per-cycle review files.
- No essay: one line per triggered FM class, and findings only where something was found.
- No re-review of an unchanged closure beyond the delta review at the SOURCE_LOCK ack.
- No outputs copied between deliveries (REV-07 flags them as padding).

## Failure classes covered

FM-01 common-mode auditor; FM-03 confounded intervention; FM-12 exposure leakage; FM-15 test edited after failure;
FM-23 lineage and answer-path drift; FM-25 misremembered facts; FM-27 self or padded review; FM-30 meta-loop Goodhart.
Definitions: `PROJECT_DOCUMENTS/KB/FAILURE_CLASSES.json`.
