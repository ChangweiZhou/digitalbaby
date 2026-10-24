# Review protocol: REVIEW.json, executed checks, canary, review debt

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`. Field types and enums: `tools/minifly_check/SCHEMAS.md`
section 13. Start from `tools/minifly_check/templates/REVIEW.template.json`, and fill every `__FILL__` from something you
executed or read at the handed-off sha. The roster gate (`MFC gate --stage roster <dir>`, T2) runs REV-01..07 and
WIT-01 against your file.

## 1. REVIEW.json walk-through, in working order

1. **Identity.** Fill delivery, reviewer_agent (your family), reviewer_session, reviewer_branch (your namespace branch,
   for example `origin/claude/review-<delivery>`), author_agent (from ENDPOINT_LOCK.author_agent) and utc.
2. **What you reviewed.** reviewed_endpoint_lock_sha256 and admission_sha256 are sha256 values of the committed file
   bytes at the sha you were handed:

   ```
   git show <sha>:<dir>/ENDPOINT_LOCK.json | sha256sum
   git show <sha>:<dir>/ADMISSION.json | sha256sum
   ```

3. **Canary seed.** Read the lock, then choose the seed (section 3). Do this before running anything else, so that
   nothing you see later can shape the seed.
4. **executed_checks.** One entry per check (section 2): {id, kind, script, script_sha256, cmd, output, output_sha256,
   cpu_cap_s, result, compared_to, match}. compared_to and match apply to kinds that compare against a committed
   artifact (rerun_admission, trace_number, replay). REV-02 requires both for rerun_admission (REV-03 reads match)
   and, for any other kind, a match that is present must be a boolean.
5. **canary.** {seed, n, detected, report}, copied from the CANARY report that `MFC canary` wrote.
6. **cheapest_shortcut_attempt.** {description, script, passed_locked_rule}, the same run as the shortcut_attempt
   entry. passed_locked_rule true forces decision BLOCK (REV-04).
7. **fm_judgments.** One line per triggered FM class (SKILL.md section 5).
8. **findings.** {id, fm_ids, claim, severity, executed, status}. Leave reconciled_by_author to the author.
   countersigned_by_reviewer and ruling_user_decision start empty.
9. **decision.** ADMIT only when no BLOCKER or MAJOR finding is open and the shortcut did not pass; otherwise BLOCK.
10. **Commit and hand off.** Commit `review/` (REVIEW.json, scripts, outputs) and `checks/CANARY_<utc>.json` on your
    branch with your `Acting-Agent:` trailer, push to your namespace, and send `<commit sha> <path>` only. The executor
    merges your commit before the first dev output (REV-06, WIT-01).

Later updates go in new commits of yours: source_lock_acks [{sha, utc, note}] before official, seal for
SEALED_CONFIRM (SEAL_PROTOCOL.md), and postrun {replay, utc} after the official run. The author's only edits are
reconciled_by_author and the reconciliation status of each finding.

At T2, once the REPORT exists: fm22_prose_adjudications [{sentence, verdict, note}] (REV-09, spec 20.20; SKILL.md
section 8a). Copy each sentence from the CLM-17P WARN row of the author's latest publish or dry publish GATE report
(or from the current text; matching ignores markup, case and spacing), give verdict ok or overclaim with a note that
cites CLAIM or VERDICT, and add any sentence the lint missed. An overclaim verdict blocks publish until the author
rewords or removes the sentence. Free-prose overclaim is a judgment check with a lint aid, not a mechanical guarantee:
the lint flags, you decide.

## 2. Executed-check recipes

Common rules for every kind:
- Placement. Put the script and its output in the delivery's `review/` directory, for example
  `<dir>/review/x1_rerun_admission.py` and `<dir>/review/x1_rerun_admission.out.json`. In REVIEW.json write the
  path form SCHEMAS.md uses (delivery-relative, such as `review/x1_rerun_admission.py`). Inside scripts and in cmd use
  only repo-relative paths, run from the repo root. Never write an absolute path (`/home/...`, `/tmp/...`) into a
  script, a cmd or an output: pass the scratch directory as an argument (`"$SCRATCH"`) and keep it out of the output.
- Scratch copy. Run delivery code only in a scratch copy of the delivery at the handed-off sha, never in the working
  tree, and write nothing under `<dir>/results/` (RUN-01 would flag an unlaunched output there):

  ```
  SCRATCH=$(mktemp -d)
  git archive <sha> <dir> | tar -x -C "$SCRATCH"
  ```

- CPU cap. Enforce it in the shell and record the same number as cpu_cap_s:

  ```
  ( ulimit -t <cpu_cap_s>; python3 -I <dir>/review/x1_rerun_admission.py "$SCRATCH" ) > <dir>/review/x1_rerun_admission.out.json
  sha256sum <dir>/review/x1_rerun_admission.py <dir>/review/x1_rerun_admission.out.json
  ```

  Size the cap from measured cost: about twice the original attempt's `cpu_children_user_s + cpu_children_sys_s` in
  `runlog/ATTEMPTS.jsonl` for a rerun, and at most 1800 s in total for a post-run replay (spec 16). A check killed by
  its cap is a result to report ('cap reached'), not a pass.
- Outputs. Write machine-readable JSON with the numbers compared and the verdict of the comparison, generated fresh
  for this delivery. Never copy an output from another review: byte-identical outputs across deliveries fail REV-07.
- Code you may import. Your scripts may import program code from `tools/minifly_check` when a recipe says so. They never
  import the delivery's evaluator code to check itself: that is the common mode the review exists to break (FM-01).

### rerun_admission

Rerun at least one technical world and compare digests.
- Prefer the causal-control arm (causal_control_arm) and the candidate arm on one technical world: the control's
  receipt carries the zero-channel claims (ADM-CTRL-*), and the write counter is executor code you must see working.
- Run the locked runner command for that world and arm in the scratch copy with the runner's `--only-world` and
  `--only-arm` flags.
- Compare with the committed receipt (compared_to = its path): channel_digests at birth, post_first and end,
  writes_by_channel, state_digest_end and realized_dose, plus the OUTPUTS rows for that (world, arm) if any.
- Also rerun one floor config of the testbed grid and compare its FLOOR_ rows. Floor learners are executor-run, and
  this rerun is the mitigation spec 15 relies on.
- Set match true only on exact equality. A mismatch (non-determinism included) needs a finding (REV-03).

### reject_input

Feed a gate or fixture an input it must reject, and show that it does.
- Target the delivery's own plumbing gates (lock.gates) and fixtures first, since the checker's own checks are already
  corpus-tested. v5.1's port gate could not fail: its fixture hard-coded an active fraction of 0.02 while the native
  path ran at 0.0354 (`PROJECT_DOCUMENTS/V51_CLAUDE_AUDIT_RECONCILIATION_20261020/AUDIT.json`, finding A4).
- Build the bad input in the scratch copy: a fixture with the wrong sign, a held prompt inside a feedback-bearing
  stream, a param off by 40%, or a prompt the runner must refuse.
- result is 'rejected' only if the intended gate or check returned false or FAIL for the intended reason. A nonzero
  exit from anything else (MISSING because the scratch copy has no git history, a crash, a usage error) is not a
  rejection. Record it and try again.

### trace_number

Trace one headline or admission number to the line that produces it, and to its coefficients.
- At review time, pick an admission number, for example ADMISSION.input_contrast.ratio, a structure meter or a
  testbed cell. After official, pick a headline number.
- Record the producing `path:line` and the coefficients it uses. Then recompute the number from raw rows or receipts
  with your own code, without importing the delivery's code or the checker's scorer: an independent derivation is the
  point.
- compared_to is the artifact path plus JSON pointer, and match is your recomputation against it (state the tolerance
  if the number is not an exact count).

### shortcut_attempt

Build the cheapest non-learning policy that could pass the locked rule. Success means BLOCK.
- The program ladder (spec 9.2) is already in the comparator, so a ladder member cannot pass by construction (LAD-02,
  RED-10). Look beyond it, at this delivery's split and format: a per-length or per-leading-byte lookup, an operand
  copy with a fixed carry guess, a positional rule fitted to the taught manifest, the arm's own stream cut at a length
  chosen per prompt length.
- Produce the policy's predictions as candidate rows from the TAUGHT and held manifests only, with no learner. Score
  them with the program interfaces (`scorer.score_rows(rows, lock, taught_by_world_arm)`, then
  `decide.evaluate(cells, lock, preconditions)`), so that the rule is the delivery's locked rule. On technical worlds
  decide returns DEV_NOT_EVIDENCE: read its shadow_verdict.
- passed_locked_rule is true if the shadow verdict (or verdict) is PASS. Then set decision BLOCK, file an incident,
  and propose the policy as a new ladder member through governance. A novel shortcut outside the ladder is the most
  valuable thing a review can find.

### oracle_upstream (required when lock.stage_chain is true)

Give each downstream stage a perfect upstream: replace stage k's output by the evaluator's ground truth for it, run
stage k+1 in the scratch copy, and score it. If stage k+1 cannot pass even with an oracle upstream, the chain cannot
test its claim (FM-05). That is a BLOCKER, and the honest exit is CANNOT_TEST_CLAIM_AS_SPECIFIED.

### replay (post-run; SEALED_CONFIRM or adoption)

`MFC review replay-plan <dir> --seed HEX --n 4` selects the histories. For SEALED_CONFIRM the seed is the
seal-derived 'replay' seed; for adoption, choose it after the official outputs are committed. Replay each (world, arm)
with the locked official command in a scratch copy, within the 1800 s total cap. Compute expected_digest (the
committed official rows) and observed_digest (your rows) with one and the same function, the canonical row digest
that REV-08 recomputes. Record them in postrun.replay and as an executed check of kind replay.

## 3. Choosing the canary seed after reading the lock

1. Read the committed ENDPOINT_LOCK at the handed-off sha: arms and roles, zero_channels, gates, ports, exposure
   streams. The seed must postdate this reading, so that the author cannot fit declarations to it.
2. Draw fresh entropy:

   ```
   python3 -I -c 'import secrets; print(secrets.token_hex(8))'
   ```

   Never take a seed from the author or from a message. Never reuse the selftest seed 'c0ffee' or a seed from another
   review.
3. Run `MFC canary <dir> --seed <hex>`. It mutates only a temporary copy; its report is `<dir>/checks/CANARY_<utc>.json`.
4. Commit the report on your branch. Its commit must follow the ENDPOINT_LOCK commit (REV-05). Copy seed, n, detected
   and the report path into REVIEW.canary.
5. If detected < n, REV-05 fails and the roster stays blocked. File the survivor as an incident (fm_class from the
   operator's target). The checker fix goes through governance; GOV-COI means the delivery's author cannot make it
   for its own delivery without a UD.

## 4. REVIEW_DEBT rows and what they block

Record debt when T2 roster spend must proceed and no reviewer of the other family is available, or when the only
review available is same-family:

```
MFC ledger append --ledger REVIEW_DEBT --row ROW.json     # {id, delivery, utc, reason, recorded_by, repaid_by: null}
```

- The delivery is labelled DEV or PILOT. DEBT-01 passes the T2 roster without a valid REVIEW only if an open debt row
  for this delivery exists and open debts program-wide stay below review_debt_max_open (3, draft).
- Open debt blocks: SEALED_CONFIRM, activation of the delivery's findings, adoption and the pointer stage (DEBT-02).
  At 3 or more open debts program-wide, no new T2 roster may start.
- A same-family subagent review is debt, not review, however thorough.
- Repayment: when the other family returns, it reviews the indebted delivery first with a full review event on the
  delivery as it stands. Then append a row with the same id and repaid_by set to the repaying reviewer and their
  REVIEW.json commit. The ledger is append-only: the open row stays and the fold reads it as repaid.
