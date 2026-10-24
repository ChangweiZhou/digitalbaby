# Knowledge ledger: findings, predictions, incidents, errata

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`. The ledgers are `PROJECT_DOCUMENTS/KB/*.jsonl`, one JSON
object per line, `merge=union`, append-only (I8; KB-02 and GOV-APPEND show only added lines). Field lists and one
example row per ledger: `tools/minifly_check/SCHEMAS.md` section 18. Seed rows and their provenance:
`PROJECT_DOCUMENTS/KB/README.md`.

Append rows only through the checker, which validates them (KB-01):

```
MFC ledger append --ledger {FINDINGS|INCIDENTS|USER_DECISIONS|REVIEW_DEBT|CANARY_COMMITMENTS} --row ROW.json
```

DELIVERY_LEDGER rows come only from `MFC claim`, and CONFIRM_LEDGER rows only from `MFC seal reveal`. Never edit or
delete a line. A correction is a new row that refers to the old one. Identical lines from a union merge dedupe; the
same id with different content is FAIL.

## 1. Row kinds (FINDINGS)

| kind | Who appends | When | refers_to |
|---|---|---|---|
| finding | the delivering author | at publish, 0-3 per delivery (CLM-11), status proposed (seed rows carry the digest's statuses) | null |
| activation | the other agent family only (KB-03) | the source is a T2 delivery with worlds_role MAIN or SEALED_CONFIRM and no open debt (KB-04, DEBT-02), after re-deriving the effect | the F-id |
| reconciliation | the other agent family only (KB-03) | re-derivation of a seed or disputed row; carries the status the re-derivation supports | the F-id |
| narrowing | anyone, with no second agent needed | evidence that the scope or effect is smaller (a stronger shortcut, a failed transfer rule) | the F-id |
| supersession | anyone | a newer finding replaces this one | the F-id |
| retraction | anyone | the finding does not hold | the F-id |
| transfer | the author of the delivery whose prediction cited it | at publish, one per cited finding per scored prediction (section 6) | the F-id |

- A finding without conditions.best_shortcut is a claim without its comparator, so CLM-11 refuses it. Fill
  conditions completely and give the source as `path:line@sha`.
- A real narrowing row to imitate: `F-20261024-claude-1` (narrows F-UF-15 with FIXED_1) in `FINDINGS.jsonl`.

## 2. Fold rules

- The current status of a finding is the most restrictive status among the finding row and every row whose
  refers_to is its id, in this order: proposed/provisional < active < unverifiable_in_repo < narrowed < superseded <
  retracted. The fold is order-independent, which is what makes union merges across branches safe.
- An activation never lifts a narrowing. Once a finding is narrowed it stays at least narrowed. New evidence goes in a
  new finding that cites the old one.
- A narrowed or superseded finding travels with the caveat of its narrowing row. Quote it only with that caveat and,
  in a pointer, with the row id (PTR-04).
- `MFC kb status` prints the folded statuses and caveats. Never fold by eye from the first row.

## 3. R-levels (computed by the checker; never declared)

| Level | Meaning for a finding | Meaning for a check |
|---|---|---|
| R0 | Recorded | Registered |
| R1 | Re-derived from source by a non-author (an activation with evidence) | Catches its own fixture |
| R2 | Holds under a re-derived metric or a changed presentation | Catches a variant fixture |
| R3 | A registered prediction citing it beat its base rate on a later delivery | Catches a sealed canary or a held-out incident |
| R4 | R3 on a different task family or substrate | — |

Only R3 and above are called 'transferable'. Every seed row starts at R0.

## 4. Activation by the other family

Do this only when you are the other agent family than the finding's author_agent. Your identity is your
`Acting-Agent:` trailer plus your branch namespace; a same-family session or subagent never counts.

1. Check the source delivery: T2, worlds_role MAIN or SEALED_CONFIRM in its CLAIM, and no open REVIEW_DEBT row. DEV
   and PILOT findings stay proposed. A T1 delivery cannot have findings activated.
2. Re-derive the effect yourself from the cited `path:line@sha`. Rerun the scoring from committed rows, or recompute
   the number with your own script. An activation without an executed re-derivation is the ledger's version of an
   UNVERIFIED review finding: do not write it.
3. Append an activation row whose evidence names your script and its output. If what you find is smaller, append a
   narrowing row instead.
   Commit it with your `Acting-Agent:` trailer and push it to your own namespace (origin/<your family>/*). KB-03 and the
   `kb status` fold take the activator from that commit (`common.commit_family`, spec 20.15), never from the row's
   author_agent or the git author: the trailer must name your family, an origin/<your family>/* branch must contain the
   commit, and the row's author_agent must equal that family. An unpushed commit gives MISSING; a trailer that its own
   namespace does not contain is identity_mismatch (FAIL), and the fold ignores the row (corpus S71, S73).
4. Legacy sources (LEARNED-BANK included) enter as provisional and wait for an executing review recorded in KB/RETRO
   (spec 17) before anyone activates them.

## 5. Prediction scoring

Registered in ENDPOINT_LOCK.predictions for the primary and each guard only (LOCK-11): point plus interval90 and/or
p_pass with p in [0.02, 0.98], a base_rate_forecast {value, source}, and basis_findings (F-ids or 'none'). `MFC claim`
scores them at publish into predictions_scored (CLM-10). You never write the scores.

With observation y, central 90% interval [l, u] (so alpha = 0.10), forecast probability p and outcome o (1 if the
target passed, else 0):

```
hit             = (l <= y <= u)
interval_score  = (u - l) + (2/alpha) * (l - y) * [y < l] + (2/alpha) * (y - u) * [y > u]    # Winkler; lower is better; 2/alpha = 20
brier           = (p - o)^2
unopened target = 'unresolved'   (no hit, no scores)
```

Worked example (the scored entry of `templates/CLAIM.example.json`): interval90 [0.35, 0.76], p_pass 0.70, observed
held exact 0.625 and the primary passed. Then hit is true, the interval score is 0.76 - 0.35 = 0.41, and Brier is
(0.70 - 1)^2 = 0.09. Had it come out at 0.30, the interval score would be 0.41 + 20 x 0.05 = 1.41. Had the primary
failed, Brier would be 0.49.

`MFC kb score` reports, per task family:
- hit rate and Brier skill BSS = 1 - Brier(forecasts) / Brier(base rate), where the base rate is the Laplace estimate
  (s + 1) / (n + 2) from s passes in n earlier scored v1 deliveries in DELIVERY_LEDGER;
- this only where the base rate lies in [0.1, 0.9] and at least 5 scored deliveries exist, otherwise INSUFFICIENT_N or
  DEGENERATE_BASE (the 53 legacy seed rows carry primary_gate 'legacy' and contribute nothing);
- per-finding transfer counts.

These are KB diagnostics. They are never an agent score, a kit score or a progress metric (I11), so nothing rewards
predicting pessimistically.

## 6. Transfer rows

For each scored prediction (not 'unresolved'), append one transfer row to every finding in its basis_findings:
`{prediction_id, delivery, ok|failed}` with kind transfer and refers_to the F-id. Use ok only when the prediction beat
its base_rate_forecast, which is the R3 criterion: for p_pass, a lower Brier than the base rate's; for an
interval-only prediction, a hit. Otherwise use failed. A finding with 2 or more failed transfers and negative skill
gets a narrowing row (anyone appends it).

## 7. Incident to check

1. Report. Append an INCIDENTS row with kind report: fm_class (from `FAILURE_CLASSES.json`), delivery, symptom,
   evidence `path:line@sha`, found_by_agent, found_utc, caught_at_discovery and checker_sha_at_discovery. For
   caught_at_discovery, run the gate that should have caught the defect with the committed checker and copy
   checker_sha256 from that GATE report (for a legacy delivery, which `gate` refuses, use `MFC retro`).
   heldout_eligible is true only under the rule in point 4.
2. Fix. If the checker missed it, the fix is a new or changed check, a corpus fixture that reproduces the incident and
   an EXPECTED.json entry whose must_fire names the check. That is a tightening through `minifly-governance` (GOV-REV,
   GOV-MONO).
3. Close. Append a row with kind closure, the same id, check_id (present in RULES.json) and fixture_id (whose
   EXPECTED.must_fire includes check_id) (KB-05). Prose, a skill sentence or a memo never closes an incident.
4. Held-out recall counts only incidents first recorded after the v1 freeze, whose fixture someone other than the
   check author wrote and committed by hash (a CANARY_COMMITMENTS row with n = 1) before any check was written. See
   `minifly-governance` references/HELDOUT_CANARIES.md section 2.

The seed incidents INC-SEED-01..23 predate the freeze (heldout_eligible false). Their closure rows are pending corpus
fixtures and do not count until KB-05 passes.

## 8. Errata

ERRATA.jsonl sits in the delivery and exists only if a published verdict changes. ERR-01 requires a row whenever
CLAIM.primary_gate changes between commits after the first publish.

| Classification | When | Allowed direction |
|---|---|---|
| BUG_FIX | The published gate came from a defect in measurement or checking that a corpus fixture reproduces (for example a carry bug in expected_hex: MET-02, corpus C01) | Any. An upgrade needs fixture_id and reviewer_signoff from the other family |
| POST_HOC | Anything else decided after seeing outcomes: a new analysis, threshold, metric, pool, stratum or world selection | Never an upgrade |

- Treat any change whose new primary_gate is PASS as an upgrade. When unsure whether a change upgrades, treat it as one.
- Commit the ERRATA row together with the regenerated CLAIM. The superseded CLAIM stays in git history; the original
  VERDICT, data and adopted pointers are never edited (I9).
- A POST_HOC idea worth testing is a new protocol on fresh worlds, never a re-label of the old result.

## 9. One-shot KB backtest (Phase 2)

The question is whether the KB improves forecasts over the objective contract alone. Run it once, not as a ritual.

1. `MFC kb backtest-snapshot --as-of 2026-10-15T00:00Z --out DIR` writes a git archive of the last commit at or
   before T.
2. A fresh session that has not seen the later deliveries reads only DIR plus redacted design briefs of the post-T
   deliveries, with results, REPORT and SUMMARY stripped by someone else. It writes two forecast files: contract-only
   (the objective contract alone) and KB-informed.
3. `MFC kb backtest-score --contract CONTRACT_FORECASTS.json --kb KB_FORECASTS.json [--outcomes OUTCOMES.json]
   [--as-of T] [--out FILE]` (both forecast files are required; exit 2 without them) computes the Brier skill of
   each against the Laplace base rate, and
   KB value = Brier(contract-only) - Brier(KB) with a paired bootstrap CI. Record the result as
   `PROJECT_DOCUMENTS/KB/RETRO/BACKTEST_<date>.json`.
4. The result is always labelled LEAK_RISK, because the container holds the full history and the session could have
   seen it. It is a falsifiable early signal, never a measured KB value.
