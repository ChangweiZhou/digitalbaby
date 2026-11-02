# MiniFly knowledge base (MRK-1, schema v1)

This directory holds the program's shared, durable state: owner constraints and decisions, research findings,
incidents, failure classes and the delivery, confirmation, review-debt and canary ledgers. The normative spec is
`tools/minifly_check/SPEC.md` (sections 8, 11, 12.3 and 13); field-level schemas live in `tools/minifly_check/SCHEMAS.md`.
Read it at the explore stage with `python3 -I tools/minifly_check/minifly_check.py kb status`.

## Files

| File | What it is | Who writes it |
|---|---|---|
| `CONSTRAINTS.json` | Protected invariants, forbidden native constructs, threshold floors, tier thresholds, default grant caps, cadence and debt limits, required arms, testbed grids (spec 13.1). Currently `DRAFT_PENDING_OWNER_DECISION` with `user_decision: null`. Protected. | Mr. Zhou decides; an agent edits it only through governance with a USER_DECISIONS row (GOV-UD, GOV-NUM). |
| `USER_DECISIONS.jsonl` | Owner decisions: verbatim words, session, channel, scope, restated numbers. Empty: no owner decision has been recorded yet (UD-0001 is pending). Protected. | Agents transcribe Mr. Zhou's verbatim words with `ledger append`; a row is never invented or paraphrased. |
| `FINDINGS.jsonl` | Unit facts (`finding` rows) and later `activation`, `narrowing`, `supersession`, `retraction`, `transfer` and `reconciliation` rows. Current status and R-level are folds over the rows. | The delivering author proposes 0-3 findings; only the other agent family activates (KB-03, KB-04); anyone may narrow or retract by appending. |
| `INCIDENTS.jsonl` | `report` rows (a defect, its evidence, whether the frozen checker caught it) and `closure` rows (`refers_to` the report, plus a check id and a corpus fixture). An incident closes only when KB-05 validates its closure. | Whoever finds the defect files the report; closures go through governance. |
| `FAILURE_CLASSES.json` | FM-01..FM-30: title, one-sentence definition, check ids (spec 18 and 12.2), fixture ids (spec 12.2), judgment question, closest archive classes (FC01..FC19), incident ids. | Governance. Reviewers and held-out canary authors read it. |
| `DELIVERY_LEDGER.jsonl` | One row per delivery: level earned, primary gate, worlds role, `opened_held_E3`, objective metric. Feeds the E3 cadence and Laplace base rates. | `minifly_check.py claim` for v1 deliveries; legacy rows are census transcriptions. |
| `CONFIRM_LEDGER.jsonl` | Confirmation attempts per lineage family with the alpha allocated (spec 10). Empty. | `seal reveal`. |
| `REVIEW_DEBT.jsonl` | Review debts recorded in single-agent mode and their repayment (spec 5). Empty. | The delivering agent records a debt; the other family repays it. |
| `CANARY_COMMITMENTS.jsonl` | Hash commitments of sealed held-out canary batches (spec 12.3). Empty. | `heldout seal`, run by a non-tool-author. |
| `GOVERNANCE/` | Cross-agent governance review records `REVIEW_<id>.json`. | The reviewer from the other family. |
| `RETRO/` | `RETRO_<date>.json`, `HELDOUT_<date>.json`, `BACKTEST_<date>.json`: annotations and scores, never verdicts. | `retro`, `heldout score`, `kb backtest-score`. |
| `AGENTS_POINTER.md` | Pointer lines for Mr. Zhou to merge into their own untracked root AGENTS.md. Protected. | Governance. |
| `ANALYSIS_20261024/` | The agent-generated history analyses the kit was designed from. Inputs, not verified facts (see its README). | Frozen. |

## Rules

- **Append-only.** Every `*.jsonl` ledger here is append-only (invariant I8; KB-02, GOV-APPEND). Never edit, reorder,
  reformat or delete a line. Corrections are new rows: a narrowing or retraction for a finding, a closure for an
  incident, an erratum for a verdict. Failed or wrong rows stay.
- **Union merge.** `.gitattributes` sets `PROJECT_DOCUMENTS/KB/*.jsonl merge=union`, so appends made on different
  branches merge without conflicts. This is why every row is one self-contained JSON object on its own line, ending
  with a newline, and why ids must be unique: readers drop byte-identical duplicate lines and reject two different
  rows with the same id.
- **One writer path.** Append with `minifly_check.py ledger append --ledger <NAME> --row ROW.json`, which validates the
  schema and assigns ids (`F-<yyyymmdd>-<agent>-<n>`, `UD-<yyyymmdd>-<n>`, and so on).
- **Protected.** `CONSTRAINTS.json`, `USER_DECISIONS.jsonl` and `AGENTS_POINTER.md` change only through governance,
  in a commit range that touches no delivery directory.
- **No fabricated owner decisions.** Only Mr. Zhou's own words become a USER_DECISIONS row.

## Seed rows (2026-10-24): provisional, pending reconciliation by the other agent

The initial rows were transcribed by Claude from the digests in `ANALYSIS_20261024/` (row field `delivery: "seed"`;
`author_agent: "claude"`, or `found_by_agent: "claude-digest"` for incident reports). They are **provisional**,
start at R0, and wait for reconciliation by Codex: re-derivation from source, then activation, narrowing or
retraction by appended rows. A seed row is not an activated finding and not an owner decision.

- `FINDINGS.jsonl`: `F-UF-01`..`F-UF-29` from the understand digest and `F-ARCH-01`..`F-ARCH-12` from the archive
  digest (archive facts are training data for held-out purposes). Statuses follow the digest and the work-package
  instructions: `F-UF-02` is `unverifiable_in_repo`; `F-UF-15` is narrowed by `F-20261024-claude-1` (FIXED_1 10.645%
  vs STOP 11.263% on NEW; on OLD FIXED_1 0.977% > STOP 0.326%), so quote P036's +22.10pp only with that caveat;
  `F-UF-03` is narrowed by `F-20261024-claude-2` (held items previously inspected); the `F-UF-16` row records the
  digest's `active` with caveat `uninformative manipulation`, but no activation row exists, so the kit's fold
  (`MFC kb status`) treats it as `provisional` with that caveat, and so should every reader; `F-UF-25` is dev only;
  everything else is `provisional`. `digest_status` keeps the digest's own wording.
- Finding ids: `F-UF-xx` (understand digest) and `F-ARCH-xx` (archive digest) are seed id forms. SPEC 8 names
  `F-<yyyymmdd>-<agent>-<n>` and seed `F-UF-xx`; `F-ARCH-xx` is an integration extension accepted by KB-01
  (tools/minifly_check/SCHEMAS.md, INTEGRATION_LOG.md section D). New rows always use `F-<yyyymmdd>-<agent>-<n>`.
- `INCIDENTS.jsonl`: one report row per SPOT-CHECKED FM instance (`INC-SEED-01`..`23`), with
  `caught_at_discovery: false` and `heldout_eligible: false` (they predate the freeze and never count as held-out).
  Closure rows (`INC-SEED-xx-CLn`) exist only where spec 12.2 lists a fixture built from that instance's source and
  names a check that fixture must fire. They carry a `pending_corpus` note and are not effective until the fixture
  is in the corpus and KB-05 passes. Reports whose fixture names no must-fire check stay open.
- `DELIVERY_LEDGER.jsonl`: one legacy row per census catalog row P001-P053, with `level_earned` and `primary_gate`
  set to `legacy`, the catalog's free text in `legacy_level_text`, and `opened_held_E3` false when that text has no
  `E3` and `unknown` otherwise (legacy rows never claim true). Fields the catalog does not state are null.
- Every seed row lists `source_paths` / `evidence_paths` checked against git at the cited commit (HEAD `f92365e4`, or
  `origin/charming-head` `f33d8dee` for LEARNED-BANK), with a `verification` field. Only path existence and line
  ranges were checked; the content was not re-derived.

Archive classes map to the closest FM classes as follows (also stored per FM in `FAILURE_CLASSES.json`):
FC01 FM-02/01; FC02 FM-14; FC03 FM-08/15; FC04 FM-09/28; FC05 FM-07/22; FC06 FM-04/06; FC07 FM-12; FC08 FM-01;
FC09 FM-03/02; FC10 FM-21/04; FC11 FM-04; FC12 FM-24/26; FC13 FM-26/25; FC14 FM-23/22; FC15 FM-22/23; FC16 FM-22/07;
FC17 FM-18/17; FC18 FM-05/16/21; FC19 FM-25/27.
