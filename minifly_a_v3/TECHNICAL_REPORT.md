# Package A V3-CLAUDE — technical qualification report (final implementation; world 190000; no science)

**What this is.** A new, separately named Package A implementation by Claude from the verified V3 causal-gate
scaffold. **Not** Muse's V2 run, not a resume of it and not an audit of it; Muse's files are unavailable and, per the
pre-lock review, are no longer a prerequisite for this version (`INTAKE.md`). No science world was run. Probe
accuracies in technical receipts were neither computed nor used.

* Final spec: `SPEC_LOCK.md` (supersedes `SPEC_DRAFT.md`). Roster: `ARM_ROSTER.json`.
* Final qualification: `results/technical_final/` (13 receipts + `TECHNICAL_AUDIT.json`, `pass: true`).
* First qualification (SPEC_DRAFT implementation, commit `555ba9c`): `results/technical/`, kept unchanged as
  technical history; it does not qualify the final implementation.
* Proposed budget: `RESOURCE_BUDGET.json` (**unapproved**). `SOURCE_LOCK.json` is **not yet written** — it is
  write-once and includes the budget, so it follows approval.

## Response to the pre-lock review (`CLAUDE_A_V3_PRELOCK_FEEDBACK_20260929`, reviewed commit `555ba9c`)

1. **Roster adopted** without Muse's files: candidates R1, R3, Z2, P1, P2, P4; controls/diagnostics R0, R1_rand,
   R0_signed, R3_randtarget, Z0_resource, Z2_rand, P0. P3/P3_shuffle NOT_INSTANTIATED, no replacement; m=26 kept with
   P3's two slots unavailable. Package B V2 E3 stays invalid.
2. **P not retuned.** `ε = 0.05` kept. My earlier "floor" wording was wrong: `1e-9` floors the *relative* weight
   before per-KC projection, not physical B. New concentration diagnostics (store 0, W branch, end of life; P0 =
   unchanged canonical B): edges with relative weight < 1e-3: P0 0, P1 48, P2 55, P4 2 (of 27,572); < 1e-6: 0 / 2 /
   2 / 2. KCs with one input carrying > 90% of the budget: 8 / 46 / 48 / 7 (of 4,765 multi-input KCs). Median fan-in
   participation ratio 4.40 / 4.35 / 4.35 / 4.40. So concentration is confined to a small set of KCs (47–66 KCs are
   active on every record); no fan-in collapse. Descriptive only.
3. **Z2 low throughput kept as a mechanism property** (`κ = 0.2`, `τ_L = 86,400 s` unchanged). Defined precisely:
   ratio = Σ gated slow L1 / Σ native slow L1 over all W-branch permitted Z events of all four stores. Z2: old stage
   (records 0–335) 2,244.6 / 17,062.6 = 0.1315; new stage (336–599) 1,564.9 / 14,473.0 = 0.1081; whole life
   3,809.5 / 31,535.6 = 0.1208. The withheld share goes to fast storage (total alpha preserved).
   **Z2_rand is now a genuinely yoked dose diagnostic**: it matches the paired Z2 receipt's realised gated slow L1 for
   the same world/branch/record/store/bucket, failing explicitly (`ZDoseNotRepresentable`) if a paired dose exceeds its
   capacity or a paired event is missing. Result: Z2_rand W total 3,809.5 = Z2's (was 4,002.7, +5.07%); no
   unrepresentable event occurred. The auditor checks this against the paired Z2 receipt (external reference).
4. **Audit claims bounded; independent reconstruction added.** `src/audit_replay.py` re-implements the spec's equations
   and drives the frozen native model without importing the implementation. On all 13 final receipts (W branch,
   store 0) it reproduces every Z event (600 per Z arm: conflicts, native/gated slow L1, target, incl. the Z2_rand
   yoke), every P update (600 per P arm: pre-norm Δ, installed change, max/min weight; both concentration
   checkpoints), R novelty for all 600 records of every R arm, and R0_signed/R3/R3_randtarget shared pre-answer values
   and R3 residuals for all 600 records. It also reproduced the first qualification's Z2, P and R receipts. New tamper
   cases: a self-consistent joint change of Z2's gated amount, target and bucket is **accepted by the log audit and
   rejected by the replay**; the same joint change in Z2_rand is rejected against the paired Z2 receipt. The 197
   rejected tamper cases are a regression battery, not exhaustive validation.
5. **Freeze prepared.** Complete closure written before the final lives: `analyze_a.py` (12 available contrasts, m=26,
   P3 unavailable), `drive_science.py` (refuses to start without a verified lock and an approved budget; yoked
   dependencies; write-once; stop on first failure), `lock.py` (package files by MANIFEST, all `src/`/`tests/`,
   spec, roster, budget, calibration, both input archives, final technical receipts, environment).

## Final qualification result

| Check | Result |
|---|---|
| Scaffold `verify_bundle.py` (full) | pass |
| Separate canonical births | 8 per R arm, 4 per Z/P arm, all `32a3726c…` |
| Full 600-record five-branch lives | 13/13 |
| V3 aggregate gate over all receipts | 13/13 |
| Independent per-record write audit (`audit_a.py`) | 13/13 |
| Independent replay reconstruction (`audit_replay.py`) | 13/13 |
| Tamper cases | 197/197 rejected (incl. the replay-only case) |
| Unit tests | 8/8 |
| One source-hash set across receipts; executed closure = current tree | yes; only `src/technical_audit.py` (this report's compiler, never executed by a life) changed afterwards |
| R calibration under final code | reproduced exactly (scales, θ, every per-world value); file unchanged |

## Resources (measured, final lives; see `RESOURCE_BUDGET.json`)

R arms 902–958 s, Z 499–522 s, P 561–574 s per technical life; peak RSS ≤ 268 MB per worker; receipts 0.34–0.85 MB.
64-world estimate 166.6 core-hours ≈ 41.7 h wall at 4 workers; 75 h is a planning allowance extrapolated from
Package B, not a measurement. Proposed hard budget: 90 active wall-hours, 360 core-hours, ≤ 2,400 s per world-arm,
≤ 800 MB per worker, ≤ 1 GB results. Receipts ≈ 410 MB.

## What is needed before science

1. Approve (or amend) `RESOURCE_BUDGET.json`; then `python src/lock.py create` writes the write-once lock.
2. A launch decision. On this host the VM computes only during tool calls, so ~42–75 h of continuous blocking waits
   across sessions are needed; receipts push every 16 and resume never reruns a committed world-arm.
