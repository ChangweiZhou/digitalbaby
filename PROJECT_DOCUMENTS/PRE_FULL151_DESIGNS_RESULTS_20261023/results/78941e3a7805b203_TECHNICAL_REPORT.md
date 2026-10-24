# Package A V3-CLAUDE — technical qualification report (world 190000; no science)

**What this is.** A new, separately named Package A implementation by Claude, built from the verified V3
causal-gate scaffold. **It is not Muse's V2 run, not a resume of it and not an audit of it.** Muse's source, locks,
receipts, stop record, P3 failure receipt and roster amendment were not available (see `INTAKE.md`). The Codex
prototype in the intake archive was read but not adopted; none of its receipts are used. No science world was run
and no science `SOURCE_LOCK.json` exists. Probe accuracies in the technical receipts were not computed or used.

Machine record: `results/technical/TECHNICAL_AUDIT.json` (`pass: true`). Spec: `SPEC_DRAFT.md` (fixed before any
technical life). Calibration: `results/calibration/R_CALIBRATION.json`.

## Result

| Check | Result |
|---|---|
| Scaffold `verify_bundle.py` (full, pinned Python 3.11 / NumPy 2.2.6 / SciPy 1.14.1 / Numba 0.61.2 / pandas 2.2.3) | pass (107 files; raw host B `872094396ed4…`, canonical B installed) |
| Separate `canonical_fresh_native()` call per store | 13 arms: 8 births per R arm, 4 per Z/P arm, all canonical `32a3726c…`, no shared fly objects |
| Full 600-record, five-branch life | 13/13 arms |
| V3 aggregate gate (`causal_branch_gate.py --receipts-dir`) | 13/13 receipts pass |
| Independent per-(record, branch, bank, store) write audit (`src/audit_a.py`; literal matrix, no runner/`branch_allows` import) | 13/13 pass (24,000 ledger rows per R receipt, 12,000 per Z/P receipt) |
| Tamper rejection (`tests/test_tamper.py`) | 194/194 rejected, incl. restored 288 `N_old_rel` old-relation writes (with and without the aggregate), a compensating pair of per-record flips with unchanged aggregates, a cross-domain pair, teacher time, dropped/duplicate row, wrong/shared birth, world ID, probe flag, branch relabel, and arm-specific gate/coefficient/Z/P corruptions |
| Unit tests (`tests/test_units.py`) | 8/8: signed interface = native at targets 0/1 (≤1e-12), zero coefficients write nothing, residual sign reversal, Z0 = native, Z2 gate L1, Z2_rand bucket L1, P support/budget/P0 unchanged/P2 stabilising term, P4 first update zero and order reversal |
| All receipts built from one source-hash set equal to the current tree | yes |

## Arms (technical roster, 13)

R1, R1_rand, R0, R0_signed, R3, R3_randtarget, Z0_resource, Z2, Z2_rand, P0, P1, P2, P4.
**P3 / P3_shuffle: not implemented** — technically unqualified/unresolved, no comparison and no replacement; the
program-wide family stays m=26 with P3's two comparisons unavailable.

R readout calibration (worlds 190101–190108, label-blind): `s_S = 1.49113`, `s_P = 1.34524`; R1 novelty
threshold `θ = 0.83544` (median novelty; 30% of records have novelty 0 because a cue recurs within 24 records).

## Label-free mechanism diagnostics (W branch, world 190000)

* R1 private writes on 41.8% of permitted records (820 store writes vs R0's 1,968); R1_rand exactly matches R1's
  per-stratum counts (820). R3 / R3_randtarget: mean |residual| 0.41, never exactly zero; identical residual
  multiset (derangement). R0_signed uses b=1 everywhere and reproduces R0's private alpha L1 exactly (69,018.75).
* Z2 passes 12.1% of the native slow share (Z2_rand 12.7%, dose-matched per bucket and event); Z0_resource 100%.
* P1/P2/P4 installed weight-change L1 1,661 / 1,716 / 5,439 over the life; P0 0. Novelty, P updates and every
  unsupervised state are identical across the five branches (audited).

## Open items before any source lock (need a decision; nothing was retuned)

1. **P weights collapse in frequently active KCs.** With `ε = 0.05` chosen from the *mean* activation rate, KCs
   active on most records drive some incoming weights to ~1e-8 (P1, P2 by records ~440–460) or to the 1e-9 floor
   (P4 by record 13), while the per-KC budget holds and the maximum weight is unchanged. Bounded and finite, but
   close to winner-take-all for those KCs. Options: keep as declared, or rescale ε by each KC's activation
   frequency. Changing it is a pre-lock design decision on technical-world numerics, not on behaviour.
2. **Z2 throughput declines over life** (13% → 6% of native slow share by life thirds) because the load saturates
   under repeated writes (`κ = 0.2`, `τ_L` = 1 day). Z2 is therefore close to "mostly fast-only" late in life.
3. **Resource estimate for 64 worlds:** 167.8 core-hours (R arms ≈ 16.5 h each, Z ≈ 9.1 h, P ≈ 10.2 h) ≈ 42 h
   wall at 4 workers, measured one arm per worker. Package B's science lives ran ~1.8× slower than its technical
   measurement, so budget ~75 h wall. Peak RSS ≤ 263 MB per worker; receipts 0.33–0.67 MB each.
4. **Roster and multiplicity are provisional** until Muse's amendment/P3 record is obtained or the absence is
   formally accepted; any science would need a new lock, receipt namespace and predeclared execution plan.

## Reproduce

Extract `input/MINIFLY_MUSE_A_V3_CAUSAL_GATE_20260929.zip` into `package/` (not committed; its hash is recorded),
create a Python 3.11 venv from `package/requirements.txt`, then: `python tests/test_units.py`,
`python src/calibrate_r.py`, `python src/drive_technical.py`, `python src/technical_audit.py`.
