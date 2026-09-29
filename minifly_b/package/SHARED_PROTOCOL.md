# Shared scientific and audit contract for the two remote packages

This document is identical in Package A and Package B. Its aim is a **single bounded mechanism comparison**, not another sequence of input-bridge patches. Both packages use the same `REFERENCE_SOURCE/MINIFLY_THREE_MECHANISM_ROUND_20260928/fixture.py` and the same continuing four-output-store byte-task scaffold. The fixture supplies 600 teaching records: 16 old directly taught fact keys and 12 old taught relation directions, then a simulated day, 16 new fact keys and 6 new relation directions, then another simulated day. Six old relation directions are withheld from all feedback. The output is emitted before the answer byte; the old relation's W/N contrast is measured on read-only clones after intervening learning.

**V2 portability rule:** `BIRTH_PORTABILITY_ADDENDUM.md` is binding. Every
candidate/control/branch uses the same exact canonical Full151 newborn B,
installed by `portable_birth.canonical_fresh_native()` before its first event.
The raw host-generated B digest is retained as a technical diagnostic, not
adopted as a new baseline. The non-B birth fingerprint must match exactly.

## Evidence and controls

- **E1:** final response to the 16 directly taught old fact keys. Exact-key recall can explain success; call it retention, not generalisation.
- **E2:** the existing never-taught spacing, fixed-prefix and inner-marker forms. The first two preserve CONTENT's effective key and are narrow presentation tests; the third changes it. None is E3.
- **Limited E3:** the six never-reinforced old relation directions after the new cohort and delay. An exact taught-pair table has no answer, but a learned one-bit symbol-class rule can solve them. A positive result is limited to this synthetic relation.
- **Causal branches:** W, `N_old_fact`, `N_old_rel`, `N_new_fact`, `N_new_rel` must see identical bytes, labels, feedback events and model time; only the indicated native write is disabled. The candidate and its matched control must have the same output organ and available external information.
- Record exposure lists and check raw bytes, effective addresses, labels, first-response timing, trained/held-out separation, balanced options, state clone isolation and exact/canonical-key plus frequency/position countermodels. Do not infer E3 from E1/E2 or from a depressed no-write baseline.

## Fixed roster and statistics

Use technical world `190000`. Worlds `190101–190108` are reserved only for any label-blind readout calibration needed by a family; calibration must not use held-out relation labels or behavioural accuracy. **Science worlds are exactly `190001–190064` in both packages.** The world is the sampling unit. Use the same fixture implementation and source hash in both packages; do not substitute a new generator, re-use existing R2/Z1/T2 trajectories or import their outcomes into parameter selection.

For each candidate `c` and its predeclared resource-matched control `b`, compute per world at the final checkpoint:

```text
E1_x = accuracy_x(W, old_fact_canonical) − accuracy_x(N_old_fact, old_fact_canonical)
E3_x = accuracy_x(W, old_relation_heldout) − accuracy_x(N_old_rel, old_relation_heldout)
ΔE1 = E1_c − E1_b
ΔE3 = E3_c − E3_b
```

There are **26 primary comparisons** (13 candidates × E1/E3) across the two packages. Report each paired-world mean and a two-sided simultaneous 95% Student-*t* interval using Bonferroni `m=26` (`t_(63, 1−0.05/(2×26))`). If paired-world variance is exactly zero, use a bounded Hoeffding interval for the known `[-2,2]` contrast rather than a zero-width victory interval. An improvement claim for an axis requires its interval's lower bound to exceed zero; otherwise report magnitude and uncertainty, including negative and null estimates. Also show candidate/control absolute W and N accuracies, taught-cue guards, per-item failures, E2, new learning and resource use. An E3 difference alone does not warrant a useful-output or integrated-core claim if W remains near chance; report the absolute W level and its uncertainty separately.

The 26 tests form **one predeclared family** even though the agents execute on separate machines. Do not use package-local `m=14` or `m=12`, inspect the other package's scores to change hypotheses, drop a weak variant, reweight endpoints, or add worlds. These intervals identify evidence for a mechanism effect relative to its own matched control; they do not by themselves prove the full project target. A candidate that improves E1 and E3 in separate architectures has **not** produced one integrated learner.

## Freeze, technical qualification and failure handling

1. Before science, implement **all assigned candidates** (or record a concrete `NOT_INSTANTIATED` blocker) and write a candidate-by-candidate `SPEC_LOCK.md`: exact equations, event timing, initialization, random streams, gains/threshold origins, state/edge/write budgets, clone/restart semantics and every matched control. Fix endpoints, world roster and `m=26` simultaneously. Parameter choices may use source literature, numerical constraints and the reserved technical/calibration worlds, **not** science outcomes.
2. Run unit and full-life **technical** checks on world `190000` for each candidate/control. Test actual write dose, graph/weight identity, finite states, no future labels, causal branch parity, read-only probes, deterministic clone/restore, source hashes and deliberate receipt corruptions. A technical score is not used to tune the candidate.
3. Measure each complete technical life for wall time, peak resident bytes and receipt size on the **remote** hardware. Set a written package hard budget before science; if 64 worlds cannot fit, mark the package `RESOURCE_NOT_FEASIBLE` and return the estimate. Do not quietly reduce n or omit expensive variants. No GPU speed claim is assumed; any accelerated implementation must pass deterministic parity with the CPU technical life.
4. Lock every implementation, auditor, fixed source file, package environment, resource budget and world roster in a new relative-path `SOURCE_LOCK.json`. Do not edit a locked scientific file after the first science receipt. Commit each world-arm atomically and never overwrite a committed receipt. An interrupted process may resume only uncommitted world-arms with the same lock; a scientific integrity failure stops the run and is reported with the actual error and committed count. No replacement worlds or automatic retry of a scientific failure.
5. Independently rederive fixture exposure, score arithmetic and source/receipt hashes without importing the science reducer as the sole oracle. Reject tampered answer, branch, world ID, dropped probe, wrong graph and wrong source hash. Analyse only after the full roster and all required audits pass.

Do not execute the included historical `run_science.py`; it is reference code for R2/Z1/T2, not a launcher for these packages. Use only package-local code/results paths and portable memory units (bytes); macOS `ru_maxrss` conventions must not be assumed on Linux.

## Required return bundle

Return `SPEC_LOCK.md`, `SOURCE_LOCK.json`, the candidate/control source and tests, `TECHNICAL_AUDIT.json`, all committed compact world-arm receipts, `FINAL_AUDIT.json`, machine-readable per-world E1/E3 metrics and intervals, `RESOURCE_REPORT.json`, and `REPORT.md`. `REPORT.md` must list the status of **every assigned variant**, its E0–E3 evidence level, controls, absolute W/N scores, uncertainty, actual exposure/countermodel audit and limits. Zip the return bundle with a hash manifest. Do not substitute screenshots, an LLM narrative, full-event CSV dumps or a score table lacking receipts/audit evidence.
