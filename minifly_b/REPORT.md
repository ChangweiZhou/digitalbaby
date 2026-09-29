# MiniFly Package B — science report (T1, T3, S1, S2, S3, S4)

Lock digest `7ba1b8e65a17012d54249a58988e5fd4f0e3117192e171c87c99d4dc518cfd9e` (verified before the run,
by every worker, and again by the analysis and final audit). Roster: 13 arms × 64 science worlds
(`190001–190064`) = **832/832 world-arm receipts committed**, `RUN_STATUS.failure = null`.
Machine-readable evidence: `results/FINAL_METRICS.json` (per-world E1/E3 and every metric, intervals),
`results/FINAL_AUDIT.json`, `results/RESOURCE_REPORT.json`, `results/technical/TECHNICAL_AUDIT.json`,
and the receipts under `results/science/<arm>/<world>.json.gz`.

## 0. Headline — read this first

1. **The E3 axis is invalid in this run, for every variant, because the causal branch was never instantiated.**
   The shared, locked platform function
   `REFERENCE_SOURCE/MINIFLY_THREE_MECHANISM_ROUND_20260928/common_platform.py::branch_allows` returns
   `branch != f"N_{stage}_{domain}"`. For relation records this compares against `"N_old_relation"` /
   `"N_new_relation"`, but the branches are named `"N_old_rel"` / `"N_new_rel"`, so **relation writes are never
   disabled**. In all 832 receipts `N_old_rel` performs all 288 old-relation writes and `N_new_rel` all 144
   new-relation writes; the end-state digests of `W`, `N_old_rel` and `N_new_rel` are identical in 832/832
   receipts, and all 8,320 relation-probe blocks (10 stage × relation-set blocks per receipt; emitted bytes,
   values and option scores) are bit-identical between `W` and `N_old_rel`. Hence **E3 ≡ 0 in every
   world and arm by construction**; the six ΔE3 intervals below are reported for completeness only and carry
   **no information about any mechanism**. They are *not* a scientific null.
   The receipt auditor did not catch this because it derives the expected write ledger from the same
   `branch_allows`; it was found post-analysis when the final audit re-derived the ledger from the
   protocol's stated semantics. The file is locked and shared with Package A, so it was not modified;
   **Package A (same platform file) is presumably affected identically** and should be checked.
2. **E1 is valid** (`N_old_fact` withholds all 768 old-fact writes in 832/832 receipts; `N_new_fact` all 768
   new-fact writes). **No candidate improves E1 over its matched control**: all six ΔE1 simultaneous
   intervals (Bonferroni m=26) include 0; point estimates are −0.036 (T1), −0.028 (T3), 0.000 (S1), −0.022
   (S2), +0.011 (S3), −0.026 (S4).
3. Integrity checks otherwise pass: receipts re-hashed and per-world E1/E3 re-derived independently (0
   mismatches), 7/7 tamper cases rejected, all resources within the pre-science hard budget.

## 1. Status of every assigned variant

Chance: facts 0.25 (4-way), relations and the held-out choice 0.50 (2-way). "E1 present" means the arm's own
E1 = W − N_old_fact on canonical old facts is > 0 (unadjusted 95%); it is retention of directly taught
exact keys, not generalisation.

| Variant | Instantiated | Matched control | Evidence level reached | ΔE1 vs control (m=26) | ΔE3 |
|---|---|---|---|---|---|
| **T1** hierarchical modular static graph | yes, 64/64 worlds (mean 21,200 accepted switches, cross-level floors held) | T0_1 (same G*, same #switches random) | E1 present (0.266); E2 forms above chance; **no advantage over control**; E3 not assessable | −0.0361 [−0.1252, +0.0530] | INVALID (branch not instantiated) |
| **T3** clustered, exact 20% sparse bridges | yes, 64/64 worlds (exact 5,514 bridges) | T0_3 | E1 present (0.275); E2 above chance; **no advantage**; E3 not assessable | −0.0283 [−0.0977, +0.0411] | INVALID |
| **S1** SET-like prune/regrow | yes; 5,248 W-branch partner changes per world | S0 (static canonical, same S state) | E1 present (0.049), W near chance (0.305); **no advantage**; E3 not assessable | +0.0000 [−0.0045, +0.0045] | INVALID |
| **S2** co-activity guided rewiring | yes; 4,438 changes/world (4,228–4,684) | S0 | E1 present (0.027), W near chance (0.280); relation outputs collapse to chance (0.48–0.52); **no advantage**; E3 not assessable | −0.0215 [−0.0582, +0.0152] | INVALID |
| **S3** homeostatic rewiring | yes; 3,156 changes/world (2,884–3,422) | S0 | E1 present (0.060), W near chance (0.319); **no advantage**; E3 not assessable | +0.0107 [−0.0279, +0.0494] | INVALID |
| **S4** sample-then-stabilise | yes; 9,628 tentatives+rollbacks/world, 356 accepts/world | S0 | E1 present (0.022), W near chance (0.276); **no advantage**; E3 not assessable | −0.0264 [−0.0609, +0.0082] | INVALID |

No variant was `NOT_INSTANTIATED` and no world or arm was dropped, replaced, retried or added.

## 2. The 12 Package-B contrasts (of the shared m=26 family)

`t_{63, 1−0.05/52} = 3.2378`. Zero paired variance ⇒ Hoeffding bound on [−2, 2] (±0.9319), per protocol.

| Candidate − control | axis | mean | sd(world) | simultaneous 95% (m=26) | interval type | improvement claim |
|---|---|---|---|---|---|---|
| T1 − T0_1 | ΔE1 | −0.0361 | 0.2201 | [−0.1252, +0.0530] | t | no |
| T1 − T0_1 | ΔE3 | +0.0000 | 0.0000 | [−0.9319, +0.9319] | Hoeffding | n/a — INVALID |
| T3 − T0_3 | ΔE1 | −0.0283 | 0.1714 | [−0.0977, +0.0411] | t | no |
| T3 − T0_3 | ΔE3 | +0.0000 | 0.0000 | [−0.9319, +0.9319] | Hoeffding | n/a — INVALID |
| S1 − S0 | ΔE1 | +0.0000 | 0.0111 | [−0.0045, +0.0045] | t | no |
| S1 − S0 | ΔE3 | +0.0000 | 0.0000 | [−0.9319, +0.9319] | Hoeffding | n/a — INVALID |
| S2 − S0 | ΔE1 | −0.0215 | 0.0906 | [−0.0582, +0.0152] | t | no |
| S2 − S0 | ΔE3 | +0.0000 | 0.0000 | [−0.9319, +0.9319] | Hoeffding | n/a — INVALID |
| S3 − S0 | ΔE1 | +0.0107 | 0.0955 | [−0.0279, +0.0494] | t | no |
| S3 − S0 | ΔE3 | +0.0000 | 0.0000 | [−0.9319, +0.9319] | Hoeffding | n/a — INVALID |
| S4 − S0 | ΔE1 | −0.0264 | 0.0854 | [−0.0609, +0.0082] | t | no |
| S4 − S0 | ΔE3 | +0.0000 | 0.0000 | [−0.9319, +0.9319] | Hoeffding | n/a — INVALID |

Per-world paired differences: `FINAL_METRICS.contrasts.<cand>.per_world_dE1|dE3`.

## 3. Absolute W / N accuracies (final checkpoint; mean over 64 worlds, unadjusted 95%)

| Arm | W old_fact | N_old_fact old_fact | E1 | W held-out | N_old_rel held-out¹ | W old-rel taught | W new-rel taught | W new_fact | N_new_fact new_fact | W old_fact @old_end |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 0.521 [0.493, 0.550] | 0.256 [0.239, 0.273] | 0.266 [0.230, 0.301] | 0.698 [0.630, 0.766] | 0.698 [0.630, 0.766] | 0.560 [0.537, 0.583] | 0.589 [0.553, 0.624] | 0.553 [0.519, 0.586] | 0.239 [0.228, 0.250] | 0.566 [0.537, 0.596] |
| T0_1 | 0.553 [0.523, 0.583] | 0.251 [0.233, 0.268] | 0.302 [0.266, 0.338] | 0.695 [0.622, 0.769] | 0.695 [0.622, 0.769] | 0.574 [0.548, 0.600] | 0.609 [0.575, 0.644] | 0.611 [0.579, 0.644] | 0.258 [0.246, 0.270] | 0.569 [0.540, 0.599] |
| T3 | 0.513 [0.481, 0.545] | 0.237 [0.222, 0.252] | 0.275 [0.243, 0.308] | 0.638 [0.570, 0.706] | 0.638 [0.570, 0.706] | 0.577 [0.549, 0.604] | 0.586 [0.553, 0.619] | 0.591 [0.554, 0.627] | 0.241 [0.230, 0.252] | 0.570 [0.540, 0.601] |
| T0_3 | 0.557 [0.526, 0.588] | 0.253 [0.236, 0.269] | 0.304 [0.267, 0.341] | 0.737 [0.669, 0.804] | 0.737 [0.669, 0.804] | 0.569 [0.546, 0.592] | 0.599 [0.564, 0.634] | 0.603 [0.566, 0.639] | 0.249 [0.237, 0.261] | 0.581 [0.551, 0.611] |
| S0 | 0.305 [0.289, 0.321] | 0.256 [0.246, 0.266] | 0.049 [0.033, 0.064] | 0.750 [0.688, 0.812] | 0.750 [0.688, 0.812] | 0.637 [0.609, 0.665] | 0.680 [0.642, 0.718] | 0.302 [0.286, 0.318] | 0.247 [0.239, 0.256] | 0.371 [0.350, 0.392] |
| S1 | 0.305 [0.288, 0.321] | 0.256 [0.246, 0.266] | 0.049 [0.033, 0.064] | 0.732 [0.667, 0.797] | 0.732 [0.667, 0.797] | 0.635 [0.607, 0.664] | 0.677 [0.639, 0.716] | 0.301 [0.285, 0.316] | 0.248 [0.239, 0.257] | 0.371 [0.350, 0.392] |
| S2 | 0.280 [0.265, 0.296] | 0.253 [0.245, 0.261] | 0.027 [0.010, 0.045] | 0.477 [0.411, 0.542] | 0.477 [0.411, 0.542] | 0.495 [0.485, 0.505] | 0.516 [0.495, 0.536] | 0.290 [0.275, 0.305] | 0.256 [0.248, 0.264] | 0.304 [0.287, 0.321] |
| S3 | 0.319 [0.301, 0.338] | 0.260 [0.250, 0.270] | 0.060 [0.040, 0.079] | 0.711 [0.647, 0.775] | 0.711 [0.647, 0.775] | 0.565 [0.543, 0.588] | 0.648 [0.611, 0.686] | 0.355 [0.331, 0.380] | 0.249 [0.239, 0.259] | 0.384 [0.359, 0.408] |
| S4 | 0.276 [0.264, 0.289] | 0.254 [0.244, 0.264] | 0.022 [0.008, 0.037] | 0.724 [0.662, 0.786] | 0.724 [0.662, 0.786] | 0.531 [0.511, 0.551] | 0.542 [0.514, 0.569] | 0.281 [0.267, 0.295] | 0.252 [0.246, 0.258] | 0.344 [0.320, 0.367] |
| Srand_1 | 0.291 [0.275, 0.307] | 0.251 [0.239, 0.263] | 0.040 [0.020, 0.060] | 0.667 [0.594, 0.739] | 0.667 [0.594, 0.739] | 0.562 [0.536, 0.589] | 0.555 [0.530, 0.579] | 0.291 [0.275, 0.307] | 0.253 [0.248, 0.258] | 0.330 [0.311, 0.349] |
| Srand_2 | 0.280 [0.267, 0.293] | 0.243 [0.234, 0.252] | 0.037 [0.023, 0.051] | 0.672 [0.602, 0.742] | 0.672 [0.602, 0.742] | 0.549 [0.528, 0.571] | 0.560 [0.534, 0.586] | 0.287 [0.272, 0.302] | 0.246 [0.238, 0.254] | 0.328 [0.310, 0.346] |
| Srand_3 | 0.292 [0.277, 0.307] | 0.250 [0.239, 0.261] | 0.042 [0.024, 0.060] | 0.745 [0.684, 0.805] | 0.745 [0.684, 0.805] | 0.573 [0.542, 0.604] | 0.542 [0.517, 0.566] | 0.289 [0.274, 0.304] | 0.247 [0.239, 0.255] | 0.330 [0.311, 0.349] |
| Srand_4 | 0.287 [0.274, 0.300] | 0.249 [0.243, 0.256] | 0.038 [0.023, 0.053] | 0.740 [0.674, 0.805] | 0.740 [0.674, 0.805] | 0.533 [0.514, 0.551] | 0.557 [0.527, 0.587] | 0.297 [0.279, 0.315] | 0.254 [0.245, 0.263] | 0.329 [0.311, 0.347] |

¹ `N_old_rel` is write-identical to `W` (Section 0), so this column is not a no-write baseline.

Readings (descriptive, not part of the m=26 family):
* **Taught-cue guards.** T arms and S0/S1/S3 keep taught-relation W above chance (0.53–0.68); S2 drives both
  taught relation sets to chance (0.495, 0.516), and S4 is near chance on taught relations (0.53–0.54). New-fact
  learning (W vs N_new_fact) is present in every arm (largest in T arms, ≈ +0.31 to +0.35; S arms +0.03 to +0.11).
* **Absolute W is low for S arms.** Old-fact W is 0.28–0.32 (chance 0.25) for every S arm including S0; T arms
  and their controls reach 0.51–0.56. S0 and T0_x differ in birth graph (canonical B vs the T2-generator G*),
  so this is a platform-level observation, not a mechanism contrast.
* **Held-out W (0.48–0.75).** Mostly above 0.5, but because no valid no-old-relation-write branch exists the
  data cannot attribute any of it to old relation teaching; it is already ≈ 0.77–0.88 at `old_end` in most
  arms and declines with later learning.
* **S1 ≈ S0.** Per-world ΔE1 sd is only 0.011: pruning/regrowing the weakest-weight edges (no weight
  learning) leaves the readout almost unchanged.

### E2 (never-taught presentation forms of old facts, W, final)

| Arm | spacing | fixed prefix | inner marker | canonical |
|---|---|---|---|---|
| T1 | 0.477 [0.448, 0.505] | 0.508 [0.479, 0.537] | 0.425 [0.399, 0.451] | 0.521 |
| T0_1 | 0.482 [0.455, 0.509] | 0.530 [0.500, 0.561] | 0.460 [0.436, 0.484] | 0.553 |
| T3 | 0.459 [0.430, 0.488] | 0.502 [0.471, 0.533] | 0.403 [0.378, 0.428] | 0.513 |
| T0_3 | 0.504 [0.475, 0.533] | 0.528 [0.500, 0.556] | 0.464 [0.438, 0.490] | 0.557 |
| S0 | 0.295 [0.280, 0.310] | 0.307 [0.290, 0.323] | 0.293 [0.277, 0.309] | 0.305 |
| S1 | 0.293 [0.278, 0.308] | 0.308 [0.290, 0.325] | 0.293 [0.277, 0.309] | 0.305 |
| S2 | 0.281 [0.266, 0.297] | 0.285 [0.271, 0.299] | 0.289 [0.274, 0.304] | 0.280 |
| S3 | 0.299 [0.280, 0.318] | 0.324 [0.305, 0.344] | 0.310 [0.290, 0.329] | 0.319 |
| S4 | 0.264 [0.254, 0.273] | 0.279 [0.265, 0.294] | 0.271 [0.258, 0.283] | 0.276 |

Spacing and prefix preserve CONTENT's effective key (narrow presentation tests); inner-marker changes it.
None of this is E3. Srand rows are in `FINAL_METRICS.absolute`.

## 4. Srand diagnostics (yoked, non-autonomous; outside the 26-test family)

Each Srand_x performed exactly its same-world candidate's realised change count at every
(branch, store, event) — verified for all 256 Srand receipts (`audit_topo`: "Srand realised count/timing").

| Sx − Srand_x | ΔE1 mean [unadjusted 95%] | ΔE3 |
|---|---|---|
| S1 − Srand_1 | +0.0088 [−0.0187, +0.0363] | 0 (invalid instrument) |
| S2 − Srand_2 | −0.0098 [−0.0321, +0.0125] | 0 (invalid instrument) |
| S3 − Srand_3 | +0.0176 [−0.0086, +0.0437] | 0 (invalid instrument) |
| S4 − Srand_4 | −0.0156 [−0.0368, +0.0056] | 0 (invalid instrument) |

No rule's partner choice beats matched-dose random rewiring on E1. Srand arms have lower taught-relation W
than S0 (0.53–0.57 vs 0.64) and S2's guided rule is worse than its random yoke on relations (0.50 vs 0.55).

## 5. Exposure and countermodel audit (`FINAL_AUDIT.json`, re-derived from the sealed fixture)

* 600 teaching records per world; each old fact key taught 12 times; the 6 held-out relation directions never
  appear in any teaching byte string (64/64 worlds); held-out options are order-balanced (3 L / 3 R targets).
* Countermodels on the held-out choice: exact/canonical pair-key table **0.50**, always-first-position
  **0.50**, taught-label frequency **0.50** (every world). Old facts: an exact-key table scores 1.00 (so E1 is
  retention only); modal-label frequency 0.25.
* First-response timing, byte-only mode, probe roster/targets, clone isolation (end digests), four-store
  canonical birth, T graph = pre-science manifest and unchanged for life, S event clock/budget, S W-branch graph
  chain rebuilt from the recorded partner changes, and S0 never changing its graph: all passed for 832/832
  receipts (`analyze_b.py` → `audit_topo.audit_receipt`).
* **Causal-branch write parity: FAILED for the relation branches** (Section 0). `W`, `N_old_fact`,
  `N_new_fact` ledgers match the intended semantics in 832/832 receipts; `N_old_rel` and `N_new_rel` match in
  0/832.
* Independent re-derivation: 0 receipt-SHA256 mismatches, 0 per-world E1/E3 mismatches vs `FINAL_METRICS`.
  Tamper replay on a committed science receipt: answer flip, branch label, world ID, dropped probe, wrong graph
  digest, wrong source hash and altered write ledger — all 7 rejected.

## 6. Resources (`RESOURCE_REPORT.json`)

Host: 4 logical CPUs (Xeon 2.1 GHz, AVX-512), 16.9 GB RAM, Python 3.11.15, numpy 2.2.6, scipy 1.14.1,
numba 0.61.2; 4 workers.

| Quantity | Measured | Hard budget |
|---|---|---|
| Learner core-hours (sum of 832 `learner_life_s`) | 80.3 h | 110 h |
| Max learner life per world-arm | 566 s (Srand_4) | 900 s |
| Peak RSS per worker | 362,958,848 B | 800,000,000 B |
| Science receipts on disk | 163,472,741 B | 1,500,000,000 B |

Per-arm learner seconds: T arms ≈ 19,690 s each; S0 22,115 s; S1–S4 22,805–23,572 s; Srand 23,425–24,388 s
(`FINAL_METRICS.resources`). Science lives ran ≈ 1.8× slower than the technical measurement (≈ 520 s vs 285 s
for S arms) at the same 4-way concurrency; still within budget.

## 7. Limits

* **E3 is not assessed** by this run for any variant (instrument defect, Section 0). Nothing here supports or
  refutes a limited-E3 effect of T1/T3/S1–S4. A valid E3 test requires fixing `branch_allows` (e.g. mapping
  `rel` ↔ `relation`) in the shared platform, a new lock and a full rerun; that is a decision for the package
  owners, not something done post hoc here.
* E1 measures exact-key retention of directly taught facts; an exact-key table would score 1.0.
* All positive E1 values are small for S arms, whose absolute W sits near fact chance.
* One package, one synthetic fixture, 64 worlds; no claim about an integrated learner.
* Srand is non-autonomous (reads its candidate's change counts) and is diagnostic only.

## 8. Host and run notes

* Raw host-generated B digest `872094396ed40c4d4be79423c9d5658eed66ae3d8f83661d2b4fa8f37efb8498`
  (`raw_B_match: false` vs the package reference); per the V2 portability addendum every store of every arm
  instead installed the canonical Full151 B `32a3726c…` (verified in every receipt); the raw digest is kept as a
  diagnostic only.
* The VM computes only while a tool call is executing, and the container is reclaimed when idle. The science
  run therefore spanned **two agent sessions**, with VM pauses and **container restarts**. After each restart
  the driver re-verified the lock (same digest) and resumed; committed receipts were skipped (write-once, never
  rerun or overwritten), and world-arms that were **in flight and uncommitted** at a pause/restart were rerun
  from scratch under the same lock. Receipts were committed and pushed append-only in batches of 16; no
  force-push.
* The driver's active-time counter for the final resumed process was 8.0 h; host wall-clock across sessions is
  not a meaningful total.
* The post-science `audit/final_audit.py` is not in `SOURCE_LOCK.json` (written after the roster completed, it
  changes no locked file and no science output).

## 9. Return bundle

`return_bundle/minifly_b_return_bundle.zip` (committed as `.partNN` pieces below GitHub's per-file limit;
reassemble with `cat minifly_b_return_bundle.zip.part* > minifly_b_return_bundle.zip`) contains
`SPEC_LOCK.md`, `SOURCE_LOCK.json`, `ARM_ROSTER.json`, `GRAPH_MANIFEST.json`, `RESOURCE_BUDGET.json`, `src/`,
`tests/`, `audit/`, `results/technical/` (incl. `TECHNICAL_AUDIT.json`), all 832 science receipts,
`results/FINAL_METRICS.json`, `results/FINAL_AUDIT.json`, `results/RESOURCE_REPORT.json`, and this
`REPORT.md`, with an internal `SHA256SUMS` manifest. Whole-zip and part hashes: `return_bundle/SHA256SUMS.txt`.
