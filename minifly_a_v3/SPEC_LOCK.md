# Package A V3-CLAUDE — SPEC_LOCK (frozen before any science world)

**Status.** A new, separately named Package A implementation by Claude from the verified V3 causal-gate scaffold
(`input/MINIFLY_MUSE_A_V3_CAUSAL_GATE_20260929.zip`, sha256 `c667ab51…`). **Not** Muse's V2 runner, not a resume of
Muse's run and not an audit of it; Muse's files are unavailable and are not a prerequisite for this version
(`INTAKE.md`). This spec supersedes `SPEC_DRAFT.md` (kept unchanged as the record for the first technical receipts in
`results/technical/`). Revisions relative to the draft, from the pre-lock review
(`CLAUDE_A_V3_PRELOCK_FEEDBACK_20260929`, commit reviewed `555ba9c`): Z2_rand is now a **yoked** dose diagnostic; P
concentration diagnostics are recorded; the P floor is stated precisely; the roster, statistics, execution plan and
audit scope are fixed below. No candidate equation or constant was changed; no technical score was used.

## 0. Common platform

* Fixture, branches, probes, timing: V3 scaffold `REFERENCE_SOURCE/MINIFLY_THREE_MECHANISM_ROUND_20260928/` —
  unmodified; `common_platform.run_fourstore_life` drives every life. Its `branch_allows` is the corrected explicit
  five-branch map. The frozen historical `run_science.py` is never executed.
* **Birth.** Every Full151 store (4 per bank; R arms have 2 banks = 8 stores) is created by its **own call** to
  `portable_birth.canonical_fresh_native()` before any conversion, clone, byte or teaching. Raw and canonical B
  digests of every store are recorded; all stores must carry canonical B `32a3726c…`.
* **Per-record write ledger.** Every native value teach performed by any store records
  `(branch, record, bank, store, write_flag_passed_to_native, applied_raw_alpha_L1)`, plus the signed-interface
  coefficients for signed arms. The independent auditor (`src/audit_a.py`) re-derives the expected write decision
  of every `(record, branch, bank, store)` from the sealed fixture and a **literal** branch matrix, never importing
  the runner or `branch_allows`, and also checks the V3 aggregate gate (`causal_branch_gate.py`).
* A permitted value write is exactly the platform's: fact records → stores 0–3, relation records → stores 0–1.
  "No-write" branches disable only that native value write; bytes, answers, timing and every unsupervised mechanism
  state update stay identical (this is the protocol's causal-branch rule).

## R family — two banks: shared FE0 (4 stores) + private CONTENT (4 stores)

Shared bank: native FE0 stores. Private bank: the frozen `content_model.VisibleContextBrain` (last-four-visible-byte
address). Both banks see identical bytes/answers/clock. The shared bank always follows the causal branch permission
`A`. Arms differ only in the private bank's write rule.

**Readout (all R arms).** `u_j = v_S,j / s_S + v_P,j / s_P`; facts emit argmax over 4 channels, relations use
`u_1 − u_0` and the fixed L/R organ. `s_S, s_P` are fixed once by **R0 calibration** on worlds `190101–190108`:
W-branch life to the end of the old stage (record 335), then read-only values of the 16 taught old fact cues and 12
taught old relation cues at `old_end_s`; `s_b = sqrt(mean_{world,cue,j}(v_b,j − mean_k v_b,k)^2)` over all four
channels (r2_spec §2 formula). Label-blind: no correctness enters. Both must be finite and ≥ 1e-8.

**R0** — private write = `A`. **R1** — private write = `A ∧ (novelty > θ)`. Novelty is computed after the cue and
before the answer byte from the private bank's pre-answer KC code `x` (the native `pending_x`, identical across the
four private stores): `novelty = 1 − max_{m ∈ buffer} Jaccard(x, x_m)` over a FIFO buffer of the last
`K = 24` presented pre-answer private codes (empty buffer → 1). The buffer is updated on **every** presented record in
every branch regardless of write permission, so novelty depends on bytes only. `θ` = median novelty over all 336
old-stage records of the 8 calibration lives (label-blind; `K=24` ≈ the number of old records in 1 h of model time).
**R1_rand** (dose diagnostic) — within each branch and stratum `(stage, domain, arrived answer)`, exactly R1's
realised count of private writes among permitted records, placed by SHA256 rank
`A3-R1RAND-v1|world|branch|stage|domain|answer|record`; reads only R1's committed per-record gate ledger counts.

**Signed teaching interface** (R0_signed, R3, R3_randtarget). For a private store with teacher coefficients
`(b, s)`: with `W(π)` the native write at punishment `π` (one `EvoLearner.event` on a clone with `plastic=True`) and
`S_n` the native non-plastic event on the live state,
`state ← S_n + b·(W(0) − S_n) + s·(W(1) − W(0))` for fast (both columns) and slow; adaptation, elapsed time and
counters are the native non-plastic event's (identical in all three evaluations; asserted). At `b=1, s∈{0,1}` this
is exactly the native binary write (tested ≤ 1e-12); at `b=s=0` it writes nothing. `A=false` ⇒ plain native
non-plastic event. Relation records: stores 2–3 non-plastic.
* **R0_signed**: `b = 1, s_j = r_j = 1 − y_j` (full native target through the signed interface).
* **R3**: `b = 0, s_j = p_j − y_j` — the private store learns only the shared bank's residual: `p` = softmax over
  active channels of the shared bank's **cached pre-answer** values `v_S,j / s_S` (temperature 1), `y` one-hot of the
  arrived answer. `s_j = r_j − r̂_j` with predicted punishment `r̂_j = 1 − p_j`. Zero residual ⇒ zero private write;
  residual sign reversal reverses the teacher-driven write.
* **R3_randtarget** (direction diagnostic): `b = 0, s = (p − y)[π]`, `π` a SHA256-keyed derangement of the active
  channels (`A3-R3RT-v1|world|branch|record`): same residual multiset and L1 dose, wrong channel correspondence.

## Z family — one bank, 4 native stores, coordinate-local durable-write gate

Writable coordinates: KCs with alpha teaching rows (`|T[:,2:4]| > 0`), as in the frozen Z1 adapter. Each store
carries a load `L_k ∈ [0,1]` per writable coordinate (birth 0). At each teach the load decays over model time,
`L ← L·exp(−Δt/τ_L)`, `τ_L = 86,400 s` (fixture day). On a permitted write the native event gives raw alpha `u_k`
and its native slow share; the frozen `Z1Learner.event_with_gate` certificate moves the gated-away slow share into
fast (total alpha and sign preserved). Gate (pre-event state only):
`c_k = 1[u_k · slow_k^pre < 0]` (the write conflicts with the durable memory's sign),
`g_k = (1 − L_k)(1 − c_k)`. After the write, for coordinates with `u_k ≠ 0`: `L_k ← L_k + κ·g_eff,k·(1 − L_k)`,
`κ = 0.2` (the frozen Z1 activity increment). No cue identity, label, future relevance or extra address.
* **Z0_resource**: identical state/updates with `g_eff = 1` (native split; the load is carried and never used).
* **Z2**: `g_eff = g`.
* **Z2_rand** (yoked dose diagnostic, outside the family): at every permitted write it permutes **its own** Z2 gate
  `g` within each `kc_side × sign(u)` bucket (SHA256 seed `A3-Z2RAND-v2|world|branch|record|store|bucket`), then
  matches exactly the **paired Z2 arm's realised gated slow L1** for the same world, branch, record, store and bucket,
  read from the committed Z2 receipt: `μ·πg` if over, `πg + λ(1 − πg)` if under. A paired target larger than this
  store's bucket capacity `Σ|slow_k|`, a nonzero target in an empty bucket, or a missing paired event is a
  qualification/scientific failure (`ZDoseNotRepresentable`), never clipped. Z2_rand therefore holds the realised
  cross-arm durable-write dose fixed while randomising its placement; its own native slow share and load differ.

## P family — one bank, 4 native stores, fixed PN→KC partners, mutable existing weights

The CSR support (`indices`, `indptr`) is the canonical birth's for life (same arrays, read-only, digest-checked).
Update clock: once per record inside `teach` (after the native value event), in **every branch** (unsupervised,
not a value write), from the pre-answer PN activity `a = p[pn_type_index]` (FE0 `p` read at the answer byte, before
it is fed) and pre-answer KC code `x` (`pending_x`). The answer identity and the write flag are not read. Probes
never teach, so they never update B. Weights in relative units `v_ik = w_ik / m_k`, `m_k` = newborn mean in-weight
of KC k. Step `ε = 0.05` (≈ 1 / expected activations per KC per life: active fraction 0.0354 × 600 records ≈ 21).
* **P1** Hebbian: `Δv = ε a_i x_k`.
* **P2** Oja: `Δv = ε x_k (a_i − x_k v_ik)` — explicit weight-dependent stabilising term `−ε x_k² v_ik`.
* **P4** timing-difference: traces `e_pre ∈ R^302`, `e_post ∈ R^5177` decay with `τ_P = 330 s` (two record
  intervals) between updates; `Δv = ε (e_pre,i x_k − a_i e_post,k)` (pre-before-post potentiates, post-before-pre
  depresses), then `e_pre ← e_pre + a`, `e_post ← e_post + x`. First update is zero; reversing two events flips it.
* After Δ: `v ← max(v + Δv, 1e-9)` (a floor on the **relative** weight before projection, not a bound on physical
  B), `w = v·m_k`, then per-KC rescale so `Σ_i w_ik` equals the newborn sum (fixed budget); physical weights may be
  positive values below 1e-9. New B is a new CSR sharing the fixed support (copy-on-write; clones never alias a mutation).
* **P0**: computes and logs the same traces/Δ bookkeeping (P1 form) and never installs it; B stays canonical.
* **Concentration diagnostics** (all P arms, store 0, after update 336 and 600): edges with relative weight < 1e-3
  and < 1e-6, per-KC fan-in participation ratio `(Σw)²/Σw²` quantiles, KCs with one input carrying > 90% of the
  budget, and the KC activation-count distribution. Descriptive only. `ε = 0.05` is retained as declared (a
  fixed-gain screen); a KC-frequency-dependent gain would be a different (homeostatic) variant and is not used. A
  null or negative result rejects this instantiation on this assay, not Hebbian/Oja/timing plasticity in general.

## P3

**Not implemented.** Kept visible as technically unqualified/unresolved (Muse's reported runtime failure is
unverified; no receipt exists). No P3 or P3_shuffle arm, comparison or replacement. The program-wide family stays
`m = 26`; P3's two comparisons are unavailable.

## Roster (ARM_ROSTER.json)

Candidates **R1, R3, Z2, P1, P2, P4**; primary controls R0 (for R1), R0_signed (R3), Z0_resource (Z2), P0 (P1, P2,
P4); diagnostics outside the family R1_rand, R3_randtarget, Z2_rand. **P3 and P3_shuffle are NOT_INSTANTIATED**
(historical failure is testimony, no recovered receipt); no replacement. 13 science arms × 64 worlds = 832 receipts.

## Endpoints and statistics (unchanged shared contract)

Science worlds exactly `190001–190064` (sampling unit: world). Per world at the final checkpoint:
`E1 = acc(W, old_fact canonical) − acc(N_old_fact, …)`, `E3 = acc(W, old_relation_heldout) − acc(N_old_rel, …)`.
Package-A contrasts: `R1−R0`, `R3−R0_signed`, `Z2−Z0_resource`, `P1−P0`, `P2−P0`, `P4−P0`, each on ΔE1 and ΔE3 =
**12 of the shared m = 26**; P3's two are reported **unavailable** (the family is not recast as 22 or 24).
Two-sided simultaneous 95% Student-t with Bonferroni `t_{63, 1−0.05/52}`; Hoeffding bound on [−2, 2] if a paired
variance is exactly zero. An improvement claim needs a lower bound > 0. Also reported: absolute W/N accuracies, E2
forms, new-cohort learning and taught-cue guards, diagnostic differences (unadjusted), mechanism diagnostics and
resources. No tuning, replacement, omission or added world after any score is seen. Package B's V2 E3 remains
invalid and is not combined with these results into a joint claim.

## Execution plan and failure handling

`src/drive_science.py` (4 workers) runs only after SOURCE_LOCK.json verifies and RESOURCE_BUDGET.json carries an
explicit approval. Receipts are write-once; R1_rand and Z2_rand for a world run after R1/Z2 for that world. Resume
after an interruption runs only uncommitted world-arms under the same lock. The first failed world-arm or budget
breach stops scheduling; the actual error and committed count are recorded; no automatic retry, replacement world
or reduced roster. Receipts are committed and pushed in batches of 16 (append-only).

## Audit scope (what is and is not established)

* `src/audit_a.py` (every receipt): literal per-(record, branch, bank, store) write decisions from the sealed fixture,
  the V3 aggregate gate, separate canonical births, probe roster, first-response timing, R gate/novelty/branch
  invariance, R1_rand strata re-derived, signed coefficients recomputed from logged pre-answer values, Z bucket
  accounting, **Z2_rand dose checked against the paired Z2 receipt** (external reference), P support/positivity and
  branch-identical unsupervised updates. It checks logged quantities; it does not reconstruct every coordinate.
* `src/audit_replay.py` (independent reconstruction from the spec, driving the frozen native model without importing
  the implementation): W-branch store-0 replay of every Z event (conflicts, native/gated slow L1, target, incl. the
  Z2_rand yoke), every P update (pre-norm Δ, installed change, max/min weight, concentration checkpoints), R novelty
  for every record and R3's shared pre-answer values and residuals. Run on all 13 final technical receipts and, in the
  final science audit, on a predeclared sample: worlds 190001–190004 for every arm.
* Deliberate tamper cases (`tests/test_tamper.py`) are a regression battery, not exhaustive validation.
