# Package A V3-CLAUDE — new implementation, technical-qualification specification (DRAFT)

**Status.** A *new, separately named* Package A implementation built by Claude from the verified V3 causal-gate
scaffold (`input/MINIFLY_MUSE_A_V3_CAUSAL_GATE_20260929.zip`, sha256 `c667ab51…`). It is **not** Muse's V2 runner, not
a resume of Muse's run, and not an audit of it: Muse's source, locks, receipts, stop record, P3 failure receipt and
roster amendment were unavailable (`input/CLAUDE_PACKAGE_A_MUSE_INTAKE_20260929.zip`, sha256 `22ef1605…`,
`README_FOR_CLAUDE.md`). The Codex prototype in that archive was read but not adopted; no prototype receipt is used.

**Scope of this stage:** technical qualification on world `190000` (and R readout calibration on the reserved
calibration worlds `190101–190108`) only. **No science world is run and no science `SOURCE_LOCK.json` is written.**
Every constant below is fixed here, before any technical life, from source constants, numerical design or
label-blind calibration statistics — never from behavioural accuracy.

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
* **Z2_rand** (resource/dose diagnostic): its own `g` is permuted within each `kc_side × sign(u)` bucket by a
  SHA256-seeded permutation (`A3-Z2RAND-v1|world|store|event`), then matched exactly to the bucket's gated slow-L1
  `Σ|slow_k| g_k` by the linear adjustment `μ·πg` (if over) or `πg + λ(1 − πg)` (if under); gates stay in [0,1].

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
* After Δ: `v ← max(v + Δv, 1e-9)`, `w = v·m_k`, then per-KC rescale so `Σ_i w_ik` equals the newborn sum (fixed
  budget). New B is a new CSR sharing the fixed support (copy-on-write; clones never alias a mutation).
* **P0**: computes and logs the same traces/Δ bookkeeping (P1 form) and never installs it; B stays canonical.

## P3

**Not implemented.** Kept visible as technically unqualified/unresolved (Muse's reported runtime failure is
unverified; no receipt exists). No P3 or P3_shuffle arm, comparison or replacement. The program-wide family stays
`m = 26`; P3's two comparisons are unavailable.

## Technical exit gate (world 190000; all 13 arms)

Per arm: separate-birth evidence for every store; complete 600-record five-branch life; V3 aggregate gate; independent
per-(record, branch, bank, store) write audit; arm-specific invariants (below); clone isolation via the platform's
read-only probe digests; finite states; resources (wall, peak RSS bytes, receipt bytes). Deliberate mutations must be
rejected, including restoring 288 old-relation writes in `N_old_rel` and a compensating pair of per-record write flips
with unchanged aggregates. Arm-specific: R — shared bank rows equal the literal matrix; private rows ⊆ permission;
R1 gate = `novelty > θ` and novelty identical across branches; R1_rand strata counts = R1's; signed coefficients
recomputed from cached pre-answer values; R0_signed ≈ native (unit test). Z — `L ∈ [0,1]`, gate formula, split
certificate, Z2_rand bucket L1 match. P — support bytes unchanged, per-KC budget preserved, finite bounded weights,
P0 B unchanged, nonzero mechanism change in P1/P2/P4.
