# Z1: coordinate-local activity/teaching eligibility on native Full151

**Status:** implementation specification, 2026-09-28. No Z1 scientific run is claimed here. The round selects one variant from each of three different mechanism families. Z1 is the representative of family Z; FE1 is excluded. Use the common **FE0-only, four-output Full151** baseline in the new mixed E1+E3 lifetime, with the same teaching interface, input route and output organ for Z0, Z1 and the random control. Native Full151 value, FE0 and clock state start fresh at time zero. The old V9.2 predictor is not connected to this value route and is not instantiated. Each of the four existing output stores receives the same Z1 rule independently and is fully costed. Do not import FE1 habituation state or code. The separate R2 mechanism is compared with its dual-bank R0; neither dual-bank state nor an R2 result is a prerequisite for Z1.

## Question and evidence boundary

Can the recent activity of an existing KC→alpha-output writable coordinate identify where a *durable* local write should land? This is a new Full151 mechanism test. The older `A_ELIG` result is useful motivation but is **not** this implementation: OMNIBUS10 substituted a normalized multi-frame KC feature for the final feature in a different graph/memory model, using `ELIG_DECAY=.72` per frame and a blank delay before teaching (`elm_omnibus10.py`, `simulate_cell` and `MemoryArm.observe`). Z1 leaves the Full151 sensory code, anatomy, readout, native teacher payload and total alpha update intact. Its only causal change is the per-coordinate fast/slow split after a native teacher arrives.

The eligible coordinate set is fixed at birth:

\[
I=\{i: |T_{i,2}|+|T_{i,3}|>0\},
\]

using the **instantiated Full151** teaching matrix after its frozen `g_support` phenotype. It is not a new fact address or a new KC. Store one float64 `z_i` only for `i∈I`, initialized to zero at the fresh native FE0 birth. The coordinate-to-array map is immutable and included in the fixed digest. The original `m.fast[:,1]`, `m.slow`, `m.adapt`, `m.B`, `m.T`, `m.Q`, feedback and clocks retain their meanings. The trace is global learner state across records and phases; `reset_stream` does not clear it.

## Exact update

Each committed byte calls the inherited `_features(t)` and produces the **pre-byte** native binary KC code `x_i(t)∈{0,1}`. In the actual `byte` path only, before `feed(b,t)`, advance each stored trace from its last absolute Full151 time `s` to `s'=elapsed_base+t` and then incorporate the current code:

\[
z_i^- = e^{-(s'-s)/30\,\mathrm{s}}z_i,\qquad
z_i^+ = z_i^-+0.20\,x_i(t)(1-z_i^-),\quad i\in I.
\]

Set `s=s'`. Reject backwards time and nonfinite state; clamp only roundoff outside `[0,1]`. A rest/flush/stream reset advances by the same exponential to the new absolute native time but has no activity increment. The 30-second decay is the inherited approximate cue-to-outcome window; `0.20` is one fixed, modest activity increment. These are **new Z1 parameters**, not an estimate or port of OMNIBUS10's frame decay. They are locked before any science-world result. Do not retune them using pilot outcomes. A 24-hour gap erases the trace numerically while leaving Full151 value state to decay by its inherited rule.

At `teach(r,t,write=True)`, the outcome byte has already supplied the pre-outcome `pending_x` and the trace update at that same `t`. Let `u_i` be the Full151 event's native signed `rawalpha_i`, including current KC activity, `T`, local DAN payload, event duration and frozen alpha gain. Let `f_i` be the already implemented Full151 `allocation_fraction(...)`, computed from **pre-event** adaptation/fast/slow state and `u_i`. In the parent, `rawslow_i=f_i u_i` and `rawfast_alpha_i=(1-f_i)u_i`. Z1 uses

\[
g_i=z_i^+\in[0,1],\qquad
\operatorname{rawslow}^{Z1}_i=g_i f_i u_i,\qquad
\operatorname{rawfast\_alpha}^{Z1}_i=(1-g_i f_i)u_i.
\]

Coordinates outside `I` retain the native event exactly; their alpha update is already zero. Gamma writes, adaptation, teacher payload and value readout are untouched. Apply the inherited post-write fast and slow exponential decays **after** this split. The identities `rawfast_alpha+rawslow=u`, `sign(rawslow),sign(rawfast_alpha)∈{0,sign(u)}`, and `u=0⇒both=0` must hold for every coordinate. `write=False` runs the inherited nonplastic event and makes neither fast nor slow alpha writes; the trace still follows the observed bytes. No label, task ID, future relevance, old-fact identity or evaluator signal enters `z` or `g`.

The candidate asks about **durable eligibility**. Reassigning the disallowed slow share to fast preserves the native instantaneous alpha update and isolates placement in the slow store from a simple total-teaching reduction. It does change later fast/slow expression, which is the intended intervention. Report actual fast and slow write mass, not only a write count.

## Required comparator arms

| Arm | Event rule | Purpose |
|---|---|---|
| `Z0` | Native Full151 FE0, `rawslow=f_i u_i`, `rawfast=(1-f_i)u_i`; the same trace is maintained but its gate is inert | Same genotype, input, output, teacher and added-state budget baseline |
| `Z1` | Equations above | Coordinate-local time eligibility |
| `Z1_BUDGET_RANDOM` | Same trace state and local teacher, but permute gate placement before applying the same fast/slow split | Test whether the location supplied by activity history matters beyond the amount of slow write |

For the random control, use a deterministic, label-blind counter-based permutation on every plastic teaching event, keyed only by a fixed arm seed and that model's `teach_seen` counter. In each bucket `(kc_side_i, sign(u_i))` among `a_i=|f_i u_i|>0`, permute the current `g_i=z_i^+` values across eligible coordinates. Let `M_b=Σ_{i∈b} a_i g_i` be Z1's potential slow-write absolute mass for the *control arm's own current prestate*. Choose the unique nonnegative scalar `λ_b` satisfying

\[
\sum_{i\in b}a_i\min(1,\lambda_b g_{\pi(i)})=M_b,
\]

and set `g_i^random=min(1,λ_b g_{π(i)})`. Solve monotonically by bisection to `1e-12 × max(1,M_b)` absolute mass error, with exact zero/full-budget branches; all `a_i>0` here have `x_i=1`, hence this trace rule gives `g_i>0` at the current byte. The bisection and permutation do not change the stored `z`, `u`, or inherited `f`. This matches slow-write L1 **separately by side and sign**, and therefore also matches total slow L1 and signed mass at each event in each arm. The matched budget is computed causally from that arm's current state, not copied from a diverged Z1 future trajectory. Match the same one-step prestate on a clone as an additional direct placement diagnostic. Record buckets with one eligible coordinate, where randomization is necessarily inert. The random arm is a diagnostic control, not a fourth candidate mechanism.

All arms need the common `W`, `N_old_fact`, `N_old_rel`, `N_new_fact`, `N_new_rel` branches with identical bytes, feedback bits, timing and **four-output choices before each teacher**. Each `N_*` branch suppresses only its named native value writes while preserving exposure and nonplastic events. Measure exact taught-cue E1 safety and never-reinforced held-out-relation E3 in the **same continuing four-store life**, including after delay and intervening learning. A `Z1_W−Z1_N` effect on held-out relations is the relevant causal E3 quantity; `Z1_W−Z0_W` alone is not sufficient evidence that earlier teaching was reused. `Z0` and `Z1_BUDGET_RANDOM` carry the same extra trace storage as Z1; Z0's trace is computed but cannot affect its write split.

## Integration points, with no parent edits

1. Place new adapter code in this round's directory. Convert a fresh `F151ByteBrain` born at model time zero; require `type(base.fe) is bc.FE0`. Do not alter `BYTE_CORE_V9/brain_byte.py`, `BYTE_CORE_1_20260922/bytecore.py`, or the MiniFly source tree. Give the Z1 and random adapters distinct version/fixed digests.
2. Override the value `EvoLearner.event` through a cloned instance of the **actual imported** `minifly/V82E/src/model_evo.py` class. The import chain is `BYTE_CORE_1_20260922/bytecore.py:build_full151` → `V88/minifly_rce_v87_followup_runner.py:NativeBackend` → hash-checked V82E `model_evo`; `NativeBackend` explicitly rejects another `model_evo` path. The V82E source hash is `ccdeaab99a889e6a74e47f95d641bba79ed95597b8bd902181e1c535942e25d7`. The similar `minifly/iteration26_evolution/src/model_evo.py` has a different file hash and is **not** the Full151 runtime source. V82E's safe patch point is after its ordinary `super().event(dt,x,r,plastic)` returns `record['rawalpha']`, `record['rawslow']`, `record['rawfast']`. Save `a=record['rawslow'].copy()` and `b=record['rawfast'].copy()`; for `plastic=True` and active `x`, set `newslow=g*a`, `newfast=b+(1−g)*a`, then patch the **post-event** states by `m.slow += (newslow−a)*exp(−dt/slow_tau)` and `m.fast[:,1] += (newfast−b)*exp(−dt/fast_tau)`. Replace the returned raw fields and adjust cumulative `raw_slow_L1` and `raw_alpha_fast_L1` by the differences of old and new absolute sums; record `g*f` as the effective slow fraction. `super()` already evaluated feedback and updated adaptation from the identical prestate; the patch changes only the current event's stored split. Ensure the `g` array is the snapshotted gate for this exact `teach` call, not recomputed from post-event state. This is algebraically equivalent to the equation above and keeps the parent's gamma path bitwise unchanged.
3. Override `F151ByteBrain.byte` or factor it locally so only the *committed* byte path updates `z` once per byte, after `_features(t)` and before `fe.feed(b,t)`. `predict` and `association_value` must not update the live trace. Override `teach` only to pass the gate into the cloned learner event at the inherited moment. `rest`, `flush`, `_commit_pending` and `reset_stream` need absolute-time trace decay without double counting. Record `trace_elapsed` in native absolute seconds so a stream-time rebase cannot move it backwards.
4. Extend `snapshot`, `restore`, `save`, `load`, `state_digest`, `fixed_digest`, `resources` and the local clone helper for `z`, `trace_elapsed`, arm kind and fixed coordinate index hash. Deep-copy `z` on every branch clone. Check exact round-trip equality and alias isolation. Do not rely on the inherited `clone_native` shallow copy for the new array.
5. The round's common assay runner must put Z0, Z1 and random on **one FE0-only four-store architecture and one continuous mixed lifetime**. The four native stores and their byte-derived teacher routing use the same fixed mapping and output organ in every arm. Score exact taught items (E1) and genuinely never-reinforced relations (potential narrow E3) at prespecified points in this same life, before any probe feedback. An old single-store FE0 E3 result is a feasibility reference, not a substitute for E3 in the four-store model; Step-13 four-output taught-item recall is an E1 reference, not an E3 result. Do not attach a private single-store choice circuit to Z1 or use the dual-bank R2 architecture as its baseline.

## Technical gate before any scientific run

* **Parent equality:** On a new technical world using the new mixed-life fixture, `Z0` must match the unmodified four-output FE0 Full151 parent at every teaching/probe receipt, **native** fast/slow/adaptation state, clock, raw fast/slow update and output byte. Z0's complete state digest differs because it includes the inert trace; compare native state separately and audit that trace independently. The inherited Step-13 and wrapped E3 qualified references can be checked on their own archived technical fixtures, but they do not replace this new-fixture equality check. In particular, the adapter must not accidentally instantiate FE1; assert `type(fe) is bc.FE0` for every arm and branch.
* **Trace causality:** Start from zero, expose one active KC and verify `z=0.20`; advance 30 seconds without activity and verify multiplication by `e^-1`; repeat activity to verify bounded growth. Verify a never-active coordinate receives no activity increment, teacher labels alone do not raise `z`, 24-hour rest decays, and `reset_stream` preserves/decays rather than clears the trace. Byte, rest and teach clock arithmetic must agree with the inherited native elapsed time.
* **Write algebra:** From the same cloned prestate with a positive and a negative nonzero `u`, check the predicted Z1 fast/slow raw and decayed state equations coordinate by coordinate, exact zero off native support, gamma/adaptation/payload equality with Z0 for that single event, and `fast+slow=u` before unequal decays. `write=False` must leave all alpha writes zero. Do not compare final fast+slow after different decay constants as if they should remain equal.
* **Budget control:** On each positive/negative side/sign bucket, verify random gate bounds, deterministic replay, changed assignment where at least two distinct gates exist, and the declared L1/signed-mass tolerance. The same-prestate clone must have identical `u`, `f`, `M_b` before gate placement. A deliberate random-seed or coordinate-map tamper must fail the technical receipt audit.
* **State and exposure:** Verify snapshot/save/load/clone equality and no shared `z`; one bit change in `z` changes state digest. Read-only probes leave the continuing model untouched. Check all five branch byte/feedback streams and native state digests; first choice precedes teacher; all held-out ordered pairs/byte forms are absent from training; native write counts and clocks match. Reject a held-out-label leak and a probe-mutation tamper.
* **Resources and source lock:** Record `|I|`, trace bytes `8|I|` per native store and `32|I|` for all four stores (upper bounds 41,416 and 165,664 bytes respectively if the inherited KC count is 5,177), cloned state and receipt costs, actual runtime/RSS, fixed array hashes, source hashes and exact world rosters. Lock equations, constants, controls, endpoints and audit code before pilot or confirmation. Pilot may assess feasibility, never choose a new `τ`, increment, bucket or endpoint.

For any reported behavior, use world-paired uncertainty; records, KCs and individual held-out probes are repeated observations inside a world. State E0 for technical checks, E1 for exact taught-cue retention, E2 only for an untouched presentation of an already taught relation, and at most narrow E3 for a never-reinforced relation whose first choice precedes feedback and whose W−N causal control passes. An exact-key table cannot answer those held-out relations, but a learned one-bit symbol rule can; exposure-independent symbol/position shortcuts must be checked. Report trained E1 safety, immediate and delayed held-out choices, later-cohort learning, actual slow budget, and all failures even when a headline contrast is positive.
