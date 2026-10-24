# A11-A reduced-scope modular redesign v2.2

**Date:** 2026-09-25  
**Status:** design direction accepted; protocol **not frozen**; **no science authorization**.  
**Relationship to v3.8.3:** this document chooses a reduced-scope, three-cell program. It neither completes nor silently replaces the original nine-cell A11-A certification. A full-grid run would require a separate explicit scope decision and protocol; it is not an implicit continuation. No frozen scientific code is changed by this document. v2.1 corrected the later-episode slot-0 donor omission; v2.2 closes four review text gaps and fixes the intended use of the limited result.

## 1. Scientific scope and execution unit

Precommit exactly three cells before inspecting any new scientific outcome:

| Module | Cell | Question | v3.8.3 rows/life | Rows at 32 seeds × 2 anchors |
|---|---|---|---:|---:|
| M1 | K8/L1 | Moderate-load reference | 30,185 | 1,931,840 |
| M2 | K16/L1 | Added active-load policy versus M1 | 30,185 | 1,931,840 |
| M3 | K8/L4 | Three additional K8 episodes versus M1 | 115,286 | 7,378,304 |

Total at the **provisional**, not yet justified, `n=32`: 11,241,984 structural event rows and 192 full lives, 18.4% of the original 60,958,272-row / 576-life A11-A plan. This is an 81.6% reduction in *planned structural rows by reducing scientific scope*, not a measured CPU speedup. M4 K8/L16 and all A11-C work are separate future decisions, not implicit extensions.

These are **modular scientific questions executed in approximately one-hour packets**. A one-hour packet is a scheduling and failure-containment unit; it does not define scientific `n` or an independent observation. M1–M3 share the same seed bank and M1 reference and form one precommitted, correlated three-cell study. They are not three statistically independent trials.

**Intended use:** a limited three-domain competence measurement that may inform later minimal-core preservation locks **only in these measured domains**. It does not choose a global B/HASH8 parent, freeze a complete minimal-core preservation envelope, or establish low-load competence. K1/L1 is deliberately deferred as a separate, pre-science-locked module if a downstream preservation decision needs a low-load reference; it is not part of the present three-cell claim. This is a scope boundary, not a finding that K1 lacks scientific value.

## 2. Master-world cross-cell nesting lock

For each independent scientific base seed `s`, generate and seal one anchor-independent master object before running either anchor or any cell:

```text
Omega_s = {
  canonical D world and D_RETURN_SET,
  A identity/source world,
  episode-1 source worlds: slot 0 = A; E[1,r] for r=1..15,
  later-episode fresh source worlds: E[e,r] for e=2..4 and r=0..15,
  source roles, truth, timing, provenance, generator/hash versions,
  separate HASH8 learner-birth record: one key identity in the frozen namespace
}
```

The sealed master **record** separates exogenous source worlds from HASH8 learner-birth state. Its world section is the **only** source of exogenous data for all six lives `(3 cells × 2 anchors)` at seed `s`; the birth key is used by HASH8 only and never passed into B. Cell events are deterministic projections `P_K,L(Omega_s)`; no cell, anchor or runner may advance a mutable source RNG to obtain its own world. A master-record hash, world-section hash, projection-rule hash, per-cell event-stream hash and birth-key digest are recorded before science; projections are regenerated from the frozen record, not from observed outcomes. Seeds are fresh relative to earlier science and predeclared as an ordered bank.

Required projection equalities:

1. M1 and M2 use the **same D and the same episode-1 sixteen source slots**, object identities, roles, outcomes and durations. M1 teaches A and F[1,1..7] and replaces slots 8..15 by the prescribed matched rests; M2 teaches A and F[1,1..15]. The difference is the frozen active-teach/rest policy at those slots, with the FPF16 elapsed-time clock matched. Neither cell draws replacement worlds.
2. M1 and M3 use the **same K8 episode-1 event history through `CORE_END(1)`**, comparing every frozen ledger field and event-stream digest. M3 then takes the v3.8.3 zero-time `A_PERSIST_SET` boundary and appends episodes 2–4. M1 legitimately takes the terminal D-return path after `CORE_END(1)`; a full-row-prefix assertion beyond that point is invalid.
3. Both anchors see identical exogenous projection events for each `(s,K,L)`, with the same HASH8 birth-key identity reused across that seed's HASH8 cells. Anchor-specific learner state and outcomes are never used to alter the projection.

Independent qualification must reject: separately generated cell worlds; changed slot identities/roles; unmatched teach/rest durations; an M1/M3 mismatch before `CORE_END(1)`; missing or misplaced L>1 boundary; missing `E[e,0]` for any later episode; A's source world reused as a later slot-0 donor; and anchor-dependent source events. Changing only the birth key must leave every B event/output digest unchanged while changing the HASH8 route map. Technical fixtures do not enter the scientific seed bank.

## 3. Estimands and interpretation lock

For anchor `a ∈ {B,HASH8}`, seed `s`, cell `g ∈ {(8,1),(16,1),(8,4)}` and inherited primary endpoint `q ∈ {current usefulness, first-use D return}`, define the seed-level score `Y[s,a,g,q]` under the existing A11 endpoint extraction and denominator rules. The new protocol must freeze the exact reducer and missing/invalid-score handling before science; event rows and individual objects are **not** inferential units.

The planned estimands are the population mean of each of the six `(cell, endpoint)` scores per anchor, plus paired seed-level contrasts:

```text
Delta_K[a,q] = E_s[Y[s,a,(16,1),q] - Y[s,a,(8,1),q]]
Delta_L[a,q] = E_s[Y[s,a,(8,4),q] - Y[s,a,(8,1),q]]
Delta_anchor[g,q] = E_s[Y[s,HASH8,g,q] - Y[s,B,g,q]]
```

`Delta_K` is the **total K16-versus-K8 active-load policy contrast on the common FPF16 clock**. For `current usefulness`, K16 also changes the scored relevant-set composition and denominator, so that native score difference alone cannot be called interference or forgetting of the shared K8 objects. Report a separately tagged shared-target diagnostic on A+F[1,1..7] in both cells if the inherited endpoint machinery can define it without changing the primary score; do not use it to replace the locked primary endpoint after seeing outcomes. First-use D return uses the same frozen D return set and is a cleaner common-target comparison.

For this limited competence-envelope purpose, **native current usefulness remains the primary system-level score**. The A+F[1,1..7] shared-target score is predeclared as a secondary diagnostic of performance on common objects, subject to a source-faithful reducer qualification. A future study whose primary question is interference on those objects must define its own primary estimand before its science starts. Neither score is relabeled after observing outcomes.

`Delta_L` is the **continued-learning lifetime contrast**, the total effect of three extra K8 episodes under the frozen policy. It includes extra elapsed time, fresh writes, matched rests and cohort replacement. It is not a pure fresh-learning effect or a pure time-decay effect. No load×lifetime interaction is estimable without K16/L4; no claim is made about unmeasured cells.

The original A11 §33 pattern labels, including “constant shift,” “load-amplified advantage” and “lifetime-decaying advantage,” are **out of scope as inferential claims**. The four anchor×contrast difference-in-differences are not members of the four primary families below. They may be displayed as clearly labeled descriptive arithmetic only; no §33 status or advantage label is issued from them.

## 4. Joint analysis and multiplicity lock

All three modules and their analyses are committed **before M1 science begins**. For each seed, the complete analysis vector contains both anchors, three cells and two primary endpoints. Resampling, if used, resamples **whole base-seed vectors** and preserves every within-seed cross-cell and cross-anchor pairing. No analysis treats lives, objects or event rows as independent samples.

The intended primary inferential objects are four explicit simultaneous 95% families:

1. Six B absolute means (`3 cells × 2 endpoints`).
2. Six HASH8 absolute means.
3. Six paired anchor differences (`3 cells × 2 endpoints`).
4. Eight within-anchor modular contrasts (`2 anchors × {Delta_K,Delta_L} × 2 endpoints`).

Each family needs a pre-science simultaneous-interval method, resampling seed and coverage/precision qualification. **Four separate 95% families do not make one joint 95% statement across all 26 quantities.** If a decision requires joint protection across families, the protocol must choose and qualify a single joint family or an explicit cross-family allocation before science. No favorable member may be promoted post hoc to a familywise claim. Secondary competence/diagnostic outputs are labeled as such with separately specified pointwise intervals, if any. Numerical intervals, decision thresholds and formal result labels remain **unfrozen** until the module-specific statistical lock is approved.

For any primary score or contrast with **zero empirical variance across complete seeds**, a zero-width resampling interval must **not** be reported as evidence of certainty: mark that component `INFERENCE_UNRESOLVED_ZERO_VARIANCE`, unless a pre-science structural proof establishes the constant value and its population meaning. A near-constant but nonzero endpoint also requires a pre-science-qualified floor/coverage method; the numerical near-constant criterion, interval fallback and simultaneous-family handling must be fixed in the statistical lock, not chosen after seeing science. This rule does not by itself delete the D-return endpoint.

There is **no formal interim inference** in the initial design. Hourly reports contain operational counts and resources only, with scientific scores sealed until the precommitted analysis point after all required complete seed vectors exist. Running M2 or M3 cannot depend on M1's scientific outcome. Any later outcome-driven addition (including M4) is conditional/exploratory unless a separate sequential selection rule was frozen before M1.

## 5. Scientific `n` and one-hour packets are separate locks

Choose `n` **once before science** from a sourced precision requirement for the four families and a cumulative CPU/RSS/disk budget. Use historical or qualification-only planning inputs; never use new M1–M3 outcomes to set `n`. `n=32` is a cost illustration, not automatically the right or one-hour sample size. If useful precision requires multiple one-hour packets, run multiple packets. If cumulative budget cannot support useful `n`, defer confirmation or explicitly designate an exploratory pilot. An `n=8` run does not become confirmatory because it fits an hour.

The 2026-09-25 FB1-based audit estimates that **this three-cell scope**, if scaled to `n=128`, might cost roughly 16–19 CPU hours. My earlier 76–93 CPU-hour warning applied to the **full nine-cell grid** at `n=128`, not to this reduced program. Neither extrapolation qualifies A11 runtime or establishes that `n=128` meets a scientific precision target. The historical 5-percentage-point practical-effect reference is not itself a 2.5-point simultaneous-interval half-width requirement.

Every packet manifest lists exact whole-life IDs `(module, seed, anchor, cell)`, ordered inputs and hashes, expected receipt keys, source/runner versions, worker count and resource caps. A complete scientific seed vector requires **all six whole-life receipts** for the frozen master object and validation of pairings; partial lives, attempts and in-flight jobs are counted separately. Receipts are sealed transactionally and reused only for the identical frozen input. Technical failure resumes that input; it never causes world or seed replacement.

The existing V88 `evaluate_life()` creates a learner, iterates the entire event list and returns a receipt at the end. **There is no qualified event-cursor resume interface.** Until an exact mid-life checkpoint passes uninterrupted-versus-resumed final-state/write, event-digest and scored-output equality for both anchors and L>1 boundaries, a packet contains only **whole lives**. The manifest must not contain arbitrary event ranges for execution. Historical FB1 receipts make short A11 lives plausible, but do not directly measure A11's matched rests, K16 or L4. A long life may still make a packet exceed one hour; the 45-minute target / 60-minute ceiling is therefore an **operational SLO, not a guaranteed hard stop**. Do not kill a life at 60 minutes and count it as completed. If a strict upper bound is required, first qualify a true mid-life checkpoint and then revise the packet contract.

Runtime qualification measures each **anchor × cell × final logging/receipt mode**, including cold start/JIT, learner construction, event processing, digest sealing and receipt validation. Equal row counts do not imply equal CPU cost: active teaching and matched rest take different paths. Record wall and CPU time, peak aggregate RSS, peak and persistent disk, and variability over technical seeds. A few timing seeds establish planning evidence, not a proof that every future life finishes below 60 minutes. The cumulative resource budget remains separate from the per-packet SLO.

## 6. Science data and completion boundary

Do not persist all scientific event rows as CSV. The preferred science receipt contains source/master/projection hashes, streaming event digest, phase counters, endpoint accumulators, per-life resource data and the compact checkpoints required for audit. Full technical ledgers remain qualification artifacts. Qualification must prove the compact streaming reducer yields the same endpoints and digests as a full technical ledger on representative cells, anchors and L>1 boundaries; failures retain only a bounded local event window. Deterministic source regeneration must be verified against the sealed master-object hash.

Named technical negative controls for this equality are `ALTER_EVENT`, `DROP_EVENT`, `REORDER_EVENT` and `WRONG_INPUT_HASH`. Each of the first three must change the expected event-stream digest and fail the independent stream-versus-ledger check; a receipt carrying a wrong master/projection/source hash must be refused before endpoint aggregation. These controls run only on technical fixtures and do not generate replacement science worlds.

Allowed narrow status labels include `A11_MODULAR_M1_COMPLETE`, `A11_MODULAR_M2_COMPLETE` and `A11_MODULAR_M3_COMPLETE` for audited cell receipts, followed by a separately defined `A11_MODULAR_THREE_CELL_ANALYSIS_COMPLETE` only after all three cells and the four families pass. A cell completion label does not imply final inference. The final artifact may describe only a **measured three-cell envelope**. It must not output the original `A11_B_COMPETENCE_ENVELOPE`, `A11_HASH8_COMPETENCE_ENVELOPE` with nine-cell meaning, or any original `A11_COMPLETE_*` status. Any later B/HASH8 parent choice or minimal-core preservation rule is confined to the three measured domains and cannot certify the six omitted cells.

## 7. Freeze gates still outstanding

This v2.2 document closes the **design-level** ambiguities in master-world nesting, limited estimand interpretation, excluded §33 claims, `n`/packet separation and whole-life restartability. It does **not** authorize a science runner. Before science code is treated as runnable, independently qualify:

- exact projection/hash fixtures and negative controls in §2;
- exact inherited endpoint reducers, missing-score policy and shared-target diagnostic feasibility;
- any future K1/L1 low-load module and its inferential family under a separate lock, before its science; it is not silently attached to this three-cell program;
- numerical precision targets, `n`, interval algorithm, resampling seed and any decision thresholds for §4;
- an independently labeled exploratory pilot, if D-return floor or endpoint variance is inspected to revise the protocol; no pilot seeds enter confirmation;
- cumulative/per-packet resource caps and anchor×cell×logging-mode timing;
- streaming-versus-full-ledger equality, sealed receipt recovery and (only if needed) exact mid-life checkpoint;
- path-sensitive independent audit and a machine-readable `NO_SCIENCE_AUTHORIZATION` → authorized transition.

Until those pass, the current scientific status remains **`NO_SCIENCE_AUTHORIZATION`**. No scientific trajectories are launched or rerun by this design revision.

This new reduced-scope A program does not wait for original A11-C's C0 triage. It also does not execute C0/C or inherit any original `A11_COMPLETE_*` status.
