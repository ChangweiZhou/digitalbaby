# Workstream A — Extended Continuous-Life Characterization of `FULL_151`

**Status:** DESIGN + QUALIFICATION PLAN — for review before any science run
**Date:** 2026-09-19
**Subject:** `FULL_151` = `151ebd9e2d054ec9ec14`, frozen. Single arm.
**Purpose:** produce the **failure map** that determines which reserve
mechanism families enter Omnibus3.

---

## 1. Scope

This experiment characterises where the frozen operational core stops working
and **why**. It adds no mechanism, tests no variant, and changes no gene.

Explicitly out of scope: any new mechanism; any gene retuning; contraction
(Stage 3B is closed as `TRANSFER_UNRESOLVED`); topology (Workstream B);
network size (Workstream C); Omnibus3 family selection itself.

One arm only: `FULL_151`. There is no comparator, because the question is not
"is X better" but "where does this core fail, and in which of four
distinguishable ways".

---

## 2. What stays frozen

* the 17-gene genome, exactly as confirmed;
* PN→KC anatomy `B`, installed from `FULL_151` (`apply_phenotype` re-draws `B`
  from a whole-genome-seeded RNG, so it must be installed explicitly);
* the fixed-reference reader/observer convention;
* the V13/V88 teaching interface, event semantics, and probe semantics;
* the corrected relevance semantics: **forgetting a retired fact is not
  failure**.

---

## 3. The reference life H₀

One standard V13/V88 life, measured from the run-of-record worlds:

| quantity | H₀ |
|---|---:|
| events | 1,816 |
| teach events | 1,728 (1,008 acquisition + 720 challenge) |
| identities | 168 |
| relevance-context blocks | 5 |
| challenges | 60 (20 STABLE, 20 ANOMALY, 20 REV) |
| truth changes | 5 |
| reactivation probes | 12 |
| probe events | 27 |

**A1 therefore requires 4H₀ ≈ 7,264 and 16H₀ ≈ 29,056 events.** No existing
generator produces lives of that length, nor matched-duration waiting, nor
variable simultaneous demand.

### 3.1 The principal risk in this workstream, stated up front

Workstream A needs **new world generators** — the first since V88. That is the
main threat to its validity, and it is why the three generators are separately
qualified. The governing requirement is:

> **Reduction to reference.** Each generator, run at its H₀ / no-distractor /
> baseline-capacity setting, must reproduce the V88 reference life's structural
> statistics exactly, and must reproduce `FULL_151`'s endpoint values within
> the pilot's between-world noise.

A generator that cannot reduce to the reference is not used, and its regime is
reported as `GENERATOR_REDUCTION_FAIL` rather than as a scientific result.

---

## 4. The three generators

All three emit only V88-semantic events (`teach`, `probe`, `rest`,
`set_relevance_context`, `revision_curve`, `change_truth`,
`reactivation_probe`). No new event kind is introduced.

### A1 — Lifetime

**Question.** Does competence decay as cumulative experience grows, with
current demand held roughly constant?

**Construction.** One continuous life. Relevance-context blocks are appended;
each switch retires the previous block's facts and introduces a fresh cohort,
so the count of *currently demanded* facts stays within ±10% of the H₀ value
while total accumulated experience grows. Checkpoints at H₀, 4H₀, 16H₀.

**Critical.** These are **checkpoints inside one life, not replicates.** The
learner is never reset between them. Reporting must never treat H₀ and 16H₀ as
independent samples of the same quantity.

**Controls.** A `NO_FURTHER_TEACHING` fork at each checkpoint (copy of the
life, probes only) separates "degraded by later learning" from "degraded by
elapsed simulated time".

### A2 — Interference

**Question.** Is degradation caused by intervening *content* or by elapsed
*time*?

**Construction.** Two arms sharing **byte-identical useful events** — the same
teaching pairs, same roles, same order, same probes, same challenge schedule.
Only the filler between useful periods differs:

| arm | filler |
|---|---|
| `DISTRACTOR` | teaching on never-demanded identities |
| `WAIT` | `rest` of exactly matched simulated duration |

**Required invariant.** For every paired life: identical useful-event count,
identical teaching opportunities per demanded fact, identical probe count and
schedule, and identical total simulated elapsed time. Only filler content
differs. This is asserted per world, not assumed.

**Distractor identities** are drawn disjoint from every demanded cohort and are
never probed.

### A3 — Capacity

**Question.** Does competence fall because too much is simultaneously demanded,
or because each fact gets too little teaching?

**Construction.** Simultaneous demanded facts at 1x, 2x, 4x the H₀ cohort, in
two separately reported conditions:

| condition | total teaching budget | exposures per fact |
|---|---|---|
| `FIXED_BUDGET` | held constant | falls as 1/k |
| `FIXED_PER_FACT` | grows with k | held constant |

Reporting them separately is the point: `FIXED_BUDGET` confounds capacity with
exposure, `FIXED_PER_FACT` isolates capacity. A result quoted without its
condition is not interpretable.

---

## 5. Endpoints — scored separately, never pooled

| endpoint | definition |
|---|---|
| `demand_prediction_pre_feedback` | correctness at the demand moment **before** the teaching signal for that event |
| `revision` | correctness on genuinely revised truth (`change_truth` cohorts) |
| `anomaly_recovery` | correctness after a single contradictory event, original truth restored |
| `first_use` | correctness at return, **before any re-teaching** (P7 exposure 0) |
| `relearn_1` / `relearn_2` / `relearn_4` | correctness after 1 / 2 / 4 renewed exposures |

`demand_prediction_pre_feedback` is the primary novel endpoint of this
workstream: it is what the animal actually has available when the world asks,
and it is not recoverable from the terminal aggregates.

No single scalar is produced. There is no leaderboard.

### 5.1 Not failure

Loss of **retired** facts — those the current relevance context no longer
demands — is **not** scored as failure anywhere: not in ranking, not in any
decision, not in any summary verdict. It is recorded as a diagnostic only, and
the retention-term firewall used in Stage 3A/3B is carried over unchanged.

---

## 6. Diagnostics and the failure-mode signature

Four failure modes must be distinguishable. Each has a signature **across**
generators; none is identifiable from a single number.

| diagnostic | source |
|---|---|
| `margin` | probe `actual` (Hz) on demanded cues |
| `decision` | probe `correct` (requires `actual ≥ 1.0` **and** `fixed ≥ pair_threshold`) |
| `access_gap` | fraction with `actual ≥ 1.0` but `correct == False` |
| `clip_fraction` | probe `clip_fraction` — expression saturation |
| `slow_L1`, `alpha_fast_L1` | persistent-state magnitude |
| `write_concentration` | Gini of per-KC write mass |
| `writable_kcs_touched` | distinct KCs written in the window |
| `exposures_per_fact` | bookkeeping, from the generator |

### 6.1 Signature table

| mode | A1 across H | A2 distractor vs wait | A3 `FIXED_BUDGET` | A3 `FIXED_PER_FACT` | state signature |
|---|---|---|---|---|---|
| **Saturation** | degrades | little/no gap | degrades | **degrades** | `clip_fraction` ↑, `write_concentration` ↑ |
| **Interference** | degrades | **distractor ≫ wait** | degrades | degrades | margin ↓, overlap of written KCs ↑ |
| **Access failure** | may be flat then drop | either | either | either | **`access_gap` ↑ while margin stays high** |
| **Insufficient exposure** | flat | no gap | **degrades** | **flat** | margin ↓, recovers fully by `relearn_1/2/4` |

The two decisive contrasts are `DISTRACTOR` − `WAIT` (interference vs time) and
`FIXED_BUDGET` − `FIXED_PER_FACT` (exposure vs capacity). `access_gap`
separates "the trace is gone" from "the trace is present but not retrieved",
which no behavioural endpoint alone can do.

A mode is called only when its **whole** signature row is satisfied. Partial
matches are reported as `MIXED` or `UNRESOLVED`, never rounded to the nearest
story.

---

## 7. Qualification plan — before any science world is opened

### 7.1 Shared

1. **Source/environment lock, enforced.** Re-hash the A runner, Stage 3A
   runner, V13, V88, `model_evo.py`, `common_evo.py`, plus python / numpy /
   scipy / numba, at the start of every stage; raise on drift. (Stage 3A's
   `verify_source_lock` pattern, reused.)
2. **Frozen parent.** `FULL_151` genome hash; `B` hash identical across every
   life; fixed-reference reader Q hash.
3. **Determinism.** Same generator + seed + checkpoint reproduces bit-identical
   output.
4. **Probe invariance.** Real probe calls on a lived learner leave persistent
   state unchanged.
5. **Cue disjointness, hard gate.** Direct set audit across every A world plus
   the historical prior-key set; namespaces disjoint from V13 (100000–300999),
   Stage 3A (390000–420999), Stage 3B (430000–460999), the bridges, Stage I.
   Proposed: 470000 / 480000 / 490000 / 500000.
6. **No-oracle.** AST check that no generator field naming relevance, future
   demand, retirement, or curriculum reaches the learner.
7. **Retention firewall.** `assert_no_retention` over every decision object,
   plus the AST check over decision functions.

### 7.2 Per generator — reduction to reference

| generator | reduction setting | must reproduce |
|---|---|---|
| A1 | H₀, one block schedule | V88 reference structural counts exactly (§3), and `FULL_151` endpoints within pilot noise |
| A2 | `WAIT` arm, zero-duration filler | same as A1 |
| A3 | k = 1x, `FIXED_PER_FACT` | same as A1 |

Any generator failing reduction is reported `GENERATOR_REDUCTION_FAIL` and its
regime is not run.

### 7.3 Per-generator specific

* **A1** — `currently_demanded_count` stays within ±10% of H₀ at every
  checkpoint; the learner is never reset; checkpoint state hashes form a chain
  (the 4H₀ state must descend from the H₀ state).
* **A2** — per-world assertion of byte-identical useful events, identical
  teaching opportunities per demanded fact, identical probe schedule, and
  matched total simulated elapsed time between `DISTRACTOR` and `WAIT`;
  distractor identities provably disjoint from all demanded cohorts and never
  probed.
* **A3** — `exposures_per_fact` exactly constant across k in `FIXED_PER_FACT`;
  total teaching events exactly constant across k in `FIXED_BUDGET`; both
  recorded per world, not assumed.

---

## 8. Protocol lock

Frozen before science, hashed, and re-verified at every later stage:

* the three generator specifications and their parameters;
* checkpoint definitions (H₀, 4H₀, 16H₀) and k levels (1x, 2x, 4x);
* the endpoint list of §5 and the exclusion of retired-fact loss;
* the diagnostic list and the §6.1 signature table **as the pre-registered
  calling rule**;
* practical margins per endpoint, from a `FULL_151`-only pilot at
  0.5 × between-life SD, frozen before any regime is inspected;
* world namespaces and seeds;
* all source and environment hashes.

---

## 9. Science schedule

Exploratory panel first, then **fresh** lives for confirmation.

| stage | lives |
|---|---|
| pilot (margins, `FULL_151` only, reduction settings) | 8 |
| A1 exploratory | 16 independent lives, checkpointed at H₀/4H₀/16H₀ |
| A2 exploratory | 16 paired lives × 2 arms |
| A3 exploratory | 16 lives × 3 k-levels × 2 conditions |
| confirmation | fresh lives, count frozen after the exploratory panel |

Cost note: a 16H₀ life is ~29k events. At the observed ~1.3 s per H₀ life the
A1 16H₀ cell alone is roughly 16 × 16 × 1.3 s ≈ 6 minutes; A3 is the largest
cell at 96 lives. The whole exploratory panel is plausibly under an hour, but
this estimate is to be replaced by a measured one during qualification.

---

## 10. Analysis

Per endpoint, per regime: paired where a pairing exists (A2 arms, A3
conditions at matched k), otherwise per-life distributions with mean, median,
95% interval, fraction improved/worsened, worst case. Student t on the life as
the inferential unit, as in Stage 3A/3B. A1 checkpoints are **within-life** and
are reported as trajectories, never as independent paired contrasts.

---

## 11. Output — the failure map

The deliverable is a table of *regime → called failure mode → evidence*, and
from it a mapping to reserve mechanism families for Omnibus3. Indicative
correspondences, to be confirmed against your ten-family reserve list:

| called mode | families it would motivate |
|---|---|
| Interference | protection/uncertainty, write localisation, banks/routing |
| Saturation | compression, sparsity/activity control, capacity allocation |
| Access failure | reinstatement / read authority, retrieval addressing |
| Insufficient exposure | spacing / adaptation budget, replay-repetition |

A mode that is not called does **not** motivate its families. `MIXED` and
`UNRESOLVED` outcomes motivate nothing and are reported as such.

---

## 12. Open questions for review

1. **Ten reserve families.** The mapping in §11 is indicative. Please supply
   or confirm the authoritative ten-family list so §11 is pinned in the lock
   rather than inferred from the audit index.
2. **16H₀ feasibility.** Runtime and memory at 29k events are estimates; if a
   16H₀ life proves impractical, the alternative is 8H₀ with the same
   checkpoint logic — this changes the lock and should be decided before it is
   frozen.
3. **Demand-context switch rate in A1.** Holding demand fixed while experience
   grows requires a retirement schedule. I propose retiring one block per new
   block after the first five; a different rate changes what "roughly fixed
   demand" means and should be your call.
4. **A2 filler intensity.** Distractor teaching at the same rate as useful
   teaching is the natural default; a lower rate would weaken the contrast and
   a higher one would confound it with total-event count.
5. **Confirmation size.** Left unfrozen deliberately until the exploratory
   panel shows the variance.

No science world will be opened until these are settled and the protocol lock
is hashed.
