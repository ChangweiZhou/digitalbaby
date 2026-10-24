# V44 — Capacity Causal Decomposition
## Where does TRACE-CORR lose continued acquisition under fixed slow-memory capacity?

**Status:** preregistration / implementation specification only.  
**Date:** 10 September 2026.  
**No V44 scientific outcome has been observed in this document.**

---

## 0. Decision in one sentence

V43 established a high-load **continued-acquisition failure** before its read-arbitration claim could qualify. V44 therefore freezes the representation, base learner, fast superposed memory, TRACE-CORR admission/consolidation rule, matching rule, value update, and read architecture, and asks one causal question:

\[
\boxed{
\text{At load 256, how much of the failure is caused by pressure in the trace tier,}
\text{ pressure in the mature-slot tier, their interaction, or neither?}
}
\]

V44 is a **diagnostic experiment**, not a repair round. It contains no new eviction heuristic, utility score, admission rule, representation, write rule, learned router, or parameter sweep.

The output of V44 chooses the *class of mechanism* that a later fixed-budget experiment is allowed to test.

---

# 1. Locked interpretation entering V44

The following assessment is treated as fixed input to this design and is not reopened inside V44.

### 1.1 V43's high-load failure is real

The completed V43 roster contained 16 seeds at each total identity load 64, 128, 256 and 512. At the original TRACE-CORR capacities \(T=512\) traces and \(C=384\) consolidated slots:

| load | cells passing all V43 acquisition checkpoints | final new-cohort native accuracy | final new-cohort mature-slot coverage | old held-out recall, SILENT-HARD |
|---:|---:|---:|---:|---:|
| 64 | 16/16 | 100.00% | 96.88% | 99.22% |
| 128 | 15/16 | 94.53% | 17.19% | 99.61% |
| 256 | **1/16** | **66.89%** | **4.59%** | **99.02%** |
| 512 | **0/16** | **44.14%** | **0.44%** | **74.61%** |

All 384 mature slots were occupied at loads 128, 256 and 512. Mean cumulative mature-slot capacity evictions were 258.31, 978.19 and 802.44 respectively.

These facts identify capacity pressure and failed continued acquisition, but they do **not** identify whether intended identities fail before trace admission, during trace residence, during matching/confirmation, after promotion, or by mature-slot replacement.

### 1.2 Mature-slot eviction is not preselected as the culprit

Earlier V39 TRACE-CORR data already exhibited a trace-tier loss mode: some target traces were removed by trace capacity before promotion, whereas mature target-slot capacity eviction was absent in that workload. V39 and V43 use different load/exposure regimes, so this does not diagnose V43; it does forbid V44 from assuming that mature LFU replacement is the answer before measuring the pipeline.

### 1.3 FAST-guided admission is not a free improvement

V41 already warned that allowing the stronger fast fallback to influence slow-memory admission can reduce target promotions and delayed recall even when mature capacity remains unused. V44 therefore preserves the inherited slow learner's **base-based admission logic** exactly. FAST may be scored for read diagnostics but may not decide whether a trace or slot is created.

### 1.4 Read arbitration is not the target of V44

V43 showed that `FIXED-50` is a strong simple read reference and that `LOCAL-BAYES-0` contains additional descriptive signal. V44 carries these forward only as read-only scoring policies. They cannot affect base writes, fast writes, trace creation, matching, promotion, revision or eviction.

`LOCAL-BAYES-E2` is retained only as historical provenance and is not a V44 candidate.

---

# 2. Scientific question and hypotheses

## Primary causal question

At total identity load 256, with the V43 stream and learning rules unchanged, compare the original capacity limits with selective removal of each capacity constraint.

Let

\[
T\in\{512,\infty\},\qquad C\in\{384,\infty\}.
\]

The four slow-memory branches are:

| branch | trace capacity | mature-slot capacity | interpretation |
|---|---:|---:|---|
| `CAP-BASE` | 512 | 384 | exact V43 physical slow-memory reference |
| `TRACE-RELIEF` | nonbinding | 384 | removes trace-pool overflow only |
| `SLOT-RELIEF` | 512 | nonbinding | removes mature-slot overflow only |
| `BOTH-RELIEF` | nonbinding | nonbinding | removes both capacity constraints |

"Nonbinding" means the capacity check is disabled for that tier on this finite stream. It does **not** mean infinite-memory storage is proposed as a deployable architecture. Persistent bytes and peak occupancy are reported honestly.

### H1 — trace pressure contributes materially

Removing trace overflow improves continued acquisition and/or durable consolidation relative to `CAP-BASE`.

### H2 — mature-slot pressure contributes materially

Removing mature-slot overflow improves continued acquisition and/or durable consolidation relative to `CAP-BASE`.

### H3 — the tiers may interact

The effect of relieving one tier may depend on whether the other tier is capped, because retained traces and slots change later candidate competition and promotions. V44 therefore estimates the full \(2\times2\) interaction rather than treating two independent one-factor comparisons as sufficient.

### H4 — capacity may be contributory but insufficient

Even `BOTH-RELIEF` may fail to restore acquisition. If so, matching, predictive confirmation, value semantics or representation becomes the primary unresolved mechanism; V44 must not force an eviction conclusion.

---

# 3. What V44 is explicitly *not* testing

V44 must not add or tune:

- a new read arbiter;
- utility/LFU/LRU/age-aware replacement;
- FAST-relative admission or consolidation;
- a new trace-admission condition;
- a new promotion rule;
- a new similarity threshold;
- separate searches over traces and slots;
- a new encoder or phase normalization;
- a new fast-store write rule;
- a new slow value update;
- replay;
- task/key/alias labels inside the learner;
- a larger *deployable* memory budget presented as the solution;
- a capacity sweep.

The two relieved capacities are **causal diagnostic interventions only**.

---

# 4. Frozen substrate

V44 inherits the V43 physical learner and V43 load semantics.

## 4.1 Stream

Scientific diagnostic panel:

- seeds `43101`–`43116`;
- total identity load: **256 only**;
- four waves A/B/C/D;
- 64 logical identities per wave;
- eight acquisition presentations per identity;
- inherited one-off-noise and revision rows;
- same row framing, target alphabet, RNG call order and stream SHA as V43 for each seed.

These seeds are deliberately reused because V44 is a paired causal follow-up to the observed V43 failure. **They are not fresh replication.** Any fixed-budget mechanism selected after V44 requires a new independent seed roster.

## 4.2 Shared BASE and FAST state

Within each seed there is exactly one BASE learner and one FAST superposed layer.

They are independent of the slow-memory capacity branch and therefore are updated **once per byte**, after all four slow branches have finalized their pre-outcome predictions.

All four slow branches receive the exact same pre-update base scores and the exact same byte outcome.

The FAST distribution may be scored but never enters slow admission or promotion.

## 4.3 Four independent slow states

The four branches begin byte-identical except for their two capacity flags.

After a capacity intervention changes which trace/slot remains resident, the slow states are expected to diverge. They must then evolve independently through the unchanged TRACE-CORR logic.

This divergence is the causal intervention; it must not be suppressed by fixed-history counterfactual scoring.

## 4.4 Unchanged TRACE-CORR semantics

Preserve exactly:

- first unsupported base error → exact probationary trace;
- exact trace reads/updates as inherited by the physical learner;
- one nearest-candidate competition across mature slots and probationary traces;
- similarity threshold \(\tau=0.85\);
- a support-1 trace has no similarity-read authority but may be a promotion proposal;
- distinct-context pre-update predictive confirmation for TRACE-CORR promotion;
- two-anchor mature slot;
- unchanged absolute-value update;
- unchanged exact-anchor revision behavior;
- mature-slot `LFU→LRU` replacement in capped branches;
- trace LRU replacement in capped branches;
- no tombstones or hidden unbounded eligibility state.

The only change between branches is whether the relevant overflow eviction is executed.

---

# 5. Read policy during V44

Acquisition/consolidation diagnosis is defined on the **physical slow states**, not on a chosen virtual read policy.

For utility reporting, score the following counterfactual readouts on each branch:

1. `SUPER` — FAST only;
2. `SILENT-HARD` — mature exact slow, silent traces, hard mature similarity, FAST fallback;
3. `FIXED-50` — V43 simple reference: mature exact slow, silent traces, 50:50 FAST/SLOW only on mature supported-similarity route G;
4. `LOCAL-BAYES-0` — branch-local evidence initialized at zero, read-only with respect to learning.

`FIXED-50` is the default deployed-reference score for V44 reports. `LOCAL-BAYES-0` is secondary and cannot determine the causal capacity verdict.

Every read policy must be scored before the true byte and must leave physical state unchanged.

---

# 6. Two stages

## Stage F — V43 forensic closure

Before interpreting the four capacity branches, replay the **unchanged `CAP-BASE` branch** with passive instrumentation on the V43 roster at loads 64, 128, 256 and 512.

Purpose:

- reconstruct where the original pipeline loses opportunities as load rises;
- verify that the new instrumentation does not alter the learner;
- obtain the same diagnostics across the healthy, transition, failing and stress regimes.

Stage F is retrospective measurement of V43 histories. It is not an independent scientific replication and may not be used to retune V44 thresholds.

If storage/runtime pressure requires staging, load 256 is mandatory; 64/128/512 may be replayed after the V44 causal panel, but their diagnostic definitions must remain fixed.

## Stage C — V44 causal capacity panel

At load 256, run the four capacity branches simultaneously for each of the 16 V43 seeds.

Top-level scientific units: **16 seeds**.  
Slow-memory states evaluated: \(16\times4=64\).  
BASE/FAST learning histories: **16**, not 64.

No branch or seed is added or removed in response to outcomes.

---

# 7. Forensic instrumentation

The V44 forensic ledger is evaluator-only. Evaluator metadata may be recorded **after** learner decisions but may never enter prediction, matching, admission, promotion, value update or replacement.

## 7.1 Every memory-transition event

Record compact rows for:

```text
seed
load
branch
record_index
byte_index
physical_time
event_type
trace_uid
slot_uid
source_trace_uid
reason
trace_count_before / after
slot_count_before / after
trace_age
trace_hits
slot_age
slot_hits
slot_last_use
```

Transition event types include at minimum:

```text
trace_admission
trace_exact_hit
trace_capacity_removal
trace_promotion_removal
promotion
slot_allocation
slot_capacity_eviction
anchor_binding_or_exact_update
supported_slot_use
```

UIDs must be monotone generation identities; array positions or reused storage positions are not valid IDs.

## 7.2 Evaluator provenance appended to event rows

After the learner has produced the transition, append only for analysis:

```text
is_designated_target_byte
logical_key_id
wave
alias
row_kind
slot_provenance_class
trace_provenance_class
```

Recommended mature-slot provenance classes:

- `target_same_key`;
- `target_cross_key`;
- `target_mixed_nontarget`;
- `nontarget_only`.

These labels describe resource ecology. **Nontarget state is not declared waste by definition**, because ordinary-byte prediction is part of the task.

## 7.3 Target-event candidate audit

On every designated target byte, before the outcome is revealed, log:

```text
exact_trace_present
exact_slot_present
best_trace_uid
best_trace_similarity
best_slot_uid
best_slot_similarity
winning_candidate_type
winning_candidate_uid
winning_similarity
candidate_support
candidate_top1
base_top1
fast_top1
```

Evaluator-side after the event may additionally derive:

```text
same_key_trace_existed
same_key_trace_above_tau
same_key_trace_won
same_key_slot_existed
same_key_slot_won
candidate_correct_for_outcome
```

Do not change lookup by searching for an evaluator-known "correct" candidate. The audit observes the locked nearest-candidate competition exactly as it happened.

## 7.4 Trace-survival and consolidation ledger

For each target-trace generation, derive:

- admission time and exposure;
- exact recurrences while resident;
- first arrival of the alternate learned rendering after admission;
- whether the trace survived until that arrival;
- similarity to that alternate rendering;
- whether it won candidate competition;
- whether its pre-update value predicted the outcome;
- whether promotion occurred;
- removal time and reason.

For each target mature-slot generation, derive:

- promotion time;
- anchor identities;
- residence lifetime;
- supported and exact uses;
- capacity-eviction time, if any;
- checkpoint presence;
- held-out alias retrieval route and correctness.

Do **not** force these into mutually exclusive causal percentages. A missing promotion can result from interacting residence, geometry, candidate competition and prediction events. Report the transition probabilities and event histories separately.

---

# 8. Checkpoint evaluation

At every inherited acquisition and wave checkpoint, evaluate all intended identities with explicit denominators.

## 8.1 Current-cohort native acquisition

Reuse the exact V43 Q0 evaluator for learned aliases 0/1 of the current cohort. Do not condition on resident traces, resident slots or successful acquisition.

For every branch and seed record:

\[
A_{w,e}
=
\text{native slow-path top-1 accuracy for all intended learned aliases}
\]

at acquisition exposures \(e\) and wave \(w\).

The primary terminal acquisition endpoint is

\[
\boxed{Y_{\rm acq}=A_{D,\mathrm{after\ D}}.}
\]

## 8.2 Mature exact-slot coverage

For each cohort and checkpoint:

\[
M_w
=
\frac{\#\{\text{intended logical identities represented by a resident mature exact slot}\}}
{\#\{\text{intended logical identities}\}}.
\]

The co-primary terminal consolidation endpoint is

\[
\boxed{Y_{\rm mature}=M_D(\mathrm{after\ D}).}
\]

Coverage is **not** substituted for acquisition accuracy; both are reported.

## 8.3 Ever-promoted fraction

For each cohort:

\[
P_w
=
\frac{\#\{\text{identities promoted at least once by checkpoint}\}}
{\#\{\text{intended identities}\}}.
\]

This separates "never consolidated" from "consolidated then lost."

## 8.4 Held-out transfer and persistence

Probe alias 2 read-only for all intended identities.

Old-A after-D held-out accuracy remains a named safety endpoint because it is the quantity V43 preserved while new acquisition failed.

Additionally report matched-delay cohort summaries:

**one later wave of interference**

\[
\frac13\left(
A_{A\to B}^{held}
+A_{B\to C}^{held}
+A_{C\to D}^{held}
\right),
\]

and **two later waves**

\[
\frac12\left(
A_{A\to C}^{held}
+A_{B\to D}^{held}
\right).
\]

These are descriptive matched-delay summaries; they do not turn D into a delayed-retention cohort that the stream does not contain.

---

# 9. Primary causal estimands

For any seed-level endpoint \(Y\), write

\[
Y_{00}=\text{CAP-BASE},\quad
Y_{10}=\text{TRACE-RELIEF},\quad
Y_{01}=\text{SLOT-RELIEF},\quad
Y_{11}=\text{BOTH-RELIEF}.
\]

Report the direct baseline contrasts:

\[
\Delta_T^{(0)}=Y_{10}-Y_{00},
\]

\[
\Delta_C^{(0)}=Y_{01}-Y_{00},
\]

\[
\Delta_{TC}=Y_{11}-Y_{00}.
\]

Also report the full factorial effects:

\[
\Delta_T
=
\tfrac12\left[(Y_{10}-Y_{00})+(Y_{11}-Y_{01})\right],
\]

\[
\Delta_C
=
\tfrac12\left[(Y_{01}-Y_{00})+(Y_{11}-Y_{10})\right],
\]

and interaction

\[
\boxed{
I
=
Y_{11}-Y_{10}-Y_{01}+Y_{00}.
}
\]

Compute these first within seed, then aggregate across the 16 seeds.

Primary endpoints for this causal table:

1. \(Y_{\rm acq}\): after-D new-cohort native acquisition;
2. \(Y_{\rm mature}\): after-D D-cohort mature exact-slot coverage.

Secondary causal endpoints:

- D ever-promoted fraction;
- A-after-D held-out accuracy;
- one-wave and two-wave matched-delay held-out accuracy;
- target trace survival to first alternate-rendering opportunity;
- promotion conditional on surviving to an eligible alternate rendering;
- mature target-slot survival after promotion;
- whole-stream FIXED-50 bpb;
- whole-stream SUPER bpb (shared reference).

---

# 10. Statistical analysis

The seed is the independent replication unit. Memory objects, aliases, records and byte events are repeated observations within a seed and are not independent replicates.

## 10.1 Primary intervals

For every named paired/factorial seed contrast, report:

- all 16 seed effects;
- equal-seed mean;
- standard deviation;
- paired Student 95% interval;
- median and IQR as robustness descriptors.

Exact sign-flip sensitivity may be reported for the two primary baseline contrasts at \(Y_{\rm acq}\), but it does not replace the effect-size interpretation.

## 10.2 Material diagnostic threshold

For the two co-primary fraction endpoints, define a **material contribution** as:

\[
\boxed{\text{mean improvement}\ge0.05\quad\text{and paired 95% lower bound}>0.}
\]

Five percentage points is a diagnostic materiality threshold, not a claim that a 5 pp gain solves persistence.

## 10.3 No survivor filtering

All 16 seeds remain in the capacity-effect analysis regardless of whether the original V43 Q0 acquisition gate failed. Failure of the baseline is the phenomenon V44 is designed to explain.

Missing/corrupt seeds make the corresponding planned contrast incomplete; they are never silently dropped to create an interval.

## 10.4 Multiplicity

The two predeclared primary resource effects on \(Y_{\rm acq}\) are:

- `TRACE-RELIEF − CAP-BASE`;
- `SLOT-RELIEF − CAP-BASE`.

If p-values are reported, Holm-adjust these two. The factorial interaction and mature-coverage effects are mechanistic co-primary/descriptive quantities reported with intervals. V44's verdict is driven by effect sizes, complete-seed intervals, and the decision logic below rather than a search over many p-values.

---

# 11. Diagnostic gates and verdict logic

These gates do not "qualify" a deployable architecture. They determine what V44 has learned about the failure mechanism.

## Q0 — exact V43 baseline reproduction

Before any causal interpretation, `CAP-BASE` must reproduce the locked V43 load-256 physical trajectory for each seed to the information available in the V43 handoff:

- stream SHA;
- BASE state/hash;
- FAST state/hash;
- slow state/hash or canonical physical fingerprint where saved;
- promotion/revision/eviction counts;
- checkpoint native acquisition;
- mature coverage;
- terminal physical probes.

Passive instrumentation must not change these values.

Any unexplained mismatch voids V44.

## Q1 — intervention isolation

For every seed:

- BASE and FAST hashes are identical across all four branches at every record checkpoint;
- all four branches receive identical pre-update base scores;
- relieved trace tier produces zero trace-capacity removals;
- relieved slot tier produces zero mature-slot capacity evictions;
- capped tiers never exceed 512 traces / 384 slots;
- no learner-visible evaluator metadata;
- read policies never change physical state.

Failure voids causal attribution.

## Q2 — does capacity matter at all?

On \(Y_{\rm acq}\), `BOTH-RELIEF − CAP-BASE` must have positive paired 95% lower bound and mean gain at least 5 pp to call capacity a **material contributor**.

If not, verdict begins `CAPACITY_NOT_PRIMARY` regardless of intermediate occupancy counts.

## Q3 — is capacity relief sufficient for acquisition on this task?

`BOTH-RELIEF` must reach the inherited acquisition standard:

- mean after-D new-cohort native acquisition \(\ge0.90\);
- every seed \(\ge0.80\).

Passing Q3 means removing capacity pressure is sufficient to restore the acquisition criterion on this controlled task. It does **not** imply an unbounded architecture is acceptable.

## Q4 — trace pressure contribution

`TRACE-RELIEF − CAP-BASE` on \(Y_{\rm acq}\) is material if its mean is at least +5 pp and paired lower bound exceeds zero.

Report the same contrast on D mature coverage and target-trace survival.

## Q5 — mature-slot pressure contribution

`SLOT-RELIEF − CAP-BASE` on \(Y_{\rm acq}\) is material under the same +5 pp / lower-bound-above-zero criterion.

Report the same contrast on D mature coverage and post-promotion target-slot survival.

## Q6 — interaction

Report

\[
I_{acq}=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]

Call the interaction materially **synergistic** if mean \(\ge+0.05\) with lower bound above zero; materially **antagonistic** if mean \(\le-0.05\) with upper bound below zero. Otherwise report `interaction_small_or_unresolved`.

## Q7 — old-memory safety descriptor

Report A-after-D held-out accuracy under all four branches. Capacity diagnosis does not require old-memory noninferiority, because an unbounded diagnostic branch may expose a tradeoff; however any intervention that restores D acquisition by substantially destroying A must be described as a tradeoff rather than a clean rescue.

Use 5 pp as the named descriptive materiality threshold for old-A loss.

---

# 12. Verdict vocabulary

The report may assign more than one mechanistic descriptor when justified, but it must choose one top-level outcome.

### `CAPACITY_NOT_PRIMARY`

Q2 fails. Removing both caps does not materially rescue D acquisition. The next experiment must focus on matching, confirmation, value semantics or representation rather than replacement.

### `CAPACITY_CONTRIBUTORY_NOT_SUFFICIENT`

Q2 passes but Q3 fails. Capacity pressure matters, but removing it does not restore the acquisition criterion. Use the forensic transition table to identify the additional bottleneck.

### `TRACE_PRESSURE_DOMINANT`

Q4 passes, Q5 does not, and the trace-relief effect exceeds the slot-relief effect by at least 5 pp on mean D acquisition without a contradictory mature-coverage pattern.

The next fixed-budget mechanism should address probationary survival / trace resource allocation, not mature-slot replacement.

### `SLOT_PRESSURE_DOMINANT`

Q5 passes, Q4 does not, and the slot-relief effect exceeds the trace-relief effect by at least 5 pp without a contradictory trace-survival pattern.

The next fixed-budget mechanism should address mature replacement/turnover.

### `MIXED_TWO_TIER_PRESSURE`

Both Q4 and Q5 pass, or neither clearly dominates while Q2/Q3 show a strong capacity rescue.

The next design must preserve the two-tier interaction rather than optimize one tier in isolation.

### `TIER_INTERACTION_DOMINANT`

The individual relief arms are weak but `BOTH-RELIEF` is strongly rescuing and Q6 shows a material positive interaction.

This is evidence that the bottleneck arises from coupled trace/slot dynamics rather than one independent cap.

### `CAPACITY_RESCUE_WITH_SAFETY_TRADEOFF`

Capacity relief restores acquisition but loses more than 5 pp of old-A held-out recall relative to `CAP-BASE`. This modifier prevents calling the effect a clean persistent-learning rescue.

---

# 13. How the forensic data choose the next fixed-budget experiment

V44 itself stops after diagnosis. It does not select and test a repair on the same observed seed roster.

The following decision table is written **before V44 outcomes**.

| V44 pattern | next allowed mechanism class |
|---|---|
| trace pressure dominates; many target traces die before alternate rendering | same-budget probationary survival / trace replacement |
| trace survives, same-key candidate frequently loses nearest-candidate competition | candidate-selection/addressing experiment |
| same-key trace wins but predictive confirmation often fails even with both caps relieved | confirmation/value/representation experiment |
| targets promote, then mature generations are capacity-evicted | mature replacement/turnover experiment |
| mature replacement dominates and frequency poorly tracks durable incremental value | only then test a causal durability/utility replacement statistic, against simple LRU/age/random controls |
| both capacities relieved yet acquisition remains poor | stop capacity heuristics; reopen matching/representation |
| capacity rescue occurs but later cohorts lose old memory | persistent-learning tradeoff remains; repair must jointly gate acquisition and retention |

Any fixed-budget V45 mechanism must use **fresh independent seeds** and must not use V44 evaluator labels as learner inputs.

---

# 14. Resource ecology analysis

V44 should not equate `nontarget` with `waste`. Instead report what consumes state and what predictive work that state performs.

For each branch compute:

- peak and terminal trace/slot counts;
- cumulative trace admissions/removals;
- cumulative promotions/slot allocations/evictions;
- trace-time and slot-time occupancy (integral of resident count over physical bytes);
- occupancy by evaluator provenance class;
- target versus nontarget exact/support uses;
- whole-stream and target/nontarget bpb for FIXED-50;
- per-tier semantic bytes;
- candidate-scan operation proxy.

Cumulative allocation and simultaneous occupancy must be reported separately. A whole-run allocation count is never converted into a physical-capacity estimate by dividing it by identity count.

---

# 15. Required outputs

Keep the archive compact but preserve enough state-transition detail to answer the diagnostic question.

Required:

```text
SUMMARY.md
REPORT.json
MANIFEST.json
PROTOCOL.md
CAUSAL_CONTRASTS.csv
CHECKPOINT_ACQUISITION.csv
MATURE_COVERAGE.csv
EVER_PROMOTED.csv
COHORT_RETENTION.csv
CONSOLIDATION_PIPELINE.csv
TRACE_SURVIVAL.csv
SLOT_SURVIVAL.csv
RESOURCE_ECOLOGY.csv
READ_POLICY_SUMMARY.csv
OCCUPANCY_CURVES.csv.gz
TRANSITION_EVENTS.npz            # or parquet/gzipped CSV
TARGET_CANDIDATE_AUDIT.npz       # target-byte events only
STATE_HASH_AUDIT.json
BRANCH_FIREWALL_AUDIT.json
EVENT_CONSERVATION_AUDIT.json
ARTIFACT_HASHES.json
elm_v44_capacity_diagnosis.py
```

Stage-F retrospective outputs must be marked `V43_FORENSIC_REPLAY` and kept separate from Stage-C V44 causal contrasts.

Do not dump full 256-way distributions for every byte. Store only the quantities needed for exact audit plus aggregate online losses. Full vectors may be retained for engineering fixtures only.

Suggested archive target: **< 30 MiB**, excluding temporary resumable checkpoints.

---

# 16. Engineering invariants

`--self-test` must verify at least:

1. exact V43 TRACE-CORR semantics on synthetic trace admission, promotion, revision and eviction fixtures;
2. `CAP-BASE`, `TRACE-RELIEF`, `SLOT-RELIEF`, `BOTH-RELIEF` start semantically identical except capacity flags;
3. below both caps, all four slow branches remain state-identical;
4. when trace cap first binds, only branches with capped traces remove a trace;
5. when slot cap first binds, only branches with capped slots remove a mature slot;
6. nonbinding trace branches never emit `trace_capacity_removal`;
7. nonbinding slot branches never emit `slot_capacity_eviction`;
8. capped branches never exceed their hard limits;
9. BASE is updated exactly once per byte and is identical across branches;
10. FAST is updated exactly once per byte and is identical across branches;
11. slow admission always receives the inherited BASE scores, never FAST or FIXED-50;
12. branch-specific read policy scoring does not modify any physical learner;
13. probes do not change hits, recency, RNG, evidence ledgers or physical state;
14. target/evaluator provenance is absent from every learner method signature and serialized physical state;
15. target-candidate audit observes but never replaces the locked winner;
16. trace admission/removal conservation holds;
17. slot allocation/eviction conservation holds;
18. monotone generation UIDs are never reused;
19. interruption/resume reproduces uninterrupted hashes and outputs;
20. source/protocol/plan mismatch rejects checkpoint reuse.

`--engineering` should additionally run at least two non-science seeds through load 256 and force both trace and slot overflow, verifying all four branches against an independently executed single-branch reference.

---

# 17. Baseline reproduction requirement

Because V44 reuses V43 seeds, baseline reproduction is unusually strong and should be exploited.

For `CAP-BASE`, require equality to V43 for every quantity that the V43 handoff permits exact comparison. If the V43 archive lacks a full physical checkpoint needed for one field, do **not** fabricate an equality claim; instead rerun the frozen V43 core and compare the two implementations on the same stream.

The V44 report must enumerate:

- fields compared exactly;
- fields compared numerically;
- fields unavailable from the saved V43 handoff;
- any rerun used to restore an exact reference.

No V44 causal result is interpreted until this audit passes.

---

# 18. Runner contract

One self-contained executable file, interruption-friendly and resumable.

Suggested interface:

```bash
python3 elm_v44_capacity_diagnosis.py --self-test
python3 elm_v44_capacity_diagnosis.py --engineering
python3 elm_v44_capacity_diagnosis.py --stage forensic --out-dir v44_results
python3 elm_v44_capacity_diagnosis.py --stage causal   --out-dir v44_results
python3 elm_v44_capacity_diagnosis.py --stage all      --out-dir v44_results
```

`--stage causal` is the core V44 experiment. `--stage forensic` is retrospective V43 closure.

A single causal seed should run all four slow branches in one stream pass.

Minimal pseudocode:

```python
for byte_event in stream:
    # Shared physical experts, pre-outcome.
    z, bank = base_encoder.encode(history)
    base_scores = base.scores(z, bank)
    fast_scores = fast.scores(history)

    packets = {}
    readouts = {}

    for branch in slow_branches:
        packets[branch] = branch.lookup(history, base_scores)
        readouts[branch] = score_virtual_read_policies(
            packet=packets[branch],
            fast_scores=fast_scores,
            branch_local_evidence=ledgers[branch],
        )

    assert shared_state_unchanged_by_scoring()

    # Outcome revealed only here.
    y = byte_event.value
    score_losses(readouts, y)
    append_post_outcome_evaluator_fields(y)

    # Shared learners update once.
    base.observe(z, bank, y)
    fast.observe(history, y)

    # Slow branches now diverge only through capacity interventions.
    for branch in slow_branches:
        transition = branch.observe(packets[branch], y)
        update_branch_local_read_evidence_after_outcome(...)
        log_transition_without_feeding_evaluator_metadata_back(...)

    assert base_and_fast_hashes_common_across_branches()
```

The exact update order must match V43/V41 wherever inherited state depends on order. The implementation may not replace this with a prose-derived approximation.

---

# 19. Staging and stopping rules

1. Run `--self-test`.
2. Run engineering references until exact invariants and V43 semantics pass.
3. Run Stage-F load-256 forensic replay first and reconcile V43 baseline.
4. Run all 16 Stage-C causal seeds; no interim scientific stopping based on effect direction.
5. Complete Stage-F 64/128/512 forensic replays if not already done.
6. Generate the fixed report and decision-tree outcome.
7. **Stop.** Do not automatically launch a fixed-budget repair.

The fact that a branch looks promising mid-run cannot alter capacities, endpoints, instrumentation, seeds or thresholds.

---

# 20. Interpretation boundaries

A successful V44 capacity diagnosis may establish, on this controlled synthetic stream, that relaxing one or both slow-memory resource constraints causally restores part or all of continued acquisition.

It will **not** establish:

- an optimal eviction policy;
- a deployable unbounded learner;
- that nontarget memories are useless;
- that FAST prediction is a sufficient storage-value signal;
- natural-language persistent learning;
- biological equivalence;
- a minimal-neuron circuit;
- independent replication of V43.

The intended achievement is narrower and more valuable at this stage:

\[
\boxed{
\text{Turn V43's high-load acquisition collapse from a symptom into a localized causal bottleneck.}
}
\]

Only after that localization should the project spend another version on a fixed-budget repair.

---

# 21. Provenance / frozen source facts

This design is based on the locked assessment immediately preceding V44 and the following prior artifacts:

- `v43_analysis_report.md` — completed 64-cell V43 results, acquisition failure, mature coverage, read-policy summaries and interpretation limit;
- `elm_v43_fast_slow_authority.py` / V43 protocol — exact V43 stream semantics, capacities, physical firewall and read-policy separation;
- `elm_v39_earned_consolidation.py` — TRACE-CORR trace/slot state machine, trace LRU, mature `LFU→LRU`, two-anchor promotion and capacity semantics;
- `progress_review_and_next_steps.md` — retrospective evidence ledger and corrections constraining V44;
- `v39_capacity_reaudit_per_seed.csv` — recomputed V39 trace/slot conservation counters used only as historical motivation, not as V44 outcome data.

The following conclusions are explicitly carried forward rather than re-litigated:

1. V43's read-arbitration qualification failed because physical acquisition validity failed at high load.
2. High old recall does not prove continued learning.
3. Aggregate capacity/eviction counters alone do not identify the causal stage of failure.
4. FAST-guided admission is not assumed safe.
5. Fixed-history read counterfactuals cannot substitute for physical branching when replacement changes future state.
6. A fixed-budget repair must be selected *after* causal decomposition and confirmed on a fresh seed roster.

