# MiniFly relevance-core follow-up: audited design specification

**Document status:** IMPLEMENTATION CONTRACT — ROUND-B AMENDMENT AUDITED AND FROZEN BEFORE ROUND B  
**Date:** 2026-09-16
**Design revision:** 1.2 (post-Round-A Round-B amendment; prospective for Round B)  
**Scope:** follow-up to the completed native RCE60 relevance-centred evolution and V87 V84E core-identification screen  
**Purpose:** define the next experiments precisely enough that implementation can be audited against the design before expensive native runs begin.

---

## 0. Executive decision

The next work is **not another global evolution run**.

Round A is now complete and frozen: `RCE_TRACE_RESTORED` passed the relevance-centered noninferiority suite against `RCE_FULL` on 48 fresh paired histories. The remaining authorized experiment is:

1. **Round B — causal operating-regime screen on the confirmed trace-restored RCE baseline.**  
   Test the V87 signals for feedback normalization and lower KC activity in a 2×3 factorial while keeping the RCE structural donor and all other parameters fixed. Add `RCE_FULL` prospectively as a secondary same-world benchmark because Round A exposed a small, consistent acquisition advantage for full RCE.

No Round C evolution is authorized by this document. A new evolution design is written only after Round B is reviewed.

---

# 1. Scientific question and non-question

## 1.1 Scientific question

Can a bounded-state MiniFly learner:

- use information that is behaviorally relevant now;
- preserve such information through subsequent unrelated learning while it remains relevant;
- revise an association when its true consequence changes;
- preserve the revised association through later learning;
- continue acquiring new associations;
- recover an association efficiently when it becomes relevant again after a period of irrelevance?

## 1.2 Explicit non-question

The experiment does **not** ask the learner to retain all previously learned or still-valid associations indefinitely.

The following are **not** primary objectives:

- age-based retention of "old" memories;
- age-based retention of "recent" memories;
- blanket overnight retention;
- retention of every currently valid but behaviorally unused association;
- low clipping as an end in itself;
- any particular fast/slow allocation fraction as an end in itself;
- maximizing write-direction cosine statistics;
- proving global neuronal minimality.

A memory becomes evaluator-relevant because the environment actually demands that cue again. Relevance is evaluator-only; no relevance bit, cue identity, age, future-use tag, or task label is exposed to the learner.

---

# 2. Source-locked facts from the completed experiments

These facts are frozen inputs to the design and must not be silently reinterpreted during implementation.

## 2.1 RCE60 result

- Native RCE60 completed 60 generations.
- Confirmed full candidate:
  - ID: `aadcb43cf170d1d018bf`
  - structural donor: `V81`
- Development-only trace-dynamics-restored candidate:
  - ID: `ca1b0a55a326a7d3e8e7`
  - structural donor: `V81`
  - exactly restores `g_alpha_gain`, `g_fast_tau`, `g_slow_tau`, and `g_fraction` to reference (`0`).
- On the eight development histories, trace restoration matched or improved every primary relevance/plasticity score and anomaly recovery.
- It was rejected solely by the `collateral_survival` reduction gate.
- It has **zero independent confirmation histories** in the completed archive.
- Therefore it is a candidate simplification, not an established core.

## 2.2 V87 result

V87 was an intervention screen on V84E, not a continuing-life confirmation of a new core.

The two most actionable signals were:

- restoring `g_feedback` to reference (`g_feedback = 0`, feedback multiplier `1.0×`);
- reducing KC active fraction to `3.0%` or `2.5%`.

These interventions improved several V87 late-life and challenge outcomes and substantially changed clipping/write-direction diagnostics. They are causal hypotheses to test on the RCE lineage; they are **not** assumed to transfer or combine additively.

## 2.3 Search-domain fact

The inherited phenotype maps

\[
\text{active fraction}=0.05\,2^{g_{\rm sparsity}}.
\]

The old RCE search box was `g_sparsity ∈ [-0.5, 0.5]`, so it could not reach either 3.0% or 2.5%:

- `3.0%` corresponds to `g_sparsity ≈ -0.736966`;
- `2.5%` corresponds to `g_sparsity = -1`.

Round B therefore tests these as **explicit assay overrides**, not by silently widening the genome box.

## 2.4 Completed Round A result and prospective Round-B amendment

Round A has now completed on **48 fresh paired histories** and is frozen. The trace-restored phenotype passed the predeclared seven-endpoint relevance-centered noninferiority suite. The result is not being altered by this amendment.

Frozen native Round-A `DECISION.json` SHA-style integrity field (`decision_hash`):

```text
897e8f969773ab426fb050a8f2cc6e8e77fbd459ff60d5770779b849a111553e
```

The scientific Round-B runner must refuse any different Round-A decision hash.

The paired Round-A result exposed one specific trade-off that is relevant to Round B:

- `RCE_TRACE_RESTORED - RCE_FULL` on P5 acquisition = approximately **-1.587 percentage points**;
- `RCE_FULL` had higher P5 acquisition on **42 of 48** paired histories;
- trace restoration nevertheless matched or improved P1, P2, P3, P4, P6, and P7 in the aggregate and passed all seven predeclared NI margins.

Because Round B has **not started**, it is legitimate to add a future-data benchmark motivated by this completed result, provided the change is explicit and does not alter the original Round-B causal estimand. Therefore this revision adds `RCE_FULL` as a **prospectively declared secondary contextual anchor** on the same 24 Round-B histories.

This amendment obeys four constraints:

1. the original six-cell feedback × activity factorial is unchanged;
2. `RCE_FULL` is excluded from factorial main effects and interactions;
3. comparisons versus `RCE_FULL` are secondary/contextual and create no automatic winner or certificate;
4. Round-A results, thresholds, and interpretation are not recomputed or retroactively modified.

The purpose of the extra anchor is narrow: determine on identical fresh worlds whether a Round-B operating-regime intervention can recover the small acquisition cost seen after trace restoration while retaining the relevance/revision gains.

---

# 3. Frozen candidate definitions

Gene order is the source-locked V82E order:

```text
0  b0
1  b_adapt
2  b_fast
3  b_slow
4  g_alpha_gain
5  g_support
6  g_teach_scale
7  g_readout
8  g_feedback
9  g_mbon
10 g_k0
11 g_fast_tau
12 g_slow_tau
13 g_fraction
14 g_sparsity
15 g_pnkc_weight
16 g_pnkc_rewire
```

## 3.1 REF

Structural donor: `REF`

```text
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

## 3.2 V84E contextual anchor

Structural donor: `V84E`

```text
[ 0.330853,
 -0.750000,
 -0.349499,
 -0.090123,
 -0.152739,
  1.000000,
 -0.233097,
 -0.076526,
  0.602103,
  0.804988,
  0.042987,
  0.054654,
  0.000000,
  0.035756,
 -0.500000,
  0.008166,
  0.000000 ]
```

## 3.3 RCE-FULL

ID: `aadcb43cf170d1d018bf`  
Structural donor: `V81`

```text
[ 0.2769583684375018,
  0.5060575709162071,
  0.7500000000000000,
 -0.5013699223698764,
  0.12733383481229846,
  0.0000000000000000,
  0.0000000000000000,
  0.0000000000000000,
  0.33202778204816097,
 -0.2930507566579318,
 -0.18361794385481717,
  0.3873791237411004,
 -0.2948909452871429,
  0.5068632844705093,
 -0.4850183373885582,
  0.0329640000000000,
  0.0000000000000000 ]
```

Derived operating values used only for audit/display:

- feedback multiplier = `2^0.33202778204816097 ≈ 1.258781413686009×`;
- active fraction = `0.05 × 2^-0.4850183373885582 ≈ 0.03572439941533491` = `3.5724399415%`.

## 3.4 RCE-TRACE-RESTORED

ID: `ca1b0a55a326a7d3e8e7`  
Structural donor: `V81`

```text
[ 0.2769583684375018,
  0.5060575709162071,
  0.7500000000000000,
 -0.5013699223698764,
  0.0000000000000000,
  0.0000000000000000,
  0.0000000000000000,
  0.0000000000000000,
  0.33202778204816097,
 -0.2930507566579318,
 -0.18361794385481717,
  0.0000000000000000,
  0.0000000000000000,
  0.0000000000000000,
 -0.4850183373885582,
  0.0329640000000000,
  0.0000000000000000 ]
```

This is a **parameter restoration**, not deletion of the fast/slow state arrays.

---

# 4. Common continuous-life protocol

Rounds A and B use the same main-life structure unless this document explicitly says otherwise.

## 4.1 Identity progression

Use the executed V86/RCE progression:

```text
48 → 72 → 96 → 120 → 132 → 144 → 156 → 168 identities
```

Equivalent block sizes:

```text
[48, 24, 24, 24, 12, 12, 12, 12]
```

No learner reset is allowed between blocks.

## 4.2 Acquisition

Each newly introduced association receives six acquisition bouts, matching the prior relevance run.

Probe:

- immediately after acquisition;
- after the standard 24-hour rest.

## 4.3 Challenge cadence

Challenges occur after identity totals:

```text
48, 120, 132, 144, 156
```

Each challenge contains 12 demanded identities:

- 4 STABLE;
- 4 ANOMALY;
- 4 REVISION.

Except for the six explicitly planned reactivation STABLE episodes, every challenge target is **challenge-fresh**: it has never appeared in an earlier relevance/challenge context. This eliminates accidental cue reuse as a confound; age of original acquisition may vary, but challenge-use history does not.

Twelve challenge cycles are retained.

- STABLE: teach the current true consequence.
- ANOMALY: teach the wrong consequence for one or two initial cycles, then restore correct evidence.
- REVISION: teach the opposite consequence and then change environmental truth accordingly.

A cue used for genuine revision may not be genuinely revised a second time in the same life.

## 4.4 Relevance semantics

`set_relevance_context` remains evaluator-only.

The current challenge target set remains the active relevance context **until the next challenge changes it**. Therefore subsequent acquisition blocks naturally test whether information that remains behaviorally relevant survives unrelated new writes.

Relevance must not be inferred from introduction age.

## 4.5 Explicit relevance reactivation

The old RCE assay allowed context switching but did not guarantee that a deactivated cue would later become relevant again. The follow-up must guarantee this.

### Reactivation source pool

Only cues that previously appeared in the **STABLE** group are eligible for the reactivation test.

For this section, **fresh** means a cue that has never appeared in any earlier challenge/relevance context. Once a STABLE cue is reserved for a future reactivation episode, it is excluded from every intervening STABLE, ANOMALY, and REVISION target set until that planned reactivation occurs. After the planned reactivation occurs, that cue is excluded from all later challenge target sets as well, so each reactivation episode is unique and cannot later be conflated with a second reuse, anomaly, or revision. This reservation is evaluator-side world construction only and is never exposed to the learner.

A reactivation cue must:

- have been behaviorally relevant in an earlier challenge;
- have subsequently become inactive because a later relevance context replaced it;
- have unchanged environmental truth;
- never have undergone genuine revision;
- not have been used previously as a reactivation cue.

### Reactivation schedule

- Challenge 0: all 4 STABLE cues are fresh.
- Challenge 1: all 4 STABLE cues are fresh.
- Challenge 2: 2 STABLE cues are reactivated from an earlier deactivated STABLE pool; 2 are fresh.
- Challenge 3: 2 STABLE cues are reactivated from an earlier deactivated STABLE pool; 2 are fresh.
- Challenge 4: 2 STABLE cues are reactivated from an earlier deactivated STABLE pool; 2 are fresh.

Thus each full life contains exactly **6 planned reactivation cue episodes**.

### Reactivation probes

For the two reactivated STABLE cues in each applicable challenge:

1. probe **zero-shot**, immediately after the relevance-context switch and before new teaching;
2. teach the correct association once, then probe;
3. after the second correct exposure, probe;
4. after the fourth correct exposure, probe.

These are main-life events: reactivation teaching is legitimate environmental reuse, not a diagnostic-only branch.

Zero-shot reactivation is diagnostic. Recovery by four correct exposures is primary.

## 4.6 Time-only tests

Retain:

- standard 24-hour rests;
- terminal 7-day rest.

Time-only retention is reported separately and never substituted for relevance persistence.

---

# 5. Primary behavioral metrics — corrected definitions

The old RCE primary metrics used history-level minima of conditional ratios. A transition with a tiny denominator could dominate the entire history. That is not used for the next decision.

Every primary metric below is computed **within each history as an explicit count ratio**:

\[
\text{history metric}=\frac{\sum \text{correct eligible behavioral decisions}}{\sum \text{eligible behavioral decisions}}.
\]

The history remains the independent resampling unit for uncertainty intervals. Individual cue events are **not** treated as independent replicates for inference.

Every result record must preserve the numerator and denominator, not only the ratio.

## P1. Current usefulness accuracy

All demanded cue decisions at every **canonical full-context immediate checkpoint** while a nonempty relevance context is active, including immediate probes after later acquisition blocks as well as completed challenge probes. Reactivation micro-probes at 0/1/2/4 exposures are excluded from P1 and belong only to the reactivation curve/P7; otherwise two reactivated cues would be overweighted relative to the rest of the demand context. This is essential because the point is to test whether currently useful information survives ongoing life, not merely whether it is correct at the instant a challenge is administered.

Numerator: demanded cues answered correctly.  
Denominator: all demanded cue decisions at those immediate probes.

## P2. Relevant-after-write accuracy

Tests relevant information after intervening writes without conditioning on whether the candidate happened to be correct before.

A **write interval** is the interval from a completed canonical day checkpoint to the next canonical full-context immediate checkpoint after one or more teaching events. Reactivation micro-probes do not terminate or create a P2/P3 interval. The interval is eligible cue-by-cue when:

- the cue is relevant at the starting checkpoint;
- the cue is still relevant at the ending immediate probe;
- environmental truth is unchanged across the interval.

This naturally includes the important case where a relevance context persists while the learner acquires unrelated new identities. A cue that is retired by a relevance-context switch receives no persistence credit across that switch.

Measure correctness after the interval.

This replaces `min(conditional_survival)` as the primary persistence endpoint.

## P3. Revised-relevant-after-write accuracy

Same as P2, restricted to cues whose environmental truth had previously been genuinely revised.

This requires both successful correction and persistence of the corrected behavior; immediate revision is still measured separately as P4.

## P4. Revision accuracy

Correctness on genuine REVISION cues at the immediate post-challenge probe under the new truth.

## P5. Acquisition accuracy

Correctness on newly introduced associations at immediate post-acquisition probes.

## P6. Anomaly recovery accuracy

Correctness on ANOMALY cues after the challenge has returned to correct teaching.

This is a plasticity/specificity guard, not a persistence endpoint.

## P7. Reactivation recovery by four exposures

Correctness on the six planned reactivation episodes after four new correct exposures.

This tests whether a previously useful but later inactive association can return to useful behavior efficiently without requiring blanket preservation throughout irrelevance.

---

# 6. Required secondary and diagnostic metrics

These metrics are mandatory to record but do **not** enter the Round A simplification gate unless explicitly stated.

## 6.1 Reactivation curve

For each reactivation episode:

- zero-shot;
- after 1 exposure;
- after 2 exposures;
- after 4 exposures.

Report count ratios and per-history distributions.

## 6.2 Conditional survival with explicit denominators

Retain candidate-conditional useful and revised-useful survival for mechanistic interpretation:

```text
correct after | candidate was correct before
```

Every row must contain:

- eligible truth-unchanged count;
- correct-before count;
- correct-after-among-correct-before count;
- resulting conditional ratio.

These are diagnostic only. No history minimum of a tiny denominator may become a primary pass/fail statistic.

## 6.3 Worst-transition stress diagnostic

Keep the old worst-transition statistic as a stress-test descriptor only.

It may identify brittle histories but may not single-handedly reject a candidate.

## 6.4 Collateral/unrelated information

Record accuracy and conditional survival for currently unused, still-valid associations.

Interpretation:

- useful for detecting indiscriminate state destruction;
- **not** a retention entitlement;
- **not** a Round A noninferiority gate;
- **not** a Round B optimization objective.

## 6.5 Time-only survival

Report 24-hour and 7-day behavior separately for:

- current relevant cues;
- revised relevant cues;
- all still-valid cues.

No generic overnight-retention statistic may be relabeled as relevant-memory persistence.

## 6.6 Obsolete outcome behavior

For genuinely revised cues, record whether the old outcome remains behaviorally preferred after the revision. This is diagnostic of failed overwrite, not an additional independent primary metric.

## 6.7 Untrained-cue behavior

Record fixed-reader and actual-only responses on untrained cues.

The old `abs(actual) >= 1` response frequency remains descriptive only. It is not called a hallucination rate and is not a core gate.

## 6.8 Clipping and write allocation

Record, but do not optimize directly:

- clipping fraction;
- cumulative fast/slow L1 write mass;
- mass-weighted slow fraction;
- allocation statistics (`z` or equivalent native quantity);
- expression margin diagnostics.

---

# 7. Round A — corrected confirmation of RCE trace restoration

## 7.1 Arms

Exactly four normal arms:

1. `REF`
2. `V84E`
3. `RCE_FULL` = `aadcb43cf170d1d018bf`
4. `RCE_TRACE_RESTORED` = `ca1b0a55a326a7d3e8e7`

The causal simplification comparison is **RCE_TRACE_RESTORED versus RCE_FULL**. REF and V84E are contextual anchors, not part of the simplification definition.

## 7.2 Structural-donor rule

`RCE_FULL` and `RCE_TRACE_RESTORED` must use the identical source-locked `V81` PN→KC donor matrix.

The B-matrix hash must match exactly between those two arms on every history.

No PN→KC mutation, jitter, or rewiring occurs in Round A.

## 7.3 Histories

Use **48 new paired histories**.

All four arms see exactly the same 48 worlds.

History generation must be complete and sealed before candidate outcomes are inspected.

### Seed derivation

Do not hand-pick sequential seeds after seeing outcomes.

For history index `i = 0..47`, derive the seed deterministically from:

```text
SHA256("MiniFly-RCEV87-FOLLOWUP-A|20260916|" + str(i))
```

Use the first 64 bits as an unsigned integer accepted by NumPy's generator. Save the resolved seed list in the run manifest before evaluation starts.

## 7.4 Raw-cue disjointness

Exact raw cue identities must be disjoint:

- within each new world;
- across all 48 Round A worlds;
- from Round B worlds;
- from prior RCE60 worlds;
- from V87 input cues when those raw inputs are available from the archived `inputs.npz`.

Representation overlap after PN→KC encoding is allowed and is scientifically meaningful.

## 7.5 Round A decision rule

Round A asks only whether the simpler trace-restored candidate is confirmed noninferior to the full RCE candidate on the relevance suite.

For each of P1–P7 compute the paired history difference:

\[
D_h=M_h(\text{TRACE})-M_h(\text{FULL}).
\]

Use a paired whole-history bootstrap with **50,000 resamples** and a one-sided 95% lower confidence bound. Resolve the bootstrap RNG seed deterministically from:

```text
SHA256("MiniFly-RCEV87-FOLLOWUP-A-bootstrap|20260916")
```

and record it in the analysis manifest.

Noninferiority margin:

```text
δ = 0.02
```

The trace-restored candidate passes Round A only if, for **every** P1–P7:

```text
lower_95_one_sided(D) >= -0.02
```

and its numeric state allocation is not larger than RCE-FULL.

### Statistical note

This is a conjunctive/intersection-union noninferiority claim: the global claim is made only if **all** seven component null hypotheses are rejected. Do not add the old redundant `useful_churn = 1 - useful_survival` as an eighth test. Do not separately gate collateral survival.

The same bootstrap history index draw must be reused across arms within each resample.

## 7.6 Round A labels

If the rule passes, the permitted label is:

```text
TRACE_RESTORATION_CONFIRMED_NONINFERIOR_ON_RELEVANCE_SUITE
```

If it fails:

```text
TRACE_RESTORATION_NOT_CONFIRMED
```

Neither label contains the phrase `minimal core`.

## 7.7 Learning-dependence controls

On the first 24 prespecified Round A histories, run `RCE_FULL` under:

- `no_learning`;
- `unpaired` teaching.

These are assay-validity controls. They are reported against P1/P4/P5/P7 and final relevant behavior. They do not create a separate candidate-selection objective.

---

# 8. Round A fast/slow causal diagnostics

The prior terminal lesion occurred after a 7-day interval, so absence of a fast-state effect at that point does not show that fast state was unnecessary during learning.

All lesion measurements are clone-only. No lesioned clone re-enters the continuing life.

## 8.1 Checkpoints

For each normal Round A arm, save diagnostic snapshots at:

1. `challenge2_immediate` — immediately after the relevance challenge at 132 identities;
2. `challenge2_plus_one_fast_tau` — a clone of checkpoint 1 rested for exactly the candidate's native `fast_tau` read from phenotype metadata;
3. `block5_immediate` — after the next 12 newly acquired identities, before the next relevance challenge;
4. `final_7day`.

## 8.2 Lesions

At every diagnostic snapshot probe current relevant cues under:

- intact clone;
- fast state zeroed;
- slow state zeroed;
- both zeroed.

Report absolute accuracy and drop relative to the intact clone.

No lesion result participates in the Round A simplification gate. It is a mechanistic test of when each state is required.

---

# 9. Round B — feedback × activity factorial on the RCE lineage

Round B is run only after Round A results are frozen. The runner does **not** auto-start Round B: it requires a separate explicit invocation and validates the integrity hash of the completed Round-A decision before generating Round-B worlds.

## 9.1 Baseline selection

If Round A passes, `BASE = RCE_TRACE_RESTORED`.

Otherwise, `BASE = RCE_FULL`.

The selected BASE specification and Round A result hash are written into the Round B manifest before Round B worlds are generated.

## 9.2 Fixed structure

All six factorial arms use:

- the exact same `V81` PN→KC donor matrix;
- the exact BASE gene vector except for `g_feedback` when explicitly changed;
- no support modification;
- no allocation modification;
- no PN→KC modification.

The six factorial arms must have identical B hashes.

## 9.3 Factor 1: feedback

Two levels:

### F-native

Leave BASE `g_feedback` unchanged.

For both possible BASE candidates:

```text
g_feedback = 0.33202778204816097
feedback multiplier ≈ 1.258781413686009×
```

### F-reference

Set only:

```text
g_feedback = 0
```

which gives feedback multiplier `1.0×`.

No other expression gene is restored. In particular, this is **not** the V87 `block_REF_expression` intervention.

## 9.4 Factor 2: active fraction

Three levels:

1. `A-native`: no override; BASE uses ≈ `3.5724399415%`;
2. `A-3pct`: exact assay override `active_fraction = 0.030`;
3. `A-2p5pct`: exact assay override `active_fraction = 0.025`.

For the two override arms, the stored genome itself is unchanged at `g_sparsity = -0.4850183373885582`.

The override is an explicit intervention field in the arm specification and must be included in the arm ID/hash. It must not masquerade as a legal V82E genome value and must not silently widen the gene box.

Native phenotype metadata must record both:

- genome-derived active fraction;
- effective overridden active fraction.

## 9.5 Factorial arms

Exactly six causal arms:

| Arm | feedback | active fraction |
|---|---:|---:|
| `Fnative_Anative` | BASE ≈1.25878× | BASE ≈3.57244% |
| `Fnative_A3` | BASE ≈1.25878× | 3.0% |
| `Fnative_A2p5` | BASE ≈1.25878× | 2.5% |
| `Fref_Anative` | 1.0× | BASE ≈3.57244% |
| `Fref_A3` | 1.0× | 3.0% |
| `Fref_A2p5` | 1.0× | 2.5% |

Add three contextual anchors on the same histories:

- `REF`;
- `V84E`;
- `RCE_FULL` as a **secondary post-Round-A benchmark**.

`RCE_FULL` is added because Round A showed a small but highly consistent acquisition advantage for full RCE relative to trace restoration. It is not a seventh factorial cell.

All contextual anchors are **excluded** from factorial main-effect and interaction calculations. `REF` and `V84E` differ in many parameters (and REF also differs in structure). `RCE_FULL` differs from the trace-restored baseline only in the four trace-dynamics parameters already tested in Round A.

If Round A had failed and selected `RCE_FULL` as BASE, the runner must not create a duplicate `RCE_FULL` anchor; in the completed native Round A, BASE is `RCE_TRACE_RESTORED`, so the scientific Round-B run contains **nine normal arms: six factorial cells plus three contextual anchors**.

## 9.6 Histories

Use **24 new paired histories**, disjoint from Round A and prior experiments.

For history index `i = 0..23`, derive seed from:

```text
SHA256("MiniFly-RCEV87-FOLLOWUP-B|20260916|" + str(i))
```

Again save resolved seeds before evaluation.

## 9.7 Round B analysis

Round B is a **causal screen**, not an automatic winner-selection stage.

Report P1–P7 for every arm and paired history differences for:

### Feedback contrasts

At each activity level:

```text
F-reference - F-native
```

### Activity contrasts

Within each feedback level:

```text
A-3pct - A-native
A-2p5pct - A-native
```

### Interaction contrasts

For both lower-activity levels, report the difference-in-differences:

```text
[(Fref, Alow) - (Fnative, Alow)]
-
[(Fref, Anative) - (Fnative, Anative)]
```

### Secondary same-world RCE-FULL contrasts

For every one of the six factorial cells, report paired differences versus `RCE_FULL` on P1-P7. These are **prospectively declared secondary contextual comparisons**, motivated by the frozen Round-A acquisition trade-off.

The key interpretive question is whether any operating-regime cell improves P5 acquisition relative to the trace-restored native cell and approaches or exceeds `RCE_FULL` on the same worlds **without materially sacrificing the relevance-centered endpoints**.

No threshold is used to select a winner automatically. In particular, these comparisons do not enter the factorial main-effect/interaction calculations and do not change the Round-B status `SCREEN_COMPLETE_NO_AUTOMATIC_WINNER`.

Use paired whole-history bootstrap intervals. Report effect size, interval, sign consistency across histories, and raw count totals.

**No scalar fitness score and no automatic `winner` label are permitted.**

A separate review decides whether the data justify a new search domain.

---

# 10. Round B mechanism diagnostics

V87's result archive did not include the exact runner source. Therefore these are explicitly defined new probes inspired by the archived V87 measurements; they must not be described as a byte-identical reproduction of the V87 probe implementation.

All mechanism branches are clone-only and never return to the continuing life.

## 10.1 Probe stages

Run mechanism probes at:

- after the 96-identity day checkpoint;
- after the 168-identity day checkpoint, before the terminal 7-day rest.

## 10.2 Deterministic cue choice

Cue identities for mechanism branches are selected from the world specification before model outcomes are known.

Do not choose a cue because one arm happens to answer it correctly.

## 10.3 Write vectors

For each teaching operation record:

```text
Δfast = alpha_fast_after - alpha_fast_before   # native m.fast[:,1]
Δslow = slow_after - slow_before               # native m.slow
```

and their L1 norms.

## 10.4 Probe scenarios

### M1. Repeat-same

From a common snapshot:

1. teach the correct consequence for one predetermined current cue; record `Δslow_1`;
2. from the resulting branch, teach the same cue/consequence again; record `Δslow_2`;
3. report cosine(`Δslow_1`, `Δslow_2`).

### M2. Current-versus-new

From two identical copies of the same snapshot:

- branch A: one correct teaching event on a predetermined current cue;
- branch B: one correct teaching event on a predetermined cue from the world's reserved **never-trained/untrained pool**, so the probe never depends on which future training block happens to remain.

Report cosine between their slow-write vectors.

### M3. Correct-versus-reverse

From two identical copies of the same snapshot and the same predetermined current cue:

- branch A: teach the current true consequence;
- branch B: teach the opposite consequence.

Report cosine between slow-write vectors.

## 10.5 Allocation diagnostics

For every mechanism event report native allocation quantities, including:

- active-cell count;
- alpha/write L1;
- fast and slow L1;
- mean and quantiles of the slow-allocation fraction where available;
- adaptation-state summaries where available.

These are explanatory measurements only.

---

# 11. Observer/readout contract

To preserve comparability with the corrected RCE runner:

1. Primary correctness uses the **fixed REF-Q reader** plus the native actual-effect threshold, exactly as in the RCE native adapter.
2. Actual-only behavior is also recorded separately.
3. Candidate-coupled reader results may be recorded diagnostically but never replace the fixed observer as the judge.
4. The REF-Q hash must be identical across all arms in a round.
5. Active-fraction overrides may change encoding, but never the observer definition.

---

# 12. Integrity, source-lock, and reproducibility contract

The implementation must fail closed on any violation below.

## 12.1 Native source lock

Use the same source-locked V82E phenotype equations and inherited MiniFly dependencies as the hardened RCE run unless a new source version is intentionally declared.

Record SHA-256 hashes for:

- `model_evo.py`;
- `common_evo.py`;
- V82E config;
- the new runner;
- the design Markdown;
- candidate/arm manifest.

The follow-up package also contains an exact collision-free two-`uint64` representation of **120,744 previously used 88-bit raw cue patterns**, assembled from the pre-RCE exclusion set, all archived native RCE60 world files, and V87 `inputs.npz`. Its file hash is source-locked in the run manifest. Round-A worlds must be disjoint from this set; Round-B worlds must additionally be disjoint from every Round-A raw cue.

No silent fallback to a fixture backend is allowed for a scientific run.

## 12.2 Candidate validation

Before any worker process starts:

- validate exact 17-gene shape;
- validate all native genes against the frozen V82E boxes;
- validate donor name;
- validate structural genes against declared donor;
- validate candidate ID/hash;
- validate explicit intervention fields separately from genome fields.

## 12.3 Structural invariants

For every causal pair/group that is supposed to share structure:

- assert exact sparse B hash equality;
- assert identical B shape and nonzero structure;
- assert no topology or weight mutation occurred.

## 12.4 World invariants

Before candidate outcomes are evaluated:

- generate and seal all worlds for that round;
- verify exact raw cue uniqueness;
- verify reactivation schedule exactly six episodes per full life;
- verify every reactivation cue previously left the relevance context;
- verify reactivation truth never changed;
- verify no cue is reactivated twice;
- verify genuine revision cues are not revised twice;
- verify all arms receive byte-identical world specifications.

## 12.5 Learner-blind relevance

Automated source/test inspection must demonstrate that:

- `set_relevance_context` is handled only by evaluator bookkeeping;
- no relevance index or tag is passed into learner methods;
- no candidate receives introduction block, age rank, future demand, or evaluator correctness.

## 12.6 Numeric-state bound

Inventory numeric learner state at life start and end.

Array shapes/dtypes and total allocated learner-state bytes must not grow during life.

Diagnostic result logs do not count as learner memory.

## 12.7 Receipts

Every completed arm × history life receives a sealed receipt containing at minimum:

- arm ID and exact intervention specification;
- candidate gene vector;
- donor and B hash;
- REF-Q hash;
- world hash;
- source-code hashes;
- environment versions;
- raw metric numerators/denominators;
- final state inventory;
- completion seal/hash.

---

# 13. Required hostile tests before native production

The code is not accepted because it merely runs one friendly preflight.

## 13.1 Unit/contract tests

At minimum test:

1. exact RCE-FULL vector and ID;
2. exact TRACE vector and ID;
3. trace restoration changes only genes 4, 11, 12, 13;
4. all gene-bound violations fail before worker submission;
5. Round B feedback restoration changes only `g_feedback` plus the resulting `F` scale;
6. active-fraction overrides do not change the stored gene vector;
7. active-fraction override changes only the effective coding fraction, not B;
8. all six factorial arms share identical B hash;
9. relevance tags never enter learner calls;
10. current relevance persists until the next context switch;
11. exactly six reactivation episodes occur;
12. reactivation cues are previously stable, deactivated, truth-unchanged, and unique;
13. P1–P7 use count pooling rather than transition minima;
14. every primary metric stores numerator and denominator;
15. `collateral_survival` is absent from Round A pass/fail logic;
16. `useful_churn` is absent as a redundant gate;
17. untrained response rate is descriptive only;
18. clipping and write cosines are descriptive only;
19. histories are paired byte-for-byte across arms;
20. exact raw cue identities are disjoint across splits/rounds/prior inputs;
21. state-reset probes operate on clones only;
22. mechanism write probes operate on clones only;
23. source-hash mismatch fails closed;
24. fixture backend cannot silently satisfy a native-run request.

## 13.2 Adversarial semantic tests

Construct synthetic result cases proving that:

- a candidate with better P1–P7 but lower collateral survival is **not** rejected by the Round A gate;
- a candidate with perfect overnight retention but poor current usefulness/revision **does** fail;
- a candidate that forgets an inactive cue but recovers it by four exposures can pass reactivation;
- a candidate preserving the obsolete pre-revision outcome does not receive revision credit;
- a one-item conditional-survival failure cannot override otherwise strong count-pooled primary behavior.

## 13.3 Native construction preflight

Before the production batch:

- construct every Round A arm natively;
- run one complete Round A life per arm;
- verify all source/B/Q hashes;
- verify P1–P7 denominators are nonzero;
- verify all clone probes complete and do not change main-state hashes.

Before Round B:

- construct all eight arms natively;
- run one complete Round B life per arm;
- assert effective active fractions exactly match specification;
- assert feedback scales exactly match specification;
- assert six factorial B hashes are identical.

## 13.4 Determinism tests

On a reduced engineering configuration, require exact final-output equality for:

- one worker versus two workers;
- uninterrupted versus stop/resume;
- repeated run with identical manifest.

A scientific run must never reuse a cached result whose arm/world/source hash differs.

---

# 14. Outputs required from Round A

At minimum:

```text
STATUS.json
RUN_MANIFEST.json
SOURCE_HASHES.json
ARM_MANIFEST.json
WORLD_MANIFEST.json
WORLD_PARTITION_AUDIT.json
NORMAL_RESULTS.jsonl.gz   # or equivalent sealed receipts
REACTIVATION_EVENTS.csv
RELEVANCE_TRANSITIONS.csv
TIME_ONLY_DIAGNOSTICS.csv
LESION_DIAGNOSTICS.csv
ARM_SUMMARY.csv
PAIRED_NONINFERIORITY.csv
CONTROL_SUMMARY.csv
REPORT.md
```

`PAIRED_NONINFERIORITY.csv` must include for P1–P7:

- full mean;
- trace mean;
- paired mean difference;
- one-sided 95% lower bound;
- NI margin;
- pass/fail;
- history count;
- pooled numerator/denominator for each arm.

---

# 15. Outputs required from Round B

In addition to the previously frozen Round-B outputs, revision 1.2 requires:

- `CONTEXTUAL_ANCHOR_SUMMARY.csv` — P1-P7 summaries for REF, V84E, and RCE-FULL;
- `CONTEXTUAL_ANCHOR_CONTRASTS.csv` — paired P1-P7 differences for each six-cell factorial arm versus RCE-FULL;
- `ARM_SUMMARY.csv` — all nine scientific Round-B arms;
- `REPORT.md` — must explicitly label RCE-FULL as a post-Round-A prospective secondary anchor and state that it did not enter factorial inference.


At minimum:

```text
STATUS.json
RUN_MANIFEST.json
SOURCE_HASHES.json
ARM_MANIFEST.json
WORLD_MANIFEST.json
NORMAL_RESULTS.jsonl.gz
REACTIVATION_EVENTS.csv
RELEVANCE_TRANSITIONS.csv
FACTORIAL_CELL_SUMMARY.csv
FACTORIAL_PAIRED_CONTRASTS.csv
CONTEXTUAL_ANCHOR_SUMMARY.csv
CONTEXTUAL_ANCHOR_CONTRASTS.csv
MECHANISM_EVENTS.csv
WRITE_DIRECTION.csv
LESION_DIAGNOSTICS.csv
REPORT.md
```

The report may say that a setting is promising or harmful on particular measured endpoints. It may **not** automatically declare a persistent core or launch a new evolution.

---

# 16. What is deliberately postponed

The following are outside this design:

- new 50–60 generation evolution;
- lowering mutation sigma below the previous 0.4 floor;
- changing PN→KC topology;
- Level-2 rewiring;
- dynamic structural eligibility grammar;
- expanding teaching support as a new optimization axis;
- treating V84E's exact allocation coefficients as frozen biology;
- widening the V82E `g_sparsity` genome box in production evolution;
- claiming that fast state is unnecessary because its terminal 7-day lesion is weak;
- claiming that lower clipping is itself the mechanism of improvement.

Any one of these requires a separate post-A/B design decision.

---

# 17. Hostile design audit

The following audit was performed against the completed RCE60/V87 archives and the exact hardened RCE runner available in the source bundle.

| Risk / previous failure mode | Design response | Status |
|---|---|---|
| Unrelated-memory retention accidentally acts as a hard persistence objective | Removed from Round A gate; retained only diagnostically | RESOLVED |
| `useful_churn = 1 - useful_survival` counted as a separate confirmation failure | Removed from decision family | RESOLVED |
| Tiny candidate-specific conditional denominators dominate via history minimum | Primary persistence replaced by pooled count-based behavioral accuracy; conditional ratios remain diagnostic | RESOLVED |
| Current usefulness sampled only at challenge moments, missing damage from later acquisition | P1 includes every canonical full-context immediate checkpoint while relevance is active, including post-acquisition probes | RESOLVED |
| Reactivation micro-probes accidentally overweight P1 | Micro-probes are excluded from P1/P2/P3 and feed only the reactivation curve/P7 | RESOLVED |
| “Relevant” silently means old/recent/currently valid | Relevance defined only by environmental behavioral demand | RESOLVED |
| Selective forgetting rewarded without testing future reuse | Explicit deactivation/reactivation episodes with zero/1/2/4-exposure curve | RESOLVED |
| Reserved future-reactivation cue accidentally reused in an intervening challenge | Fresh/reserved target semantics explicitly forbid intervening reuse; a cue is retired from later challenges after its one planned reactivation | RESOLVED |
| Incidental challenge reuse masquerades as relevance reactivation | Every non-reactivation challenge target is challenge-fresh | RESOLVED |
| Reactivation conflated with true revision | Reactivation pool restricted to prior STABLE, truth-unchanged, never-revised cues | RESOLVED |
| Round-B design changed after seeing Round A | Change is prospectively frozen before any Round-B data; only adds RCE-FULL as secondary anchor; six-cell factorial estimand unchanged | RESOLVED |
| V87 feedback result assumed to transfer to RCE | Tested causally on RCE in Round B; no transfer assumption | RESOLVED |
| V87 2.5%/3% sparsity settings inaccessible in old RCE gene box | Use explicit assay override; do not pretend genome box was widened | RESOLVED |
| Feedback and sparsity effects assumed additive | 2×3 factorial includes interaction contrasts | RESOLVED |
| Different structures confound causal factorial | All six factorial cells share V81 donor/B hash | RESOLVED |
| REF/V84E anchors accidentally enter factorial causal effect | Explicitly excluded from factorial contrasts | RESOLVED |
| Terminal 7-day reset used to infer fast-state irrelevance during learning | Add immediate, one-fast-tau, post-new-learning, and final lesion probes | RESOLVED |
| Clipping/write cosine becomes proxy fitness | Explicitly diagnostic only | RESOLVED |
| V87 runner source absent from result archive | New mechanism probes are explicitly defined and not claimed as byte-identical V87 reproduction | RESOLVED |
| Friendly preflight misses invalid arms/operator behavior | Contract tests + adversarial semantic tests + full native construction preflights | RESOLVED |
| Candidate intervention silently changes multiple genes | Exact vectors and intervention deltas frozen in manifest/tests | RESOLVED |
| Active-fraction override hidden inside genome | Intervention stored separately and included in arm identity/hash | RESOLVED |
| World/cue leakage between rounds | Precomputed sealed worlds + exact raw-cue disjointness audit against a bundled 120,744-pattern prior set; Round B additionally excludes all Round-A cues | RESOLVED |
| Round B starts before Round-A interpretation is frozen | Separate explicit invocation; Round-A decision hash must validate before Round-B world generation | RESOLVED |
| Relevance tag leaks to learner | Explicit learner-blind contract and source-level test | RESOLVED |
| Serial/parallel/resume divergence | Determinism tests required before production | RESOLVED |
| Round B screen automatically launches another evolution | Prohibited; new evolution requires separate reviewed design | RESOLVED |

## 17.1 Remaining limitations that are not design bugs

1. **Finite synthetic-life benchmark.** Passing it does not establish indefinite open-world memory.
2. **Evaluator-defined demand.** The learner is tested under changing behavioral demand, but this experiment does not yet isolate a general autonomous relevance-inference mechanism.
3. **Round A is confirmatory but finite.** Trace restoration is independently supported on the frozen 48-history relevance suite, not proven optimal in an open world.
4. **V87 mechanism source was not bundled.** Exact historical mechanism-probe implementation cannot be reconstructed with source-level certainty from the V87 result archive alone.
5. **Round B is a screen.** Twenty-four paired histories can identify useful directions, but this design deliberately does not convert a screened cell into a final core certificate.
6. **Parameter restoration is not neuronal deletion.** Even a successful trace restoration identifies a simpler parameterization, not a globally minimal circuit.
7. **RCE_FULL contextual comparisons are post-Round-A motivated.** They are prospective for fresh Round-B data but remain secondary and must not be rebranded as part of the original factorial hypothesis family.

---

# 18. Implementation acceptance checklist

The implementation is ready for a native production run only after every item below is checked and archived.

```text
[ ] Runner version and design SHA-256 recorded
[ ] Source hashes match source-locked V82E environment
[ ] Exact arm vectors/IDs reproduce this document
[ ] Round A worlds generated before outcomes and all 48 are unique
[ ] Bundled 120,744-pattern prior exclusion file hash matches manifest
[ ] Prior RCE/V87 exact raw cues excluded exactly (not probabilistically)
[ ] Six reactivation episodes verified in every world
[ ] Relevance never enters learner calls
[ ] P1–P7 count formulas unit-tested
[ ] Reactivation 0/1/2/4 micro-probes cannot enter P1/P2/P3
[ ] No collateral gate exists
[ ] No redundant useful_churn gate exists
[ ] Round A paired NI implementation matches Section 7.5
[ ] All Round A B/Q hashes satisfy contract
[ ] Lesion branches never modify continuing state
[ ] Serial/parallel/resume engineering tests match exactly
[ ] Full native Round A preflight succeeds for all four arms
[ ] Round B cannot start without frozen Round A result hash
[ ] Completed Round-A decision version/hash accepted explicitly by amended runner
[ ] Scientific Round B contains six factorial arms + REF + V84E + RCE_FULL (9 normal arms)
[ ] RCE_FULL absent from FACTORIAL_CELL_SUMMARY and FACTORIAL_PAIRED_CONTRASTS
[ ] RCE_FULL present in contextual anchor summary/contrast outputs
[ ] Six Round B factorial arms share exact V81 B hash
[ ] Active-fraction overrides are exact and separately manifested
[ ] Feedback normalization changes only g_feedback/F scale
[ ] Round B mechanism probes are clone-only
[ ] No automatic evolution trigger exists in code
```

---

# 19. Provenance used for this audited design

The design was checked against the following available artifacts:

- `RCE60_V87_joint_review/README.md`
- `RCE60_V87_joint_review/AUDIT.json`
- `RCE60_V87_joint_review/CANDIDATES_FROM_EXISTING_RESULTS.json`
- `RCE60_V87_joint_review/rce_reduction_gate_audit.csv`
- `RCE60_V87_joint_review/rce_rejected_simpler_candidate.csv`
- `RCE60_V87_joint_review/rce_parameter_reversion_development.csv`
- `RCE60_V87_joint_review/rce_relevant_transitions.csv`
- `RCE60_V87_joint_review/v87_late_summary.csv`
- `RCE60_V87_joint_review/v87_write_direction_means.csv`
- exact hardened RCE runner: `joint_review/rce/minifly_relevance_core_evolution60_hardened.py`
- native V82E phenotype source: `joint_review/native_source/model_evo.py`
- V87 result report and manifests under `joint_review/v87/V87_CORE_ID/`

No new native MiniFly simulation was performed to write this design. The document specifies the next experiment; it does not convert post-hoc findings into confirmed biological conclusions.
