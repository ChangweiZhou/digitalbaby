# V50 — Integrated Functional Fly Persistent-Learning Core
## One maintained neuron-level reference system before any further mechanism search

**Status:** refined release design / implementation-frozen before science  
**Date:** 11 September 2026  
**Primary evidence source for connectivity:** uploaded BANC v888 raw metadata and edge list  
**Historical basis:** complete V20–V49 mechanism audit, including early ELM/neuron implementations and the separate local-learning thread  
**Target platform:** Apple Silicon Mac M3, macOS, Python 3.11+

---

# 0. Research constitution — no drift

The fixed project objective is:

> Build a **minimal, bounded-state, biologically based fly-like persistent-learning core** that learns online from sequential experience, preserves still-useful knowledge through intervening learning, generalizes beyond exact presentations, revises genuinely changed associations, and does not require replay, Transformer training, or unbounded object storage.

V50 is **not** another mechanism tournament.

It exists to repair the central failure of V20–V49 project execution:

> many useful components were tested in isolation, but we repeatedly changed the computational object instead of maintaining one functioning reference learner and improving or reducing it.

V50 therefore freezes one coherent reference circuit and demands one repeatable learning phenotype before any additional fly motif, slot policy, representation trick, routing system, eligibility rule, information gate, or large graph simulation is introduced.

The design may not be reinterpreted as:

- a slot-memory experiment;
- a topology-superiority contest;
- a whole-byte-model benchmark;
- a ranking of all historical neuron families;
- evidence that the selected neuron equation is biologically exact.


## 0.1 Release refinements completed before scientific execution

The first implementation pass exposed one numerical mismatch inherited from V49: `eta=4.0` was calibrated for a 256-way byte output interface and caused a single contradictory 8-way cue consequence to overwrite an otherwise stable association.

Using only disjoint engineering seeds `50971` and `50972`, the **same local rule** was checked at smaller scales. `eta=0.5` was frozen because it preserved later-cohort acquisition and eight-exposure revision while removing the one-update overwrite pathology on both engineering seeds. No graph parameter, neuron parameter, cue generator parameter, or science seed was used in this calibration.

Therefore the V50 release fixes:

```text
KC→MBON learning rate eta = 0.5
```

The initial V50 release also deliberately omits the optional `ELM-KC` and `LEAKY-KM-REWIRE` explanatory arms. The exact historical ELM cell was not recovered cleanly enough to label a new implementation as that historical cell, and topology superiority is not part of the functional-core qualification. Both remain later explanatory comparisons after the reference phenotype exists.

The Mac execution is **functional-first**. Stage I runs by default. Stage II byte transfer is a separate command and is scientifically interpreted only after Stage I qualifies. This prevents the expensive serialization stress test from replacing the primary functional milestone.

---

# 1. What V50 must establish

The same bounded core must demonstrate all of the following on one controlled cue→consequence task:

1. **initial acquisition** — learn a set of arbitrary cue→consequence associations;
2. **continued acquisition** — learn later associations without ceasing to learn;
3. **first-return recall** — answer old cues after intervening learning **before their outcomes are shown again**;
4. **held-out rendering generalization** — recognize noisy/rendered variants of learned cues;
5. **one-off corruption resistance** — do not replace a stable association because of one contradictory outcome;
6. **true revision** — change the association after repeated consistent contradictory evidence;
7. **synaptic persistence** — memory survives a reset of transient neuron activity;
8. **learning dependence** — learned performance disappears when the learned KC→MBON synapses are reset;
9. **causal pairing dependence** — unpaired outcome exposure does not create cue-specific competence;
10. **bounded state** — persistent learned state is fixed with lifetime.

This is a **component milestone**, not the final project goal.

After the phenotype exists, the same architecture is transferred without structural retuning to the existing byte-stream stress test.

---

# 2. Why the primary assay is cue→consequence rather than the 14-byte serialization

The V49 byte workload asks one small raw mushroom-body-like circuit to do several jobs simultaneously:

- predict fixed framing syntax;
- predict six random identity-string characters;
- identify the current object;
- predict the object-associated target byte;
- revise associations.

That is a legitimate integration challenge.

It is not an isolated test of whether a fly-like associative circuit can learn and preserve associations.

V50 therefore separates:

## Stage I — functional associative core

Direct sensory cue sequences arrive on PN channels.

The learner predicts a subsequently observed consequence.

No identity labels, cohort labels, phase labels, or target-position labels enter the learner.

The observed consequence itself is the self-supervised teaching signal.

## Stage II — transfer/integration stress

Run the same architecture and fixed learning constants on the existing 14-byte stream.

This stage remains visible and mandatory for reporting, but Stage-I success is not invalidated merely because one 124-kB associative circuit is not yet a complete byte language model.

Conversely, Stage-I success may not be advertised as solving the byte-learning objective.

---

# 3. Raw-BANC substrate

V50 reuses the **actual embedded BANC PN→KC and KC→MBON graph blocks** already audited in V49.

Source files:

```text
banc_888_meta.feather
SHA256 819bbcff476e52702d6f8d8604ce1f12d1d7b11942281df2f49df2a73a6f15a5

banc_888_edgelist_simple_v2.feather
SHA256 363fdef3813b72a5e45a42f17034cd5a544b654c838929cecf6e1ce5f60625cb
```

Audited full bilateral populations:

```text
PN      699
KC      4,552
MBON    104
```

Raw graph blocks:

```text
PN→KC pair rows         22,417
PN→KC count sum        168,315

KC→MBON pair rows       15,523
KC→MBON count sum       28,415
```

V50 does **not** require the raw graph to beat a rewire in order for the functional core to count as working.

Raw-vs-rewire remains an explanatory control only.

This corrects the earlier drift in which topology-specific superiority became an implicit prerequisite for an engineering reference system.

---

# 4. Core architecture

The V50 reference core is:

\[
\boxed{
\text{PN sensory cue sequence}
\rightarrow
\text{stateful KC population}
\rightarrow
\text{plastic KC→MBON synapses}
\rightarrow
\text{consequence prediction}
}
\]

No persistent object records.

No replay.

No slots.

No learned routing.

No PAIR or signed representation.

No APL controller.

No MBON recurrence in the primary core.

No DAN network in V50.

The consequence residual is an explicitly engineered teacher interface. It is not described as the fly's recovered DAN physiology.

---

# 5. Sensory cue construction directly in PN space

V50 Stage I does **not** use the V49 random `96-context → PN` adaptor.

The cue is generated directly as a PN population pattern.

This removes one arbitrary interface from the functional core assay.

## 5.1 Identity prototypes

Each cue identity \(q\) has a fixed latent PN prototype:

\[
u_q\in\mathbb R^{699}.
\]

For each identity:

- choose exactly 32 active PN channels without replacement;
- active amplitudes sampled once from Uniform[0.7, 1.0];
- all other PN channels zero.

Identity prototypes are generated from the seed and never supplied to the learner as labels.

Different identities may overlap in active PNs by chance.

## 5.2 Cue rendering

Every presentation consists of four sensory microframes.

For each microframe:

- retain 80% of the identity's active PNs;
- replace the remaining 20% with the same number of randomly chosen inactive PN channels;
- multiply retained active amplitudes by independent log-normal or Gaussian jitter with SD 0.08;
- clip to [0, 1].

Thus different renderings of one cue share latent structure but are not identical.

Training uses rendering families 0 and 1.

Held-out probes use independent rendering family 2.

The learner is never given rendering IDs.

## 5.3 Reset between trials

All transient neuron state is reset to zero before each independent cue trial.

Therefore across-trial persistence cannot be carried by lingering KC activity.

Within a four-frame cue, stateful cells may integrate information.

---

# 6. Raw PN→KC drive

Use the same fixed count-normalized raw PN→KC projection across all cell arms.

For raw recorded count \(C^{PK}_{pk}\):

\[
a^{PK}_{pk}
=
\frac{\sqrt{C^{PK}_{pk}}}
{\sqrt{\sum_{p'} C^{PK}_{p'k}}+\epsilon}.
\]

At sensory microframe \(t\):

\[
d_{k,t}
=
\sum_p a^{PK}_{pk}u_{p,t}.
\]

This drive is identical across all matched cell arms.

---

# 7. Primary KC cell — minimal stateful leaky cell

The V50 reference neuron is intentionally simple.

For every KC:

\[
s_{k,t}
=
\rho s_{k,t-1}
+
(1-\rho)\tanh(d_{k,t}).
\]

Fixed:

\[
\boxed{\rho=0.75}.
\]

No per-cell learning occurs inside the KC.

No BPTT or hidden gradient enters the KC.

At the final cue microframe:

1. rank KCs by \(|s_k|\);
2. retain the largest 5%:

\[
K_{\rm active}=228;
\]

3. set all others to zero;
4. L2-normalize the retained vector:

\[
z_q=
\frac{\operatorname{TopK}_{228}(s)}
{\|\operatorname{TopK}_{228}(s)\|_2+\epsilon}.
\]

This vector drives MBON prediction and local plasticity.

The fixed \(\rho\) is an engineering assumption.

V50 does not tune it scientifically.

---

# 8. Static-cell lesion

`STATIC-KC` is not a competing architecture family.

It is a lesion of within-cue private temporal state.

At each frame:

\[
s_{k,t}
=
\tanh(d_{k,t}).
\]

Only the final microframe is used to produce the top-5% KC vector.

Thus `STATIC-KC` and `LEAKY-KC` have:

- the same raw graph;
- the same PN frames;
- the same active KC count;
- the same KC→MBON learning rule;
- the same outcome teacher;
- the same number of learned parameters.

The only difference is within-cue state.

This tests whether a minimal stateful cell repairs a named deficit rather than running a broad neuron-type tournament.

---

# 9. Expressive historical ELM reference — conditional secondary block

The historical audit found several genuine stateful ELM implementations, but the exact six-cell P4 comparison was not completely recovered.

V50 therefore forbids silently inventing a new arm and calling it the historical ELM.

An `ELM-KC` secondary arm is **not included in the initial V50 release**. It may be added only in a later explanatory comparison if the exact intended historical cell equation/source used for the relevant local temporal comparison is recovered and frozen.

If recovered, the comparison must:

- receive the identical PN→KC drives;
- use the same 4,552 KC identities;
- preserve private state per KC;
- have no BPTT or hidden-state backpropagation during V50;
- report its larger private state and fixed parameters separately;
- use the identical KC→MBON learner and exposure schedule.

If the source cannot be recovered cleanly:

\[
\boxed{\text{do not include ELM-KC in V50}}
\]

and record it as a pending secondary comparison.

V50 qualification does not depend on an ELM result.

---

# 10. Consequence alphabet and MBON readout

Stage I uses:

\[
\boxed{8\text{ possible consequences}}.
\]

This is deliberately smaller than the 256-byte alphabet.

The goal is to test persistent associative learning, not byte serialization.

## 10.1 Fixed outcome code

For every science seed generate eight normalized Rademacher codewords:

\[
c_y\in\{-1,+1\}^{104},
\qquad
\|c_y\|_2=1.
\]

The same codebook is used by all matched arms for that seed.

Report codeword pairwise cosine statistics.

No codebook sweep.

## 10.2 KC→MBON memory

Use the raw KC→MBON support:

\[
M_{ij}\in\{0,1\}.
\]

One learned scalar exists per raw edge:

\[
\theta_{ij}.
\]

Initialize:

\[
\theta_{ij}=0.
\]

For V50 primary functional proof use **binary raw support**:

\[
w_{ij}=\theta_{ij}.
\]

Do not use raw recorded-count scaling in the primary core.

V49 showed count heterogeneity can alter stability/prediction tradeoffs; V50 first asks whether the coherent learner works at all.

Recorded-count weighting can be restored later as a reduction/interpretation question.

Persistent learned payload:

\[
15,523\text{ float64 values}\approx124\text{ kB}.
\]

## 10.3 Prediction

MBON scores:

\[
h_j=\sum_i M_{ij}\theta_{ij}z_i.
\]

Outcome logits:

\[
\ell_y=\tau^{-1}c_y^\top h.
\]

Use:

\[
\boxed{\tau=0.25}
\]

as in the calibrated V49 MBON code interface.

Probability:

\[
p(y)=\operatorname{softmax}(\ell)_y.
\]

The prediction is recorded before the consequence is revealed.

---

# 11. Local teaching and plasticity

After the environmental consequence \(y\) is revealed, compute:

\[
\bar c=\sum_y p(y)c_y.
\]

Standardized MBON teaching residual:

\[
e
=
\sqrt{104}(c_y-\bar c).
\]

For every active existing KC→MBON edge:

\[
\boxed{
\theta_{ij}
\leftarrow
\theta_{ij}
+
\eta\,z_i e_j
}
\]

with:

\[
\boxed{\eta=0.5}.
\]

This is the same output-coordinate normalization/calibration family used in the final V49 runner.

The rule is local with respect to the defined interface:

- presynaptic KC activity \(z_i\);
- MBON-specific consequence residual \(e_j\);
- anatomical edge \(M_{ij}\).

It is **not** claimed to be the recovered biological DAN learning rule.

---

# 12. Functional learning schedule

Use fresh science seeds.

Every seed creates four disjoint cohorts:

\[
A,\ B,\ C,\ D.
\]

Each cohort contains:

\[
\boxed{32\text{ identities}}.
\]

Total identities:

\[
128.
\]

Every cohort has balanced consequence assignment:

- 8 consequence types;
- 4 identities per consequence per cohort.

Different identities may share the same consequence.

This prevents the system from solving the task by treating outcome class as identity.

---

# 13. Training exposures

Each identity receives:

\[
\boxed{8\text{ acquisition presentations}}
\]

using training renderings 0 and 1 in alternating/randomized order.

Presentations are shuffled within each exposure.

For each presentation:

1. reset transient KC state;
2. deliver four cue microframes;
3. predict consequence;
4. reveal consequence;
5. apply one KC→MBON synaptic update.

Only one durable synaptic update occurs per cue→consequence event.

There are no framing/key-byte writes in Stage I.

---

# 14. Read-only probe discipline

A probe:

1. resets transient state;
2. delivers a cue rendering;
3. records consequence prediction;
4. does **not** reveal a consequence to the learner;
5. does not modify KC→MBON synapses.

Probe purity is asserted by state hash.

First-return recall therefore occurs before any re-teaching.

---

# 15. A-cohort internal roles

Within cohort A, split identities deterministically by seed:

- 16 **untouched** identities;
- 8 **one-off corruption** identities;
- 8 **revision** identities.

These role labels are evaluator/schedule metadata.

They do not enter the learner.

---

# 16. Stage-I sequence

## 16.1 Train A

Train all 32 A identities for 8 presentations.

Probe:

- A trained rendering;
- A held-out rendering.

Checkpoint: `after_A`.

## 16.2 Train B

Train 32 new B identities for 8 presentations.

Before any A cue is trained again, perform **first-return** read-only probes on all A identities.

Checkpoint: `A_first_return_after_B`.

Then probe B.

## 16.3 One-off corruption

For each of the 8 A-noise identities:

- present one training rendering once;
- reveal one incorrect consequence, selected uniformly from the other seven outcomes;
- allow the learner to update normally.

Then read-only probe the original A consequence.

This tests whether one contradictory event catastrophically overwrites a stable association.

## 16.4 Train C

Train 32 new C identities for 8 presentations.

Probe:

- untouched A;
- B;
- C;
- held-out renderings.

## 16.5 True revision

For each of the 8 A-revision identities choose a new consequence distinct from the original.

Present the revised consequence repeatedly for:

\[
1,\ 2,\ 4,\ 8
\]

cumulative revision exposures.

After exposures 1, 2, 4 and 8 perform read-only probes.

Record:

- new-consequence probability/accuracy;
- old-consequence probability;
- revision bpb.

## 16.6 Train D

Train 32 new D identities for 8 presentations.

Final read-only probes:

- untouched A trained rendering;
- untouched A held-out rendering;
- noise A original consequence;
- revision A new consequence;
- B trained + held-out;
- C trained + held-out;
- D trained + held-out.

No old outcome is shown before these probes.

---

# 17. Primary arms

V50 Stage-I science uses only four mandatory arms.

## F0 — `LEAKY-RAW`

Primary functioning candidate:

- raw PN→KC;
- leaky KC state;
- top-5% KC code;
- raw binary KC→MBON support;
- local consequence update.

## F1 — `STATIC-RAW`

Cell-state lesion.

Identical to F0 except no within-cue KC state.

## F2 — `LEAKY-NOLEARN`

Same leaky circuit with:

\[
\eta=0.
\]

Controls hand-designed cue/readout bias.

## F3 — `LEAKY-UNPAIRED`

Same leaky circuit and same number of updates, but consequence exposure is unpaired.

Implementation:

For each acquisition exposure within a cohort, permute the consequence assignments across identities before revealing them to the learner.

Preserve exactly:

- consequence marginal counts;
- number of updates;
- cue presentations.

Because the pairing changes across exposures, no stable cue-specific association is available.

This is a causal-learning control, not a candidate.

---

# 18. Optional explanatory raw-topology control

After the primary functional result is known, V50 may report:

## F4 — `LEAKY-KM-REWIRE` (deferred from the initial V50 release)

This explanatory control is not run in the first V50 science pass. If added later, use the same degree-preserving KC→MBON rewire machinery as V49 with binary edges.

This control asks whether the functioning phenotype depends on raw topology.

Its result is explanatory only.

A functioning `LEAKY-RAW` core does **not** fail qualification merely because the rewire matches it.

This corrects the earlier topology-superiority drift.

---

# 19. Persistent-state localization lesions

These are read-only post-training interventions on clones of the trained `LEAKY-RAW` system.

## 19.1 Transient-state reset

All normal Stage-I probes already begin from:

\[
s=0.
\]

Therefore successful first-return/final recall already demonstrates memory surviving transient activity reset.

Record this explicitly.

## 19.2 Synapse-reset lesion

After the final training schedule:

1. clone the trained model;
2. set every learned KC→MBON \(\theta_{ij}=0\);
3. preserve raw graph and fixed cell parameters;
4. run final probes from zero transient state.

If learned association performance remains high after synapse reset, the claimed synaptic memory interpretation fails.

## 19.3 Synapse-shuffle lesion

Secondary localization:

- preserve the multiset of learned \(\theta\) values;
- shuffle them within each MBON across existing raw incoming edges;
- run final probes.

This tests whether the learned edge-specific pattern matters, rather than only the weight distribution.

It is explanatory, not a qualification prerequisite.

---

# 20. Primary functional endpoints

All endpoints are measured across science seeds.

## 20.1 Acquisition

- A accuracy after A;
- B accuracy after B;
- C accuracy after C;
- D accuracy after D;
- prequential accuracy/loss by exposure number;
- presentations to first cross 70% and 80% cohort accuracy.

## 20.2 Persistence

- A first-return after B;
- untouched A after C;
- untouched A after D;
- B after D;
- C after D.

## 20.3 Generalization

For every cohort:

- trained-rendering accuracy;
- held-out-rendering accuracy;
- held-out bpb.

## 20.4 Noise resistance

On A-noise identities:

\[
\text{accuracy on original consequence after one wrong update}.
\]

## 20.5 Revision

On A-revision identities:

- revised accuracy after 1,2,4,8 changed exposures;
- old consequence probability;
- new consequence probability;
- revision bpb.

## 20.6 State localization

- intact final probe;
- synapse-reset final probe;
- synapse-shuffle final probe.

## 20.7 Resource accounting

- learned synaptic bytes;
- transient KC-state bytes;
- fixed graph bytes;
- fixed cell-parameter bytes;
- operations/event;
- wall-clock;
- peak RSS.

---

# 21. Functional phenotype gates

These gates define a **working component**, not final project success.

No topology-specific gate exists.

## G0 — implementation validity

Require:

- raw graph integrity;
- identical cue streams across matched arms;
- identical consequence codebook within seed;
- identical exposure counts;
- probe purity;
- transient-state reset between trials;
- no slots/replay/growing identity-indexed persistent state;
- no evaluator labels inside the learner;
- all 16 science seeds complete.

Failure:

`IMPLEMENTATION_INVALID`.

## G1 — initial acquisition

`LEAKY-RAW` after A:

\[
\boxed{\text{mean trained accuracy}\ge0.80}
\]

and no seed below 0.60.

Failure:

`CORE_DOES_NOT_ACQUIRE`.

## G2 — continued acquisition

`LEAKY-RAW` after D:

\[
\boxed{\text{mean D trained accuracy}\ge0.80}
\]

and no seed below 0.60.

Failure:

`CORE_STOPS_LEARNING`.

## G3 — first-return persistence

A first-return after B:

\[
\boxed{\text{mean accuracy}\ge0.60}
\]

and paired improvement over `LEAKY-NOLEARN` ≥0.30 with lower CI >0.

Failure:

`NO_FIRST_RETURN_MEMORY`.

## G4 — long-delay persistence

Untouched A after D:

\[
\boxed{\text{mean trained-rendering accuracy}\ge0.50}.
\]

Held-out A after D:

\[
\boxed{\text{mean held-out accuracy}\ge0.40}.
\]

Both must exceed `LEAKY-NOLEARN` by at least 0.20 with lower CI >0.

Failure:

`NO_DURABLE_ASSOCIATIVE_MEMORY`.

## G5 — causal pairing

`LEAKY-RAW` must beat `LEAKY-UNPAIRED` on untouched-A-after-D accuracy by at least:

\[
\boxed{0.20}
\]

with paired lower CI >0.

Failure:

`PAIRING_NOT_CAUSAL`.

## G6 — one-off corruption resistance

For A-noise identities after one incorrect outcome:

Original-consequence accuracy must remain:

\[
\boxed{\ge0.50}
\]

and must not fall more than 0.25 absolute relative to the immediately preceding correct probe.

Failure:

`ONE_OFF_NOISE_OVERWRITES_MEMORY`.

## G7 — revision

After 8 consistent revised exposures:

\[
\boxed{\text{new-consequence accuracy}\ge0.60}
\]

and mean new-consequence probability must exceed old-consequence probability.

Failure:

`CORE_CANNOT_REVISE`.

## G8 — synaptic localization

After KC→MBON synapse reset:

accuracy on untouched A final probes must fall by at least:

\[
\boxed{0.25}
\]

relative to intact trained `LEAKY-RAW`,

with paired CI excluding zero.

Failure:

`MEMORY_NOT_LOCALIZED_TO_LEARNED_SYNAPSES`.

## G9 — bounded state

Persistent learned state is fixed:

\[
15,523\text{ KC→MBON scalars}
\]

and does not grow with identities or exposures.

Transient state is reported separately.

Failure:

`STATE_BUDGET_FAIL`.

---

# 22. Cell-state question

The primary V50 scientific claim does **not** require the leaky cell to beat STATIC.

The cell comparison is explanatory:

\[
\text{Does private temporal state repair a named temporal/sensory deficit?}
\]

Report:

- acquisition difference;
- held-out rendering difference;
- first-return difference;
- noise/revision difference;
- resource difference.

Do not declare the static cell obsolete if the leaky cell matches it on the basic association task.

A richer cell earns inclusion only when it repairs a named failure.

---

# 23. Secondary temporal-order assay

To give cell state a task where it can matter specifically, V50 includes a small separate assay.

This does not affect the persistent-learning gates above.

## 23.1 Construction

Generate cue triples of PN subpatterns:

\[
X,\ Y,\ Z.
\]

Create paired sequences:

\[
X\rightarrow Y\rightarrow Z
\]

and

\[
Y\rightarrow X\rightarrow Z.
\]

Both sequences:

- contain the same three subpatterns;
- have the same final microframe \(Z\);
- have the same aggregate PN activity distribution;
- differ only in temporal order.

Assign different consequences.

Use held-out jittered versions of \(X,Y,Z\) for testing.

## 23.2 Question

`STATIC-KC` sees the same final frame and therefore has no direct order state.

`LEAKY-KC` can carry order-sensitive recency information.

Report whether LEAKY exceeds STATIC.

This is a named temporal deficit.

If LEAKY does not help:

> do not add more complex neuron models merely because they are more biologically elaborate.

If LEAKY helps:

> the exact historical ELM reference may be worth restoring in a later cell-level comparison.

---

# 24. Stage-II byte-stream transfer

Stage II uses the **same architecture constants**:

- raw PN→KC graph;
- selected KC cell equation;
- raw binary KC→MBON support;
- same KC sparsity;
- same local update form;
- no slots/replay.

A deterministic fixed context→PN adaptor is required because the byte stream does not provide biological PNs.

The Stage-II outcome alphabet is 256 bytes, using a fixed 104-dimensional byte codebook.

V49's calibrated byte interface may be reused.

## 24.1 Purpose

Stage II asks:

> Does a core that genuinely learns/persists in the direct associative assay transfer at all to the current byte integration task?

It is not allowed to redefine Stage-I success.

## 24.2 Required reporting

Report:

- whole-stream bpb;
- target bpb;
- newest-cohort acquisition;
- old target recall;
- held-out rendering recall;
- update count by framing/key/target evaluator class;
- plasticity-state bytes.

## 24.3 Passive oracle lesions

For diagnosis only, run read-only or separately trained oracle lesions:

- target-only plasticity;
- no-framing plasticity;
- no-key plasticity;
- random count-matched plasticity.

These arms are **nondeployable**.

They may identify what overwhelms the associative core.

They may not be promoted into a candidate that secretly knows byte roles.

This incorporates Claude's useful workload decomposition without returning the project to slot admission.

---

# 25. Stage-II interpretation

Possible outcomes:

## Functional Stage I + weak Stage II

Interpretation:

> the associative core exists, but the serialization/interface or all-byte teaching schedule overwhelms it.

Next work should improve the interface/teaching decomposition while preserving the functioning core.

Do not discard the core.

## Weak Stage I + weak Stage II

Interpretation:

> the defined raw-fly core still lacks a basic persistent-learning mechanism.

Do not proceed to bigger load tests or digital-fly embodiment.

Diagnose the failed phenotype inside the fixed core.

## Strong Stage I + improved Stage II

Interpretation:

> the reference core is worth preserving and minimizing.

Only then begin controlled graph/cell/feedback reduction or expansion.

---

# 26. Science seeds

Fresh V50 seeds:

```text
50901 50902 50903 50904
50905 50906 50907 50908
50909 50910 50911 50912
50913 50914 50915 50916
```

Engineering seeds:

```text
50971+
```

No science seed may be inspected during engineering calibration.

Seed is the inferential unit.

Cue identities, MBONs, KCs, exposures and probes are not independent scientific replicates.

---

# 27. Fixed constants

Primary Stage-I constants:

```text
PN count                      699
KC count                      4552
MBON count                    104

cue active PN prototype       32
cue microframes               4
rendering retain fraction     0.80
rendering amplitude SD        0.08

KC leak rho                   0.75
KC active fraction            0.05
KC active count               228

outcome classes               8
MBON temperature tau          0.25
KC→MBON learning rate eta     0.5

cohorts                       A,B,C,D
identities/cohort             32
acquisition exposures         8
A untouched                   16
A one-off-noise               8
A revised                     8
revision exposures            1,2,4,8
```

No scientific sweep over these constants is permitted in V50.

Engineering changes for numerical stability must be completed on engineering seeds before the science source hash is frozen.

---

# 28. Required outputs

```text
SUMMARY.md
GATES.json
MANIFEST.json
PROTOCOL.md

RAW_GRAPH_AUDIT.csv
CUE_GENERATOR_AUDIT.csv
OUTCOME_CODEBOOK_AUDIT.csv

PREQUENTIAL_ACQUISITION.csv
FIRST_RETURN.csv
COHORT_RETENTION.csv
HELDOUT_RENDERINGS.csv
NOISE_RESISTANCE.csv
REVISION.csv

CELL_COMPARISON.csv
TEMPORAL_ORDER_ASSAY.csv

STATE_LOCALIZATION.csv
SYNAPSE_RESET.csv
SYNAPSE_SHUFFLE.csv

STATE_COSTS.csv
RUNTIME.csv
SAMPLE_BUDGET.csv

BYTE_TRANSFER_SUMMARY.csv
BYTE_PLASTICITY_CLASS_AUDIT.csv

jobs/
checkpoints/
```

---

# 29. Engineering / Mac M3 plan

One self-contained Python runner plus launch scripts.

## 29.1 Shared computation

Within a seed:

- cue prototypes/renderings shared across arms;
- PN microframes shared;
- raw PN→KC sparse arrays shared;
- outcome codebook shared;
- evaluation schedule shared.

Each learning arm has independent KC transient state and learned KC→MBON weights.

For efficiency, prediction/update traverses only raw KC→MBON edges leaving the currently active 228 KCs. This is algebraically identical to scanning all 15,523 edges because inactive KCs contribute exactly zero.

## 29.2 Multiprocessing

Default:

- 4 macOS `spawn` workers;
- one seed per worker for Stage I;
- BLAS/Accelerate threads fixed to 1.

Environment before NumPy:

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## 29.3 Caffeinate

Primary functional milestone (default release command):

```bash
caffeinate -dimsu python3 elm_v50_functional_fly_core.py \
  --stage functional \
  --workers 4 \
  --out-dir v50_results
```

After Stage-I qualification, run the byte transfer separately:

```bash
caffeinate -dimsu python3 elm_v50_functional_fly_core.py \
  --stage byte \
  --workers 4 \
  --out-dir v50_results
```

`--stage all` remains available for an uninterrupted combined run but is not the recommended first execution.

## 29.4 Resume

- atomic per-seed results;
- rolling checkpoint for unfinished seeds;
- cache identity includes source/design/raw-graph hashes;
- SIGINT/SIGTERM checkpoint;
- rerun reuses matching completed jobs;
- final upload excludes large rolling checkpoints.

---

# 30. Engineering firewall

Before any science run:

1. compile;
2. raw PN→KC / KC→MBON integrity test;
3. deterministic cue/rendering test;
4. outcome-codebook validity test;
5. probe-purity state hash;
6. static/leaky common-input audit;
7. one engineering-seed A acquisition check;
8. forced interruption/resume equivalence;
9. two-worker and four-worker determinism;
10. full engineering seed through A→D/noise/revision;
11. synapse-reset lesion check;
12. no science seed inspection.

The source hash is frozen only after these pass.

---

# 31. No-drift decision tree

## If G1 fails — core does not acquire A

Do **not** open slots, pathways, DAN, APL, recurrence or representation learning.

Inspect only:

- sensory coding magnitude;
- outcome residual scaling;
- raw KC activity health;
- local update correctness.

This is an interface/learning-rule failure.

## If G1 passes but G2 fails — stops learning later cohorts

The core has a plasticity-capacity problem.

Stay inside the same reference system.

Do not change neuron family and memory architecture simultaneously.

## If acquisition works but G3/G4 fail — catastrophic interference

The coherent core has reproduced the original research problem.

This is then the right place to test **one** biological plasticity/consolidation mechanism while preserving the functioning reference.

Examples may include eligibility, modulatory teaching or synaptic turnover, but none is preselected by V50.

## If G6 fails — one wrong event overwrites memory

The reference core needs noise/consolidation dynamics.

Do not solve this by hiding the event with evaluator labels.

## If G7 fails — cannot revise

The reference core is too rigid.

Any future protection mechanism must retain a revision route.

## If G8 fails — memory is not synaptic

Do not describe the system as a persistent synaptic learner.

Locate the surviving information before proceeding.

## If Stage I passes

Freeze the functioning reference implementation.

All future experiments are patches/lesions against that reference.

Do not replace it merely because Stage-II byte transfer is weak.

---

# 32. What V50 intentionally does not answer

V50 does not determine:

- the true fly DAN teaching rule;
- whether APL is physiologically inhibitory in this export;
- whether MBON recurrence matters under some other dynamical model;
- whether raw BANC topology beats every rewire;
- whether ELM, ALIF, GRU or LIF cells are superior;
- whether byte prediction can be solved by this core alone;
- whether any explicit episodic memory is biologically impossible;
- whether the final minimal core has fewer than 1,000, 10,000 or 100,000 neurons.

It establishes the reference phenotype first.

---

# 33. Pre-run prediction

The main pre-run expectation is intentionally simple:

1. `LEAKY-RAW` should acquire A and later D under the direct cue→consequence interface.
2. If the current raw local synaptic rule has any genuine persistent-learning capacity, A first-return after B should remain well above no-learning/unpaired controls.
3. Resetting transient KC state should not erase learned associations because every probe starts from zero state.
4. Resetting learned KC→MBON synapses should destroy most learned competence.
5. One-off corruption should hurt less than repeated true revision if the system has any useful stability/plasticity balance.
6. The static-cell lesion may or may not match the leaky core on the basic association task; that does not decide the functional milestone.
7. The temporal-order assay is expected to favor a stateful cell if within-cue temporal state is computationally useful.

Failure of these expectations is informative because the same reference system is retained rather than replaced.

---

# 34. Success criterion

A V50 functional core is established only if G0–G9 all pass.

The allowed claim is then:

> **A bounded raw-BANC-based circuit with explicit stateful neurons and local KC→MBON synaptic learning can acquire sequential cue→consequence associations, preserve a substantial fraction through intervening learning, generalize to new renderings, resist one-off corruption, revise repeated changes, and store the learned information in synaptic state rather than lingering activity.**

That would still be a component result.

It would finally give the project a maintained working organism-like learning core to minimize, lesion and extend—rather than another isolated mechanism verdict.
