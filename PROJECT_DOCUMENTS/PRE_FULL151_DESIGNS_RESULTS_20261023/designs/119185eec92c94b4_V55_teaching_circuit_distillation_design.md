# V55 — Data-Driven Distillation of the Fly Teaching Circuit

**Status:** design freeze candidate  
**Date:** 13 September 2026  
**Primary purpose:** recover a minimal, reproducible teaching-circuit architecture from real fly connectome data before inventing another persistent-learning rule  
**Discovery specimen:** BANC v888  
**Replication specimen:** hemibrain v1.2.1  
**Cross-specimen alignment:** curated **cell-type / homologous-class** matching only  
**Behavioral BYTE benchmark:** explicitly out of scope for the primary V55 result  
**Expected handoff:** `V55_TEACHING_SPEC.json` for a later behavioral integration round

---

# 0. Why V55 exists

V50–V54 established a relatively stable associative substrate:

\[
PN \rightarrow KC \rightarrow MBON
\]

with bounded KC→MBON learned state.

But the teaching side remained largely engineered:

- outcome residuals;
- scalar gains;
- novelty/surprise rules;
- phase-conditioned heuristics;
- eligibility constructions;
- fast/slow and complementary stores.

Many of those constructions failed, traded retention against learning, or were matched by simpler controls.

Meanwhile, the uploaded BANC and hemibrain data contain a much richer biological teaching circuit:

\[
\text{KC / MBON / other upstream neurons}
\rightarrow
\text{DAN}
\rightarrow
\text{MB-associated targets}.
\]

The recent audit found that direct MBON feedback is only a minority of recorded DAN input, while other mapped neurons and KCs contribute substantial input. Different source classes also arrive with different dendrite/axon distributions.

V55 therefore asks a different question:

> **What teaching-circuit computation is actually supported by the connectome, and how far can that circuit be compressed without destroying the structures that replicate across specimens?**

V55 does **not** start by proposing a new learning rule.

---

# 1. Constitution: what may and may not be joined

## 1.1 Same-specimen joins are allowed

### BANC

Neuron-level joins may use only BANC-v888-compatible data:

```text
banc_888_meta
banc_888_edgelist_simple
banc_888_edgelist_split
banc_888_neurotransmitter_prediction
```

and any later BANC-only derived tables.

### Hemibrain

Neuron-level joins may use only hemibrain-v1.2.1-compatible data:

```text
hemibrain_121_meta
hemibrain_121_split_edgelist
hemibrain_olf_data
```

and same-specimen hemibrain queries.

## 1.2 Cross-specimen alignment is type-level only

Use curated:

```text
cell_type_cross_matching.csv
```

to construct homologous / corresponding **cell-type classes**.

Allowed cross-specimen comparisons include:

- cell-type input composition;
- output composition;
- connection-density distributions;
- matched-type adjacency;
- transmitter class;
- broad receiving territory;
- motif frequency;
- operator rank / compressibility;
- module membership stability.

Forbidden:

- treating a BANC neuron as the same individual as a hemibrain neuron;
- copying missing edges from one specimen into the other;
- filling a BANC circuit with hemibrain synapses;
- merging neuron IDs into one graph;
- averaging coordinates across specimens.

## 1.3 External physiology is not anatomy

DoOR and other physiological datasets do not enter V55's teaching-circuit reconstruction.

They may be used in later sensory simulations, but are not measurements of BANC or hemibrain DAN activity.

---

# 2. Populations

V55 focuses on the mushroom-body teaching system.

Primary classes:

```text
KC
MBON
PAM DAN
PPL1 DAN
APL
DPM
```

Additional upstream neurons are included if they directly contact selected DANs.

Do not collapse all "other" neurons into one anonymous category before analysis.

For every outside-core upstream neuron retain, when available:

```text
body ID
cell type
superclass
class
subclass
hemilineage
region
side
transmitter prediction / curated transmitter evidence
```

Unknown annotations remain unknown.

---

# 3. Discovery / replication separation

## 3.1 BANC = discovery specimen

BANC is primary because the current V50/V51 core already uses BANC PN/KC/MBON identities.

V55 may discover candidate teaching motifs in BANC.

## 3.2 Hemibrain = replication specimen

Hemibrain independently tests whether those motifs recur.

A BANC motif is not promoted merely because a vaguely similar connection exists in hemibrain.

Replication must be evaluated at the matched-type or matched-class level.

## 3.3 Codex output

If the parallel Codex analysis produces additional matched-type tables, they enter as **replication evidence only**.

They cannot modify BANC neuron-level discovery results.

---

# 4. Block A — DAN input census

For every selected DAN in each specimen independently, partition incoming recorded connection count into:

```text
KC
MBON
DAN
APL
DPM
PN / direct sensory-class neurons
ascending / descending neurons
other annotated central neurons
unknown
```

Report both:

1. raw recorded connection counts;
2. fractions of the DAN's total recorded incoming count.

Repeat using receiving-compartment information where available:

```text
post dendrite
post axon
post primary neurite
post soma
post unknown
```

No threshold above count ≥1 is used for the primary census.

Threshold sensitivity is reported separately for:

```text
1, 2, 5, 10
```

because weak DAN contacts are abundant and high thresholds can erase the circuit.

---

# 5. Block B — DAN-family heterogeneity

Do not assume all DANs implement one computation.

Within each specimen, group DANs by annotated type/family:

```text
PAM types
PPL1 types
other explicitly dopamine-annotated types
```

For each group report:

- source-class input fractions;
- receiving-territory fractions;
- total input degree;
- MBON feedback fraction;
- KC fraction;
- outside-core fraction;
- DAN↔DAN recurrence;
- transmitter evidence of major upstream types.

Primary question:

> **Are there reproducible DAN families with distinct input mixtures, or does the circuit look approximately homogeneous after degree normalization?**

This is descriptive unless supported by matched controls.

---

# 6. Block C — trace outside-core DAN inputs one hop upstream

For every outside-core neuron \(u\) that directly contacts a selected DAN, compute within the same specimen:

\[
f^{MBON}_u
=
\frac{\text{incoming recorded count from MBONs}}
{\text{total incoming recorded count}},
\]

and analogously:

\[
f^{KC}_u,\;
f^{sensory}_u,\;
f^{ascending}_u,\;
f^{DAN}_u.
\]

Weight summaries both:

1. equally by upstream neuron;
2. by the neuron's outgoing recorded count to DANs.

Report the full distribution, not only the mean.

This block distinguishes:

```text
feedback-dominated relays
mixed-input relays
mostly-independent input neurons
```

without assigning semantic labels such as "reward" unless supported by annotations.

---

# 7. Block D — direct and two-hop MBON→DAN operators

Construct, within each specimen:

\[
F_{\rm direct}
:
MBON\rightarrow DAN
\]

from direct recorded connections.

Construct:

\[
F_{\rm relay}
:
MBON\rightarrow U\rightarrow DAN
\]

where \(U\) is the outside-core upstream population.

The relay operator must preserve per-neuron intermediate normalization choices explicitly.

Report at least two normalizations:

## D1 — raw count composition

\[
F_{\rm relay}^{count}
=
A_{MU}A_{UD}.
\]

## D2 — row-normalized relay composition

Normalize each upstream neuron's incoming and outgoing vectors before composition to prevent high-degree relay neurons from dominating solely through degree.

These operators are **structural transfer operators**, not physiological response models.

---

# 8. Block E — decomposition of the teaching circuit

For BANC and hemibrain independently decompose:

```text
direct MBON→DAN
two-hop MBON→relay→DAN
KC→DAN
other-upstream→DAN
```

using:

- SVD / singular spectrum;
- nonnegative matrix factorization where input is nonnegative;
- bipartite community detection;
- matched degree-preserving nulls.

Report:

\[
r_{50},\;r_{80},\;r_{90},\;r_{95}
\]

where \(r_x\) is the minimum rank explaining \(x\%\) of Frobenius energy for the stated operator.

Do not promote "rank 8" or any other historical number unless recovered from the new analysis under the current definition.

---

# 9. Block F — matched-type cross-specimen replication

Use curated type matching to aggregate each specimen independently to cell-type level.

Construct type-level matrices such as:

\[
A^{BANC}_{MBONtype\rightarrow DANtype}
\]

and:

\[
A^{HB}_{MBONtype\rightarrow DANtype}.
\]

Likewise for:

```text
KC subtype → DAN type
outside-core type → DAN type
DAN type → MBON type
DAN type → KC subtype
```

where supported by the uploaded data.

## 9.1 No edge imputation

Only shared matched types enter direct matrix comparisons.

Missing type in one specimen:

```text
NA
```

not zero, unless the type is confirmed present and the edge absent.

## 9.2 Replication metrics

For the shared type set report:

- Spearman correlation of normalized edge weights;
- cosine similarity of each matched DAN-type input profile;
- Jaccard overlap of top-k source types;
- module-assignment agreement;
- direction-of-effect agreement for source fractions.

Permutation nulls shuffle type labels within broad biological classes, not across arbitrary populations.

---

# 10. Block G — transmitter-aware stratification

Use curated transmitter evidence before raw classifier argmax where they disagree.

Preserve transmitter **class**:

```text
acetylcholine
GABA
glutamate
dopamine
serotonin
octopamine
tyramine
histamine
unknown / mixed
```

Do **not** automatically convert transmitter identity into synaptic sign.

In particular:

- GABAergic input may be treated as inhibitory only in explicitly labelled sensitivity analyses;
- glutamatergic effects are not assigned one universal sign;
- raw BANC transmitter predictions remain probabilistic evidence.

Report whether conserved DAN-input modules are enriched for particular transmitter classes.

---

# 11. Block H — receiving-territory organization

Using split connectivity separately in each specimen, ask whether source classes reach different DAN regions.

For every DAN type/family estimate:

\[
P(
\text{post territory}
\mid
\text{source class}
).
\]

Primary contrasts include:

```text
outside-core vs KC
outside-core vs MBON
KC vs MBON
direct MBON vs relay-neuron input
```

Matched controls preserve:

- DAN identity;
- source-class degree;
- total incoming count.

Question:

> **Does source identity predict receiving territory beyond what degree alone explains?**

This is structural evidence for spatial integration, not proof of local computation.

---

# 12. Block I — DAN output-side structure

Independently characterize outgoing recorded connections from DANs to:

```text
KC
MBON
APL
DPM
other MB neurons
outside-core neurons
```

Report:

- coverage;
- count distribution;
- receiving territory;
- type specificity;
- threshold sensitivity.

Because BANC and hemibrain have very different apparent DAN→KC pair coverage, comparisons must be normalized within specimen.

Do not infer absence of biological modulation from a missing recorded pair.

---

# 13. Primary V55 hypotheses

V55 tests structural hypotheses, not a BYTE score.

## H1 — teaching is not feedback-only

Outside-core and/or KC input contributes substantial recorded input to DANs beyond direct MBON feedback.

Supported only if the effect survives:

- per-DAN normalization;
- PAM/PPL1 restriction;
- both specimens qualitatively;
- type-level replication where mapping exists.

## H2 — relays integrate feedback with other input

Many MBON-recipient DAN-upstream neurons receive the majority of their recorded input from non-MBON sources.

Supported if this is true in both specimens at matched-type / distribution level.

## H3 — source classes are spatially organized

Different input source classes show different receiving-territory distributions on DANs beyond matched degree controls.

## H4 — the teaching circuit is compressible but not rank-one

A modest number of reproducible structural modes explain much of the MBON/relay/DAN routing, while rank-one approximation loses substantial structure.

No component count is predeclared as biologically special.

## H5 — conserved type-level motifs exist

At least some DAN/MBON/KC teaching motifs replicate across BANC and hemibrain at matched cell-type level substantially better than type-shuffled nulls.

---

# 14. What would justify a later behavioral teaching circuit

V55 produces a behavioral handoff only if the following are all true:

1. H1: feedback-only is structurally incomplete;
2. H2: feedback and independent input converge in upstream/DAN pathways;
3. H3 or H5: at least one nontrivial organization principle survives matched controls;
4. the resulting structure can be reduced to a bounded circuit of practical size.

If these fail, do **not** invent a V56 "DAN network" merely because DANs exist.

---

# 15. Reduced teaching-circuit specification

The main artifact is:

```text
V55_TEACHING_SPEC.json
```

It must describe only structures supported by V55.

Possible fields:

```json
{
  "discovery_specimen": "BANC_v888",
  "replication_specimen": "hemibrain_v1.2.1",
  "dan_types": [],
  "input_source_modules": [],
  "feedback_modules": [],
  "territory_modules": [],
  "transmitter_classes": [],
  "operator_ranks": {},
  "cross_specimen_replication": {},
  "unsupported_assumptions": []
}
```

No dynamical equation is inserted unless it follows from an explicit V55 modeling test.

---

# 16. Conditional Block J — minimal computational surrogate

This block runs **only if Blocks A–I identify a reproducible two-source or multi-source teaching motif**.

It is not the primary V55 result.

The surrogate asks:

> Can the discovered structural motif be reproduced by a much smaller bounded circuit without losing its BANC and hemibrain type-level statistics?

Possible state:

```text
event/input modules
feedback modules
DAN-family units
local-output modules
```

The surrogate is fit to **structural operators only**, not BYTE labels.

Use train/test splits over:

- DAN types;
- upstream types;
- held-out edges.

Compare against:

```text
rank-1 feedback-only
degree-matched random
feedback-only NMF
input-only NMF
full input+feedback factorization
```

Qualification requires improvement on held-out structural reconstruction, not training fit.

---

# 17. What V55 explicitly does not do

V55 does not:

- run V54 Markov gains;
- design a new scalar gate;
- run the BYTE benchmark as primary evidence;
- introduce fast/slow memory;
- introduce eligibility traces;
- repair cue learning;
- use DoOR;
- infer reward semantics from unknown upstream neurons;
- convert all neurotransmitters into signed weights;
- copy hemibrain edges into BANC;
- copy BANC edges into hemibrain;
- force a fixed number of DAN channels;
- interpret NMF factors as biological cell types automatically.

---

# 18. Null models

Every claimed structural motif must be compared with appropriate nulls.

At minimum:

## N1 — degree-preserving bipartite rewiring

Preserve in/out degrees within the relevant bipartite graph.

## N2 — type-preserving rewiring

Preserve broad source/target biological class while randomizing individual edges.

## N3 — territory-preserving null

Where testing cell-type specificity within territories, preserve the source's total contacts to each broad territory.

## N4 — transmitter-class-preserving null

When testing transmitter enrichment, preserve source transmitter counts.

A result that disappears under a better-matched null is not promoted.

---

# 19. Cross-specimen decision logic

## `CONSERVED_TEACHING_MOTIF`

At least one nontrivial type-level motif:

- appears independently in BANC and hemibrain;
- exceeds matched nulls in both;
- has compatible source/target composition.

## `SPECIMEN_SPECIFIC_STRUCTURE`

Strong motif in only one specimen or inconsistent direction across specimens.

Do not use as the sole basis for the next learner architecture.

## `FEEDBACK_ONLY_SUFFICIENT_STRUCTURALLY`

Outside-core / KC additions add little structure beyond feedback under the defined reconstruction tests.

This would argue against the present V55 motivation.

## `TEACHING_CIRCUIT_HIGH_DIMENSIONAL_UNCOMPRESSIBLE`

Structural modes exist but no bounded surrogate reconstructs them substantially better than nulls.

Do not force an arbitrary small architecture.

## `NO_REPRODUCIBLE_TEACHING_STRUCTURE`

No motif survives the matched null and cross-specimen criteria.

This is a legitimate negative result.

---

# 20. Implementation principles

One analysis script:

```text
elm_v55_teaching_circuit_distillation.py
```

Mac M3 compatible.

It should:

- use local Feather/CSV files;
- cache every derived edge table;
- avoid loading unrelated whole-brain matrices repeatedly;
- save specimen-specific tables separately;
- never serialize a cross-specimen neuron-level graph;
- include file SHA256s in manifest;
- allow BANC-only and hemibrain-only stages;
- permit cross-specimen aggregation only after both specimen stages finish.

Recommended stages:

```bash
python3 elm_v55_teaching_circuit_distillation.py --self-test

caffeinate -dimsu python3 elm_v55_teaching_circuit_distillation.py \
  --stage banc \
  --out-dir v55_results

caffeinate -dimsu python3 elm_v55_teaching_circuit_distillation.py \
  --stage hemibrain \
  --out-dir v55_results

python3 elm_v55_teaching_circuit_distillation.py \
  --stage compare \
  --out-dir v55_results

python3 elm_v55_teaching_circuit_distillation.py \
  --stage distill \
  --out-dir v55_results
```

---

# 21. Required outputs

```text
SUMMARY.md
REPORT.md
PROTOCOL.md
MANIFEST.json
GATES.json
V55_TEACHING_SPEC.json

# Specimen-specific
BANC_DAN_INPUT_CENSUS.csv
HB_DAN_INPUT_CENSUS.csv

BANC_DAN_TERRITORY.csv
HB_DAN_TERRITORY.csv

BANC_UPSTREAM_RELAYS.csv
HB_UPSTREAM_RELAYS.csv

BANC_DIRECT_MBON_DAN.csv
HB_DIRECT_MBON_DAN.csv

BANC_RELAY_MBON_DAN.csv
HB_RELAY_MBON_DAN.csv

BANC_DAN_OUTPUTS.csv
HB_DAN_OUTPUTS.csv

BANC_OPERATOR_SPECTRA.csv
HB_OPERATOR_SPECTRA.csv

BANC_MODULES.csv
HB_MODULES.csv

# Cross-specimen, type-level only
MATCHED_TYPES.csv
TYPE_LEVEL_CONNECTIVITY_BANC.csv
TYPE_LEVEL_CONNECTIVITY_HB.csv
TYPE_REPLICATION.csv
MODULE_REPLICATION.csv
TRANSMITTER_REPLICATION.csv

# Nulls
NULL_DEGREE.csv
NULL_TYPE.csv
NULL_TERRITORY.csv
NULL_TRANSMITTER.csv

# Optional surrogate
SURROGATE_RECONSTRUCTION.csv
SURROGATE_HELDOUT.csv
```

---

# 22. Self-test / audit firewall

Before scientific interpretation:

1. verify all BANC joins use BANC IDs only;
2. verify all hemibrain joins use hemibrain IDs only;
3. assert no neuron-level table contains IDs from both specimens;
4. verify matched-type comparison is performed only after type aggregation;
5. verify missing type is `NA`, not automatically zero;
6. verify direct and relay operators use the declared normalizations;
7. verify threshold-1 primary census;
8. verify threshold sensitivity does not overwrite primary counts;
9. verify nulls preserve the declared marginals;
10. verify transmitter class is not silently converted to sign;
11. verify APL raw-vs-curated disagreement remains visible;
12. verify unknown source types remain unknown;
13. verify all source files and derived files are hashed;
14. verify repeated run determinism.

Failure:

```text
IMPLEMENTATION_INVALID
```

---

# 23. Interpretation boundary

A positive V55 can establish:

> the adult fly connectomes contain a reproducible, structured teaching circuit whose organization is richer than direct MBON feedback alone and can be summarized by a bounded set of conserved motifs.

It cannot establish:

- what any specific DAN firing pattern means;
- that a particular neuron encodes reward;
- that the circuit computes prediction error;
- that a structural factor is a dynamical state;
- that the BYTE learner should use the same exact neuron identities;
- that transmitter class uniquely determines sign;
- that a compressed factor is biologically a cell population.

Those require later modeling or physiological evidence.

---

# 24. V55 stop rule

Do not convert a visually interesting graph motif into a V56 mechanism unless it:

1. survives a matched null;
2. is not produced by degree alone;
3. replicates across BANC and hemibrain at the appropriate type level;
4. admits a bounded reduced representation;
5. has a clear mapping to an operation that can be falsified behaviorally.

If no structure satisfies all five, V55 should end with a negative result rather than another speculative teaching architecture.

---

# 25. Why this is the correct next version

V54 showed that an invented scalar teaching heuristic can confidently select the wrong events.

The connectome audit showed that the real teaching circuit is neither:

```text
one scalar feedback channel
```

nor obviously:

```text
one static sparse mask.
```

But the audit has not yet earned any specific dynamical equation.

Therefore the disciplined next move is:

\[
\boxed{
\text{measure}
\rightarrow
\text{replicate}
\rightarrow
\text{distill}
\rightarrow
\text{only then model}
}
\]

V55 is the **measure / replicate / distill** round.

It deliberately postpones the next behavioral mechanism until the fly data have actually constrained what that mechanism should be.
