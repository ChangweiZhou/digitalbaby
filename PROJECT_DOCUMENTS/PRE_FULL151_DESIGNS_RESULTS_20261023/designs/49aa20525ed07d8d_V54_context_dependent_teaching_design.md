# V54 — Context-Dependent Teaching on the Frozen Fly-Like Core

**Status:** implementation-ready design  
**Date:** 12 September 2026  
**Primary question:** can a causal, learner-available teaching signal reduce persistent interference without blocking new acquisition, revision, or ordinary byte prediction?  
**Platform target:** Apple Silicon Mac M3  
**Primary substrate:** frozen V50/V51 PN→KC→MBON core, as exercised in V52/V53  
**Version role:** one controlled teaching-signal experiment; not a new memory architecture

---

# 0. Executive decision

V54 does **not** introduce:

- a fast/slow synaptic memory;
- a second persistent memory store;
- slots;
- replay;
- a new cue-learning algorithm;
- a new phase-normalization family;
- a new eligibility buffer;
- learned routing;
- new compartment masks;
- a new PN→KC graph;
- target-position labels.

The sole proposed mechanism is:

\[
\boxed{
\text{modulate the existing KC}\rightarrow\text{MBON update
according to whether long context explains the observed byte better than a bounded local predictor}
}
\]

The persistent learned state remains the existing KC→MBON weight vector.

The primary candidate is called:

```text
CONTEXT-GAIN-1
```

It uses a bounded **one-byte Markov predictor** only to modulate teaching strength.  
It does **not** contribute directly to the readout.

A two-byte Markov version is retained only as an expensive robustness diagnostic:

```text
CONTEXT-GAIN-2
```

It cannot be the minimal-production winner.

---

# 1. Evidence frozen before V54

V54 is designed around the following established findings.

## 1.1 V53 representation × writing factorial

At load 128, fresh V53 seeds showed approximately:

| Interface | Write policy | Old A after D | New D after D | \(U=\min(A,D)\) |
|---|---|---:|---:|---:|
| V51-LAG | ALL | 0.012 | 0.059 | 0.006 |
| V51-LAG | TARGET-ONLY | 0.219 | 0.352 | 0.203 |
| INVARIANT-CUE | ALL | 0.340 | 0.789 | 0.340 |
| INVARIANT-CUE | TARGET-ONLY | 0.926 | 0.990 | 0.922 |

For INVARIANT-CUE:

\[
\Delta U_{\rm target-only-all}
\approx +0.582
\]

with a paired 95% interval approximately:

\[
[+0.517,\,+0.647].
\]

Thus, in this fixed fixture and learner:

1. improving the associative cue materially helps;
2. suppressing non-target writes materially helps even when the cue is good.

These are intervention effects. V54 does **not** claim they imply two mandatory biological modules.

## 1.2 TARGET-ONLY is not a deployable solution

TARGET-ONLY sharply improves associative retention but has poor whole-stream prediction:

\[
\text{whole-stream bpb}\approx 7.94
\]

versus approximately:

\[
3.86
\]

for INVARIANT-CUE + ALL.

Therefore the V54 problem is not:

> identify evaluator target bytes and ignore everything else.

It is:

> preserve durable context-dependent associations **while continuing to learn the rest of the stream**.

## 1.3 Fast/slow storage is not a new hypothesis

Historical experiments already tested:

- two-state and three-state fast/slow synapses;
- feedback consolidation;
- fast generalizing + slow durable complementary memory;
- multiple related consolidation constructions.

Those experiments do not prove that all multi-timescale dynamics are impossible, but they remove “add fast and slow memory” as a fresh V54 idea.

## 1.4 V25 FIFO is not autonomous attribution

V25 demonstrated that an ordered, complete delayed-feedback channel can use the correct pending trace.

It did **not** show that an autonomous learner identifies which event in an unsegmented stream deserves credit.

OMNIBUS2's completed unselected-trace arms did not recover the oracle known-delay result.

V54 therefore does not introduce another delayed trace-selection problem.

## 1.5 Corrected fly-data interpretation

The corrected hemibrain randomization does **not** establish that PN→KC wiring avoids same-glomerulus convergence, nor does it establish that PN→KC wiring creates invariance.

The anatomical data support only a more modest constraint:

- sensory PN→KC connectivity;
- KC output/readout connectivity;
- dopamine-related/modulatory connectivity

are anatomically distinguishable pathways.

The fly data do **not** supply the V54 gain equation.

V54 therefore tests an engineered teaching computation on the preserved anatomical core. It does not claim to reconstruct DAN physiology.

---

# 2. V54 hypothesis

Let:

\[
p_t(y)
\]

be the existing V50/V51 context-conditioned prediction for the current byte before the byte is observed.

Let:

\[
q_t(y)
\]

be a bounded local predictor that uses only the immediately preceding byte.

After the actual byte \(y_t\) is observed, compute:

\[
G_t
=
\log_2 p_t(y_t)
-
\log_2 q_t(y_t).
\]

Interpretation:

- \(G_t>0\): the long/context-dependent system explained the observation better than the local surface predictor;
- \(G_t<0\): local surface statistics explained it better.

V54 asks:

> Does using \(G_t\) only to modulate the existing synaptic teaching update improve persistent association without sacrificing ordinary stream learning?

This is **not** asserted to be dopamine, mutual information, or a biological credit signal.

It is an experimentally falsifiable engineering hypothesis.

---

# 3. Frozen learner substrate

Unless an arm explicitly says otherwise, freeze exactly:

- BANC PN→KC graph;
- 699 PNs;
- 4,552 KCs;
- 104 MBONs;
- leaky KC dynamics;
- 5% KC code / 228 active KCs;
- existing KC→MBON anatomical support;
- 15,523 float64 learned KC→MBON scalars;
- original V50/V51 output codebook;
- original byte serialization;
- original V51 learning rate \(\eta=0.5\);
- no replay;
- no slots;
- no consolidation state;
- no eligibility trace buffer;
- no forgetting term;
- no new recurrent circuit;
- no learned router.

The only change in the V54 candidate is a scalar multiplier on the existing per-byte KC→MBON update.

---

# 4. Representation conditions

V54 uses two representation conditions, kept conceptually separate.

## R0 — `V51-LAG`

The actual current byte interface.

Purpose:

> test whether the teaching mechanism transfers to the unresolved real BYTE interface.

## R1 — `INVARIANT-CUE`

The exact V53 oracle diagnostic:

- at evaluator target events only, substitute the invariant four-key-byte cue;
- at all other byte events, retain the original V51 representation.

This is **nondeployable**.

Purpose:

> isolate teaching/write interference under a representation already known to support strong persistent association.

A positive result only in R1 means:

```text
TEACHING_GAIN_CONDITIONAL_ON_USABLE_CUE
```

It does not solve autonomous cue formation.

---

# 5. Auxiliary local predictor

## 5.1 Primary local predictor — `MARKOV1`

Use the immediately preceding byte only.

Maintain:

\[
N[a,b]
\]

for previous byte \(a\) followed by byte \(b\), plus:

\[
N[a]=\sum_b N[a,b].
\]

Prediction:

\[
q_t(b\mid a)
=
\frac{N[a,b]+1}
     {N[a]+256}.
\]

If no previous byte exists, use a global additive-one byte-frequency predictor.

Update counts **after** scoring the current byte.

Properties:

- causal;
- role-blind;
- no record length;
- no target-position input;
- no identity input;
- bounded state;
- fixed alphabet size.

Suggested storage:

- `uint32[256,256]`;
- `uint32[256]`;
- global `uint32[256]`.

Approximate learned state:

\[
\sim 0.26\ {\rm MB}.
\]

This is charged to the mechanism.

## 5.2 Robustness predictor — `MARKOV2`

Reuse the exact V51 Markov2 baseline:

- last two bytes when available;
- one-byte fallback;
- global fallback;
- additive-one smoothing.

This costs roughly tens of MB and is therefore **diagnostic only**.

If only MARKOV2 works:

```text
CONTEXT_TEACHING_REQUIRES_EXPENSIVE_LOCAL_MODEL
```

V54 is not qualified as a minimal architecture.

---

# 6. Teaching modulation rule

The primary arm uses `MARKOV1`.

Before observing \(y_t\):

1. compute the frozen-core prediction \(p_t\);
2. compute the local prediction \(q_t\).

After observing \(y_t\):

\[
G_t
=
\log_2 \max(p_t(y_t),\epsilon)
-
\log_2 \max(q_t(y_t),\epsilon).
\]

Use:

\[
g_{\min}=0.25,
\qquad
\tau=1\ {\rm bit}.
\]

Define:

\[
\boxed{
g_t
=
\operatorname{clip}
\left(
2^{G_t/\tau},
g_{\min},
1
\right).
}
\]

Therefore:

- if context is at least as predictive as the local model:
  \[
  g_t=1;
  \]
- one bit worse:
  \[
  g_t\approx0.5;
  \]
- two or more bits worse:
  \[
  g_t=0.25.
  \]

The existing V50 update becomes:

\[
\boxed{
\Delta w_{ij}
=
g_t\,
\eta\,
z_i\,
\delta_j.
}
\]

No other learner equation changes.

### Why a nonzero floor is mandatory

A hard \(G_t>0\) gate would have two obvious failure modes.

**Cold start:** before an association is learned, the context system may not yet outperform the local predictor.

**Revision:** after a genuine mapping change, the old context prediction can initially be confidently wrong.

A zero floor can therefore suppress exactly the examples necessary for acquisition or revision.

V54 does not assume \(g_{\min}=0.25\) solves those problems. It makes them directly testable.

There is **no gain sweep** in the scientific run.

---

# 7. Primary arm roster

The primary mechanistic experiment runs under **INVARIANT-CUE**.

## A0 — `ALL`

Normal V51 all-byte learning.

\[
g_t=1.
\]

Deployable baseline.

## A1 — `TARGET-ONLY`

Evaluator-gated target-only writing.

Nondeployable ceiling/reference.

It is not eligible to qualify.

## A2 — `CONTEXT-GAIN-1`

Primary V54 candidate.

Uses MARKOV1 and the gain equation above.

Deployable with respect to stream information.

## A3 — `CONTEXT-GAIN-2`

Same mechanism using exact V51 MARKOV2.

Diagnostic only due state cost.

## A4 — `CONST-BLOCK-MATCH`

Nondeployable budget control.

First run A2 and record its gain sequence:

\[
g_1,\ldots,g_T.
\]

Split time into fixed **512-byte chronological blocks**.

Within each block \(B\), use:

\[
\bar g_B
=
\frac{1}{|B|}
\sum_{t\in B} g_t
\]

for every event in that block.

This preserves local modification budget and broad temporal nonstationarity while destroying event-specific selection.

It uses no byte roles.

## A5 — `SHUFFLED-BLOCK-MATCH`

Nondeployable specificity control.

Within each 512-byte block:

- retain the exact multiset of A2 gains;
- deterministically permute gains across byte events.

This preserves:

- gain distribution;
- gain sum;
- blockwise time dependence;

while destroying context/event correspondence.

## A6 — `SURPRISE-BLOCK-MATCH`

Nondeployable alternative-criterion control.

Within each fixed 512-byte block:

1. compute local-model surprise:
   \[
   S_t=-\log_2 q_t(y_t);
   \]
2. take the exact A2 gain multiset from that block;
3. assign the largest gains to the most surprising events.

Thus A6 has the same per-block gain distribution and budget as A2.

If A6 matches A2, the evidence supports generic surprise prioritization rather than contextual predictive advantage.

---

# 8. Instrumentation invariance test

The MARKOV1 predictor must not affect the learner except through \(g_t\).

Engineering arm:

```text
AUX-NEUTRAL
```

runs MARKOV1 normally but forces:

\[
g_t=1.
\]

It must reproduce `ALL` bit-for-bit for:

- main-memory weights;
- predictions;
- probes;
- update counts.

This is a self-test, not a science arm.

---

# 9. Primary experimental condition

Run the complete A0–A6 roster under:

```text
INVARIANT-CUE
```

at load:

\[
128.
\]

Fresh seeds:

```text
54601 ... 54616
```

No parameter selection occurs on these seeds.

The primary claim is conditional:

> Can context-dependent teaching improve the persistent-learning tradeoff when a usable associative cue is already available?

---

# 10. Raw-interface transfer condition

If `CONTEXT-GAIN-1` clears the primary mechanism gate, run a transfer block under:

```text
V51-LAG
```

using the same fixed parameters.

Arms:

- `ALL`;
- `TARGET-ONLY` — oracle ceiling only;
- `CONTEXT-GAIN-1`;
- `CONST-BLOCK-MATCH`;
- `SHUFFLED-BLOCK-MATCH`.

Do not retune \(g_{\min}\), \(\tau\), MARKOV order, or learning rate.

This block answers:

> Does the teaching mechanism help the actual unresolved BYTE interface?

A negative raw result does not erase a valid conditional teaching result.

---

# 11. Required endpoints

## 11.1 Persistent association

Report:

- A held-out after A;
- A first return after B;
- A after C;
- untouched A after D;
- B/C/D after D;
- revised A after D.

Primary persistent endpoint:

\[
\boxed{
U
=
\min(
A_{\rm afterD,heldout},
D_{\rm afterD,heldout}
).
}
\]

## 11.2 Acquisition

For every cohort:

- prequential target bpb by exposure;
- target accuracy by exposure;
- exposure-to-criterion;
- right-censor failures.

Especially report:

- exposure 1;
- exposure 2;
- exposure 4;
- exposure 8.

## 11.3 Revision

Report revised target accuracy and probability after:

\[
1,\ 2,\ 4,\ 8
\]

consistent revised exposures.

Also report revised mapping after D.

## 11.4 Isolated corruption

Preserve V51/V53 one-off wrong-target probe.

Measure whether the mechanism:

- resists isolated corruption;
- still accepts sustained revision.

These two outcomes must not be collapsed into one “stability” score.

## 11.5 Ordinary stream learning

Report:

- whole-stream bpb;
- framing bpb;
- key bpb;
- target bpb;
- classwise accuracy.

Persistent-memory improvement is not sufficient if ordinary stream prediction collapses toward TARGET-ONLY behavior.

---

# 12. Same-outcome selective-revision diagnostic

This is evaluator-only probing.

For each revised identity \(r\), find unrevised identities \(u\) satisfying:

\[
y_u^{old}=y_r^{old}
\]

where possible.

After revising \(r\):

measure on \(u\):

- probability of the unchanged old outcome;
- probability of \(r\)'s new outcome;
- top-1 accuracy.

This tests whether the mechanism implicitly confuses:

\[
\text{same output category}
\]

with:

\[
\text{same identity}.
\]

No such relation enters the learner.

---

# 13. Teaching-gain diagnostics

For A2/A3 report:

- mean gain by evaluator byte class:
  - framing;
  - key;
  - target;
- mean gain by byte position;
- mean gain by cohort;
- mean gain by exposure;
- mean gain before/after revision;
- distribution of \(G_t\);
- fraction at:
  - floor;
  - interior;
  - full gain;
- correlation with:
  - local surprise;
  - main-model surprise;
  - target correctness;
  - update norm.

These are diagnostics only.

A result is not accepted merely because the learned gain happens to correlate with evaluator target position.

---

# 14. Update-budget accounting

For every arm report:

\[
\sum_t g_t,
\]

\[
\sum_t \|\Delta w_t\|_2,
\]

\[
\sum_t \|\Delta w_t\|_1,
\]

number of nonzero update events, and final:

\[
\|w\|_2.
\]

A2, A4, A5, and A6 share the same A2 per-block gain budget by construction.

Actual update norms may still differ because model states/residuals diverge.

If A2's cumulative update norm differs from every matched control by more than **20%**, label the primary specificity interpretation:

```text
UPDATE_NORM_CONFOUNDED
```

and do not claim selection-specific superiority from gain matching alone.

---

# 15. Primary gates

All comparisons are paired by seed.

## G0 — implementation validity

Required:

- AUX-NEUTRAL bitwise equals ALL;
- MARKOV1 predictions are causal;
- local predictor updates after scoring;
- no byte role enters A2/A3;
- no record period enters A2/A3;
- no target position enters A2/A3;
- all probes read-only;
- deterministic resume;
- worker-count determinism.

Failure:

```text
IMPLEMENTATION_INVALID
```

## G1 — persistent benefit over ALL

For A2 under INVARIANT-CUE:

\[
\Delta U
=
U_{A2}-U_{ALL}
\ge 0.10
\]

and the paired 95% lower bound must be above zero.

## G2 — not merely lower-rate learning

A2 must beat `CONST-BLOCK-MATCH` on \(U\):

\[
\Delta U\ge0.05
\]

with paired lower bound above zero.

## G3 — event correspondence matters

A2 must beat `SHUFFLED-BLOCK-MATCH` on \(U\):

\[
\Delta U\ge0.05
\]

with paired lower bound above zero.

## G4 — contextual information beats generic surprise

A2 must beat `SURPRISE-BLOCK-MATCH` on \(U\).

For mechanism qualification require:

\[
\Delta U > 0
\]

with paired lower bound above zero.

No additional 5-point effect threshold is imposed here.

## G5 — new acquisition preserved

For new-D held-out accuracy:

\[
D_{A2}
\ge
D_{ALL}-0.05.
\]

Also report the full exposure curve.

## G6 — revision preserved

At 8 revised exposures:

\[
Rev8_{A2}
\ge
Rev8_{ALL}-0.05.
\]

And revised-after-D:

\[
RevD_{A2}
\ge
RevD_{ALL}-0.05.
\]

## G7 — ordinary stream prediction preserved

\[
{\rm bpb}_{whole,A2}
-
{\rm bpb}_{whole,ALL}
\le +0.10.
\]

## G8 — no category merging

On unrevised same-old-outcome identities, the revision of a paired identity may reduce correct old-outcome probability by no more than 0.05 relative to ALL.

---

# 16. Qualification vocabulary

## `CONTEXT_TEACHING_SUPPORTED`

All G0–G8 pass under INVARIANT-CUE.

Interpretation:

> a causal context-versus-local teaching signal improves persistent learning beyond matched modification-budget controls, conditional on a usable cue.

## `CONTEXT_TEACHING_RAW_TRANSFER`

`CONTEXT_TEACHING_SUPPORTED` plus the raw V51-LAG transfer block shows:

\[
\Delta U\ge0.05
\]

over raw ALL with paired lower bound above zero and whole-stream noninferiority.

This is stronger, but still not autonomous cue learning.

## `RATE_EFFECT_ONLY`

A2 beats ALL but fails G2.

## `TIMING_SELECTION_NOT_SPECIFIC`

A2 fails to beat shuffled gains.

## `GENERIC_SURPRISE_SUFFICIENT`

A2 fails to beat surprise-ranked gains.

## `RETENTION_BY_UNDERLEARNING`

Persistent retention improves but G5 or G6 fails.

## `STREAM_PREDICTION_TRADEOFF`

G7 fails.

## `OUTPUT_CATEGORY_MERGING`

G8 fails.

## `CONTEXT_TEACHING_REQUIRES_EXPENSIVE_LOCAL_MODEL`

A3 works but A2 does not.

## `NO_CONTEXT_TEACHING_GAIN`

A2 does not clear G1.

---

# 17. Why MARKOV1 is primary

The local predictor should model **surface-local regularity**, not solve the same associative task.

One previous byte is sufficient to capture many local serialization transitions:

- newline → prefix;
- prefix transitions;
- separator-related local structure;
- repeated framing.

At the target byte, however, a one-byte model sees primarily the immediately preceding separator and target marginal distribution.

It therefore cannot directly recover six-byte identity.

That makes it a useful contrast against the long/context-dependent fly core.

MARKOV2 is retained because a false negative could otherwise arise from a grossly underfit local predictor.

But its much larger state prevents it from qualifying as the minimal candidate.

---

# 18. Cold-start interpretation

For the first presentation of a new identity, both models may be near uniform:

\[
p_t(y)\approx q_t(y)\approx1/256.
\]

Then:

\[
G_t\approx0
\]

and:

\[
g_t=1.
\]

Thus the rule does not automatically suppress first acquisition.

This is an important difference from a hard positive-information gate.

It is still possible that later early-learning events are downweighted.

That possibility is measured by the acquisition curves and G5.

---

# 19. Revision interpretation

After a well-learned association changes, the context model can initially assign very low probability to the new target.

Then:

\[
G_t<0
\]

and V54 attenuates, but never eliminates, learning:

\[
g_t\ge0.25.
\]

Therefore revision can be slower.

V54 explicitly treats that as a falsification opportunity.

A persistent learner that improves old retention by refusing to revise is not a success.

---

# 20. Fly-data role in V54

The biological data play a **constraint**, not equation-selection, role.

V54 uses the following modest inference:

> sensory representation and modulatory/plasticity control can be represented as distinct functional pathways without inventing another persistent memory store.

V54 does **not** infer from the connectome:

- \(G_t\);
- \(g_{\min}\);
- \(\tau\);
- MARKOV order;
- dopamine sign;
- compartment-specific update equations.

No cross-specimen neuron join enters the experiment.

DoOR does not enter V54.

No new large connectome dataset is required.

---

# 21. Explicit exclusions

The following are out of scope for V54:

- new cue-learning mechanism;
- repaired Claude RELIABILITY-v1;
- PAIR retraining;
- phase-normalization sweep;
- fast/slow weights;
- fast/slow stores;
- slot memory;
- replay;
- FIFO/delayed-trace competition;
- surprise-only gate as a production candidate;
- novelty gate;
- MESU;
- covariance protection;
- random bank routing;
- new KC sparsity level;
- new PN→KC topology;
- learned DAN compartments;
- APL recurrence;
- MBON recurrence;
- load-256 scaling.

The absolute-surprise arm exists only as a matched criterion control.

---

# 22. Seeds

Engineering:

```text
54771, 54772, 54773, 54774
```

Primary science:

```text
54601 ... 54616
```

The mechanism and all thresholds are frozen before science-seed inspection.

There is no screen-seed candidate selection.

MARKOV1 is primary by design.

MARKOV2 is diagnostic by design.

---

# 23. Mac M3 execution design

Single self-contained script:

```text
elm_v54_context_teaching.py
```

Recommended modes:

```bash
python3 elm_v54_context_teaching.py --self-test

caffeinate -dimsu python3 elm_v54_context_teaching.py \
    --stage primary \
    --workers 2 \
    --out-dir v54_results

caffeinate -dimsu python3 elm_v54_context_teaching.py \
    --stage raw-transfer \
    --workers 2 \
    --out-dir v54_results

python3 elm_v54_context_teaching.py \
    --stage aggregate \
    --out-dir v54_results
```

The full-stream learner should cap scientific multiprocessing at:

```text
2 workers
```

on the Mac M3 unless engineering benchmarks demonstrate lower memory pressure.

Every seed × representation × arm job is atomically cached.

Cache identity must include:

- V54 namespace/version;
- source SHA256;
- V51/V53 source SHA256;
- seed;
- representation;
- arm;
- local predictor order;
- \(g_{\min}\);
- \(\tau\);
- block size;
- workload/load;
- codebook hash.

Interrupted jobs may be rerun individually.

---

# 24. Required output files

```text
SUMMARY.md
REPORT.md
PROTOCOL.md
MANIFEST.json
IMPLEMENTATION_AUDIT.json
GATES.json

TERMINAL.csv
PERSISTENCE.csv
FIRST_RETURN.csv
ACQUISITION_CURVES.csv
EXPOSURES_TO_CRITERION.csv
REVISION_CURVES.csv
REVISED_AFTER_D.csv
NOISE_RESISTANCE.csv
SAME_OUTCOME_REVISION.csv

BYTE_CLASS_METRICS.csv
PREQUENTIAL.csv.gz

GAIN_EVENTS.csv.gz
GAIN_BY_CLASS.csv
GAIN_BY_POSITION.csv
GAIN_BY_EXPOSURE.csv
GAIN_DISTRIBUTION.csv

UPDATE_BUDGET.csv
STATE_COSTS.csv
RUNTIME.csv

CONTROL_CONTRASTS.csv
PRIMARY_CONTRASTS.csv

jobs/
checkpoints/
```

---

# 25. State accounting

At minimum report separately:

```text
kc_mbon_learned_bytes
markov1_learned_bytes
markov2_learned_bytes
fixed_graph_bytes
fixed_adaptor_bytes
codebook_bytes
transient_kc_bytes
cached_gain_bytes
```

Qualification of A2 requires bounded state independent of number of identities and stream length.

Gain caches used for nondeployable matched controls are diagnostic storage and must be reported separately.

They do not count as deployable learner state, but they cannot be hidden.

---

# 26. Primary interpretation matrix

| Result | Interpretation |
|---|---|
| A2 beats ALL + constant + shuffled + surprise, preserves acquisition/revision/stream | Context-dependent teaching supported |
| A2 beats ALL but constant matches it | Lower effective learning rate explains benefit |
| Shuffled gains match A2 | Gain amount/timing distribution matters, not event content |
| Surprise-ranked gains match A2 | Generic surprise is sufficient; contextual advantage not established |
| Retention rises but new D/revision fall | Stability obtained by underlearning |
| Oracle-cue passes, raw fails | Teaching result conditional on cue; representation remains blocker |
| Raw also improves | First deployable evidence that teaching modulation helps the actual byte interface |
| MARKOV2 works, MARKOV1 fails | Mechanism requires a much larger local model; minimal claim fails |
| None pass | Retire this specific M098 teaching construction |

---

# 27. What V54 can and cannot establish

A positive V54 can establish:

> within the frozen byte learner, a causal context-versus-local predictive comparison can allocate durable KC→MBON plasticity more effectively than matched reduction, shuffled allocation, and generic surprise controls.

It cannot establish:

- that the fly computes \(G_t\);
- that dopamine represents predictive information;
- that cue formation is solved;
- that the invariant-cue oracle is deployable;
- that all learning problems reduce to teaching modulation;
- that MARKOV1 is biologically realistic;
- that the fly connectome proves this equation;
- that load scaling is solved.

---

# 28. Stop rule

If `CONTEXT-GAIN-1` fails the primary oracle-cue mechanism gate, do **not** respond by:

- sweeping \(g_{\min}\);
- sweeping \(\tau\);
- trying many Markov orders;
- adding fast/slow state;
- adding a trace buffer;
- adding another memory protector.

The tested hypothesis is then:

```text
context-vs-local predictive advantage, in this fixed bounded modulation form
```

and should be reported as unsupported.

If it succeeds, the result earns a later question:

> can the same teaching principle coexist with an autonomously learned usable cue?

That is not part of V54.
