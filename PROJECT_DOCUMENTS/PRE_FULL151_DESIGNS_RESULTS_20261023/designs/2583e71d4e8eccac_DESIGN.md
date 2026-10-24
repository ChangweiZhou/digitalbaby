# MiniFly Workstream A6 — Structured-Drift Value Matrix and Centering Headroom Audit

**Status:** Proposed next experiment  
**Model:** frozen `FULL_151`  
**Role:** read-only / counterfactual state-surgery audit  
**Learner modification:** none  
**Primary scientific question:** Is the structured component \(M\) that dominates A5 primarily a useful self-learning direction with collateral foreign harm, or a removable systematic bias that wastes bounded capacity?

---

## 1. Why A6 is the next experiment

A5 separated two phenomena that had previously been conflated.

The ordinary centered finite-capacity term behaves approximately as expected:

\[
\sigma_{\rm centered}
\propto
\sqrt{\frac{L_{\rm eff}}{d_{\rm eff}^{rw}}},
\]

while the observed foreign-write damage is overwhelmingly attributed to the structured component \(M\).

Across the completed capacity curve, \(M\) already dominates at the first nonzero foreign load and remains dominant at full load. The centered component \(C\) and the \(M\times C\) interaction remain much smaller.

This means that the next scientific problem is **not** to increase nominal memory capacity, add slots, add a relevance mechanism, or begin turnover.

The unresolved question is more basic:

\[
\boxed{
\text{What functional role does the structured write component }M\text{ play?}
}
\]

There are at least three qualitatively different possibilities.

### Hypothesis A — removable drift

The structured component contributes little to the memory being written but systematically harms other relevant memories.

Then a centering or balancing mechanism may recover a large fraction of current wasted capacity.

### Hypothesis B — useful-but-conflicting write direction

The same structured component is strongly beneficial to the source memory and harmful to other memories.

Then naive centering would trade acquisition for retention. The problem is not simply “remove the mean”; it is to separate self-benefit from foreign harm geometrically.

### Hypothesis C — block/state-specific artifact

The within-block role×context mean explains damage descriptively but does not define a stable direction across worlds/blocks.

Then a fixed global centering vector is not a plausible mechanism. The next theory must be state-dependent or locally inferred.

A6 is designed to distinguish these cases before any online intervention is attempted.

---

## 2. Frozen constraints

A6 makes no change to:

- neuron count;
- anatomy;
- `FULL_151` genes;
- reader;
- slow/fast time constants;
- plasticity rule;
- writable support;
- teaching signal;
- state budget;
- external memory;
- relevance information available to the learner.

No optimizer, evolution, learned gate, routing table, memory slot, or fitted behavioral threshold is introduced.

The experiment is an analysis and causal state-surgery audit only.

---

## 3. Reuse of A5 rather than another full capacity sweep

A5 has already established the entire \(K=1,\ldots,8\) load curve.

A6 therefore does **not** rerun all 112 lifetimes merely to reproduce that result.

Use three principled load conditions per target rotation:

\[
K=1,
\qquad
K=2,
\qquad
K=8.
\]

These are not arbitrary selected loads.

- \(K=1\) is the exact zero-foreign-learning baseline.
- \(K=2\) is the first nonzero foreign-load condition, where premature structured damage already appears.
- \(K=8\) is the inherited full FPF8 load.

For \(K=2\), retain both deterministic A5 rosters:

- `forward`;
- `reverse`.

For \(K=8\), forward/reverse are identical and remain one shared arm.

Thus each block requires at most:

\[
1 + 2 + 1 = 4
\]

instrumented trajectories, and the \(K=1\) trajectory is a baseline rather than a foreign-source arm.

Across eight target rotations this is at most 32 replayed trajectories, with existing A5 cache/state reused wherever instrumentation permits.

---

## 4. Why the A5 role×context mean is analysis-only

A5 used four prespecified source groups:

\[
g\in
\{
(\mathrm{acquisition},0),
(\mathrm{acquisition},1),
(\mathrm{challenge},0),
(\mathrm{challenge},1)
\}.
\]

For each target checkpoint \(T\), the foreign slow-state difference was decomposed as

\[
D_T=M_T+C_T.
\]

This was mathematically useful, but the labels “acquisition” and “challenge” are experimenter descriptions.

A6 must therefore preserve the distinction:

\[
\boxed{
M \text{ is an analysis object, not a deployable learner variable.}
}
\]

The purpose of A6 is to determine what kind of mechanism would eventually be needed—not to smuggle \(M\) into the learner.

---

## 5. Event-level source ledger

For every non-target teaching event \(t\), capture:

- source cohort \(c(t)\);
- source fact identity;
- source role \(r(t)\in\{0,1\}\);
- source context \(q(t)\in\{\mathrm{acquisition},\mathrm{challenge}\}\);
- event start/end times;
- slow-state pre/post vectors;
- exact slow innovation
  \[
  \xi_t=s_{t+1}-a_t s_t;
  \]
- actual scalar slow decay factor \(a_t\);
- source truth before/after the event;
- whether the event is an acquisition or challenge write;
- the full frozen non-slow state required for exact readout probes.

For non-target slots in the matched target-only baseline, the corresponding slow innovation must be zero to numerical precision.

The captured foreign-source ledger must reconstruct the A5/A4 \(D\) state in transition order.

---

## 6. Decay-weighted source-state decomposition

At checkpoint \(T\), define the retained state contributed by source cohort \(c\) and source group \(g\):

\[
S_{c,g}(T)
=
\sum_{\substack{t<T\\c(t)=c\\g(t)=g}}
A_{tT}\xi_t,
\]

where

\[
A_{tT}
=
\exp\left(
-\frac{T-t_{\rm end}}{\tau_s}
\right).
\]

Then

\[
D_T
=
\sum_{c\neq c_{\rm target}}
\sum_g
S_{c,g}(T).
\]

This identity is a qualification condition.

No scientific result is reported if the grouped source states fail to reconstruct the exact D ledger.

---

## 7. Destination cohorts and the value matrix

For every active destination cohort \(\nu\), evaluate the behavioral effect of each retained source-state component.

Let the native shared state at checkpoint \(T\) be \(s_T\).

For a retained source component \(S\), define the exact leave-component-out effect:

\[
V_{\nu,T}(S)
=
F_{\nu,T}(s_T)
-
F_{\nu,T}(s_T-S).
\]

Interpretation:

- \(V_{\nu,T}(S)>0\): the component helps destination cohort \(\nu\);
- \(V_{\nu,T}(S)<0\): the component harms destination cohort \(\nu\).

No linear approximation is required for this primary quantity.

For each source cohort \(c\), construct the matrix

\[
\boxed{
\mathcal V_{c\to\nu,g}(T)
=
V_{\nu,T}(S_{c,g}(T)).
}
\]

This is the central A6 object.

---

## 8. Self-benefit and foreign effect

For each source cohort \(c\) and source group \(g\), define the exact self effect

\[
B_{c,g}(T)
=
\mathcal V_{c\to c,g}(T).
\]

For foreign destinations, do **not** immediately compress to one score.

Report the complete set

\[
\{
\mathcal V_{c\to\nu,g}(T):
\nu\neq c
\}.
\]

Also report descriptive aggregates:

\[
H^{\rm mean}_{c,g}(T)
=
\frac{1}{K-1}
\sum_{\nu\neq c}
\mathcal V_{c\to\nu,g}(T),
\]

and

\[
H^{\rm sum}_{c,g}(T)
=
\sum_{\nu\neq c}
\mathcal V_{c\to\nu,g}(T).
\]

These are descriptive summaries, not optimization objectives.

The key visualization is the pair

\[
\boxed{
\left(
B_{c,g},
H^{\rm mean}_{c,g}
\right).
}
\]

A component with strong positive self benefit and strong negative foreign effect represents a genuine stability/plasticity conflict.

A component with small self benefit and strong negative foreign effect is candidate wasted capacity.

---

## 9. Separate mean and centered source components

Within each source group, define the empirical mean innovation

\[
\mu_{b,g}
=
E_{t\in(b,g)}[\xi_t]
\]

for block \(b\).

At checkpoint \(T\),

\[
M_{c,g}(T)
=
\mu_{b,g}
\sum_{\substack{t<T\\c(t)=c\\g(t)=g}}
A_{tT},
\]

and

\[
C_{c,g}(T)
=
S_{c,g}(T)-M_{c,g}(T).
\]

Then

\[
S_{c,g}=M_{c,g}+C_{c,g}.
\]

A6 evaluates the exact self/foreign effects of:

\[
M_{c,g},
\qquad
C_{c,g},
\qquad
S_{c,g}.
\]

This determines whether the source cohort's own benefit is also concentrated in the same structured mean component that generates foreign damage.

---

## 10. Exact within-source M/C Shapley decomposition

For one source-group state

\[
S=M+C,
\]

use the local background

\[
s_{\rm bg}=s_T-S.
\]

For destination cohort \(\nu\), define

\[
G_\nu(X)
=
F_\nu(s_{\rm bg}+X)-F_\nu(s_{\rm bg}).
\]

Then compute

\[
G_\nu(M),
\qquad
G_\nu(C),
\qquad
G_\nu(M+C).
\]

The exact two-component Shapley attributions are

\[
\psi_{M,\nu}
=
\frac12
\left[
G_\nu(M)
+
G_\nu(M+C)-G_\nu(C)
\right],
\]

\[
\psi_{C,\nu}
=
\frac12
\left[
G_\nu(C)
+
G_\nu(M+C)-G_\nu(M)
\right].
\]

They must satisfy

\[
\psi_{M,\nu}+\psi_{C,\nu}
=
G_\nu(S)
\]

to numerical precision.

This produces a direct answer to:

> Is the structured mean \(M\) useful to the source cohort itself, or does its value mostly appear as foreign damage?

---

## 11. Nested structured models

The four-way role×context decomposition used by A5 may be more detailed than necessary.

A6 therefore compares four prespecified nested mean models:

### G0 — global mean

One mean vector:

\[
\mu.
\]

### G1 — role-conditioned mean

Two means:

\[
\mu_{r=0},
\qquad
\mu_{r=1}.
\]

### G2 — context-conditioned mean

Two means:

\[
\mu_{\rm acquisition},
\qquad
\mu_{\rm challenge}.
\]

### G3 — role×context mean

Four means:

\[
\mu_{q,r}.
\]

No further partition is introduced after inspecting results.

For every model report:

- D-state reconstruction attributable to the structured component;
- target foreign-harm contribution;
- source self-benefit contribution;
- residual centered contribution;
- number of mean directions;
- effective rank of the mean-direction matrix.

The goal is to find the **simplest descriptive structure supported by the data**, not to declare a deployable rule.

---

## 12. Cross-block stability is mandatory

A within-block mean can explain a trajectory even if the mean direction changes arbitrarily from block to block.

Such an object would not support a fixed global balancing mechanism.

For every source group \(g\), measure pairwise alignment:

\[
\cos(\mu_{b,g},\mu_{b',g})
\]

for all block pairs \(b\neq b'\).

Also construct the block-by-vector mean matrix

\[
U_g
=
\begin{bmatrix}
\mu_{0,g}^\top\\
\vdots\\
\mu_{7,g}^\top
\end{bmatrix}.
\]

Report:

- singular values of \(U_g\);
- participation-ratio effective rank;
- pairwise cosine distribution;
- principal structured subspace across all groups.

No arbitrary “alignment threshold” is introduced.

---

## 13. Leave-one-block-out mean prediction

To distinguish a reusable structured direction from a descriptive same-block average, perform a leave-one-block-out analysis.

For held-out block \(b\), estimate

\[
\widehat{\mu}^{(-b)}_g
=
\frac{1}{7}
\sum_{b'\neq b}\mu_{b',g}.
\]

Do not use block \(b\) when constructing its predictor.

Using the held-out block's actual event times and group labels, construct

\[
\widehat M^{(-b)}_T
=
\sum_g
\widehat{\mu}^{(-b)}_g
\sum_{t\in g,b}
A_{tT}.
\]

Then measure:

1. alignment with the held-out native \(M_{b,T}\);
2. linear target projection;
3. exact target damage under state insertion/removal;
4. source-cohort self effect;
5. foreign-cohort effect.

This is the primary generalization test for whether the structured drift has a reusable low-dimensional direction.

---

## 14. Analysis-only centering headroom surgery

A6 includes one deliberately nondeployable causal audit.

For held-out block \(b\), use only the leave-one-block-out means

\[
\widehat\mu_g^{(-b)}.
\]

At checkpoint \(T\), define the predicted structured state

\[
\widehat M_T^{(-b)}.
\]

Construct the counterfactual state

\[
s_T^{\rm centered}
=
s_T-\widehat M_T^{(-b)}.
\]

Evaluate all active cohorts under:

- native state \(s_T\);
- held-out-centered state \(s_T^{\rm centered}\).

Report separately:

\[
\Delta U_{\rm source},
\]

\[
\Delta U_{\rm target},
\]

\[
\Delta U_{\rm other\ active}.
\]

This is **not** a deployable algorithm.

It is an answer to the causal question:

> If the stable cross-block structured mode were removed from stored slow state, how much existing useful memory would be lost and how much foreign damage would be recovered?

No future truth, relevance labels, or evaluator state is added to the learner because the learner is not modified at all.

---

## 15. No write-budget compensation in A6

Do not rescale \(C\) after subtracting \(M\).

Do not inject a random vector of matched norm.

Do not preserve total \(L_1\) or \(L_2\) slow-write mass by an arbitrary compensating transformation.

Those operations answer different questions and introduce unnecessary degrees of freedom.

A6 asks first whether the structured component carries behavioral value.

If later intervention design requires fixed-budget controls, they should be introduced in the intervention round after the causal object has been identified.

---

## 16. Linear read/write diagnostics

Exact behavioral state surgery is primary.

For interpretation, also compute the linear source-destination kernel

\[
K_{\nu,c,g}
=
r_\nu^\top M_{c,g}
\]

and centered counterpart

\[
K^C_{\nu,c,g}
=
r_\nu^\top C_{c,g}.
\]

This allows direct comparison with the A3/A4 read-write geometry.

Report:

- diagonal self terms;
- off-diagonal foreign terms;
- singular values of the source-destination transfer matrix;
- symmetric and antisymmetric parts;
- effective rank.

Do not infer mechanism from the linear kernel when exact nonlinear probes disagree.

---

## 17. Source-context decomposition

The A4/A5 conditional means suggest that acquisition and challenge writes may behave differently.

A6 therefore reports separate transfer matrices for:

\[
\mathrm{acquisition},
\qquad
\mathrm{challenge},
\]

and for the two role values within each context.

The key question is not “which group is bad?” but:

\[
\boxed{
\text{which source context has poor self-benefit / foreign-harm geometry?}
}
\]

For example:

- high self benefit + high foreign harm: genuine conflict;
- low self benefit + high foreign harm: wasteful drift candidate;
- high self benefit + low foreign harm: desirable write mode;
- low self benefit + low foreign harm: weak or irrelevant stored component.

These are descriptive regions, not thresholded classes.

---

## 18. K=2 versus K=8 comparison

A5 showed that structured damage appears immediately at \(K=2\).

A6 therefore explicitly compares the transfer geometry at:

\[
K=2
\]

and

\[
K=8.
\]

Questions:

1. Is the same structured mean direction already present at first foreign load?
2. Does self-benefit / foreign-harm geometry remain approximately stable with load?
3. Does the structured subspace rotate as state accumulates?
4. Does nonlinearity amplify an otherwise similar low-load transfer pattern?
5. Does the centered component become more behaviorally important only at high load?

The forward/reverse \(K=2\) arms are kept separate before averaging so order dependence is visible.

---

## 19. Primary outputs

Produce:

```text
PREFLIGHT.json
DESIGN_AUDIT.json

SOURCE_LEDGER_SUMMARY.json
GROUP_STATE_CLOSURE.json

SELF_FOREIGN_TRANSFER.json
SELF_FOREIGN_TRANSFER_K2.json
SELF_FOREIGN_TRANSFER_K8.json

MEAN_CENTERED_SHAPLEY.json
NESTED_MEAN_MODELS.json

MEAN_STABILITY.json
MEAN_SINGULAR_SPECTRA.npz
LOBO_MEAN_PREDICTION.json

CENTERING_HEADROOM.json
LINEAR_TRANSFER_KERNEL.json

per_block/
    block_00.json
    ...
    block_07.json

SUMMARY.json
DECISION.md
MANIFEST.json
```

---

## 20. Technical qualification

Science results are reportable only if:

1. frozen `FULL_151` source hashes match;
2. A2/A5 world and plan hashes match;
3. required A5 \(K=1\), \(K=2\), and \(K=8\) endpoint metrics regress to the completed A5 run;
4. event source labels reproduce the frozen schedule exactly;
5. source-event counts agree with the transformed worlds;
6. transition-order source replay reconstructs the slow-state D ledger;
7. grouped source states satisfy
   \[
   D=\sum_{c,g}S_{c,g};
   \]
8. each source state satisfies
   \[
   S_{c,g}=M_{c,g}+C_{c,g};
   \]
9. every exact M/C Shapley decomposition closes;
10. LOBO predictors use no held-out block data;
11. counterfactual centering uses only LOBO mean vectors;
12. no behavioral result determines whether an arm is run or reported.

Behavioral performance is never a technical qualification gate.

---

## 21. Internal tests required before native execution

The runner must include synthetic tests for:

- source cohort/context/role ledger construction;
- matched-rest source silence;
- transition-order D reconstruction;
- grouped-state reconstruction;
- exact \(S=M+C\);
- exact leave-component-out effect;
- exact two-component Shapley identity;
- nested global/role/context/role×context models;
- leave-one-block-out data separation;
- cross-block mean matrix singular-spectrum calculation;
- source-destination matrix indexing;
- K=2 forward/reverse preservation;
- K=8 shared-arm identity;
- exact A5 endpoint regression fixture;
- cache isolation by runner version;
- source audit confirming no optimizer, no action cap, no behavioral capacity threshold, and no online centering rule.

A native block-0 pilot should run before expanding to all eight blocks.

---

## 22. What A6 is allowed to conclude

### Outcome A — structured mode is mostly foreign harm

If held-out \(M\) has:

- small source self benefit;
- large negative foreign effect;
- stable cross-block direction;
- positive centering headroom;

then A6 supports a subsequent **balancing/centering intervention round**.

It still does not identify the deployable local signal.

### Outcome B — structured mode is useful but conflict-heavy

If held-out \(M\) strongly benefits its source cohort and strongly harms other cohorts, then naive centering is not appropriate.

The next mathematical problem becomes:

\[
\boxed{
\text{find a local transformation that preserves the self component while reducing foreign projection.}
}
\]

This points toward geometry-aware eligibility or inhibition, not simple subtraction.

### Outcome C — structured mode does not generalize across blocks

If within-block \(M\) dominates but LOBO mean directions poorly reproduce held-out effects, then the A5 decomposition is descriptive rather than a fixed global correction target.

The next round should search for the local state variables that predict the harmful direction.

### Outcome D — context-specific asymmetry

If one source context has much worse self-benefit / foreign-harm geometry than another, the next round should target the local state distinction corresponding to that context rather than all slow writes.

### Outcome E — centered capacity term becomes important only at K=8

This would preserve the A5 interpretation:

- structured drift is premature wasted capacity;
- centered finite-dimensional collision is the genuine high-load background capacity limit.

Both should remain separate in future models.

---

## 23. Explicit exclusions

A6 does not introduce:

- relevance labels;
- forgetting policies;
- LFU/LRU-style heuristics;
- memory slots;
- task IDs;
- context IDs inside the learner;
- replay;
- extra neurons;
- extra persistent memory;
- learned gating;
- topology evolution;
- dynamic rewiring;
- consolidation stages;
- an online mean-subtraction mechanism.

These remain out of scope.

---

## 24. Why turnover remains deferred

The completed A5 run shows that the current learner loses substantial useful performance because of structured interference before centered finite-capacity collision dominates.

If turnover were introduced now, two qualitatively different losses would be mixed:

\[
\text{intentional capacity reuse}
\]

and

\[
\text{premature structured corruption}.
\]

A6 therefore finishes the capacity-efficiency audit first.

Only after the avoidable structured component is understood should the project ask which memories ought to be forgotten.

---

## 25. Scientific stopping rule

A6 ends once the following four questions are answered:

\[
\boxed{
Q1:
\text{Does }M\text{ provide significant self-memory benefit?}
}
\]

\[
\boxed{
Q2:
\text{How much foreign harm does the same component create?}
}
\]

\[
\boxed{
Q3:
\text{Are its directions stable enough across blocks to define a reusable low-dimensional mode?}
}
\]

\[
\boxed{
Q4:
\text{Does held-out centering recover foreign performance without destroying source performance?}
}
\]

No online intervention is designed inside A6.

---

## 26. Desired theoretical outcome

The desired reduced model is a source-destination transfer law.

For source cohort \(c\), destination cohort \(\nu\), and source context \(g\),

\[
\Delta U_{\nu}
\approx
\mathcal T_{\nu c g}^{(M)}
+
\mathcal T_{\nu c g}^{(C)}
+
R_{\nu c g}^{\rm nonlinear}.
\]

The central object is the structured transfer matrix

\[
\boxed{
\mathcal T^{(M)}.
}
\]

If its diagonal entries are strongly positive while off-diagonal entries are systematically negative, MiniFly has a true distributed-memory stability/plasticity conflict.

If its diagonal entries are weak but off-diagonal entries are negative, MiniFly is wasting bounded capacity through a removable shared bias.

If the matrix varies substantially across blocks, the required correction is state-dependent rather than global.

This is the information needed before designing any biologically plausible balancing mechanism.

---

# Final question

A6 is designed to answer:

\[
\boxed{
\text{Is the structured }M\text{ mode a useful memory signal with collateral damage,
or primarily wasted capacity that can be centered away?}
}
\]

That question should be settled before introducing either an online centering rule or relevance-weighted turnover.
