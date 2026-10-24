# V47 — Signed Predictive Representation Learning
## Can a bounded online representation learn both what should be invariant and what must remain distinct, so one fixed distributed slow memory avoids V45/V46 cross-identity interference?

**Status:** refined locked design / executable implementation specification  
**Date:** 11 September 2026  
**Target platform:** Apple Silicon Mac M3, macOS, Python 3.11+  
**Scientific results observed before this document:** V45 and V46 only, plus the earlier Candidate-C representation results and Claude's post-hoc audit of V40–V44. No V47 outcome has been observed.

---

# 0. Research constitution — no drift

The project objective remains fixed:

> Build a **bounded-state, non-Transformer, online persistent learner** that learns from sequential experience, retains useful knowledge through intervening learning, generalizes beyond exact presentations, revises obsolete knowledge, and does not require replay or full retraining.

V47 follows the branch explicitly locked in the V46 design:

> If learned overlapping pathways fail to reduce selective interference, do **not** tune pathway count/routing hyperparameters and do **not** return to slots. The next branch is **positive invariance + negative near-collision representation learning**.

V47 therefore changes **one scientific object only**:

\[
\boxed{\text{the frozen representation used by the same bounded global distributed slow memory}}
\]

The durable substrate, write schedule, deployed reader, FAST system, BASE system, state budget, persistent workload, and evaluation protocol remain the V45/V46 global-matrix reference.

V47 may **not** add or tune:

- permanent object slots;
- slot capacity, admission, eviction, LFU/LRU, or utility replacement;
- slow pathways, routing, banks, compartments, or expert mixtures;
- Candidate-B factor-coded outputs;
- TRACE-CORR or any new object-level corroboration mechanism in the persistent stream;
- new slow-write schedules or write-rate sweeps;
- replay/rehearsal;
- online movement of the final representation during A→B→C→D;
- Transformer/backpropagation training;
- larger slow matrices;
- load-dependent state growth;
- evaluator identity, wrapper, cohort, target-position, or revision labels inside the learner.

If V47 fails, these exclusions are not relaxed mid-run.

---

# 1. Evidence entering V47

## 1.1 V45: positive invariance is real but increases some harmful sharing

V45 established that Candidate-C-style PAIR representation learning is reproducible.

At load 256:

\[
\Delta(\text{PAIR geometry margin}-\text{RAW}) \approx +0.1073.
\]

PAIR also improved delayed held-out bpb relative to RAW under matched writes by approximately

\[
-0.0582\text{ bit},
\]

but this did not meet the locked 0.10-bit persistent-gain threshold.

The direct write-interference audit showed the key failure:

- useful transfer to alternate renderings of the same identity increased;
- useful transfer among different identities with the same outcome increased;
- **harmful different-identity / different-outcome cross-talk also increased.**

Thus PAIR learned useful *positive equivalence* but did not sufficiently preserve distinctions between superficially similar experiences that require incompatible predictions.

V45 design SHA256:  
`00eb0a01b183f83604d938ca4d427efcfa372a1d02297772cced062ef508b4c8`

V45 completed summary SHA256:  
`432886dbdadfe9dbe8ac59e6b7b85463ae48d642a47f82a4c1b817023d402d95`

## 1.2 V46: downstream learned routing did not repair the representation collision

V46 retained PAIR but replaced the one global slow matrix by learned overlapping slow pathways.

The preregistered verdict was:

`NO_SELECTIVE_INTERFERENCE_GAIN`.

At load 256:

\[
I_{\rm diff}^{\rm PRED}-I_{\rm diff}^{\rm GLOBAL}
=
+0.0004446\text{ bpb/write},
\]

95% CI approximately

\[
[+0.0001788,+0.0007104].
\]

Thus prediction-shaped pathway responsibility made the harmful different-outcome interference **worse**, not better.

The pathway system did not collapse:

- all eight pathways remained active;
- effective pathway count stayed near eight;
- acquisition passed;
- state stayed bounded.

Therefore the failure is not explained by dead routing or acquisition invalidity.

V46's locked stop condition already specified the next branch:

> learn both **positive invariance** and **negative near-collision separation** in the representation itself.

V46 design SHA256:  
`fc5df017aee4311509b514d7f6b511d06b0b73d746958aaf6f66cff7f746800a`

V46 completed summary SHA256:  
`d5bcc377070c7e0e11dcf2cc61b9501cc3de2eb900081aca4b392383dd07779c`

## 1.3 Claude's audit contributes one additional principle

Claude's audit of V40–V44 identified a useful distinction:

- the predictor used for **behavior/readout** should be strong and generalizing;
- the signal used to decide what deserves durable learning should not automatically be the same strong predictor, because good one-shot generalization can make genuinely new information look already known.

The useful part for V47 is more general than slots:

> A deliberately **myopic short-context predictor** can identify locally predictable surface structure. Negative representation teaching should emphasize conflicts that remain surprising to such a short-context predictor.

V47 incorporates that idea **only as a development-stage filter on negative teaching**.

It is **not** used for slot admission, because V47 has no slots.

The necessity of this short-context filter is tested separately; V47's core signed-representation hypothesis is allowed to survive even if the filter proves unnecessary.

---

# 2. Primary hypothesis

PAIR learns only:

\[
\text{corroborated positive relation}
\Rightarrow
\text{suppress their difference}.
\]

V47 adds a second learner-visible signal:

\[
\text{high-similarity near collision + incompatible observed outcome}
\Rightarrow
\text{do not suppress the distinguishing direction}.
\]

The core hypothesis is:

> A representation that learns both **positive invariance** and **negative collision protection** will preserve PAIR's useful same-identity transfer while lowering different-identity/different-outcome write interference enough to produce a material persistent-memory gain in the unchanged fixed-size global slow matrix.

V47 does **not** learn a new slow-memory architecture.

It tests whether V45/V46 failed because the representation learned only one side of the equivalence problem.

---

# 3. Fixed substrate inherited from V45/V46

Unless explicitly changed below, V47 reuses V45's global distributed-memory substrate.

## 3.1 Persistent scientific loads

Primary qualification load:

\[
P=256.
\]

Secondary scaling load:

\[
P=128.
\]

Load 256 determines qualification.

Load 128 is explanatory and enters the scaling gate but cannot rescue a load-256 failure.

## 3.2 Persistent A→B→C→D workload

Reuse unchanged:

- four sequential cohorts A→B→C→D;
- 8 acquisition presentations per identity;
- no replay;
- held-out rendering probes;
- intervening learning;
- inherited one-off noise and revision structure;
- all intended identities remain in denominators;
- evaluator-only identity/role labels never enter the learner.

## 3.3 FAST, BASE, slow matrix, writes, reader

Reuse V45/V46 unchanged:

- same BASE predictor;
- same FAST predictor;
- same fixed sparse slow projection;
- same 1,024 slow units;
- same `1024 × 256` learned slow output matrix;
- same 51 active slow units per durable write;
- same slow learning rate;
- same BASE-miss persistent write schedule;
- same 50:50 FAST/SLOW deployed reader.

All V47 arms receive **exactly the same persistent write events**.

No arm changes its write count or timing.

This is essential:

\[
\boxed{\text{any persistent difference must arise from representation geometry, not selective writing}}
\]

---

# 4. Scientific arms

V47 runs six distributed arms inside each seed/load trajectory.

All share the same stream, FAST, BASE, durable-write events, sparse projection size, slow-weight count, slow learning rate, and read rule.

## 4.1 `RAW`

No learned representation suppression.

Reference for the original 96-dimensional context code.

## 4.2 `PAIR20`

The validated Candidate-C/V45 PAIR representation:

- learn a rank-20 positive nuisance subspace \(U_+\);
- remove all 20 learned positive-pair directions.

This is the direct incumbent representation that V47 must improve.

## 4.3 `PAIR-RANDOM-PROTECT8`

Same learned positive basis \(U_+\), but randomly protect eight orthonormal combinations inside the 20-dimensional positive subspace.

Therefore only twelve directions are actually suppressed.

Purpose:

> control for simply removing fewer PAIR directions.

The random protected subspace is fixed by seed and independent of outcomes.

## 4.4 `SIGNED-NOGATE`

Learn which parts of \(U_+\) are involved in high-similarity conflicting outcomes, but accept all qualifying outcome conflicts regardless of short-context predictability.

This tests positive + negative representation learning without Claude's short-context filter.

## 4.5 `SIGNED-SHUFFLED-NEG`

Use the same number of gated negative updates, matched by coarse surprise and time stratum, but break the correspondence between the current near-collision and the negative source context.

This is nondeployable and can never qualify.

Purpose:

> determine whether correct negative relation content matters, rather than generic repulsive regularization.

## 4.6 `SIGNED-GATED` — primary arm

Positive PAIR invariance plus negative collision protection, where negative teaching is accepted only when both conflicting observations are surprising to a deliberately myopic one-byte predictor.

This is the V47 candidate.

---

# 5. Two-stage development protocol

The final representation remains frozen during A→B→C→D.

V47 therefore separates representation learning from persistent-memory evaluation.

The two development stages use **disjoint identities**.

No development identity appears in the persistent workload.

## 5.1 Stage P — positive invariance learning

Purpose:

> reproduce the already validated Candidate-C PAIR mechanism without changing it.

Use:

- 192 development identities;
- wrappers 0–3;
- two development repeats;
- Candidate-C causal top-K proposal/corroboration;
- positive verifier cap = 1,024;
- proposal threshold = 0.58;
- proposal top-K = 12;
- positive Oja/Sanger rank = 20;
- positive Oja rate = 0.06.

The learner receives only context bytes, its bounded verifier state, and the observed next byte.

It does not receive:

- identity ID;
- wrapper ID;
- true pair label;
- evaluator target-role labels.

After Stage P:

1. orthonormalize and freeze the rank-20 basis

\[
U_+\in\mathbb R^{20\times96};
\]

2. destroy/reset the Stage-P verifier.

`PAIR20` uses this \(U_+\) directly.

## 5.2 Stage N — negative collision learning

Use a second, disjoint set of **128 development identities**.

Each identity again appears under wrappers 0–3 with two development repeats.

Unlike Stage P, Stage N is consumed as the complete 14-byte record stream so that the myopic predictor can learn ordinary local transitions as well as long-context outcomes.

Development state remains bounded.

No Stage-N identity appears later in A→D.

---

# 6. The myopic short-context predictor

V47 uses a deliberately weak predictor only during Stage N.

It is not part of deployed prediction.

It is destroyed before A→D.

## 6.1 Predictor definition

Use a one-byte Markov table:

\[
P_{\rm short}(y_t\mid b_{t-1}).
\]

State:

- `count[256,256]` unsigned integer counts;
- `total[256]`;
- Laplace/Dirichlet pseudocount \(\alpha=1\).

Before observing \(y_t\):

\[
p_{\rm short}(y_t)
=
\frac{\operatorname{count}[b_{t-1},y_t]+1}
{\operatorname{total}[b_{t-1}]+256}.
\]

Short-context surprise:

\[
S_t=-\log_2 p_{\rm short}(y_t).
\]

Only after all current-event development decisions are recorded is the Markov table updated.

Thus the surprise signal is causal.

## 6.2 Fixed surprise threshold

Negative teaching is considered "long-context relevant" when:

\[
\boxed{S_t\ge4.0\text{ bits}}
\]

for both the current event and its proposed conflicting source event.

Four bits means the one-byte predictor assigned probability at most \(1/16\).

This threshold is fixed before V47 and is not tuned.

`SIGNED-NOGATE` ignores this threshold.

The predictor never sees evaluator target-position labels.

---

# 7. Negative near-collision verifier

Stage N asks a bounded causal question:

> Among earlier contexts that look unusually similar **after PAIR cleaning**, is there a context whose already-observed consequence conflicts with the current consequence?

## 7.1 Representation used for proposals

For a rolling 12-byte context \(c_t\):

1. construct the same normalized RAW 96-dimensional code;
2. apply the frozen positive PAIR transform

\[
z_+(c_t)=
\operatorname{norm}
\left(
x_t-U_+^\top U_+x_t
\right).
\]

The proposal verifier operates on \(z_+\).

## 7.2 Bounded verifier

Fixed parameters:

- trace cap = 1,024;
- top-K proposals = 12;
- cosine similarity threshold

\[
\boxed{\tau_-=0.70}.
\]

The 0.70 threshold is fixed from the pre-V47 observation that PAIR places same-identity alternate renderings near ~0.9 while difficult different-identity comparisons are typically lower; it is not tuned on V47 outcomes.

Each verifier trace stores only bounded learner-available state needed to reproduce the proposal:

- 12-byte context;
- observed outcome;
- observation time/LRU metadata;
- short-context surprise at the source event.

No identity or wrapper label is stored.

Eviction is LRU and bounded.

## 7.3 Causal sequence

For every Stage-N byte event after a complete 12-byte context exists:

1. compute the current PAIR-cleaned representation;
2. retrieve the fixed top-K candidate set **before** observing \(y_t\);
3. compute the short-context predictive distribution before \(y_t\);
4. observe \(y_t\);
5. among the pre-outcome candidates with similarity ≥0.70 and stored outcome different from \(y_t\), choose the highest-similarity conflicting candidate;
6. classify the event for `SIGNED-NOGATE`;
7. additionally classify it for `SIGNED-GATED` only if both current and source short surprises are ≥4 bits;
8. only then update the short-context Markov table and the bounded verifier.

There is no future lookahead.

---

# 8. How negative teaching modifies PAIR

V47 deliberately does **not** amplify arbitrary new dimensions.

It only protects dimensions that PAIR itself would otherwise erase.

This makes the experiment a clean repair of PAIR rather than a second independent representation learner.

## 8.1 Negative evidence in PAIR coordinates

For a qualifying negative pair with RAW normalized difference

\[
d_-=x_{\rm current}-x_{\rm source},
\]

normalize:

\[
\hat d_-=\frac{d_-}{\|d_-\|+\epsilon}.
\]

Project into the positive nuisance basis:

\[
a=U_+\hat d_-\in\mathbb R^{20}.
\]

If \(a\) is small, the negative conflict lies mostly outside the PAIR nuisance subspace and PAIR would not erase much of it.

If \(a\) is large, PAIR is attempting to remove a direction that participates in an observed conflict.

Accumulate the bounded 20×20 negative covariance:

\[
K_- \leftarrow K_- + a a^\top.
\]

Separate matrices are accumulated for:

- all conflicts (`NOGATE`);
- gated conflicts (`GATED`);
- shuffled gated conflicts (`SHUFFLED`).

All are development-only bounded state.

## 8.2 Protected negative subspace

At the end of Stage N, eigendecompose the normalized \(K_-\).

Take the top:

\[
\boxed{r_-=8}
\]

orthonormal directions:

\[
B_-\in\mathbb R^{8\times20}.
\]

These are combinations of PAIR nuisance directions that most strongly participate in observed near-collision conflicts.

No outcome information is used after Stage N.

## 8.3 Final signed transform

Define the projector onto protected negative combinations:

\[
Q_-=B_-^\top B_-.
\]

The signed representation removes only the part of the positive nuisance space orthogonal to \(Q_-\):

\[
\boxed{
z_{\rm signed}(x)
=
\operatorname{norm}
\left[
x
-
U_+^\top (I_{20}-Q_-) U_+x
\right].
}
\]

Interpretation:

- PAIR says: "these 20 combinations repeatedly varied across positively corroborated experiences, so ignore them";
- negative learning says: "these 8 combinations also repeatedly distinguished dangerous conflicts, so do **not** erase them."

Thus V47 learns **what to ignore and what not to ignore** without adding a second downstream memory system.

Effective suppressed rank is 12.

`PAIR-RANDOM-PROTECT8` uses the identical equation but replaces \(B_-\) by a seed-fixed random rank-8 basis in \(\mathbb R^{20}\).

---

# 9. Shuffled-negative specificity control

`SIGNED-SHUFFLED-NEG` must preserve the main nuisance factors of negative development while destroying content correspondence.

For every gated negative update, record a development-only event tuple containing:

- current context;
- chosen source context;
- current outcome;
- source outcome;
- current surprise;
- source surprise;
- event time;
- similarity.

After Stage N is complete, create the shuffled control event-by-event. For each genuine gated negative event, choose a **wrong** source context subject to:

- current outcome still differs from the shuffled source outcome;
- coarse current-surprise bin is matched whenever possible: `[4,5)`, `[5,6)`, `[6,7)`, `>=7`;
- coarse source-surprise bin is matched whenever possible using the same bins;
- among eligible wrong sources, minimize the absolute difference between the genuine pair's PAIR cosine and the shuffled pair's PAIR cosine;
- add a small penalty for temporal-block distance and repeated reuse of one source, but do **not** sacrifice similarity matching merely to preserve an exact 256-event block.

Search relaxes in this order:

1. both current- and source-surprise bins matched;
2. source-surprise bin matched;
3. global outcome-conflicting fallback.

Source reuse is permitted if needed to make the shuffled pair genuinely similarity-matched. This is intentional: the control prioritizes matching the **difficulty of the near collision** over exact preservation of the source-event multiset.

The shuffled control must preserve exactly:

- number of negative updates;
- current-event times;
- current contexts.

It must preserve as closely as possible:

- source-surprise distribution;
- PAIR-similarity distribution;
- temporal locality.

The content correspondence between a current near-collision and its actual conflicting source is destroyed.

Report for every seed:

- update count;
- surprise-matching fallback counts;
- source-surprise mean absolute mismatch;
- genuine versus shuffled mean PAIR similarity;
- fraction of unique shuffled source events used;
- mean temporal-block distance.

This refinement is locked before any V47 scientific result is observed.

This arm is nondeployable and cannot qualify.

---

# 10. Representation geometry endpoints

Geometry is measured on fresh persistent identities that were absent from both development stages.

For each representation arm, report:

1. same identity / held-out rendering cosine;
2. different identity / same outcome cosine;
3. different identity / different outcome cosine;
4. positive invariance margin:

\[
M_+=
\operatorname{sim}(\text{same identity})
-
\operatorname{sim}(\text{different same-outcome});
\]

5. conflict-separation margin:

\[
M_-=
\operatorname{sim}(\text{same identity})
-
\operatorname{sim}(\text{different different-outcome}).
\]

The primary representation contrast is:

\[
\text{SIGNED-GATED} - \text{PAIR20}.
\]

The desired pattern is:

- same-identity similarity remains high;
- different-outcome similarity falls materially;
- \(M_-\) increases.

---

# 11. Direct write-interference assay

Reuse V45/V46's fixed bounded deterministic probe-bank method.

For a sampled slow write:

\[
W\rightarrow W+\Delta W,
\]

measure probe loss before and after the write.

Categories:

- `same_identity_alt`;
- `same_outcome_diff_identity`;
- `different_outcome_diff_identity`;
- overall.

Define:

\[
T_{\rm same}
=
\operatorname{mean}\Delta\ell
\quad
\text{on same-identity alternate-rendering probes},
\]

and

\[
I_{\rm diff}
=
\operatorname{mean}\Delta\ell
\quad
\text{on different-identity/different-outcome probes}.
\]

More negative \(T_{\rm same}\) means more useful transfer.

Lower \(I_{\rm diff}\) means less harmful cross-talk.

The central mechanistic prediction is:

\[
\boxed{
I_{\rm diff}^{\rm SIGNED}
<
I_{\rm diff}^{\rm PAIR}
}
\]

without materially worsening \(T_{\rm same}\).

---

# 12. Persistent behavioral endpoints

For every arm and load report:

- whole-stream bpb;
- target-byte bpb;
- D trained acquisition accuracy;
- delayed held-out accuracy after D;
- delayed held-out bpb after D;
- revision accuracy and bpb at exposures 1,2,4,8;
- prequential acquisition loss;
- representation geometry;
- direct interference metrics;
- persistent-state bytes;
- development-state peak bytes;
- raw development presentations;
- persistent presentations;
- wall-clock runtime and peak RSS.

All intended identities remain in denominators.

Conditional retention among successfully learned identities may be reported descriptively but cannot replace unconditional endpoints.

---

# 13. Scientific seeds

Fresh V47 scientific seeds:

```text
47701 47702 47703 47704
47705 47706 47707 47708
47709 47710 47711 47712
47713 47714 47715 47716
```

No V45/V46 scientific seed is reused.

Engineering/smoke/resume tests must use disjoint non-scientific seeds, e.g. `47971+`.

Seed is the inferential unit.

Repeated events, identities, probes, writes, or byte positions are not treated as independent replicates for scientific confidence intervals.

---

# 14. Locked constants

```text
persistent loads          = 256 primary, 128 secondary
persistent repeats        = 8
positive dev identities   = 192
negative dev identities   = 128
positive wrappers         = 0,1,2,3
negative wrappers         = 0,1,2,3
positive dev repeats      = 2
negative dev repeats      = 2

positive verifier cap     = 1024
positive proposal top-K   = 12
positive proposal tau     = 0.58
positive rank             = 20
positive Oja rate         = 0.06

negative verifier cap     = 1024
negative proposal top-K   = 12
negative proposal tau     = 0.70
short-context order       = 1 byte
short Dirichlet alpha     = 1
short surprise threshold  = 4.0 bits
protected negative rank   = 8

slow units                = 1024
slow active units/write   = 51
slow outputs              = 256
slow learning rate        = inherited V45 value
deployed FAST/SLOW mix    = 0.5 / 0.5

persistent slow-state cap = 3.75 MiB
```

No scientific sweep over these constants is allowed in V47.

---

# 15. Ordered gates

Gates are evaluated in this order.

A failure voids downstream qualification but all descriptive outputs are still reported.

## G0 — implementation validity

Require:

- all 16 primary load-256 seeds complete;
- all scientific arms share identical persistent streams;
- all arms share the identical persistent write-event schedule;
- no slots/pathways/growing memory objects exist;
- slow learned-weight count is identical across arms;
- write count is identical across arms;
- read-only probes do not mutate state;
- development identities are disjoint across Stage P, Stage N, and persistent A→D;
- short-context predictor is destroyed before A→D;
- final representation is frozen throughout A→D;
- state hashes and resume identity checks pass.

Failure → `IMPLEMENTATION_INVALID`.

## G1 — positive PAIR development validity

Across 16 primary seeds require:

- mean positive pair updates ≥300;
- mean evaluator-only false-positive pair rate ≤5%;
- rank \(U_+=20\) in every seed;
- orthogonality error within implementation tolerance;
- PAIR20 geometry margin over RAW ≥0.05 with paired 95% lower bound >0.

Failure → `POSITIVE_DEVELOPMENT_INVALID`.

This gate confirms the already-established PAIR effect still exists.

## G2 — negative development validity

For `SIGNED-GATED` require:

- mean gated negative updates ≥100 per seed;
- no seed has fewer than 50 gated negative updates;
- protected rank \(r_-=8\) in every seed;
- current and source surprise both satisfy the 4-bit rule by construction;
- negative covariance is nondegenerate in every seed;
- shuffled control has exactly the same gated-update count per seed.

Report, but do not gate on:

- target-position enrichment of gated versus ungated negative events;
- same-identity-conflict versus different-identity-conflict composition;
- similarity distribution;
- short-predictor bpb by byte position.

Failure → `NEGATIVE_DEVELOPMENT_INVALID`.

## G3 — acquisition validity

At load 256:

- `PAIR20` mean D trained accuracy ≥0.90;
- `SIGNED-GATED` mean D trained accuracy ≥0.90;
- neither arm may have a seed below 0.80 D trained accuracy.

Failure → `INVALID_ACQUISITION`.

## G4 — signed representation geometry

At load 256, relative to `PAIR20`, `SIGNED-GATED` must:

### Preserve positive invariance

Same-identity held-out-rendering cosine may decrease by no more than 0.05 in paired mean, and the paired 95% lower bound on

\[
\operatorname{sim}_{\rm SIGNED}
-
\operatorname{sim}_{\rm PAIR}
\]

must be ≥ −0.08.

### Improve conflict separation

Different-identity/different-outcome similarity must decrease by at least:

\[
\boxed{0.05}
\]

in paired mean with paired 95% upper bound <0.

Equivalently, conflict-separation margin \(M_-\) must improve by at least 0.05 with paired 95% lower bound >0.

Failure → `NO_SIGNED_GEOMETRY_GAIN`.

## G5 — selective interference mechanism

At load 256 compare `SIGNED-GATED` with `PAIR20`.

### Harmful cross-talk

Let

\[
\Delta I=
I_{\rm diff}^{\rm SIGNED}
-
I_{\rm diff}^{\rm PAIR}.
\]

Require:

\[
\boxed{
\operatorname{mean}(\Delta I)\le-0.00075\text{ bpb/write}
}
\]

and paired 95% upper bound <0.

This retains the exact materiality target that V46 failed in the opposite direction.

### Useful-sharing noninferiority

Let

\[
\Delta T=
T_{\rm same}^{\rm SIGNED}
-
T_{\rm same}^{\rm PAIR}.
\]

More negative is better.

Require paired mean ≤ +0.05 bpb/write and paired 95% upper bound ≤ +0.08.

Failure → `NO_SELECTIVE_INTERFERENCE_GAIN`.

## G6 — primary persistent gain

At load 256:

\[
\Delta L=
L_{\rm heldout}^{\rm SIGNED}
-
L_{\rm heldout}^{\rm PAIR}.
\]

Require:

\[
\boxed{
\operatorname{mean}(\Delta L)\le-0.10\text{ bit}
}
\]

with paired 95% upper bound <0.

This is the same persistent materiality threshold used in V45/V46.

Failure after G5 passes → `INTERFERENCE_REDUCED_NO_PERSISTENT_GAIN`.

## G7 — negative-learning specificity

`SIGNED-GATED` must beat both controls on delayed held-out bpb at load 256:

- `PAIR-RANDOM-PROTECT8`;
- `SIGNED-SHUFFLED-NEG`.

For each:

\[
\boxed{
L_{\rm control}-L_{\rm SIGNED}\ge0.05\text{ bit}
}
\]

with paired 95% lower bound >0.

If random protection matches:

`RANK_REDUCTION_SUFFICIENT`.

If shuffled negative evidence matches:

`NEGATIVE_CONTENT_SPECIFICITY_FAILED`.

These outcomes do not qualify signed collision learning.

## G8 — short-context filter assessment

This gate is **secondary** and does not determine whether signed negative learning itself qualifies.

Compare:

\[
\text{SIGNED-GATED}
\quad\text{vs}\quad
\text{SIGNED-NOGATE}.
\]

Define the short-filter mechanism as supported if:

\[
L_{\rm NOGATE}-L_{\rm GATED}\ge0.03\text{ bit}
\]

with paired 95% lower bound >0,

and `SIGNED-GATED` has lower harmful \(I_{\rm diff}\).

If this gate fails while G4–G7 pass:

\[
\boxed{\texttt{SIGNED\_LEARNING\_SUPPORTED\_SHORT\_GATE\_UNNECESSARY}}
\]

The simpler ungated signed learner should then be preferred.

This prevents Claude's additional short-context idea from becoming a mandatory explanation unless the data support it.

## G9 — prediction and revision safety

At load 256, `SIGNED-GATED` versus `PAIR20`:

- whole-stream bpb paired mean difference ≤ +0.02 bit;
- whole-stream bpb paired 95% upper bound ≤ +0.04;
- revision-8 accuracy may not decrease by more than 5 percentage points in mean;
- revision-8 bpb paired 95% upper bound ≤ +0.10 bit.

Failure → `PERSISTENCE_GAIN_WITH_SAFETY_COST`.

## G10 — scaling

Define positive persistence gain:

\[
G_L=
L_{\rm PAIR,L}-L_{\rm SIGNED,L}.
\]

Require:

\[
\boxed{
G_{256}\ge0.75\max(G_{128},0.05)
}
\]

together with G6.

Failure → `GAIN_COLLAPSES_WITH_LOAD`.

## G11 — bounded state

At both loads:

- persistent slow state ≤ 3.75 MiB;
- persistent learned-weight count is load-independent;
- final representation state is fixed-size;
- no identity/event-indexed persistent object state grows without a hard cap;
- development-only verifier and short-context states are destroyed before A→D;
- persistent state does not materially grow from load 128 to 256.

Development peak state and sample budget are reported separately and are not hidden.

Failure → `STATE_BUDGET_FAIL`.

---

# 16. Verdict logic

Evaluate in order.

1. G0 fail → `IMPLEMENTATION_INVALID`
2. G1 fail → `POSITIVE_DEVELOPMENT_INVALID`
3. G2 fail → `NEGATIVE_DEVELOPMENT_INVALID`
4. G3 fail → `INVALID_ACQUISITION`
5. G4 fail → `NO_SIGNED_GEOMETRY_GAIN`
6. G5 fail → `NO_SELECTIVE_INTERFERENCE_GAIN`
7. G6 fail → `INTERFERENCE_REDUCED_NO_PERSISTENT_GAIN`
8. G7 fail → classify as `RANK_REDUCTION_SUFFICIENT` or `NEGATIVE_CONTENT_SPECIFICITY_FAILED`
9. G9 fail → `PERSISTENCE_GAIN_WITH_SAFETY_COST`
10. G10 fail → `GAIN_COLLAPSES_WITH_LOAD`
11. G11 fail → `STATE_BUDGET_FAIL`
12. G4–G7, G9–G11 pass:
   - if G8 passes → `SIGNED_REPRESENTATION_QUALIFIED`
   - if G8 fails → `SIGNED_LEARNING_SUPPORTED_SHORT_GATE_UNNECESSARY`

No V47 result may be described as solving general persistent learning.

---

# 17. Required outputs

V47 must write:

```text
SUMMARY.md
GATES.json
MANIFEST.json
PROTOCOL.md

DEVELOPMENT_POSITIVE.csv
DEVELOPMENT_NEGATIVE.csv
SHORT_CONTEXT_AUDIT.csv
NEGATIVE_SHUFFLE_AUDIT.csv

REPRESENTATION_GEOMETRY.csv
INTERFERENCE_SUMMARY.csv
PERSISTENCE_CONTRASTS.csv
REVISION.csv
TERMINAL.csv

STATE_COSTS.csv
SAMPLE_BUDGET.csv
RUNTIME.csv

jobs/
development/
checkpoints/
```

`MANIFEST.json` must include:

- source SHA256;
- design SHA256;
- scientific seeds;
- all fixed constants;
- platform/Python/NumPy versions;
- thread environment;
- arm definitions;
- result completeness.

---

# 18. Mac M3 implementation plan

## 18.1 One seed/load job contains all six scientific arms

For exact pairing and efficiency, one worker handles one seed/load and shares:

- persistent byte stream;
- BASE state;
- FAST state;
- persistent write schedule;
- development representation artifacts;
- immutable sparse projections.

Only the arm-specific final representation and slow learned matrix diverge.

This avoids multiplying expensive front-end computation by six.

## 18.2 Multiprocessing

Default:

- 4 macOS `spawn` workers;
- one seed/load per worker;
- BLAS/Accelerate threads fixed to 1 per worker.

Environment before importing NumPy:

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## 18.3 Caffeinate

Scientific launcher:

```bash
caffeinate -dimsu python3 elm_v47_signed_representation.py \
  --stage all \
  --workers 4 \
  --out-dir v47_results
```

Primary-first launcher:

```bash
caffeinate -dimsu python3 elm_v47_signed_representation.py \
  --stage primary \
  --workers 4 \
  --out-dir v47_results
```

## 18.4 Resume

Requirements:

- atomic per-seed development artifacts;
- atomic per-seed/load scientific results;
- one rolling atomic checkpoint for unfinished persistent jobs;
- identity hash includes source, design, seed, load, stream, development artifacts, and fixed constants;
- completed matching jobs are reused;
- mismatched caches are rejected;
- SIGINT/SIGTERM saves current scientific job state;
- rerunning the same command resumes.

Do not compress the rolling scientific checkpoint if compression becomes a material Mac bottleneck. Final upload bundles may be compressed.

## 18.5 Engineering firewall

Before scientific execution:

1. `python3 -m py_compile`
2. `--self-test`
3. 2 engineering seeds in smoke mode
4. forced interruption/resume equivalence
5. 2-worker and 4-worker determinism
6. one full-size **engineering** load-256 trajectory
7. verify all required CSV/JSON outputs
8. verify arm-order invariance
9. verify development/persistent identity disjointness
10. verify exact common persistent write schedules
11. verify short-context predictor and development verifiers are absent from persistent state
12. verify no scientific seed has been inspected.

Engineering seeds must be disjoint from 47701–47716.

---

# 19. Interpretation table

| Outcome | Interpretation |
|---|---|
| G4–G7 pass | Positive + negative representation learning materially improves the same bounded global slow memory |
| G4 passes, G5 fails | Geometry changed, but harmful write cross-talk did not fall enough; the representation change is not mechanistically sufficient |
| G5 passes, G6 fails | Harmful interference fell, but not enough to produce the required delayed-persistence gain |
| G6 passes, G7 fails | Generic rank reduction/repulsion explains the behavioral gain; negative semantic content is not established |
| G4–G7 pass, G8 fails | Signed negative learning works, but Claude's short-context filter is unnecessary |
| G4–G7 pass, G8 passes | Both negative content and short-context filtering are supported |
| G10 fails | Gain is another low-load effect and does not change scaling |
| G4 fails | Positive/negative representation branch is closed in this form |

---

# 20. Stop conditions and next branch

## If G4 fails

The signed-representation construction itself failed to preserve useful invariance while separating conflicts.

Do **not** tune:

- protected rank;
- 0.70 threshold;
- 4-bit surprise threshold;
- PAIR rank;
- Oja rate.

Close this representation branch in its current form.

The next mainline branch becomes the already identified separate idea:

> **short-context-surprise-gated distributed slow plasticity**

with RAW/PAIR representation held fixed.

That tests Claude's broader principle at the plasticity level rather than by changing representation.

## If G4 passes but G5 fails

The geometry is correct but does not reduce actual write interference enough.

Do not tune representation hyperparameters.

Move to short-context-surprise-gated distributed plasticity.

## If G5 passes but G6 fails

Negative representation learning reduces local interference but not enough to produce persistent gain.

Close the representation branch and test plasticity allocation rather than further metric learning.

## If G4–G7 pass but G8 fails

Keep the simpler `SIGNED-NOGATE` representation for confirmation.

The short-context filter is not part of the supported mechanism.

## If V47 qualifies

Do **not** immediately add slots, pathways, factors, replay, or more representation machinery.

First perform an independent fresh-seed confirmation of the exact fixed V47 mechanism.

Only after confirmation should the project address **continual online representation adaptation without coordinate drift**, because V47 still freezes the learned representation before A→D.

---

# 21. What V47 can and cannot establish

A positive V47 can establish:

- a bounded causal representation can use positive and negative experience relations;
- negative near-collision evidence can protect discriminative directions that positive PAIR would otherwise erase;
- this representation can reduce harmful cross-identity interference in a fixed distributed slow store;
- any incremental value of the one-byte short-context filter can be isolated;
- the gain can remain bounded-state and survive load doubling from 128 to 256.

It cannot establish:

- unbounded lifetime memory;
- continual online movement of the representation coordinates;
- natural-language competence;
- whole-fly biological equivalence;
- that one-byte surprise is the correct universal importance signal;
- optimal rank, thresholds, or learning rates;
- that all forms of explicit episodic memory are unnecessary.

---

# 22. Fixed pre-run prediction

The preregistered qualitative prediction is:

1. `PAIR20` will reproduce V45/V46 positive invariance but retain excess different-outcome cross-talk.
2. `SIGNED-GATED` will preserve most same-identity similarity while lowering different-identity/different-outcome similarity.
3. That selective geometry change will lower harmful per-write cross-talk by at least 0.00075 bpb/write.
4. If this mechanism is behaviorally important, delayed held-out loss at load 256 will improve by at least 0.10 bit relative to `PAIR20`.
5. `PAIR-RANDOM-PROTECT8` and `SIGNED-SHUFFLED-NEG` will not match the primary arm.
6. The one-byte short-context filter is a separately falsifiable refinement: it may or may not beat `SIGNED-NOGATE`.

Failure of items 2–5 closes the signed-representation mechanism in this form.
