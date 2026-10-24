# V46 — Learned Overlapping Slow Pathways
## Can experience learn *where* durable sharing should occur, so a bounded slot-free slow memory preserves useful transfer without V45's cross-identity interference?

**Status:** refined locked design / executable implementation specification  
**Date:** 11 September 2026  
**Target platform:** Apple Silicon Mac M3, macOS, Python 3.11+  
**Scientific results observed before this document:** V45 only, plus the prior BANC connectivity re-audit. No V46 outcome has been observed.

---

# 0. Research constitution — no drift

The project objective remains fixed:

> Build a **bounded-state, non-Transformer, online persistent learner** that learns from sequential experience, retains useful knowledge through intervening learning, generalizes beyond exact presentations, revises obsolete knowledge, and does not require replay or full retraining.

V46 follows the branch explicitly specified in the V45 design for the case in which learned invariance is valid but one global distributed slow matrix still fails at high load.

The one scientific question is:

\[
\boxed{
\text{Can a shared learned representation route experience into several overlapping, experience-specialized slow pathways,}
\\
\text{so useful same-identity sharing is preserved while harmful different-identity/different-outcome cross-talk is reduced?}
}
\]

V46 changes **one architectural operation only**: the allocation of the existing 1,024 slow synaptic units among several simultaneously active learning pathways.

V46 may **not** add, tune, or search over:

- permanent object slots;
- slot eviction or replacement rules;
- TRACE-CORR or any new corroboration verifier;
- factor-coded outputs from Candidate B;
- a new PAIR/invariance algorithm;
- online adaptation of the PAIR subspace during A→B→C→D;
- replay or rehearsal;
- learning-rate sweeps;
- multiple synaptic-time-constant sweeps;
- hand-labelled identity, wrapper, alias, cohort, target-position, or revision information inside the learner;
- hand-designed biological compartments or BANC-derived pathway labels;
- increased slow state as load increases;
- Transformer/backpropagation training;
- whole-fly recurrent simulation.

No slot-based arm is run in V46. Historical slot results remain context only and cannot qualify or disqualify V46.

If V46 fails, the exclusions above are not relaxed mid-run.

---

# 1. Evidence entering V46

## 1.1 V45 established the representation manipulation but not persistent gain

V45's locked verdict was `NO_PERSISTENT_GAIN`.

At the primary load 256:

- PAIR-OJA geometry replicated strongly:

\[
\Delta(\text{PAIR margin}-\text{RAW margin}) = +0.1073
\]

with 95% CI \([+0.1017,+0.1130]\).

- PAIR improved delayed held-out loss relative to RAW under the matched random write schedule:

\[
\text{PAIR-RAND}-\text{RAW-RAND}=-0.05819\text{ bit},
\]

95% CI \([-0.06710,-0.04928]\), but this was below the preregistered 0.10-bit materiality threshold.

- the same representation gain was much larger at load 128:

\[
\approx-0.136\text{ bit},
\]

showing a scaling loss as identity load doubled.

- corroboration added no detectable persistent benefit once PAIR was present, and its bounded verifier itself collapsed in event coverage at load 256. V46 therefore removes corroboration from the candidate mechanism rather than redesigning it.

V45 design SHA256:  
`00eb0a01b183f83604d938ca4d427efcfa372a1d02297772cced062ef508b4c8`

V45 runner SHA256:  
`c8eefb54fe965b2afce3dcf5aafc6603eab4b00b56b70b587e22d94573a287bc`

V45 completed summary SHA256:  
`432886dbdadfe9dbe8ac59e6b7b85463ae48d642a47f82a4c1b817023d402d95`

## 1.2 V45 localized the remaining problem to *where sharing occurs*

At load 256, V45's PAIR representation strengthened desirable transfer:

- same identity, alternate rendering: approximately

\[
-0.3294\text{ bpb/write}
\]

under `PAIR-RAND`;

- different identity, same outcome:

\[
-0.3219\text{ bpb/write}.
\]

But harmful different-identity/different-outcome interference increased to approximately

\[
+0.00383\text{ bpb/write},
\]

versus the RAW matched control's lower harmful cross-talk.

Thus the observed V45 failure was not that PAIR failed to learn useful sharing. It was that **all identities still wrote into one common slow synaptic matrix**.

V46 therefore does not modify PAIR. It tests whether experience can learn *where in the slow substrate* PAIR-mediated sharing should be expressed.

## 1.3 BANC motivates overlapping specialization, not literal copied compartments

The supplied BANC v888 graph contains, at the inclusive recorded-count cutoff:

- 4,552 annotated KCs and 104 MBONs;
- 15,523 recorded KC→MBON connection pairs;
- among connected KCs, median recorded MBON fanout = 4;
- 81.10% of connected KCs reach more than one recorded MBON;
- MBON input sets have generally low pairwise overlap (mean Jaccard ≈ 0.0189, median 0 at cutoff 1);
- KC-family composition across MBON inputs is strongly concentrated relative to a degree/hemisphere-stratified permutation reference.

These measurements support a **shared representational population participating in several differentiated output/learning routes**. They do not identify eight biological pathways, a top-3 rule, or the equations below. Those are computational hypotheses constrained by the graph rather than copied from it.

BANC re-audit SHA256:  
`701634b78a6989df2a09f8667998828c69b660c61035a1e084cb40f242ef8c3e`

Previous negative results remain binding:

- V34 fixed compartments did not solve persistence;
- Candidate B's hand-designed modular topology was matched by degree-preserving rewiring.

Therefore V46 tests **experience-learned pathway responsibility**, not another static partition.

---

# 2. Primary hypothesis

V45 used one global slow store:

\[
\text{PAIR representation}\rightarrow W_{\rm global}\in\mathbb R^{1024\times256}.
\]

V46 retains exactly the same total 1,024 slow units and the same total

\[
1024\times256
\]

learned output weights, but divides the slow units into eight equal pathway blocks. Every experience uses **three pathways simultaneously**.

The primary prediction is:

> A pathway assignment learned from both input similarity and each pathway's own predictive history will keep alternate renderings of the same identity on overlapping synaptic support while gradually separating superficially similar experiences whose consequences conflict.

This predicts a selective change:

\[
\boxed{
\text{harm to different identity / different outcome}\downarrow
}
\]

without eliminating

\[
\boxed{
\text{benefit to same identity / alternate rendering}.
}
\]

The candidate is successful only if this mechanistic change produces a material delayed-persistence gain at load 256 under fixed state and write budget.

---

# 3. Fixed substrate inherited from V45

Unless explicitly changed below, V46 reuses V45 unchanged.

## 3.1 Stream and scientific loads

Primary load:

\[
P=256\text{ total identities}.
\]

Secondary scaling load:

\[
P=128.
\]

Persistent workload:

- four sequential cohorts A→B→C→D;
- 8 acquisition presentations per identity;
- held-out rendering probes;
- intervening learning;
- inherited one-off noise and revision structure;
- no replay;
- all intended identities remain in denominators.

Load 256 determines qualification. Load 128 is explanatory and enters the scaling gate but cannot rescue a load-256 failure.

## 3.2 Representation development

Use the V45 development protocol unchanged:

- 192 development identities;
- wrappers 0–3;
- two development repeats;
- all 256 possible next-byte outcomes available;
- causal top-K proposal/corroboration;
- PAIR rank = 20;
- PAIR Oja/Sanger rate = 0.06;
- proposal threshold = 0.58;
- proposal top-K = 12;
- development trace cap = 1,024.

Development identities are disjoint from A–D identities.

After development:

- retain only the frozen rank-20 PAIR subspace;
- destroy/reset the development verifier;
- initialize FAST, BASE, all slow weights, pathway state, and stream history from scratch.

The PAIR subspace remains frozen throughout A→D. V46 does not test continual coordinate drift.

## 3.3 FAST / BASE / deployed reader

Reuse V45's BASE and FAST trajectories exactly.

All candidate slow arms receive the same BASE-miss durable-write events. No selective corroboration schedule is used.

For every arm:

\[
p_{\rm deployed}=\tfrac12p_{\rm FAST}+\tfrac12p_{\rm SLOW}.
\]

The 50:50 composition is fixed and not tuned.

## 3.4 Slow sensory projection and write rule

Inherited constants:

- slow units = 1,024;
- slow contacts/unit = 12;
- active slow units/write = 51;
- output dimension = 256;
- slow learning rate = 0.6.

The fixed sparse projection from the frozen PAIR-cleaned 96-dimensional representation to 1,024 slow unit drives is unchanged from V45.

For any selected 51-unit active set \((i,v)\), the deployed slow score remains

\[
s=v^\top W_i.
\]

After observing outcome \(y\), every arm uses the same inherited V45 delta update:

\[
e=-s,\qquad e_y\leftarrow e_y+1,
\]

\[
W_i\leftarrow W_i+0.6\,\frac{v}{v^\top v}e^\top.
\]

**No arm changes the synaptic write equation.** V46 changes only which slow units form \((i,v)\).

---

# 4. Pathway construction

## 4.1 Fixed resource partition

At initialization, deterministically permute the 1,024 slow units from the seed and divide them into:

\[
M=8
\]

pathway blocks of

\[
128\text{ slow units each}.
\]

This partition changes neither the total number of units nor the total learned weights.

A routed experience selects

\[
R=3
\]

of the eight pathways.

Within each selected pathway, choose the 17 units with largest absolute fixed slow drive. Thus:

\[
3\times17=51
\]

active units, exactly matching V45's write sparsity.

Concatenate the 51 selected drive values and normalize them jointly to unit norm before scoring/writing.

The global control instead chooses the 51 largest absolute drives over all 1,024 units exactly as V45 did.

Therefore every arm has:

- the same number of learned weights;
- the same total slow units;
- the same number of active units/write;
- the same slow learning rate;
- the same BASE-miss write events;
- the same FAST and BASE trajectories.

Any performance difference cannot be credited to a larger state, fewer writes, or fewer synaptic rows touched per event.

## 4.2 Shared routing state allocation

Every arm, including controls, is charged storage for:

- eight 96-dimensional pathway prototype vectors;
- eight usage statistics;
- identical counters and route metadata allowance.

Unused/dummy routing state remains allocated in controls so state accounting is exact.

Initial prototype vectors are deterministic normalized Gaussian vectors from the scientific seed. `FROZEN-ROUTE`, `INPUT-LEARNED`, `PRED-RESP`, and `SHUFFLED-FEEDBACK` begin from **identical prototypes within a seed**.

---

# 5. Scientific arms

Exactly six distributed arms are run. There is no slot arm.

| arm | pathway choice | prototype learning | purpose |
|---|---|---|---|
| `GLOBAL-PAIR` | global top-51 | none | V45 single-matrix reference |
| `HASH-ROUTE` | deterministic context hash → 3 paths | none | stable partitioning without input geometry |
| `FROZEN-ROUTE` | fixed random prototypes → top-3 | none | fixed input-similarity routing |
| `INPUT-LEARNED` | learned prototypes → top-3 | input-only responsibility | self-organized clustering without predictive teaching |
| **`PRED-RESP`** | learned prototypes → top-3 | input + pathway predictive success | primary mechanism |
| `SHUFFLED-FEEDBACK` | learned prototypes → top-3 | same responsibility values assigned to wrong selected paths | mechanism-specificity veto |

No arm receives identity labels or evaluator pair labels.

---

# 6. Causal pathway routing

Let \(x_t\in\mathbb R^{96}\) be the normalized PAIR-cleaned representation of the current 12-byte context.

For pathway \(p\), let \(c_p\) be its normalized 96-dimensional prototype.

Every route decision is made **before** observing the current byte.

## 6.1 Usage homeostasis

`FROZEN-ROUTE`, `INPUT-LEARNED`, `PRED-RESP`, and `SHUFFLED-FEEDBACK` maintain a bounded usage EMA \(u_p\), initialized at

\[
R/M=3/8.
\]

After every routing decision:

\[
u_p\leftarrow(1-\rho)u_p+\rho\,\mathbf 1[p\text{ selected}],
\]

with fixed

\[
\rho=1/512.
\]

The routing logit is

\[
a_p=\langle x_t,c_p\rangle-\lambda(u_p-R/M),
\]

with fixed

\[
\lambda=0.10.
\]

Select the top three pathways by \(a_p\).

Within the selected set \(S_t\), define the pre-outcome routing prior

\[
g_p=\frac{\exp(a_p/T)}{\sum_{j\in S_t}\exp(a_j/T)},
\]

with fixed

\[
T=0.20.
\]

These constants are locked before scientific execution and are not tuned on V46 outcomes.

## 6.2 `HASH-ROUTE`

Hash the exact 12-byte context together with the scientific seed. Use the digest to choose three distinct pathways uniformly without replacement.

The same exact context always selects the same three pathways.

No learned routing state affects this arm.

## 6.3 `FROZEN-ROUTE`

Use the common initial random prototypes and the homeostatic routing equation above, but never update the prototypes.

This arm asks whether fixed random input-space routing plus overlapping pathway use is already sufficient.

## 6.4 `INPUT-LEARNED`

After the observed byte arrives, update selected prototypes using only the pre-outcome routing prior \(g_p\):

\[
q_p^{\rm input}=g_p.
\]

Prototype update:

\[
c_p\leftarrow\operatorname{normalize}
\left[(1-\eta_cq_p)c_p+\eta_cq_px_t\right],
\]

with fixed

\[
\eta_c=0.02.
\]

This is an online balanced input-clustering control. It does not use predictive success.

## 6.5 `PRED-RESP` — primary mechanism

Before the weight update, each selected pathway makes its own local prediction from its 17 selected slow units.

For pathway \(p\), normalize its 17 local drive values within pathway and compute:

\[
s_p=v_p^\top W_{p,i}.
\]

After the byte \(y_t\) is revealed, calculate the pathway's pre-update predictive probability

\[
r_p=p_p(y_t)=\operatorname{softmax}(s_p)_{y_t}.
\]

Define predictive responsibility

\[
q_p^{\rm pred}
=
\frac{g_pr_p^{\beta}}{\sum_{j\in S_t}g_jr_j^{\beta}},
\]

with fixed

\[
\beta=1.
\]

Then update prototypes with the same equation as above using \(q_p^{\rm pred}\).

The synaptic weight update itself remains the ordinary V45 update on the full 51-unit active set. `q_p` **does not scale the immediate slow weight update**. It changes only future routing.

This restriction is deliberate: V46 tests learned pathway responsibility, not another write-magnitude rule.

## 6.6 `SHUFFLED-FEEDBACK`

Compute \(q_p^{\rm pred}\) exactly as `PRED-RESP` does.

Before prototype updates, deterministically permute the three responsibility values among the three selected pathway IDs using a seed/event-index hash.

Thus this arm preserves exactly, event by event:

- selected pathway count;
- responsibility-value multiset;
- total prototype-update mass;
- synaptic write event;
- number of slow units updated;
- slow learning rate.

Only the correspondence between a pathway's own predictive success and its prototype update is destroyed.

If this control matches `PRED-RESP`, the claimed prediction-shaped specialization mechanism is retired.

---

# 7. Shared read and write ordering

For every byte event:

1. build the PAIR-cleaned representation from past bytes only;
2. compute BASE and FAST predictions;
3. compute every V46 arm's slow route and pre-outcome slow prediction;
4. form deployed 50:50 FAST/SLOW probabilities and score them;
5. reveal the observed byte;
6. if the shared BASE was wrong, apply one slow synaptic write in **every arm** using that arm's already-selected 51-unit active set;
7. update routing usage statistics;
8. update prototypes according to the arm rule using only information available at this point;
9. continue the stream.

No arm can change BASE or FAST history.

No arm can decide whether a durable slow write occurs. The BASE-miss write schedule is common to all arms.

---

# 8. Direct pathway and interference diagnostics

V46 must establish *how* any terminal gain arose.

## 8.1 Pathway usage

For each arm and seed/load record:

- pathway selection counts \(n_p\);
- per-event pathway selection rates \(s_p=n_p/N_{\rm route}\), whose sum is \(R=3\);
- normalized pathway occupancy \(\pi_p=n_p/\sum_jn_j\), whose sum is one;
- effective pathway count

\[
N_{\rm eff}=\exp\left(-\sum_p\pi_p\log \pi_p\right);
\]

- maximum and minimum **per-event selection rates** \(s_p\);
- prototype cosine matrix over time;
- prototype movement norm;
- responsibility entropy for learned arms.

## 8.2 Path-set overlap

Evaluator-only identity/outcome labels classify read-only route comparisons into:

- same identity / alternate rendering;
- different identity / same outcome;
- different identity / different outcome.

For each pair, report Jaccard overlap between the selected three-path sets.

This is diagnostic only; evaluator labels never affect learner routing.

A useful pathway mechanism should increase the gap

\[
J_{\rm same\ identity}-J_{\rm different\ outcome}.
\]

No numerical qualification threshold is attached to this descriptive geometry endpoint.

## 8.3 Per-write interference audit

Reuse the V45 interference audit procedure unchanged wherever possible.

For a deterministic bounded sample of slow-write events, measure probe loss immediately before and immediately after the write for:

- same identity / alternate rendering;
- different identity / same outcome;
- different identity / different outcome;
- aggregate probe bank.

The key harmful endpoint is:

\[
I_{\rm diff}=\text{mean }\Delta\text{bpb/write on different-identity/different-outcome probes}.
\]

The key useful-sharing endpoint is:

\[
T_{\rm same}=\text{mean }\Delta\text{bpb/write on same-identity/alternate-rendering probes}.
\]

Lower/more negative is better for both metrics, but the intended selective effect is:

- materially lower \(I_{\rm diff}\) than `GLOBAL-PAIR`;
- no material loss of the beneficial same-identity transfer measured by \(T_{\rm same}\).

The seed, not individual probe events, remains the sampling unit.

---

# 9. Persistent-learning endpoints

At matched checkpoints report per arm:

- current-cohort trained accuracy;
- held-out rendering accuracy;
- held-out rendering bpb;
- old-cohort delayed accuracy and bpb;
- cohort A after A/B/C/D;
- cohort B after B/C/D;
- cohort C after C/D;
- cohort D after D;
- revision accuracy/bpb after exposures 1,2,4,8;
- whole-stream bpb;
- target and nontarget bpb;
- total slow writes;
- total byte presentations.

The primary persistence endpoint is the same load-256 delayed held-out bpb family used in V45, so the 0.10-bit materiality threshold is directly comparable.

All intended identities remain in denominators; retention cannot be conditioned only on identities that happened to learn successfully.

---

# 10. Fresh seeds and statistical unit

Scientific seeds:

```text
46601 46602 46603 46604
46605 46606 46607 46608
46609 46610 46611 46612
46613 46614 46615 46616
```

These are disjoint from V45's 45501–45516 seeds and from engineering seeds.

The **seed is the replication unit**.

All arm comparisons are paired within seed/load.

Report paired Student 95% intervals over the 16 seed-level contrasts. No event, byte, pathway, identity, or probe is treated as an independent replicate.

V46 uses predeclared ordered gates rather than multiplicity-adjusted omnibus discovery. Secondary tables are descriptive unless named in a gate below.

---

# 11. Predeclared gates

Gates are evaluated in order. Failure of an upstream validity gate voids downstream qualification but does not suppress descriptive output.

## G0 — implementation identity

All required invariants pass:

- same persistent stream bytes for every arm within seed/load;
- same BASE/FAST physical histories across arms;
- identical BASE-miss write-event indices across all arms;
- exactly 1,024 slow units and \(1024\times256\) learned slow weights per arm;
- exactly 51 slow units touched per slow write in every arm;
- identical slow learning rate = 0.6;
- PAIR representation frozen during A–D;
- no arm contains permanent slots;
- evaluator labels are absent from deployable state;
- probes/readouts do not mutate learner state.

Failure → `IMPLEMENTATION_INVALID`.

## G1 — development validity

Across 16 seeds:

- learned PAIR rank = 20 in every seed;
- mean accepted pair updates ≥ 300;
- mean evaluator-only false-pair rate ≤ 5%;
- PAIR geometry margin exceeds RAW by ≥ 0.08 with paired lower 95% bound > 0.

Failure → `DEVELOPMENT_INVALID`.

## G2 — routing health

At load 256 for `PRED-RESP`:

- effective pathway count \(N_{\rm eff}\ge5.0\) in every seed at the terminal checkpoint;
- no pathway receives <2% of selections over the full A–D persistent stream;
- no pathway receives >65% of selections;
- all eight pathways receive at least one slow weight update in every seed.

Failure → `ROUTING_COLLAPSE`.

These thresholds prevent a nominal eight-pathway model from degenerating into one or two effective banks.

## G3 — acquisition validity

At load 256:

- `GLOBAL-PAIR` mean D trained accuracy ≥ 0.90;
- `PRED-RESP` mean D trained accuracy ≥ 0.90;
- neither arm may have a seed below 0.80 D trained accuracy.

Failure → `INVALID_ACQUISITION`.

## G4 — selective interference mechanism

At load 256 compare `PRED-RESP` with `GLOBAL-PAIR` on the seed-level mean per-write interference summaries.

### Harmful cross-talk requirement

Let

\[
\Delta I=I_{\rm diff}^{\rm PRED}-I_{\rm diff}^{\rm GLOBAL}.
\]

Require:

\[
\boxed{\operatorname{mean}(\Delta I)\le-0.00075\text{ bpb/write}}
\]

and the paired 95% upper bound < 0.

The 0.00075 threshold is roughly one fifth of V45's observed ~0.0038 bpb/write harmful PAIR cross-talk at load 256 and is fixed before V46.

### Useful-sharing noninferiority

Let

\[
\Delta T=T_{\rm same}^{\rm PRED}-T_{\rm same}^{\rm GLOBAL}.
\]

Because more negative is better, require the paired 95% upper bound to be ≤ +0.05 bpb/write.

This prevents qualification by simply destroying all sharing.

Failure → `NO_SELECTIVE_INTERFERENCE_GAIN`.

## G5 — primary persistent gain

At load 256:

\[
\Delta L=
\text{heldout-after-D bpb}_{\rm PRED}
-
\text{heldout-after-D bpb}_{\rm GLOBAL}.
\]

Require:

\[
\boxed{\operatorname{mean}(\Delta L)\le-0.10\text{ bit}}
\]

with paired 95% upper bound < 0.

This is the same materiality scale V45 failed to reach with one global PAIR matrix.

Failure after G4 passes → `INTERFERENCE_REDUCED_NO_PERSISTENT_GAIN`.

## G6 — learned-responsibility specificity

At load 256, `PRED-RESP` must beat **each** simpler/control route on delayed held-out bpb:

- `INPUT-LEARNED`;
- `FROZEN-ROUTE`;
- `HASH-ROUTE`;
- `SHUFFLED-FEEDBACK`.

For every contrast require:

\[
\boxed{\text{control bpb}-\text{PRED bpb}\ge0.05\text{ bit}}
\]

with paired 95% lower bound > 0.

Interpretation if individual controls match:

- `INPUT-LEARNED` matches → `INPUT_CLUSTERING_SUFFICIENT`;
- `FROZEN-ROUTE` or `HASH-ROUTE` matches → `STATIC_PARTITIONING_SUFFICIENT`;
- `SHUFFLED-FEEDBACK` matches → `FEEDBACK_SPECIFICITY_FAILED`.

These are scientifically meaningful outcomes but do **not** qualify the prediction-shaped responsibility mechanism.

## G7 — prediction and revision safety

At load 256, `PRED-RESP` versus `GLOBAL-PAIR`:

- whole-stream bpb difference must be ≤ +0.02 bit in mean and paired upper 95% bound ≤ +0.04;
- revision-8 accuracy may not be lower by more than 5 percentage points in mean;
- revision-8 bpb paired upper bound must be ≤ +0.10 bit.

Failure → `PERSISTENCE_GAIN_WITH_SAFETY_COST`.

## G8 — scaling

Define positive persistence gain

\[
G_L=
\text{GLOBAL heldout bpb}_L-
\text{PRED heldout bpb}_L.
\]

Require:

\[
G_{256}\ge0.75\max(G_{128},0.05).
\]

Together with G5, this prevents a repeat of V45's pattern in which an apparent representation benefit was much larger at 128 and collapsed as load doubled.

Failure → `GAIN_COLLAPSES_WITH_LOAD`.

## G9 — bounded state

Every distributed arm must satisfy, at both loads:

- persistent slow state ≤ V45 cap = 3.75 MiB;
- identical learned slow-weight count across loads;
- identical prototype/state allocation across loads;
- no event/identity-indexed state grows without a hard cap;
- no arm's persistent state grows materially from load 128 to 256 except fixed-size counters whose allocation is already charged.

Failure → `STATE_BUDGET_FAIL`.

---

# 12. Verdict logic

Evaluate in this order:

1. G0 fail → `IMPLEMENTATION_INVALID`
2. G1 fail → `DEVELOPMENT_INVALID`
3. G2 fail → `ROUTING_COLLAPSE`
4. G3 fail → `INVALID_ACQUISITION`
5. G4 fail → `NO_SELECTIVE_INTERFERENCE_GAIN`
6. G5 fail → `INTERFERENCE_REDUCED_NO_PERSISTENT_GAIN`
7. G6 fail → classify by matching control as described above
8. G7 fail → `PERSISTENCE_GAIN_WITH_SAFETY_COST`
9. G8 fail → `GAIN_COLLAPSES_WITH_LOAD`
10. G9 fail → `STATE_BUDGET_FAIL`
11. all gates pass →

\[
\boxed{\texttt{QUALIFIED\_LEARNED\_PATHWAYS}}
\]

A simpler control that performs as well is a successful scientific simplification, not a reason to retune `PRED-RESP` mid-run.

---

# 13. Secondary analyses — descriptive only

These do not change the formal verdict:

- path-set Jaccard by evaluator category;
- pathway prototype cosine evolution;
- effective pathway count over A/B/C/D;
- responsibility entropy versus load;
- per-path outcome entropy;
- per-path cohort mixture;
- same-outcome versus different-outcome path overlap;
- load-128 component tables;
- FAST-only and slow-only endpoint decomposition;
- historical V45 comparisons shown as context only.

No post-hoc subgroup may replace the preregistered load-256 all-identity endpoints.

---

# 14. Why the controls identify the claim

V46 is not a generic mixture-of-experts contest.

- `GLOBAL-PAIR` asks whether pathways matter at all.
- `HASH-ROUTE` asks whether stable partitioning alone is enough.
- `FROZEN-ROUTE` asks whether a fixed similarity routing geometry is enough.
- `INPUT-LEARNED` asks whether ordinary online clustering is enough.
- `SHUFFLED-FEEDBACK` keeps the prediction-responsibility magnitudes but puts them on the wrong pathways.
- `PRED-RESP` is the only arm in which a pathway's own predictive success causally shapes that pathway's future responsibility.

All six arms have the same durable-write schedule, same output-weight count, same slow learning rate, same active slow-unit count, same PAIR representation, and bounded state.

Therefore, if `PRED-RESP` uniquely succeeds, the supported claim is narrow:

> **Prediction-shaped pathway responsibility improves the allocation of a fixed distributed slow synaptic substrate.**

It would not establish that the fly implements these exact equations or that eight pathways are biologically privileged.

---

# 14A. Implementation refinements locked before execution

The following clarifications are fixed during implementation and do **not** alter the scientific mechanism:

1. Pathway routing/prototype learning occurs on every byte event for which a full 12-byte context exists. Slow **synaptic** writes still occur only on the shared BASE-miss schedule. This follows the ordering in §7 and prevents routing learning from being confounded with a different durable-write schedule.
2. Read-only probes compute routes from the current prototype/usage state but do not update usage, prototypes, weights, FAST, or BASE.
3. `HASH-ROUTE` and `GLOBAL-PAIR` are charged the same prototype/usage allocation as learned routing arms even though those arrays are dummy state for these controls.
4. The interference audit samples at most 128 BASE-miss write events per seed/load by a deterministic seed/event hash and uses at most 48 prior-identity probes, matching V45's bounded audit scale. Sampling affects diagnostics only, never learning.
5. `GLOBAL-PAIR` must reproduce V45 `PAIR-ORDINARY` slow encoding and write semantics on an identical engineering stream to numerical tolerance.
6. For pathway-usage entropy, normalized occupancy is \(\pi_p=n_p/\sum_j n_j\). The 2% and 65% routing-health thresholds continue to use the distinct per-event selection rate \(s_p=n_p/N_{\rm route}\).
7. Scientific checkpoints are rolling atomic uncompressed pickle files; final result artifacts are compressed. This avoids making checkpoint compression an accidental Mac M3 bottleneck and changes no scientific state.

# 15. Mac M3 implementation plan

V46 should be implemented as one self-contained Python file and one command.

## 15.1 Shared computation inside a seed/load job

All six arms run inside one physical trajectory and share:

- stream bytes;
- development PAIR subspace;
- BASE state;
- FAST state;
- fixed 1,024-unit sparse projection;
- precomputed 1,024 slow drives per context;
- path partition;
- evaluator probe definitions.

Each arm owns only:

- its independent \(1024\times256\) slow matrix;
- its routing prototypes/usage statistics;
- its arm-local counters.

This preserves exact pairing and avoids recomputing the expensive sparse front end six times.

## 15.2 Parallelism

Default scientific execution:

- 16 seeds;
- primary load 256 first;
- secondary load 128 only after primary completion unless explicitly requested together;
- 4 macOS-safe `spawn` workers;
- BLAS / Accelerate / OpenMP thread count = 1 per worker.

Environment variables set before NumPy import:

```bash
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
OMP_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

Recommended launcher:

```bash
caffeinate -dimsu python3 elm_v46_learned_pathways.py \
  --stage primary \
  --workers 4 \
  --out-dir v46_results
```

Then:

```bash
caffeinate -dimsu python3 elm_v46_learned_pathways.py \
  --stage full \
  --workers 4 \
  --out-dir v46_results
```

The full-stage command reuses finished primary jobs.

## 15.3 Checkpoint / resume

Each seed/load job must use:

- one rolling atomic uncompressed pickle checkpoint;
- job identity containing source SHA256, design SHA256, seed, load, stream hash, development hash, and all fixed constants;
- resume only if identity matches exactly;
- atomic final-result JSON/NPZ replacement;
- corruption rejection rather than silent reuse.

Checkpoint at least:

- after development load;
- after each cohort A/B/C/D;
- every 64 persistent records within a cohort.

Scientific state and RNG state required for deterministic continuation must be checkpointed.

## 15.4 Engineering tests before science

Engineering seeds must be disjoint from 46601–46616.

Required tests:

1. compile/import;
2. deterministic development repeat;
3. representation rank/normalization;
4. path partition is complete and disjoint;
5. every routed arm touches exactly 51 slow units/write;
6. `GLOBAL-PAIR` reproduces V45 PAIR-ORDINARY semantics on an identical tiny stream;
7. arm-order permutation leaves per-arm results unchanged;
8. probes are read-only;
9. no evaluator identity enters deployable state;
10. path responsibility is computed from pre-update predictions;
11. shuffled-feedback preserves eventwise responsibility multiset;
12. forced interruption + resume is bit-identical to uninterrupted run;
13. two-worker and four-worker `spawn` runs are deterministic;
14. state cap accounting includes prototypes, usage, fixed projection, PAIR matrix, and counters.

Engineering results may invalidate the implementation. They may **not** be used to tune the locked scientific constants. A parameter change after engineering requires a new design identity before science.

---

# 16. Required outputs

The scientific runner must write:

- `MANIFEST.json`
- `PROTOCOL.md`
- `GATES.json`
- `SUMMARY.md`
- `REPORT.json`
- `DEVELOPMENT_AUDIT.csv`
- `TERMINAL.csv`
- `CHECKPOINT_METRICS.csv.gz`
- `PERSISTENCE_CONTRASTS.csv`
- `ROUTING_HEALTH.csv`
- `PATHWAY_USAGE.csv`
- `PATHWAY_OVERLAP.csv`
- `PATHWAY_PROTOTYPES.csv.gz`
- `RESPONSIBILITY_SUMMARY.csv`
- `INTERFERENCE_SUMMARY.csv`
- `INTERFERENCE_EVENTS.csv.gz`
- `REVISION.csv`
- `STATE_COSTS.csv`
- `SAMPLE_BUDGET.csv`
- `RESUME_AUDIT.json`
- `FIREWALL_AUDIT.json`

Final upload bundle:

```text
v46_results_upload.zip
```

The upload bundle excludes rolling checkpoints and large transient caches.

---

# 17. Stop conditions and next branch

V46 is not an open-ended pathway search.

## If `QUALIFIED_LEARNED_PATHWAYS`

Do **not** add factor coding immediately. First confirm on fresh seeds at load 512 and on at least one changed nuisance/stream family while keeping the V46 mechanism fixed.

Only after confirmation should the project address continual online PAIR-representation adaptation and coordinate compatibility.

## If `INPUT_CLUSTERING_SUFFICIENT`

Prefer the simpler input-only self-organized pathway learner. Retire predictive-feedback complexity unless a new independent task specifically requires it.

## If `STATIC_PARTITIONING_SUFFICIENT`

The learned-responsibility claim fails. Keep the simpler bounded partitioning mechanism as an engineering result and investigate why ordinary stable segregation is enough before adding biology-inspired teaching machinery.

## If `FEEDBACK_SPECIFICITY_FAILED`

Prediction-shaped feedback is not identified; retire it.

## If G4 passes but G5 fails

The pathway mechanism changes local interference but not enough to solve persistence. Do not tune pathway count/routing temperatures. Move to the already identified alternative representation question:

> learn both **positive invariance** and **negative near-collision separation** from experience.

That becomes the next representation branch.

## If G4 fails

The learned-pathway hypothesis is closed in this form. Do not return to slots. The next branch is likewise positive+negative representation learning, because V45/V46 would then show that changing *where* a shared matrix writes does not adequately control identity collisions.

---

# 18. What V46 would and would not establish

A passing V46 would establish, on this controlled byte-level continual-learning family, that:

1. reusable learned representation can feed a bounded distributed memory;
2. learned overlapping pathway responsibility can selectively reduce harmful cross-identity interference;
3. this yields a material persistence gain at load 256 without permanent object slots, replay, growing state, or reduced acquisition;
4. the gain cannot be explained by simple hashing, static similarity partitioning, input-only clustering, or misassigned feedback.

It would **not** establish:

- whole-fly biological fidelity;
- that eight pathways or three active routes are biologically correct numbers;
- continual representation learning without coordinate drift;
- language-model competence;
- unlimited storage;
- real-world generality.

The claim remains deliberately narrow and falsifiable.

---

# 19. Provenance

Primary source artifacts used to lock this design:

- `V45_invariance_guided_distributed_consolidation_design.md`  
  SHA256 `00eb0a01b183f83604d938ca4d427efcfa372a1d02297772cced062ef508b4c8`
- `elm_v45_invariance_guided_consolidation.py`  
  SHA256 `c8eefb54fe965b2afce3dcf5aafc6603eab4b00b56b70b587e22d94573a287bc`
- completed V45 `SUMMARY.md`  
  SHA256 `432886dbdadfe9dbe8ac59e6b7b85463ae48d642a47f82a4c1b817023d402d95`
- BANC v888 graph re-audit `GRAPH_REAUDIT.json`  
  SHA256 `701634b78a6989df2a09f8667998828c69b660c61035a1e084cb40f242ef8c3e`

The BANC measurements motivate the organizational hypothesis only. The V46 equations, pathway count, routing cardinality, homeostatic constants, and qualification thresholds are computational design choices fixed before V46 outcomes are observed.
