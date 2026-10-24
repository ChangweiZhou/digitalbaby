# V35 — Shared Solver vs Residual Ownership

## Status

**Design / preregistration only. No V35 scientific outcomes have been observed.**

V35 is designed after the completed V34 result and the subsequent re-audit of the BANC-derived fly learning circuit.

It deliberately stops adding new KC widths, sparsity fractions, routing schemes, covariance caps, MESU variants, or temperament gates.

The round asks two sequential questions:

1. **Shared-solver question:** can a true cumulative recursive least-squares solver preserve arbitrary old byte associations on the already validated V33 lag representation?
2. **Ownership question:** if not, does separating a shared temporal/statistical predictor from a bounded, item-local residual memory solve the persistence problem?

The maximum scientific workload is **64 trajectories**. Stage II is conditional and uses a disjoint fresh seed block fixed before Stage I outcomes.

---

# 1. Why V35 exists

## 1.1 What V33 established

V33 showed that binding byte identity to temporal position was necessary but not sufficient.

The lag-preserving 96-dimensional representation substantially improved:

- same-key alias similarity;
- alias routing consistency;
- revision transfer;
- representation effective rank.

Full historical covariance then gave an independent old-memory benefit over ordinary writing.

However, strict untouched-A retention remained far below a persistent-learning standard.

Therefore:

\[
\boxed{\text{representation collision and shared-write interference were both real}}
\]

but neither repair produced durable memory.

## 1.2 What V34 established

V34 tested a fly-inspired structural hypothesis:

\[
\text{lag code}
\rightarrow
\text{fixed KC-like expansion}
\rightarrow
\text{sparse/local compartment participation}.
\]

The strongest candidate, C4-P, acquired A and D completely but retained only **3.125%** of untouched A after D.

More importantly, V34 substantially improved the geometry that had looked pathological in V33:

- effective data rank increased substantially;
- late covariance cap activation fell from nearly universal to only a few percent;
- the uncapped directional amplification fell strongly;
- related aliases showed more active-set/compartment overlap than unrelated keys.

Yet long-horizon retention still collapsed.

Thus V35 treats the following hypothesis as **demoted**:

\[
\cancel{
\text{low rank / cap saturation is the main cause of long-term forgetting}
}
\]

Sparse expansion and local wiring can delay interference for one wave, but V34 did not establish durable storage.

## 1.3 The remaining shared-memory confound

The strongest shared neural writers used so far are still greedy online update rules.

The protected directional writer asks, approximately:

> How can the current example be corrected while disturbing historical score directions as little as possible?

It does **not** maintain the exact joint optimum of all accumulated examples.

Therefore one decisive confound remains:

\[
\boxed{
\text{Does a good shared solution exist, but our greedy writer fail to track it?}
}
\]

V35 Stage I answers this directly with genuine recursive ridge / recursive least squares.

---

# 2. Fly-data motivation retained for V35

The fly audit is used only for structural inspiration. Static connectivity is **not** treated as proof of biological learning function.

The BANC-derived files support the following observations.

## 2.1 The canonical MB seed circuit is not structurally closed

The semantic seed set contains:

- 4,552 KCs;
- 699 antennal-lobe projection neurons;
- 317 mushroom-body DANs;
- 104 MBONs;
- 2 DPM neurons;
- 1 APL neuron;

for 5,675 mandatory seed neurons.

That semantic-only circuit retains only about **32.3%** of seed-incident synaptic mass in the learning-core audit.

## 2.2 Important multi-hop pathways use surrounding circuitry

At three hops, relative to the full discovery universe:

- semantic-only `AL_PN -> KC` reachability retains about **42.4%**;
- adding the broader olfactory networks raises this to about **98.9%**.

More strikingly:

- semantic-only `MBON -> MB_DAN` three-hop reachability retains about **24.7%**;
- the olfactory-network extension reaches only about **32.6%**;
- adding the superior-brain shell reaches about **92.9%**;
- the 7,000-neuron PPR core reaches about **94.6%**.

The structural lesson used in V35 is therefore:

\[
\boxed{
\text{associative readout is embedded in a larger feedback/context circuit}
}
\]

rather than being a self-contained feedforward memory box.

## 2.3 Readout coordination and teaching locality differ

The architecture extraction also shows:

- substantial MBON-to-MBON recurrent connectivity;
- much more cross-lobe MBON interaction than cross-lobe DAN-to-DAN interaction;
- predominantly local DAN recurrent structure.

V35 does **not** claim this computes a particular prediction error.

The minimal machine-learning abstraction is only:

\[
\boxed{
\text{shared prediction}
\rightarrow
\text{local responsibility}
\rightarrow
\text{local residual update}
}
\]

rather than sending the same global teaching update into every participating memory state.

## 2.4 Two-system interpretation

The fly re-audit, V33, and V34 together motivate a division of labor:

\[
\boxed{
\text{shared temporal/statistical predictor}
+
\text{item-local residual memory}
}
\]

The shared system learns recurring byte-stream structure.

The local system owns exceptional/context-specific residual associations that should not be rewritten by unrelated future contexts.

This is the V35 Stage II hypothesis.

---

# 3. Non-negotiable interface

All V35 learner arms remain causal:

\[
\text{past bytes}
\rightarrow
\text{prediction}
\rightarrow
\text{actual next byte}
\rightarrow
\text{one online update}.
\]

The learner receives **no**:

- task ID;
- wave ID;
- logical key ID;
- alias ID;
- memory ID;
- revision flag;
- corruption flag;
- future byte;
- evaluator classification.

Evaluator metadata may be used only for scoring and read-only diagnostics.

There is no replay of untouched A.

There is no reset at record, key, or wave boundaries.

---

# 4. Fixed workload

Reuse the exact V33/V34 stream generator without redesign.

Each trajectory contains:

\[
A \rightarrow B \rightarrow C \rightarrow D
\]

with:

- 16 arbitrary six-byte logical keys per wave;
- 64 logical keys total;
- 32 initial target presentations per key;
- two alternating learned renderings per key;
- one third rendering used only as a held-out alias probe;
- four A keys corrupted once after B;
- four disjoint A keys revised eight times in rendering 0 after C;
- eight untouched A keys that never appear during training after A;
- D as an additional unrelated interference wave.

Per trajectory:

- **64** unique logical keys;
- **128** learned target renderings;
- **64** never-trained held-out target renderings;
- **2,084** target prediction events;
- **29,176** total next-byte prediction events.

Unique examples and repeated presentations must always be reported separately.

Seeds, not byte events, are the independent statistical units.

---

# 5. V35 architecture: two prespecified stages

V35 contains:

- **Stage I:** 4 arms × 8 fresh seeds = 32 trajectories.
- **Stage II:** 4 arms × 8 different fresh seeds = 32 trajectories, run only if the Stage-I shared-solver qualification fails.

Maximum:

\[
\boxed{64\text{ scientific trajectories}}
\]

and

\[
64\times29,176
=
1,867,264
\]

next-byte predictions if both stages run.

No parameters or Stage-II architecture choices may be changed after Stage-I outcomes.

---

# 6. Stage I — Shared Solver Test

## 6.1 Question

\[
\boxed{
\text{Can exact cumulative online ridge regression rescue the shared parametric memory?}
}
\]

Stage I intentionally returns to the simpler V33 hard-bank lag substrate.

Do **not** use the V34 KC expansion.

This isolates the solver.

## 6.2 Fresh seeds

Scientific seeds:

\[
\boxed{35101,35102,\ldots,35108}
\]

must be audited against available prior manifests before the scientific run.

If any collision is discovered before outcomes are generated, replace the entire block with the next unused consecutive eight-seed block and lock it before scientific execution.

Engineering/smoke seeds must be disjoint.

## 6.3 Common representation and routing

Use the exact V33 lag96 representation:

- last 12 raw bytes;
- eight signed binary bits per byte;
- chronological order;
- zero-fill before 12 bytes are available;
- one L2 normalization.

Thus:

\[
z_t\in\mathbb R^{96}.
\]

Use the exact V33/V34 fixed eight-bank router and RNG order.

No router learning or bank allocation changes.

## 6.4 Stage-I arms

| Arm | Representation / routing | Writer | Purpose |
|---|---|---|---|
| **H-O** | V33 lag96 + hard 8-bank route | ordinary V33 writer | fresh ordinary baseline |
| **H-P** | same | protected covariance directional writer | validated interference-reduction baseline |
| **H-RLS** | same | exact online ridge / RLS | shared-solver test |
| **Dict** | exact last-12-byte context | existing exact count dictionary | explicit-storage diagnostic ceiling |

Exactly four fixed arms.

---

# 7. H-RLS definition

For each of the eight banks maintain:

\[
W_b\in\mathbb R^{96\times256}
\]

and

\[
P_b\in\mathbb R^{96\times96}.
\]

Initialize:

\[
W_b=0,
\qquad
P_b=I.
\]

This corresponds to ridge coefficient:

\[
\lambda=1.
\]

For the routed bank and normalized current code \(z\), let the observed next-byte one-hot vector be \(y\in\mathbb R^{256}\).

Prediction before update uses:

\[
s=W^\top z
\]

and the same V33 softmax temperature for reported predictive probabilities.

The RLS update is:

\[
g=
\frac{Pz}{1+z^\top Pz},
\]

\[
W^+
=
W+
g\left(y-W^\top z\right)^\top,
\]

\[
P^+
=
P-
\frac{Pzz^\top P}{1+z^\top Pz}.
\]

There is:

- no 0.6 correction coefficient;
- no covariance-direction cap;
- no replay;
- no finite horizon;
- no forgetting factor;
- no outcome-dependent regularization.

This arm is intended to maintain the exact ridge-regression solution:

\[
W_t
=
\arg\min_W
\left[
\sum_{\tau\le t}
\|y_\tau-W^\top z_\tau\|_2^2
+
\|W\|_F^2
\right]
\]

within each routed bank.

---

# 8. Mandatory RLS verification

Before scientific execution, the runner must include deterministic tests that compare online RLS against direct batch ridge on small synthetic matrices.

For several random and adversarial sequences verify:

\[
W_{\mathrm{RLS}}
\approx
(Z^\top Z+I)^{-1}Z^\top Y
\]

to strict float64 tolerance.

Also verify:

- \(P=P^\top\) numerically;
- \(P\) remains positive definite within tolerance;
- repeated identical observations agree with direct ridge;
- orthogonal observations agree with direct ridge;
- changing output labels agrees with direct ridge;
- read-only probes do not modify \(W\) or \(P\);
- pause/resume produces identical learned-state hashes.

Failure of these checks blocks scientific execution.

---

# 9. Stage-I primary endpoints

Primary:

\[
\boxed{\text{untouched-A accuracy after D}}
\]

averaged over its two trained renderings.

Always display beside it:

\[
A_{\text{after A}}.
\]

Also report:

- A after B;
- A after C;
- D-current accuracy;
- target-only bits/byte;
- non-target bits/byte;
- whole-stream bits/byte;
- first/second/fourth presentation acquisition;
- revision exact / known alias / held-out alias after 1, 2, 4, 8 changed observations;
- one-off corruption damage;
- state bytes;
- update FLOP proxy.

---

# 10. Stage-I acquisition validity

An arm cannot establish persistence by failing to learn.

Acquisition validity requires all of:

1. mean A-after-A \(\ge 90\%\);
2. minimum seed A-after-A \(\ge 75\%\);
3. mean D-current \(\ge 90\%\).

Invalid acquisition must be reported as such; normalized retention cannot rescue an invalid arm.

---

# 11. Stage-I statistical comparisons

Use complete paired two-sided 95% Student-\(t\) intervals across the eight seeds.

The main comparison is:

\[
\boxed{\text{H-RLS} - \text{H-P}}.
\]

Report the paired contrast for:

- A-after-D;
- D-current;
- target bpb;
- whole-stream bpb;
- revision endpoints.

No byte-level pseudo-replication.

---

# 12. Stage-I shared-solver qualification

Stage I qualifies shared parametric memory strongly enough to **skip Stage II** only if H-RLS satisfies **all** of:

### S1 — valid acquisition

The acquisition-validity criteria above.

### S2 — absolute persistence

\[
\boxed{
\operatorname{mean}(A_{\text{after D}})
\ge 75\%
}
\]

on untouched A.

### S3 — meaningful improvement over protected directional writing

\[
\operatorname{mean}
(A_{\text{after D}}^{RLS}
-
A_{\text{after D}}^{H-P})
\ge 20\text{ pp}
\]

and the paired 95% lower bound is \(>0\).

### S4 — current-learning noninferiority

The paired 95% lower bound for

\[
D_{\text{current}}^{RLS}
-
D_{\text{current}}^{H-P}
\]

must be at least:

\[
-5\text{ pp}.
\]

### S5 — no severe sample-efficiency collapse

At the second target presentation, H-RLS may not be more than 15 percentage points worse than H-P in mean accuracy.

This prevents a slow batch-like learner from qualifying solely through eventual accumulation.

If **all S1–S5 pass**, Stage II is skipped.

If **any** fail, Stage II runs exactly as preregistered below.

A result such as 50–70% terminal retention is scientifically important but does **not** count as solving the shared-memory problem; Stage II still runs.

---

# 13. Stage-I interpretation

## Outcome A — RLS reaches durable retention

If H-RLS reaches at least 75% untouched-A after D while learning D and remaining reasonably sample-efficient, then:

\[
\boxed{
\text{a good shared solution exists; previous online writers were the main failure}
}
\]

The next round should focus on:

- sample efficiency;
- computationally local approximations to RLS;
- compression of \(P\);
- longer A→B→C→D→E… streams;
- natural byte data.

Do not run Stage II.

## Outcome B — RLS improves substantially but remains below 75%

Then the solver mattered, but shared coefficients still do not provide enough ownership.

Run Stage II.

## Outcome C — RLS remains near H-P / collapses

Then the evidence becomes strong that the problem is not mainly the greedy writer.

Run Stage II.

---

# 14. Stage II — Residual Ownership Test

Stage II is completely specified before Stage-I outcomes.

It is run only when Stage I fails the shared-solver qualification.

## 14.1 Question

\[
\boxed{
\text{Does separating shared statistical prediction from item-local residual state provide durable ownership?}
}
\]

The architecture explicitly contains two private channels:

### Shared temporal/statistical channel

Answers:

> What normally happens in this byte stream?

### Position-bound residual-memory channel

Answers:

> What context-specific correction has been learned for this particular recent history?

They interact only at prediction and error feedback.

---

# 15. Stage-II fresh seeds

Scientific seeds:

\[
\boxed{35201,35202,\ldots,35208}.
\]

These are distinct from Stage I and all previous scientific blocks.

As with Stage I, perform a manifest collision audit before any outcomes are generated.

Stage-II parameters may not be altered after Stage-I results.

---

# 16. Stage-II base predictor

Use the exact previously validated **V32 current temporal ordinary byte learner** as the shared base predictor.

This is the lineage in which distributed byte input is processed through the temporal fast/slow contrast representation rather than the V33 exact lag code.

The implementation must be copied without parameter retuning and verified against an archived/reference engineering result before scientific execution.

Conceptually its role is:

\[
\boxed{\text{shared statistical prediction}}
\]

not long-term arbitrary factual storage.

The base predictor continues to learn online throughout A→B→C→D.

Its state is never frozen at a wave boundary.

---

# 17. Stage-II private key channel

Independently compute the exact V33 lag96 code:

\[
k_t\in\mathbb R^{96}.
\]

This code is used **only** to address residual memory.

The temporal base representation and lag key remain private until their predictions are combined.

There is no learned early fusion.

Thus V35 explicitly tests the two-channel principle:

\[
\boxed{
\text{compressed temporal statistics}
\quad+\quad
\text{position-bound identity}
}
\]

suggested independently by V33 and the deeper fly re-audit.

---

# 18. Stage-II arm roster

| Arm | Base predictor | Residual memory | Purpose |
|---|---|---|---|
| **BASE** | exact V32 temporal ordinary | none | statistical predictor baseline |
| **SHARED-R** | same | one shared lag96 residual matrix | residualization without item ownership |
| **EPI-R** | same | bounded approximate episodic residual slots | candidate ownership mechanism |
| **EXACT-R** | same | unbounded exact-context residual table | item-local diagnostic ceiling |

Exactly four arms × eight seeds = 32 conditional trajectories.

No extra arm is added based on Stage-I outcomes.

---

# 19. Common Stage-II prediction sequence

At every byte:

1. Compute temporal base state.
2. Compute lag96 private key.
3. Base produces logits:

\[
\ell_B.
\]

4. Residual memory, if present, produces correction:

\[
m_t.
\]

5. Final logits are:

\[
\ell_t=\ell_B+m_t.
\]

6. Predict:

\[
p_t=\operatorname{softmax}(\ell_t/T)
\]

with the locked V32/V33 temperature.

7. Actual next byte \(y_t\) is observed.
8. The base learner updates from its **own base prediction**, exactly as its standalone implementation would.
9. Residual memory receives the **final residual** and updates only its own state.

This prevents the memory from changing what the base considers a normal statistical example.

---

# 20. SHARED-R definition

SHARED-R asks whether the base/residual decomposition alone is enough.

Maintain one shared residual matrix:

\[
R\in\mathbb R^{96\times256},
\qquad
R_0=0.
\]

Residual correction:

\[
m_t=R^\top k_t.
\]

After observing \(y_t\), let:

\[
e_t
=
\operatorname{onehot}(y_t)-p_t.
\]

Update:

\[
q_0=\frac{k_t}{k_t^\top k_t}
\]

and

\[
R^+
=
R+
0.6\,q_0e_t^\top.
\]

Thus SHARED-R separates ordinary stream statistics from exception learning, but all exceptions still share one parametric residual matrix.

It is the critical control for EPI-R.

---

# 21. EPI-R bounded episodic residual memory

EPI-R is the candidate architecture.

It maintains at most:

\[
\boxed{M=256\text{ slots}}.
\]

Each slot contains:

\[
(\mu_i,v_i,n_i,t_i)
\]

where:

- \(\mu_i\in\mathbb R^{96}\): normalized context key centroid;
- \(v_i\in\mathbb R^{256}\): local logit-residual correction;
- \(n_i\): support/confidence counter;
- \(t_i\): last-use timestamp.

No semantic identity is stored.

Approximate float64 slot state is:

\[
256(96+256)\times8
\approx
0.69\text{ MiB}
\]

plus small counters.

This is bounded and smaller than the historical H-P full state.

---

# 22. EPI-R retrieval

For current normalized lag key \(k_t\), compute cosine similarity:

\[
s_i=\mu_i^\top k_t.
\]

A slot is eligible if:

\[
\boxed{s_i\ge0.70}.
\]

The value 0.70 is fixed before V35 from the already observed V33 lag-code separation between same-key and different-key contexts. It is not tuned on V35 outcomes.

Retrieve at most the four highest-similarity eligible slots.

For eligible slot \(i\), define raw responsibility:

\[
a_i=(s_i-0.70)^2.
\]

Normalize:

\[
w_i=
\frac{a_i}{\sum_j a_j}
\]

when at least one eligible slot exists.

Memory correction:

\[
m_t
=
\sum_i w_i v_i.
\]

If no eligible slot exists:

\[
m_t=0.
\]

Record the number of retrieved slots and responsibility entropy.

---

# 23. EPI-R local feedback update

After the actual byte arrives, compute final residual:

\[
e_t
=
\operatorname{onehot}(y_t)-p_t.
\]

Only retrieved slots receive credit:

\[
v_i^+
=
v_i+
0.6\,w_i e_t.
\]

This is the minimal V35 abstraction of:

\[
\boxed{
\text{readout}
\rightarrow
\text{responsibility-weighted local teaching feedback}
}
\]

suggested by the broader MBON/feedback-shell architecture.

Unretrieved slots are bit-for-bit unchanged.

There is no global revision detector.

A repeated contradiction naturally updates the same locally retrieved state.

---

# 24. EPI-R key binding

When a slot is retrieved, its key centroid updates slowly toward the current lag representation:

\[
\tilde\mu_i
=
(1-\eta_k w_i)\mu_i
+
\eta_k w_i k_t,
\]

\[
\mu_i^+
=
\frac{\tilde\mu_i}{\|\tilde\mu_i\|},
\]

with fixed:

\[
\boxed{\eta_k=0.10}.
\]

This allows two sufficiently similar aliases to share and gradually consolidate a local address without any logical-key label.

Record accidental merges between evaluator-known different logical keys as a diagnostic only.

---

# 25. EPI-R allocation

Allocation is generic and causal.

Let base-only surprisal of the observed byte be:

\[
h_B=-\log_2 p_B(y_t).
\]

If:

1. no existing slot has \(s_i\ge0.70\); and
2. base surprisal is:

\[
\boxed{h_B>4\text{ bits}},
\]

allocate a new slot.

The 4-bit threshold is fixed before V35. It means the shared predictor assigns less than \(1/16\) probability to the actual outcome.

The new slot starts with:

\[
\mu=k_t,\qquad
v=0,\qquad
n=1.
\]

Then apply the same local 0.6 residual update.

This use of surprise is **not** interpreted as a revision detector.

Its only role is resource allocation:

> create local state when the shared statistical predictor is substantially wrong and no sufficiently similar local state exists.

This distinction must be retained in the report.

---

# 26. EPI-R support and deterministic eviction

Whenever a slot is retrieved on an event with base surprisal \(>4\) bits, increment:

\[
n_i\leftarrow n_i+1.
\]

When all 256 slots are occupied and a new allocation is required:

1. evict the slot with minimum \(n_i\);
2. break ties by oldest \(t_i\);
3. break any remaining tie by lowest slot index.

This favors repeatedly confirmed exceptions over one-off surprising events.

Critically, a high-support slot is **not frozen**. Repeated contradictory outcomes can still revise its \(v_i\).

Record:

- allocations;
- evictions;
- slot support distribution;
- slot age;
- occupancy by wave;
- evaluator-known logical-key purity;
- alias sharing;
- accidental cross-key merges.

---

# 27. EXACT-R diagnostic

EXACT-R uses the same temporal base but replaces the bounded approximate slot store with an unbounded exact last-12-byte residual table.

For each exact causal context \(c\), maintain its own residual vector:

\[
v_c.
\]

Prediction:

\[
\ell=\ell_B+v_c
\]

when the context has been seen, else \(v_c=0\).

After the next byte:

\[
v_c^+
=
v_c+
0.6
\left[
\operatorname{onehot}(y)-p
\right].
\]

Different exact contexts never modify each other's local residual state.

This diagnostic:

- is not bounded;
- does not solve alias generalization automatically;
- is not a deployable final architecture.

Its purpose is to test the pure **ownership** hypothesis under the same base/residual decomposition.

---

# 28. Stage-II primary scientific comparisons

## R1 — does residual decomposition alone help?

Compare:

\[
\boxed{\text{SHARED-R} - \text{BASE}}
\]

on untouched A after D, target acquisition, and whole-stream bpb.

This asks whether separating exceptions from shared temporal statistics already reduces interference.

## R2 — does item-local ownership beat shared residual coefficients?

Primary Stage-II comparison:

\[
\boxed{\text{EPI-R} - \text{SHARED-R}}.
\]

For strong support require:

- acquisition validity in both arms;
- mean untouched-A after-D gain \(\ge20\) pp;
- paired 95% lower retention bound \(>0\);
- D-current noninferiority lower bound \(\ge-5\) pp.

## R3 — absolute bounded-memory qualification

EPI-R is a promising persistent-memory candidate only if:

\[
\boxed{
\operatorname{mean}(A_{\text{after D}})\ge75\%
}
\]

while:

\[
\operatorname{mean}(D_{\text{current}})\ge90\%.
\]

## R4 — item-local ceiling

EXACT-R should establish whether local residual ownership itself can preserve the exact trained contexts.

If EXACT-R also loses untouched A badly, the implementation or residual formulation is suspect; do not blame capacity management.

---

# 29. Stage-II revision requirements

Report revision after:

\[
1,\ 2,\ 4,\ 8
\]

changed observations for:

1. exact revised rendering;
2. previously learned alternate rendering;
3. held-out alias.

A promising EPI-R candidate should reach, by four revised observations:

\[
\text{exact}\ge90\%,
\]

\[
\text{known alias}\ge70\%,
\]

\[
\text{held-out alias}\ge50\%.
\]

These are secondary qualification criteria after persistence and current learning.

Do not hide failure behind exact-context performance.

---

# 30. Stage-II sample-efficiency requirements

Always report accuracy at:

- first target occurrence;
- second target occurrence;
- fourth;
- eighth;
- terminal acquisition.

For EPI-R versus SHARED-R, report paired differences at the second occurrence.

The architecture is not considered sample-efficient merely because 32 repeated presentations eventually fit every mapping.

The report must state explicitly:

- unique logical examples;
- unique rendered contexts;
- total presentations;
- slot allocations.

---

# 31. Noise / corruption diagnostics

The one-off A corruption cohort must remain separate from untouched retention and true revision.

For each arm report:

- probability of the old correct target immediately after one corruption;
- recovery after subsequent normal observations, if any;
- whether EPI-R allocated a new slot or modified an existing one;
- change in slot value norm;
- support counter before and after corruption.

This tests whether item ownership degenerates into memorizing every noisy event.

No extra noise gate is added in V35.

---

# 32. Whole-stream behavior

The deeper fly audit motivates a two-system learner partly because V33 showed a tradeoff:

- exact positional memory helped arbitrary target associations;
- compressed temporal representation was better for ordinary stream structure.

Therefore Stage II must report separately:

\[
\text{whole-stream bpb},
\]

\[
\text{target-only bpb},
\]

\[
\text{non-target bpb}.
\]

A promising EPI-R candidate must not worsen whole-stream bpb relative to BASE by more than:

\[
\boxed{0.10\text{ bits/byte}}
\]

in mean.

Also report the paired 95% interval.

Persistence is not sufficient if the auxiliary memory destroys ordinary stream prediction.

---

# 33. State accounting

For every arm report separately:

- trainable/shared coefficient bytes;
- covariance bytes;
- episodic key bytes;
- episodic value bytes;
- counters/metadata;
- fixed encoder state;
- resident NumPy state;
- peak temporary state if practical.

Do not describe the 256-slot memory as bounded without reporting occupancy and evictions.

Do not count repeated examples as new information.

---

# 34. Read-only diagnostics

The evaluator may compute, without changing state:

## Stage I

- bank occupancy;
- RLS \(P\) effective rank / condition diagnostics;
- direct-ridge discrepancy on checkpoints;
- old-score displacement;
- per-wave retention;
- fitted margin distributions.

## Stage II

- same-key vs different-key lag cosine;
- slot logical-key purity;
- number of logical keys per slot;
- number of slots per logical key;
- alias co-retrieval;
- accidental cross-key merge rate;
- retrieval responsibility entropy;
- occupancy and eviction by wave;
- proportion of predictions using memory;
- base surprisal on allocation events;
- memory correction norm;
- base vs memory contribution to final logits.

No diagnostic may change allocation, routing, retrieval, learning rates, or prediction.

---

# 35. Mandatory observer tests

For at least one full engineering trajectory per candidate arm:

1. run with all evaluator diagnostics enabled;
2. run with diagnostics disabled;
3. require exact final learned-state hashes and prequential loss equality.

Any discrepancy blocks science.

---

# 36. Statistical protocol

Eight seeds are the independent units in each stage.

Use complete paired two-sided 95% Student-\(t\) intervals.

Do not calculate confidence intervals over byte events.

Always report absolute levels beside differences.

Primary endpoints:

- untouched A after D;
- A after A;
- D-current.

Secondary:

- A after B/C;
- acquisition curves;
- revision curves;
- bpb;
- noise;
- slot diagnostics.

No post-hoc arm selection.

---

# 37. Interpretation tree

## Stage I passes

If H-RLS satisfies S1–S5:

\[
\boxed{\text{shared parametric storage survives the solver test}}
\]

Skip Stage II.

V36 should work on efficient/local approximations and longer streams.

## Stage I fails; EXACT-R succeeds; EPI-R succeeds

Then the evidence favors:

\[
\boxed{
\text{shared statistical learning}
+
\text{bounded item-local residual ownership}
}
\]

as the main architecture.

V36 should stress:

- capacity saturation;
- eviction;
- longer horizons;
- natural text;
- autonomous alias formation;
- compression.

## Stage I fails; EXACT-R succeeds; EPI-R fails

Then explicit ownership works, but the bounded approximate retrieval/allocation mechanism is inadequate.

The next problem is not another shared writer.

It is:

\[
\boxed{\text{addressing / capacity management}}
\]

for episodic state.

## Stage I fails; SHARED-R succeeds strongly

Then early base/residual separation, rather than item-local ownership, was the critical missing architecture.

Simplify before adding slots.

## Stage I fails; EXACT-R also fails

Do not infer that item-local memory is impossible.

First audit the base/residual combination and exact-table implementation, because exact local contexts should not be overwritten by unrelated contexts by construction.

## EPI-R retains but cannot revise

Then ownership has become too rigid.

The next mechanism should be a **slot-local contradiction/revision trace**, not a global familiarity/error gate.

## EPI-R revises but old slots are evicted

Then the primary bottleneck is bounded resource management.

That is a legitimate persistent-memory problem and should be treated separately from interference.

---

# 38. What V35 deliberately does not test

V35 does **not** test:

- another KC width;
- another sparsity percentage;
- APL-like homeostasis;
- another number of compartments;
- MESU;
- finite covariance horizon;
- novelty gating as a revision detector;
- learned router centers;
- task-aware memory allocation;
- semantic memory IDs;
- replay;
- Transformer attention;
- natural-language scale;
- developmental representation plasticity.

These remain outside the round.

---

# 39. Fly interpretation limits

The BANC-derived connectome supplies structural motivation only.

V35 does **not** claim that:

- MBON recurrence is literally an error network;
- lateral horn is an episodic memory;
- DPM is a language-model context state;
- DAN feedback implements the exact residual equations above;
- the heuristic MBON/DAN overlap table reconstructs anatomical dopamine compartments exactly;
- fly memory solves arbitrary byte-key storage.

The machine-learning proposal is an abstraction inspired by:

- recurrent preprocessing;
- associative expansion;
- local modulatory state;
- recurrent readout;
- external feedback shells;
- parallel processing.

The empirical V35 outcome, not the analogy, determines whether the architecture survives.

---

# 40. Execution contract

When implemented, V35 should follow the project execution standard:

- one self-contained Python file;
- one command;
- Mac M3 compatible;
- one BLAS thread unless explicitly overridden;
- automatic `caffeinate` on macOS;
- atomic per-trajectory commits;
- deterministic resume;
- source/protocol hashes;
- fixed manifest;
- compact outputs;
- no giant per-byte prediction dumps.

Suggested command:

```bash
python3 elm_v35_shared_solver_residual_memory.py --install-deps
```

The same command should resume.

Stage II should start automatically **only** when the locked Stage-I qualification evaluates false.

The archive must record why Stage II did or did not run.

---

# 41. Required result package

The final ZIP should contain:

- executing V35 `.py`;
- this locked protocol;
- `RUN_MANIFEST.json`;
- `REPORT.json`;
- `SUMMARY.md`;
- `SUMMARY.csv`;
- per-seed compact records;
- acquisition curves;
- retention-by-wave table;
- revision curves;
- loss decomposition;
- RLS verification diagnostics;
- slot occupancy/allocation/eviction diagnostics if Stage II runs;
- state/resource accounting;
- observer-effect audit;
- checksums.

No model matrices or per-byte raw logs are required unless a verification failure occurs.

---

# 42. Preregistered scientific claim hierarchy

V35 may support progressively stronger claims.

### Claim 1 — shared solver matters

Supported if H-RLS significantly and materially exceeds H-P while retaining current learning.

### Claim 2 — shared parametric memory is sufficient on this workload

Supported only if H-RLS passes the full 75% absolute persistence qualification.

### Claim 3 — residual decomposition matters

Supported if SHARED-R materially exceeds BASE after Stage-I failure.

### Claim 4 — item-local ownership matters

Supported if EPI-R materially exceeds SHARED-R at matched current learning.

### Claim 5 — bounded persistent residual memory is promising

Supported only if EPI-R reaches at least 75% untouched-A after D, learns D, remains reasonably sample-efficient, revises, and does not materially damage whole-stream prediction.

Do not collapse these claims into one headline.

---

# 43. Central V35 falsifiable hypothesis

The round is built around the following fork:

\[
\boxed{
\begin{array}{c}
\textbf{Either}\\[2mm]
\text{a true cumulative shared solver preserves the arbitrary facts,}\\
\text{showing that previous forgetting was largely an optimization failure,}\\[3mm]
\textbf{or}\\[2mm]
\text{persistent facts require state with local ownership,}\\
\text{while shared parameters learn the reusable statistical structure.}
\end{array}
}
\]

V35 is intended to decide between these possibilities with the smallest experiment that is still causally interpretable.

---

# 44. Evidence basis used to motivate this design

Source-derived structural facts used above come from the supplied BANC-derived project outputs:

- `fly_learning_core_results/REPORT.md`
- `fly_learning_core_results/MOTIF_REACHABILITY.csv`
- `architecture_distillation/REPORT.md`

The stronger functional interpretation—shared statistical predictor plus residual/episodic ownership—is a **machine-learning hypothesis inferred from those structural motifs together with V33/V34**, not a claim directly contained in the connectome.

Historical workload and V34 outcome details are taken from:

- `v33_protocol.md`
- `v34_protocol.md`
- `v34_implementation_and_verification.md`
- `v34_scientific_results_audit.md`

---

## Final design summary

### Stage I — guaranteed

\[
4\text{ arms}\times8\text{ seeds}=32
\]

Compare:

\[
\boxed{\text{ordinary vs protected covariance vs true RLS vs dictionary}}
\]

on the unchanged lag96 hard-bank workload.

### Stage II — conditional and already locked

If shared RLS does not qualify:

\[
4\text{ arms}\times8\text{ new seeds}=32
\]

compare:

\[
\boxed{
\text{BASE}
\quad\text{vs}\quad
\text{SHARED-R}
\quad\text{vs}\quad
\text{EPI-R}
\quad\text{vs}\quad
\text{EXACT-R}.
}
\]

Maximum total:

\[
\boxed{64\text{ trajectories}.}
\]

The intended architectural endpoint, if supported, is:

\[
\boxed{
\text{temporal statistical prediction}
+
\text{position-bound residual ownership}
+
\text{local responsibility-weighted feedback}
}
\]

rather than another globally shared persistent-memory matrix.
