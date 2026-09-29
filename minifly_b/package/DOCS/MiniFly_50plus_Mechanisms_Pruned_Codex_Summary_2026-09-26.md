# MiniFly persistent-learning project — pruned 50+ mechanism summary for Codex

**Prepared:** 2026-09-26  
**Purpose:** replace the over-broad 108-entry implementation ledger with a focused handoff centered on the **50+ biological/computational mechanisms identified in the earlier Drosophila mechanism survey**, while merging the experimentally relevant design history through BYTE_CORE V9.8.  
**Companion:** `MiniFly_Codex_Design_History_Master_2026-09-26.md` remains the exhaustive implementation/version ledger. This document is the **decision-oriented mechanism map** Codex should read first.

---

## 0. How to use this document

The historical mechanism survey contained two overlapping lists:

- a **31-entry systems/circuit/molecular matrix**;
- a **25-entry persistent-learning mechanism list**.

Together they produced **56 named entries**, with substantial overlap (for example Rac1 forgetting appears in several forms; Orb2 appears both as a molecular scaffold and as a tag/capture mechanism; sleep appears as DAN reactivation, DPM consolidation, and sleep/wake gating). The project later compressed these into a smaller computational grammar.

This document preserves the variants but **does not treat every biological label as a separate module**.

Use these status labels:

- **CORE / ALREADY REPRESENTED** — important computation, but already present or abstracted sufficiently that it should not be opened as a new mechanism search.
- **P1 — ACTIVE PRIORITY** — justified by current evidence and V9.8; reasonable near-term mechanism/architecture work.
- **P2 — CONDITIONAL RESERVE** — scientifically open, but only test after its specific failure signature appears.
- **DEPRIORITIZED** — biologically plausible, but current project evidence says it is a poor next move or is dominated by a better representative.
- **CLOSED EXACT** — do not rerun that exact engineering implementation unchanged; this is not a family-level impossibility claim.
- **DIAGNOSTIC / ORACLE** — useful as a causal upper bound or lesion, not a deployable core.
- **OUT OF CURRENT SCOPE** — downstream motor/navigation or otherwise not a persistent-memory-core mechanism.

### Critical rule

If one mechanism has five materially different variants, **keep all five variants in the ledger**. A failure of one exact variant does not erase the rest of the family.

---

# 1. Current scientific state after V9.7–V9.8

The most important new distinction is:

\[
\boxed{\text{episodic capacity} \neq \text{persistent reusable learning}.}
\]

### V9.7

`HASH8` substantially improved taught-pair memory at approximately matched deployed byte cost relative to the geometric-bank comparator. It therefore established that **isolated state/banked episodic storage can buy capacity**.

But its write-harm comparison did not establish repaired interference, and exact hashing did not preserve representational neighborhood structure.

### V9.8

`REF` showed a promising causal signal on never-reinforced combinations of previously taught features:

- E3 immediate held-out: **80.6% W vs 52.8% N**;
- after new learning + delay: **69.4% W vs 55.6% N**, imprecise at 12 worlds.

`HASH8` retained strong trained-pair memory but did not show corresponding E3 transfer.

E2 showed why exact hash routing is suspect:

- trained/withheld renderings had mean active-support Jaccard about **0.956**;
- exact KC code usually changed;
- only **12/72 = 16.7%** of trained/withheld renderings went to the same HASH8 bank.

With eight banks, 12.5% is the random same-bank baseline, so the hash largely destroys local geometry.

### Current architecture hypothesis

The project should no longer aim for maximal isolation \(K\to I\). The useful target is:

\[
\boxed{K = I + A_{\rm rel} + E_{\rm nuisance}}
\]

where:

- \(I\) = self/episodic retention;
- \(A_{\rm rel}\) = useful relation/feature sharing;
- \(E_{\rm nuisance}\) = destructive cross-talk.

The emerging architecture is therefore:

\[
\boxed{V(x)=V_{\rm shared}(x)+V_{\rm episodic}(x)}
\]

with shared state supporting transfer and private/banked state supporting capacity.

This V9.8 result is the main reason the mechanism priorities below differ from the older “protect everything from every other write” view.

---

# 2. What is already in the working substrate and should not be reopened as a separate mechanism

Several biological entries from the 50+ survey are **foundational computations**, not missing add-ons.

## 2.1 Local associative KC→MBON-like writing / dopamine-like teaching

The project already has local cue-dependent writes gated by observed outcomes/teaching signals. Therefore the following literature labels should not each become a new module:

- presynaptic heterosynaptic LTD;
- KC→MBON learning-induced plasticity;
- compartmental dopamine teaching;
- D1-like positive teaching gate;
- reward-prediction-error-style scalar teaching;
- direct dopamine behavioral valence.

The unresolved issue is **quality, locality, timing and allocation of teaching**, not whether a teaching signal exists at all.

## 2.2 Fast/slow state and channel structure

Within the later `FULL_151` implementation family, Stage 3A found these true deletions harmful:

- alpha-fast store `m.fast[:,1]`;
- slow state `m.slow`;
- adaptive allocation;
- support expansion;
- feedback pathway `m.F`.

This is **dependence inside the selected family**, not a global minimality theorem, but it means “add a generic fast/slow state” is no longer a useful new proposal.

## 2.3 Delayed eligibility already has positive evidence

OMNIBUS10 found `A_ELIG` robustly improved old and new held-out loss across backgrounds. In the selected `LEAK_HOM` background, it reduced old-A and new-D held-out loss, though it did not improve every sample-efficiency metric.

Thus eligibility is not merely a speculative biological idea; it is an **empirically supported component family**.

## 2.4 Sparse expansion is already a core operating principle

The current line already uses sparse KC-like coding. The unresolved question is whether sparsity should remain hard/static or be generated/adapted by circuit dynamics; “use a sparse expansion layer” itself is not a new mechanism.

---

# 3. P1 — mechanisms/architectures that remain active priorities

These are the families that current evidence most directly justifies.

---

## P1-A. Dynamic eligibility / synaptic tagging / recipient selection

### Computational question

\[
\boxed{\text{WHERE / WHO should receive a durable write?}}
\]

This is the most consistently supported unresolved mechanism family.

### Variants that must remain distinct

1. **`A_ELIG` (OMNIBUS10)**  
   Causal multi-frame KC eligibility trace bridging cue→consequence delay. Positive empirical evidence.

2. **`ELIG_TRACE_PORT`**  
   Source-faithful port of the historical eligibility equation into a later architecture, with `ELIG_ADDR_SHUFFLE` as an address-specific causal control. Designed; not automatically equivalent to OMNIBUS A_ELIG in another substrate.

3. **Coordinate-local `Z_i(t)` gate**  
   One bounded state per writable coordinate using only current activity, local teaching/error, fast/slow state, adaptation/resource state, and frozen anatomy.

4. **STC tag + limited capture resource**  
   Separate local tag from scarce consolidation resource. This is the correct computational abstraction of STC/mTOR/ERK/Orb2-style biological stories; do **not** emulate each molecule separately.

5. **Evolved local rule `Z_i(t)`**  
   Bounded algebraic expression using primitives such as `+,-,*,abs,tanh,clip` over local signals. This was the intended next route after broad 17-gene evolution stopped yielding clear architecture gains.

6. **Structural eligibility**  
   Eligibility for where structural/topological change may occur. This is distinct from synaptic-value eligibility and belongs at the M1/M6 boundary.

### Why it survives pruning

- A_ELIG is one of the cleanest replicated positive mechanism signals.
- A2–A6 showed destructive writes are strongly coordinate- and direction-dependent.
- V9.8 adds a new constraint: eligibility must preserve **useful overlap for related inputs**, not merely suppress overlap.

### New V9.8 requirement

A good eligibility rule must satisfy both:

\[
Z_i(x)\approx 0 \quad \text{for unrelated destructive writes}
\]

and

\[
Z_i(x)\approx Z_i(x') \quad \text{when }x,x'\text{ share task-relevant structure}.
\]

So the next eligibility work should include **relational-transfer preservation**, not only old-memory protection.

---

## P1-B. Shared relational state + private episodic state / compartment heterogeneity

This is now the main architectural synthesis implied by V9.8.

### Variants/history

1. **Native shared REF-like state** — low/moderate episodic capacity but preliminary feature reuse.
2. **4-bank fixed/inferred routing** (`context_fixed`, `context_inferred`).
3. **8-bank fixed/inferred routing** (`silent_fixed`, `silent_inferred`).
4. **FB2 / FB4 / FB8** isolated alpha-store line — strong first-return gains but increasing state cost; capacity/isolation confounded.
5. **GEOM4** — geometric routing, expensive router.
6. **HASH4 / HASH8** — cheap exact-code router; HASH8 strong trained-pair memory but poor V9.8 transfer.
7. **`H` heterogeneous compartments** — small number of compartment classes with different learning/retention rules.
8. **Stable/flexible compartment split** — conceptual two-type or small-K local policies.
9. **Shared + episodic residual** — current synthesis: shared substrate learns reusable feature structure; banked/private state stores exceptions or high-specificity residuals.
10. **BROADCAST8 diagnostic** — write to all banks while retaining banked reads; diagnostic only, useful to test whether fragmented writes kill transfer.
11. **Canonical-bank forced-read lesion** — evaluator-only V9.8 follow-up for E2; tests whether read addressing alone causes the failure.

### What was pruned inside this family

- **Exact hash routing as the whole memory architecture is not a core candidate.** It appears to trade away relational geometry.
- “More banks” (`HASH16`, `HASH32`) is **not** the next scientific move until address fragmentation is causally localized.
- Fixed banks remain useful as episodic components, not as proof that the brain should be fully partitioned.

### Why this is P1

V9.8 directly exposed the capacity/generalization tradeoff. This is no longer a theoretical topology concern; it is a measured architecture-level conflict.

---

## P1-C. Representation geometry + input-side plasticity + topology

This combines two historically separate families because V9.8 shows they must now be studied together, while still keeping their operations distinct.

### Input/representation variants (M10-like)

1. V24 `E0` clean-target backprop development — privileged historical control, not local candidate.
2. V24 `E1` observed-target backprop development — historical control.
3. V24 `E2` prescribed fixed circuitry.
4. V24 `E3` local sensory afferent development + homeostasis.
5. Dense Hebbian/Oja encoder adaptation.
6. Novelty-gated encoder change.
7. Shuffled/rate-matched gate controls.
8. Norm gate.
9. Temporal contrast / fast-minus-slow sensory channels — partial positive history.
10. Variance salience selection — task-specific; deprioritized as generic rule.
11. `P` bounded local PN→KC adaptation.
12. ITDP-like timing-dependent input plasticity.
13. `PHASE-FROZEN / PHASE-LIVE / LAG` stationarity panel.
14. V53/V54 `INVARIANT-CUE`, `XOR-OFF`, phase-Z, reliability-learning line from the separate BANC V50 lineage.

### Static/dynamic topology variants (M6-like)

1. Early prune-only development.
2. Random regrowth.
3. Guided regrowth using gradient-derived scores — nonlocal historical control.
4. New-edge lesion.
5. Topology-only transfer with weights/state reset.
6. Global cosine-kNN rewiring / hubs / EMA routing — early locality/causality issues.
7. `g_pnkc_rewire` — **not true adjacency rewiring**; row/weight reassignment in frozen encoding.
8. `g_pnkc_weight` perturbation — weight geometry, not edge replacement.
9. `g_support` — writable-support expansion, not topology rewiring.
10. A11 static topology panel: `G0_NATIVE`, `G1_WEIGHT_SHUFFLE`, `G2_REWIRE_10`, `G3_REWIRE_50` — designed, not fully executed.
11. Slow bounded prune/grow / partner-reweight `S` — open.
12. `FIXED / RANDOM / SET-like / HEBB / HOMEOSTATIC / SAMPLE+STABILIZE` dynamic structural families — proposed/open.
13. Hierarchical modular / bounded-degree expander / sparse bridge geometry — theoretical proposal, not validated.
14. Topology × commitment variants `K0/K1/K2/K3`, where only topology-blind global K0 has been tested as an exact prototype.

### Why P1 now

The project has direct evidence that:

- changing address geometry can increase capacity;
- exact hashing can destroy transfer;
- support location/count matters inside some architectures;
- true event-level structural rewiring is **largely untested**.

The goal is not “rewire because brains rewire.” The goal is to find geometry that suppresses nuisance coupling while preserving relational coupling.

---

## P1-D. Local resource budgeting / adaptive allocation / slow metaplasticity

### Variants

1. `pair_importance` usage protection.
2. `pair_importance_strong`.
3. `neuron_availability` lower-state-cost protection.
4. `pair_importance_shuffled` address control.
5. `bayes_stable` diagonal uncertainty.
6. `bayes_reopening` uncertainty + reopening/process noise.
7. `E_META` OMNIBUS diagonal uncertainty/metaplasticity.
8. Adaptive allocation in `FULL_151` — required by Stage-3A deletion within that family.
9. Fixed/equal-budget controls — showed some “adaptive allocation” benefits were really budget/gain effects.
10. Plasticity quota / competition across time — proposed; must be causal online, not future-window oracle.
11. `M` slow local resource state.
12. Receiver-availability/saturation states from V69–V71 allocation work.
13. Mitochondrial/subcellular budgeting as biological inspiration — computationally represented as bounded local write/resource state, not literal mitochondria.

### Pruning inside the family

- `pair_importance_strong` and overly rigid protection should not be revisited unchanged: they preserve old state by blocking new learning/revision.
- `E_META` is **not** currently justified as a standalone winner: it improved old retention but harmed new acquisition in OMNIBUS.
- resource budgeting remains active because `FULL_151` depends on adaptive allocation and because saturation/no-receiver-freedom was observed.

### Current role

M9 should regulate **how much plasticity is available**, while M1 decides **where it goes**. Do not collapse them.

---

## P1-E. Teaching/error quality as a cross-cutting core axis

This was not one of the literature “reserve mechanism” families, but the experimental history makes it impossible to omit from the current design map.

Relevant variants:

1. ordinary local target write;
2. normalized/error-based teaching;
3. full-KC NLMS/global-error oracle/reference;
4. consequence-specific DAN-like teaching;
5. delayed eligibility-gated teaching;
6. B slow→fast baseline reassignment downstream of teaching;
7. V52 TARGET-ONLY vs ALL byte writing;
8. class/role-masked write lesions as diagnostics.

Evidence includes a large teaching-normalization gain in one late line and the high full-KC NLMS ceiling. Therefore a new memory mechanism should not be credited for repairing a defect that actually comes from weak teaching/error information.

This is **CORE / ACTIVE CONTROL AXIS**, not a new biological module.

---

# 4. P2 — conditional reserve families

These remain scientifically open, but should not be the immediate next implementation unless their trigger appears.

---

## P2-A. Selective active forgetting / reopening

### Variants to preserve

1. V23 `active_forgetting`.
2. `surprise_immediate`.
3. `surprise_smoothed`.
4. `importance_surprise_release`.
5. `bayes_reopening`.
6. OMNIBUS `C_FORGET` targeted wrong-winner decay.
7. `CONTRACT_0.05` / global `R`.
8. `BR`, `KR`, `BKR` combinations.
9. selective downstream `R_sel`.
10. homeostatic forgetting / targeted bidirectional extinction.
11. biologically motivated Rac1/Cofilin actin-remodeling abstraction.
12. single-DAN acquisition/erasure duality abstraction.
13. D2-like anti-persistence gate.

### What is already known

- Global/unconditional reopening can improve revision/new learning but damages unrehearsed return and retention.
- `C_FORGET` was a false negative under blanket retention: it improved several competence/revision outcomes while worsening generic old-memory preservation.
- Surprise ≠ obsolescence; simple surprise-triggered erasure is not selective enough.

### Status

**P2 / selective family open; global unconditional variants CLOSED EXACT as core mechanisms.**

Trigger before opening: replicated evidence that stale slow state blocks revision or that a selective erase policy could separate obsolete from dormant-useful traces.

---

## P2-B. Latent/silent memory and reinstatement

### Variants

1. `silent_fixed`.
2. `silent_inferred`.
3. `context_fixed`.
4. `context_inferred`.
5. `P_REINSTATE`: correct/wrong/no reminder.
6. K-probation evaluator reveal `s -> s+p`.
7. proposed ACTIVE↔LATENT `L` state.
8. recent-cache hard/soft retrieval controls.
9. forgotten-but-stored engram abstraction.

### Current evidence

- Banked stores work as storage, but reminder assays did not establish **selective reinstatement**.
- K-probation reveal gave little rescue after decay; this only closes that short-lived probation trace as the latent store.

### Status

**P2 / OPEN.** Trigger: dormant return is poor while a few renewed exposures rapidly restore performance, suggesting hidden information rather than complete deletion.

---

## P2-C. Offline consolidation / sleep / rest-dependent upgrading

### Variants

1. sleep-dependent DAN reactivation;
2. DPM-mediated consolidation;
3. sleep/wake MBON microcircuit gating;
4. circadian consolidation window;
5. spaced-training disinhibition / pERK-like gating;
6. bounded offline `O` without stored episode archive;
7. replay reservoir8 / reservoir32;
8. recent replay8 / recent replay32;
9. fastslow_replay8 / fastslow_replay32;
10. immediate→intermediate→slow commitment;
11. recurrent post-event consolidation window;
12. CREB/mTOR/ERK/Orb2 molecular persistence abstracted into commitment/capture rather than separately simulated.

### What is pruned

- generic replay buffers are not preferred core mechanisms; they add presentations/state and had modest/mixed historical gains;
- one global K commitment clock is CLOSED EXACT;
- simple generic `B_CONSOL` was harmful in OMNIBUS;
- spacing effects in late experiments were largely explainable by adaptation recovery except at high dose, so “sleep solves persistence” is not currently supported.

### Status

**P2 / OPEN BUT DEFERRED.** Trigger: delay-only branch loses transfer while interference-free rest produces reproducible improvement not explained by ordinary state recovery.

---

## P2-D. Circuit-generated sparsity / APL-like inhibition

### Variants

1. hard/static top-k sparse KC code;
2. proposed circuit-generated suppressive/decorrelating `A`;
3. raw KC↔APL scalar suppressive controller (V48);
4. competitive top-response plasticity;
5. random-subset matched control;
6. bottom-k control;
7. local normalization/spillover controls;
8. `g_sparsity` evolution/coding-fraction sweeps;
9. recurrent inhibitory mechanism matched for mean activity.

### What is known

- V48 scalar APL abstraction was NEGATIVE EXACT.
- hard sparse coding remains useful.
- this family becomes important only if topology/generalization studies show that adaptive inhibition can preserve relational geometry better than hard thresholding.

### Status

**P2. V48 scalar implementation CLOSED EXACT; broader circuit family OPEN.**

---

## P2-E. Internal-state / motivational gating

### Variants

1. stubborn/adaptable heterogeneous populations;
2. V68 active-population reversal-vs-unrelated classifier;
3. `V` bounded causal internal-state learning gate;
4. incremental long-vs-short information gate;
5. octopamine/metabolic/sugar-state gate;
6. D1/D2 balance as state-dependent persistence control;
7. cholinergic single-trial-LTM gate;
8. history/motivation dopamine integration;
9. sleep/circadian state as modulator;
10. stress/age neuromodulatory state.

### What is known

- V68 showed state was highly decodable (AUROC ~0.951) but the chosen control policy harmed learning/retention.
- Therefore **information in state is not enough; the control law matters**.

### Status

**P2 / DEFERRED.** Trigger: a reproducible failure depends on current need/state and cannot be repaired by representation/write-allocation changes.

---

# 5. Mechanisms pruned from the active search

This section means **do not spend the next round on these unchanged**.

## 5.1 CLOSED EXACT implementations

- global same-sign temporal commitment `K` / legacy `THREE` as the missing core;
- global unconditional reopening `R` / `CONTRACT_0.05` as a complete core;
- OMNIBUS generic `B_CONSOL`;
- OMNIBUS `D_LOCAL` raw-locality mask;
- V48 scalar APL controller;
- V34 four sparse compartments with local covariance;
- deep `cascade8`; `cascade4` also not competitive as-is;
- exact projection16/48 strict/leaky protected-subspace implementations;
- additive consolidation bonus V19;
- V23 feedback-consolidation exact implementation;
- strong pair-importance protection unchanged;
- exact surprise-immediate/smoothed rules as selective forgetting mechanisms;
- exact hashing as the **entire** memory architecture after V9.8;
- unsigned linear MBON recurrence at the tested gain;
- raw output-gain/input-addressing V20–V22 exact variants.

## 5.2 DEPRIORITIZED engineering families

- explicit unbounded dictionaries / exact slots as the target biological core;
- replay archives/reservoirs as the default solution;
- RLS/full covariance as deployable mechanisms — retain only as diagnostics/oracles;
- global orthogonalization/Gram-Schmidt protection;
- attention/Sinkhorn/global-workspace/MoE composite systems from early ELM work;
- pretrained/Transformer teacher paths;
- learned global adapters requiring hidden backprop;
- indexed hash/tree/HNSW retrieval as a “learning mechanism” — database engineering, not the core problem.

## 5.3 OUT OF CURRENT SCOPE for the persistent-memory core

These may be real fly mechanisms but do not answer the present core question:

- ring-attractor visual landmark anchoring;
- LAL pro-goal/anti-goal steering;
- UpWind navigation integration;
- direct motor-valence pathways downstream of MB memory;
- detailed central-complex navigation implementation;
- literal retrotransposon capsid transport unless a future failure specifically calls for intercellular message transport;
- literal gap-junction modeling unless recurrent communication becomes the demonstrated bottleneck.

---

# 6. Condensed experimental evidence timeline merged from the design-history master

This is the minimum historical context needed to interpret the pruning decisions.

## V20–V23: anatomy and broad mechanism screen

- V20–V22 tested raw feedback/output gain, input-only anatomical addressing, and joint input/readout addressing. None established a broad raw-anatomy advantage.
- V23 ran **52 arms × 6 seeds**. The main useful signals were separate banks, moderate protection/availability, diagonal uncertainty, and some explicit allocation systems. The main failures were progressive rigidity, slot saturation, poor revision, and large unequal state costs.
- The important lesson was not “banks win”; it was that **selective storage can reduce overwrite**, while global protection easily freezes learning.

## V24–V35: locality, representation compatibility and write geometry

- Local sensory development was separated from privileged clean-target development.
- Moving representations caused address/content compatibility problems; freezing or splitting address/content helped only partially.
- Competition/salience heuristics did not produce a robust generic frontier improvement.
- Temporal contrast / lag-aware encoding produced useful task signal.
- Full covariance/coactivation-shaped writes produced a meaningful retention gain but required large nonlocal state and revised poorly.
- Exact RLS retained history strongly but was nonlocal and revision-limited.

## V36–V49: read authority, bounded memory, invariance and raw graph

- Read authority (`CORR2`, fixed mixtures) showed that **storage and expression are separable**.
- Corroborated probation (`TRACE-CORR`) reduced clutter-driven permanent admission in explicit-memory systems.
- Bounded superposition generalized at low load but degraded with load; exact hashing reduced transfer.
- V44 localized a mature-slot capacity bottleneck in that explicit-store architecture.
- Candidate C / V45–V47 produced a narrow but reproducible positive invariance signal, without solving persistent learning.
- V48 scalar APL inhibition was negative.
- V49 raw FlyWire/BANC-style graph alignment learned current information but retained old information poorly; raw topology superiority was unresolved.

## V50–V55 BANC functional-core/byte line

- V50 established a functional cue→consequence learning core on **BANC v888**.
- V51 showed higher-load CUE remained usable while BYTE failed catastrophically.
- V52 showed **TARGET-ONLY writing** produced a large causal byte improvement; all-byte writes were a major interference source.
- V53 showed the remaining gap was strongly representation-related: `INVARIANT-CUE × TARGET-ONLY` was far above deployable representations, and XOR contamination was another important factor.
- V54 pursued deployable cue formation; no result established a solved representation learner.
- V55 cross-substrate replication was designed/self-tested but not scientifically completed.

## OMNIBUS10: common-organism mechanism factorial

OMNIBUS10 crossed seven cell configurations with five training mechanisms over **3,920 conceptual trajectories**.

Most important result:

- `A_ELIG` robustly improved both old and new held-out loss.
- `B_CONSOL` worsened both old and new endpoints in the common assay.
- `D_LOCAL` was strongly harmful in that exact raw-locality implementation.
- `E_META` improved old retention but materially harmed new acquisition.
- `C_FORGET` harmed blanket old retention but later relevance-aware reanalysis showed small acquisition/revision benefits, so it remains a tradeoff rather than a family-level negative.

This is the strongest reason M1 survives as P1 while generic consolidation/locality do not.

## V61–V71: receiver allocation and interference geometry

- The project moved from “what store?” to **which coordinates should receive writes?**
- Restricted frozen-state solvers often found safer alternative write allocations, proving some local headroom existed.
- But a deployable local rule did not reliably identify those allocations.
- At high dose, many groups had no receiver freedom because of saturation/singleton structure.

This motivates combining M1 eligibility with M9 resource state rather than adding another generic memory layer.

## V72–V79: relapse, teaching quality, reopening and support

- Revision could succeed immediately and then relapse.
- Old slow state could block durable revision.
- `CONTRACT_0.05` improved some revision/new-learning endpoints but harmed old return.
- Teaching normalization/error learning produced a large improvement in one line.
- A full-KC NLMS/global-error reference was far above native performance, demonstrating teaching/error headroom.
- Equal-budget controls showed some allocation gains were actually budget/gain effects.
- Support expansion produced a modest positive signal.

## V79E–V81E and recovery

- Evolution searched a 17-gene phenotype including allocation, support, teaching, readout, feedback, sparsity and PN→KC perturbation.
- V81E had objective/feasibility defects; later recovery/re-evolution was required.
- `g_pnkc_rewire` did **not** implement true lifetime adjacency rewiring.

## FULL_151 / Stage 3 / A2–A6

Within the recovered FlyWire `FULL_151` implementation family, deletion confirmation found these required:

- support expansion;
- adaptive allocation;
- alpha-fast state;
- slow state;
- feedback pathway.

A2–A6 then causally localized interference:

- target-only controls dramatically improved target behavior;
- slow-state transplant recovered a large fraction of lost behavior;
- foreign writes contributed substantial negative target margin;
- LOBO subtraction improved average behavior but harmed many individual memories, rejecting a simple universal nuisance direction.

This strongly supports **structured write geometry**, not one scalar forgetting/consolidation fix.

## B/K/R and fixed banks

- `B` baseline slow→fast reassignment produced repeat positive evidence and remained an incumbent component.
- exact global `K` commitment and global `R` reopening did not improve the overall frontier enough to become core components.
- FB2/FB4/FB8 showed increasing isolated-store capacity/first-return performance at increasing state cost, but capacity and isolation were confounded.

## BYTE_CORE V9.7–V9.8

- V9.7 HASH8 showed that cheap exact-address banks can buy taught-pair capacity.
- V9.8 showed REF has a preliminary reusable-feature signal on never-reinforced E3 combinations, while HASH8 does not.
- E2 demonstrated that highly similar sparse representations can still hash to different banks.

This is the current pivot: **isolation is useful only if it does not erase relational geometry.**

---

# 7. Original 56-entry literature mechanism coverage map

This table exists so Codex can verify that pruning did not silently lose the earlier “50+ mechanisms.” The two survey lists overlap; duplicates/refinements are intentionally retained.

## 7.1 Original 31-entry systems/circuit/molecular matrix

| # | Original mechanism | Current computational mapping | Disposition |
|---:|---|---|---|
| 1 | Presynaptic heterosynaptic LTD | local teaching/write substrate | CORE / ALREADY REPRESENTED |
| 2 | Input-timing-dependent plasticity (ITDP) | M10 input plasticity + M1 eligibility | **P1** |
| 3 | Dual-valence compartmental reinforcement | heterogeneous compartments / teaching channels | CORE + **P1 architecture** |
| 4 | Octopaminergic cAMP/PKA potentiation | internal-state gate / teaching gain | P2 |
| 5 | Rac-Cofilin cytoskeletal plasticity | selective active forgetting | P2 |
| 6 | Direct DAN→MBON monosynaptic modulation | local teaching/feedback routing | CORE; not standalone new module |
| 7 | Compartment recurrent excitatory loop | local recurrent consolidation/heterogeneity | P2; exact recurrence not established |
| 8 | MP1 DAN / MVP2 dual-receptor oscillatory loop | post-event commitment/termination + state gate | P2 |
| 9 | APL recurrent microglomerular inhibition | circuit-generated sparsity | P2; scalar V48 CLOSED EXACT |
| 10 | Cholinergic gating of single-trial LTM | internal-state/commitment gate | P2 |
| 11 | UpWind neuron integration | downstream navigation/action | OUT OF CURRENT SCOPE |
| 12 | Electrical coupling via Innexin gap junctions | communication dynamics | DEPRIORITIZED pending communication failure |
| 13 | CREB-mediated de novo transcription | durable commitment | P2 abstraction; do not simulate molecule literally |
| 14 | Transient MB-specific early gene induction | commitment window/resource state | P2 / folded into M1/M4 |
| 15 | Spaced-training circuit disinhibition | rest/spacing-dependent commitment | P2; spacing mostly not yet mechanistically unique |
| 16 | Sleep-dependent DAN reactivation | offline consolidation | P2 |
| 17 | Orb2 prion-like amyloid aggregation | tag/capture + durable commitment | **P1 abstraction / P2 durability**, not literal amyloid model |
| 18 | 3'UTR spatial targeting of orb2 | structural/local eligibility | **P1** |
| 19 | dArc1/dArc2 capsid intercellular transfer | intercellular gain/message transport | DEPRIORITIZED |
| 20 | Single-DAN dual-function acquisition/erasure | selective active forgetting | P2 |
| 21 | Permanent vs transient active forgetting | selective reopening family | P2 |
| 22 | KC-MBON sleep/wake microcircuit gating | offline/state-dependent consolidation | P2 |
| 23 | Homeostatic rebound via KC microcircuits | resource/sparsity homeostasis | P1/P2 boundary; use abstract resource control |
| 24 | Ring-attractor visual landmark anchoring | central-complex navigation | OUT OF CURRENT SCOPE |
| 25 | Reward-prediction-error TD computation | teaching/error axis | CORE / CONTROL AXIS |
| 26 | LAL pro-goal vs anti-goal steering | motor policy | OUT OF CURRENT SCOPE |
| 27 | Cross-modal structural reweighting | input plasticity + topology | **P1** |
| 28 | History/motivation dopaminergic integration | internal-state gating | P2 |
| 29 | Direct dopaminergic behavioral valence | teaching/action signal | CORE / not a persistence add-on |
| 30 | Mitochondrial subcellular budgeting | local resource/metaplasticity | **P1 abstraction** |
| 31 | Adult structural remodeling of clock neurons | slow structural plasticity | **P1 family**, biology is indirect to MB task |

## 7.2 Original 25-entry persistent-learning list

| # | Original mechanism | Current computational mapping | Disposition |
|---:|---|---|---|
| 1 | Synaptic Tagging and Capture (STC) | M1 eligibility + capture resource | **P1** |
| 2 | Learning-induced KC→MBON plasticity | local value writer | CORE |
| 3 | cAMP/PKA signaling for LTM | teaching/commitment gain | CORE abstraction / P2 molecular detail |
| 4 | CREB-dependent transcription | durable commitment | P2 abstraction |
| 5 | mTOR-dependent protein synthesis | scarce consolidation resource | **P1 M1 abstraction / P2 offline** |
| 6 | Heterogeneous DAN teaching signals | teaching-channel heterogeneity / compartments | CORE + **P1 architecture** |
| 7 | D1-like dopamine receptors | positive persistence gate | CORE abstraction / P2 detail |
| 8 | D2-like dopamine anti-persistence | selective forgetting | P2 |
| 9 | Internal sugar-sensor modulation | internal-state gating | P2 |
| 10 | Octopamine–dopamine internal-state gating | M7 | P2 |
| 11 | Rac1-dependent active forgetting | M2 | P2 |
| 12 | Dopaminergic circuits for active forgetting | M2 | P2 |
| 13 | Forgotten-memory storage / latent engrams | M3 | P2 |
| 14 | Sleep-dependent engram reactivation | M4 | P2 |
| 15 | DPM sleep-memory link | M4 / compartmental recurrence | P2 |
| 16 | Circadian modulation of consolidation | M7/M4 timing gate | DEPRIORITIZED unless timing failure appears |
| 17 | Orb2 amyloid-like memory scaffold | M1/M4 durable local capture | **P1 abstraction** |
| 18 | ERK/MAPK durability signaling | intermediate commitment/resource gate | folded into M1/M4; no separate module |
| 19 | Actin-remodeling active forgetting | M2 | P2 |
| 20 | Epigenetic & miRNA persistence | slow metaplastic/commitment state | DEPRIORITIZED; too molecularly specific now |
| 21 | MBON ensemble code for valence/persistence | heterogeneous shared/private readout channels | **P1 architecture** |
| 22 | MB as internal-state sensor | M7 | P2 |
| 23 | Stress/age modulation of persistence | M7/M9 pathology/state robustness | DEPRIORITIZED for core construction |
| 24 | Representational drift + consolidation | representation/topology problem | **P1 problem class**, not a single mechanism |
| 25 | Computational MB continual-learning models | external reference family | REFERENCE, not a mechanism |

### Coverage conclusion

All 56 original labels are accounted for. After removing duplicates, molecular refinements, downstream motor systems and already-represented core operations, the truly distinct unresolved computational questions reduce to roughly:

\[
\boxed{
\begin{array}{l}
\text{eligibility / tag-and-capture}\
\text{shared-vs-private compartment architecture}\
\text{representation / topology adaptation}\
\text{resource budgeting / metaplasticity}\
\text{selective reopening}\
\text{latent expression}\
\text{offline commitment}\
\text{circuit-generated sparsity}\
\text{internal-state gating}
\end{array}}
\]

The first four are active P1 priorities; the latter five are conditional P2 reserves.

---

# 8. Exact historical variants Codex should remember but not rerun blindly

## Eligibility / write selection

- `A_ELIG` — positive.
- V69–V71 PROP / OPP / BALANCED receiver allocation — restricted safe-witness diagnostics; no decisive deployable selector.
- local-policy/refit — failed validation.
- `ELIG_TRACE_PORT` — designed source-faithful port.
- evolved `Z_i(t)` — still untested as the intended compact local-rule search.

## Consolidation

- `fast_slow2`, `fast_slow3` — not competitive overall.
- `feedback_consolidation` — exact negative screen.
- `cascade4`, `cascade8` — stability/plasticity tradeoff; depth 8 very poor.
- `B_CONSOL` — harmful in OMNIBUS.
- `TRACE-CORR` — useful in explicit-storage clutter reduction, not a biological core result.
- global K/THREE — CLOSED EXACT.
- bounded offline `O` — still open, never equated with replay buffer.

## Forgetting/reopening

- V23 active/surprise variants — insufficient selectivity.
- `C_FORGET` — competence tradeoff, not family negative.
- `CONTRACT_0.05` / R — revision benefit + return cost.
- selective `R_sel` — still open.

## Metaplasticity/protection

- pair importance — retention benefit, revision/acquisition cost.
- neuron availability — similar benefit at lower state cost.
- Bayesian/diagonal uncertainty — partial retention, new-learning cost.
- `E_META` — old retention positive, D acquisition substantially worse in OMNIBUS.
- adaptive allocation — required in FULL_151 family.

## Banks/episodic storage

- 4-bank fixed/inferred.
- 8-bank fixed/inferred.
- FB2/FB4/FB8 resource curve.
- GEOM4.
- HASH4/HASH8.
- HASH8 is **not** a general persistent-learning core after V9.8; keep as episodic comparator.

## Representation/topology

- hard sparse code.
- temporal contrast.
- local development E3.
- support expansion.
- static weight shuffle / 10% / 50% rewires (designed A11).
- true slow lifetime prune/grow remains largely untested.

---

# 9. Current experimental order after V9.8

The next work should be diagnostic and architecture-guided, not another 50-mechanism tournament.

## Step 1 — replicate reusable feature learning in REF

Run the E3 family on untouched worlds with a prespecified primary persistent endpoint.

Crucially split the immediate checkpoint into:

```text
DELAY_ONLY
DELAY_PLUS_INTERFERENCE
```

so decay and overwrite are not confounded.

Use margin as a secondary continuous endpoint and held-out choice W−N as the continuity endpoint.

## Step 2 — add a second relation family that defeats the one-bit shortcut

Use a relation requiring more than one stored class bit per symbol, e.g. a 3-state cyclic relation. Preserve balance of symbol position/outcome exposure.

Goal: determine whether REF has merely learned a tiny lookup compression or a more general reusable feature representation.

## Step 3 — causal HASH8 address lesions

### E2 forced canonical-bank read

At evaluation only, read the bank used by the semantically equivalent trained rendering. Do not use label information.

If this rescues E2, read-address instability is causal.

### BROADCAST8 diagnostic

During training, broadcast the same update to every bank while retaining normal hashed reads.

If this restores E3 transfer, fragmented writes are causal.

Neither intervention is a deployable learner; both are mechanism diagnostics.

## Step 4 — if V9.8 replicates, build one hybrid shared+episodic candidate

Do **not** jump to HASH16/HASH32.

Candidate form:

\[
V(x)=V_{shared}(x)+V_{episodic}(x),
\]

where episodic writing is novelty/error/residual-gated and shared state remains available to related inputs.

This is the natural synthesis of the surviving compartment/bank and relational-geometry mechanisms.

## Step 5 — then test P1-A eligibility on the hybrid

The eligibility rule must now be evaluated on four axes:

1. taught acquisition;
2. nuisance-format invariance;
3. never-reinforced relational transfer;
4. persistence of that transfer through subsequent learning.

Do not optimize only taught-pair accuracy.

## Step 6 — static topology/support geometry before dynamic rewiring

Only after the hybrid/eligibility baseline is stable:

- native vs weight-shuffle;
- degree-preserving rewire 10%;
- degree-preserving rewire 50%;
- support count/placement controls.

If static geometry does not matter, do not open dynamic M6 merely because structural plasticity is biologically real.

## Step 7 — open P2 families only on a matching failure

- dormant return failure -> latent memory M3;
- delay-only decay/rest benefit -> offline M4;
- stale-state revision block -> selective M2;
- state-dependent systematic gain error -> M7;
- hard-sparsity/topology instability -> M5.

---

# 10. Model-lineage warning

Do not treat all V-numbers as one continuous organism.

### BANC V50 line

V50→V55 used a BANC v888 PN→KC→MBON line. It produced the first strong cue→consequence functional core and byte-interface diagnostics.

### FlyWire / MiniFly / FULL_151 line

Later MiniFly used FlyWire v630 and produced the evolution/recovery/Stage-3/A2–A6/B/FB line. `FULL_151` is an identifier, not 151 neurons.

### BYTE_CORE V9.x line

V9.x uses its own frozen V9.2 checkpoint and source locks. `REF`, `GEOM4`, `HASH8` etc. must be interpreted inside that lineage.

Cross-line conceptual conclusions are allowed; cross-line score comparisons are not automatically meaningful.

---

# 11. Core mathematical framework Codex should preserve

For memory channel \(m\), coordinate \(i\), event \(t\):

\[
\boxed{
\Delta s_{i,t}^{(m)}
=
C_{i,t}^{(m)}T_{i,t}^{(m)}B_{i,t}^{(m)}Z_{i,t}^{(m)}-F_{i,t}^{(m)}
}
\]

where:

- \(T\): teaching / WHAT;
- \(B\): budget / HOW MUCH;
- \(Z\): eligibility / WHERE-WHO;
- \(C\): durable commitment / WHETHER TO KEEP;
- \(F\): reopening/removal.

Retrieval is separate:

\[
\hat y=\mathcal R(x,s,h).
\]

For read/write geometry:

\[
K_{\nu\mu}(\Delta t)=r_\nu^\top G(\Delta t)w_\mu.
\]

The current target is:

\[
\boxed{K=I+A_{rel}+E_{nuisance}}
\]

not merely \(K=I\).

This is the conceptual bridge between the earlier 50+ mechanisms and V9.8: a successful persistent learner must **protect unrelated memories while still allowing related experiences to share state**.

---

# 12. Codex implementation rules

1. **Do not resurrect a CLOSED EXACT variant unchanged** because its biological family remains open.
2. **Do not declare a family dead** because one exact prototype failed.
3. **Do not treat molecular labels as separate mandatory modules** when one bounded computational abstraction tests the shared operation.
4. **Do not optimize taught-pair accuracy alone.** Always include relational transfer and persistent transfer after V9.8.
5. **Do not confuse bank count with topology, or support expansion with rewiring.**
6. **`g_pnkc_rewire` was not true lifetime adjacency rewiring.**
7. **Do not call a nonlocal covariance/RLS oracle a candidate core.**
8. **Do not introduce task IDs, future relevance, obsolete/will-return labels, or evaluator identity into the learner.**
9. **Account for all persistent state and fixed routing arrays.** Fixed bank tables, structural traces, partner lists and support masks are not free.
10. **Separate diagnostic interventions from trained learners.** Forced bank reads, BROADCAST8, LOBO subtraction and oracle recipient solvers are causal diagnostics unless implemented causally online.
11. **Separate delay from interference** whenever a final probe follows both elapsed time and new learning.
12. **Keep model lineage explicit** before importing any historical result.

---

# 13. One-page current priority map

| Priority | Family / synthesis | Why now | What NOT to do |
|---|---|---|---|
| **P1** | Dynamic eligibility / STC / `Z` | strongest clean mechanism signal; write harm is coordinate-specific | do not evolve a huge global controller first |
| **P1** | Shared relational + private episodic architecture | V9.8 exposes capacity-vs-transfer conflict | do not simply add HASH16/32 |
| **P1** | Representation + topology / M6+M10 | exact hashing breaks neighborhood structure; true rewiring largely untested | do not conflate support expansion with topology |
| **P1** | Resource budgeting / M9 | adaptive allocation required in FULL_151; receiver saturation observed | do not equate rigid protection with intelligence |
| **CORE** | Teaching/error quality | large headroom remains; target-only and normalized teaching matter | do not credit memory architecture for teacher defects |
| **P2** | Selective forgetting / M2 | useful revision tradeoff, but global erase harms return | do not rerun global R as core |
| **P2** | Latent memory / M3 | still biologically plausible; prior reveal/reminder tests insufficient | do not call bank storage proof of silent engram |
| **P2** | Offline consolidation / M4 | open, but current spacing/sleep evidence not decisive | do not add replay archive by default |
| **P2** | Circuit sparsity / M5 | could stabilize useful geometry if hard top-k becomes limiting | do not rerun V48 scalar APL unchanged |
| **P2** | Internal-state gating / M7 | state is decodable but prior policy failed | do not confuse classifier AUROC with useful control |

---

# 14. Bottom line

The earlier 50+ mechanisms have **not disappeared**. They have been reduced to a small number of distinct computational questions after removing duplicates, molecule-level refinements, downstream motor circuits, and exact engineering variants that already failed.

The active project is now best described as:

\[
\boxed{
\text{persistent learning}
=
\text{good teaching}
+
\text{selective writing}
+
\text{relationally coherent shared state}
+
\text{bounded episodic capacity}
+
\text{resource control}
+
\text{conditional commitment/reopening/topology adaptation}.
}
\]

V9.8 specifically warns against solving catastrophic interference by forcing every memory into an independent address. The next architecture must suppress **nuisance coupling** without eliminating **useful relational coupling**.

That is the lens Codex should use when evaluating every surviving mechanism above.


---

# 15. Source basis for this summary

Primary recovered artifacts used to build this handoff:

- `Fly Brain Learning Mechanisms (1).docx` / `Fly Brain Learning Mechanisms.docx` — original 31-entry matrix + 25-entry persistent-learning list.
- `00_MASTER_ROADMAP.md` — compression of 47 literature mechanisms into reserve families and failure-trigger logic.
- `FULL_PROJECT_REVIEW_AND_LEDGER(1).md` — M001–M108 implementation/version audit through V49.
- `v23_results_review.md` — exact 52-arm V23 roster and costs.
- `OMNIBUS10_persistent_learning_experiment_design.md`, `elm_omnibus10.py`, and `V51_frozen_core_scope_and_eligibility_design.md` — common-organism mechanism definitions and reanalysis.
- `CORE_RECOVERY_mechanism_qualification_bridges_v1(1).md` — relevance-aware reopening/forgetting reinterpretation.
- `00_MASTER_ROADMAP.md`, Stage-3 protocol/result artifacts, A2–A6 audit materials, and B/K/R/FB designs — later FlyWire line.
- `MiniFly_Codex_Design_History_Master_2026-09-26.md` — prior exhaustive handoff used as the chronology cross-check.
- V9.8 report supplied in the current conversation — latest unseen-combination evidence and HASH8 transfer/address diagnosis.

Evidence language in this document is deliberately scoped to exact implementations. A biological mechanism is not declared false merely because its MiniFly proxy failed.
