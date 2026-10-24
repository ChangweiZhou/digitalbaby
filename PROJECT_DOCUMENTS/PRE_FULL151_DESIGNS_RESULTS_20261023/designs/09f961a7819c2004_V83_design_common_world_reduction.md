# MiniFly V83 — Common-world anchor audit and cross-pole reduction

**Status:** proposed protocol; no V83 scientific trajectories have been run.

## 0. Frozen project objective

V83 is not a new evolutionary search and does not introduce a new biological mechanism. Its purpose is to determine which parts of the existing V81E and V82E results survive on common experimental ground, and whether their apparent plasticity/protection differences can be reduced to explicit code-level blocks.

The persistent-core target remains a bounded, local learner that can acquire new associations, retain established associations, revise obsolete associations, survive misleading observations, and continue learning without external replay, task IDs, cue-indexed storage, or growing mutable state.

## 1. Why V83 is needed

V81E and V82E were evaluated on different confirmation histories and were selected under different evolutionary procedures. Their aggregate summaries therefore cannot establish that they are genuinely complementary on the same worlds.

Several audit issues must also be removed before mechanistic attribution:

1. The historical `FrozenReader` froze the RBF head but used each genotype's modified `Q` projection. V83 makes the **reference-Q projection reader** primary and retains the genotype-coupled reader only as a shadow diagnostic.
2. PN→KC jitter is deterministically seeded from the whole genome. Naive gene swaps therefore redraw the structural perturbation. V83 freezes the realized PN→KC matrix to the donor of the declared static block for every hybrid.
3. V82E's `g_support` scales teaching onto **all 544 readout-only eligible KCs when nonzero**; it does not choose a fraction of cells. V83 reports the actual expanded-cell count and does not describe this gene as dynamic eligibility.
4. Per-genotype conditional loss uses different initially-correct cohorts. V83 retains per-memory rows and adds a **shared-initially-correct identity analysis** on the intersection of memories initially correct in REF, V81E and V82E.
5. V82E's pooled resource summaries concealed stratum-level slow-budget failures. V83 reports resource use per history/load/case as well as pooled values.

## 2. Arms

V83 uses the exact 17-gene V81E and V82E champions delivered in their source/result bundles.

### Anchors

- `REF`: all 17 genes zero.
- `V81`: exact V81E champion.
- `V82`: exact V82E champion.

### Predeclared block swaps

For attribution only, the genome is partitioned into two non-overlapping **code-level blocks**. These are not claimed to be independent biological modules.

**STATIC_INTERFACE block**

- `g_support`
- `g_teach_scale`
- `g_readout`
- `g_feedback`
- `g_mbon`
- `g_k0`
- `g_sparsity`
- `g_pnkc_weight`
- `g_pnkc_rewire`

This block determines the fixed encoding/interface/circuit arrays for a lifetime. `g_readout` is included here even though `Q` participates in expression and teaching. The block name is operational, not physiological.

**PLASTICITY_DYNAMICS block**

- `b0`
- `b_adapt`
- `b_fast`
- `b_slow`
- `g_alpha_gain`
- `g_fast_tau`
- `g_slow_tau`
- `g_fraction`

Two hybrids are fixed before any V83 outcomes:

- `S81_D82`: STATIC_INTERFACE from V81, PLASTICITY_DYNAMICS from V82.
- `S82_D81`: STATIC_INTERFACE from V82, PLASTICITY_DYNAMICS from V81.

For each hybrid, the realized PN→KC `B` matrix is copied from the exact champion that donated the STATIC_INTERFACE block. Thus swapping dynamics does not redraw structural noise.

### Targeted ablations

Two narrow arms are also predeclared because they directly test prominent interpretations from earlier rounds:

- `V82_NOSUPPORT`: exact V82 except `g_support = 0`; uses the exact V82 PN→KC realization.
- `V81_NATIVE_ALLOC`: exact V81 except `b0=b_adapt=b_fast=b_slow=0`; uses the exact V81 PN→KC realization.

These arms are mechanistic probes, not candidate persistent cores.

Total: **7 arms**.

## 3. Common fresh worlds

Default: **24 fresh histories**. Every arm sees the same raw PN patterns, outcome assignments and evaluator schedules within a history.

Each history contains 240 raw PN cues = 120 cue pairs. Each cue is an 8-of-88 PN pattern. New patterns are rejected if their support exactly duplicates any pattern discoverable in prior project `inputs.npz` / interface files. Roles use the inherited balanced 6/6 construction within blocks of 12 pairs.

The candidate's own encoder transforms the shared raw PN world into KC activity. This is intentional: coding sparsity and PN→KC perturbations are part of the evolved phenotype.

Loads 48 and 96 are evaluated within the same history.

## 4. Primary observer

The **primary V83 reader** uses the RBF head together with the unmodified reference `Q` projection. It is identical across all arms.

For every endpoint V83 also records:

- actual-behavior margin and actual-only choice;
- fixed-reference-reader margin and joint choice (**primary**);
- historical genotype-coupled reader margin and joint choice (**shadow audit**).

A result that exists only under the genotype-coupled reader cannot support a persistent-core claim.

V83 also records alpha-output clipping frequency at each probe.

## 5. Branched common-panel assay

For comparability with V82E, each arm builds its own foundation from zero using its own autonomous rule.

At loads 48 and 96, clone the foundation and evaluate:

- `STABLE`: 12 correct-contingency bouts;
- `ANOMALY1`: one contradictory bout, then 11 correct bouts;
- `ANOMALY2`: two contradictory bouts, then 10 correct bouts;
- `REV`: 12 reversed-contingency bouts;
- `NEW`: six bouts on a previously unseen association.

For every branch:

1. evaluate at the inherited common clock;
2. evaluate a non-mutating +24 h clone;
3. continue the actual common-clock branch through 12 additional NEW associations;
4. evaluate immediately and on a non-mutating +24 h clone.

As in V82E, the learner never sees case names, target age, future outcomes, old-memory margins, reader values, or evaluator labels.

### No-challenge baseline

Each challenge has a matched branch with the same sensory schedule and challenge learning disabled, followed by the learner's own ordinary mixed-learning suffix. This separates challenge-learning collateral from passage of time and subsequent learning.

## 6. Per-memory identity accounting

V83 retains one row per old memory at day endpoints, including:

- raw memory index;
- whether it was initially correct;
- final actual/fixed-reader/coupled-reader choices;
- actual and reader margins;
- corresponding no-challenge baseline choice/margin.

The primary cross-genotype identity analysis uses, for each shared world/endpoint, the intersection of old memories that were initially correct under **all three anchors**: REF, V81 and V82.

For that same identity set report:

- loss rate in each anchor;
- recovery is not allowed to cancel a loss because all selected memories started correct;
- actual-only and fixed-reader versions separately.

This directly tests the earlier V81 churn / V82 protection narrative without changing denominators across genotypes.

## 7. Learned-state-dependence control

At each foundation checkpoint, clone the learner, zero alpha/gamma fast state and slow state, retain fixed phenotype and adaptation, and re-probe the learned cues.

Report the drop from learned foundation performance to plastic-reset performance. A genotype whose behavior remains largely intact after plastic reset requires additional scrutiny for fixed-phenotype or observer shortcuts.

No reset branch participates in learning or qualification.

## 8. Secondary uninterrupted-life assay (anchors only)

REF, V81 and V82 additionally face one continuous, non-branching lifetime on the same raw world.

1. Acquire pairs 0–47.
2. Select 12 disjoint old pairs deterministically from the history seed: 4 STABLE, 4 ANOMALY1, 4 REV.
3. Interleave 12 cycles over those targets. STABLE receives the original role; REV receives the opposite role in every cycle; ANOMALY1 receives the opposite role only in the first cycle and the original role thereafter.
4. Advance the **actual learner** by 24 h.
5. Evaluate all 48 pairs against their current contingencies.
6. Acquire pairs 48–71, advance actual time by 24 h, evaluate all 72.
7. Acquire pairs 72–95, advance actual time by 24 h, evaluate all 96.

This panel is secondary because it is new. It addresses the limitation that previous day probes were observation clones rather than experienced delays.

## 9. Primary metrics

No single weighted score determines the scientific interpretation.

### Plasticity / learning

- NEW target joint choice at day;
- REV target joint choice at day;
- post-MIXED REV day;
- mixed NEW accuracy at day;
- foundation/capacity day.

### Stability / identity

- absolute old accuracy;
- loss among each arm's initially-correct old memories;
- loss on the **shared-initially-correct identity set**;
- worst old-margin damage;
- 95th-percentile old-margin damage;
- challenge-relative old change versus matched no-write baseline.

### Resource / mechanism

- cumulative total alpha L1;
- cumulative slow L1;
- slow/alpha ratio;
- gamma L1;
- expanded-cell count;
- active KC fraction;
- alpha clipping fraction;
- mutable-state bytes.

### Observer audit

- actual-only choice;
- fixed-reader joint choice;
- genotype-coupled-reader joint choice;
- disagreement rates between the three.

## 10. Statistical unit

The independent unit is the **history**.

Loads, cases, phases, memories and horizons are repeated measurements and are never treated as independent `n`.

For key contrasts:

1. aggregate repeated cells within each history first;
2. use paired history-level bootstrap 95% intervals;
3. report sign counts;
4. do not make dozens of stratum-wise adjusted tests the sole scientific verdict.

Strata remain visible descriptively.

## 11. Predeclared V83 questions

### Q1 — Does V81's learning advantage reproduce on common worlds with a truly frozen reader?

Evidence for reproduction requires the history-level mean V81−REF fixed-reader macro contrast to be positive and its actual-only contrast to have the same direction. The historical +9.7 pp effect is a reference magnitude, not a hard replication threshold.

### Q2 — Does V82's protection advantage survive on the same memory identities?

The key endpoint is shared-initially-correct old-memory loss, supported by worst-old damage. Better final aggregate old accuracy alone is insufficient because foundation quality and recoveries can differ.

### Q3 — Is either historical result reader-dependent?

A large benefit under the coupled reader but not actual-only / fixed-reader scoring is flagged `OBSERVER_COUPLING`.

### Q4 — Which code-level block carries the V81/V82 phenotype difference?

Compare the two parents with `S81_D82` and `S82_D81`.

Because `Q`, feedback and support have coupled roles, V83 interprets this only as a **gene-block interaction**. It does not claim that STATIC_INTERFACE and PLASTICITY_DYNAMICS are independent biological modules.

### Q5 — Can a static hybrid shift the plasticity–stability frontier?

A hybrid is interesting if it improves learning relative to V82 while preserving V82-like shared-memory survival, or improves protection relative to V81 without losing V81-like learning, on the same common worlds.

This is a development/mechanism result only. V83 does not promote a hybrid to the accepted core.

### Q6 — What do the two targeted ablations establish?

- `V82_NOSUPPORT` tests the contribution of the all-544-cell teaching extension in V82's background.
- `V81_NATIVE_ALLOC` tests whether V81's evolved local fast/adaptation allocation is necessary in its own background.

Negative results are background-specific; they do not universally close support expansion or fast/slow allocation.

## 12. Interpretation flags

The analysis may emit the following flags; they are descriptive, not model promotion labels.

- `ANCHOR_PATTERN_REPRODUCED`
- `ANCHOR_PATTERN_NOT_REPRODUCED`
- `OBSERVER_COUPLING`
- `V81_CHURN_REPRODUCED`
- `V82_SHARED_IDENTITY_PROTECTION_REPRODUCED`
- `STATIC_INTERFACE_DOMINANT`
- `PLASTICITY_DYNAMICS_DOMINANT`
- `STRONG_BLOCK_INTERACTION`
- `HYBRID_FRONTIER_MOVED`
- `NO_CLEAN_HYBRID`
- `SUPPORT_EXTENSION_CONTRIBUTES`
- `V81_ALLOCATION_CONTRIBUTES`
- `FIXED_PHENOTYPE_SHORTCUT_WARNING`

V83 must not end by inventing a V83b threshold or extra arm after seeing outcomes.

## 13. What V83 deliberately does not do

- no new evolution;
- no larger genome;
- no interpolation chosen after outcomes;
- no SPACE combination;
- no new teaching rule;
- no dynamic eligibility;
- no true rewiring;
- no cell deletion;
- no reader refit;
- no change to physiological parameters beyond those already encoded in the delivered champions/hybrids.

## 14. Deliverables

The runner must produce:

- `inputs.npz` and input audit;
- per-job receipts so interrupted runs resume safely;
- `endpoints.csv`;
- `memory_day.csv`;
- `foundation.csv`;
- `resources.csv`;
- `shared_identity_history.csv`;
- `continuous_life.csv`;
- `history_summary.csv`;
- `pairwise_contrasts.csv`;
- `decision.json`;
- `REPORT.md`;
- source/config hashes.

## 15. Decision after V83

V83 is a pruning experiment.

- If the V81/V82 anchor distinction does not survive common worlds and fixed observation, stop treating the champions as complementary poles; improve the evolutionary ecology before mechanistic recombination.
- If the distinction survives and one block swap shifts the frontier, compress that hybrid on a **new** fresh panel before any larger evolutionary search.
- If the distinction survives but neither static hybrid combines the strengths, the next hypothesis may be state-dependent switching or another dynamic local rule—but that is a V84 question, not a V83 rescue.
- If V82's support ablation collapses protection, support extension earns a focused follow-up. If it does not, demote support expansion.
- If V81's native-allocation ablation preserves its gain, demote the earlier fast/adaptation commitment interpretation.

The purpose is to convert two attractive evolutionary narratives into common-world causal evidence before opening additional degrees of freedom.
