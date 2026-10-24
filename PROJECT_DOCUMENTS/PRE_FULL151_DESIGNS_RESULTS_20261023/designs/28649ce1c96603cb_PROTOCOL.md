# V39 — Earned Consolidation Under Hard Capacity
## From one-shot exact traces to bounded generalizable object memory

**Status:** design / preregistration only. No V39 scientific outcomes have been observed.

**Proposed scientific seeds:** `39101`–`39108`  
**Independent base streams:** 8  
**Memory policies:** 6 branches per seed, sharing the same base-learning trajectory  
**Primary mechanism:** `TRACE-CORR` — probationary exact trace → corroborated two-anchor slot  
**Primary read rule:** locked V37 `CORR2`  
**Hard capacities:** `T = 512` probationary traces; `C = 384` consolidated slots  
**No PBF as a load-bearing mechanism. No parameter sweep. No automatic follow-up.**

---

# 0. Decision in one sentence

V37 established that a memory must **earn authority** before generalizing to unseen similar inputs. V38 showed that graded authority can improve utility further, but the slot-specific predictive-evidence mechanism failed its preregistered shuffled-evidence specificity veto.

V39 therefore returns to the strongest scientifically supported variable — **cross-context corroboration** — and asks a new question:

\[
\boxed{\text{Does a one-shot experience need a full long-term slot before it has earned reusable identity?}}
\]

The primary hypothesis is:

> **One surprising experience deserves immediate exact recall, but only a second distinct, predictively confirmed rendering deserves expensive, similarity-addressable long-term consolidation.**

The intended architecture is therefore

\[
\boxed{
\text{shared predictive base}
+
\text{bounded probationary exact traces}
+
\text{bounded corroborated object slots}
}
\]

rather than the V38 regime in which essentially every eligible context immediately becomes a permanent full slot.

---

# 1. Why generic eviction is not the right first capacity experiment

V38 makes the source of memory growth unusually clear.

Across the eight V38 `PRED-V` seeds at `after_D`:

- mean total slots: **1460.5**;
- mean target/object slots: **63.75**;
- mean nontarget-only slots: **1396.75**;
- mean nontarget multi-context slots: **232.75**;
- mean nontarget singleton slots: **1164.0**;
- nontarget singletons as a fraction of all slots: **79.7%**;
- mean exact member contexts: **1862.4**;
- mean incremental episodic payload (values + address state + PBF ledger): **7.16 MB**;
- mean total payload including the fixed representation and shared base: **9.51 MB**;
- mean operation proxy: about **3.40 × 10^9** multiply-add units.

The dominant waste is therefore not primarily:

> “important memories live forever and need a better eviction priority.”

It is:

> **“a one-off base error immediately allocates the same expensive long-term object machinery as a repeatedly useful identity.”**

A generic LRU/LFU tournament would treat the symptom after the allocation has already happened. V39 instead tests whether most of those allocations should never occur.

---

# 2. What V39 inherits, what it rejects, and what it is allowed to change

## 2.1 Frozen scientific substrate

V39 freezes the following successful or settled choices:

- the byte-level causal next-byte task;
- the V36–V38 A→B→C→D workload and its revision/noise structure;
- the shared base learner and its learning rule;
- lag96 address geometry;
- similarity threshold
  \[
  \tau=0.85;
  \]
- absolute local value semantics (`V`);
- PRED-V style **pre-update predictive confirmation**;
- no replay;
- no task ID, key ID, alias ID, wave ID, revision flag, noise flag, or evaluator metadata in the learner;
- V37 `CORR2` as the **primary deployable read-authority rule** for an unbound similarity hit on a consolidated slot.

## 2.2 V38 PBF is not promoted to a foundational primitive

V38 produced an excellent engineering policy (`PBF`) but failed Q10: its slot-specific evidence assignment did not beat the shuffled-evidence control by the preregistered materiality threshold.

Therefore V39 must **not** use PBF evidence to decide:

- which trace is promoted;
- which slot is evicted;
- which memory is protected;
- which write is accepted.

PBF may be reported as an optional read-only engineering diagnostic only if this adds negligible complexity. It has no qualification role.

## 2.3 V39 changes one scientific variable

V39 changes **admission/consolidation state semantics**:

- a first eligible context becomes a bounded probationary trace;
- a full similarity-addressable slot is allocated only after cross-context corroboration.

Everything else should be held as fixed as possible.

---

# 3. Engineering preflight: remove representation waste before measuring capacity

A capacity experiment is uninterpretable if the measured “memory requirement” is mostly duplicate Python state.

The V38 implementation stores several quantities redundantly:

- `member_vectors[c]`, although lag96 address vectors are deterministic from the 12-byte context;
- `local[c]`, which duplicates a reference to the slot value already recoverable from membership;
- both `sums[j]` and normalized `prototypes[j]`;
- a permanent `eligible` set even though, for the PRED-V path, a newly eligible context is immediately allocated/bound on the same event;
- full slot/member containers whose Python overhead is separate from their semantic content.

V39 therefore begins with a **mandatory lossless-compaction preflight**.

## 3.1 `COMPACT-LEGACY` representation

Implement the original unbounded PRED-V semantics with no scientific change, but:

1. reconstruct member lag96 vectors from the stored 12-byte context when needed;
2. remove `local` as a separate state table;
3. store the unnormalized prototype accumulator only and normalize on lookup;
4. derive eligibility from resident episodic state where mathematically equivalent rather than retaining a duplicate permanent set;
5. keep the 256-way value vector in float64 for V39 — do **not** introduce quantization as another variable.

## 3.2 Preflight equivalence gate P0

Before any V39 science seed runs, `COMPACT-LEGACY` must reproduce the legacy V38-style PRED-V trajectory on at least two full verification seeds.

Required equivalence:

- same base predictions at every event;
- same tentative candidate identities and similarities to numerical tolerance;
- same PRED-V confirmation decisions;
- same allocations and bindings;
- same slot values to numerical tolerance;
- same target acquisition/retention/revision curves;
- same CORR2 predictions and losses;
- same final semantic state hash after canonical serialization.

If P0 fails, the V39 scientific run does not start.

**This compaction is engineering hygiene, not the V39 scientific mechanism.**

---

# 4. The bounded memory primitives

V39 introduces two explicit state types.

## 4.1 Probationary exact trace

A trace represents one context that has earned local memory because the shared base was wrong, but has **not** yet earned generalizable object identity.

A trace stores only bounded state:

- exact 12-byte context;
- absolute float64 256-way local value;
- exact hit count;
- last-use / insertion counter;
- no separately stored lag96 vector (reconstruct from context);
- no prototype;
- no similarity-generalization authority.

Hard cap:

\[
T=512.
\]

When the trace pool is full, evict the least-recently-used trace; ties break by oldest insertion order.

Eviction intentionally forgets that one-shot memory. There is **no hidden tombstone or unbounded eligibility table** preserving it.

### Trace read semantics

- exact context hit → trace value has full absolute authority;
- merely similar context → trace is a **proposal only** and has zero current read authority.

Thus V37's support-1 lesson is preserved:

\[
\boxed{\text{one experience may own itself, but may not yet generalize.}}
\]

## 4.2 Consolidated two-anchor slot

A slot represents an identity that has earned generalizable memory.

A consolidated slot stores:

- at most **two distinct exact anchor contexts**;
- one float64 256-way absolute value;
- one 96-dimensional prototype accumulator / prototype state;
- access count;
- last-use counter;
- no additional permanent exact members beyond the two anchors.

Hard cap:

\[
C=384.
\]

A consolidated slot has CORR2 support by construction: it exists only after two distinct contexts have established the identity.

### Why two anchors are sufficient for this round

The V36–V38 target geometry already shows the relevant structure:

- first learned rendering ↔ second learned rendering similarity is just above \(\tau\);
- the held-out third rendering is below threshold against either raw rendering alone;
- the prototype formed by the two learned renderings places the held-out alias above threshold and gives the successful unseen-alias transfer.

Thus two anchors are not an arbitrary compression number. They are the **minimum state already demonstrated to generate the desired identity basin**.

Additional similar contexts may reinforce the slot value when causally confirmed, but they do not enlarge the exact-member table or prototype state in V39.

## 4.3 Consolidated-slot eviction

The same fixed policy is used by every bounded arm:

1. evict the slot with the smallest access count;
2. break ties by oldest last use;
3. break any remaining ties by oldest allocation index.

This is `LFU→LRU`, deliberately simple and evaluator-blind.

V39 does **not** claim this is the optimal eviction mechanism. It is common infrastructure so that admission policies can be compared under the same downstream pressure.

---

# 5. Primary mechanism: TRACE-CORR

For each observed 12-byte context \(c\), prediction happens before the true next byte \(y\) is revealed.

## 5.1 Exact hit

### Exact probationary trace

If \(c\) is in the trace table:

- predict with the trace's absolute value;
- after observing \(y\), update that trace with the unchanged V-rule;
- increment exact hit / recency metadata;
- **do not promote merely because the same exact context recurred**.

### Exact consolidated anchor

If \(c\) is one of a slot's two anchors:

- predict with the slot's absolute value;
- after observing \(y\), update the slot value exactly as V37/V38 would update an exact bound member;
- this preserves one-shot true revision on the learned anchor renderings.

## 5.2 Unbound context: candidate search

If there is no exact hit, compute the nearest candidate over:

1. consolidated-slot prototypes;
2. probationary trace addresses reconstructed from their contexts.

Only candidates with cosine similarity \(\ge\tau\) are retained.

If both pools contain candidates, use the single highest-similarity candidate. Ties are resolved deterministically by state type then allocation index, fixed before science runs.

## 5.3 Prediction-time authority

### Candidate is a consolidated slot

The slot has already earned two-anchor support, so locked `CORR2` applies:

\[
p(y\mid c)=p_j(y).
\]

### Candidate is a probationary trace

Support is one, so it has **zero generalized authority**:

\[
p(y\mid c)=p_B(y\mid c).
\]

The trace candidate exists only for the post-outcome corroboration test.

### No candidate

Use the shared base.

## 5.4 Post-outcome state transition

Compute, from frozen pre-update state:

- base top-1;
- candidate top-1 if a candidate exists.

Then update the shared base exactly as before.

### Case A — base was correct

Do not create or promote episodic memory for this previously unbound context.

The base already explains the event; V39 does not spend scarce long-term state on it.

### Case B — base was wrong; trace candidate exists and predicted \(y\)

This is the primary consolidation event.

Require the current context to be distinct from the trace's exact context.

Promote:

\[
\text{trace}(c_1)
+
(c_2,y)
\longrightarrow
\text{slot}(c_1,c_2).
\]

Procedure:

1. remove \(c_1\) from the trace pool;
2. allocate a two-anchor slot (evicting one full slot first only if \(C\) is already reached);
3. initialize the slot value from the trace value;
4. apply the current observation's unchanged V-update once;
5. set the prototype from the two distinct anchor addresses;
6. store exactly the two anchor contexts;
7. initialize access metadata deterministically.

### Case C — base was wrong; consolidated slot candidate predicted \(y\)

The candidate identity was useful, but V39 does not add another permanent member.

- reinforce/update the slot value using the unchanged V-rule;
- update access metadata;
- do not change the two-anchor identity state.

### Case D — base was wrong; no candidate or candidate predicted the wrong byte

Create a new probationary exact trace for \(c\).

If \(T\) is full, evict one trace first by the fixed trace-LRU rule.

This is the only admission path for a new unsupported experience.

---

# 6. Why this is a genuine consolidation hypothesis rather than “smaller cache”

TRACE-CORR makes a strong structural prediction.

A useful arbitrary object in the current task has two distinct learned renderings that share a target consequence. The first rendering should become a trace. The second rendering should be invisible to that trace at prediction time, but if the trace's pre-update value correctly predicts the second rendering's surprising target, the pair should consolidate.

By contrast:

- a singleton transient context never supplies cross-context corroboration;
- repeated exact noise may strengthen exact recall but does not establish a reusable identity neighborhood;
- generic contexts for which the base already predicts correctly do not earn new episodic state;
- a wrong similarity match fails pre-update predictive confirmation and does not consolidate.

Thus the proposed state transition is:

\[
\boxed{
\text{surprising one-shot experience}
\rightarrow
\text{probationary exact trace}
\rightarrow
\text{distinct-context predictive corroboration}
\rightarrow
\text{generalizable long-term slot}
}
\]

This directly reuses the scientifically validated V37 distinction between support 1 and support 2.

---

# 7. V39 memory policies

All policies share the same data stream and shared-base trajectory within each seed. Auxiliary memory state branches independently.

## 7.1 LEGACY-U — unbounded compact legacy reference

`COMPACT-LEGACY` PRED-V semantics with locked CORR2 read authority and no capacity limit.

Purpose:

- fresh-seed reference for target quality and bpb;
- denominator for memory/operation reduction;
- not deployable as a bounded learner.

## 7.2 ANCHOR2-U — unbounded two-anchor representation diagnostic

Immediate full-slot allocation as in legacy, but every slot stores at most two exact anchors and fixed two-anchor identity state.

No slot cap.

Purpose:

- isolate whether the two-anchor state restriction itself changes target behavior before capacity pressure is introduced.

If ANCHOR2-U materially damages target persistence or held-out revision relative to LEGACY-U, V39 cannot attribute later effects cleanly to admission and the primary consolidation claim is void.

## 7.3 FULL-CAP — hard cap without probationary admission

Same two-anchor slot representation as ANCHOR2-U, but:

- every first base-miss immediately allocates a full slot;
- slot cap \(C=384\);
- no trace tier;
- common `LFU→LRU` slot eviction.

This is the primary “just cap/evict the old architecture” control.

## 7.4 TRACE-CORR — primary deployable V39 arm

The two-tier policy in Section 5:

- trace cap \(T=512\);
- slot cap \(C=384\);
- distinct-context predictive confirmation required for promotion;
- common `LFU→LRU` full-slot eviction;
- CORR2 read authority for consolidated slots.

## 7.5 TRACE-EXACT2 — recurrence control

Same two-tier state and same caps as TRACE-CORR, except promotion is earned by the **second exact occurrence of the same probationary context**, not by a distinct context.

The promoted slot receives recurrence credit but only one unique anchor prototype.

Purpose:

\[
\boxed{\text{Does generic recurrence/frequency explain the benefit, or is cross-context identity corroboration necessary?}}
\]

This arm should preserve many frequently repeated exact memories but is predicted to generalize poorly to held-out aliases whose similarity requires the two-rendering prototype.

## 7.6 SHUFFLED-PAIR — nondeployable resource-matched control

This control receives the same **promotion count and promotion event times** as TRACE-CORR for each seed, supplied by a shadow TRACE-CORR schedule that cannot affect the control's predictions.

At each scheduled promotion, instead of promoting the causally matched trace, select a random resident trace from the same exact-hit-count class and nearest feasible age quartile; if the matched stratum is empty, fall back deterministically to the full trace pool.

Pair that random trace with the current context and execute the same allocation/value-update budget.

Thus SHUFFLED-PAIR matches approximately:

- number of full slots created;
- promotion timing;
- trace removals;
- value update count;
- hard capacities.

But it deliberately destroys **which prior experience is deemed the corroborating identity**.

This is the decisive causal control for the consolidation story. It is evaluator-assisted and explicitly nondeployable.

---

# 8. Capacity-stress tail

The ordinary A→B→C→D stream already creates more singleton nontarget slots than the proposed trace capacity. Nevertheless V39 should include a fixed, compact stress tail so that boundedness is tested deliberately rather than incidentally.

Append the tail **after the existing `after_D` checkpoint**. No old target item is replayed for learning during this tail.

The tail uses the same 14-byte framing family and no learner-visible type labels.

## 8.1 `SINGLETON` clutter

Generate **768** new unique framed contexts, each presented exactly once with an arbitrary target from the same 24-byte target alphabet.

Purpose:

- force the probationary trace pool to turn over;
- make immediate-full-allocation policies consume/evict long-term slots;
- test whether one-off experience can be forgotten without destroying consolidated objects.

## 8.2 `EXACT-REPEAT` clutter

Generate **128** additional unique contexts, each presented exactly twice with the same target, separated in time.

No alternate rendering is ever shown.

Purpose:

- distinguish generic recurrence from cross-context corroboration;
- give TRACE-EXACT2 a fair class of events it is designed to consolidate.

## 8.3 Schedule

Use a fixed two-pass schedule:

- pass 1: all 768 singleton rows plus first occurrences of the 128 repeated contexts, seed-shuffled;
- pass 2: second occurrences of the 128 repeated contexts, interleaved with an additional deterministic permutation of remaining singleton-style filler windows if needed for spacing.

The exact generator, row count, and stream SHA are locked in the run manifest before science execution.

## 8.4 Read-only post-tail probes

After the tail, probe without learning:

- untouched A keys;
- revised A keys at their final true values;
- B/C/D target keys;
- known learned renderings;
- held-out alias 2;
- a sample of stress-tail singleton and exact-repeat contexts;
- base-only counterfactual.

This creates a direct persistence-through-intervening-experience endpoint.

---

# 9. Primary hypotheses

## H1 — admission, not merely eviction, is the dominant capacity lever

TRACE-CORR should preserve substantially more old target memory than FULL-CAP under the same consolidated-slot capacity because one-off contexts remain in the bounded trace tier instead of immediately evicting long-term objects.

## H2 — one-shot exact learning survives probation

A target's first eligible rendering should be recalled on its first exact recurrence from the trace tier without waiting for consolidation.

Thus the mechanism must not solve capacity by giving up sample efficiency.

## H3 — cross-context corroboration earns unseen-alias transfer

After a second distinct learned rendering confirms the first trace, TRACE-CORR should consolidate a two-anchor slot and restore held-out alias transfer close to the unbounded reference.

## H4 — exact recurrence is not equivalent to identity corroboration

TRACE-CORR should beat TRACE-EXACT2 on held-out alias transfer while remaining competitive on exact recurrence and whole-stream utility.

## H5 — the identity-specific pairing matters beyond promotion rate

TRACE-CORR should beat SHUFFLED-PAIR despite matched promotion timing/count and nearly matched state occupancy.

If SHUFFLED-PAIR matches TRACE-CORR, the scientific consolidation claim is rejected.

## H6 — boundedness is literal

At no event may TRACE-CORR exceed:

\[
\boxed{512\text{ traces},\quad384\text{ slots},\quad768\text{ stored slot anchors}.}
\]

No hidden eligibility, tombstone, replay, or unbounded member table may exist outside these caps.

---

# 10. Qualification gates

All thresholds below are fixed before V39 results are observed.

## Q0 — engineering equivalence

P0 `COMPACT-LEGACY` equivalence passes. Otherwise no scientific interpretation.

## Q1 — two-anchor representation adequacy

ANCHOR2-U versus LEGACY-U:

- A-after-D loss no worse than **2 pp**;
- held-out revision @4 loss no worse than **5 pp**;
- whole-stream bpb no worse by more than **0.05**.

Failure voids the bounded admission comparison because the representation restriction itself is damaging the mechanism.

## Q2 — target acquisition under TRACE-CORR

Mean target acquisition after the standard learning exposures:

\[
\ge 90\%.
\]

Every seed must remain \(\ge 80\%\).

## Q3 — old persistence after D

For untouched A targets before the capacity-stress tail:

\[
\text{TRACE-CORR A-after-D}\ge90\%\text{ mean},
\]

with every seed \(\ge80\%\).

## Q4 — persistence after clutter

After the full capacity-stress tail, on the union of stable A/B/C/D target objects excluding deliberately obsolete labels:

\[
\text{current-target accuracy}\ge90\%\text{ mean},
\]

with every seed \(\ge80\%\).

This is the main lifelong-retention endpoint.

## Q5 — revision remains functional

On the existing true-revision keys:

- exact anchor revision @4 \(\ge90\%\);
- known other learned rendering @4 \(\ge90\%\);
- held-out alias @4 \(\ge90\%\);
- obsolete-value probability after 8 revision presentations \(\le0.10\).

## Q6 — whole-stream utility

TRACE-CORR whole-stream bpb must be no more than **0.10 bpb worse** than LEGACY-U on the same seed.

The same 0.10 noninferiority margin applies to nontarget bpb.

## Q7 — literal capacity

For every event in every seed:

- trace count \(\le512\);
- consolidated-slot count \(\le384\);
- stored consolidated anchor contexts \(\le768\);
- no auxiliary context-address table exceeds the sum of the explicitly bounded resident contexts;
- no replay buffer exists;
- no historical eviction log is required for prediction or learning.

Any violation fails V39 regardless of accuracy.

## Q8 — semantic memory reduction

Peak incremental episodic semantic payload for TRACE-CORR must satisfy both:

1. absolute bound \(\le 2.30\) MiB, excluding the unchanged shared base/fixed encoders and evaluator logs;
2. \(\le40\%\) of LEGACY-U's peak incremental episodic payload on the same seed.

Report actual Python container overhead separately; it cannot be used to hide an unbounded structure.

## Q9 — admission beats immediate allocation under the same slot cap

TRACE-CORR − FULL-CAP on post-clutter current-target accuracy:

- paired mean advantage \(\ge5\) percentage points;
- paired 95% lower interval \(>0\).

If FULL-CAP is already noninferior and equally bounded, probationary admission has not earned its complexity.

## Q10 — distinct-context corroboration beats exact recurrence

TRACE-CORR − TRACE-EXACT2 on held-out alias accuracy after consolidation/revision:

- paired mean advantage \(\ge10\) percentage points;
- paired 95% lower interval \(>0\).

At the same time, TRACE-CORR exact-recurrence accuracy must be noninferior within **5 pp**.

This is the primary mechanism-specific gate.

## Q11 — identity-specific promotion beats shuffled pairing

TRACE-CORR must beat SHUFFLED-PAIR on at least one of:

- post-clutter current-target accuracy by \(\ge10\) pp;
- held-out alias accuracy by \(\ge10\) pp;
- whole-stream bpb by \(\ge0.10\);

and the corresponding paired interval must exclude zero in TRACE-CORR's favor.

If Q11 fails, the claim that the correct corroborating identity matters is rejected.

## Q12 — one-shot trace sample efficiency

For target contexts that become probationary traces:

- first exact recurrence top-1 accuracy \(\ge90\%\) mean;
- no more than one external target presentation is required to create the exact trace.

## Q13 — immediate E1→E2 consolidation transfer

For a fresh logical target:

- after E1: exact E1 recurrence is remembered, but held-out alias has no required generalized authority;
- after the first distinct E2 confirmation: the slot exists immediately;
- the first subsequent held-out alias probe has top-1 accuracy \(\ge90\%\) mean.

This prevents a hidden many-shot consolidation mechanism.

---

# 11. Evaluator-only admission diagnostics

These diagnostics may use key/type metadata **only after the run**. They cannot influence the learner.

Report:

- fraction of original 64 target logical keys that reach consolidated slots;
- number of target traces evicted before consolidation;
- singleton clutter trace admission rate;
- singleton clutter consolidation rate;
- exact-repeat clutter consolidation rate;
- nontarget consolidated slots;
- target consolidated slots;
- promotion precision: fraction of consolidated target pairs belonging to the same evaluator-known logical key;
- cross-key false promotion count;
- slot evictions by target/nontarget evaluator class;
- trace residence-time distribution;
- slot access-count distribution;
- peak and terminal occupancy of each tier.

The expected TRACE-CORR signature is:

\[
\boxed{
\text{high target consolidation}
+
\text{near-zero singleton consolidation}
+
\text{large trace churn}
+
\text{stable consolidated target slots}
}
\]

but qualification is determined by Q gates, not by visually pleasing diagnostics.

---

# 12. Resource model and hard upper bound

V39 must report semantic payload from first principles, not only `sys.getsizeof`.

With float64 values and no cached trace vectors, a conservative payload budget is approximately:

### One probationary trace

- context: 12 bytes;
- 256 float64 value: 2048 bytes;
- counters/metadata: at most 24 bytes.

Budget:

\[
\le 2084\text{ bytes/trace}.
\]

At \(T=512\):

\[
\le 1.02\text{ MiB}.
\]

### One consolidated slot

- two anchor contexts: 24 bytes;
- 96 float64 prototype accumulator: 768 bytes;
- 256 float64 value: 2048 bytes;
- counters/metadata: at most 32 bytes.

Budget:

\[
\le 2872\text{ bytes/slot}.
\]

At \(C=384\):

\[
\le 1.05\text{ MiB}.
\]

Allowing bounded index arrays/maps, the preregistered semantic episodic cap is therefore:

\[
\boxed{2.30\text{ MiB}.}
\]

The unchanged shared base and fixed encoders are reported separately and do not count toward Q8, because V39 specifically tests the episodic-memory growth mechanism.

---

# 13. Operation accounting

The current unbounded PRED-V lookup cost grows approximately linearly with slot count. V39 should report operation proxies for:

- base prediction/update;
- consolidated-slot similarity scan;
- probationary-trace similarity scan;
- address reconstruction for traces;
- value scoring;
- allocation/eviction operations.

Do **not** claim sublinear retrieval in V39. The bounded scan is still linear in the fixed capacities:

\[
O((T+C)D),
\]

but because \(T+C=896\) is constant, lookup cost no longer grows with lifetime experience.

Approximate-nearest-neighbor indexing / LSH is explicitly deferred until a bounded semantic memory has first been established.

---

# 14. Hard implementation invariants

V39 is invalid if any invariant fails.

### I1 — one shared base trajectory per seed

All memory policies in a seed see the same base pre-update distribution and the same base update. Auxiliary policy predictions cannot change base learning.

### I2 — prediction precedes outcome

Trace/slot candidate choice and prediction use only pre-outcome state. Predictive confirmation is evaluated only after the true byte arrives.

### I3 — no evaluator metadata

No memory policy can access logical key, alias, target/nontarget class, stress-tail type, wave name, noise/revision status, or probe type.

### I4 — no hidden unbounded eligibility

TRACE-CORR may remember an unsupported context only if that context is currently resident in the bounded trace tier or is an explicit consolidated anchor.

An evicted trace is genuinely forgotten.

### I5 — only two exact slot anchors

A consolidated slot cannot accumulate an unbounded member set. Additional confirmed similar contexts may reinforce the value but do not become permanent exact members in V39.

### I6 — probes are read-only

All evaluator probes leave base, trace, slot, counters, recency, and RNG state unchanged.

### I7 — deterministic eviction

Given the same state and event, every deployable arm's eviction choice is deterministic.

Only SHUFFLED-PAIR has a dedicated preregistered RNG.

### I8 — matched control schedule cannot leak into deployable arms

TRACE-CORR never reads SHUFFLED-PAIR or evaluator state. SHUFFLED-PAIR may read the shadow TRACE-CORR promotion schedule only because it is a nondeployable causal control.

### I9 — resume determinism

Resume must reject source/protocol/plan hash mismatch and reproduce uninterrupted outputs exactly.

---

# 15. Required outputs — deliberately compact

Previous rounds produced unnecessarily large archives. V39 should default to aggregate scientific outputs and avoid per-byte CSV dumps.

Required:

- `SUMMARY.md`
- `REPORT.json`
- `SUMMARY.csv`
- `PROTOCOL.md`
- `RUN_MANIFEST.json`
- `STATUS.json`
- `STATE_COSTS.json`
- `OCCUPANCY_CURVES.csv` — checkpoints/event blocks, not every byte
- `ADMISSION_SUMMARY.csv`
- `EVICTION_SUMMARY.csv`
- `E1_E2_CONSOLIDATION.csv`
- `RETENTION_BY_WAVE.csv`
- `POST_CLUTTER_RETENTION.csv`
- `REVISION_CURVES.csv`
- `PAIRED_CONTRASTS.csv`
- `CONTROL_AUDIT.json`
- `COMPACTION_EQUIVALENCE.json`
- one self-contained executable `elm_v39_earned_consolidation.py`
- artifact hashes.

Per-event traces are **off by default**. An optional `--audit-events` flag may emit a compressed detailed log for one specified seed, but this is not part of the scientific archive.

Target archive size: **well below 20 MB**.

---

# 16. Engineering tests before science

`--self-test` must verify at least:

1. compact legacy state is semantically equivalent on synthetic sequences;
2. first base miss creates one trace and no full slot in TRACE-CORR;
3. exact trace recurrence gets full local authority;
4. similar support-1 trace has zero current authority;
5. distinct similar context + pre-update predictive confirmation promotes exactly once;
6. promotion removes the source trace and creates exactly one two-anchor slot;
7. same exact recurrence alone never promotes TRACE-CORR;
8. TRACE-EXACT2 does promote after the specified exact recurrence;
9. consolidated slot has full CORR2 authority on a new similar context;
10. exact anchor contradiction updates the whole slot value immediately;
11. a third confirmed similar context does not increase permanent anchor/member count;
12. trace pool never exceeds 512 under overflow;
13. slot pool never exceeds 384 under overflow;
14. slot eviction removes both anchor lookup entries and all slot state;
15. evicted trace leaves no eligibility/tombstone residue;
16. probes do not alter recency counters;
17. SHUFFLED-PAIR matches the primary promotion count/timing on a synthetic schedule;
18. SHUFFLED-PAIR changes identity assignment;
19. all deployable arms share identical base hashes;
20. deterministic resume/restart reproduces hashes.

`--verify-full` should additionally run one complete engineering seed and verify:

- P0 compact-legacy equivalence;
- all hard capacities at every event;
- no leaked evaluator metadata;
- no output artifact exceeds the configured logging budget.

---

# 17. Decision tree after V39

## Outcome A — TRACE-CORR passes Q0–Q13

Interpretation:

\[
\boxed{\text{corroborated consolidation establishes a bounded persistent associative memory on this workload.}}
\]

The architecture has now separated:

1. shared statistical prediction;
2. one-shot exact episodic traces;
3. reusable object-level long-term memory;
4. corroborated read authority.

**Next round:** deliberately exceed the number of genuinely reusable objects that the consolidated capacity can hold. That is the first scientifically legitimate round for testing eviction/consolidation priority among *useful* memories rather than filtering transient clutter.

## Outcome B — TRACE-CORR preserves targets but Q9 fails because FULL-CAP matches it

Interpretation: the trace tier is not scientifically necessary at this capacity/workload. The benefit comes from the fixed cap/common eviction or engineering compaction.

Do not elaborate probationary consolidation.

## Outcome C — TRACE-CORR beats FULL-CAP but Q10 fails because TRACE-EXACT2 matches it

Interpretation: recurrence/frequency is enough; cross-context corroboration has not earned a mechanistic role in admission.

Use the simpler recurrence-based admission rule in the next capacity stress.

## Outcome D — SHUFFLED-PAIR matches TRACE-CORR

Interpretation: correct identity-specific pairing is not established as causal. Reject the corroboration-consolidation story despite any good headline memory numbers.

## Outcome E — target traces are evicted before E2

Interpretation: the trace tier is too small **for this fixed schedule**, not proof that the consolidation idea is wrong.

Do not tune `T` post hoc. Report failure and redesign the task/capacity relationship in the next preregistered round.

## Outcome F — two-anchor representation itself fails Q1

Interpretation: exact-member state contains important information beyond the minimal two-rendering prototype. Do not proceed with capacity claims; first identify which additional member statistic is necessary.

---

# 18. What V39 may claim if successful

A successful V39 would establish, on the controlled byte workload, that:

- one-shot local learning does not require permanent full-slot allocation;
- a bounded trace tier can preserve immediate exact recall;
- distinct-context predictive corroboration can promote selected experiences into reusable object memory;
- the promoted memory preserves old knowledge through substantial intervening transient experience;
- held-out alias transfer and rapid revision survive the bounded state transition;
- total episodic state and candidate-scan cost are bounded independently of lifetime stream length;
- matched cap/recurrence/shuffled-pair controls support the claim that admission structure, not merely fewer allocations, matters.

It would **not** establish:

- optimal eviction among more useful memories than capacity can hold;
- robustness to one-off false revision;
- natural-language semantic identity;
- sublinear memory retrieval;
- large-scale language-model competitiveness;
- biological equivalence to mushroom-body consolidation.

The intended claim is deliberately narrower:

\[
\boxed{
\text{Immediate exact memory can be cheap and temporary; reusable identity memory must be earned.}
}
\]

---

# 19. Why V39 should be this round

The project has finally reached a point where “capacity” can be decomposed rather than attacked by another heuristic sweep.

The sequence of established questions is now:

\[
\begin{aligned}
\text{V35:}&\quad \text{Who owns a memory?}\\
\text{V36:}&\quad \text{Which renderings belong to one object?}\\
\text{V37:}&\quad \text{When may that object memory speak?}\\
\text{V38:}&\quad \text{Can graded confidence improve authority? Yes functionally; specific PBF mechanism not established.}\\
\textbf{V39:}&\quad \mathbf{\text{When does an experience deserve scarce long-term memory at all?}}
\end{aligned}
\]

The V38 state audit gives a direct empirical reason to ask this now: nearly four-fifths of full slots are nontarget singletons.

If V39 succeeds, the next capacity question becomes much harder but also much cleaner: **what to forget when all remaining memories have already earned consolidation?**

That is the point at which eviction should become the research problem.

---

# 20. Provenance and locked inputs

V39 design is based on the completed V37/V38 program state and, specifically, the following V38 artifacts:

- `SUMMARY.md` / `REPORT.json` — V38 mechanism verdict and qualified/failing gates;
- `STATE_COSTS.json` — unbounded slot/member/payload/operation growth;
- `ADDRESSING.json` — target vs nontarget slots and multi-context counts;
- `PROTOCOL.md` — locked V38 substrate and causal restrictions;
- `elm_v38_predictive_evidence.py` — exact PRED-V allocation/member/value representation audited for state redundancy.

Key V38 facts frozen into this design:

- PBF is an engineering-positive but mechanism-unqualified result because Q10 failed;
- CORR2 remains the strongest clean deployable authority reference;
- mean after-D PRED-V slot count is 1460.5;
- mean target slots are 63.75;
- mean nontarget singleton slots are 1164.0 (~79.7% of all slots);
- mean incremental episodic payload is ~7.16 MB;
- storage remains unbounded in V38.

No V39 threshold may be changed after V39 scientific outcomes are observed.


## Locked implementation choices (before science)
- Ties: slot before trace, then oldest allocation index. Similarity uses float64.
- Slot access counts start at 1; promotion starts at source exact-hit count + 1.
  One access is counted on an exact hit, authoritative slot read, or confirmed write;
  at most one increment per event. Lookup/probes never touch counters.
- TRACE-EXACT2 promotes on the second exact presentation, including admission.
  Its single unique anchor has CORR2 support 1. A later distinct confirmed context
  can fill anchor 2 (as ANCHOR2-U); subsequent identity state is frozen.
  Similar traces never cause promotion in TRACE-EXACT2.
- SHUFFLED-PAIR follows TRACE-CORR online after all predictions are frozen. Source
  matching uses exact hit count, then nearest age quartile (rank of insertion age
  among resident traces); ties prefer the lower quartile. Empty hit class falls
  back to the entire eligible trace pool. Current context is excluded as source.
  If the current context is already an anchor, its owning slot is removed first;
  if it is a trace, that trace is removed. These extra removals are reported.
  No resident source means a missed promotion, never a fabricated trace. Any
  schedule mismatch makes Q11 invalid (fails qualification).
- Stress tail: 768 singleton and 128 repeated new uppercase/digit six-byte keys,
  alias-0 framing. First pass shuffles 896 rows; second pass shuffles 128 repeats.
  No optional fillers: exactly 1024 rows / 14336 bytes. Smoke scales to 24+8*2.
- Whole/nontarget bpb gates include the tail; pre-tail bpb is separately reported.
  Q6, Q8 and Q10 recurrence margins must hold on each paired seed. Other level
  gates use means and the explicitly specified per-seed floors.
- Q10/Q11 held-out endpoint is revision@4 alias 2. Q4/Q9 use all 64 current
  objects, both learned aliases, after clutter; alias-2 accuracy is separate.
- Q12 denominator: first subsequent exact acquisition presentation for every
  target context admitted as a trace (including intervening eviction/promotion).
  Q13 uses first exposure of E2, not later successful consolidation; includes
  failures to confirm or consolidate. Immediate E2 existence is separately gated.
- Completed seed checkpoints are atomic/checksummed. Interrupted seeds restart;
  no pickle or hidden live-state checkpoint is required. Source, protocol, plan,
  stream hashes and engineering P0 evidence are locked. No scientific results
  are used for implementation decisions.
