# V43 — Fast–Slow Authority
## When may mature similarity-based object memory override a fast bounded memory?

**Status:** preregistration / implementation specification only.

**Primary scientific question**

\[
oxed{
	ext{When both fast superposed memory and mature explicit object memory have an answer,
which one should control prediction?}
}
\]

**Primary candidate:** `LOCAL-BAYES-E2`  
**Primary specificity control:** `SHUFFLED-ALPHA`  
**Simpler controls:** `FIXED-50`, `GLOBAL-BAYES`  
**Physical learner:** V41 `SLOTS/READ-HYBRID` learning trajectory, frozen as the common substrate  
**Primary structural change from V42:** probationary traces are silent for prediction  
**Mature exact anchors:** hard slow authority  
**Only mature supported-similarity reads are arbitrated**

**Scientific panel:** 16 fresh seeds × loads {64, 128, 256, 512} = 64 physical trajectories.  
All read policies are scored on the same physical trajectory for each seed/load.

No parameter sweep. No new encoder. No new write rule. No new admission rule. No utility eviction. No phase-normalization branch.

---

# 1. Motivation

V42 decomposed the remaining READ-HYBRID penalty approximately as

\[
L_{m READ}-L_{m SUPER}
pprox
+0.0576_{m probationary\ exact}
+
0.0619_{m supported\ similarity}
-
0.0129_{m mature\ exact}
\quad{m bpb}.
\]

The implications are:

1. mature exact consolidated anchors are mildly beneficial;
2. hard probationary-trace authority is costly;
3. supported mature-slot similarity is necessary for durable held-out alias recall but costly on ordinary prediction;
4. the fast superposed layer already supplies strong one-shot behavior.

V43 therefore freezes the memory architecture and manipulates only mature similarity authority.

Proposed hierarchy:

\[
oxed{
egin{array}{c}
	ext{shared predictive base}\
\downarrow\
	ext{fast bounded superposed memory}\
\downarrow\
	ext{silent probationary traces}\
\downarrow\
	ext{mature explicit object slots}
\end{array}}
\]

---

# 2. Non-negotiable architectural firewall

Every seed/load has exactly one physical learner trajectory.

The physical learner contains:

- shared base;
- fast superposed store;
- slow explicit trace/slot learner;
- unchanged admission, consolidation, revision, eviction, and capacity machinery.

Read policy must never affect learning:

\[
oxed{
	ext{read policy}

otightarrow
	ext{base write, fast write, trace creation, slot promotion, revision, eviction}.
}
\]

The slow learner must use exactly the same internal fallback and admission logic as the locked V41 `SLOTS/READ-HYBRID` shared learning path.

**The scored fast/slow mixture is never fed back into the physical learner.**

All V43 policies are counterfactual readouts over identical physical state.

Any policy-specific physical-state divergence voids the experiment.

---

# 3. Exact route taxonomy

Every event is classified before the true byte is observed into exactly one route.

## A — mature exact anchor

Exact context is one of a mature slot's retained anchors.

Primary architecture:

\[
p=p_S.
\]

No V43 authority manipulation.

## T — probationary exact trace

Exact context exists only in the probationary trace tier.

Primary architecture:

\[
p=p_F.
\]

Trace state remains available for post-outcome corroboration/consolidation but is silent for prediction.

Historical `READ-HYBRID` may hard-read it as a baseline.

## G — mature supported similarity

No exact mature anchor, but a mature slot is a supported similarity candidate under the locked addressing rule.

This is the **only** route V43 manipulates.

Available experts:

\[
p_F,\qquad p_S.
\]

## F — fast fallback

No mature exact anchor and no supported mature similarity route.

\[
p=p_F.
\]

Route classification must be deterministic and policy-independent.

---

# 4. Fast and slow experts

`p_F` is the full pre-update distribution of the V41 SUPER fast tier.

It is not the base-only distribution.

The base distribution `p_B` is logged only for diagnostics.

For route G, `p_S` is the mature slot's ordinary pre-update probability distribution.

For route A, `p_S` is the exact mature-anchor prediction.

For route T, a counterfactual `p_T` may be logged but is not deployed by the proposed architecture.

All distributions come from the same physical pre-update state.

---

# 5. Specialist evidence

For each mature slot generation \(j\), maintain a virtual evidence ledger

\[
E_j(t)
=
\sum_{	au<t,\;	au\in G_j}
\log_2
rac{p_{S,j,	au}(y_	au)}
     {p_{F,	au}(y_	au)}.
\]

Include only prior route-G events for that slot generation.

Exclude:

- mature exact anchors;
- probationary trace events;
- unsupported nearest candidates;
- evaluator probes;
- events after eviction;
- events from a previous generation of a reused slot index.

Interpretation:

\[
E_j>0
\]

means slow similarity has historically outpredicted the fast tier for this object.

Convert evidence to authority:

\[
oxed{
w_j
=
rac{2^{E_j}}{1+2^{E_j}}
=
\sigma(E_j\ln2).
}
\]

On route G:

\[
oxed{
p=w_jp_S+(1-w_j)p_F.
}
\]

Use a numerically stable logistic implementation. Store `E_j` in float64. Do not clip the scientific ledger; only protect numerical evaluation of the logistic.

---

# 6. Evidence update order

For each event:

1. construct current context;
2. hash complete physical state;
3. compute `p_B` and `p_F`;
4. classify route A/T/G/F;
5. compute `p_S` or `p_T` as applicable;
6. score all V43 read policies;
7. record all pre-outcome features;
8. reveal `y`;
9. record policy losses;
10. update virtual evidence ledgers;
11. execute one common physical learner update;
12. process promotion/revision/eviction ledger lifecycle;
13. verify no policy changed physical state.

No policy may inspect `y` before step 8.

---

# 7. Slot-generation lifecycle

Never key evidence by a reusable array index.

Each mature memory needs a monotone unique `slot_uid` / `generation_uid`.

## Consolidation

Two candidate initializations:

### LOCAL-BAYES-0

\[
E_j=0.
\]

### LOCAL-BAYES-E2

At E2 promotion:

\[
oxed{
E_j^{(0)}
=
\log_2
rac{p_T(y_{E2})}
     {p_F(y_{E2})}.
}
\]

`p_T` and `p_F` must both be computed before the E2 outcome and before promotion.

This statistic may not alter whether promotion occurs.

## Revision

When a slot-wide semantic revision commits:

\[
E_j\leftarrow0.
\]

## Eviction

Destroy the generation ledger.

No tombstone.

## Re-consolidation

Allocate a fresh generation and fresh ledger.

No evidence inheritance.

---

# 8. Read policies

All policies share the same physical trajectory.

## SUPER

Fast everywhere.

## READ-HYBRID

Historical baseline:
- A: slow;
- T: trace;
- G: slow;
- F: fast.

## SILENT-HARD

Primary structural baseline:
- A: slow;
- T: fast;
- G: hard slow;
- F: fast.

## ANCHOR-ONLY

- A: slow;
- T: fast;
- G: fast;
- F: fast.

## FIXED-50

On G:

\[
p=.5p_S+.5p_F.
\]

Else use the primary architecture.

## GLOBAL-BAYES

Maintain

\[
E_G
=
\sum_{	au<t,	au\in G}
\log_2rac{p_S(y_	au)}{p_F(y_	au)}
\]

and use

\[
w_G=\sigma(E_G\ln2).
\]

## LOCAL-BAYES-0

Per-slot evidence initialized at zero.

## LOCAL-BAYES-E2

Per-slot evidence seeded from the E2 trace-vs-fast log likelihood ratio.

**Primary V43 candidate.**

If the exact pre-E2 quantities cannot be obtained without altering the physical learner, drop this arm before science. Do not approximate it post hoc.

## SHUFFLED-ALPHA

Primary nondeployable specificity control, computed offline from the saved authority-event table.

Take the actual pre-outcome LOCAL authority weights \(w_t\) on route G.

Within each seed × load, stratify by:

- support: `2`, `3–4`, `5+`;
- slot-age quartile;
- prior similarity-use count: `0–1`, `2–3`, `4+`.

Within each permutable stratum, deterministically permute the LOCAL weights across events:

\[
p_t=w_{\pi(t)}p_{S,t}+(1-w_{\pi(t)})p_{F,t}.
\]

This must preserve exactly within each permutable stratum:

- authority-weight multiset;
- total authority mass;
- number of events;
- fast and slow predictions;
- physical trajectory.

It destroys the alignment between earned authority and the object/event that earned it.

Singleton strata are reported and excluded from the decisive shuffled contrast.

At least 80% of route-G events used in the specificity test must lie in permutable strata.

## EVENT-ORACLE

Diagnostic only:

\[
\ell_O(t)=\min(\ell_F(t),\ell_S(t)).
\]

## SLOT-ORACLE

Diagnostic only: for each slot generation, retrospectively select whichever fixed expert, FAST or SLOW, had lower cumulative route-G log loss.

---

# 9. Oracle headroom

For route-G events:

\[
L_H=\sum\ell_S
\]

under hard similarity authority, and

\[
L_O=\sum\min(\ell_S,\ell_F).
\]

Available headroom:

\[
oxed{H=L_H-L_O.}
\]

For policy \(m\):

\[
oxed{
C_m
=
rac{L_H-L_m}{L_H-L_O}.
}
\]

This is the fraction of theoretically available arbitration gain captured.

If oracle headroom is negligible, do not interpret tiny policy differences as a mechanism discovery.

---

# 10. Authority-event instrumentation

For every A/T/G event, record compact pre-outcome features:

```text
seed
load
event_index
physical_time
route
slot_uid
generation_uid
support
slot_age
time_since_last_use
prior_similarity_use_count
similarity_top1
similarity_top2
similarity_margin
fast_top1
slow_top1
fast_slow_agree
fast_entropy
slow_entropy
fast_margin
slow_margin
js_divergence
local_E_before
local_w_before
revision_age
was_revised
```

After `y` is revealed, append:

```text
true_byte
pB_true
pF_true
pS_true
slow_minus_fast_loss
```

Do not dump every ordinary fast-only byte event.

Top-1 accuracy for preregistered policies should be accumulated online. Full 256-way distributions need not be written to disk unless needed for engineering audit.

Recommended storage: Parquet or NPZ.

---

# 11. Exploratory predictability diagnostic

On route G define

\[
Z_t=\mathbf 1\{\ell_S(t)<\ell_F(t)\}.
\]

Using only pre-outcome features, fit nondeployable diagnostics:

1. logistic regression;
2. spline/GAM if available;
3. gradient boosting as a flexible ceiling.

Validation must be **leave-one-seed-out**, never random event splitting.

Report:

- ROC-AUC;
- log loss;
- calibration;
- fraction of EVENT-ORACLE headroom captured.

This does not determine qualification.

Its purpose is to answer whether the fast-vs-slow mistake is predictably identifiable from causal state.

---

# 12. Engineering invariants

Run at least two non-science engineering seeds at loads 64 and 256.

## E0 — physical reproduction

With arbitration disabled, reproduce the locked V41/V42 physical trajectory:

- base hashes;
- fast-store hashes;
- slow-memory hashes;
- promotion/revision/eviction counts;
- terminal physical probes.

Any unexplained mismatch stops V43.

## E1 — policy-state firewall

Scoring all policies must leave the physical hash unchanged.

Policies may maintain only virtual arbitration ledgers.

## E2 — complete physical hashing

Hash must include:

- all learned arrays;
- traces/slots;
- allocation counters;
- slot-generation counters;
- RNG state that can influence future learning;
- admission/eviction counters.

This explicitly prevents the earlier bug where evaluator probes advanced hidden RNG/buffer state absent from the hash.

## E3 — probe purity

Evaluator probes may not alter:

- physical state;
- virtual ledgers;
- RNG positions used by learning.

## E4 — route exclusivity

Every event maps to exactly one of A/T/G/F.

Counts must sum to total events.

Exact mature anchors may not also count as G.

## E5 — causality

Assert that:

- all policy predictions finalize before `observe(y)`;
- ledger updates occur after `y`;
- E2 seed uses pre-E2 distributions;
- no future metadata enters current prediction.

## E6 — generation safety

Engineering fixture:

- birth;
- eviction;
- slot-index reuse;
- re-consolidation.

Verify zero evidence inheritance.

## E7 — revision reset

Engineering fixture with positive evidence followed by committed revision.

Verify exact reset to zero.

## E8 — shuffled exactness

For each permutable stratum:

- identical authority-weight multiset before/after permutation;
- identical total authority mass;
- substantial reassignment;
- no physical-state access beyond saved features.

---

# 13. Scientific panel

Fresh seeds:

```text
43101–43116
```

Loads:

\[
64,\;128,\;256,\;512.
\]

Total physical trajectories:

\[
16	imes4=64.
\]

Every trajectory scores every read policy.

Load 512 is a stress cell. If the underlying physical learner itself fails acquisition there, mark it `physical_capacity_invalid`; do not use it to judge arbitration.

No adaptive seed addition based on science outcomes.

---

# 14. Primary endpoints

1. whole-stream bpb;
2. route-G bpb;
3. durable old held-out accuracy;
4. oracle-headroom capture \(C_m\).

Secondary:

- target/nontarget bpb;
- anchor bpb vs fast;
- trace bpb vs fast;
- E1/E2 held-out accuracy;
- revision exact/known/held-out;
- route counts;
- authority mass;
- fast/slow agreement;
- evidence distributions;
- SLOT-ORACLE loss.

---

# 15. Statistical analysis

The independent scientific unit is the seed.

Authority events are not independent replicates.

## Planned paired contrasts

At each load:

1. LOCAL-E2 − SILENT-HARD
2. LOCAL-E2 − FIXED-50
3. LOCAL-E2 − GLOBAL-BAYES
4. LOCAL-E2 − SHUFFLED-ALPHA
5. LOCAL-E2 − SUPER

Use paired seed-level effects and intervals.

## Hierarchical load analysis

Fit seed-blocked/hierarchical models across valid loads for:

- whole bpb;
- G-route bpb;
- durable held-out success.

Use \(\log_2(	ext{load})\) as the ordered pressure axis.

Estimate:

- load-specific means;
- policy × load interaction;
- held-out \(L_{50}\) where identifiable;
- probability/order of policy dominance over the observed load range.

## Multiplicity

The decisive mechanistic family is:

- LOCAL vs SILENT-HARD;
- LOCAL vs GLOBAL;
- LOCAL vs FIXED-50;
- LOCAL vs SHUFFLED.

If frequentist p-values are reported, use Holm adjustment across these four planned contrasts.

Effect sizes and intervals remain primary.

---

# 16. Qualification gates

## Q0 — physical validity

E0–E8 pass.

Every science cell used in inference has:

- no physical-state divergence;
- finite probabilities/losses;
- physical learner acquisition ≥90% on intended current mature memories.

## Q1 — arbitration headroom

At loads 128 and 256:

- at least 500 G-route events per seed;
- hard-G loss exceeds EVENT-ORACLE by ≥0.20 bits/G-event on average.

Otherwise arbitration is not a meaningful bottleneck.

## Q2 — mature anchors remain safe

Across 128 and 256:

\[
\ell_A^{slow}-\ell_A^{fast}\le0
\]

on average, with no seed mean disadvantage worse than +0.05 bits/A-event.

If this fails, unconditional anchor authority must be reopened.

## Q3 — silent traces are viable

Relative to READ-HYBRID, SILENT-HARD must:

- improve or match whole bpb;
- preserve E1 held-out within 2 pp;
- preserve physical consolidation counts exactly.

## Q4 — primary prediction gain

At load 256:

\[
	ext{bpb}_{LOCAL-E2}
-
	ext{bpb}_{SILENT-HARD}
\le-0.03
\]

with paired 95% upper bound below zero.

## Q5 — durable alias noninferiority

At load 256:

\[
	ext{heldout}_{LOCAL-E2}
\ge
	ext{heldout}_{SILENT-HARD}-0.02.
\]

Paired lower bound for LOCAL − SILENT-HARD must exceed −0.05.

## Q6 — specificity veto

At load 256 on route G:

\[
	ext{bpb}_{LOCAL-E2}
-
	ext{bpb}_{SHUFFLED-ALPHA}
\le-0.05
\]

with paired 95% upper bound below zero.

At least 80% of compared G events must belong to permutable strata.

If Q6 fails, object/event-specific learned authority is not established.

## Q7 — complexity must earn itself

If FIXED-50 or GLOBAL-BAYES is within all of:

- 0.01 whole bpb;
- 0.02 G-route bpb;
- 2 pp durable held-out accuracy

of LOCAL-E2 across loads 128 and 256, prefer the simpler policy.

## Q8 — oracle capture

At load 256:

\[
C_{LOCAL-E2}\ge0.50.
\]

## Q9 — end-to-end target

At load 256:

\[
	ext{whole bpb}_{LOCAL-E2}
\le
	ext{whole bpb}_{SUPER}+0.05
\]

while Q5 holds.

If Q4/Q6/Q8 pass but Q9 fails, the mechanism is supported but the architecture gap remains.

## Q10 — scaling stability

From load 128 to 256:

- LOCAL advantage over SILENT-HARD must not reverse sign;
- durable held-out degradation must not materially exceed SILENT-HARD.

Load 512 remains descriptive unless the physical substrate stays valid.

---

# 17. Decision logic

## Outcome A — local specialist arbitration established

Q0–Q6 and Q8 pass, LOCAL beats SHUFFLED, and no simpler rule matches under Q7.

Conclusion:

> mature similarity authority is genuinely object/event specific and can be earned from the slot's own past predictive advantage over fast memory.

## Outcome B — global arbitration is enough

GLOBAL matches LOCAL.

Use one global scalar.

## Outcome C — fixed mixture is enough

FIXED-50 matches adaptive policies.

Use FIXED-50.

## Outcome D — similarity should remain silent

ANCHOR-ONLY/SUPER dominates and no adaptive policy preserves durable held-out transfer economically.

Revisit identity/addressing rather than adding more gates.

## Outcome E — LOCAL looks best but fails shuffled specificity

Adaptive authority magnitude helps, but correct object/event assignment is not established.

Do not carry local evidence forward.

## Outcome F — little oracle headroom

Q1 fails.

Stop read-policy search.

---

# 18. Forbidden changes

V43 must not modify:

- base architecture;
- superposed write rule;
- slow write rule;
- trace admission;
- two-context consolidation;
- trace/slot capacities;
- revision;
- eviction;
- lag representation;
- similarity threshold;
- output temperature;
- stream semantics.

Do not add:

- PBF-vs-base authority;
- phase normalization;
- new plasticity;
- replay;
- learned router;
- utility eviction;
- fast-memory unloading;
- revision-confidence logic.

---

# 19. Required outputs

```text
SUMMARY.md
REPORT.json
MANIFEST.json
PROTOCOL.md
TERMINAL.csv
LOAD_CURVES.csv
ROUTE_SUMMARY.csv
AUTHORITY_EVENTS.parquet   # or .npz
AUTHORITY_BY_SUPPORT.csv
AUTHORITY_BY_AGE.csv
POLICY_CONTRASTS.csv
ORACLE_HEADROOM.csv
SLOT_ORACLE.csv
SHUFFLED_ALPHA_AUDIT.json
STATE_HASH_AUDIT.json
REVISION_LEDGER_AUDIT.json
```

Optional:

```text
PREDICTABILITY_DIAGNOSTIC.csv
PREDICTABILITY_FEATURES.csv
```

Do not dump full ordinary per-byte logs.

---

# 20. Runner contract

One self-contained executable file.

Suggested interface:

```bash
python3 elm_v43_fast_slow_authority.py --self-test
python3 elm_v43_fast_slow_authority.py --engineering
python3 elm_v43_fast_slow_authority.py --out-dir v43_results
```

One seed/load trajectory must score all policies in one pass.

Minimal pseudocode:

```python
for event in stream:
    physical_hash_before = physical_state_hash()

    pB = base.predict(context)
    pF = fast.predict(context)

    route = slow.classify_route(context)      # A/T/G/F
    pS, pT, slot_uid = slow.counterfactual_distributions(context, route)

    predictions = score_all_read_policies(
        route=route,
        pF=pF,
        pS=pS,
        pT=pT,
        virtual_ledgers=ledgers,
        slot_uid=slot_uid,
    )

    assert physical_state_hash() == physical_hash_before

    y = event.next_byte

    record_losses(predictions, y)
    update_virtual_ledgers_after_outcome(route, slot_uid, pS, pF, y)

    transition = physical_learner.observe(
        context=context,
        y=y,
        internal_fallback=LOCKED_V41_FALLBACK,
    )

    apply_ledger_lifecycle(
        transition=transition,
        pre_event_pT=pT,
        pre_event_pF=pF,
        y=y,
    )

    assert no_policy_specific_physical_state()
```

`LOCKED_V41_FALLBACK` must be the exact value/path used by the shared V41 SLOTS/READ-HYBRID physical learner. Do not infer or replace it from prose.

---

# 21. Interpretation boundary

A successful V43 would establish only:

\[
oxed{
	ext{fast distributed memory and slow durable object memory can coexist with a causal,
low-state arbitration rule that preserves durable alias recall while recovering much of
the fast tier's prediction quality.}
}
\]

It would not establish:

- optimal forgetting;
- natural-language persistent learning;
- noise-vs-revision discrimination;
- biological equivalence;
- universal optimality of Bayesian specialist arbitration.

Only after V43 should the project return to:

1. marginal-utility survival under true scarcity;
2. fast-memory unloading after consolidation;
3. noise versus genuine revision;
4. real byte-corpus validation.
