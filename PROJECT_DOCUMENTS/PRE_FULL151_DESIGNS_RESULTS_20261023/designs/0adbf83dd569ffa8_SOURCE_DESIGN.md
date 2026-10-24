# MiniFly FB1 v3 — Fixed-Address Banked-Storage Resource Curve
## Second hostile-audit revision

**Date:** 2026-09-21  
**Status:** implementation-ready design **only after all qualification gates pass**  
**Parent:** `FULL_151 + exact frozen B`  
**Historical bridge:** completed post-V5 `S_LOCAL / S_FIXED` experiment  
**Science roster:** `B / FB2 / FB4 / FB8`  
**Main sample:** 96 canonical + 128 K8 independent worlds = 896 science trajectories

---

# 0. Scientific question

FB1 v3 asks exactly:

\[
\boxed{
\textbf{Conditional on one frozen 8-way content-address dictionary,
what competence/risk/resource curve is produced by 2, 4, or 8 isolated alpha stores?}
}
\]

It does **not** ask:

- whether isolation rather than raw extra state is the unique cause;
- whether two/four/eight banks are globally optimal;
- whether the router dictionary is minimal;
- whether fixed banks are literal fly anatomy;
- whether other learning mechanisms are unnecessary;
- whether a minimal persistent-learning core has already been found.

The formal experiment ends after the frozen 2/4/8 panel.

If fixed banks win, the capacity-versus-isolation causal question is deferred to the separately defined future gate `FB2-EQ`.

---

# 1. Current evidence motivating FB1

The completed post-V5 study established:

## 1.1 Adaptive banked storage versus B

On 152 fresh K8 worlds:

\[
S_{\rm LOCAL}-B
=
+5.688\text{ pp}
\]

for first-use return, with simultaneous interval:

\[
[3.681,\ 7.696]\text{ pp}.
\]

`S_LOCAL` also improved K8 current usefulness, acquisition, anomaly recovery, acquisition +24h, revision +24h, and usefulness +7d.

## 1.2 Fixed router versus adaptive router

On the predeclared 48-world K8 mechanism-control subset:

```text
B        ≈ 49.002% first-use return
S_LOCAL  ≈ 54.297%
S_FIXED  ≈ 73.351%
```

and:

\[
S_{\rm LOCAL}-S_{\rm FIXED}
=
-19.054\text{ pp},
\]

with corrected interval:

\[
[-22.295,\ -15.812]\text{ pp}.
\]

Thus the tested center-adaptation rule is not retained in FB1.

This does **not** imply that every dynamic write gate is harmful.

## 1.3 State cost

Historical allocated state:

```text
B        =   165,688 bytes
S_FIXED  = 1,159,744 bytes
```

A fixed 8-bank win is therefore an architecture signal, not yet a minimal-core result.

---

# 2. Updated mathematical interpretation

The current theory separates write address from read address.

For storage level \(K\):

\[
\pi_t^{(K)}(x)
\]

denotes the physical write address used at event \(t\), while:

\[
\rho_T^{(K)}(x)
\]

denotes the physical read address at checkpoint \(T\).

For one-hot routing:

\[
\pi_t^{(K)}(x),\rho_T^{(K)}(x)\in\{e_0,\ldots,e_{K-1}\}.
\]

Define:

\[
A_{\rm WR}^{(K)}(x;t,T)
=
\rho_T^{(K)}(x)^\top\pi_t^{(K)}(x).
\]

In FB2/FB4/FB8, the router is fixed and the current task reuses the same encoded cue, so:

\[
\boxed{
A_{\rm WR}=1
}
\]

for an exact same cue code by construction.

This is a **qualification invariant**, not a new behavioral discovery.

The historical adaptive-router replay is the only place in FB1 where write/read drift is an empirical diagnostic.

---

# 3. History-conditioned read–write kernel remains background theory

For one realized history \(H\):

\[
\delta z_{t+1}
=
A_t^H\delta z_t+B_t^H\delta u_t,
\]

\[
\Phi_{T\leftarrow t+1}^{H}
=
A_{T-1}^{H}\cdots A_{t+1}^{H}.
\]

For write direction \(w_\mu(t,H)\) and query sensitivity \(r_\nu(T,H)\):

\[
K_{\nu\mu}^{H}(T,t)
=
r_\nu(T,H)^\top
\Phi_{T\leftarrow t+1}^{H}
w_\mu(t,H).
\]

FB1 does not construct a dense Jacobian and does not assume factorization into an address term.

A10 already establishes important source restrictions:

- alpha intervention does not alter the gamma-teaching block;
- adaptation follows the same exogenous input stream;
- B's important closed-loop response includes native allocation and alpha feedback.

Therefore FB1 must not cite “shared gamma/adaptation” as evidence of unmeasured cross-bank coupling.

The relevant statement is narrower:

> direct alpha storage is partitioned, while each selected bank still participates in the native B/allocation/alpha-feedback loop when it is active.

---

# 4. Current task has no held-out rendering variable

The current MiniFly runner encodes each world cue once and reuses that code.

Therefore FB1 contains no experimental category called:

```text
same identity / different rendering
```

and no such samples are fabricated.

For the fixed router:

```text
same exact cue write→read consistency = structural invariant
```

and is used only for qualification.

The science diagnostics instead distinguish:

1. **same cue identity**;
2. **the other cue in the same association pair**;
3. **foreign cue identities from other associations**.

The two cues inside one pair are never silently classified as “unrelated.”

---

# 5. Frozen 8-way address dictionary

For router seed \(s\), create:

\[
c_0^{(s)},\ldots,c_7^{(s)}
\]

using the already-qualified source rule:

```python
unit(default_rng(seed).normal(size=(8, 5177)))
```

with the exact historical normalization operation.

For current static KC activity \(x\):

\[
\hat x
=
\frac{x}{\max(\lVert x\rVert_2,\epsilon)}.
\]

Define route class:

\[
a_s(x)
=
\arg\max_{j=0,\ldots,7}
(c_j^{(s)})^\top\hat x.
\]

Use the source's deterministic first-index tie rule.

Router centers never update.

The router cannot access:

- punishment/outcome;
- truth;
- fact/pair ID;
- cohort;
- context;
- checkpoint;
- relevance;
- score.

---

# 6. Four fixed router dictionaries

Use:

```text
R0 = 2026092101
R1 = 2026092111
R2 = 2026092121
R3 = 2026092131
```

`R0` is the historical fixed-router seed.

`R1–R3` are predeclared alternative fixed dictionaries.

These are **four fixed architecture strata**, not random draws supporting a universal random-router claim.

All FB arms within one world use the same router dictionary.

Router-seed namespace is separate from world/donor-seed namespace.

---

# 7. Balanced nested 8→4→2 storage hierarchy

The hierarchy is frozen from center geometry only.

No science outcome participates.

## 7.1 Eight to four physical banks

Enumerate all 105 perfect matchings of the eight address classes.

Select the matching maximizing:

\[
\sum_{(i,j)\in M}
c_i^\top c_j.
\]

Exact ties at stored float64 precision are broken lexicographically using canonical sorted pair tuples.

This gives four banks containing exactly two address classes each.

## 7.2 Four to two physical banks

For each FB4 pair \(A\), define:

\[
m_A
=
\frac{\sum_{j\in A}c_j}
{\left\|\sum_{j\in A}c_j\right\|_2}.
\]

Enumerate the three perfect matchings of the four pair centroids.

Choose the matching maximizing total centroid cosine, with the same lexicographic tie rule.

This gives two banks containing exactly four address classes each.

## 7.3 Frozen partitions

### R0

```text
FB4: {0,4} {1,7} {2,6} {3,5}
FB2: {0,3,4,5} {1,2,6,7}
```

### R1

```text
FB4: {0,6} {1,4} {2,5} {3,7}
FB2: {0,2,5,6} {1,3,4,7}
```

### R2

```text
FB4: {0,3} {1,7} {2,6} {4,5}
FB2: {0,2,3,6} {1,4,5,7}
```

### R3

```text
FB4: {0,2} {1,7} {3,6} {4,5}
FB2: {0,1,2,7} {3,4,5,6}
```

Freeze:

```text
ROUTER_STORAGE_HIERARCHY.json
```

including centers, pair scores, tie results, and final mappings.

---

# 8. Address-class balance is not event-load balance

The hierarchy balances the number of address classes per bank.

It does not guarantee equal traffic.

Historical R0 event traffic under the new FB2 grouping is approximately:

```text
36.49% / 63.51%
```

for the two physical banks.

Therefore every science world reports actual:

- address-class occupancy;
- physical-bank write occupancy;
- physical-bank read occupancy;
- physical-bank write mass.

A low-K failure is interpreted only for this frozen hierarchy.

It is not a theorem that \(K\) physical banks are insufficient under all possible partitions.

---

# 9. Implementation interface: address count and bank count are separate

The old wrapper's:

```text
configure(banks=K)
```

is forbidden as the FB2/FB4 implementation shortcut because it changes both:

- number of router centers;
- number of physical stores.

FB1 v3 requires:

```text
n_address_classes = 8            # fixed
n_storage_banks   = K            # K ∈ {1,2,4,8}
centers.shape     = (8, 5177)    # fixed
route_class(x)    -> 0..7
route_to_bank[K]  -> 0..K-1
```

Both write and read paths call the same frozen:

```text
route_to_bank[K][route_class(x)]
```

mapping.

Address-class statistics and physical-bank statistics are logged separately.

---

# 10. State semantics

## 10.1 Shared native state

Exactly one shared copy of all state shared by the qualified `S_FIXED` implementation, including the inherited native non-alpha state.

## 10.2 Banked alpha state

For FB\(K\), maintain \(K\) independent copies of:

- alpha-fast;
- slow.

## 10.3 Fixed router state

All FB2/FB4/FB8 arms retain the same eight router centers for a given router-seed stratum.

FB1 reduces banked alpha storage only.

It does not yet compress the router dictionary.

---

# 11. Exact event transition

The implementation mechanically inherits the qualified fixed-bank wrapper.

For one event:

1. compute route class from the current static cue representation;
2. map route class to physical bank;
3. load the **pre-event** selected bank alpha-fast/slow state into the native scratch locations;
4. run the exact native B event transition **once**;
5. native allocation, native teaching write, B correction, native decay, and shared-state update occur once in source order;
6. save the native post-event alpha-fast/slow state back to the selected bank;
7. apply exactly one passive native decay for the same elapsed interval to each unselected bank;
8. never update router centers.

Forbidden implementation:

```text
pre-decay selected bank
→ call native event
```

because that double-applies the selected-bank active transition.

Forbidden implementation:

```text
call native event
→ separately update shared native state again
```

because that duplicates the source transition.

---

# 12. Event classes that must be qualified separately

Qualification covers:

```text
active + plastic
active + nonplastic
idle/rest
long-rest/dormancy
read-only probe
```

For each, compare the new wrapper against the exact qualified source behavior.

Source semantics override prose shorthand.

---

# 13. Read semantics

For an ordinary query:

1. compute current static cue code;
2. compute frozen route class;
3. map to physical bank;
4. expose that bank's alpha fast/slow state to the unchanged frozen scorer/reader;
5. score;
6. restore live state.

A read-only probe must not:

- advance time by itself;
- update a bank;
- update shared state;
- update router state;
- consume learner RNG.

All biological elapsed time is represented by explicit native/passive transitions in the world schedule.

---

# 14. Science arms

```text
B
FB2
FB4
FB8
```

## B

Exact incumbent.

## FB2

Eight address classes share two alpha banks.

## FB4

Eight address classes share four alpha banks.

## FB8

Eight address classes use eight independent alpha banks.

For router seed R0, FB8 must reproduce the qualified historical fixed-router semantics.

---

# 15. Qualification-only FB1 identity

`FB1_IDENTITY` maps all eight route classes to one physical alpha bank.

It is not a science arm.

The identity test must use **nonzero pre-event states**.

Required procedure:

1. run B to a predeclared nonzero checkpoint;
2. clone the exact B state;
3. initialize the one-bank wrapper from that cloned state without clearing it;
4. continue B and FB1 on identical subsequent events;
5. compare every tested transition and probe.

Test:

- active/plastic;
- active/nonplastic;
- idle;
- long rest;
- subsequent read-only probes.

Required equality objects include:

- alpha fast/slow;
- shared state;
- native write summaries;
- predictions;
- endpoint numerators/denominators;
- final digest.

A wrapper that achieves identity only from a freshly zeroed state fails qualification.

---

# 16. Historical FB8 bridge

Use a replay bridge because compact historical output alone is not enough for detailed state equivalence.

Freeze one old K8 control world:

```text
KQ
```

before replay.

Procedure:

1. replay locked historical `S_FIXED` on KQ;
2. verify its compact digest/metrics against the old archive;
3. run new FB8/R0 on the same KQ;
4. compare route sequence, states, writes, predictions, and endpoints.

The old world is technical qualification only.

It is never reused as fresh science.

Object-specific tolerances are frozen before replay.

---

# 17. Stable-retention window is identified by checkpoint labels, not global seconds

For each world and panel, locate:

```text
from_stage = block7_day
to_stage   = final_7day
```

from the actual event/checkpoint structure.

For every world, qualification verifies:

\[
t_{\rm final7d}
-
t_{\rm block7d}
=
604800\text{ s}
\]

and:

```text
fact_specific_teaching_events_in_window = 0
```

Record both absolute times.

Expected historical examples:

```text
canonical:
1693440 → 2298240 s

K8:
5685120 → 6289920 s
```

The absolute times are panel/world metadata, not universal constants.

---

# 18. Stable-start correctness coverage

Stable retention is sparse, so FB1 separates:

1. whether the candidate and B know material at the start of the passive window;
2. whether mutually-known material is later lost.

For fixed eligible population \(E_w\), define:

\[
C_{m,w}^{\rm start}
=
\frac{
\#\{j\in E_w:m_j\text{ correct at block7_day}\}
}{
|E_w|
}.
\]

Formal dense guard:

\[
\boxed{
D_{c,w}^{\rm start}
=
C_{c,w}^{\rm start}
-
C_{B,w}^{\rm start}.
}
\]

Higher is better.

This prevents low stable-loss mass from being interpreted favorably merely because a candidate knew less material at the start.

---

# 19. Sparse common-known stable-loss mass

For candidate \(c\):

\[
g_j
=
1\{B_j\text{ correct at block7_day}\}
1\{c_j\text{ correct at block7_day}\}.
\]

Define:

\[
L_{m,w}
=
\frac1{|E_w|}
\sum_{j\in E_w}
g_j
1\{m_j\text{ wrong at final_7day}\}.
\]

Then:

\[
\boxed{
D_{c,w}^{\rm stable}
=
L_{c,w}-L_{B,w}.
}
\]

Positive means more loss mass under the candidate.

This estimand is:

> common-known stable-loss mass per total eligible fact.

It is **not** the conditional forgetting probability among only common-known memories.

Always publish:

```text
eligible_n
B_correct_start_n
candidate_correct_start_n
common_known_n
B_common_known_losses_n
candidate_common_known_losses_n
D_stable
```

If a world has:

```text
common_known_n = 0
```

then `D_stable=0` is algebraically defined as zero loss mass, but:

```text
stable_support = 0
```

and must not be described as evidence of successful retention.

Also publish the descriptive conditional loss rate among common-known facts whenever its denominator is nonzero.

---

# 20. Sparse stable-loss risk uses a separate inference family

Stable-loss is removed from the dense guard family.

It gets its own distribution-free risk family.

For each FB arm versus B:

```text
canonical D_stable
K8 D_stable
```

giving:

\[
M_R=6.
\]

Allocate:

\[
\alpha_R=0.01.
\]

Because:

\[
D_{c,w}^{\rm stable}\in[-1,1],
\]

use a simultaneous Hoeffding risk envelope:

\[
\boxed{
\epsilon_n
=
\sqrt{
\frac{
2\log(2M_R/\alpha_R)
}{n}
}.
}
\]

For the frozen sample sizes:

```text
canonical n=96  → half-width ≈ 38.43 pp
K8 n=128        → half-width ≈ 33.28 pp
```

These intervals are intentionally conservative.

Because the science roster is exactly balanced over the fixed design strata, the all-world mean used here equals the corresponding equal-stratum mean. The Hoeffding envelope is therefore applied to the same balanced target estimand, without estimating within-stratum variances.

They are not advertised as high-precision retention certification.

A world with `common_known_n=0` remains in the frozen roster and contributes zero to the **loss-mass** estimand because no mutually-known fact can contribute loss mass there; its support flag is nevertheless zero and it is never described as evidence of retention.

If total common-known support across a panel is zero:

```text
STABLE_RISK_UNSUPPORTED
```

rather than “zero risk.”

The ordinary paired-t stable-loss interval may still be shown descriptively, but it cannot replace the frozen Hoeffding family after results are visible.

---

# 21. Dense formal guard family

Stable-loss is replaced by stable-start correctness coverage in the dense family.

## Canonical dense guards

1. current usefulness;
2. acquisition;
3. anomaly recovery;
4. acquisition/new learning +24h;
5. revision +24h;
6. usefulness +7d;
7. first-use return;
8. stable-start correctness coverage.

## K8 dense guards

1. current usefulness;
2. acquisition;
3. anomaly recovery;
4. acquisition/new learning +24h;
5. revision +24h;
6. usefulness +7d;
7. stable-start correctness coverage.

Thus:

\[
8+7=15
\]

dense guards per candidate.

For:

```text
FB2-B
FB4-B
FB8-B
```

there are:

\[
M_G=45.
\]

Allocate:

\[
\alpha_G=0.02.
\]

Use the predeclared stratified paired-t procedure described below.

No historical-SD equivalence margin is introduced.

---

# 22. Primary family

Primary endpoint:

```text
K8_FIRST_USE_RETURN
```

before any re-teaching/reactivation exposure.

Contrasts:

```text
FB2 - B
FB4 - B
FB8 - B
FB4 - FB2
FB8 - FB4
```

Thus:

\[
M_P=5.
\]

Allocate:

\[
\alpha_P=0.02.
\]

Total formal family budget:

\[
\boxed{
\alpha_P+\alpha_G+\alpha_R
=
0.02+0.02+0.01
=
0.05.
}
\]

The frozen formal comparison counts are:

```text
primary resource-curve comparisons = 5
dense guard comparisons            = 45
sparse stable-risk comparisons     = 6
total formal intervals             = 56
```

These three families answer different questions and are not collapsed into one pass/fail family.

---

# 23. Four independent result dimensions

Do not collapse the round into one ordinal label.

Every FB arm reports:

## 23.1 Execution status

```text
EXECUTION_COMPLETE
EXECUTION_INCOMPLETE
```

## 23.2 Primary status

```text
PRIMARY_GAIN_CONFIRMED
PRIMARY_GAIN_NOT_CONFIRMED
PRIMARY_INFERENCE_INCOMPLETE
```

## 23.3 Dense protection status

```text
NO_CONFIRMED_DENSE_GUARD_HARM
CONFIRMED_DENSE_GUARD_HARM
DENSE_GUARD_INFERENCE_INCOMPLETE
```

## 23.4 Stable-retention risk status

For the distribution-free stable-loss interval:

```text
STABLE_RISK_BENEFIT_CONFIRMED
STABLE_RISK_UNRESOLVED
STABLE_RISK_HARM_CONFIRMED
STABLE_RISK_UNSUPPORTED
```

Definitions:

- `BENEFIT_CONFIRMED`: the simultaneous upper bound is below zero;
- `HARM_CONFIRMED`: the simultaneous lower bound is above zero;
- `UNRESOLVED`: the interval contains both the no-harm boundary and at least one harmful value;
- `UNSUPPORTED`: the panel contains no common-known support at all.

There is deliberately no status called “safe” or “no confirmed harm” for this sparse endpoint.

The distribution-free interval may remain too wide to be practically informative.

---

# 24. Research-candidate rule

Define:

```text
ADVANCE_AS_RESOURCE_CANDIDATE
```

if and only if:

1. execution is complete;
2. primary gain versus B is confirmed;
3. all dense guard intervals are estimable;
4. no dense guard has confirmed harm.

Stable-loss risk does **not** block this research-candidate label.

Instead it travels as a separate status.

Therefore a valid final combination is:

```text
ADVANCE_AS_RESOURCE_CANDIDATE
+
STABLE_RISK_UNRESOLVED
```

This means:

> the architecture deserves further causal/resource study, but old-memory retention safety is not established.

It is not a deployment/safety certificate.

---

# 25. Smallest beneficial resource point

Among arms satisfying:

```text
ADVANCE_AS_RESOURCE_CANDIDATE
```

define:

```text
SMALLEST_ADVANCING_TESTED_ARM
```

as the smallest \(K\in\{2,4,8\}\).

This is a project-management label.

It does not prove:

- larger K is unnecessary;
- K is globally minimal;
- untested partitions with fewer resources cannot work.

Adjacent-bank contrasts remain separately reported.

---

# 26. Address diagnostics use only task-defined categories

For a scored cue \(q\) at its actual checkpoint \(T_q\), define prior eligible plastic cue-presentation events:

\[
\mathcal W(q,T_q).
\]

Classify prior write events into:

## Same cue

same logical cue identity as \(q\).

## Pair opponent

the other cue belonging to the same paired association as \(q\).

## Foreign

distinct logical cue identity and not the pair opponent.

There is no `different rendering` category in FB1.

---

# 27. Pair-opponent bank overlap

For each scored association pair:

\[
(x^{(1)},x^{(2)}),
\]

define:

\[
O_{\rm pair}^{(K)}
=
1\{
\pi_K(a(x^{(1)}))
=
\pi_K(a(x^{(2)}))
\}.
\]

Average uniformly over scored association pairs within world/checkpoint.

Do not call pair opponents “foreign.”

---

# 28. Event-uniform foreign overlap

For query cue \(q\), let:

\[
\mathcal W_F(q,T_q)
\]

be prior foreign plastic presentation events.

Define:

\[
O_{{\rm foreign,event}}^{(K)}(q,T_q)
=
\frac{
\#\{t\in\mathcal W_F:
\text{bank}(t)=\text{bank}(q)\}
}{
|\mathcal W_F|
}.
\]

Repeated teaching presentations count as repeated write events.

Each scored query cue receives equal weight when forming the world-level checkpoint summary.

If the foreign-write denominator is zero, record numerator/denominator and mark the query unsupported.

---

# 29. Correct stored-slow mass accounting

The current B wrapper's source `rawslow` already represents the B-corrected pre-decay slow write:

\[
\boxed{
rawslow=q_s-c.
}
\]

Do **not** subtract \(c\) again.

For one selected-bank event with native slow decay factor \(d_s\), define the immediate event-end stored slow contribution:

\[
\boxed{
m_t
=
\left\|
d_s\cdot rawslow_t
\right\|_1.
}
\]

Qualification must verify numerically:

\[
s_{\rm post}
-
d_s s_{\rm pre}
=
d_s\cdot rawslow
\]

at the source operation order/tolerance.

This is a write-accounting identity.

It is not a causal effect on future behavior.

---

# 30. Online retained-mass accumulator

To avoid per-event storage, evaluator-only diagnostics maintain streaming sufficient statistics.

For each bank \(b\):

\[
M_b^{t+1}
=
d_{s,t}M_b^t
+
1\{b=b_t\}m_t.
\]

For each logical cue \(j\) and bank \(b\), maintain:

\[
M_{j,b}^{t+1}
=
d_{s,t}M_{j,b}^t
+
1\{j=j_t,b=b_t\}m_t.
\]

These are sums of absolute passive-retained write contributions.

They are not the signed final slow state and not the full closed-loop causal effect.

At query \(q\), use total-bank and per-cue accumulators to subtract:

- same-cue contribution;
- pair-opponent contribution;

from the foreign pool.

---

# 31. Retained-mass-weighted foreign overlap

For query \(q\) with physical bank \(b_q\):

\[
F_{b_q}
=
M_{b_q}
-
M_{q,b_q}
-
M_{{\rm pair}(q),b_q}.
\]

Total foreign retained mass:

\[
F_{\rm all}
=
\sum_b M_b
-
\sum_b M_{q,b}
-
\sum_b M_{{\rm pair}(q),b}.
\]

Define:

\[
O_{{\rm foreign,mass}}^{(K)}
=
\frac{F_{b_q}}{F_{\rm all}}
\]

when \(F_{\rm all}>0\).

This is the primary mass-weighted routing diagnostic.

It is computed online.

No raw per-event ledger is written to disk.

---

# 32. Lifetime heterogeneity without per-event age logs

Do not persist event-level age records.

Maintain evaluator-only per logical cue:

```text
first_teaching_time
last_teaching_time
write_count
physical_bank
```

At each predeclared query/checkpoint, compute per-bank distributions of:

```text
time since first teaching
time since last teaching
number of prior teaching presentations
```

using fixed age bins:

```text
[0,1h)
[1h,6h)
[6h,24h)
[1d,3d)
[3d,7d)
[7d,∞)
```

Report counts and retained-mass totals by bin.

No post-hoc quantile definition is introduced.

---

# 33. Online diagnostic persistence schema

The experiment processes approximately millions of plastic cue presentations.

Per-event diagnostic JSON/CSV persistence is forbidden.

Freeze:

```text
STREAM_DIAGNOSTIC_SCHEMA.json
```

before science.

Permanent evidence is aggregated by:

```text
world
arm
router seed
rotation, when applicable
checkpoint/query class
route class
physical bank
age bin
```

Store:

- counts;
- numerators/denominators;
- retained-mass sums;
- tail transitions;
- endpoint summaries;
- digests.

World/cue logical definitions are stored once in the world file/manifest.

No raw per-event trace is retained except the tiny qualification fixtures explicitly named before science.

---

# 34. Science sample

Fresh science worlds exclude all previous science namespaces.

## Canonical

\[
n=96.
\]

Balance:

```text
4 router dictionaries
× 24 worlds
```

## K8

\[
n=128.
\]

Joint balance:

```text
4 router dictionaries
× 8 fixed rotations
× 4 independent worlds
```

Each independent K8 world uses eight non-reused donor worlds.

Required base worlds:

\[
\boxed{
96+128\times8=1120.
}
\]

Main science trajectories:

\[
\boxed{
4\times(96+128)=896.
}
\]

No sample-size expansion after results are visible.

---

# 35. Historical exclusion must scan all project roots

Before world generation, freeze:

```text
HISTORICAL_SOURCE_ROOTS.json
```

including at least:

```text
/Users/pencilbard/Downloads/second_try_copy
/Users/pencilbard/Documents/second_try/CORE_RECOVERY_A10/v2
```

plus explicit manifests/locks for:

```text
A8
A9
B/K/R V5
A10 Phase A/B
Post-V5 S_LOCAL/S_FIXED
evolution / validation / final-confirmation archives
```

Build one exclusion union containing actual historical:

```text
donor seeds
world seeds
pattern keys
cue/raw-pattern hashes
composite world hashes
```

Validate zero intersection against the new 1,120 base worlds.

Do not rely on:

- project-name change;
- directory-local max index;
- filename uniqueness.

Router dictionary seeds use a separate namespace and are never treated as world seeds.

---

# 36. K8 inferential strata include rotation

K8 is balanced over both:

- router dictionary \(s\in\{0,1,2,3\}\);
- fixed rotation \(r\in\{0,\ldots,7\}\).

The inferential strata are therefore **router-seed × rotation** cells, not router seed alone.

Define stratum:

\[
h=(s,r),
\]

giving:

\[
H=32
\]

fixed K8 strata, each with:

\[
n_h=4
\]

independent worlds.

For contrast \(c\) and endpoint \(e\):

\[
d_{c,e,h,i}
=
Y_{c,e,h,i}
-
Y_{{\rm ref},e,h,i}.
\]

Estimate:

\[
\boxed{
\hat\mu
=
\frac1{32}\sum_{h=1}^{32}\bar d_h.
}
\]

Variance:

\[
\widehat{\operatorname{Var}}(\hat\mu)
=
\frac1{32^2}
\sum_h\frac{s_h^2}{n_h}.
\]

When estimable, use Satterthwaite degrees of freedom:

\[
\nu
=
\frac{
\left(
\sum_h s_h^2/n_h
\right)^2
}{
\sum_h
(s_h^2/n_h)^2/(n_h-1)
}.
\]

This directly respects the fixed router×rotation balance.

---

# 37. Canonical inferential strata

Canonical uses the four fixed router dictionaries as strata.

Each contains:

\[
n_s=24.
\]

Use the equal-router estimator:

\[
\hat\mu
=
\frac14\sum_s\bar d_s
\]

with the corresponding stratified variance and Satterthwaite degrees of freedom.

---

# 38. Degenerate variance rule

If the predeclared stratified variance is not estimable because the Satterthwaite numerator/denominator degenerates:

- do not switch to an unstratified test after seeing results;
- do not emit a zero-width t interval;
- do not substitute a bootstrap because it is narrower.

Report:

```text
UNRESOLVED_ZERO_VARIANCE
```

unless an independent analytic invariant proves the contrast exactly.

---

# 39. Formal primary inference

Primary family:

```text
K8_FIRST_USE_RETURN
```

contrasts:

```text
FB2 - B
FB4 - B
FB8 - B
FB4 - FB2
FB8 - FB4
```

Thus:

\[
M_P=5.
\]

Allocate:

\[
\alpha_P=0.02.
\]

Use simultaneous two-sided Bonferroni intervals with the frozen 32-stratum K8 estimator.

Historical `S_FIXED-B` first-use SD:

\[
8.201\text{ pp}
\]

is used only as a planning proxy.

At \(n=128\), plausible stratified-df planning gives a primary half-width of roughly:

```text
2.13–2.25 pp
```

under that historical scale.

This is not a guarantee for FB4−FB2 or FB8−FB4.

---

# 40. Dense guard inference

Dense family:

\[
M_G=45.
\]

Allocate:

\[
\alpha_G=0.02.
\]

Use simultaneous two-sided Bonferroni stratified paired-t intervals.

For positive-valued guards, confirmed harm means the upper interval bound is below zero.

No noninferiority margin is invented.

---

# 41. Sparse retention-risk inference

Sparse stable-loss family:

\[
M_R=6,
\qquad
\alpha_R=0.01.
\]

Use the frozen simultaneous Hoeffding envelope on:

\[
D_{c,w}^{\rm stable}\in[-1,1].
\]

This is intentionally conservative.

The family is not used to claim precise absence of forgetting.

It exists to prevent switching inferential methods after observing sparse outcomes.

---

# 42. Per-stratum disclosure

For every formal K8 contrast publish:

- overall equal-stratum estimate;
- four router-seed marginal estimates;
- eight rotation marginal estimates;
- all 32 router×rotation cell means.

For canonical publish:

- overall estimate;
- each router-seed estimate.

No formal “every router works” claim is made unless separately predeclared.

Opposite-sign strata are highlighted.

---

# 43. Precision and power sensitivity

The main sample remains fixed:

```text
96 canonical
128 K8
```

The design is intended to:

- re-confirm a large fixed-bank effect;
- characterize the resource curve;
- detect large adjacent-bank differences.

It is not designed to guarantee detection of 1 pp adjacent-bank increments.

Using historical `S_FIXED-B` SD 8.201 pp only as a planning proxy:

```text
true 3 pp effect → ~88% approximate power at n=128
true 2 pp effect → ~43%
true 1 pp effect → ~6%
```

These calculations are not validated power for new adjacent-bank contrasts.

No data-dependent sample increase is authorized.

---

# 44. Formal status vector

Every candidate receives separate status fields.

## Execution

```text
EXECUTION_COMPLETE
EXECUTION_INCOMPLETE
```

## Primary

```text
PRIMARY_GAIN_CONFIRMED
PRIMARY_GAIN_NOT_CONFIRMED
PRIMARY_INFERENCE_INCOMPLETE
```

## Dense guards

```text
NO_CONFIRMED_DENSE_GUARD_HARM
CONFIRMED_DENSE_GUARD_HARM
DENSE_GUARD_INFERENCE_INCOMPLETE
```

## Stable risk

```text
STABLE_RISK_BENEFIT_CONFIRMED
STABLE_RISK_UNRESOLVED
STABLE_RISK_HARM_CONFIRMED
STABLE_RISK_UNSUPPORTED
```

## Resource candidacy

```text
ADVANCE_AS_RESOURCE_CANDIDATE
DO_NOT_ADVANCE
CANDIDACY_INCOMPLETE
```

These fields are not collapsed into one ordinal category.

---

# 45. Resource-candidate rule

Set:

```text
ADVANCE_AS_RESOURCE_CANDIDATE
```

when:

1. execution is complete;
2. primary gain versus B is confirmed;
3. every dense guard interval is estimable;
4. no dense guard has confirmed harm.

Stable-loss risk remains an independent status.

Therefore this is valid:

```text
ADVANCE_AS_RESOURCE_CANDIDATE
STABLE_RISK_UNRESOLVED
```

and must be reported exactly that way.

---

# 46. Smallest advancing tested arm

Among resource candidates, define:

```text
SMALLEST_ADVANCING_TESTED_ARM
```

as the smallest tested \(K\).

This means only:

> the smallest **tested** bank count meeting the frozen research-candidate rule.

It does not prove:

- global minimality;
- equivalence to larger K;
- safety;
- absence of a better untested partition.

---

# 47. Resource accounting

Separate:

```text
deployment mutable state
banked alpha state
shared native mutable state
fixed router centers
fixed B baseline
native transient/scratch state
evaluator-only diagnostic state
shared read-only anatomy/model assets
peak process RAM
runtime
```

Evaluator-only diagnostics are not learner memory.

Scratch arrays are not persistent state.

Known planning anchors under the current float64 layout:

```text
FB2 ≈   662,656 bytes
FB4 ≈   828,320 bytes
FB8 ≈ 1,159,648 bytes
```

Historical measured:

```text
S_FIXED = 1,159,744 bytes
B       =   165,688 bytes
```

The new measured ledger supersedes these planning values.

---

# 48. Descriptive resource Pareto set only

Generate:

```text
DESCRIPTIVE_RESOURCE_PARETO_SET.csv
```

from observed competence/risk/resource summaries.

It is explicitly descriptive.

Mean-based Pareto membership is not called statistically proven optimality.

Only 2/4/8 bank counts and the frozen hierarchy have been searched.

---

# 49. No formal diminishing-return label

Because bank increments and byte increments differ:

```text
2 → 4
4 → 8
```

do not define equal resource steps.

Report descriptive slopes:

\[
\frac{Y_4-Y_2}
{\text{bytes}_4-\text{bytes}_2},
\qquad
\frac{Y_8-Y_4}
{\text{bytes}_8-\text{bytes}_4}.
\]

Do not attach a formal “diminishing returns” status.

---

# 50. Capacity versus isolation remains unresolved by FB1

FB1 changes both:

- amount of independent alpha state;
- degree of storage isolation.

Therefore a fixed-bank win does **not** prove isolation is the causal source.

If an FB arm advances, the next authorized causal gate is:

```text
FB2-EQ
```

whose target is:

> partitioned fixed-address storage versus a shared/non-isolated architecture with the same total deployable state budget.

FB2-EQ is not part of FB1 because a source-faithful equal-total-state shared control is not yet defined.

No “isolation mechanism established” language is permitted before FB2-EQ or another equally direct control.

---

# 51. Historical adaptive-router address-drift diagnostic

This is retrospective and cannot affect FB1 candidacy.

Select exactly four archived K8 worlds using metadata only:

- choose the lowest archived control-world ID in rotations 0, 2, 4, and 6;
- freeze them in `ADDRESS_DRIFT_MANIFEST.json` before replay.

Replay:

```text
S_LOCAL
S_FIXED
```

on each.

Total historical drift panel:

\[
4\text{ worlds}\times2\text{ arms}
=
8\text{ replay trajectories}.
\]

For each logical cue, compare:

```text
write-time route class
first-use read-time route class
route mismatch
first-use margin
```

`S_FIXED` should satisfy exact same-code address stability.

`S_LOCAL` mismatch is an empirical historical diagnostic.

The old `route_changes` adjacent-event counter is not used as a substitute.

---

# 52. Qualification execution budget

Maximum qualification full-life equivalents:

\[
12.
\]

Freeze the schedule:

## Q-run C

One archived canonical qualification world:

```text
B
FB2
FB4
FB8
```

= 4 trajectories.

## Q-run K

One archived K8 qualification world, chosen as the rotation-0 world used in the historical drift panel:

```text
B
FB2
FB4
FB8
```

= 4 trajectories.

Subtotal:

\[
8.
\]

## FB1 identity continuations

Canonical + K8:

```text
FB1_IDENTITY
FB1_IDENTITY
```

= 2 full-life-equivalent continuations.

Subtotal:

\[
10.
\]

## Old fixed-source bridge

Replay old locked:

```text
S_FIXED
```

on Q-run K.

= 1 trajectory.

Subtotal:

\[
11.
\]

## Short deterministic fixtures

All transition, clone, serialization, probe, compact-summary, and report-pipeline synthetic fixtures share a combined budget of:

```text
≤1 full-life equivalent
```

for accounting.

Qualification cap:

\[
\boxed{12}.
\]

---

# 53. Total non-science replay budget

The historical address-drift panel contains 8 trajectory slots.

The Q-run K `S_FIXED` replay is shared with that panel and counted once.

Therefore:

```text
qualification unique full-life equivalents ≤ 12
additional unique drift replays       ≤ 7
```

and:

\[
\boxed{
\text{non-science unique full-life equivalents}\le19
}
\]

with a hard administrative ceiling of:

\[
20.
\]

Retries caused by crashes are separately recorded as attempts and do not create new scientific units.

No failed world is replaced with a new seed.

---

# 54. Historical exclusion contract

Freeze:

```text
HISTORICAL_SOURCE_ROOTS.json
```

and explicit manifest hashes.

At minimum include:

```text
/Users/pencilbard/Downloads/second_try_copy
/Users/pencilbard/Documents/second_try/CORE_RECOVERY_A10/v2
```

plus explicit source manifests for:

```text
A8
A9
B/K/R V5
A10 Phase A/B
Post-V5 S_LOCAL/S_FIXED
evolution
validation
final-confirmation
```

Construct one exclusion union across all known:

```text
world seeds
donor seeds
pattern keys
raw cue hashes
composite world hashes
```

Before launch verify:

```text
historical ∩ FB1 = ∅
```

for the relevant namespaces.

---

# 55. Compact streaming diagnostics

The experiment must not persist one row per plastic presentation.

Historical scale implies approximately 11.6 million plastic cue presentations across the three FB arms.

Per-event permanent logging is prohibited.

Maintain only streaming sufficient statistics.

Maximum intended evaluator state scales approximately with:

```text
O(number of cues × number of banks)
```

rather than number of lifetime events.

Permanent results aggregate by:

```text
world
arm
router seed
rotation
checkpoint class
route class
physical bank
age bin
```

---

# 56. Required streaming diagnostic state

Evaluator-only state may include:

```text
per-bank write/read counts
per-route-class counts
per-cue write counts
per-bank retained absolute slow-mass accumulators
per-cue × bank retained absolute slow-mass accumulators
first/last teaching time per cue
tail numerators/denominators
endpoint numerators/denominators
fixed age-bin counts
```

This state is not learner state and is accounted separately.

It may never affect the learner.

---

# 57. World-level persisted record

For each world/arm persist at least:

```text
world_id
world_hash
event_stream_hash
arm
source/config hash
router seed
K8 rotation, if applicable
formal endpoint numerators/denominators
formal paired effects
stable-start coverage components
stable-loss components
stable support count
absolute Q/R tail components
route utilization
bank utilization
pair-opponent overlap
foreign event overlap
foreign retained-mass overlap
age-bin summaries
resource/runtime summary
numeric audit maxima
final-state digest
write digest
completion/error status
```

Do not persist the full diagnostic event stream.

---

# 58. End-to-end reporting qualification

Before new science world generation, process the technical qualification records through:

```text
world commit
→ compact aggregation
→ stratified analysis
→ zero-variance handling
→ Hoeffding stable-risk analysis
→ status vector
→ tables
→ figures
→ REPORT.md
→ ZIP
→ MANIFEST/hash verification
```

The plotting/report environment is part of the frozen execution environment.

This gate must pass before science.

---

# 59. Runtime process contract

Freeze:

```text
workers = 4
```

unless qualification demonstrates resource failure.

Run under a Mac process wrapper that prevents system sleep during active execution.

Persist:

```text
RUN_LOCK.json
RUN_HEARTBEAT.json
```

with:

- protocol hash;
- source hash;
- PID/host;
- start time;
- last completed world;
- active world IDs;
- retry count;
- current disk usage;
- current status.

Heartbeat updates atomically.

A stale lock requires explicit resume logic; it does not authorize a new roster.

---

# 60. Atomic world semantics

One science world is committed only after:

```text
B
FB2
FB4
FB8
```

all complete and validate.

If interrupted:

- discard/mark incomplete temporary outputs for that world;
- rerun the same frozen world;
- do not replace it;
- do not retain an easier-arm-only complete case.

`896` is the planned count of unique completed science trajectories, not a promise that crash retries produce no extra executions.

---

# 61. Disk contract

Qualification must measure actual compact-output sizes.

Planned limits:

```text
persistent new experiment files ≤ 64 MiB
peak incremental disk           ≤ 256 MiB
shareable final bundle          < 10 MB
```

Historical world-size projection for the frozen roster is approximately:

```text
19.25 MiB
```

when each actual world is stored once.

This leaves headroom for compact diagnostics, but the limit is not assumed to pass until measured.

If evidence cannot fit without dropping required fields/samples:

```text
DISK_BUDGET_UNRESOLVED
```

and science does not launch.

---

# 62. Runtime planning is not a statistical decision

Historical measured per-life runtimes are used only for execution planning.

They do not alter:

- candidate eligibility;
- bank hierarchy;
- sample size;
- formal endpoint families.

No faster bank arm receives more worlds.

No slow bank arm loses worlds.

---

# 63. Qualification gates

Science begins only if all pass.

## Q1 Parent

Exact B reproduces parent.

## Q2 Address/storage separation

Every FB arm keeps:

```text
8 router centers
8 route classes
K physical stores
```

with read/write using the same frozen 8→K mapping.

## Q3 FB1 nonzero-state identity

One-bank wrapper reproduces B from cloned nonzero state.

## Q4 FB8 historical bridge

New FB8/R0 reproduces old `S_FIXED` replay.

## Q5 Event ordering

Selected bank loaded pre-event; native transition once; selected bank saved; unselected banks passive-decayed once.

## Q6 Checkpoint clock

For every qualification world:

```text
block7_day → final_7day = 604800 s
teaching events in window = 0
```

with actual absolute times stored.

## Q7 Frozen hierarchy

All four center sets and exact FB4/FB2 partitions reproduce from seeds.

## Q8 K8 manifest

Exactly:

```text
4 router seeds × 8 rotations × 4 worlds
```

with donor uniqueness.

## Q9 Read-only probes

No state/RNG mutation.

## Q10 Save/restore

Identical subsequent transitions after round trip.

## Q11 B-corrected slow write accounting

Verify:

```text
rawslow = q_s - c
post - d_s*pre = d_s*rawslow
```

within frozen operation-specific tolerance.

## Q12 Sparse retention pipeline

Stable-start, stable-loss mass, support flags, Hoeffding family, and unsupported state reproduce from synthetic/technical fixtures.

## Q13 Streaming schema

Formal results reproduce from compact sufficient statistics without raw per-event storage.

## Q14 Historical exclusion

All roots/manifests scanned; new 1,120 base worlds disjoint.

## Q15 End-to-end delivery

Figures/report/ZIP/manifest complete successfully.

Failure of any gate blocks science.

---

# 64. Final interpretation table

Every final report contains:

```text
arm
K
execution_status
primary_status
dense_guard_status
stable_risk_status
resource_candidate_status

K8 first-use effect vs B
canonical effects
K8 guard effects
stable-start effect
stable-loss risk envelope
stable support

absolute Q/R costs
pair-opponent overlap
foreign event overlap
foreign retained-mass overlap
bank-load heterogeneity

deployment mutable bytes
fixed router bytes
total deployable bytes
runtime
peak RAM
```

---

# 65. Descriptive resource Pareto output

Produce:

```text
DESCRIPTIVE_RESOURCE_PARETO_SET.csv
```

using observed means/resources.

It is descriptive only.

Do not call Pareto membership statistically established superiority.

---

# 66. Stopping boundary

After the frozen 896 science trajectories:

do **not** add:

- bank3;
- bank6;
- bank16;
- different hierarchy;
- traffic-balanced hierarchy;
- behavior-optimized hierarchy;
- router adaptation;
- router learning-rate changes;
- outcome-aware routing;
- bank-specific B vectors;
- bank-specific decay;
- extra confirmatory worlds.

Final FB1 statuses may include:

```text
FIXED_BANK_GAIN_REPLICATED
FIXED_BANK_GAIN_NOT_REPLICATED
CONFIRMED_RESOURCE_TRADEOFF
NO_FIXED_BANK_RESOURCE_CANDIDATE
STABLE_RISK_UNRESOLVED
INFERENCE_INCOMPLETE
```

The fixed-address 2/4/8 resource curve then closes.

---

# 67. Next causal gate

If at least one arm is:

```text
ADVANCE_AS_RESOURCE_CANDIDATE
```

the next causal question is:

```text
FB2-EQ
```

which must compare partitioned storage to a non-isolated/shared architecture with matched total deployable state.

FB1 itself cannot answer:

\[
\text{capacity}
\quad\text{versus}\quad
\text{isolation}.
\]

No isolation-specific core claim is permitted before that test or an equivalent direct control.

---

# 68. Relation to A10 and A11

A10 is complete for sequencing.

FB1 retains the complete native B/allocation/alpha-feedback loop.

A11 remains a separate broader architecture-characterization program.

FB1 should finish before the final A11 parent is selected.

An FB1 research candidate does not automatically become A11 parent.

If stable-storage architecture remains promising, FB2-EQ may precede that parent choice.

---

# 69. Required artifacts

```text
PROTOCOL_LOCK.json
SOURCE_LOCK.json
B_PARENT_LOCK.json

HISTORICAL_SOURCE_ROOTS.json
HISTORICAL_EXCLUSION.json
WORLD_MANIFEST.json
ENDPOINT_DICTIONARY.json

ROUTER_SEED_MANIFEST.json
ROUTER_STORAGE_HIERARCHY.json
ADDRESS_DRIFT_MANIFEST.json
STREAM_DIAGNOSTIC_SCHEMA.json

QUALIFICATION_PLAN.json
QUALIFICATION.json
HISTORICAL_BRIDGE_CHECK.json
ADDRESS_DRIFT_DIAGNOSTIC.json

RESOURCE_LEDGER.csv
STORAGE_BUDGET_AUDIT.json
RUN_LOCK.json
RUN_HEARTBEAT.json

WORLD_LEVEL_ENDPOINTS.csv.gz
PRIMARY_INTERVALS.csv
DENSE_GUARD_INTERVALS.csv
STABLE_RISK_INTERVALS.csv
SEED_ROTATION_EFFECTS.csv
TAIL_BY_WORLD.csv.gz

ROUTE_UTILIZATION.csv.gz
BANK_UTILIZATION.csv.gz
PAIR_OPPONENT_OVERLAP.csv.gz
FOREIGN_OVERLAP.csv.gz
WRITE_LOAD_BY_BANK.csv.gz
LIFETIME_HETEROGENEITY.csv.gz

RESOURCE_CAPABILITY_TABLE.csv
DESCRIPTIVE_RESOURCE_PARETO_SET.csv

FINAL_DECISION.md
REPORT.md
MANIFEST.json
```

---

# 70. Final target

FB1 v3 is deliberately narrower than a minimal-core certification.

Its exact deliverable is:

\[
\boxed{
K\in\{2,4,8\}
\longmapsto
\left(
\text{first-use return},
\text{ongoing competence},
\text{sparse retention risk},
\text{address collisions},
\text{state cost}
\right)
}
\]

conditional on:

- four predeclared fixed 8-way address dictionaries;
- one predeclared balanced nested coarsening rule;
- the exact incumbent B closed loop.

This experiment is strong enough to tell us whether the historical fixed-bank signal survives at substantially lower banked-state cost.

It is intentionally not strong enough to claim that isolation is the unique causal mechanism or that the smallest successful tested bank count is the globally minimal persistent-learning core.
