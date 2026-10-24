# V49 — Raw-BANC Mushroom-Body Synaptic Memory
## Does the actual PN→KC→MBON alignment provide a bounded persistent-learning substrate, and does raw MBON recurrence add value beyond matched graph controls?

**Status:** refined locked design / implementation specification  
**Date:** 11 September 2026  
**Primary evidence source:** uploaded BANC v888 raw metadata and edge-list tables  
**Target platform:** Apple Silicon Mac M3, macOS, Python 3.11+  

---

# 0. Fixed research objective

The project objective remains:

> Find a **minimal, bounded-state, non-Transformer, online persistent-learning core whose architecture is materially constrained by the raw fly connectome**.

V49 does **not** assume that any previous software memory abstraction is biologically correct.

In particular, V49 does not use:

- permanent object slots;
- object-level trace/promotion/eviction machinery;
- learned pathway routing;
- Candidate-C PAIR representation learning;
- signed representation learning;
- factor-coded output learning as a scientific variable;
- replay;
- Transformer/backpropagation training;
- APL control from V48;
- DAN learning rules in the primary experiment.

Previous versions are used only as evidence about failed/successful implementations and as workload/evaluation infrastructure.

The V49 scientific substrate is the raw BANC:

\[
\boxed{
\text{fixed PN sensory layer}
\rightarrow
\text{raw PN→KC expansion}
\rightarrow
\text{sparse KC code}
\rightarrow
\text{raw KC→MBON plastic synapses}
}
\]

with raw MBON recurrence tested separately in Block B.

---

# 1. Raw-data provenance

The runner must embed graph arrays extracted directly from:

```text
banc_888_meta.feather
banc_888_edgelist_simple_v2.feather
```

Source SHA256:

```text
banc_888_meta.feather
819bbcff476e52702d6f8d8604ce1f12d1d7b11942281df2f49df2a73a6f15a5

banc_888_edgelist_simple_v2.feather
363fdef3813b72a5e45a42f17034cd5a544b654c838929cecf6e1ce5f60625cb
```

The scientific Mac runner should not require a Feather/Arrow dependency.

Instead, before freezing the V49 runner:

1. extract the required raw graph blocks from the supplied Feather files;
2. embed the resulting integer IDs, sparse indices and recorded counts into the runner/bundle;
3. store the two source hashes above;
4. self-test the embedded graph against the audited totals below.

No external literature or web-derived connectivity may enter the V49 graph.

---

# 2. Raw-BANC facts used by V49

The raw audit over the supplied export found:

## 2.1 Population sizes

```text
PN    699
KC    4,552
MBON  104
DAN   301
DPM   2
APL   1
```

V49 Block A uses the **full bilateral PN/KC/MBON populations**:

\[
N_{PN}=699,\qquad
N_{KC}=4552,\qquad
N_{MBON}=104.
\]

This avoids selecting one hemisphere simply for engineering convenience.

## 2.2 Raw PN→KC

```text
PN→KC pair rows:        22,417
recorded count sum:    168,315
```

This graph is fixed in every V49 arm.

## 2.3 Raw KC→MBON

```text
KC→MBON pair rows:      15,523
recorded count sum:     28,415
```

Among KCs with at least one recorded MBON output:

```text
connected KCs:          3,587
mean MBON fanout:       4.33
median fanout:          4
multi-output fraction:  81.10% at count >= 1
```

Threshold sensitivity:

```text
count >= 3: multi-output fraction ≈ 43.06%
count >= 5: multi-output fraction ≈ 14.95%
```

This motivates testing the actual recorded-count heterogeneity rather than converting the graph immediately to a binary support mask.

## 2.4 Raw MBON recurrence

All MBON→MBON rows:

```text
pair rows:              1,245
recorded count sum:     15,137
```

Excluding self-connections:

```text
directed edges:         1,141
recorded count sum:     10,140
```

Thresholds on non-self edges:

```text
count >= 5:    410 edges
count >= 10:   215
count >= 20:   114
count >= 50:    41
```

The raw reciprocal directed-edge fraction is approximately:

\[
48.35\%.
\]

This is tested independently in Block B.

---

# 3. Central V49 questions

V49 has two distinct scientific blocks.

## Block A — synaptic-memory substrate

> Does preserving the **actual alignment** between PN→KC sensory expansion and KC→MBON plastic connectivity improve bounded persistent learning relative to graph controls with the same size, degree and weight-count statistics?

The key word is **alignment**.

Raw PN→KC defines which sensory mixtures activate each BANC KC.

Raw KC→MBON defines which MBONs that same KC can modify.

A control that randomizes KC identities between those two graphs preserves each graph separately while destroying their joint alignment.

This is the cleanest V49 test.

## Block B — MBON recurrent state

> With the KC→MBON substrate held fixed, does the actual non-self MBON recurrence improve persistent learning more than degree/weight-matched recurrent controls?

Block B tests recurrence as a fixed state mechanism.

It does not introduce a new plasticity rule.

Block A and Block B have independent verdicts.

Block B may run even if raw KC→MBON support fails to beat its controls, provided the raw KC→MBON substrate passes acquisition validity. This prevents one raw motif's failure from automatically voiding a different raw motif.

---

# 4. Common workload

Reuse the current controlled online persistent-learning workload for continuity.

## 4.1 Scientific loads

Primary:

\[
P=256.
\]

Secondary:

\[
P=128.
\]

Load 256 determines qualification.

Load 128 describes scaling and may not rescue a load-256 failure.

## 4.2 Stream

Reuse:

- four sequential cohorts A→B→C→D;
- 8 acquisition presentations per identity;
- no replay;
- held-out rendering probes;
- intervening learning;
- one-off noise;
- revision;
- ordinary byte prediction;
- all intended identities retained in denominators.

Prediction always occurs before the observed byte is used for learning.

Evaluator identity, cohort, role, wrapper and revision labels are unavailable to the learner.

---

# 5. Fixed sensory interface: 96 context coordinates → 699 BANC PNs

The byte task does not naturally provide fly PN channels.

Therefore V49 needs one explicitly engineering-only adaptor.

This adaptor is:

- fixed within a scientific seed;
- random but deterministic from that scientific seed;
- identical across every matched arm and both loads for that seed;
- independently regenerated across scientific seeds;
- nonplastic;
- not part of any raw-topology claim.

Varying this nuisance mapping across seeds prevents one lucky synthetic context→PN assignment from being mistaken for a raw-connectome effect.

## 5.1 Input vector

Use the existing normalized 96-dimensional signed context representation:

\[
x_t\in\mathbb R^{96}.
\]

## 5.2 Fixed sparse context→PN adaptor

For each of 699 PN channels:

- choose 8 distinct context coordinates using a fixed adaptor seed;
- assign fixed Rademacher signs \(\pm1/\sqrt8\).

Then:

\[
u_p
=
\tanh\left(
\sum_{q\in A_p}
r_{pq} x_q
\right).
\]

The adaptor is deterministically reconstructed **per scientific seed** and is identical across all matched arms and both loads for that seed. Different scientific seeds receive different adaptor draws. This treats the synthetic sensory interface as a nuisance variable rather than a single lucky fixed mapping.

The adaptor exists only because the synthetic byte workload lacks biological sensory channels.

No claim may attribute a V49 result to this adaptor.

---

# 6. Fixed raw PN→KC encoder

Let \(C^{PK}_{pk}\) be the raw recorded count on PN \(p\) → KC \(k\).

For each KC, define a fixed normalized raw scale:

\[
a^{PK}_{pk}
=
\frac{\sqrt{C^{PK}_{pk}}}
{\sqrt{\sum_{p'} C^{PK}_{p'k}}+\epsilon}.
\]

KC drive:

\[
d_k
=
\sum_{p}
a^{PK}_{pk} u_p.
\]

Apply a fixed nonlinearity:

\[
v_k=\tanh(d_k).
\]

Then retain the largest 5% of KCs by absolute activity:

\[
K_{\rm active}
=
\operatorname{round}(0.05\times4552)
=
228.
\]

Set all other KCs to zero and L2-normalize the retained vector:

\[
z=
\frac{\operatorname{TopK}_{228}(v)}
{\|\operatorname{TopK}_{228}(v)\|_2+\epsilon}.
\]

This KC representation is common to all Block-A arms.

No APL recurrence is included.

V48 already tested one scalar APL abstraction and it did not improve selective interference.

---

# 7. Fixed 104-dimensional MBON output code for 256 bytes

The raw graph contains 104 MBONs, whereas the task predicts 256 byte values.

V49 therefore uses a fixed codebook:

\[
C\in\{-1,+1\}^{256\times104}.
\]

## 7.1 Codebook construction

For each scientific seed, generate one deterministic seed-specific Rademacher codebook. The same codebook is shared by every matched arm and both loads for that seed; a different codebook is used in the next scientific seed.

Generate normalized Rademacher codewords:

\[
\|c_y\|_2=1.
\]

The codebook is deterministically generated **per scientific seed**, and is identical across all matched arms and both loads for that seed. Different seeds receive different codebooks so a raw-topology result cannot depend on one lucky 256→104 embedding.

Before scientific execution report:

- maximum pairwise codeword cosine;
- mean absolute pairwise cosine;
- rank of the 256×104 codebook.

No codebook sweep is permitted.

## 7.2 Decoding

Given MBON state:

\[
h\in\mathbb R^{104},
\]

byte logits are:

\[
\ell_y
=
\tau^{-1} c_y^\top h.
\]

Use the engineering-calibrated fixed temperature:

\[
\boxed{\tau=0.25}.
\]

This value is frozen after engineering-only acquisition checks on disjoint non-scientific seeds and shared across all arms. No science-seed sweep is permitted.

Probability:

\[
p(y)=\operatorname{softmax}(\ell)_y.
\]

Prediction loss is bits/byte.

---

# 8. Plastic KC→MBON synapses

This is V49's durable memory.

There are **no permanent object records**.

For each existing KC→MBON edge \(i\to j\), store one signed plastic parameter:

\[
\theta_{ij}.
\]

All nonedges remain exactly zero.

Persistent learned state therefore scales with the fixed number of raw graph edges, not with number of past experiences.

## 8.1 Raw count scale

For each MBON \(j\), let raw existing-edge counts be \(C^{KM}_{ij}\).

For the count-sensitive arms:

\[
s_{ij}
=
\frac{\sqrt{C^{KM}_{ij}}}
{\sqrt{
\frac{1}{d_j}
\sum_{i:M_{ij}=1}
C^{KM}_{ij}
}
+\epsilon}.
\]

Thus within each MBON:

\[
\operatorname{mean}_{i:M_{ij}=1}
s_{ij}^2
\approx1.
\]

This preserves relative recorded-count heterogeneity without letting high-degree/high-count MBONs automatically receive a larger total learning scale.

For binary arms:

\[
s_{ij}=1.
\]

Effective edge weight:

\[
w_{ij}=s_{ij}\theta_{ij}.
\]

Initialize:

\[
\theta_{ij}=0.
\]

The raw counts do not provide an initial learned association.

They only shape the fixed support and/or relative edge scale.

---

# 9. Feed-forward MBON prediction

For an arm with support \(M_{ij}\):

\[
h_j
=
\sum_i
M_{ij}s_{ij}\theta_{ij}z_i.
\]

The active KC vector is sparse, so only active existing edges need to be traversed.

Byte prediction uses the fixed 104-dimensional codebook.

---

# 10. Local self-supervised KC→MBON plasticity

After predicting byte \(y_t\), observe the true byte.

Let:

\[
\bar c
=
\sum_y p(y)c_y.
\]

The normalized 104-dimensional codebook makes individual residual coordinates approximately \(1/\sqrt{104}\) smaller than a one-hot 256-output residual. Standardize this coordinate scale before applying the common local rule:

\[
\tilde e
=
\sqrt{104}\,(c_{y_t}-\bar c).
\]

For every active existing KC→MBON edge:

\[
\boxed{
\theta_{ij}
\leftarrow
\theta_{ij}
+
\eta\,
s_{ij}\,
z_i\,
\tilde e_j.
}
\]

Use the engineering-calibrated fixed post-standardization learning rate:

\[
\boxed{\eta=4.0}
\]

for every Block-A and Block-B arm.

Together with the fixed decoder temperature

\[
\boxed{\tau=0.25},
\]

this scale was frozen using only the disjoint engineering seeds before any V49 scientific seed was inspected. The calibration exists because the 104-dimensional normalized code changes output-coordinate scale relative to earlier 256-one-hot learners. It is not a graph hyperparameter and is not swept on scientific seeds.

This is an engineering three-factor-like local output rule:

- presynaptic KC activity \(z_i\);
- MBON-specific observed-outcome residual \(\tilde e_j\);
- existing anatomical edge mask.

It is not claimed to be the fly's true plasticity equation.

There is:

- no backpropagation through PN→KC;
- no replay;
- no object trace;
- no slot;
- no gradient through time.

---

## 10.1 Engineering-only calibration record

The initial implementation check showed that directly reusing the old one-hot rate `0.6` with a normalized 104-dimensional code produced an artificially small teaching coordinate. Engineering-only calibration therefore standardized the residual by `sqrt(104)` and compared decoder/update scale only on seeds disjoint from `49901–49916`.

The frozen scientific values are:

```text
MBON code temperature          = 0.25
standardized residual factor   = sqrt(104)
post-standardization eta       = 4.0
```

No scientific sweep over these values is permitted.

# 11. Block A arms

Run all arms inside one seed/load trajectory.

They share:

- stream;
- fixed context→PN adaptor;
- raw PN→KC activity;
- active KCs;
- codebook;
- learning rate;
- byte outcomes;
- evaluation schedule.

Only the KC→MBON support/count structure differs.

## A0 — `RAW-ALIGNED`

Use actual raw KC identities and raw KC→MBON support/counts.

This is the primary arm.

## A1 — `KC-ID-SHUFFLE`

Apply one seed-fixed permutation \(\pi\) to KC rows of the entire KC→MBON support/count matrix:

\[
M'_{ij}=M_{\pi(i)j}.
\]

PN→KC is unchanged.

This preserves exactly:

- KC→MBON row-degree multiset;
- MBON in-degrees;
- edge count;
- count-value multiset;
- each permuted KC output pattern.

But it destroys the alignment between:

\[
\text{the sensory mixture that defines BANC KC }i
\]

and

\[
\text{the MBON plastic targets of BANC KC }i.
\]

This is the primary alignment control.

## A2 — `KM-DEGREE-REWIRE`

Degree-preserving bipartite edge swaps on KC→MBON.

Preserve:

- every KC out-degree;
- every MBON in-degree;
- total edge count.

After rewiring, assign to each MBON the exact original incoming count multiset, randomly permuted over its rewired KC inputs.

Thus each MBON preserves its raw incoming count distribution.

This destroys higher-order raw support topology while preserving degree and per-MBON count statistics.

## A3 — `KM-WEIGHT-SHUFFLE`

Keep exact raw support.

Within each MBON, permute the raw recorded counts among its incoming KC edges.

This preserves:

- raw support;
- KC degrees;
- MBON degrees;
- each MBON's incoming count multiset.

It destroys which raw KC receives each stronger/weaker recorded connection.

## A4 — `KM-BINARY`

Keep exact raw support.

Set:

\[
s_{ij}=1
\]

for every existing edge.

This tests whether raw support alone is sufficient and whether recorded-count heterogeneity contributes additional value.

---

# 12. Block A scientific questions

V49 Block A separates three claims.

## 12.1 Joint sensory/output alignment

Primary contrast:

\[
\text{RAW-ALIGNED}
-
\text{KC-ID-SHUFFLE}.
\]

If RAW wins, there is evidence that the actual alignment of PN→KC sensory identity with KC→MBON plastic targets matters.

## 12.2 Support topology beyond degree

Contrast:

\[
\text{RAW-ALIGNED}
-
\text{KM-DEGREE-REWIRE}.
\]

If RAW wins, raw KC→MBON organization contains value beyond the degree sequence and per-MBON weight distribution.

## 12.3 Recorded-count organization

Contrasts:

\[
\text{RAW-ALIGNED}
-
\text{KM-WEIGHT-SHUFFLE}
\]

and

\[
\text{RAW-ALIGNED}
-
\text{KM-BINARY}.
\]

These ask whether the raw weak/strong edge pattern contributes beyond support alone.

No interpretation of raw counts as physiological synaptic strengths is assumed.

They are tested only as the quantitative information present in the supplied graph.

---

# 13. Direct interference audit for Block A

Reuse the bounded deterministic probe-bank methodology from recent rounds.

For a fixed deterministic sample of **target-byte KC→MBON writes only**:

1. record probe losses before the update;
2. apply the local update;
3. record probe losses after.

Whole-stream learning still updates on every byte. The interference audit is restricted to target-byte writes because only there do evaluator identity/outcome categories have a coherent association-level meaning; ordinary prefix/header/separator writes must not be relabeled as identity-specific interference.

Categories:

- same identity / alternate rendering;
- different identity / same outcome;
- different identity / different outcome;
- old cohort;
- overall.

Define:

\[
T_{\rm same}
=
E[\Delta\ell\mid \text{same identity alt}],
\]

\[
I_{\rm diff}
=
E[\Delta\ell\mid \text{different identity, different outcome}].
\]

More negative \(T_{\rm same}\) is beneficial transfer.

Lower \(I_{\rm diff}\) is less harmful cross-talk.

A raw-topology claim should not rest solely on terminal accuracy.

---

# 14. Block A endpoints

For every arm/load report:

- D trained target accuracy;
- A after D held-out accuracy;
- A after D held-out bpb;
- whole-stream bpb;
- target bpb;
- prequential acquisition bpb;
- revision accuracy and bpb at exposures 1, 2, 4, 8;
- same-identity KC cosine/Jaccard;
- different-identity KC cosine/Jaccard;
- direct write interference categories;
- cumulative KC→MBON update norm;
- learned-state bytes;
- fixed-graph bytes;
- operations/byte;
- wall-clock time;
- peak RSS.

---

# 15. Block A ordered gates

## A-G0 — raw graph integrity

Require exact embedded counts:

```text
PN        = 699
KC        = 4552
MBON      = 104
PN→KC     = 22,417 edges
PN→KC sum = 168,315
KC→MBON   = 15,523 edges
KC→MBON sum = 28,415
```

Require embedded source hashes to equal Section 1.

Require all arms to have exactly the declared degree/count invariants.

Failure:

`RAW_GRAPH_INTEGRITY_FAIL`.

## A-G1 — code/readout validity

Require:

- fixed codebook rank = 104;
- codewords finite and normalized;
- no duplicate codewords;
- all arms use identical codebook;
- RAW-ALIGNED mean D trained target accuracy at load 256 ≥0.75;
- no RAW-ALIGNED seed below 0.60.

This gate is an absolute learnability floor, not a qualification advantage. The primary alignment gate below separately requires RAW acquisition to be noninferior to `KC-ID-SHUFFLE`, so RAW cannot obtain a persistence advantage merely by learning less.

Failure:

`RAW_KM_ACQUISITION_INVALID`.

## A-G2 — alignment mechanism

At load 256 compare RAW-ALIGNED to KC-ID-SHUFFLE.

Require all three:

### Acquisition noninferiority

At load 256, RAW D trained accuracy minus `KC-ID-SHUFFLE` D trained accuracy must have paired mean ≥ −0.05 and paired 95% lower bound ≥ −0.08.

### Persistent effect

\[
L_{\rm heldout}^{RAW}
-
L_{\rm heldout}^{KCshuffle}
\le
-0.05\text{ bit}
\]

with paired 95% upper bound <0.

### Interference effect

\[
I_{\rm diff}^{RAW}
-
I_{\rm diff}^{KCshuffle}
\le
-0.00025\text{ bpb/write}
\]

with paired 95% upper bound <0.

Same-identity useful transfer may not worsen by more than +0.05 bpb/write in paired mean.

Failure:

`NO_KC_ID_ALIGNMENT_GAIN`.

## A-G3 — topology beyond degree

At load 256:

\[
L_{\rm heldout}^{RAW}
-
L_{\rm heldout}^{REWIRE}
\le
-0.05\text{ bit}
\]

with paired 95% upper bound <0,

or RAW must reduce harmful \(I_{\rm diff}\) by at least 0.00025 bpb/write with upper bound <0 while remaining whole-stream noninferior.

Failure does not void A-G2.

Classification:

`DEGREE_STRUCTURE_SUFFICIENT`.

Pass:

`RAW_SUPPORT_TOPOLOGY_SUPPORTED`.

## A-G4 — raw count organization

This is secondary.

Define count organization supported if RAW beats both:

- KM-WEIGHT-SHUFFLE;
- KM-BINARY;

on held-out bpb by at least 0.03 bit with paired 95% upper bounds <0,

without whole-stream degradation >0.02 bit.

If not:

`RAW_COUNTS_NOT_NEEDED`.

This does not invalidate a support/alignment result.

## A-G5 — prediction/revision safety

RAW vs strongest matched control:

- whole-stream bpb mean difference ≤ +0.02;
- paired 95% upper bound ≤ +0.04;
- revision-8 accuracy loss ≤5 pp mean;
- revision-8 bpb upper bound ≤ +0.10.

Failure:

`RAW_GAIN_WITH_SAFETY_COST`.

## A-G6 — scaling

Let:

\[
G_L
=
L_{\rm control,L}
-
L_{\rm RAW,L}
\]

for the strongest relevant control.

Require primary load-256 gain to remain at least:

\[
0.75\max(G_{128},0.03).
\]

Failure:

`RAW_GAIN_COLLAPSES_WITH_LOAD`.

## A-G7 — bounded state

Learned persistent state must be load-independent.

No state may scale with number of past identities.

Failure:

`STATE_BUDGET_FAIL`.

---

# 16. Block A verdict vocabulary

Possible Block-A conclusions:

- `RAW_KM_ALIGNMENT_SUPPORTED`
- `RAW_KM_ALIGNMENT_AND_TOPOLOGY_SUPPORTED`
- `RAW_KM_ALIGNMENT_TOPOLOGY_COUNTS_SUPPORTED`
- `DEGREE_STRUCTURE_SUFFICIENT`
- `RAW_COUNTS_NOT_NEEDED`
- `NO_KC_ID_ALIGNMENT_GAIN`
- `RAW_KM_ACQUISITION_INVALID`
- `RAW_GAIN_WITH_SAFETY_COST`
- `RAW_GAIN_COLLAPSES_WITH_LOAD`
- `STATE_BUDGET_FAIL`
- `RAW_GRAPH_INTEGRITY_FAIL`

Do not collapse these into one binary success/failure label.

The key scientific distinction is whether:

1. KC identity alignment matters;
2. higher-order raw support matters;
3. raw count assignment matters.

---

# 17. Block B — raw MBON recurrent state

Block B is a separate motif test.

It may run after Block A completes if `RAW-ALIGNED` satisfies A-G1 acquisition validity.

Block B qualification does **not** require Block A raw-topology superiority.

All Block-B arms use the exact same RAW-ALIGNED PN→KC and KC→MBON substrate.

Only MBON recurrence differs.

---

# 18. Raw MBON recurrence matrix

Construct the 104×104 raw non-self matrix:

\[
R^{raw}_{jk}
=
C^{MM}_{jk},
\qquad
j\ne k.
\]

Use the 1,141 non-self raw directed edges.

No transmitter sign is inferred.

Recorded counts are unsigned graph quantities.

## 18.1 Stability normalization

For each recurrence-control matrix separately:

1. take \(\sqrt{\text{recorded count}}\) on existing edges;
2. set diagonal to zero;
3. normalize the matrix to a fixed spectral radius:

\[
\boxed{\rho(R)=0.25}.
\]

This removes arbitrary recurrence-gain tuning.

Every recurrence control has the same target spectral radius.

No scientific recurrence-gain sweep is allowed.

---

# 19. Recurrent MBON dynamics

For online byte event \(t\):

Feed-forward MBON drive:

\[
f_t=W_{KM}^\top z_t.
\]

Recurrent state:

\[
\boxed{
h_t
=
f_t
+
R h_{t-1}.
}
\]

`NO-RECURRENCE` uses:

\[
R=0,
\]

so the feed-forward reference is **exactly the Block-A RAW-ALIGNED MBON state** rather than a new nonlinear baseline.

Because every recurrent control is normalized to \(\rho(R)=0.25<1\), the linear recurrence is contractive in the absence of new feed-forward drive and requires no additional arbitrary activation nonlinearity.

For stream execution, recurrent state persists continuously across byte events.

For held-out read-only probes:

- initialize \(h=0\);
- feed the 12-byte context sequentially using the common sensory pipeline;
- score the final state;
- do not mutate learned weights or persistent online recurrent state.

This makes probe interpretation independent of accidental preceding stream context.

The recurrent state itself is fixed-size transient state, not permanent object memory.

---

# 20. Block B local learning rule

KC→MBON plasticity uses the same local residual rule as Block A.

For recurrent arms, the MBON prediction residual is computed from the recurrent MBON state \(h_t\).

Only KC→MBON synapses remain plastic.

MBON→MBON recurrence is fixed.

This isolates recurrence.

---

# 21. Block B arms

## B0 — `RAW-KM-FF`

Reuse Block-A RAW-ALIGNED feed-forward result.

## B1 — `RAW-KM+RAW-MM`

Raw non-self MBON recurrence with recorded-count scaling and spectral-radius normalization.

Primary recurrence arm.

## B2 — `RAW-KM+MM-DEGREE-REWIRE`

Degree-preserving directed MBON recurrence rewire.

Preserve:

- non-self edge count;
- each MBON recurrent in-degree;
- each MBON recurrent out-degree.

Assign the raw target-node incoming count multiset to rewired incoming edges.

Normalize spectral radius to 0.25.

## B3 — `RAW-KM+MM-WEIGHT-SHUFFLE`

Keep exact raw non-self MBON recurrence support.

Within each postsynaptic MBON, permute incoming recorded counts.

Normalize spectral radius to 0.25.

---

# 22. Block B endpoints

In addition to persistent endpoints, report:

- recurrent-state L2 norm;
- state autocorrelation by lag;
- cross-MBON covariance;
- saturation fraction;
- amount of KC→MBON update norm needed to reach acquisition;
- recurrent contribution norm relative to feed-forward drive;
- same/different identity state similarity;
- direct write interference.

The recurrence claim is strongest if raw recurrence:

1. improves persistence;
2. or reduces required permanent synaptic modification at matched acquisition;
3. and beats matched recurrence controls.

---

# 23. Block B ordered gates

## B-G0 — recurrence integrity

Require exact raw non-self recurrence:

```text
104 MBONs
1,141 non-self edges
recorded count sum 10,140
```

All recurrence controls must preserve declared degree/count invariants.

Every recurrent matrix spectral radius after normalization:

\[
0.25\pm10^{-6}.
\]

Failure:

`MM_GRAPH_INTEGRITY_FAIL`.

## B-G1 — recurrent-state health

Require:

- finite state on every event;
- recurrent-state and feed-forward-state norms are reported;
- no science seed may have >1% of recurrent events with >95% of MBON coordinates at absolute value >20;
- RAW-MM mean D trained accuracy ≥0.75;
- no RAW-MM science seed below 0.60;
- RAW-MM D trained accuracy is noninferior to feed-forward RAW-KM: paired mean difference ≥ −0.05 and paired 95% lower bound ≥ −0.08.

Failure:

`RECURRENT_DYNAMICS_INVALID`.

## B-G2 — raw recurrence persistent gain

At load 256:

\[
L_{\rm heldout}^{RAWMM}
-
L_{\rm FF}
\le
-0.05\text{ bit}
\]

with paired 95% upper bound <0,

or raw recurrence must reduce cumulative KC→MBON update norm by ≥10% at matched acquisition and delayed held-out noninferiority.

If neither:

`NO_MBON_RECURRENCE_GAIN`.

## B-G3 — recurrence specificity

RAW-MM must beat both:

- MM-DEGREE-REWIRE;
- MM-WEIGHT-SHUFFLE;

by at least 0.03 held-out bit with paired upper bound <0,

or show a materially better update-efficiency/interference tradeoff under locked criteria defined in the runner.

Failure:

`RECURRENCE_TOPOLOGY_NOT_SPECIFIC`.

## B-G4 — safety

RAW-MM vs FF:

- whole-stream bpb increase ≤0.02 mean;
- upper CI ≤0.04;
- revision-8 accuracy decline ≤5 pp;
- revision-8 bpb upper CI ≤+0.10.

Failure:

`RECURRENCE_GAIN_WITH_SAFETY_COST`.

## B-G5 — scaling

Any load-256 gain must remain at least 75% of the load-128 gain.

Failure:

`RECURRENCE_GAIN_COLLAPSES_WITH_LOAD`.

---

# 24. Scientific seeds

Fresh V49 seeds:

```text
49901 49902 49903 49904
49905 49906 49907 49908
49909 49910 49911 49912
49913 49914 49915 49916
```

Engineering seeds:

```text
49971+
```

Science seeds must not be inspected before protocol freeze.

Seed is the inferential unit.

Repeated bytes, identities, MBONs, KCs and writes are not independent scientific replicates.

---

# 25. State and compute accounting

Report separately:

## Fixed state

- context→PN adaptor;
- raw PN→KC sparse graph;
- KC→MBON support/count scales;
- fixed byte→MBON codebook;
- MBON recurrence matrices.

## Learned persistent state

Only:

\[
\theta_{ij}
\]

for existing KC→MBON edges.

For raw support this is:

\[
15,523
\]

float values.

Float64 payload:

\[
15,523\times8
\approx124\text{ kB}.
\]

This is substantially smaller than recent 2–4 MiB learned slow matrices, but no result may be described as better solely because it uses less memory.

## Transient state

- PN activations;
- KC activity;
- MBON state;
- recurrence state;
- probe buffers.

No learned state may grow with lifetime.

---

# 26. Implementation requirements

The runner should be one self-contained `.py` file plus optional Mac launchers.

It must embed the extracted BANC arrays or a compressed literal representation.

It must not require the raw Feather files on the Mac.

## 26.1 Common computation

One seed/load worker should share across arms:

- stream;
- context vector;
- context→PN activation;
- raw PN→KC drive;
- sparse KC code;
- byte codebook;
- target byte;
- probe schedule.

Only KC→MBON learned parameters and recurrent states differ by arm.

## 26.2 Mac M3

Default:

- 4 `spawn` workers;
- BLAS/Accelerate threads = 1 per worker;
- `caffeinate -dimsu`;
- atomic per-job cache;
- rolling uncompressed checkpoint if needed;
- completed jobs reusable;
- source/design/graph hashes included in cache identity.

## 26.3 Engineering firewall

Before scientific execution:

1. compile;
2. raw graph integrity self-test;
3. codebook self-test;
4. exact degree/count-control audits;
5. smoke run on non-scientific seeds;
6. forced interruption/resume equivalence;
7. 2-worker/4-worker determinism;
8. one full-size engineering load-256 trajectory;
9. no science seed inspection;
10. verify all arms receive same stream and presentation count;
11. verify fixed/learned state accounting.

---

# 27. Required output files

```text
SUMMARY.md
GATES.json
MANIFEST.json
PROTOCOL.md

RAW_GRAPH_AUDIT.csv
CODEBOOK_AUDIT.csv
KM_ARM_AUDIT.csv
MM_ARM_AUDIT.csv

TERMINAL.csv
CHECKPOINT_ACQUISITION.csv
COHORT_RETENTION.csv
REVISION.csv

KC_GEOMETRY.csv
MBON_STATE_GEOMETRY.csv
INTERFERENCE_SUMMARY.csv
UPDATE_NORM_SUMMARY.csv

STATE_COSTS.csv
RUNTIME.csv
SAMPLE_BUDGET.csv

jobs/
checkpoints/
```

The final upload bundle should exclude large rolling checkpoints.

---

# 28. Interpretation rules

## If RAW-ALIGNED beats KC-ID-SHUFFLE

This supports a specific raw-BANC claim:

> the identity alignment between sensory PN→KC expansion and KC→MBON plastic targets contributes to persistent-learning performance.

It does not prove the biological plasticity rule.

## If RAW-ALIGNED matches KC-ID-SHUFFLE

The joint KC identity alignment is not supported in this abstraction.

Do not tune the input adaptor repeatedly to rescue it.

## If RAW beats KC-ID-SHUFFLE but matches degree-rewire

The relevant structure may be degree/statistical rather than detailed raw support.

## If raw support works but binary/shuffled counts match

The support topology matters, but raw recorded-count organization does not add detectable value.

## If raw counts beat binary and shuffled counts

The quantitative weak/strong organization present in the raw graph contributes beyond support alone.

## If RAW-MM beats rewired recurrence

The actual MBON recurrent organization contributes beyond generic recurrence.

## If generic recurrence helps but RAW-MM matches rewire

Recurrence as a computational primitive may help, but the raw recurrence topology is not supported.

---

# 29. No-drift stop conditions

V49 is **not** a hyperparameter search.

If Block A fails:

Do not sweep:

- KC sparsity;
- context→PN adaptor size;
- learning rate;
- count exponent;
- codebook dimension;
- edge thresholds.

Record the substrate result.

Block B may still test recurrence independently if acquisition remains valid.

If Block B fails:

Do not sweep:

- recurrence gain;
- recurrence depth;
- spectral radius;
- recurrence time constants;
- MBON subsets.

Record the motif as unsupported in this abstraction.

If both Block A and Block B fail raw-vs-control specificity:

> stop claiming that these raw graph motifs, at this resolution and with this simple local learning rule, provide the missing persistent-learning mechanism.

Then return to a **learning-rule question** while keeping topology simple, rather than continuing to add raw motifs.

If one or both pass:

> only then consider a later graph-distilled KC/MBON/DAN core, with motif deletion controls.

---

# 30. What V49 can establish

A positive Block A can establish:

- a fixed sparse raw-BANC KC→MBON graph can act as a bounded synaptic memory substrate;
- the actual alignment with PN→KC sensory wiring matters beyond graph marginals;
- raw support and raw recorded-count organization can be separately evaluated.

A positive Block B can establish:

- raw MBON recurrence supplies useful dynamic state beyond feed-forward MBON output;
- any benefit is specific to raw recurrence organization rather than generic recurrence.

V49 cannot establish:

- that raw recorded counts are physiological synaptic strengths;
- neurotransmitter signs;
- the true fly plasticity rule;
- DAN credit assignment;
- APL physiology;
- whole-fly cognition;
- unlimited lifetime memory;
- language learning.

---

# 31. Pre-run qualitative predictions

The primary predictions are:

1. `RAW-ALIGNED` will outperform `KC-ID-SHUFFLE` if the PN→KC/KC→MBON identity alignment carries useful structure.
2. If the graph's detailed support matters beyond degree, `RAW-ALIGNED` will beat `KM-DEGREE-REWIRE`.
3. If raw count heterogeneity matters, RAW will beat both `KM-WEIGHT-SHUFFLE` and `KM-BINARY`.
4. Raw MBON recurrence will improve persistence or reduce required KC→MBON update norm only if its actual organization carries information beyond generic recurrence.
5. A failure of these predictions closes the corresponding raw motif at this abstraction; it does not trigger parameter tuning.

---

# 32. Recommended execution order

1. Build/extract/embed raw PN→KC, KC→MBON and MBON→MBON arrays.
2. Run engineering firewall.
3. Run Block A at load 256 first.
4. Complete Block A load 128.
5. Run Block B if RAW-ALIGNED acquisition validity is satisfied.
6. Aggregate Block A and B separately.
7. Only after reading both verdicts decide whether a later graph-distilled core is justified.

This staging preserves attribution and avoids constructing a complex digital-fly model before its component motifs have earned inclusion.
