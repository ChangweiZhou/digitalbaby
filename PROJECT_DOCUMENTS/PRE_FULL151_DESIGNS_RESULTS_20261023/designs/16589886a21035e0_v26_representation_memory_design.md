# V26: testing the representation beneath persistent memory

Implementation design, 7 September 2026. This document describes the implemented independent runner and fixed roster, not results from its default six-seed experiment. It incorporates the separately uploaded Claude V26 source and its completed screening results. The final verification record below reflects the shipped source.

## 1. Goal and revised recommendation

The project's goal remains a non-Transformer learner that acquires useful knowledge from sequential experience, retains it through intervening learning, and revises it without full retraining. Unique observations and total presentations are primary resources. Synthetic recall, structured transfer and context are mechanism tests toward that goal.

V26 asks two connected questions:

1. Which simple sensory and temporal representations expose useful information to banked local learning?
2. Can a local representation update improve learning on a changed sensory distribution while old associations remain accessible?

The default memory remains eight separately writable banks with ordinary-rate supervised writes. The representation itself is now open to a broad comparison: sparse random afferents, balanced or sensory-selected afferents, direct channel-preserving temporal features, denser activity, simpler linear processing, and explicit two-branch computations. The developed E3 encoder remains an historical reference.

The current ELMCell is not assumed to equal a biological neuron or to be the indispensable unit of computation. Useful computational constraints are bounded local state, an explicit local update pathway, sparse or otherwise accounted connectivity, a defined teaching pathway, and separately measurable storage. The primitive may be a leaky feature or a small branch calculation if the experiment earns its additional cost.

The user has authorized broad testing. Accordingly, V26 broadens the representation experiment while preserving fixed contrasts and a fixed replication subset. It does not reopen the previously screened storage families, add replay or representation transport, introduce global backpropagation, or declare a hierarchy learned merely because projections are stacked.

## 2. What carries forward from V25

The previous review reported all 348 V25 trajectories complete. Its strongest design implications are these:

| V25 result | V26 response |
|---|---|
| Minimal temporal encoding was competitive with E3 at substantially smaller stored numeric cost | Retain both references; make minimal features the economical starting point |
| Ordinary writes learned much faster after one labeled example than slow writes | Ordinary writes are the baseline; slow writes are a measured retention tradeoff |
| Freezing addresses rescued a substantial portion of live-encoder loss | Retain a controlled address/content 2×2 experiment with fixed routing prototypes |
| Changing content could still damage memory with addresses fixed | Require content-drift diagnostics and old-memory evaluation |
| Stable/plastic populations mitigated live losses | Test the same concept in a simpler locally learning representation |
| The sensory gate had no convincing update-budget control | Add random, reduced-rate, same-state proposal-norm and shuffled-gate controls |
| Blended reading improved structured transfer | Record hard and blended predictions before the same supervised write |
| FIFO recovered delayed learning under ordered complete outcomes | Retain an ordered control and explicitly violate the ordering prior |
| Int8 worked in tested frozen-E3 settings | Test the new minimal/int8 combination directly |

These are reasons to conduct the comparisons. They do not establish that storage is solved, that minimal features are universally superior, or that the new combinations already work.

## 3. Corrections to Claude and the other-thread plan

### Z-scoring was already present

The V25 minimal encoder computed a frozen-calibration z-score, clipped it, and selected the largest absolute values separately in two 48-cell populations. A supposed new “z-scored top-k” arm would duplicate the baseline.

V26 therefore retains `z_topk` as its reference and introduces **raw** top-k as an explicit ablation. Dense and soft-threshold activity remain separate tests. The raw arm changes standardization and clipping as well as which activity magnitudes win; it is interpreted as that stated package.

### The old calibration distribution did not reveal the structured signal

V25's 64 minimal calibration observations came from an isotropic sensory generator. The unusually high-variance informative coordinate was added only in the structured task. It could not be discovered from those old calibration observations by variance ranking.

V26 keeps legacy calibration intact for the legacy tasks. New geometry tasks receive 64 separately generated, counted, unlabeled observations from their initial task distribution. All compared custom encoders receive the same calibration observations. Variance-selection arms compute generic sensory variance over all ticks and coordinates; they receive neither a target coordinate nor clean latent patterns.

Variance selection and feature calibration require separate processing of those observations. They reuse the same 64 unique observations but incur two observation evaluations; that work is accounted. The shift tasks calibrate on A only and provide no B observations before the shift.

### Seventy-five percent is not the measured encoder's ceiling

For perfectly known but unordered signs with balanced labels 00, 01, 10 and 11, the ideal expected accuracy is 75%. This arithmetic does not imply that order is worth only 4.15 points because an imperfect ordered-input baseline scored 79.15%. Neither the empirical ordered score nor 75% bounds the contribution of order in the tested learner.

V26 includes a balanced order-only task using 01 and 10, where an order-blind predictor has an ideal 50% benchmark. Its four-output memory interface is retained; only two labels occur. Swapped-input probes also measure whether the same trained learner uses order. The dense observable-input diagnostic remains an empirical, task-informed comparator, never an “observable ceiling.”

### Salience is a hypothesis, not a generic learning solution

The original informative coordinate has higher variance than its nuisance coordinates. Variance selection can exploit that covariance structure legitimately, but that does not make it task-independent representation learning. V26 tests equal coordinate marginals, a distributed informative direction, and nuisance with greater variance than the signal. Useful information extraction must be distinguished from following a convenient generator statistic.

### FIFO and optimistic reward require precise claims

FIFO can assign complete outcomes that arrive in originating-event order. It cannot in general infer the correct origin after unmarked outcomes arrive out of order. Both regimes are now explicit.

A common optimistic offset applied to every action leaves the immediate argmax unchanged. The V25 optimistic learner benefits through changed action-value errors and subsequent learning, not a separate instantaneous action-selection effect. V26 compares zero-prior learning, optimistic learning, and increased epsilon without claiming that these are identical interventions.

### The separately completed Claude V26 screen supplies one important positive result

The subsequently uploaded `elm_fly_representation_v26.py` and results archive are a distinct experiment. Their source hash matches their manifest, and the archive covers 40 configurations × six seeds = 240 completed screening trajectories. It contains no independent confirmation results. Its screening seeds are 401, 419, 433, 449, 461 and 479.

The strongest recoverable finding is a simple residual temporal filter, emitting `2*fast - slow`. In that implementation it improves structured blended accuracy from 63.05% to 73.88%, a paired screening difference of **10.84 pp [6.49,15.19]**, winning all six seeds. Association hard-read retention rises by **2.36 pp [0.86,3.86]**, with approximately unchanged prequential accuracy. Its report's gate excludes the temporal arm by name and therefore fails to surface this important comparison.

This is a reason to include a precise residual-filter comparison before running our independent experiment. It weakens a strong assertion that only afferent dilution matters. It does not prove that the improvement specifically comes from order: that archive lacks the corresponding paired swapped-order controls and independent replication.

The residual filter is added to the new runner's geometry, robustness, live-learning and fixed-confirmation panels. It is a **controlled mechanism replication**, not an attempt to duplicate Claude's entire baseline package, which changes random initialization, clipping and population normalization relative to V25. Our existing pure fast-minus-slow filter remains a distinct arm.

The audit also identifies why the uploaded adaptation claims do not close the representation/storage problem:

| Uploaded implementation finding | Consequence | Corrected independent design |
|---|---|---|
| Its recurrence metric pools all 2,040 nonfirst appearances | Reported 91–98% is not learning after one example | Count exactly one second lifetime appearance per identity |
| Its shift schedule includes 406–431 A presentations after B begins | Old knowledge is still being trained | Assert zero A training presentations after the declared transition |
| Its A probes use new latent identities conditional on labels | Old-cohort performance measures rule transfer, not arbitrary old-association recall | Preserve same-latent old noisy probes and a separate new-unseen B set |
| A blank input can alter live weights through initial renormalization | Frozen/live differences include gain correction unrelated to a sensory learning proposal | Preserve matched initial row norms; a zero proposal cannot normalize them to new magnitudes |
| Its stable cells share global competition and normalization with plastic cells | Stable weights do not create a stable feature partition | Independent 48-cell competition/normalization and frozen calibration |
| Its “normalized top-k” divides by SD after already z-scoring | That arm tests double variance weighting | Separate current z-score, explicit raw ranking and dense activity |
| Salience calibration remains isotropic and transformations affect only task inputs | The named task-variance hypothesis is not actually manipulated | Transform and account for initial-distribution calibration consistently |
| Export columns are fixed from a Block-A row | Per-seed old/new cohort endpoints are lost | Export the union of fields and retain all declared cohort endpoints |
| Confirmation requires an unrelated Block-A winner | Promising Block-B rows are never independently confirmed | Run the fixed confirmation subset regardless of screening scores |

The uploaded fully live shift arm's overall improvement is a real result under that implemented protocol, but it does not show retention of arbitrary old associations without replay. Its “Pareto improvement” gate also allows a retention loss within a margin and does not establish one-example sample efficiency. Its adaptation idea remains worth a corrected test.

The new roster is therefore **informed by V25 and the uploaded Claude screen, then fixed before any primary or confirmation outcome from this independent runner**. It is not claimed to have been selected before all prior experimental scores existed. Fresh seed sets avoid the newly discovered overlap with the uploaded screen.

## 4. Fixed experiment size and replication

The default roster has **199 primary configuration/task combinations × six seeds**, plus **26 predetermined confirmation combinations × six independent seeds**: **1,350 trajectories**.

Primary seeds are 601, 607, 613, 617, 619 and 631. Confirmation seeds are 641, 643, 647, 653, 659 and 661. Both sets are disjoint from the uploaded Claude screening seeds and from each other. Primary and confirmation results are identified separately. Configuration identity includes cohort, task and full parameters. No score from this runner is consulted when constructing either roster.

| Panel | Configurations per primary seed | Purpose |
|---|---:|---|
| Historical references | 9 | E3/minimal and ordinary/slow continuity on three legacy tasks |
| Precision | 3 | Minimal encoder combined with int8 storage |
| Geometry and temporal primitives | 69 | 23 fixed variants on three tasks |
| Robustness | 36 | Nine predetermined variants on four signal/order controls |
| Address/content causality | 24 | Two-pathway interventions under a sensory shift |
| Practical stable/plastic population | 12 | Three architectures on four shift tasks |
| Local update rules and gates | 12 | Six modifications on two shift tasks |
| Richer primitives under live learning | 18 | Test whether promising primitive classes survive adaptation |
| Observable-input diagnostics | 10 | Locate usable signal remaining in raw observed pulses |
| Delayed teaching | 3 | Immediate, ordered FIFO and unordered FIFO |
| Scalar reward | 3 | Zero prior, optimism and increased exploration |
| **Primary total** | **199** | |

The fixed 26-row confirmation subset includes four legacy association consolidation rows; five geometry rows on permuted structured transfer, including the residual temporal filter; four rows on equal-marginal transfer; three rows on high-variance nuisance; six practical/gate rows on structured shift; and the four random-afferent causal paths on association shift. Every confirmation comparator is included on the same confirmation seeds. This subset tests specific predetermined mechanisms, not whichever primary rows happen to score highest.

The run contains 2,766 declared comparison instances, counting per-seed comparisons and the hard/blended views of an identical trained state. These are not 2,766 independent experiments.

## 5. Shared representation and storage interfaces

Custom encoders emit 96 content coordinates. Their output is split into two 48-coordinate populations, each normalized independently, preventing the adaptable half from changing the stable half through a shared population norm. The default sparse code keeps 12 coordinates in each half. Features are signed engineering variables; the implementation does not call them literal nonnegative firing rates.

For the standard minimal primitive,

\[
d_j(t)=\tanh\!\left(\sum_{k\in C_j}w_{jk}x_k(t)\right),
\qquad
u_j(t+1)=(1-\alpha_j)u_j(t)+\alpha_jd_j(t),
\]

with \(\alpha_j=1-e^{-1/\tau_j}\) and taus 2, 6 and 16. Each observation is a defined 12-tick episode. Temporal traces reset at its beginning. Representation weights may persist and learn between and within observations; the temporal state does not persist across episodes.

Frozen calibration means and standard deviations define the default standardized activity. Top-k uses absolute standardized values after clipping to ±5. Population normalization is an explicit engineering operation. It is not an undisclosed learning rule.

The memory is the retained V25 bank implementation: eight banks, four target outputs, ordinary learning rate 0.6, and locally learned routing unless the contrast specifies fixed prototypes. Selected-bank supervision supplies the bank update. The hard view reads one bank and the blended view combines banks. Both views can be evaluated before the same supervised write without altering its learning rule. Slow references use 0.3. Main encoder comparisons use float64 stored readouts; int8 is a separately tested stored format with temporary floating arithmetic.

The scalar-reward and delayed-teaching comparisons use fixed routing prototypes to isolate teaching from address learning. The ordinary observable diagnostic uses one memory and thus differs in both representation and storage organization from a banked comparator.

## 6. Geometry and primitive panel

Each row below runs on legacy association, permuted structured transfer and permuted context: 23 × 3 = 69 configurations per primary seed.

| Variant | Manipulation |
|---|---|
| `random8` | Current eight-contact minimal reference |
| `random4` | Four random contacts per cell |
| `random16` | Sixteen random contacts per cell |
| `random32` | Dense random contact projection |
| `balanced8` | Exactly balanced population coverage, eight contacts per cell |
| `balanced16` | Balanced coverage, sixteen contacts per cell |
| `variance8` | Eight contacts sampled with calibration-variance bias |
| `variance_coverage8` | Variance bias tempered by accumulated population coverage |
| `axis` | One input coordinate per unit, three taus per coordinate |
| `raw_topk` | Select from unstandardized activity |
| `dense` | Standardized, clipped, normalized activity without top-k |
| `soft_threshold` | Signed shrinkage at standardized magnitude 1, then normalization |
| `single6` | A common tau of 6 |
| `wide4` | Tau set 1, 4, 16 and 64 |
| `contrast` | Fast-minus-slow temporal traces |
| `residual_contrast` | Twice the original fast trace minus a four-times-slower trace |
| `two_branch` | Two contact subgroups with an explicit nonlinear interaction |
| `branch_order` | Antisymmetric interaction between two branches' fast and slow traces |
| `linear` | Linear sensory activation in the standard leaky unit |
| `balanced8_dense` | Balanced contacts combined with dense activity |
| `axis_dense` | Channel-preserving temporal basis without top-k |
| `variance8_dense` | Variance-selected contacts without top-k |
| `variance_coverage8_dense` | Variance/coverage contacts without top-k |

Every variant is compared with `random8` on the same task and seed. Each combined dense variant also has an explicit comparison against its own sparse geometry parent. Thus improved contact selection is not automatically attributed to removing sparsification, or vice versa.

For the same fan-in, contact-geometry alternatives retain the same initial weight coefficients as the random baseline. Their topology differs; a separate random draw of coefficients is not silently added as another factor. Fan-in changes legitimately change contact count and stored state and are charged accordingly.

Balanced coverage constructs three shuffled Latin rectangles. Each input coordinate has exactly \(3f\) contacts across 96 cells at fan-in \(f\), with no within-cell duplicate contacts. This is a fixed engineering geometry rather than learned biological wiring.

Variance bias uses the empirical variance over all calibration ticks, adds a small uniform floor, and samples without replacement within each cell. The coverage variant divides the sampling weight by one plus each coordinate's accumulated degree. It is a simple coverage-control heuristic, not a trained anti-Hebbian circuit.

The `axis` basis assigns 32 coordinates to each of three taus with unit coefficients. It neither knows the useful coordinate nor isolates the two pulse windows by position. Its success would indicate that preserving input coordinates and temporal weightings can be useful without random mixing. It remains sensitive to how the task is represented, which is why the rotated task is included.

### Richer temporal computations

The contrast variant maintains tau pairs (1,4), (2,8) or (4,16) and emits fast minus slow activity. The distinct residual-contrast variant retains the original fast taus 2, 6 and 16, adds slow taus 8, 24 and 64, and emits twice the fast trace minus the slow trace. Both trace families are fixed sensory computations; neither is a learned temporal hierarchy.

The two-branch variant splits eight contacts into two groups of four. With branch activations \(b_1,b_2\), its instantaneous drive is

\[
\phi\left(\tfrac12(b_1+b_2)+\tfrac12b_1b_2\right),
\]

followed by the standard leaky trace. This tests an explicit nonlinear subunit calculation.

The branch-order variant gives each branch a fast tau-2 trace and slow tau-16 trace and emits

\[
q_j=a_{j,\mathrm{fast}}b_{j,\mathrm{slow}}
-a_{j,\mathrm{slow}}b_{j,\mathrm{fast}}.
\]

It has no residual linear feature. Its failure may therefore expose information lost by this pure interaction feature; it does not reject all possible dendritic units. Its four temporal traces cost more working state even though output width and total contacts are unchanged.

These richer sensory primitives are included because the user authorized breadth and the bedrock discussion raised their relevance. They do not introduce another layer of write protection, hundreds of banks, a scratchpad, or a claim of biological simulation.

## 7. Signal and order robustness

The following nine variants are fixed before this runner's outcomes: random8, variance/coverage8, axis, axis-dense, dense, contrast, residual-contrast, branch-order and balanced8. Each runs on four tasks: 36 configurations per primary seed.

| Task | What changes | Interpretation |
|---|---|---|
| `structured_equal` | Every input coordinate has the same signed-uniform marginal law, including higher moments | A variance rule cannot identify the label coordinate from population marginal statistics |
| `structured_rotated` | Apply one dense orthogonal rotation to the informative and nuisance directions | Tests a distributed signal, not coordinate-0 selection |
| `structured_nuisance` | Nuisance SD is 1.3, greater than the informative coordinate's SD | Tests whether salience follows large irrelevant variation |
| `order_only` | Only balanced 01 and 10 labels occur | Clean dependence on temporal order; ideal unordered benchmark 50% |

The ordinary permuted structured task uses a fixed random input-coordinate permutation per seed. The learner never receives the permutation or the informative direction. A constant hidden coordinate convention therefore cannot be hardcoded by the encoder.

The informative pulse coordinate has a balanced sign and uniform absolute amplitude in [0.75,1.15]. The original nuisance SD is 0.65. Equal-marginal nuisance follows the same independent signed-uniform law as the signal, rather than merely matching its variance while leaving another obvious marginal cue. The four-output label is twice the first sign category plus the second sign category.

Training and ordinary transfer probes in these structured tasks have disjoint latent pulse identities. A high score must transfer to new pulse nuisance patterns. Refresh phases retain the stationary shared rule and are not called label revision.

Permuted context inserts an unrelated distractor between the first context pulse and the late cue. Controlled probes change the early context while retaining the same late cue and distractor. Paired context scores, both-correct rates, and history-erasure controls are reported. This remains within a defined 12-tick episode; unsegmented cross-episode working memory is outside what this task demonstrates.

## 8. A locally learning representation

The custom live encoder modifies existing sensory contact weights only. It does not rewire contacts, learn tau, train a private MLP, receive association labels or apply backpropagation. Its local proposal is a normalized Oja-like rule.

For cell \(j\), retain its initial weight norm \(r_j\). Let

\[
\widehat w_j=w_j/r_j,
\qquad
\widehat x_j=x_{C_j}/\max(\|x_{C_j}\|,\epsilon),
\qquad
y_j=\widehat w_j^\top\widehat x_j.
\]

The proposed unit-weight change is

\[
p_j=y_j(\widehat x_j-y_j\widehat w_j).
\]

For eligible cells passing a declared gate,

\[
w_j\leftarrow r_j\,
\frac{\widehat w_j+\eta s p_j}
{\|\widehat w_j+\eta s p_j\|},
\]

where default sensory learning rate \(\eta=0.001\), and \(s=1\) except for the stated rate or proposal-norm control. Numerical denominators have explicit small positive floors.

Normalizing each local sensory vector makes the rule bounded but changes the statistics it emphasizes. It is not identical to unconstrained covariance PCA on raw inputs, and it does not guarantee selection of the label-relevant direction. Representation benefit is an outcome to measure, not a premise.

At each sensory tick, the current weight values generate the activity first, then the unsupervised proposal updates weights. Later ticks in the same episode can use those updated weights. Supervised online predictions occur before the memory receives the current label, but after this within-episode sensory processing. This timing is intentional and identical across the declared live comparisons.

Competitive Oja retains the same proposal but permits only the largest absolute local linear responses in each 48-cell population to update, intersected with the plastic-cell mask. It is an explicit population competition operation and a different sensory update rule; it is not claimed to optimize the downstream target loss.

### Gates and their controls

On the stable/plastic architecture, the following are tested on association shift and structured shift:

| Modification | Exact purpose |
|---|---|
| Competitive Oja | Change which local proposals are eligible |
| Novelty gate | Accept proposal magnitude above a cell's past EMA mean plus 0.5 EMA SD |
| Random gate | Accept eligible cells independently with probability 0.35 |
| Rate 0.35 | Apply all eligible proposals at 0.35 times the sensory rate |
| Proposal-norm control | Scale all eligible proposals to the Frobenius norm of the same-state hypothetical novelty-selected proposal |
| Shuffled gate | Preserve the same-state novelty gate's number of accepted eligible cells but randomly assign which eligible cells receive them |

EMA decay is 0.98. The novelty decision precedes the EMA update for the current observation. A cell's first observed proposal is accepted. Random streams are explicit state and included in the memory ledger.

The random and rate controls are predetermined approximations, not matched by inspecting outcome scores. The proposal-norm control matches the submitted weighted proposal before row renormalization at its own current state. Different trajectories can produce different future proposals and different actual applied weight changes. The shuffled control matches its own current gate count, not the entire novelty arm's future sequence. Actual proposals, accepted counts, nonzero changes and applied norms are reported.

## 9. Sensory shift and memory compatibility

The shift tasks start with old arbitrary-label associations A, then introduce new identities B in a new sensory subspace. No task identifier, phase identifier, hidden identity, inverse transform or future example is supplied to a learner.

A and B have disjoint eight-dimensional high-variance subspaces, placed behind the same per-seed random input permutation. Background SD is 0.12. The ordinary active-subspace SD is 0.65. Calibration contains independent A observations only.

At default sizes, 72 old identities are acquired and some undergo genuine revision before the shift. The subsequent 192 new identities arrive in eight waves. Once B begins, A has **zero further training presentations**. A is accessible only to the evaluator's read-only probes.

| Task | B learning problem |
|---|---|
| `shift_association` | Arbitrary identity associations; genuine B-only revisions |
| `shift_structured` | Stationary two-sign rule, with independent B transfer probes |
| `shift_structured_equal` | Same rule, but equal marginal laws across the eight active B coordinates |
| `shift_structured_nuisance` | Same rule, with nuisance larger than the informative B coordinate |

Ordinary shift probes are new noisy renderings of the same latent identities, supporting exact association recall. Structured shift additionally has `new_unseen`: independent B latent episodes, never used for development or training. At default sizes this extra set has 48 label-balanced identities with three independent probe variants. It measures transfer beyond B identity storage.

The favorable shift creates a circumstance where local sensory adaptation could help represent a newly active subspace. It does not guarantee success. If Oja changes weights but never improves new-unseen performance or acquisition per example, its old-memory damage should first be interpreted as ineffective representation learning on this task.

### Causal 2×2

All four paths retain and process an identical frozen/live encoder pair. The live branch learns even when its output is not used. This equalizes duplicate-encoder cost and sensory exposure.

| Path | Address features | Content features |
|---|---|---|
| FF | Frozen | Frozen |
| LL | Live | Live |
| FL | Frozen | Live |
| LF | Live | Frozen |

Routing prototypes are fixed in these causal comparisons. Each path uses the same bank shape, memory learning rate and branch initialization. FL and LF are compared both with FF and with LL.

The 2×2 is run for random8/Oja, balanced16/Oja and random8/competitive Oja, on both association shift and structured shift: 3 × 4 × 2 = 24 configurations. This crosses the routing/content hypothesis with a modest change in input coverage and local learning rule without a storage tournament.

### Practical single-population models

Each practical model uses 48 address coordinates, all 96 content coordinates and a locally learned router. The three conditions are fully frozen, all 96 cells plastic, or first 48 stable and last 48 plastic. Output calibration and the two population normalizations remain fixed and separate.

All three run on the four shift tasks: 12 configurations. The stable scaffold is compared with both frozen and all-plastic controls. A learned prototype can still change its address assignment even when the address coordinates are stable; stable coordinates must not be confused with fixed routing decisions.

Additional live-bedrock tests use contrast, residual-contrast and branch-order primitives, each frozen and stable/plastic, on the two main shift tasks. Balanced16 also receives frozen, fully live and stable/plastic comparisons. These 18 rows prevent a promising static primitive from being adopted without examining what happens when its afferents learn.

No memory is silently moved, replayed, re-indexed, transported to a new feature basis or allocated by evaluator identity. The old-memory-access problem must be solved or exposed by the actual learning trajectory.

## 10. Observable-input diagnostics

Ten diagnostics copy observed ticks 0 and 5 into 64 ordered dense coordinates, pad to width 96, normalize, and use an ordinary one-bank local memory. They run on legacy structured/context; permuted, equal, rotated and nuisance structured tasks; order-only; and the three structured shift tasks.

This diagnostic knows the observable pulse positions. It does not receive clean pulses, the informative coordinate, target signs, the inverse permutation/rotation, or extra training labels. Under a rotation it sees the rotated observations.

Because it changes geometry, sparsity and storage organization, its comparison with a banked minimal encoder does not isolate one mechanism. It estimates what this particular task-informed local readout can obtain directly from the observations. A high score motivates representation work; a low score limits how a failure of another encoder should be interpreted. Its score is never a Bayes limit or an empirical ceiling.

## 11. Teaching and reward consolidation

These six rows use the frozen legacy minimal encoder and fixed bank routing on the association stream. They hold representation learning out of the attribution problem.

### Categorical feedback

- Immediate feedback is the reference.
- Variable-delay FIFO receives complete outcomes delivered in originating-event order, including FIFO tie order.
- Unordered variable-delay FIFO receives the same type of outcomes without enforcing originating-event order.

The learner has a bounded 32-slot trace store and no external identity pointer. The evaluator may retain identities to diagnose whether the FIFO mechanism selected the true originating event; those identities are not supplied to the learner.

The unordered comparison is a limitation test of the attribution prior. It does not claim to implement a solution to unmarked out-of-order credit. An outcome delivery-time schedule is part of the environment, not an inferred property of arbitrary sequential data.

### Scalar reward

- Existing zero-prior reward learning, epsilon 0.1.
- Optimistic action-value prior, epsilon 0.1.
- Existing zero-prior learning, epsilon 0.3.

All receive the same number of scalar outcomes. Increased epsilon changes exploration frequency. Optimism changes the chosen-action prediction error and therefore subsequent learning. These are separate interventions and are not pooled as one exploration effect.

Reward metrics include reward actually earned online, late acquisition/revision and final recall. A final-retention gain cannot replace the objective of earning useful reward while learning.

The teaching/reward panels act and learn through their declared hard behavior view. Their online prequential and reward figures describe that actual behavior and are shared when final memory is reported under multiple retrieval views. They do not represent independently acting blended policies. In contrast, the standard supervised encoder panels explicitly measure both hard and blended predictions before every shared target-driven write.

## 12. Endpoints and decision rules

### Primary endpoints

1. **Prequential accuracy:** every prediction is recorded before its supervised memory update. In standard supervised panels, hard and blended views are evaluated separately on the same state. Teaching/reward online figures describe the actual hard behavior policy, as specified above.
2. **Next-recurrence accuracy:** one prediction per identity at its second stream appearance, after one prior labeled appearance. At default acquisition settings the two appearances are distinct noisy renderings and precede label revision.
3. **Final current-label recall:** held-out noisy observations of learned identities evaluated after intervening learning.

The primary V26 report uses the explicit field `terminal_next_recurrence_accuracy`. The older phrase “second-presentation accuracy” refers to the interpretation above, not a second independently retrained experiment. Counts accompany this metric.

Late acquisition and genuine revision endpoints remain useful but are secondary to the learning-speed evidence: reaching high accuracy after six presentations does not establish fast learning from one example. Structured same-rule refreshes are not called revision. In `shift_structured` and its equal/nuisance variants, `late_revision` refers only to the genuine **pre-shift A revisions**; B uses a stationary rule and has no revision endpoint. `shift_association` includes genuine B-only revisions after the shift. These meanings must not be pooled as if both measured late adaptation to changing B labels.

### Shift promotion

A live representation candidate requires all of the following against its matched frozen comparison:

- A positive paired advantage on B `new_unseen` where that endpoint exists.
- Old A recall within a predeclared five-percentage-point noninferiority margin.
- Competitive learning per presentation, including new B prequential/recurrence measurements.
- Explicit accounting of learned state, duplicated encoders, calibration and sensory update work.
- No A training replay or evaluator-to-learner leakage.

New-rule transfer and old exact recall must be evaluated jointly. A method that wins one while silently abandoning the other is a tradeoff. Association-shift results without a new-unseen task cannot by themselves pass the structured representation-benefit requirement.

Terminal aggregate accuracy can conceal old/new tradeoffs, so the report separates old-current, new-current and new-unseen outcomes. No method is promoted solely for staying below an arbitrary route-change threshold.

### Paired uncertainty and confirmation

Contrasts use seed-paired differences with Student 95% intervals over independent seeds. Intervals are descriptive and unadjusted for the broad screen. An inconclusive noninferiority interval is not a demonstrated failure; a borderline screening win is not robust superiority. Confirmation repeats the predeclared subset on new seeds and is reported separately.

The reference, slow-write and precision comparisons explicitly gate terminal recall, prequential accuracy and next-recurrence accuracy together, using a five-point margin. They do not silently reuse only phase-end plasticity gates. The other scientific contrasts remain descriptive comparisons, with joint shift decisions handled explicitly in the report.

## 13. Mechanistic diagnostics and accounting

The evaluator records representation changes for the same observations, route changes, old-memory predictions, and finite bank write effects. Adjacent-phase causal probes cross old/new address coordinates and old/new content coordinates under frozen previous bank weights. The remaining write effect is evaluated separately.

These are ordered adjacent-phase MSE interventions, not percentages of the final accuracy gap or a unique global decomposition of forgetting. The problematic V25 adaptive-prototype same-route displacement diagnostic is removed rather than reused as if it were valid.

Encoder diagnostics include input coverage, contact counts, calibration statistics, activity saturation, standardized clipping, feature-zero fractions, eligible cell-ticks, accepted gates, nonzero updates, proposal norms, applied-change norms and declared random state. An open gate is not assumed to be a nonzero update.

The resource ledger distinguishes:

- Unique rendered sensory observations and repeated labeled presentations.
- Unlabeled calibration identities, calibration observation evaluations and E3 developmental presentations.
- Numeric retained state, mutable versus fixed arrays, contact indices and gate random state.
- Extra transient temporal traces and duplicated frozen/live branches.
- Learner sensory computation, bank updates and evaluator probe computation.

At default sizes a standard stream has 264 training identities, two fixed noisy training renderings per identity, three passes per acquisition phase, and 2,304 labeled presentations. These are 528 unique rendered training inputs, not 2,304 unique examples. The three probe variants are evaluation only. Legacy E3 additionally retains its counted 64-identity, six-pass local development and calibration cost.

Cached sensory trajectories share evaluator computation across compatible bank/path configurations. This is permissible because encoder updates use sensory observations only, not labels or memory errors. Every simulated learner is charged its own conceptual processing and retained state; cache reuse is not described as zero-cost learning. Cache size is bounded, and losing it can require recomputation of an unfinished trajectory without changing completed results.

## 14. Interpreting possible outcomes

| Outcome | Supported next conclusion |
|---|---|
| Coverage/variance changes help only where signal variance is distinctive | A useful task-specific salience rule, with limited generality |
| Dense codes help across several geometries | Current competition/sparsification discards relevant information on those tasks |
| Channel-preserving traces are strong, but weaken after rotation | Unmixed coordinates help when the input basis exposes structure; representation robustness remains open |
| Temporal interactions improve order-only and paired context tests | Added temporal computation earns evidence on a task that needs it |
| A richer primitive wins static tests but loses after local adaptation | Static information extraction and continual representation compatibility remain separate problems |
| Local adaptation improves B transfer but damages A | A genuine representation/storage tradeoff has been demonstrated |
| Stable addressing rescues LL but content still damages A | Both access pathways remain relevant, consistent with the V25 causal result |
| No local rule improves B over frozen features | The tested representation-learning rule/task pairing did not earn a benefit; do not declare storage coupling solved or impossible |
| A live candidate improves B and preserves A with good online learning | A substantive step toward persistent representation learning, limited to the tested tasks and resources |
| Unordered FIFO fails | The ordered-feedback prior matters; no claim that arbitrary delayed credit is solved |

V26 does not provide a scaling law, a language model, unsegmented long-horizon working memory, an anatomical fly simulation or unbounded exact memory. Those limitations are part of interpreting the implemented experiment, not reasons to stop the authorized broader representation test.

## 15. Single-file execution and review artifacts

The deliverable is one self-contained Python runner plus this design. The runner retains automatic macOS `caffeinate`, exact source/configuration identity, completed-job resume, a fixed manifest and a compact results ZIP. The intended default command is:

```bash
python3 elm_fly_representation_memory_v26.py --install-deps
```

The runner writes the source, configuration, run completeness, per-seed endpoints, declared paired contrasts, separate confirmation rows, joint shift decisions, causal and control diagnostics, and numeric-state/exposure accounting. Completed jobs are reused on resume; an interrupted individual job can be recomputed.

To resume, run `python3 elm_fly_representation_memory_v26.py --resume auto` from the original working directory, or supply the specific output folder. Successful jobs are preserved; interrupted jobs restart. Resume rejects a different source hash or scientific configuration. `--list-jobs` prints the fixed roster, and `--panels` can limit execution when desired. A filtered run reports absent controls explicitly.

Results include `REPORT.md`, `SUMMARY_BY_SEED.csv`, the full compressed metric and diagnostic tables, declared paired comparisons, `COMPATIBILITY_BY_SEED.csv`, `COMPATIBILITY_GATES.csv`, cost/development ledgers, the manifest, configuration, source and run-completeness records. The compact ZIP excludes the larger individual job records retained in the run folder. Upload that ZIP for review.

The broad roster is the default. A smoke configuration is only an execution check. Optional panel/arm selections are explicit and must report missing declared comparisons instead of inventing a matched parent. The complete default experimental result remains for the user to run unless explicitly reported otherwise.

## 16. Verification record

The shipped source passed its deterministic mechanism, information-boundary, runtime and reporting checks, including 41 encoder variants and the residual-filter arithmetic. All **199 primary configurations** completed the small workflow. A deliberate stop after seven completed jobs and resume preserved those jobs and completed the remaining 192.

All **199 configurations also completed the full 2,304-presentation stream on validation seed 251, with zero failures**. This includes every implemented live rule, causal path, task, counterfactual probe, teaching condition and reward control. Each trajectory recorded exactly 264 second lifetime appearances, rather than pooling all repeats. The custom minimal association baseline matched the legacy minimal baseline exactly.

Seven retained V25 references reproduce checkpoint accuracy across all 21 phases and both saved read views, plus all available hard-view prequential and second-appearance measurements. These are continuity checks, not fresh estimates of candidate superiority.

All 34 declared full-size structured-shift compatibility pairs retained their old-current, new-unseen, new-prequential and new-second endpoints. With only one validation seed, their joint gates correctly remain inconclusive. The full-size archive retains these per-seed endpoints and the compatibility tables. A final packaging-only change adds an exact copy of the executing source to future archives; source-copy integrity, all 199 small configurations and resume were rechecked on the shipped package. All scientific code is byte-identical to the full-size-validated version. No pilot score was used to choose a winning configuration or revise the new primary/confirmation roster.

The source's SHA256 is `1263a33ea101036ce267b0226269bef4b251f5d575745c8f7486b5e71792a22b`.

The **full 1,350-trajectory primary-plus-confirmation experiment remains for the user's run**. Validation seed 251 and small-workflow seed 17 are separate from both study cohorts. Execution checks establish implementation coverage; they do not establish that a scientific candidate works. Automatic macOS `caffeinate` launch and cleanup were checked with a mocked macOS environment; the actual execution environment for validation was Linux.

## 17. Provenance

This design merges the uploaded `v26_experiment_recommendation(1).md`, the recovered `v26_merged_research_plan.md`, the complete V25 review and the shipped V25 implementation, plus the subsequently uploaded Claude V26 source/results and the independent `claude_source_audit.md` / `claude_results_review.md` calculations. Disagreements are corrected by source and saved-data inspection. Its exact runnable roster is generated by `v26_jobs`; `v26_design_policy` records its principal interpretations in the manifest.

The uploaded Claude source SHA256 is `159ed6992e37e99d591f82e37aaca06703541f5509f30ccb817baad89e5c1f7b`. Its V25 dependency hash is `eafb1cf48e5014619a13669b10cc6ce327cdeb20d8fa718723f7cf967e5e56b3`. The independent runner has a different filename and source identity; the two experiments' trajectories and gates must not be merged under one version label.

The design's mathematical update rules and task definitions describe this implementation. They are not claims that a cited biological mechanism has been faithfully reproduced, nor that this version implements every previously discussed alternative.
