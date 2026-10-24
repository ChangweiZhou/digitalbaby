# V51 — Frozen-core scope and eligibility confirmation

**Experiment namespace:** `V51-SCOPE-ELIG`  
**Status:** refined design for implementation; no V51 trajectories have been run  
**Date:** 12 September 2026  
**Platform:** Apple Silicon Mac M3; one self-contained Python runner and one-command launch  
**Evidence:** uploaded V50 results/source, uploaded OMNIBUS10.1.1 results, the other-thread OMNIBUS10 source/design, and the user's supplied Claude review. No external literature is used.

## 1. The objective and the decision this experiment serves

The project objective remains a minimal, bounded-state, biologically based fly-like persistent-learning core, materially informed by the supplied raw fly connectome. A particular version's reporting convention does not determine the project's future. This document specifies one experiment, not a new research constitution.

V50 established a useful **128-identity direct-association operating point**. OMNIBUS10 supplied a broad within-harness component screen. The next useful work is to preserve that reference and measure its scope, while independently checking the one small nominated addition—not to replace the system again.

V51 answers:

1. Does the unchanged V50 core retain its functional phenotype when direct-association load increases from 128 to 256 identities?
2. What happens at both loads in the existing serialized-byte harness?
3. Does the exact OMNIBUS10 multi-frame eligibility write vector improve the preserved reference on fresh seeds, and at what acquisition, retention, revision, and state costs?

**No V50 result is retroactively invalidated by failure at 256.** The appropriate conclusion would be that the demonstrated operating envelope is narrower than hoped. Likewise, a successful direct-association core is not a complete byte predictor.

### The recommended design in one table

| Axis | Levels | Interpretation |
|---|---|---|
| Task/interface | `CUE8`, `BYTE256` | Two existing task bundles; not a pure serialization intervention |
| Identity load | 128, 256 | Four cohorts of 32 or 64 identities, eight acquisition presentations per identity |
| Learning configuration | `V50-REF`, `ELIG-WRITE` | Same circuit and reader; final KC code versus exact omnibus multi-frame eligibility vector for writing |
| Replication | 16 fresh seeds | Seeds are the inferential units |

There are **64 seed × task × load jobs**, each containing both candidates and the causal controls. The two candidates account for **128 learned trajectories**, not 128 independent replications.

## 2. What the source audit changed

This section distinguishes checked results from proposed decisions. All numerical reanalyses use the current 112 completed omnibus job files / 3,920 trajectory rows; archived `.stale-*` jobs are excluded. Intervals below are descriptive paired Student 95% intervals over 16 seed-level contrasts.

### 2.1 Conditional factorial results

For A, B, C, and E, average the on-minus-off contrast within each seed over the seven cells and eight backgrounds with `D_LOCAL=0`. For D, average over seven cells and sixteen remaining backgrounds. Positive loss differences are worse.

| Factor | Old untouched-A held-out loss change, bits | D held-out loss change, bits | Number of background means improving old-A loss |
|---|---:|---:|---:|
| `A_ELIG` | −0.07268 [−0.09920, −0.04616] | −0.07079 [−0.08483, −0.05675] | 56/56 |
| `B_CONSOL` | +0.02771 [+0.01862, +0.03680] | +0.10917 [+0.10375, +0.11459] | 0/56 |
| `C_FORGET` | +0.08673 [+0.08318, +0.09028] | −0.00567 [−0.00750, −0.00383] | 0/56 |
| `D_LOCAL` | +1.36554 [+1.20175, +1.52933] | +1.58250 [+1.51006, +1.65494] | 0/112 |
| `E_META` | −0.02639 [−0.03264, −0.02013] | +0.09913 [+0.09656, +0.10170] | 56/56 |

The 56 and 112 counts describe robustness across background configurations. They are **not independent hypothesis tests**, and they do not replace seed-level uncertainty.

Calling a factor “conditionally positive” because it interacts favorably with a strongly harmful factor is not sufficient for inclusion in the operating core. The selected core must be evaluated in its actual background.

### 2.2 The selected eligibility addition has a smaller effect than the factorial average

For `LEAK_HOM`, B=C=D=E=0, compare A1 directly with A0:

| Endpoint | A0 | A1 | A1 − A0, paired 95% interval |
|---|---:|---:|---:|
| Old-A held-out loss after D | 1.36948 | 1.33495 | −0.03453 [−0.05541, −0.01365] bits |
| Old-A held-out accuracy after D | 81.64% | 83.20% | +1.56 [−1.54, +4.66] percentage points |
| New-D held-out loss | 0.90207 | 0.86798 | −0.03409 [−0.04762, −0.02055] bits |
| Original answer after one corruption | 91.41% | 96.09% | +4.69 [+0.56, +8.81] percentage points |
| Revision accuracy immediately after exposure 8 | 97.66% | 96.09% | −1.56 [−4.89, +1.77] percentage points |
| Revised-answer accuracy after subsequent D learning | 88.28% | 90.63% | +2.34 [−2.65, +7.34] percentage points |
| Mean acquisition prequential loss over A–D × exposures 1–8 | — | — | **+0.02727 [+0.02424, +0.03029] bits** |

Thus the nominated addition improves some final probabilities but is **not already demonstrated to improve acquisition sample efficiency**. The original V50 reference must remain in V51.

The omnibus A0 row is a within-omnibus control, not an exact rerun of V50: its cue sampling, rendering reuse, random-number streams, and revision presentation details differ. Cross-version means cannot substitute for paired comparisons.

### 2.3 Specific corrections to the supplied interpretation

- `D_LOCAL` is the raw-DAN-derived plasticity mask with capped magnitude compensation. It is **not** the covariance factor. `COV96` is a separate nonlocal oracle. The actual covariance oracle also worsened old-A loss in all 112 seed × cell comparisons, but its seed-averaged cost is **+0.37819 [+0.32712, +0.42926] bits**, not +1.36554.
- The reported ELM-versus-leak cell contrast is an **improvement-oriented** contrast, averaged over factorial backgrounds. A near-zero interval does not prove biological or universal equivalence. The tested ELM-like cells are explicitly not the unrecovered historical P4 cell.
- Excluding `E_META` from V51 is reasonable: the measured tradeoff is unattractive for this task and its per-edge state doubles. However, dividing a D loss cost by an A loss gain does not define a “net benefit” without endpoint weights.
- `C_FORGET` has a genuine revision-speed tradeoff. With D=0, its mean revision-4 accuracy effect is +3.06 [+2.20, +3.91] percentage points, alongside the old-memory cost above. It is not in V51, because V51 does not change the revision schedule. Its status is not “all forgetting mechanisms disproved.”
- `CORR2` is a **read-only attenuation** control, unlike event selection or consolidation. Its worse score cannot be pooled with other corroboration mechanisms as three identical failures. Candidate A also had higher delayed accuracy under CORR than RANDOM-MATCH at load 256 (65.84% versus 63.57%), with a whole-stream prediction cost. No universal “corroboration only works for isolated storage” principle is established.
- V50 `NO-KEY` worsening whole-stream byte loss does not establish that key writes help delayed target retention. Whole-stream prediction and target-association utility must both be measured.

## 3. What `A_ELIG` actually implements

Use the implementation, not the biological label, as the contract.

Let `S(v)` mean the existing deterministic top-228 sparse selection followed by L2 normalization. The leaky KC state during four input microframes is:

\[
s_t = 0.75s_{t-1}+0.25\tanh(d_t),\qquad s_0=0.
\]

The read code is:

\[
z=S(s_4).
\]

The omnibus eligibility vector is:

\[
a_0=0,\qquad a_t=0.72a_{t-1}+S(s_t),\quad t=1,2,3,4,
\]

followed by a blank-step multiplication and normalization:

\[
\widetilde z=S(0.72a_4).
\]

Reads and the output residual always use **z**, not the eligibility code. Only the presynaptic write vector changes to **\(\widetilde z\)** when eligibility is enabled.

Because `S(cv)=S(v)` for positive c away from the numerical zero guard, the final blank-step factor cancels. The source does **not** establish a nontrivial delay-tolerance curve merely by multiplying a vector by 0.72 and normalizing it again. V51 preserves this exact operation for comparison and calls it **multi-frame eligibility-shaped writing**. It does not claim to have established long-delay credit assignment.

A separate controlled teacher-delay experiment would be needed for that stronger claim; it is not silently added here.

## 4. Frozen biological substrate and engineering assumptions

### 4.1 Raw-data-derived components

Reuse the exact embedded graph from the V50 release:

| Population / block | Size |
|---|---:|
| PN | 699 |
| KC | 4,552 |
| MBON | 104 |
| PN→KC directed pairs | 22,417 |
| Sum of PN→KC recorded counts | 168,315 |
| KC→MBON directed pairs | 15,523 |
| Sum of KC→MBON recorded counts | 28,415 |

Retain root identities, edge order, PN→KC count normalization, and binary KC→MBON support. Do not re-extract with a different annotation rule or silently drop disconnected cells.

### 4.2 Engineering assumptions retained

- Homogeneous leaky KC equation above; no claim that its precise time constant is measured physiology.
- Fixed 5% KC selection, 228 active entries, L2 normalization.
- One float64 learned scalar per existing KC→MBON edge, initialized to zero.
- MBON consequence codebook, temperature 0.25, learning rate 0.5, residual multiplier \(\sqrt{104}\).
- No consolidation, active forgetting, DAN locality mask, metaplasticity, learned router, permanent object slots, PAIR transformation, recurrence, replay, or new cell family.
- “Hard read” means using the complete learned anatomical readout without CORR2/SHADOW attenuation. Predictions remain probability distributions, not one-hot hard decisions.

The raw graph constrains the substrate. It does not supply the outcome codebook or prove the teaching rule. No rewiring-superiority test is required for this functional scope experiment.

## 5. Two preserved learning configurations

### `V50-REF`

Read with z and update active existing edges using z:

\[
h_j=\sum_{i:(i,j)\in E}\theta_{ij}z_i,
\qquad p=\operatorname{softmax}(C h/0.25),
\]

\[
e_j=\sqrt{104}\left(C_{y,j}-\sum_v p_vC_{v,j}\right),
\qquad \Delta\theta_{ij}=0.5z_ie_j.
\]

Prediction precedes the observed outcome. No hidden-layer gradients are used.

### `ELIG-WRITE`

Read and compute e from the same z as `V50-REF`, using this arm's own learned weights and prediction. Update instead with:

\[
\Delta\theta_{ij}=0.5\widetilde z_i e_j.
\]

The graph, outcome stream, reader, learning rate, and number of update opportunities are shared. Eligibility changes neither the consequence nor its arrival time. Each arm learns its own synaptic trajectory.

Both write vectors have at most 228 active normalized KC coordinates. This does **not** guarantee identical numbers of anatomical synapses touched or identical update norms: KC degrees and residuals differ. Record these quantities; do not silently rescale them to manufacture equality.

`ELIG-WRITE` is a nominated, frozen addition undergoing confirmation. It does not replace `V50-REF` before the results are seen.

## 6. Experimental layout, namespaces, and seeds

### 6.1 Main panel

| Task | Load 128 | Load 256 |
|---|---|---|
| `CUE8` | Preserved V50 operating point on new seeds | Direct-association envelope extension |
| `BYTE256` | Byte task at matched identity load | Byte task at the older failure regime's identity load |

Every cell contains the same two learning configurations. No cell is omitted because another cell performed poorly. Only implementation or data-integrity failures pause execution.

### 6.2 Scientific seeds

```
51601 51602 51603 51604
51605 51606 51607 51608
51609 51610 51611 51612
51613 51614 51615 51616
```

Engineering seeds: `51671–51674`. Historical seeds may be used for explicitly labelled source-regression checks, never counted as new confirmation evidence.

The runner must verify disjointness from V50's `50901–50916` and OMNIBUS10's published seed roster.

### 6.3 Exact provenance

Namespace old experiments as `V50-FUNCTIONAL`, `V50-BYTE`, and `OMNIBUS10.1.1`, not as alternative uses of V51. Each new job ID includes experiment namespace, seed, task, load, source hash, design hash, graph hash, feature rule and generator version.

## 7. Why this is a scope matrix, not a complete explanation of the V49→V50 gap

The two native tasks differ in more than format:

| Property | CUE8 | BYTE256 |
|---|---|---|
| Input | Four noisy PN-space cue frames | Four chunks of a rolling 12-byte context passed through V50's fixed adaptor |
| Prediction alphabet | 8 consequences | 256 bytes |
| Association outcomes used by generator | Balanced 8-way assignments | Inherited 24 target letters |
| Teaching frequency | Once per cue→consequence | Once per byte, including framing and key characters |
| Timing represented by KC state | Four views of one cue | Four ordered chunks of one context window |
| Training renderings | Fresh V50 jitter/replacements per exposure | Inherited repeated serialized wrappers and key perturbations |

Crossing task with load prevents the simple mistake of comparing only cue/128 with byte/256. It estimates load effects **within each task bundle** and whether the eligibility contrast changes with task/load.

It does not separately identify input encoding, outcome alphabet, instruction frequency, or serialization. No result may be phrased “serialization alone caused X.” An alphabet- and teaching-matched bridge would be a separate experiment, justified only after seeing which failure needs explanation.

Increasing load also increases intervening observations/updates while holding per-identity exposure fixed. That is the intended continual-learning burden, not an isolated test of memory bytes independent of interference duration.

Both tasks start from zero learned weights. This is **architecture reuse across interfaces**, not transfer of trained synapses from the 8-outcome task into an unrelated 256-output codebook.

## 8. CUE8 generator and schedule

### 8.1 Preserve V50, not the omnibus generator

Port `CueWorld`, its seeding scheme, rendering rule and consequence schedule from the supplied V50 source. Parameterize cohort size only: 32 at total load 128 and 64 at total load 256.

- Each prototype has 32 active PNs out of 699 and amplitudes in [0.7,1.0].
- Four frames per cue; 26 retained prototype channels per frame and 6 replacement channels, following V50's rounding.
- Preserve V50's amplitude jitter implementation, including its treatment of replacement channels.
- Preserve family alternation and fresh per-exposure render variants. Do **not** replace these with omnibus's two cached training renderings repeated unchanged.
- Eight outcomes, balanced within each cohort.
- Reset all KC and eligibility activity before every cue trial.
- One teacher event and one update opportunity per trial.

The source's “held-out rendering family” is independently sampled from the same perturbation law. It is not an unseen transformation-family test. Report it as unseen-rendering generalization.

### 8.2 Schedule

1. Train A for eight exposures; score A after each exposure and at the completed wave.
2. Train B for eight exposures; first-return A probes before any old cue is taught again.
3. Show each A-noise cue once with a wrong consequence; probe the original answer before and after.
4. Train C for eight exposures; probe A/B/C.
5. Revise A-revision cues for eight consistent changed exposures; score after 1, 2, 4 and 8, and record every pre-update revision prediction.
6. Train D for eight exposures; final cohort probes, including the revised A mapping **after D**, not just immediately after revision-8.

A roles preserve V50 proportions: half untouched, one quarter noise, one quarter revision. Roles are evaluator/environment metadata only.

Total cue teacher events, including noise and revision:

\[
N_{\rm cue}(L)=8L+L/16+8L/16=8.5625L.
\]

Thus 1,096 events at L=128; 2,192 at L=256. The schedule does not stop an identity's training early when it reaches an accuracy criterion.

## 9. BYTE256 generator, causal interface and scoring repair

Reuse V50 `make_byte_stream` with an explicit load argument, the same prefix/headers/separators, six-character keys, wrapper transformations, 24 target letters, eight acquisition exposures and inherited noise/revision schedule.

The learner receives only the true observed byte history and, after prediction, the next byte. It does not receive a record boundary, phase, key ID, target-role indicator, cohort, or revision marker.

### 9.1 Sensory interface

At every byte event:

1. Take the last 12 observed bytes; use the V50 signed-bit/permutation encoding, with padding only for unavailable initial history.
2. Split those 96 coordinates into four ordered three-byte chunks.
3. Apply the same V50 fixed context→PN adaptor to each chunk.
4. Compute the homogeneous leaky KC sequence from zero, as in V50 ByteCore.
5. Read using the final sparse code; write using final or eligibility code according to the candidate.
6. Append the true byte to sensory history after prediction and teaching.

Eligibility is computed within this four-frame window. It does not persist across records or introduce a growing history. This is the exact architectural meaning of transferring the operation; it is not the same sensory interpretation as four noisy views of a cue.

### 9.2 Teaching and sample counts

Both candidates update on **all bytes**, not only target positions. Native byte counts are 15,344 at load 128 and 30,688 at load 256. Count initial padding events as in V50; report a first-full-context subset separately rather than silently dropping them.

The evaluator tags byte roles as framing, key, or target solely for output tables. Each table must carry n, loss sum, correct count, probability sum, and their means.

The previous `target_probability` column actually averaged true-byte probability across all bytes. Replace it with the unambiguous `mean_true_byte_probability` and generate real per-class metrics. Assert:

\[
\sum_c N_c=N_{\rm whole},\qquad
\sum_c \mathrm{loss\_sum}_c=\mathrm{loss\_sum}_{\rm whole}.
\]

### 9.3 Association-specific probes

At the same logical checkpoints as CUE8, query the context immediately preceding each identity's target without exposing its target. Use trained wrappers separately from held-out wrapper 2. The cloned evaluator constructs only the query history; no synaptic updates occur while constructing a probe.

Report all of the following separately:

- whole-stream byte prediction;
- framing/key/target prequential curves;
- target old-cohort retention before re-teaching;
- trained-wrapper and held-out-wrapper target recall;
- original-answer survival after isolated corruption;
- revised-answer learning at 1/2/4/8 and revised-answer retention after D.

The BYTE256 functional thresholds need not equal CUE8's 8-choice thresholds. Low absolute byte-target accuracy must still be shown, not hidden behind a whole-stream average.

## 10. Causal controls without another mechanism screen

Each task/load job includes:

| Branch | Reads | Writes | Role |
|---|---|---|---|
| `V50-REF` | Final KC code | Final KC code | Preserved learner |
| `ELIG-WRITE` | Final KC code | Multi-frame eligibility code | Nominated addition |
| `NOLEARN` | Final KC code | None | Decoder/prior reference, shared by the two feature-write variants |
| `UNPAIRED-REF` | Final KC code | Final code, unpaired teacher | Pairing-dependence control |
| `UNPAIRED-ELIG` | Final KC code | Eligibility code, unpaired teacher | Pairing-dependence control |

Total: 320 model trajectories across the 64 jobs, but only 16 independent seed blocks. Controls are not architecture candidates.

For CUE8, use the inherited within-cohort/exposure teacher permutation; preserve teacher counts/marginals and score against the actual environmental associations. Permutations must vary across exposures and be shared by both unpaired branches.

For BYTE256, independently permute **target teaching outcomes** within acquisition cohort/exposure or revision exposure. Keep all actual input bytes unchanged and continue teaching framing/key bytes normally. This control intentionally uses evaluator role information to break associative teaching only; it is explicitly nondeployable. Do not substitute shuffled bytes into future sensory history. Check pairing destruction and target teacher marginals; exact moment-by-moment teaching contents are necessarily changed.

### Post-training localization, for both real learners

Clone the trained state and compare on **identical probe realizations**:

- intact learned synapses;
- all learned KC→MBON weights reset;
- learned weights shuffled within each MBON's existing incoming edges.

Clear KC and eligibility state for every probe. Equalize the sensory realization, codebook, and tie handling across lesions. These are read interventions, not retraining.

V50 sometimes used different probe variants for intact/reset/shuffle and before/after noise. V51 adds a fixed paired probe panel to remove that avoidable sampling variation. This is a measurement refinement; it does not alter training. A regression output can retain legacy V50 probe definitions under explicit names.

## 11. Primary endpoint: sample efficiency, not a terminal leaderboard

### 11.1 Record predictions before teaching

For every acquisition presentation log:

`seed, task, load, arm, cohort, identity_eval_only, exposure_index, prior_exposures, correct, true_probability, loss_bits, observations_so_far, updates_so_far`.

For byte input, log all byte roles but compute association sample-efficiency endpoints on target events. An identity's e-th prequential prediction has only **e−1 prior teaching presentations of that identity**. Do not call its exposure index the number of examples already learned.

### 11.2 Exposure-to-criterion with censoring

For each seed, arm, task, load and cohort, let \(a_e\) be prequential association accuracy on its e-th exposure.

Define a sustained 80% crossing as the first e in 1,…,7 for which:

\[
a_e\ge0.80\quad\text{and}\quad a_{e+1}\ge0.80.
\]

Record the crossing index, the number of prior exposures (e−1), and the observation/update count at confirmation. An isolated first crossing at e=8 is labelled `late_unconfirmed_crossing`; it is not a sustained crossing and must not be averaged as though confirmed.

If no sustained crossing occurs, record **right-censored after eight exposures**. Do not remove that cohort from averages or present a fabricated ninth exposure as observed data.

Report per-cohort curves, the seed-level fraction of cohorts crossing, and a restricted acquisition-delay summary. One explicit bounded summary is:

\[
D_{80}=\sum_{e=1}^{8}\mathbf 1[\text{a sustained crossing has not yet been confirmed by }e].
\]

This is an eight-round delay score, not an estimated uncensored time-to-success. Smaller is better. A cohort never crossing contributes eight.

### 11.3 Continuous co-primary acquisition measure

Because the byte task may not approach 80%, also use:

\[
\overline L_{\rm acq}=
\frac14\sum_{q\in\{A,B,C,D\}}
\frac18\sum_{e=1}^{8}L_{q,e}.
\]

For CUE8, this is bits/consequence; for BYTE256 it is target bits/byte. These are analyzed **within task**, never pooled as one comparable absolute-loss average.

The sample-efficiency reporting family is therefore: censored exposure-to-criterion, restricted delay D80, and cumulative prequential association loss. Terminal accuracy cannot override a substantial acquisition-curve cost.

### 11.4 Unique information and compute

Report distinct latent identities, distinct observed input realizations, repeated presentations, teacher events, raw input observations, graph operations, and elapsed time. Four cue frames and fourteen byte updates are not interchangeable samples.

No claim of “same cost” across the two tasks is assumed.

## 12. Functional phenotype remains mandatory evidence

Sample efficiency is primary for comparing the nominated addition, but persistence is still the project's purpose. A learner that reaches a criterion quickly and forgets everything is not an improvement.

For each learner separately, retain V50's direct-cue component thresholds as descriptive operating-point checks:

| Component, CUE8 | Preserved threshold |
|---|---|
| Initial A trained accuracy | Mean ≥0.80; no seed <0.60 |
| New D trained accuracy | Mean ≥0.80; no seed <0.60 |
| A first return after B | Mean ≥0.60 and paired improvement over NOLEARN ≥0.30, 95% interval lower bound >0 |
| Untouched A after D, trained-family probe | Mean ≥0.50 |
| Untouched A after D, unseen-rendering probe | Mean ≥0.40 and improvement over NOLEARN ≥0.20, lower bound >0 |
| Stable-pairing dependence | Untouched-A improvement over its matched UNPAIRED control ≥0.20, lower bound >0 |
| One-off corruption | Original-answer mean ≥0.50 and before/after loss in accuracy no greater than 0.25 |
| Revision immediately after exposure 8 | New-answer mean ≥0.60 and new-answer probability exceeds old-answer probability on the same queries |
| Learned-state localization | Intact-minus-reset accuracy ≥0.25 with lower bound >0 |
| Bounded learned state | 15,523 scalars throughout |

Also report trained/held-out B and C after D; noise-original accuracy after D; revised-answer accuracy after D; and all revision curves. Revised accuracy at exposure 8 and revised accuracy after D remain distinct endpoints.

Do not stop scoring other components after the first missed threshold. The purpose is to map failure, not to hide remaining information behind one verdict.

At load 128, a failure to reproduce the operating point is investigated before making a new architecture claim. At 256, failure is an **envelope limitation**, not deletion of the established 128 result.

## 13. Statistical analysis

### 13.1 Replication and pairing

- Seed is the replication unit; there are 16.
- Compute contrasts within seed and experimental condition before averaging.
- Identities, exposures, byte positions, probe variants and factorial backgrounds do not become extra independent replicates.
- Do not pool historical V50 or omnibus seeds into the confirmation roster.
- The same seed across tasks is a blocking device, not a claim that a PN cue and a serialized key are the same sensory sample.
- Load-128 and load-256 generators obey the same within-task law but are not assumed to contain identical nested random realizations. Log hashes and label frequencies explicitly.

### 13.2 Planned comparisons

For each endpoint Y:

\[
\Delta_{s,T,L}=Y_{s,T,L,\rm ELIG}-Y_{s,T,L,\rm REF}.
\]

Report eligibility effects at each task/load, and within-task eligibility × load interactions:

\[
J_{s,T}=\Delta_{s,T,256}-\Delta_{s,T,128}.
\]

Also report the load change of each learner within each task. Between-task differences are task-bundle descriptions; do not use them to identify a single mechanistic cause.

Use paired Student intervals for continuous seed summaries and a seed-cluster bootstrap as a sensitivity check for accuracy/delay summaries. Never bootstrap individual byte events as independent samples.

Primary inferential families:

1. four ELIG-minus-REF acquisition-loss comparisons (2 tasks × 2 loads);
2. four ELIG-minus-REF D80 comparisons;
3. four ELIG-minus-REF delayed old-association-loss comparisons.

Show unadjusted intervals and Holm-adjusted two-sided p-values within each declared family. Other endpoints and interactions are explicitly secondary/explanatory. Missing primary seeds leave the corresponding estimate provisional; do not silently analyze only successful seeds.

### 13.3 Practical interpretation without an invented overall score

Report a benefit/cost table rather than subtracting unrelated endpoints.

For this experiment, flag as practically small (not automatically equivalent):

- acquisition loss differences below 0.02 bit in magnitude;
- delayed association loss differences below 0.03 bit;
- accuracy differences below 0.02;
- revision-exposure-8 accuracy costs below 0.05.

These are prospective reporting margins chosen for V51, not biological constants or retroactive tests of the omnibus. Formal noninferiority requires the corresponding interval inside the declared margin, not merely a nonsignificant difference.

`ELIG-WRITE` is eligible for default adoption only if fresh evidence shows a useful benefit in the intended operating condition, retains the functional phenotype, and rules out unacceptable acquisition/revision costs. Otherwise keep `V50-REF` as default and describe the addition as a tradeoff or unconfirmed. It is permitted for both learners to work.

## 14. Byte-model reference

Include an independent fixed-order Markov baseline for BYTE256, scored on exactly the same predictions. It is never used to gate the core's updates or to blend its output.

Proposed fixed baseline: empirical order-2 counts with Dirichlet pseudocount 1 per next byte. Use order-1 backoff only where fewer than two history bytes are available, then unigram/uniform initialization. No learned phase, target parser or order search.

A dense uint32 order-2 table uses 256³ counters (64 MiB); totals/backoff state are additional and must be reported. This is a **diagnostic computational baseline, not a state-matched biological-core competitor**. It must obey predict-then-update and read-only probe rules. Uniform probability is also reported.

A first-order cheaper baseline may be reported from the same backoff counts as a descriptive view, but no model is selected after inspecting byte outcomes.

Task-target accuracy and loss are not replaced by Markov-relative whole-stream performance.

## 15. State and biological-claim accounting

| Quantity | V50-REF | ELIG-WRITE |
|---|---:|---:|
| Learned KC→MBON float64 state | 124,184 bytes | 124,184 bytes |
| Working KC state, 4,552 float64 values | 36,416 bytes | 36,416 bytes |
| Eligibility accumulator | Not needed | **36,416 additional bytes** |
| Raw graph, codebook, adaptor | Fixed, reported separately | Same within task/seed |

The omnibus cost table's transient-cell count does not separately charge its feature-cache eligibility accumulator. V51 must charge a deployable implementation's extra state even if experiment-side feature caching avoided storing it in each simulated arm.

Fixed codebooks differ in byte count: 8×104 float64 values for CUE8 versus 256×104 for BYTE256. This is another reason not to call the tasks resource-identical.

Deterministic feature caches are optional computation caches, not learned memory. The reference deployable path computes features from the present observation/window without identity lookup. By default V51 should share a current event's features across arms and discard them afterward. A cache must have a hard cap, be counted in RSS, and pass cache-disabled prediction equivalence. Do not inherit a large unreported dense per-context cache.

For each learner report:

- event count and nonzero update count;
- active KCs and touched anatomical synapses;
- sum of update L1 norms;
- sum of update L2 norms;
- sum of squared update L2 norms;
- final weight norms;
- fixed state, learned state, transient state, diagnostic/evaluator storage, cache bytes and peak RSS.

Matching cue counts or normalized KC vectors does not match realized synaptic update mass.

## 16. Engineering plan for Mac M3

### 16.1 Deliverable expected at implementation

One executable `elm_v51_core_scope.py`, embedding the audited V50 graph, plus optional launch/status scripts. No raw Feather dependency on the Mac is needed for ordinary execution. The script should require only the already-used NumPy/SciPy stack; no ML framework or GPU is required.

Proposed command:

```
python3 elm_v51_core_scope.py --stage all --workers 4 --out-dir v51_results
```

On macOS the launcher or runner uses `caffeinate -dimsu` for long runs. Default four `spawn` workers, one thread per BLAS/Accelerate worker. One job contains both learner variants, causal controls, and the relevant fixed Markov baseline.

Run order interleaves task/load cells across seed blocks; do not abandon the byte or 256 arm after seeing another cell fail. `--stage cue`, `--stage byte`, and `--stage aggregate` may be supported for interruption or resource management; partial reports must identify missing conditions.

### 16.2 Resume and audit

- Atomic completed job files plus bounded rolling checkpoints for unfinished jobs.
- Checkpoint all future-relevant learner/RNG/evaluator state and event cursor.
- Check source/design/graph/generator hashes before reuse.
- Reject or quarantine stale outputs; never combine them with current results.
- Report `physical simulations`, `learner branches`, `seed blocks`, and `completed cells` separately.
- Do not mutate source identity merely for an extra analysis table; keep science and report hashes separate when a genuine reporting-only hotfix is needed.
- Make startup, checkpoint and completion messages visible so an active run does not appear stalled.

### 16.3 Required engineering checks

1. Exact graph IDs/counts and codebook validity.
2. V50 reference numerical equivalence on identical historical input sequences; compare parameters, predictions and update counts, not merely terminal percentages.
3. Omnibus eligibility algebra equivalence on identical four-frame fixtures.
4. Test that only the write vector changes under `ELIG-WRITE`; both read using final KC code.
5. Record the scalar blank-delay cancellation; do not turn it into a purported delay benchmark.
6. Test both task/load cells in smoke mode, with full role-class accounting.
7. Probe and lesion purity, identical sensory inputs for paired interventions, independent evaluator RNG.
8. Teacher permutation preserves intended marginals and does not alter real byte sensory history.
9. No-learning weights remain exactly zero; normal outcomes unavailable before prediction.
10. Fixed memory sizes throughout and explicit eligibility accumulator accounting.
11. Arm-order invariance, cache-disabled equality, and one-worker/two-worker/four-worker determinism.
12. Forced interruption/resume equality for both tasks, including interrupted aggregation.
13. Reporting tests for all-censored E80, final-only threshold crossing, missing seeds, constant zero variance and different outcome alphabets.
14. No V51 scientific outcomes inspected during engineering. Numerical bug fixes are documented separately from any change of learning rule.

Engineering tests must not retune eta, rho, eligibility decay, graph selection, task noise or criterion to make the phenotype pass. If a reproducible inherited bug prevents execution, report it and revise the implementation contract explicitly before science.

## 17. Required outputs

```
SUMMARY.md
MANIFEST.json
IMPLEMENTATION_AUDIT.json
OPERATING_ENVELOPE.csv
PHENOTYPE_BY_CONDITION.csv
CONFIGURATION_RECOMMENDATION.csv

PREQUENTIAL_ASSOCIATION.csv.gz
ACQUISITION_CURVES.csv
EXPOSURES_TO_CRITERION.csv
SAMPLE_EFFICIENCY_CONTRASTS.csv
TASK_LOAD_CONTRASTS.csv

FIRST_RETURN.csv
COHORT_RETENTION.csv
NOISE_RESISTANCE.csv
REVISION_CURVES.csv
REVISED_AFTER_D.csv
STATE_LOCALIZATION.csv

BYTE_CLASS_METRICS.csv
BYTE_ASSOCIATION_PROBES.csv
MARKOV_BASELINE.csv
BYTE_ACCOUNTING_ASSERTIONS.json

STATE_COSTS.csv
UPDATE_NORMS.csv
SAMPLE_BUDGET.csv
RUNTIME.csv
PROTOCOL.md
jobs/
checkpoints/
```

Per-class files preserve raw sums/counts, not only averages. Final uploads exclude large working checkpoints and any redundant feature caches. Stage-I and Stage-II files must never overwrite each other's same-named rows without a task column.

## 18. Interpretation matrix

| Observation | Permitted interpretation / action |
|---|---|
| Reference reproduces CUE8/128 and also works at 256 | The direct-association operating envelope extends to the higher tested load |
| Reference works at 128 but not 256 | Preserve 128 result; characterize which phenotype component and learning curve breaks |
| Eligibility helps after fresh confirmation without unacceptable cost | Adopt it only within the demonstrated envelope; retain plain reference as a lesion |
| Eligibility improves terminal loss but slows acquisition | Report a measured tradeoff; do not call it more sample-efficient |
| Both variants work on cues but fail bytes at the same load | The native byte task bundle exposes an interface/teaching/output limitation; source not yet isolated |
| Byte failure persists even at 128 | Load doubling alone cannot explain the byte gap |
| Both tasks degrade as load increases | Report load-dependent burden under each task; do not infer an unlimited capacity law from two points |
| Covariates/control probes expose a bug | Correct the measurement/implementation before interpreting learned behavior |
| Reset or unpairing leaves apparent competence | Locate information leakage, shortcut or residual state before a synaptic-learning claim |

No new mechanism is automatically triggered by a table cell. The working reference is retained, and any later intervention must address the measured failure rather than satisfy an assistant-authored version chain.

## 19. What stays out of V51

There is no new topology search, cell tournament, APL/DAN/routing addition, slot mechanism, metaplasticity variant, forgetting-rate sweep, replay, or growing memory. There is no load-512 branch in this initial matrix and no shortened revision schedule. Those would change the questions or cost; they are not silently appended.

`B_CONSOL`, `C_FORGET`, `D_LOCAL`, `E_META`, `COV96`, and `CORR2` remain documented exact constructions with measured costs. Exclusion from this run is a resource decision, not a universal biological refutation.

The unknown exact historical P4 neuron comparison is not declared completed by the omnibus ELM-like proxy. Conversely, that missing historical comparison is not a reason to add richer cells to a task already handled by the simple leaky reference.

## 20. Ledger status corrections to carry forward

Preserve the original through-V49 audit as a dated historical artifact; update its **current status**, not the historical evidence.

| Ledger item | Updated status / caution |
|---|---|
| Overall core | A bounded raw-BANC-based functional associative core is demonstrated at V50's 128-identity operating point; broad scope/minimality unresolved |
| M005 simple leaky cell | Working reference; V50 temporal and persistence controls support a useful stateful-cell contribution |
| M001–M004, M006–M014 richer/historical cell families | Add omnibus exact-proxy evidence where applicable; do not mark unrecovered P4/GRU/LIF tests completed |
| M011 heterogeneity | Negative exact omnibus fixed heterogeneous schedules on this loop; earlier task-specific findings remain part of history |
| Eligibility | Positive omnibus screen for multi-frame write shaping; nominated for fresh paired confirmation; long-delay credit assignment not established |
| M058 covariance | `COV96` separate proxy failed here (+0.378 bits old loss); do not attribute D_LOCAL's +1.366-bit mask effect to it or erase V30 |
| M063 read authority | CORR2 attenuation worsened this distributed readout; earlier explicit-memory results remain distinct |
| M103 phase+2 slot admission | Withdraw the unsupported occupancy/capacity/"key junk" interpretation; one-seed admission decomposition remains a historical diagnostic, not a confirmed mainline learner |

## 21. Provenance and implementation anchors

**Computed local hashes:**

- V50 runner `elm_v50_functional_fly_core.py`: `16cbb5ecc00e020c7407b5bbf3c33345f3c003281a0516ab26d791bf04a4dc81`
- V50 refined design: `96297122caea7ffffab2d38b32159911edf6b694ff662b50f29d1af28fb360a4`
- Uploaded `v50_results_upload.zip`: `9ba8a9899b1ad11255ac2b9c5c443e3d0ef4af977e870b9340dbe25d0befdfa7`
- Uploaded `omnibus10_results.zip`: `455372c67ec853d0992d808b8a1e0c95f7a79fee700cba8e2b61076457d50e13`

**Recorded in the supplied manifests/source:**

- OMNIBUS10.1.1 source: `11934e4706caf3830fd3082f739c24d7a179a815e22c3e2b504a745443fb9197`
- Raw metadata: `819bbcff476e52702d6f8d8604ce1f12d1d7b11942281df2f49df2a73a6f15a5`
- Raw edge list: `363fdef3813b72a5e45a42f17034cd5a544b654c838929cecf6e1ce5f60625cb`
- V50 embedded graph payload: `ae64d7759469fb0c98bf0d6b801345ebb8884997bc60c32384a041684c0507e5`

**Source anchors:**

- V50: `RawSensory.cue_codes`, `CueWorld.frames`, `KM.predict/update`, `train_cohort`, `functional_trial`, `ByteCore.features`, `make_byte_stream`, `byte_transfer_trial`.
- OMNIBUS10: `simulate_cell`, `topk_sparse`, `MemoryArm.observe`, `factor_arms`, `train_wave`, `score_ids`, `Cov96Arm`, and its `FACTOR_SPECS.json` / `CELL_SPECS.json`.
- Analysis evidence: `TRAJECTORY_SUMMARY.csv`, `CHECKPOINT_PROBES.csv`, `PREQUENTIAL.csv`, `COVARIANCE_ORACLE.csv`, `READ_AUTHORITY_PROBES.csv`, `STATE_COSTS.csv`.

The accompanying audit code recomputes the numerical source checks without executing any learner. This document supplies the V51 plan; it does not claim that its implementation or scientific results already exist.
