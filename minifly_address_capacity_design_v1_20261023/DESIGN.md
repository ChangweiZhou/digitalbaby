# MiniFly address capacity experiment V1

Proposed design, 23 October 2026. Design delivery only; implementation and experiments have not started.

Keep the confirmed STOP-ONLY component and test one native storage intervention: expand the available KC population while retaining the original active-count limits, gradual teaching schedule and native update equations. First use saved responses to determine how much of the stopping benefit survives stronger stopping references. Keep EOS reweighting, mature-life training and another protection gate outside this comparison.

The practical question is whether the extension retains more correct OLD content while still learning NEW content, at an explicitly measured cost. A reduction in address overlap is a mechanism measurement, not the behavioral success criterion.

## 1 Evidence and the question still open

The [official confirmation report](../minifly_stop_only_confirm_v1_20261023/REPORT.md) establishes an **E1 stopping component**. Under STD storage, final complete OLD recall was 0.33% and complete NEW recall was 11.26%, versus zero for the direct codec. Mean conditional stopping gain was 22.10 percentage points, with paired 95% interval 20.47–23.72 points. This does not establish reliable persistent memory or unseen-addition generalization.

The [development revision](../minifly_recoverability_v1_r2_dev_20261023/REVISED_DESIGN.md) records a separate null protection result. CREDIT attenuated plastic-increment magnitude by only 0.312% during OLD and 1.335% during NEW; mean active multiplier was 0.9952. Retire that recipe without declaring the whole protection family disproven.

Matched continuations with NEW writes disabled support a write-associated source of OLD loss in this workload. They do not prove that passive evolution can never change a margin's sign: the readout combines signed contributions with different time constants and nonlinearities. Likewise, the gradual schedule gives NEW about 2.30 times OLD's mean nominal strength, but that alone does not explain the loss. Jumping a fresh life to maximum strength would also change total teaching dose.

The new hypothesis is narrower: **a larger pool of distinct input responses, accessed with the same active-count limits and comparable unlabelled output gain, may reduce destructive cross-writes**. This is an engineering extension of the native substrate. It is neither an unchanged anatomical reproduction nor a test of population size independently of how the additional input connectivity is constructed.

## 2 Goal and evidence boundary

The project goal remains retained own-output generalization to unseen addition problems per training sample and actual compute. The [arithmetic track](../PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json) remains unchanged. This bounded side experiment diagnoses taught random-association storage at target-independent surrogate addresses: **E1**, preceded by a saved-response stopping analysis at the same evidence level.

| Required audit | Contract for this experiment |
| --- | --- |
| Exposure | OLD and NEW keys are each explicitly taught their full random byte value and EOS, fourteen times. Each cohort has two values of every length 0–8. The first scored response at each checkpoint is read-only and receives no correction. No scored value is called an untaught relation. |
| Recitation countermodel | An exact-key table and a canonicalized-key table can reproduce every taught association. This keeps a positive result at E1. They are evaluator references, never storage backends. Report training-only byte/length priors and absent-key emissions as well. |
| Operative state | Native fast, slow and adaptation state produce the payload signs. The existing frozen STD EOS heads may change stopping only. No key, answer table, value length, cohort label or evaluator correctness enters inference. |
| Attribution | Paired write/no-write, within-length shuffled OLD targets and matched passive NEW continuations separate teaching, correspondence and later writes. They do not establish E3 reuse. |
| Promotion | A favorable pilot supports this particular storage extension only. No host integration, new core adoption, arithmetic claim or automatic confirmation follows. |

There is no arithmetic-specific representation, attention, Transformer, ELM, replay store, private allocation per test key, new task or externally fitted payload decoder. No new EOS head is fitted.

## 3 Saved response stopping analysis

Use the immutable responses from the completed 32-pair confirmation. Do not restore a native learner to generate new responses or repeat training. Keep the official verdict intact; these additional comparisons are **post-confirmation descriptive diagnostics**, not new confirmation claims.

In every comparison the emitted payload bytes come from the same real native tensor `D`. Only the EOS decisions differ:

| Readout | EOS information and decision |
| --- | --- |
| DIRECT and STOP | Reproduce the saved outputs with the original codec or each of the three frozen STD STOP heads. |
| FIXED LENGTH | Report all fixed endpoints 0–8 separately. Emit real native payload signs through that endpoint, then EOS. Do not choose the best endpoint on confirmation results. Length 1 is an explicit nonempty reference; length 0 explains the empty default. |
| POSITION ONLY | Supply zero for all 162 response features and the native EOS skip, retain the real slot one-hot, and use the same frozen STOP head. This is an ablation, not a newly optimized positional model. |
| EOS FOREIGN | Supply the EOS head with another record's entire response tensor, including its skip margin, but emit payload from real `D`. Use a fixed one-record cyclic shift within the same life, cohort and checkpoint, across lengths, with no target-dependent matching. |

The FOREIGN comparison must not reproduce the historical lesion that changed both payload and EOS. At all slots, assert bit-for-bit identity of real payload signs across these new readouts. Empty values use slot 0. Missing EOS by slot 8 is truncation. Fixed length is an openly engineered reference, not learned stopping.

Report unconditional complete exact bytes plus own EOS, payload exact, early/late stopping, truncation and per-length scores, including empties separately. Also report conditional stopping on the **same real-payload-correct nonempty subset** for every readout; publish its numerator and denominator, and preserve undefined subsets. Never calculate eligibility from the donor/zero tensor. The three original component checkpoints are OLD_end, NEW_final and OLD_S_NEW; also publish final written OLD and the other saved checkpoints.

Reduce two value lives within each pair and average all three heads where applicable. Show all 32 pair differences and descriptive paired intervals. No best head, endpoint, checkpoint or subset is selected. RAW can be tabulated as a separate descriptive appendix if the analysis budget permits; STD is mandatory.

If STOP mainly matches a fixed/positional rule, narrow the explanation to stopping calibration. If real tensor input outperforms the fixed references and EOS FOREIGN, this supports memory-dependent stopping in this assay. Neither finding repairs payload storage, and neither selects a different expansion recipe. Finish this report before new native training; the storage comparison below remains defined independently of its outcome.

## 4 The two native arms

| Arm | Available KCs per compartment | Active limits | Native plasticity |
| --- | ---: | --- | --- |
| CANON | 5,177 | Original two side-specific counts `K_s` | Frozen native signed update, allocation gate off |
| EXPAND | 20,708 = 4 × 5,177 | The same `K_s`, not four times as many | Identical update equations and constants |

Fourfold expansion is a single engineering contrast, not an optimum or a size sweep. It gives an interpretable fourfold availability change while preserving integer side proportions and empirical donor counts. Under independent uniform supports, expected overlap is `K²/N`; that model predicts a fourfold reduction. Actual recruitment need not be uniform, so the prediction must be measured rather than assumed.

Both arms retain nine independent value compartments, float64 state, the original PN-type input interface, `DT=30/14`, neutral subtraction, BEGIN/adaptation reset, clocks, rests, residual, fast/slow allocation and decay. Keep `gated=False`; CREDIT and replay are off. All dimensioned mutable state and checkpoint fields must match the new N, including inherited bookkeeping that is functionally inactive but still allocated. Report its cost.

### Connectivity and state construction

Do not call the existing size helper with a larger integer. In [fresh_bank](../minifly_v5_1_dev_20261020/v51_common.py), noncanonical sizes resample selected row arrays, scale Q and reader_Q by `old/n`, retain a fixed active fraction and leave B unchanged. That is not this intervention.

Construct CANON exactly as before. Construct EXPAND from a separately verified canonical newborn and a frozen unlabelled extension, as follows:

1. Retain the original B columns, Q/T/reader_Q rows and KC-side assignments as the first block. No learned checkpoint is expanded.
2. Add three blocks. Within each side, each block uses every canonical KC once as a donor, in a seeded permutation. Copy each donor's Q, T and reader_Q row and side together. This preserves their empirical joint distribution over the whole population; do not independently shuffle teaching and output routes.
3. Give each added KC a new sparse input column. For its side, define the admissible PN-row set as the union of nonzero canonical B rows on that side. Draw the donor's number of distinct input rows without replacement, weighted by canonical edge-inclusion counts on that side. Randomly assign the donor's original nonzero weight multiset to those rows. Zero-degree donors stay zero-degree. Keep the original PN-type mapping. This preserves degree and weight multisets and total edge count `4E0`; it does not preserve the specimen's entire connectivity or all correlations.
4. Use one frozen extension seed, `202610238201`. Generate the three blocks once before any assessment keys or values. Do not resample a graph after inspecting recruitment or recall. Report duplicate effective input columns after aggregation by PN type, including dead columns; anatomical PN-row differences can still be identical in the 88-channel interface.
5. Initialize every fast/slow value to zero and adaptation to one. No prior values, per-key slots or enlarged pretrained answer module are supplied. All global feedback matrices and scalar dynamics remain the parent's.

The sparse matrix dimensions must be `PN_rows × N`; Q, T, reader_Q and kc_side must have N rows; fast/slow/adaptation and every retained bookkeeping tensor must agree. No padding or broadcasting may conceal a dimension mismatch. Freeze and hash this complete constructed birth.

### Fixed activity and calibration

Set `K_s = ceil(f0 × N0_s)` from the canonical frozen active fraction and side counts, then install these exact two limits in both arms. STD scores are `(B.T @ p[pn_type_index] − mu)/max(sigma,floor)`. Select the largest strictly positive scores within each side up to `K_s`, using KC index to break ties, exactly as in CANON. Report actual counts and shortfalls; do not recruit negative scores merely to fill a quota.

Reuse the original 2,048 unlabelled 88-channel calibration vectors. Original-column means, standard deviations and floor remain unchanged; compute additional-column moments from those same vectors and use the same frozen floor. No assessment key, value, length or success statistic determines calibration.

Do **not** apply `N0/N` output scaling at fixed K. Before values, compute each arm's calibration activities and the mean drive of each raw Q column. For EXPAND only, multiply Q column j by the positive scalar

`g_j = mean_cal(x_CANON @ Q_CANON[:,j]) / mean_cal(x_EXPAND @ Q_donor[:,j])`.

Apply the analogous independently computed ratios to reader_Q, using canonical reader_Q as the reference. Require finite positive denominators and numerators; failure stops qualification rather than selecting a replacement graph. Record all eight ratios. CANON is not renormalized. T, alpha_scale, residual strength, native thresholds and STOP-head scales remain unchanged.

This matches four first-moment output drives, not the full distribution, teacher-to-readout coupling or nonlinear self-write gain. Test the latter on calibration-only fixtures below. If gains differ materially, report those differences with the behavioral result; do not describe it as a clean size-only or perfectly dose-matched effect. No label-derived gain correction is permitted after qualification.

## 5 Shared experience and readout

Reserve eight fresh pairs `202610238101–202610238108`, each with two independently generated value lives. Each life supplies the same keys, receptor vectors, values, pass orders, time increments and feedback to both arms. Collision-check exact keys against all previous external-memory rosters before values are generated. Only the constructed KC supports differ. Results remain conditional on one canonical anatomy and one fixed extension.

Reuse the original surrogate mapping and eighteen OLD, eighteen NEW and eight absent keys. Each cohort has two values of each length 0–8, uniformly sampled byte payloads, and fourteen passes. Preserve the continuing counter n across OLD and NEW:

`a(n) = (1/16) × (1/16 + (15/16) × min(n/1024,1))`

`c = a(n) × (tanh(native_margin/2) − observed_target)/2`.

Expected output bytes and EOS are external training signals; EOS masks payload writes. No artificial relationship between different values is provided. Mean nominal strength remains approximately 0.01109 for OLD and 0.02551 for NEW. Warmup restarts only at each independent empty life, never between its cohorts. The first one-hour rest follows OLD, and a second follows NEW. No maturity counter is imported and no hidden preceding experience is free.

For each size, run W, N and SH. W permits observed writes. N sees all the same events with value writes off. SH uses the parent's within-length cyclic reassignment of OLD values and original NEW teaching. At W's first-rest checkpoint fork S_NEW, which sees the same NEW events with writes off, then the same final rest. SH and S_NEW are diagnostic branches, not extra candidates.

Use the parent's fixed common-horizon SNAP query: BEGIN on a private copy, all nine slots without teaching or inter-slot ticks, actual minus neutral response. Emit bytes from the signs of the eight averaged native payload channels. Report direct EOS and all three already-frozen STD STOP heads with their original RMS scales; the primary complete-output summary averages those heads. No expansion-specific refit, new normalization from assessment responses or threshold adjustment is allowed. Distribution shift can reduce STOP performance even when native payloads improve; report those two effects separately.

## 6 Endpoints and development decision

The storage co-endpoints are final OLD and final NEW **nonempty payload exactness**, using evaluator length only to measure stored content. It is unavailable to the learner. For every pair, average its two lives before comparing EXPAND minus CANON. Publish acquisition at passes 1, 4 and 14, both rests, bit/byte accuracy, all-value and nonempty scores, and per-length results.

A **favorable storage pilot** requires final OLD payload gain strictly positive in all eight pairs, and final NEW payload gain nonnegative in all eight. This is a conservative observed Pareto screen, not a noninferiority confidence claim or a reliability threshold. Publish the paired means and descriptive 95% Student-t intervals even if the screen fails. No cohort average may conceal individual pair regressions, and eight pairs are not a new official confirmation.

In addition, a teaching-mediated interpretation requires positive average EXPAND W-minus-N payload effects for both cohorts. Report EXPAND-versus-CANON differences in those effects and W-versus-SH OLD effects. The difference

`(OLD_S_NEW − OLD_final)_CANON − (OLD_S_NEW − OLD_final)_EXPAND`

measures whether the later-write-associated OLD penalty is reduced. A better OLD_end acquisition level alone does not establish that explanation. S_NEW contrasts are matched within size; different sizes need not have identical prestates or self-gain.

Mandatory practical outcomes are final OLD and NEW complete nonempty output with own EOS, jointly by pair, alongside their native payload ceilings. If the storage screen passes, label the same frozen-readout integration favorable only if complete OLD improves in every pair and complete NEW does not fall in any pair. Otherwise retain a storage-component result only. Undefined conditional stopping subsets stay undefined and cannot pass a stopping-specific gate.

| Observed outcome | Next decision |
| --- | --- |
| Smaller overlap without better retained content | No storage advance. Do not scale again automatically. |
| Better OLD purchased by worse NEW | Retention/acquisition trade-off; no favorable pilot. |
| Better storage but frozen STOP regresses | Storage evidence only; document readout distribution shift before proposing any separately scoped recalibration. |
| Better OLD and NEW at higher cost | Report the accuracy/state/compute trade-off. Do not call it a compute-efficiency improvement. |
| Favorable storage and complete-output screen with acceptable measured cost | Propose a separate fresh confirmation; no automatic run or adoption. |

“Acceptable cost” is a reported engineering judgment, not an adjustable behavioral gate. No numerical deployment target is invented here. Plot both arms' OLD/NEW scores against **all external records and cumulative measured CPU**, using the declared checkpoints and measured operating costs. Report where one arm is Pareto-dominated; do not claim matched-compute accuracy outside the observed checkpoints or extend training to create a favorable crossing.

## 7 Measurements that explain the outcome

Record supports separately for side, output slot and OLD/NEW cohort. Publish pairwise intersections, Jaccard values, identical-code frequency, overlap with structurally teachable cells, and teacher/readout-weighted overlap. For calibration inclusion probabilities p_i, report `sum(p_i²)`, activity-frequency distribution, dead cells and `N_eff = (sum p_i)² / sum(p_i²)` within each side/slot. Compare these with N and actual K. Raw neuron count is not effective diversity.

Measure self-write and cross-write effects on eight fixed unlabelled calibration addresses, before assessment values. For each address, each of nine compartments and both synthetic coefficient signs, use an identical newborn prestate, coefficient magnitude `a(0)/2`, and matched passive clone. Record the queried margin change after the write minus passive evolution, on the written address and the other seven. This is **288 partial native fixture events across both sizes**, excluding their read-only probes; every intervention starts from its own pristine fixture state. Use paired signed self changes and destructive-to-self ratios where denominators are nonzero. Do not fit Q, T or teaching strength to these measurements.

During assessment, record proposed/applied plastic-increment L1 and L2, per-compartment fast0/fast1/slow totals, residuals, simulated time, nominal strength, clipping and active support counts. At the fixed checkpoints decompose signed pre-clipping gamma-path, alpha-fast and alpha-slow input contributions, then record actual clipped outputs. These additive pre-clipping terms are explanatory measurements, not an exact additive attribution of nonlinear recall.

Do not run a whole-core clone or an OLD-query sweep after every write. The calibration intervention matrix and fixed checkpoint probes bound this diagnostic cost. Passive evolution remains exactly the native operation; a single-timescale decay argument is not substituted for the measured S_NEW branch.

## 8 Counts and resource ceilings

This is a bounded development pilot, with no screen followed by automatic sample extension.

| New assessment work | Count |
| --- | ---: |
| Pairs × value lives × sizes | 8 × 2 × 2 = 32 combinations |
| W/N/SH full base histories | 96 × 504 records = 48,384 records |
| W S_NEW continuations | 32 × 252 records = 8,064 records |
| Main newly executed applications | **56,448 records / 282,240 events** |
| Shared OLD prefixes in logical branch accounting | 8,064 record equivalents |
| Main logical record equivalents | 64,512 |
| Qualification ceiling in addition to main work | 96 full balanced synthetic records / 480 events, plus the 288 partial calibration events |

There are eight assessment units, 96 base histories and 32 correlated continuations. STOP heads, two value lives and forks do not create additional independent pairs. Calibration has 2,048 unlabelled receptor vectors per size, not 2,048 secretly taught value records. Existing head-training exposure, calibration and birth provenance remain part of the lifetime information ledger; reuse avoids new fitting but does not erase their cost.

Saved-response analysis has a 300 cumulative process-tree CPU-second, 600 active wall-second and 2 GiB sampled RSS ceiling. Native qualification, roster, reduction, independent artifact audit and packaging share a separate 3,600 CPU-second, 5,400 active wall-second and 2 GiB sampled RSS ceiling, one worker and one numerical thread. Require measured 2 GiB admission headroom. Check a conservative projection after qualification and the first complete planned pair, which counts in the roster. If it cannot fit, preserve the partial package and stop without shrinking the roster, dropping controls, switching seeds or resetting budgets.

Report construction/calibration, encoding, native updates, queries, copies, diagnostics, audits and packaging separately. EXPAND has four times the cell-associated state and four times the B edges before overhead. Its total memory and runtime need not be exactly fourfold. Fixed K does not remove the inherited global projection, sorting, passive sweeps or audit copies. No sparse-compute or scaling claim follows.

## 9 Qualification and delivery

Before any fresh value is generated, lock the complete imported source and canonical birth closure, expanded assets, calibration, code manifest, head files, pair roster and report/reduction definitions. Verify:

- CANON reproduces the parent fixtures; expanding and restoring does not alter the original first block before the declared gain calibration.
- Every N-shaped array and sparse matrix dimension agrees; extension generation is deterministic, donor/degree/weight invariants hold, and all gain ratios and calibration moments are finite.
- Exact active limits are installed; shortfalls are logged; fixture values/labels cannot affect supports or gains. Duplicate input responses are disclosed rather than filtered using task outcomes.
- Disabled writes, passive evolution, residual masks, adaptation resets and clocks match their declared semantics. Restoring each size preserves all operative mutable state.
- Queries leave persistent state unchanged; payload signs match direct decoding under all EOS controls; inference receives no keys, target length, identity, correctness, labels or cohort flags.
- Synthetic 0–255 payloads and EOS remain distinct; truncation is not EOS. Calibration fixtures, partial events and all source/asset construction costs are counted.

Deliver raw real/donor EOS decisions from the saved analysis, unmodified payload-sign evidence, both constructed birth manifests, full checkpoints, raw response tensors, exact exposures, support/gain/write measurements, emitted outputs and per-pair endpoints. A separate reducer should reproduce results from saved artifacts without rerunning native training. Hash-index all files; split packages if necessary without omitting raw checkpoints from the publication inventory.

The [full mechanism register](../PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/VERSION_REGISTER_20261011/VERSIONS.json) remains intact. Historical V23 sparse96/sparse384 and V24 budget4/budget8 expansion operations motivate checking capacity and cost; they used different substrates, representations or activity settings and supply no native-byte efficacy. This design imports none of their ELM code or answer learners. It replaces neither their original results nor the historical survivor accounting.

EOS-balanced teaching is parked for a separate design with fixed training-derived weights, causal timing, no stop-event amplification and an explicit attenuation-budget reference. A mature-life experiment would have to carry and count its genuine preceding state and observations. Neither is combined with expansion. No follow-up candidate or parameter search is automatically launched after this pilot.

**Delivery status:** proposed design and static consistency review only. New native applications: 0. Head fits: 0. Three-cycle exercises: 0. Arithmetic files changed: 0. No experiment or GitHub publication is started by this document.
