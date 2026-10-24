# V67 / Iteration 12 — write formation and locally available information

13 September 2026. Completed diagnostic experiment; no repair pilot qualified.

**The tested routes do not yet yield a selective persistent-learning repair.** Removing the modeled γ-history contribution from teaching reduces collateral disturbance by 26%, but also reduces new-association gain by 34% and correct choices by 12.4 percentage points. A simple matched slow-state rescaling produces less disturbance and more acquisition than this intervention. The proposed local fast/slow rule also fails its selection test. The full 5,177-KC learner and fixed observer remain the reference; no learning model is promoted.

V67 does establish a useful correction to the interpretation of V66: write magnitude carries information about disturbance size, even when it barely predicts the *sign* of the effect. It also separates a strong retrospective explanation of interference from the much weaker information available to a local controller before its consequences are known.

## What was actually run

The [preparation](../v67_preparation/DESIGN.md) was adopted with numerical, pre-outcome clarifications in [PROTOCOL.md](PROTOCOL.md) and [config.json](config.json). The user then authorized execution. The experiment retained iteration5/bi, instantiated through the frozen [full reference](../iteration10/models/reference.npz), with [iteration8/current_RBF64](../iteration8/models/current_RBF64.npz). It uses 165,688 mutable bytes, without cue-specific memory banks, KC deletion, new anatomical data, physiological refitting or observer refitting.

Three stages completed:

| Stage | Evidence | Role |
|---|---|---|
| R: magnitude and risk | 5,796 eligible delayed V66 records from 16 existing streams | Exploratory reanalysis; no new trajectories |
| F: formation | 16 fresh streams, each with 96 associations | Matched interventions on the formation of each later write |
| L: local information | Four focal memories per stream, four conditions, twelve bouts | Test discrimination and the proposed gate using actual local state |

The new bank contains 3,072 distinct cues, disjoint from 21,240 prior unique cues. Each cue activates eight of the existing 88 synthetic input channels. Consequence orientations are counterbalanced. One association receives six A/B bouts: 30 seconds A, 135 seconds rest, 30 seconds B, 135 seconds rest, repeated six times. These inputs exercise anatomy-constrained routing; they are not measured odor-evoked PN activity.

All 64 focal memories were eligible at the four-association checkpoint. The campaign produced 161,952 formation records and 196,608 sampled local feature records. Inferences and intervals use whole streams, not these rows as independent animals or learners. Reported 95% intervals use 2,000 stream resamples. Both F and L are development diagnostics; the reserved independent pilot bank was never generated.

## R: magnitude matters to disturbance, but does not specify its direction

Let E be a write's delayed effect on a previously learned, oriented memory margin, relative to matched exposure with new writes disabled. Negative E is collateral loss. Across streams, whole-write norm has mean Spearman correlation 0.0048 with E (95% interval −0.0180 to 0.0255), but 0.1448 with |E| (0.1162 to 0.1773). Slow-write norm correlates 0.2214 with |E| (0.1923 to 0.2517). The product of absolute slow-state alignment and slow-write norm correlates 0.8215 with |E| (0.7783 to 0.8576).

Fixed ridge models, with training-fold standardization and whole-stream holdout, give:

| Target | Direction features | Magnitude features | Combined | With interactions |
|---|---:|---:|---:|---:|
| Signed E | 0.7573 | −0.0034 | 0.7572 | 0.7969 |
| Disturbance \|E\| | 0.6207 | 0.0337 | 0.6635 | 0.6923 |
| Squared disturbance E² | 0.4342 | 0.0224 | 0.4655 | 0.5670 |
| Negative loss max(−E,0) | 0.5443 | 0.0031 | 0.5528 | 0.7344 |
| Tail indicator E < −1 Hz | 0.0711 | 0.0002 | 0.0777 | 0.1768 |

Values are **mean held-out-stream R²**, not V66's pooled R². For nonnegative targets, the direction block uses absolute alignment. The magnitude-only |E| R² interval spans zero (−0.0252 to 0.0718); interaction R² is 0.6923 (0.5514 to 0.7829). Its |E| mean absolute error is 0.0950 Hz, versus 0.1085 for direction alone and 0.2064 for magnitude alone. |E| and E² have identical Spearman ranks by construction, so those correlations are not separate evidence.

This qualifies any earlier claim that weak signed correlation rules out magnitude control. It does **not** establish homeostasis as a causal solution. These retrospective features include focal alignment and realized writes that a deployed local controller is not granted. Linear risk predictions are left unclipped; calibration tables retain invalid negative predictions. They are diagnostic regressions, not calibrated probability controllers.

![Magnitude and direction audit](figures/01_magnitude_risk.png)

Sources: [correlations](results/R_correlation_summary.csv), [holdout scores](results/R_LOSO_summary.csv), [calibration](results/R_calibration_summary.csv), [alignment strata](results/R_alignment_summary.csv).

## F: γ-history feedback supports useful acquisition as well as interference

At each matched pre-write state, the teaching payload is evaluated as p(g,a), with g and a denoting the modeled γ and α history contributions. INTACT uses p(g,a), G0 uses p(0,a), A0 uses p(g,0), and GA0 uses p(0,0). The γ→α indirect path, clipping and naive feedback are included. These are teaching-port counterfactuals, not deletion of stored memory or biological neuron silencing. Actual output expression retains the live memory state.

NO_WRITE disables new plastic increments while preserving passive decay and sensory exposure. HALF_WRITE halves every new raw increment. Every intervention branches from the same pre-state; only INTACT continues the ordinary stream. Therefore the results describe one later learning block on parent-generated histories, **not** full longitudinal performance of a G0 learner.

Primary results one day after the block, averaged within and then across streams:

| Arm | Negative focal disturbance D, Hz ↓ | New-association gain over NO_WRITE, Hz ↑ | New joint correct choices ↑ |
|---|---:|---:|---:|
| INTACT | 0.131162 | 4.022350 | 80.30% |
| G0 | 0.096839 | 2.658480 | 67.93% |
| A0 | 0.131162 | 4.022350 | 80.30% |
| GA0 | 0.096839 | 2.658480 | 67.93% |
| HALF_WRITE | 0.055845 | 1.649700 | 53.87% |
| Slow-norm sham | 0.091350 | 2.770905 | 69.36% |
| NO_WRITE | 0 | 0 | 31.39% |

D is mean max(−E,0). New gain is the oriented margin difference from the matched NO_WRITE branch. Joint choice requires the frozen reader and actual memory contribution to agree on a correct decision. NO_WRITE's 31.39% reflects generalization from previously learned associations; it is **not** a zero-state false-alarm result.

G0 reduces D by **0.03432 Hz** (0.03012 to 0.03868), or 26.2%. Its new gain falls **1.36387 Hz** (−1.39783 to −1.33132), or 33.9%, and joint choices fall **12.36 percentage points** (−13.93 to −10.80). It passes the minimum disturbance-reduction gate but fails acquisition noninferiority: allowed losses were 0.25 Hz and five percentage points.

The evaluator-only sham starts from the NO_WRITE endpoint and scales the INTACT increment so its slow-state endpoint norm equals G0's. All 1,472 comparisons match; scale factors range from 0.3730 to 1.1287, including values above one. It matches the **slow** norm, not all components or their time courses. Compared with this sham, G0 has **more** negative disturbance by 0.00549 Hz (0.00291 to 0.00819) and **less** new gain by 0.11242 Hz (0.10079 to 0.12312). G0 also fails the damage-advantage gate against HALF_WRITE. This is enough to reject the prepared route lead for a pilot; it does not make the endpoint sham an executable validated repair.

A0 is effectively a numerical near-null in this frozen model: new gain changes about 8.3×10⁻⁹ Hz; choice counts do not change. The largest α-feedback coefficient is only 1.37×10⁻⁹, whereas the largest γ-feedback coefficient is about 0.327. The finite output-rate bound verifies this small effect without rounding α feedback to zero. Maximum instantaneous factorial interaction is 4.3×10⁻¹⁴. These facts limit this fitted model's α-route test; they do not establish that biological α feedback is unimportant.

![Feedback formation tradeoff](figures/02_feedback_tradeoff.png)

Sources: [arm summaries](results/F_summary.csv), [paired contrasts](results/F_paired_contrasts.csv), [write components](results/F_write_summary.csv), [sham coverage](results/F_sham_coverage.csv).

## L: the tested local state gives weak discrimination and no selective gate

The four conditions are true reversal, one anomalous A/B bout followed by the original contingencies, stable reinforcement, and repeated learning of a predetermined unrelated fresh pair. Reversal and anomaly have exactly identical first-bout states and local signals. Any classifier using only that prefix must perform at chance on this contrast; both tested tiers give AUROC exactly 0.5.

The primary feature set contains current effective KC activity, adaptation, pre-write α-fast and slow states, routed α teaching, raw α drive and its sign agreements with those states. A secondary tier adds γ-fast and routed γ teaching. Neither includes cue, KC, focal or stream IDs, condition labels, bout number, observer output, future outcomes or old-memory alignment. Labels are used only by the evaluator. Thirty-two active KC coordinates are uniformly sampled per presentation with a shared deterministic sampling rule; gate-mass calculations use every active coordinate.

One fixed L2 logistic model per contrast and tier is trained on bouts 2–12, leaving each entire stream out. All 64 fits converge. There is no feature, threshold or nonlinear-model search.

| Features | Reversal vs unrelated: post-prefix AUROC | Reversal vs anomaly: post-prefix AUROC |
|---|---:|---:|
| α-local | 0.6081 [0.5766, 0.6330] | 0.6089 [0.5958, 0.6208] |
| With γ signals | 0.6068 [0.5896, 0.6228] | 0.6081 [0.5947, 0.6201] |

These values pool bouts 2–12 within each held-out stream before averaging stream AUROCs. Per-bout AUROCs in the figure answer a different ranking question; for example the α-local unrelated contrast reaches about 0.70 at bout four. Its per-bout average must not be substituted for the prespecified pooled score. There is modest information above chance, but neither contrast reaches the 0.75 gate. This bounds the tested features, sample and linear classifier, not every conceivable local computation.

![Information available during learning](figures/03_local_information.png)

The proposed parameter-free gate blocks a slow increment when it opposes the current slow state and lacks agreement with α-fast state. We compute hypothetical keep/block decisions on **INTACT trajectories**; the gate is not deployed. For late bouts 6–12:

| Test | Observed | Required |
|---|---:|---:|
| Conflicting slow-write mass retained during true reversal | 76.25% [72.64, 79.13], 16 streams | ≥80% |
| Mass retained during harmful unrelated repetition | 77.68% [67.55, 87.24], 12 streams | Substantially below reversal |
| Paired reversal-minus-unrelated retention gap | −2.47 pp [−11.54, +6.38], 12 streams | ≥20 pp, interval above zero |

Harmful unrelated episodes are defined, for evaluation only, by old-oriented delayed E < −0.1 Hz after twelve bouts. There are 22 such episodes among 64; twelve streams have both required conflict-mass denominators. Coverage passes. The paired gap uses those twelve streams, so it is not the subtraction of the two means with different denominators. Repetition can permit conflicting writes in both conditions; the tested rule does not selectively preserve reversal. These are diagnostic mass estimates, not measured retention under a gated learner.

![Trace gate selection test](figures/04_trace_gate_kill_test.png)

Sources: [classifier scores](results/L_classifier_summary.csv), [gate scores](results/L_trace_gate_summary.csv), [episode outcomes](results/L_episode_summary.csv), [gate decisions](results/candidate_gates.csv).

## The unchanged parent still has a load-dependent limitation

The fresh ordinary panel measures all learned associations at snapshot endpoints. After one additional day, joint correct choices are:

| Associations per stream | Correct / tested | Fraction |
|---:|---:|---:|
| 12 | 179 / 192 | 93.23% |
| 24 | 339 / 384 | 88.28% |
| 48 | 613 / 768 | 79.82% |
| 96 | 1,089 / 1,536 | 70.90% |

These are descriptive pooled counts on new cues and joint criteria, not a matched re-estimate of every previous capacity table. At load 96, recent/middle/old groups score 67.77%, 71.21% and 73.79%. Age and intervening learning history differ together; this is not a causal passive-decay comparison. The full 5,177-KC parent loses distinctions without any compression in V67.

![Full-parent ordinary panel](figures/05_parent_capacity.png)

Sources: [capacity](results/F_parent_capacity.csv), [age groups](results/F_parent_age_groups.csv).

## Decision, reliability and limits

The [machine-readable decision](results/decision.json) selects **no candidate**. Both predeclared leads fail eligibility. Accordingly, no half-gain repair pilot, deployed trace-gate trial, budget calibration, independent validation campaign or physiology refit was run. A failed eligibility test is not a direct experiment showing that every intermediate feedback gain fails. No fallback sweep was added. This completes the bounded V67 plan.

There are 21 passing startup checks and **70 passing final checks**. INTACT exactly reproduces the parent; restoration, continuation, same-state output expression, adaptation, clone isolation and direct main-stream replay pass. An independent four-port replay using original parent steps also agrees exactly. The maximum reader-versus-actual effect discrepancy is 1.42×10⁻¹⁴. Frozen parent inputs and source hashes are verified, including 60 transitive V65 assets; the V66 and V67 preparation archives remain unchanged.

One verification-loader error occurred: parent imports changed the module search path, causing a generic `analyze` import to resolve to V66's collector. The verifier was corrected to collect V67 files explicitly. No simulation, outcome, analysis code or decision changed. The [correction record](checks/verification_loader_correction.json) preserves both hashes and references the original failure. Final [acceptance checks](checks/acceptance_checks.csv) and [verification summary](checks/final_verification.json) record the successful rerun.

Summed simulation-worker time was 303.5 seconds with two single-threaded workers. Peak resident memory for one worker was 221.9 MB, not total machine usage. No additional download was needed. Raw data and checkpoints remain in the project for audit and are excluded from delivery.

The biological interpretation remains bounded by the [physiology source audit](../physiology_source_audit_20260913/REPORT.md): some old fitted targets differ from final published tables, several mappings remain unresolved, and the synthetic input representation is not measured PN physiology. Effective state coordinates and output clipping are not physical synapse weights or measured synaptic bounds. This model has no separate persistent teacher state. Strong local correlations inside it do not identify a conserved fly mechanism.

**What V67 adds to the end goal:** γ-history feedback is not a free source of interference that can simply be removed; it also supports acquisition. The tested fast/slow signs fail to distinguish useful revision from harmful repeated learning selectively enough. Magnitude remains relevant to disturbance size, but neither magnitude alone nor retrospective knowledge of the victim memory supplies a validated local control rule. The next defensible design would need evidence for information that separates those cases, or a prospective control that improves the gain–disturbance tradeoff. V67 does not supply that repair or establish a distilled persistent-learning core.

The compact bundle contains this report, final aggregate metrics, five figures in PNG/PDF, offline diagnostic fits, new source/configuration, acceptance checks and a manifest of unchanged dependencies. See [README.md](README.md) for reproduction and [MANIFEST.json](MANIFEST.json) for paths and hashes.
