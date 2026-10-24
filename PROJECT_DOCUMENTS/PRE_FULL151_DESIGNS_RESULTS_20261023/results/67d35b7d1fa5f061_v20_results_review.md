# V20 results review

Reviewed 6 September 2026. This review uses the uploaded full default run, not the earlier smoke run. No neural training or hyperparameter search was rerun. A small, explicitly described matrix calculation checks the controller at initialization.

## Judgment

**V20 implements an ELM–fly routing connection, but demonstrates no useful retention advantage from that connection.** Its exact, compressed, shuffled, outcome-only and feedback-disabled local learners have identical headline accuracies in each of three seeds. They acquire and revise associations effectively, while substantially forgetting associations whose outcomes have not changed.

This is a negative result for the implemented gain-modulated delta rule on this pilot. It is not evidence that fly learning structure cannot be distilled. It also cannot establish that the eight-pattern compression preserves a useful persistence mechanism: disabling the controller entirely reaches the same endpoints.

## Sources and integrity

- Uploaded archive: `elm_fly_v20_20260906_152006_26173_results.zip` (15,683 bytes).
- Source: `elm_fly_structural_learning_v20.py`, version 20.0.0.
- Source SHA-256: `f0dad8a713d2f2889f65685ae1f1e53295c65fe8debcbfbd95dcb4fd48881e87`. Independently computed and identical to the archived configuration.
- Embedded anatomy SHA-256: `7689815a557dd77bc30da723ad4820649d68c14478d72259b765e9ff84918f8a`. Loaded and checked during the initialization-only calculation.
- Examined `CONFIG.json`, `COSTS.json`, `INTERPRETATION.json`, `MATCHED_COMPARISONS.csv`, `METRICS.csv`, `REPORT.md`, `SUMMARY.csv`, `VERIFICATION.json`, and `WRITE_DIAGNOSTICS.csv`.
- Means and matched differences were recomputed from the CSV files. An independent code review checked mechanism, evaluation, controls and exposure accounting.

## Measured results

All entries are percent correct, averaged across seeds 17, 29 and 43. Probes are held-out noisy variants of the trained cue identities, with four possible outcome labels; chance accuracy is 25%. These are not tests of new semantic categories.

| Measurement | Fly exact | Fly compressed eight | Fly shuffled | Outcome only | Feedback disabled | Cumulative ridge |
|---|---:|---:|---:|---:|---:|---:|
| A after learning A | 98.61 | 98.61 | 98.61 | 98.61 | 98.61 | 100.00 |
| A after learning B | 49.54 | 49.54 | 49.54 | 49.54 | 49.54 | 98.61 |
| B after learning B | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 99.07 |
| Final unchanged A | 39.51 | 39.51 | 39.51 | 39.51 | 39.51 | 99.38 |
| Final unchanged B | 56.17 | 56.17 | 56.17 | 56.17 | 56.17 | 99.38 |
| Final revised associations | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 46.30 |
| Final all current associations | 60.88 | 60.88 | 60.88 | 60.88 | 60.88 | 86.11 |

Exact fly A retention after B is 38.89%, 45.83%, and 63.89% across the three seeds. Ordinary delta learning matches each value exactly. The mean acquisition-to-retention loss is 49.07 percentage points. Final accuracy on the combined unchanged A and B associations is 47.84%, versus 99.38% for ridge.

The equality concerns these endpoints, not every internal quantity or every intermediate checkpoint. For example, during the first B pass, seed 43 exact fly has one more correct A probe and one fewer correct B probe than feedback-disabled. During the first revision pass, seed 29 exact fly has one fewer correct A probe. These small differences do not produce a sustained advantage.

## The controller is running

There is no evidence that the fly arm accidentally calls the baseline or fails to modify weights. Its measured gains, update norms and pre-update errors differ from the baseline, and the source dispatches the arms separately.

The presentation-weighted average of the exact controller's per-update across-port gain standard deviation is 0.04264, around mean gain one. The corresponding pass averages range from 0.02674 to 0.06192. Its average update norm is 3.57306, versus 3.55963 for feedback-disabled, a ratio of 1.00377.

Thus the implemented controller changes learning, but the differences are modest and do not improve the principal accuracy outcomes. Mean-one gains alone do not guarantee matched update norms because gains and errors can correlate; the logged norms show that gross learning-rate attenuation is not the explanation here.

## What the update can and cannot select

Let h be the current feature vector, W the durable associative matrix, delta the current output error, and g the controller's vector of output gains. V20 uses

\[
\Delta W=\frac{\eta}{10^{-12}+h^T h}\,h(g\odot\delta)^T.
\]

For an earlier cue with feature vector a, its immediate output change is therefore exactly

\[
\Delta z(a)=\frac{\eta\,a^T h}{10^{-12}+h^T h}\,(g\odot\delta).
\]

The factor a^T h captures overlap between the earlier and current cues. V20's fly controller rescales output columns; it does not change that overlap factor or independently address different presynaptic connections within a column. At the configured feedback strength, every gain remains strictly between 0.5 and 1.5. There is no per-synapse protected state or selective write exclusion beyond the ordinary sparse feature vector.

Output gains could improve retention in some circumstances. The equation does not prove an impossibility theorem. It shows why this particular integration has limited means of preserving an old association when its features overlap the current one. The same connections remain eligible to change, with somewhat different output multipliers.

There is also a representational dilution worth recording. Four task labels are encoded as random orthogonal vectors in 104 output ports, rather than assigned a measured functional meaning at those ports. If C is the 4-by-104 codebook, C C^T = 104 I. For a fixed gain vector the projected update operator in the four label directions is

\[
M=C\,\operatorname{diag}(g)\,C^T/104.
\]

Using the embedded mappings and the exact controller at W=0, for each of the four outcomes in each of the three archived seeds, the relative Frobenius distance ||M-I||/||I|| is only 0.806–2.020%. Across those twelve cases, diagonal entries range from 0.97995 to 1.02568 and the largest absolute off-diagonal entry is 0.01921.

This initialization-only calculation uses no training. It supports the interpretation that averaging gains through the random output codes makes their direct task effect small. It does not measure the effect throughout learning: gains later depend on predictions, and modulation can create components outside the initial code span.

## How to interpret ridge

Ridge shows that the same frozen ELM features support retaining both A and B on this stream. Accordingly, a wholesale inability of the representation to support those associations is not the immediate explanation for the local learner's failure. Ridge uses an additional 96-by-96 inverse covariance, so this is not an equal-memory demonstration.

Its reversal score requires a precise qualification. Each revised sensory variant has three old-label presentations followed by three new-label presentations. Cumulative ridge weights all six equally. Its final objective includes conflicting old and new labels for the same input, and therefore favors a compromise. The 46.30% revised-label accuracy does not establish a general inability of covariance-based learners to adapt, nor a fly-specific revision advantage: feedback-disabled also reaches 100% on revised associations.

The useful contrast is between a local rule that readily replaces associations but damages unrelated ones, and cumulative statistics that preserve associations but continue counting obsolete targets. Neither meets the full intended persistent-learning goal in this experiment.

## Scope, costs and safeguards

- Each seed uses 48 online cue identities, 96 unique sensory training variants, and 120 distinct sensory-input/outcome pairs when changed outcomes are counted separately.
- Each arm receives 144 labeled presentations on A, 144 on B, and 72 on the revised subset: 360 online labeled presentations.
- Encoder development uses 64 separate sensory inputs for six passes (384 presentations), plus 64 calibration presentations. Development plus calibration plus online learning totals 808 presentations per arm; development is physically shared across arms.
- There are 96 scalar-activity ELM cells, each with six private transient states, and 9,984 durable associative coefficients. The encoder has 7,752 parameters; ridge additionally stores 9,216 covariance coefficients. Fixed routing matrices, topology and calibration are additional resources, so neuron count alone is not a complete memory budget.
- Transient activity resets between experiences, and probe calls are read-only. The learner's update interface receives the current feature vector and outcome, not cue identity, phase identity or reversal flags.
- A probes occur after each B pass, without intervening A training or reminders. `A_first_return_after_B` denotes the endpoint after the final B pass, not literally the first A probe since A training.
- The archive contains aggregate metrics rather than individual probe predictions or final weight checkpoints. It supports equality of endpoint accuracies, not a claim that every model makes exactly the same prediction on every probe.
- The effective normalized four-hop compression error used by this learner is 18.56%; the earlier raw three-hop value of 6.73% is a different metric.
- Three seeds are an exploratory pilot. No significance or equivalence claim is justified.

## Narrow next step

Keep the ELM representation, cue stream and exposure budget fixed. Change the location of the intervention: make the distilled teaching-delivery structure select eligible input-feature/synapse updates, rather than merely rescale output coordinates. Claude's measured direct teaching-to-association return mapping is a candidate structural input, with the ELM-to-distilled-association mapping made explicit as an engineering assumption.

Compare structured addressing with shuffled addressing that matches write sparsity and scale, a learned-output-feedback lesion, and ordinary local learning. Retain ridge as a reference. Defer further feedback-rank comparisons until an intact controller produces a benefit worth preserving.

The decisive diagnostic is the damage to old-cue predictions caused by new-cue writes, assessed alongside new acquisition and revision. Old probes remain evaluator-only and must never determine learner updates. A controller should reduce collateral damage at comparable new-learning effectiveness; slowing all learning does not count as memory protection.

This is a proposal for the next single mechanistic change, not a claim that anatomical routing already supplies a validated protection rule. V20 gives no reason to increase neuron count or launch a larger benchmark campaign yet.
