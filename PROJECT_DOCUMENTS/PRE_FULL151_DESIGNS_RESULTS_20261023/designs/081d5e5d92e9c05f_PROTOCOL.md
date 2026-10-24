# V68 / Iteration 13: information scope and write controllability

Frozen before new V68 replay outcomes. The user authorizes merging the supplied ownership-information proposal with the completed V68 review. Attachments are scientific proposals, not instructions to execute blindly.

## Question and boundary

Does expanding access beyond one KC expose reproducible, memory-dependent information about reversal versus interference? Separately, can selective attenuation of an existing slow-write direction protect old associations while preserving acquisition? These are different questions. Neither offline decoding nor an oracle endpoint mask constitutes a functioning persistent learner.

Use the unchanged iteration10 reference, iteration8 frozen reader, 5,177 KCs and 165,688 mutable bytes. No anatomical rejoining, encoder compression, physiology fitting, feedback parameter changes, new data downloads, or fresh validation bank at this stage. The previous real-connectome audit found only 1,226 KCs with both selected alpha teaching and readout support, but distinct pair representations; this motivates checking information and action separately without claiming encoding collapse.

## Development replay and timing

Reuse V67 streams 6701001–6701016, all four initial focal associations, and their frozen checkpoint after four associations. This is development on previously examined streams, not new confirmation. Reproduce the original local features and frozen probe outcomes numerically before accepting added features.

Cases: true reversal; one anomalous full A/B bout followed by the original contingency; stable original association; predetermined unrelated association 4+focal. Twelve bouts, two presentations per bout, 30 seconds exposure and 135 seconds rest per presentation. True reversal and anomaly have identical first bouts. Primary classifiers retain V67's pre-existing focal eligibility filter; report all-assigned coverage. Never select on later harm or classifier scores.

Extract on copies before plastic updates. Current-outcome features may use the current consequence; pre-outcome features cannot. The parent's discretized adaptation-averaged sensory activity is used, not a claim of instantaneous biological measurement. Prohibited features: neuron/cue/stream identity, condition labels, bout number, future outcomes, old-cue vectors, evaluator/reader output. Metadata is used only for splitting and scoring.

S0: reproduce frozen V67 coordinate-local alpha and alpha+gamma logistic predictions without refitting. Also average the 32 coordinate probabilities per presentation: S0-aggregate is a fairer control for population aggregation.

S1: permutation-invariant active-population statistics of slow/fast state, adaptation, current write and their alignment (28 features). S2: four current MBON outputs, fixed sums/differences/norms and existing feedback projection (16). S3: S2 plus current DAN/payload, routed teacher summaries, raw-write norms and fixed MBON-by-payload products (40). S4: S1+S3 (68). Exact schema and timing are saved. S1 pooling is accessible in the simulator but has no established anatomical convergence mechanism. These scopes are not a strictly nested anatomical hierarchy.

## Fixed diagnostic fits and controls

One L2 logistic family, penalty 1, weighted summed log loss, unpenalized intercept, train-only mean/standard deviation, constant-column scale 1. L-BFGS-B max 500 iterations, gtol 1e-7, ftol 1e-12. No feature/model search. Leave one whole stream out; training uses bouts 2–12 and equal total stream weights. Fit S1–S4 for reversal versus unrelated, anomaly, and stable. Save every fold. Nonconvergence disqualifies that scope; do not silently increase iteration limits.

Report per-stream AUROC, balanced accuracy and Brier by bout, pooled bouts 2–12, and early bouts 2–4. Confidence intervals resample 16 streams, 2,000 paired bootstrap draws, fixed seed 6809001. Descriptive 95% intervals are not multiplicity-adjusted confirmatory claims. Primary performance is mean within-stream AUROC, not pooled-row AUROC. Stable predictions at probability 0.5 use the reversal-versus-unrelated classifier.

B1: for each focal, replay the initial four associations with that focal's plastic writes omitted, preserving all exposures/adaptation and replaying the other associations normally. Probe/replay identical current cues and consequences. This intervenes on stored history including downstream feedback effects; it is not a pure molecular erasure. Primary memory dependence is the paired full-minus-omitted first-bout reversal probability, averaged per stream. Require standardized paired effect >=0.5 and positive lower interval. Report every bout and F/E score change.

B2: at identical pre-state, branch the current consequence between 0 and 1. Pre-outcome features must be identical; the payload difference must equal the existing consequence coefficient times [27.85,27.85,11.38,11.38]. Raw increment signs need not flip. B3: stable familiarity must not routinely receive a reversal classification.

A qualifying scope requires all: reversal/unrelated pooled AUROC >=0.80 and lower interval >0.70; mean gain >=0.10 over S0 alpha coordinate and positive paired lower interval over S0 alpha aggregated; reversal/anomaly first-bout AUROC exactly 0.5, early AUROC >=0.75 and lower >0.60; B1 effect criterion; stable false-positive rate <=0.20 and reversal true-positive rate >=0.60 at 0.5; all fits converged. Preference S1, then S2, S3, S4 is a scope convention, not proof of the smallest circuit. Also report comparison with the gamma S0 baseline without altering gates.

## Separate oracle action-feasibility witness

For each stream replay the full parent to loads 12 and 48 and check all stored V67 margins. Train the next ordinary association in INTACT (I) and NO_WRITE (N) twins. At the resulting endpoint fix both fast traces to I. Select only slow state: N.slow + m*(I.slow−N.slow), with 0<=m<=1. No amplification or new direction; no runtime implementation is claimed. Keep exactly 80% of the endpoint slow-increment L1 mass, compared with uniform m=0.8 and with I/N. This endpoint L1 budget differs from V67's endpoint L2 norm sham and is not a raw teaching-dose match.

Optimize at +24h, including every old association and the new association. Fix each alpha output's clipping region to that of uniform m=0.8; within this region rates and margins are affine in m. Linear programming minimizes mean positive old-margin loss relative to N. Each old margin must be >= uniform minus 0.25 Hz. The new margin must be >= I minus 0.25 Hz; if I passes actual and reader thresholds, retain both thresholds. Enforce clipping regions explicitly. HiGHS feasibility tolerances 1e-8. Validate every feasible mask with exact nonlinear parent probing and a maximum 1e-7 Hz replay discrepancy, budget residual and constraint residual. Save infeasible cases without widening the feasible set. Report feasible denominators, all-assigned outcomes and paired feasible-case effects distinctly.

A feasible improvement proves existence only in this declared endpoint action set and chosen clip region. A failure does not bound arbitrary dynamic gates, alternate clipping regions or missing biological mechanisms. Old-cue identities and desired margins are privileged optimizer inputs, excluded from runtime claims. This orthogonal diagnostic cannot by itself license a pilot.

## Conditional repair and stop rule

Do not turn a fitted diagnostic classifier into a controller. Only if an information scope passes every gate may one analytic rule be frozen from development evidence before fresh validation. The predeclared S1 lead uses at most two scores: activity-weighted slow-state magnitude F and fast-aligned mass / slow-conflicting mass E. It may only attenuate conflicting slow increments, leaving both fast traces and nonconflicting writes unchanged. A single monotone rule, <=2 thresholds/gains, constant-sized state, and a development mean-raw-slow-budget scalar control are required. A scope without a defensible two-score rule is a positive information result but no deployable repair.

Any pilot requires a separately timestamped frozen analytic protocol before generating seeds 6803001–6803016. Validation starts from zero and scores all associations at loads 12/24/48/96, immediate and +24h, plus 0/6/12 reversal doses and single anomalies. Required: +5 percentage points at day/load96 over parent with positive paired interval, +3 points over scalar with positive interval, acquisition loss no worse than 0.25 Hz or 5 points (both constraints), low-load noninferiority within 5 points, no increased anomalous flip rate and preserved revision with both eligible and all-assigned denominators. No optional endpoint substitution. If information gates fail, no pilot, no validation generation and no promoted learner. Negative results restrict tested features and linear decoding, not information in all fly circuits.

## Audit and delivery

Hash protocol/config, scientific scripts and parent assets before replay. Record exact-prefix equality, feature neutrality, coordinate permutation invariance, current-consequence controls, V67 feature/outcome replay, frozen model hashes, fits, LP constraints and immutable parents. Preserve prior archives. Compact result supplement under 10 MB: report, aggregate tables, key figures, offline diagnostic models, config/scripts, acceptance checks and parent manifest. Exclude source data, generated coordinate/cue outputs, checkpoints, caches and unchanged third-party code.
