# Proposed V186 · Sensory timescale × update cadence

Status: design only. Parent: the executable V181 NORMAL learner and FE-PERM/KC sensor. V185's 44/80 ridge witness leaves the finite readout ceiling unresolved on its separate associative roster. This design addresses the prose prediction result in the supplied memorization-to-representation review.

## Question and rationale

Does changing the FE trace from 0.95 to 0.50 improve the deployed local learner on fresh text at equal exposure? Does it help at the inherited update cadence, or only with more frequent optimizer updates? The review reports large gains with decay 0.50 and smaller batches on two opened text pairs. Those results nominate a candidate and an interaction. The chosen decay was inspected on those worlds, and the review's fitted KC probe and V181 native prediction route use different feature access. Neither is an upper bound on what the complete architecture can learn. The review also reports that its best short-trace readout still trails the project's calibrated count reference on M1.

## Arms and causal contrast

| Arm | FE decay | Max targets per chronological update |
|---|---:|---:|
| Parent | 0.95 | 256 |
| Sensor only | 0.50 | 256 |
| Cadence only | 0.95 | 32 |
| Joint | 0.50 | 32 |

Apply the decay at both existing FE sites: read-time encode and state transition. Use the same receptor tables, permutation, top-k tie break, sparse fan-in, 37,313 learned scalars, V181 NORMAL update, learning rate, feedback and dropout policy. Pair birth and example order within each world. Log dropout and batch grouping. Batch size changes update count, gradient grouping, optimizer clocks and compute. It is a cadence package and cannot isolate step count alone. Use every residual batch and the same target order. Preserve the inherited teaching schedule, without adding warmup exposures. Inputs and outputs are raw bytes with a separate EOS class. No task-specific features, attention, Transformer or ELM.

## Data, lifetime and controls

- Run the parent on an existing technical fixture for 0.95 parity. M1/M2 are opened development worlds and can only support parity or replay of the review. Before qualification lock two fresh disjoint A/B document pairs, exact source hashes, record boundaries, splits, deterministic births and edit IDs. Use the first two eligible pairs from a prospectively ordered source list, with eligibility based only on provenance, overlap and target budgets. Use held records from each document distinct from training and calibration, with normalized duplicate screening. Neither held outcomes nor candidate predictions enter selection.
- Preserve the V181 physical schedule per arm: A 160×2, B 48×3, EDIT 24×2 and NOEDIT 24×2. B follows the actual A state. Both C branches start from identical serialized B3 weights, moments, RNG and clock. Lock the V181 correction construction, scoring positions and held splits before training. Score A and unseen B after A2, A and B after B3, then still-valid A/B and corrected versus obsolete responses after C. Keep target ordering and sensor resets identical across arms.
- Fit unigram, bigram and calibrated interpolated KN-5 on the same permitted observations, using train/calibration only. Refresh stage-specific controls with the matching observed history. An observed exact-prefix table with count fallback is the recitation control. Report held performance stratified by whether the observed suffix already appeared in training, using locked suffix lengths 1–5. Held prose reuses byte patterns and supports predictive development evidence, with no claim of held arithmetic relations.
- On cached codes only, compare within-record distant-prefix changes that preserve the last 1, 2 or 4 observed bytes with position-matched pairs that change the suffix. Report KC overlap at fixed active counts and each decay's collision rate. These diagnostics test whether short decay actually changes the geometry in the predicted direction. A regularized convex readout can be added only if separately budgeted and locked before held scoring. Report its input features, solver convergence and calibration budget. It stays evaluator-side.

## Endpoints and decision

Primary endpoint is A2 held byte-plus-EOS bits from the deployed model's predictive distribution on observed prefixes. Define positive sensor gain as loss at 0.95 minus loss at 0.50 at the same batch size. Report both contrasts and their difference in each world. The batch-32 sensor contrast is the primary test of the review's strongest configuration. Require a positive gain in both fresh worlds, with record-paired bootstrap intervals reported per world and no pooling of records as independent worlds. Report the batch-256 contrast and the cadence-only effect even if the primary fails.

A sensor gain with no 256-batch gain supports a conditional effect. A cadence gain without a sensor gain supports the cadence explanation at this dose. For progression to a new-world review require the short-trace batch-32 arm to beat matched unigram and bigram on A2 held A in both worlds. Report the calibrated KN-5 gap as a separate benchmark. Require a positive matched-decay contrast on B3 held A and B, and on still-valid panels after EDIT. Report the B3 minus A2 loss on held A separately to expose forgetting even when a candidate has a better absolute score. Require correction to raise changed-target new-versus-obsolete log odds relative to NOEDIT. Use only predeclared checkpoints. Report every failed guard.

Decode held continuations from fixed prefixes using greedy native bytes and native EOS, capped at 64 outputs. Report exact complete continuation, own EOS, premature EOS and truncation separately. Flag any gain in predictive loss that accompanies worse exact completion or EOS behavior. Predictive-only improvement may justify representation research. Progression toward deployment requires successful own-byte behavior and subsequent confirmation.

## Admission and stopping

Two worlds × four arms × 560 = 4,480 candidate record presentations, including both correlated C branches. Freeze actual target counts, with at most 128 byte/EOS targets per training record, a scientific upper bound of 573,440 targets. Charge technical exposures against the 5,000-record and 600,000-target candidate caps. Log count-model fits, read-only evaluations and any probe work separately, including their CPU and storage. Keep the compact envelope of 40,000 learned scalars, 8 MiB deployed state, 1 MiB full mutable checkpoint, 900 CPU seconds, 1,800 active wall seconds, 1 GiB peak RSS and 256 MiB complete outputs. Profile the batch-32 arm and all required outputs before admission. Cache bounded active-index arrays and materialize at most one batch. Count checkpoint copies and audit output. Stop on any failed resource admission. Do not silently reduce dose or drop forks.

The next technical exercise must show exact parent parity at 0.95, single-scalar source difference, causal pre-feedback predictions, complete target accounting, clean split provenance, deterministic forks and measured resource admission. No scientific execution has occurred under this design.

## Follow-up boundary

Test the two-path idea only after the sensor contrast is resolved. The next question would be whether a long-trace episodic writer preserves taught recall while a separately trained short-trace predictor helps on new contexts, under matched total memory and compute. Test both roles with separate endpoints and causal branch controls. The present factorial has no mixed path or consolidation mechanism, and it cannot establish why every earlier architecture failed.
