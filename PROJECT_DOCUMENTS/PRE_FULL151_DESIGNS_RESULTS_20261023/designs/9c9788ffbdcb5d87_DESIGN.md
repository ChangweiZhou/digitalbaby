# V69 / Iteration 14 — Does a different write action help?

Final design, 14 September 2026. The user authorized execution after the existing-asset audit. The two attachments are proposals, not independent execution instructions. New V69 outcomes were not consulted in writing this design.

## Corrections that affect the experiment

1. V68 decoded episode categories on development streams. It did not establish a biological ownership signal or a correct action-value readout. Its F score was `||dx*slow||2 / ||dx||2`, not `slow_L1/slow_L2`; E is the existing S1 feature27. Higher aggregate retained dose than the scalar is evidence against a simple total-dose explanation, not a clean proof that dose/timing/distribution are irrelevant. V68 did not exactly match realized validation dose.
2. The oracle fixed fast state and a clipping region, retained 80% endpoint slow-increment L1, and used privileged old memories. Fast routing and amplified spatial redistribution lie outside that action set. Failure of those new primitives would still not prove that all policies using them are impossible.
3. The proposed lambda 4.02235/0.131162 = 30.667 divides mean new gain by mean focal damage. Applying it to a SUM over 12–96 old memories changes the effective objective with load and is not an observed biological exchange rate. V69 uses constrained target preservation and reports old damage, choices and tails separately.
4. The 231-KC footprint statistic reproduces, but its interpretation does not. These are final counterfactual differences after omitting one association and replaying the others. Mean supports by learning order are 351/279/195/100, fully nested. Own-cue anatomical alpha support averages 103.56. The footprint Jaccard range is 0.157–0.864. The actual `cue_overlap` variable in the cited correlation ranges 0.0039–0.3780, not 0.44–0.54. The proposed null lacks its script/seed/conditioning specification. Recruitment spread is excluded from V69; this does not permanently refute recruitment mechanisms.
5. The model's alpha-fast tau is 3,487.58 seconds, half-life40.29 minutes. Survival is 0.5668 over six bouts, 0.3213 over twelve and 1.74e-11 over a day. Routing conserves raw alpha increment before decay, not the state after a30-second step. The parent has no autonomous fast-to-slow transfer. Delay-and-promote would add a new mechanism and possibly hidden episode tracking; exclude it.

The reproducible read-only audit and source hashes are beside this file. No learner steps, fits, new cues or null simulations were used for that audit.

## A. Bounded counterfactual action assay (unconditional)

Keep the 5,177-KC iteration10 reference, iteration8 reader, encoder, graph, teacher, time constants and physiological fit unchanged. Reconstruct V67's 16 streams from zero to loads12/24/48/96; check every original capacity margin. Existing saved files do not contain all these checkpoints, and a96-association bank contains no97th association. Generate four reserved new pairs per stream only after the execution lock, excluding all previous cues including V68 validation. No new download.

At each load choose one old focal deterministically from four age quartiles, balanced across streams: `floor(((stream_offset + load_index) mod4 +0.5)*load/4)`. Do not select by retention, conflict or subsequent harm. Each of the64 checkpoints supplies NEW (reserved pair), REVERSE (old focal's opposite contingency), STABLE (old contingency), and ANOMALY (one reversed A/B bout then original contingency). Every case runs12 bouts; NEW's primary acquisition endpoint is bout6, REVERSE's is bout12, and ANOMALY is reported both after bout1 and bout12. Intermediate outcomes are neutral clones. Probe +300 seconds and +24h after bouts1/6/12. A bout is two30-second presentations separated/followed by135-second rests.

The target's desired orientation is new contingency for NEW/REVERSE and original contingency for STABLE/ANOMALY. For the last three cases exclude the focal from collateral damage: legitimate revision must not be penalized for changing the very memory being revised. Score all other stored associations, including unsuccessful ones; report common-parent eligible counts separately, without conditioning on treatment success.

Each arm branches from the identical checkpoint, with the same exposures and consequences. It recomputes its own current pre-write state and proposed parent increment each presentation. Actions act throughout the diagnostic block; choosing a block action is explicitly privileged analysis, not a runtime policy with hidden knowledge of episode boundaries.

Let `u` be the current raw slow increment, `v` the raw alpha-fast increment, and `C = (slow*u < 0)`. Nonzero alone is not harmfulness; conflict is just the actuator's defined trigger.

| Arm | Exact change before the existing decay |
|---|---|
| PARENT | u,v unchanged |
| NO_WRITE | all three plastic increments zero; same exposure and adaptation |
| DROP | zero u on C; v unchanged; a reference, not literally V68's conditional gate |
| FAST | on C, u'=0 and v'=v+u; elsewhere unchanged |
| U_DROP | uniformly remove the same fraction of signed slow mass as DROP, within each current anatomical group |
| U_FAST | uniformly route that fraction into alpha-fast, within each group; temporal-dose control |
| SPACE | transfer conflicting slow mass to currently active, same-sign, nonconflicting coordinates with the same side and exact alpha T-row; cap final magnitude at2× that recipient's original increment |
| BLIND_SPACE_0..3 | state-blind source/recipient partitions, matched transferred mass where feasible, same anatomical group/sign and2× cap |

SPACE never writes a zero-teacher or inactive coordinate. For each side/T-row/sign group, Cmass is conflicting absolute mass and Rmass is nonconflicting absolute mass. Transfer `M=min(Cmass,Rmass)`. Remove M proportionally from conflicting donors and add M proportionally to recipients. Leave unmovable mass at donors. Thus total signed/group L1 mass is conserved, including zero-capacity fallback to PARENT. No weak-occupancy threshold or amplification sweep. This operates inside an effective anatomical group; a biological circuit carrying out the transfer is not established.

For each blind sham, up to16 predetermined coordinate permutations form a prefix donor set with capacity >=M and remaining receiver capacity >=M. Use the first feasible partition, without looking at stored state or outcomes. Remove/add M proportionally in those sets. If none is feasible, leave that group unchanged and record unmatched mass. Four fixed sham seeds are averaged within stream; they are not independent animals. SPACE cannot qualify if mean matched transferred mass is below90% at either primary load.

In block controls each arm evolves on its own trajectory. U controls match the conflict fraction computed on their own current state, and blind shams match the SPACE transfer amount on their own current state. They are **instantaneous matched-action controls**, not guaranteed exactly equal realized block dose. Record cumulative raw signed/group/trace mass and transfer coverage. If a primary comparison differs by >1% in the relevant block dose (slow for DROP, both alpha traces and their combined mass for FAST, slow and moved mass for SPACE), that comparison is inconclusive and cannot license progression. No post-outcome rescaling of states or retuning. Exact budget identities are tested from a common pre-state.

## Outcomes and finite-menu oracle

Relative to the exposure-matched NO_WRITE branch, report target actual/reader gain and joint choice, mean and total positive collateral margin damage, worst collateral damage, and every old margin/choice. The primary improvement is mean positive damage reduction at NEW/bout6/+24h, at **both loads48 and96**. Loads12/24 are descriptive and low-load guards; no 'some load' selection.

For each FAST or SPACE family, a privileged oracle can choose only PARENT or that primitive for the block. Choose the primitive only if, relative to PARENT, target actual and reader margins are each within0.25Hz at both +300s and +24h, every parent-correct target remains jointly correct, no old joint choices are lost, no old margin falls by >0.25Hz, and mean positive collateral damage decreases at +24h. Otherwise choose PARENT. Retain all assigned episodes in estimates. This is an optimistic finite-menu witness, not an oracle over arbitrary temporal policies. Report raw primitive averages too.

For each primary load require: oracle benefit >=0.02Hz AND >=20% of parent mean damage; a positive stream-bootstrap lower interval; primitive selected in >=25% of streams; positive benefit beyond its control (apply the SAME oracle selections to the corresponding U_FAST or mean blind-space control); and the dose/coverage checks above. Both48 and96 must pass for the same family. Two candidate families are prespecified: use95% two-sided paired stream intervals (one-sided0.025 per family), requiring intersection across primary loads. This is a progression screen on known development histories, not a final statistical claim of a species mechanism. Also report worst losses and target gains; no scalar utility weight.

If neither family passes, stop mechanism/selector development and report the scoped negative result. No rescue threshold, delay promotion, recruitment change or replacement family. A failed dose/coverage gate means unresolved control, not a disproved action. Choose at most one passing family by the larger minimum damage reduction across primary loads, FAST only as a numerical tie-break.

## B. Action-value information (conditional)

Only after A passes, freeze an additional extraction/analysis implementation before new labels. At parent trajectory pre-write states (bouts1/2/6 and12 where relevant, both presentations), compare PARENT with one application of the chosen action, then return both branches to the parent rule for the same remaining episode. Labels are paired target-margin and collateral-damage effects; current-state features cannot see that future rollout. Split all conditions, snapshots, outcomes and sham draws of a stream together.

Use at most eight declared S1 quantities: slow L1, slow L2, slow coherence, alpha-fast L2, slow/write cosine, slow conflict-mass fraction, alpha-fast/write cosine, and fast-aligned/slow-conflict ratio. Fixed ridge penalty1 predicts paired target actual/reader margin change and damage reduction. Train-only normalization. Compare singles, all28 pairs, then the six possible additions to the best pair, entirely inside a four-fold grouped inner split for each outer held-out stream. Maximum three statistics, no category classifier, no all28-feature runtime model. Select the smallest subset meeting target noninferiority and positive damage benefit in inner predictions; otherwise choose PARENT. No feature selection on outer labels. Report policy value/regret and abstention, not classification accuracy alone; compare always-PARENT and the best fixed action chosen in the training fold.

B passes only if held-out selected actions preserve target margins/choices, reduce mean old damage by at least0.01Hz with positive paired intervals at both primary loads, and beat both fixed references. Otherwise conclude only that the tested compact selector is inadequate. This does not establish absence of all biological readouts.

A good snapshot predictor is not an analytic mechanism. V69 does not automatically convert fitted weights into a biological controller. Any conditional runtime pilot must first freeze a concrete <=3-statistic rule and explicitly label fitted coefficients as engineered if used. No hand-translated replacement after seeing held-out outcomes. New validation inputs must be generated after that rule is frozen.

## C. Contract for any conditional runtime pilot

At most one candidate;16 fresh globally disjoint streams from zero, all stored associations at12/24/48/96, acquisition, parent-common-eligible and all-assigned revision, anomalies, and a development-calibrated scalar/temporal dose control. The current V68 rates are references, never absolute acceptance thresholds for a different bank. Require +5pp day/load96 over concurrent parent and +3pp over control, positive paired intervals; acquisition actual margin within0.25Hz AND choices within5pp; day/load12/24 noninferiority within5pp; day/dose12 reversal noninferiority within5pp under both denominators; no increased anomalous false flips. Also run identical12-bout reversal dose spread across24h, with parent and candidate tested on that schedule and readouts aligned to the final bout. Report dose, span and retention delay separately. Temporal routing without later slow storage is not consolidation. Schedule robustness is required for promotion; a failed spread schedule is a scope limit, not proof all temporally local learning is useless.

If no concrete rule satisfying B exists, V69 ends as an action/selector diagnostic without a runtime pilot. This is preferable to V68's deliberately tested failed-calibration fallback. No learner promotion from an oracle or decoder alone.

## Execution and audit contract

Science uses the existing single-threaded NumPy/Numba environment, maximum two workers, stream-by-stream output. No large source tables loaded and no new download. Resource estimate is provisional: assess measured first-worker cost; a30-minute wall budget for A stops with explicit incomplete status, not relaxed gates or selective stream omission. Resume unchanged if needed. Do not claim V68's91-second runtime predicts this much larger assay.

Before outcomes: source/config/protocol/input hashes, parent equality, open-action/null equivalence, current-coordinate permutation checks where applicable, exact raw and post-decay mass identities, cap/route support, no-write exposure, and same-prefix anomaly controls. Exact single-step replay tolerance1e-9Hz; aggregate independent checks1e-7Hz. V68's8.9e-15 observed error is not a universal1e-12 requirement. Probes use copies and never change training state. Reproduce stored parent capacities. All changes and failures retained; no specimen stitching, physiology refit or new kernel timescale.

Deliver REPORT.md, aggregate metrics, key figures, model/config/source files, checks, source manifest and a compact <10MB result ZIP. Exclude raw trajectories/checkpoints, datasets, caches, prior archives and unchanged code. Preserve V68 and all earlier releases.
