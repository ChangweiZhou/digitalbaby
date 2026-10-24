# V76 / Iteration 21: calibrated residual teaching across a native state interface

Prospective diagnostic protocol, 14 September 2026. The accepted full5177 reference and frozen reader are unchanged. No model is automatically promoted. The review and papers motivate controlled contrasts, not presumed answers. No iTDS/evidence state or new biological circuit is installed.

## Why this experiment

V75 showed a6.23pp benefit of prediction subtraction in the adapted/clipped route bridge, but label-only still exceeded native learning by25pp. Its93.89% sign accuracy coexisted with75.55% of probes below1Hz. This round separates residual subtraction from geometric normalization, fixes score units before outcomes, restores native gamma learning and alpha-fast/slow superposition, and observes delayed joint expression. It does not claim to implement a biological DAN residual circuit.

## Locked units and arms

All four primary hybrid arms use one fixed R0=2Hz: twice the frozen1Hz reader/actual threshold. In the simple conditional-mean example with10% symmetric label noise,0.8R0 is1.6Hz; that motivates headroom but is not a fit or guarantee. Prespecified R0=1 and4 normalized-error arms are sensitivity checks, never a post-result selection. Physical output bounds, Q, T, sensory reader, PN→KC encoding, all lifetimes and the native alpha allocation fraction remain unchanged across the four primary cells.

The matched factorial is LABEL_FIXED, ERROR_FIXED, LABEL_NORM, ERROR_NORM. LABEL uses rR0; ERROR uses rR0−score, r=2*revealed label−1. Labels can be corrupted. No clean target, cue ID, task ID, past tag, future outcome or delayed clone enters the update. The score is the present pair difference of observed alpha outputs minus the frozen sensory reader, in Hz. Both cues are observed using the inherited non-mutating5s clone convention. This simultaneous pair-level information remains an engineering permission.

For adapted activity dx, v=(dx_A−dx_B)mean(T_alpha), g=(dx_A−dx_B)mean(Q_alpha)restricted to T support. Update u=0.5*signal*v/(epsilon+D), epsilon=1e−12. FIXED uses a single median D from independent unlabeled calibration geometry. NORM uses per-pair g·v, clamped to[0.1,10] times the fixed median; gain extremes/zero support/clamps are reported. Zero geometry receives no update. This does not incorporate the derivative of clipping and does not promise a fixed actual step near saturation.

Calibration uses128 distinct unlabeled pairs and1024 random presentations; discard256 transient observations when taking the denominator median. Native adaptation evolves, all memory writes are disabled. No readout score, correct answer, noise realization, reward, candidate performance or test data chooses the median. All scientific arm constants and this recipe precede preparation and locking.

## State and event semantics

Every hybrid retains the full native fast(gamma/alpha), slow(alpha), adaptation, fixed feedback and output calculation. During the unchanged30s cue/135s gap sequence for A and B, gamma writes remain native; native alpha forcing is disabled. A pending alpha vector computed from the current pair is committed at the end of the330s pair. Its split is the native fraction f to slow and1−f to alpha-fast. Existing memory decays over the physical pair; the new end-of-pair vector receives no extra330s decay. Gamma's rule is identical across hybrids, but its values can diverge through feedback from their different states.

This is an external pair-teaching bridge with a native state interface, not a one-line replacement of the DAN equation. Native alpha routing originally differs by event/compartment; pooling T, observing pairs and committing at pair end are explicit changes. The residual is not recomputed from future responses during the pair. Preserve this semantic boundary in interpretation. No continuous simulator rewrite occurs.

ERROR_NORM_SLOW uses the same algorithm but puts all new alpha updates into slow; pre-existing alpha-fast state still decays normally. It is an allocation diagnostic, not a physiological candidate. REF is the unmodified native learner. V75_BRIDGE exactly replays the V75 adapted/clipped/route-error computation with R0=1, no gamma/alpha-fast plasticity and its original post-update decay. It is an anchor; differences from hybrids are bundled, not a pure allocation contrast.

Models keep165688 native mutable numeric bytes. The hybrid retains a pending5177-element update vector (41416 bytes) across the two-cue sequence, in addition to current pair inputs and temporary geometry. Report that workspace separately. No learned per-cue table is introduced. Encodings cached by the evaluator are deterministic fixed transforms of input PN patterns, not hidden learned state. The legacy wrapper allocates additional scratch state for parity; report its actual implementation overhead, not a compression claim.

## A: fresh acquisition, noisy learning and delayed expression

16 fresh histories; all nine arms on ANOMALY and REVERSAL streams. Each has64 randomly assigned cue pairs and1024 prequential observations, with256 warmup observations excluded. ANOMALY has10% independent flipped teaching labels. REVERSAL flips the latent mapping every256 observations. The evaluator holds the final observed block's mapping fixed for post-stream probes; it does not introduce an unobserved1025th-trial reversal.

Record forced reader sign accuracy, reader/actual signed margins, below1Hz fraction, and joint actual>=1Hz plus reader>=1Hz. After the stream, independently probe0/day clones for all64 pairs. Then branch the end state into12 new six-bout associations and a sensory/time-matched no-write control; probe old and new associations at0/day again. Retention clones do not advance the learning trajectory. Record old mean/worst positive actual-margin damage against no-write, new joint acquisition, state norms and update diagnostics. Removing alpha-fast in separate evaluator clones diagnoses expression dependence; it never supplies learning or stopping information.

Primary inference: on ANOMALY, compare the four matched differences ERROR_FIXED−LABEL_FIXED, ERROR_NORM−LABEL_NORM, LABEL_NORM−LABEL_FIXED, ERROR_NORM−ERROR_FIXED, plus their interaction, for online joint accuracy and END/day joint accuracy (10 two-sided exact history sign-flip tests, one Holm family). A positive material effect requires mean>=5pp and adjusted p<=.05.16 histories are the units;5000 paired bootstrap intervals accompany contrasts. Other metrics, scale/allocation arms, mixed transfer and REVERSAL are prespecified descriptive diagnostics, not outcome-selected winners.

## B: shared native prestates and revision/interference

16 separate fresh histories train unchanged REF on96 associations,6 bouts each, saving prestates at loads48/96. At each load challenge middle/newest focal memories under REV, STABLE and ANOMALY1 (one opposite bout then original contingency). Each of the eight nonlegacy arms branches from the same native prestate for12 bouts. REF18 supplies a fixed18-bout resource anchor from that same state; it is not an autonomous controller or a dose-matched mechanistic contrast.

LAST0/day records immediately after the final scheduled bout and its delay. COMMON pads all12-bout branches with6 inactive bouts' elapsed time to match18 bouts; assess0/day. MIXED then adds12 new six-bout associations; assess0/day. For every branch, a no-write clone receives the same sensory sequence/time with all writing off. All probes are independent. LAST12 versus LAST18 has different elapsed time and is not a matched-clock efficacy claim.

Retain target actual/reader/joint results, evaluator-only alpha-fast-erased target, old absolute joint accuracy, fraction of old initially-correct choices lost, positive mean/worst actual-margin damage against no-write, and direct worst positive loss versus REF12 at each endpoint. The worst victim is defined inside each assignment; victims may differ. Record new mixed acquisition separately. Average ages/loads within history for pooled intervals and report the entire stratified grid.

Diagnostic transfer qualification, for each non-REF12 arm: pooled REV target gain>=10pp with lower95%>0 at COMMON/day and MIXED/day; for every load/case at those two day phases, old accuracy lower>=−2pp, old-loss upper<=2pp, no-write-relative worst-damage difference upper<=0.05Hz, direct worst loss versus REF12 upper<=0.05Hz; STABLE/ANOMALY1 target lower>=−2pp and mixed-new accuracy lower>=−2pp. These correlated intersection conditions are engineering safeguards. All data are retained even if they fail. Passing does not establish physiology, natural-odor competence, compression or an autonomous persistent-learning core. No promotion this round.

## Integrity and delivery

Use32 independent scientific jobs, two CPU processes and single-thread BLAS, caffeinate, atomic hashed receipts, complete-grid verification and verified resume. All PN patterns are globally unique for this round and exact-disjoint from the eight named prior banks and interface calibration/evaluation; KC overlaps remain. Calibration/scientific panels are disjoint. These are synthetic streams through one specimen-derived interface, not independent flies.

Before science: verify encoded EVENT parity against the unchanged native schedule, exact V75 bridge replay, native gamma rule, alpha split and support, commit/decay timing, zero residual/support, gain clamps, immutable probes, no-write outcome invariance, clean-label/future-probe noninterference and rest composition. Lock config, protocol, all new scientific code, inputs, calibration, proposal and parent hashes. No post-outcome model/config/grid edits. Independent result audit reconstructs scores and victim metrics from saved vectors and verifies complete receipts/grids. Report all failures and causal limits.

Deliver REPORT.md, final tables, key figures, config/new scripts, acceptance checks and a manifest pointing to unchanged parents, target<10MB. Exclude raw per-seed vectors/checkpoints, source datasets, prior archives, caches and unchanged parent code. Preserve existing accepted reference and earlier bundles.
