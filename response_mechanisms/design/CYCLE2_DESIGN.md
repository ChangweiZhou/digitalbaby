# Cycle 2: design revision after independent code audit

Prospective revised design, recorded before world290001 is executed. Cycle1 results remain exploratory. No final-world generator invocation has occurred.

## Material revision: correct the timing hypothesis
The cycle1 T implementation updated PN/KC timing traces once at cue completion and consequently tested cross-record order, not within-cue input timing. The independent code reviewer identified this mismatch before interpreting any pilot result. Preserve cycle1 as a failed instantiation of the intended timing hypothesis, not as its scientific negative.

Cycle2 T will form an antisymmetric timing eligibility on every observed cue byte. Separate cue-only FE0 sensor activity a_t and its current KC code x_t drive e += pre_trace*a_post − a_pre*post_trace with both traces exponentially decayed at the real interbyte interval. Traces reset at each record boundary. Tau10s matches the sensory trace timescale; the update sums over the 12 cue bytes and divides by their fixed count12. Epsilon .05, positive-weight floor and fixed per-KC incoming mass remain. All timing eligibility is completed before the teacher arrives. PN→KC weights change after the native teacher has captured its original KC code, so prediction and teacher agree on representation. T_OFF performs the same calculations with actual B write disabled. Teacher/newline bytes never enter this added sensor. This is a deliberately new within-cue timing variant and must not be equated with frozen A V3 P4.

## Shared label-blind output calibration
Set the same gain for J, J_ADD and J_SHUFFLE to 0.4237781016501581, the RMS of cycle1 FE0 W pre-teacher outputs after subtracting each row's channel mean. This uses all64 training observations (old and new); neither labels nor correctness enter the rule. It replaces provisional unit gain. Calibration is from an exploratory world only, fixed across every future arm/world. This is a units calibration, not a fit for performance.

## Better causal attribution and measurement
- Record decision-space decomposition of W−N_old, plus raw and uncentered differences
- Assert the complete probe grid is literally row-major 4×4 and each training bout contains every cue exactly once
- Report target F/G/H component strengths and all underlying mappings. Random overall balance does not mean zero main effects
- J's eligibility is coordinate-local recent×current receptor activity. Its update is a channel-owned normalized bank update with a sum-of-squared-eligibility normalization, not a strictly independent per-synapse learning rule. This shared postsynaptic normalization is an explicit architectural resource
- J_ADD and J_SHUFFLE have the same allocation, decay, limits, normalization equation and learning rate. Their realized updates need not have equal L1 because their own prediction residuals differ. Record realized L1 and do not call cycle1/2 shuffling a dose-yoked causal isolation. Consider a properly dose-yoked control in the next audit if necessary
- Source and runtime locks must cover all imported transitive repository files and exact Python, NumPy, SciPy and Numba versions. No external model weight download or install is needed

## Cycle2 pilot
World290001 only; four old and four new repetitions, all seven arms. Retain all outcomes and errors. Audit changes before running. Final worlds remain unopened. Added event timing work may change runtime; revise budget from this measurement, not from an unmeasured estimate.
