# V144-R1 — cycle revisions and remaining execution boundary

This amends the [published V144 design](../minifly_v144_addition_readiness_design_20261024/DESIGN.md) prospectively. The fixed goal, native parent, gradual warmup, no-arithmetic-structure constraint, byte/EOS metric and resource ceilings are unchanged. Historical V143 sources/results and V144's original published memo remain intact.

## R1.1 — preserve capacity breadth in the early shortlist

The original rule shortlisted the three best configurations globally at 64 passes. Complete answers could still all be wrong at that point, or smaller models could learn earlier. A CPU tie-break would then discard all larger models before the prolonged-practice test. That would undermine the purpose of a broader capacity screen.

Before any Stage C fixture result or full-screen result is opened, replace that rule with **one finalist from each H in {128,256,512}**. Within a hidden-size group rank by mean development complete orbit accuracy at 64 passes, then lower development teacher-forced BCE per token, then lower measured CPU, then lexicographic (learning-rate, decay) order. The secondary loss is development-selection information, not held-generation success. The final common configuration is still chosen after 4,096 passes by development BP complete orbit accuracy and the original final tie-breaks.

The grid still has 27 configurations, three finalists, at most 66 development starts and 18 holdback starts. Exposure and CPU ceilings do not increase. H1/H2 remain unopened by the exercise. Failure of an early small fixture does not permit shrinking the addition endpoint to parity or hand-constructing arithmetic features.

## R1.2 — exact inherited front end and meaningful gradient fixtures

Use the parent's actual FE-PERM event API, including its timing, for both prompt and observed/generated prefix processing. A literal-decay approximation is not bit-identical. Finite-difference fixtures must exercise nonzero gradients rather than accidentally inactive hidden units. These changes repair implementation/qualification, not the scientific architecture.

## R1.3 — validity is separate from behavioral success

The checker must reject source/asset hash failure, unlocked runtime modules, missing/duplicate/extra histories, closure failure, invalid settings, missing metrics and partial histories. Its CLI returns nonzero for rejection. Passing this checker certifies only the covered validity conditions. It does not turn technical fixtures or reference output into native addition efficacy.

Use actual saved KC arrays/hashes for FROZEN invariance. Use generated output bytes and own EOS for complete-answer reduction. A repeated trained relation is not an independent held relation; neither orientations nor output tokens are independent worlds.

## R1.4 — scope of this three-cycle implementation

The exercise implements and runs the paired native D2 diagnostic and a 64-pass ordinary-addition interface trial on `V144-EXERCISE-C2`, using BP/DFA/feature-frozen references. This world is developmental and is never a holdback or confirmation replicate. The original four named CAP-D/H worlds are not used for tuning or rehearsal by this exercise.

`capacity.py` deliberately rejects the full-screen and holdback world names and any request beyond 64 passes. Its three fixture fits have a separate 180-process-CPU-second ceiling each. This operational guard cannot be removed merely to call the exercise a completed capacity search.

The complete 27-configuration sweep, 4,096-pass continuations, data-scaling roster and 18 holdback fits have **not** been run by the exercise. Their finite dispatcher, exact resource admission and checkpoint/continuation manifests still need implementation and qualification before an explicit request to run that screen. No native addition efficacy roster or new mechanism arm is admitted automatically.

The reference fixture may establish acquisition and own stopping under this limited dose. Its outputs are not a promotion test for MiniFly. Persistent E3 would still require the no-prior-teaching and shuffled-content controls, actual later learning, retained held performance, and native integration specified in the original design.

## R1.5 — telemetry coverage

The native runner saves each pre-write residual/margin and actual strength schedule, per-bit feature-update norms, total feature/decay/bias-update norms, active-unit fraction, code arrays/hashes, class-conditioned overlap and hidden effective rank. Raw `G e` vectors and per-event ReLU masks are not retained; no claim about the full direction of credit assignment is permitted from aggregate norms. Applied signed coefficients are determined by the source-locked residual/strength/clipping rule; per-compartment raw-alpha/clipping telemetry is not separately instrumented.

These limitations do not prevent answering whether a substantial informative residual and nonzero feature updates persist. They do prevent calling this a complete direction-level causal credit audit. Any later directional intervention must first save those missing quantities on its own fresh, bounded development stream. Do not rerun the two D2 histories to retrofit them.

## Publication and resources

Reports include all development/fixture attempts and disclose setup commands whose CPU was not captured by the process ledger. Unknown campaign cost stays unknown. Native state cloning and global KC competition are inherited costs; no scalable sparse-computation claim follows. Byte/gradient fixtures and native diagnostic records are listed separately from addition feedback applications.

Raw event logs, KC arrays and checkpoints remain local. Source, revisions, compact metrics and audits may be prepared for publication, but this exercise does not itself push results or start another experiment.
