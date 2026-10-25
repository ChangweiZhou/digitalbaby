# V151 cycle 1 — design and implementation audit

The user's three-cycle instruction authorizes the bounded interventions in the previously delivered V151 design. The design directory remains a pre-execution snapshot. Runtime authorization is recorded separately; no addition training, holdback access, trajectory replay or GitHub push is included.

Implementation audit before the first native intervention:

- A/B coefficients are computed from the same pre-state values, frozen code and target. Actual delivered coefficients include eta and the inherited observed-target warmup once.
- The intervention wrapper calls the unchanged native signed interface, does not feed the target, and pairs each event with identical timing and sensory state. Its zero and ordinary passive paths are separately checked.
- Operator perturbations are actual signed coefficients, with two predeclared radii; they never advance the observation counter and never become a learner-side preconditioner.
- Gain fitting uses only immediate centered displacement from the two calibration sources. All six multipliers are locked before any scored site is executed. Missing matches stay missing.
- C is frozen in the primary comparison. The separate code factorial preserves the pre-state update and explicitly includes its direct code-gradient term in the effect audit.
- The finite event/readout caps, native internal evaluations, prefix inputs, clone/hash overhead and one-thread limits are accounted separately. Checkpoints are immutable; all counterfactuals are discarded.
- Calibration target branches have only one immediate norm readout. Operator branches, scored branches and C-factorial branches have four query paths. This preserves the design's exact accounting distinction.

Qualified cycle-1 technical checks are recorded in CYCLE1_PREFLIGHT.json: output gradients, full 257-class reachability, clone isolation, no target feeding, matched zero/passive timing, native pulse orientation, and cross-process native resume without resetting warmup. The resume check uses six repeated observed-target applications on a discarded technical state. It is not a replay of V150 or a new trained history.

Numerical tolerances and the gain noise floor were specified in PRE_FIT_CLARIFICATIONS.json before any target was applied. They qualify computation, not scientific efficacy. A counterfactual response on a taught cue is E1 at most; derivative/interface checks are E0. The later scored assay cannot establish E3 arithmetic generalization.
