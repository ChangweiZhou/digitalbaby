# MiniFly v5 design review

Reviewed Claude commit **8e90d2e6c57d0394ce6246b9e9b69a75a9306ed4**, 20 October 2026. The received design and its v4 review are preserved verbatim in this folder. This is a design/data review only: **zero new training, no implementation, no three-cycle run, no GitHub upload**.

**Recommendation: revise before implementation of the full ladder.** The direction is better than v4's prerequisite ordering, but the new reference rule and pass gates are not yet reliable specifications. None of these fixes requires arithmetic-specific representations, attention, Transformers, ELM or backpropagation. Preserve gradual warmup and the primary goal: **retained exact own-output unseen-addition generalization per all training exposure and actual compute**.

## What improves

Learning byte selectivity and boundaries before evaluating the resulting codes addresses a real weakness in v4. Distinguishing fast adaptation from permanent commitment is a useful hypothesis. A strict slow-allocation floor repairs the literal no-early-commitment claim, and partial learning curves will make failures easier to diagnose. These are reasons to refine v5, not evidence that its mechanism already learns arithmetic.

The project contract keeps taught association (E1) as a necessary diagnostic and unseen relation reuse (E3) as the objective. Neither stable clusters, correct format, a fitted external head nor high confirmation counts establish E3. V5 is an unimplemented proposal; v4 demonstrated native primitive/interface properties but did not qualify exact acquisition in its new port at the tested dose.

## Findings, in priority order

### F01 — P1: a running mean of learned values is not a neutral output baseline

Source: [M5, lines 68–72](DESIGN_RECEIVED.md#L68).

M5 sets r_j to the running mean of the same learned z_j that it then thresholds and uses for teaching. That can absorb a persistent learned bit bias, not merely the unwanted native offset. For a stable learned value z=c, the proposed update makes r approach c, so the decision margin approaches zero and the error-corrective signal need not vanish. Context variation or a change in the frequency of output types can move this threshold without any change to the stored association. This is a failure mode to rule out, not a proof that every implementation must fail.

The design also leaves unclear whether references update on observed teaching events, generated guesses, evaluation queries or rest. Updating persistent references during evaluation would make answers depend on query order and violate the read-only evaluation contract. The native offset is activity/adaptation-dependent, not a universal −0.98 scalar: that number came from one fixture.

**Correction:** separate neutral physiological baseline from learned output signal. A target-free matched nonplastic/birth reference at the same activity, time and adaptation is one candidate; a calibrated reference frozen after a declared phase is another. Do not substitute the mean of the learned signal without validation. Specify initialization, timescale, update permissions and pre/post-event timing. Test always-on, always-off, imbalanced and context-varying bit targets, all payload bytes, own EOS, rest and a matched no-write branch. Training and generation must use the identical prepared reference. These are generic port checks, not supplied arithmetic rules.

### F02 — P1: listening can leak held-out answers and cannot be a free information budget

Sources: [lines 95–108](DESIGN_RECEIVED.md#L95), [D0, lines 115–122](DESIGN_RECEIVED.md#L115).

D0 explicitly includes correct answers and updates M2/M3. “No answer information reaches M2/M3” is therefore false as an exposure claim; the accurate statement is that they receive no explicit correctness/error target. Learning from answer bytes is legitimate pretraining, but it must be counted. With small operand ranges, sampling rather than replaying a fixed roster still repeats many whole facts. Sampling a fresh presentation does not guarantee a never-seen semantic pair.

**Correction:** reserve addition test orbits before any listening, fitting, calibration, supervised teaching or replay. Exclude both a+b and b+a for a held pair from every answer-bearing learner stream. Keep train, development/selection and final test partitions distinct. Count D0 bytes, unique complete examples, repeated presentations, expected-answer bytes, native updates, oracle fitting and replay separately. Match pretraining information across arms. D2 subtraction and D3 decimals should remain parked for the current addition-only study, rather than silently entering D0.

### F03 — P1: stability can let collapsed or oversegmented codes pass R1/R2

Sources: [M2/M3, lines 39–53](DESIGN_RECEIVED.md#L39), [R1/R2, lines 132–133](DESIGN_RECEIVED.md#L132).

A stable winner does not imply that different bytes remain distinguishable. A single class for all bytes gives perfectly stable assignments and consistently no boundaries. One class per byte can give consistently excessive boundaries. Neither qualifies useful segmentation.

There is also a conflict between stable single winning units and requiring all 64 units to be used evenly when the addition payload alphabet has far fewer than 64 distinct byte types. Homeostasis can keep rotating winners just to use otherwise idle neurons. The raw-bit bypass preserves access to information, but does not repair a misleading gate for the byte-value path used by M4.

**Correction:** measure type-weighted byte confusion/aliasing and reconstruction as well as assignment stability; allow unused capacity where appropriate. Define boundary consistency against a meaningful noncollapse/non-oversegmentation diagnostic, not simple repeatability. Evaluator-only structural annotations may score the learned segmentation but must never teach or select boundaries on the final test set. Freeze M2 identities before accumulating the context profiles that give those identities meaning, or explicitly specify how profiles are remapped as M2 changes.

### F04 — P1: R3 contains arithmetic and removes a useful acquisition check

Sources: [change table, line 29](DESIGN_RECEIVED.md#L29), [R3, line 134](DESIGN_RECEIVED.md#L134).

Correct answer length is partly a numerical result: 49+49 has two digits, while 99+1 has three. Requiring correct EOS at the true answer length before arithmetic rungs can create another circular prerequisite. Syntax, predicted length, known-reply recall and arithmetic correctness are different measurements.

The central hypothesis predicts that unconfirmed traces may fade; it does not predict that the fast store should be unable to recall a controlled taught reply immediately. Eliminating that E1 sanity test would again make a broken port hard to distinguish from a representation problem.

**Correction:** split R3 into generic port acquisition with own EOS, syntactic output validity, semantic length accuracy, and full unseen-answer accuracy. Keep immediate and delayed taught-cue recall as diagnostics, never as promotion evidence. Report exact answer length as an arithmetic diagnostic until that ability is learned; do not use it as proof of format-only competence. Permit bounded repeated calibration examples without turning the main fresh-example curriculum into pass-based training.

### F05 — P1: the dose calculation uses attenuated v4 writes as full-strength constants

Sources: [lines 126–128](DESIGN_RECEIVED.md#L126), [v4 measurement](../minifly_v4_dev_20261020/ops/PORT_BASELINE_DIAGNOSIS.json), [measurement code, line 8](../minifly_v4_dev_20261020/ops/port_baseline.py#L8).

The quoted measurement used coefficient (tanh(z/2)−target)/32: the residual divided by two and then multiplied by 1/16. Actual coefficients were approximately +0.01711 and −0.04539, producing matched-control changes −0.00877 and +0.02327. Those shifts do not establish a universal 0.01–0.02 change at full strength. A [separate coefficient-0.5 fixture](../minifly_v4_dev_20261020/ops/SIGN_AUDIT.json) measured a change of −0.16411. Saturation and state make naive linear extrapolation unsafe as well.

**Correction:** calibrate the proposed M5 reference and actual full-strength fast write with matched passive controls across representative states and active-cell counts. Derive finite dose caps from those response curves and measured feature recurrence. Do not claim that 3,000/30,000/100,000 records follow from the present measurements. Full-strength fast teaching is a new warmup hypothesis to test; fast decay alone does not prove that coupled consolidation, representations and references cannot commit prematurely.

### F06 — P1: segment closure and the first output frame need a causal specification

Sources: [M3/M4, lines 51–65](DESIGN_RECEIVED.md#L51).

Even ideal learned classes do not implement the stated timing automatically. Consider an illustrative number-run class N and separator class O for `12+34=`. This is an evaluator trace, not a supplied classifier:

| Moment | Closed segments, newest first | Open segment |
|---|---|---|
| After `+` | `12` | `+` |
| After first byte of `34` | `+`, `12` | `3` |
| Before first answer prediction | `34`, `+`, `12` | `=` |
| After the fly emits its first digit | `=`, `34`, `+`, `12` | emitted digit |

The equals segment is not closed before the first output byte. Its closure follows the class of an actually emitted byte. The segment ages shift at that event, and “steps since last boundary” is not automatically a monotonic output-position counter: both the `=` arrival and the first digit arrival can reset it. Using the expected digit's class to close `=` early would leak a future target.

**Correction:** lock a before/after event table for sensing, class lookup, closure, buffer insertion, prediction, teaching and own-output feedback. Specify whether boundary bytes are retained as their own segment (v5 requires this; v4's wrapper dropped the boundary byte). Use a separately declared generic reply-step counter if needed; a generic IO counter is not an arithmetic rule. Never supply the true answer length or expected future boundary to the encoder. Require identical pre-feedback frames when only the withheld target is changed.

### F07 — P2: R4 is neither a best-possible bound nor a cheap “analysis only” rung

Sources: [R4 and assertion, lines 135–140](DESIGN_RECEIVED.md#L135), [sizes, line 166](DESIGN_RECEIVED.md#L166).

A finite ridge fit is not the best possible linear classifier, and its failure is not a theorem about the native nonlinear/adaptive readout. Conversely, success of a head fitted on expected answers does not show that native local learning acquires it. High feature dimension alone does not imply that 128 examples cannot generalize; sample needs depend on the feature family, regularization and target complexity. The v4 oracle was a useful failed diagnostic, not a capacity proof.

At 100,000 problems and an illustrative five target events each, 2^20 KC features require about **3.81 TiB** if stored densely in float64. A dense dual Gram matrix needs about **1.82 TiB**. If v4's 2% activity is retained, even float32 values plus 32-bit sparse indices need roughly **78 GiB** for the feature matrix. These are assumption-based storage calculations, not measured v5 workloads; a streamed solver could avoid retaining these matrices. The head itself is much smaller.

**Correction:** call R4 an empirical feature-readout diagnostic. Lock a streamed/sparse solver, cache policy, stopping rule, CPU and peak-RSS caps before invoking the grid. Use development data for N/solver choices and reserve a fresh final test. Count fitting labels and all oracle compute. Begin with a measured small prefix and do not enlarge N just because the test score is disappointing.

### F08 — P2: commitment, spacing and post-freeze drift are still underspecified

Sources: [M6/M7, lines 74–78](DESIGN_RECEIVED.md#L74), [warmup, lines 99–108](DESIGN_RECEIVED.md#L99).

Lock q's update and decay, sign treatment, whether the hurdle uses |q|, per-record limits, minimum simulated spacing, record identity, replay eligibility, f_max and the exact above-threshold function. Otherwise consistent negative writes may be blocked, replay may manufacture “independent” confirmation, or repeated generic format bits may dominate commitment. Define identity without using unobserved future answer bytes to change a current write decision.

Also reconcile “frozen” with later low-rate M2/M3 updates and ongoing combination-cell homeostasis. Slow key drift is still key drift, especially after native values have been stored. Specify what is truly frozen, when versions change, whether replay re-encodes old records, and what that costs.

**Correction:** publish equations and a persistent-state/update-permission table. Validate contradictory signs, zero writes, repeated identities, separated fresh identities, zero replay, replay after rest and q decay. Report native fast/slow mass and predictive utility separately from confirmation counts. Strict delayed allocation is not itself evidence that arithmetic rules win.

### F09 — P2: “combination cell” needs an executable activation/wiring definition

Source: [M4, lines 60–66](DESIGN_RECEIVED.md#L60).

Four inputs can support a conjunction under a suitable nonlinearity and threshold, but summing four arbitrary signed inputs with global top-k does not guarantee an AND detector. The design must specify excitation/inhibition, required coincidences, input-family quotas, thresholds, time/segment features and competition. It must also show that a useful mix of units exists at the actual prediction frames, rather than assuming semantic columns are exposed.

**Correction:** define the generic local coincidence rule and audit its actual selectivity. Match feature-family degree, weight distribution, neuron count, active count and homeostasis between topographic and random wiring. Do not quietly select wiring by digit, operand, operator or carry labels. If a generic dendritic coincidence mechanism is added, identify it as a new intervention.

### F10 — P2: lock the causal E3 and retention protocol before the long rungs

Sources: [R5/R6, lines 136–137](DESIGN_RECEIVED.md#L136), [controls, lines 151–158](DESIGN_RECEIVED.md#L151).

“No-write” must name the prior episode and every persistent state it disables: native stores, q, output references, M2/M3, thresholds and replay bookkeeping. Shared D0 representations can be held fixed to test the incremental value-learning episode, but the permitted claim must say that. Queries must freeze all learned state, including output references.

Partial digit/carry metrics must state whether prefixes are teacher-forced or generated. Missing/extra digits, invalid bytes and truncation need explicit treatment. Carry annotations may be evaluator-only; they must not enter the learner. Chance/frequency and whole-key recitation baselines, paired worlds, retained OLD after intervening addition learning, and fresh confirmation tests remain necessary. R5's per-digit threshold does not imply the final whole-answer gate.

**Correction:** keep addition-only R5/R6 for this study, define the causal histories and OLD/NEW/rest chronology, and specify a sample/compute comparison against C-native. Separate engineering screening from statistical confirmation. Partial curves diagnose progress; they do not replace exact own-output E3.

## Corrections to the accompanying v4 retrospective

The retrospective is right that v4 did not settle its central mechanism hypothesis, and that my implementation's reversed sign and replay defect must be excluded from efficacy. It overstates several conclusions:

- The 0.01–0.02 event shift was measured with attenuated coefficients, not the proposed full-strength rule.
- Too few samples and too many features do not by themselves prove that any feature family must fail to generalize. Coverage failure before learned boundaries is especially ambiguous.
- Stable unlearned random closures existed; meaningful learned segmentation was not demonstrated. “No segments ever formed” is not the literal implemented behavior.
- V4 reported oracle-byte scores and teacher-forced byte/residual diagnostics, not only whole-answer accuracy. The partial curves proposed for v5 are still a useful improvement.
- One-off recall before decay remains a valid fast-memory sanity check. Failing it does not disprove consolidation, but removing it would hide port failures.

These corrections preserve the original v4 verdict and all frozen artifacts; they do not retrospectively turn an engineering failure into a success or a family-wide rejection.

## Minimal revised ladder

1. **Port and native bridge first:** isolate the baseline from learned values; verify all bit/EOS directions, readonly generation, rest and bounded taught-reply acquisition. Profile the actual intended write strength. Retain a v3/native anchor on a shared addition range for diagnosis, not as an alternative new answer mechanism.
2. **One locked addition-only listening split:** count every heard answer byte; learn byte identities, then context classes; check aliasing, collapse and causal segment timing before freezing.
3. **Learned-code coverage:** use the frozen representations and a resource-capped external diagnostic with train/development/test separation.
4. **Matched native learning:** compare strict gated/native allocation, topographic/random and segment lesions on identical information streams. Keep immediate taught recall and syntax/length/digit curves as diagnostics.
5. **Primary E3 test:** exact own-output unseen addition after delay and intervening addition learning, against no-prior-teaching/shuffled/recitation controls, plotted against all samples and actual compute. Only then consider later operations or trimming.

The fast-store/commitment split is allowed as a hypothesis. It is not treated as a proven synonym for gradual warmup. No arithmetic relation, digit/place/carry feature, regex delimiter, answer-length hint or decoding constraint is needed for these corrections.

## Review limits and next action

This review used the two received Markdown files, the published v4 implementation/measurements and the standing objective contract. It ran only arithmetic/resource calculations and read-only source inspection, not a new neural training test. The mean-reference and boundary issues are design failure modes to resolve; no v5 performance was measured. The next action is a written protocol revision closing F01–F06 and locking the remaining definitions/resources before another three-cycle implementation exercise.
