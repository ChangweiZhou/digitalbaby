# V144 — three bounded development cycles

24 October 2026. **Three cycles completed with limitations. Unseen-addition improvement is not established.**

The exercise implemented the integrity fixtures, paired native learning-signal diagnostic and short complete-addition/EOS reference trials in the [published design](../minifly_v144_addition_readiness_design_20261024/DESIGN.md). Gradual warmup, raw byte IO, addition as the goal, and the bans on arithmetic-specific structure, attention, Transformers and ELM kernels were retained. No new replay, normalization or bank-selection mechanism was introduced.

The most useful result is diagnostic: **the native learner retains a substantial informative residual and continues feature updates; complete addition still shows weak held transfer at the tested dose even when the reference produces its own EOS.** Neither observation is a new native arithmetic efficacy result.

## What ran

| Work | Actual execution | Evidence boundary |
|---|---|---|
| Cycle1 | Source/asset restoration, token/trace/gradient fixtures, fail-closed rejection fixtures, native instrumentation equality | E0 qualification; synthetic/development exposure only |
| Cycle2 native | One original D2 world, LEARN/FROZEN,150 passes each,22,500 total native records, one-hour final rest | Parity component diagnostic; same development world, not independent V143 confirmation |
| Cycle2 reference | One V144-EXERCISE-C2 world,3 modes,64 passes each,14,592 addition records | E1 acquisition and descriptive held addition; independent reference, not the native fly |
| Cycle3 | Synthetic resume equality, actual checker exit tests, source/array/output reduction, prospective selection revision | Technical qualification/revision, no new scientific campaign |

The27-configuration Stage C search,4,096-pass continuations, data scaling and18 holdback fits were not run. CAP-H1/H2 were not opened. No formal native addition campaign or new core adoption occurred.

## Native learning-signal result

The final per-prompt LEARN/FROZEN outputs exactly match V143's published development D2 outputs. Extra probes preserve complete persistent state, and FROZEN learned-bank code hashes remain invariant.

| Final balanced held parity accuracy | LEARN | FROZEN |
|---|---:|---:|
| Learned bank |82.35%|33.09%|
| Actual summed system |69.85%|48.90%|

These are development parity results. They do not replace V143's locked four-world partial verdict or test ordinary native addition/EOS.

For LEARN:

| Mean over training window | Passes10–20 | Passes125–150 |
|---|---:|---:|
| Absolute informative-bit residual |0.4381|0.04267|
| Absolute residual of each constant bit |0.008980|0.0008309|
| Feature-update L2 norm before decay |0.08269|0.009550|
| Decay-update L2 norm |0.0003785|0.0005603|

Late feature updates are about17 times larger than decay updates. The informative bit still dominates the per-bit update norms. Thus taught-item success did not remove all upstream teaching. The strong starvation explanation is unsupported for this exact recipe; slow or poorly directed learning remains possible. This observation is temporal, not a causal intervention on relative learning speeds.

The learned hidden effective rank falls from17.75 to2.95; final within-class KC Jaccard is0.8065 and between-class Jaccard0.7608. FROZEN remains at rank17.75, with within/between overlaps0.2415/0.2461. The learned representation is highly shared but retains class-related structure. Increased overlap alone does not prove a harmful collapse, especially for a two-class task.

Raw per-event credit vectors/gate masks and raw-alpha histories were not retained. Norm telemetry cannot identify every direction-level credit failure. No native histories were rerun to retrofit these missing fields.

## Complete addition result at64 passes

All models use the same byte trace, nine-bit byte/EOS output port, H128, learning rate0.03, decay0.001 and1,024-record gradual ramp. Models have no arithmetic features/parser, target length, forced EOS, output mask to digits, numeric decoder or beam rescue. The expected sum is computed only by the evaluator that supplies training feedback and scores outputs.

The55 unordered operand pairs are split into40 training and15 held orbits, with neither orientation of a held orbit reinforced. This fixture has76 oriented training prompts and24 held prompts. Every model receives4,864 training presentations and11,904 observed target tokens, including EOS. No held response is fed back. The learner's own generated prefix drives subsequent output steps; teacher-forced scores are distinct diagnostics.

| Reference mode | Taught complete orbit accuracy | Held complete orbit accuracy | Held payload-only accuracy | EOS within8 decisions | EOS at correct held length |
|---|---:|---:|---:|---:|---:|
| BP |77.50%|6.67%|8.33%|100%|86.67%|
| DFA |81.25%|3.33%|4.17%|100%|73.33%|
| Feature-frozen lesion |15.00%|6.67%|8.33%|100%|56.67%|

Payload-only accuracy above averages oriented prompts; complete accuracy and correct-length EOS average within orbit first. Termination by itself is not correct stopping or arithmetic success. Complete scoring includes the entire payload and its own EOS.

Feature learning helps taught acquisition at this dose, but BP does not beat frozen features on held complete answers; DFA is lower. This remains compatible with overfitting or insufficiently effective feature learning, not a missing-EOS-only explanation. Taught accuracy is also incomplete, so it is not a pure generalization-only deficit.

All six declared countermodels scored0 on this particular held set. There are only15 held orbits in one developmental world. A few correct predictions and a weak baseline do not establish robust transfer, native capability or statistical significance. The broader capacity screen remains unexecuted; the short fixture cannot prove larger models or longer training impossible.

The scalar-only and blurry-feedback phases were not tested. Success under them cannot be inherited from this supervised warmup exercise.

## Revisions justified by the exercise

1. A helper named `common.py` collided with the old native import; it is now uniquely named. Source restoration includes lazy-import dependencies and assets missed by the old import-only closure inventory.
2. Reference byte traces now use the exact inherited event API. The literal0.95 approximation failed bitwise equality.
3. Gradient fixtures select nonzero entries; an initially inactive-unit test was vacuous. The qualified maximum finite-difference error is3.92e-11.
4. The future screen preserves one finalist per hidden size. Early64-pass complete-score/CPU ranking could otherwise discard all larger, slower learners. Teacher-forced development loss is only a tie-break, never a substitute for own-generation success. This amendment preceded opening the reference trial results.
5. The report distinguishes any own EOS from EOS at the correct answer length and from complete exact accuracy.

The prospective [V144-R1 revision](REVISED_DESIGN.md) preserves the goal and original budgets. It does not license hand-built arithmetic or add mechanisms to compensate for poor results.

## Audit and accounting

The author-run [artifact audit](FINAL_AUDIT.json) verifies five execution source locks with zero mismatches,48 saved code arrays against their hashes, finite metrics/events, source-locked D2 agreement, exact reference scoring and actual completion of the developmental roster. Checker subprocesses return0 for valid input and2 for hash/closure/missing-control/settings failures. Synthetic resume produces exactly equal model state.

| Record applications | Count |
|---|---:|
| Native paired diagnostic |22,500|
| Addition reference trials |14,592|
| Native qualification fixtures |48|
| Synthetic reference fixtures, including resume rehearsal |36|
| **Total** |**37,176**|

The addition trials reveal35,712 target tokens; synthetic reference fixtures add92. Native diagnostic/fixture records each supply one observed parity answer byte. Total target tokens for these record applications are58,352. Serialization/finite-difference probes are technical queries, not additional full record applications. Their measured process costs remain in the attempt evidence.

Retained native diagnostic CPU is1,602.73 seconds; retained addition fixture CPU is2.52 seconds. These differ in task and implementation and are not a native-versus-reference efficiency comparison. Several setup/import/fixture commands preceded complete process-ledger coverage. Their CPU is partially known from logs, so the above numbers are not an all-in campaign cost. No compute-efficiency improvement is claimed. Cloning/global competition remain inherited native costs; the exercise does not establish computational sparsity.

Four preparation/fixture failures are preserved: initial missing native configuration, namespace collision, approximate trace mismatch and missing lazy constructor import. None reached native teaching; the last had already executed three synthetic reference warmup records, which are included above. Repeated successful qualification fixtures and the synthetic resume duplicate work are counted. The finite-difference fixture repair is also preserved in cycle notes.

Native and reference workers ended normally. Caffeinate followed active work; nothing from this exercise remains running. Raw event logs, KC arrays and checkpoints remain local. SUMMARY.json/SCREEN_TABLE.csv are compact results; ATTEMPT_LEDGER.jsonl is the local collated event ledger. This is not an independent scientific acceptance audit or a formal result package.

## Next step

Finish and qualify the finite Stage C dispatcher/checkpoint admission protocol, then run the already bounded capacity screen under explicit execution authorization. Keep H1/H2 untouched until a configuration is frozen. Do not add replay, homeostasis, arbitration, extra arithmetic data or another model family now. The screen should decide whether a broader/longer reference can learn complete addition at a useful cost, and which failure remains before a native integration experiment.

The present result justifies continuing that diagnosis. It does not justify adopting a new core, declaring the addition goal solved, or shifting the goal back to parity.

Cycle evidence: [cycle1](cycles/CYCLE1.md), [cycle2](cycles/CYCLE2.md), [cycle3](cycles/CYCLE3.md), [revised plan](REVISED_DESIGN.md), [compact summary](SUMMARY.json), [audit](FINAL_AUDIT.json).
