# V181-R1 — mini local-plasticity qualification revision

Parent: V181-MINI-CGLP-D1, GitHub commit `51b1f6b36f769c8a204125895970b023c08a6f6e`. V180 remains stopped and unchanged. This revision does not allocate V182, enlarge the learner, change a scientific threshold, or authorize the official experiment.

The three technical cycles pass. The scientific result remains **NOT RUN**. The compact kernel, evaluator and resource envelope are qualified; official lifecycle/receipt/reducer integration must be completed before launch, using these frozen conventions rather than introducing another architecture.

## Cycle 1: extract and verify the genuinely compact circuit

Extract only canonical sparse B, the independently fingerprinted PN receptor indices and KC sides, receptor table, FE-PERM timing and per-side TopK. No native model, compartment, fast/slow value bank or older learned network is constructed. Execute the literal parent frontend and encoder definitions through AST extraction and compare empty state, EOS timing, punctuation, NUL, future-prefix controls, and all 256 bytes in ascending/descending order. Also compare 896 historically cached target positions from complete old source records. The historical cache is an interface reference, not a pretrained feature supplier.

The physical KC-H parameter tensor is 64×256. Every KC has three or four afferent uses; no H neuron repeats an input. H/T/O contain exactly 37,313 learned float32 scalars. Empty sensory state is 5,177 zeros; it does not receive an extra learned empty-context feature. Sensor arrays are write-protected after extraction.

Measured sensor arrays: 556,836 bytes. Conservative single-model inference-state accounting: 1,248,828 bytes, including one sensor, feedback, both compact and execution index representations, weights and inference buffers. Framework RSS is reported separately. A full mutable weights/moments/clock/dropout checkpoint occupies 458,410 bytes. These meet 8 MiB/1 MiB caps without lower precision or omitted state.

## Cycle 2: audit the actual local update

Use one physical forward and actual inverted-dropout mask. Compute output-local error and separate fixed FH/FT broadcasts; no learned-layer backpropagation. Validate H/T/output local derivatives with independent float64 finite differences, including hidden biases. This validates the declared local surrogate, not a global backpropagation gradient.

Use complete-record IDs for the coherence split. The unequal technical split has 24 versus 38 target sites and two distinct records per group. Weighted microgradient recombination differs from the full gradient by at most 7.45e-9 in this fixture. Unreliable record groups take the 0.25 floor.

NORMAL agrees with an independent Torch AdamW reference within the declared float32 tolerance. Update moments using the unmodified full gradient, then gate the realized H/T weight-and-bias movement, including decoupled decay. O receives unit gain. At a shared state, the independently checked uniform RMS control matches selective squared movement; realized float32 movement has relative energy difference 1.24e-6. This is rounding tolerance, not permission for a different strength control.

## Cycle 3: bounded continuous-life and resource tests

Exactly 32 complete technical record updates and 1,064 actual byte/EOS target updates were applied across NORMAL, FIXED-LOW, RMS-SHAM and CGLP. A uses four short complete records selected by length before fitting, B uses two complete records, and C uses one EDIT and one NOEDIT record per arm. This fixture tests mechanics; it is deliberately not a smaller efficacy experiment or a representative learning curve.

Each C fork restores identical B weights, both Adam moments, clock and dropout RNG. The forks advance to equal clocks and RNG states after equal-length complete records. Serialization preserves mutable state. Evaluation and own-byte generation preserve model, moments, clock and RNG; generation uses the model's own EOS and reports 64-step truncation separately.

Separate static reference calculations inspected eight record views/124 target views without applying an optimizer update. The independent full-batch stress fixture uses 256 synthetic targets and zero optimizer updates. Evaluator-side count profiling fits 64 countermodels with 6,656 record/335,520 target presentations. These costs are disclosed separately rather than counted as neural science or hidden under the 32-update figure.

The full-plan preflight includes sensory extraction, all outcome panels, worst-case 64-output own generation, the entire count grid and storage reserve. It projects 262 CPU seconds and 93,331,768 output bytes with the 1.5× reserve. Observed process peak RSS is 422,330,368 bytes. These are projections, not a completed main run or a wall-time guarantee under OS contention. Runtime must enforce the original 900 CPU-second, 1,800 wall-second, 1 GiB RSS and 256 MiB output limits; no automatic budget increase or retry.

## Frozen conventions clarified before scientific fitting

- Seeds are unsigned first-eight-byte SHA256 integers modulo 2^63−1. The incidence namespace is `V181|KC-H-birth`; weight and independent dropout namespaces are `V181|WORLD|birth` and `V181|WORLD|dropout`. Fixed feedback uses independent Torch generators 21 and 22. Weight variances and zero biases remain those of D1.
- Resolve the correction-selector wording by maximizing distinct changed training-record count first, total context occurrence count second, then the registered SHA tie. Replacement and record selection use only source training bytes. The selected contexts satisfy both context and modal-symbol occurrence minima; this clarification does not change the selected fixtures.
- Immutable IDs in the official runner must name original source records. A corrected record retains its original ID in EDIT and NOEDIT. Repetitions are exposures, not new independent records. No phase ID or edited-label hint enters the network.
- Use chronological batches of at most 256 real targets and include the actual residual; reset transient sensor state at real record boundaries only. Preserve optimizer/RNG/weights across the entire life, including A→B→C. No extra warmup, replay, phase-based gating or best-checkpoint restore is added.
- Cache active KC indices, not an all-corpus dense target×5,177 matrix. Densify at most one training/evaluation batch. Include all sensor/index/buffer bytes in deployment accounting.
- Reuse exact model state for all panels. Preserve SAME_KC versus ALTERED_KC versus changed-label separation. Score EOS separately. Keep unigram and IKN order1–5 × discounts .5/.75/.9 on the evaluator side; choose only on calibration, never by neural held outcomes.

## Data admission and unchanged decision

Both corpus pairs are authenticated and the fixed train/held records have no exact overlap. M1 has 41 changed-label held targets and 2,903 unchanged-label identical-KC targets; M2 has 35 and 1,661. Both correction fixtures qualify without changing sources, rules or coverage thresholds. M2 is opened development text, not sealed confirmation.

The scheduled official test remains two worlds × four arms ×560 complete presentations =4,480 neural records, now counted at 227,148 actual neural byte/EOS target applications. All original acquisition, correction, retention, competence, strength-control and composite gates remain unchanged. The technical tests do not establish any of them.

E0 engineering evidence only: neither memory improvement nor learned linguistic structure, reasoning, arithmetic, E2 or E3 transfer has been demonstrated here. Repeated prefixes, byte frequency and exact-context recitation remain countermodels. Gradual exposure and the existing arithmetic track remain unchanged. If the mini learner fails competence, report that result; do not expand it or return to V180 automatically.
