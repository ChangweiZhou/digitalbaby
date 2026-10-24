# Next bounded experiment: answer-revealing warmup, then answer withdrawal

This is a **design**, not an already launched evolution run. Freeze the V2 sealed benchmark and result. The first objective is to find a byte-stream learner that can actually use brief, post-choice correct-answer teaching on *new* one-operator expressions. Only then pay for a population search or longer expressions.

## Claim, level, and promotion

The initial claim is an E0/E3 feasibility question: can prior teaching improve first-response accuracy on never-reinforced expressions in this synthetic arithmetic family? Merely fitting taught examples is E1; lower MAE without a convincing exact or operation-balanced gain is a useful diagnostic, not the desired arithmetic gate. A later persistent-core claim additionally requires delay, intervening learning, and a causal prior-teaching state control. No benchmark score here licenses a general NLP claim.

## Exposure and teaching interface

Each trial presents a newline, sequential ASCII expression bytes, and `=`. The learner emits an integer before any feedback. In the warmup only, show the correct result *after that choice*, preferably as ordinary result bytes with a short separator through the same byte input channel. Numeric-target weight fitting may be kept as an explicitly privileged capacity control, but it must not be reported as the byte-teaching result. During later graded-only trials, return just the locked 0/−1…−5 unsigned consequence; scored trials reveal neither target nor grade. Keep the evaluator's arithmetic parser outside the learner.

For a one-operator life, reserve at most 384 distinct warmup expressions, 128 distinct graded-only expressions, 64 validation expressions, and 128 sealed scored expressions. This 704-expression plan fits the existing exactly balanced Stage-1 finite support of 840 expressions (division is the limiting stratum with 210). Canonicalize away spacing before exclusion. A curriculum can compare 0, 32, 128, and 384 warmup answers and assess performance after 0, 8, 32, and 128 grade-only experiences. Use fresh worlds and multiple seeds; never tune on the sealed set. Keep the original V2 task generator as a named benchmark, and disclose any later answer-balanced or nontrivial-zero variant as a different fixture.

## Minimal architecture and controls

Use a trainable temporal byte encoder with a small writable working state and output head; test it beside the frozen Full151 sensor and in a parallel Full151+byte route. Match parameter and experience budgets for KC-only, byte-only, and combined routes. Do not provide parsed operands, operator-specific answer tables, a symbolic arithmetic evaluator, or test-time target hints. First establish that a simple positive control can solve the fixture, then ask which learned route succeeds. A future topology search can include gated registers or multiple recurrent timescales, but each added family must be a distinct, auditable intervention.

At the same warmup checkpoint, fork identical learner state and future inputs into: (1) frozen readout/state, (2) true grade-only plasticity, and (3) same-histogram shuffled grades. Separately compare correct-answer warmup with permuted-answer warmup and a sensory-matched no-prior-teaching branch. Exact-key lookup, constant-zero, zero-shot, answer-frequency, and a symbolic non-fly solver are required controls. Read out unseen-expression exact accuracy, MAE, 90th-percentile absolute error, and every operator separately. Evaluate from cloned states so repeated read-only tests do not alter subsequent training histories. A one-off grade-phase hit is not progress if another operator or the worst error tail deteriorates.

The cheap representation gate is prospective and deliberately modest: across independent sealed worlds, the true-answer arm must gain at least 10 percentage points in exact accuracy over **both** constant-zero and matched no-prior-teaching controls, with positive gain on all four operators and no worse 90th-percentile error. Then graded-only continuation must retain the gain relative to its frozen post-warmup fork; if it damages it, repair the update rule before evolution. This gate only advances the search. Stronger absolute exact thresholds can start near 25%, then rise toward 50% and above as expression length increases; they should be fixed before each new confirmation, not fitted after seeing sealed results.

## Evolution after the gate

Breed within family from a rank-ordered elite set on novel-expression accuracy, per-operator coverage, error tail, and causal improvement. Preserve elites and lineage IDs. An absolute elimination threshold controls certification and curriculum advancement, while a **relative** threshold keeps parents available before certification. Do not fill 20% of slots with unrelated immigrants or reset an entire family when it misses a too-strict absolute gate. If all candidates fail the fixed progression gate, hold the stage, revise the representation or learning rule in a separately named experiment, or stop; do not call the next random population an evolved descendant.

Before a full run, microbenchmark 8–32 individuals end-to-end, including Full151 sensing, warmup learning, grade-phase updates, and sealed scoring. The prior fixed-feature V2 run completed 131,072 evaluations in roughly 91 minutes; a 512-label raw-byte GRU fit here took about 4–6 seconds **per individual**, so copying V2's search size would cost roughly 145–218 serial hours before grading and sensor overhead. Use compact aggregate checkpoints and a measured time cap; do not launch a six-figure candidate sweep on that assumption.

## Hostile stopping interpretations

- Correct-answer warmup lowers MAE but not exact novel accuracy: representation or learning objective still fails the arithmetic gate; do not advance difficulty.
- True answers and shuffled answers help equally: task priors, exposure, or answer-frequency effects explain the gain.
- Warmup succeeds, true grades hurt relative to freezing: repair grade-only credit assignment or retention; do not attribute the failure to capacity.
- Byte-only equals Full151+byte: the result belongs to the engineered byte route, not native MiniFly topology.
- Only repeated expressions improve: E1 recitation, not reusable arithmetic information.
- Exact gain survives withdrawal but fails delayed/intervening-learning probes: an E3 family-specific learner, not yet a persistent learning core.
