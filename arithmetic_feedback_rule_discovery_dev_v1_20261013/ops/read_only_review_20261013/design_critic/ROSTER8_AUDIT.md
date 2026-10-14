# Prospective audit of the revised eight-arm recipe

The parent supplied a further saved-state diagnosis: the cue matrices retain rank 65; R1 mean cosine rises from .542 before the equals byte to .824 after it and .929 after REPLY; EOS conditional prediction is correct, and teacher-forced complete answers equal autonomous complete answers (41/199). A first-byte least-squares diagnostic improves when the ridge penalty is removed, but fitting payload and EOS contexts together remains weak. These observations support testing representation discrimination and categorical learning together. They do not prove erased information, a hard rank capacity limit, or a linear-classifier accuracy ceiling.

## Roster judgment

RAW-RLS, G-SQ, G-CE, L-SQ, L-CE, G-FROZEN-CE, L-FROZEN-CE and LEGACY form a defensible fixed roster. G is the unchanged depth-1 GRU reader; L is a bounded last-32-byte chronological embedding buffer followed by a generic 64-unit tanh MLP. SQ versus CE compares the objective under synchronous record learning. Frozen CE twins test whether learned representation contributes. Only G-CE/L-CE are selectable, if declared before outcomes; a later good SQ result cannot be silently promoted.

M059 supplies a generic lag/context motif rather than arithmetic positions. M001/M024 supply generic nonlinear cells and representation adaptation; their original evidence restrictions remain. This is an engineered architecture experiment. The G/L contrast changes the whole reader recipe, its state and its parameter count; it is not an equal-parameter isolated memory-mechanism contrast. Actual production CPU and persistent/scratch state must be charged. Equal sample counts do not imply equal computational cost.

## Required precise definitions

1. The lag buffer stores public controls and payload identities in chronological order. It contains observed prior reply bytes during training and generated prior reply bytes during inference. It cannot contain future target bytes, an answer-length field, an operation field, numeric values or evaluator metadata.
2. Padding must be zero vectors in unused slots, distinguishable from the learned NUL-byte embedding. BEGIN resets the transient buffer; EOS participates as a legal target. Right alignment and the 32-event window are fixed generic context policies. Long-stream eviction is disclosed and logged, not hidden by the short arithmetic prompts.
3. The joint recipes freeze all weights during a framed record, compute the teacher-forced objective from the actual arriving feedback bytes plus EOS, and perform one generic update only at actual EOS. Generated EOS and no-write EOS cannot step the optimizer. A new BEGIN aborts pending permitted loss exactly as specified.
4. The encoder/head optimizer, gradient averaging, clipping and zero head initialization must be identical within G-SQ/G-CE and L-SQ/L-CE. SQ and CE have different loss scales; report their gradient norms and clipping frequency. Do not claim that a common numerical learning rate perfectly isolates the statistical objective.
5. Learned/frozen twins share birth parameters and the same head update algorithm. Frozen means all reader parameters, including byte embeddings, stay fixed. Optimizer moments exist only for genuinely trainable arrays; no padding arrays manufacture a resource match.
6. Parameter/state counts include the categorical head and optimizer moments. A flat 32 by 16 reader creates a much larger input matrix than the GRU. The buffer is not free, and simultaneous record-gradient scratch is additional state.

## Outcome and attribution safeguards

Do not turn teacher-forced accuracy into the result. The diagnosis makes first-byte and conditional-payload discrimination the leading acquisition diagnostic; EOS handling is already correct on these saved examples. Exact autonomous output plus EOS remains scored in all primary probes.

The same actual teaching records, held equation-family split, six budgets, matched W/N_OLD/SHUFFLE_OLD histories, binary new cohort, rest and retained held probes must be used in every arm. Evaluator family membership is never supplied to the core. Claims remain limited to interleaved single binary expressions, not genuinely mixed expressions with precedence.

The offline fit is a read-only counterfactual diagnostic. It must not initialize the new model, select its learning rate, select held labels, or enter its scored branch. Its 71.9% result is not evidence that the architecture can attain that accuracy online at the stated sample cost.

Keep existing comparative sample-score J and paired common-production-CPU score G rules. G must include all encoder/head learning and internal computation, including the larger lag input matrix; probes, snapshots and audits remain separately reported operations costs. A conditional confirmation run can retain the original 12-comparison family for the selected candidate (eight per-operation causal contrasts plus J/G against matched frozen and legacy). Additional architecture/reference contrasts need explicitly descriptive status or a separately declared family. DEV selection needs a deterministic tie rule and may select at most one candidate.

No arbitrary acquisition percentage should be introduced after these fixtures. Better taught recall only proves acquisition. A small held gain remains a small E3 effect; failure of an all-operation comparative criterion remains a valid no-advance result. The fully correct held grid is a separately scoped capability claim and cannot be inferred from merely positive mean contrasts.

## Qualification

Independently finite-difference the complete record loss and head/encoder updates for both readers and both losses, including repeated byte identities, NUL, EOS and buffer eviction. Test chunking parity, state snapshots with pending loss, pure generation, no-write counters/moments, evaluator isolation and actual sample-budget jobs. Preserve every failed technical fixture. Fresh formal namespaces, dispatcher, eight-/ten-hour boundary and output packaging need qualification before science admission; none is authorized by this audit.

The prior jobs took seconds, but a new joint learner must be timed as a complete job. The eight-arm roster need not create an hours-long confirmation for every arm: fixed DEV followed by at most one preregistered candidate and its fixed control roster keeps interpretation and runtime bounded. Do not grow sample counts, vary widths or add replay while treating the same version as unchanged.
