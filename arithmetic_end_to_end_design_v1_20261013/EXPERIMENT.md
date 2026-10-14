# E2E-RULE V1 — learn arithmetic through one end-to-end byte model

13 October 2026. **Revised design only. No model implementation, training, three-cycle qualification, DEV, confirmation or background launch is authorized by this document.** Static data enumeration is reported separately. The completed RAW-RULE/STREAM preparations, adopted core and frozen results are unchanged.

## 1. Goal and permitted representation

Improve **retained exact own-output generalization to unseen four-operation problems and bounded mixed expressions per external training sample and actual computation**, using observed feedback. The entire operative prediction path is one trainable model:

    raw byte identities → learned embedding → learned sequence computation → learned 257-way logits → own bytes and own EOS

No number representation, digit ordering, operand slot, operator ID, arithmetic module, place-value channel, answer template, supplied intermediate calculation, symbolic solver, numeric loss or restricted vocabulary is present in that path. Every parameter that represents input, context or output is trainable. Generic architectural choices—recurrence, learned position embeddings, causal attention, normalization and finite buffers—are allowed, but cannot encode arithmetic semantics. All configurations use the same code across operations and phases. Model parts may change; the task objective does not.

Training data and the independent evaluator necessarily generate correct arithmetic answers and audit task membership. Their scalar values, ASTs, family identifiers and phase/operation labels never enter learner calls. Mixed-expression feedback contains **only the complete observed answer**, not a chain of calculations. A learned reuse of earlier examples is allowed; automatically manufacturing arithmetic-related training examples inside the learner is forbidden.

This is an engineering backprop comparison. It does not claim an unchanged Full151, biological locality, NLP competence or a solved general persistent core. Intended evidence is bounded E3; taught recall is E1, formatting changes alone E2. Exact answers remain mandatory; numeric proximity is diagnostic only.

## 2. Why the design changes

The saved technical evidence separates failures that earlier summaries combined:

- A's joint CE models often learn immediate EOS. Over the last128 G-CE records, payload predictions are0/174 while EOS is128/128; declining loss is not arithmetic acquisition. G-SQ's final31/202 is the constant output `0`.
- The same generic GRU learns B's taught addition and multiplication perfectly after many repeats. This rules out a categorical byte-I/O impossibility, not short-budget inefficiency.
- B ANSWER-D unseen addition is15/36 before interference at64 passes, falling to8/36 at256 passes. Approximate structure appears, but more memorization does not yield reliable exact rules.
- Correct prior addition speeds taught multiplication, while unseen products remain near zero. Old held addition declines with later learning. Old taught addition was not measured after multiplication, so its loss cannot be asserted.
- The unrelated cohort produces large updates while learning almost none of its keys. A correct derivative is not a guarantee of useful optimizer dynamics.

Sources: [saved-receipt reduction](../arithmetic_two_designs_20261013/read_only_diagnosis_20261013/SAVED_RECEIPT_DIAGNOSIS.json), its read-only reducer, and immutable cycle receipts. These are single-world technical findings, not new population claims or family-wide exclusions.

The revision removes the old normalized-head/RMS recipe from the primary comparison, uses a conventional differentiable model implementation, changes blocked teaching into balanced microblocks, measures old taught facts as well as old held relations after later learning, and adds actual mixed-expression learning as the intervening task. It does not substitute a simpler success criterion.

## 3. Four fixed configurations

| ID | End-to-end model | Raw-record replay | Role |
|---|---|---|---|
| G0 | Generic standard GRU | None | Reference |
| G1 | Identical GRU | Reservoir64, at most one prior record per permitted external record | Candidate |
| T0 | Small causal Transformer | None | Candidate |
| T1 | Identical Transformer | Same reservoir/replay policy | Candidate |

Within G and within T, birth arrays match exactly across replay conditions and causal histories, but mutable storage is independently owned. Across architectures, match worlds, data, optimization and feedback; charge real parameter, computation and memory differences. This is a complete model-recipe comparison, not proof that attention alone caused an effect.

Inventory lineage: M001/M007 recurrence; M031/M033 coherent classification and optimization; M026 learned causal prefix access; M050/U047 real-experience reactivation. M059/LAG32, REL2 fixed projected products and RAW V1's additional zero-input GRU steps remain separately tested variants. Neither attention nor replay is described as an untouched family. The full overlapping108/136/254 inventories and older71-row view remain linked in [the complete review](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/inventory/FULL_INVENTORY_REVIEW.json). No survivor census or old verdict is rewritten.

## 4. Generic input, models and output

All256 payload bytes are categorical identities. Input-only BEGIN257 and REPLY258, plus actual EOS256, are public generic interface controls. NUL, LF, spaces, `=` and parentheses are ordinary payload. Arithmetic payloads are relabeled by one uniform random permutation of all256 byte IDs per world, applied equally to every model/history and all input/output payloads. The model never receives the permutation or its inverse. This removes any opportunity to rely on ASCII magnitude; it is data relabeling, not a semantic encoder.

A frame is BEGIN, actual prompt, REPLY, actual observed reply, actual EOS. No task or operation side-channel exists. For ordinary observation streams, the same interface can have an empty prompt and an observed continuation. The core code and learning rule remain unchanged; language capability is not tested here.

### G model

Learned259×16 input embedding; one standard64-unit GRU layer with two bias vectors per gate; learned affine64→257 output. No unit normalization of `[h,1]`, fixed features, hash address or arithmetic component. Use the published framework GRU convention, including its reset-gate placement, and identify the exact implementation version in qualification. This is not a claim that it is numerically identical to the previous custom GRU. Parameter count:36,593.

### T model

Learned259×32 input embedding;128 learned position embeddings; two pre-normalized causal Transformer blocks, four heads, width32, FF64 with GELU; learned final LayerNorm and affine32→257 output. All weights, biases, positions and normalization scales are learned. Dropout0. Parameter count:38,017. No fixed sinusoidal/numeric embedding, operand-specific head, task route or symbolic scratchpad.

Causal and absent-padding masks enforce chronology and missing positions only; vocabulary/output masks are forbidden. At prediction the current position can read earlier raw-token states through learned Q/K/V. Attention does not receive any label describing what those positions mean. Architectures and parameter counts are near-sized, not artificially made equal with unused weights. Model plus two float32 Adam moments is439,116 bytes for G and456,204 for T; gradients, caches, replay and runtime state are additional.

Initialization is task-independent: byte/position embeddings are Gaussian with standard deviation1/sqrt(embedding width); standard GRU matrices and its two gate biases use the framework uniform hidden-width bound; other affine weights use Xavier-uniform with zero biases; LayerNorm starts gamma1/beta0. Output matrices are not initialized to zero. All these arrays remain trainable. Derive modelseed from the first8bytes of SHA256(`E2E_RULE_V1|world|architecture`), reduced modulo2^63; replay does not enter the modelseed. Use an independently derived `world|replay` RNG shared across the G1/T1 data-matched policies, never the evaluator RNG. Initialize arrays in sorted parameter-name order with the explicit distributions above; do not inherit changing constructor defaults. Pin framework behavior and version before qualification.

Both accept variable-length streams. G has a bounded128-event per-target credit horizon and recurrent compressed state; T retains the last128 events with learned chronological positions0–127 and recomputes under current parameters. Overflow is counted and disclosed. Public BEGIN resets transient working context only. These are bounded context policies, not unlimited memory. For G, each permitted target differentiates through at most128 preceding transitions from a detached boundary state; all target gradients contribute to the record mean. Older target gradients are not discarded. Short frames can share one equivalent full graph; long frames use bounded per-target accumulation. T shifts to the last128 raw IDs, reindexes learned positions from0 and recomputes them under current parameters. Never retain stale learned embeddings/KV arrays across an update; short causal vectorization must agree with prefix-by-prefix decisions.

At test, feed prompt bytes only and then generate entirely from the model's own prior choices. The whole257-way distribution stays available. The model emits its own EOS. Retain the16-payload-byte cap plus a final EOS decision; reaching a cap is TRUNCATED, not chosen termination. Empty, malformed, extra-byte and truncated answers fail exact scoring. Generation changes neither parameters, optimizer, replay state nor learning RNG.

## 5. One end-to-end feedback objective and update policy

Use response-byte plus actual-EOS cross-entropy, mean within each observed record. Prompt bytes participate in the learned computation and gradients; their identities are not auxiliary arithmetic targets. The generic public REPLY determines which arriving observations carry response loss. There are no digit/EOS class weights, length supervision, numeric error gradients, parsed-answer rewards or automatic feedback for generated guesses.

All parameters stay fixed during the external frame. Predictions precede each arriving reply target. At actual permitted EOS, form the record gradient and take one joint update through input embedding, sequence model and output. Use framework automatic differentiation and float32 consistently for training and inference. No pretrained model is used.

Fixed optimizer: AdamW, learning rate0.001, betas(0.9,0.999), epsilon1e-8, decay0.0001 for all parameters. Clip global gradient norm to1. Then compute the complete proposed AdamW displacement, including adaptive preconditioning and decay. Bound its RMS displacement across trainable scalars to the nominal learning rate:

    d = proposed_theta - current_theta
    applied_d = d * min(1, learning_rate * sqrt(P) / ||d||_2)

P is the actual trainable parameter count. Compute the gradient, Adam age and moment proposal in the fixed optimizer order; produce float32 candidate parameter arrays, measure complete displacement with float64 sum-of-squares, and apply the single cap scalar back in float32. Record rounding tolerance in qualification. Optimizer age/moments advance from the permitted gradient even when displacement is capped, while an entirely unpermitted record changes none of them. Temporary proposal arrays and their memory traffic are charged. The displacement cap covers decay as well as the data update. This is **AdamW with an explicit generic displacement cap**, not vanilla AdamW or a proven cure. Log raw gradient norm, proposed/applied displacement, clipping, loss and component norms. Zero-rate/no-write behavior blocks decay and moment changes. The cap provides a finite update budget and is not a new success threshold.

Implementation references: [official AdamW specification](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW), [official GRU convention](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU). Standard architecture/optimizer choices remain hypotheses in this task; the [attention](https://arxiv.org/abs/1706.03762) and [episodic-replay](https://arxiv.org/abs/1902.10486) papers do not validate this MiniFly result in advance.

### Replay is real stored experience, not a representation helper

A64-record reservoir stores only raw prompt and actually permitted observed reply bytes, with a hard4,096-byte payload bound. Eligibility requires at most64 raw payload bytes per record, so64 records cannot exceed4,096 payload bytes; larger records are ineligible and logged. Metadata is additional paid storage. All benchmark records fit. Store no hidden states, numbers, ASTs, answers from the evaluator, generated guesses or unseen queries. Use standard reservoir sampling over permitted external observations; duplicate external presentations remain duplicate paid observations.

Before inserting the current record, sample at most one stored prior record. Re-encode it with the same pre-update parameters used for the current external record. Both losses use only earlier actual bytes with the same causal shift; ignore only absent padding and the generic public prompt segment, never an output class or EOS. Average the current-record and replay-record losses in **one joint optimizer step**; there is not a second optimizer update. When empty, use current record only. Then insert the newly observed raw record. Thus G0/G1 and T0/T1 have equal permitted optimizer-step counts, while replay consumes extra forward/backward computation and bounded storage. At inference the reservoir is never consulted for an answer: every byte/EOS comes from the same neural logits. This is not an equal-compute comparison; the common-compute endpoint pays the extra work.

No-write disables replay insertion, reservoir counts, sampling RNG, gradients, optimizer, decay and parameter updates. N histories cannot secretly store forbidden feedback for later replay. Existing permitted memory may be replayed only when a later external record itself allows learning. Probes and generated bytes are never insertion events.

## 6. Task data: four operations, then bounded composition

No counting warmup, arithmetic intermediate targets or manufactured equivalents are supplied by the model.

### Phase A: binary four-operation experience

Evaluator seed operands0–15 give136 unordered SUM families and136 PRODUCT families. Hold32 families of each kind from the105 families with both seeds≥2; teach104 families per kind. Close the split over exchange and signed inverse subtraction and exact integer division, omitting zero divisors. All literal/signed equivalent primitive relations in the held family remain absent from permitted binary and mixed feedback. Independent audits check input/output byte support.

This is a declared broader limited-precision data range than the prior0–7/0–9 trials. Its scores cannot be compared to those trials as a model-only effect. All new configurations use this same generator and split. Expanding the table is not asserted to force a compact rule.

External budgets per operation64,128,256,512,1024,2048:8,192 records at the endpoint. At every microblock take one record per operation in a fresh shuffled order. Each operation cycles its shuffled permitted pool, without replacement until that pool is exhausted. Each repeat is another external sample. No operation label is passed into the model. This removes the previous growing blocks of one operation.

### Phase B: mixed-expression experience and retention

Construct two-operation expressions by joining two **taught Phase-A primitive facts** at their shared intermediate value. Both primitives must be in the teaching pool, not merely have familiar digits. The joining process and intermediate value are evaluator-only. For example, the learner sees a complete parenthesized expression and its final observed answer; it is never shown the decomposition as a field or trace.

Use the16 ordered inner/root operator combinations equally. Group mixed expressions by a canonical evaluator-only tree that closes commutative swaps and same-operator associativity. All syntactic forms of a group stay together. For each operator combination select32 teaching groups and16 held groups:512 taught and256 never-taught mixed groups. Choose one fixed representative raw form per held mixed group using a separate seeded data draw before any model runs; the256 representative queries define the primary score. All other forms remain withheld and can be separately labeled descriptive probes, with separately charged costs. Taught-query diagnostics use only literal forms that actually appeared in the permitted W teaching prefix, not a hypothetical representative never shown. Average binary scores within each family and operation, and mixed scores equally across held groups within each operator combination, so groups with more renderings cannot dominate. Use fully parenthesized raw expressions in this version; precedence without parentheses is not claimed. All available commutative/associative forms of a held group are absent from reinforcement and replay. Targets are ordinary signed integer strings transformed by the world's byte permutation. The primitive teaching pool has nonnegative literal operands; this joined domain therefore has nonnegative literal leaves and usually nonnegative inner results, although final subtraction replies may be negative. It is not arbitrary signed mixed arithmetic. Rendered held mixed input/reply byte support must be present in the full mixed teaching pool, independently audited before learning.

Mixed external budgets64,128,256,512,1024,2048. Microblocks contain one example from each of the16 operator combinations in a shuffled order. Within a combination cycle its teaching groups uniformly, then sample an available rendered form. All repeats are paid. No rewriting or synthetic-example generation occurs in the learner or replay subsystem.

Because every mixed primitive is a previously taught relation, held mixed expressions test a new **composition of potentially learned pieces**, not arithmetic facts hidden in an auxiliary input. Whole held binary families remain absent even as mixed intermediates. Models do not receive the grouping or know whether a problem belongs to Phase A/B.

A design-only validator enumerates these pools, checks separation and support, and confirms sufficient groups for every operator combination on four reserved data-only seeds. This is not a learned positive control. Later admission reconstructs every scientific world's manifest before a model runs. A deterministic within-world data draw may retry only for declared coverage/pool feasibility, never for model scores; retry counts and chosen manifests are frozen before learning.

## 7. Four causal histories with shared prefixes

| History | Binary episode | Mixed episode | Question |
|---|---|---|---|
| W | Correct feedback, learning | Correct feedback, learning | Operative learner |
| N_BINARY | Same observed bytes, whole learning episode disabled | Correct feedback, learning | Is earlier binary learning reused? |
| SH_BINARY | Length-stratified shuffled binary answers, learning | Correct feedback, learning | Does correct earlier relation structure matter? |
| N_MIXED | Shares W binary endpoint | Same mixed feedback, learning disabled | What is changed by later learning? |

SH_BINARY shuffles replies within operation and answer-length strata in fixed budget increments, preserving EOS/length and reply marginals. Report equal-label matches; do not choose a derangement by inspecting model behavior. Reservoirs contain exactly the replies actually observed and permitted in their own histories.

Execute three binary prefixes, then clone W for W and N_MIXED continuations. Execute four mixed continuations. A full world/configuration job therefore uses:

    3 × 8192 + 4 × 2048 = 32768 external record applications

W itself has10,240 external observations. Prefixes/forks are correlated, not independent lives. There are22,528 permitted original-record optimizer updates per full job. A replay configuration performs22,525 additional stored-record forward/backward applications at most; three initially empty reservoirs have no prior record. Replays add no external information and no extra optimizer steps.

At every binary and mixed checkpoint, read-only probes score both held binary and held mixed problems, taught problems, termination and first-error diagnostics. After the mixed endpoint probe **old taught and old held binary problems**, newly taught and held mixed problems, then repeat after86,400 simulated seconds. Idle advance is a disclosed no-op; persistence pressure comes from actual later learning, not a biological forgetting clock.

## 8. Exact ability, samples and compute

Let A(n) be half the equal-operation exact held binary score and half the equal-operator-combination exact held mixed score after n external W-history observations. Own payload and own EOS must both match. Probe birth as well as all fixed checkpoints. No early stopping or choice of best checkpoint is allowed from a held curve.

J is a lifecycle ability/sample area with separate final retention gates; it is not an area of independently interference-tested scores at every earlier checkpoint. Primary sample efficiency J is the normalized left-continuous staircase area of A(n) over0–10,240 external observations. This charges all prior binary learning before mixed performance becomes available. Publish the entire curve and separate task scores; high taught recall or numeric proximity does not enter J.

For each candidate-W versus G0-W pair, G is the corresponding area over their common actual production-compute horizon, without interpolation or extrapolation. Availability costs include original sensory processing, teacher-forced forward/backward, optimizer, reservoir sampling/re-encoding, and the mandatory scored own-output query batch at that checkpoint. Count the shared W binary prefix once, its W mixed continuation and mandatory W readout costs. Never divide whole-job CPU by histories: N/SH controls and peer jobs are separate execution costs. Mark query readout separately so training-only efficiency is also visible. Exclude repeated validation, digesting, compression and orchestration overhead; report them separately. Report FLOPs, logical memory traffic, peak state, per-query latency, target-byte counts, original repetitions, unique feedback and replay counts. No idle padding is used to pretend replay costs equal computation.

Final retained binary scores F_o are reported for all four operations after the mixed episode/rest. Mixed J_MIX is an ability summary: the equal-budget mean across its six fixed checkpoints and16 operator combinations. It is not a sample-efficiency integral. Publish its exposure coordinates8,192+m; the joint J and G charge the full prior prefix. Taught mixed acquisition W−N_MIXED is a diagnostic against apparent retention obtained by refusing to learn. If new mixed learning is absent, report that explicitly; preservation alone earns no claim of adaptive persistence. A genuinely perfect zero-shot composition result is reported as already acquired capability, not penalized for lacking further improvement on solved queries.

Countermodels use the respective history's actually permitted original and replayed feedback; W is compared against its own W-envelope. Each mode table uses raw permuted prompt identities and raw reply bytes. Ties choose the lexicographically smallest raw reply; unseen suffix/length keys fall back to the global reply mode; an unseen byte-bag falls back to the length mode then global; empty teaching state abstains. These rules never parse numbers. Exact trained-key lookup cannot answer a held group; the canonical-group lookup is a disclosed evaluator-only E3 countermodel whose group key is never passed to a model or frequency control. Countermodels are fixed before learning: exact and canonicalized trained-key tables, global reply mode, length mode, last-two-byte mode and byte-bag mode with feedback-only fallbacks. None has access to held targets or the evaluator's primitive composition. The envelope is evaluated per checkpoint; it is an evaluator diagnostic, never a gate or feature inside the core. Retained comparisons use only feedback that the respective history was allowed to learn; no evaluator family key or learned-rule solver is added to shortcut controls.

## 9. Selection, confirmation and bounded claims

DEV: four fresh paired worlds202610133001–202610133004, four configurations,16 jobs. Only G1,T0,T1 are candidate recipes; G0 is the reference. At most one recipe advances. Require positive mean J and G differences against G0, positive retained binary causal/shortcut contrasts in every operation, and positive mixed causal/shortcut contrasts. These are a screen, not significance. Rank eligible candidates by J gain, then G gain, then lower production cost, then fixed order G1,T0,T1. No gate or hyperparameter changes after DEV.

CONFIRM:24 fresh worlds202610134001–202610134024, with G0, selected candidate and one fixed diagnostic peer. Mapping: selectedG1→peerT1; selectedT0→peerT1; selectedT1→peerT0. No late promotion of the peer. Maximum72 confirmation jobs; zero if DEV has no eligible candidate.

For the single selected recipe, the17 primary one-sided world-mean comparisons are:

- For each of four final retained binary operations, W exceeds N_BINARY, SH_BINARY and its shortcut envelope:12 tests.
- Mixed J_MIX W exceeds N_BINARY, SH_BINARY and its shortcut envelope:3 tests.
- Joint J and common-compute G exceed G0:2 tests.

Apply Holm correction at0.05 to this declared17-test family. World, not problem, checkpoint, replay or history, is the independent unit. Use paired world-mean inference, report two-sided95% intervals; constant-contrast handling must be independently qualified before locking and may not claim zero uncertainty merely from zero sample variance. If any required contrast fails, no confirmed efficiency adoption. For nonconstant paired differences, use the preregistered one-sided paired Student-t world-mean test with its reported uncertainty. For a constant observed contrast with mean d on the declared[-1,1] range, use p=min(1,exp(-n*d*d/2)) if d>0, otherwise p=1; its95% interval is d±sqrt(2*log(40)/n), clipped to[-1,1]. This is a conservative bounded-mean Hoeffding treatment, not an artificial zero-variance t result. No postrun choice between inferential methods is allowed.

This is a comparative bounded gain, not mastery of arithmetic or unrestricted mixed calculation. Publish individual curves and limitations even on a negative verdict. The diagnostic peer and approximate errors cannot rescue failed primary gates.

Maximum88 full jobs and2,883,584 executed external record applications, conditional on advancement. Maximum additional internal replay applications1,261,400 under the fixed peer mapping. These are program execution counts; W's sample-efficiency denominator remains10,240, not all counterfactual histories combined.

## 10. Qualification and execution boundary

The next authorized preparation, if requested, performs three complete design–audit–technical-test–revision cycles within this design: generic byte/gradient/optimizer checks; complete-life causal/resource checks; then all four recipes plus final reducer/auditor/package checks. No formal result can be claimed from these technical worlds.

Required checks include: all256 byte identities; random relabeling; EOS/truncation; no arithmetic imports or privileged fields in model code; current-prefix versus future-byte causal masking; generic position/padding behavior; complete joint gradients; standard GRU gate convention; AdamW moments and **full proposed displacement including decay**; whole-phase no-write including reservoir RNG; pure queries/generated output; independently owned mutable forks; exact recorded exposure and primitive/mixed group absence; correct replies never leaking into shuffled/prohibited reservoirs; historical replay target provenance; batch-step and sample counts; source lock and tamper rejection. Technical target bytes and fixtures never count as science.

Measure a complete job including all histories, finalization and new query costs before promising runtime. Use two detached workers, one numerical thread each, supervisor-linked caffeinate and a read-only monitor if the experiment is later authorized. A conservative3× measured slowest full-job/finalization estimate must fit the declared eight-hour admission window. Whole-job boundaries and a ten-hour hard session cap apply. If projection does not fit, present an explicit session schedule before launch; do not silently reduce n, change budgets or auto-resume. No runtime estimate from the old NumPy core is treated as a measurement of these new kernels.

All model/optimizer parameters above are generic fixed starting choices, not known-optimal arithmetic hyperparameters. Qualification may repair implementation or clarify a contradiction with a preserved audit trail. Behavioral redesign after a negative technical result must be explicitly versioned; no patching under the same frozen scientific lock.
