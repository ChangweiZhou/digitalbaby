# MiniFly · V190 · Cued integrated reliability

**Status** DESIGN_ONLY. **Date** November 3, 2026, New York time. **Inputs** MINIFLY_MEMO_ROUND4_20261104.md and the consolidated branch at d9845798955838784532b9daf9b2da393068e897. This is an integration experiment in the memo's hashed byte-memory and successor-map lineages. Local numbering: V187 completed, V188 allocated with an incomplete verdict, and the published unnumbered familiarity trial occupies V189 in the local sequence. V190 is this proposal. Remote registry unchanged.

## Decision

Does first-encounter reliability keep both relation readouts useful in a continuing learner that also sees prose, learns facts and revises obsolete answers? Give every arm the same environmental query cue so the comparison isolates routing among experts from recognizing a new question syntax. Compare first-encounter weighting with equal weighting and repeat-accuracy weighting on **the same trained experts**.

The memo reports 97.2% comparison and 99.3% parity with first-encounter weighting on isolated synthetic streams. Its table scores the answer byte. The prior published familiarity experiment, using a different implementation, had 0/256 complete answers with the gate and 47/256 with memory alone. A query cue can select the memory. Whether that memory emits the correct byte and then EOS still needs measurement. Treat the memo numbers as reported development evidence and pin the underlying scripts and manifests before claiming reproduction.

## Interface and learner

Each record presents observed raw bytes and genuine boundaries. An environmental query-start event carries an opaque route tag for fact or relation queries. The tag contains no answer, stopping length, held status or correctness label. It is available to all arms and persists during generation. Teaching markers and targets arrive after each prediction. An ordinary prose record can mention a familiar key and carries no query cue. During generation the selected expert or mixture feeds back its own bytes and independently emits EOS or hits an eight-output cap. The evaluator never switches to EOS on its behalf.

The existing all-byte lag-hash predictor learns on the entire observed stream. Marked fact answers train the sparse bag-of-n-gram memory. Marked relation answers train both relation experts on their own local errors. Train genuine EOS in the relevant expert as well. Use the memo's distinct fixed sparse populations fed by TD(0) successor maps with discounts +0.8 and −0.8. Learn their maps from observed background transitions and freeze them for relation teaching in this design. This freeze isolates the mixer and must be disclosed if it differs from the memo's implementation. Audit the exact sensory extraction, update signs, step sizes, projection, cue handling, and serialization before locking. Disclose any explicit operand fields in the source interface. A symbolic harness cannot stand in for an operative byte learner. Preserve any inherited warmup and charge it to the exposure ledger. No extra warmup, generated-target training, replay, attention, backpropagation, ELM or arithmetic rules.

At each marked relation payload site, record both experts' greedy predictions before any update. After the target arrives, update each eligible expert's correct and total counts. Start with one pseudocorrect and one pseudoincorrect vote per expert and supplied tag. At inference, normalize the fourth power of each smoothed accuracy and mix the experts' normalized 257-class distributions. EOS contributes expert learning but no reliability vote. Only completed observations can affect a future mixture.

| Policy | Relation read | Purpose |
|---|---|---|
| PREQ | One reliability vote on the first encounter of each training prompt | Candidate |
| REPEAT | One vote on that prompt's second encounter | Compare reliability from repeated examples at equal endpoint vote count |
| UNIFORM | Equal weights on the same two experts | Test whether weighting helps over adding two experts |
| Single population | Each expert on its own | Identify available competence and EOS failures |

Use a bounded exact-byte prompt registry with collision-checked fingerprints and the provided tag. Train each relation pair with one rendering and two presentations. A generic registry may count prior identical prompts. The learner may not see evaluator pair IDs or a first-encounter flag. REPEAT votes later than PREQ, so the contrast tests the vote schedules at matched endpoint counts. Report timing and vote coverage. All read policies share expert weights, experience, updates and initial state. Their predictions do not control training. Prose-only equality across them is an isolation check by construction.

## Locked stream

Use three independent development worlds, seeded from the first 64 bits of SHA-256 of MINIFLY|V190|CUED-INTEGRATION|WORLD|i for i = 0, 1, 2. Freeze each complete roster and use independent salted draws for symbols, values, order, wrappers, corrections and filler.

- **Background.** Eight observed laps over 16 randomly relabeled cyclic symbols. Each 17-byte record includes the wrap edge. The candidate receives bytes and boundaries, without coordinates, cyclic distances, parity or expected relation outputs.
- **A.** Teach 32 facts and the cyclic comparison relation. Each fact is a random key with an independent one-byte value. Each occurs twice across two teaching wrappers. Comparison distinguishes distances 1–7 from 9–15. Omit 0 and 8 from both relations to match domains.
- **B.** Teach 96 new facts and the alternating parity relation. Continue from the actual A state. Interleave one nonquery prose record per four teaching records in each phase. Include known-key mentions. Independently relabel the two relation tags and their binary answer byte alphabets each world.
- **Held relations.** Exclude distances {3, 6, 10, 13} under both tags. The set is closed under reversal and balanced for both relations. Each tag has 160 taught ordered pairs and 64 never-taught pairs. Teach each training pair twice with the second appearance strictly later. Audit no held pair, its reverse or a re-rendering receives answer feedback. Measure each relation after its acquisition, both after B, and both after correction.
- **Correction.** Fork EDIT and NOEDIT from one serialized B snapshot. Select eight A keys before training. EDIT gives each a different independently assigned value twice. NOEDIT gives the old value twice, with the same prompts, order, cues and filler. The remaining A and B facts stay valid. Include no phase label in the learner. Evaluate original and held fact wrappers, obsolete answers and untouched A/B facts. Retain the complete state at birth, A, B, EDIT and NOEDIT.

Withhold entire fact wrapper families from both training and filler. Audit exact prompt and suffix overlap and include matched unmarked prose mentions. Use one query record per four prose records in the mixed held stream. Balance fact and relation queries where both apply. Query probes receive no feedback and do not mutate the continuing state. Score prediction bits per observed payload byte and EOS separately from autonomous generation.

## Decision and controls

Primary endpoint is exact own answer byte **and own EOS** on each held distance panel. Report byte accuracy, EOS, premature stops, extra bytes and truncation separately. Require PREQ at or above 95% complete accuracy on each relation in each world after B and after EDIT. This uses the memo's joint target with complete-output scoring. Require PREQ to match or beat UNIFORM and REPEAT on each relation after B, and beat each strictly on at least one relation in at least two worlds. The world is the paired unit. Forks and individual pairs do not create extra independent worlds.

For a positive integration result, require all three guards:

1. The cued fact memory reaches at least 80% complete recall on each held wrapper after B, with taught-wrapper and byte-only figures shown separately.
2. EDIT produces more corrected complete answers than NOEDIT, while preserving the count of still-valid facts recalled at B on the same prompts. Report obsolete responses and both relation panels in both children.
3. Ordinary prose predictions match across policies, and PREQ does not raise mixed-stream payload loss versus UNIFORM. Report the fact and relation EOS losses separately.

Use train-only frequency and operand marginals, exact and reverse-pair lookup, and a locked nearest-neighbour rule for relations. For facts use trained-prefix lookup and canonical-key lookup with the supplied query cue. An evaluator-side rule reader of the observed cycle checks assay determinacy. It cannot supply candidate outputs. Evaluate born relation readouts with the same learned maps but no relation teaching, and verify the cue is independent of correct outputs within each tag. The routing contrast does not isolate the causal effect of successor structure. That would require an additional matched prior-exposure control. Taught fact results can support changed-presentation recall (E2 ceiling). Held relation results can support development transfer on this distance family. They cannot certify E3, arithmetic or adoption.

If the cued memory still fails complete generation, record the integration failure at the fact readout rather than blaming automatic timing. If the best single expert and the mixtures fail the relation target, distinguish byte accuracy from EOS before attributing the failure to the mixer. If all mixers tie above target, the additional reliability state has no demonstrated benefit. If PREQ passes, retain the exact cue-assisted package for fresh-world review.

Keep count-on addition separate. For that later experiment, require a Latin-square or marginal-completion baseline, a structure-shuffled successor control, a matched no-scratchpad arm and holdouts with audited multiplicity of completions. Sparse missing cells may be recoverable from marginals, so high accuracy on those cells cannot identify a learned arithmetic operation.

## Resource and launch gates

Per world: 8 background + 256 fact + 640 relation + 32 correction + 232 prose = **1,168 training presentations**. Across three worlds, **3,504 presentations**, including both correlated correction branches. At 127 payload bytes plus EOS per record the ceiling is **448,512 target sites**. Count actual map transitions, expert writes, reliability votes, probes and both branches. A single observed target can update several experts, so report per-module operations separately. No duplicated expert history is needed for the read policies.

Use proposed caps of 5,000 training presentations and 600,000 target sites including technical and warmup, 900 CPU seconds, 1,800 active wall seconds, 1 GiB process RSS, 8 MiB deployed state and full checkpoint, and 256 MiB total outputs. Predictor and memory arrays at 2,048 and 4,096 rows by 257 float32 columns occupy about 6.03 MiB before bias. Four relation banks at 256 by 257 float32 columns add about 1.00 MiB before maps, scratch, counters and cue state. This is a planning envelope. Source audit and actual full-roster profiling must establish whether the original bank sizes and all state fit. Do not silently shrink the source to pass.

Before training, lock exact source and data, qualify cue timing, first/repeat ledgers, held-distance and reversal census, finite output probabilities, authentic EOS and checkpoint restoration. Rehearse the actual entry point through both correction forks, reduction, JSON serialization, receipt verification and archive readback. Bind the rehearsal to source hashes and measure full-roster resources. A failed data, source or resource check stops admission. If requested, run three successive refinement → audit → test cycles, with the scientific test run only after admission.

**Work performed** Read-only memo and repository review, then design. Implementation, data lock, technical execution and scientific results are pending.
