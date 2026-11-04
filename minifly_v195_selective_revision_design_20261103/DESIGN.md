# MiniFly V195 · Selective revision in a long-life fact memory

**Status** DESIGN_ONLY. **Date** November 3, 2026, New York time. **Input** MINIFLY_MEMO_ROUND6_20261104.md, supplied by the owner. **Evidence anchors** Consolidated 7fb2ff17c347396a6030f6a6a69e174432b35d09 and parallel ca74be1fd03158fd78e7ff0569e1f5c21b42498f. V195 is the requested identifier. This document is a local proposal and does not allocate or relabel any intervening version.

## Decision and evidence

Can a protected, shared fact memory suppress obsolete answers after an observed correction without sacrificing recall of other facts? Test whether concentrating the same write strength on rows that support the former answer helps, and whether explicitly depressing that answer contributes beyond reopening alone.

The memo reports two synthetic lives per arm. At 65,536 rows with importance-scaled writes, all-current fact accuracy was 94.5% and 94.3%, latest-correction accuracy 100% and 96.9%, and obsolete answers 4.4% and 4.9%. Broad contradiction-triggered release lowered all-current accuracy to 70.6% and 67.8%. These are memo-reported first-byte results with a supplied query cue. The source files named in the memo were not found in the checked work folders. Recover and pin them before claiming source equivalence. The present document specifies a bounded successor experiment. An implementation reconstructed from the memo must identify itself as such and qualify its own reference.

The parallel float16 result applies to a different fixed-lag store. It establishes a useful engineering candidate, not numerical parity for importance-scaled writes. V193 studies relational readout reuse and does not validate a fact-memory intervention. V192 concerns successor-map routing and has no scientific outcome. Keep those decisions separate.

- [Round 6 memo](/Users/pencilbard/Downloads/MINIFLY_MEMO_ROUND6_20261104.md)
- [Parallel storage results](https://github.com/ChangweiZhou/digitalbaby/blob/06c6077591fa8538fd8b23629eade36eb4e76e00/minifly_next_phase/STORAGE_RESULTS.json)
- [Parallel V193 publication](https://github.com/ChangweiZhou/digitalbaby/commit/ca74be1fd03158fd78e7ff0569e1f5c21b42498f)

## Fixed interface and substrate

Use one all-byte lag-hash predictor and one 65,536-row bag-of-n-grams fact store. The latter uses a 16-byte causal suffix, bigrams and trigrams, and four fixed addresses per n-gram. Use the pinned source address and local update conventions if recovered. Otherwise freeze a reconstruction with unique active rows and feature weight 1/sqrt(active rows). Count fixed hashes, collisions, projection buffers and every mutable array. An environmental query-start signal permits the fact memory to read. It provides no key ID, correction flag, expected value, response length or stop signal. Nonquery bytes use the predictor. Teaching arrives after each prediction, and only marked answer events write fact weights. Train genuine EOS at the end of taught fact responses in the same 257-way head. On a query, feed the model's own bytes back through the raw suffix and stop at its own EOS or an eight-output cap.

Use the same source-matched importance accumulator and protected base write in every arm. The proposed reconstruction uses nonnegative per-row importance I, incremented by absolute active feature weight on each eligible observed fact write, with a multiplier 1/(1+I/5). First predict, then receive the target, then write. Log the corresponding memo byte-only score separately from the stricter own-byte/EOS outcome. Keep predictor state, teaching and query schedules, legitimate boundary signals and any inherited warmup the same across arms. No consolidation, sleep replay, attention, backpropagation, ELM or arithmetic features.

Treat row specificity as a **testable proxy**. At the marked payload site, record the full readout's old winner before feedback. A revision trigger is a non-EOS winner that differs from the observed target, with a lead over the runner-up and at least two prior marked activations on eligible rows. Choose one margin threshold using only separate engineering worlds and lock it. An eligible active row must itself favor the old winner. Among eligible rows, select the lowest prior marked-activation-count quartile, breaking ties by row ID. Log empty sets and do the base write on those events. This bounded selector uses only local row statistics and raw-context activations. It may select rare wrapper rows or mistake a confident collision for a correction. The evaluator measures these cases with its hidden key and revision labels, which never enter the model. No exact key table or explicit owner map is part of the candidate.

## Mechanism and controls

Let U0 be the protected source update at an event, computed from the current arm's pre-update state. In the proposed reconstruction it is the 0.6 NLMS step using the 257-way one-hot error, weighted by the row importance multiplier. All arms update the same importance and prior-activation counters. On triggered payload events, construct the following proposed directions. On all other events use U0.

| Arm | Proposed direction |
|---|---|
| BASE | Ordinary importance-scaled write |
| REOPEN | Temporarily restore the unprotected row multiplier on selected rows for this write. Preserve the accumulated importance afterward. |
| DECAY | On selected rows add a component that decreases the old winner and increases the observed target by the same amount. Before normalization its magnitude equals that row's base target-column movement in absolute value. |
| BOTH | Combine transient reopening and the decay component. Compute the decay magnitude from U0, before reopening. |
| SHAM | BOTH with the selected row indices permuted to a same-size subset of active rows, with a fixed training-only seeded schedule. Keep the trigger and old/target columns the same. |

Normalize each proposed active-row update to the Frobenius norm of U0 computed at **that same arm and state**. If U0 is zero or the candidate direction degenerates, use U0 and record why. This removes a simple write-strength explanation at a common state. Divergent histories can still have different cumulative motion. Report proposed and actually stored row and class movement, triggered and selected counts, and training targets per arm. Before scientific lock, qualify one persistent dtype across all arms. The planned 48 MiB deployed envelope assumes float16 persistent fact weights with float32 update scratch. Measure actual realized write norms after casting, especially saturation and zero movement. Predeclare a numerical tolerance using engineering fixtures. If quantization defeats same-state dose matching or baseline recall, stop admission. Do not import the other lineage's float16 result as evidence of parity.

## Long-life roster and readout

Use four fresh paired synthetic worlds from the first 64 bits of SHA-256 of MINIFLY|V195|WORLD|i for i = 0 to 3. Use separate deterministic streams for keys, values, order, wrappers, filler and corrections. Randomly assign independent one-byte values from a fixed balanced alphabet. The external scorer checks all 256 payload values and native EOS, with no candidate vocabulary mask. Use one chronological life per arm. There is no state reset between waves.

- Eight waves of 128 new facts. Present each new fact four times, twice in each of two teaching wrappers. After every wave from wave one through seven, correct 32 prespecified keys from the previous wave to a different independently drawn value. Teach each correction four times, twice per wrapper. This yields 1,024 distinct facts and 224 single-key revision events over seven rounds. This design does not measure repeated re-correction of the same key.
- Insert one ordinary prose record per four teaching records, including unmarked mentions of known keys. Freeze filler, cue positions and boundary tokens before training. Use the same stream and observation budget across arms. Do not include a second relational learner in this fact-writer comparison.
- Probe at the end of each wave before and after that wave's correction, and at the end of life. Preserve separate panels for untouched facts, all-current facts, wave zero, each correction cohort, and the latest correction. Test two wholly held wrapper families and the trained wrappers. Audit exact-prefix leakage, active-code collisions, and cue/value balance. Probe through read-only copies before feedback, with matched query cues. Preserve original and revised labels so the scorer distinguishes an obsolete response from other errors.
- Use a trained-prefix table, a canonical-key table updated at the same feedback events, and frequency and wrapper-only baselines. The canonical table can solve the taught-key task by construction. A positive result supports bounded cued memory retention and correction and changed-presentation recall, at most E2 under the objective contract. It supplies no evidence for question recognition, arithmetic or E3.

## Decision rule

Primary comparison is BOTH versus BASE on final held-wrapper **obsolete complete answers** among the 224 corrected facts. A complete answer is exactly one emitted payload byte followed by own EOS. Report obsolete first bytes, extra payload, premature EOS and truncation separately, so failure to stop cannot masquerade as removal of the old answer. The preservation panel comprises untouched facts and the oldest untouched wave. Report all-current, latest-correction, and per-round results as separate panels. Report predictor-only prequential payload bits on the identical observed filler and held-filler prediction separately from cued retrieval.

A package can advance to a fresh-world review only if BOTH meets all of the following on each fresh world and both held wrapper families:

- At most 1% obsolete **first-byte** answers among corrected facts, and fewer obsolete complete answers than BASE on the pooled paired-world mean. Report raw counts, including how many worlds improved.
- All 32 latest corrections produce the new byte and own EOS. At the end of life, exact accuracy across all correction cohorts is at least BASE's.
- All-current exact recall is at least 94%, the memo's approximate benchmark carried over as a proposed stricter complete-output target. Exact untouched and oldest-untouched recall do not fall below BASE on the same held prompts.
- The improvement over BASE also exceeds SHAM on the paired mean. Report the two single-mechanism contrasts and the factorial interaction. If a single intervention achieves the same guard vector and ties or exceeds BOTH on obsolete answers, favor the simpler arm for prospective review. Do not choose doses or thresholds from the test roster.

Report paired-world scores and signed differences. Four worlds give development evidence. Treat items within a life as correlated. If BASE loses the ability to produce complete answers on trained cues, stop at source/endpoint qualification before interpreting the candidate. If the effect disappears against SHAM, the specific row-selection claim has failed. If a candidate suppresses old winners but increases nonanswers or impairs retention, reject the package. Further consolidation and real-prose tests remain separate.

## Resource admission and three-cycle plan

Per world and arm: 4,096 new-fact presentations + 896 correction presentations + 1,248 filler records = **6,240 training records**. Four worlds × five arms = **124,800 physical presentations**. Maximum 64 payload bytes plus EOS per record gives **8,112,000 byte/EOS sites**. Count actual distinct records, presentations, marked answer writes, EOS writes, update arithmetic, sparse read traffic, checkpoint copies, and probes separately. No global table scan or whole-state copy per event. Derive an exact manifest census before launch.

Proposed ceilings are **135,000 records and 8.8 million training sites including technical**, 7,200 CPU seconds, 10,800 active wall seconds, 1 GiB peak process RSS, 48 MiB per deployable learner and complete checkpoint, and 384 MiB retained output. A 65,536 × 257 float16 fact matrix is 32.125 MiB before predictor, indices, counts and scratch. Use one world/arm worker at a time. Retain compact checkpoint metrics and state hashes at all read-only probes and two designated full engineering/reproduction states. Science states are not resume points unless a further checkpoint budget is approved before admission. Count any owner archive and every retained copy. This is a larger storage lineage than V192, with an explicit physical envelope.

1. **Refine → audit → test the clean reference.** Resolve source and dtype. Use only a bounded clean fixture to qualify native EOS, known-cue recall, and hand-computed updates. Do not invoke a complete historical lifetime to read an early checkpoint. Record exact technical spend first.
2. **Refine → audit → test the causal intervention.** Check trigger ordering, raw-context-only feature selection, unchanged accumulated importance, post-cast write dose, same-size sham coverage and realistic collision canaries. Test two facts with overlapping addresses and a true correction. A canary is an engineering check.
3. **Refine → audit → test production execution.** Rehearse the actual launch path on a reduced synthetic life through matched arms, native generation, finite serialization, receipt verification and archive readback. Freeze the source and full roster before launch. Profile the actual longest records and all output paths. Include prior technical costs in the admission ledger before deciding whether the full comparison fits.

**What ran for this document** Memo and source-status review and design only. No implementation, qualification, scientific execution, independent certification or repository publication.
