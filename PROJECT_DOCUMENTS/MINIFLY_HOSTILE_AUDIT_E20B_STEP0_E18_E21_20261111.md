# Hostile audit of E20-B Stage 1, Target Test Step 0 v2, E18b/E18c/E23, and E21 resume

11 November 2026

This note records the read-only audit at consolidated revision `376092e`. The merged harness and E24 brief were pending at that snapshot. The E20-B final results and the joint audit of those results and probes 9.93/9.95 await the completed record. [Project state](https://github.com/ChangweiZhou/digitalbaby/blob/376092e6ed9204c14343eca17cee89d057c4429d/PROGRAM_STATE_20261107.md)

## E21 harness merge and E24 dispatch

**Rerun all 47 checks on the actual merged revision before enabling resume.** The worker’s 47 passes and the coordinator’s smoke tests cover different revisions. The pre-merge worker code also leaves three substantive gaps:

- Admission omits source hashes, Torch version and dtype, the supported-command list, and the qualification-receipt gate.
- A torn final birth receipt can be parsed before the valid checkpoint is loaded. Exercise a kill during final publication and verify recovery from the checkpoint.
- E23 shuffled targets consume a private Torch generator. Rebuilding input batches and restoring global Torch RNG leaves that generator out of step after a direct merge. Keep that shape outside resume support until the private state is restored and parity-qualified.

Use a fresh test directory and bind the receipt to the exact merged sources. Include wrong-source, wrong-birth, wrong-dose, mixed-stage, and final-publication fixtures. [E21 amendments](https://github.com/ChangweiZhou/digitalbaby/blob/376092e6ed9204c14343eca17cee89d057c4429d/OPUS_TASK_E21_RESUMABLE_LIVES_20261110.md#L46-L57) · [Worker admission and finalization](https://github.com/ChangweiZhou/digitalbaby/blob/8b3ab4e7ad57a60aea81353aba1a0de5db2d5b1e/MOD97_LOCAL_RULE_20261106/exp16_integrated.py#L969-L1043) · [E23 target shuffle](https://github.com/ChangweiZhou/digitalbaby/blob/ed07bf61/MOD97_LOCAL_RULE_20261106/exp16_integrated.py#L342-L371)

**E24 awaits its published brief.** The revised forecast is nine new births plus two host-check reruns, about 4½ hours. The informative comparison includes answer-only and every-byte backprop alongside the matched sign-feedback conditions at the same dose. Answer weighting can establish the effect of weighting on the tested configuration. The comparison can identify an interaction between objective and update rule under the tested conditions. [Review 9.97](https://github.com/ChangweiZhou/digitalbaby/blob/376092e6ed9204c14343eca17cee89d057c4429d/MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md)

## E20-B Stage 1 and dispatch

**Accept the recorded technical qualification and file the premature start as a separate deviation.** The G0 birth curve and all 16 common score blocks across seven life checkpoints match in the committed receipts. Losses, exposure, panel hash, starting-weight hash, and 1,536 optimizer steps agree. The worker’s check records equal tensors and passing architecture, scorer and admission fixtures. Stage 2 started before the required review and release.

Record the three interrupted G0 lives and their later authorized reruns from the same births. Include the interrupted work in the cost ledger. The resumable driver uses receipt completion fields to skip work and logs child failures without propagating them. Before interpreting outcomes, verify all 24 completion receipts and weight hashes independently of the driver’s terminal line. G1 had exited and G2 had started at the inspected worker revision. [Stage 1 checks](https://github.com/ChangweiZhou/digitalbaby/blob/fdc450d862aadef592b0154d56daaeebb68134fc/MOD97_LOCAL_RULE_20261106/exp45/e20b_qual/E20B_CHECK.json) · [Interrupted attempts](https://github.com/ChangweiZhou/digitalbaby/blob/ddc9460a8ed30c2606cac91dfa747c835ee0939d/MOD97_LOCAL_RULE_20261106/exp45/e20b/interrupted_20261111/README.md) · [Stage 2 driver](https://github.com/ChangweiZhou/digitalbaby/blob/ddc9460a8ed30c2606cac91dfa747c835ee0939d/MOD97_LOCAL_RULE_20261106/exp45/e20b/run_stage2_resumable.sh)

## Target Test Step 0 v2

**Correct the acceptance rules before the generator-only handoff.** The sealed pool, independent family values, and state restoration address the earlier audit.

1. Shuffling intact `X:vv c:uu` lines preserves true mappings within a stable phase. Use it as an alignment control and the matched ordinary-byte stream as the extra-observation control. Audit disclosures available before each probe.
2. Identical suffixes imply zero both-correct twins for a deterministic suffix responder. Single answers can match by collision. Cue recency may reflect a post-change value or a stale answer from a different context. Measure exact current and obsolete-answer rates. Use exact rendered prompts in the positive long-window lookup fixture.
3. Check identifiability from the actual prefix at each probe, including first-encounter reveals and wider-view disclosures. Track old and current revision values separately. Use a ground-truth oracle and a separate prefix-only solver. For causal comparisons match the prior-teaching control on timing and exposure.
4. Report the actual number of scored semantic twins separately from filler renderings. The draft simulation dropped an equal-outcome twin in one world.

[Step 0 v2](https://github.com/ChangweiZhou/digitalbaby/blob/376092e6ed9204c14343eca17cee89d057c4429d/OPUS_TASK_TARGET_TEST_STEP0_20261111.md#L18-L47) · [Simulation receipt](https://github.com/ChangweiZhou/digitalbaby/blob/376092e6ed9204c14343eca17cee89d057c4429d/PROJECT_DOCUMENTS/TARGET_STEP0_DRAFT_CHECKS.json)

## E18b, E18c and E23

**Close the line as planned, with corrections to the report.** Read-only recomputation of 27 birth receipts and six E18c life receipts reproduced all 14 registered verdicts.

- E18b shows that every-byte targets damaged the tested sign-feedback configuration. Embedding movement establishes update reach. The mechanism behind the behavioral collapse remains open.
- E18c’s relay arm gained 12.5, 13.6 and 14.8 percentage points on final F1 over the cut arm. This is a birth-plus-life comparison. The cross-host E22 comparison is descriptive and leaves the added value of local heads unresolved.
- E23’s base arm collapsed, leaving the timescale question unanswered. Its structural comparator is fitted on the scored panel with access to the true target class, which limits what the head’s miss establishes.
- The E18c composition table entries are percentages of correct final suffixes. The hard-pair whole-answer counts out of 1,200 were 0/2/1 for cut and 3/5/1 for relay across the three seeds. Report both endpoints with their units.

[Report](https://github.com/ChangweiZhou/digitalbaby/blob/ed07bf61/MOD97_LOCAL_RULE_20261106/E18B_E18C_E23_REPORT_20261111.md) · [Scorer](https://github.com/ChangweiZhou/digitalbaby/blob/ed07bf61/MOD97_LOCAL_RULE_20261106/e18b_e23_score.py#L79-L82)
