# V154 — three-cycle native bridge diagnosis

**Status: bounded exercise complete; independent audit accepted. No established improvement in unseen addition.** The user assigned V154 because Claude is working on V153. The provisional design was preserved, the execution used separate folders and the fresh V154 RNG/world namespace. No core is adopted.

The objective remains retained complete own-byte/EOS unseen-addition generalization per total feedback and actual compute. Candidate learning stayed in the native Full151/R_center path with gradual warmup and local signed writes. External linear fits were evaluator instruments; they never generated native answers or supplied native updates. No arithmetic feature/rule, attention, Transformer, ELM candidate, forced EOS or extra task was introduced.

## Native acquisition and retention

All counts below are complete payload plus the model's own EOS. Each row is one data world, with W/SH_OLD/N_OLD as correlated histories. OLD-end and NEW-end scores have 64 questions each; final OLD is after NEW and idle.

| Configuration | World suffix | OLD end | NEW end | Final W OLD | Immediate exact /1,536 | Held admitted |
|---|---:|---:|---:|---:|---:|---|
| N_2B_I_FAST | 5401 | 0/64 | 0/64 | 1/64 | 3 | no |
| N_2B_I_FAST | 5402 | 0/64 | 0/64 | 0/64 | 4 | no |
| N_2B_I_SLOW | 5401 | 0/64 | 0/64 | 0/64 | 2 | no |
| N_2B_I_SLOW | 5402 | 0/64 | 0/64 | 0/64 | 1 | no |
| N_2B_P_FAST | 5401 | 6/64 | 5/64 | 4/64 | 126 | yes |
| N_2B_P_FAST | 5402 | 4/64 | 1/64 | 1/64 | 98 | no |
| N_2B_P_SLOW | 5401 | 8/64 | 5/64 | 3/64 | 167 | yes |
| N_2B_P_SLOW | 5402 | 7/64 | 1/64 | 2/64 | 115 | yes |
| N_A_P_SLOW | 5401 | 7/64 | 8/64 | 5/64 | 197 | yes |
| N_A_P_SLOW | 5402 | 7/64 | 1/64 | 0/64 | 135 | yes |

Immediate complete-answer acquisition was 848/15360 across updating native presentations. Immediate success and end-of-phase retrieval are distinct; neither a transient correct answer nor a lower teacher-forced loss establishes retained rule learning.

## Conditional unseen addition

All acquisition/retention reports and admission decisions were hash-locked across the roster before any held score. Unready groups have no held score; they are not assigned zero accuracy. These two worlds provide descriptive development evidence only.

### Paired input-path effects

All entries are percentage-point differences in complete accuracy; the immediate column is per updating presentation. These are descriptive within-world contrasts, not independent trial counts. Native effective write gains differ across front ends, so these are complete input-path effects.

| World / measure | PERM at fast | PERM at slow | Slow at identity | Slow at PERM | Interaction |
|---|---:|---:|---:|---:|---:|
| 5401 / OLD_after | +9.38 | +12.50 | +0.00 | +3.12 | +3.12 |
| 5401 / NEW_after | +7.81 | +7.81 | +0.00 | +0.00 | +0.00 |
| 5401 / immediate_exact | +8.01 | +10.74 | -0.07 | +2.67 | +2.73 |
| 5402 / OLD_after | +6.25 | +10.94 | +0.00 | +4.69 | +4.69 |
| 5402 / NEW_after | +1.56 | +1.56 | +0.00 | +0.00 | +0.00 |
| 5402 / immediate_exact | +6.12 | +7.42 | -0.20 | +1.11 | +1.30 |

| Configuration/world | W | SH_OLD | N_OLD | W−SH_OLD | W favorite | W position-frequency |
|---|---:|---:|---:|---:|---:|---:|
| N_2B_P_FAST/5401 | 3/50 | 1/50 | 2/50 | +2/50 | 0/50 | 0/50 |
| N_2B_P_SLOW/5401 | 3/50 | 1/50 | 2/50 | +2/50 | 0/50 | 0/50 |
| N_2B_P_SLOW/5402 | 0/50 | 0/50 | 1/50 | +0/50 | 1/50 | 3/50 |
| N_A_P_SLOW/5401 | 1/50 | 0/50 | 1/50 | +1/50 | 0/50 | 0/50 |
| N_A_P_SLOW/5402 | 0/50 | 0/50 | 0/50 | +0/50 | 1/50 | 3/50 |

Read the branch-specific recitation/frequency counters in ANALYSIS.json beside these scores. A small positive difference is not confirmation; no held outcome selects a new arm, dose or mechanism.

The paired slow-permutation results do not reproduce a useful unseen gain across worlds: W/SH_OLD is 3/1 in world 5401 and 0/0 in world 5402. N_A is 1/0 and 0/0. Fast permutation is 3/1 in world 5401, with world 5402 not admitted. The first-world positives exceed their own W favorite/position counters, but do not justify a cross-world generalization or sample/compute-efficiency claim. No new core is adopted.

## Existing-checkpoint EOS attribution

Six selected V152 checkpoints were queried read-only on their taught OLD or OLD+NEW questions. No V152 training was rerun. Each ablation generated its own prefix.

| Reference/snapshot | Full exact | Full empty EOS | No-q exact | No-q empty EOS | No-q CAP | Q-only exact |
|---|---:|---:|---:|---:|---:|---:|
| R_BP/OLD_W | 0 | 64 | 0 | 0 | 64 | 0 |
| R_BP/W | 0 | 128 | 0 | 0 | 128 | 0 |
| R_LE0/OLD_W | 0 | 64 | 0 | 0 | 64 | 0 |
| R_LE0/W | 0 | 128 | 0 | 0 | 128 | 0 |
| R_LE/OLD_W | 0 | 64 | 0 | 0 | 64 | 0 |
| R_LE/W | 0 | 128 | 0 | 0 | 128 | 0 |

Removing q can change stopping at a jointly trained state without recovering correct complete answers. A no-q checkpoint intervention does not establish the result of training with q fixed at zero. The stored pairwise margins exactly decompose context and bias contributions; original checkpoint hashes and model states remained unchanged. No Q0 training followed.

## Fixed representation and writer diagnostics

The front-end factorial uses identity/PERM × the exact FE0 10-second constant/the historical slow 0.95-per-byte trace. RHO, binary X and observed activity are aligned to the native prospective read time. Raw-feature collisions and normalized-probe collisions are reported separately. No finite probe score is called an information-theoretic ceiling.

| Front/tap | World suffix | OFF_257 validation BPS | LOC_257 | OFF_C16 | LOC_C16 |
|---|---:|---:|---:|---:|---:|
| I_FAST/RHO | 5401 | 2.3740† | — | — | — |
| I_FAST/X | 5401 | 2.2126† | 2.3926 | 4.1566 | 5.0918 |
| I_FAST/OBS | 5401 | 2.1445† | 2.3186 | 4.0499 | 5.1090 |
| I_FAST/RHO | 5402 | 2.5205† | — | — | — |
| I_FAST/X | 5402 | 2.3857† | 2.4390 | 3.9003 | 4.7990 |
| I_FAST/OBS | 5402 | 2.3469† | 2.3937 | 3.7949 | 4.8186 |
| I_SLOW/RHO | 5401 | 2.6804† | — | — | — |
| I_SLOW/X | 5401 | 2.4801† | — | — | — |
| I_SLOW/OBS | 5401 | 2.3984† | — | — | — |
| I_SLOW/RHO | 5402 | 2.8129† | — | — | — |
| I_SLOW/X | 5402 | 2.6543† | — | — | — |
| I_SLOW/OBS | 5402 | 2.5954† | — | — | — |
| P_FAST/RHO | 5401 | 2.0027 | — | — | — |
| P_FAST/X | 5401 | 2.0924 | — | — | — |
| P_FAST/OBS | 5401 | 2.0995 | — | — | — |
| P_FAST/RHO | 5402 | 2.0520 | — | — | — |
| P_FAST/X | 5402 | 2.3000 | — | — | — |
| P_FAST/OBS | 5402 | 2.3153 | — | — | — |
| P_SLOW/RHO | 5401 | 2.0855 | — | — | — |
| P_SLOW/X | 5401 | 2.1723 | 2.2080 | 3.8534 | 4.6331 |
| P_SLOW/OBS | 5401 | 2.1845 | 2.2166 | 3.8678 | 4.6981 |
| P_SLOW/RHO | 5402 | 2.0855 | — | — | — |
| P_SLOW/X | 5402 | 2.3154 | 2.2805 | 3.2473 | 4.4452 |
| P_SLOW/OBS | 5402 | 2.3236 | 2.2901 | 3.2634 | 4.5056 |

No raw RHO/X/OBS view had an exact conflicting-target collision in these recorded contexts. That is not a proof of sufficient useful context. All twelve identity-front OFF_257 fits remained numerically unresolved; their factorial loss comparisons are finite-fit observations. On P_SLOW/OBS the qualified full-port batch validation losses were 2.1845 and 2.3236 BPS, versus 3.8678 and 3.2634 for C16. Yet next-byte argmax accuracy was similar, so the BPS gap concerns the declared code/scale/regularization family and probability calibration; it does not establish better complete native answers. Local C16 was worse in BPS than matched batch C16 at the same budget. These findings motivate an output/write-transfer audit, not an assertion that all information was destroyed or that a larger decoder solves addition.

† A fit stopped without the declared numerical qualification. It is a finite fitted instrument, not an established representability ceiling; no rate grid or additional optimizer budget was added. LOC/OFF comparisons share their output parameterization and objective. The native path has different reachability, decay, feature scaling and event timing, so its gap cannot be attributed solely to locality.

## Three cycles

1. **Design, audit, qualification, revise:** preserved provisional V153, established V154 identity and fresh streams, qualified native/FE endpoint parity, all output classes, semantic state serialization, local gradients and read-only prefix sharing. Raw and normalized collision interpretations were separated before training.
2. **Causal history audit, test, revise:** executed the fixed native histories independently of the external/BP diagnostics. Saved all immediate/acquisition/retention outputs, froze global admission, and applied the declared held gate without extra examples or retries. Checkpoint attribution stayed read-only.
3. **Representation/readout test, independent audit, revise:** corrected a duplicate timing-field logging defect through a separately locked operations continuation, reused the saved first fit/cache without replay, collected the declared untaught features, ran the finite same-family instruments, reduced saved outputs/receipts in a separate process, rejected mutated response fixtures, and revised the next proposal from those specific limits. No completed trajectory was replayed for a report.

## Exposure, cost and limits

- 43,274 physical/presented record applications and 123,532 reply/EOS target positions: native 25,600, feature collection 1,280, online instruments 16,384, technical 10.
- 42,806 own/teacher-forced query pairs and 8,704 instrument teacher-forced record evaluations. Evaluator labels are accounted separately from feedback writes.
- 1,824 batch objective/gradient evaluations and 5,340,192 virtual target reuses. Internal optimizer reuse is not hidden as new independent experience.
- Main controller, including its waited independent audit: 3244.05 CPU seconds; peak main RSS 566.2 MiB. One worker and one numerical thread. Saved-response postprocessing and packaging costs are recorded separately.

The target budget includes warmup, repeated presentations and write-inhibited histories. There are 128 unique taught relations/world, not one new arithmetic fact per presentation. Exact and canonicalized lookup can recite taught facts and cannot substantiate E3. External evaluators, pretrained checkpoint attribution and native learning results remain separate. Frozen historical sources and the arithmetic-design pointer were preserved; no automatic promotion, confirmation, extra n or GitHub upload occurred.

Recorded departures and the preserved/resolved logging failure are detailed in PROTOCOL_REVIEW.md. Native curves are plotted in ACQUISITION_TARGETS.png and ACQUISITION_CPU.png. Actual KC dimension was 5,177, not a historical approximate KC count. Full-KC instruments have 1,330,489 parameters (about 10.15 MiB per float64 matrix); C16 instruments have 82,832 (about 0.632 MiB). L-BFGS uses additional vectors. Global encoding sorts and minimum counted clone movement are retained in COUNTS.json; peak RSS includes the interrupted controller and its continuation.

See REVISED_DESIGN.md, SUMMARY.json, FINAL_AUDIT.json, ANALYSIS.json and the compact RESULT_BUNDLE.zip. Raw datasets, detailed per-query rows and checkpoints remain local.
