# Step 8 — training-only output baseline improves taught-cue emission

**Verdict: `E1_ONLINE_BASELINE_QUALIFIED` under the frozen Step 8 rules.** All 32 prespecified diagnostic worlds (11901–11932) completed, after one excluded technical world (11900). The independent receipt audit passed. This is an **E1 trained-association result**, not an E3 transfer result: no held-out relation cue was probed.

## Question and intervention

Step 7's two-store emitter chose `1` when native `value_1−value_0 > 0`. Its eight-world feasibility pilot stopped below its old-taught 70% guard. Step 8 kept the same V9.8 relation generator, Step 6 fading-pair input, two native Full151 stores, fixed `!` outcome, external Boolean teacher, opponent teaching, 144 old and 72 disjoint new records, W/N old-write contrast, and two simulated day gaps. The **only output intervention** was a branch-local scalar threshold: the mean of the last 12 unlabeled, prefeedback training margins. It emitted `1` when `value_1−value_0 > b`. Raw and calibrated choices were recorded from the **same native state**. The scalar did not read labels or test cues and did not alter Full151 teaching or memory state.

## Frozen primary result

At the final probe, after the second day and intervening new learning, calibrated W answered **273/384 = 71.09%** of old directly taught cues correctly, versus raw W **241/384 = 62.76%**. The paired mean gain across 32 worlds was **+8.33 percentage points**, with two-sided 95% Student-*t* interval **+1.34 to +15.33 pp**. The interval lower bound was positive. Calibrated N, which had no old writes, scored **184/384 = 47.92%**. The calibrated final W−N trained-cue difference was **+23.18 pp** [**+17.43, +28.93 pp**]. These two intervals use the world, not individual cue presentations, as the unit.

| Directly taught cue accuracy | W raw | W calibrated | N calibrated |
|---|---:|---:|---:|
| Old teaching end, 12 old cues | 278/384 = 72.40% | **323/384 = 84.11%** | 50.00% |
| After first day, old cues | 249/384 = 64.84% | 284/384 = 73.96% | 50.00% |
| After new cohort, old cues | 230/384 = 59.90% | 294/384 = 76.56% | 48.18% |
| Final day, old cues | 241/384 = 62.76% | **273/384 = 71.09%** | 47.92% |
| New cohort end, six new cues | 144/192 = 75.00% | **179/192 = 93.23%** | **181/192 = 94.27%** |

All six prespecified guards passed: positive final calibrated-minus-raw interval; old-end W ≥80%; final W ≥70%; positive final W−N interval; and new-end W and N ≥65%. The same architecture's raw W old-end score here was 72.40%, compared with 64.58% in Step 7's **different eight pilot worlds**. That cross-run difference is not a treatment effect. The valid comparison is calibrated versus raw on the **same 32 learner states**.

## Limits visible in the receipts

The gain is not universal across worlds: final calibrated W beats raw in **20/32**, ties in six, and is worse in six. At final, calibrated W is correct on **154/192 (80.21%)** of answer-`0` cues and **119/192 (61.98%)** of answer-`1` cues. Only **16/32** worlds individually reach 70% final calibrated old-cue accuracy. The frozen gate was an aggregate world-level criterion; passage does not mean reliable performance in every world or balanced accuracy across labels. The new-cohort task was learned strongly at new-end in both branches, but its final-day accuracy fell to 76.56% W and 85.42% N under calibrated output.

This experiment shows that the fixed zero-threshold output rule was a **contributing, actionable interface limitation** for these taught cues. It does not prove that threshold choice was the only Step 7 bottleneck, that the input code supports never-taught pair transfer, or that Full151 can derive its own teaching label. The added scalar is engineered output state and must be named in any descendant architecture. An exact-key associative table could pass the scored E1 task.

## Execution and audit

The source lock was written before the technical world. Its full timeline ran in 5.86 s; the 32 diagnostic worlds ran in 191.14 s total under `caffeinate`, each with one atomic compact receipt. No completed world was rerun or replaced and no scientific failure receipt was emitted. The independent [audit](results/AUDIT.json) regenerated the source fixtures, verified 432 training rows and 120 probe rows per world, recomputed every FIFO baseline, both output decisions, stage metrics, intervals and verdict, and rejected eight intentional changes to training/probe data, source identity and roster. It audits the recorded native values and the runner's state-nonmutation checks; it does not independently replay native trajectories.

See the frozen [design](DESIGN.md), [source lock](results/SOURCE_LOCK.json), [summary](results/SUMMARY.json), and [world receipts](results/worlds). The next E3 question requires a **separate source-locked protocol and fresh worlds**, with a predeclared trained-cue positive control that addresses the final answer-`1` weakness and per-world variability. No Step 7 result is retrospectively changed, and no E3 claim follows from this E1 result.
