# Full151 output Step 5 — equal-write competitor-content crossover

## Result and scope

**Prespecified verdict: `CUE_CONTENT_INTERFERENCE_DEMONSTRATED`.** With the FE0 input, four copied Full151 output stores, answer-byte teaching rule, and native write count held fixed, changing the *two other question strings* made the learned output for `0+1=` fail. In the reference panel A it answered `1` in **16/16** paired schedule worlds; in panel B it did so in **1/16**. The paired difference in correct-class margin, B minus A, was **−0.2428** (two-sided 95% Student-*t* interval **[−0.2536, −0.2319]**). Every one of the 16 paired differences was negative. All three decision gates frozen in [DESIGN.md](DESIGN.md) passed.

This is an **E1, exact-taught-expression** diagnostic. It shows that the identity of competing input strings matters even when teaching dose, answer-byte order, exposure count, clock, output rule, and learner birth state are matched. It does not show changed-presentation transfer, unseen-sum generalization, a ten-digit output system, or autonomous stopping. An exact-key memory could pass the reference panel.

## Paired intervention and outcomes

Every panel taught `0+0=→0` and the focal `0+1=→1`, plus one question for answer 2 and one for answer 3. Each of 16 new schedule identities used the **same cue-index order** in A, C, D, and B, with 12 exposures per fact. The index order also fixed the answer-byte sequence. The W branch made four native plastic writes per record, **192 per life**; the N branch received the same panel-specific stream with writes disabled. Thus this is a contrast of question-byte content under the existing four-channel teaching rule, not a change in the number of updates.

| Panel | Answer-2 question | Answer-3 question | Focal `0+1=` W correct | Mean focal W margin | W correct across all 64 probes |
|---|---|---|---:|---:|---:|
| A | `0+2=` | `1+2=` | 16/16 | +0.1662 | 64/64 |
| C | `1+1=` | `1+2=` | 6/16 | −0.0097 | 54/64 |
| D | `0+2=` | `2+1=` | 0/16 | −0.0977 | 48/64 |
| B | `1+1=` | `2+1=` | 1/16 | −0.0766 | 49/64 |

The focal margin is output-class-1 value minus the **largest** competing class value. The other three W facts were correct in **16/16 worlds in every panel**; all observed W deficits were confined to `0+1=`. N answered `0+1=` correctly in **0/16** in each panel, and its apparent 16/16 on `0+0=` reflects the default tie outcome rather than learning. The prespecified A–B contrast is the inferential result. C, D, and the 2×2 interaction are descriptive diagnostics: C−A mean −0.1759, D−A mean −0.2639, and B−C−D+A mean +0.1970. Since the maximizing rival class changes among panels, that interaction cannot be read as a synaptic interaction.

The D change, `1+2=` to `2+1=`, preserves the two operand symbols while changing their order and final operand. By itself it destroyed the focal answer in all 16 schedules. That makes late-symbol/order-sensitive address interference a concrete hypothesis, but this experiment does **not** identify which KC features or Full151 updates caused the effect. The C change also alters symbol frequencies. Neither contrast alone isolates the front end from downstream plasticity.

## Relation to earlier results

Step 3 showed small-panel one-byte recitation was possible, but the crowded panel's `0+1=` failed. Step 4's write lesions restored that answer when competing records were prevented from updating memory; those lesions changed teaching dose and sacrificed other facts. Step 5 resolves the narrower ambiguity: **even with all four answers taught equally often and all four stores updated by the same rule, competitor *content* determines whether the focal association survives.** It does not imply that Full151 is intrinsically defective. The existing system can learn all four facts in panel A and fails under the different address geometry of the other panels.

This also explains why a read-only distance threshold is insufficient. Native values depend on the full sequence of overlapping positive and neutral updates and on which rival wins the argmax. Any proposed encoder must be judged by learned behavior on a crowded panel, in addition to its static code geometry.

## Audit and next decision

The [independent reducer](audit.py) checked the frozen source digest, 16 complete paired receipts, input hashes, identical cue-index and teacher-byte schedules, byte-derived routing, write counts, probes and argmax, recomputed the paired interval, and rejected six injected receipt corruptions. The excluded technical identity 11500 passed before science; no failed scientific run was restarted. Full machine-readable results are in [SUMMARY.json](results/SUMMARY.json), with individual [world receipts](results/worlds) and [source lock](results/SOURCE_LOCK.json).

The next useful bounded iteration is a **candidate input-address qualification**, not an all-100-fact run. First screen a fixed, label-free encoder without learning for full-expression distinctness, nonidentical but related shifted presentations, and long-string counterexamples such as `ABACA`/`ACABA`. Then hold Full151, the four-output teaching rule, panels, and schedules fixed for a crowded E1 behavior test and a never-taught changed-presentation E2 probe. A candidate qualifies only if learned output improves without erasing other facts, and E2 exceeds its no-write control. Target-only teaching is a separate policy experiment: it reduces native plastic writes from four to one per record and therefore must not be folded into this equal-write comparison. The larger objective still needs delayed E3 reuse and a no-prior-teaching causal control.
