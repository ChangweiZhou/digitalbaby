# E39 jobs L0 and L1: the small world — report (16 November 2026)

Brief: `OPUS_TASK_E39_SMALL_WORLD_PROTOCOL_L0_L1_20261115.md`, revision 2 (released), on `codex/minifly-consolidated-20261023` at 2b65392f. Work on `claude/gallant-edison-p0qa68`: L0 at 7411e743, the three L1 rungs at 0547bacf (R0), d328bc0f (vu512) and e0a9f550 (h96vu512); this report in the commit that adds it. Owner's instruction for this session: do revision 2, build L0 and report its nine checks (check 1's digests must match), then run L1 without waiting unless a check fails.

Evidence level: none for L0 (an instrument, **TECHNICAL_PASS**). L1 is a calibration of that instrument; its figures are E2-type readings in the small world and are claims about nothing at full scale. **Nothing is adopted.** L2 is **NOT_RUN**.

Host: Linux x86-64 container, 4 cores, Python 3.13.18, torch 2.15.0+cpu (installed for this session), numpy 2.5.3; one thread per process, at most three processes, launched detached without a time limit. Peak memory: **not measured**.

## Summary

1. **L0: all nine checks pass** (`L0/E39_L0_CHECKS.md`, machine record `L0/E39_L0_CHECKS.json`). The generator is byte-identical to the specification file and gives the audit's six digests in full (w1 05e48881… / ef5afac9…, w2 3d7ba088… / 983ff313…, w3 ea204b19… / dfc3270f…); the reviewer's second implementation gives the same six on this host, agrees with this scorer on every quantity of two 16-visit lives, and rebuilds the event files in every column but `correct`.
2. **L1: the gate fails at every rung of the ladder.** The ladder as written is exhausted: R0 (default), then visits of 512 updates, then H = U = 96 with visits of 512. In no rung did more than zero worlds pass both parts. No rung is adopted, so L2 has no size to start from; what follows is the coordinator's decision.
3. **Why it fails, from the listed readings.** The instrument is not on its floor everywhere; it is on its floor for the linear operations. Max and min are learned within their first visit (mean first-visit level 0.55 at R0, 0.76 and 0.80 at the higher rungs; 16 of the 20 such tasks reach 0.8 at R0, all 20 at the higher rungs). The 16 linear operations, 192 of the 212 tasks met under `naive`, stay near chance: **1 of 192 reaches 0.8 at R0**, 20 at vu512 and 30 at h96vu512. Since most tasks are linear, the median first-visit level of later tasks is 0.03–0.06 at R0 and 0.03–0.25 at the higher rungs (part 2), and the first group, which in every world holds at least one linear task, has both tasks at 0.8 in 0, 0 and 1 of 8 worlds (part 1).
4. **The sweep learns less than backpropagation**, in every world at every rung, and the gap grows with the dose: mean score over 64 visits `naive` 11.3 / 18.3 / 19.6%, `sweep` 9.6 / 12.4 / 12.3% (R0 / vu512 / h96vu512). On the linear operations the sweep barely moves: 1, 4 and 2 of 192 tasks reach 0.8.
5. **Cost is not the obstacle.** Forecast CPU for a 1,024-visit life under `naive`: 1.0, 1.9 and 2.6 CPU-hours (bound 6).

## The learner and the ledger

| | R0, vu512 | h96vu512 |
|---|---|---|
| Network | `sw_net.SWNet`: `book` 258×8 (BOS row padding), `l1` 8→64, `GRU` 64→64, `lm`, `l2` 64→64, `l3` 64→257 | the same at 96 |
| Parameters (embedding included) | 52,625 | 102,353 |
| Optimizer state | AdamW moments 105,250, 13 step counters | 204,706, 13 |
| Buffer, anchor | none (L1 runs no replay arm), 0 | none, 0 |
| Update | `exp26_candidate.step`'s text: `exp45.batch_tensor`, `arch26.forward(m, x, local, True)`, answer-masked cross entropy, one AdamW step (lr 0.001, betas 0.9/0.98, decay 0.25, embedding table without decay) | same |
| Stream per update | 64 items: 48 taught (all learned from), 16 asked (answered first, greedy 3 bytes incl. the end symbol, never learned from) | same |
| Updates per life | 16,384 (R0); 32,768 (vu512) | 32,768 |
| Taught presentations per life | 786,432 (R0); 1,572,864 | 1,572,864 |
| Answered asked items per life | 262,144 (R0); 524,288 | 524,288 |

Unique taught pairs: round(p²/2) per task (32 to 684); a task met in a life gets 6,144 (R0) or 12,288 taught presentations per visit.

**CPU, measured.** L0 check lives 0.09 CPU-hours and the check assembly under a minute; L1: R0 1.06, vu512 2.00, h96vu512 2.69 CPU-hours; L1 in all 5.74 CPU-hours over 48 lives. CPU-seconds per visit (mean of eight worlds): R0 `naive` 3.53, `sweep` 3.90; vu512 6.66, 7.35; h96vu512 8.97, 9.88.

**The learner's time constant at each rung** (brief: a roster that changes the visit length restates it). 1 / (lr × decay) = 4,000 updates: 15.6 visits at R0; 7.8 visits at vu512 and h96vu512, where a short absence (1–4 visits) is 0.13–0.51 constants, a medium one (5–32) 0.64–4.1 and a long one (33+) 4.2 or more.

**What is committed** (the brief's rule, written into `.gitignore`): every receipt, every score table, the per-rung tables `E39_L1_<rung>.json`, and event files and states of world 1 for each arm; event files and states of worlds 2–8 by their hashes in the receipts.

## The gate, rung by rung (under `naive`)

| Rung | Part (1): worlds | Part (2): median first-visit level of tasks n ≥ 3, worlds 1–8 | Part (2): worlds in 0.10–0.90 | Worlds passing both | Forecast, 1,024 visits | Gate |
|---|---|---|---|---|---|---|
| R0: H = U = 64, 256 updates a visit | 0 of 8 | 0.062, 0.047, 0.062, 0.031, 0.047, 0.047, 0.031, 0.062 | 0 | 0 | 1.03 h | fails (low) |
| vu512: 512 updates a visit | 0 of 8 | 0.094, 0.125, 0.062, 0.062, 0.188, 0.031, 0.062, 0.125 | 3 | 0 | 1.92 h | fails (low) |
| h96vu512: H = U = 96, 512 a visit | 1 of 8 (world 7) | 0.094, 0.125, 0.078, 0.109, 0.250, 0.062, 0.078, 0.062 | 3 | 0 | 2.62 h | fails (low) |

Every life's stream digest equals `sw_world.digest` of its world at its visit length (48 of 48), so every rung's digests are recorded in the receipts and in `E39_L1_<rung>.json`. The first groups (worlds 1–8) are: max mod 26 with 2a+2b mod 19; −2a+2b mod 21 with 2a+b mod 20; −2a−2b mod 18 with 2a−b mod 33; a−2b mod 32 with −a−2b mod 10; −a+b mod 18 with −a−2b mod 32; 2a−2b mod 11 with 2a−b mod 12; −2a−b mod 18 with −2a−2b mod 34; −a+b mod 20 with −a+b mod 24. Only world 1's starts with an operation the learner takes at once (max), and its partner never reaches 0.8 at any rung.

## The figures L1 reports

All per-life tables are in `E39_L1_TABLES.md`; every task's first-visit level and first-visit cost in `E39_L1_first_visit.csv` (with operation and modulus); score by age and the blind predictor's in `E39_L1_score_by_age.csv`. Pooled summaries over the eight worlds follow.

**Score by age** (percent of asked items, mean over worlds, in four blocks of 16 visits) and the blind predictor (5.0–7.0% over the life, by world, at every rung):

| Rung | Arm | Visits 0–15 | 16–31 | 32–47 | 48–63 | Score trend (second half), worlds positive |
|---|---|---|---|---|---|---|
| R0 | naive | 9.1 | 12.3 | 10.9 | 13.1 | 6 of 8 |
| R0 | sweep | 7.4 | 10.9 | 9.6 | 10.6 | 5 of 8 |
| vu512 | naive | 11.4 | 18.1 | 19.9 | 23.7 | 5 of 8 |
| vu512 | sweep | 10.2 | 13.4 | 13.0 | 13.1 | 4 of 8 |
| h96vu512 | naive | 10.8 | 17.1 | 22.8 | 27.9 | 5 of 8 |
| h96vu512 | sweep | 9.3 | 12.2 | 13.2 | 14.6 | 5 of 8 |

**First-visit level by order of meeting** (mean, pooled over worlds; counts of tasks in brackets) and the **trend of learning** (slope of first-visit level on ln n, n ≥ 3):

| Rung | Arm | First group | n = 3–10 (64) | n = 11–20 (76) | n ≥ 21 (56) | Trend of learning: mean, worlds positive |
|---|---|---|---|---|---|---|
| R0 | naive | 0.051 | 0.073 | 0.109 | 0.133 | 0.028, 5 of 8 |
| R0 | sweep | 0.049 | 0.068 | 0.102 | 0.114 | 0.017, 4 of 8 |
| vu512 | naive | 0.096 | 0.107 | 0.201 | 0.244 | 0.054, 6 of 8 |
| vu512 | sweep | 0.088 | 0.076 | 0.146 | 0.155 | 0.032, 5 of 8 |
| h96vu512 | naive | 0.084 | 0.101 | 0.220 | 0.253 | 0.088, 7 of 8 |
| h96vu512 | sweep | 0.092 | 0.086 | 0.140 | 0.150 | 0.030, 5 of 8 |

The rise with n is in max and min: their first-visit level is 0.55 / 0.76 / 0.80 against 0.05 / 0.12 / 0.12 for the linear operations (`naive`, R0 / vu512 / h96vu512). A max or min task met later may find its symbol's operation already learned under other moduli (the symbol is shared across moduli); that is not separated here, and the composition of tasks by order is the world's, not the learner's.

**Tasks that reach the 0.8 criterion within the life** (cost reported, not compared; tasks met, eight worlds):

| Rung | Arm | Linear, c1 ≠ c2 (144) | Linear, c1 = c2 (48) | Max (8) | Min (12) |
|---|---|---|---|---|---|
| R0 | naive | 0 | 1 | 8 | 8 |
| R0 | sweep | 0 | 1 | 8 | 8 |
| vu512 | naive | 12 | 8 | 8 | 12 |
| vu512 | sweep | 2 | 2 | 8 | 12 |
| h96vu512 | naive | 23 | 7 | 8 | 12 |
| h96vu512 | sweep | 1 | 1 | 8 | 12 |

**The first group at the end of each of its visits**: in `E39_L1_TABLES.md`, per arm and world. At R0 only the max task of world 1 and the −2a−2b mod 18 task of world 3 reach the 0.8 criterion (after 22,336 and 39,102 taught items); the other fourteen first-group tasks end every visit between 0.00 and 0.25. At h96vu512 world 7's two tasks reach 0.8 late (after 171,379 and 187,283 taught items).

**Kept at return, the visit's first update** (percent right, pooled over items of all returns in eight worlds; 560 short, 230 medium and 22 long returns, the same at every rung since the schedule does not depend on the visit length):

| Rung | Arm | Short: wrap / no wrap | Medium: wrap / no wrap | Long: wrap / no wrap |
|---|---|---|---|---|
| R0 | naive | 9.2 / 10.9 | 5.2 / 7.9 | 1.8 / 9.0 |
| R0 | sweep | 7.0 / 11.9 | 4.6 / 8.4 | 4.6 / 4.5 |
| vu512 | naive | 12.4 / 18.0 | 5.9 / 6.2 | 2.8 / 6.0 |
| vu512 | sweep | 9.9 / 11.8 | 5.8 / 7.6 | 4.6 / 4.5 |
| h96vu512 | naive | 12.5 / 17.6 | 4.6 / 10.1 | 1.8 / 4.5 |
| h96vu512 | sweep | 7.7 / 14.7 | 4.5 / 7.6 | 3.7 / 6.0 |

Per world, with the first-32 version beside it, in the tables. Wrap and no-wrap items differ in more than wrapping: `needs_wrap` is nearly constant for half the linear operations (the audit), so the split mixes operations.

**Symmetric operations by `mirror`** (percent right, pooled over worlds):

| Rung | Arm | Symmetric, mirror = 1 | Symmetric, mirror = 0 | Non-symmetric | Blind on mirror = 1 |
|---|---|---|---|---|---|
| R0 | naive | 28.6 | 24.5 | 5.4 | 7.1 |
| R0 | sweep | 24.6 | 20.8 | 4.6 | 7.1 |
| vu512 | naive | 35.2 | 31.2 | 12.5 | 7.1 |
| vu512 | sweep | 30.5 | 26.0 | 6.3 | 7.1 |
| h96vu512 | naive | 34.9 | 31.1 | 14.4 | 7.1 |
| h96vu512 | sweep | 28.5 | 23.9 | 6.9 | 7.1 |

The mirror shortcut is worth about 4 points (3.8 to 4.6): items whose mirrored pair was taught are only a little easier than those whose mirror is itself asked. What the symmetric operations score, they score mostly as operations (max and min), not by recall plus commutativity.

## To set beside the full-scale figures of E36 to E38

These are pooled over all returns, learned tasks and tasks at chance together; the blind predictor is at 5–7%.

**How fast a task is lost when the stream leaves it.** Level at the end of the previous visit (mean of its last 32 asked items) against the share right on the return's first update:

| Rung | Arm | Short: before → kept | Medium | Long |
|---|---|---|---|---|
| R0 | naive | 13.0 → 9.7 | 11.2 → 6.1 | 8.1 → 4.5 |
| vu512 | naive | 21.0 → 14.2 | 20.4 → 6.0 | 19.2 → 4.0 |
| h96vu512 | naive | 21.1 → 14.1 | 22.4 → 6.5 | 17.5 → 2.8 |
| vu512 | sweep | 14.4 → 10.5 | 13.4 → 6.4 | 9.8 → 4.5 |
| h96vu512 | sweep | 12.9 → 9.9 | 14.0 → 5.5 | 11.2 → 4.5 |

Under `naive` at 512 updates a visit, a short absence (0.13–0.51 time constants) keeps about two thirds of the level (14 of 21 points); after a medium one (0.64–4.1 constants) the first update scores 6.0–6.5%, at the blind predictor's level.

**How fast it is restored at return.** Returns at which the visit's own last 32 reach 0.8, and the median taught items within the visit needed: `naive` R0 short 55 of 560 (300 items), medium 15 of 230 (2,194), long 1 of 22; vu512 80 of 560 (1,233), 25 of 230 (4,274), 3 of 22 (7,920); h96vu512 87 of 560 (1,600), 31 of 230 (3,715), 4 of 22 (10,010). Under the sweep, 37 to 64 of 560 short returns, 14 to 19 of 230 medium. Of all returns that reach 0.8 under `naive`, max and min make 44 of 71 at R0 and about half at the higher rungs (53 of 108, 55 of 122); under the sweep nearly all (47 of 62, 55 of 84, 56 of 57); the linear returns that reach it belong to the few linear tasks learned at all.

**First-visit level of late tasks against the first.** The first group's mean first-visit level is 0.05 (R0) and 0.08–0.10 (higher rungs); tasks met as the 21st or later average 0.13 (R0) and 0.24–0.25 (higher rungs) under `naive`, 0.11–0.16 under the sweep. The first group's first visit starts from the network's initial values; later tasks start from a network with a life behind it, and the mix of operations differs by world.

## Predictions

| | Prediction (the brief's) | Chance | Outcome |
|---|---|---|---|
| P39.1 | The reviewer's second implementation gives the three digests at the first attempt | 75% | Judged in the brief: false from the text alone, true from the file's rules. Here: its `SpecWorld` gives all six on this host. |
| P39.3 | At the default size a 1,024-visit life is forecast within 6 CPU-hours | 75% | **True**: 1.0 (`naive`) and 1.1 (`sweep`) CPU-hours, from L0 check 8 and the R0 lives. |
| P39.5 | Some rung of the ladder passes within the 6-hour bound | 70% | **False**: no rung passes (0, 0, 0 worlds of 8); every rung is within the cost bound. |
| P39.8 | The revised gate passes at the default size | 35% | **False**: 0 of 8 worlds. |
| P39.9 | At the default size both first-group tasks reach 0.8 within 64 visits in six of eight worlds | 60% | **False**: 0 of 8 (two single tasks reach it, in worlds 1 and 3). |
| P39.6 | In L1, the mean first-visit level of later tasks is above the first group's in six of eight worlds under `naive` | 65% | No rung was adopted, so it is judged at each: **true at R0 (7 of 8)**, false at vu512 (5 of 8), true at h96vu512 (6 of 8). Near the floor at R0 (0.05 against 0.10 in the pooled mean); see the caveat above on what the first visit starts from. |
| P39.7 | `naive` and `sweep` differ by less than 10 points in mean score over the 64 visits in six of eight worlds | 60% | **True at every rung**: 8, 7 and 6 of 8 worlds (largest differences 7.4, 10.8 and 12.5 points). At R0 because both are near the floor; at the higher rungs the sweep is below `naive` in all eight worlds, by 5.9 and 7.3 points on average. |

## What this does not show

- Nothing at full scale. The small world is smaller in every dimension than E26's; no claim here holds until repeated there.
- Nothing about long lives. L1's lives are 64 visits; the question the protocol is built for (whether a learner is stuck over 1,024 visits) needs L2, which cannot start until a size is adopted.
- No claim that the linear operations are unlearnable by this network. They are unlearned within 16,384 to 32,768 updates of a life spread over many tasks; whether more updates per visit, another rate or decay, or a wider network would change that was not tried beyond the ladder.
- The order-of-meeting rise in first-visit level is not shown to be learning to learn: it is concentrated in max and min, whose symbol is shared across moduli.
- No independent re-run: the reviewer's code was used for checks 1 and 7 only. No formal checker or seal exists; none is claimed.

## Decision record

- **Anchor.** `claude/gallant-edison-p0qa68` from 2b65392f; brief E39 revision 2; learner `sw_net.SWNet` (52,625 / 102,353 parameters); arms `naive` and `sweep`.
- **Question.** Is the instrument off its floor and its ceiling for L2 at some rung of the ladder, within 6 CPU-hours a 1,024-visit life?
- **Falsifier.** The revised gate at each rung.
- **What ran.** L0: nine checks, all pass. L1: 48 lives (three rungs × 2 arms × 8 worlds), 5.74 CPU-hours, every one complete; nothing else.
- **Result.** The gate fails at all three rungs, always low. The floor is the linear operations', not the instrument's: max and min are learned at once, linear operations near chance at R0 and rarely past 0.8 at the higher rungs. The sweep learns less than backpropagation at every rung. Cost is 1.0 to 2.6 CPU-hours a long life.
- **Disposition.** STOP as the brief says: no rung adopted; L2 not started; nothing changed in the generator, runner or scorer.
- **Next action.** None by me. A new rung, task set or learner setting is a change the brief does not authorise, and a generator change needs an audit. For whoever decides: the cost bound leaves about 2.3 times the updates of h96vu512 within 6 CPU-hours.
