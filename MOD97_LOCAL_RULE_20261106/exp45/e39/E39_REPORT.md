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

---

# Addendum (16 November 2026): jobs L1a and L1b, revision 5

Brief: `OPUS_TASK_E39_SMALL_WORLD_PROTOCOL_L0_L1_20261115.md`, revision 5, on `codex/minifly-consolidated-20261023` at 8183db86 (the brief is unchanged at that branch's head 517a05a8 when this was written). Work on `claude/gallant-edison-p0qa68`: the check-10 digests at 1d217996 (committed before any comparison); the solo and birth streams, the third arm and the scorer with birth groups at b260bbb0; a fix to the solo progress line at 63145a05; L1a's 72 runs, its table and check 13's record at 0c985f3c; this addendum in the commit that adds it. Owner's instruction for this session: do jobs L1a and L1b of revision 5. Commit the check-10 digests first and make them equal the reviewer's nine. Run L1a's 72 solo runs at 64, 96 and 192 units and table them. Apply the size rule, and if no size qualifies, stop and report. Report the gate, the listed figures and P39.10 to P39.21, with no readings beyond the brief's.

Evidence level: L1a is a calibration measurement in the small world (E2-type readings, claims about nothing at full scale). **The size rule fails at all three sizes**, so the brief's stop applies. **L1b is NOT_RUN**: none of its 24 lives, and none of checks 11, 12 or 15 or the births' half of check 14. **Nothing is adopted.**

## Summary

1. **Check 10 passes**: the runner's nine stream digests (six solo streams of world 1, births of worlds 1 to 3) equal the reviewer's in full (`L1b/E39_CHECK10.json`). They were committed at 1d217996 before the reviewer's code or full digests were read.
2. **L1a: all 72 runs complete.** They cover 12 tasks by the brief's rule, worlds 1 and 2, sizes 64, 96 and 192, arms `naive` and `sweep`, 16,384 updates each, all on one source.
3. **No run of the 72 reaches the criterion** (share right ≥ 0.8 over the task's asked items of the last 64 updates) within 16,384 updates. That holds at any size, under either arm, in either world, in any band.
4. **The size rule: no size qualifies.** Under `naive`, 0 of the 8 upper-band tasks reach the criterion within 8,192 updates at 64, at 96 and at 192 units (six needed). Per the brief: **stop and report**. "The task space or the learner's settings are the next thing to change, which is a decision for the reviewer and me and not a rung."
5. **192 units is not the size**, so the brief's forecast for a 1,024-visit life at 192 units before L1b does not apply and was not made.
6. **Cost:** 7.35 CPU-hours for the 72 runs. Peak resident memory per process: 379 MiB at 64 and 96 units, 388 MiB at 192.

## L1a: the tasks and the runs

The rule's tasks (`python3 sw_l1a.py tasks`). World 1's match the brief's list. World 2's are 6, 24, 0, 13, 7 and 4, all among the six the brief names.

| World | Band | c1 ≠ c2 | c1 = c2 |
|---|---|---|---|
| 1 | 8 to 13 | task 20, −a + 2b mod 9, 40 taught pairs | task 2, a + b mod 8, 32 |
| 1 | 18 to 24 | task 5, a − b mod 23, 264 | task 1, 2a + 2b mod 19, 180 |
| 1 | 30 to 37 | task 3, −a − 2b mod 33, 544 | task 12, −a − b mod 37, 684 |
| 2 | 8 to 13 | task 6, −a + b mod 12, 72 | task 24, −a − b mod 11, 60 |
| 2 | 18 to 24 | task 0, −2a + 2b mod 21, 220 | task 13, −2a − 2b mod 18, 162 |
| 2 | 30 to 37 | task 7, a − 2b mod 35, 612 | task 4, 2a + 2b mod 37, 684 |

**Runs.** Each run is `sw_run.py solo <arm> <world> <task> --H <H> --updates 16384`. It starts from the network's initial values (`T.manual_seed(S)`, then the constructor; 52,625, 102,353 and 349,841 parameters at 64, 96 and 192) and uses the solo stream `Random("world|S|solo|<task>")` with no draw for the task. The rate and decay are the defaults (lr 0.001, decay 0.25, betas 0.9/0.98, embedding table without decay). Each update teaches 48 items and asks 16, and the asked items are answered before the update. A run has 786,432 taught presentations and 262,144 answered asked items; `visit` is 0 throughout. At most three processes ran at a time, one thread each, the largest size first.

**Source.** Every receipt records the same sources: `sw_run.py` 664551bd…, `sw_score.py` df79c0d9…, and `sw_world.py` 355e15e6… (the specification file, byte-identical), with no file modified against the commit. In all 36 world-1 receipts the first-256-update stream digest equals check 10's digest for that task. Each of the 12 streams has one full-stream digest across its six runs.

**A first launch was aborted.** The first three runs (192 units, world 1, tasks 20 and 2) failed at update 2048. The cause was a `BufferError` in the solo progress line: a numpy view of the growing event array was kept alive. Nothing but their logs was written (`L1a/aborted/`, with `NOTE.txt`). The fix (63145a05) changed only that line. All 72 runs were then launched from the start, and no output of the aborted launch is used. Its CPU, about 0.06 CPU-hours by the runs' rate, is not measured and not in the totals.

## L1a: updates to criterion, by size and arm

The table the brief asks for, per size and arm, gives each task's updates to criterion with its count of taught pairs. It is in `L1a/E39_L1A_TABLE.md` (machine record `L1a/E39_L1A.json`), with every listed quantity per run. **Every one of the 72 entries for updates to criterion is "not reached in 16,384 updates."**

The other recorded quantities:

- **Updates to a share of 0.5:** reached in 4 of 72 runs:
  - world 1, task 12 (−a − b mod 37): `naive` at 96 units 10,624; `naive` at 192 units 13,248; `sweep` at 64 units 6,784;
  - world 2, task 4 (2a + 2b mod 37): `naive` at 96 units 5,888.

  The other 68 never reach 0.5.
- **32-item window ≥ 0.8** (the second column, which decides nothing): met in the same 4 runs, at updates 11,237, 15,614, 7,266 and 5,975 (seen 539,376, 749,472, 348,768 and 286,800). Not met in the other 68.
- **Mean training loss over the last 64 updates:** at most 0.033 in every run, and below 0.0001 in all 24 runs of the band 8 to 13.
- **Share right over the last 256 asked items** (wrap and no-wrap separately in `E39_L1A_TABLE.md`):

| World | Band | Task | Taught pairs | 64 `naive` | 64 `sweep` | 96 `naive` | 96 `sweep` | 192 `naive` | 192 `sweep` |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8–13 | 20 (−a+2b mod 9) | 40 | 0.004 | 0.000 | 0.000 | 0.000 | 0.012 | 0.000 |
| 1 | 8–13 | 2 (a+b mod 8) | 32 | 0.023 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 18–24 | 5 (a−b mod 23) | 264 | 0.242 | 0.043 | 0.062 | 0.047 | 0.102 | 0.094 |
| 1 | 18–24 | 1 (2a+2b mod 19) | 180 | 0.039 | 0.047 | 0.164 | 0.016 | 0.004 | 0.016 |
| 1 | 30–37 | 3 (−a−2b mod 33) | 544 | 0.109 | 0.102 | 0.168 | 0.062 | 0.090 | 0.078 |
| 1 | 30–37 | 12 (−a−b mod 37) | 684 | 0.277 | 0.691 | 0.590 | 0.367 | 0.426 | 0.473 |
| 2 | 8–13 | 6 (−a+b mod 12) | 72 | 0.004 | 0.000 | 0.000 | 0.008 | 0.023 | 0.000 |
| 2 | 8–13 | 24 (−a−b mod 11) | 60 | 0.020 | 0.008 | 0.000 | 0.000 | 0.023 | 0.000 |
| 2 | 18–24 | 0 (−2a+2b mod 21) | 220 | 0.031 | 0.031 | 0.020 | 0.047 | 0.035 | 0.047 |
| 2 | 18–24 | 13 (−2a−2b mod 18) | 162 | 0.184 | 0.359 | 0.223 | 0.297 | 0.250 | 0.312 |
| 2 | 30–37 | 7 (a−2b mod 35) | 612 | 0.137 | 0.023 | 0.117 | 0.141 | 0.391 | 0.242 |
| 2 | 30–37 | 4 (2a+2b mod 37) | 684 | 0.324 | 0.094 | 0.562 | 0.273 | 0.398 | 0.367 |

## The size rule

A size qualifies if, under `naive`, at least six of the eight upper-band tasks (world 1's tasks 1, 3, 5 and 12; world 2's tasks 0, 13, 7 and 4) reach the criterion within 8,192 updates.

| Size | Parameters | Upper-band tasks within 8,192 updates (`naive`) | Qualifies |
|---|---|---|---|
| H = U = 64 | 52,625 | 0 of 8 | no |
| H = U = 96 | 102,353 | 0 of 8 | no |
| H = U = 192 | 349,841 | 0 of 8 | no |

**No size qualifies. L1b is not run** (24 lives, three arms: NOT_RUN). The brief names the next thing to change as the task space or the learner's settings, and the choice is the reviewer's and the coordinator's, not a rung. The forecast for a 1,024-visit life at 192 units is asked for only if 192 is the size, so it is not reported.

## Checks 10 to 15

| Check | Status | Record |
|---|---|---|
| 10. Solo and birth streams | **Passes**: 9 of 9 digests equal the reviewer's in full (`e39_world.py spec streams`). The worker's nine were committed first (1d217996), from the brief's text. | `L1b/E39_CHECK10_DIGESTS_WORKER.json`, `L1b/E39_CHECK10.json` |
| 11. The life after a birth is the life | NOT_RUN (no L1b lives) | — |
| 12. Birth and resume | NOT_RUN (no L1b lives) | — |
| 13. The scorer with birth groups | **Passes**, run on its own because it needs no life. The hand-made event file (two birth tasks, two later tasks, four visits) gives every hand-computed value. The reviewer's fixture (`e39_world.py` at ae61164e, rebuilt with a zero mirror column) passes its own assertions on this scorer. The reviewer's `spec scorer` self-test exits 0. Without birth groups, the 48 L1 score tables are reproduced exactly (0 differ). | `L1b/E39_CHECK13.json` |
| 14. The new quantities, by a second implementation | **Half in place, not passed.** The event files of all 72 solo runs are committed (`L1a/sw_solo_*.npz`, worlds 1 and 2). World 1's births are NOT_RUN. The recomputation is the reviewer's and has not been done. As a same-author consistency check only, updates to 0.8 and to 0.5 and the last-256 share were recomputed from the 72 event files and agree with every receipt. | `L1a/` |
| 15. The third arm's start | NOT_RUN (no L1b lives) | — |

## Predictions (revision 4 and 5)

| | Prediction (the brief's) | Chance | Outcome |
|---|---|---|---|
| P39.10 | L1a: 96 units qualifies | 45% | **False**: 0 of 8 upper-band tasks within 8,192 updates (six needed). |
| P39.11 | L1a: 64 units qualifies | 25% | **False**: 0 of 8. |
| P39.20 | L1a: 192 units qualifies | 60% | **False**: 0 of 8. |
| P39.21 | If 192 units is the size used, a life of 1,024 visits with its birth is forecast within 6 CPU-hours | 40% | **Not judged**: its condition is not met (192 is not the size used), and no forecast was made. |
| P39.12 | L1a: under the sweep at 96 units, at least four of the eight upper-band tasks reach the criterion within 16,384 updates | 25% | **False**: 0 of 8. |
| P39.13 | L1a, world 1, `naive`, 96 units: the c1 = c2 task reaches the criterion in fewer updates than the c1 ≠ c2 task of the same band in at least two of three bands (a band where neither reaches does not count) | 50% | **False**: in all three bands neither task reaches it, so no band counts. |
| P39.18 | L1a: no task of the band 8 to 13 reaches the criterion under `naive` at either size, in either world | 55% | **True** as written (64 and 96 units): 0 of 4 at each. At 192 units also 0 of 4. |
| P39.14 | L1b: the birth is complete in at least six of eight worlds under `naive` | 60% | Not judged: L1b NOT_RUN. |
| P39.15 | L1b: the gate passes | 35% | Not judged: L1b NOT_RUN (the brief stops before it). |
| P39.16 | L1b: the mean first-visit level of linear tasks after the birth is at least twice L1's under `naive` | 55% | Not judged: L1b NOT_RUN. |
| P39.17 | L1b: the sweep's own birth is complete in at least six of eight worlds | 15% | Not judged: L1b NOT_RUN. |
| P39.19 | L1b: `sweep-after-naive`'s mean first-visit level of later tasks is within 0.10 of `naive`'s in six of eight worlds | 40% | Not judged: L1b NOT_RUN. |

## Cost

| Size | `naive`: CPU-hours (mean CPU-seconds a run) | `sweep` | Peak resident memory, one process |
|---|---|---|---|
| 64 | 0.66 (197) | 0.73 (218) | 379 MiB |
| 96 | 0.89 (267) | 0.99 (298) | 379 MiB |
| 192 | 1.95 (584) | 2.13 (640) | 388 MiB |

The total is 7.35 CPU-hours: 3.27 for the 48 runs at 64 and 96 units, 4.08 for the 24 at 192. The brief estimated about 3.5 and 3. Check 13 took under a minute. The L1b driver (`smallworld/sw_l1b.py`) and the assembly of checks 11 to 15 (`smallworld/sw_checks_l1b.py assemble`) are committed and have not been run.

**What is committed:** all 72 receipts and event files, the run logs, the driver log, `E39_L1A.json` and `E39_L1A_TABLE.md`. The 72 end states (`*_state_end.pt`, 155 MB) are kept by their hashes in each receipt (`states[].file_sha256`, weights and optimizer hashes), not as files.

## What this does not show

- Nothing at full scale.
- No claim that these tasks cannot be learned alone by this network. They are not learned to the criterion within 16,384 updates at the default rate and decay, from one draw of initial values per world; nothing else was tried.
- Nothing about births or lives with a birth; L1b is not run.
- No independent recomputation: check 14's is the reviewer's. No formal checker or seal exists; none is claimed.

## Decision record (L1a and L1b)

- **Anchor.** `claude/gallant-edison-p0qa68`; brief E39 revision 5 (8183db86); learner `sw_net.SWNet` at H = U = 64, 96 and 192; arms `naive` and `sweep`; generator unchanged (355e15e6…).
- **Question.** Does one linear task alone reach the criterion within 8,192 updates at some size, so that L1b's birth has a size?
- **Falsifier.** The size rule: six of eight upper-band tasks under `naive`, at the smallest qualifying size.
- **What ran.** Check 10, 72 L1a runs (7.35 CPU-hours, all complete) and check 13. Nothing else.
- **Result.** No run reaches the criterion; 0 of 8 upper-band tasks at each size. No size qualifies.
- **Disposition.** STOP, as the brief says. L1b NOT_RUN. No size adopted. The generator, the adopted core and the learner's settings are unchanged.
- **Next action.** None by me. Per the brief, the task space or the learner's settings are next, a decision for the reviewer and the coordinator.
