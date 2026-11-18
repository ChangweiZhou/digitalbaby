# Audit of revision 3 of the E39 brief: one linear task alone (L1a) and the life with a birth (L1b) (16 November 2026)

From the outside reviewer. Audited: `OPUS_TASK_E39_SMALL_WORLD_PROTOCOL_L0_L1_20261115.md` at `a3275395` on `codex/minifly-consolidated-20261023`, the part added in revision 3 (from "L0 and L1 as I read them" to the end), against the released part of the same brief, the L1 report on `claude/gallant-edison-p0qa68`, and my second implementation.

Everything here is read-only. Nothing was trained, and no code of another branch was run. The commands below are run from the repository root on `claude/vibrant-cerf-sje726`; `C` stands for `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_rev3_checks.py`.

## 1. Summary

The two jobs are the right next step and the specification is close. Six changes are required before release. Three of them concern how the criterion is measured and how L1a's result is turned into L1b's setting; without them the brief is likely to end in a stop for reasons it does not examine. After the six changes no re-audit is needed.

The generator is untouched: `sw_world.py` as merged has the same SHA-256 as the specification file (355e15e6…), and the released text changed in its title and first paragraph only (`git diff 95f2d80f a3275395 -- OPUS_TASK_E39_SMALL_WORLD_PROTOCOL_L0_L1_20261115.md | grep -n '^-[^-]'`).

## 2. Findings

### 2.1 L1a's selection rule is unambiguous and gives six tasks; what they are (`C A`)

Applied to all 540 tasks of world 1, the rule gives:

| Band | c1 ≠ c2 | c1 = c2 |
|---|---|---|
| 8 to 13 | task 20, −a + 2b mod 9, 40 taught pairs | task 2, a + b mod 8, 32 taught pairs |
| 18 to 24 | task 5, a − b mod 23, 264 taught pairs | task 1, 2a + 2b mod 19, 180 taught pairs |
| 30 to 37 | task 3, −a − 2b mod 33, 544 taught pairs | task 12, −a − b mod 37, 684 taught pairs |

- The six are distinct and none is a probe task.
- Tasks 1, 2 and 3 are three of the four birth tasks of world 1. That is useful: L1a measures them alone and L1b trains them jointly.
- Half of the six have c1 = c2. Of the 192 linear tasks met in L1 a quarter do, and of the birth tasks of the eight worlds 5 of 30 linear ones do (`C B`). A median over the six therefore leans toward the symmetric tasks.
- All six runs at a size start from the same initial values (`T.manual_seed(1)`), so the six differ in the task only.

### 2.2 The 32-item criterion is met well below a true level of 0.8, and differently in the two jobs (`C D`)

A window of 32 asked items reaches 0.8 when 26 are right. The chance of that at a lower true level:

| True level | One window | L1a: some window during 1,000 updates | Birth: some check during 1,024 updates, the other three tasks sure |
|---|---|---|---|
| 0.55 | 0.002 | 0.59 | 0.03 |
| 0.60 | 0.009 | 0.99 | 0.14 |
| 0.65 | 0.036 | 1.00 | 0.45 |
| 0.70 | 0.113 | 1.00 | 0.85 |

In L1a the window is examined at every asked item, sixteen times an update, so "updates to criterion" is in effect the time to a true level of about 0.55 to 0.6. In the birth the windows are examined once every 64 updates and must be met by all four tasks at once, which happens at about 0.65 to 0.7. So the two jobs measure different things under one name, neither is 0.8, and L1a's figure understates what a birth needs.

With every asked item of the task over the last 64 updates (1,024 items alone, about 256 a task in a birth) the chance of a false pass at a true level of 0.70 is 0.0002 and below.

### 2.3 The rule that sets L1b from L1a is computable, but the median does not fit the birth (`C E`)

- **Computable in every case.** The median of six is the mean of the third and fourth. If the fourth is "not reached" (above 16,384) the median is above 8,192 whatever the third is. No tie can leave it undecided.
- **The two numbers fit each other.** Alone, 8,192 updates give a task 393,216 taught items. In a birth on four tasks, 32,768 updates give each task 393,216. The 6-hour bound is not at risk: a life of 1,024 visits with a birth at the cap is 2.0 CPU-hours at 64 units and 2.7 at 96 (`C H`).
- **The median does not fit.** A size qualifies when three of its six tasks are within 8,192 updates, whatever the other three do. A birth needs all four of its tasks inside the same allowance. If half the tasks at a size need more, and tasks were independent, one birth in sixteen would complete; the gate asks six worlds of eight.

### 2.4 Tasks with few taught pairs (`C B`, `C C`)

Two of L1a's six tasks have 32 and 40 taught pairs. What is on file about such tasks:

- In L1 at the highest rung under `naive`, 2 of the 38 linear tasks with modulus 8 to 13 reached the criterion within the life, against 12 of 44 at 18 to 24 and 10 of 51 at 30 to 37. Their best level at the end of any visit averages 0.19, against 0.41 and 0.32.
- At full scale a fresh network given 792 pairs of each operation memorises them: 97 to 98% on taught pairs, about 20% on untaught ones (E35-A, arm `G`, a tenth of the pairs; E40's fresh runs agree).

So it is likely, though not shown, that a fresh network cannot learn a linear task of small modulus alone to the criterion on untaught pairs, for want of examples and not for want of size. L1a will show whether this is so. The brief does not say what follows if it is:

- Under the median rule a size still qualifies with both small tasks "not reached", provided the other four reach.
- In L1b, four of the eight worlds have a birth task of modulus 13 or less: worlds 1 (a + b mod 8), 4 (−a − 2b mod 10), 6 (three of its four) and 7 (a − b mod 12). If such a task stays unlearned, those births run to the cap, at most four worlds complete, and part (1) of the gate fails whatever the later tasks do.

### 2.5 The birth (`C B`, `C G`)

- **It is what the recommendation meant**, in outline: joint training on the first tasks, then the life as specified, each arm by its own rule, the birth counted in the ledger.
- **"The last 32 asked items" of a task is defined.** A task gets about 4 asked items an update, so 32 items are its last 8 updates or so, and the guard "at least 32 asked items" covers the improbable case.
- **The birth groups.** Worlds 1 and 5 have three linear tasks and one of max or min; the other six worlds have four linear tasks.
- **An arm whose birth does not complete** is defined: recorded as not complete, and its life runs. The brief itself gives the sweep 15% to complete in six worlds. Its life is then the life of a learner that was not born, and the comparison of the two arms in L1b says nothing about the sweep as a rule for a life.
- **Births of different length** are handled by the ledger and by counting the birth in any computation to threshold. That is sufficient, provided comparisons between arms in L1b are read as descriptions; the gate is `naive`'s alone, which the brief says.
- **What follows the birth.** In every world the life begins with visits to groups 0 and 1 only: the first group that is not a birth group is met at visit 6 to 29 (29 in world 3, after 27 visits on group 0 alone), and the birth groups take 14 to 43 of the 64 visits. The later tasks number 12 to 32 a world. The schedule is the generator's and should stay; the report should show it.

### 2.6 The scorer

- **Leaving the birth tasks out of the first-visit quantities and of the trend of learning is right.** Their first visit in the life is not a first meeting.
- **Two definitions are missing.** (a) The trend of learning was fitted from n = 3, to leave out a first group that started from nothing. With a birth that reason is gone, and the brief does not say from which n the trend is fitted. (b) "The absence counted from the end of the birth" does not say what the absence of visit 0 is. Visit 0 is group 0 in every world, and the classes begin at 1.
- **The gate does not follow from E40.** E40's learner had twelve tasks behind it and met the same four operations under a new modulus. Here a later task brings, as a rule, an operation and a modulus the learner has not met, after a birth on four tasks. A level near the ceiling is possible but less likely than one near the floor. The brief names no response to either failure, so either ends in a stop.

### 2.7 The two streams can be reproduced from the text (`C F`)

I added both streams to `e39_world.py` from the text alone (`SpecWorld.solo_batches`, `SpecWorld.birth_batches`). The brief states no digests; mine, for the first 256 updates, with the world digest's bytes per item, taught then asked:

| Stream | SHA-256 |
|---|---|
| solo, world 1, task 1 | 1fd661d4df3adec6c5224a57bb424170e56aeb029ffd3efe2e5e05d61127fcb8 |
| solo, world 1, task 2 | 84af069bc95624b1d8478ed037a683dc46d35ea9602b0612b06d45fe452b64d4 |
| solo, world 1, task 3 | 435f5c4638f01a12fc14ba55ad225d259e618559ac71e9418664fc0e5e94daa7 |
| solo, world 1, task 5 | 81271cee73bc7a2e48066c03ea1ace39214b973a0e8bb8b7a2d0c13ac6ecfbfa |
| solo, world 1, task 12 | 69fc5bf96bedb8410a29b65b8df249b7977dd6fb92fbf2e9d087796f224aa002 |
| solo, world 1, task 20 | 3de3b1eda53ced30a52b6510bda303e60883282a7322385c8627bb858c9c6d61 |
| birth, world 1 | 8ac9eb07e8e1beb2a66ee12a3c11eb9d41d5260395540e2a2e1f3d463ac5a61c |
| birth, world 2 | db31da0c84814890710c9502d10d7a4edb4fa56e276308c5d3d5616e1081c199 |
| birth, world 3 | 1e012e3c01a63ec10cb872872b124f7f6369a57bb2f6dfb3b9f0e460db4a8800 |

The same command prints them: `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py spec streams`. The three audited digests of the lives still come out of the same file (`... e39_world.py spec stats`).

Where a second reader could differ:

- **A draw for the task in a solo stream.** The brief says none is made, and that sentence is needed: `randrange(1)` consumes generator state, so a runner that calls `visit_batches` with a one-task group produces other items (digest 823bb2c6… for task 2 in place of 84af069b…).
- **One generator or one per visit.** A life has a generator per visit; the text of both new streams reads as one generator for the whole run, which is what I implemented. It should say so.
- **What is hashed.** The brief says "the same digest" without saying that it is the world digest's rule over each update's taught then asked items.
- **Which solo stream.** Check 10 says "one solo stream" and does not name it.
- **The `visit` column of a solo event file** is not stated.

### 2.8 Checks, cost, predictions

- **Checks 10 to 13** cover the streams, the unchanged life, resume and the scorer. One thing is not covered: the new quantities themselves (updates to criterion in L1a, the update at which a birth ends) are computed by one implementation only.
- **Cost** (`C H`). L1a's 24 runs come to 1.75 CPU-hours from L1's measured rates, as the brief says. L1b's "at most about 4 CPU-hours" holds at 64 units (3.99); at 96 units sixteen lives with births at the cap are 5.36.
- **Predictions** P39.10 and P39.11 are stated on the median and need restating if the rule changes. P39.13 does not say how a band counts in which neither task reaches the criterion.

## 3. Required changes

1. **One statistic for the criterion, in both jobs.** A task is at the criterion at a check, made every 64 updates, if the share right over all of its asked items in the last 64 updates is at least 0.8. L1a's updates to criterion and the end of a birth both use it. The 32-item figure stays in L1a's table as a second column, for comparison with L1.
2. **The rule that sets L1b.** A size qualifies if, under `naive`, at least three of the four tasks of the two upper bands (tasks 1, 3, 5 and 12) reach the criterion within 8,192 updates. Use the smaller qualifying size; if neither qualifies, stop as the brief says. The two tasks of the band 8 to 13 are run and tabled with their taught-pair counts, and are not counted. Restate P39.10 and P39.11 on this rule.
3. **The end of a birth.** A birth is complete at the first check at which at least three of its four tasks are at the criterion. The share right of all four is recorded at every check and at the end. The cap stays at 32,768 updates. Part (1) of the gate uses this definition.
4. **Two definitions for the scorer.** The birth counts as a visit before visit 0, so a birth task's first visit in the life has an absence of its visit number plus one, and visit 0 of group 0 has absence 1. The trend of learning is fitted over every task that is not a birth task, from n = 1. My scorer implements both (`score(..., birth_tasks=...)`), with a hand-computed fixture of two birth tasks and two later ones (`python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py spec scorer`), which the worker may use for check 13.
5. **The streams and their check.** State that each solo run and each birth has one generator for all its updates; that the digest of check 10 is the world digest's rule over each update's taught then asked items; that check 10 covers all six solo streams and the births of worlds 1, 2 and 3, against the digests in 2.7; and what the `visit` column holds in a solo event file. Add a check 14: the worker commits the event files of the 24 solo runs and of world 1's two births, and I recompute the updates to criterion and the update at which each birth ends. Check 14 does not hold the runs.
6. **The cost of L1b at 96 units** is 5.4 CPU-hours with births at the cap, not 4.

## 4. Recommendations

- **A third arm in L1b: the sweep's life after the `naive` birth.** P5 allows any optimizer in development. Eight more lives, about 1 to 1.3 CPU-hours. Without it the sweep's life cannot be read whenever its own birth is incomplete, which the brief expects, and the question returns in L2.
- **A second set of initial values for L1a.** The same rule in world 2 picks tasks 0, 4, 6, 7, 13 and 24, with seed 2: 24 more runs, 1.75 CPU-hours. The rule of change 2 then reads "at least six of the eight upper-band tasks". As written the choice of size rests on four tasks and one set of initial values.
- **Name the one further step for each way the gate can fail,** so that neither ends in a new round. If the median first-visit level is above 0.90: the same sixteen lives with visits of 256 updates. If it is below 0.10 with births complete: a birth on groups 0 to 3 with "six of eight" and a cap of 65,536.
- **Report beside the gate:** per world, the visit at which the first later group is met and the birth tasks' levels at their last visit before it; in L1a, for every run whether it reached or not, the share right over the last 256 asked items and the mean training loss over the last 64 updates, which separates a network that has memorised the taught pairs from one that has learned nothing.
- **P39.13:** say that a band in which neither task reaches the criterion does not count as "fewer".

## 5. What I could not check

- **The runner.** No code exists yet for the two streams, the birth or `--birth-groups`; whether it follows the text is what checks 10 to 14 are for.
- **L1's tables.** Section 2.4 reads the worker's score files. I did not recompute them from event files; my scorer agreed with the worker's on L0's check lives by the worker's report.
- **That small-modulus tasks cannot be learned alone.** It is an inference from L1 and from full scale. L1a tests it, and changes 2 and 3 cost nothing if it is wrong.
- **CPU figures** are the worker's measured rates, not mine.

## 6. Files

- `e39_rev3_checks.py`: parts A to H, as cited above.
- `e39_world.py`: the two streams, `stream_digest`, the scorer's `birth_tasks`, the fixture `hand_made_birth`, and the command `spec streams`.
