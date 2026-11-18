# E60: the long life under the sweep that starts from back-propagation's birth (report, 18 November 2026)

**Brief.** `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/TASK_E60_THE_SWEEPS_LONG_LIFE_FROM_BACK_PROPAGATIONS_BIRTH_20261118.md`, revision 3, on `claude/vibrant-cerf-sje726` at `9bb87033`. The coordinator released it in decision 25 (`COORDINATOR_DECISIONS_20261118.md` on the default branch at `392b3eca`), and the reviewer forwarded it.

**Branch.** `claude/e44-l2-sweep-long-lives`, the branch of the helper of the long lives.

**Evidence level.** E2: a first measurement, which confirms nothing. No formal checker gate was run, and this report has not been verified. Every reading in it holds at the learning rate 0.001 only.

## 1. The question and the terms

The owner's requirement B1 (15 November) is that our learner may be slower than the conventional learner at first and overtake it in the course of continued learning. The verified long lives compared the arm `sweep` with the arm `naive`. Those two arms differ in the rule by which the life learns and also in the birth from which it starts. This job adds a third arm, so that two lives differ in the rule alone.

The terms are the brief's (section 2):

- **Rules.** *Back-propagation* (rule B) is the exact gradient. *The sweep* (rule S) passes the error backward through each linear map by the signs of its weights times their mean size.
- **Births.** A *birth* is E41's network trained beforehand under one rule: `exp45/e41/e41_<B|S>_h256_f50_<seed>.pt`, at H = 256 and U = 128, on half the pairs of add and sub modulo 89, for 40,960 updates.
- **Visits and lives.**
  - A *visit* is 256 updates on one family of tasks.
  - A *life* is 1,024 visits in a *world*. A world fixes the families and the order of the visits.
  - Worlds 1 and 4 use seed 73129, world 2 uses seed 93683 and world 3 uses seed 116089. **Worlds 1 and 4 share their births.**
- **Reduction items and kinds of visit.**
  - A *reduction item* is an asked item of add or sub whose answer needs the reduction by the modulus.
  - A later visit is a *continuation* if it directly follows the family's previous visit, and a *return* otherwise.
- **Arms.** An *arm* is a birth together with the rule of the life. None of the three arms has a replay store, and every life starts with a fresh optimizer.
  - `naive`: the birth of rule B, with a life by back-propagation.
  - `sweep`: the birth of rule S, with a life by the sweep.
  - `sweep-after-naive`: the birth of rule B, with a life by the sweep.
- **Levels.** D2 is the share right on the reduction items asked in a visit.
  - The *early level* is the mean of D2 over visits 0 to 63.
  - The *late level* (D4) is the mean of D2 over the last quarter of the visits of a life read at 256 or at 1,024 visits.
- **Trends.**
  - The *score trend* (D5) is the least-squares slope of D2 on ln(v + 1) over the returns among visits V/2 to V − 1.
  - The *trend of learning* (D7) is the slope, on ln n, of the level each family reaches at the end of its first visit. Here n is the family's place in the order of first meeting.
- **D13.** An arm is *still learning* over a set of worlds if the interval of D5 or of D7 lies wholly above zero, and *stuck* otherwise. A trend is *declining* if its interval lies wholly below zero.
- **Two intervals.** Every reading is given with each of them; both are two-sided 90% intervals of the mean, computed with `l2_score.interval90` and read with `l2_score.verdict`.
  - *Over worlds 1 to 4*: the mean of the four unrounded values ± 1.18 standard deviations (t with 3 degrees of freedom).
  - *Over worlds 1 to 3*: the mean of the three values ± 1.686 standard deviations (t with 2 degrees of freedom).
  - A difference is *lower* or *higher* when the interval excludes zero, and *not distinguished* otherwise.
- **The limit beside every reading over worlds 1 to 4:** worlds 1 and 4 start from the same births (seed 73129), so the four values are not those of four independent learners.

## 2. Runs

Four lives of `sweep-after-naive`, one in each of worlds 1 to 4. Each was started from `MOD97_LOCAL_RULE_20261106/smallworld` with the brief's command:

`python3 sw_run.py life sweep-after-naive <w> 1024 --world lw --H 256 --U 128 --learner <abs. path of exp45/e41/e41_B_h256_f50_<seed>.pt> --visit-updates 256 --wd 0.25 --lr 0.001 --list-every 8 --decay-distances none --save-states 64,128,256,384,512,640,768,896,1024 --out <abs. path of exp45/e44/L2/sweep_after_naive/lives>`

- **Driver and limits.** The lives ran through the main worker's driver `smallworld/l2_driver.py`, unchanged, from `sweep_after_naive/san_jobs.json` (written by `smallworld/l2_san.py jobs`). At most three processes ran at once, with one thread each.
- **Host.** Python 3.13.18, torch 2.15.0+cpu and numpy 2.5.3, recorded in every receipt.
- **Timing.** Worlds 1 to 3 ran from 15:08 to 18:34 UTC. World 4 ran from 18:34 to 21:57 UTC.
- **No interruptions.** No life was interrupted or resumed (the receipts' `resumes` are empty).
- **Cost.** The receipts give 3.411, 3.412, 3.411 and 3.359 CPU-hours, 13.59 in all. Check 6 forecast 13.85.
- **Unchanged code.** The runner and the scorer were not changed. The new scripts are `smallworld/l2_san.py`, `l2_san_checks.py`, `l2_san_cut.sh`, `l2_san_restart.sh` and `l2_san_report.py`.
- **Saving.** The latest state of each running life was committed and pushed every 25 minutes. To keep the disk from filling, superseded states and the states of the check lives were deleted from the disk. The latter were never committed; their hashes are in the check receipts.

## 3. Checks before the lives

The receipt is `sweep_after_naive/checks/E60_CHECKS.json`, written by `python3 smallworld/l2_san_checks.py report`. All six checks passed.

1. **Sources.** The twelve source files and the four unchanged files have the SHA-256 recorded in `sweep/checks/E44_L2_SWEEP_CHECKS.json`. The receipt lists the SHA-256 of the new scripts as they stood when it was written. `l2_san_restart.sh` and `l2_san_report.py` were written after the receipt and are not in it.
2. **Admission and refusal.** In each of worlds 1 to 4, a one-visit life of `sweep-after-naive` with the birth of rule B started. The object `learner` of its receipt equals, key for key, that of the receipt of `naive` of the same world on `origin/claude/e44-l2-first-long-lives` (at `45e9ba16`). The same command for world 1 with the sweep's birth was refused by the runner: `AssertionError: arm sweep-after-naive lives from E41's rule B, not S`, exit 1 (`checks/c2_refused.txt`). While that refused start ran, for a few seconds, three check lives were running, so four processes ran at once for that time.
3. **Calibration.** World 1's eight-visit calibration life in turn gives all six hashes of E44's receipt `exp45/e44/F1/lw_sweep-after-naive_w1_v8_turn.json`: the arrays of the event and list files, the weights, the optimizer, the learned batches and the stream.
4. **The same start as `naive`.** In each world, the 2,000 rows of the birth family answered before visit 0 equal those of `naive`'s list file in task, a, b, g1, g2, g3 and correct. The numbers right are 1,893, 1,885, 1,860 and 1,893. `by_visit[0].stream_sha256` is the same in both.
5. **Resumption.** The life of check 5 was killed with signal 9 after its log printed the line of visit 5 and before the line of visit 6 (`checks/c5_cut_kill.txt`). It was continued from the state after 4 visits by `smallworld/l2_san_cut.sh`. It gives the whole run's final hashes of the weights and the optimizer, the arrays hashes and file SHA-256 of the event and list files, and `learned_sha256`.
6. **Cost.** 12.06 CPU-seconds a visit, alone on the host. The forecast for the four lives was 13.85 CPU-hours, against the brief's limit of 19.

## 4. The lives against `naive`

Each new life's receipt has, compared with the receipt of `naive` of the same world (`final/E60_TABLES.md`, section "The new lives against `naive`"):

- the same object `learner`, key for key;
- the same stream digest at each of its 1,024 visits;
- the same stream digest over the whole life.

The two lives in each world therefore start from the same weights and see the same items in the same order; they differ in the rule of the life alone. The scores at birth (E41's receipts' final rows; A is per cent right on held-out add and sub items, R per cent right on held-out reduction pairs) are:

| Seed | Birth of rule B: A / R | Birth of rule S: A / R |
|---|---|---|
| 73129 | 94.50 / 94.65 | 95.83 / 91.95 |
| 93683 | 96.17 / 94.25 | 93.50 / 92.55 |
| 116089 | 96.17 / 93.00 | 92.83 / 90.90 |

## 5. Results

### 5.1 Where the figures come from

All figures are from `sweep_after_naive/final/E60_TABLES.md` and `E60.json`, written by `python3 smallworld/l2_san_report.py final`. That script imports `l2_score` and uses its `quantities`, `interval90` and `verdict` only.

- **The new lives.** The score files `sweep_after_naive/scores/lw_sweep-after-naive_w<w>_v1024_V<V>.json` hold every quantity of the scorer for each life (D2 to D10, D14, D15), written by `l2_san_report.py scores`.
- **`naive` and `sweep`.** Their figures are the score files of commit `738e2d61` (`sweep/main/` and `sweep/scores/`), read with `git show`.
- **Rounding.** Every figure is rounded once from the unrounded value.
- **Per-visit curves.** These are in `final/curves_w<w>.csv`.

### 5.2 Levels

| Arm | Age | World 1 | World 2 | World 3 | World 4 |
|---|---|---|---|---|---|
| `naive` | first 64 visits | 0.782 | 0.717 | 0.633 | 0.759 |
| `naive` | 256 visits | 0.582 | 0.696 | 0.692 | 0.685 |
| `naive` | 1,024 visits | 0.669 | 0.756 | 0.742 | 0.732 |
| `sweep-after-naive` | first 64 visits | 0.689 | 0.574 | 0.521 | 0.648 |
| `sweep-after-naive` | 256 visits | 0.486 | 0.546 | 0.633 | 0.525 |
| `sweep-after-naive` | 1,024 visits | 0.631 | 0.632 | 0.706 | 0.658 |
| `sweep` | first 64 visits | 0.772 | 0.620 | 0.639 | 0.713 |
| `sweep` | 256 visits | 0.589 | 0.512 | 0.611 | 0.599 |
| `sweep` | 1,024 visits | 0.650 | 0.628 | 0.682 | 0.672 |

### 5.3 The nine cells

All readings hold at the learning rate 0.001 only. The readings over worlds 1 to 4 carry the limit on shared births. There are three pairs:

- **R**, `sweep-after-naive` − `naive`: the same birth, the rules differ.
- **B**, `sweep` − `sweep-after-naive`: the same rule of the life, the births differ.
- **RB**, `sweep` − `naive`: both differ.

In every world and at every age RB = R + B exactly (`E60.json`, `RB_equals_R_plus_B_largest_deviation` = 0.0).

| Pair | Age | World 1 | World 2 | World 3 | World 4 | Over worlds 1 to 4 | Over worlds 1 to 3 |
|---|---|---|---|---|---|---|---|
| R | first 64 visits | −0.0936 | −0.1431 | −0.1115 | −0.1118 | −0.115 [−0.139, −0.091], lower | −0.116 [−0.158, −0.074], lower |
| R | 256 visits | −0.0951 | −0.1505 | −0.0591 | −0.1591 | −0.116 [−0.172, −0.060], lower | −0.102 [−0.179, −0.024], lower |
| R | 1,024 visits | −0.0381 | −0.1238 | −0.0355 | −0.0735 | −0.068 [−0.116, −0.019], lower | −0.066 [−0.150, +0.019], not distinguished |
| B | first 64 visits | +0.0836 | +0.0453 | +0.1179 | +0.0656 | +0.078 [+0.042, +0.114], higher | +0.082 [+0.021, +0.143], higher |
| B | 256 visits | +0.1028 | −0.0337 | −0.0219 | +0.0738 | +0.030 [−0.050, +0.111], not distinguished | +0.016 [−0.112, +0.143], not distinguished |
| B | 1,024 visits | +0.0191 | −0.0042 | −0.0235 | +0.0144 | +0.001 [−0.021, +0.024], not distinguished | −0.003 [−0.039, +0.033], not distinguished |
| RB | first 64 visits | −0.0100 | −0.0977 | +0.0064 | −0.0461 | −0.037 [−0.091, +0.017], not distinguished | −0.034 [−0.128, +0.061], not distinguished |
| RB | 256 visits | +0.0076 | −0.1842 | −0.0810 | −0.0853 | −0.086 [−0.178, +0.007], not distinguished | −0.086 [−0.248, +0.076], not distinguished |
| RB | 1,024 visits | −0.0191 | −0.1279 | −0.0590 | −0.0592 | −0.066 [−0.120, −0.013], lower | −0.069 [−0.162, +0.024], not distinguished |

### 5.4 Pair R: the score trend, the change, and D13

| Quantity | World 1 | World 2 | World 3 | World 4 | Over worlds 1 to 4 | Over worlds 1 to 3 |
|---|---|---|---|---|---|---|
| Difference of D5 at 1,024 visits | +0.0286 | +0.0265 | −0.0092 | +0.4664 | +0.128 [−0.138, +0.394], not distinguished | +0.015 [−0.020, +0.051], not distinguished |
| Change: R at 1,024 visits less R over the first 64 visits | +0.0555 | +0.0193 | +0.0760 | +0.0382 | +0.047 [+0.019, +0.076], higher | +0.050 [+0.002, +0.099], higher |

A change above zero means the difference moved in favour of `sweep-after-naive`.

In world 4 the score trend of `sweep-after-naive` over the second half is +0.460. Its curve falls from 0.648 (visits 0 to 63) to 0.421 (visits 576 to 639) and then rises to 0.675 (visits 960 to 1,023). The other three worlds have slopes of +0.019 to +0.068. That one world sets the four-world mean of the D5 difference.

**D13 for `sweep-after-naive` at 1,024 visits:**

- Over worlds 1 to 4: score trend +0.143 [−0.107, +0.393], trend of learning +0.137 [+0.115, +0.160]. It is **still learning**, and no trend is declining.
- Over worlds 1 to 3: score trend +0.037 [−0.009, +0.083], trend of learning +0.144 [+0.115, +0.173]. It is **still learning**.

### 5.5 The three fixed sentences

**Over worlds 1 to 4**, at the rate 0.001. Worlds 1 and 4 start from the same births, so the four values are not those of four independent learners.

1. From the same birth, the life under the sweep is **lower than** the life under back-propagation over the first 64 visits (−0.115 [−0.139, −0.091]). It is **lower than** it at 256 visits (−0.116 [−0.172, −0.060]) and **lower than** it at 1,024 visits (−0.068 [−0.116, −0.019]).
2. In 4 of the four worlds the difference `sweep-after-naive` − `naive` is nearer to zero at 1,024 visits than over the first 64 visits; in 0 it changed sign, and in 0 it is not nearer to zero. Over these worlds the change (the difference at 1,024 visits less the difference over the first 64 visits) is **higher** (+0.047 [+0.019, +0.076]). In every world e < l < 0, so the deficit of `sweep-after-naive` became smaller in worlds 1, 2, 3 and 4. **Overtaking is not said**, because pair R is not *higher* at 1,024 visits.
3. Of the difference between the arms `sweep` and `naive` at 1,024 visits (RB, −0.066 [−0.120, −0.013], lower), the part with the birth held equal is R (−0.068 [−0.116, −0.019], lower). The part with the rule held equal is B (+0.001 [−0.021, +0.024], not distinguished). **RB is assigned to the rule**, because RB is distinguished, R is distinguished on the same side and B is not distinguished. This reading counts the births of seed 73129 twice.

**Over worlds 1 to 3**, at the rate 0.001:

1. From the same birth, the life under the sweep is **lower than** the life under back-propagation over the first 64 visits (−0.116 [−0.158, −0.074]). It is **lower than** it at 256 visits (−0.102 [−0.179, −0.024]) and **not distinguished from** it at 1,024 visits (−0.066 [−0.150, +0.019]).
2. In 3 of the three worlds the difference `sweep-after-naive` − `naive` is nearer to zero at 1,024 visits than over the first 64 visits; in 0 it changed sign, and in 0 it is not nearer to zero. Over these worlds the change is **higher** (+0.050 [+0.002, +0.099]). The deficit of `sweep-after-naive` became smaller in worlds 1, 2 and 3. **Overtaking is not said.**
3. Of the difference between the arms `sweep` and `naive` at 1,024 visits (RB, −0.069 [−0.162, +0.024], not distinguished), the part with the birth held equal is R (−0.066 [−0.150, +0.019], not distinguished). The part with the rule held equal is B (−0.003 [−0.039, +0.033], not distinguished). **Nothing is assigned**, because RB is not distinguished.

**Where the two readings differ.** Pair R at 1,024 visits, and so sentence 1's last clause and sentence 3's assignment, differ between the readings:

- Over worlds 1 to 4, pair R is *lower* and RB is assigned to the rule.
- Over worlds 1 to 3, pair R is *not distinguished* and nothing is assigned.

In both readings, the gap that the arm `sweep` shows against `naive` at 1,024 visits is about the size of pair R. Pair B (the birth) is near zero at 1,024 visits (+0.001 and −0.003), although it is *higher* over the first 64 visits in both readings.

### 5.6 Keeping (D8, D9, D10, D15)

The figures below are read from the score files at V = 1,024. Without a store, `sweep-after-naive` keeps as little as the other two arms.

- **Kept at return (D8).** After an absence of 33 visits or more, the share right on the first update of a return is 0.007 to 0.012 in the four worlds, against 0.005 to 0.015 under `naive`.
- **Kept on the list (D9).** For the same absences, the mean ratio is 0.013 to 0.020, against 0.008 to 0.012 under `naive`.
- **First 16 updates of such a return (D10).** The share right is 0.071 to 0.093, against 0.093 to 0.115 under `naive`.
- **D15.** Pooled over the four worlds, kept at return after 33 visits or more does not rise with the number of earlier visits to the family. By the five classes of earlier visits it is 0.005, 0.013, 0.006, 0.012 and 0.011, with 56, 102, 170, 237 and 434 returns.

D14 (what the wrong answers are) is in each score file under `wrong_answers_stream` and `wrong_answers_list`.

## 6. Predictions (the reviewer's, on file before any life)

The predictions are scored over worlds 1 to 3, as decision 25 rules. The outcomes over worlds 1 to 4 are given beside them, with the limit on shared births. All hold at the rate 0.001 only.

| | Statement | Chance (worlds 1 to 3) | Outcome, worlds 1 to 3 | Chance (worlds 1 to 4) | Outcome, worlds 1 to 4 |
|---|---|---|---|---|---|
| P60.1 | Pair R over the first 64 visits is *lower* | 75% | **Yes** (−0.116 [−0.158, −0.074]) | 80% | **Yes** (−0.115 [−0.139, −0.091]) |
| P60.2 | Pair R at 1,024 visits is *lower* | 45% (*n.d.* 50%, *higher* 5%) | **No**, *not distinguished* (−0.066 [−0.150, +0.019]) | 55% | **Yes** (−0.068 [−0.116, −0.019]) |
| P60.3 | In all the worlds of the reading the difference of pair R is *nearer to zero* at 1,024 visits | 60% | **Yes** (3 of 3) | 50% | **Yes** (4 of 4) |
| P60.4 | Pair B at 1,024 visits is *not distinguished* | 60% | **Yes** (−0.003 [−0.039, +0.033]) | 55% | **Yes** (+0.001 [−0.021, +0.024]) |
| P60.5 | `sweep-after-naive` is *still learning* at 1,024 visits | 85% | **Yes** | 90% | **Yes** |
| P60.6 | Checks 1 to 5 pass | 85% | **Yes** | (the same) | **Yes** |

## 7. Limits

- **Shared births.** There are four worlds on three birth seeds. Worlds 1 and 4 share their births, so the reading over four worlds counts one pair of births twice. Over worlds 1 to 3 the interval is the mean ± 1.686 standard deviations, and with it neither RB nor R is distinguished at 1,024 visits.
- **Pair R is not a comparison on equal terms.** The birth was formed by back-propagation, and the sweep starts below `naive` on it: pair R over the first 64 visits is −0.115. The reverse arm, a life by back-propagation from the sweep's birth, is not run, because the runner has no such arm (decision 26). The design has three of the four cells of rule and birth.
- **One setting each.**
  - Learning rate: 0.001 only.
  - Visit length: 256 updates, calibrated for `naive`.
  - No store.
  - Lives of 1,024 visits; nothing is said beyond them.
- **Different kinds of visit.** The first 64 visits are mostly continuations (23 to 47 of 64 in worlds 1 to 4), and the late window of 1,024 visits is mostly returns (218 to 223 of 256). So the change in sentence 2 compares a difference measured on continuations with a difference measured on returns.
- **Fresh optimizer.** The optimizer is fresh at the start of every life, so the first updates of each life are not those of a learner that continues.
- **World 4.** World 4's score trend for `sweep-after-naive` (+0.460) comes from a dip and a recovery within its second half. It alone sets the four-world D5 difference.

## 8. Committed files

The list below is from `git ls-tree -r --name-only HEAD MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep_after_naive MOD97_LOCAL_RULE_20261106/smallworld`, restricted to `sweep_after_naive/` and `smallworld/l2_san*`. It was taken at the commit that adds this report and appended in the next commit, which changes only this file.
