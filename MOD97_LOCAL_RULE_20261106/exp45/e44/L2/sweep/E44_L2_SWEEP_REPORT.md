# E44, job L2: the long lives of the sweep, beside release 1 (report, 18 November 2026)

**Branch** `claude/e44-l2-sweep-long-lives`, which starts from the main worker's branch `claude/e44-l2-first-long-lives` at commit `b218f351`. **Instructions:** `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/WORKER_E44_L2_SWEEP_LONG_LIVES.md` on `claude/vibrant-cerf-sje726`. They were written by the outside reviewer, who coordinated the programme in the owner's absence. **Definitions** D1 to D15 are those of the roster note `OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md`, revision 2. **Evidence level:** E2 (four worlds, one birth per rule and seed). No formal checker gate was run. The outside reviewer verified the report at commit `3879225d` from the event files, list files, receipts and states (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/E44_L2_SWEEP_VERIFICATION_20261118.md` with `e44_l2_sweep_verify.py`, on `claude/vibrant-cerf-sje726` at `b3865b5e`). This revision applies the corrections the reviewer asked for (its message of 18 November, 10:47 UTC). They change no run, receipt, score file or table.

The question comes from the owner (15 November): a learner may be slower than the conventional learner at first and overtake it as learning continues. These eight lives test that at 256 and 1,024 visits, for the learning rule called the sweep against back-propagation. They say nothing beyond 1,024 visits.

**Two limits hold for every reading below.**

- **Arms, not rules.** "The sweep" in a reading means an arm: `sweep` (or `sweep+replay`), that is, E41's birth made under the sweep and then trained with the sweep. "Back-propagation" means the arm `naive` (or `replay`): back-propagation's birth trained with back-propagation. The two arms of a pair differ in the birth as well as in the rule. The sweep's births answer R = 91.95, 92.55 and 90.90 per cent at seeds 73129, 93683 and 116089, against 94.65, 94.25 and 93.00 for back-propagation's (section 4). No difference in this report is assigned to the rule alone, at any age.
- **One learning rate.** Every reading holds at the learning rate 0.001 (and weight decay 0.25) only. This rate is the one every experiment has used. In the revision of the instructions at `9eda3569`, the reviewer states that it was taken from the literature on such tasks, given to both rules and never varied for either. Job E52 is testing other rates (section 7).

## 1. Terms

- **The sweep** is the learning rule that passes the error backward by the signs of the weights. In the runner it is `local=True`, used by the arms `sweep` and `sweep+replay`.
- **Back-propagation** is the exact gradient, used by the arms `naive` and `replay`. The lives of `naive` and `replay` are the main worker's (release 1). They are read from its branch.
- A **visit** is 256 updates on one family of tasks. A **life** is a schedule of visits in a **world**: a seeded generator of families and of the order in which they are visited (`smallworld/lw_world.py`).
- The **store** is a replay buffer as large as the network: 55,209 items of 23 bytes. In an arm with the store, half of each update's 48 learned items (ρ = 0.5) are drawn from it.
- A **reduction item** is an asked item of add or sub whose answer needs the reduction modulo the family's modulus (D1).
- The **late level** is the mean share right on reduction items over the last quarter of the visits (D4).
- The **score trend** is the slope of the score by age on ln(v + 1), taken over the returns in the second half of the life (D5).
- The **trend of learning** is the slope, on ln n, of the level each family reaches at the end of its first visit (D7). Here n is the family's place in the order of first meeting.
- **Kept at return** is the share right on the first update of a later visit, before any learning in that visit (D8).
- A later visit to a family is a **continuation** if it follows the family's previous visit directly (an absence of 1 visit), and a **return** if at least one other family's visit lies between them (an absence of 2 or more visits) (D3).
- The **blind predictor** answers every asked item of a task with the answer of the last item of that task that was learned, without reading the operands. The event file's `blind` column records whether that answer would have been right (`sw_run.py`, the line that sets `last_answer`). Its late level is the same mean as D4 over this column, a chance line for D2.
- An arm is **stuck** at a length if neither its score trend (D5) nor its trend of learning (D7) has a 90% interval wholly above zero. It is **still learning** if at least one of them has. A trend is **declining** if its interval lies wholly below zero (D13). "Stuck" therefore means only that no rising trend was found at this precision.
- A comparison of two arms is **closed** at 1,024 visits if the late-level difference is *higher* at both 256 and 1,024 visits, or *lower* at both, and if at 1,024 visits neither the difference in D5 nor the difference in D7 is distinguished on the opposite side. Otherwise it is **open** (D12).
- An interval is the two-sided 90% interval of the mean over the four worlds (Student's t with three degrees of freedom: the mean ± 1.18 standard deviations). A difference is **higher** or **lower** when the interval excludes zero, and otherwise **not distinguished** (D11). In the fixed sentences, *behind* and *ahead* stand for *lower* and *higher* in late level, *steeper* and *flatter* for *higher* and *lower* in the score trend, and *level* and *the same* for *not distinguished*. *Level*, *the same*, *not changed* and *not seen* never mean that the arms were shown to be equal. They mean only that an interval includes zero, and the interval is given wherever the word is used. *Ahead* and *behind* are used only for the reading over the four worlds, never for a single world.

## 2. Runs

| Arm | Worlds | Visits | Birth |
|---|---|---|---|
| `sweep` | 1, 2, 3, 4 | 1,024 | E41's half-width birth under the sweep, `exp45/e41/e41_S_h256_f50_<seed>.pt` |
| `sweep+replay` (ρ = 0.5) | 1, 2, 3, 4 | 1,024 | the same |

The seeds are 73129 for worlds 1 and 4, 93683 for world 2 and 116089 for world 3. Every setting is release 1's: `--world lw --H 256 --U 128 --visit-updates 256 --wd 0.25 --lr 0.001 --list-every 8 --decay-distances none --save-states 64,128,256,384,512,640,768,896,1024`. The jobs are listed in `sweep_jobs.json`, written by `smallworld/l2_sweep.py jobs`. They ran under the main worker's job driver `smallworld/l2_driver.py`, unchanged, with at most three processes at once and one thread each. The host was Linux x86-64 with Python 3.13.18, torch 2.15.0+cpu and numpy 2.5.3. Every receipt records these versions.

The runner `smallworld/sw_run.py` (SHA-256 `41714c5b…`) and the scorer `smallworld/l2_score.py` (SHA-256 `6969417e…`, as at `b218f351`) were not changed. The new code is in new files:

- `l2_sweep.py`: the jobs and the saving to the branch.
- `l2_sweep_checks.py`: the checks.
- `l2_sweep_report.py`: the scoring and the tables.
- `l2_sweep_restart.sh`: the restart script.

**Interruptions.** The machine stopped twice while lives ran. The first stop came at about 21:50 UTC on 17 November, when the session was idle. The second came at about 00:37 UTC on 18 November, while the session was held open; the host's kernel version changed, which suggests the machine was moved. At each stop the three running lives were continued with `life --resume` from their latest state:

- World 1 under both arms and world 2 under `sweep`: from visit 64 at 22:12 UTC, and from visit 640 at 00:38 UTC.
- The other five lives ran without interruption.

Each life's `.starts.jsonl` and the `resumes` field of its receipt record these continuations. Check 4 (section 3) shows that resumption is exact only in a life of 8 visits cut at visit 4, with other list settings than the long lives and a store that fills during that visit. It does not cover the long lives. The long lives have their own evidence. Their logs print the visits computed twice: 104 visits in `sweep` world 1, 100 in `sweep+replay` world 1 and 106 in `sweep` world 2. Each of these visits prints the same share right and loss both times. The event files are continuous, with 1,024 visits of 256 updates each. The reviewer's verification also finds the resumed lives continued from the states their receipts name. The CPU-hours in the receipts are those of the visits kept. The visits that were lost and computed again are not counted: about 24 visits after the first stop and about 78 after the second, for each of the three lives.

**Cost.** The receipts give 29.86 CPU-hours for the eight lives, from 3.65 to 3.80 CPU-hours a life (`final/E44_L2_SWEEP_TABLES.md`, last section). Check 6 forecast 30.1 CPU-hours.

## 3. Checks before the lives

The receipt is `checks/E44_L2_SWEEP_CHECKS.json`, written by `python3 smallworld/l2_sweep_checks.py report`. All six checks passed.

1. All twelve source hashes of `sw_run.py` and of its imports equal those under `sources_sha256` in the main worker's receipt `exp45/e44/L2/checks/E44_L2_CHECKS.json`. The scorer, the driver and `sw_score.py` equal their files at `b218f351`.
2. World 1's eight-visit calibration life under `sweep` (in turn, 256 updates a visit, the defaults of the two new arguments) gave on this host all six hashes of E44's receipt `exp45/e44/F1/lw_sweep_w1_v8_turn.json`. The six hashes are those of the list, the events, the final weights, the optimizer state, the learned batches and the stream. Its event and list columns are equal item by item; 25,124 of 32,768 asked items were answered right in both.
3. `sweep+replay` with ρ = 0 equals `sweep` on world 1 for one visit: the same hashes and event file. It loads the sweep's birth.
4. A `sweep+replay` life of world 1, stopped at visit 4 of 8 and resumed, gives the hashes, event file and list file of the life run without interruption. The store's capacity is 55,209 items and is never exceeded.
5. The three births under the sweep have file and tensor hashes equal to their E41 receipts. Through the runner, A and R (per cent right on E41's held-out panels) equal the receipts' final rows:

   | Seed | A | R |
   |---|---|---|
   | 73129 | 95.83 | 91.95 |
   | 93683 | 93.50 | 92.55 |
   | 116089 | 92.83 | 90.90 |

6. Cost, each arm alone on the host for 4 visits of world 1: `sweep` took 12.98 CPU-seconds a visit and `sweep+replay` 12.92. The main worker measured 11.89 for `naive` and 11.81 for `replay`. The forecast was 3.77 CPU-hours a life and 30.1 for the eight lives.

## 4. The births

The two rules start from different births. Each life starts from E41's birth made under its own rule. At birth, before visit 0, the sweep's lives answered 91.9% (seed 73129), 92.5% (93683) and 90.9% (116089) of the birth family's 2,000 reduction items right. E41's receipts give the following for A and R:

| Seed | Back-propagation A / R | Sweep A / R |
|---|---|---|
| 73129 | 94.50 / 94.65 | 95.83 / 91.95 |
| 93683 | 96.17 / 94.25 | 93.50 / 92.55 |
| 116089 | 96.17 / 93.00 | 92.83 / 90.90 |

The sweep's births are 2.7 and 3.3 points lower in A at seeds 93683 and 116089 (1.3 points higher at 73129), and 1.7 to 2.7 points lower in R at every seed. These figures are in `final/E44_L2_SWEEP_TABLES.md`, section "The births", and in `final/E44_L2_SWEEP.json` under `births`.

## 5. Results

All figures below come from `final/E44_L2_SWEEP_TABLES.md` and `final/E44_L2_SWEEP.json`. They were written by `python3 smallworld/l2_sweep_report.py tables final` from:

- the score files `scores/<life>_V<visits>.json`, written by `l2_sweep_report.py scores`;
- `main/<life>_V<visits>.json`, written by `l2_sweep_report.py main`. These score the main worker's committed event and list files with the same scorer. The files were read from `claude/e44-l2-first-long-lives` at commits `56e77d86` (world 1 and `naive` world 2), `a594f1cd` (`naive` world 3 and `replay` worlds 2 and 3) and `f997d701` (world 4).

Each score file records the commit and the file hashes it read.

### 5.1 Late level (D4)

| Visits | Arm | World 1 | World 2 | World 3 | World 4 |
|---|---|---|---|---|---|
| 256 | `sweep` | 0.589 | 0.512 | 0.611 | 0.599 |
| 256 | `naive` | 0.582 | 0.696 | 0.692 | 0.685 |
| 256 | `sweep+replay` | 0.665 | 0.525 | 0.588 | 0.669 |
| 256 | `replay` | 0.577 | 0.755 | 0.707 | 0.659 |
| 1,024 | `sweep` | 0.650 | 0.628 | 0.682 | 0.672 |
| 1,024 | `naive` | 0.669 | 0.756 | 0.742 | 0.732 |
| 1,024 | `sweep+replay` | 0.784 | 0.574 | 0.789 | 0.769 |
| 1,024 | `replay` | 0.730 | 0.812 | 0.835 | 0.709 |

The blind predictor's late level is 0.014 to 0.016 in every life.

### 5.2 Between the rules (D11, D12)

Each entry is the sweep's figure minus back-propagation's: the mean over the four worlds, with its 90% interval.

| Pair | Visits | Late level | Kept at return, 5–32 visits away | Kept at return, 33 or more | Score trend | Trend of learning |
|---|---|---|---|---|---|---|
| `sweep` − `naive` | 256 | −0.086 [−0.178, +0.007] n.d. | −0.001 [−0.019, +0.018] n.d. | −0.004 [−0.021, +0.013] n.d. | −0.050 [−0.111, +0.011] n.d. | −0.031 [−0.053, −0.009] **lower** |
| `sweep` − `naive` | 1,024 | −0.066 [−0.120, −0.013] **lower** | +0.001 [−0.009, +0.010] n.d. | +0.004 [−0.005, +0.012] n.d. | +0.036 [−0.039, +0.111] n.d. | −0.018 [−0.046, +0.011] n.d. |
| `sweep+replay` − `replay` | 256 | −0.063 [−0.228, +0.103] n.d. | −0.084 [−0.172, +0.004] n.d. | −0.086 [−0.236, +0.064] n.d. | −0.002 [−0.109, +0.104] n.d. | +0.003 [−0.098, +0.105] n.d. |
| `sweep+replay` − `replay` | 1,024 | −0.042 [−0.206, +0.121] n.d. | −0.088 [−0.236, +0.059] n.d. | −0.083 [−0.242, +0.076] n.d. | −0.004 [−0.059, +0.051] n.d. | +0.000 [−0.070, +0.070] n.d. |

(n.d.: not distinguished.)

**The fixed sentences** (instructions, section 6). Each reads an arm against an arm, birth and rule together, at the learning rate 0.001 only.

- **Without the store** (`sweep` against `naive`): over 1,024 visits the arm `sweep` is **behind** in late level, by −0.066 [−0.120, −0.013]. Its score trend over the second half is **the same as** `naive`'s; that is, the difference, +0.036 [−0.039, +0.111], is not distinguished. *Behind* is the only reading of the main comparisons whose interval excludes zero, and it treats worlds 1 and 4, which share their births (seed 73129), as two replications. If those two worlds are averaged into one birth, the interval over the three births is −0.075 [−0.154, +0.003], which includes zero. That figure uses t with 2 degrees of freedom; it is a sensitivity reading, not the roster note's definition. In each of the four worlds `sweep`'s late level is below `naive`'s at 1,024 visits, by 0.019 to 0.128. At 256 visits the arm `sweep` is **level**: −0.086 [−0.178, +0.007].
- **With the store** (`sweep+replay` against `replay`): over 1,024 visits the arm `sweep+replay` is **level** in late level: −0.042 [−0.206, +0.121], an interval 0.33 wide. Its score trend over the second half is **the same as** `replay`'s: −0.004 [−0.059, +0.051]. Both differences change sign across the worlds. World 2 (seed 93683) carries most of the negative mean in late level (−0.238 at 1,024 visits). In worlds 1 and 4 `sweep+replay`'s late level is above `replay`'s, by 0.054 and 0.061.

**Third clause** (the reviewer's addition, section 5.7): with and without the store, the change of the difference from the first 64 visits to the late window of 1,024 visits is **not distinguished**. Without the store it is −0.029 [−0.060, +0.001]; with the store, +0.000 [−0.138, +0.139].

**Overtaking in time.** This would show as a steeper score trend together with a late level that is behind or level. It is **not seen** up to 1,024 visits, at the rate 0.001, with or without the store. *Not seen* means that no steeper trend and no grown difference was distinguished; it does not exclude overtaking. The trend difference without the store is +0.036 [−0.039, +0.111], and every arm is still learning at 1,024 visits. Without the store, the arm `sweep`'s score trend in the second half is steeper than `naive`'s in three of the four worlds: by 0.073 (world 1), 0.056 (world 2) and 0.073 (world 4). In world 3 it is less steep, by 0.059. Over the four worlds the difference is not distinguished. The sweep's trend of learning is lower than back-propagation's at 256 visits without the store. At 1,024 visits it is not distinguished.

**Closed (D12).** Neither comparison is closed at 1,024 visits. Without the store, the late-level difference is *not distinguished* at 256 visits and *lower* at 1,024. With the store, it is *not distinguished* at both lengths.

**The curves.** Section "The curves of score by age" of the tables gives the mean score by blocks of 64 visits; `final/curves_w<world>.csv` gives it per visit. The table below lists the blocks in which the sweep's mean is at or above back-propagation's (blocks numbered 0 to 15).

| World | `sweep` ≥ `naive` | `sweep+replay` ≥ `replay` |
|---|---|---|
| 1 | blocks 1 to 5, 14, 15 | blocks 1 to 15 |
| 2 | none | none |
| 3 | block 0 only | block 0 only |
| 4 | none | blocks 1 to 15 |

So the curves do not cross in a common direction across the worlds. In world 1 the sweep's curve falls below `naive`'s after block 5 and returns to it in the last two blocks (0.649 against 0.640 over visits 960 to 1,023). In world 2 the curves of the two rules do not meet under either arm.

### 5.3 Stuck or still learning (D13)

Each entry is the mean over the four worlds, with its 90% interval.

| Visits | Arm | Score trend | Trend of learning | Statement |
|---|---|---|---|---|
| 256 | `sweep` | −0.042 [−0.153, +0.069] | −0.018 [−0.079, +0.043] | stuck |
| 256 | `sweep+replay` | +0.030 [−0.002, +0.061] | −0.021 [−0.093, +0.051] | stuck |
| 1,024 | `sweep` | +0.050 [−0.012, +0.113] | +0.081 [+0.056, +0.106] | still learning |
| 1,024 | `sweep+replay` | +0.048 [+0.036, +0.061] | +0.108 [+0.067, +0.149] | still learning |
| 1,024 | `naive` (for comparison) | +0.015 [−0.016, +0.046] | +0.098 [+0.083, +0.114] | still learning |
| 1,024 | `replay` (for comparison) | +0.052 [−0.005, +0.109] | +0.108 [+0.078, +0.137] | still learning |

At 256 visits, "stuck" means only that no trend was found at this precision; no trend is *declining*. The statements at 256 visits rest on four worlds, not on the eight that the roster note names for release 1's arms.

### 5.4 Keeping (D8, D9, D10)

**Without the store, almost nothing is kept under either rule.** Kept at return after an absence of 33 or more visits, up to 1,024 visits, is 0.009 to 0.020 for `sweep` and 0.005 to 0.015 for `naive` in the four worlds. On the list (D9), the mean ratio of the share right to the share at the family's last visit, for absences of 33 or more, is 0.014 to 0.018 under `sweep` and 0.008 to 0.012 under `naive`.

The first 16 updates of such a return reach 0.079 to 0.104 under `sweep` and 0.093 to 0.115 under `naive` (D10). The last 64 updates reach 0.687 to 0.775 under `sweep` and 0.743 to 0.836 under `naive`.

**With the store, both rules keep about half.** Kept at return after 33 or more visits is 0.34 to 0.51 under `sweep+replay` and 0.45 to 0.60 under `replay`. `sweep+replay` is *higher* than `sweep` by +0.432 [+0.342, +0.522] at 1,024 visits.

### 5.5 What the wrong answers are (D14)

These figures are pooled over the four worlds, at returns after 33 or more visits, up to 1,024 visits, on the stream (3,812 answers per arm).

| Arm | Right | Answer under the modulus of the family taught last | Answer under the modulus of an earlier family | Another two-digit number | Chance line |
|---|---|---|---|---|---|
| `sweep` | 0.014 | 0.421 | 0.219 | 0.201 | 0.008 |
| `naive` | 0.010 | 0.519 | 0.172 | 0.145 | 0.007 |
| `sweep+replay` | 0.447 | 0.017 | 0.294 | 0.194 | 0.014 |
| `replay` | 0.526 | 0.011 | 0.228 | 0.190 | 0.007 |

Without the store, both rules mostly answer as for the family taught last. The sweep does so less often than back-propagation, and more often gives the answer of an earlier family or another number. The list's rows show the same pattern (tables, section D14).

### 5.6 Does a memory become more lasting as a family is met again? (D15)

Kept at return, pooled over the four worlds up to 1,024 visits, by the number of earlier visits to the family (1; 2 to 3; 4 to 7; 8 to 15; 16 or more):

| Arm | Absence (visits) | 1 | 2 to 3 | 4 to 7 | 8 to 15 | 16 or more |
|---|---|---|---|---|---|---|
| `sweep` | 33 or more | 0.015 | 0.018 | 0.009 | 0.014 | 0.015 |
| `sweep+replay` | 2 to 4 | 0.106 | 0.291 | 0.417 | 0.553 | 0.657 |
| `sweep+replay` | 5 to 32 | 0.054 | 0.178 | 0.298 | 0.497 | 0.602 |
| `sweep+replay` | 33 or more | 0.070 | 0.165 | 0.320 | 0.458 | 0.598 |

The `sweep+replay` cells hold 54 to 533 returns each.

**`sweep`:** in the class of absences of 33 visits or more, neither quantity rises with the number of earlier visits. Kept at return stays between 0.009 and 0.018, and the mean share over the first 16 updates is 0.077, 0.096, 0.072, 0.074 and 0.094 across the classes of earlier visits. The statement is not made for the other classes. In the class of 2 to 4 visits, for example, the share over the first 16 updates goes 0.084, 0.115, 0.108, 0.106 and 0.134.

**`sweep+replay`:** both quantities rise with the number of earlier visits in every class of absence. For absences of 33 or more, kept at return goes from 0.070 to 0.598, and the share over the first 16 updates from 0.262 to 0.642. Under `replay`, in the class of 33 visits or more, kept at return rises from 0.115 to 0.658, and the share over the first 16 updates from 0.337 to 0.696. A family met more often has more of its items in the store, so this table cannot tell a memory that lasts longer by itself from more rehearsal.

### 5.7 Early level and the change of the difference (the reviewer's addition)

The reviewer added this reading on 17 November at 22:35 UTC, after the lives had started; no run changed for it. The **early level** of a life is the mean of D2 over its first 64 visits (visits 0 to 63). For each pair, the **change** in each world is the late difference minus the early difference, each difference being the arm of the sweep minus the arm of back-propagation. A change below zero means that the arm of the sweep lost ground relative to back-propagation's between the two windows; above zero, that it gained. Its mean and 90% interval over the four worlds are formed as in D11. The words *grown*, *not changed* and *shrunk*, and the requirement to read them by that interval, are the reviewer's, from its message of 22:35 UTC on 17 November (now the appendix of the verification document). Here *not changed* is written as *not distinguished*. Like every reading in this report, this one holds at the learning rate 0.001 only. The figures are in `final/E44_L2_SWEEP_EARLY.md` and `final/E44_L2_SWEEP_EARLY.json`, written by `python3 smallworld/l2_sweep_early.py` from the same score files.

| Pair | Early difference | Late difference at 1,024 visits | Change |
|---|---|---|---|
| `sweep` − `naive` | −0.037 [−0.091, +0.017] | −0.066 [−0.120, −0.013] | −0.029 [−0.060, +0.001]: **not distinguished** |
| `sweep+replay` − `replay` | −0.043 [−0.100, +0.015] | −0.042 [−0.206, +0.121] | +0.000 [−0.138, +0.139]: **not distinguished** |

With the late window at 256 visits, the change is −0.049 [−0.107, +0.010] without the store and −0.020 [−0.168, +0.127] with it; both are *not distinguished*. **Third clause of the fixed sentences:** the change of the difference from the first 64 visits to the late window of 1,024 visits is **not distinguished**, with or without the store. With the store its interval is 0.28 wide. Without the store the change is negative in all four worlds (−0.0091, −0.0302, −0.0654, −0.0131), so the arm `sweep` lost ground in each. Over the four worlds it is not distinguished: −0.0294 [−0.0597, +0.0008].

The two windows hold different kinds of visit. In visits 0 to 63 most visits are continuations: 44, 23, 47 and 38 of 64 in worlds 1 to 4. In the late window of 1,024 visits most are returns: 218 to 223 of 256. The change therefore compares a difference measured mostly on continuations with a difference measured mostly on returns.

In these lives the sweep's early level is not above back-propagation's: the early difference is −0.037. The reviewer's figure from the calibration, by which the arm `sweep` is ahead of `naive` by +0.161 before any life, is a different quantity measured in different lives (E44_GATE_FOR_THE_SWEEP_20261117.md). The two are not compared here.

## 6. Predictions (the reviewer's, on file before any life)

| | Statement | Chance | Outcome |
|---|---|---|---|
| PS.1 | `sweep` at 1,024 visits is *stuck* | 50% | **No.** It is *still learning*: the trend of learning is +0.081 [+0.056, +0.106]. The score trend was not found. |
| PS.2 | `sweep` is *behind* `naive` in late level at 1,024 visits | 45% | **Yes**, at the rate 0.001. −0.066 [−0.120, −0.013], with worlds 1 and 4 counted as two replications although they share their births. With the two averaged into one birth: −0.075 [−0.154, +0.003] (t with 2 degrees of freedom; a sensitivity reading, not the roster note's definition). |
| PS.3 | `sweep+replay` is *higher* than `sweep` in late level at 1,024 visits | 70% | **No**, at the rate 0.001. +0.071 [−0.029, +0.171], not distinguished. `sweep+replay`'s late level exceeds `sweep`'s in three worlds; in world 2 it is lower by 0.054. |
| PS.4 | The sweep's score trend over the second half is *steeper* than back-propagation's, without the store | 15% | **No**, at the rate 0.001. +0.036 [−0.039, +0.111]: not distinguished (*the same*). |
| PS.5 | The answer of PS.2 at 256 visits is the same as at 1,024 visits | 70% | **No**, at the rate 0.001. *Level* at 256 visits (−0.086 [−0.178, +0.007], not distinguished), *behind* at 1,024 (−0.066 [−0.120, −0.013]; see PS.2 for the shared births). |

## 7. Limits

- **Learning rate.** The learning rate 0.001 is the one every experiment has used. The instructions, in their revision at `9eda3569`, state that it was taken from the literature on such tasks, given to both rules and never varied for either. (The first issue of the instructions, at `3b4bdda5`, from which this work started, said that it "was chosen for back-propagation". The report's first text repeated that phrase.) Job E52 is testing other rates for both rules. If it finds that the sweep gains from another rate at this width, the reviewer has said these lives will be asked for again at that rate. The comparison above holds only at 0.001 and weight decay 0.25.
- **Births.** The two arms of a pair start from different births, each made under its own rule. Every difference in this report, at every age, is a difference between arms (birth and rule together), not between rules.
- **Worlds and seeds.** There are four worlds and three birth seeds, and worlds 1 and 4 share a seed. The 90% intervals over four worlds are wide.
- **World 2.** World 2's `sweep+replay` life is the low outlier with the store: its median first-visit level is 0.234 at 256 visits and 0.451 at 1,024. It sets the sign of the mean difference with the store. Nothing here explains it.
- **Horizon.** Nothing is said beyond 1,024 visits. Both comparisons are open by D12, and a longer life needs a larger task space.
- **States.** The states committed are those after 256 and 1,024 visits of world 1's two lives. Every other state was removed from the branch's tree at the end. Their hashes are in the receipts under `states`, except the states after 64 and 640 visits from which three lives were resumed: those are recorded under `resumes` only. States written while lives ran stay in the branch's history.

## 8. Committed files

The list below is from `git ls-tree -r --name-only HEAD MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep MOD97_LOCAL_RULE_20261106/smallworld/l2_sweep*` at commit `184298f7`, which adds this report; the list was appended in the next commit, `87db9681`, which changes only this file. The three files of section 5.7 were added later, and the list was updated by hand to include them.

```
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/E44_L2_SWEEP_REPORT.md
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/E44_L2_SWEEP_CHECKS.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c2_calib/c2_calib.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c2_calib/c2_calib.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c2_calib/c2_calib.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c2_calib/lw_sweep_w1_v8_turn.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c2_calib/lw_sweep_w1_v8_turn.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c2_calib/lw_sweep_w1_v8_turn_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c3_sweep/c3_sweep.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c3_sweep/c3_sweep.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c3_sweep/c3_sweep.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c3_sweep/lw_sweep_w1_v1.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c3_sweep/lw_sweep_w1_v1.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c3_sweep/lw_sweep_w1_v1_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c3_sweep_replay_rho0/c3_sweep_replay_rho0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c3_sweep_replay_rho0/c3_sweep_replay_rho0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c3_sweep_replay_rho0/c3_sweep_replay_rho0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c3_sweep_replay_rho0/lw_sweep+replay_w1_v1_rho0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c3_sweep_replay_rho0/lw_sweep+replay_w1_v1_rho0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c3_sweep_replay_rho0/lw_sweep+replay_w1_v1_rho0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_cut/c4_cut.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_cut/c4_cut.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_cut/c4_cut.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_cut/c4_resumed.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_cut/c4_resumed.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_cut/c4_resumed.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_cut/lw_sweep+replay_w1_v8_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_cut/lw_sweep+replay_w1_v8_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_cut/lw_sweep+replay_w1_v8_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_whole/c4_whole.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_whole/c4_whole.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_whole/c4_whole.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_whole/lw_sweep+replay_w1_v8_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_whole/lw_sweep+replay_w1_v8_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c4_whole/lw_sweep+replay_w1_v8_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c6_sweep/c6_sweep.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c6_sweep/c6_sweep.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c6_sweep/c6_sweep.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c6_sweep/lw_sweep_w1_v4.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c6_sweep/lw_sweep_w1_v4.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c6_sweep/lw_sweep_w1_v4_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c6_sweep_replay/c6_sweep_replay.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c6_sweep_replay/c6_sweep_replay.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c6_sweep_replay/c6_sweep_replay.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c6_sweep_replay/lw_sweep+replay_w1_v4_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c6_sweep_replay/lw_sweep+replay_w1_v4_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/c6_sweep_replay/lw_sweep+replay_w1_v4_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/checks/check_jobs.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/final/E44_L2_SWEEP.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/final/E44_L2_SWEEP_EARLY.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/final/E44_L2_SWEEP_EARLY.md
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/final/E44_L2_SWEEP_TABLES.md
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/final/curves_w1.csv
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/final/curves_w2.csv
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/final/curves_w3.csv
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/final/curves_w4.csv
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/interim_world1/E44_L2_SWEEP.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/interim_world1/E44_L2_SWEEP_TABLES.md
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/interim_world1/curves_w1.csv
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/interim_world1/curves_w2.csv
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/interim_worlds123/E44_L2_SWEEP.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/interim_worlds123/E44_L2_SWEEP_TABLES.md
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/interim_worlds123/curves_w1.csv
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/interim_worlds123/curves_w2.csv
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/interim_worlds123/curves_w3.csv
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w1_v1024_rho0.5.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w1_v1024_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w1_v1024_rho0.5.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w1_v1024_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w1_v1024_rho0.5.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w1_v1024_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w1_v1024_rho0.5_state_v1024.pt
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w1_v1024_rho0.5_state_v256.pt
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w2_v1024_rho0.5.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w2_v1024_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w2_v1024_rho0.5.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w2_v1024_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w2_v1024_rho0.5.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w2_v1024_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w3_v1024_rho0.5.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w3_v1024_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w3_v1024_rho0.5.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w3_v1024_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w3_v1024_rho0.5.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w3_v1024_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w4_v1024_rho0.5.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w4_v1024_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w4_v1024_rho0.5.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w4_v1024_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w4_v1024_rho0.5.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep+replay_w4_v1024_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w1_v1024.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w1_v1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w1_v1024.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w1_v1024.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w1_v1024.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w1_v1024_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w1_v1024_state_v1024.pt
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w1_v1024_state_v256.pt
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w2_v1024.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w2_v1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w2_v1024.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w2_v1024.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w2_v1024.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w2_v1024_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w3_v1024.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w3_v1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w3_v1024.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w3_v1024.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w3_v1024.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w3_v1024_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w4_v1024.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w4_v1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w4_v1024.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w4_v1024.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w4_v1024.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/lives/lw_sweep_w4_v1024_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/logs/check_driver.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/logs/saver.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/logs/sweep_driver.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_naive_w1_v1024_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_naive_w1_v1024_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_naive_w2_v1024_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_naive_w2_v1024_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_naive_w3_v1024_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_naive_w3_v1024_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_naive_w4_v1024_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_naive_w4_v1024_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_replay_w1_v1024_rho0.5_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_replay_w1_v1024_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_replay_w2_v1024_rho0.5_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_replay_w2_v1024_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_replay_w3_v1024_rho0.5_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_replay_w3_v1024_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_replay_w4_v1024_rho0.5_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/main/lw_replay_w4_v1024_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep+replay_w1_v1024_rho0.5_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep+replay_w1_v1024_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep+replay_w2_v1024_rho0.5_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep+replay_w2_v1024_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep+replay_w3_v1024_rho0.5_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep+replay_w3_v1024_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep+replay_w4_v1024_rho0.5_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep+replay_w4_v1024_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep_w1_v1024_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep_w1_v1024_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep_w2_v1024_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep_w2_v1024_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep_w3_v1024_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep_w3_v1024_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep_w4_v1024_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/scores/lw_sweep_w4_v1024_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/sweep_jobs.json
MOD97_LOCAL_RULE_20261106/smallworld/l2_sweep.py
MOD97_LOCAL_RULE_20261106/smallworld/l2_sweep_checks.py
MOD97_LOCAL_RULE_20261106/smallworld/l2_sweep_early.py
MOD97_LOCAL_RULE_20261106/smallworld/l2_sweep_report.py
MOD97_LOCAL_RULE_20261106/smallworld/l2_sweep_restart.sh
```
