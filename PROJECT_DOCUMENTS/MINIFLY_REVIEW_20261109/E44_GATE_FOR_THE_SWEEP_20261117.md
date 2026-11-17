# The gate of E44's calibration, computed for the sweep (17 November 2026)

A read-only computation for the outside reviewer's file. Nothing was trained, no code of another branch was run, and every figure below comes from a command run on a committed file, named with its path and its ref in section 9. The script is `e44_gate_sweep.py` in this folder. It reuses by import the functions of the audit's script `e44_l2_checks.py` (this folder, unedited), which read an event file and define the gate's figure.

## 1. The question

E44's calibration chose the length of a visit, 256 updates, by a gate that the brief asked of the arm `naive` only. Long lives of 1,024 visits under `sweep` and `sweep+replay` are now running at that length. The inventory of the record lists as never done "a visit length calibrated for the `replay` arm or for the sweep, in place of the one calibrated under `naive`". Four things were asked:

1. the gate's own figure for the arm `sweep` at 64, 128 and 256 updates a visit, beside `naive`'s;
2. for both arms at 256 updates, the level within a first visit by quarter of the visit;
3. what the calibration's files hold for an arm with a store;
4. whether 256 updates meets the gate's own condition for the sweep, by how much the two arms differ, and what that means for a comparison of the two rules over a life.

## 2. Terms

- A *family* is the four operations add, sub, max and min at one modulus. A *reduction pair* is a pair of add or sub whose answer needs the modulus (for add, a + b is at least the modulus; for sub, a − b is below zero).
- An *update* takes 64 items of one family: 16 *asked* items, which are answered first and never learned from, and 48 *taught* items, from which the network then learns. An asked pair is never a taught pair. A *reduction item* is an asked item on a reduction pair.
- A *visit* is a number of updates on one family (64, 128 or 256 in the calibration). A *first visit* is a family's first. A *life* is a sequence of visits in a *world*, which is a seed that fixes the families, their items and the stream. Worlds are numbered 1 to 8.
- The *calibration* (job F1 of E44) ran lives of eight visits in which visit v is family v, so that every visit is a first visit. *Families 1 to 7* are the second to the eighth family taken; family 0 is the first.
- *Back-propagation* is the exact gradient. *The sweep* is the project's second learning rule, which sends the error back through each linear map by the signs of its weights.
- A *birth* is a network trained beforehand on the project's first family (mod 89) in experiment E41, by back-propagation (rule B) or by the sweep (rule S). *Accuracy R* is the birth's share right on 2,000 held-out reduction pairs of that family.
- An *arm* is a birth together with the rule by which the life learns. `naive`: the birth of rule B, a life by back-propagation. `sweep`: the birth of rule S, a life by the sweep. `sweep-after-naive`: the birth of rule B, a life by the sweep. None of the three has a store.
- A *store* is a buffer of past taught items; an arm with a store (`replay`, `sweep+replay`) learns half of each update, 24 items, from it and the other 24 from new taught items.
- *The family's figure* over some updates of a visit is the mean of two shares: the share right on the asked reduction items of add, and the same for sub. *The first-visit level* is the family's figure over the last quarter of the updates of the family's first visit (the last 16, 32 or 64 updates at 64, 128 or 256 updates a visit).
- *The gate's figure* of a world is the median of the first-visit level over families 1 to 7.
- *The gate* (the brief's words, for one visit length, "under `naive`"): "In at least six of eight worlds, the median over the second to the eighth family taken (families 1 to 7 in the file's numbering) of the first-visit level on reduction pairs is between 0.20 and 0.90; and a life of 1,024 visits at that length is forecast within 6 CPU-hours. Use the shortest visit length that passes." *The band* is 0.20 to 0.90; a world is *inside* if its figure is in the band, both ends counted, as the audit's script counts them. *The forecast* is the receipts' CPU-seconds a visit, averaged over the eight worlds, times 1,024.
- An *event file* holds one row for every asked item of a life: the update, the visit, the task, whether the answer was right, whether the pair needs the reduction. A *receipt* is the JSON file that the runner writes beside it.
- *Pooled* means over the 56 first visits of families 1 to 7 in the eight worlds. A *90% interval* is the mean of eight per-world values plus or minus 1.895 standard errors (Student's t with 7 degrees of freedom).

Shell names: `D=origin/codex/minifly-consolidated-20261023` (at 7cabb13a); `L=origin/claude/e44-l2-first-long-lives` (at 9519302d); `F1=MOD97_LOCAL_RULE_20261106/exp45/e44/F1`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`.

## 3. The answers in brief

1. Under `sweep` the gate's figure is inside the band in 3 of 8 worlds at 64 updates a visit, in 8 of 8 at 128 and in 8 of 8 at 256. Under `naive` the counts are 0, 4 and 8. The cost condition is met at every length for both arms.
2. At 256 updates both arms are still rising at the end of a first visit, by amounts that are not distinguished: from the third to the last quarter `naive` rises by 0.065 and `sweep` by 0.062. Neither has stopped rising. The difference between the arms, 0.12 in the mean over families, is formed within the first 64 updates of a visit and does not change afterwards.
3. The calibration's files hold nothing for an arm with a store: none of its 115 receipts records a store.
4. The visit length of 256 updates meets both of the gate's conditions for the arm `sweep`. The gate's own rule would have chosen 128 for it. The paragraph is section 8.

## 4. The gate's figure for `sweep`, beside `naive` (item 1)

Source: the 72 event files `$F1/lw_<arm>_w<1..8>_v8[_vu64|_vu128]_turn.npz` on `$D` and their receipts `.json`; `e44_gate_sweep.py G`.

| Arm | Updates a visit | Gate's figure, worlds 1 to 8 | Mean | Inside the band | Forecast, 1,024 visits | Both conditions met |
|---|---|---|---|---|---|---|
| naive | 64 | 0.059 0.112 0.068 0.065 0.068 0.053 0.084 0.133 | 0.080 | 0 (8 below) | 0.97 h | no |
| naive | 128 | 0.169 0.434 0.169 0.322 0.213 0.096 0.195 0.301 | 0.237 | 4: worlds 2, 4, 5, 8 (4 below) | 1.94 h | no |
| naive | 256 | 0.368 0.566 0.301 0.457 0.491 0.368 0.436 0.636 | 0.453 | 8 | 3.85 h | yes |
| sweep | 64 | 0.186 0.355 0.122 0.182 0.240 0.112 0.152 0.233 | 0.198 | 3: worlds 2, 5, 8 (5 below) | 1.01 h | no |
| sweep | 128 | 0.452 0.458 0.396 0.416 0.420 0.345 0.422 0.441 | 0.419 | 8 | 2.07 h | yes |
| sweep | 256 | 0.669 0.656 0.538 0.561 0.737 0.537 0.622 0.595 | 0.614 | 8 | 4.23 h | yes |
| sweep-after-naive | 64 | 0.026 0.031 0.036 0.036 0.042 0.059 0.029 0.031 | 0.036 | 0 (8 below) | 1.00 h | no |
| sweep-after-naive | 128 | 0.040 0.047 0.017 0.067 0.085 0.054 0.051 0.045 | 0.051 | 0 (8 below) | 2.00 h | no |
| sweep-after-naive | 256 | 0.157 0.275 0.117 0.187 0.182 0.200 0.212 0.240 | 0.196 | 3: worlds 2, 7, 8 (5 below) | 4.03 h | no |

No world is above the band in any row. World 6 under `sweep-after-naive` at 256 is 0.19972, below the band's lower limit although it prints as 0.200.

**The shortest length at which both conditions are met** is 256 for `naive`, 128 for `sweep`, and none of the three for `sweep-after-naive`.

**How the figures were checked.** The script reads 1,376,256 answers of the 72 lives. For `naive` and `sweep` the figures are the audit's own table (`gate_table` of `e44_l2_checks.py`), in which every asked item is marked again from the reviewer's own stream and asserted equal to the event file in task, reduction mark and right or wrong; the third arm goes through the same two functions. The `naive` rows equal the report's table and the audit's section 2.1 to the third decimal (asserted), and the audit's own command `e44_l2_checks.py C` prints the same six rows for `naive` and `sweep`. Two further routes give the same 576 family figures with a largest difference of zero: the event file's own columns alone (the audit's `l2_quantities`), and the worker's committed table `$F1/E44_F1_first_visit.csv`. The worker's table also agrees on the 384 figures of the whole visit and of its first 16 updates at 256, which tests the window bounds used in section 5.

**How fine a figure is.** The last quarter of a first visit of 256 updates holds 221 to 284 asked reduction items of add and sub (median 250). One family's figure there is the mean of two shares of about 125 items each, so its sampling error is about 0.03 (at a share of one half, 0.045 a share and 0.032 for the mean of the two).

**The difference between the arms**, gate's figure of one arm minus the other's, worlds 1 to 8:

| Difference | Updates a visit | Worlds 1 to 8 | Mean | 90% interval | Worlds above zero |
|---|---|---|---|---|---|
| sweep − naive | 64 | +0.127 +0.243 +0.055 +0.117 +0.172 +0.059 +0.067 +0.101 | +0.118 | +0.074 to +0.161 | 8 |
| sweep − naive | 128 | +0.282 +0.024 +0.226 +0.094 +0.207 +0.249 +0.227 +0.140 | +0.181 | +0.122 to +0.240 | 8 |
| sweep − naive | 256 | +0.301 +0.089 +0.237 +0.104 +0.246 +0.169 +0.186 −0.041 | +0.161 | +0.089 to +0.234 | 7 |
| sweep-after-naive − naive | 256 | −0.210 −0.292 −0.185 −0.270 −0.308 −0.169 −0.224 −0.396 | −0.257 | −0.307 to −0.206 | 0 |
| sweep − sweep-after-naive | 256 | +0.511 +0.381 +0.421 +0.374 +0.554 +0.337 +0.410 +0.355 | +0.418 | +0.367 to +0.469 | 8 |

With the mean over families 1 to 7 in place of the median, the figures at 256 are 0.495 0.449 0.398 0.468 0.527 0.427 0.474 0.529 for `naive` and 0.630 0.588 0.481 0.582 0.613 0.593 0.633 0.583 for `sweep`; the difference is +0.117 (90% interval +0.091 to +0.144; above zero in 8 worlds).

**The learners.** The receipts record one birth file an arm, the same in all eight worlds and at all three lengths: `exp45/e41/e41_B_h256_f50_73129.pt` for `naive` and for `sweep-after-naive` (accuracy R at birth 94.65%), `exp45/e41/e41_S_h256_f50_73129.pt` for `sweep` (91.95%). `sweep` and `naive` therefore differ in two things, the birth and the rule of the life. `sweep` and `sweep-after-naive` differ in the birth only, and `sweep-after-naive` and `naive` in the rule of the life only.

## 5. The level within a first visit of 256 updates (item 2)

Source: the 24 event files at 256 updates a visit on `$D`; `e44_gate_sweep.py Q`. The family's figure is taken over each quarter of the visit's updates (updates 0 to 63, 64 to 127, 128 to 191, 192 to 255), for families 1 to 7. The last quarter is the first-visit level. The script asserts that the quarters, and the eighths used below, add up to the whole visit in items and in right answers for every family.

**`naive`**

| World | Mean over families 1 to 7, quarters 1 2 3 4 | Rise 1→2, 2→3, 3→4 | Median over families 1 to 7, quarters 1 2 3 4 |
|---|---|---|---|
| 1 | 0.163 0.351 0.438 0.495 | +0.188 +0.087 +0.058 | 0.091 0.169 0.239 0.368 |
| 2 | 0.083 0.261 0.373 0.449 | +0.178 +0.113 +0.075 | 0.073 0.273 0.533 0.566 |
| 3 | 0.088 0.233 0.354 0.398 | +0.144 +0.121 +0.044 | 0.059 0.124 0.217 0.301 |
| 4 | 0.121 0.296 0.388 0.468 | +0.175 +0.092 +0.080 | 0.149 0.257 0.338 0.457 |
| 5 | 0.125 0.357 0.494 0.527 | +0.232 +0.136 +0.033 | 0.057 0.240 0.470 0.491 |
| 6 | 0.105 0.255 0.339 0.427 | +0.149 +0.084 +0.088 | 0.083 0.223 0.308 0.368 |
| 7 | 0.132 0.360 0.413 0.474 | +0.228 +0.053 +0.061 | 0.078 0.238 0.283 0.436 |
| 8 | 0.158 0.341 0.448 0.529 | +0.183 +0.106 +0.081 | 0.215 0.451 0.558 0.636 |
| Pooled | 0.122 0.307 0.406 0.471 | +0.185 +0.099 +0.065 | 0.082 0.239 0.362 0.447 |

**`sweep`**

| World | Mean over families 1 to 7, quarters 1 2 3 4 | Rise 1→2, 2→3, 3→4 | Median over families 1 to 7, quarters 1 2 3 4 |
|---|---|---|---|
| 1 | 0.232 0.445 0.550 0.630 | +0.213 +0.105 +0.080 | 0.238 0.504 0.543 0.669 |
| 2 | 0.178 0.439 0.553 0.588 | +0.262 +0.114 +0.035 | 0.186 0.441 0.664 0.656 |
| 3 | 0.132 0.332 0.449 0.481 | +0.200 +0.118 +0.032 | 0.100 0.283 0.477 0.538 |
| 4 | 0.178 0.412 0.533 0.582 | +0.234 +0.121 +0.050 | 0.140 0.418 0.539 0.561 |
| 5 | 0.206 0.453 0.534 0.613 | +0.248 +0.081 +0.079 | 0.237 0.544 0.662 0.737 |
| 6 | 0.227 0.434 0.529 0.593 | +0.206 +0.095 +0.064 | 0.221 0.366 0.493 0.537 |
| 7 | 0.220 0.455 0.552 0.633 | +0.236 +0.096 +0.081 | 0.179 0.490 0.580 0.622 |
| 8 | 0.213 0.424 0.511 0.583 | +0.211 +0.087 +0.072 | 0.244 0.501 0.567 0.595 |
| Pooled | 0.198 0.424 0.526 0.588 | +0.226 +0.102 +0.062 | 0.183 0.430 0.547 0.595 |

In the pooled rows the first two columns are the mean over the 56 first visits and the last column is the median of the 56. The median column's last entry in each world row is the gate's figure of section 4. The pooled share right over all reduction items of the 56 first visits (14,092 to 14,159 items a quarter) is 0.123 0.306 0.403 0.468 for `naive` and 0.198 0.423 0.524 0.586 for `sweep`.

**The rise from each quarter to the next**, mean over the eight worlds with its 90% interval, and the number of the 56 first visits in which the later quarter is the higher:

| Arm | Quarter 1 → 2 | Quarter 2 → 3 | Quarter 3 → 4 |
|---|---|---|---|
| naive | +0.185 (+0.163 to +0.206); 56 of 56 | +0.099 (+0.082 to +0.116); 55 of 56 | +0.065 (+0.052 to +0.078); 50 of 56 |
| sweep | +0.226 (+0.211 to +0.241); 56 of 56 | +0.102 (+0.092 to +0.112); 52 of 56 | +0.062 (+0.048 to +0.075); 46 of 56 |
| sweep-after-naive | +0.063 (+0.051 to +0.075); 54 of 56 | +0.062 (+0.050 to +0.073); 51 of 56 | +0.054 (+0.045 to +0.063); 51 of 56 |

Every rise is above zero in all eight worlds. The third quarter's level is 0.862 of the last quarter's under `naive` and 0.895 under `sweep`.

**The difference between `sweep` and `naive` in each quarter** (mean over families 1 to 7 in each world, then over the worlds): +0.076 (+0.059 to +0.093), +0.118 (+0.092 to +0.144), +0.120 (+0.085 to +0.156) and +0.117 (+0.091 to +0.144), above zero in all eight worlds in each quarter. The sweep's rise from the third to the last quarter minus `naive`'s is −0.003 (−0.023 to +0.017); from the second to the last quarter it is −0.001 (−0.024 to +0.023).

**By eighth of the visit** (32 updates each), pooled mean:

| Arm | Eighths 1 to 8 | Slope over the last four eighths, per eighth |
|---|---|---|
| naive | 0.071 0.173 0.271 0.343 0.385 0.426 0.452 0.489 | +0.034 (+0.028 to +0.040) |
| sweep | 0.111 0.285 0.389 0.458 0.511 0.541 0.582 0.594 | +0.029 (+0.023 to +0.035) |
| sweep-after-naive | 0.038 0.081 0.107 0.139 0.171 0.199 0.229 0.249 | +0.026 (+0.023 to +0.030) |

`sweep` minus `naive`, pooled, by eighth: +0.040 +0.113 +0.118 +0.115 +0.126 +0.114 +0.130 +0.104. The sweep's slope over the last four eighths minus `naive`'s is −0.005 (−0.014 to +0.004). The sweep's rise minus `naive`'s from each eighth to the next is +0.073, +0.006, −0.003, +0.011, −0.012, +0.015 and −0.025, with half-widths of the 90% interval of 0.031, 0.028, 0.023, 0.021, 0.026, 0.023 and 0.007.

**What the tables say.**

- **Neither arm has stopped rising at the end of a visit of 256 updates.** Both rise from the third to the last quarter in all eight worlds, by 0.065 and 0.062, and the two rises are not distinguished. The case that was asked about, the sweep still rising where `naive` has stopped or the reverse, is not what the files show.
- **The difference between the two arms is formed early.** The sweep rises faster only in the first step, from the first to the second eighth (+0.073, all eight worlds). From the second eighth on, that is after 32 updates, the pooled difference stays between 0.104 and 0.130, and from the second quarter on the mean over worlds stays at 0.12.
- **In the last 32 updates the sweep's rise is the smaller one** (+0.012 against +0.037; the difference is −0.025, below zero in all eight worlds). I give this as an observation and not as a finding. It is one of seven steps, it was looked at after the data were seen, the step before it goes the other way (+0.015), and over the last half of the visit the two slopes are not distinguished.
- **`naive`'s level at the end is reached by `sweep` half a visit earlier.** `naive`'s pooled level over its last eighth is 0.489; `sweep`'s pooled level first reaches that value in its fifth eighth (updates 128 to 159). The gate's figures say the same: `sweep` at 128 updates a visit (mean 0.419) is near `naive` at 256 (0.453).
- **Family 0**, which the gate does not count, has the same form: 0.149 0.396 0.552 0.616 under `naive` and 0.290 0.538 0.600 0.650 under `sweep` (means of eight worlds by quarter).

## 6. An arm with a store (item 3)

Source: `e44_gate_sweep.py S`.

**The calibration's files hold nothing for an arm with a store.** Under `MOD97_LOCAL_RULE_20261106/exp45/e44/` on `$D` there are 485 files, 115 of them receipts of lives or branches: 96 under `F1/` (48 of `naive`, of which 24 are the set with decay 0; 24 of `sweep`; 24 of `sweep-after-naive`), 16 under `F1/lap2/` (8 of `naive`, 8 of `sweep`) and 3 check lives of `naive`. Every one records `replay: false`, a store that was not used, and zero replayed items. No file name contains "replay". The brief says so: "The replay arms of E39 remain in the runner and are not run here."

**The nearest figures the calibration holds are those at 128 updates a visit, and they are not the same thing.** An arm with a store learns 24 new items an update, so a visit of 256 updates teaches 256 × 24 = 6,144 new items, the number that a visit of 128 updates teaches without a store (128 × 48). At 128 updates the gate's figure is inside the band in 4 of 8 worlds under `naive` and in 8 of 8 under `sweep` (section 4). A life with a store differs from those lives in two ways that these figures do not contain: it makes twice as many updates, and half of each update is made of stored items of other families. The figures at 128 updates are therefore no estimate of the gate under `replay` or `sweep+replay`.

**Outside the calibration**, among the 40 fetched refs only `$L` (at 9519302d) holds event files of an arm with a store: five check lives of `replay`. There is no event file of `sweep+replay` on any fetched ref. One of the five contains a first visit of a second family: world 5, four visits by the schedule (families 0, 1, 0, 1), `replay` at ρ = 0.5 (`exp45/e44/L2/checks/c7_replay/lw_replay_w5_v4_rho0.5.npz`), with its partner under `naive` (`…/c7_naive/lw_naive_w5_v4.npz`), both from the birth `e41_B_h256_f50_93683.pt`:

| Arm | Family 1's figure by quarter of its first visit | Family 0's last quarter |
|---|---|---|
| replay | 0.019 0.088 0.124 0.238 | 0.170 |
| naive | 0.020 0.124 0.279 0.509 | 0.235 |

This is one family in one world, and the gate's figure is a median over seven families in each of eight worlds. It shows that the files which the long lives will write contain what is needed, and that in this one case the first-visit level under `replay` is about half of `naive`'s and near the band's lower limit. The trial of the long lives will give the figure for `sweep+replay`.

## 7. The births: how far the calibration's figures hold for the long lives

Source: `e44_gate_sweep.py B`.

The calibration used the births of seed 73129 in every world. The long lives use, in world w, the births of seed (73129, 93683, 116089)[(w − 1) mod 3]: worlds 1 and 4 the calibration's births, world 2 those of seed 93683, world 3 those of seed 116089. Accuracy R at birth, from E41's receipts on `$D` (`exp45/e41/e41_<B|S>_h256_f50_<seed>.json`):

| Seed | Rule B | Rule S |
|---|---|---|
| 73129 | 94.65% | 91.95% |
| 93683 | 94.25% | 92.55% |
| 116089 | 93.00% | 90.90% |

The worker of the first long lives ran the calibration life of worlds 2 and 3 again under `naive` with the world's own birth (`$L`, `exp45/e44/L2/births/lw_naive_w<2|3>_v8_turn.npz`; the same stream, asserted item by item). The gate's figure is 0.573 in world 2 (the calibration's: 0.566; difference +0.007) and 0.313 in world 3 (0.301; +0.011). Single families move by more: −0.149 to +0.122 in world 2 and −0.505 to +0.129 in world 3. **No such life exists under `sweep`:** the folder `L2/births/` on `$L` holds these two event files only, and no event file of an eight-visit life in turn of a sweep arm exists outside the calibration's folder on any of the 40 fetched refs. The sweep's figures of section 4 are therefore measured with one birth file, that of seed 73129, and whether they hold with the other two births of rule S is not in any committed file.

## 8. Does 256 updates a visit meet the gate's condition for the sweep (item 4)

At 256 updates a visit the arm `sweep` meets both conditions of the gate: its figure is inside 0.20 to 0.90 in all eight worlds (0.537 to 0.737), and a life of 1,024 visits is forecast at 4.23 CPU-hours against a limit of 6. It meets them at 128 updates as well (0.345 to 0.458; 2.07 hours), so the gate's own rule, "use the shortest visit length that passes", gives 128 for the sweep and 256 for `naive`: 256 is a length at which the sweep passes, and it is not the length the gate would have chosen for it. At 256 the sweep's figure is above `naive`'s by 0.161 as a mean over the eight worlds (90% interval 0.089 to 0.234; above in seven worlds), and by 0.117 when the mean over families replaces the median; the difference is formed within the first 64 updates of a visit and does not change afterwards, and both arms are still rising at the end of the visit, by 0.06 from the third quarter to the last. For a comparison of the two rules over a life at this visit length this says three things about the instrument and nothing about the rules. First, neither arm begins below the band or above it, so the level of both can be read as rising or as falling. Second, the two arms do not begin at the same level: a difference of 0.12 to 0.16 at the end of a first visit is present within the first eight visits of a life, so a difference of that size and sign in a quantity that averages the score over visits repeats what the calibration already shows and is not evidence of what a long life adds; only changes within an arm, such as its trends, are free of this difference, and the two arms have unequal room for a rise before the band's upper limit (0.90 less the mean of the eight figures: 0.29 for `sweep`, 0.45 for `naive`). Third, the difference belongs to the arm and not to the rule of the life: `sweep` starts from the birth made under the sweep and `naive` from the birth made under back-propagation, one birth file each, and the sweep's life from back-propagation's birth is 0.257 below `naive`, with three of eight worlds inside the band. At this visit length the instrument therefore compares two learners, each a birth together with its rule, that begin at unequal levels; it cannot assign a difference between them to the rule of the life.

## 9. Commands and run times

All from the repository root, on `claude/vibrant-cerf-sje726`, with `python3 -I` (Python 3.13.17, numpy 2.5.3), one thread (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`), on a host of four cores shared with other jobs. Files of other refs are read with `git show <ref>:<path>` and listed with `git ls-tree`; nothing is fetched, checked out or written.

| Command | Content | Run time |
|---|---|---|
| `python3 -I $R/e44_gate_sweep.py` | all four parts | 75 s |
| `python3 -I $R/e44_gate_sweep.py G` | section 4 | 71 s |
| `python3 -I $R/e44_gate_sweep.py Q` | section 5 | 70 s |
| `python3 -I $R/e44_gate_sweep.py S` | section 6 | 67 s |
| `python3 -I $R/e44_gate_sweep.py B` | section 7 | 70 s |
| `python3 -I -B $R/e44_l2_checks.py C` | the audit's own part C: the six rows of `naive` and `sweep` | 44 s |

Every command is prefixed with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Every part first reads and marks the 72 event files, which takes about 65 seconds; the parts themselves take about 3 seconds each. Each part was run alone and the four together with the script as it now is (SHA-256 in section 11), and the single runs print what the joint run prints. The printed output is not kept in the repository; the commands regenerate it.

The files read:

| Ref | Path | Used for |
|---|---|---|
| `$D` at 7cabb13a | `$F1/lw_{naive,sweep,sweep-after-naive}_w{1..8}_v8[_vu64\|_vu128]_turn.npz` (72 event files) and `.json` (72 receipts) | sections 4 and 5 |
| `$D` | `$F1/E44_F1_first_visit.csv` | the third route of section 4 |
| `$D` | every `.json` under `MOD97_LOCAL_RULE_20261106/exp45/e44/` (115 receipts) | section 6 |
| `$D` | `MOD97_LOCAL_RULE_20261106/exp45/e41/e41_{B,S}_h256_f50_{73129,93683,116089}.json` | section 7 |
| `$L` at 9519302d | `MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay/lw_replay_w5_v4_rho0.5.{npz,json}`, `…/c7_naive/lw_naive_w5_v4.{npz,json}` | section 6 |
| `$L` | `MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/lw_naive_w{2,3}_v8_turn.{npz,json}` | section 7 |
| `claude/vibrant-cerf-sje726` | `$R/e44_l2_checks.py` (last changed in ef8b512b), `$R/e44_world_reader.py`, `$R/np_gru.py` | imported, unedited |

The words of the gate are those of `OPUS_TASK_E44_LONG_LIFE_WORLD_IN_THE_FULL_SCALE_FORMAT_20261116.md` on `$D` ("The gate, per visit length, under `naive`"), and the arms' births those of `MOD97_LOCAL_RULE_20261106/exp45/e44/E44_REPORT.md` on `$D` ("The learner and the ledger"), read with `git show`.

## 10. What I could not check

- **No life was run.** Every figure is recomputed from committed event files. That the files are what the runner wrote is taken from their receipts; the marking of each answer against the reviewer's own stream shows that the items and the reduction marks are the stream's, and it does not show that the recorded answers are the network's.
- **The sweep's gate with the births of the long lives.** The calibration has one birth file an arm. For `naive` two further lives exist and the gate's figure moved by 0.007 and 0.011 (section 7); for `sweep` none exists. The difference of 0.161 between the arms is a difference between two birth files and the lives grown from them. How much it would vary over births of other seeds is not measured.
- **The intervals** take the eight worlds as the repeated unit. The worlds differ in families and stream and share the birth file, so the intervals describe the spread over worlds and not over births.
- **An arm with a store.** The gate's figure under `replay` or `sweep+replay` is in no committed file. The one first visit under `replay` in section 6 is a single family from a check life, on a ref that its worker is still pushing to; I read it at 9519302d.
- **First visits later in a life.** The calibration's first visits are the first eight visits of a life. In a life by the schedule a new family arrives among returns, hundreds of visits in. Whether the first-visit level of either arm stays inside the band there is what the long lives measure.
- **The forecast** uses the CPU-seconds recorded in the receipts, which I cannot measure, and leaves out the time of the list, as the brief's and the audit's forecast do.
- **The lives on `$L`.** They were run with the changed runner. That the changed runner reproduces the calibration life of world 1 is the worker's check; I did not repeat it.
- **Independence of the routes.** Routes 1 and 2 of section 4 are both the reviewer's functions. Route 3 is the worker's scorer. The three agree exactly, which leaves open only an error common to the reviewer's reading of the brief and the worker's.

## 11. A note on what was written

Two files were written: this document and `e44_gate_sweep.py` (SHA-256 `6c85640770865d96be9c70fadd443cb2c0019602da21cde450247423c3026424`). The script sets `sys.dont_write_bytecode` and writes nothing. One thing beyond the two files happened and is recorded here: a trial import of the audit's script, made before the script set that flag, caused Python to write its compiled copy `__pycache__/e44_l2_checks.cpython-313.pyc` in this folder. That file is a cache of the unedited audit script, the folder `__pycache__/` is ignored by git, and I left it as it is.
