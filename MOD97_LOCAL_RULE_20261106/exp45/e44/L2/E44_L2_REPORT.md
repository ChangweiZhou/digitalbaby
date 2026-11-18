# E44, job L2, release 1: the first long lives — report (18 November 2026)

Branch `claude/e44-l2-first-long-lives`. The roster note is `OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md`, revision 2, on this branch. The worker entered it from the reviewer's audit at ef8b512b while the coordinator was away. The note's definitions D1 to D15 are used throughout and are not repeated here.

Every figure below comes from one of two files beside this report:

- `E44_L2.json`, with `E44_L2_TABLES.md` as its tables, written by `python3 smallworld/l2_report.py final`;
- for the trial, `trial/E44_L2_TRIAL.json` and `trial/E44_L2_TRIAL_REPORT.md`.

Both are computed from the lives' event files and list files in `lives/`. The level of evidence is E2, as the note states.

**What the intervals count.** A statement about arms is made over worlds, with a 90% interval (Student's t over worlds), as the note's D11 fixes it. The intervals count **worlds, not births**:

- the eight worlds at 256 visits start from three birth files;
- the four long worlds start from three, since worlds 1 and 4 share seed 73129.

Within a seed the worlds differ, but the learner at the start does not. The reviewer's verification averaged the worlds of each seed into one value (three values, t with 2 degrees of freedom; not the note's definition). The readings that **no longer exclude zero** under that averaging are:

- `naive-wd0` − `naive` in late level at 256 visits: −0.110 [−0.306, +0.086], and with it PL2.4;
- `replay-wd0` − `replay` in late level: −0.161 [−0.432, +0.109], and with it PL2.9;
- `naive-wd0` − `naive` in kept at return after 33 visits or more: +0.027 [−0.040, +0.095], and with it PL2.5. That reading also rests on few items: 18 right answers of 471 under `naive-wd0` against 4 under `naive`, over 121 returns;
- `replay-wd0` − `replay` in score trend and in trend of learning;
- "still learning" for `naive` at 256 visits: its trend of learning becomes +0.070 [−0.068, +0.209]. Its lower bound by worlds is +0.003, and the trend is negative in three of the eight worlds.

The readings that **survive** are:

- `replay` higher than `naive` in kept at return, in every class and at both lengths;
- `naive-wd0` higher than `naive` in kept at return after 5 to 32 visits: +0.072 [+0.011, +0.133];
- `naive-wd0` lower than `naive` in trend of learning: −0.126 [−0.220, −0.032];
- both long arms still learning at 1,024 visits: trend of learning +0.100 [+0.084, +0.116] and +0.111 [+0.062, +0.160].

The intervals are also wide. For example, `replay` − `naive` in late level is +0.047 ± 0.058 at 1,024 visits and +0.015 ± 0.032 at 256. "Stuck" means only that no improving trend was found at that precision. *(Corrected on 18 November after the reviewer's verification, `E44_L2_VERIFICATION_20261118.md` on the reviewer's branch.)*

**Words used below.**

- *Declining*: a trend whose 90% interval lies wholly below zero (the note's section "How the result is stated").
- *Closed*: as D12 defines it.

## 1. What was run

**The runner.** The runner is `smallworld/sw_run.py`, sha256 41714c5b…. It is the runner at 39e4e594 with the note's two changes, made in commit f249b419:

- `--list-every K` and `--decay-distances` (with `none`); the birth family's items follow both arguments;
- `load_learner` accepts the replay arms.

Every one of the 32 receipts names this same runner file.

**The host.** Python 3.13.18, torch 2.15.0+cpu and numpy 2.5.3, one thread per process (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`). The machine had 4 cores; at most three processes ran at once, through `smallworld/l2_driver.py`.

**The learners.** Each life starts from E41's half-width backpropagation birth (H = 256, U = 128, 317,457 parameters) of the world's seed:

- seed 73129 for worlds 1, 4 and 7;
- seed 93683 for worlds 2, 5 and 8;
- seed 116089 for worlds 3 and 6.

**The settings.** Visits of 256 updates; `--list-every 8 --decay-distances none`. The arms:

| Arm | Store | Decay in the life |
|---|---|---|
| `naive` | none | 0.25 |
| `replay` | ρ = 0.5: 24 of 48 items from a reservoir store of 55,209 items | 0.25 |
| `naive-wd0` | none | 0 |
| `replay-wd0` | as `replay` | 0 |

**The roster.** The roster ran in the order of the audit's recommendation 1.

| Part | Lives | Visits | Completed |
|---|---|---|---|
| Trial: world 1, `naive` and `replay` | 2 | 1,024 | 01:42 UTC |
| Long: worlds 2 to 4, `naive` and `replay` | 6 | 1,024 | 08:02 UTC |
| Short: worlds 5 to 8, `naive` and `replay` | 8 | 256 | 09:46 UTC |
| Without decay: worlds 1 to 8, `naive-wd0` and `replay-wd0` | 16 | 256 | 14:38 UTC |
| Learning probe: 4 visits over families 52 to 55, from the states after 64, 256 and 1,024 visits of the eight long lives | 24 branches | — | — |

All 32 lives and all 24 branches completed; nothing failed.

**Cost.**

| Item | CPU-hours | Source |
|---|---|---|
| The 32 lives | 46.98 | sum of the receipts' totals |
| Probe branches | 0.26 | — |
| Checks and the two 8-visit lives of the other births | 0.58 | — |
| Lost to interruptions | 1.34 | 408 visits computed and then computed again (the logs), at the lives' mean of 11.80 CPU-seconds a visit |

The forecast was 48.2 before the trial and 48.0 after it.

**Interruptions and resumes.** The record of each start is in `lives/*.starts.jsonl`, and each receipt lists its resumes.

| When (UTC) | What happened | Lives affected | What was done |
|---|---|---|---|
| 17 Nov, 21:15 | The machine was stopped after the session went idle | The trial's two lives and world 2's `naive` long life, at visit 57, before any state was saved | Started again from the beginning at 22:07 |
| 18 Nov, 00:57 | Container restart | World 1's `naive` and `replay` and world 2's `naive`, at visits 790 to 797 | Continued at 00:59 with `life --resume` from their states after 768 visits: `lives/lw_naive_w1_v1024_state_v768.pt` (sha256 17b8f9254d18b3958509d1bc86a5406e94111c9002cb8c0ea5d7e62d6b205246), `lives/lw_replay_w1_v1024_rho0.5_state_v768.pt` (sha256 48596256a5b7f2257941cfd30908cd734b74d2c156347aea099fb26511a6982e), `lives/lw_naive_w2_v1024_state_v768.pt` (sha256 71919f4b8812a1dccbaad4692d37472cc8ac0853668b8d1030c7ac37bed81672) |
| 18 Nov, about 05:20 | Container restart | World 4's two long lives, at visits 309 and 315 | Resumed from their states after 256 visits: `lives/lw_naive_w4_v1024_state_v256.pt` (sha256 47339b8d931b1809ef9e3a5cdd550360013e0f45e384bb03ec6a5f9336f94bd4), `lives/lw_replay_w4_v1024_rho0.5_state_v256.pt` (sha256 2ccdfafc445a9f5c1a01f4f9e4dd69dc33cec4fdbbce6e6147540cdff13f586f) |
| 18 Nov, about 05:20 | Container restart | World 5's `replay` life, at visit 37, with no state | Started again from the beginning |

Each receipt records its resume under `resumes`, with the path and file hash of the state it was resumed from; that state is not in the receipt's list `states`, because a resumed life's list is the one held in the state it was resumed from. The states were removed from the tree afterwards and remain in the branch's history. The note's check 5 showed that a resumed life equals an uninterrupted one in weights, optimizer state, events, learned batches and list. It showed this only for one `replay-wd0` life of 8 visits in world 1, at `--list-every 2 --decay-distances 1,2,4` (the lives used 8 and `none`). That life was cut after visit 4, before its store was full: the store fills at update 1,150, in visit 4. The check therefore does not exercise a resumption with a full store, nor the lives' own settings.

For the five resumed long lives the evidence is of another kind:

- every visit computed twice prints the same share right and loss before and after the resumption (408 visits computed twice in all, none different);
- the event files are continuous, and equal to the regenerated stream on every column that can be recomputed;
- the states loaded have the SHA-256 that the receipts record.

These records make a difference unlikely, but they do not show equal weights: no uninterrupted run of these lives exists, and the log lines are rounded. *(Corrected on 18 November after the reviewer's verification, `E44_L2_VERIFICATION_20261118.md` on the reviewer's branch.)*

## 2. Checks

All checks passed (`checks/E44_L2_CHECKS.json`, commit bd985865).

**Checks of the runner change, (a) to (d).**

- (a) With the defaults, the runner at 39e4e594 and the changed runner give, on this host, the same hashes of the list, the events, the final weights and optimizer state, and the learned batches. Both equal E44's committed receipt.
- (b) The three world-5 lives of 32 visits match the audit's section 7 exactly:

  | Life | Keys | Items |
  |---|---|---|
  | defaults | 241 | 165,200 |
  | `--list-every 4 --decay-distances 1,2,4,8` | 77 | 38,800 |
  | `--list-every 4 --decay-distances none` | 57 | 27,600 |

  The reviewer's part K agrees for all three, and every row present in two runs is equal item by item.
- (c) Across those three runs, the events, weights, optimizer state and learned batches are equal.
- (d) The settings are carried into receipts and states, and kept on resume.

**The note's checks 1 to 7.** All pass:

- check 1: the generator's digests, and the list-item digests of all 12 roster lives in the audit's section 7;
- check 2: the births of seeds 93683 and 116089 equal their E41 receipts;
- check 3: `replay` with ρ = 0 equals `naive`;
- check 4: the store's capacity is 55,209 items;
- check 5: the resume of a `replay-wd0` life is exact (at settings other than the lives', and before its store was full; section 1);
- check 6: the arms without decay read back decay 0.0;
- check 7: 11.7 to 12.1 CPU-seconds a visit in every arm.

**The audit's recommendation 3.** Two 8-visit lives with the other two births gave median first-visit levels of 0.573 (world 2, seed 93683) and 0.313 (world 3, seed 116089). Both are above the 0.20 floor.

**Checks on the lives.**

- Every one of the 32 list files has the reviewer's row keys, by part K, with the items in the reviewer's order.
- The scorer `smallworld/l2_score.py` equals the reviewer's part J on D2 to D10, for the trial's two lives at 256 and 1,024 visits.
- Every `replay` life ends with its store full (55,209 items) and replays from its second update.
- Every visit's mean loss is finite: the largest is 0.934 (`naive-wd0`, world 2).

**One harness error of mine.** The first attempt of the check report ran the reviewer's script without a file it imports. It is kept as `checks/E44_L2_CHECKS_attempt1_harness_error.json`.

## 3. The trial

The trial's report is `trial/E44_L2_TRIAL_REPORT.md`, commit 118b6047, sent to the reviewer at 01:47 UTC.

- No stop condition held.
- `replay`'s median first-visit level was 0.595, beside `naive`'s 0.757. It is not below 0.20, so `replay` at ρ = 0.25 is not added to release 2 on that account.

## 4. Results

**Late level and first-visit level.** Means over worlds (`E44_L2.json`, `per_life`):

| Arm | Visits | Worlds | Late level D4 [90% interval] | on returns | Median first-visit level D6 |
|---|---|---|---|---|---|
| `naive` | 1,024 | 4 | 0.724 [0.679, 0.769] | 0.709 | 0.801 |
| `replay` | 1,024 | 4 | 0.771 [0.699, 0.843] | 0.771 | 0.672 |
| `naive` | 256 | 8 | 0.669 [0.640, 0.699] | 0.642 | 0.633 |
| `replay` | 256 | 8 | 0.684 [0.645, 0.722] | 0.687 | 0.416 |
| `naive-wd0` | 256 | 8 | 0.574 [0.504, 0.645] | 0.568 | 0.430 |
| `replay-wd0` | 256 | 8 | 0.541 [0.450, 0.631] | 0.546 | 0.273 |

The blind predictor's late level is 0.015 in every arm.

**Kept at return and kept on the list.** Means over worlds:

| Arm | Visits | Kept at return (D8), absence 1 | absence 5 or more | First 16 updates, absence 5 or more (D10) | Kept on the list (D9), 5 or more |
|---|---|---|---|---|---|
| `naive` | 1,024 | 0.802 | 0.011 | 0.107 | 0.011 |
| `replay` | 1,024 | 0.750 | 0.537 | 0.611 | 0.661 |
| `naive` | 256 | 0.696 | 0.017 | 0.110 | 0.021 |
| `replay` | 256 | 0.668 | 0.481 | 0.523 | 0.676 |
| `naive-wd0` | 256 | 0.660 | 0.085 | 0.210 | 0.116 |
| `replay-wd0` | 256 | 0.615 | 0.462 | 0.486 | 0.784 |

**Relearning at return (D10)** by class of absence: the family's figure over the first 16 updates and over the last 64 updates of the return, beside the last 64 updates of its previous visit. Means over worlds; the numbers of returns and items are those of the schedule, the same under every arm.

| Arm | Visits | Absence | Returns | Items | First 16 | Last 64 | Previous visit, last 64 |
|---|---|---|---|---|---|---|---|
| `naive` | 1,024 | 2 to 4 | 1,057 | 4,172 | 0.155 | 0.818 | 0.798 |
| `naive` | 1,024 | 5 to 32 | 1,210 | 4,699 | 0.110 | 0.807 | 0.811 |
| `naive` | 1,024 | 33 or more | 999 | 3,812 | 0.102 | 0.801 | 0.814 |
| `replay` | 1,024 | 2 to 4 | 1,057 | 4,172 | 0.644 | 0.757 | 0.743 |
| `replay` | 1,024 | 5 to 32 | 1,210 | 4,699 | 0.614 | 0.764 | 0.761 |
| `replay` | 1,024 | 33 or more | 999 | 3,812 | 0.607 | 0.770 | 0.764 |
| `naive` | 256 | 2 to 4 | 738 | 2,880 | 0.174 | 0.730 | 0.712 |
| `naive` | 256 | 5 to 32 | 516 | 1,949 | 0.117 | 0.716 | 0.722 |
| `naive` | 256 | 33 or more | 121 | 471 | 0.082 | 0.736 | 0.737 |
| `replay` | 256 | 2 to 4 | 738 | 2,880 | 0.561 | 0.662 | 0.645 |
| `replay` | 256 | 5 to 32 | 516 | 1,949 | 0.518 | 0.660 | 0.645 |
| `replay` | 256 | 33 or more | 121 | 471 | 0.574 | 0.717 | 0.687 |
| `naive-wd0` | 256 | 2 to 4 | 738 | 2,880 | 0.331 | 0.640 | 0.626 |
| `naive-wd0` | 256 | 5 to 32 | 516 | 1,949 | 0.223 | 0.619 | 0.613 |
| `naive-wd0` | 256 | 33 or more | 121 | 471 | 0.163 | 0.624 | 0.608 |
| `replay-wd0` | 256 | 2 to 4 | 738 | 2,880 | 0.524 | 0.587 | 0.572 |
| `replay-wd0` | 256 | 5 to 32 | 516 | 1,949 | 0.491 | 0.572 | 0.559 |
| `replay-wd0` | 256 | 33 or more | 121 | 471 | 0.494 | 0.585 | 0.551 |

**Stuck or still learning (D13).** The improving side is above zero.

| Arm | Visits | Score trend D5 [90% interval] | D5 over all visits of the half (not used) | Trend of learning D7 [90% interval] | Coefficient of ln n in the joint fit with ln(taught pairs) | Statement |
|---|---|---|---|---|---|---|
| `naive` | 1,024 | +0.015 [−0.016, +0.046] | +0.016 [−0.015, +0.047] | +0.098 [+0.083, +0.114] | +0.099 [+0.083, +0.115] | still learning |
| `replay` | 1,024 | +0.052 [−0.005, +0.109] | +0.062 [+0.017, +0.107] | +0.108 [+0.078, +0.137] | +0.109 [+0.080, +0.138] | still learning |
| `naive` | 256 | +0.047 [−0.009, +0.102] | +0.074 [+0.037, +0.111] | +0.070 [+0.003, +0.136] | +0.067 [+0.009, +0.126] | still learning |
| `replay` | 256 | +0.095 [+0.025, +0.166] | +0.107 [+0.047, +0.167] | +0.041 [−0.034, +0.115] | +0.039 [−0.037, +0.115] | still learning |
| `naive-wd0` | 256 | +0.031 [−0.042, +0.104] | +0.051 [−0.043, +0.144] | −0.049 [−0.119, +0.021] | −0.047 [−0.116, +0.022] | stuck |
| `replay-wd0` | 256 | −0.006 [−0.081, +0.069] | +0.020 [−0.064, +0.104] | −0.059 [−0.131, +0.014] | −0.056 [−0.129, +0.018] | stuck |

No interval lies wholly below zero, so no arm is called *declining*. Two values in this table were first printed rounded twice (`replay`'s D5 bound at 1,024 visits, −0.006, and its D5 mean at 256, +0.096); they are now −0.005 and +0.095. *(Corrected on 18 November after the reviewer's verification, `E44_L2_VERIFICATION_20261118.md` on the reviewer's branch.)*

**Between arms (D11).** Per world differences; mean [90% interval]:

| Comparison | Visits | Late level | Kept at return, 5 to 32 | 33 or more | D5 | D7 |
|---|---|---|---|---|---|---|
| `replay` − `naive` | 256 | +0.015 [−0.018, +0.047] not distinguished | +0.459 [+0.413, +0.506] higher | +0.511 [+0.409, +0.613] higher | not distinguished | not distinguished |
| `replay` − `naive` | 1,024 | +0.047 [−0.011, +0.105] not distinguished | +0.531 [+0.440, +0.622] higher | +0.519 [+0.422, +0.616] higher | not distinguished | not distinguished |
| `naive-wd0` − `naive` | 256 | −0.095 [−0.166, −0.024] lower | +0.075 [+0.045, +0.105] higher | +0.031 [+0.003, +0.058] higher | not distinguished | −0.119 [−0.161, −0.076] lower |
| `replay-wd0` − `replay` | 256 | −0.143 [−0.245, −0.041] lower | not distinguished | not distinguished | −0.102 [−0.181, −0.022] lower | −0.099 [−0.154, −0.045] lower |

**The horizon (D12).** The comparison of `naive` and `replay` is **open** at 1,024 visits: the late-level difference is *not distinguished* at both lengths. Every statement here holds up to the number of visits it names. By the note, a longer life needs a larger task space, which is a generator change and an audit.

**What the wrong answers are (D14).** Pooled over worlds and over absences of 2 visits or more, on the first update of a return at 1,024 visits (12,683 answers):

| Class of answer | `naive` | `replay` |
|---|---|---|
| right | 0.013 | 0.549 |
| the right answer under the modulus of the family taught last | 0.499 | 0.013 |
| the right answer under an earlier family's modulus | 0.141 | 0.183 |
| the sum or difference without the reduction | 0.126 | 0.017 |
| another two-digit number | 0.196 | 0.212 |
| chance (a never-met modulus in the place of the last family's) | 0.007 | 0.008 |

On the list at 1,024 visits (2,685,000 answers), `naive` gives the last family's modulus 0.543, and `replay` an earlier family's modulus 0.258. The tables file holds only these rows pooled over absences of 2 visits or more; D14 by class of absence follows. The first version of this paragraph said that the tables file held every absence, which was wrong. *(Corrected on 18 November after the reviewer's verification, `E44_L2_VERIFICATION_20261118.md` on the reviewer's branch.)*

**D14 by class of absence**, pooled over worlds. The columns are, in order: answers; right; no reduction; the last family's modulus; an earlier family's modulus; 89; an operand; the never-met modulus; another two-digit number; chance.

| Arm, visits, part | Absence | Answers | right | no red. | last fam. | earlier fam. | 89 | operand | never-met | other | chance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `naive`, 1,024, stream | 2 to 4 | 4,172 | 0.016 | 0.113 | 0.469 | 0.122 | 0.012 | 0.013 | 0.005 | 0.249 | 0.006 |
| `naive`, 1,024, stream | 5 to 32 | 4,699 | 0.013 | 0.133 | 0.508 | 0.133 | 0.006 | 0.010 | 0.007 | 0.190 | 0.007 |
| `naive`, 1,024, stream | 33 or more | 3,812 | 0.010 | 0.130 | 0.519 | 0.172 | 0.009 | 0.009 | 0.005 | 0.145 | 0.007 |
| `naive`, 1,024, list | 2 to 4 | 198,600 | 0.012 | 0.128 | 0.520 | 0.143 | 0.007 | 0.008 | 0.007 | 0.175 | 0.008 |
| `naive`, 1,024, list | 5 to 32 | 1,143,800 | 0.009 | 0.131 | 0.533 | 0.157 | 0.006 | 0.006 | 0.006 | 0.151 | 0.007 |
| `naive`, 1,024, list | 33 or more | 1,342,600 | 0.008 | 0.127 | 0.555 | 0.198 | 0.006 | 0.004 | 0.005 | 0.096 | 0.007 |
| `replay`, 1,024, stream | 2 to 4 | 4,172 | 0.578 | 0.017 | 0.016 | 0.136 | 0.009 | 0.007 | 0.010 | 0.226 | 0.010 |
| `replay`, 1,024, stream | 5 to 32 | 4,699 | 0.542 | 0.018 | 0.011 | 0.189 | 0.011 | 0.006 | 0.007 | 0.217 | 0.007 |
| `replay`, 1,024, stream | 33 or more | 3,812 | 0.526 | 0.017 | 0.011 | 0.228 | 0.016 | 0.004 | 0.007 | 0.190 | 0.007 |
| `replay`, 1,024, list | 2 to 4 | 198,600 | 0.538 | 0.019 | 0.012 | 0.204 | 0.009 | 0.007 | 0.008 | 0.204 | 0.008 |
| `replay`, 1,024, list | 5 to 32 | 1,143,800 | 0.518 | 0.019 | 0.012 | 0.228 | 0.010 | 0.007 | 0.008 | 0.197 | 0.008 |
| `replay`, 1,024, list | 33 or more | 1,342,600 | 0.520 | 0.019 | 0.012 | 0.291 | 0.012 | 0.004 | 0.008 | 0.134 | 0.009 |
| `naive`, 256, stream | 2 to 4 | 2,880 | 0.028 | 0.098 | 0.317 | 0.055 | 0.020 | 0.022 | 0.006 | 0.455 | 0.006 |
| `naive`, 256, stream | 5 to 32 | 1,949 | 0.020 | 0.097 | 0.334 | 0.068 | 0.015 | 0.021 | 0.006 | 0.438 | 0.006 |
| `naive`, 256, stream | 33 or more | 471 | 0.008 | 0.140 | 0.382 | 0.098 | 0.011 | 0.028 | 0.006 | 0.327 | 0.006 |
| `naive`, 256, list | 2 to 4 | 71,600 | 0.025 | 0.096 | 0.358 | 0.063 | 0.016 | 0.019 | 0.007 | 0.415 | 0.008 |
| `naive`, 256, list | 5 to 32 | 211,400 | 0.017 | 0.104 | 0.389 | 0.073 | 0.009 | 0.018 | 0.005 | 0.384 | 0.005 |
| `naive`, 256, list | 33 or more | 9,200 | 0.011 | 0.129 | 0.420 | 0.092 | 0.004 | 0.014 | 0.002 | 0.327 | 0.003 |
| `replay`, 256, stream | 2 to 4 | 2,880 | 0.519 | 0.018 | 0.021 | 0.051 | 0.010 | 0.019 | 0.011 | 0.351 | 0.012 |
| `replay`, 256, stream | 5 to 32 | 1,949 | 0.478 | 0.017 | 0.016 | 0.071 | 0.010 | 0.017 | 0.006 | 0.385 | 0.006 |
| `replay`, 256, stream | 33 or more | 471 | 0.480 | 0.015 | 0.008 | 0.113 | 0.006 | 0.004 | 0.015 | 0.359 | 0.015 |
| `replay`, 256, list | 2 to 4 | 71,600 | 0.485 | 0.015 | 0.019 | 0.076 | 0.006 | 0.017 | 0.009 | 0.374 | 0.010 |
| `replay`, 256, list | 5 to 32 | 211,400 | 0.466 | 0.016 | 0.018 | 0.095 | 0.009 | 0.015 | 0.010 | 0.371 | 0.010 |
| `replay`, 256, list | 33 or more | 9,200 | 0.463 | 0.017 | 0.010 | 0.127 | 0.019 | 0.015 | 0.010 | 0.338 | 0.010 |

The share of answers that are not two digits followed by the end symbol is at most 0.001 in every row.

**Does a memory become more lasting by itself as a family is met again? (D15.)** This is the reviewer's table, pooled over worlds. Each cell gives kept at return (D8) and, after the slash, the mean of the first 16 updates; the columns are the number of earlier visits to the family. The numbers of returns and of items in every cell, the same under every arm, are in `E44_L2_TABLES.md`, which also holds the rows of `naive` and `replay` at 256 visits and of `replay-wd0`. For example, at 1,024 visits after absences of 2 to 4 visits the five cells hold 59, 103, 154, 267 and 474 returns, with 216, 405, 585, 1,028 and 1,938 items. After absences of 33 or more at 256 visits, the cell of one earlier visit holds 7 returns and 19 items.

| Arm, visits | Absence | 1 | 2 to 3 | 4 to 7 | 8 to 15 | 16 or more |
|---|---|---|---|---|---|---|
| `naive`, 1,024 | 2 to 4 | 0.014 / 0.123 | 0.010 / 0.148 | 0.021 / 0.143 | 0.019 / 0.141 | 0.014 / 0.173 |
| `naive`, 1,024 | 5 to 32 | 0.005 / 0.097 | 0.027 / 0.103 | 0.018 / 0.109 | 0.008 / 0.106 | 0.010 / 0.116 |
| `naive`, 1,024 | 33 or more | 0.000 / 0.086 | 0.013 / 0.117 | 0.005 / 0.089 | 0.009 / 0.093 | 0.014 / 0.111 |
| `replay`, 1,024 | 2 to 4 | 0.093 / 0.331 | 0.360 / 0.492 | 0.480 / 0.555 | 0.566 / 0.637 | 0.714 / 0.754 |
| `replay`, 1,024 | 5 to 32 | 0.094 / 0.320 | 0.279 / 0.409 | 0.395 / 0.530 | 0.564 / 0.624 | 0.693 / 0.720 |
| `replay`, 1,024 | 33 or more | 0.115 / 0.337 | 0.265 / 0.452 | 0.409 / 0.542 | 0.565 / 0.612 | 0.658 / 0.696 |
| `naive-wd0`, 256 | 2 to 4 | 0.031 / 0.119 | 0.116 / 0.223 | 0.090 / 0.247 | 0.145 / 0.332 | 0.219 / 0.414 |
| `naive-wd0`, 256 | 5 to 32 | 0.028 / 0.105 | 0.035 / 0.105 | 0.038 / 0.151 | 0.070 / 0.190 | 0.166 / 0.334 |
| `naive-wd0`, 256 | 33 or more | 0.000 / 0.059 | 0.000 / 0.080 | 0.025 / 0.098 | 0.018 / 0.136 | 0.060 / 0.213 |

One sentence for each arm, with the figures it rests on:

- **`naive`.** Kept does not rise with the number of earlier visits in any class of absence: it stays between 0.000 and 0.027 at 1,024 visits, and between 0.000 and 0.115 at 256 visits, the 0.115 resting on 26 items. The figure after the first 16 updates is higher after 16 or more earlier visits than after one in every class and at both lengths, though not at every step between them: at 1,024 visits 0.173 against 0.123, 0.116 against 0.097, 0.111 against 0.086; at 256 visits 0.202 against 0.099, 0.151 against 0.102, 0.099 against 0.076.
- **`replay`.** Both quantities rise with the number of earlier visits in every class of absence and at both lengths, at every step. At 1,024 visits kept rises from 0.093, 0.094 and 0.115 to 0.714, 0.693 and 0.658, and the first 16 updates from 0.331, 0.320 and 0.337 to 0.754, 0.720 and 0.696.
- **`naive-wd0`.** Both quantities are higher after 16 or more earlier visits than after one, in every class at 256 visits, though not at every step: kept after 2 to 4 visits is 0.031, 0.116, 0.090, 0.145, 0.219, and after 33 or more 0.000, 0.000, 0.025, 0.018, 0.060. *(Corrected on 18 November after the reviewer's verification, `E44_L2_VERIFICATION_20261118.md` on the reviewer's branch.)* Kept rises from 0.031, 0.028 and 0.000 to 0.219, 0.166 and 0.060, and the first 16 updates from 0.119, 0.105 and 0.059 to 0.414, 0.334 and 0.213.
- **`replay-wd0`.** Both quantities rise with the number of earlier visits in every class at 256 visits, at every step. Kept rises from 0.113, 0.156 and 0.000 to 0.630, 0.630 and 0.592, and the first 16 updates from 0.150, 0.244 and 0.188 to 0.626, 0.637 and 0.618.

**The learning probe.** Each cell is the mean over the four probe families (52 to 55, visited in that order) of the first-visit level (D6), then the mean over the four long lives' worlds:

| Arm | After 64 visits | After 256 | After 1,024 |
|---|---|---|---|
| `naive` | 0.427 | 0.669 | 0.826 |
| `replay` | 0.267 | 0.570 | 0.743 |

Every branch is in `E44_L2_TABLES.md`.

**The sizes of the weights.** The norm of every tensor is recorded at every eighth visit of the lives without decay (`norms/`, `E44_L2_norms.csv`). The table gives the Euclidean norm of all weights but the byte table, then the norm of the GRU's recurrent matrix. The born weights are 52.8 and 18.5 as means over the eight lives without decay, and 52.3 and 18.3 as means over the four long lives, from which the rows with decay start.

| Arm | Lives | After 64 visits | After 256 visits | After 1,024 visits |
|---|---|---|---|---|
| `naive-wd0` | 8 | 88.8 / 35.7 | 126.3 / 52.8 | — |
| `replay-wd0` | 8 | 89.1 / 36.7 | 122.9 / 54.8 | — |
| `naive` (decay 0.25), the long lives | 4 | 67.6 / 18.6 | 70.8 / 17.6 | 76.1 / 17.2 |
| `replay` (decay 0.25), the long lives | 4 | 64.9 / 19.2 | 69.3 / 19.2 | 72.3 / 18.6 |

The rows with decay are read from the long lives' saved states. Without decay, both norms rise at every recorded age (64, 128, 192, 256 visits; `E44_L2_TABLES.md`). With decay, the means over the four long lives of the recurrent matrix's norm stay between 17.2 and 19.2. Single lives with decay range from 16.00 to 20.08 over their saved states (104 states of the 16 lives with decay), and the mean over the four lives of 256 visits under `replay` reaches 19.5. *(Corrected on 18 November after the reviewer's verification, `E44_L2_VERIFICATION_20261118.md` on the reviewer's branch.)* The states used to read them were deleted afterwards and are not committed, as the reviewer asked.

## 5. Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| PL2.1 | `naive` at 1,024 visits is *stuck* | 45% | does not hold: still learning, by D7 [+0.083, +0.114] |
| PL2.2 | `replay` is *higher* than `naive` in late level at 1,024 visits | 75% | does not hold: not distinguished, +0.047 [−0.011, +0.105] |
| PL2.3 | `replay` is *higher* in kept at return after long absences, 256 visits | 85% | holds: +0.511 [+0.409, +0.613], over 121 returns with 471 items after 33 or more; at 5 or more, +0.464 [+0.419, +0.508], over 637 returns with 2,420 items |
| PL2.4 | `naive-wd0` against `naive`, late level, 256 visits: *not distinguished* | 60% | does not hold by worlds: lower, −0.095 [−0.166, −0.024]; would hold when each birth is counted once |
| PL2.5 | `naive-wd0` is *higher* in kept at return after long absences, 256 visits | 30% | holds by worlds: +0.031 [+0.003, +0.058], over 121 returns with 471 items (18 right under `naive-wd0`, 4 under `naive`); not when each birth is counted once (section header). At 5 or more, +0.068 [+0.038, +0.097], over 637 returns with 2,420 items |
| PL2.6 | `replay` at 1,024 visits is *still learning* | 55% | holds, by D7 [+0.078, +0.137] |
| PL2.7 | The trend of learning under `naive` at 1,024 visits has an interval above zero | 25% | holds: [+0.083, +0.114] |
| PL2.8 | The comparison of `naive` and `replay` closes at 1,024 visits | 70% | does not hold: open |
| PL2.9 (the reviewer's) | `replay-wd0` against `replay`, late level, 256 visits: *not distinguished* | 60% | does not hold by worlds: lower, −0.143 [−0.245, −0.041]; would hold when each birth is counted once |

**The reviewer's expectations.**

- (i) `naive`'s late level at 1,024 visits between 0.40 and 0.55: **does not hold**. The worlds' late levels are 0.669, 0.756, 0.742 and 0.732 (mean 0.724).
- (ii) `replay`'s median first-visit level below 0.20 in at least three of eight worlds at 256 visits: **does not hold**. No world is below 0.20; the lowest is 0.328.
- (iii) Under `naive`, at returns after two or more visits, more than a tenth of answers are the right answer under the last family's modulus: **holds**, at 0.499 of 12,683 answers at 1,024 visits (0.469 to 0.519 by class of absence), and 0.329 of 5,300 at 256.

## 6. Where the instructions could not be followed exactly, and what I chose

- **The weight sizes.** They were read from states saved at every eighth visit rather than written by the runner. This avoided a third runner change; the reviewer agreed at 20:43 UTC on 17 November.
- **The trial's start.** The trial ran beside world 2's `naive` long life from the start, as the instructions allow.
- **The order of the probe branches.** The driver ran the probe branches when a place was free, so some ran before their lives' later visits. Each branch reads only its own saved state, so the order does not matter.
- **Committed states.** Only world 1's states after 256 and 1,024 visits remain in the tree, as the note says (`lives/*_state_v256.pt`, `*_state_v1024.pt`). Other states were committed while lives ran, as the instructions asked, and removed from the tree afterwards; they remain in the branch's history.
- **D14's chance line.** The never-met modulus is family 55's, a probe family. No life's schedule meets it; the probe branches do, but D14 reads only the lives.

## 7. What this cannot show

- Anything about the sweep; release 2 covers that, and a helper session runs the sweep's long lives.
- A tuned replay share, a tuned store or the best decay: one value each was run.
- Anything beyond 1,024 visits or 52 families, or at full scale.
- **Why D15 rises.** Figures rise with the number of earlier visits under all four arms:
  - under `naive`, only the figure over the first 16 updates;
  - under `replay`, `naive-wd0` and `replay-wd0`, both figures.

  This table cannot show whether the rise comes from the earlier visits themselves or from what accompanies them. A family met more often has more of its items in the store (under `replay` and `replay-wd0`), and its returns come later in the life. In the long lives the median visit of the returns after one earlier visit and after 16 or more is:
  - 355 and 451 after absences of 2 to 4 visits;
  - 479.5 and 563 after 5 to 32;
  - 639.5 and 650.5 after 33 or more.

  In the lives of 256 visits the same medians are 114 and 122, 113 and 152.5, and 180 and 211. The reviewer's verification split the returns into the two halves of the life (not a definition of the note):
  - under `naive` at 1,024 visits, the figure over the first 16 updates rises from 0.087 to 0.146 in the first half (85 and 641 returns), but only from 0.118 to 0.123 in the second (84 and 800 returns);
  - under `replay`, kept at return rises within each half (0.105 to 0.713; 0.096 to 0.671);
  - under `naive-wd0` at 256 visits, kept at return rises within each half (0.026 to 0.229; 0.030 to 0.153).

  The age of the life may therefore account for most of `naive`'s rise, and not for the rise under `replay` and `naive-wd0` within these halves. The split is coarse, the halves still differ in age within them, and the store's contents remain confounded under `replay`. The table separates neither the store nor the age of the life from the number of earlier visits under any arm. *(Corrected on 18 November after the reviewer's verification, `E44_L2_VERIFICATION_20261118.md` on the reviewer's branch.)*
- **The learning rate.** It was 0.001 in every life and was never varied.
- **The births.** There were three seeds, one birth file each, and the intervals count worlds (section header).
- **The visit length.** It is 256 updates, calibrated under `naive`. Under `replay` a visit teaches 6,144 new items, the number of the calibration at 128 updates a visit. `replay`'s median first-visit level is lower than `naive`'s in every world: −0.216 [−0.265, −0.167] at 256 visits and −0.128 [−0.201, −0.055] at 1,024. The comparison at this visit length is therefore not neutral between the arms.
- **The lives without decay.** They were read only up to 256 visits; no statement on them extends to 1,024.
- **No end state.** Both long arms were still learning at 1,024 visits by D7, so no level reached is an end state.
- **The widths of the intervals.** The intervals beside the differences read are wide (section header); a difference that is not distinguished is not a finding of equality.
- **Composition**, which this world does not teach.
- **Decay alone beside what is kept.** Release 1 does not compute it (`--decay-distances none`; the audit, 1.3 c).
- **The level of evidence** is E2. *(Corrected on 18 November after the reviewer's verification, `E44_L2_VERIFICATION_20261118.md` on the reviewer's branch.)*

## 8. Committed files

The list below is `git ls-tree -r --name-only HEAD` for the files this job added or changed. It was taken at the commit that first added this report (without this list); the list was then appended in the next commit, which changes no other file.

At commit 1e4eceae (`git ls-tree -r --name-only 1e4eceae`, restricted to this job's paths):

```
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/E44_L2.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/E44_L2_REPORT.md
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/E44_L2_TABLES.md
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/E44_L2_norms.csv
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/E44_L2_score_by_age_blocks.csv
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/birth_w2_93683.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/birth_w2_93683.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/birth_w2_93683.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/birth_w3_116089.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/birth_w3_116089.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/birth_w3_116089.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/lw_naive_w2_v8_turn.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/lw_naive_w2_v8_turn.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/lw_naive_w2_v8_turn_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/lw_naive_w3_v8_turn.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/lw_naive_w3_v8_turn.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/births/lw_naive_w3_v8_turn_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/E44_L2_CHECKS.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/E44_L2_CHECKS_attempt1_harness_error.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/a_base/lw_naive_w1_v8_turn.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/a_base/lw_naive_w1_v8_turn.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/a_base/lw_naive_w1_v8_turn_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/a_new/a_new.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/a_new/a_new.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/a_new/a_new.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/a_new/lw_naive_w1_v8_turn.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/a_new/lw_naive_w1_v8_turn.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/a_new/lw_naive_w1_v8_turn_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_default/b_default.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_default/b_default.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_default/b_default.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_default/lw_naive_w5_v32.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_default/lw_naive_w5_v32.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_default/lw_naive_w5_v32_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_k4_d1248/b_k4_d1248.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_k4_d1248/b_k4_d1248.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_k4_d1248/b_k4_d1248.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_k4_d1248/lw_naive_w5_v32.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_k4_d1248/lw_naive_w5_v32.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_k4_d1248/lw_naive_w5_v32_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_k4_none/b_k4_none.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_k4_none/b_k4_none.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_k4_none/b_k4_none.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_k4_none/lw_naive_w5_v32.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_k4_none/lw_naive_w5_v32.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/b_k4_none/lw_naive_w5_v32_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c3_naive/c3_naive.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c3_naive/c3_naive.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c3_naive/c3_naive.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c3_naive/lw_naive_w1_v1.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c3_naive/lw_naive_w1_v1.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c3_naive/lw_naive_w1_v1_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c3_replay_rho0/c3_replay_rho0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c3_replay_rho0/c3_replay_rho0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c3_replay_rho0/c3_replay_rho0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c3_replay_rho0/lw_replay_w1_v1_rho0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c3_replay_rho0/lw_replay_w1_v1_rho0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c3_replay_rho0/lw_replay_w1_v1_rho0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_cut/c5_cut.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_cut/c5_cut.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_cut/c5_cut.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_cut/c5_resumed.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_cut/c5_resumed.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_cut/c5_resumed.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_cut/lw_replay_w1_v8_rho0.5_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_cut/lw_replay_w1_v8_rho0.5_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_cut/lw_replay_w1_v8_rho0.5_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_whole/c5_whole.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_whole/c5_whole.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_whole/c5_whole.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_whole/lw_replay_w1_v8_rho0.5_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_whole/lw_replay_w1_v8_rho0.5_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c5_whole/lw_replay_w1_v8_rho0.5_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_naive/c7_naive.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_naive/c7_naive.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_naive/c7_naive.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_naive/lw_naive_w5_v4.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_naive/lw_naive_w5_v4.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_naive/lw_naive_w5_v4_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_naive_wd0/c7_naive_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_naive_wd0/c7_naive_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_naive_wd0/c7_naive_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_naive_wd0/lw_naive_w5_v4_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_naive_wd0/lw_naive_w5_v4_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_naive_wd0/lw_naive_w5_v4_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay/c7_replay.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay/c7_replay.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay/c7_replay.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay/lw_replay_w5_v4_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay/lw_replay_w5_v4_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay/lw_replay_w5_v4_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay_wd0/c7_replay_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay_wd0/c7_replay_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay_wd0/c7_replay_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay_wd0/lw_replay_w5_v4_rho0.5_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay_wd0/lw_replay_w5_v4_rho0.5_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/c7_replay_wd0/lw_replay_w5_v4_rho0.5_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/checks/check_jobs.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v1024.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v1024.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v1024.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v1024.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v1024_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v1024_state_v1024.pt
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v1024_state_v256.pt
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v256_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v256_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v256_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v256_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v256_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v256_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w2_v1024.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w2_v1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w2_v1024.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w2_v1024.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w2_v1024.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w2_v1024_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w2_v256_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w2_v256_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w2_v256_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w2_v256_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w2_v256_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w2_v256_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w3_v1024.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w3_v1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w3_v1024.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w3_v1024.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w3_v1024.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w3_v1024_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w3_v256_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w3_v256_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w3_v256_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w3_v256_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w3_v256_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w3_v256_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w4_v1024.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w4_v1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w4_v1024.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w4_v1024.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w4_v1024.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w4_v1024_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w4_v256_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w4_v256_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w4_v256_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w4_v256_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w4_v256_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w4_v256_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w5_v256.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w5_v256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w5_v256.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w5_v256.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w5_v256.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w5_v256_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w5_v256_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w5_v256_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w5_v256_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w5_v256_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w5_v256_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w5_v256_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w6_v256.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w6_v256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w6_v256.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w6_v256.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w6_v256.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w6_v256_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w6_v256_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w6_v256_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w6_v256_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w6_v256_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w6_v256_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w6_v256_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w7_v256.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w7_v256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w7_v256.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w7_v256.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w7_v256.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w7_v256_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w7_v256_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w7_v256_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w7_v256_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w7_v256_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w7_v256_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w7_v256_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w8_v256.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w8_v256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w8_v256.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w8_v256.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w8_v256.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w8_v256_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w8_v256_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w8_v256_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w8_v256_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w8_v256_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w8_v256_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w8_v256_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v1024_rho0.5.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v1024_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v1024_rho0.5.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v1024_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v1024_rho0.5.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v1024_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v1024_rho0.5_state_v1024.pt
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v1024_rho0.5_state_v256.pt
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v256_rho0.5_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v256_rho0.5_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v256_rho0.5_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v256_rho0.5_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v256_rho0.5_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v256_rho0.5_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w2_v1024_rho0.5.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w2_v1024_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w2_v1024_rho0.5.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w2_v1024_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w2_v1024_rho0.5.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w2_v1024_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w2_v256_rho0.5_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w2_v256_rho0.5_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w2_v256_rho0.5_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w2_v256_rho0.5_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w2_v256_rho0.5_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w2_v256_rho0.5_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w3_v1024_rho0.5.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w3_v1024_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w3_v1024_rho0.5.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w3_v1024_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w3_v1024_rho0.5.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w3_v1024_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w3_v256_rho0.5_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w3_v256_rho0.5_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w3_v256_rho0.5_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w3_v256_rho0.5_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w3_v256_rho0.5_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w3_v256_rho0.5_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w4_v1024_rho0.5.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w4_v1024_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w4_v1024_rho0.5.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w4_v1024_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w4_v1024_rho0.5.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w4_v1024_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w4_v256_rho0.5_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w4_v256_rho0.5_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w4_v256_rho0.5_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w4_v256_rho0.5_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w4_v256_rho0.5_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w4_v256_rho0.5_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w5_v256_rho0.5.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w5_v256_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w5_v256_rho0.5.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w5_v256_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w5_v256_rho0.5.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w5_v256_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w5_v256_rho0.5_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w5_v256_rho0.5_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w5_v256_rho0.5_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w5_v256_rho0.5_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w5_v256_rho0.5_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w5_v256_rho0.5_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w6_v256_rho0.5.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w6_v256_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w6_v256_rho0.5.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w6_v256_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w6_v256_rho0.5.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w6_v256_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w6_v256_rho0.5_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w6_v256_rho0.5_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w6_v256_rho0.5_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w6_v256_rho0.5_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w6_v256_rho0.5_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w6_v256_rho0.5_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w7_v256_rho0.5.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w7_v256_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w7_v256_rho0.5.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w7_v256_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w7_v256_rho0.5.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w7_v256_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w7_v256_rho0.5_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w7_v256_rho0.5_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w7_v256_rho0.5_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w7_v256_rho0.5_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w7_v256_rho0.5_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w7_v256_rho0.5_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w8_v256_rho0.5.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w8_v256_rho0.5.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w8_v256_rho0.5.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w8_v256_rho0.5.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w8_v256_rho0.5.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w8_v256_rho0.5_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w8_v256_rho0.5_wd0.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w8_v256_rho0.5_wd0.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w8_v256_rho0.5_wd0.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w8_v256_rho0.5_wd0.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w8_v256_rho0.5_wd0.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w8_v256_rho0.5_wd0_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/logs/check_a_base.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/logs/check_driver.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/logs/roster_driver.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/logs/saver.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w1_v1024_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w1_v256_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w2_v1024_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w2_v256_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w3_v1024_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w3_v256_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w4_v1024_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w4_v256_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w5_v256_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w5_v256_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w6_v256_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w6_v256_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w7_v256_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w7_v256_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w8_v256_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_naive_w8_v256_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w1_v1024_rho0.5_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w1_v256_rho0.5_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w2_v1024_rho0.5_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w2_v256_rho0.5_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w3_v1024_rho0.5_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w3_v256_rho0.5_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w4_v1024_rho0.5_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w4_v256_rho0.5_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w5_v256_rho0.5_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w5_v256_rho0.5_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w6_v256_rho0.5_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w6_v256_rho0.5_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w7_v256_rho0.5_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w7_v256_rho0.5_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w8_v256_rho0.5_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/norms/lw_replay_w8_v256_rho0.5_wd0_norms.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br1024_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br1024_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br1024_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br1024_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br1024_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br1024_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br256_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br256_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br256_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br256_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br256_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br256_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br64_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br64_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br64_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br64_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br64_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w1_v1024_br64_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br1024_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br1024_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br1024_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br1024_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br1024_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br1024_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br256_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br256_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br256_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br256_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br256_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br256_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br64_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br64_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br64_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br64_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br64_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w2_v1024_br64_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br1024_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br1024_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br1024_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br1024_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br1024_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br1024_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br256_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br256_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br256_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br256_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br256_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br256_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br64_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br64_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br64_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br64_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br64_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w3_v1024_br64_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br1024_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br1024_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br1024_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br1024_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br1024_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br1024_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br256_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br256_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br256_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br256_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br256_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br256_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br64_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br64_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br64_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br64_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br64_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_naive_w4_v1024_br64_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br1024_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br1024_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br1024_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br1024_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br1024_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br1024_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br256_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br256_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br256_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br256_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br256_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br256_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br64_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br64_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br64_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br64_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br64_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w1_v1024_rho0.5_br64_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br1024_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br1024_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br1024_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br1024_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br1024_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br1024_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br256_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br256_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br256_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br256_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br256_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br256_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br64_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br64_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br64_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br64_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br64_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w2_v1024_rho0.5_br64_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br1024_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br1024_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br1024_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br1024_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br1024_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br1024_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br256_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br256_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br256_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br256_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br256_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br256_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br64_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br64_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br64_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br64_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br64_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w3_v1024_rho0.5_br64_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br1024_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br1024_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br1024_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br1024_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br1024_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br1024_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br256_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br256_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br256_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br256_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br256_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br256_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br64_g52-53-54-55.done
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br64_g52-53-54-55.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br64_g52-53-54-55.log
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br64_g52-53-54-55.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br64_g52-53-54-55.starts.jsonl
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/probe/lw_replay_w4_v1024_rho0.5_br64_g52-53-54-55_list.npz
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/roster_jobs.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w1_v1024_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w1_v1024_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w1_v256_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w2_v1024_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w2_v1024_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w2_v256_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w3_v1024_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w3_v1024_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w3_v256_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w4_v1024_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w4_v1024_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w4_v256_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w5_v256_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w5_v256_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w6_v256_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w6_v256_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w7_v256_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w7_v256_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w8_v256_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_naive_w8_v256_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w1_v1024_rho0.5_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w1_v1024_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w1_v256_rho0.5_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w2_v1024_rho0.5_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w2_v1024_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w2_v256_rho0.5_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w3_v1024_rho0.5_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w3_v1024_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w3_v256_rho0.5_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w4_v1024_rho0.5_V1024.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w4_v1024_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w4_v256_rho0.5_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w5_v256_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w5_v256_rho0.5_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w6_v256_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w6_v256_rho0.5_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w7_v256_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w7_v256_rho0.5_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w8_v256_rho0.5_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/scores/lw_replay_w8_v256_rho0.5_wd0_V256.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/trial/E44_L2_TRIAL.json
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/trial/E44_L2_TRIAL_REPORT.md
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/trial/E44_L2_TRIAL_TABLES.md
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/trial/E44_L2_TRIAL_first_visits.csv
MOD97_LOCAL_RULE_20261106/exp45/e44/L2/trial/E44_L2_TRIAL_score_by_age.csv
MOD97_LOCAL_RULE_20261106/smallworld/l2_checks.py
MOD97_LOCAL_RULE_20261106/smallworld/l2_compare.py
MOD97_LOCAL_RULE_20261106/smallworld/l2_driver.py
MOD97_LOCAL_RULE_20261106/smallworld/l2_lives.py
MOD97_LOCAL_RULE_20261106/smallworld/l2_report.py
MOD97_LOCAL_RULE_20261106/smallworld/l2_restart.sh
MOD97_LOCAL_RULE_20261106/smallworld/l2_score.py
MOD97_LOCAL_RULE_20261106/smallworld/sw_run.py
OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md
```
