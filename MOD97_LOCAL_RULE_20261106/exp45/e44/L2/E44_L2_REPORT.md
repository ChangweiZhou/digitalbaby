# E44, job L2, release 1: the first long lives — report (18 November 2026)

Branch `claude/e44-l2-first-long-lives`. The roster note is `OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md`, revision 2, on this branch. The worker entered it from the reviewer's audit at ef8b512b while the coordinator was away. The note's definitions D1 to D15 are used throughout and are not repeated here.

Every figure below comes from one of two files beside this report:

- `E44_L2.json`, with `E44_L2_TABLES.md` as its tables, written by `python3 smallworld/l2_report.py final`;
- for the trial, `trial/E44_L2_TRIAL.json` and `trial/E44_L2_TRIAL_REPORT.md`.

Both are computed from the lives' event files and list files in `lives/`. A statement about arms is made over worlds, with a 90% interval (Student's t over worlds), as the note fixes it. Four worlds at 1,024 visits and eight at 256 give wide intervals. "Stuck" means that no improving trend was found at that precision.

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
| Lost to interruptions (estimated) | about 1.4 | about 410 visits lost before a restart and run again |

The forecast was 48.2 before the trial and 48.0 after it.

**Interruptions and resumes.** The record of each start is in `lives/*.starts.jsonl`, and each receipt lists its resumes.

| When (UTC) | What happened | Lives affected | What was done |
|---|---|---|---|
| 17 Nov, 21:15 | The machine was stopped after the session went idle | The trial's two lives and world 2's `naive` long life, at visit 57, before any state was saved | Started again from the beginning at 22:07 |
| 18 Nov, 00:57 | Container restart | World 1's `naive` and `replay` and world 2's `naive`, at visits 790 to 797 | Continued at 00:59 with `life --resume` from their states after 768 visits |
| 18 Nov, about 05:20 | Container restart | World 4's two long lives, at visits 309 and 315 | Resumed from their states after 256 visits |
| 18 Nov, about 05:20 | Container restart | World 5's `replay` life, at visit 37, with no state | Started again from the beginning |

The note's check 5 showed that a resumed life equals an uninterrupted one in weights, optimizer state, events, learned batches and list. No figure depends on an interruption.

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
- check 5: the resume of a `replay-wd0` life is exact;
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

**Stuck or still learning (D13).** The improving side is above zero.

| Arm | Visits | Score trend D5 [90% interval] | Trend of learning D7 [90% interval] | Statement |
|---|---|---|---|---|
| `naive` | 1,024 | +0.015 [−0.016, +0.046] | +0.098 [+0.083, +0.114] | still learning |
| `replay` | 1,024 | +0.052 [−0.006, +0.109] | +0.108 [+0.078, +0.137] | still learning |
| `naive` | 256 | +0.047 [−0.009, +0.102] | +0.070 [+0.003, +0.136] | still learning |
| `replay` | 256 | +0.096 [+0.025, +0.166] | +0.041 [−0.034, +0.115] | still learning |
| `naive-wd0` | 256 | +0.031 [−0.042, +0.104] | −0.049 [−0.119, +0.021] | stuck |
| `replay-wd0` | 256 | −0.006 [−0.081, +0.069] | −0.059 [−0.131, +0.014] | stuck |

No interval lies wholly below zero, so no arm is called *declining*.

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
| the sum or difference without the reduction | 0.126 | — |
| another two-digit number | 0.196 | 0.212 |
| chance (a never-met modulus in the place of the last family's) | 0.007 | 0.008 |

On the list at 1,024 visits (2,685,000 answers), `naive` gives the last family's modulus 0.543, and `replay` an earlier family's modulus 0.258. All absences and both lengths are in `E44_L2_TABLES.md`.

**Does a memory become more lasting by itself as a family is met again? (D15.)** This is the reviewer's table, pooled over worlds, with the full table in `E44_L2_TABLES.md`. Each cell gives kept at return (D8) and, after the slash, the mean of the first 16 updates; the columns are the number of earlier visits to the family.

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

- **`naive`.** At 1,024 visits neither quantity rises with the number of earlier visits in any class of absence: kept stays between 0.000 and 0.027, and the first 16 updates between 0.086 and 0.173. At 256 visits kept does not rise either: it stays between 0.000 and 0.115, and the 0.115 rests on 26 items. The first 16 updates at 256 visits rise from 0.099 to 0.202 after absences of 2 to 4 visits, and not in the other two classes (0.102, 0.086, 0.101, 0.097, 0.151 and 0.076, 0.071, 0.061, 0.073, 0.099).
- **`replay`.** Both quantities rise with the number of earlier visits in every class of absence and at both lengths. At 1,024 visits kept rises from 0.093, 0.094 and 0.115 to 0.714, 0.693 and 0.658, and the first 16 updates from 0.331, 0.320 and 0.337 to 0.754, 0.720 and 0.696.
- **`naive-wd0`.** Both quantities rise with the number of earlier visits in every class at 256 visits. Kept rises from 0.031, 0.028 and 0.000 to 0.219, 0.166 and 0.060, and the first 16 updates from 0.119, 0.105 and 0.059 to 0.414, 0.334 and 0.213.
- **`replay-wd0`.** Both quantities rise with the number of earlier visits in every class at 256 visits. Kept rises from 0.113, 0.156 and 0.000 to 0.630, 0.630 and 0.592, and the first 16 updates from 0.150, 0.244 and 0.188 to 0.626, 0.637 and 0.618.

**The learning probe.** Each cell is the mean over the four probe families (52 to 55, visited in that order) of the first-visit level (D6), then the mean over the four long lives' worlds:

| Arm | After 64 visits | After 256 | After 1,024 |
|---|---|---|---|
| `naive` | 0.427 | 0.669 | 0.826 |
| `replay` | 0.267 | 0.570 | 0.743 |

Every branch is in `E44_L2_TABLES.md`.

**The sizes of the weights.** The norm of every tensor is recorded at every eighth visit of the lives without decay (`norms/`, `E44_L2_norms.csv`). The table gives the Euclidean norm of all weights but the byte table, then the norm of the GRU's recurrent matrix; the born weights are 52.8 and 18.5.

| Arm | Lives | After 64 visits | After 256 visits | After 1,024 visits |
|---|---|---|---|---|
| `naive-wd0` | 8 | 88.8 / 35.7 | 126.3 / 52.8 | — |
| `replay-wd0` | 8 | 89.1 / 36.7 | 122.9 / 54.8 | — |
| `naive` (decay 0.25), the long lives | 4 | 67.6 / 18.6 | 70.8 / 17.6 | 76.1 / 17.2 |
| `replay` (decay 0.25), the long lives | 4 | 64.9 / 19.2 | 69.3 / 19.2 | 72.3 / 18.6 |

The rows with decay are read from the long lives' saved states. Without decay, both norms rise at every recorded age (64, 128, 192, 256 visits; `E44_L2_TABLES.md`). With decay, the recurrent matrix stays between 17.2 and 19.2. The states used to read them were deleted afterwards and are not committed, as the reviewer asked.

## 5. Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| PL2.1 | `naive` at 1,024 visits is *stuck* | 45% | does not hold: still learning, by D7 [+0.083, +0.114] |
| PL2.2 | `replay` is *higher* than `naive` in late level at 1,024 visits | 75% | does not hold: not distinguished, +0.047 [−0.011, +0.105] |
| PL2.3 | `replay` is *higher* in kept at return after long absences, 256 visits | 85% | holds: +0.511 [+0.409, +0.613]; at 5 or more, +0.464 [+0.419, +0.508] |
| PL2.4 | `naive-wd0` against `naive`, late level, 256 visits: *not distinguished* | 60% | does not hold: lower, −0.095 [−0.166, −0.024] |
| PL2.5 | `naive-wd0` is *higher* in kept at return after long absences, 256 visits | 30% | holds: +0.031 [+0.003, +0.058]; at 5 or more, +0.068 [+0.038, +0.097] |
| PL2.6 | `replay` at 1,024 visits is *still learning* | 55% | holds, by D7 [+0.078, +0.137] |
| PL2.7 | The trend of learning under `naive` at 1,024 visits has an interval above zero | 25% | holds: [+0.083, +0.114] |
| PL2.8 | The comparison of `naive` and `replay` closes at 1,024 visits | 70% | does not hold: open |
| PL2.9 (the reviewer's) | `replay-wd0` against `replay`, late level, 256 visits: *not distinguished* | 60% | does not hold: lower, −0.143 [−0.245, −0.041] |

**The reviewer's expectations.**

- (i) `naive`'s late level at 1,024 visits between 0.40 and 0.55: **does not hold**. The worlds' late levels are 0.669, 0.756, 0.742 and 0.732 (mean 0.724).
- (ii) `replay`'s median first-visit level below 0.20 in at least three of eight worlds at 256 visits: **does not hold**. No world is below 0.20; the lowest is 0.328.
- (iii) Under `naive`, at returns after two or more visits, more than a tenth of answers are the right answer under the last family's modulus: **holds**, at 0.499 of 12,683 answers at 1,024 visits.

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
- Whether the rise of D15 under `replay` and `naive-wd0` comes from the family's earlier visits themselves or from what accompanies them. A family met more often has more of its items in the store, and its weights have grown with it.

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
