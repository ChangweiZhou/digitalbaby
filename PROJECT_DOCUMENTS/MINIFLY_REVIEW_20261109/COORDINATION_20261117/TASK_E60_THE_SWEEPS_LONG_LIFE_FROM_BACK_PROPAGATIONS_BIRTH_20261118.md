# E60: the long life under the sweep that starts from back-propagation's birth, so that two lives differ in the rule alone (18 November 2026, revision 2)

**Status.** Written by the outside reviewer at the coordinator's decision 21 (`COORDINATOR_DECISIONS_20261118.md` on the default branch at 0cf288ce). Revision 1 was checked by an agent of the reviewer that had not seen the reviewer's reasoning (`CHECK_OF_TASK_E60_BRIEF_20261118.md` with `check_e60_brief.py` beside this file). The check found seventeen defects and gave a text for each; revision 2 enters them, and the last section lists them. The check is not the independent audit that the programme's rules ask for. **The brief is released only by the coordinator, and no life is started before that.** The number E60 is the reviewer's; the coordinator may renumber it. The job is the second half of item 23 of `PLAN.md` beside this file and is meant for the helper of the long lives, on its present branch. The level of evidence is the one the coordinator's briefs call E2: a first measurement, which confirms nothing.

## 1. The question

Mr. Xu's statement of 15 November (requirement B1 of the plan): our learner may be slower than the conventional one at first and overtake it in the course of continued learning.

The sweep's long lives have been run and verified (`$R/E44_L2_SWEEP_VERIFICATION_20261118.md`; `$R/MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md`, section 33.16). Without the store the arm `sweep` ended below the arm `naive` in all four worlds, and no overtaking was seen up to 1,024 visits. The verification's first limit is that the two arms differ in two things at once: the rule by which the life learns, and the birth from which it starts. The sweep's births answer less than back-propagation's before a life begins. A difference between those arms cannot be assigned to the rule.

**This job asks: when the life under the sweep starts from the same birth as the life under back-propagation, is it lower, higher or not distinguished from it over the first 64 visits (the early level), and at 256 and at 1,024 visits (the late level), and how does that difference stand beside the difference that the sweep's own birth gave?**

## 2. Terms and shell names

`M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`; `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch; files under `$R` are read from `$V`); `W1=origin/claude/e44-l2-first-long-lives` (the main worker's branch). Files of other branches are read with `git show`; no other branch is checked out.

- *Back-propagation* (rule B) is the exact gradient. *The sweep* (rule S) passes the error backward through each linear map by the signs of its weights times their mean size.
- A *visit* is 256 updates on one *family* (add, sub, max and min under one modulus). In each update the network answers 16 *asked* items and then learns from 48 *taught* items. A *life* is 1,024 visits in a *world*, which fixes the families and the order of the visits. A *reduction item* is an asked item of add or sub whose answer needs the reduction by the modulus. A later visit to a family is a *continuation* if it follows the family's previous visit directly and a *return* otherwise.
- A *birth* is the network that E41 trained beforehand under one rule, at the half width (H = 256, U = 128), on half the pairs of the *birth family* (add and sub modulo 89; E41's first family), for 40,960 updates: `$M/exp45/e41/e41_<B|S>_h256_f50_<seed>.pt`. The roster note gives each world a seed: world 1, 73129; world 2, 93683; world 3, 116089 (world 4 uses world 1's births and is not part of this job).
- An *arm* is a birth together with the rule of the life. `naive`: the birth of rule B, a life by back-propagation. `sweep`: the birth of rule S, a life by the sweep. `sweep-after-naive`: the birth of rule B, a life by the sweep. None of the three has a store. Every life starts with a fresh optimizer.
- *The roster note* is `OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md` at the root of your branch (revision 2); its definitions hold here. This brief uses: D2, the share right on the asked reduction items of a visit; D4, the *late level*, the mean of D2 over the last quarter of the visits of a life read at a length V (256 or 1,024 visits); D5, the *score trend*: the least-squares slope of D2 on ln(v + 1) over the returns among visits V/2 to V − 1 (the scorer's `score_trend`; `score_trend_all_visits` is reported beside it and not used); D7, the *trend of learning* (`trend_of_learning`); D13, read for an arm over the three worlds: *still learning* if the interval over three worlds of D5 or of D7 lies above zero, *stuck* otherwise, and *declining* for an interval wholly below zero. The *early level* is the mean of D2 over visits 0 to 63, as in section 5.7 of your report.
- *The interval over three worlds* of a quantity is the two-sided 90% interval of the mean of its three unrounded per-world values: mean ± t × s / √3, with s the sample standard deviation (divisor 2) and t = 2.920, the 0.95 quantile of Student's t with 2 degrees of freedom; that is the mean ± 1.686 s. It is computed with `l2_score.interval90` on exactly three values and read with `l2_score.verdict`. A difference is *lower* or *higher* when the interval excludes zero and *not distinguished* otherwise. If a per-world value is missing, the quantity is *not read*, and no sentence or prediction that uses it is decided. No other word is used for a difference: not "level", "the same", "behind", "ahead", "grown", "shrunk" or "not changed".

## 3. What the record says

Every figure of this section was read by the reviewer from committed files, and the check recomputed each of them (`check_e60_brief.py`).

- **The verified lives, worlds 1 to 3** (your `final/E44_L2_SWEEP_EARLY.md` at 738e2d61; `naive` then `sweep`). Early level: 0.782 and 0.772; 0.717 and 0.620; 0.633 and 0.639. Late level at 256 visits: 0.582 and 0.589; 0.696 and 0.512; 0.692 and 0.611. Late level at 1,024 visits: 0.669 and 0.650; 0.756 and 0.628; 0.742 and 0.682.
- **`sweep` − `naive` over these three worlds.** Early: −0.0100, −0.0977, +0.0064; mean −0.034 [−0.128, +0.061], not distinguished. At 256 visits: +0.0076, −0.1842, −0.0810; mean −0.086 [−0.248, +0.076], not distinguished. At 1,024 visits: −0.0191, −0.1279, −0.0590; mean −0.069 [−0.162, +0.024], not distinguished. With three worlds the interval is wide: the difference at 1,024 visits, which is lower over four worlds by the roster note's interval, is not distinguished over three.
- **The calibration of E44** (`$R/E44_GATE_FOR_THE_SWEEP_20261117.md`, with `e44_gate_sweep.py`) is the only place where `sweep-after-naive` has been run: lives of eight visits, each visit a new family, in eight worlds, all from the births of seed 73129. At 256 updates a visit the gate's figure (the median over families 1 to 7 of the first-visit level on reduction items, the first-visit level of a family being the mean of add's and sub's share right over the last 64 updates of its first visit), averaged over the eight worlds, is 0.196 for `sweep-after-naive` (0.117 to 0.275 over the worlds; inside the gate's band of 0.20 to 0.90 in 3 of 8), 0.453 for `naive` and 0.614 for `sweep`. `sweep-after-naive` − `naive` is −0.257 (90% interval over eight worlds −0.307 to −0.206; lower in all eight). `sweep` − `sweep-after-naive` is +0.418 (+0.367 to +0.469; higher in all eight). Within a first visit, the family's figure, averaged over the 56 first visits of families 1 to 7, rises through the eight eighths of the visit from 0.038 to 0.249 under `sweep-after-naive` and from 0.071 to 0.489 under `naive`.
- **What does not exist:** a life of `sweep-after-naive` longer than eight visits; one with returns to a family; one from the births of seeds 93683 or 116089; one with the roster's settings (the list every 8 visits, saved states). The reverse arm, a life by back-propagation from the sweep's birth, has never been run, and the runner has no such arm (`load_learner` in `sw_run.py` refuses a birth whose rule is not the one the arm names).

## 4. The runs

| Arm | Worlds | Visits | Birth (`--learner`) |
|---|---|---|---|
| `sweep-after-naive` | 1, 2, 3 | 1,024 | `$M/exp45/e41/e41_B_h256_f50_<seed>.pt`, seed 73129, 93683, 116089: the file that the receipt of `naive` in the same world records |

Three lives. Each life is started from `$M/smallworld` by

`python3 sw_run.py life sweep-after-naive <w> 1024 --world lw --H 256 --U 128 --learner <absolute path of $M/exp45/e41/e41_B_h256_f50_<seed>.pt> --visit-updates 256 --wd 0.25 --lr 0.001 --list-every 8 --decay-distances none --save-states 64,128,256,384,512,640,768,896,1024 --out <absolute path of $M/exp45/e44/L2/sweep_after_naive/lives>`,

run through the main worker's driver `smallworld/l2_driver.py` (unchanged) with a jobs file `sweep_after_naive/san_jobs.json` of the form of your `sweep/sweep_jobs.json` (stem `lw_sweep-after-naive_w<w>_v1024`). The driver continues an interrupted life from its latest state and passes `--out`. A life resumed by hand is resumed with `life --resume <state> --out <the life's folder>`; without `--out` the runner writes into `exp45/e39/`.

**You do not change the runner or the scorer.** If you find you need to, stop and write to the reviewer. Order: the checks; the three lives together. At your measured 3.65 to 3.80 CPU-hours a life, about 11.5 CPU-hours; with the checks, about 12. If check 6 forecasts more than 15, stop and ask.

Outputs go under `$M/exp45/e44/L2/sweep_after_naive/`; new scripts are named `smallworld/l2_san*`; no file of your earlier work or of the main worker is overwritten.

## 5. Checks before the lives (a receipt for each, in `sweep_after_naive/checks/`)

1. **Sources.** Each file named in `sources_sha256` of your `sweep/checks/E44_L2_SWEEP_CHECKS.json` (five under `smallworld/` and seven in `$M`) and each file named in its `checks.1.unchanged_files_sha256` (`l2_score.py`, `sw_score.py`, `l2_driver.py`, `sw_run.py`) has the recorded SHA-256. The receipt of this check also lists the SHA-256 of every new script `smallworld/l2_san*`.
2. **The birth of rule B is admitted, and the sweep's birth is refused** (the admission is the coordinator's condition in decision 21). For each of worlds 1, 2 and 3: a life of one visit of `sweep-after-naive` with the birth of rule B (the command of section 4 with `1` visit and `--out …/checks/c2_w<w>`) starts, and the object `learner` of its receipt equals, key for key (file, file_sha256, tensors_sha256, rule, seed, final_row with the birth's A and R, and the rest), the object `learner` in the receipt of `naive` of the same world on `$W1`. And for world 1: the same command with the sweep's birth `e41_S_h256_f50_73129.pt` is refused by the runner; the receipt of this check holds the text of the refusal. The runner does not compare the birth's seed with the world. Each long life's receipt is therefore compared in the same way with that of `naive`, and the report states the result for the three lives.
3. **The calibration reproduced.** On this host, `python3 sw_run.py life sweep-after-naive 1 8 --world lw --H 256 --U 128 --learner <the birth of rule B of seed 73129> --in-turn --visit-updates 256 --save-states 8 --out …/checks/c3_calib`, with neither `--list-every` nor `--decay-distances` given (their defaults), gives the hashes of E44's receipt `$M/exp45/e44/F1/lw_sweep-after-naive_w1_v8_turn.json`: the arrays hashes of the event file and of the list file, the final hashes of the weights and of the optimizer, `learned_sha256` and `stream_sha256`. Check 3 passes only if all six are equal. If they are not, say which differ and go on only if the event file's scores agree to the last item; this fall-back allows the lives to start but is not a pass.
4. **The same start as `naive`.** In check 2's one-visit life of each world, the 2,000 rows of the list file with at_visit = 65535 and source = 0 (the birth family's items answered before visit 0), in the columns task, a, b, g1, g2, g3 and correct, equal the same rows of the list file of `naive` of the same world on `$W1`; and `by_visit[0].stream_sha256` equals the one in that receipt.
5. **Resumption, with the lives' own settings.** From `$M/smallworld`, `python3 sw_run.py life sweep-after-naive 1 8 --world lw --H 256 --U 128 --learner <the birth of rule B of seed 73129> --list-every 8 --decay-distances none --save-states 4,8 --out <dir>`, run whole with `<dir>` = `…/sweep_after_naive/checks/c5_whole`. Run it again with `<dir>` = `…/checks/c5_cut`, and kill that process with signal 9 after its log has printed the line of visit 5 and before it prints the line of visit 6 (the log counts visits from 0). Continue it with `life --resume …/checks/c5_cut/lw_sweep-after-naive_w1_v8_state_v4.pt --out …/checks/c5_cut` (the state written when visits 0 to 3 were done). The two receipts give the same final hashes of the weights and of the optimizer, the same arrays hashes and file SHA-256 of the event file and of the list file, and the same `learned_sha256`. The receipt of this check records the last visit printed before the kill.
6. **Cost.** CPU-seconds a visit, and the forecast for the three lives.

## 6. Saving and resuming

The host stops a machine a few minutes after a session's turn ends and has also restarted machines during open turns. Start every life so that it can be resumed. Commit and push, at least every 30 minutes, the receipts, event files and list files of every finished life and check, and the log and the latest saved state of every running life (a running life has no receipt, event file or list file: the runner writes them after the last visit). Continue an interrupted life from its last saved state as section 4 says. For every life that was resumed the report gives the state it was resumed from with its SHA-256, the number of visits computed twice, and the statement, read from the logs, whether those visits print the same share right and loss in the run before the stop and in the run after it.

## 7. What is recorded and stated

For each of the three lives, the scorer's quantities (`l2_score.quantities` on its event and list files, at V = 256 and at V = 1,024): D2 to D10, D14 and D15; and the early level, the mean of `score_by_age` over visits 0 to 63 at V = 1,024. D11 is the rule of comparison (the interval over three worlds); D12 is not read; D13 is read for `sweep-after-naive` over the three worlds. The figures of `naive` and `sweep` are those of the score files on your branch at 738e2d61 (`sweep/main/lw_naive_w<w>_v1024_V<V>.json` and `sweep/scores/lw_sweep_w<w>_v1024_V<V>.json`). Write `smallworld/l2_san_report.py`, which imports `l2_score` and uses its `quantities`, `interval90` and `verdict` only. It does not use `l2_sweep_report.py` or `l2_sweep_early.py` for the reading, because they read four worlds and write words that this brief excludes. It writes the table, the three sentences, D13, the curves and the comparison of the stream digests into `sweep_after_naive/final/`.

**The fixed reading** (decision 21). Three pairs, each in worlds 1, 2 and 3:

- pair R, `sweep-after-naive` − `naive`: the same birth, the rules differ;
- pair B, `sweep` − `sweep-after-naive`: the same rule of the life, the births differ;
- pair RB, `sweep` − `naive`: both differ (the verified lives).

In every world and at every age the per-world differences satisfy RB = R + B, and so do their means (before rounding; printed figures may differ by one unit in the last decimal). The intervals and the words do not add.

For each pair, at three ages (the early level; the late level at 256 visits; the late level at 1,024 visits): the three per-world differences, their mean and the interval over three worlds, and the word *lower*, *higher* or *not distinguished*. One table holds the nine cells with pair RB beside pairs R and B. For pair R also: the difference of D5 at 1,024 visits with its interval; D13 for `sweep-after-naive`; and the change of the difference, late level at 1,024 visits less early level, by world and with its interval.

Three sentences are fixed beforehand.

1. "From the same birth, the life under the sweep is [lower than / higher than / not distinguished from] the life under back-propagation over the first 64 visits, [..] at 256 visits and [..] at 1,024 visits," with the three means and intervals.
2. On the change for pair R. For each world let e be the difference `sweep-after-naive` − `naive` over the first 64 visits and l the difference at 1,024 visits, both unrounded. The world is *nearer to zero* if e and l have the same sign and |l| < |e|. It has *changed sign* if e and l have opposite signs. Otherwise it is *not nearer to zero*. The sentence is: "In k of the three worlds the difference `sweep-after-naive` − `naive` is nearer to zero at 1,024 visits than over the first 64 visits, in j it changed sign, and in m it is not nearer to zero; over the three worlds the change (the difference at 1,024 visits less the difference over the first 64 visits) is [lower / higher / not distinguished]," with the mean and the interval. A *higher* change means that the difference moved in favour of `sweep-after-naive`. *Overtaking* is said if and only if pair R is *lower* over the first 64 visits and *higher* at 1,024 visits. In a world in which e < l < 0 the report says "the deficit of `sweep-after-naive` became smaller in world w", and this is not called overtaking.
3. On what the birth gave: "Of the difference between the arms `sweep` and `naive` at 1,024 visits (RB, mean and interval), the part with the birth held equal is R (mean and interval) and the part with the rule held equal is B (mean and interval)." RB is assigned to the rule if and only if RB is distinguished, R is distinguished on the same side as RB, and B is not distinguished. It is assigned to the birth if and only if RB is distinguished, B is distinguished on the same side as RB, and R is not distinguished. In every other case the three are given, nothing is assigned, and the report names the condition that failed.

Beside the table: the scores at birth (the receipts' `final_row`, A and R) of the two births of each seed; the curves of the score by age of the three arms in each world; and the digest of every visit's stream of each new life set beside that of `naive` of the same world (they are to be equal, visit for visit).

## 8. Limits that the report states, whatever the result

- Three worlds, one birth seed each; the interval over three worlds is the mean ± 1.686 standard deviations, and the difference between `sweep` and `naive` at 1,024 visits over these worlds, −0.069 [−0.162, +0.024], is not distinguished by it.
- Pair R is not a comparison on equal terms: the birth was formed by back-propagation, and the calibration shows that the sweep starts far below on such a birth. The reverse arm (a life by back-propagation from the sweep's birth) is not run, because the runner has no such arm; the design has three of the four cells of rule and birth.
- One learning rate, 0.001; one visit length, 256 updates, calibrated for `naive`, at which the gate's figure of `sweep-after-naive` was inside the gate's band in 3 of 8 calibration worlds; no store; lives of 1,024 visits; nothing is said beyond them.
- The first 64 visits are mostly continuations (23 to 47 of 64 in worlds 1 to 3) and the late window of 1,024 visits mostly returns (220 to 223 of 256); the change of sentence 2 compares a difference on continuations with a difference on returns.
- The optimizer is fresh at the start of every life, so that the first updates of each life are not those of a learner that continues.

## 9. Predictions (the reviewer's, on file before any life)

| | Statement | Chance |
|---|---|---|
| P60.1 | Pair R over the first 64 visits is *lower* | 75% |
| P60.2 | Pair R at 1,024 visits is *lower* | 45% (*not distinguished* 50%, *higher* 5%) |
| P60.3 | In all three worlds the difference of pair R is *nearer to zero* at 1,024 visits than over the first 64 visits, as sentence 2 defines it (a world that changed sign does not count) | 60% |
| P60.4 | Pair B at 1,024 visits is *not distinguished* | 60% (*higher* 35%, *lower* 5%) |
| P60.5 | `sweep-after-naive` is *still learning* at 1,024 visits by D13 over the three worlds | 85% |
| P60.6 | Checks 1 to 5 pass, each as its text defines passing | 85% |

## 10. Rules

- At most three processes at once, one thread each, launched detached and outside any wrapper with a time limit; the versions of Python, torch and numpy that your receipts of the eight lives record, recorded again in every receipt.
- **Do not end your turn while a check or a life is running:** the machine is stopped a few minutes after a turn ends. Wait in steps of at most ten minutes (one waiting command at a time), and commit and push work in progress at least every 30 minutes. The reviewer sends its messages with the priority that joins a running turn. If the session is nevertheless stopped, continue as section 6 says when it is started again.
- Push only to your branch. Never push to the default branch, to the main worker's branch or to any other; open no pull request.
- The report, `sweep_after_naive/E60_REPORT.md`, in standard academic English: full sentences; every term defined at first use; every figure with the file or command that gives it; each figure rounded once, from the unrounded value; "the sweep" in a reading always named as the arm it is; the limit to the rate 0.001 beside every reading between arms; the limits of section 8; the outcome of each prediction; the committed files listed from `git ls-tree -r --name-only` under `$M/exp45/e44/L2/sweep_after_naive/` and for `$M/smallworld/l2_san*` only.
- Write no name of a language model in a committed file or commit message, apart from what the host and the record impose: the names of branches, the file name of the roster note and the name of the message server. Commit trailers `Acting-Agent: worker session_<your session id>`, as in your earlier commits.
- If anything in this brief cannot be done as written, stop and write to the reviewer before running.

## 11. Reporting

Send the reviewer a short message (tool `send_message` of the claude-code-remote server, session `session_01YBMaMvtGtQvDFcgppgThrZ`) at each of these points: the checks passed, or which failed; world 1's life finished, with the early level and the late levels at 256 and 1,024 visits beside those of `naive` and `sweep`; the three lives finished, with the table of section 7; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines.

## 12. History

Revision 1 (18 November 2026, 1a1a3b4b): written at the coordinator's decision 21.

Revision 2 (18 November 2026): the seventeen defects of the check are entered, each with the check's text unless noted.

| Defect | Weight | What was wrong | Where it is corrected |
|---|---|---|---|
| 1 | high | The rule on messages told the worker to end its turn, which stops the machine and its lives; the message tool can join a running turn | Section 10 |
| 2 | high | "Nearer to zero" had no rule for a difference that changes sign; "overtaking" had only a necessary condition | Section 7, sentence 2; P60.3 |
| 3 | high | A difference that is not distinguished could be assigned to the rule or to the birth | Section 7, sentence 3 |
| 4 | moderate | Check 1 named hashes that the receipt does not hold and left out files that resuming needs | Check 1 |
| 5 | moderate | D5 described as the slope over all visits; D13 and the interval not fully fixed; no rule for a missing world | Section 2; P60.5 |
| 6 | moderate | "D1 to D15 for a life" unclear; the worker's report scripts read four worlds and write excluded words | Section 7, first paragraph |
| 7 | moderate | Check 5: visits counted from 0, and two runs with one file stem in one folder | Check 5 |
| 8 | moderate | No command gave `--out`; a life resumed by hand would write into another experiment's folder | Section 4 |
| 9 | moderate | Check 2 claimed "only that birth", which decision 21 does not ask and the runner does not test; rows of check 4 not named | Checks 2 and 4 |
| 10 | moderate | Check 3: the two arguments and the six hashes not named; its fall-back left P60.6 undecided | Check 3; P60.6 |
| 11 | moderate | "One half of a two-by-two design" (it is three of four cells); two limits missing | Section 8 |
| 12 | moderate | Section 6 asked for files that a running life does not have | Section 6 |
| 13 | low | Two labels of the calibration's figures | Section 3 |
| 14 | low | The rules on names conflicted with the prescribed trailer and with the brief's own text | Section 10. The check proposed to name branches and the message server by other words; the reviewer keeps their exact names, which the worker needs, and exempts them in the rule |
| 15 | low | RB = R + B stated without its limit | Section 7 |
| 16 | low | "At 64 visits" against "over the first 64 visits" | Section 1 |
| 17 | low | "256 cells"; "first family"; the ref for files under `$R`; the memo unnamed; P60.4's chances | Sections 1 and 2; P60.4 |
