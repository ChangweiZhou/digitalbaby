# E60: the long life under the sweep that starts from back-propagation's birth, so that two lives differ in the rule alone (18 November 2026, revision 1)

**Status.** Written by the outside reviewer at the coordinator's decision 21 (`COORDINATOR_DECISIONS_20261118.md` on the default branch at 0cf288ce). Revision 1 is not yet checked. It is released only by the coordinator, and no life is started before that. The number E60 is the reviewer's; the coordinator may renumber it. The job is the second half of item 23 of `PLAN.md` beside this file and is meant for the helper of the long lives, on its present branch. The level of evidence is the one the coordinator's briefs call E2: a first measurement, which confirms nothing.

## 1. The question

Mr. Xu's statement of 15 November (requirement B1 of the plan): our learner may be slower than the conventional one at first and overtake it in the course of continued learning.

The sweep's long lives have been run and verified (`$R/E44_L2_SWEEP_VERIFICATION_20261118.md`; memo section 33.16). Without the store the arm `sweep` ended below the arm `naive` in all four worlds, and no overtaking was seen up to 1,024 visits. The verification's first limit is that the two arms differ in two things at once: the rule by which the life learns, and the birth from which it starts. The sweep's births answer less than back-propagation's before a life begins. A difference between those arms cannot be assigned to the rule.

**This job asks: when the life under the sweep starts from the same birth as the life under back-propagation, is it lower, higher or not distinguished from it at 64, at 256 and at 1,024 visits, and how does that difference stand beside the difference that the sweep's own birth gave?**

## 2. Terms and shell names

`M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`; `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `W1=origin/claude/e44-l2-first-long-lives` (the main worker's branch); `D=origin/codex/minifly-consolidated-20261023` (the default branch). Files of other branches are read with `git show`; no other branch is checked out.

- *Back-propagation* (rule B) is the exact gradient. *The sweep* (rule S) passes the error backward through each linear map by the signs of its weights times their mean size.
- A *visit* is 256 updates on one *family* (add, sub, max and min under one modulus). In each update the network answers 16 *asked* items and then learns from 48 *taught* items. A *life* is 1,024 visits in a *world*, which fixes the families and the order of the visits. A *reduction item* is an asked item of add or sub whose answer needs the reduction by the modulus.
- A *birth* is the network that E41 trained beforehand under one rule, at the half width (256 cells), on half the pairs of the first family, for 40,960 updates: `$M/exp45/e41/e41_<B|S>_h256_f50_<seed>.pt`. The roster note gives each world a seed: world 1, 73129; world 2, 93683; world 3, 116089 (world 4 uses world 1's births and is not part of this job).
- An *arm* is a birth together with the rule of the life. `naive`: the birth of rule B, a life by back-propagation. `sweep`: the birth of rule S, a life by the sweep. `sweep-after-naive`: the birth of rule B, a life by the sweep. None of the three has a store. Every life starts with a fresh optimizer.
- *The roster note* is `OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md` at the root of your branch (revision 2); its definitions D1 to D15 hold here. This brief uses: D2, the share right on the asked reduction items of a visit; D4, the *late level*, the mean of D2 over the last quarter of the visits of a life read at a given length (256 or 1,024 visits); D5, the trend of the score over the second half of the life; D7, the trend of learning; D13, *stuck* or *still learning*. The *early level* is the mean of D2 over visits 0 to 63, as in section 5.7 of your report.
- *The interval over three worlds* of a quantity is the mean of its three per-world values plus or minus t(0.95, 2 degrees of freedom) × sd / √3, that is the mean ± 1.686 standard deviations (t = 2.920). A difference is *lower* or *higher* when the interval excludes zero and *not distinguished* otherwise. No other word is used for a difference: not "level", "the same", "behind", "ahead", "grown", "shrunk" or "not changed".

## 3. What the record says

Every figure of this section was read by the reviewer from committed files; the check of this brief recomputes them.

- **The verified lives, worlds 1 to 3** (your `final/E44_L2_SWEEP_EARLY.md` at 738e2d61; `naive` then `sweep`). Early level: 0.782 and 0.772; 0.717 and 0.620; 0.633 and 0.639. Late level at 256 visits: 0.582 and 0.589; 0.696 and 0.512; 0.692 and 0.611. Late level at 1,024 visits: 0.669 and 0.650; 0.756 and 0.628; 0.742 and 0.682.
- **`sweep` − `naive` over these three worlds** (the reviewer's arithmetic on the four-decimal differences of that file). Early: −0.0100, −0.0977, +0.0064; mean −0.034 [−0.128, +0.061], not distinguished. At 256 visits: +0.0076, −0.1842, −0.0810; mean −0.086 [−0.248, +0.076], not distinguished. At 1,024 visits: −0.0191, −0.1279, −0.0590; mean −0.069 [−0.162, +0.024], not distinguished. With three worlds the interval is wide: the difference at 1,024 visits, which is lower over four worlds by the roster note's interval, is not distinguished over three.
- **The calibration of E44** (`$R/E44_GATE_FOR_THE_SWEEP_20261117.md`, with `e44_gate_sweep.py`) is the only place where `sweep-after-naive` has been run: lives of eight visits, each visit a new family, in eight worlds, all from the births of seed 73129. At 256 updates a visit the median level at the first visit of a family (reduction items; families 1 to 7) is 0.196 for `sweep-after-naive` (0.117 to 0.275 over the worlds), 0.453 for `naive` and 0.614 for `sweep`. `sweep-after-naive` − `naive` is −0.257 (90% interval over eight worlds −0.307 to −0.206; lower in all eight). `sweep` − `sweep-after-naive` is +0.418 (+0.367 to +0.469; higher in all eight). Within a first visit the pooled share right of `sweep-after-naive` rises through the eight eighths of the visit from 0.038 to 0.249, where `naive` rises from 0.071 to 0.489.
- **What does not exist:** a life of `sweep-after-naive` longer than eight visits; one with returns to a family; one from the births of seeds 93683 or 116089; one with the roster's settings (the list every 8 visits, saved states). The reverse arm, a life by back-propagation from the sweep's birth, has never been run and the runner has no such arm (`load_learner` in `sw_run.py` refuses a birth whose rule is not the one the arm names).

## 4. The runs

| Arm | Worlds | Visits | Birth (`--learner`) |
|---|---|---|---|
| `sweep-after-naive` | 1, 2, 3 | 1,024 | `$M/exp45/e41/e41_B_h256_f50_<seed>.pt`, seed 73129, 93683, 116089: the file that the receipt of `naive` in the same world records |

Three lives. Every other setting is that of your eight lives: `--world lw --H 256 --U 128`, visits of 256 updates, weight decay 0.25, learning rate 0.001, `--list-every 8 --decay-distances none`, `--save-states 64,128,256,384,512,640,768,896,1024`. **You do not change the runner or the scorer.** If you find you need to, stop and write to the reviewer. Order: the checks; the three lives together. At your measured 3.65 to 3.80 CPU-hours a life, about 11.5 CPU-hours; with the checks, about 12. If check 6 forecasts more than 15, stop and ask.

Outputs go under `$M/exp45/e44/L2/sweep_after_naive/`; new scripts are named `smallworld/l2_san*`; no file of your earlier work or of the main worker is overwritten.

## 5. Checks before the lives (a receipt for each, in `sweep_after_naive/checks/`)

1. **Sources.** `sw_run.py`, the scorer `l2_score.py` and the other source files of `smallworld/` have the SHA-256 that your receipt `sweep/checks/E44_L2_SWEEP_CHECKS.json` records.
2. **The birth is admitted, and only that birth** (the coordinator's condition, decision 21). For each of worlds 1, 2 and 3: a life of one visit of `sweep-after-naive` with the birth of rule B starts, and its receipt's record of the learner (file, file hash, tensors hash, rule B, score at birth) equals, field for field, the record of the learner in the receipt of `naive` of the same world on `$W1`. And for world 1: the same command with the sweep's birth `e41_S_h256_f50_73129.pt` is refused by the runner; the receipt of this check holds the text of the refusal.
3. **The calibration reproduced.** On this host, world 1's calibration life of eight visits under `sweep-after-naive` at 256 updates a visit (`--in-turn`; E44's receipt is `$M/exp45/e44/F1/lw_sweep-after-naive_w1_v8_turn.json`), with the defaults of the two newer arguments, gives the hashes of that receipt (list, events, final weights and optimizer). If it does not, say which differ and go on only if the event file's scores agree to the last item.
4. **The same start as `naive`.** In check 2's one-visit life of each world, the answers to the birth family's items before visit 0, as the list file records them, equal those in the list file of `naive` of the same world on `$W1`, item for item; and the digest of the first visit's stream equals the one in that receipt.
5. **Resumption, with the lives' own settings.** A life of world 1 under `sweep-after-naive` of 8 visits with `--list-every 8 --decay-distances none --save-states 4,8`, run whole; and the same life whose process is killed from outside (signal 9) during visit 6 and continued with `life --resume` from the state after visit 4. The two end on the same hashes of weights and optimizer and on equal event and list files. The receipt records the visit at which the process was killed.
6. **Cost.** CPU-seconds a visit, and the forecast for the three lives.

## 6. Saving and resuming

The host stops a machine a few minutes after a session's turn ends and has also restarted machines during open turns. Start every life so that it can be resumed; commit and push the receipts, the event and list files and the latest saved state of every running life at least every 30 minutes; continue an interrupted life with `life --resume` from its last saved state. For every life that was resumed the report gives the state it was resumed from with its SHA-256, the number of visits computed twice, and the statement, read from the logs, whether those visits print the same share right and loss in the run before the stop and in the run after it.

## 7. What is recorded and stated

Everything the roster note records for a life (D1 to D15), computed with the scorer for the three lives, at 256 and at 1,024 visits; and the early level.

**The fixed reading** (decision 21). Three pairs, each in worlds 1, 2 and 3:

- pair R, `sweep-after-naive` − `naive`: the same birth, the rules differ;
- pair B, `sweep` − `sweep-after-naive`: the same rule of the life, the births differ;
- pair RB, `sweep` − `naive`: both differ (the verified lives). In every world and at every age RB = R + B.

For each pair, at three ages (the early level; the late level at 256 visits; the late level at 1,024 visits): the three per-world differences, their mean and the interval over three worlds, and the word *lower*, *higher* or *not distinguished*. One table holds the nine cells with pair RB beside pairs R and B. For pair R also: the difference of D5 at 1,024 visits with its interval; D13 for `sweep-after-naive`; and the change of the difference, late level at 1,024 visits less early level, by world and with its interval.

Three sentences are fixed beforehand.

1. "From the same birth, the life under the sweep is [lower than / higher than / not distinguished from] the life under back-propagation over the first 64 visits, [..] at 256 visits and [..] at 1,024 visits," with the three means and intervals.
2. On the change for pair R: "In k of the three worlds the difference `sweep-after-naive` − `naive` is nearer to zero at 1,024 visits than over the first 64 visits; over the three worlds the change is [lower / higher / not distinguished]." *Overtaking* is said only if pair R is *higher* at 1,024 visits. A deficit that becomes smaller is stated in those words and is not called overtaking.
3. On what the birth gave: "Of the difference between the arms `sweep` and `naive` at 1,024 visits (RB, mean and interval), the part with the birth held equal is R (mean and interval) and the part with the rule held equal is B (mean and interval)." The difference RB is said to belong to the rule only if R is distinguished and B is not; to the birth only if B is distinguished and R is not; otherwise the three are given and nothing is assigned.

Beside the table: the scores at birth of the two births of each seed; the curves of the score by age of the three arms in each world; and the digest of every visit's stream of each new life set beside that of `naive` of the same world (they are to be equal, visit for visit).

## 8. Limits that the report states, whatever the result

- Three worlds and three birth seeds; the interval over three worlds is 1.69 standard deviations wide on each side, and a difference of the size found between `sweep` and `naive` (−0.07 at 1,024 visits) is not distinguished by it.
- Pair R is not a comparison on equal terms: the birth was formed by back-propagation, and the calibration shows that the sweep starts far below on such a birth. The reverse arm (a life by back-propagation from the sweep's birth) is not run, because the runner has no such arm. The design is therefore one half of a two-by-two design of rule and birth.
- One learning rate, 0.001; one visit length, calibrated for `naive`; no store; lives of 1,024 visits; nothing is said beyond them.
- The optimizer is fresh at the start of every life, so that the first updates of each life are not those of a learner that continues.

## 9. Predictions (the reviewer's, on file before any life)

| | Statement | Chance |
|---|---|---|
| P60.1 | Pair R over the first 64 visits is *lower* | 75% |
| P60.2 | Pair R at 1,024 visits is *lower* | 45% (*not distinguished* 50%, *higher* 5%) |
| P60.3 | In all three worlds the difference of pair R is nearer to zero at 1,024 visits than over the first 64 | 60% |
| P60.4 | Pair B at 1,024 visits is *not distinguished* | 60% (*higher* 35%) |
| P60.5 | `sweep-after-naive` is *still learning* at 1,024 visits | 85% |
| P60.6 | Checks 1 to 5 pass | 85% |

## 10. Rules

- At most three processes at once, one thread each, launched detached and outside any wrapper with a time limit; the versions of Python, torch and numpy that your receipts of the eight lives record, recorded again in every receipt.
- A message reaches you only when your turn ends. Keep waits to ten minutes at most, so that a message is read within that time, and push work in progress every 30 minutes.
- Push only to your branch. Never push to the default branch, to the main worker's branch or to any other; open no pull request.
- The report, `sweep_after_naive/E60_REPORT.md`, in standard academic English: full sentences; every term defined at first use; every figure with the file or command that gives it; each figure rounded once, from the unrounded value; "the sweep" in a reading always named as the arm it is; the limit to the rate 0.001 beside every reading between arms; the limits of section 8; the outcome of each prediction; the committed files listed from `git ls-tree`. No name of any language model in any committed file or commit message; commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says.
- If anything in this brief cannot be done as written, stop and write to the reviewer before running.

## 11. Reporting

Send the reviewer a short message (tool `send_message` of the claude-code-remote server, session `session_01YBMaMvtGtQvDFcgppgThrZ`) at each of these points: the checks passed, or which failed; world 1's life finished, with the early level and the late levels at 256 and 1,024 visits beside those of `naive` and `sweep`; the three lives finished, with the table of section 7; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines.

## 12. History

Revision 1 (18 November 2026): written at the coordinator's decision 21; to be checked by an agent that has not seen the reviewer's reasoning; then to the coordinator for release.
