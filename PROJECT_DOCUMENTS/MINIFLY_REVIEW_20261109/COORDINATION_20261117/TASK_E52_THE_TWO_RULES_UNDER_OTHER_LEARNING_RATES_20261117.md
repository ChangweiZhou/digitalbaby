# E52: the two learning rules under other learning rates (17 November 2026, revision 2)

Evidence level E2. **Status.** Written by the outside reviewer, acting for the coordinator, who cannot be reached while Mr. Xu is away (his instruction of 17 November: "我人在外面你自己安排", arrange it yourself). Revision 1 was checked by an agent of the reviewer that had not seen the reviewer's reasoning (`CHECK_OF_TASK_E52_BRIEF_20261117.md` beside this file); revision 2 enters its ten findings, listed at the end. That check is not the independent audit the programme's rules ask for, and the coordinator has not seen the brief. It is item 8 of `PLAN.md` beside this file. A trial on one seed: 28 runs, about 26 CPU-hours, and up to 4 more for the further runs.

Shell names: `D=origin/codex/minifly-consolidated-20261023` (the default branch); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `W46=origin/claude/wonderful-edison-8hx6i6` (E46's worker); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. Host: Linux x86-64, Python 3.13, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit.

## Terms

- *The network* is `arch20`'s recurrent network at a width (H, U): a byte table, a layer `l1` (8 to H), a GRU of U units run by `arch26.forward`, and a readout. *Present width*: H = 512, U = 256. *Half width*: H = 256, U = 128. `exp41_scale.new_net(seed, H, U)` builds it.
- *Back-propagation* (rule B) is the exact gradient. *The sweep* (rule S) passes the error backward through each linear map by the signs of its weights times one scale for the map, the mean size of the map's weights (`exp3_state.py`, line 100); it is `local=True` in `exp26_candidate.step`.
- *The first family* is add, sub, max and min under modulus 89. *The taught fraction* f is the share of the pairs of operands that the stream teaches; the others are never taught. Score A is the share right on never-taught pairs of add and sub at 12 letters; score R the share right on never-taught pairs of add and sub whose answer needs the modulus, without letters. Both are E41's.
- *Updates to 80* on a score: the first scored row from which three scored rows in a row are at or above 80% (E41's three-row rule; rows every 256 updates). A run that does not reach 80 on a score has no count on it; its last recordable count is its dose less 512.
- *The optimizer* of E41, which is E26's, is AdamW with learning rate 0.001, betas (0.9, 0.98) and weight decay 0.25 on every parameter but the byte table (`exp26_candidate.make_opt`). The rate was taken from the literature on such tasks and given to both rules; the decay was chosen for the sign rule on earlier learners and back-propagation took it over in E26 (the inventory of the record, `INVENTORY_DENSE_LINE_20261117.md`).
- *A case* is a rule, a width and a taught fraction (eight cases). *The lag* is the sweep's updates to 80 divided by back-propagation's, at one width and fraction.

## The question

Every comparison of the sweep with back-propagation in the record was made at one learning rate of AdamW, 0.001, which neither rule was ever run without. E41 found the sweep behind, most at the half width and with a quarter of the pairs. Mr. Xu's question of 15 November, whether our learner, slower at first, overtakes later, is put in lives run at the half width. **Is the sweep's lag a property of the rule, or of the one learning rate at which both rules have been run?**

## What the record says (working rule 1)

E41 (`exp41_scale.py`; three seeds; the reviewer's recount of the receipts). Updates to 80 on A and on R at learning rate 0.001, and the lag, for seeds 73129, 93683, 116089; "–" is not reached within the dose.

| Width, fraction | Sweep, A | Back-propagation, A | Lag on A | Sweep, R | Back-propagation, R | Lag on R |
|---|---|---|---|---|---|---|
| half, 0.50 | 12,288; 10,240; 10,496 | 11,520; 7,424; 7,680 | 1.07, 1.38, 1.37 | 15,872; 13,312; 20,480 | 13,056; 8,704; 8,960 | 1.22, 1.53, 2.29 |
| half, 0.25 | 17,408; 14,080; 24,832 | 10,240; 6,144; 7,680 | 1.70, 2.29, 3.23 | 35,840; 24,576; 30,464 | 16,896; 7,424; 9,216 | 2.12, 3.31, 3.31 |
| present, 0.50 | 5,632; 4,864; 5,376 | 5,120; 4,096; 4,352 | 1.10, 1.19, 1.24 | 6,656; 4,864; 6,400 | 6,400; 4,096; 5,120 | 1.04, 1.19, 1.25 |
| present, 0.25 | 8,704; 9,984; – | 4,352; 3,584; 4,864 | 2.00, 2.79, – | –; –; – | 7,680; 4,352; 5,888 | – |

- With a quarter of the pairs the sweep reaches 80 on R in 3 of 6 runs at these two widths, all three at the half width. At the present width its R over the last three rows is 80.4, 78.8 and 73.9; on seed 73129 the three rows are 79.95, 80.75 and 80.35, so the three-row rule fails there by one answer of 2,000.
- At the half width with a quarter of the pairs the sweep's R is unsteady from row to row on seed 73129 (root mean square change 7.7 points over the last 16 rows, against 3.7 under back-propagation). A smaller step is the ordinary remedy for an unsteady update.
- Seed 73129 has the smallest lag of the three seeds in every comparison that can be formed. The trial is therefore run on seed 93683.
- **E46** (`exp46_decay.py` on `$W46`; the same birth at decay 0, 0.05 and 0.25, present width, f = 0.50, three seeds): the decay does not change the updates to threshold under either rule, within the band two thirds to 1.5; under back-propagation two seeds lie on its upper boundary without decay. Under the sweep the final level of R is lower without decay on all three seeds. The decay sets the size of the weights: on seed 73129 under the sweep the norm of the recurrent map at 24,576 updates is 28.5 at decay 0.25, 44.7 at 0.05 and 72.0 without decay. The size of the weights is part of the sweep, whose error scale is the mean size of a map's weights.
- **AdamW's learning rate has not been varied in any birth of the present network.** E43's first addendum ran plain gradient steps at 0.01, 0.03 and 0.1 on a continuation, under back-propagation on one seed.

## Runs

The birth of E41 (the first family alone, 0 to 12 letters, batches of 64, `exp41_scale`'s stream, panels and scores) with one thing changed: the learning rate.

| Factor | Values |
|---|---|
| Learning rate | 0.0003, 0.001, 0.003 |
| Rule | S, B |
| Width | half, present |
| Taught fraction | 0.50, 0.25 |
| Seed | 93683 |
| Dose | At 0.001 and at 0.003: E41's, 40,960 updates at the half width and 12,288 at the present width. At 0.0003: E41's × 10/3, rounded up to a row: 136,704 and 40,960. Rows every 256 updates. Final levels at 0.0003 are tabled at the end of the run and at E41's dose |

Twenty-four runs. The eight runs at 0.001 repeat E41's runs of seed 93683 with the new runner, so that the norms and the length test exist at every row for all three rates; their final `tensors_sha256` must equal E41's receipts' (check 1). Order: the eight at 0.001, then the eight at 0.003, then the eight at 0.0003.

**The decay stays at 0.25.** AdamW's update is w ← (1 − rate × decay) w − rate × step. Both terms carry the rate, so with the decay left at 0.25 the whole update is scaled, and the size at which the decay holds a weight that the loss keeps pushing does not move. The size of a weight held against noise still changes, by about the square root of the rate. Every scored row therefore records E46's seven norms (`exp46_decay.norms` on `$W46`), and the report tables them at each threshold and at the end. The byte table stays without decay.

**A control: rate × decay held at 0.00025** (decay `0.25 * 0.001 / rate`, computed by that expression), which moves the size of the weights in proportion to the rate. It is run at the half width with half of the pairs, under both rules, at each new rate, at the main runs' dose: four runs, read beside the main runs.

**One more value where the best is at the upper end.** For a case, if the best rate on R (on A where the case is not reached on R) is 0.003, the case is also run at 0.01 at E41's dose with decay 0.25; a tie is not a best rate. If the best is then 0.01, it is reported as "at or above 0.01". No run is added below the lowest rate; a best rate there is reported as "at or below 0.0003". A case that is not reached on both scores gets no further run. At most eight further runs.

## The runner

A new file `exp52_rate.py` in `$M`; no existing source is edited. It is `exp41_scale.run` with the learning rate and the decay as arguments: after `opt = C.make_opt(m)`, every parameter group's `lr` is set to the run's rate, and `weight_decay` is left as `make_opt` set it (in the four control runs group 0's is set to `0.25 * 0.001 / rate`).

- **The receipt** is E41's format with these differences. The fields `optimizer` and `optimizer_read_back` are not taken from `exp26_candidate.opt_record`, which writes 0.001 and 0.25 whatever the groups hold. The runner writes `dict(name="AdamW", lr=<the run's rate>, weight_decay=<the run's decay>, betas=[0.9, 0.98], groups=exp20_reference.opt_groups(m, opt))`, and asserts the groups equal to the expected ones at the start and at the end of the run, as `exp46_decay.py` does. The receipt records `exp52_rate.py` as the runner, with `exp41_scale.py` among the sources.
- **At every scored row** the runner also records E46's seven norms and the length test of E42 and E46 on pairs that the run's own stream never teaches: `E.panel(t, E.split(t, f, seed)[1], seed, ks=(12, 40), n=300)` with `t = ("add", 89, b"p")` and f the run's fraction. (`exp11_cell.split` seeds its shuffle with the fraction, so the pairs held out at 0.50 are not held out at 0.25; the panel must be drawn from the run's own split.) At f = 0.50 this is E46's panel. If its 300 items at 12 letters are those of the row's `A_counts["add89"]`, the two counts are asserted equal; if they are not the same items, say so in the report.
- **A run whose row has `finite_losses` false** is stopped at that row and recorded as diverged; it counts as not reached, and no higher rate is run in its case.
- Receipts in `$M/exp45/e52/e52_<rule>_h<H>_f<25|50>_lr<rate>_<seed>.json` with the rate written by `f"{rate:g}"` (`lr0.0003`, `lr0.001`, `lr0.003`, `lr0.01`); the control runs carry `_product` before the seed. Final weights beside them.

## Checks before the runs (a receipt for each)

1. **The runner is E41's at 0.001.** (a) With learning rate 0.001 and decay 0.25, 2,048 updates at the present width, f = 0.50, seed 73129 give the `tensors_sha256` that E41's check 1 records for that dry run under each rule (`exp41_scale.DRY2048`: 79618ab2 under S, adf933d7 under B, their first eight digits). (b) Each of the eight runs at 0.001 on seed 93683 ends with the `tensors_sha256` of E41's receipt of the same case, and its A, R and taught counts equal E41's at every row. If a run of (b) fails, stop and report before any run at another rate.
2. **The two numbers act as set, and nothing else differs.** From E41's start (seed 73129, present width, f = 0.50, its first batch) under back-propagation, for each new rate: (a) one update at (rate, the run's decay) and one at (rate, decay 0) differ in every tensor of group 0 by rate × decay × the starting tensor within 1%, and not at all in the byte table (E46's check 2, with the rate in the place of 0.001); (b) the change of every tensor in one update at (rate, decay 0) is rate / 0.001 times its change at (0.001, decay 0) within 1%. (c) In every run the groups read back at the start and at the end equal the expected ones, with betas (0.9, 0.98). (d) In every run the batch digest at row 4,096 equals the one in E41's receipt of the same seed and fraction.
3. **Cost.** A dry run of 256 updates at each width with CPU-seconds an update and a scored row, and a forecast. If the forecast for the 28 runs exceeds 26 CPU-hours by more than a quarter, report before starting.

## How the result is stated (fixed beforehand)

Each reading is made on R and on A separately, by E41's three-row rule.

- *The best learning rate on a score*, for a case: the rate with the fewest updates to 80 on it. Counts that differ by at most one row (256 updates) are tied; a tie that includes 0.001 is given to 0.001, and any other tie is reported as a tie. Where no rate reaches 80 on the score, the case is *not reached* on it and no best rate is named.
- *Faster, slower or the same at another rate*: the run at 0.001 and the run at the other rate are described by E41's rule for two runs (`exp41_scale.describe_step`), with the run at 0.001 in the place of the narrower run. The case is *faster* below two thirds of the count at 0.001, *slower* above 1.5 times, *the same* between, and *not decided* or *neither reaches* as that rule says when a run does not reach 80. *The rule gains from its own rate* at a width and fraction if it is faster at some rate on R and not slower at that rate on A; where no rate reaches 80 on R, if it is faster at some rate on A.
- *The lag on a score*, at a width and fraction: the sweep's updates to 80 over back-propagation's, each at its best rate on that score, beside the same ratio at 0.001. Where a count is missing the lag is given as a bound from the last recordable count, or as "not formed" when both are missing. The lag at the rules' own rates is *smaller* (*larger*) only if it is below two thirds of (above 1.5 times) the lag at 0.001, *gone* if it is smaller and at most 1.0, and otherwise *the same within the scatter of one seed*.
- For each case the final levels of A and R (mean of the last three rows), the length test at 40 letters (last row, and mean of the last three rows) and the seven norms are tabled at every rate, so that a rate that is faster to 80 and ends lower is seen. One row of the length test is a weak figure (in E46 it moves by 15 to 30 points over the last twelve rows of a run); the report says so beside it.
- The control runs are tabled beside the main runs of the same case and rate, with the norms; the report says in one sentence for each rule and rate whether the count to 80 falls in the same class (faster, the same, slower) under both ways of setting the decay.

The sentence the job can license, for one seed: "at this width and fraction the sweep's lag behind back-propagation is [the same within the scatter of one seed / smaller / gone / larger] when each rule is run at its own best learning rate among those tried". Nothing is said about more than one seed until the other seeds are run.

## After the trial

The reviewer reads the trial. The two other seeds are then run for every width and fraction at which either rule gains from its own rate or the lag changes class. If the sweep gains at the half width, the sweep's long lives (plan item 6) are asked for again at that rate.

## Predictions (the reviewer's, on file before any run)

| | Statement | Chance |
|---|---|---|
| P52.1 | Under back-propagation the best learning rate on R is 0.001 or 0.003, or a tie that includes one of them, at all four widths and fractions | 75% |
| P52.2 | The sweep gains from its own rate at one or more of the four widths and fractions | 30% |
| P52.3 | At the present width with a quarter of the pairs taught the sweep's mean of R over the last three rows is at least 85 at some rate tried (at 0.001 it is 80.4, 78.8 and 73.9 on the three seeds) | 35% |
| P52.4 | At the half width with a quarter of the pairs, the lag on R at the rules' own rates is *smaller* than at 0.001 (where it is 3.31 on this seed) | 25% |
| P52.5 | Back-propagation gains from its own rate at none of the four widths and fractions | 75% |
| P52.6 | At the half width with a quarter of the pairs, the sweep's R is steadier at 0.0003 than at 0.001 (a smaller root mean square change from row to row over the last 16 rows) | 65% |

The explanation under test, with what refutes it: the sweep's lag at the half width and at a quarter of the pairs is in part a matter of the learning rate. It is refuted for this seed if the sweep gains at no width and fraction, or if, wherever the sweep gains, back-propagation gains too and the lag is not smaller.

## What it cannot show

- The other numbers of the optimizer (the betas, the epsilon), or another optimizer; the scale inside the sweep (plan item 18).
- A learning rate that changes during a run.
- Widths other than the two; fractions other than the two; more than one seed.
- Anything about a life: the rates are compared on a first learning. A birth made at another rate is not admitted to a life by `exp26_candidate.admit_e26`, which compares the recorded optimizer with the constants.

## Rules for the worker

- At most three processes, one thread each (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), launched detached and outside any wrapper with a time limit. torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` in a virtual environment; versions recorded in every receipt.
- The container can restart without warning and loses what is not pushed. Commit and push receipts and logs as runs finish, and the partial receipt of a long run at least every two hours. A run interrupted by a restart is started again from its beginning, after its partial receipt and log are moved to `$M/exp45/e52/interrupted/` (the runner refuses an existing path).
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only: fetch with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read with `git show`. Where you use a file of E46's runner, copy it byte for byte, record its SHA-256 and the commit it came from, and import it unchanged.
- Commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says. No name of any language model in any committed file or commit message.
- The report, `E52_REPORT.md`, in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files listed from `git ls-tree`.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: the checks passed (or which failed); the eight runs at 0.001 finished (do they equal E41's?); the runs at 0.003 finished, with their table; the runs at 0.0003 and the controls finished; any further runs finished; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The three checks; the 28 runs and any further runs the rule above calls for; `E52_REPORT.md` with the table of every run (updates to 80 on A and on R, final levels, the length test, the norms), the readings for each case and for each width and fraction, the control beside the main runs, the outcome of each prediction, and the list of committed files.

## What changed from revision 1 (the check's ten findings)

| | Change |
|---|---|
| 1 | The decay stays at 0.25 in the main runs. Revision 1 held rate × decay fixed and said that this left one thing changed; it moves the size of the weights in proportion to the rate, and the size of the weights is the sweep's error scale. The product rule is now a control of four runs, and the norms are recorded at every row |
| 2 | The readings use the record's band (two thirds to 1.5) and are made on R and on A separately; ties, cases not reached and missing counts are defined. Revision 1's mark of 0.8 lay inside the scatter of one seed. The trial seed is 93683, since seed 73129 has the smallest lag of the three in every comparison |
| 3 | The rate 0.0003 is run at 10/3 of E41's dose; at E41's dose it could only have been read as slower. No run below 0.0003 |
| 4 | The length test is drawn from the run's own held-out pairs (at a quarter of the pairs, 68 of the 300 pairs of revision 1's panel are taught) and is scored at every row; the cases are run again at 0.001 so that it and the norms exist at all three rates |
| 5 | A diverged run is stopped and counted as not reached; "case" replaces the two uses of "cell" |
| 6 | The receipt writes the run's own rate and decay; `exp26_candidate.opt_record` would have written 0.001 and 0.25 for every run |
| 7 | Check 2 tests that one update uses the two numbers; check 3 has a limit |
| 8 | Figures corrected: the lag at the present width with half of the pairs is 1.10, 1.19 and 1.24 on the three seeds, not 1.1; "4 of 9" spanned three widths; E46's sentence carries its qualification; plain gradient steps were run at three rates in E43's first addendum |
| 9 | Cost restated: about 26 CPU-hours for the 28 runs |
| 10 | The decay of the control is given as an expression; the receipt's name has a format; an interrupted run's partial receipt is moved before the run is started again |
