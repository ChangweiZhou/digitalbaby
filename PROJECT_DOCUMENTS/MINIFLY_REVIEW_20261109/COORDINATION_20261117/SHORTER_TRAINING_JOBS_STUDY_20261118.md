# How the planned training runs can be made shorter or smaller without losing what they are run for: a study from the record (18 November 2026)

**For** the project's reviewer, its coordinator (Mr. Zhou) and its owner (Mr. Xu). **Written by** an agent of the reviewer, between 21:36 and 21:56 UTC on 18 November 2026 (clock read with `date -u` at 21:36, 21:51, 21:53 and 21:55 UTC); the work was interrupted by a restart of the container at about 21:45 and resumed from the script as it stood on disk. **Status:** advice. It decides nothing, changes no brief and starts nothing. Nothing was trained for it; no existing file was changed.

**Figures.** Every number below is printed by `shorter_jobs_study.py` in this folder, with the number of its section in brackets (for example `[2.1]`). The script reads 193 receipts with a curve (E41, E42, E47, E49, E50, E52, E54 on the default branch `origin/codex/minifly-consolidated-20261023`; E57 on `origin/claude/e50-fixed-input-code-017dSE`) and the 132 receipts of E58 and E59, through `git show`, and writes nothing. It runs in about 18 seconds: `python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/shorter_jobs_study.py` from the repository's root, after fetching E57's branch as the task describes. Copies of runs kept as references in other folders, interrupted runs, dry runs and qualification runs are left out. The source read is that named in the task (`exp50_code.py`, `exp42_cells.py`, `exp11_cell.py`, `exp45.py`, `e50_sparse_code.py`, with `e42_cells.py`, `e49_interface.py` and `exp3_state.py` for the maps they call).

## Terms

- **A row** is one scoring of the network (every 256 updates on record). **Updates to 80** on a panel is the update of the first of three consecutive rows at or above 80 on that panel; a run **reaches** a panel if such rows exist within its updates. **The level** at an update is the mean of the three rows ending there.
- **Settled** (a run, at an update): both R and A have reached 80 and the third of the three rows on each has been scored.
- **The words** are those of E61's revision 3: on a panel, *sufficient* if the run reaches 80 within 1.25 times the record's updates (on R 19,200 for the exact gradient and 42,880 for the sweep; on A 17,600 and 44,480), *slower* if it reaches 80 later within its updates, otherwise *not reached*. The study applies them to every run as a test of whether a change alters a reading; they have no other meaning for runs outside E61.
- **A changed reading**: the stopped or rescored run differs from the full run in whether R or A is reached, in the updates to 80 on either, or in either word.
- **The record's network** is E50's `C46-one`; **E47's network** has 1,024 cells. **S** is the sweep, **B** the exact gradient. **A line** is one machine of four cores that runs three processes at a time.
- **Executed multiply-adds** are those the code performs; **multiply-adds in use** are those of weights inside the masks, which the receipts and briefs count.

## 1. Where the time goes

**1.1 Cost by network and rule.** On the 88 runs of 256-cell networks with the standard schedule (E42, E49, E50, E54, E57) an update costs 0.10 to 0.15 CPU-seconds and a row 5.6 to 7.5 CPU-seconds (median 6.64), and scoring takes a median 17.8% of the CPU of a run (range 12.5 to 21.4%) [1.1]. The networks with periods (E42's P4, P8, P16, P8-source; E54's P8) cost 0.14 to 0.21 s an update and 9 to 12 s a row. The record's network costs 0.113 (B) and 0.134 (S) s an update, 6.7 s a row, 1.90 and 2.18 CPU-hours a run of 49,152 updates. E47's network of 1,024 cells costs 0.40 s an update and 34 s a row, 3.6 CPU-hours for 24,576 updates [1.1]. Over 81 matched pairs, an update under the sweep costs a median 1.07 times one under the exact gradient (0.97 to 1.29) [1.2]. Wall time is a median 1.022 times the CPU time of updates and scoring (10th to 90th percentile 1.013 to 1.037) [1.3]: the runs are bound by computation, not by waiting.

**1.2 How an update divides.** In `e50_sparse_code.forward` and `e49_interface.masked`, a masked map is computed as a dense matrix product with the mask multiplied into the weights. The arithmetic performed is therefore that of the dense maps, not that of the weights in use [1.4]:

| map | forward, executed (in use) | backward, executed | share of forward and backward |
|---|---|---|---|
| input map (512 code values to 768 gate inputs) | 393,216 (35,328) | 393,216 (weights only) | 0.293 |
| cells (recurrent map) | 196,608 (8,448) | 393,216 | 0.219 |
| `lm` | 131,072 | 262,144 | 0.146 |
| `l2` | 262,144 | 262,144 (weights only) | 0.195 |
| `l3` | 131,584 | 263,168 | 0.147 |
| total | 1,114,624 (568,576) | 1,573,888 | 1 |

The readout holds 92.3% of the forward multiply-adds in use (`l2` 46.1%), the figure the task quotes, but 47.1% of those executed (`l2` 23.5%); with the backward pass the readout is 48.8% of the executed work, the cells 21.9%, the input map 29.3%, and the backward pass 58.5% [1.4]. The two counts answer different questions: the count in use describes the network, the executed count predicts the CPU.

**1.3 A model of the cost.** On 51 runs of E41 and E52 under the exact gradient (128, 256 and 512 cells), the CPU-seconds of an update are 0.0140 + 0.0845 times the executed forward multiply-adds of a position in millions; the model gives 0.0405, 0.1085 and 0.3692 s against medians of 0.0402, 0.1078 and 0.3656, and 0.1082 for the record's network against 0.1131 observed under B [1.5]. Scaled by the sweep's factor and with the scoring cost proportional to the forward, it gives for 49,152 updates and 193 rows [1.6]:

| network | B | S |
|---|---|---|
| the record's network | 1.83 | 2.11 |
| readout without `l2` (units reading 256, 32 or 8 cells, masked as now) | 1.45 | 1.66 |
| the same, with the masked `lm` computed sparsely (32 cells; 8 cells) | 1.28; 1.26 | 1.47; 1.45 |
| no hidden layer, 257 output units reading 32 cells | 1.16 | 1.33 |
| R = 2, 4, 8 exchanges a byte | 2.12, 2.70, 3.86 | 2.44, 3.10, 4.43 |

A readout whose units read 32 or 8 cells costs, as the code now computes it, the same as one whose units read 256: only the removal of `l2` saves work (about 20%). The R grid agrees with the order's estimates (2.2, 2.6, 3.4 an hour a run on average over the rules).

**1.4 E47's network with more exchanges.** E47's source was not read. If the part of its cost above that of E42's `F10` (0.277 s an update, 27 s a row) is the cells' part and grows in proportion to R, a run of 24,576 updates costs about 11.5 CPU-hours at R = 4 and 22 at R = 8, and item 4's four runs about 67 CPU-hours against the order's 26 [1.7, 9.1]. This is an assumption about code not read; item 4's cost check must settle it before release, and the brief's limit should be set from that check, not from the order's figure.

## 2. Stopping a run when its reading is settled

All rules were applied to the 193 complete curves (6.30 million updates, 217 CPU-hours) [2.0]; 135 reach 80 on R, 144 on A, 135 on both.

**2.1 Stop when settled.** A run stopped when it is settled, or n rows later, keeps every reading defined on updates to 80 by construction: whether R and A are reached, the updates to each, and the words. On the record it would have saved 40.2% of all updates with n = 0, 35.9% with n = 8, and 24.6% with n = 32 (78.5, 66.3 and 37.8 CPU-hours), with no changed reading [2.1]. **This is the rule with the largest saving at no changed reading.** Most of the saving comes from the dense runs of E41 and E52, which settle early. On the 27 runs of 256-cell networks with 49,152 updates (E50, E54, E57), which are closest to the planned jobs, only 9 reach both panels, and the rule with n = 8 that never stops before 24,576 saves 12.5% of their updates (37.5% in a run that reaches) [2.1, 9.0].

**2.2 What a stopped run loses.** (i) *The level at the end.* In 134 runs that settle before their end, the level of R at the end exceeds the level at the settling row by a median 9.3 points (10th to 90th percentile 2.5 to 13.0), by more than 5 points in 111 runs [2.2]. A stopped run reports the level at its stop, which is not the level at the end and cannot be compared with the record's levels at the end. (ii) *The level at 24,576*, which the briefs quote beside updates to 80: lost in 93 runs if the stop may come earlier [2.2]. (iii) *The length test at the end*: in 103 runs the level at 40 letters rises a median 9.1 points after settling (more than 5 in 86) [2.2]. (iv) *A later fall*: in 8 runs R fell below 80 for three rows after reaching it (lowest later rows 64.8 to 77.3; for example E41 `h256_f25` S, seed 73129, reached at 35,840, fell to 66.2 and ended at 82.8; two runs at the rate 0.003 fell to 67.5 and 68.5) [2.2]. Updates to 80 are unchanged by such a fall, but a stopped run cannot show it. None of the 8 is a 256-cell run with limited connections except E42's `A10` B (lowest 76.9).

For this reason the recommended form is: **stop 8 rows after the run is settled, and never before update 24,576**; report the level at the stop as such; and run to the full length every run whose reading is quoted as a level (any run named a reference, and any run that does not reach). It saves 18.8% of the record's updates (22.5 CPU-hours), with no changed reading [2.1].

**2.3 Stop as "not reached".** Rules of the form "stop when the level of R (or of T) at update u is below x" were tried for u = 4,096 to 24,576 and x = 2 to 60 [2.3]. The largest saving with no changed reading is "R below 10 at 24,576": 3.1% of the updates (4.3 CPU-hours). The next rules change readings: "R below 20 at 24,576" saves 6.9% but stops a run (E52 S, `h256_f50`, rate 0.0003, product control) that later reached 80 on A at 76,800; "R below 5 at 16,384" saves 3.1% and stops the same run [2.3]. The margin is narrow where the planned jobs live: E57's `in16 row` under the sweep had R at 22.8 at 16,384 and 44.9 at 24,576 and reached 80 on R at 44,800 and on A at 36,096; E50's `C46-one` under the sweep had R at 51.5 at 16,384 and reached at 34,304 [2.3, 6.1]. A rule on T is no better: "T below 50 at 16,384" stops E57's `in16 row` S (T 43.1), which reached [2.4]. Rules on the training loss (not fallen by a share y over w updates, from 8,192 on) change readings in 2 to 14 runs for every setting except the weakest (no fall at all over 8,192 updates: 7 runs stopped, 1.5% saved, none changed) [2.5].

Such a stop also changes the content of a reading even when it does not change the class: the fixed sentences of items 2 and 3 ("does not reach 80 within 49,152 updates; levels at the end A [a], R [r]") need the end. E50's `C16-in` under B ended at 79.4 on R and reached A at 40,704; a stop at 16,384 would have reported R at 48.0 and no reach on A. **A stop as "not reached" is therefore advised only as a screening stage whose result is worded as such (section 6), not as a saving in the runs that carry readings.**

**2.4 Stop at the limit of *sufficient*.** On E49, E50 and E57 runs of 49,152 updates, stopping at 1.25 times the record's updates when not settled saves 0.9% and turns *slower* into *not reached* in 2 of 7 runs (E50 `C16-in` B on A; E57 `in16 row` S on R) [2.6]. It is not advised.

## 3. Scoring less

**3.1 Fewer rows.** With rows every 512 updates (three consecutive rows at that spacing), reached or not changes in 2 runs on R and 1 on A, updates to 80 on R move by a median 128 (up to 3,584) and 26 of 300 pairs of runs move across the ratio 1.25; with rows every 1,024, 3 and 4 runs change, the shift is up to 7,936, and 25 pairs move across 1.25 [3.1, 3.4]. The differences between arms that the readings rest on are smaller than these errors in a material share of cases: of the 300 pairs (same experiment, seed and rule) that both reach R, 44 differ by less than 1,024 updates and 92 lie within a ratio of 1.25 [3.4]. A fixed coarser schedule is not advised.

**3.2 Rows that follow the run.** Scoring every 1,024 updates while R and A are both below 40 and every 256 updates once either is at or above 40, with the rows at 24,064, 24,320 and 24,576 and the last three always scored, changes no reading on the record and keeps 75.7% of the rows (at a threshold of 30, 78.2%; at 50, one reading changes, on A by 256 updates) [3.2]. It is advised, but it saves little once the scorer is corrected (section 7).

**3.3 Smaller panels.** Resampling the recorded counts row by row (hypergeometric draws of 250 of each 1,000 items of R, of 150 of each 300 of A): with R of 500 items, reached or not on R changes in 0.48% of draws, a word in 0.93%, and the error in updates to 80 on R has a 90th percentile of 1,280 and a 99th of 5,120 updates; with A of 300 items, reached or not on A changes in 0.66% of draws and a word in 1.15%; with both, a word changes in 2.0% [3.3]. The standard deviation of a share near 80 rises from 0.89 to 1.79 points on R and from 1.63 to 2.31 on A [3.3]. The draws are independent between rows; a fixed smaller panel would make the rows' errors correlated, so these figures describe the size of the error, not its exact pattern. Since the pairs that the readings compare often differ by less than 1,280 updates (section 3.1), smaller panels are not advised.

**3.4 Work that the panels repeat.** The length test at 12 letters consists of the same 300 items as A's `add89` panel (same pairs, same random generator, same order): in all 19,453 rows on record its count equals A's `add89` count [3.5]. It need not be scored twice. The A and T panels also score 300 items each of `max89` and `min89`, which enter neither A nor T but only E50's figure of the four operations; dropping them would change that figure's content and is listed in section 7 only as an option. The length test at 40 letters is 16.4% of the scoring work [7.1]; scoring it only at 24,576 and at the end would lose the record's curve of length, which no planned reading uses, and saves 16% of a row's work under the present scorer, under 2% after section 7's changes.

## 4. A shorter or smaller task

**4.1 The record.** No run on record was trained with fewer than 0 to 12 letters, so the effect of 0 to 4 letters on the planned networks is not on record. What is on record: (i) Trained on 0 to 12 letters, networks extrapolate to 40 letters: among 108 runs at or above 80 at 12 letters at the end, the median at 40 letters is 92.6 and 98 are at or above 80 [4.3]. (ii) A (12 letters) reaches 80 before R (no letters, but only pairs with a carry or a borrow) in 110 of 135 runs, by a median 1,280 updates [4.3]; the difficulty of R lies in its pairs, not its length. (iii) Fewer pairs (E54, 49,152 updates): with 25% of the pairs the three networks reach R at 7,680 to 40,960 updates against 6,400 to 15,104 with 50% in E42; with 15% and 10% none reaches R [4.2]. Fewer pairs make a run neither shorter nor cheaper an update. (iv) Half the width (E41, 128 cells against 256, dense): 0.040 against 0.109 s an update, but the sweep needs more updates (with half the pairs, a median 15,872 against 6,400 on R), so the CPU to reach 80 is similar under S (about 680 against 740 CPU-seconds) and lower under B [4.1].

**4.2 Items of 10 to 14 positions.** A batch is as long as its longest item: with 0 to 12 letters the mean item has 16 positions but the mean batch 21.99, so 27.3% of the positions computed are padding [7.4]; with 0 to 4 letters a batch has 14 positions. The model gives 0.069 to 0.074 s an update against 0.108 under B, a saving of 32 to 36% of the training CPU; scoring is unchanged if the panels are [4.4]. **What would be lost:** with 12 letters, the first operand stands 16 and 17 positions before the first answer; with at most 4, at most 9. Items 2, 4 and 7 ask how far the network passes what it holds across positions and exchanges (R exchanges a byte; relay to cells away from the patch; E exchanges of error); their answers depend on that distance, and A at 12 letters would become a test of lengths never taught, so that A and the length test would change meaning. Every reference (E50's `C46-one` and `C16-in`, E47, E42's `G`) was trained on 0 to 12 letters and would have to be run again. **Not advised for the nine jobs.** It could be considered for item 8 (batches of 1 and 8, where the cost is highest) only if its references are run under the same items.

## 5. The rate

On the dense network (E41 at 0.001; E52 and E52's other seeds at 0.003; 10 matched runs a rule; 128 and 256 cells; 25% and 50% of the pairs) the rate 0.003 reaches 80 sooner [5.2]: under the sweep in a median 0.56 of the updates on R (0.34 to 1.00) and 0.54 on A, with 3 runs reaching R only at 0.003 (all at 256 cells, 25% of the pairs, within 12,288 updates); under the exact gradient in 0.70 on R (0.45 to 0.97) and 0.76 on A. Risks on record: under B, 3 of 10 runs end more than 2 points lower on R at 0.003; under S, 1 of 10; the largest fall from the best row at 0.003 is 10.4 points (B, 256 cells, 25%, seed 73129, where A also came later, 8,192 against 4,352); no run at 0.003 diverged; at 0.01, 5 of 7 runs end below 80 on R and only 2 reach it [5.1, 5.3]. Because the sweep gains more than the exact gradient (0.56 against 0.70), a change of rate changes the ratio between the rules, which is itself a reading of the record (E61's 2.53 on A). All of this is on the dense network; nothing on record shows the rate on limited connections or a sparse input. **A change of the default rate is not advised;** the rate belongs to item 9, where it is the variable.

## 6. Screening in place of full grids

**6.1 What early levels predict.** Of the 27 runs on 256 cells with 49,152 updates, the slowest that reached had R at 4.8, 12.4 and 22.8 at 8,192, 12,288 and 16,384 (E57 `in16 row`, S), while runs that never reached had R up to 58.5 at 16,384 (E54 `G_f15`, B) [6.1]. Early levels do not separate the arms that reach from those that do not. A first stage of short runs therefore cannot decide the reading of an arm; it can only decide the order of running.

**6.2 The exact gradient bounds the sweep.** In all 81 matched pairs (same arm, seed, rate and dose), the sweep never reached R when the exact gradient did not; the exact gradient reached first or alone in 57 [6.2]. Running B first in each arm, and S after it, is therefore a safe order; it is not a reason to omit S where B learns, nor, strictly, where B fails (the record contains no counter-example, but the planned rules and readouts are new).

**6.3 Two values first.** For each of the first three jobs, a first stage of two grid values decides whether the middle value is run:

- *Item 2* (R = 2, 4, 8 and two controls, both rules). Run R = 2 and R = 8 under both rules, with the controls. Run R = 4 only if R = 8 reaches 80 on R and A and R = 2 does not (the order's outcome needs the smallest sufficient R, and R = 4 then brackets it). Saves 5.8 CPU-hours by the model (20% of the job) in every other outcome [6.3]. The order's own sequence (R = 8 and R = 2 first) already allows this.
- *Item 3* (four readouts, both rules). Run arms (i) (256 cells, no `l2`), (iii) (8 cells) and (iv) first; run arm (ii) (32 cells) only if (i) reaches and (iii) does not. Saves 3.1 CPU-hours (about 26% of the job by the model) when (i) and (iii) agree [6.3].
- *Item 4* (R = 4 and 8 on E47's network). Run R = 8 under both rules first; run R = 4 only if R = 8 brings the outer rings into use. Saves R = 4's cost (about 23 CPU-hours at 24,576 updates under the assumption of section 1.4) when R = 8 does not; the reverse order saves R = 8's cost (about 44) when R = 4 already does [6.3]. Because the question is whether time for relay brings the outer rings into use at all, R = 8 first is the order that answers it soonest.

In every case the rule of decision is the brief's own reading of the stage-1 runs at their full length, so the screening changes no reading; only arms whose answer would be implied by the others are omitted, and the report states which were omitted and why. Where two values suffice: R = 4 in item 2 and arm (ii) in item 3 when their neighbours agree; in item 4 one of the two values when R = 8 does not help.

## 7. Savings in the code that leave every result unchanged in definition

**7.1 The scorer.** `exp11_cell.respond` generates each byte by running the whole item again (`forward(m, p, ...)` on the prompt and the bytes generated so far) and computes the readout at every position, although only the last position's output is used. The work of a row divides as A 28.2%, T 28.2%, R 20.1%, the length test at 12 letters 7.0%, at 40 letters 16.4% [7.1]. By the executed multiply-adds [7.2]:

| change (scoring only) | work, share of now | CPU-s a row (now 6.66) |
|---|---|---|
| length test at 12 letters taken from A's `add89` | 0.930 | 6.19 |
| readout only at the last position | 0.556 | 3.70 |
| the state carried: each generated byte one new position | 0.213 | 1.42 |
| carried state and the input map as a table of 258 rows | 0.088 | 0.59 |
| the same and the duplicate removed | 0.082 | 0.55 |

The last figures leave out the fixed cost of Python's loops, which the model of training puts at about 13% of an update; a row will probably cost about 1 to 1.5 CPU-seconds, against 6.6, so that the scoring share of a run falls from about 17% to about 3 to 4%. Grouping the panels of equal prompt length (A, T and the length test at 12 all have 19 bytes) into batches of 256 would remove the short last batches (44 items of each 300) but was not measured. **How to show the counts unchanged:** score the saved weight files of the record (start, at 24,576 and at the end, for E50's six runs and two of E42's) with both scorers and compare every generated answer of all 5,000 items; the receipts' counts of the last row and the answers files of the length test must be reproduced exactly. A difference can come only from a near tie between two outputs in the last bit (the present scorer itself computes the earlier positions in products of different sizes at each step); any such item must be listed. The scorer can be changed for every planned job, with no bridge run, once this comparison is clean.

**7.2 Training.** The loss is taken at the three answer positions, but the readout is computed at all 22 positions of a batch, and the input map multiplies the fixed code (a function of the byte alone) at every position. By the model [7.3]: the readout at the answer positions only, 0.070 s an update under B against 0.108 (0.083 against 0.128 under S); with the input map as a table of 258 rows computed once an update, 0.043 (0.051). These change the order of summation in the gradients, so a run is not bit-identical to the present code and, over thousands of updates, follows another path within the same definition. **How to show it:** the first batch's loss equal to the stored value bit for bit; the gradient of each map on one batch equal to 1e-6 relative; then either the weights after 256 and 1,024 updates bit-identical (which would make the change free), or, if they are not, one bridge run (section 8). These changes would roughly halve every job's training CPU; they are not counted in the protocol's figures below.

**7.3 Processes and threads.** The same computation (E49's `in16` B, rerun bit for bit as E57's `in16 none` B: identical R, A and loss at every row) cost 0.115 s an update in E49's queue (two processes at a time beside the rest of a stage, as its receipt records) and 0.139 beside three others, 21% more, and 17% more a row [1.3]. On a line of four cores, four processes instead of three would raise throughput by about 4/3 / 1.21 = 1.10 and lengthen each run by about a fifth. Among E41 and E52's runs of 128 cells, zero, one or two processes beside made no difference (0.040 to 0.044 s) [1.3]. Two threads a process are not on record (every receipt records one thread); at matrices of 64 by 256 by 768 the gain of a second thread is uncertain, and three processes of two threads would oversubscribe four cores. Neither is advised without a timing of 512 updates under each arrangement on the line itself.

## 8. What must not change, or must be bridged

To be compared with the record, a new run must keep: the panels (their items, seeds and sizes: A and T 600 of `add89` and `sub89` at 12 letters, R 2,000 items without letters, the length test 300 at 12 and 300 at 40); greedy generation with at most four bytes; the definition of updates to 80 (three consecutive rows 256 updates apart, so that the adaptive schedule of section 3.2 must score every 256 updates once either panel is at 40); the stream (the seed's order of items, 64 items a batch, 0 to 12 letters, half the pairs); AdamW, the rate 0.001 and the decay 0.25; the start; and the dose named in the reading ("within 49,152 updates"). The references a planned reading names (E50 `C46-one` S: 80 on R at 34,304, on A at 35,584, R at the end 87.2; B: 15,360, 14,080, 94.2; `C16-in` S: neither, R at the end 43.6; B: A at 40,704, R at the end 79.4; E42 `G`: 6,656 and 6,400 on R; E47 `one`: 17,664 under S and 11,008 under B on R) [8.1] must not be rerun under a stopping rule.

A change of the training code that is not bit-identical (section 7.2) needs one bridge run: E50's `C46-one` under B (the cheapest reference that reaches, 1.9 CPU-hours, with its full 49,152 updates and the old schedule of rows), whose updates to 80 should fall within the record's resolution of a few hundred updates of 15,360 and 14,080; on one seed this shows only that the change does not move a reading by more than the run-to-run variation, which the record does not measure for that network. A change of the items, the rate, the panels or the pairs cannot be bridged by one run; it makes a new series that needs its own references.

## 9. A recommended protocol for the nine jobs

1. **Stopping:** stop a run 8 rows after it is settled (80 held on R and on A for three rows) and never before update 24,576; report the level at the stop as such. Run to the full length every run that does not settle, every reference arm, and every run whose level is quoted.
2. **Scoring schedule:** every 1,024 updates while R and A are both below 40, every 256 updates after either reaches 40; always the rows at 24,064, 24,320, 24,576 and the last three.
3. **Panels:** unchanged; the length test at 12 letters taken from A's `add89` answers instead of being generated again.
4. **Scorer:** the state carried between generated bytes, the readout at the last position, the input map as a table; adopted after the rescoring of saved weights shows every answer unchanged.
5. **Training code:** unchanged for the first three jobs; the faster training of section 7.2 only after a bit-identity test or a bridge run.
6. **Items:** 0 to 12 letters, as now. **Rate:** 0.001, as now (the rate is item 9's variable).
7. **Screening:** the two-value first stages of section 6.3 for items 2, 3 and 4; within each arm the exact gradient first.
8. **Processes:** three a line, one thread each, until a timing on the line shows otherwise.

**Expected cost** (CPU-hours; the order's count is kept as the present figure and the protocol's ratios from the model are applied to it; "none reaches" assumes no run settles and so no early stop; "all reach" assumes every run settles as fast as the record's 256-cell runs that settle; wall hours on one line by a longest-first assignment to three processes) [9.0 to 9.2]:

| order's item | runs | as counted | protocol, none reaches | protocol, all reach | further saving by screening | wall hours, as counted → protocol |
|---|---|---|---|---|---|---|
| 2 relay at the input that failed | 10 | 26 | 21.4 | 13.4 | about 20% if R = 4 is omitted | 9.6 → 8.0 |
| 3 the readout made local | 8 | 10 | 8.4 | 5.2 | about 26% if arm (ii) is omitted | 3.8 → 3.2 |
| 4 relay on E47's network | 4 | 26 (67 by E47's receipts) | 19.3 (about 50) | the same (dose 24,576) | 23 or 44 by section 6.3 | 8.9 → 6.7 |
| 5 the assembled network | 10 | 25 | 21.3 | 13.3 | – | 10.0 → 8.5 |
| 6 requirement A | 8 | 8 | 6.8 | 6.8 | – | 3.0 → 2.6 |
| 7 a local rule | 10 | 20 | 17.0 | 10.6 | – | 8.0 → 6.8 |
| 8 items per change, reset, plain steps | 8 | 25 | 21.3 | 13.3 | – | 9.4 → 8.0 |
| 9 requirement B | 15 | 35 | 29.8 | 18.6 | – | 11.7 → 9.9 |
| 10 connections set by each cell | 4 | 10 | 8.5 | 5.3 | – | 5.0 → 4.3 |

The order's items 2 to 10 sum to 185 CPU-hours (the task's round figure is 160). Under the protocol they come to about 154 if no run settles and about 106 if every run settles (item 4 taken at the order's count; by E47's receipts add about 30); the faster training of section 7.2, if bridged, would remove a further 35 to 60% of the training CPU in either case. The ranges of the "all reach" column are those of the record's 256-cell runs (a run that settles saves 37.5% of its updates on average); for items 5 to 10, whose arms are not yet defined, the scoring share is taken as 16% and the saving is only indicative.

**Risks, stated plainly.**

1. *Item 4 is probably under-counted.* If E47's cost grows with the number of exchanges as section 1.4 assumes, its four runs cost about 67 CPU-hours, not 26, and a line is occupied about 22 hours, not 9. The assumption rests on receipts, not on the source of E47's runner.
2. *A stopped run cannot be quoted at its end.* The level at the end, the length test at the end and a later fall below 80 (8 runs on record) are unknown for a stopped run; a report that mixes stopped and full runs in one table of levels would compare unlike quantities. Every table must say which runs were stopped and where.
3. *The rules were chosen on the record.* The stopping rule changes no reading by its construction, but the adaptive schedule's threshold (40) and the screening order were checked on curves that are mostly dense networks and the exact gradient; on new rules and readouts a run may jump from below 40 to settled within 1,024 updates (the record's sharpest case at a threshold of 50 moved A by 256 updates). The scorer's change, if adopted without the rescoring check, could alter counts at near ties. A change of the training code, of the items or of the rate, if adopted, would make the new runs incomparable with every reference of the record unless bridged as section 8 says.
