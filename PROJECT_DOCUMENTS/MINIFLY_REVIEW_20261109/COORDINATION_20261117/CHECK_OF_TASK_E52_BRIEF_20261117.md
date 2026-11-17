# Check of the E52 brief before release (17 November 2026)

By an agent of the reviewer that had not seen the reviewer's reasoning. Checked: `TASK_E52_THE_TWO_RULES_UNDER_OTHER_LEARNING_RATES_20261117.md` beside this file, revision 1, as committed at `46991a56` (sha256 `800bd6b5…`). It was read against the default branch at `7cabb13a` and against E46's branch `origin/claude/wonderful-edison-8hx6i6` at `b76f5979`. Everything here is read-only: nothing was trained, torch was not used, and no code of another branch was run. The figures come from the script in section 4, whose parts A to G are named where they are used, or from the file and lines named beside them. This is not the independent audit that the programme's rules ask for.

**Verdict.** The brief should not be released as it stands. Defects 1 to 3 would let the job finish and report a result that does not answer its question. Defect 4 spoils one score in half of the runs. Defects 5 to 10 are repairs of wording. With the changes the job costs 12 to 22 CPU-hours instead of 8, depending on the choices in defect 9.

## 1. Defects, most serious first

### 1. Holding learning rate × decay fixed does not leave one thing changed; it moves the size of the weights in proportion to the rate (kinds b and c)

The brief's statement about torch is right. In `torch/optim/adam.py` (`_single_tensor_adam`; `AdamW` is `Adam` with `decoupled_weight_decay=True`) the decay is `param.mul_(1 - lr * weight_decay)` and the step is `param.addcdiv_(exp_avg, denom, value=-step_size)` with `step_size = lr / bias_correction1`. E46's check 2 measured the same on torch 2.15.0+cpu at rate 0.001: the decay recovered from one update is 0.24998 and 0.05001 for 0.25 and 0.05, and the byte table is bit-equal (`exp45/e46_qual/check2_73129.json` on E46's branch).

The brief's reason for its rule is mistaken. Both terms of the update carry the rate. If the decay stays at 0.25, the whole update is scaled by one factor and the balance between the step and the shrinkage is E41's. If rate × decay is held at 0.00025, the step changes and the shrinkage does not, so the balance changes by the factor of the rate: 3.3 times each way, ten times between the two new rates. Two things then vary together, which is what the paragraph says it avoids.

What shows that this matters here:

- Part G (an illustration with one weight, not the network). Under the brief's rule a weight that the loss keeps pushing is held at 0.30 and 3.0 times E41's size at 0.0003 and 0.003, and a weight driven by noise at 0.29 and 2.9 times. With the decay kept at 0.25 the figures are 0.99 and 1.00 for a pushed weight, and 0.51 and 1.78 for a weight driven by noise. Neither rule leaves the weights alone; the brief's moves them more.
- Part B. The weights of this network are held by the decay. On seed 73129 under the sweep the norm of the recurrent map is 28.75 at 12,288 updates and 28.50 at 24,576 under decay 0.25, 44.34 and 44.71 under 0.05, and 59.22 and 71.99 without decay; the input map, `lm` and `l2` behave alike, and back-propagation gives the same picture. Part F: the mean size of a decayed weight in E41's saved networks is 0.04 to 0.06, the order that noise alone gives in part G (0.045).
- The size of the weights is part of the sweep. The error passed back through a map is `S = weight.sign() * weight.abs().mean()` (`exp3_state.py`, line 100), and the network has no normalisation.
- E46 found the sweep's final R lower without decay on all three seeds (96.3, 98.8, 98.1 under 0.25 against 92.8, 97.3, 92.9), that is, with larger weights. In those runs the recurrent map's norm is 2.1 to 2.5 times its norm under 0.25 (part B). At 0.003 the brief's rule sets the decay to 0.083, and by part G the weights would be about three times E41's, the size at which E46 saw the sweep end lower.
- At 0.0003 the brief's rule sets the decay to 0.8333. No decay above 0.3 has been run on this network in the code I read (0.3 in E20-B, `exp20_reference.py` line 53; 0.25 since E26; 0, 0.05 and 0.25 in E46). The further value 0.0001 would run at decay 2.5.

So a gain or a loss at another rate could not be put down to the rate.

**Change.** Replace the paragraph "The decay a update is held fixed" by:

> **The decay stays at 0.25.** AdamW's update is w ← (1 − rate × decay) w − rate × step. Both terms carry the rate, so with the decay left at 0.25 the whole update is scaled and the size at which the decay holds a weight that the loss keeps pushing does not move. The size of a weight held against noise still changes, by about the square root of the rate (0.5 and 1.8 times E41's). Every scored row therefore records E46's seven norms (`exp46_decay.norms`, lines 160 to 163 on `origin/claude/wonderful-edison-8hx6i6`), and the report tables them at each threshold and at the end. The byte table stays without decay. The other rule, rate × decay held at 0.00025 (decay `0.25 * 0.001 / rate`, computed by that expression), moves the weights in proportion to the rate. It is run as a control at the half width with half of the pairs under both rules at each new rate (four runs) and is read beside the main runs.

In "The runner" replace "and group 0's `weight_decay` to the run's decay" by "and `weight_decay` is left as `make_opt` set it (in the four control runs group 0's is set to `0.25 * 0.001 / rate`)". In "One more value" replace "with the decay set by the same rule" by "with decay 0.25". If the reviewer keeps the brief's rule as the main one, the least that is needed is to strike "two things would vary together", to say that the decay and with it the size of the weights vary with the rate, to record the norms, and to add four runs at decay 0.25.

### 2. The readings would be decided by the scatter of one seed (kind c)

"Gains" is a ratio of at most 0.8, or reaching 80 where the run at 0.001 does not. Both can come from noise.

- Part B. E46 changed only the decay, seed by seed, and the record describes the result as "stays". On R the count at decay 0.05 is 0.80 to 1.20 of the count at 0.25, and the lesser of the two other decays is exactly 0.80 of the count at 0.25 in two of six cases (both rules on seed 116089). The brief's rule would call both a gain.
- Part A. Within one run the single-row count is 0.71 to 0.80 of the three-row count on R in five of the seven cells of seed 73129 that reach 80. One or two rows that dip below 80 therefore move a count by a fifth to a quarter. R changes by 2.2 to 7.7 points (root mean square) from one row to the next over the last 16 rows of these runs. The scored items are the same at every row, so this is change in the network and not sampling.
- Part A. Between E41's three seeds at 0.001 the least count is 0.44 to 0.73 of the largest on R and 0.57 to 0.86 on A.
- E41 and E46 call a ratio between two thirds and 1.5 "stays" (`exp41_scale.describe_step`, lines 421 to 429). A ratio of 0.8 is inside that band.
- Part A. In the cell of P52.3 (sweep, present width, a quarter of the pairs, seed 73129) R at the last three rows of the run at 0.001 is 79.95, 80.75 and 80.35. The first is 1,599 answers of 2,000. The three-row rule fails by one answer. "Reaches 80 at its best rate and does not at 0.001" would be decided by that answer.
- Part A. Seed 73129 has the smallest lag at 0.001 of the three seeds in all seven comparisons that can be formed. At half of the pairs it is 1.07 on A and 1.22 on R at the half width, and 1.10 and 1.04 at the present width. "Smaller", "larger" and "gone" have no boundary in the brief, and P52.4 sits in the cell where the lag is 1.07.
- The best rate is the least of three noisy counts, one of which is the count at 0.001, so a rule can only appear to gain.
- Undefined cases: the score on which "gains" and "the lag" are taken when the best rate was chosen on A or on the final level; the lag when the sweep's best rate is chosen on A and back-propagation's on R; the lag at 0.001 where the sweep does not reach 80; a tie between rates (counts are multiples of 256); "gains as much" in the sentence on refutation.

**Change.** Replace the first three bullets of "How the result is stated" by:

> Each reading is made on R and on A separately, by E41's three-row rule. A run that does not reach 80 on a score has no count on it; its last recordable count is its dose less 512.
>
> - *The best learning rate on a score*: the rate with the fewest updates to 80 on it. Counts that differ by at most one row (256 updates) are tied; a tie that includes 0.001 is given to 0.001, and any other tie is reported as a tie. Where no rate reaches 80 on the score, the cell is *not reached* on it and no best rate is named.
> - *Faster, slower or the same at another rate*: the run at 0.001 and the run at the other rate are described by E41's table of four cases (`exp41_scale.describe_step`, lines 421 to 429), with the run at 0.001 in the place of the narrower run. The rule is *faster* below two thirds of the count at 0.001, *slower* above 1.5 times, *the same* between, and *not decided* or *neither reaches* as the table says when a run does not reach 80. *The rule gains from its own rate* in a cell if it is faster at some rate on R and not slower at that rate on A; where no rate reaches 80 on R, if it is faster at some rate on A.
> - *The lag on a score*: the sweep's updates to 80 over back-propagation's, each at its best rate on that score, beside the same ratio at 0.001. Where a count is missing the lag is given as a bound from the last recordable count, or as "not formed" when both are missing. The lag at the rules' own rates is *smaller* (*larger*) only if it is below two thirds of (above 1.5 times) the lag at 0.001, *gone* if it is smaller and at most 1.0, and otherwise *the same within the scatter of one seed*.

In the sentence on refutation replace "or if back-propagation gains as much wherever the sweep does" by "or if, in every cell in which the sweep gains, back-propagation gains too and the lag is not smaller". Restate P52.3 on the level: "the sweep's mean of R over the last three rows is at least 85 at some rate tried (at 0.001 it is 80.4, 78.8 and 73.9 on the three seeds)". Move P52.4 to the half width with a quarter of the pairs, where the lag at 0.001 on seed 73129 is 1.70 on A and 2.12 on R. Advised, at no cost: run the trial on seed 93683, whose lag at 0.001 is not the smallest of the three seeds in any cell (1.38 on A and 1.53 on R at the half width with half of the pairs). E41's runs and weights of that seed are on the default branch, and check 1 stays on seed 73129 as in E41. The figures for seed 73129 in this file would then be replaced by that seed's from part A.

### 3. At E41's dose the rate 0.0003 is not read (kind c)

Part C. If the updates to 80 grow as 1 / rate, a run at 0.0003 needs 10/3 of E41's count. On seed 73129 that exceeds the last recordable row in all seven cells that reach 80 on R at 0.001, and in six of eight on A; the two that fit are back-propagation at the half width. The runs at 0.0003 are eight of the sixteen. They could say only "slower than 0.001", which needs no run.

What the brief would wrongly conclude:

- The final levels and the length test at 0.0003 would be those of unfinished runs and would be read as "ends lower".
- P52.1 would come true by construction.
- The fallback "highest mean of R over the last three rows" compares unfinished runs and favours the faster rate.
- The one reading for which a lower rate matters has no room. At the half width with a quarter of the pairs the sweep's R over the last three rows is 82.8, 84.3 and 82.4 against back-propagation's 89.8, 96.2 and 91.8, and on seed 73129 its R changes by 7.7 points from row to row against 3.7 (part A). A smaller step is the ordinary remedy for an unsteady update.
- The further value 0.0001 would be run at a tenth of the updates it needs.

At the present width E41's dose also ends in mid-rise at 0.001: in E46 the sweep's R on seed 73129 goes from 86.9 at 12,288 updates to 96.3 at 24,576. A "final level" there measures how far a run has come.

**Change.** Replace the row "Dose" of the table by:

> | Dose | At 0.003: E41's, 40,960 updates at the half width and 12,288 at the present width. At 0.0003: E41's × 10/3, rounded up to a row: 136,704 and 40,960. Rows every 256 updates. Final levels at 0.0003 are tabled at the end of the run and at E41's dose. |

Replace "if the fewest updates to 80 on R (on A where R is not reached in any run of the cell) are at 0.0003, the cell is also run at 0.0001; if at 0.003, also at 0.01" by "if the best rate on R (on A where the cell is not reached on R) is 0.003, the cell is also run at 0.01 at E41's dose; a tie is not a best rate. If the best is then 0.01, it is reported as 'at or above 0.01'. No run is added below the lowest rate; a best rate there is reported as 'at or below it'. A cell that is not reached on both scores gets no further run." The best rate is the one defined in defect 2. A cheaper form is 0.0005 at twice E41's dose (defect 9). If the cost must stay at 8 CPU-hours, leave the lower rate out and say that the trial asks only whether a higher rate helps.

### 4. The length test named would contain taught pairs at a quarter of the pairs, and no length test exists at 0.001 (kinds a and c)

`exp11_cell.split` seeds its shuffle with the fraction (line 238: `Random(seed * 1000 + round(frac * 100) + m * 7 + code[0])`), so the pairs held out at 0.50 are not held out at 0.25. Part E: of the 300 pairs of the brief's test, 68 are taught by the stream at a quarter of the pairs on seed 73129 (79 and 75 on the other seeds). E41 guards its own panel of this kind with `e26panel = (H == 512 and fp == 50)` (`exp41_scale.py`, line 160). The call as printed, `split(.5, seed)[1]`, also lacks the task; it is copied from a text field of E46's receipts. E41's runs at 0.001 scored no length test, so "tabled at every rate" cannot be done from the receipts.

**Change.** Replace the sentence "At the last row the runner also scores the length test … has it)." by:

> At every scored row the runner also scores the length test of E42 and E46 on pairs that the run's own stream never teaches: `E.panel(t, E.split(t, f, seed)[1], seed, ks=(12, 40), n=300)` with `t = ("add", 89, b"p")` and f the run's fraction. At f = 0.50 this is E46's panel (`exp46_decay.length_panel`). Its count at 12 letters is asserted equal to the row's `A_counts["add89"]`, since they are the same 300 items. For 0.001 the test and the seven norms are taken from E41's saved weights `exp45/e41/e41_<rule>_h<H>_f<25|50>_<seed>.pt`, loaded into `exp41_scale.NetW(H, U)` after their sha256 is checked against the receipt's `weights_sha256`. E41 saved its last row only, so the table gives the last row at every rate, and the mean of the last three rows beside it for the new rates.

One row of this test is a weak figure: in E46's runs of seed 73129 at decay 0.25 the count at 40 letters lies between 72.7 and 89.3% under the sweep and between 67.7 and 96.0% under back-propagation over rows 9,216 to 12,288. If the comparison with 0.001 at 40 letters or in the norms is to carry weight, the eight cells are run again at 0.001 by the new runner (3.9 CPU-hours). Their final `tensors_sha256` must then equal E41's, which is the strongest form of check 1.

### 5. The rule for one more value and the handling of a failed run are not defined in every case (kinds a and c)

The rule is silent on a tie between an end rate and 0.001, and on a cell in which no rate reaches 80 on either score. After a fourth run the best may again be at the end, and the brief does not say how that is reported. A run whose loss becomes non-finite continues to its dose, because `exp41_scale.run` only records `finite_losses`. "Cell" means a rule, width and fraction in "Runs" (eight cells) and a width and fraction in the predictions and in "After the trial" (four).

**Change.** The replacement in defect 3 covers the first three. Add to "The runner": "A run whose row has `finite_losses` false is stopped at that row and recorded as diverged; it counts as not reached, and no higher rate is run in its cell." Write "cell (rule, width, fraction)" in "Runs" and "width and fraction" in the predictions and in "After the trial".

### 6. The receipts would record learning rate 0.001 and decay 0.25 for every run (kind b)

`exp26_candidate.opt_record` (lines 79 and 80) writes `lr=LR, weight_decay=WD`, the module's constants, whatever the groups hold. `exp41_scale.run` uses it for `optimizer` (line 164) and `optimizer_read_back` (line 197), and the brief asks for E41's format. E46 met the same fault for the decay and wrote its own record (`exp46_decay.py`, lines 178 to 180); that record still writes `lr=C.LR`. Nothing else on the path of a run depends on the rate or the decay: the eleven sources hold no scheduler, `make_opt` asserts the groups before the change (line 75), and the groups are read only by `exp20_reference.opt_groups` (line 97). `exp26_candidate.admit_e26` (lines 185 and 186) compares a birth's recorded optimizer with the constants; it is not on this job's path, but a birth made at another rate could not be admitted to a life by it.

**Change.** Add to "The runner": "The fields `optimizer` and `optimizer_read_back` are not taken from `exp26_candidate.opt_record`, which writes 0.001 and 0.25 whatever the groups hold. The runner writes `dict(name="AdamW", lr=<the run's rate>, weight_decay=<the run's decay>, betas=[0.9, 0.98], groups=exp20_reference.opt_groups(m, opt))`, and asserts the groups equal to the expected ones at the start and at the end of the run, as `exp46_decay.py` does at lines 174 and 240. The receipt records `exp52_rate.py` as the runner, with `exp41_scale.py` among the sources."

### 7. Check 2 cannot fail on what it is for, and check 3 has no limit (kind e)

Check 2 reads back from the groups what the runner has just written into them. It does not show that an update uses the values. The batch digest depends on the seed and the fraction only (`exp41_scale.stream`, lines 91 to 93), so no rate can change it. Check 3 names no forecast at which the worker reports before starting; E41's check 6 had one. Check 1 is sound.

**Change.** Replace check 2 by:

> 2. **The two numbers act as set, and nothing else differs.** From E41's start (seed 73129, present width, f = 0.50, its first batch) under back-propagation, for each new rate: (a) one update at (rate, the run's decay) and one at (rate, decay 0) differ in every tensor of group 0 by rate × decay × the starting tensor within 1%, and not at all in the byte table (E46's check 2, `exp46_decay.py` lines 306 to 333, with the rate in the place of 0.001); (b) the change of every tensor in one update at (rate, decay 0) is rate / 0.001 times its change at (0.001, decay 0) within 1%. (c) In every run the groups read back at the start and at the end equal the expected ones, with betas (0.9, 0.98). (d) In every run the batch digest at row 4,096 equals the one in E41's receipt of the same seed and fraction (bbd71ab7… at 0.50 and a71b1ca1… at 0.25 on seed 73129).

Add to check 3: "If the forecast for the first runs exceeds the brief's figure by more than a quarter, report before starting."

### 8. Figures and statements about the record (kind d)

- "1.1 times the updates at the present width with half of the pairs taught." Part A: at the present width the three seeds give 1.10, 1.19 and 1.24 on A and 1.04, 1.19 and 1.25 on R; the ratio of the medians the brief itself quotes is 5,376 / 4,352 = 1.24. The figure 1.10 is the median of nine cases over three widths (`E41_E46_VERIFICATION_20261117.md`, line 154) and also seed 73129's own. Write "1.2 times (1.10, 1.19 and 1.24 on the three seeds)".
- "more at the half width and with a quarter of the pairs, where it reaches 80 on R in 4 of 9 runs." The nine runs are the three widths at a quarter of the pairs: three of three at the half width, none at the present width, one at the double width. Write "more at the half width; with a quarter of the pairs it reaches 80 on R in 3 of 6 runs at the two widths used here, all three at the half width".
- "The optimizer of every experiment since E26 … weight decay 0.25." E46 ran decays 0 and 0.05, and `PLAN.md` lists the first long lives with no decay. Write "The optimizer of E41, which is E26's, is …".
- "E46: the decay does not change the updates to threshold under either rule." Part B: under back-propagation two seeds need 1.50 times the updates to 80 on A without decay (5,120 to 7,680 and 4,096 to 6,144), the boundary of "stays". The reviewer's own verification asks that this be carried with the sentence (item 6). Add "(within the band two thirds to 1.5; under back-propagation two seeds lie on its upper boundary without decay)".
- "The learning rate has not been varied in any run of the present network." `e43_addendum.py` on the default branch ran plain gradient steps (SGD) at 0.01, 0.03 and 0.1 under back-propagation on seed 73129 (receipts `exp45/…/e43_unrelated-plain_lr<rate>_B_73129.json`). Write "AdamW's learning rate has not been varied in any birth of the present network".

The other figures of "What the record says" are right (section 2).

### 9. Cost (kind f)

Part D, from the CPU-seconds of updates and scoring that E41's receipts record: the eight cells once cost 3.90 CPU-hours on seed 73129 (3.98 and 4.12 on the other seeds). The sixteen runs cost 7.8 to 8.2, and with the checks (about 0.2) the brief's "about 9" holds. The eight further runs bring the total to 11.7 to 12.4, which the status line does not say. After the changes: decay kept and 0.0003 at 10/3 of the dose cost 16.9 (7.2 at the half width, 5.8 at the present width, 3.9 for 0.003). The four control runs of defect 1 add 4.7, of which 3.6 are the two at 0.0003. With 0.0005 at twice E41's dose in the place of 0.0003 the main runs cost 11.7, and with controls at 0.003 only, 12.8. Further runs at 0.01 add at most 3.9, and running the eight cells again at 0.001 (defect 4) adds 3.9.

**Change.** In "Status" replace "about 9 CPU-hours" by the figure for the design chosen, with "and up to 3.9 more for the further runs".

### 10. Smaller points (kind a)

- The decays are printed rounded (0.8333, 0.08333). `0.0003 * 0.8333` is 0.00024999. With `0.25 * 0.001 / rate` the product is exactly 0.00025 in double precision at 0.0003, 0.003, 0.0001 and 0.01. Give the expression wherever the product rule is used.
- `<rate>` in the receipt's name has no format. Write "`lr0.0003`, `lr0.003`, `lr0.01` (`f"{rate:g}"`)".
- "A run interrupted by a restart is started again from its beginning" is refused by the runner's own guard, `R_.refuse_existing(rp, wp, …)` (`exp41_scale.py`, line 153), while the partial receipt is in place. Add "after its partial receipt and log are moved to `exp45/e52/interrupted/`".
- Check 2 says "in a run at 0.003" without naming the cell; the replacement in defect 7 names it.

## 2. Verified and found sound

- **The optimizer.** `exp26_candidate.make_opt` (lines 72 to 76) calls `exp16_integrated.adamw16(m, 0.001, 0.25, ())` (lines 176 to 187) with `E16_BOOKDECAY=off` set by `exp20_reference` (line 24). There are two groups: every parameter but `book.weight` with decay 0.25, and `book.weight` with decay 0; betas (0.9, 0.98). All 54 receipts of E41 read back exactly these groups.
- **Setting the groups after `make_opt` is enough for the update.** torch reads `group["lr"]` and the group's `weight_decay` at every step. E46 changed a group in the same way and measured the effect.
- **Check 1.** `exp41_scale.DRY2048` (line 218) holds 79618ab2 and adf933d7. E41's three check-1 receipts record 79618ab29390e6ec… and adf933d7f2ab1a8f… for 2,048 updates at the present width, f = 0.50, seed 73129. E46's check 1a reproduced both with added scores.
- **The digests.** Every E41 receipt of seed 73129 has bbd71ab7ca27b72e at row 4,096 for f = 0.50 and a71b1ca10b24a36b for f = 0.25.
- **E41's figures.** The medians and ranges of updates to 80 on A, the sweep's three present-width values at a quarter of the pairs (8,704, 9,984, not reached), and "does not reach 80 on R at the present width on any seed" are right under the three-row rule (part A).
- **E46.** The sweep's final R is lower without decay on all three seeds (part B).
- **The quotation from E41's brief** is word for word (its line 138). The doses, the three-row rule (`first_row`, lines 390 to 397) and the row interval are as the brief says. Sixteen runs and at most eight further runs are counted rightly.
- **The host.** All 54 receipts record torch 2.15.0+cpu, Python 3.13.18, x86_64.
- **E41's runs at 0.001.** The eight of seed 73129 are complete with finite losses, and their weights are on the default branch with the sha256 their receipts record; the same holds for seed 93683.
- **E46's runner** has the length test (lines 151 and 152). Its count at 12 letters equals score A's count for add in all 1,746 rows of E46's receipts, so it is score A's panel with a column at 40 letters.
- **The range of rates.** At 0.001 neither rule shows a loss spike or a non-finite loss on seed 73129, and where a run reaches 80 it does so between 4,352 and 35,840 updates. A factor of 3.3 each way is a reasonable first bracket once defects 1 and 3 are repaired.

## 3. Not checked

- No check of the brief was executed, because torch is not installed. The torch source read is the main branch on GitHub, not release 2.15.0. E46's measurement of the shrinkage is at rate 0.001 only.
- How the updates to 80 change with the rate. Defect 3 assumes 1 / rate; no run in the record tests it. If 0.001 is well above a rule's best rate, 0.0003 is read in more cells than part C says.
- Whether either rule is stable at 0.003 or 0.01.
- Part G is one weight under an invented gradient. It shows the direction and rough size of the effect in defect 1, not its size in the network.
- E42's length test was not read; that it is E46's rests on E46's docstring.
- Other rates in the record were searched only by a pattern over `MOD97_LOCAL_RULE_20261106/*.py` on the default branch. The inventory of `PLAN.md`, item 3, is the proper check. The optimizers of experiments after E41 other than E46 were not read.
- Bit-equality of check 1 on the worker's host.
- The statements about the question of 15 November and about the width of the long lives were taken from `PLAN.md` and the verification document.

## 4. The commands

Save the block to a file outside the repository and run it from the repository root: `OPENBLAS_NUM_THREADS=1 python3 -I <file> <A|B|C|D|E|F|G>`. It reads receipts and weights with `git show` and writes nothing.

```python
"""Check of the E52 brief: every figure of the check, from the receipts read with `git show`. Read-only; standard library and numpy.
Run from the repository root:  OPENBLAS_NUM_THREADS=1 python3 -I <this file> <A|B|C|D|E|F|G>"""
import hashlib, io, json, subprocess, sys, zipfile
from random import Random
from statistics import median

D = "origin/codex/minifly-consolidated-20261023"; W = "origin/claude/wonderful-edison-8hx6i6"; M = "MOD97_LOCAL_RULE_20261106"
SEEDS = (73129, 93683, 116089); CELLS = [(H, fp) for H in (256, 512) for fp in (50, 25)]
show = lambda ref, path: subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout
e41 = lambda rule, H, fp, s: json.loads(show(D, f"{M}/exp45/e41/e41_{rule}_h{H}_f{fp}_{s}.json"))
e46 = lambda rule, wd, s: json.loads(show(W, f"{M}/exp45/e46/e46_{rule}_wd{wd}_{s}.json"))


def to80(curve, key, three=True):              # exp41_scale.first_row (lines 390-397), threshold 80
    v = [(r["step"], r[key]) for r in curve]
    for i, (u, a) in enumerate(v):
        if (not three and a >= 80) or (three and i + 2 < len(v) and all(x >= 80 for _, x in v[i:i + 3])):
            return u


f = lambda v: "n.r." if v is None else f"{v:,}"
part = sys.argv[1]
if part == "A":      # E41 at 0.001: thresholds, the scatter between seeds, single-row against three-row, the lag on each seed
    for key in "AR":
        for H, fp in CELLS:
            t = {r: [to80(e41(r, H, fp, s)["curve"], key) for s in SEEDS] for r in "SB"}
            one = {r: [to80(e41(r, H, fp, s)["curve"], key, False) for s in SEEDS] for r in "SB"}
            for r in "SB":
                ok = [x for x in t[r] if x]
                print(f"{key} h{H} f{fp} {r}: " + "  ".join(f"{f(a)} [{f(b)}]" for a, b in zip(t[r], one[r]))
                      + (f"  median {median(ok):,}  least/most {min(ok) / max(ok):.2f}" if len(ok) == 3 else ""))
            print(f"   sweep over back-propagation, per seed: " + ", ".join("n.r." if a is None else f"{a / b:.2f}" for a, b in zip(t["S"], t["B"])))
    n = [to80(e41("S", H, 25, s)["curve"], "R") is not None for H in (256, 512, 1024) for s in SEEDS]
    print("sweep, f 0.25, 80 on R reached: half", sum(n[:3]), "present", sum(n[3:6]), "double", sum(n[6:]), "of 3 each")
    c = e41("S", 512, 25, 73129)["curve"]
    print("sweep h512 f25 73129, R at the last three rows:", [x["R"] for x in c[-3:]], "counts", [x["R_counts"] for x in c[-3:]])
    for H, fp in CELLS:
        for r in "SB":
            c = e41(r, H, fp, 73129)["curve"][-17:]
            print(f"   {r} h{H} f{fp} 73129: rms change of R from row to row over the last 16 rows {(sum((b['R'] - a['R']) ** 2 for a, b in zip(c, c[1:])) / 16) ** .5:.1f} points; taught at the end {c[-1]['taught']:.1f}")
elif part == "B":    # E46: one seed at three decays (a change the record describes as "stays")
    for key in "AR":
        for r in "SB":
            for s in SEEDS:
                v = [to80(e46(r, wd, s)["curve"], key) for wd in ("025", "005", "000")]
                print(f"{key} {r} {s}: decay 0.25, 0.05, 0: {v}; 0.05 over 0.25: {v[1] / v[0]:.2f}; least of the other two over 0.25: {min(v[1:]) / v[0]:.2f}")
    for r in "SB":
        c = {wd: e46(r, wd, 73129)["curve"] for wd in ("025", "005", "000")}
        for g in ("recurrent", "input_map", "lm", "l2", "l1", "l3"):
            print(f"norm {r} 73129 {g}: start {c['025'][0]['norms'][g]:.2f}; at 12,288 and 24,576 under 0.25: {c['025'][48]['norms'][g]:.2f}, {c['025'][-1]['norms'][g]:.2f}; "
                  f"under 0.05: {c['005'][48]['norms'][g]:.2f}, {c['005'][-1]['norms'][g]:.2f}; under 0: {c['000'][48]['norms'][g]:.2f}, {c['000'][-1]['norms'][g]:.2f}")
        print(f"final R (last three rows) {r}:", {s: [round(sum(x['R'] for x in e46(r, wd, s)['curve'][-3:]) / 3, 1) for wd in ("025", "005", "000")] for s in SEEDS})
elif part == "C":    # would 0.0003 be read within E41's dose, if updates to 80 grow as 1 / rate?
    for H, fp in CELLS:
        for r in "SB":
            c = e41(r, H, fp, 73129); last = c["dose"] - 512
            for key in "RA":
                u = to80(c["curve"], key)
                print(f"{r} h{H} f{fp} {key}: at 0.001 {f(u)}; times 10/3 " + ("-" if u is None else f"{u * 10 / 3:,.0f} against the last recordable row {last:,}: {'read' if u * 10 / 3 <= last else 'NOT read'}"))
elif part == "D":    # cost from E41's CPU-seconds
    tot = {}
    for s in SEEDS:
        for H in (256, 512):
            tot[(s, H)] = sum(e41(r, H, fp, s)[k] for r in "SB" for fp in (50, 25) for k in ("cpu_updates", "cpu_scoring")) / 3600
        print(f"seed {s}: the eight cells once: half {tot[(s, 256)]:.2f} + present {tot[(s, 512)]:.2f} = {tot[(s, 256)] + tot[(s, 512)]:.2f} CPU-h; sixteen runs {2 * (tot[(s, 256)] + tot[(s, 512)]):.2f}; twenty-four {3 * (tot[(s, 256)] + tot[(s, 512)]):.2f}")
    a, b = tot[(73129, 256)], tot[(73129, 512)]
    print(f"0.0003 at 10/3 of the dose: half {a * 10 / 3:.2f}, present {b * 10 / 3:.2f}; with 0.003 at E41's dose: {(a + b) * (1 + 10 / 3):.2f} CPU-h")
elif part == "E":    # the length test at a quarter of the pairs; exp11_cell.split (lines 235-241) and panel (262-265) copied
    def split(task, frac, seed):
        op, m, code = task; pairs = [(x, y) for x in range(m) for y in range(m)]
        Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs); k = round(frac * len(pairs)); return pairs[:k], pairs[k:]
    t = ("add", 89, b"p"); s = 73129
    red = Random(f"e41|{s}|50|add").sample(sorted(p for p in split(t, .5, s)[1] if p[0] + p[1] >= 89), 1000)
    print("copy of split gives E41's check-7 digest (f433a55ca52ae34d):", hashlib.sha256(json.dumps([list(p) for p in red]).encode()).hexdigest()[:16])
    for s in SEEDS:
        taught25 = set(split(t, .25, s)[0])
        print(f"seed {s}: of the 300 pairs of the brief's length test, taught by the stream at a quarter of the pairs: {sum(p in taught25 for p in Random(s + 3).sample(split(t, .5, s)[1], 300))};"
              f" drawn from split(t, .25, seed)[1] instead: {sum(p in taught25 for p in Random(s + 3).sample(split(t, .25, s)[1], 300))}")
elif part == "F":    # the sizes of E41's saved weights (raw float32 storages in state-dict order), without torch
    import numpy as np
    for r, H in (("S", 512), ("B", 512), ("S", 256), ("B", 256)):
        z = zipfile.ZipFile(io.BytesIO(show(D, f"{M}/exp45/e41/e41_{r}_h{H}_f50_73129.pt"))); root = z.namelist()[0].split("/")[0]
        a = [np.frombuffer(z.read(f"{root}/data/{i}"), dtype="<f4").astype(float) for i in range(13)]      # 0 book; 4 and 6 the recurrent map and its bias
        w = np.abs(np.concatenate(a[1:]))
        print(f"{r} h{H} f50 73129: recurrent norm {np.sqrt((a[4] ** 2).sum() + (a[6] ** 2).sum()):.2f}; all weights under decay: mean size {w.mean():.4f}, largest {w.max():.2f}, above 1.2: {(w > 1.2).sum()} of {w.size}")
elif part == "G":    # an illustration, not the network: where AdamW (betas 0.9, 0.98) holds a weight under noise and under a constant push
    import numpy as np
    rng = np.random.default_rng(0)
    for name, lr, wd in (("E41", .001, .25), ("brief 0.0003", .0003, .25 * .001 / .0003), ("brief 0.003", .003, .25 * .001 / .003), ("decay kept 0.0003", .0003, .25), ("decay kept 0.003", .003, .25)):
        out = []
        for push in (True, False):
            w = np.zeros(500); m = np.zeros(500); v = np.zeros(500)
            for t in range(1, 60001):
                g = -np.ones(500) if push else rng.standard_normal(500)
                m = .9 * m + .1 * g; v = .98 * v + .02 * g * g
                w = w * (1 - lr * wd) - lr * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .98 ** t)) + 1e-8)
            out.append(float(np.sqrt((w * w).mean())))
        print(f"{name}: rate {lr}, decay {wd:.4g}, product {lr * wd:.3g}: held under a constant push {out[0]:.2f}; rms under noise {out[1]:.4f}")
```

Part F reads the saved weights as raw storages in the order of the state dict. The reading is confirmed by its first line: the recurrent norm of the sweep's present-width network is 28.75, the value E46's receipt records for the same run at row 12,288.
