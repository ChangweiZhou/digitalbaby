# Check of the brief of E54, revision 1 (17 November 2026)

Written for the outside reviewer by an agent that has not seen the reviewer's reasoning. The brief is `TASK_E54_THE_LIMITED_AND_THE_PERIODIC_NETWORK_WITH_FEWER_PAIRS_20261117.md` beside this file (its SHA-256 begins `c7c0ec01`). The work was read-only. Nothing was trained, torch is not installed, no code of another branch was executed, and the files of other branches were read with `git show`. This is the check before release that the brief's status line names. It is not the independent audit that the programme's rules ask for.

**Shell names.** `D=origin/codex/minifly-consolidated-20261023` (at 7cabb13a); `W=origin/claude/gallant-dijkstra-i9zgsr` (at 44f91edc; the branch of E42's receipts, which the brief does not name); `M=MOD97_LOCAL_RULE_20261106`. A line number without a file is a line of `$D:$M/exp42_cells.py`. `APP <part>` is the script of the appendix, run from the repository root as `OPENBLAS_NUM_THREADS=1 python3 -I <the script> <part>`. It uses the standard library and `git show` only.

**Terms.** As in the brief. A *level* is the mean of a score over the three scored rows that end at the named update. *Updates to 80* is the first of three consecutive scored rows at or above 80. The *present width* is the network of 256 cells (`h512` in E41's receipts); the *half width* is E41's network of 128 cells (`h256`). Seeds are always given in the order 73129, 93683, 116089.

## Verdict

The brief should not be released as it stands. The code it builds on is sound for the purpose: the dense configuration `G` is E41's network at the present width, update for update, so the third check can hold. Every figure that the brief takes from E42 and from E41 was recomputed and is right, with two exceptions of wording (defect 7). The defects are in the reading of the result and in the checks.

- Defects 1 and 2 change what the job would conclude. The brief reads a level at a fixed update as an end result. At a tenth of the pairs under back-propagation the dense network is still rising at any dose within reach, and at a quarter under the sweep the limited network will, on E42's ratios, still be rising at update 24,576. A network that is only slower would be reported as one that generalises worse, and the explanation under test would be called refuted. Under back-propagation the margin of 10 points is inside the spread between seeds.
- Defect 3 puts taught pairs into the length test: 68 of its 300 pairs at a quarter and 29 at a tenth.
- Defects 4 to 6 concern what the worker is given. The brief has no value of theta and does not name the branch that holds it, it describes a change to `exp42_cells.run` that cannot be made as described, and two of its checks cannot fail for the thing they are named after.
- Defects 7 and 8 are corrections of statements and smaller points.

If the texts given under each defect are applied, I see no need for a further check of the brief.

## Defects, the most serious first

### 1. A level at a fixed update is read as an end result, so that a slower network is reported as one that generalises worse

**What is wrong.** The brief licenses "generalises better than, as well as or worse than the dense one" and "needs fewer samples" from levels at the end of a dose, and it calls the explanation refuted if no limited network is 10 points ahead at a tenth. Six facts of the record stand against this.

(a) *At a tenth under back-propagation the dense network has not finished at any dose within reach.* At the half width its level of R at update 40,960 is 44.0, 61.7 and 53.0, and it rose by 6.2, 5.9 and 3.5 points over the last 8,192 updates. The present width is behind the half width at update 12,288: 14.8, 28.9 and 22.2 against 26.8, 39.5 and 39.1. `G` will therefore be in the middle of its rise at update 49,152. (`APP e41`.)

(b) *The limited and the periodic network are known to be slower.* At a half they needed 1.12 to 2.95 times `G`'s updates to 80 (`APP e42`; on seed 73129, on A, `F10` 2.36 times under the sweep and 1.70 under back-propagation, `P8` 1.36 and 1.25). On E41's half-width curves at a tenth, a learner that follows the dense network's curve 1.5 times more slowly is 7.1, 12.6 and 4.6 points of R behind at update 40,960, and one twice as slow 11.9, 13.3 and 8.8 (`APP spread`). "Behind the dense network" by 10 points is what the slowness measured in E42 produces by itself, whatever the end result.

(c) *At a quarter under the sweep the dose of 24,576 is too short for `F10`.* In E41 the dense network of seed 73129 under the sweep reaches 80 on A at update 8,704 and has not reached 80 on R by update 12,288 (its last four rows of R are 83.85, 79.95, 80.75 and 80.35), so `G`'s updates to 80 on R are 12,032 or more. `F10`'s ratio to `G` on R at a half was 2.27, 2.95 and 2.24 under the sweep. With the smallest of these `F10` reaches 80 on R at update 26,950 at the earliest, which is after the end of the dose. At a half itself `F10` under the sweep was still rising when its dose ended (R rose by 4.5, 6.8 and 5.7 points over the last 8,192 updates) and stood 7.0, 6.0 and 7.7 points of R below `G`. (`APP e41`, `APP e42`, `APP spread`.)

(d) *No rule connects the words to a reading at a quarter.* "How the result is stated" defines *ahead* and *behind* for a tenth only. For a quarter it lists readings and gives no rule for "better than", "as well as" and "worse than".

(e) *The licence at a quarter cannot be earned, so the refutation rests on P54.3 alone.* "Reaching 80 on R at f = 0.25 where `G` does not" is excluded under back-propagation by the record: E41's dense network of this seed reaches 80 on R at update 7,680, and check 3 requires `G` to be that run. Under the sweep the threshold of 80 lies inside the scatter of the dense network's own late level: at the half width R is 82.8, 84.3 and 82.4 at update 40,960, and the scatter of R from row to row at a quarter is 2.0 to 2.4 points (`APP spread`). The explanation is then "refuted for this seed" whenever P54.3 holds, to which the brief gives 75%. Under the sweep P54.3 holds if all three networks stay where the dense one is in E41 (R 6.9 to 10.8 with the taught pairs at 97%), and that says only that none of them generalises from a tenth.

(f) *The two fractions lie far on either side of the fraction at which the dense network begins to generalise.* Its level of R at update 12,288 is 73.9 to 90.6 at a quarter and 6.9 to 28.9 at a tenth. A limited network that needed 15% of the pairs where the dense one needs 20% would show nothing at either fraction. The job can find a lowering of the needed fraction as far as a tenth. It cannot refute a smaller one.

**Repair.** In "Runs", the dose of the three runs under the sweep at f = 0.25 becomes 49,152 (2.9 CPU-hours more at the rates of E42's seed 73129), and P54.2 keeps update 24,576 as its limit. In "How the result is stated", the first two bullets are replaced by:

> - A *level* is the mean over the three scored rows that end at the named update. A curve *has flattened* if its level of R rose by less than 3 points over the last 8,192 updates. A difference between two levels counts as a difference of end results only if the lower of the two curves has flattened. Otherwise the reading is *not decided within the dose*, and the report gives both levels and both rises.
> - At f = 0.25: updates to 80 on A and on R with E42's words and bands (`describe` in `e42_tables.py` on E42's branch: a ratio below 2/3 is *fewer*, above 1.5 *more*, otherwise *about the same*, with its rules for a run that does not reach 80). Beside each ratio stands the ratio of the same seed at a half, from E42's receipts: on A, `F10` 2.36 under the sweep and 1.70 under back-propagation, `P8` 1.36 and 1.25; on R, `F10` 2.27 and 1.48, `P8` 1.27 and 1.12. Then the levels of A, of R and of the length test at the end. A network *generalises better* than `G` if its level of R is 10 points or more above `G`'s and `G`'s curve has flattened, *worse* if it is 10 points or more below and its own curve has flattened, and *as well* if the difference is under 10 points and both have flattened.
> - At f = 0.10: the levels of A and of R at updates 12,288, 24,576 and 49,152, and the rise of R over the last 8,192 updates. A network *generalises from a tenth* if R reaches 80 within the dose. It is *ahead of the dense network at update 49,152* if its level of R is higher by 10 points under the sweep or by 20 points under back-propagation, and *behind* if it is lower by as much. These two words describe one update. They become *better* and *worse* only under the condition of the first bullet.

The sentence that begins "'Needs fewer samples' is licensed only by" is replaced by:

> "Needs fewer samples" is licensed only (i) by a limited network that generalises from a tenth where `G` does not, or (ii) by *generalises better* at either fraction. A ratio of updates to 80 at a quarter that is smaller than the ratio at a half is reported as "loses less speed from fewer pairs" and licenses nothing further.

The sentence that begins "It is refuted for this seed if" is replaced by:

> The job cannot refute it. The two fractions lie on either side of the fraction at which the dense network begins to generalise (in E41 its R at update 12,288 is 74 to 91 at a quarter and 7 to 29 at a tenth), so a lowering of that fraction that stops short of a tenth is not visible. If no licence is earned, the report says "on this seed no lowering to a tenth of the pairs was found" and lists the readings that were not decided.

P54.3 is restated with the two margins before any run. The report on P54.2 adds that its threshold lies inside the scatter of the dense network's late level. By the rule of the first bullet `F10` under the sweep at a half is itself *not decided* at update 24,576, which is where the rule is meant to bite. The limit of 3 points is one to two times the scatter of a three-row level at a quarter (row scatter 2.0 to 4.0 points); the reviewer may prefer nine rows for the two levels of that test.

### 2. Under back-propagation the margin of 10 points is inside the spread between seeds, and the chosen seed is the one on which the dense network is lowest

**What is wrong.** The level of R of the dense network alone, at one setting, over E41's three seeds (`APP e41`):

| Setting | Sweep | Range | Back-propagation | Range |
|---|---|---|---|---|
| f = 0.10, present width, update 12,288 | 6.9, 10.8, 9.0 | 3.9 | 14.8, 28.9, 22.2 | 14.2 |
| f = 0.10, half width, update 12,288 | 7.0, 6.9, 8.7 | 1.7 | 26.8, 39.5, 39.1 | 12.7 |
| f = 0.10, half width, update 24,576 | 8.1, 8.1, 9.7 | 1.7 | 32.9, 53.7, 47.4 | 20.8 |
| f = 0.10, half width, update 40,960 | 10.5, 10.0, 10.3 | 0.5 | 44.0, 61.7, 53.0 | 17.7 |
| f = 0.25, present width, update 12,288 | 80.4, 78.8, 73.9 | 6.4 | 83.7, 90.6, 86.0 | 6.9 |
| f = 0.25, half width, update 40,960 | 82.8, 84.3, 82.4 | 1.9 | 89.8, 96.2, 91.8 | 6.4 |

At a tenth under back-propagation one network with one setting differs between seeds by 12.7 to 20.8 points of R. A difference of 10 points between two networks on one seed is not above that. Using the same seed for both networks does not remove the spread: in E42 at a half, the difference of R between `F10` or `P8` and `G` at one update during the rise (updates 4,096, 6,144 and 8,192) differs between seeds by 5 to 23 points (`APP spread`). The scatter from row to row is smaller (0.3 to 1.0 points at a tenth at the present width, 1.6 to 4.8 at the half width under back-propagation during the rise), so the margin is above the spread between rows under both rules and above the spread between seeds under the sweep only.

Seed 73129 is the seed with the lowest dense level at a tenth under back-propagation, at both widths and at every update of the table. "Ahead of the dense network" is easiest to obtain on it. From this one seed the brief would report that a limited network needs fewer samples under back-propagation on a lead that the dense network's own other two seeds exceed.

**Repair.** The margin of 20 points under back-propagation at a tenth, as in the text of defect 1. In "What it cannot show", the first bullet becomes:

> - More than one seed; widths other than the present one. The dense network alone differs between seeds by 13 to 21 points of R at a tenth under back-propagation (E41), and seed 73129 is its lowest seed there. Every sentence of the report about a tenth under back-propagation names the seed and gives this spread beside the difference found.

An option for the reviewer to weigh: `G` alone at f = 0.10 on seeds 93683 and 116089 (four runs, 6.7 CPU-hours at the rates of seed 73129) would give the spread of the dense network at update 49,152 at the present width.

### 3. The length test would hold taught pairs

**What is wrong.** The brief says "the panels are `exp42_cells.panels(seed, f)`" and "The length test stays E42's". Line 230 draws the length panel from the pairs held out at a half, whatever f is:

    LP = E.panel(LEN_TASK, E.split(LEN_TASK, .5, seed)[1], seed, ks=(12, 40), n=300)

The splits at two fractions are separate shuffles and not parts of one: `$D:$M/exp11_cell.py` line 238 seeds the shuffle with `seed * 1000 + round(frac * 100) + m * 7 + code[0]`. Of the 1,980 addition pairs taught at a quarter on seed 73129, 980 are among those taught at a half and 1,000 among those held out at a half. E42's length panel of seed 73129 therefore holds 68 pairs that are taught at a quarter and 29 that are taught at a tenth (`APP length`; 79 and 75, and 34 and 31, on the other two seeds).

At a tenth the dense network of seed 73129 answers taught additions at 12 letters at 96 and 98% (289 and 295 of 300 under the sweep and back-propagation) and untaught ones at 10 and 20% (29 and 61 of 300). The 29 taught pairs can add up to 9.7 points to the length test, which is the size of the brief's margin. The test would then also differ between networks by how well each answers taught pairs at 40 letters, which is not the question.

**Repair.** In "The runner", the sentence "The length test stays E42's." is replaced by:

> The panels of A, of R and of taught pairs are the first three that `exp42_cells.panels(seed, f)` returns. Its fourth, the length panel, is not used: line 230 draws it from the pairs held out at a half, of which 68 of 300 are taught at a quarter and 29 at a tenth on this seed. The length panel is `E.panel(LEN_TASK, E.split(LEN_TASK, f, seed)[1], seed, ks=(12, 40), n=300)` with `exp42_cells.LEN_TASK`. Before the first update the runner asserts that no pair of the panels of A, of R and of the length test is among the taught pairs of its operation at this fraction, and that every pair of the taught panel is.

A fifth check is added:

> 5. **The length panel.** In every scored row `length_counts["12"]` equals `A_counts["add89"]`, as it does in all eighteen receipts of `G`, `F10` and `P8` of E42. Built in this way, the length panel at 12 letters is the addition panel behind A.

At f = 0.50 this panel is E42's own, so check 1 is not affected. (`APP spread`, last line, for the eighteen receipts.)

### 4. Check 1 cannot detect a wrong periodic network; theta has no value in the brief; E42's branch is not named

**What is wrong.** `P8` needs theta. Line 75 is `THETA = None`, and lines 313 and 314 stop a run of a changing configuration that has no `--theta`. The brief says "E42's value of theta" and gives neither the value nor a place. The value is 0.5. It stands as `theta` in `$W:$M/exp45/e42/e42_P8_S_73129.json`, as `chosen_theta` in `$W:$M/exp45/e42_qual/check10_73129.json`, and in a copy of the receipt on the default branch, `$D:$M/exp45/e43/addendum2/e42_start/e42_P8_S_73129.json`. The brief names neither `$W` nor that copy. E42's receipts of `G`, its dry runs, `e42_tables.py` and `exp42_ckpt.py` are not on the default branch.

No check would notice a wrong theta or a fault in the copied replacement of connections. Check 1 runs `F10`, which replaces nothing, for 256 updates. A connection can be given up only once it is 512 updates old (`$D:$M/e42_cells.py` line 24, `EVENT, MIN_AGE = 256, 512`, and line 180), and E42's dry run of `P8` has its first replacements at update 512 (0, 0, 173, 49 and 208 at rows 0 to 1,024). Check 2 is the stream, check 3 is `G`, check 4 is the cost. Check 1 also compares the hash of the weights only, and that hash does not depend on the panels or on what the rows record.

**Repair.** A shell name is added: `W=origin/claude/gallant-dijkstra-i9zgsr` (E42's receipts under `$M/exp45/e42/`, its checks under `$M/exp45/e42_qual/`, `e42_tables.py` and `exp42_ckpt.py` under `$M/`). In "Runs", "`P8` with E42's value of theta" becomes "`P8` with theta = 0.5 (`theta` in `$W:$M/exp45/e42/e42_P8_S_73129.json`)". Check 1 becomes:

> 1. **At f = 0.50 the runner is E42's.** Two dry runs of the new runner at f = 0.50: `F10` under the sweep, 2,048 updates, and `P8` under the sweep with theta 0.5, 1,024 updates. Each is compared with the same run made with `exp42_cells.py` on this host and with E42's receipt: `$W:$M/exp45/e42_qual/check8_73129_runs/e42_F10_S_dry2048_73129.json` (`tensors_sha256` begins `837dadc1`) and `$W:$M/exp45/e42_qual/check7_73129_runs/rep0/e42_P8_S_dry1024_73129.json` (begins `1b78f306`; replaced 0, 0, 173, 49, 208). Equal are the `tensors_sha256`, the digests of the mask and of `made`, the counts replaced, and the loss, the counts and the batch digest of every row.

The four dry runs cost about 15 CPU-minutes at E42's rates.

### 5. Check 3 can hold, but as written it is not a check before the runs and it cannot fail

**What is wrong.**

(a) The check needs 12,288 updates of `G` under each rule. That is the first half of the two `G` runs at a quarter, or 0.84 CPU-hours if it is run apart. The section is headed "Checks before the runs", and the brief does not say which is meant.

(b) The sentence in brackets lets any difference be put down to the host's arithmetic. It gives no test of whether the arithmetic differs, and no tolerance for the comparison of scores that replaces the hash. Two runs that differ in arithmetic part company row by row, so "compare the scores at every row" has no outcome that counts as a failure. The record gives no ground for the allowance: E41's and E42's receipts were made on hosts of two kernel builds (the `platform` of the receipts), and `G` of E42 equals E41's run at a half in the loss, in the counts and in the digest of all 49 rows to update 12,288, for all three seeds and both rules (`APP same`).

(c) The hash covers the weights. It does not test that the panels of A and of R are E41's at that fraction, which is what makes the scores comparable with E41's. The rows do.

**Repair.** Check 3 becomes:

> 3. **`G` is E41's network.** Beforehand `python3 -I exp42_cells.py check 1 --qual exp45/e54/qual` passes on this host (2,048 updates of `G` at a half give E26's hashes under both rules; about 9 CPU-minutes). If it does not, stop and ask; no comparison of scores takes its place. The check itself is read from the two `G` runs at f = 0.25, which are launched first, and no run at f = 0.10 is launched before it has passed. In every row to update 12,288 the loss, the counts of A, of R and of taught pairs and the batch digest equal those of `$D:$M/exp45/e41/e41_<rule>_h512_f25_73129.json`. The first comparison is at row 256, and a difference stops every run. At update 12,288 the `tensors_sha256` equals the receipt's. The two `G` runs at f = 0.10 are compared in the same way with `e41_<rule>_h512_f10_73129.json`.

### 6. `exp42_cells.run` cannot be given a taught fraction as an argument

**What is wrong.** The brief says: "It is `exp42_cells.run` with the taught fraction as an argument." `run` (line 308) has no such argument. It builds its panels and its stream inside, with a half written in:

    325    A, Rs, TA, LP = panels(seed)
    326    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]

No existing source may be edited, so the worker must either copy the 94 lines of `run` or replace `E.split` or `panels` from outside while `run` executes. The brief does not say which, and the second way would also change what line 230 does. Other lines assume E42 or a half. Line 336 writes the `data` string with `.5`. Lines 303 to 305 name the receipt `e42_...` and add `_dry<dose>` for any dose other than 24,576, so the full run at a tenth would be named a dry run. Line 330 writes `experiment="E42"`. Lines 405 to 410 record the hash of `exp42_cells.py` as the runner's. E42's `run` has no assertion that only taught pairs are presented; E41's has one (`$D:$M/exp41_scale.py` lines 131 to 138 and 188).

**Repair.** The second sentence of "The runner" becomes:

> `exp42_cells.run` (lines 308 to 401) takes no fraction and has a half written into lines 325 and 326. The new file therefore holds a copy of its text with the following changes and no others, and the report shows the difference between the two texts. The panels (as described below) and the stream are built with f (lines 325 and 326), and the `data` string says so (line 336). The receipt has `experiment="E54"`, a field `fraction` and the name given below (lines 303 to 305 and 330). `provenance` adds the hash of `exp54_fraction.py` (lines 405 to 410). The `tensors_sha256` is also recorded at update 12,288. Every batch passes E41's assertion that only taught pairs are presented (`exp41_scale.taught_keys` and `exp41_scale.assert_taught`). Nothing inside `exp42_cells` or `exp11_cell` is replaced while a run executes.

### 7. Three statements about the record and the plan

(a) "Final levels are close (A 90 to 98)." Over the eighteen runs A at the end is 91.2 to 99.1. `F10` under the sweep ends 6.6, 4.5 and 7.4 points of A and 7.0, 6.0 and 7.7 points of R below `G`, on every seed, and is still rising (`APP e42`, `APP spread`). The sentence becomes: "At the end of the dose A is 91 to 99. `F10` under the sweep is 4.5 to 7.4 points of A below `G` on every seed, and its R was still rising by 4.5 to 6.8 points over the last 8,192 updates. The other differences from `G` at the end are under 5 points of R."

(b) "Still rising by 3 to 5 points in 2,048 updates" (half width, a tenth, back-propagation). This holds for the last 2,048 updates (R rose by 3.3, 4.9 and 4.3 points). Over the last 8,192 updates R rose by 6.2, 5.9 and 3.5 points in all, and the scatter of R from row to row there is 1.6 to 4.8 points (`APP e41`, `APP spread`). The clause becomes: "A 47.1, 66.6, 56.4 and R 44.0, 61.7, 53.0, with R rising by 3.5 to 6.2 points over the last 8,192 updates".

(c) "It is item 10 of `PLAN.md`." Item 10 (line 57 of `PLAN.md`) has a second half: "the dense network at the half width with a tenth of the pairs at three times the dose". The brief leaves it out and does not say so. That run would tell whether the dense curve at a tenth flattens below 80 or arrives, which the readings at a tenth need (defect 1 a). Either the sentence becomes "It is the first half of item 10 of `PLAN.md`; the second half, the dense network at the half width with a tenth of the pairs at three times the dose, is not in this job", or the two runs are added: `python3 -I exp41_scale.py run <S|B> 256 10 73129 --dose 122880 --out exp45/e54`, about 3.3 CPU-hours for both at the rates in E41's receipts (0.040 and 0.044 CPU-seconds an update, 1.5 a scored row). Their rows to update 40,960 must equal those of E41's receipt, and line 143 of `exp41_scale.py` will put `_dry122880` into the receipt's name. I did not run that command.

### 8. Smaller points

- "The four operations on the panel behind A are recorded apart at every row." E42's format already does this: `A_counts` holds `add89`, `sub89`, `max89` and `min89` (lines 224 and 235, and every row of E42's receipts). The brief should say that nothing is to be added, or a worker may build a second record.
- "Means of three rows" should read "the three rows ending at the named update", since E41's and E42's tables use the last three.
- Restarts. E42 lost runs to restarts of the container (`$W:$M/exp45/e42/interrupted/`) and then wrote `$W:$M/exp42_ckpt.py`, whose self-test shows resumed dry runs equal to uninterrupted ones in hash, rows and digest (`$W:$M/exp45/e42_qual/ckpt_selftest.json`). The rule on restarts should name it as the base of a runner that can resume. The longest run of this job is 2.4 CPU-hours.
- theta was chosen on the first 4,096 updates at a half. In `P8` under the sweep of seed 73129 the share of the connections replaced at an event is 7.6% over the first sixteen events and 19.8% over the last sixteen (`APP spread`). The report should give these shares at each fraction, since the dose at a tenth is twice E42's.
- E42's words and bands are defined in code in `$W:$M/e42_tables.py` (lines 63 to 72 and 87 to 88), which is not on the default branch. The brief should cite it.

## Verified and found sound

- **The three configurations** are as the brief describes them: lines 58 to 70, with `$D:$M/e42_cells.py` lines 21 to 24 (16 by 16 grid, radius 3; periods 1, 2, 4, 8 for 128, 64, 32, 32 cells; replacement every 256 updates).
- **E42's updates to 80 on A**: all eighteen figures of the brief are right (`APP e42`). theta is 0.5 in every receipt of `P8`.
- **E41's levels**: every level of A and of R that the brief cites at a quarter and at a tenth, at both widths, is right to the decimal given, as are the taught levels of 97 to 99% and the rises at the present width (`APP e41`). The rise of R at a quarter under the sweep is 1.7, 3.3 and 6.0 points over the last 2,048 updates, which the brief rounds to "2 to 6".
- **The stream** named in the brief is E41's (`$D:$M/exp41_scale.py` lines 91 to 93). Rebuilt without torch from `exp11_cell.py`, `exp45.batch_tensor` and `exp26_candidate.Digest`, it gives the digests of E41's receipts at row 4,096: `a71b1ca10b24a36b` at a quarter and `e120924e5af546f9` at a tenth (`APP stream`). Check 2 can pass, and it fails if the stream is built otherwise.
- **The panels of A, of R and of taught pairs** from `exp42_cells.panels(seed, f)` are E41's at the same fraction: lines 222 to 229 are the expressions of `exp41_scale.py` lines 106 to 113. The scores A and R of E54 are comparable with E41's. The length panel is the exception (defect 3).
- **`G` is E41's network at the present width, so check 3 can hold.** The network: line 156 calls `arch26.new_net(seed)`, whose thirteen tensors equal those of E41's `new_net(seed, 512, 256)` (E41's check 1, passed for the three seeds). For `G` the layer changes no weight and adds no parameter (`e42_cells.py` lines 85 to 100), and its draws use `random.Random` only (lines 35 to 46). The optimizer: line 162 is `exp26_candidate.make_opt`, as in `exp41_scale.py` line 156. The update: lines 200 to 208 with lines 97 to 99 make the call of `exp26_candidate.step` (its lines 84 to 91), which E41 uses (`exp41_scale.py` line 190). The stream's generator is its own (`exp11_cell.py` line 246). Between updates E42's runner computes two gradients on the first batch without touching the optimizer (lines 279 to 289 and 363), scores the panels without gradients (`exp11_cell.py` lines 187 to 220), checks the mask, and replaces connections in changing configurations only (lines 367 to 371). None of this draws from the stream's generator or from torch's, and training draws from neither once the network is built. The receipts agree: at a half, `G` of E42 equals E41's run in all 49 rows to update 12,288, in the loss to the last digit, for the three seeds and both rules (`APP same`).
- **Nothing but the taught fraction is varied** in the code, apart from the dose at a tenth, which the brief declares, and the length panel (defect 3). theta, the optimizer, the batches and the letters are E42's.
- **No other run of a limited or periodic network at another fraction exists** on any fetched branch: every runner that imports the cell layer (`exp42_cells.py`, `e43_addendum2.py`, `exp47_local.py`, `exp49_interface.py`, `exp50_code.py`) teaches a half.
- **The cost.** E42's receipts of seed 73129 give 0.85, 0.82, 0.87, 0.86, 1.20 and 1.18 CPU-hours for `G`, `F10` and `P8` under the sweep and back-propagation at 24,576 updates (`APP e42`). The six runs at a quarter cost 5.8 CPU-hours and the six at a tenth 11.5 (updates and replacement doubled, 193 scored rows for 97): 17.3 in all, and 22.9 to 23.5 at the rates in the receipts of the other two seeds. "About 20 CPU-hours" is sound. The repairs of defects 1, 4 and 5 add 3.3. The receipt of a running run is written every 4,096 updates (line 373), that is every 8 to 12 minutes, so the two-hour rule can be met.
- **E48.** The four figures agree with the reviewer's verification note (`E48_RESULT_VERIFICATION_20261117.md`, line 140: 91.1 against 85.6 and 97.0 against 90.2). I did not recompute them from E48's receipts.

## Not checked

- No run was made. Whether the hashes agree on the worker's host, and the CPU times on that host, are not known.
- How `F10` and `P8` behave at a quarter or a tenth. The projections in defect 1 carry E42's ratios at a half over to a quarter; nothing on record says that they hold there.
- E48's figures from its receipts; section 22.12 of the reviewer's memo; the request of the audit of E50 about the four operations; the commit trailers of `AGENTS_POINTER.md`; the session named for `send_message`.
- The replica of the stream was compared with E41's receipts at row 4,096 of seed 73129 only.
- `tools/minifly_check` was not run, since this check may write one file only.

## Appendix: the script behind the figures

Parts: `length`, `stream`, `e42`, `e41`, `same`, `spread`, or `all`. It reads receipts with `git show` and writes nothing.

```python
import hashlib, json, statistics, subprocess, sys
from array import array
from random import Random
D, W, M = "origin/codex/minifly-consolidated-20261023", "origin/claude/gallant-dijkstra-i9zgsr", "MOD97_LOCAL_RULE_20261106"
SEEDS = (73129, 93683, 116089)
show = lambda ref, path: json.loads(subprocess.run(["git", "show", f"{ref}:{M}/exp45/{path}"], capture_output=True, check=True).stdout)
e41 = lambda rule, H, fp, s: show(D, f"e41/e41_{rule}_h{H}_f{fp}_{s}.json")
e42 = lambda cfg, rule, s: show(W, f"e42/e42_{cfg}_{rule}_{s}.json")
part = sys.argv[1] if len(sys.argv) > 1 else "all"

# ---- the task, copied from exp11_cell.py lines 224-265 (random.Random only)
EOS, BOS, BATCH = 256, 257, 64
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
LET = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ap = lambda op, m, a, b: {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]
def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(LET) for _ in range(k)]
    return list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")], list(f"{ap(op, m, a, b):02d}".encode()) + [EOS]
def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]
def batches(f, seed):
    rng = Random(seed + 77); flat = [(t, pr) for t in DEV for pr in split(t, f, seed)[0]]
    def stream():
        while True:
            order = list(range(len(flat))); rng.shuffle(order)
            for i in order:
                t, (a, b) = flat[i]
                yield render(t, a, b, rng.randint(0, 12), rng)
    st = stream()
    while True:
        yield [next(st) for _ in range(BATCH)]
def digest(f, seed, n):          # exp26_candidate.Digest over exp45.batch_tensor (int64 x, y; bool mask), without torch
    h = hashlib.sha256(); bt = batches(f, seed)
    for _ in range(n):
        items = next(bt); rows = [p + t for p, t in items]; L = max(map(len, rows)) + 1
        data = [[BOS] + r + [BOS] * (L - 1 - len(r)) for r in rows]
        x = b"".join(array("q", r[:-1]).tobytes() for r in data); y = b"".join(array("q", r[1:]).tobytes() for r in data)
        mask = b"".join(bytes(1 if len(p) <= j < len(p) + len(t) else 0 for j in range(L - 1)) for p, t in items)
        for t in (x, y, mask):
            h.update(str((len(rows), L - 1)).encode()); h.update(t)
    return h.hexdigest()

if part in ("all", "length"):
    for seed in SEEDS:
        lp = Random(seed + 3).sample(split(DEV[0], .5, seed)[1], 300)                      # exp42_cells.py line 230 with exp11_cell.panel
        for f in (.25, .10):
            taught = set(split(DEV[0], f, seed)[0])
            a_add = Random(seed + 3).sample(split(DEV[0], f, seed)[1], 300)                # the add panel behind A at f
            print(f"length panel, seed {seed}, f {f}: {sum(p in taught for p in lp)} of 300 pairs are taught; it shares {len(set(lp) & set(a_add))} pairs with the add panel behind A")
    t25, t50 = set(split(DEV[0], .25, 73129)[0]), set(split(DEV[0], .5, 73129)[0])
    print(f"seed 73129, addition: {len(t25)} pairs taught at a quarter, of which {len(t25 & t50)} are taught at a half")
    for rule in "SB":
        r = e41(rule, 512, 10, 73129)["curve"][-1]
        print(f"E41 dense, present width, f 0.10, {rule}, last row: addition right on {r['taught_counts']['add89']} of 300 taught and {r['A_counts']['add89']} of 300 untaught pairs at 12 letters")
if part in ("all", "stream"):
    for fp, f in ((25, .25), (10, .10)):
        want = next(r["data_digest"]["sha256"] for r in e41("S", 512, fp, 73129)["curve"] if r["step"] == 4096)
        print(f"stream digest at 4,096, f {f}: {digest(f, 73129, 4096)[:16]}; E41's receipt {want[:16]}")

def to80(c, key):
    v = [r[key] for r in c]
    return next((c[i]["step"] for i in range(len(v) - 2) if min(v[i:i + 3]) >= 80), None)
def m3(c, key, at):
    rows = [r[key] for r in c if at - 512 <= r["step"] <= at]; assert len(rows) == 3
    return sum(rows) / 3
if part in ("all", "e42"):
    for cfg in ("G", "F10", "P8"):
        for rule in "SB":
            cs = [e42(cfg, rule, s) for s in SEEDS]
            print(cfg, rule, "to 80 on A", [to80(r["curve"], "A") for r in cs], "on R", [to80(r["curve"], "R") for r in cs],
                  "A at end", [round(m3(r["curve"], "A", 24576), 1) for r in cs], "R at end", [round(m3(r["curve"], "R", 24576), 1) for r in cs],
                  "rise of R over the last 8,192", [round(m3(r["curve"], "R", 24576) - m3(r["curve"], "R", 16384), 1) for r in cs],
                  "theta", cs[0]["theta"], "CPU-h", [round((r["cpu_updates"] + r["cpu_scoring"] + r["cpu_replacement"]) / 3600, 2) for r in cs])
if part in ("all", "e41"):
    for H, fp in ((512, 25), (512, 10), (256, 25), (256, 10)):
        for rule in "SB":
            cs = [e41(rule, H, fp, s) for s in SEEDS]; d = cs[0]["dose"]; R = [m3(r["curve"], "R", d) for r in cs]
            print(f"E41 H={H} f=0.{fp} {rule} at {d}: A", [round(m3(r["curve"], "A", d), 1) for r in cs], "R", [round(x, 1) for x in R],
                  "range of R between seeds", round(max(R) - min(R), 1), "taught", [round(m3(r["curve"], "taught", d), 1) for r in cs],
                  "rise of R over the last 2,048", [round(m3(r["curve"], "R", d) - m3(r["curve"], "R", d - 2048), 1) for r in cs],
                  "over the last 8,192", [round(m3(r["curve"], "R", d) - m3(r["curve"], "R", d - 8192), 1) for r in cs],
                  "to 80 on A", [to80(r["curve"], "A") for r in cs], "on R", [to80(r["curve"], "R") for r in cs],
                  *((f"R at 12,288 and 24,576", [(round(m3(r["curve"], "R", 12288), 1), round(m3(r["curve"], "R", 24576), 1)) for r in cs]) if d > 24576 else ()))
if part in ("all", "same"):
    for rule in "SB":
        for s in SEEDS:
            a = {r["step"]: r for r in e41(rule, 512, 50, s)["curve"]}; b = {r["step"]: r for r in e42("G", rule, s)["curve"]}
            keys = ("loss", "A_counts", "R_counts", "taught_counts", "data_digest")
            print(f"G of E42 against E41 H=512 f=0.50, {rule} {s}: rows equal in loss, counts and digest:", sum(all(a[u][k] == b[u][k] for k in keys) for u in a), "of", len(a))
if part in ("all", "spread"):
    for cfg in ("F10", "P8"):
        for rule in "SB":
            g = [e42("G", rule, s)["curve"] for s in SEEDS]; c = [e42(cfg, rule, s)["curve"] for s in SEEDS]
            print(f"E42 f=0.50, level of R of {cfg} minus G under {rule}, by seed:",
                  {u: [round(m3(a, "R", u) - m3(b, "R", u), 1) for a, b in zip(c, g)] for u in (4096, 6144, 8192, 24576)})
    mr = lambda c, at: m3(c, "R", round(at / 256) * 256)
    for s in SEEDS:
        c = e41("B", 256, 10, s)["curve"]
        print(f"dense, half width, f=0.10, B, seed {s}: points of R by which a learner k times slower would be behind at 40,960:",
              {k: round(mr(c, 40960) - mr(c, 40960 / k), 1) for k in (1.5, 2.0)})
    def scatter(c):               # sd of the differences between successive rows over the last 2,048 updates, over root 2
        r = [x["R"] for x in c[-9:]]
        return round(statistics.pstdev([b - a for a, b in zip(r, r[1:])]) / 2 ** .5, 1)
    for H, fp in ((512, 10), (256, 10), (512, 25)):
        print(f"row-to-row scatter of R, dense, H={H} f=0.{fp}:", {rule: [scatter(e41(rule, H, fp, s)["curve"]) for s in SEEDS] for rule in "SB"})
    print("E41 S H=512 f=0.25 seed 73129, R of the last four rows:", [r["R"] for r in e41("S", 512, 25, 73129)["curve"][-4:]])
    for rule in "SB":
        r = e41(rule, 256, 10, 73129)
        print(f"dense, half width, {rule}: CPU-seconds an update {r['cpu_updates'] / r['dose']:.4f}, a scored row {r['cpu_scoring'] / len(r['curve']):.2f}")
    r = e42("P8", "S", 73129)["curve"]
    print("P8 S 73129, share of the 2,560 connections replaced an event: first 16 events", round(sum(x["replaced_since_previous_row"] for x in r[1:17]) / 16 / 2560, 3),
          "last 16", round(sum(x["replaced_since_previous_row"] for x in r[-16:]) / 16 / 2560, 3))
    print("length_counts['12'] == A_counts['add89'] at every row, E42 receipts:",
          sum(all(x["length_counts"]["12"] == x["A_counts"]["add89"] for x in e42(c, ru, s)["curve"]) for c in ("G", "F10", "P8") for ru in "SB" for s in SEEDS), "of 18")
```
