# Check of the brief of E54, revision 2 (17 November 2026)

Written for the outside reviewer by an agent that has not seen the reviewer's reasoning. The brief is `TASK_E54_THE_LIMITED_AND_THE_PERIODIC_NETWORK_WITH_FEWER_PAIRS_20261117.md` beside this file, at commit a1da70da (its SHA-256 begins `05a28b97`; it was unchanged when this check ended). The work was read-only. Nothing was trained, torch is not installed, no code of another branch was executed, and the files of other branches were read with `git show`. The check began at 22:01 UTC, was interrupted by a restart of the container at about 22:09, and ended at 22:32 UTC. It is the second check that the brief's status line names. It is not the independent audit that the programme's rules ask for.

**Shell names.** `D=origin/codex/minifly-consolidated-20261023` (at 7cabb13a); `W=origin/claude/gallant-dijkstra-i9zgsr` (at 44f91edc); `M=MOD97_LOCAL_RULE_20261106`. "The brief" is revision 2; "the first check" is `CHECK_OF_TASK_E54_BRIEF_20261117.md`. A line of the brief is cited as "brief, line n". `APP <part>` is the script of the appendix, run from the repository root as `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I <the script> <part>`; it uses the standard library and `git show` only. `APP1 <part>` is the first check's script (`check_e54_brief.py`).

**Terms.** As in the brief. A *level* is the mean of a score over the three scored rows that end at the named update. The *nine-row rise* at an update is the mean of R over the nine scored rows that end at that update less the mean over the nine scored rows that end 8,192 updates earlier; the brief calls a curve *flattened* if this rise, taken at the last update, is below 3 points. Seeds are given in the order 73129, 93683, 116089.

## Verdict

Release after the listed changes; do not release as it stands. The eight findings of the first check are entered. Every hash and path that the brief cites is right, and so is every figure that could be recomputed, apart from the order of two rates and a rounding (defect 8). The third taught fraction can be produced by the code named, and its digest is what the stream gives. The defects are of two kinds.

- Defects 1 to 3 change what the job would conclude. The rule "has flattened" passes curves that are still rising: applied to the curves on record it is wrong in 6 of the 30 cases in which it can be tested, by 5 to 12 points of R, and it is wrong on the dense network at a tenth under back-propagation, which is the case it was made for. Licence (i) for "needs fewer samples" has no margin and restores the licence at a quarter that the first check removed. The second reading, the sweep against back-propagation, has no margin at a tenth and no assignment of its words.
- Defects 4 and 5 stop the job or lose its runs. A copy of `exp42_ckpt.py` with the listed changes "and no others" fails at its first start on the worker's branch, and several lines tied to E42 are not in the list. Detached runs die when the worker's session is idle, part 4 refuses to start again after an interruption, and the weights are excluded from commits by `.gitignore`.
- Defects 6 to 10 are the checks, cases that the words do not cover, statements about the record, smaller points and language.

If the texts given below are applied, I see no need for a third check. The reviewer should read the new list of changes of the runner (defect 4) once against `exp42_ckpt.py`, since that list is now long.

## Defects, the most serious first

### 1. The rule "has flattened" passes curves that are still rising

**What is wrong.** The brief (line 32) calls a curve of R flattened if its nine-row rise at the last update is below 3 points. Every decided reading rests on this: *worse* needs the network's own curve to have flattened, *better* and the licence need `G`'s, *as well* needs both.

(a) *Applied to the record, the rule is wrong where it can be tested.* E41's twelve runs at the half width have 40,960 updates, so the rule can be applied before their end and compared with what the level did afterwards. At the five updates 20,480 to 36,864 of the twelve curves the rule says "flattened" 30 times of 60. In 6 of the 30 the level then rose by 5 points or more before the run ended (`APP rule2`):

| Run (E41, half width) | Rule says flattened at | Level then | Level at 40,960 |
|---|---|---|---|
| f = 0.10, back-propagation, 73129 | 24,576 (rise 2.1) | 32.9 | 44.0 |
| f = 0.10, back-propagation, 93683 | 28,672 (rise 2.9) | 50.2 | 61.7 |
| f = 0.25, sweep, 73129 | 32,768 (rise 2.5) | 76.6 | 82.8 |
| f = 0.25, back-propagation, 93683 | 20,480 (rise 2.4) | 86.1 | 96.2 |
| f = 0.25, back-propagation, 93683 | 24,576 (rise 2.4) | 90.8 | 96.2 |
| f = 0.25, back-propagation, 116089 | 20,480 (rise 2.7) | 84.5 | 91.8 |

(b) *It is wrong on the case it was made for.* The first check's finding 1 (a) was that the dense network at a tenth under back-propagation has not finished at any length within reach. Its three curves at the half width rise in steps and not evenly. On seed 73129 the nine-row rise at updates 16,384 to 40,960 is 6.8, 4.6, 2.1, 6.2, 4.3, 3.6 and 5.2 (`APP flat`). At the end of the run, update 40,960, the rule says "flattened" for seed 93683 (rise 2.7) at a level of 61.7, although the level itself rose by 5.9 points over the same 8,192 updates; the brief's own record section (line 44) cites that curve as still rising.

(c) *A learner that is only slower is called flattened.* A learner that follows the dense network's own curve 1.5 or 2 times more slowly has, by construction, the dense network's end result. Read at update 49,152 on E41's curves at the half width at a tenth under back-propagation, its nine-row rise is 0.1 and 1.1 points on seed 73129 and 2.4 and 1.9 on seed 116089 (`APP slow`). The rule calls all four flattened. (On seed 93683 the rise is 6.9 and 5.4, and the rule does not.) With a difference of the margin, the word is then *worse*. This is the error of finding 1.

(d) *Why.* The limit of 3 points over 8,192 updates allows 18 points over a run of 49,152 updates, which is more than the margin of 10 and about the margin of 20. One span cannot tell a plateau between two steps from the end of a curve. The bracket in line 32 (a scatter of 0.9 to 1.9 points for a difference of two nine-row means) is right as arithmetic and treats the rows as independent around a smooth curve; the curves on record are not smooth. Nine rows in place of three did not remove the weakness: the three-row form of the first check's repair text says "flattened" at 30 of the same 60 points and is wrong in 8 (`APP rule3`).

(e) *A rule with two spans is right on the same record.* If the rise must be under 3 points in size over the last 8,192 updates and also over the 8,192 updates before them, the rule says "flattened" at 15 of the 60 points, and in none of the 15 did the level rise by 5 points afterwards (`APP rule2`). At the end of E41's half-width runs it gives: a tenth under the sweep flattened on all three seeds; a tenth under back-propagation on none; a quarter under the sweep on seed 93683 only; a quarter under back-propagation on seed 116089 only. It will leave more readings *not decided* than the brief's rule. The record offers no cheaper rule that is right on its own curves.

(f) *A falling curve.* By the wording "less than 3 points above", a curve that falls counts as flattened, whatever the fall. No curve on record falls by 3 points over 8,192 updates (`APP rule3`), but in `P8` the share of connections replaced at an event grows through the run (9.3%, 15.8%, 18.3% and 19.6% of the 2,560 connections in the four quarters of E42's run of seed 73129 under the sweep), and E54's runs are twice as long.

**Repair.** In "Terms", the last bullet (brief, line 32) is replaced by:

> - Two *rises* are given for every curve of R. The *last rise* is the mean of R over the nine scored rows that end at the last update of the run, less the mean over the nine scored rows that end 8,192 updates earlier. The *rise before it* is that second mean less the mean over the nine scored rows that end 16,384 updates before the last update. A curve of R *has flattened* if both rises are smaller than 3 points in size. A curve with a rise of minus 3 points or less has not flattened, and the report says that it fell. (One span is not enough. Applied before the end of E41's twelve runs at the half width, at updates 20,480 to 36,864, a test on the last rise alone says "flattened" 30 times of 60, and in 6 of the 30 the level then rose by 5.4 to 11.5 points before the run ended; the test on both rises says it 15 times, and in none of them did the level rise by 5 points. The scatter of R from row to row is 2.0 to 4.0 points at a quarter of the pairs in E41, so a difference of two means of nine rows has a scatter of 0.9 to 1.9 points.)

In "How the result is stated", in the bullet **Levels**, "the rise of R by the nine-row rule" becomes "the two rises of R". In the table of changes, the cell of row A2 that begins "With the rule of finding 1" becomes:

> Addition. With the rule of finding 1, a run of 24,576 updates leaves most readings *not decided*: at a half, none of E42's 18 curves has flattened at update 24,576 by the rule of two rises, since the 8,192 updates before the last 8,192 still hold part of the main rise; by the last rise alone 6 of the 18 had not flattened, and 6 of the 12 comparisons with `G` were not decided

### 2. Licence (i) has no margin, restores the licence at a quarter that the first check removed, and "flattened below 80" is not defined

**What is wrong.** The brief (line 98) licenses "needs fewer samples" (i) "by a limited or periodic network that generalises from a fraction at which `G`'s curve of R has flattened below 80".

(a) *No margin.* Every other decided word needs a difference of 10 or 20 points, chosen against the spread between seeds (finding 2). Licence (i) needs none. A network with three rows at 80 and a `G` that has flattened at a level of 79 earn the main statement of the job on a difference of one or two points. At f = 0.15, where the brief sets the margin at 20 because no spread is known, licence (i) is earned by a network at 81 against a `G` flattened at 70, which the comparison of end results in the same report calls *as well*.

(b) *The threshold lies inside the scatter of a flattened curve.* The first check, finding 1 (e), removed the licence at a quarter for this reason, and its repair text limited licence (i) to a tenth. The brief extends it to every fraction. At a quarter under the sweep E41's dense network at the half width, after it had reached 80 (three rows in a row), fell below 80 again in 9 of 21, 5 of 65 and 11 of 42 of its later scored rows, with single rows down to 66.2, 75.5 and 77.0 (`APP record`). At the present width the last twelve rows of seed 73129 lie between 75.7 and 83.9, and four of them are at or above 80.

(c) *Two definitions meet.* "Generalises from the fraction" is "R reaches 80 within the run" (line 91), an event that can lie early in the run. "Flattened below 80" is not defined; read as "has flattened, and its level at update 49,152 is below 80", it holds for a `G` that reached 80 at update 20,000 and stands at 79.6 at the end. `G` then generalises from the fraction and the network is licensed for generalising where `G` "has flattened below 80".

**Repair.** The paragraph that begins "'Needs fewer samples' is licensed, for one seed, only" (brief, line 98) is replaced by:

> "Needs fewer samples" is licensed, for one seed, only by *generalises better* at some fraction, that is, by a level of R at update 49,152 above `G`'s by the margin or more while `G`'s curve has flattened. For each such case the report adds whether the network generalises from the fraction and whether `G` does. Reaching 80 where `G` does not licenses nothing by itself, because the threshold can lie inside the scatter of a flattened curve (at a quarter under the sweep E41's dense network at the half width fell below 80 again in 9 of 21, 5 of 65 and 11 of 42 scored rows after it had reached 80). For a network that reaches 80 where `G` does not, and that does not generalise better, the report says one of two things. If `G`'s curve has flattened: "reaches 80 on R at this fraction; the dense network has flattened at [its level]; the difference of [d] points is inside the margin of [m]". If `G`'s curve has not flattened: "reaches 80 on R at this fraction within the run, and the dense network does not; whether the dense network would reach 80 with more updates is not decided". A ratio of updates to 80 that is smaller at a smaller fraction than at a half is reported as "is slowed less by the smaller fraction" and licenses nothing further.

P54.8 then reads: "At no fraction and under neither rule does `F10` or `P8` generalise better than `G`", with the same chance.

### 3. The second reading, the sweep against back-propagation, cannot be applied as written, and the sentence of the record on which it rests says more than the brief's rule allows

**What is wrong.**

(a) The bullet (brief, line 95) says "with the same margins and the same condition of flattening". The margins (line 92) are given by rule: at f = 0.10 they are 10 points under the sweep and 20 under back-propagation. A comparison between the two rules at f = 0.10 has no margin. Of the nine pairs of this reading, three cannot be read.

(b) The words *better*, *worse* and *as well* are defined for a network against `G`. Which rule takes the place of the network, and whose curve must have flattened, is not written.

(c) "On the record the sweep therefore needs more taught pairs than back-propagation and not fewer" (line 45). The figures of the bullet are right (see "Verified"). The conclusion is drawn from levels at fixed updates, which finding 1 forbids. By the brief's own definitions no fraction on record is enough for back-propagation and not for the sweep: at a quarter at the half width both reach 80, at the present width the sweep's run ends in its rise, and at a tenth neither reaches 80. By the rule of defect 1 the difference is one of end results in 4 of the 12 pairs on record (two widths, two fractions, three seeds): at a tenth at the half width on all three seeds (the sweep has flattened near 10; back-propagation is 33.5, 51.7 and 42.7 points above), and at a quarter at the half width on seed 93683 (11.9 points). The other eight are not decided (`APP record`, `APP rule2`).

**Repair.** The bullet **The sweep against back-propagation** (line 95) is replaced by:

> - **The sweep against back-propagation** within each configuration and fraction, as a second reading of the same eighteen runs (nine pairs). The difference is the sweep's level of R at update 49,152 less back-propagation's. The margin is 10 points at f = 0.25 and 20 points at f = 0.15 and at f = 0.10. The sweep *generalises better* than back-propagation if the difference is the margin or more and back-propagation's curve has flattened; *worse* if the difference is minus the margin or less and the sweep's own curve has flattened; the two are *not distinguished* if the difference is smaller than the margin in size and both curves have flattened. In every other case the reading is *not decided within the run*. The report gives both levels and the two rises of each curve.

In "What the record says", the last two sentences of the bullet **The two rules with fewer pairs** (line 45) are replaced by:

> By the rule fixed below, the difference is one of end results in 4 of the 12 pairs on record: at a tenth at the half width, on all three seeds, the sweep's curve has flattened near 10 and back-propagation's level is 33 to 52 points above it, although neither reaches 80; at a quarter at the half width the sweep is 11.9 points below on seed 93683, with its curve flattened. The other eight pairs are not decided. In no pair is the sweep above back-propagation, so nothing on record suggests that the sweep needs fewer taught pairs. What is open is whether a limited or a periodic network changes this.

### 4. The list of changes to `exp42_ckpt.py` is not complete; taken at its word, the copy stops before its first update

**What is wrong.** The brief (line 65) makes the runner "a copy of `$W:$M/exp42_ckpt.py` with the following changes and no others". I read `exp42_ckpt.py` (234 lines) in full. The worker's branch starts from the default branch (`LOG.md` beside this file, section 3: "Each session is a new one, started from the default branch"), and the brief forbids executing `exp42_ckpt.py` from `$W`. Line numbers below are lines of `$W:$M/exp42_ckpt.py`.

(a) *Line 68 stops every run.* `me = dict(wrapper="exp42_ckpt.py", wrapper_sha256=R.X16.sha_file(HERE / "exp42_ckpt.py"), ...)` is executed at every start. `sha_file` reads the file (`exp16_integrated.py`, line 263). `exp42_ckpt.py` is not on `$D`, so on the worker's branch the read raises an error. If the worker saves a copy of `exp42_ckpt.py` beside the new file to make the difference, the run starts and records the hash of a file that did not run. Change 4 speaks of `provenance` only, which is another place (line 79).

(b) *Line 44*: `dose = dose or R.DOSE`. `R.DOSE` is 24,576. A run started without `--dose` has E42's length, and by change 4 it is then named a dry run.

(c) *Lines 48 and 51*: the receipt's path is `R.receipt_path(...)`, which writes `e42_` and compares the length with 24,576 (`exp42_cells.py`, lines 303 to 305). A new function is needed; change 4 gives the name and not this. The file of saved state is `<stem of the receipt>.ckpt.pt`. With the new name the stem carries the fraction, so that a saved state of one fraction is not found by a run of another; the brief should say that the stem is kept, since this is what separates the fractions.

(d) *Line 96*: on resuming, the assertion compares configuration, rule, seed and length, and line 97 the starting structure. None of these differs between fractions. Only the replayed digest (line 101) would stop a run that had found the saved state of another fraction. The fraction belongs in line 96.

(e) *Lines 99 and 100, and change 3.* On resuming, the stream is replayed to the saved update and each batch goes to the digest only. Change 3 says "every batch passes E41's assertion". Whether the replayed batches are asserted is left open. In substance nothing is wrong either way: the replayed batches were asserted by the process that presented them, and line 101 shows that the replay equals them. The assertion costs nothing in the replay and makes the sentence true of every process.

(f) *Change 5 has no place.* The receipt held in memory (`rec`) is part of the saved state (line 127). If the hash at update 12,288 is written into `rec` after `checkpoint(u)` (line 148) and the run is stopped before update 13,312, the resumed run starts from the state saved at 12,288, never passes that update again, and the hash is lost. Check 3 needs that hash. It must be written after `score_row(u)` (line 144) and before the receipt and the state are written (lines 146 and 148).

(g) *The self-test* (lines 179 to 214, and lines 223 to 224 of `main`) names E42's folder `exp45/e42_qual`, four of E42's dry-run receipts, and a shell script `exp45/e42/selftest_case.sh`. None of these is on the worker's branch. Under "no others" it stays in the file as code that cannot run. Check 6 replaces it and the list does not say so.

(h) *Change 6 contradicts the copied text.* "Nothing inside `exp42_cells`, `e42_cells` or `exp11_cell` is replaced while a run executes." Line 59, `R.install(layer)`, replaces `exp11_cell.forward` by a function that runs this run's layer (`exp42_cells.py`, lines 190 to 197). The first check's repair text had the same sentence; its purpose was to exclude a replacement of `E.split` or of `panels` from outside.

(i) *How f reaches `run_ckpt`* is not said. `main` (lines 217 to 230) has no argument for it.

(j) *Smaller ties to E42*: the printed lines begin "e42" (lines 104, 120, 150, 175), and line 171 reads the environment variable `E42_BESIDE`.

Sound as they stand (see "Verified"): the option `--stop-after` (lines 147 to 151 and 228), which check 6 needs; the names of the other output files, which follow the receipt's stem (lines 49 and 50); the replay of the stream; `Random(f"e42|replace|{seed}")` (line 66), which must stay for check 1.

**Repair.** In "The runner", the sentence "The new file is a copy of ... the difference between the two files." and the numbered list (brief, lines 65 to 72) are replaced by:

> The new file is a copy of `$W:$M/exp42_ckpt.py` with the following changes and no others, and the report shows the difference between the two files. Line numbers in this list are lines of `exp42_ckpt.py`.
>
> 1. **The fraction.** `run_ckpt` takes the taught fraction f. `main` (lines 226 to 230) reads it from a required option `--fraction <50|25|15|10>` and passes that number divided by 100. The stream (line 64) is `R.E.batches([(t, R.E.split(t, f, seed)[0]) for t in R.E.DEV], seed, R.E.KMAX["M"])`, which is E41's stream for that fraction (`$D:$M/exp41_scale.py` lines 91 to 93), and the `data` string (line 76) says so.
> 2. **The panels** (line 63). The panels of A, of R and of taught pairs are the first three that `R.panels(seed, f)` returns; they are E41's panels at that fraction. Its fourth, the length panel, is not used: line 230 of `exp42_cells.py` draws it from the pairs held out at a half, of which 68 of 300 are taught at a quarter and 29 at a tenth on this seed. The length panel is `R.E.panel(R.LEN_TASK, R.E.split(R.LEN_TASK, f, seed)[1], seed, ks=(12, 40), n=300)`. At every start, before the first update or the replay, the runner asserts three things: no pair of the panels of A, of R and of the length test is among the taught pairs of its operation at this fraction; every pair of the taught panel is; and the items of the length panel at 12 letters are the items of the addition panel behind A.
> 3. **Taught pairs only.** `item_key`, `taught_keys` and `assert_taught` (`$D:$M/exp41_scale.py` lines 126 to 138) are imported from `exp41_scale` after `exp42_cells`, or copied as text. None of them uses the table `FRACS`, which has no 0.15. `assert_taught` is applied to every batch in the loop of updates (before line 136) and to every batch of the replay on resuming (line 100).
> 4. **Length and names.** The default number of updates (line 44) is 49,152 in place of `R.DOSE`. A function of the new file takes the place of `R.receipt_path` (line 48): the receipt is `e54_<config>_<rule>_f<50|25|15|10>_<seed>.json` under `$M/exp45/e54/`, with `_dry<updates>` before the seed for any number of updates other than 49,152. The file of saved state keeps the receipt's stem as its name (line 51), so that it carries the fraction, under `$M/exp45/e54/ckpt/` (line 31); it is not committed. The receipt has `experiment="E54"` and `fraction=f` (line 70), and the assertion on resuming (line 96) compares `fraction` as well.
> 5. **Provenance.** Line 68 records `wrapper="exp54_fraction.py"` and the hash of that file; as it stands the line reads `exp42_ckpt.py`, which is not in the worker's tree, and stops the run. `provenance` (line 79) adds the hash of `exp54_fraction.py`, and of `exp41_scale.py` if it is imported.
> 6. **Hashes on the way.** At updates 12,288 and 24,576, `R.R_.tensors_sha(m.state_dict())` is written into the receipt as `tensors_sha256_at["12288"]` and `tensors_sha256_at["24576"]`, directly after `score_row(u)` (line 144) and before the receipt is written (line 146) and the state is saved (line 148), so that a resumed run still holds it.
> 7. **The self-test** (lines 179 to 214, and lines 223 and 224) is left out: it names E42's folders and a shell script that are not in the worker's tree. Check 6 takes its place. The option `--stop-after` (lines 147 to 151 and 228) stays.
> 8. **Names in the output.** The printed lines (104, 120, 150 and 175) say `e54`, and line 171 reads `E54_BESIDE`.
> 9. Apart from `R.install(layer)` (line 59), which is E42's own replacement of `exp11_cell.forward`, nothing inside `exp42_cells`, `e42_cells`, `exp11_cell` or `exp41_scale` is replaced while a run executes.

### 5. As the rules for the worker stand, runs will be lost

**What is wrong.**

(a) *Detached processes die when the session is idle.* The brief (line 129) has runs "launched detached". `LOG.md` beside this file, 22:11 of 17 November, eleven minutes after the brief was written: the machine of the long lives' worker "was stopped at 21:15, about five minutes after its turn ended ... The three lives died"; "On this host detached processes do not outlive an idle session; the worker now holds its turn open with a waiting command". The four workers then running were warned by message. The brief does not say this, and a worker that follows it loses its runs unless the starting message adds the warning. The job needs at least 13 hours of wall time at three processes.

(b) *Part 4 refuses to start again.* "A run of part 4 is started again from its beginning" (line 130). `exp41_scale.run` refuses to start if its receipt or its weights exist (`$D:$M/exp41_scale.py` line 153, with `exp20_reference.py` lines 125 to 127), and it writes the receipt every 4,096 updates (lines 193 and 194). After an interruption the same command fails on the partial receipt. E42's worker moved such files to `exp45/e42/interrupted/` with the time appended to the name.

(c) *Part 4 needs 1.57 and 1.69 CPU-hours without interruption* (`APP cost`). The reviewer's container was restarted at about 21:50 and at about 22:09 on 17 November (`LOG.md`), and this check's at about 22:09; the files in E42's folder of interrupted runs are stamped at three times within 36 minutes on 16 November (20:37, 20:46, 21:12). The brief gives no limit to the number of fresh starts.

(d) *The weights are not committed.* `.gitignore` on `$D` has the line `*.pt`. "Commit and push receipts and logs" (line 130) therefore leaves every weights file out, and "Done" (line 142) does not list them. Revision 1 said "final weights beside them"; revision 2 dropped the clause. Rule 4 of `PLAN.md` has every result "verified from receipts and weights before it is read". E42's worker committed its weights by force: 90 files, 325 MB for 45 runs, about 7 MB a run (`APP paths`).

(e) *Relative paths.* `--out exp45/e54` and `--qual exp45/e54/qual` (lines 59 and 80) are relative to the current folder, and `exp41_scale.py` and `exp42_cells.py` take them as given. Run from the repository root, they write outside `$M`.

**Repair.** In "Rules for the worker", the second bullet (line 130) is replaced by:

> - The container can restart without warning, and the host stops a session that is idle together with every process that it started, detached ones included (`LOG.md` beside this file, 17 November, 22:11: three runs of another worker died in this way about five minutes after its turn ended). While runs are in progress, keep the turn open with a waiting command, and read receipts between waits. After a stop or a restart, a run of parts 1 to 3 is resumed by its own command from its saved state (check 6). A run of part 4 cannot resume: move its partial receipt to `$M/exp45/e54/interrupted/` with the time appended to its name, because `exp41_scale.run` refuses to start while the receipt exists, and start the run again from its beginning. If a run of part 4 is interrupted a second time, stop part 4 and report; parts 1 to 3 do not depend on it. Commit and push receipts and logs as runs finish, and the receipt of a running run at least every two hours (the runner writes it every 4,096 updates). The final weights of every run are committed with `git add -f`, since `.gitignore` excludes `*.pt`: in parts 1 to 3 the files `.pt` and `_recurrent.pt`, about 7 MB a run, with `_structure.json`; in part 4 the file `.pt`. The files of saved state are not committed.

In "Runs" (line 59), the sentence "Part 4 is E41's own runner, unchanged: ..." becomes:

> Part 4 is E41's own runner, unchanged, run from `$M`: `python3 exp41_scale.py run <S|B> 256 10 73129 --dose 122880 --out exp45/e54`. Its receipt is `exp45/e54/e41_<S|B>_h256_f10_dry122880_73129.json` (line 143 of `exp41_scale.py` puts `_dry122880` into the name), with `experiment="E41"` and the weights beside it.

In check 3 (line 80), "Beforehand `python3 exp42_cells.py check 1 --qual exp45/e54/qual` passes on this host" becomes "Beforehand `python3 exp42_cells.py check 1 --qual exp45/e54/qual`, run from `$M`, passes on this host". In "Done" (line 142), "the twenty runs" becomes "the twenty runs with their receipts and final weights".

### 6. Checks 4, 5 and 6, and the part of check 3 that concerns part 4

**What is wrong.**

(a) *Check 5 is read where it can hardly fail.* "In every scored row ... Read at the first scored rows of every run" (line 82). At row 0 both counts are 0 for any panel. In the next seven rows of E41's four dense runs of seed 73129 at the present width (a quarter and a tenth, both rules) the count on the addition panel is 0 to 27 of 300, mostly below 10, so that the counts of two different panels can agree by chance, and how many rows "the first scored rows" are is not said. What the check is named after, that the length panel at 12 letters is the addition panel behind A, can be shown exactly before the first update by comparing the two lists of items. The identity itself is right: `E.panel` seeds its sample with `seed + 3` and renders the first number of letters first (`exp11_cell.py` lines 262 to 265), so the two panels at 12 letters are the same items when both are drawn from the pairs held out at f.

(b) *Check 4 cannot fail.* It has no limit (line 81).

(c) *Check 6.* The two runs have the same name, so the second is refused unless each has its own folder (`refuse_existing`, lines 54 and 56 of `exp42_ckpt.py`). "Equal ... in every row" cannot hold for the CPU and wall times that every row carries; E42's self-test leaves them out (line 184).

(d) *Check 3, part 4* (last sentence of line 80). When the rows are read and what a difference stops is not said. The comparison itself is well defined: E41's receipt has 161 rows to update 40,960 with the loss, the three sets of counts and the batch digest (`APP paths`). No hash can be compared: E41's receipt has its hash at update 40,960, and `exp41_scale.run` records one only at the end of a run.

Checks 1, 2 and 3 (parts 1 and 3) can fail for what they are named after, and say when they are carried out and what stops; their hashes and paths are right (see "Verified").

**Repair.** Checks 4, 5 and 6 are replaced by:

> 4. **Cost.** From the dry runs of checks 1, 3 and 6 the report gives the CPU-seconds an update and a scored row for `G`, `F10` and `P8`, and a forecast for each part. If the forecast for the whole job is above 51 CPU-hours, which is its cost at the rates of the slowest receipts on record, stop and ask before part 1 is launched.
> 5. **The length panel.** At every start the runner asserts that the items of the length panel at 12 letters are the items of the addition panel behind A (change 2), and at every scored row that `length_counts["12"]` equals `A_counts["add89"]`, as it does in all eighteen receipts of `G`, `F10` and `P8` of E42. Either assertion stops the run. The receipt of every run records that the first assertion held, in a field `length_panel_is_addition_panel`.
> 6. **A resumed run equals an uninterrupted one.** `P8` under the sweep at f = 0.25, 2,048 updates, twice, each with its own folder for receipts and for saved state (`--out`, `--ckdir`): once without interruption, and once with `--stop-after 1024` and then started again by the same command without that option. The two receipts are equal in `tensors_sha256`, in the batch digest and in every field of every row except the CPU and wall times, and the second receipt names update 1,024 under `checkpointing.resumed`.

The last sentence of check 3 becomes:

> In part 4 the rows to update 40,960 equal those of `e41_<rule>_h256_f10_73129.json` in the same quantities; they are read at row 256 and at every writing of the receipt, and a difference stops part 4 only. No hash is compared there, since `exp41_scale.run` records one only at the end of a run.

### 7. Cases that the words do not cover

**What is wrong.** The task of the worker is to apply the words to twelve comparisons with `G` (two networks, two rules, three fractions) and nine between the rules. With the repairs of defects 1 to 3 the three words and *not decided* cover every combination of difference and flattening. The following still need a decision.

(a) *`describe` when one run does not reach 80.* The brief cites lines 63 to 72 of `e42_tables.py`, "with its rules for a run that does not reach 80". Those rules take the lengths of the two runs as arguments, and the arguments are formed in `thr_desc` (lines 75 to 83) as the run's updates less 512. With 49,152 updates on both sides the limit is 32,256 updates with that subtraction and 32,768 without it (`APP record`). A worker who reads lines 63 to 72 only has to choose.

(b) *Both curves flattened far below 80.* At a tenth under the sweep E41's dense network stays at R 7 to 11 with the taught pairs at 97%. If all three networks do the same, the brief's word is that the limited network "generalises as well" as the dense one, where none of them generalises. With a margin of 20 points the same word covers a network that is 19 points lower. The word should say what was found: the two were not distinguished at the margin.

(c) *P54.2*: "reaches 80 on R by update 24,576". Updates to 80 is the first of three rows. Whether the first row or the third must lie at or before update 24,576 is open, which is a difference of two rows. Revision 1 had "within its dose" of 24,576, that is, the third row.

(d) *Max and min* (line 93): "apart from add and sub at every reading". Which quantity, and at which updates, is not said. They are on the panel behind A only (300 items each).

**Repair.** The bullet **Updates to 80** (line 89) is replaced by:

> - **Updates to 80** on A and on R, with E42's words and bands (`describe` in `$W:$M/e42_tables.py`, lines 63 to 72, called as in `thr_desc`, lines 75 to 83). If both runs reach 80, a ratio to `G` below 2/3 is *fewer*, above 1.5 *more*, otherwise *about the same*. For a run that does not reach 80, the length entered is its updates less 512, here 48,640. If only `G` reaches 80, the network needs *more* when `G`'s updates to 80 are 32,256 or fewer, and the reading is *not decided* otherwise. If only the network reaches 80, it needs *fewer* when its updates to 80 are 32,256 or fewer, and the reading is *not decided* otherwise. If neither reaches 80 the reading is *neither reaches*. Beside each ratio stands the ratio of the same seed at a half, given above.

The bullet **The comparison of end results** (line 92) is replaced by:

> - **The comparison of end results.** The difference is the network's level of R at update 49,152 less `G`'s. A network *generalises better* than `G` if the difference is the margin or more and `G`'s curve has flattened; *worse* if the difference is minus the margin or less and the network's own curve has flattened; the two are *not distinguished* if the difference is smaller than the margin in size and both curves have flattened. In every other case the reading is *not decided within the run*. In all four cases the report gives both levels, the two rises of each curve and the margin. Where both curves have flattened and neither network generalises from the fraction, the report says "neither generalises from this fraction" before the word. The margin is 10 points at f = 0.25 under both rules and at f = 0.10 under the sweep, and 20 points at f = 0.10 under back-propagation and at f = 0.15 under both rules (at f = 0.15 no run exists from which a spread could be taken).

In "The question" (line 36), "generalise better than the dense one, as well, or worse" may stay. P54.2 becomes: "At f = 0.25 under the sweep, the updates to 80 on R of `P8` are 24,064 or fewer, so that the three rows end by update 24,576. (The threshold lies inside the scatter of the dense network's own late level at this fraction, so the outcome says little by itself)". The bullet **Max and min** (line 93) becomes:

> - **Max and min** beside add and sub: the share right on the untaught pairs of max and of min at 12 letters (`A_counts["max89"]` and `A_counts["min89"]`, 300 items each), as levels at updates 12,288, 24,576 and 49,152. Their answers need no reduction by the modulus.

### 8. Statements about the record and about the job

(a) *Two measures of a rise under one name.* The rises in "What the record says" are differences of levels, that is, of three-row means: "4.5 to 6.8" (line 42), "1.7, 3.3 and 6.0" (line 43), "3.5 to 6.2" (line 44). The brief's rule uses nine-row means, which give other figures for the same curves: 7.4, 8.8 and 7.1 for `F10` under the sweep at a half; 5.2, 2.7 and 3.8 for the half width at a tenth under back-propagation (`APP flat`). After "with the script in the appendix of the check" (line 40) add:

> The rises quoted in this section are differences of levels, which are means of three rows. The rule of flattening uses means of nine rows, which give other figures for the same curves: for the half width at a tenth under back-propagation 5.2, 2.7 and 3.8 points where the levels give 6.2, 5.9 and 3.5.

(b) *"It cannot refute a lowering that lies between two of the fractions tested"* (line 117). The first check's text was "The job cannot refute it". The brief's sentence implies that a lowering to a tested fraction can be refuted. With one seed, margins of 10 and 20 points and readings left not decided, a missing licence is not a refutation. The two sentences "The job can find ... tested." become:

> The job can find a lowering of the needed fraction as far as 0.15 or as far as a tenth. It cannot refute one: a lowering that lies between two of the fractions tested is not visible, and at a tested fraction a missing licence on one seed, with readings left not decided, is not a refutation.

(c) *The order of two rates* (line 61). The brief gives its pairs of figures in the order sweep, back-propagation. E41's receipts of seed 73129 at the half width have 0.0435 CPU-seconds an update under the sweep and 0.0399 under back-propagation (`APP cost`). "(0.040 and 0.044 CPU-seconds an update" becomes "(0.044 and 0.040 CPU-seconds an update under the sweep and under back-propagation".

(d) *Rounding.* From the receipts unrounded, each of parts 1 to 3 costs 11.54 CPU-hours and the job 38.5; the brief's 11.6 and 38.6 come from adding the six rounded figures. No change is needed.

### 9. Smaller points

- **Shell names** (line 24). `V` and `R` are defined and not used, and `R` is also the name of a score. The memo of line 123 is `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md`, section 22.12. Delete `R` and write that path in line 123. Keep `V` and say what it is for: "The brief, `PLAN.md` and `LOG.md` are on `$V`." After the definition of `W` add: "The worker's branch starts from `$D`. `exp41_scale.py` and E41's receipts are on `$D` and not on `$W`; `exp42_ckpt.py`, `e42_tables.py` and E42's receipts are on `$W` and not on `$D`, and are read with `git show`."
- **"Not decided within the dose"** (table of changes, row 1) and "not decided within the run" (line 92) are one term with two names. Use the second in both places.
- **"The first family"** (line 30) is defined and not used again. The sentence becomes: "The four operations are add, sub, max and min under modulus 89."
- **The hash at update 24,576** (change 5) is used by no check and no reading of this job. Say what it is for, or leave it out.
- **`python3` without `-I`.** The first check's two commands were `python3 -I ...`; the brief has `python3 ...`. Both files put their own folder on the path themselves, and the command recorded in a receipt is the same either way. No change is needed; one form should be used throughout.
- **"A receipt for each"** (heading of the checks). Checks 2, 3 and 5 are read from the runs. Say that their receipt is a short file under `$M/exp45/e54/qual/` that names the run, the rows compared and the outcome.

### 10. Language

The brief asks the report for standard academic English, and the same holds for the brief. The following fail it: a fragment in place of a sentence, a term not defined at first use, a metaphor, or a label that a reader outside the project cannot understand.

| Brief, line | Text | Fault | Replacement |
|---|---|---|---|
| 3 | "Evidence level E2." | fragment; "E2" not defined | "The job is at evidence level E2 of the programme's scale of evidence, which is defined in [the file, to be named by the reviewer]." |
| 3 | "One seed, about 39 CPU-hours." | fragment | "The job has one seed and costs about 39 CPU-hours." |
| 9, 18, 42, 43, 59 | "dose" | label not defined | "length of the run", or in "Terms": "The *dose* of a run is its number of updates." |
| 12, 65, 78 | "dry run" | not defined | in "Terms": "A *dry run* is a run of fewer updates than the full length, made to test the runner." |
| 24 | "Host: Linux x86-64, Python 3.13, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit." | fragment | "The host is Linux on x86-64 with Python 3.13 and torch 2.15.0+cpu. At most three processes run at a time, each with one thread, and none is launched inside a wrapper that has a time limit." |
| 12, 24 | "receipt" | not defined | in "Terms": "A *receipt* is the JSON file that a run writes, with its settings, its scored rows and its hashes." |
| 28 | "replaced during learning by E42's rule, with E42's value theta = 0.5" | the rule and theta not defined | "replaced during learning by E42's rule: at every 256th update (an *event*) a cell gives up each connection that is at least 512 updates old and whose strength is below theta times the mean strength of the cell's connections, and takes a new one at random in its neighbourhood (`replace` in `e42_cells.py`, lines 173 to 195). theta is 0.5, E42's value" |
| 30 | "at 12 letters", "without letters" | "letters" not defined | before the sentence: "An item is the code of its operation, the first operand, the sign of the operation, k capital letters that carry no information, the second operand and the equals sign; 'at 12 letters' means k = 12." |
| 30 | "panel", "stream" | not defined | "A *panel* is a fixed set of test items. The *stream* is the sequence of batches of 64 taught items." |
| 31 | "Rows are scored every 256 updates." | "row" not defined | "The scores are taken every 256 updates; one such set of scores is a *scored row*." |
| 38 | "(working rule 1)" | reference not given | name the file of the working rules, or delete the bracket |
| 44 | "the untaught pairs come late under back-propagation, if at all, and the run on file ends before it can be said which" | metaphor; idiom | "under back-propagation the untaught pairs are answered correctly only late in the run, if they are at all, and the recorded run ends before this can be decided" |
| 47 | "(a hint, not fixed beforehand)" | loose | "(an observation that was not predicted beforehand)" |
| 59 | "Seed 73129." | fragment | "The seed is 73129." |
| 59 | "Part 4 may fill a free process at any time after the checks." | metaphor | "A run of part 4 may be started whenever fewer than three processes are running, once the checks have passed." |
| 61 | "Cost at the rates of E42's receipts ...: 11.6 CPU-hours for each of parts 1 to 3, ..." | no verb | "At the rates of E42's receipts of seed 73129 (...), each of parts 1 to 3 costs 11.6 CPU-hours, part 4 costs 3.3 (...) and the checks about 0.6. The job costs 38.6 CPU-hours, and up to 51 at the rates in the receipts of the other two seeds." |
| 78, 79 | "batch digest" | not defined | in "Terms": "The *batch digest* is the SHA-256 over all batches presented so far, recorded at every scored row." |
| 98, 125 | "whether the dense network would arrive"; "would arrive" | metaphor | "would reach 80 with more updates" |
| 98 | "loses less speed from fewer pairs" | loose | "is slowed less by the smaller fraction" |
| 98, 117 | "is licensed", "licenses", "If no licence is earned" | label not defined | in "Terms": "The report is *licensed* to make a statement if the condition fixed in this brief for that statement holds."; line 117: "If no licence is given" |
| 102 | "(the reviewer's, on file before any run)" | idiom | "(the reviewer's, recorded before any run)" |
| 113 | "does `F10` or `P8` earn 'needs fewer samples'" | metaphor | the text of P54.8 in defect 2 |
| 121 | "More than one seed; widths other than the present one in parts 1 to 3." | fragment | "The job has one seed and, in parts 1 to 3, one width." |
| 122 to 125 | the four further bullets of "What it cannot show" | fragments | begin each with "The job does not test" or "The job does not show" |
| 129 | "At most three processes, one thread each (...), launched detached and outside any wrapper with a time limit." | fragment | "Run at most three processes, each with one thread (...), launched detached and outside any wrapper that has a time limit." |
| 133 | "The report, `E54_REPORT.md`, in standard academic English: ..." | no verb | "The report, `E54_REPORT.md`, is written in standard academic English: ..." |
| 142 | "The six checks; the twenty runs; `E54_REPORT.md` with ..." | fragment | "The job is done when the six checks, the twenty runs and `E54_REPORT.md` are committed. The report has ..." |

## Verified and found sound

### A. The eight findings of the first check

| Finding | Entered | Departure from the repair text, and whether the purpose is kept |
|---|---|---|
| 1 | Yes | Nine rows for the test of flattening in place of three: the first check allowed it; the weakness of the test is in both forms (defect 1). One set of words for all fractions in place of *ahead* and *behind* at a tenth: kept, since every undecided case gives both levels and both rises. 49,152 updates for all runs: more than the repair asked. Licence (i) extended from a tenth to every fraction and tied to `G`'s flattening: the tie keeps the purpose; the extension loses it at a quarter (defect 2). "Cannot refute": weakened (defect 8 b). P54.3 restated with the two margins; the note on P54.2 entered in the prediction. |
| 2 | Yes | Margin of 20 at a tenth under back-propagation; the sentence on the seed and the spread moved from "What it cannot show" to "How the result is stated" and extended to f = 0.15. Kept. The second reading lacks a margin at a tenth (defect 3). The option of `G` alone on two more seeds was not taken; it was an option. |
| 3 | Yes | The runner's sentence is the repair text. Check 5 has an added sentence on when it is read, which weakens it (defect 6 a). |
| 4 | Yes | The shell name, theta = 0.5 with both places, and check 1 are the repair text, with the rows of the replaced counts and a sentence on the age of 512 updates added. Kept. |
| 5 | Yes | The repair text, with "no run of part 2 or 3" for "no run at f = 0.10", `-I` dropped, and a sentence on part 4 added. Kept. |
| 6 | Yes | The text copied is `exp42_ckpt.py` in place of `exp42_cells.run`. Each element of the repair is in the list of changes. The list is not complete for the new text (defect 4), so the purpose, that the worker decides nothing about the runner, is kept in part. |
| 7 | Yes | (a) and (b) are the repair texts. For (c) the two runs were added as part 4 with the command, the name and the comparison to update 40,960. Kept. |
| 8 | Yes | Four of the five points as asked. Of `e42_tables.py` the brief cites lines 63 to 72 and not 87 to 88; it does not use the words of lines 87 to 88, so nothing is lost there. Lines 75 to 83 are needed (defect 7 a). |

### B. The four additions

1. **The third fraction, 0.15.**
   - *The code can produce it.* `exp11_cell.split` (lines 235 to 241) rounds the fraction twice: `round(frac * 100)` in the seed of the shuffle and `round(frac * len(pairs))` for the number taught. At 0.15 these are 15 and 1,188 of 7,921. `exp42_cells.reduction_sample` (line 219) names its sample `e41|73129|15|<op>`. `taught_held` and `panels(seed, f)` (lines 212 to 229) pass f through and look nothing up. The pairs that need the reduction number 3,333 for add and 3,338 for sub among the 6,733 held out, so the samples of 1,000 and of 300 can be drawn (`e54_r2_figs.py`). The four shuffles of one run have four different seeds at every fraction. The shuffle of sub at 0.15 has the seed of the shuffle of max at 0.10, since 15 + 113 = 10 + 118; the two are in different runs and nothing follows from it (`APP split`).
   - *`exp41_scale.FRACS` has no 0.15, and nothing the brief takes depends on it.* `item_key`, `taught_keys` and `assert_taught` (lines 126 to 138) use `taught_held` and `E.split` only. `FRACS` is read by `run`, `check2`, `check5`, `check7` and `write_tables`; of these only `run` is used, in part 4, with 10. The tables `CHECK7_SAMPLES` and `E35A_DIGEST_4096` are not used.
   - *The name.* `f * 100` is 15.0 at 0.15, and `round` and `int` both give 15.
   - *The digest.* `e54_r2_figs.py`, run again, gives `a6f5362d355c59fc` at 0.15, `a71b1ca10b24a36b` at 0.25 and `e120924e5af546f9` at 0.10; the last two are those of E41's receipts. The script computes what the brief says: it loads the first check's copy of the stream (`split`, `render`, the stream of `exp11_cell.py` lines 244 to 259, and `exp26_candidate.Digest` over `exp45.batch_tensor`), which I compared with those sources line by line, and runs none of that script's parts.
   - *The margins at 0.15* (20 points under both rules) are defensible: the only spreads on record for a network in the middle of its rise are 12.7 to 20.8 points, and none exists at 0.15.
   - *The choice of 0.15* is sensible for the purpose stated. The dense network generalises at a quarter (R 74 to 91 at update 12,288) and not at a tenth (7 to 29), so the fraction at which it begins lies between them, and 0.15 is near the geometric mean of the two (0.158). One limit follows from the record: if `G` is in the middle of its rise at 0.15, as it is at a tenth under back-propagation, most readings there will be not decided.
2. **One length of run.** Nothing in the readings still takes 24,576 or 12,288 as the end. Update 12,288 remains as the end of E41's runs in check 3, 24,576 as the limit of P54.2 and as a row at which levels are given, and both as rows of the hashes. Two places outside the readings still hold 24,576: line 44 of `exp42_ckpt.py` (defect 4 b) and the subtraction in `thr_desc` (defect 7 a). `describe` behaves as the repair text of defect 7 says. The cost: see E.
3. **The definition of flattening.** `e54_r2_flat.py` is right: its nine rows are those from 2,048 updates before the named update to that update, and it counts a rise of 3 or more as not flattened. Run again, it gives 6 of 18 for E42 at a half: `G` under the sweep on seed 73129 (4.2), `F10` under the sweep on all three seeds (7.4, 8.8, 7.1) and under back-propagation on seeds 73129 and 116089 (6.8, 5.5), as the brief says. Two curves pass by 0.03 and 0.14 points (`P8` of seed 116089, 2.97 and 2.86). The defects of the rule and of its use are defects 1, 2, 3 and 7.
4. **The runner.** Beyond defect 4: a saved state of one fraction is not found by a run of another as long as the file keeps the receipt's stem; if it were found, the replayed digest (line 101) would stop the run, and lines 96 and 97 would not. The replay on resuming is cheap: E42's receipt of `F10` under back-propagation, seed 116089, resumed at update 9,216 with 7.3 CPU-seconds of replay, which is about 40 seconds at update 49,152. Four of E42's eighteen receipts of `G`, `F10` and `P8` carry a resumed entry, so resuming has worked on a worker's host (`APP paths`). `--stop-after` stops only at a saved state, that is, at a multiple of 1,024 below the length; 1,024 of 2,048 is one.
5. **The sweep against back-propagation in the record.** Level of R under the sweep and under back-propagation, and the difference (`APP record`): a quarter, present width, 80.4, 78.8, 73.9 against 83.7, 90.6, 86.0 (3.3, 11.8, 12.0); a quarter, half width, 82.8, 84.3, 82.4 against 89.8, 96.2, 91.8 (7.0, 11.9, 9.3); a tenth, present width, 6.9, 10.8, 9.0 against 14.8, 28.9, 22.2; a tenth, half width, 10.5, 10.0, 10.3 against 44.0, 61.7, 53.0. "3 to 12" and "7 to 11 against 15 to 62" are right.
6. **Part 4.** The command is valid when run from `$M`: `main` (lines 538 to 542) asserts the rule, 256 in `WIDTHS`, 10 in `FRACS` and the seed, and 122,880 is a multiple of 256. The receipt is `e41_<S|B>_h256_f10_dry122880_73129.json`. The run refuses to start only if that receipt, its weights or the temporary weights file exists (defect 5 b). The comparison with E41's receipt to update 40,960 is well defined (defect 6 d).

### C. The checks: hashes and paths

All are as the brief gives them (`APP paths`): `check8_73129_runs/e42_F10_S_dry2048_73129.json` begins `837dadc1`; `check7_73129_runs/rep0/e42_P8_S_dry1024_73129.json` begins `1b78f306`, with theta 0.5 and 0, 0, 173, 49 and 208 connections replaced at rows 0 to 1,024, and with the digests of the mask and of `made`; `theta` in `e42_P8_S_73129.json` and `chosen_theta` in `check10_73129.json` are 0.5; `ckpt_selftest.json` passes in four cases, each resumed at 1,024; the four receipts `e41_<S|B>_h512_f<25|10>_73129.json` have 49 rows to update 12,288 and the two `e41_<S|B>_h256_f10_73129.json` 161 rows to 40,960, each with the loss, the three sets of counts and the digest in every row; the digests at row 4,096 are `a71b1ca10b24a36b` and `e120924e5af546f9`; `exp42_cells.py check 1` compares with E26's two hashes (lines 414 and 415). `$W` is at 44f91edc. Lines 63, 64 and 76 of `exp42_ckpt.py` are lines 325, 326 and 336 of `exp42_cells.py`; line 143 of `exp41_scale.py` forms the name; lines 91 to 93 and 126 to 138 are as cited.

### D. The predictions

Each of P54.1 to P54.10 can be decided from the runs as specified, with two exceptions that the repairs remove: P54.2 by two rows (defect 7 c), and P54.8 through licence (i) (defect 2). None contradicts the record. P54.8 does not imply P54.3: a network above `G` by the margin while `G` has not flattened makes P54.3 false and leaves P54.8 true; the two chances (80% and 85%) are consistent. P54.1 is decided by `describe`'s word *more*; `G` reaches 80 on A at updates 8,704 and 4,352 at a quarter in E41, far below the limit of 32,256.

### E. Figures of "What the record says" and of "Runs"

- **The ratios on seed 73129** (`APP record`): on A, `F10` 2.364 and 1.700, `P8` 1.364 and 1.250; on R, `F10` 2.269 and 1.480, `P8` 1.269 and 1.120.
- **E42 at the end**, the spreads between seeds, "13 to 21", "seed 73129 has the lowest dense level", the last four rows of E41's sweep at a quarter, and the levels of E41: as the first check's script gives them (`APP1 all`, run again).
- **The scatter**: a row scatter of 2.0 and 4.0 points gives 0.94 and 1.89 for a difference of two nine-row means (`APP record`).
- **The cost** (`APP cost`): the six runs of seed 73129 took 0.85, 0.82, 0.87, 0.86, 1.20 and 1.18 CPU-hours, 5.77 in all; at 49,152 updates, with 193 scored rows for 97, 11.54. Part 4: 1.69 and 1.57, 3.26 in all. The checks: 0.60. The job: 38.5; at the rates of the other two seeds 50.7 and 49.8. "11.6", "3.3", "38.6" and "up to 51" are right to the rounding of defect 8 (d). The longest run is 2.39 CPU-hours.
- **The share replaced**: 7.6% and 19.8% in the first and the last sixteen events of `P8` under the sweep, seed 73129 (`APP1 spread`).

## Not checked

- No run was made. Whether the hashes agree on the worker's host, the CPU times there, and how often that host stops or restarts are not known.
- How `G`, `F10` and `P8` behave at 0.15, and how the rule of two rises behaves at the present width. The test of the rule (defect 1) uses E41's curves at the half width, which are the only ones long enough.
- Whether the copy of the stream without torch gives the stream of the worker's torch build at 0.15. It gives E41's digests at 0.25 and 0.10 for seed 73129 at row 4,096; no receipt exists at 0.15.
- E48's four figures from its receipts; section 22.12 of the reviewer's memo beyond its existence; the definition of evidence level E2 and of "working rule 1"; the commit trailers of `AGENTS_POINTER.md` (the file exists on `$D`); the session named for `send_message`.
- That no run of a limited or a periodic network at another fraction exists was not looked for again; branches of workers now running were not fetched.
- The repaired list of changes (defect 4) was written from a reading of `exp42_ckpt.py`; it was not tried, since nothing may be run.
- `tools/minifly_check` was not run, since this check may write one file only.

## Appendix: the script behind the figures

Parts: `paths`, `flat`, `slow`, `cost`, `record`, `rule2`, `rule3`, `split`, or `all`. It reads receipts and sources with `git show` and writes nothing.

```python
import json, subprocess, sys
D, W, M = "origin/codex/minifly-consolidated-20261023", "origin/claude/gallant-dijkstra-i9zgsr", "MOD97_LOCAL_RULE_20261106"
SEEDS = (73129, 93683, 116089)
def raw(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{M}/{path}"], capture_output=True, check=True).stdout
show = lambda ref, path: json.loads(raw(ref, "exp45/" + path))
e41 = lambda rule, H, fp, s: show(D, f"e41/e41_{rule}_h{H}_f{fp}_{s}.json")
e42 = lambda cfg, rule, s: show(W, f"e42/e42_{cfg}_{rule}_{s}.json")
part = sys.argv[1] if len(sys.argv) > 1 else "all"
ROW = 256
def mean_rows(c, key, at, n):                      # mean of `key` over the n scored rows that end at update `at`
    rows = [r[key] for r in c if at - (n - 1) * ROW <= r["step"] <= at]; assert len(rows) == n, (at, len(rows))
    return sum(rows) / n
m3 = lambda c, at, key="R": mean_rows(c, key, at, 3)
m9 = lambda c, at: mean_rows(c, "R", at, 9)
rise9 = lambda c, at: m9(c, at) - m9(c, at - 8192)  # the brief's rule: flattened if this is below 3
def to80(c, key):
    v = [r[key] for r in c]
    return next((c[i]["step"] for i in range(len(v) - 2) if min(v[i:i + 3]) >= 80), None)

if part in ("all", "paths"):                       # every hash and path that the six checks cite
    r = show(W, "e42_qual/check8_73129_runs/e42_F10_S_dry2048_73129.json")
    print("check 1, F10 S dry 2,048:", r["tensors_sha256"][:8], "dose", r["dose"], "rows", len(r["curve"]), "complete", r["complete"])
    r = show(W, "e42_qual/check7_73129_runs/rep0/e42_P8_S_dry1024_73129.json")
    print("check 1, P8 S dry 1,024:", r["tensors_sha256"][:8], "theta", r["theta"], "replaced by row", [x["replaced_since_previous_row"] for x in r["curve"]],
          "has structure digests", sorted(r["structure_end_sha256"]))
    print("theta in e42_P8_S_73129.json:", e42("P8", "S", 73129)["theta"], "; chosen_theta in check10_73129.json:", show(W, "e42_qual/check10_73129.json")["chosen_theta"])
    st = show(W, "e42_qual/ckpt_selftest.json")
    print("ckpt_selftest.json: pass_", st["pass_"], "; cases", {k: (v["tensors_equal"], v["curve_equal"], v["digest_equal"], v["resumed_at"]) for k, v in st["selftest"].items()})
    for fp in (25, 10):
        for rule in "SB":
            for H in (512, 256):
                r = e41(rule, H, fp, 73129); d4 = next(x["data_digest"]["sha256"] for x in r["curve"] if x["step"] == 4096)
                print(f"e41_{rule}_h{H}_f{fp}_73129.json: dose {r['dose']}, rows {len(r['curve'])}, digest at 4,096 {d4[:16]}, tensors {r['tensors_sha256'][:8]}, row keys {sorted(r['curve'][1])}")
    src = raw(D, "exp42_cells.py").decode().splitlines()
    print("exp42_cells.py lines 414-415:", src[413].strip(), src[414].strip())
    for ref, p in ((W, "exp45/e42/interrupted"), (W, "exp45/e42/ckpt"), (W, "exp45/e42/selftest_case.sh"), (D, "exp45/e42"), (D, "exp45/e43/addendum2/e42_start/e42_P8_S_73129.json")):
        q = subprocess.run(["git", "ls-tree", "--name-only", f"{ref}:{M}/{p}"], capture_output=True, text=True)
        q2 = subprocess.run(["git", "cat-file", "-t", f"{ref}:{M}/{p}"], capture_output=True, text=True)
        print(f"{ref.split('/')[-1][:12]}:{p}:", q2.stdout.strip() or "ABSENT", (q.stdout.split()[:6] if q.returncode == 0 else ""))
    wts = subprocess.run(["git", "ls-tree", "-r", "-l", f"{W}:{M}/exp45/e42"], capture_output=True, text=True).stdout.splitlines()
    pt = [l for l in wts if l.endswith(".pt")]
    print("E42's committed .pt files:", len(pt), "; total bytes", sum(int(l.split()[3]) for l in pt), "; examples", [l.split()[-1] for l in pt[:3]])
    n = 0
    for cfg in ("G", "F10", "P8"):
        for rule in "SB":
            for s in SEEDS:
                r = e42(cfg, rule, s); ck = r.get("checkpointing")
                if ck and ck["resumed"]:
                    n += 1; print(f"E42 {cfg} {rule} {s}: resumed at", [(x["at_update"], round(x["cpu_replay"], 1)) for x in ck["resumed"]])
    print("E42 receipts of G, F10, P8 with a resumed entry:", n, "of 18")

if part in ("all", "flat"):                        # the rule 'has flattened' applied to the curves on record
    for cfg in ("G", "F10", "P8"):
        for rule in "SB":
            print(f"E42 {cfg} {rule}: nine-row rise at 24,576", [round(rise9(e42(cfg, rule, s)["curve"], 24576), 2) for s in SEEDS],
                  "; three-row", [round(m3(e42(cfg, rule, s)["curve"], 24576) - m3(e42(cfg, rule, s)["curve"], 16384), 2) for s in SEEDS])
    for H, fp in ((512, 25), (512, 10), (256, 25), (256, 10)):
        for rule in "SB":
            cs = [e41(rule, H, fp, s)["curve"] for s in SEEDS]; d = cs[0][-1]["step"]
            print(f"E41 H={H} f=0.{fp} {rule}: nine-row rise at {d}", [round(rise9(c, d), 2) for c in cs], "; three-row", [round(m3(c, d) - m3(c, d - 8192), 2) for c in cs],
                  "; level", [round(m3(c, d), 1) for c in cs])
    # the rule applied at earlier updates of the long curves, beside what the level did afterwards
    print("the rule applied before the end of a curve: (update, nine-row rise, level then, level at the end)")
    for H, fp, rule in ((256, 10, "B"), (256, 25, "S"), (256, 25, "B"), (256, 10, "S")):
        for s in SEEDS:
            c = e41(rule, H, fp, s)["curve"]; d = c[-1]["step"]
            print(f"  E41 H={H} f=0.{fp} {rule} {s}:", [(u, round(rise9(c, u), 1), round(m3(c, u), 1), round(m3(c, d), 1)) for u in range(12288, d + 1, 4096)])
    for cfg, rule in (("F10", "S"), ("F10", "B"), ("G", "S")):
        for s in SEEDS:
            c = e42(cfg, rule, s)["curve"]
            print(f"  E42 {cfg} {rule} {s}:", [(u, round(rise9(c, u), 1), round(m3(c, u), 1), round(m3(c, 24576), 1)) for u in range(12288, 24577, 4096)])

if part in ("all", "slow"):                        # a learner that follows the dense curve k times more slowly
    def slowed(c, k, dose):                        # R at update u is the dense curve's R at u / k (linear between rows)
        R = {r["step"]: r["R"] for r in c}
        def at(u):
            x = u / k; lo = int(x // ROW) * ROW; hi = min(lo + ROW, max(R))
            return R[lo] + (R[hi] - R[lo]) * (x - lo) / ROW if hi > lo else R[lo]
        return [dict(step=u, R=at(u)) for u in range(0, dose + 1, ROW)]
    for s in SEEDS:
        c = e41("B", 256, 10, s)["curve"]
        for k in (1.5, 2.0, 3.0):
            sc = slowed(c, k, 49152)
            print(f"dense, half width, f=0.10, B, seed {s}, a learner {k} times slower, at update 49,152: level {m3(sc, 49152):.1f}, dense level {m3(c, 40960):.1f} at 40,960, "
                  f"nine-row rise {rise9(sc, 49152):.1f} -> {'has flattened' if rise9(sc, 49152) < 3 else 'has not flattened'}")
    for s in SEEDS:
        c = e41("S", 256, 25, s)["curve"]
        for k in (1.5, 2.0):
            sc = slowed(c, k, 49152)
            print(f"dense, half width, f=0.25, S, seed {s}, a learner {k} times slower, at update 49,152: level {m3(sc, 49152):.1f}, dense level at the same update of its own run "
                  f"{m3(c, 40960):.1f} (40,960), nine-row rise {rise9(sc, 49152):.1f}")

if part in ("all", "cost"):
    tot = {}
    for s in SEEDS:
        h = {(cfg, rule): (lambda r: (r["cpu_updates"] + r["cpu_scoring"] + r["cpu_replacement"]) / 3600)(e42(cfg, rule, s)) for cfg in ("G", "F10", "P8") for rule in "SB"}
        # at 49,152 updates: updates and replacement doubled, 193 scored rows for 97
        h2 = {}
        for cfg in ("G", "F10", "P8"):
            for rule in "SB":
                r = e42(cfg, rule, s); h2[(cfg, rule)] = (2 * r["cpu_updates"] + r["cpu_scoring"] * 193 / 97 + 2 * r["cpu_replacement"]) / 3600
        p4 = 0.0
        for rule in "SB":
            r = e41(rule, 256, 10, s); pu, ps = r["cpu_updates"] / r["dose"], r["cpu_scoring"] / len(r["curve"])
            p4 += (122880 * pu + 481 * ps) / 3600
            if s == 73129: print(f"E41 h256 f10 {rule} 73129: {pu:.4f} CPU-s an update, {ps:.2f} a scored row; 122,880 updates and 481 rows: {(122880 * pu + 481 * ps) / 3600:.2f} CPU-h")
        tot[s] = (sum(h.values()), sum(h2.values()), p4)
        print(f"seed {s}: six runs at 24,576: {sum(h.values()):.2f} CPU-h {[round(v, 2) for v in h.values()]}; at 49,152: {sum(h2.values()):.2f}; twice the first: {2 * sum(h.values()):.2f}; part 4: {p4:.2f}; "
              f"three parts + part 4 + 0.6: {3 * sum(h2.values()) + p4 + 0.6:.1f} (with twice: {6 * sum(h.values()) + p4 + 0.6:.1f}; with part 4 at 3.3: {6 * sum(h.values()) + 3.3 + 0.6:.1f})")
    r = {(cfg, rule): e42(cfg, rule, 73129) for cfg in ("G", "F10", "P8") for rule in "SB"}
    per = lambda x, u: (x["cpu_updates"] + x["cpu_replacement"]) / 24576 * u + x["cpu_scoring"] / 97 * (u // 256 + 1)
    chk = 2 * per(r[("F10", "S")], 2048) + 2 * per(r[("P8", "S")], 1024) + per(r[("G", "S")], 2048) + per(r[("G", "B")], 2048) + per(r[("P8", "S")], 2048) + per(r[("P8", "S")], 2048)
    print(f"the checks (two F10 S of 2,048, two P8 S of 1,024, G S and G B of 2,048, two P8 S of 2,048): {chk / 3600:.2f} CPU-h")
    print("longest single run at 49,152 on seed 73129's rates:", round(max((2 * x["cpu_updates"] + x["cpu_scoring"] * 193 / 97 + 2 * x["cpu_replacement"]) / 3600 for x in r.values()), 2), "CPU-h")

if part in ("all", "record"):
    g = {rule: e42("G", rule, 73129)["curve"] for rule in "SB"}
    for cfg in ("F10", "P8"):
        for rule in "SB":
            c = e42(cfg, rule, 73129)["curve"]
            print(f"seed 73129, {cfg} {rule}: ratio to G on A {to80(c, 'A') / to80(g[rule], 'A'):.3f}, on R {to80(c, 'R') / to80(g[rule], 'R'):.3f}")
    for H, fp in ((512, 25), (256, 25), (512, 10), (256, 10)):
        S = [m3(e41("S", H, fp, s)["curve"], e41("S", H, fp, s)["dose"]) for s in SEEDS]; B = [m3(e41("B", H, fp, s)["curve"], e41("B", H, fp, s)["dose"]) for s in SEEDS]
        print(f"E41 H={H} f=0.{fp}: R under S {[round(x, 1) for x in S]}, under B {[round(x, 1) for x in B]}, B minus S {[round(b - a, 1) for a, b in zip(S, B)]}")
    import math
    print("scatter of a difference of two nine-row means for a row scatter of 2.0 and 4.0:", round(2.0 * math.sqrt(2) / 3, 2), round(4.0 * math.sqrt(2) / 3, 2),
          "; of a difference of two three-row levels:", round(2.0 * math.sqrt(2 / 3), 2), round(4.0 * math.sqrt(2 / 3), 2))
    Dd = 49152 - 512                                # describe's bounds at 49,152 updates (thr_desc: the dose less 512)
    print("describe with one run not reaching 80, both of 49,152 updates: 'more' if G's updates to 80 are at most", max(u for u in range(0, 49153, 256) if Dd / max(u, 1) >= 1.5),
          "; 'fewer' if the network's are at most", max(u for u in range(0, 49153, 256) if u / Dd <= 2 / 3), "; with the dose itself in place of the dose less 512:",
          max(u for u in range(256, 49153, 256) if 49152 / u >= 1.5), max(u for u in range(0, 49153, 256) if u / 49152 <= 2 / 3))
    c = e41("S", 512, 25, 73129)["curve"]
    print("E41 S h512 f25 73129: R of the last 12 rows", [x["R"] for x in c[-12:]], "; rows at or above 80:", sum(x["R"] >= 80 for x in c))
    for s in SEEDS:
        c = e41("S", 256, 25, s)["curve"]; u80 = to80(c, "R")
        after = [x["R"] for x in c if x["step"] >= u80]
        print(f"E41 S h256 f25 {s}: to 80 on R {u80}; afterwards {len(after)} rows, of which below 80: {sum(v < 80 for v in after)}; min {min(after)}, max {max(after)}; level at 40,960 {m3(c, 40960):.1f}")

if part in ("all", "rule2"):                       # the brief's rule and two stricter ones, applied before the end of E41's long curves
    rules = {"brief (one span of 8,192)": lambda c, u: rise9(c, u) < 3,
             "two spans, each under 3": lambda c, u: abs(rise9(c, u)) < 3 and abs(rise9(c, u - 8192)) < 3,
             "8,192 and 16,384, each under 3": lambda c, u: abs(m9(c, u) - m9(c, u - 8192)) < 3 and abs(m9(c, u) - m9(c, u - 16384)) < 3}
    for name, fn in rules.items():
        calls = wrong = 0; ex = []
        for H, fp in ((256, 10), (256, 25)):
            for rule in "SB":
                for s in SEEDS:
                    c = e41(rule, H, fp, s)["curve"]
                    for u in range(20480, 36865, 4096):
                        if fn(c, u):
                            calls += 1; gain = m3(c, 40960) - m3(c, u)
                            if gain >= 5:
                                wrong += 1; ex.append(f"{rule} f=0.{fp} {s} at {u}: {m3(c, u):.1f} then {m3(c, 40960):.1f}")
        print(f"{name}: 'has flattened' at {calls} of 60 points (12 half-width curves, updates 20,480 to 36,864); the level rose by 5 points or more afterwards in {wrong}: {ex}")
    for name, fn in rules.items():
        print(f"{name}: at the end of the curve:",
              {f"E41 h256 f{fp} {rule}": [fn(e41(rule, 256, fp, s)["curve"], 40960) for s in SEEDS] for fp in (25, 10) for rule in "SB"},
              {f"E42 {cfg} {rule}": [fn(e42(cfg, rule, s)["curve"], 24576) for s in SEEDS] for cfg in ("G", "F10", "P8") for rule in "SB"})

if part in ("all", "rule3"):                       # the first check's three-row form of the rule, for comparison
    calls = wrong = 0
    for H, fp in ((256, 10), (256, 25)):
        for rule in "SB":
            for s in SEEDS:
                c = e41(rule, H, fp, s)["curve"]
                for u in range(20480, 36865, 4096):
                    if m3(c, u) - m3(c, u - 8192) < 3:
                        calls += 1; wrong += m3(c, 40960) - m3(c, u) >= 5
    print(f"three-row form (one span of 8,192): 'has flattened' at {calls} of 60 points; the level rose by 5 points or more afterwards in {wrong}")
    # a fall: does any curve on record fall by 3 points or more between nine-row means 8,192 updates apart?
    falls = []
    for H, fp in ((256, 10), (256, 25)):
        for rule in "SB":
            for s in SEEDS:
                c = e41(rule, H, fp, s)["curve"]
                falls += [(rule, fp, s, u, round(rise9(c, u), 1)) for u in range(12288, 40961, 2048) if rise9(c, u) <= -3]
    for cfg in ("G", "F10", "P8"):
        for rule in "SB":
            for s in SEEDS:
                c = e42(cfg, rule, s)["curve"]
                falls += [(cfg, rule, s, u, round(rise9(c, u), 1)) for u in range(12288, 24577, 2048) if rise9(c, u) <= -3]
    print("nine-row falls of 3 points or more over 8,192 updates on record:", falls)
    # P8: share of connections replaced an event over the run, seed 73129
    for rule in "SB":
        r = e42("P8", rule, 73129)["curve"]; ev = [x["replaced_since_previous_row"] / 2560 for x in r[1:]]
        print(f"P8 {rule} 73129: share replaced an event, first 16 {sum(ev[:16]) / 16:.3f}, last 16 {sum(ev[-16:]) / 16:.3f}; by quarter of the run", [round(sum(ev[i:i + 24]) / 24, 3) for i in range(0, 96, 24)])

if part in ("all", "split"):                       # the places where a fraction is rounded (exp11_cell.py lines 238 and 240; exp42_cells.py line 219)
    DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
    for f in (0.50, 0.25, 0.15, 0.10):
        seeds = [73129 * 1000 + round(f * 100) + m * 7 + code[0] for _, m, code in DEV]
        print(f"f {f}: f * 100 = {f * 100!r}, round {round(f * 100)}, int {int(f * 100)}; taught pairs an operation {round(f * 7921)}; "
              f"the four shuffles of one run have distinct seeds: {len(set(seeds)) == 4}; reduction sample named e41|73129|{round(f * 100)}|add")
    allseeds = {(f, op): 73129 * 1000 + round(f * 100) + m * 7 + code[0] for f in (0.50, 0.25, 0.15, 0.10) for op, m, code in DEV}
    print("equal shuffle seeds across fractions:", [(a, b) for a in allseeds for b in allseeds if a < b and allseeds[a] == allseeds[b]])
    # the comparisons with G at a half under the brief's one-span rule and a margin of 10 points
    und = 0
    for cfg in ("F10", "P8"):
        for rule in "SB":
            for s in SEEDS:
                g, c = e42("G", rule, s)["curve"], e42(cfg, rule, s)["curve"]; d = m3(c, 24576) - m3(g, 24576)
                fg, fc = rise9(g, 24576) < 3, rise9(c, 24576) < 3
                und += not ((d >= 10 and fg) or (d <= -10 and fc) or (abs(d) < 10 and fg and fc))
    print("E42 at a half, the twelve comparisons with G at update 24,576 by the brief's rule: not decided", und, "of 12")
```
