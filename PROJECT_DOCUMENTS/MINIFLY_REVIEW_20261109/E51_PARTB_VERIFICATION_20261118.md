# Verification of the trial of E51, Part B: how much rehearsal keeps the first family (seed 73129)

18 November 2026. Written for the outside reviewer by an agent that was given the worker's files and the brief, and not the reviewer's expectations. It is a recomputation from saved receipts and saved weights. It does not replace the independent audit that the programme's rules ask for.

## Verdict

**Confirmed.** Every figure of the worker's tables of Part B (`E51_TABLES_B.md`: 34 rows of figures, 274 cells, the two readings "the least share that keeps", the two readings "smaller steps leave more", the two header lines and the two sentences of B2) and every value of its readings file (`E51_READ_B.json`: 296 values) was recomputed from the receipts by the script of this verification, with no difference. The three tables of the report's section 6 agree cell by cell. All eighteen run receipts of Part B (the sixteen runs and the two runs of check 8) are complete and record the settings their names state; the data digest of every scored row was recomputed from this verification's own streams and agrees, which confirms the share k of every run of B1, the order of gaps and reviews of B2 and the stream of B3. The five predictions are scored correctly in the report (PB.1 failed, PB.2 held, PB.3 failed, PB.4 held, PB.5 held), each by the rule that the brief fixed beforehand. The four checks 8 to 11 record what the brief asks. From the saved weights of two runs, the first family's counts, the new task's scores and the last probe were reproduced exactly.

**Figures that differ.**

1. The report states (section 6, B2) that the run ends at "4,616 updates under S, 4,448 under B". The receipts give 2,640 under S and 2,496 under B (8 gaps of 256 updates plus reviews of 592 and 448 updates); the worker's own tables give 2,640 and 2,496. The report's figures are wrong.
2. The report states that a share of 1/4 leaves the reduction score "7.6 (S) ... points below row 0". The receipts give 96.565 − 89.019 = 7.546 points, which rounds to 7.5; 7.6 is the difference of the two rounded figures. The figure for B, 9.2 (9.206), is right. The difference changes no reading.
3. A minor inaccuracy of text: the report says the second addendum's scored rows are "0, 64, 256, 1,024"; its receipt has 0, 64, 256, 1,024, 2,048 and 4,096. The point it makes (the addendum has no probe, so its first probe row at 80% cannot be given) stands.
4. Not a difference, but a matter of definition: "Part B took 4.95 CPU-hours" is the sum over the sixteen runs and the two runs of check 8; the sixteen runs alone took 4.64 CPU-hours.

**Sentences of the report that say more than the figures.**

1. "*With rehearsal at intervals the first family does not become cheaper to restore*" (rule B). The sentence applies the brief's fixed rule correctly (cycle 8 needs 48 review updates; the rule asks for at most 40, half of cycle 1's 80). The figures support a narrower statement: under back-propagation every review after the first needs fewer updates than the first (32 to 64 against 80), and the review of cycle 8 is 40% shorter than that of cycle 1, short of the half that the rule asks by one probe interval of 16 updates.
2. "*The explanation under test ... is refuted under both rules by its own criterion.*" This follows the brief's fixed criterion (more is left: 58.25 and 72.50 points against 0.00 and 0.00). The figures do not show that the earlier reviews alone account for the difference: the gap of cycle 1 is also the gap in which the new task is first learned (its held-out score rises from 0 to 6.0% under S and from 0 to 98.0% under B), whereas from cycle 4 on the new task is already at 35 to 99% when the gap begins. The comparison that the criterion makes is therefore between gaps that differ in more than the number of earlier reviews. The report itself adds that what is left is not monotone in the cycle and that the trial has one seed.

No other sentence of the report's sections on Part B says more than its figures.

## Terms, material and method

**Terms**, as the brief defines them. *The network* is the recurrent network of `arch20` at the present size (a byte table, a layer `l1` of 512 units, a recurrent layer of 256 GRU cells, a readout). *Rule B* is back-propagation (the exact gradient); *rule S* is the sweep, which passes the error backward through each linear map by the signs of its weights times one scale. *The first family* is add, sub, max and min under modulus 89; its *reduction pairs* are the pairs of add and sub whose answer needs the modulus (3,916 of each operation). *The reduction score* is the mean of the two percentages of whole answers right on the reduction pairs without letters (E43's score 1). *The new task* is E43's `unrelated` task; its *held-out score* is the percentage of whole answers right on E43's panel of 300 never-taught items. *The birth* is E26's network of the seed, trained on the first family alone. *An update* is one optimizer step on a batch of 64 items. *A row* is a number of updates at which E43's full scores are recorded; *a probe row* is one at which only the light probe is recorded. *The probe* consists of 400 reduction pairs (200 of add, 200 of sub, drawn by `Random("e51|73129|probe")`) and the new task's held-out panel; *the first-family probe* is the percentage of the 400 answered right. *The share k* in B1 means that each item is a first-family item with chance 1/k. *A receipt* is the file of results that a run or a check writes.

The brief's fixed readings (section "How Part B's result is stated"): **kept** at a share, if the reduction score at update 4,096 is within 5 points of row 0; **the least share that keeps**, the smallest share that is kept with every larger share also kept; **the review shortens**, if the review of cycle 8 needs at most half the updates of the review of cycle 1; **more is left**, if the first-family probe at the end of the gap of cycle 8 is at least 20 points above that of cycle 1; **smaller steps leave more**, if the first-family probe at the first probe row at which the new task's held-out score is at 80% is at least 20 points higher at a smaller rate than at 0.001.

**Material.** The worker's branch `claude/e51-reserve-017dSEpW` at 608f6b4c: the runner `exp51_partb.py`, the reader `e51_read.py`, the report `E51_REPORT.md`, the eighteen run receipts and four check receipts of Part B, eighteen weight files, `E51_TABLES_B.md` and `E51_READ_B.json`. E43's receipts and the second addendum's receipts were read on the worker's branch and compared with those on the default branch at 7cabb13a; the four that are used are byte-identical.

**Method.** Nothing was trained, torch was not used, and no code of another branch was executed: every file was read with `git show` into memory. The script is `e51_partb_verify.py` beside this file, called *the reader* below. It imports the verification of Part A (`e51_trial_verify.py`) for the items, the streams, the digest of the batches, the reader of weight files, the forward pass in double precision and the generation of answers; that script's forward pass was itself checked against Part A's receipts. "Part 1" and the like name the part of the reader that prints a figure.

## 1. The figures from the receipts (part 1)

**Completeness and settings (part 1.1).** Each of the eighteen receipts is complete and records: the tag of its file name; rule and `local` consistent with each other (S with the sweep, B without); seed 73129; batch 64; weight decay 0 in both groups of the optimizer, at the start and at the end; the birth `exp45/e26/e26_dev_<rule>_73129.pt`, whose file hash and tensors hash part 3 checks against the file; the number of updates its set requires (4,096 for B1; 4,096, 13,824 and 40,960 for B3 at 0.001, 0.0003 and 0.0001; for B2 the sum of the gaps and reviews); one optimizer whose step count equals the number of updates; the weight file of the last update; and no modified source at the time of the run. Every receipt records the same hash of the runner, which equals the hash of `exp51_partb.py` at 608f6b4c. B1 records k, a rate of 0.001, E43's six rows (0, 64, 256, 1,024, 2,048, 4,096) and the probe every 64 updates to 1,024 and every 256 after. B2 records a gap of 256, a limit of 1,024, eight cycles, a probe every 16 review updates, a rate of 0.001, and E43's rows at update 0 and at the end. B3 records its rate in the settings, and both groups read back at the start and at the end carry that rate; the probe is every 16 updates to 256 and every 64 after; E43's rows are at 0, 64, 4,096 and the end.

**The streams (part 1.1).** The reader built each run's batches from its own streams (the unrelated stream, the first family's taught half, and the choice `Random("e51|73129|joint|k").randrange(k) == 0`; for check 8 the addendum's `Random("e43|73129|joint").randrange(2)`) and recomputed the data digest at every scored row. All agree. The share of first-family items is 0.2504, 0.1259, 0.0627 and 0.0316 for k = 4, 8, 16, 32 (the receipts' item counts agree), and 0.4997 for check 8. For B2 the reader rebuilt the 2,640 (S) and 2,496 (B) batches by taking the receipt's review lengths, and the digest agrees; this confirms that the gaps took the unrelated stream alone and the reviews took the first family's stream continued from cycle to cycle. For B3 one pass of 40,960 batches of the unrelated stream reproduces the digests at rows 0, 64, 4,096, 13,824 and 40,960 of all six runs.

**The probe items (part 1.2).** The reader's own draw of the 400 probe items equals the indices recorded in all sixteen receipts.

**B2's cycles from the trace (part 1.3).** Each receipt's eight cycle records equal what the trace of probes gives. Every gap has its four probes, and every review stopped at the first probe at or above the goal.

**The worker's tables and readings (parts 1.4, 1.5).** All agree, as stated in the verdict.

**The report's figures (parts 1.6, 1.7).**

| Figure in the report | From the receipts | |
|---|---|---|
| B1 table, 12 rows; B2 table, 8 rows; B3 table, 6 rows | the same | agrees |
| 1/4 leaves the reduction score 7.6 (S) and 9.2 (B) points below row 0 | 7.546 and 9.206 | S differs (rounds to 7.5) |
| What is left falls steadily as the share falls | S 97.1, 89.0, 86.8, 63.2, 44.2, 0.0; B 96.7, 88.9, 78.4, 57.5, 34.6, 0.0 (k = 2, 4, 8, 16, 32, none) | agrees |
| B2 probe at update 0 and goal: S 95.25% and 85.73%; B 98.25% and 88.43% | 95.25 and 85.725; 98.25 and 88.425 (the tables print 88.42) | agrees |
| Every review reached its goal within 1,024 | yes, in all sixteen cycles | agrees |
| Unrelated held at the end of each gap 98 to 100% from cycle 2; cycle 1 S 6.0%, B 98.0% | S 6.0, 98.7, 100, 100, 99.7, 99.7, 100, 100; B 98.0, then 100 | agrees |
| At the end of the reviews of cycles 1 to 3: 0 to 35% (S), 0.3 to 70% (B); from cycle 4: 88 to 100% | S 0.0, 0.0, 35.3, then 88.3 to 99.7; B 0.3, 0.3, 69.7, then 91.7 to 98.7 | agrees |
| End of the run: 4,616 updates (S), 4,448 (B) | 2,640 and 2,496 | **differs** |
| Reduction score at the end 87.9 (S), 91.9 (B); unrelated 299/300 and 294/300 | 87.93, 91.91; 299, 294 | agrees |
| S: cycle 8 review 32 against 80; more left 58.25 against 0.00 | the same | agrees |
| B: 48 against 80; 72.50 against 0.00 | the same | agrees |
| S: what is left after the gaps of cycles 5 to 8: 74.5, 65.0, 54.3, 58.3 | 74.50, 65.00, 54.25, 58.25 | agrees |
| B3: probe at update 64 under S: 0.0%, 21.5%, 62.0% | the same | agrees |
| B3: when the new task is at 80%, the probe is at 0 to 0.25% at every rate | 0.00 to 0.25 | agrees |
| Check 10: 381 (S) and 393 (B) of 400 at update 0 | the same (part 3) | agrees |
| Check 11: 6.1 CPU-hours forecast; 4.95 taken | 6.125; 4.954 with the two runs of check 8, 4.643 without | agrees, with that definition |

## 2. The five predictions (part 2)

| | Statement in the brief | Deciding figures | Outcome | Report |
|---|---|---|---|---|
| PB.1 | A share of a quarter (k = 4) keeps the first family, under both rules | Reduction score at row 0 and at 4,096: S 96.57 and 89.02 (7.55 points below); B 98.14 and 88.93 (9.21 points below). Kept requires at most 5 | failed | failed: correct |
| PB.2 | A share of 1/32 does not keep it, under either rule | S 52.34 points below row 0 (45.8% of row 0); B 63.51 below (35.3%) | held | held: correct |
| PB.3 | The review shortens, under back-propagation | B: cycle 1 needs 80 review updates, cycle 8 needs 48; at most 40 required. (S: 80 and 32, shortens) | failed | failed: correct |
| PB.4 | More is left after a gap at cycle 8, under back-propagation | B: first-family probe at the end of the gap 0.00 (cycle 1) and 72.50 (cycle 8); at least 20 points required. (S: 0.00 and 58.25) | held | held: correct |
| PB.5 | Smaller steps do not leave more, under either rule | First-family probe at the first probe row with the new task at 80%: S 0.00, 0.00, 0.00; B 0.00, 0.25, 0.25 (rates 0.001, 0.0003, 0.0001); at least 20 points more required | held | held: correct |

**Each reading follows the rule fixed beforehand.** The worker's reader (`e51_read.py`) applies exactly the brief's rules: kept as an absolute difference of at most 5 points of the reduction score; the least share as the first break in the sequence 1/2, 1/4, 1/8, 1/16, 1/32; the review shortens as `cycle 8 ≤ cycle 1 / 2`; more is left as a difference of at least 20 points; smaller steps leave more as a difference of at least 20 points at the first probe row with the new task at 80%. The reader's own implementation from the brief's text gives the same verdicts. The phrase "within 5 points of row 0" admits a second reading (the score at 4,096 as a percentage of row 0, within 5 of 100); it gives the same verdict at every share (part 2.1). The B3 phrase "the last probe row at which the first-family share is still at least half of its value at update 0" also admits two readings (the last such row in the run, or the row before the first drop below half); both give the same rows (0, 16 and 96 under S; 0, 16 and 48 under B; part 2.4), and the new task's score there is 0.0% in every case.

**Figures behind the report's wording (part 2.2 to 2.4).** For B1, the claim that the new task is learned "as fast as or faster than with a share of 1/2" can only be checked on E43's scored rows, because the second addendum has no probe: under S every share first reaches 80% on the held-out panel at row 1,024, as k = 2 does; under B every share reaches it at row 256 (84 to 95%), where k = 2 is at 65.7% and first reaches 80% at row 1,024. The claim is supported on that grid. The first probe row at 80% and the first probe row from which the score stays at 80% are the same in all eight runs of B1. For B2 the figures of the sentence about the review under B are given in the verdict. For B3 the probe curves show that the first family is lost over more updates at smaller rates (under S at update 64: 0.0, 21.5 and 62.0%), and that at every rate it is near 0% before the new task's held-out score passes 80% (at updates 384, 960 and 2,304 under S; 224, 448 and 1,024 under B).

## 3. The checks before the runs of Part B (part 3)

- **Check 8** (B1's runner is the addendum's at k = 2). The receipt records, for each rule, that scores 1 and 2 at row 64 equal the addendum's, that the tensors hash at 4,096 is the addendum's (S `15dc8293`, B `f89d3089`), and that the data digest is the same. The reader compared the two runs' receipts with the addendum's receipts on the default branch: scores 1 and 2 at row 64 equal, score 1 equal at all six rows, data digests equal, tensors hashes equal. This is what the brief asks.
- **Check 9** (B3 at 0.001 is E43's run). The receipt records equal counts at rows 64 and 4,096, equal tensors hashes at 4,096 (S `88b52850`, B `7c70f8b0`) and a rate of 0.001 read back from both groups. The reader confirmed all three against E43's receipts on the default branch, and the data digests at rows 0, 64 and 4,096 also agree. This is what the brief asks.
- **Check 10** (the probe at update 0). The receipt records 381 (S) and 393 (B) of 400 for the probe, the same counts for the same items in a generation over all 7,921 pairs of add and of sub, that all 400 are reduction pairs, and that they are distinct. The brief asks for equality with "the same 400 items in E43's score 1 at row 0"; E43's receipt keeps only counts, so the worker regenerated score 1's answers. The reader loaded the birth (file and tensors hashes equal to the receipts'), found 381 and 393 with its own forward pass, and found that its answers on all reduction pairs give 7,563 (S) and 7,686 (B), the numbers in E43's receipt at row 0. The probe at update 0 is 381 or 393 in all five runs of each rule. The check records what the brief asks, by the only route the record permits.
- **Check 11** (cost). The receipt records a forecast of 6.125 CPU-hours for sixteen runs, from Part A's measured CPU-seconds per update of the arm `open` (0.0934 for S, 0.0881 for B; the reader recomputed them from Part A's receipts) with B2 at its upper limit. Two terms are constants of the runner rather than measurements (22.0 seconds for a full row of E43's scores and 1.5 seconds for a probe). The brief asks for a forecast from Part A's measured CPU-seconds per update, and that is what the receipt gives.
- **Order.** The receipts of checks 8, 9 and 11 were written at commit f2733afb, check 10's at aa88d07f. The logs show that the two runs of B3 at 0.0001 started in the same second as the first run of check 8, that is, before checks 8 and 9 had passed. Checks 8 and 9 require runs by their nature (check 9's run is itself the B3 run at 0.001), and the report states this order. No check failed, so the order changed nothing.

## 4. From the weights (part 4)

All eighteen weight files of Part B have the SHA-256 that their receipts record. Two runs were reproduced with the reader's forward pass from their final weights:

| Run | Tensors hash | First family (reduction pairs add+sub of 3,916+3,916; other pairs; max; min; held-out at 12 letters) | New task held / taught | Last probe (first family, held) |
|---|---|---|---|---|
| B1, k = 8, rule B, row 4,096 | equal | 2,883+3,257 (78.40); 3,228+2,984; 7,735; 7,639; 236/230/296/295 — equal to the receipt | 299 / 298 of 300 — equal | 329 of 400, 299 — equal |
| B3, rate 0.0003, rule S, row 13,824 | equal | 0+0 (0.00); 0+0; 0; 0; 0/0/0/0 — equal to the receipt | 300 / 300 of 300 — equal | 0 of 400, 300 — equal |

The second sample tests the forward pass on the first family only through counts of zero; the non-trivial counts of the first sample and of the birth in part 3 (381 and 393 of 400; 7,563 and 7,686 reduction pairs) test it with nonzero counts.

## 5. What Part B does and does not show

Part B starts from one network per rule that was trained on four operations under modulus 89, and teaches it an unrelated task. With no rehearsal the first family's reduction pairs are lost completely while the new task is learned. When one item in k is a first-family item, the share of the reduction score that remains after 4,096 updates falls as k grows: 98.6 to 100.5% of its start at a share of 1/2, 91 to 92% at 1/4, 80 to 90% at 1/8, 59 to 65% at 1/16 and 35 to 46% at 1/32; only the share of 1/2 is within the brief's 5 points. The new task reaches at least 98.7% at every share. When the first family is rehearsed only in reviews between gaps of 256 unrelated updates, nothing of it is left after the first two gaps, and from the fourth gap on 46 to 78% of the probe remains after a gap; the review that restores it to 90% of its start needs fewer updates in the last cycle than in the first (80 and 32 under the sweep, 80 and 48 under back-propagation), though not in every cycle (128 in cycle 2 under the sweep). Smaller learning rates without rehearsal delay the loss but do not prevent it: at every rate the first family is at 0 to 0.25% of the probe by the time the new task is at 80%, and at 0 at the end. These are the results of one seed and one birth per rule. Part B does not show how the results vary between seeds; it does not separate the effect of earlier reviews in B2 from the fact that the new task is already learned when the later gaps begin; it does not test shares between 1/2 and 1/4, a share that changes with time, more than two tasks, or runs longer than those of the brief.

## What could not be checked

The updates were not repeated, so the training itself is verified only through the data digests, the hashes and the scores of the saved weights. Only the final weights of each run are saved, so the rows and probe rows before the end, including every probe of B2's cycles, are verified from the receipts only. Of the final weights, two runs were scored with the forward pass; the other sixteen were checked by their file hashes alone. The CPU times are the receipts' own and could not be measured again. The partial receipts of the restarted B3 runs at 0.0003 (in `partB/aside/`) were not examined beyond their presence.

## Running time

Run from the repository root as `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e51_partb_verify.py [1 2 3 4]`, on a machine with 4 cores, one thread: part 1, 21 seconds; part 2, under 1 second; part 3, 8 seconds; part 4, 16 seconds.
