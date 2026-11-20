# The loop at the smallest size: report (19 and 20 November 2026)

**By** the outside session that wrote the audits beside this folder. **Authorisation:** Mr. Xu, 19 November 2026, to this session: "So the current agents are obviously stuck. If you have more thoughts, try to implement them to unblock the current situation". **Status:** a small case, in the sense of the reviewer's status rows 137 to 153. Development level: five seeds, one stream each, not repeated by anyone else, not verified by the coordinator. It is not a version of the adopted core and claims no checker gate.

Everything below is from `cells_loop.py` and `look_inside.py` in this folder (numpy only), from the plans `PLAN.md` and `PLAN_2.md`, which were committed before the runs they govern, and from the result files in `res/`. Commands are in section 7.

**Sections 10 to 16 were added on 20 November,** after Mr. Xu's words of that day to this session ("Keep working, see how much you can push along the road map. Feel free to revise anything you like.") and his three messages during the work, which are quoted in sections 13 and 14. The plans `PLAN_4.md` to `PLAN_11.md` were each committed before the runs they govern, with the development outcomes entered before any reported seed was read. The programs are `cells2.py`, `hold.py`, `hold_x.py`, `prose.py`, `agree.py` and the compiled kernels `kern.c` to `kern4.c`; the tables of the reported seeds are in `report4.txt`. The status is the one stated above: development level, five reported seeds where a table says so and one development seed elsewhere, nothing repeated or verified by anyone else.

## 1. In brief

1. **The loop runs.** A system in which the cell that names the next byte forms its mismatch when the byte arrives, and the cells in between change their own strengths from that mismatch relayed to them over return connections, learns the text it reads. Nothing passes backward over the line; the return connections are learned by a local product and are never copied from the forward strengths.
2. **On single-digit sums nobody answers a sum never shown, back-propagation included.** Five reported seeds, 60,000 lines: sums shown 97.5% under the local relay and 96.2% under the comparator, against 12.5% with no cells in between; sums never shown 0% at the median in every arm that learns (5% where the cells in between do not change). The yardstick that stopped the agents on 19 November ("it holds a table, it does not add") is one that the comparator fails as well. My reading: an unseen pair of single digits is not a new combination of anything taught, and 80 facts do not force an order of the digits.
3. **On two-digit numbers the cells answer sums never shown.** There a pair never shown is a new combination of digits that were shown. Five reported seeds, 400,000 lines, 256 cells in between, nothing about digits, places or carrying laid down: the whole answer is right on **88.0% of sums never shown** (82.5 to 92.7 over the seeds), as often as on sums shown (88.4%). With no cells in between: 0%. With cells in between that do not learn: 0%. With return connections left at their random birth values: 83.9%. The comparator, in the three forms run: 58.6%, 62.6% and 70.3%.
4. **What the agents laid down by hand has formed by learning.** In the saved networks the strengths of most cells from the digit lines lie in the order of the digits; cells weigh the two tens digits equally, so that they add them; and in the cells that read all four digits a tens digit weighs about ten times a units digit (section 5).
5. **What does not work yet:** from yes or no alone the same system has 8.9% of never-shown sums right at this length. And the system is not yet as a cell has it in five named respects (section 6).

### Added on 20 November

6. **The runs are fast now.** My programs were the slow kind: numpy with a Python loop over every byte. Their inner loops are compiled, checked against the numpy programs on every option, and give the same scores; 400,000 lines take 16 to 34 seconds where they took about three minutes, and the runs without the delay chain take minutes where they took half an hour and more. Of the sparsity there is, the input's was used from the start and the fading is now done exactly as one scale factor; cells that are silent most of the time give about 70% where logistic cells give 84% (section 13).
7. **Closer to what a cell has, one item at a time** (five seeds; every line read: 84.2%). A cell that reads 64 of the 140 lines and rewires by its own strengths: 80.8%; 32 lines: 42.7% with rewiring, 7.5% without. One sign for each cell: 70.7% (section 10).
8. **The relay works over two steps**: 128 + 128 cells reach what 256 reach in one step, 84.4% (section 10).
9. **Keeping, in parts** (five seeds). The loss of the first task is total within a few thousand lines and is not the fading; a little over half of it is written in the answering cell; what was learned returns ten to seventeen times faster than it was learned. When the second task is written into new cells, one old line in 32 keeps 67% of the first task, where it keeps 16% otherwise (section 11).
10. **Cells that hold bytes themselves, without the chain of delay cells** (section 12). Single-digit sums: 98.3% (five seeds). One digit held across 0 to 8 letters (five seeds): 91% and 99% on the two texts with one crossing between cells, 28% and 45% without, the same at every length of gap, and as good at 10 to 16 letters, which were never taught. Two-digit sums: not learned with the present byte alone (5%); with the two bytes before it on lines as well, where the earlier system had nine, 66% of never-shown sums with the crossing and 23% without (five seeds). With letters between the two numbers: not learned yet (8%, development seed).
11. **Two small language tasks** (section 14, development seed). The next byte of prose never read: 46% and 52% on two books, level with a table of the last two to three bytes and below a table of four. A verb agreeing with its subject across words in between: 100% on pairs of noun and verb never shown together; with a noun of the other number in between, 100% with the crossing on two seeds of three (55% on the third) and chance without it on all three.
12. **Corrections of my own earlier statements** are in section 12 (a reading that did not survive its test), section 13 (a reading from the development seed that the reported seeds did not bear out) and section 14 (what the comparator of sections 2 to 6 is).

## 2. The system, in the words of the plan

One line for each byte value; a chain of delay cells, so that the present byte and the bytes before it are present on lines. H cells in between: each adds its strengths from the active lines, less a threshold, and gives the logistic function of that. One answering cell with a branch for each byte value in use; it reads the cells in between and nothing else, and names the byte of its largest branch. When the next byte arrives, the answering cell has for each branch a mismatch (1 less its output for the byte that arrived, 0 less its output for the others). It changes its own strengths by mismatch times the sending cell's output. Each cell in between receives the mismatches over return connections, summed with those connections' strengths, and changes its strengths from the lines that are active, and its threshold, by that sum times h (1 - h), its own output's sensitivity. A return connection changes by the same product as the forward strength between the same two ends, and both fade at the same slow rate, which brings them to agree (their cosine is 1.00 at the end of every run with fading).

One stream, one byte at a time, no batch, nothing reset, nothing averaged over lines, no pass backward over a line, no quantity taken from another cell's strengths.

The arms: `none` (no cells in between: the answering cell reads the lines); `fixed` (cells in between that never change); `relay-fixed` (return connections at their random birth values); `relay-learned` (the candidate); `comparator` (the signal to a cell in between computed from the answering cell's forward strengths: back-propagation through this network, which a cell does not have; **corrected in section 14**: with one step of relay this arm is the local rule itself with return strengths born at nothing); `relay-learned` under yes or no (the reviewer's rule Q in the answering cell: only the branch that named the byte has a mismatch, and on yes the others).

## 3. First plan: single-digit sums (the reviewer's `addtext.py` text)

Lines such as `7+8=15`, 20 of the 100 pairs never shown (the same 20 as `addtext.py` draws for the seed), 60,000 lines, 32 cells in between, the setting chosen on a development seed (`PLAN.md`, section 8). Whole answer right, %, each of the five reported seeds and the median (`report1.txt`):

| Arm | Sums shown | Sums never shown |
|---|---|---|
| `none` | 10.0, 11.2, 12.5, 12.5, 12.5; **12.5** | 0, 0, 0, 0, 0; **0** |
| `fixed` | 2.5, 2.5, 3.8, 2.5, 6.2; **2.5** | 5, 5, 0, 0, 10; **5** |
| `relay-fixed` | 90.0, 100, 95.0, 95.0, 97.5; **95.0** | 5, 0, 5, 0, 0; **0** |
| `relay-learned` | 97.5, 100, 92.5, 100, 97.5; **97.5** | 0, 5, 0, 0, 0; **0** |
| `comparator` | 96.2, 98.8, 92.5, 93.8, 98.8; **96.2** | 0, 5, 5, 0, 0; **0** |
| `relay-learned`, yes or no | 32.5, 42.5, 33.8, 31.2, 33.8; **33.8** | 0, 0, 0, 0, 0; **0** |

- The hypothesis of the first plan (the cells come to answer sums never shown) is **refuted in that form**, by its own rule: 0% at the median with sums shown learned. It is refuted for the comparator in the same way.
- The cells in between do learn from the relay: 97.5% against 12.5% without them and 2.5% when they do not change. The local rule and the comparator cannot be told apart on this text.
- Of the 99 never-shown pairs of the five seeds whose sum occurs among the pairs shown, the candidate has 1 right. On the development seeds neither 240,000 lines nor a fading of the cells' strengths changed this, for either rule (`PLAN.md`, section 8).
- Predictions: P1 held (the comparator at 90 or more on 5 of 5 seeds), P3 held, P5 held, P6 held; P2, P4 and P7 failed; P8 did not apply.

For the reviewer's rows: a table of pairs by 300 cells gave 92 to 96 shown and 0 never shown (row 143). Thirty-two cells that learn give the same picture. The difference from the systems that reached 80 to 100 on sums never shown is the one the audit names: in those the program forms the sum.

## 4. Second plan: two-digit numbers

Lines such as `07+85=092`, a and b from 0 to 99 at fixed width (the first family's way of writing a number), 2,000 of the 10,000 pairs never shown, 400,000 lines, 256 cells in between, nine delay cells, the setting chosen on development seeds (`PLAN_2.md`, section 7). Scored on 500 pairs shown and 500 never shown; each figure is the mean of a run's last three readings; whole answer right, % (`report2.txt`, `report2_comparator_own.txt`):

| Arm | Sums shown, median | Sums never shown: each seed; median | Never shown by place, median: hundreds, tens, units |
|---|---|---|---|
| `none` | 0.0 | 0, 0, 0, 0, 0; **0.0** | 78, 1, 0 |
| `fixed` | 0.0 | 0, 0, 0, 0, 0; **0.0** | 65, 5, 0 |
| `relay-fixed` | 84.7 | 81.9, 85.5, 89.3, 83.9, 82.5; **83.9** | 95, 87, 100 |
| `relay-learned` | 88.4 | 82.5, 92.7, 90.3, 88.0, 85.2; **88.0** | 98, 92, 100 |
| `comparator`, at the relay's setting | 62.1 | 54.8, 63.4, 58.6, 57.9, 70.7; **58.6** | 90, 67, 100 |
| `comparator`, at its own best setting of the grid | 65.3 | 62.0, 77.1, 62.6, 68.7, 59.1; **62.6** | 92, 74, 100 |
| `comparator`, answering cell started at random (added after the registered report) | 70.5 | 71.6, 70.3, 76.9, 67.6, 68.3; **70.3** | 85, 81, 100 |
| `relay-learned`, yes or no | 8.4 | 2.0, 9.6, 5.7, 11.2, 8.9; **8.9** | 93, 42, 40 |

- The hypothesis of the second plan **holds** in the sense intended: the candidate has at least 50% of sums never shown right at the median and is not below the comparator. The plan's wording was "within 15 points of the comparator"; it was written expecting the relay at or below the comparator, and the relay is 18 to 29 points above the comparator's three forms. I state the wording's failure to foresee this and do not count it against the hypothesis.
- Sums never shown are answered as often as sums shown, in every arm that learns. The tens digit, which needs the carry from the units, is the place that fails.
- Random return connections that never change are enough for most of it (83.9%); learned ones add about four points.
- **I do not read the comparator's lower figures as the local rule being better than back-propagation.** The comparator here is plain steps of one size on one stream, tuned over 16 settings on one development seed. It is not the record's reference, which uses batches and AdamW. One further variant was run after the registered report (`report2_comparator_random_start.txt`): with the answering cell's strengths started at random values, as the return connections are, the comparator has 71.6, 70.3, 76.9, 67.6 and 68.3% of sums never shown right, 70.3 at the median. So part of the difference is the start, and I have not established the rest. What the plan asked is whether the relay stays within reach of the comparator; it does.
- From yes or no alone the system has learned little at 400,000 lines (units 40%, tens 42%). On single digits the same arm stood at 33.8% of sums shown where the rule told the byte stood at 97.5%. Learning from yes or no through the relay is much slower; whether it arrives with more lines was not run.
- Predictions: Q2, Q3, Q4 and Q6 held; Q5 failed; Q1 ("within 15 points of the comparator") held in the sense that the relay is not more than 15 points below, and not by its letter, since the relay is more than 15 points above.

For the reviewer's rows on numbers from 0 to 99 (written there without leading zeros, so the figures are beside one another and not on one text): with nothing laid down, a pool of 4,000 random cells gave 24 to 32 on sums never shown at 120,000 lines; with holding cells, a carry cell and gates laid down, 88 at 240,000 lines and 95 at 960,000 (rows 145, 146).

## 5. What the cells in between have formed

`look_inside.py` reads the saved strengths of the candidate's five reported networks. For each cell and each delay at which a digit of one of the two numbers stands, it sets the cell's ten strengths from the digit lines against the digits 0 to 9. A cell is *ordered* at a delay when the correlation is at least 0.9 in size. At birth no cell of 256 is ordered at any delay.

At the byte `=` the first number's tens digit stands 5 bytes back and its units digit 4, the second number's tens digit 2 and its units digit 1 (`look_inside_output.txt`):

| Seed | Cells of 256 ordered at one or more of these four delays | Ordered at delays 5 and 2 with one sign (the two tens digits) | Slope at the second over the first there, median | Ordered at all four with one sign | In those, slope at a tens digit over a units digit: median (range) |
|---|---|---|---|---|---|
| 73129 | 161 | 146 | 1.00 | 25 | 9.8 (6.9 to 15.5) |
| 93683 | 163 | 140 | 1.00 | 26 | 9.1 (5.9 to 18.0) |
| 116089 | 160 | 146 | 0.99 | 26 | 9.2 (6.6 to 12.2) |
| 406364 | 176 | 152 | 1.00 | 25 | 10.8 (6.8 to 20.0) |
| 545353 | 161 | 138 | 0.99 | 25 | 10.2 (7.4 to 15.6) |

Three readings, mine.

1. **The order of the digits has formed.** More than half of the cells in every network carry a digit as an amount, with the ten strengths in the order of the digits. Nothing in the program knows that order; on single digits (section 3) it did not form.
2. **Cells that add have formed.** About 145 cells in each network have the same slope for the two numbers' digits of one place (ratio 1.00 at the median), so that their drive follows the sum of the two digits. The reviewer's lines reached equal strengths with a rule made for the purpose (status row 147); here they follow from the relay. One byte later the same two delays hold the two units digits, so the same cells then add the units, which the tens digit needs for the carry.
3. **Place value has formed.** About 25 cells in each network read all four digits with one sign, and in them a tens digit weighs about ten times a units digit (medians 9.1 to 10.8): their drive follows the value of a + b. That is what the hundreds digit, named at `=`, depends on.

Limits of this look: 0.9 is my level; what the answering cell does with these cells was not traced; the tens digit is still wrong in about 8% of sums.

## 6. What is shown, what is not, and what is still not as a cell has it

**Shown, at development level:** that a mismatch formed in the cell that names the byte, relayed once over learned return connections, teaches cells in between a text; that on a text with parts those cells answer never-shown combinations as often as shown ones; that the order of the digits, equal strengths and place value form without being laid down.

**Not shown:** anything on the first family (task byte, letters, the reduction by the modulus), on a tissue, on cells that hold a byte themselves, on keeping what was learned, on cost beside the comparator. One text each, five seeds, one stream a seed. The comparator was tuned on one development seed. Lines are scored with the right bytes read in, as in the reviewer's programs; for a whole answer this equals the system's own answer.

**Laid down by the program:** one line a byte value; the chain of delay cells; the sizes; random birth values.

**Still not as a cell has it** (stated in the plan before any run): a strength may be of either sign; the mismatch is signed, one value for each branch of the answering cell; there is one answering cell with a branch for each byte value; every cell in between reads every line; the cells in between do not send to one another. Three of these were tried once each on development seed 1 and are exploratory (`res/dev2x_*.json`): with 128 cells in between 77% of never-shown sums are right and with 64 cells 69%, against 85% with 256; with each cell reading a random 64 of the 130 lines 55%, still rising, and with 32 lines 25%; with 800,000 lines 88%, against 85% at 400,000.

**NOT_RUN:** numbers written without leading zeros; more than two digits; subtraction; longer learning under yes or no; more than one answering cell; cells of one sign; a second task.

## 7. Commands and cost

    python3 cells_loop.py check
    python3 cells_loop.py dev                                                        # PLAN.md, development grid (72 runs, seed 1)
    python3 cells_loop.py report H=32 eta1=1/16 eta2=1/4 lam=0 tag=rep1              # report1.txt
    python3 cells_loop.py report text=d2 H=256 eta1=1/4 eta2=1/16 lam=1e-5 lines=400000 every=50000 tag=rep2
    python3 cells_loop.py report text=d2 H=256 eta1=1 eta2=1/16 lam=0 lines=400000 every=50000 tag=rep2own only=comparator
    python3 cells_loop.py run rep2save_relay-learned_D_<seed> text=d2 arm=relay-learned seed=<seed> H=256 eta1=1/4 eta2=1/16 lam=1e-5 lines=400000 every=50000 save=1
    python3 look_inside.py res/rep2save_relay-learned_D_*.npz

A run of the first plan is 13 CPU-seconds; of the second about 3 CPU-minutes (3.6 million bytes). The candidate of the second plan has 40,205 strengths, of which 3,328 are return connections. Everything in this folder took about four and a half CPU-hours, most of it the development grids.

## 8. What I would have the agents do with it

1. Take the loop as the object to build on: the mismatch formed where the byte is named, relayed over learned return connections, each cell changing its own strengths from what reaches it. It is forty lines (`System.attempt_and_learn`).
2. Measure generalisation only on texts whose unseen items are new combinations of taught parts, with the comparator beside every figure. Single-digit sums are not such a text.
3. Remove what is not yet as a cell has it one item at a time under the same score: yes or no in place of the arriving byte; sparse reading; strengths of one sign for each cell; several answering cells that come to one voice (the reviewer's rows 137 and 140); cells that hold bytes themselves, with a trace at each connection, in place of the delay chain. Then the first family beside the record's reference, and then a second task, which is where the question of keeping begins.

## 9. Third plan: keeping (added after Mr. Xu's word "feel free to experiment")

`PLAN_3.md` (committed before any run), `keep.py`, `res/keep_*.json`. The system of the second plan reads 400,000 lines of addition and then 200,000 lines of subtraction (`aa-bb=ccc`, larger number first). Whole answer right on 500 never-shown pairs of each task, %, five reported seeds and the median:

| Arm | Addition after its own phase | Addition at the end | Subtraction at the end |
|---|---|---|---|
| `plain` | 82, 82, 86, 85, 88 | 2, 2, 2, 1, 2; **1.8** | 82, 90, 91, 86, 86; **86.0** |
| `stiffen` (each strength divides its step by 1 + the total size of its own past changes / 100) | 80, 80, 85, 85, 84 | 3, 2, 2, 2, 3; **2.2** | 76, 79, 86, 66, 76; **76.0** |
| `rehearse` (one line in eight of the second phase is an addition line) | as `plain` | 73, 71, 69, 72, 69; **71.2** | 83, 83, 88, 80, 84; **83.2** |
| `mixed` (both from the start, 600,000 lines) | - | 77, 76, 66, 80, 73; **75.6** | 81, 78, 72, 82, 85; **81.2** |

- **Nothing of addition is kept without rehearsal** (1.8%), as in every line of the record. The cells that learn by the relay forget as the dense network does.
- **A connection that stiffens with its own use does not help.** On the development seed, constants of 0.1 to 10 (the plan's grid) stopped all learning, and constants of 100 to 100,000 (added when those failed) protected nothing; no value lies between. This agrees with the reviewer's finding on the dense network (memo, 29.3, reading 4) and with the advice memo's thesis: what is kept is decided by where a thing is written, not by how firmly.
- **One line in eight of rehearsal keeps addition near the level that learning both together reaches** (71 against 76), while subtraction is learned (83).
- **Two forms of rehearsal made by the system itself failed, on the development seed only** (exploratory): when the system writes lines from its own outputs and learns from them, its lines turn into the new task within a few hundred lines, because its prediction of the operator byte turns; and an answering cell that changes only the branch of the arriving byte and of the byte wrongly named, so that the old operator's branch is left standing, learns far worse (30%) and still keeps nothing.
- Predictions: K1 held; K2 failed; K3 held (71.2 and 83.2); K4 held (75.6 and 81.2).

What this gives is the first measured part of keeping on cells that learn: the loss is total, stiffening is not the way, and a small share of rehearsal suffices. The open parts are rehearsal that the system makes itself, and writing a new task where the old one left room.

## 10. Fourth plan, parts A and B: closer to what a cell has; the relay over two steps

`PLAN_4.md` (committed before its runs; the development outcomes were entered before the reported seeds were read), `cells2.py`, `report4.txt`, `res/c2_r_*.json`. The system of sections 4 and 9 (256 cells in between, two-digit sums, 400,000 lines, the relay with learned return connections), with one thing changed at a time. Whole answer right on 500 never-shown pairs, %, mean of the last three readings; the five reported seeds and their median. (This runner has one more byte value, the minus sign, than the second plan's, so its plain system is drawn afresh: 84.2 here against 88.0 there.)

| | 73129 | 93683 | 116089 | 406364 | 545353 | median |
|---|---|---|---|---|---|---|
| every cell reads all 140 lines | 84.2 | 82.7 | 86.4 | 81.8 | 86.7 | **84.2** |
| 32 lines a cell, drawn at birth and kept | 14.5 | 7.3 | 7.5 | 15.9 | 7.3 | **7.5** |
| 32 lines a cell; every 4,096 bytes each cell withdraws its weakest connection and tries another line | 44.4 | 40.9 | 42.7 | 45.1 | 31.6 | **42.7** |
| 64 lines a cell, with the same rewiring | 68.1 | 83.0 | 84.3 | 80.8 | 73.0 | **80.8** |
| one sign for each cell (a quarter of the cells inhibitory; no strength from a line below zero) | 70.7 | 73.9 | 64.0 | 62.1 | 72.3 | **70.7** |
| the relay over two steps, 128 + 128 cells | 82.5 | 85.9 | 85.0 | 82.6 | 84.4 | **84.4** |

On the development seed only: 16 lines a cell with rewiring 10.9; one sign together with 32 lines and rewiring 41.8; two steps with 256 + 256 cells 84.3.

- **Reading few lines costs much, and a cell that rewires by its own strengths recovers a large part.** With 32 of the 140 lines fixed at birth, 7.5%. When each cell withdraws the weakest of its connections and tries a line it does not read (the cell compares nothing but its own strengths), 42.7%, still rising at the end of every run. With 64 lines and the same rewiring, 80.8%, within four points of reading every line.
- **One sign for each cell costs about 14 points** (70.7 against 84.2).
- **The relay works over two steps.** Cells B receive the mismatch from the answering cell over learned return connections, change, and pass on what they received, times their own sensitivity, to cells A over a second set of learned return connections. 128 + 128 cells reach what 256 cells reach in one step (84.4 against 84.2). The roadmap's step 3 ("the correction is relayed inward") is met at this size. The tens digit, which I had expected a second step to help, is not clearly better: 89, 91, 92, 89, 93 against 91, 87, 90, 90, 91 over the seeds.
- Predictions: A1 failed (42.7, where I had said at least 60); A2 and A4 failed on the development seed and were not run on the reported seeds; A3 held (70.7); B1 held (84.4); B2 held on three seeds of five by two to four points and failed on two by one to two, which is no difference I would claim.

## 11. Fourth plan, part C: keeping, in small parts

Addition for 400,000 lines, then subtraction for 200,000 lines, then addition again for 40,000 lines; never-shown pairs of each task; five reported seeds (`report4.txt`).

| | 73129 | 93683 | 116089 | 406364 | 545353 | median |
|---|---|---|---|---|---|---|
| addition after its own phase | 84.2 | 82.7 | 86.4 | 81.8 | 86.7 | 84.2 |
| addition 2,000 lines into subtraction | 18.2 | 18.4 | 8.6 | 12.4 | 20.4 | 18.2 |
| addition after subtraction | 2.0 | 1.6 | 1.8 | 1.4 | 2.0 | **1.8** |
| subtraction after its phase | 82.0 | 89.8 | 91.4 | 86.0 | 86.0 | 86.0 |
| addition, with the answering cell's strengths put back as addition left them | 44.6 | 56.8 | 36.4 | 51.4 | 40.8 | **44.6** |
| addition, with the cells in between put back instead | 13.2 | 3.8 | 2.6 | 2.6 | 2.6 | **2.6** |
| lines of addition needed to reach 70%: at first | 60,000 | 60,000 | 60,000 | 70,000 | 70,000 | |
| ... when taught again after subtraction | 4,000 | 4,000 | 6,000 | 4,000 | 6,000 | |
| addition 2,000 lines into being taught again | 55.0 | 60.0 | 44.2 | 58.4 | 42.0 | 55.0 |
| subtraction after addition is taught again | 2.8 | 3.0 | 1.6 | 2.6 | 3.0 | 2.8 |
| one addition line in 32 during subtraction: addition at the end | 20.8 | 12.2 | 15.6 | 17.2 | 9.8 | **15.6** |
| ... subtraction at the end | 85.6 | 85.8 | 90.4 | 84.2 | 85.8 | 85.8 |
| 128 new cells for subtraction, nothing of addition rehearsed: addition at the end | 5.0 | 3.0 | 3.2 | 4.8 | 4.4 | **4.4** |
| 128 new cells and one addition line in 32: addition at the end | 75.2 | 62.8 | 67.0 | 64.4 | 67.8 | **67.0** |
| ... subtraction at the end | 78.6 | 72.8 | 75.4 | 73.4 | 77.6 | 75.4 |

Read with section 9 (one addition line in 8 kept 71%).

1. **The loss is fast and total, and the fading is not its cause.** After 2,000 lines of subtraction addition is at 18%; after 200,000 at 1.8%. On the development seed the same happens with the fading of the answering cell's strengths switched off for the second phase (17% after 2,000 lines, 1.4% at the end).
2. **It is written in both places.** With the answering cell's strengths put back as addition left them, and the cells in between left as subtraction made them, addition is at 44.6%: a little over half of the loss is undone (53% at the median, 41 to 69 over the seeds). With the cells in between put back instead, 2.6%. Over the second phase the answering cell's strengths change by 1.1 times their size and the cells' strengths by 0.4 of theirs.
3. **What was learned comes back ten to seventeen times faster than it was learned** (70% after 4,000 to 6,000 lines against 60,000 to 70,000, at the spacing of the readings), and subtraction is then lost in its turn (2.8%).
4. **Less rehearsal than one line in 8 does not keep it**: one line in 32 keeps 15.6%.
5. **Where the second task is written decides how much rehearsal is needed.** In the control `reserve` the program gives subtraction 128 new cells and leaves untouched everything addition wrote (the old cells, the answering cell's strengths from them, its constants). With nothing rehearsed addition still falls to 4.4%: nothing of it was overwritten, but the new cells also speak on addition lines, where nothing has taught them to be silent. With one addition line in 32, addition is kept at 67.0% and subtraction learned to 75.4%, where the same rehearsal without new cells keeps 15.6%. So the loss has two parts: overwriting, which goes when the new task is written elsewhere, and the new cells' voice on old lines, which a little of the old task removes.
6. Two further controls, development seed only (`res/c2_d_a2*.json`): with two answering cells, the program choosing by the operator which one answers and learns, addition still falls to 16% because the cells in between move; if they are not allowed to move, addition stays at 83% and subtraction reaches only 30 to 37%; a cell in between that stiffens with its own use gives 14 to 17% and 86 to 91%.
7. Predictions: C1 held as a share of the loss (53%) and failed as a score (44.6% where "at least half" could be read as 50%); the wording was mine and was loose. C2 failed (15.6). C3 held (a ratio of 0.06 to 0.10 on every seed, at the spacing of the readings).

What is still open in R5 is unchanged in kind and narrower in extent: a way for the cells themselves, and not the program, to settle where a new task is written; and rehearsal that the system makes itself.

## 12. Fifth, sixth, seventh and ninth plans: cells that hold bytes themselves (roadmap R3)

`PLAN_5.md`, `PLAN_6.md`, `PLAN_7.md`, `PLAN_9.md` (each committed before its runs; development outcomes entered before the reported seeds), `hold.py`, the instrument `hold_x.py`, `look_hold.py`, `res/hold_*`.

**The system.** In sections 2 to 11 a chain of nine delay cells, given by the program, kept the last bytes present on lines. Here it is removed. 256 cells; each reads the lines of the present byte and 16 other cells, drawn at birth, one byte late. Each has a gate and a candidate, both logistic, and its state is gate times its state before plus (1 - gate) times the candidate: with the gate near 1 it holds, near 0 it takes in. (The gated cell in its smallest form; that the gated cell may stay is recorded in `WHAT_A_NEURON_HAS`, section 7, item 1, among Mr. Xu's corrections.) The answering cell, the mismatch and its learned return connections are those of section 2. Every strength keeps a trace that runs forward: how that strength has shaped its own cell's present state. A strength changes by what is relayed to its cell times its trace. Nothing passes backward over the line.

**The crossing** (`cross=1`). A cell that reads another passes back to it what it received from the answering cell, times its own sensitivity to that connection; the cell that is read changes by what its readers send times each of its traces as it stood one byte earlier. One crossing, at the last step, still forward in time. Section 12.4 says what this assumes.

### 12.1 Nothing to bridge: single-digit sums (fifth plan)

Sums shown, whole answer right, %, 240,000 lines, five reported seeds: cells that learn **98.3** (97.9, 98.3, 98.8, 97.9, 99.2); cells that never change 2.1 (1.2, 1.2, 3.8, 2.5, 2.1). No crossing. The cells come to hold the first number for the three bytes it is needed. (Sums never shown are not answered, as by every learner on this text: section 3.) H1 held.

### 12.2 One digit held across a gap of letters (seventh plan)

Lines `a+<letters>b=c`: between the operator and the second number, 0 to 8 letters (four letter values, drawn afresh for every line), where the first family has its letters. Text `g1`: c = a + b. Text `m1`: c = (a + b) mod 10, so that the first number is of use only together with the second. Whole answer right on the 80 sums shown, at gaps of 0, 4 and 8 letters; 960,000 lines.

| Sums shown, % (mean over the three gaps, last three readings) | 73129 | 93683 | 116089 | 406364 | 545353 | median |
|---|---|---|---|---|---|---|
| `m1` (c = (a + b) mod 10): no crossing, eta1 = 1 | 27.6 | 21.2 | 28.2 | 40.1 | 18.3 | **27.6** |
| `m1` (c = (a + b) mod 10): no crossing, eta1 = 4 | 25.7 | 40.0 | 31.2 | 16.4 | 24.9 | **25.7** |
| `m1` (c = (a + b) mod 10): **one crossing, the reader's own strengths** | 91.2 | 81.9 | 100.0 | 78.1 | 100.0 | **91.2** |
| `m1` (c = (a + b) mod 10): one crossing, return strengths grown from nothing | 92.4 | 42.2 | 100.0 | 31.8 | 93.5 | **92.4** |
| `g1` (c = a + b): no crossing, eta1 = 1 | 39.2 | 45.0 | 45.7 | 44.4 | 33.1 | **44.4** |
| `g1` (c = a + b): no crossing, eta1 = 4 | 45.0 | 43.3 | 69.2 | 33.5 | 59.9 | **45.0** |
| `g1` (c = a + b): **one crossing, the reader's own strengths** | 99.9 | 94.2 | 99.0 | 98.6 | 97.4 | **98.6** |
| `g1` (c = a + b): one crossing, return strengths grown from nothing | 95.6 | 98.5 | 99.0 | 88.9 | 96.4 | **96.4** |

At the last reading, by gap (0, 4, 8 letters), median of the five seeds: `m1` with the crossing 98, 98, 96; `m1` without (eta1 = 1) 22, 24, 25; `g1` with the crossing 99, 100, 100; `g1` without (eta1 = 1) 45, 46, 42.

- **With one crossing the cells learn to hold a digit across the letters, and the answer is as good after eight letters as after none.** Five reported seeds, the reader's own strengths: `m1` 91.2% at the median (78 to 100 over the seeds), `g1` 98.6% (94 to 100); at the end of the runs the three gaps differ by one or two points at the median. Return strengths grown from nothing do as well on `g1` (96.4) and are unreliable on `m1` (92, 42, 100, 32, 93).
- **Without the crossing a part is learned and no more, at either step.** `m1`: 27.6% with eta1 = 1 and 25.7% with eta1 = 4; `g1`: 44.4% and 45.0% (medians of five seeds). At the better step of each, the crossing adds 64 points on `m1` and 54 on `g1`. On the development seed the arm without the crossing had 11 to 18% at 240,000 lines, the plan's first length, which I first took for nothing learned; with four times the lines it learns a part, at either step.
- **What has formed** (`look_hold.py`, `look_hold_output.txt`). At birth no cell's state before the second number is mostly accounted for by the first number, and nine cells of 256 keep their gate above 0.95 on letters; yet a readout fitted to the cells' states names the first number after eight letters 79% (`m1`) and 86% (`g1`) of the time, so the digit is not lost at birth: it is in no form the answering cell can use. After learning with the crossing about 150 cells keep their gate shut on letters, 17 to 22 cells carry the first number, and the readout is right 97 to 100% of the time (medians of the five seeds). **The holding carries to gaps never taught**: taught 0 to 8 letters, these networks answer at 10, 12 and 16 letters 96, 95 and 94% of the time on `m1` and 100% on `g1` (at 0, 4 and 8 letters: 98, 98, 96 and 99, 100, 100). Without the crossing as many gates shut (about 155) and the readout still finds the first number 90% of the time, but the answers are at 22 to 25% and 42 to 46%: the number is there to be read and is not in a form from which the cells compute the answer (the last clause is my reading).
- **Which return strength is needed** (development seed, `m1`, 960,000 lines unless said): the reader's own strength (`exact`) 98.1; a separate strength that starts at nothing and receives the forward strength's changes (`grown`) 99.6; the form that needs only a common signal (`local2`, section 12.4) 51.4 on `m1` and 77.8 on `g1`; a return strength built from the reader's signal at the moment of the mismatch alone (`local`) 10.6 on `m1` and 53.6 on `g1`: a reader that has taken in the second number holds it for one byte before the answer is named, and at that moment its sensitivity is near zero; the sign of the strength alone (`sign`) 11.7 at 240,000 lines; a return strength made to agree with the forward one by fading both (`learned`, the construction of section 2) 12.1, because the fading of the strengths between cells by itself stops the learning (12.4 with exact return strengths and the same fading).
- **Beside a pass backward over the line** (the instrument `hold_x.py`, which is not something a cell has; development seed, `m1`, 240,000 lines, the same step): one crossing 36 to 37% (the forward program: 49%), two crossings 87%, unlimited 79%. More crossings learn faster; with one, that seed went on to 98 to 100% by 960,000 lines. At the three smaller steps I tried first the instrument learned nothing in that time, which I had misread as a failure of back-propagation.
- Predictions of the seventh plan: M1 held, M2 held, M3 failed narrowly (below), M4 failed at the steps named and is met at the forward program's step, M5 failed, M6 failed.

**A reading of mine, half borne out.** After the fifth plan I wrote that a cell's own path teaches holding where the byte held is useful to the answering cell by itself, and that the crossing is needed only where it is useful together with another byte. On the development seed, at the plan's first length, that failed outright: without the crossing both texts were at 12 to 18%. On the reported seeds, at four times the length, the plain sum is at 44% without the crossing and the sum mod 10 at 28%: 17 points apart at the gap of 8 letters, where I had said at least 20 (M3). So where the byte held is useful by itself a cell's own path teaches some holding, and far from enough; the agreement task of section 14 is a case where it is enough. I cannot yet say beforehand which case a text will be.

### 12.3 Two-digit sums without the chain of nine (sixth and ninth plans)

Never-shown sums, development seed, 2,400,000 lines unless said:

| Lines a cell reads | no crossing | one crossing |
|---|---|---|
| the present byte only | 1.1 (by place 79, 20, 9) | 5.0 (93, 38, 10) |
| the present byte and the one before | not run | 63.6, still rising: 71 at the last reading (99, 78, 93) |
| the present byte and the two before | 35.0 (94, 46, 94) | **72.4** (98, 75, 100); `grown` 64.8; `local2` 51.2 |
| the full chain of nine, no cells that hold (sections 4 and 10) | | 84.2 at 400,000 lines (five seeds) |

Five reported seeds, the present byte and the two before it on lines, 1,600,000 lines (`report4.txt`):

| Never-shown sums, % | 73129 | 93683 | 116089 | 406364 | 545353 | median |
|---|---|---|---|---|---|---|
| no crossing | 16.2 | 17.3 | 27.4 | 23.4 | 29.2 | **23.4** |
| **one crossing, the reader's own strengths** | 65.8 | 66.6 | 67.3 | 60.1 | 62.1 | **65.8** |
| ... the same arm on sums shown | 66.4 | 66.8 | 68.0 | 65.2 | 69.9 | **66.8** |

By place at the end (hundreds/tens/units), the five seeds: without the crossing 96/46/47, 95/34/60, 97/46/71, 97/36/80, 96/31/96; with it 99/66/100, 99/74/98, 98/73/99, 99/72/91, 98/68/98.

- **With the present byte alone the two-digit text is not learned**, with or without the crossing, in 2,400,000 lines; the instrument with unlimited crossings had 22% at 600,000 lines and was rising. The sixth plan's hypothesis (one crossing is enough) was refuted as registered.
- **My reading of why, and the test of it.** For the units digit of the first number a cell must take in the second byte after the end mark, and nothing on its lines says "second"; a cell that tells the second byte from the first would have to be taught by the holding cell that needs it, which is itself taught by a reader: two crossings. If the two bytes before the present one are on lines as well, both digits of a number are before the cell when the byte after it arrives, and one crossing reaches everything. That is the ninth plan, and it held: 72% of never-shown sums on the development seed and 66% at the median of the five reported seeds, against 23% without the crossing, and as often as sums shown (67%); three of the four digits are held by cells for up to five bytes. One delay line gave nearly as much on the development seed.
- **With letters between the two numbers the system does not yet learn** (eleventh plan: lines `aa+<0 to 8 letters>bb=ccc`, two delay lines, development seed, 2,400,000 lines): 3.1% of never-shown sums without the crossing and 7.7% with it, alike at every gap; by place 97, 76, 16. Its hypothesis was refuted as registered. The tens digit is right three times in four and the units digit hardly above a guess: the carry, which needs only a rough hold of the first number's units digit, is carried across the letters, and the exact digit is not. One digit was held exactly across the same gaps in 12.2, and two digits for five bytes above; two digits held exactly and kept apart across a gap is what has not been learned in this time. This is where the way to the first family stands.
- So most of the chain of nine can go, at a cost: 66% where the full chain gives 84%. What is kept of it is very little, the byte or two just past, which a cell could have from the delay of its own connections; cells that hold do the rest.
- Predictions: eleventh plan T1 failed, T3 held, T2 held only in the empty sense that every gap is equally low. Sixth plan K1, K2, K3 and K5 failed; K4 and K6 were **not run** (the instrument is numpy only and was not lengthened). Ninth plan: N1 held; N2 failed (two delay lines without the crossing already give 35% on the development seed, and 23% at the median of the reported seeds, where I had said at most 20); N3 held; N4 failed (one delay line is 9 points below, not 20); N5 held.

### 12.4 What the crossing assumes

In sections 2 to 11 the answering cell's mismatch reaches every cell alike, and each cell weighs it with return strengths it holds itself. The crossing between cells, in the form that was run on the reported seeds (`exact`), is different in one respect: what a reader passes back is its own signal times its own sensitivity times its own strength from the cell it reads. No strength of another cell is used, but **the value differs from one cell it reads to the next**.

The record bears on this in two ways, and it is not mine to settle between them. `WHAT_A_NEURON_HAS_20261118.md`, section 7, item 2, in the reviewer's wording of the page after Mr. Xu's correction: "What does not exist is an instruction computed separately for each cell and delivered to it, and the use by one cell of another cell's weights; a signal that is the same for many cells is not excluded." Section 16 of the same page, Mr. Xu, in the reviewer's translation: "The strength of a connection between cells is determined jointly by the two ends of the connection; it is a local quantity."

So a third form was built, **`local2`**, in which nothing differs from one cell read to another: every reader sends, the same to all the cells it reads, what it received, its gate and its two sensitivities, at every byte; a cell that is read keeps return strengths and a trace for each of its readers, built from those numbers and its own earlier outputs, and weighs the reader's signal with its own return strengths. On the development seed it learns on every text it was tried on and is the weakest of the forms that learn: the agreement task with a phrase in between 100% (as the reader's own strengths); the plain gap text 77.8% (95.3 with the reader's own strengths, 29.2 with no crossing); the gap text mod 10 51.4% (98.1; 22.6); two-digit sums with two delay lines 51.2% (72.4; 35.0). My prediction for it (at least 90% on the mod-10 text) failed. It was not run on the reported seeds. So holding across a gap does not depend on a value that differs from connection to connection; at this size it is learned faster and further with one.

## 13. Eighth plan: speed

Mr. Xu, 20 November, during the work: "Are you using the slow version or the fast version of the code? Previously fable agents optimized the code to run around 2 minutes. Your code seems rather slow." and "The idea is to utilize sparsity to speed up the code as much as possible, and enable fast feedback loop in our research. So see if this can be done. In general, if you got stuck, figure out the root cause first and act." `PLAN_8.md`.

**I was using the slow kind**: my own numpy programs with a Python loop over every byte. The agents' compiled kernels are for the tissue's per-connection step and do not compute this system, but their method applies as it stands. The causes of the slowness, in the order of their size, and what was done about each:

| Cause | Remedy | Gain |
|---|---|---|
| A Python loop over every byte, some twenty array operations each | the inner loops in C, called through ctypes: `kern.c` and `fastk.py` (`fast=1`; every option of both programs), `kern4.c` (the gated cells with a short chain) | 5.5 times (the plain system); 8 to 10 times (the gated cells) |
| Every strength of the answering cell and every return strength faded at every byte; strengths laid out by cell, so that the ten active lines are scattered | one scale factor in place of the fading, exactly; strengths laid out by line: `kern3.c` (`fast=3`; the plain system) | 2.2 times more |
| Lines drawn and pairs scored one at a time in Python | the lines up to the next reading drawn at once (the same stream, checked); all pairs scored at once | small for the runs here; not measured apart |
| Every cell active at every byte, so that every sum and every change runs over all cells | not removable by rewriting: the cell itself has to be silent most of the time (below) | see below |

**What a run costs now** (CPU seconds on this machine when it runs nothing else; `speed_output.txt`):

| | numpy | compiled |
|---|---|---|
| the plain system (256 logistic cells, every line read), 400,000 lines of two-digit sums (3.6 million bytes) | 188 | 34 (`fast=1`); 16 (`fast=3`) |
| the same with 64 lines a cell and rewiring | 317 to 341 (machine busy) | 76 (`fast=1`) |
| the relay over two steps, 128 + 128 cells | 380 to 534 (machine busy) | 239 (`fast=1`) |
| silent cells, 256; 512 | not run at length | 11; 24 (`fast=2`) |
| gated cells, a gap text, 240,000 lines (2.5 million bytes), with the crossing | 528 | 52 |
| gated cells, two-digit sums, 200,000 lines (2 million bytes), with the crossing | 360 | about 45; 50 with two delay lines |

**The compiled programs compute what the numpy programs compute.** `check_fast.py` runs both on the same seed and lines for every option in use (35 cases) and compares every strength: the largest difference over the largest strength is between 1e-15 and 2e-11 (`check_fast_output.txt`); a wrong term would show as a difference of order one. Twelve full-length runs made with both give the same score to the digit printed (`report4.txt`, last block: for instance 82.5 and 82.5, 7.3 and 7.3). At the larger steps used with the gated cells the two drift apart after some thousands of lines, as two orders of summation will in a system this sensitive, and single readings then differ; I have compared levels there and not readings. The numpy programs remain the reference.

**Sparsity.** What is sparse in the input was used from the start: of the 140 lines 10 are active at a byte, and only their strengths are read and changed. What remained dense had two causes. The fading is removed exactly (above). The other is that a logistic cell is never silent, and that is a property of the cell, not of the program. So the eighth plan tried cells that are silent below a threshold (output = drive cut off at 0 and at 1; each cell moves its threshold slowly so that its mean output stays near a target), with a kernel that works only on the cells that are active (`kern2.c`, `fast=2`; it agrees with the numpy program to 1e-13).

Development seed, 400,000 lines, never-shown sums (`PLAN_8.md`, section 5):

| Cells | active at a byte | never-shown sums | CPU s |
|---|---|---|---|
| 256 logistic | 256 | 81.5 | 34 (`fast=1`), 16 (`fast=3`) |
| 256 silent, target 1/8, no upper cut-off | 25 | 56.4 | 6.7 |
| 256 silent, target 1/4 | 53 | 61.0 | 9.5 |
| 512 silent, target 1/16 | 53 | 66.3 | 12.5 |
| 512 silent, target 1/4 | 140 | 81.6 | 22 |
| 1,024 silent, target 1/8 | 183 | 78.1 | 33 |

On the five reported seeds (cut off at 0 and 1, target 1/4, thresholds moving at 1e-5 a byte; `report4.txt`): 256 silent cells **71.3** (72.2, 71.9, 54.1, 42.0, 71.3; about 50 active at a byte; 9.6 s); 512 silent cells **70.0** (62.8, 70.0, 69.5, 74.2, 83.2; about 140 active; 22.9 s); 256 logistic cells 84.2 (33 s with `fast=1`).

- **Silent cells learn and answer never-shown sums, at about 70% where logistic cells give 84%**, with a fifth to a quarter of the cells active and a wider scatter over the seeds (42 to 83). The active-cell kernel runs 256 of them in 10 to 11 s, where 256 logistic cells take 34 s with the first dense kernel and 16 s with the lazy one: about 3.3 and 1.5 times faster, at 13 points less.
- **A reading of mine from the development seed did not survive the reported seeds.** There the score had seemed to follow the number of active cells (56% with 25 active, 61 to 66% with about 50, 82% with 140), and I wrote in `PLAN_8.md` that 512 silent cells equal 256 logistic ones. On the five reported seeds 256 silent cells with 50 active give 71.3% and 512 with 140 active give 70.0%. I withdraw it.
- The first grid failed for a reason worth recording: with the thresholds moving at 1e-3 a byte, the score rose and then fell away in most settings (for instance 44, 39, 36, 28, 13, 13, 8, 6). The thresholds were moving as fast as the strengths were learning; at 1e-5 the fall is gone.
- Predictions (all on the development seed, as registered): S1 and S2 failed (61.0 at best with 256 cells, where I had said 70; 56.4 at the target 1/8, where I had said 60); S3 held (five times faster than the first dense kernel at the target 1/8); S4 failed (66.3 at its target of 1/16).

**What the feedback loop is now.** A run of the plain system takes 16 to 34 seconds where it took about three minutes, so that five seeds of an arm are back in a minute on four cores; the gap texts and the two-digit text without the full chain, which I could not afford to run to the end before, take three to ten minutes a run. After any change to a numpy program or a kernel, `python3 check_fast.py` says within a minute whether the two still agree.

**What would make it faster still, not done.** The gated cells' kernel still moves every trace at every byte; the traces of the strengths from lines can be kept lazily in the same way as the fading (my estimate: 1.5 times). Two return matrices are kept where one and a constant would do. Single precision would double the throughput of every kernel and change the rounding. Cells that are silent and read few lines would let the drives be computed from the active lines alone; both cost score at this size (this section and section 10).

## 14. Tenth plan: two small language tasks

Mr. Xu, 20 November, during the work: "If possible, can you also try some small scale nlp tasks while waiting? thanks!" `PLAN_10.md` (committed before the runs), `prose.py`, `agree.py`, `res/prose_*.json`, `res/agree_*.json`. One development seed, three for one comparison; nothing here was run on the reported seeds.

### A. The next byte of real prose

The system of the second plan, unchanged (one line a byte value at each of ten delays; logistic cells in between; one answering cell; the mismatch relayed), reads the first nine tenths of a book eight times, one byte at a time, naming each next byte before it arrives; the book is put in lower case with 37 byte values. The score is the next byte named right on the last tenth, which is never read. Two books already in this repository (the NLTK Gutenberg selection used by an earlier MiniFly version): Sara Cone Bryant's *Stories to Tell to Children* (216,437 bytes read, 24,049 never read) and Lewis Carroll's *Alice* (128,206 and 14,246).

| Next byte right on text never read, % | Bryant | Alice |
|---|---|---|
| always the most frequent byte (the space) | 19.0 | 18.2 |
| no cells in between (the answering cell reads the 370 lines) | 35.5 | 37.0 |
| 1,024 cells that never change | 32.7 | not run |
| table of the last 2 bytes | 41.3 | 44.0 |
| **cells taught by the relay, return strengths born as in the earlier plans**: 256 cells; 1,024 cells | 45.8; 46.2 | not run; 51.5 |
| table of the last 3 bytes | 49.5 | 54.4 |
| **the same with return strengths born at nothing (the comparator of the earlier plans)**: 256 cells; 1,024 cells | 48.9; 50.2 | not run; 55.5 |
| table of the last 4 bytes; of the last 5 | 53.5; 54.8 | 58.3; 59.6 |

On Bryant with 256 cells and three seeds: 45.8, 45.9, 46.2 against 48.9, 48.8, 48.7.

- **The loop learns real prose.** Cells taught by the relayed mismatch name 46% (Bryant) and 52% (Alice) of the next bytes of text they never read, against 33 to 37% for cells that never change or for no cells, and 19% for the commonest byte.
- **It does not beat a table.** With ten bytes on its lines the system is level with the table of the last two to three bytes and below the tables of four and five. I had predicted it level with or above the table of three; that failed (L2, L3). My reading, not tested: on a text this small, counting is hard to beat with one layer of 256 to 1,024 cells, and the answering cell here fits its branches by least squares, which estimates rare bytes poorly.
- **The relay is three to four points below its comparator here, and the cause is the birth of the return strengths.** With return strengths born at 0.3, 0.1, 0.03 and 0.01 of their earlier size the relay gives 47.0, 47.9, 49.2 and 48.8 (256 cells), the last two level with the comparator's 48.9. On two-digit sums it is the other way round: born at nothing 51.1, at 0.03 59.5, at 0.1 75.3, at 0.3 75.9, at the earlier size 81.5 (development seed; `res/c2_d_rs*.json`). The size of the return strengths at birth is a number fixed by us that matters, in opposite directions on the two texts; about 0.1 to 0.3 loses a little on each.
- Predictions: L1 failed (4.0 points below, where I had said within 3); L2 and L3 failed; L4 and L5 held. The hypothesis is refuted in its second half (the table of three) and stands at the edge in its first.

**A correction this forces on sections 2 and 6.** I wrote there that the comparator's signal is "back-propagation through this network, which a cell does not have". In a network with one step of relay that is wrong. If the return strengths start at nothing, as the answering cell's own strengths do, and receive the same local product, the two are equal at every byte with nothing copied. The comparator is therefore the local rule itself with one particular birth of the return strengths, and what sections 3 and 4 compare is two births of the same rule. The sentence stands only for the comparator whose answering cell is born at random (section 4, 70.3%), where the return strengths would have to be born equal to forward strengths that are not zero, and for networks with more than one step.

### B. A verb agrees with its subject across words in between

Sentences in bytes, `the <noun>[s] <words in between> <verb>[s].`: ten nouns and ten verbs of three letters, the subject singular or plural at random; a singular subject takes the verb with `s` (`the cat still naps.`), a plural subject the verb without (`the cats still nap.`); 0 to 4 words in between. In the text `ag0` these are adverbs. In `ag1` each is, one time in three, a phrase with a noun and a number of its own (`the cat also by the owls naps.`), which is the classic test of agreement: the nearer noun must not decide. 20 of the 100 pairs of noun and verb are never shown together. The system is the gated cells of section 12 with two delay lines (`chain=2`), 256 cells, 300,000 sentences. Scored: the byte after the verb's third letter (`s` or the full stop) on the never-shown pairs, both numbers; a system that has only learned the form of a sentence is right half the time.

| Development seed, never-shown pairs, % right | adverbs in between: 0, 2, 4 words | a phrase with the subject's number | a phrase with the other number |
|---|---|---|---|
| `ag0`: cells that never change | 48, 48, 50 | | |
| `ag0`: own path only | 100, 100, 100 | | |
| `ag0`: one crossing (`exact`; `grown`) | 100, 100, 100; 100, 100, 100 | | |
| `ag1`: cells that never change | 60, 45, 46 | 66 | 23 |
| `ag1`: own path only | 100, 100, 100 | 100 | **48** |
| `ag1`: one crossing, `exact` | 100, 100, 100 | 100 | **100** |
| `ag1`: one crossing, `grown` | 100, 100, 100 | 100 | 81 |

Sentences whose phrase carries the other number, three seeds (1, 2, 3): own path only 48, 46, 51; one crossing with the reader's own strengths 100, 100, 55; with return strengths grown from nothing 81, 100, 80; with the common-signal form of section 12.4, 100 (seed 1 only). On the third seed the arm with the reader's own strengths wavers between 50 and 82 over its readings and ends at 55.

- **The rule is learned and carried to pairs never shown**, across four words in between as well as across none; on `ag0` from the first reading (50,000 sentences) on.
- **Where nothing has to be protected, the cell's own path is enough.** On `ag0` the arm without the crossing is perfect. The number that is held decides between two branches of the answering cell by itself, so the cell that holds it is paid directly. This is the case I had expected on the plain gap text of section 12 and did not find there; here it shows.
- **Where a later noun of the other number has to be ignored, only the arms with the crossing do it, and not on every seed.** Taught by its own path alone the system is at chance on such sentences on all three seeds (48, 46, 51), at every reading. With one crossing and the reader's own strengths it answers by the subject on two seeds of three (100, 100) and wavers on the third (55). I had expected the crossing arm to make this error throughout (G4), reasoning that telling the subject from a later noun needs a cell taught over two crossings; that was wrong.
- Predictions: G1 and G2 held (100); G3 failed (the arm without the crossing is at 100, where I had said at most 70); G4 failed. The hypothesis is refuted as written (without the crossing the cells do hold the number). What the crossing adds, where it adds anything, is narrower than I wrote: holding against a later noun, on two seeds of three.
- A slip of mine in `agree.py`, with no effect on the scores: the count of bytes named right in the stream, which the program printed, lost what the kernel had added between readings. It is corrected and the printed counts of these runs are not used.

## 15. Where the revised roadmap stands, and what I would do next

The table of `ROADMAP_REVISION_PROPOSED_20261119.md`, section 2, with its states as of 20 November (that file is updated to match).

| | Step | State |
|---|---|---|
| R1 | The loop at the smallest size | reached at development level on two-digit sums (sections 4 and 10); to be repeated by another line, which now takes minutes |
| R2 | The same as a cell has it, one item at a time | sparse reading with rewiring: 81% with 64 lines, 43% with 32; one sign: 71%; the relay over two steps: 84%; silent cells: about 70%; all against 84%. Yes or no alone: 9% (section 6), not reached. Several answering cells and feedback through dedicated cells: not started |
| R3 | Cells that hold bytes themselves | reached in two of three parts. One digit across a gap of varying length: 91% and 99% with one crossing between cells (five seeds), and as good at gaps never taught. Two-digit sums: 66% of never-shown sums with two delay lines in place of nine and the crossing (five seeds); not learned with none. The two together (two-digit numbers with letters between them): 8% on the development seed, not reached |
| R4 | The first family | not started. Beyond R3 it needs two-digit numbers with letters between them (tried once, development seed: 8%, not reached; section 12.3), the task byte, and the reduction by the modulus |
| R5 | Keeping, in parts | (a) what is lost and how fast it returns, (b) how little rehearsal: measured on five seeds; (c) rehearsal the system makes itself: two forms failed (section 9); (d) writing the new task elsewhere: works when the program chooses where, with one old line in 32; a way for the cells to choose is open |
| R6 | A life; cost | a run costs a fifth to a twelfth of what it did; cost beside the comparator is not measured |

What I would do next, in this order.

1. Have another line repeat sections 4, 12.2 and 12.3 from the programs here. With the kernels that is an hour.
2. Put to Mr. Xu and Mr. Zhou which form of the crossing is allowed (section 12.4). A form that needs only a signal common to all the cells a cell reads is built (`local2`); it learns on every text it was tried on and is weaker. If only that form is allowed, it is the one to improve.
3. From R3 towards R4: two-digit numbers with letters between them are the next obstacle (8% in 2,400,000 lines; the carry is held and the exact digit is not). I would try, in this order and one at a time: longer runs, since a single digit took 400,000 to 900,000 lines to be held; the reader's signal passed over two crossings; more cells. Only after that the first family's own items beside the record's reference.
4. R5 (d): let the cells settle where a new task is written. Two candidates I would test first, neither of them run: cells that stiffen with age while new cells keep joining; cells that stay silent on lines they were not taught.
5. On prose: an answering cell that estimates rare bytes better than a least-squares fit does; and more than one step between the lines and the answer, which section 10 shows the relay can carry.

## 16. Commands, cost, and what was not run (20 November)

    python3 fastk.py; python3 fastk2.py; python3 fastk3.py; python3 fastk4.py        # build the kernels (gcc)
    python3 check_fast.py                                                            # every option, numpy against compiled
    python3 cells2.py <label> seed=<seed> [fan=32|64 rewire=4096] [dale=1] [H=128 H2=128] fast=1          # section 10
    python3 cells2.py <label> seed=<seed> plan=add:400000,sub:200000,add:40000 swap=1 fine=1 fast=1      # section 11
    python3 cells2.py <label> seed=<seed> plan=add:400000,sub:200000 [reh=32] [reserve=128] fast=1
    python3 cells2.py <label> seed=<seed> clip=1/4 eta1=1/64 kap=1e-5 [H=512] fast=2                    # section 13
    python3 hold.py <label> text=d1 seed=<seed> eta1=1 lines=240000 every=40000 fast=1                  # section 12.1
    python3 hold.py <label> text=m1|g1 gap=8 seed=<seed> eta1=1 [cross=1 ret=exact|grown|local2] lines=960000 every=80000 fast=1 save=1
    python3 hold.py <label> text=d2 seed=<seed> eta1=1 chain=2|1 [cross=1 ret=exact] lines=1600000 every=200000 fast=1
    python3 hold.py <label> text=g2 gap=8 seed=1 eta1=1 chain=2 [cross=1 ret=exact] lines=2400000 every=200000 fast=1   # eleventh plan
    python3 hold_x.py <label> text=m1 gap=8 seed=1 eta1=1 X=0|1L|1|2|inf lines=240000 every=40000       # the instrument (numpy)
    python3 look_hold.py m1|g1 res/hold_<label>.npz ...
    python3 prose.py tables; python3 prose.py <label> eta1=1 H=1024 [arm=comparator|fixed|none] [rscale=0.03]
    python3 agree.py <label> text=ag0|ag1 [cross=1 ret=exact|grown|local2]
    python3 report4.py > report4.txt

`fast=1` runs every option compiled; `fast=3` is the plain system with the fading as one scale factor; `fast=2` is for silent cells; without `fast` the numpy program runs, which is the reference.

**Sizes.** The delay-chain system as in section 7, with 140 lines here (a fourteenth byte value). The gated cells: 256 cells, 16 other cells read by each; strengths that learn: 7,168 from the lines of the present byte (21,504 with two delay lines), 8,192 between cells, 512 thresholds, about 3,600 at the answering cell; one trace for each of a cell's strengths, and one earlier value of it with the crossing.

**Cost.** Everything of 20 November took about 20 CPU-hours on four cores: 76 recorded runs in numpy (8.7 CPU-hours) and 241 recorded runs compiled (9.2 CPU-hours), and about two hours of runs that were stopped or made as checks.

**NOT_RUN.** The instrument at X = 0 and X = 1 on two-digit sums (the sixth plan's K4 and K6). 16 lines a cell, one sign with 32 lines, and two steps of 256 + 256 on the reported seeds. One delay line (`chain=1`), the common-signal crossing (`local2`) and two-digit numbers with letters between them on the reported seeds; the last of these with more lines, more cells or two crossings. The first family. Subtraction or keeping with the gated cells. Silent cells together with sparse reading, rewiring, one sign or two steps. The language tasks on the reported seeds, on other books, or at the level of words. A comparator for the agreement task. Cost beside the comparator. Repetition by anyone else.
