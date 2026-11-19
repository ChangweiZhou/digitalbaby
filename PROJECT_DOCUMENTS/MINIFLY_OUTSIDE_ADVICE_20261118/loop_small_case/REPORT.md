# The loop at the smallest size: report (19 November 2026)

**By** the outside session that wrote the audits beside this folder. **Authorisation:** Mr. Xu, 19 November 2026, to this session: "So the current agents are obviously stuck. If you have more thoughts, try to implement them to unblock the current situation". **Status:** a small case, in the sense of the reviewer's status rows 137 to 153. Development level: five seeds, one stream each, not repeated by anyone else, not verified by the coordinator. It is not a version of the adopted core and claims no checker gate.

Everything below is from `cells_loop.py` and `look_inside.py` in this folder (numpy only), from the plans `PLAN.md` and `PLAN_2.md`, which were committed before the runs they govern, and from the result files in `res/`. Commands are in section 7.

## 1. In brief

1. **The loop runs.** A system in which the cell that names the next byte forms its mismatch when the byte arrives, and the cells in between change their own strengths from that mismatch relayed to them over return connections, learns the text it reads. Nothing passes backward over the line; the return connections are learned by a local product and are never copied from the forward strengths.
2. **On single-digit sums nobody answers a sum never shown, back-propagation included.** Five reported seeds, 60,000 lines: sums shown 97.5% under the local relay and 96.2% under the comparator, against 12.5% with no cells in between; sums never shown 0% at the median in every arm that learns (5% where the cells in between do not change). The yardstick that stopped the agents on 19 November ("it holds a table, it does not add") is one that the comparator fails as well. My reading: an unseen pair of single digits is not a new combination of anything taught, and 80 facts do not force an order of the digits.
3. **On two-digit numbers the cells answer sums never shown.** There a pair never shown is a new combination of digits that were shown. Five reported seeds, 400,000 lines, 256 cells in between, nothing about digits, places or carrying laid down: the whole answer is right on **88.0% of sums never shown** (82.5 to 92.7 over the seeds), as often as on sums shown (88.4%). With no cells in between: 0%. With cells in between that do not learn: 0%. With return connections left at their random birth values: 83.9%. The comparator, in the three forms run: 58.6%, 62.6% and 70.3%.
4. **What the agents laid down by hand has formed by learning.** In the saved networks the strengths of most cells from the digit lines lie in the order of the digits; cells weigh the two tens digits equally, so that they add them; and in the cells that read all four digits a tens digit weighs about ten times a units digit (section 5).
5. **What does not work yet:** from yes or no alone the same system has 8.9% of never-shown sums right at this length. And the system is not yet as a cell has it in five named respects (section 6).

## 2. The system, in the words of the plan

One line for each byte value; a chain of delay cells, so that the present byte and the bytes before it are present on lines. H cells in between: each adds its strengths from the active lines, less a threshold, and gives the logistic function of that. One answering cell with a branch for each byte value in use; it reads the cells in between and nothing else, and names the byte of its largest branch. When the next byte arrives, the answering cell has for each branch a mismatch (1 less its output for the byte that arrived, 0 less its output for the others). It changes its own strengths by mismatch times the sending cell's output. Each cell in between receives the mismatches over return connections, summed with those connections' strengths, and changes its strengths from the lines that are active, and its threshold, by that sum times h (1 - h), its own output's sensitivity. A return connection changes by the same product as the forward strength between the same two ends, and both fade at the same slow rate, which brings them to agree (their cosine is 1.00 at the end of every run with fading).

One stream, one byte at a time, no batch, nothing reset, nothing averaged over lines, no pass backward over a line, no quantity taken from another cell's strengths.

The arms: `none` (no cells in between: the answering cell reads the lines); `fixed` (cells in between that never change); `relay-fixed` (return connections at their random birth values); `relay-learned` (the candidate); `comparator` (the signal to a cell in between computed from the answering cell's forward strengths: back-propagation through this network, which a cell does not have); `relay-learned` under yes or no (the reviewer's rule Q in the answering cell: only the branch that named the byte has a mismatch, and on yes the others).

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
