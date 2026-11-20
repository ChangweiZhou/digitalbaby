# nano/unite, first plan: do the clock and the cells that count compose? (20 November 2026)

**By** the session that holds this line of the outside work. **Authorisation.** Mr. Xu, 20 November 2026, in Chinese, in the coordinator's translation as given to this line: "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." and then "Good, keep trying." The task itself (one system out of the pieces) is the coordinating session's. **Source.** Branch `claude/nano-unite-8mmhfm`, made from 91f6221; `nano/ahead2/ahead.py` as of 7765872, `nano/skip/skip.py` as of 7016262, `loop_small_case/cells2.py` as of f3e632c; what was taken from each is named at the head of `unite.py`. **Status.** Development level. No checker gate, seal or outside review is claimed. Written before any run of `unite.py` except its check (`check_output.txt`: the compiled kernel against the plain Python reference, nine cases, all PASSED, agreement to 1e-14).

## 1. The question

Two pieces were shown apart today. (1) `nano/skip`: a trial clock for the delay chain of the relay system, so that the letters of the text `g2` never enter the chain. (2) `nano/sparse`, repeated by `nano/ahead2`: cells that count, keyed on a few delays, read by an answering cell with a learned trust, with look-ahead; no relay system; plain two-digit sums. **Do they compose into one system?**

## 2. The system (exact)

*The cells.* As `ahead.py` with look-ahead (its arithmetic, rewritten for a chain with a clock; not that file's `lag` and `levown=0` options). A cell is keyed on at most four of nine lines and counts the byte that arrives next, the byte after it and the third byte (255 kinds). The answering cell forms, in this order, a guess of the byte three ahead, of the byte two ahead, and its answer; 130 kinds of cells are keyed on at most three lines and the guess three ahead (counting the next byte and the one after), 130 kinds on at most three lines and the guess two ahead (counting the next byte). A trust for each family, kind and level of experience (40 levels), taught by the answering cell's own mismatch; counts are whole numbers and only grow. Constants as `ahead.py`: eta 0.1, e 2, plain trusts born at 0.02, others at nothing.

*The nine lines.* Line 1 is the present byte, always. Lines 2 to 9 are the chain: the last eight bytes that entered it. When a byte arrives, the byte that was present enters the chain or is passed over, according to the entry of its byte value (as `skip.py`). With every entry open this is `ahead.py`'s history of nine bytes.

*The look-ahead is in real bytes*: a cell counts the bytes that arrive one, two and three bytes after its reading, whatever the clock does. (Whether it must instead be in steps of the chain is one of the things to find out; see 5.)

*The clock* (`skip.py`, second form, nothing retuned at the start): 17 entries, one for each byte value. Every 256 bytes each entry is closed with chance logistic(u) kept within 0.01 and 0.99, or open. The answering cell sends out one number at every byte, the same to all entries: the sum over the byte values of its squared mismatch (one for the byte that arrived, less its share; for the first of its three statements, the answer). An entry keeps the mean of that number, in units of its running deviation, over its closed and its open periods (fading by 1/3000 a period) and u = logit(0.2) + 2 t, t the difference of the two means in units of its uncertainty. 68 numbers.

*Size.* 17 byte values (digits, `+`, `=`, end mark, four letters; `-` does not occur in this step and has no line). Cells that can be addressed: 10,524,960 plain and 2 x 7,195,250 keyed on a guess. Every figure will carry the number of cells that have counted and of counts above zero at that moment. On plain sums `nano/ahead2` reports 3.27 million cells and 34.6 million counts; with letters and a chain that changes by chance I expect more, and that is part of the result. This is not a small system and is not called one.

*Text and score.* `g2`: `aa+<0 to 8 letters>bb=ccc`, end mark; 500 pairs scored and never shown, 8,000 shown (the split of `cells2.py`; the 500 are those of `hold.py` and `skip.py`; the stream's draws are not those of `skip.py`). Scored on the 500 never-shown pairs with 0, 4 and 8 letters (the letters fixed by the pair, as `hold.py`): the line is read up to the `=`, then **the system writes the three digits from its own bytes**; whole answer right, %. Beside it the same with the end mark also required. Every entry in its likelier state, no drawing. (`skip.py` scores the relay system with the true answer bytes before each place; the cells are scored the harder way.)

## 3. Runs

Readings every 10,000 lines; 200,000 lines; mean of the last three readings beside the last reading.

A. Development seed 1: `clock=oracle`, `clock=none`, `clock=trial` at `skip.py`'s numbers (alpha 2, p0 0.2, P 256, win 3000). If the trial clock does not close the letters, at most three further settings of (alpha, p0, P, win), reported as tuned on this text and seed; then whatever locates the failure.
B. The five reported seeds 73129, 93683, 116089, 406364, 545353, three arms at equal size and length: the united system (`clock=trial`, the setting from A); `clock=none`; `clock=oracle` (the program passes over the letters: what a clock can at best give).
C. Beside them, not rerun: the relay system with the trial clock, from `nano/skip/res/p2C_evid_s*.json` (same held pairs; 1,200,000 lines, readings each 100,000; at 200,000 lines its five reported seeds read 65.0, 75.2, 57.4, 78.0, 92.2, median 75.2, and 96.7 at the end; scored with true answer bytes).
D. Where the united system and another arm are within ten points at the median: seeds 1001 to 1010 for both.

## 4. Hypothesis, and what refutes it

**H1 (the coordinator's).** The united system has at least 80% of never-shown sums right at every gap (0, 4, 8 letters) within 200,000 lines; median of the five reported seeds, mean of the last three readings. **Refuted if** the median is under 50% at the gap of 8. Between 50 and 80: not established.

If it fails, the result is where: (a) the letters are not closed; (b) they are closed too late for the cells; (c) they are closed in time and the cells still fail (then: the counts made before, or the look-ahead).

## 5. What I do not know, and how each is read from the runs

1. *Is the counting cells' mismatch as good a judge for the clock as the relay system's?* Read from: the reading at which exactly the letters are first closed, each seed; whether any digit, `+` or `=` is closed at the end. My expectation: the cells learn the lines without letters within a few thousand lines, so the evidence for closing a letter should come earlier than in the relay system; but with every entry closed in one period in five at first, the ten digits are all open in only one period in nine, and cells count much that is misaligned.
2. *Must the look-ahead reach ahead in the chain and not in real bytes?* Read from: how often the guesses made at the `=` (three ahead: the units digit; two ahead: the tens digit) are right, at each gap, against `ahead2`'s 100% and 89%. From the `=` on no letter remains, so I expect no clash. Only if these are clearly lower with the clock than in the oracle arm is a form in chain steps written and run.
3. *What does a closed entry do to the counts made before it closed?* Read from: the cells that have counted and are keyed on a now-closed byte value at a chain line (they cannot be reached again: dead weight), and the size of the united system against the oracle arm's.

## 6. Predictions (mine, before the runs)

| | Prediction | Chance |
|---|---|---|
| U1 | `clock=oracle`, median of reported seeds: at least 85% (one figure for the three gaps) | 0.80 |
| U2 | `clock=none`: under 5% at gaps 4 and 8 (median) | 0.85 |
| U3 | `clock=none`: at least 60% at gap 0 (median) | 0.55 |
| U4 | development seed, `skip.py`'s numbers: exactly w x y z closed at 200,000 lines | 0.45 |
| U5 | H1 holds (at least 80% at every gap) | 0.35 |
| U6 | H1 is not refuted (at least 50% at gap 8) | 0.55 |
| U7 | no digit closed at the last reading of any reported-seed run | 0.60 |
| U8 | where the letters are closed, the guess three ahead at the `=` is right on at least 99% at the gap of 8 | 0.70 |
| U9 | the united system holds at least 1.5 times the cells of the oracle arm at 200,000 lines (median) | 0.80 |
| U10 | the united system at 200,000 lines is at or above the relay system with the trial clock at 200,000 lines (75.2) | 0.45 |

## 7. Cost, and what is not run

About 3 gigabytes and (estimated from `ahead2`) 8 to 15 CPU-minutes a run; 18 runs and the development runs: about 4 CPU-hours on 4 cores. **Not run in this step:** subtraction and keeping (step 2, only if this step holds at 60% or more); three-digit numbers (step 3); 26 letters; letters inside the numbers; the relay system again; any gate.

## Outcome of A, the three arms at `skip.py`'s numbers (entered 20 November 2026; development seed 1; `res/dev_*_1.json`)

Never-shown sums, the system's own answer bytes, whole answer right, %, at gaps 0 / 4 / 8.

| Arm | 10,000 lines | 20,000 | 40,000 | 60,000 | 100,000 | 200,000 | last three | cells that have counted; counts above zero (at 200,000) |
|---|---|---|---|---|---|---|---|---|
| `clock=oracle` | 86 | 93 | 94 | 88 | 85 | 76 | 76.8 | 3,203,799; 41.9 million |
| `clock=none` | 51 / 0 / 0 | 57 / 0 / 0 | 71 / 0 / 0 | 78 / 0 / 0 | 85 / 0 / 0 | 88 / 0 / 0 | 86.7 / 0.0 / 0.0 | 7,183,179; 97.7 million |
| `clock=trial` (the united system) | 50 / 49 / 24 | 77 | 89 | 94 | 74 | 59 | 58.1 | 10,262,104; 109.3 million |

(one figure where the three gaps are equal.) The trial clock has w x y closed at 10,000 lines and exactly w x y z from 20,000 lines to the end; no other entry is closed at any reading. **U4 held.**

What this shows already (reading, mine; one seed):
1. **The clock is found, and early**: within 20,000 lines, where the relay system's clock needed 100,000 to 200,000. The counting cells' mismatch is a good judge.
2. **The two pieces compose**: at 40,000 to 60,000 lines the united system answers 89 to 94% of never-shown sums at every gap, level with the oracle arm.
3. **Then the score falls with further reading, in the oracle arm too** (94 at 40,000, 76 at 200,000) and further in the united system (59). The place that falls is the tens digit (75.6 and 58.8 at the end; hundreds 96 and 95; units 100). `nano/sparse` reports this fall for its cells of four delays on plain sums (80 at 40,000 lines, 59 to 68 at 400,000; its `REPORT.md`, section 6) and locates it in the growing weight of a cell's counts; there a cell speaking at the weight of at most 16 counts held its level. That line did not run its look-ahead with the cap, and says so.
4. So on this seed H1, read at 200,000 lines as this plan fixed it, is not met (58.1), and is not refuted (above 50); and the reason is not the composition but the known fall of piece (2), made worse by something in the united system that the oracle arm does not have.
5. `clock=none` does better at the gap of 0 than I expected at the end (88) and nothing at 4 and 8, as expected. It holds 7.2 million cells.
6. The united system holds 3.2 times the oracle arm's cells; 3,542,373 of them are keyed on a letter at a chain line and can never be reached again.

## Addition 1 (20 November 2026, written before its runs): where the fall comes from

B goes on as planned for `clock=trial` and `clock=oracle` (200,000 lines). **Changed for cost, said beforehand:** `clock=none` on the reported seeds is run for 60,000 lines only (its gaps of 4 and 8 read 0.0 at every one of the twenty readings on the development seed); my time allows about ten more long runs.

Two diagnostic arms, development seed 1 first, each one change to the united system:
- `freeze=30000`: after 30,000 lines no entry is tried any more (each stays in its likelier state). If the united system then follows the oracle arm, its further fall comes from the trials that go on (each of the 13 open entries is still closed in one period in a hundred, so in about one period in eight some digit or sign is out of the chain, the lines are misaligned, the answering cell is confidently wrong and the trusts are taught on it). If not, it comes from what was counted in the first 30,000 lines.
- `ncap=16` (`nano/sparse`'s number, not chosen here; my reading of its prose and its line of code: a cell that has counted more than 16 times speaks as if it had counted 16 times, in the same proportions; its level of experience is still taken from its true count), on the oracle arm and on the united system. The check has a case with the cap (`check_output.txt`, ten cases PASSED).

If on the development seed the cap holds the united system's level, a fourth arm on the five reported seeds: the united system with `ncap=16`, and H1 is read for it as a separate statement (**H1c**: at least 80% at every gap at 200,000 lines with the cap; refuted under 50% at the gap of 8). The uncapped system's H1 is scored as written whatever the capped one does.

| | Prediction | Chance |
|---|---|---|
| V1 | `freeze=30000`, development seed: last three within 8 points of the oracle arm (76.8) | 0.50 |
| V2 | `ncap=16` on the oracle arm: reading at 200,000 lines within 5 points of its reading at 40,000 | 0.50 |
| V3 | `ncap=16` on the united system: last three at least 80 at every gap | 0.40 |
| V4 | the cap lowers the peak (the reading at 40,000 lines) by at least 3 points in the oracle arm | 0.50 |

## Outcome of addition 1 (entered 20 November 2026; development seed 1; `res/dev_*_1.json`)

Never-shown sums, every gap the same from 20,000 lines on, %:

| Arm | 40,000 lines | 100,000 | 200,000 | last three | tens digit right at the end | guess of the tens digit at the `=` right | cells; counts above zero |
|---|---|---|---|---|---|---|---|
| `clock=oracle` | 93.6 | 85 | 75.6 | 76.8 | 75.6 | 82.2 | 3.20 million; 41.9 million |
| the united system | 93.8 | 74 | 58.8 | 58.1 | 58.8 | 68.0 | 10.26 million; 109.3 million |
| the united system, no entry tried after 30,000 lines | 94.2 | 84.2 | 76.2 | 75.8 | 76.2 | 83.8 | 8.45 million; 84.0 million |
| `clock=oracle`, a cell speaking at the weight of at most 16 counts | 92.4 | 86.0 | 84.0 | 84.5 | 84.0 | 89.0 | 3.12 million; 41.3 million |
| the united system with the same cap | 91.0 | 85.0 | 82.4 | 82.2 | 82.4 | 84.0 | 9.95 million; 105.6 million |

**V1 held** (75.8 against 76.8). **V2 failed** (84.0 at 200,000 lines against 92.4 at 40,000: 8.4 points lower, not within 5). **V3 held** (82.2). **V4 failed** (the cap lowers the oracle arm's reading at 40,000 lines by 1.2 points, not 3).

Reading (mine; one seed):
1. **The fall has two parts and both are located.** (a) The cells' own fall with more reading, which the oracle arm has without any clock (94 to 76) and which `nano/sparse` found on plain sums. (b) A further 18 points that come from the clock's trials going on after the letters are found: with no entry tried after 30,000 lines the united system follows the oracle arm to the point (75.8 against 76.8), although it still holds the 5 million cells counted while the chain was changing. So the counts made before the entries closed do not hurt; what hurts is that each of the 13 open entries is still closed in one period in a hundred, for as long as the system lives.
2. **The cap of `nano/sparse` carries over to the look-ahead system and to the united one, in part**: 84 and 82 at 200,000 lines where there were 76 and 58, but both still 8 to 9 points under their reading at 40,000 lines. With the cap the trials cost little (82.2 against 84.5).
3. The cap costs twice the CPU in my kernel (no table of logarithms on that path); for that reason, and for time, **the fourth arm (the united system with the cap on the five reported seeds) is not run**, though V3 held and this plan said it would be. H1c is therefore neither tested nor claimed.

## Outcome of B and C (entered 20 November 2026; five reported seeds 73129, 93683, 116089, 406364, 545353 in that order; `res/trial_*.json`, `res/oracle_*.json`, `res/none_*.json`)

Never-shown sums `aa+<letters>bb=ccc`, the system writing its own three answer digits, whole answer right, %. One figure where the gaps of 0, 4 and 8 letters are equal (they are from the reading at which the four letters are closed).

| Arm | 20,000 lines (median) | 40,000: each seed; median | 60,000 (median) | 100,000 (median) | 200,000: each seed; median | last three readings: each seed; **median** | size at 200,000 (median): cells that have counted; counts above zero |
|---|---|---|---|---|---|---|---|
| **the united system** (`clock=trial`) | 77.2 / 76.2 / 76.2 | 93.8, 93.6, 94.0, 91.6, 96.2; 93.8 | 92.2 | 78.4 | 58.0, 53.4, 50.6, 49.0, 52.6; 52.6 | 58.3, 54.5, 52.7, 50.3, 53.7; **53.7** | 10,219,049; 109.6 million (3,584,797 of the cells keyed on a letter at a chain line) |
| letters passed over by the program (`clock=oracle`) | 92.6 | 89.4, 93.4, 92.0, 93.6, 90.6; 92.0 | 86.8 | 85.2 | 78.6, 72.2, 72.2, 74.6, 73.8; 73.8 | 79.3, 73.9, 72.5, 76.5, 75.1; **75.1** | 3,205,792; 41.9 million |
| no clock (`clock=none`), gaps 0 / 4 / 8 | see below | see below | see below | development seed: 85 / 0 / 0 | development seed: 88 / 0 / 0 | development seed: **86.7 / 0.0 / 0.0** | development seed: 7,183,179; 97.7 million |
| the relay system with the trial clock (`nano/skip`, its files; true answer bytes before each place; 55,570 numbers that learn) | not read | not read | not read | 1.6, 3.0, 1.6, 2.0, 6.2; 2.0 (gap 0) | 65.0, 75.2, 57.4, 78.0, 92.2; 75.2 | at 1,200,000 lines: **96.7** | 55,642 numbers |

The clock in the united system: exactly w x y z closed from 20,000, 30,000, 30,000, 20,000, 20,000 lines to the end; no other entry closed at any reading of any seed. The guess of the units digit made at the `=` (three bytes ahead) is right on 100% at the gap of 8 on every seed at 40,000 and at 200,000 lines; the guess of the tens digit (two ahead) on 89.8 to 94.4% at 40,000 lines in the united system (89.2 to 93.4 in the oracle arm) and on 62.2 to 66.8% at 200,000 (77.0 to 80.8). By place at the end (hundreds, tens, units; median): 96.6, 52.6, 100 and 98.2, 73.8, 100. With the end mark also required the figures are the same.

**H1 is not met**: 53.7 at every gap at 200,000 lines, where 80 was asked. **It is not refuted by its letter** (the median is above 50 at the gap of 8), narrowly: one seed is at 50.3, and the readings are still falling. I score it as written and do not move the reading point after the fact. What is also true, and was not the hypothesis as I fixed it: at 40,000 lines the united system is at 93.8 (91.6 to 96.2) at every gap, and at 60,000 at 92.2.

**The three things I did not know** (section 5):
1. *The judge.* The counting cells' mismatch is a good judge for the clock, and a faster one than the relay system's: the letters are closed within 20,000 to 30,000 lines on all six seeds, against 100,000 to 200,000, and nothing else is closed. The two speeds fit in the direction that helps: the cells learn the aligned lines quickly, so closing a letter pays quickly.
2. *The look-ahead.* In real bytes it does not clash with the clock: the guess three ahead at the `=` is right on 100%, the guess two ahead as often as in the oracle arm while both are at their best. No form in chain steps was written.
3. *The counts made before the entries closed.* They do not hurt the answers (addition 1: with no entry tried after 30,000 lines the united system follows the oracle arm), but they are 3.2 times the oracle arm's cells, and a third of the united system's cells are keyed on a letter at a chain line and can never be reached again. What does hurt is the trials that go on.

**Predictions.** U1 failed (75.1 at the reading the plan fixed; 92.0 at 40,000 lines). U2: see below. U3: see below. U4 held. U5 failed. U6 held, narrowly (53.7). U7 held. U8 held (100). U9 held (3.2 times). U10 failed (52.6 against 75.2 at 200,000 lines; at 40,000 lines the relay system has not yet started and the united system is at 93.8).

**Where the result stands (reading, mine).** The two pieces compose: one system, with no relay, finds the letters by a rule that names no byte and answers 94% of never-shown sums at every gap after 40,000 lines, a fifth of the lines at which the relay system with the same clock reaches 75%. It does not hold that level: by 200,000 lines it is at 54% and falling. The fall is located (addition 1, development seed): about half is the counting cells' own fall with more reading, which the oracle arm shows without any clock; the other half is the clock's trials going on for life. The relay system with the same clock goes on rising to 96.7% at 1,200,000 lines with 55,642 numbers; the united system holds 10 million cells and 110 million counts at 200,000 lines. Size and the lack of a stable level are the two things this composition does not have.
