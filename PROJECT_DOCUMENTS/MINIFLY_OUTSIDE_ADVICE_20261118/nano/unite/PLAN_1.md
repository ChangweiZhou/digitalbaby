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
