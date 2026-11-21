# A cell leaves for what it has not met the room that cells of its kind and age have needed (roadmap second revision, section 15; the last number set per text in the answering path) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。" (my translation: "Given the bitter lesson, holding on to the task of addition may not be the cleverest thing to do; go on trying general ideas."). **Source.** `claude/friendly-darwin-8mmhfm` at 65e98bd09; `gate.py` with the option `room=1`, checked against numpy (`check23_output.txt`); nothing else was run with it. Written before any run. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

1. `PLAN_22.md`: one rule now sets the step of every trust on every text; what is left per text is the smoothing of a cell's counts, a number added to every count (0.02 on prose, 2 on sums; sums with prose's value lose 8 points).
2. That number says how much a cell allows for a follower it has not met. `PLAN_18.md` showed what hangs on it: in a stream that holds nothing new a cell is right to allow nothing, and in a stream that goes on bringing new things it is wrong. So the right allowance is not a property of a text chosen by me; it is something each kind of cell can measure: **how often has a cell of my kind and my age been followed by something it had not met?**
3. General-purpose compressors of the kind that back off through contexts gained much when they stopped fixing that allowance and estimated it from the contexts' own record.

## 2. The one change

The answering cells already sort a speaking cell by its kind and its level (ten levels of how often it has counted, four of how many different followers it has met). For each kind and level the system counts two things as it reads: how many times a cell of that kind and level spoke, and at how many of those the byte that came was one that cell had never counted. A cell's message then gives to all the followers it has not met, together, the share (new + 1) / (spoke + 2), spread evenly over them, and to those it has met the rest in proportion to their counts. No number is added to any count. Nothing else changes: the answering cells and their rule of the step are those of `PLAN_22.md`, addition 1 (the sum starting at one), with the number 0.03, on both kinds of text.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The last number per text can go: with the learned room the first cell is within 0.003 nats of the fixed smoothing on each of six books, and on sums in an open stream (one line in a hundred new from line 40,000 on; `PLAN_18.md`) the second cell ends 400,000 lines at 85 or more on both seeds. *Refuted if* any book is more than 0.010 worse, or the second cell ends under 75 in the open stream on either seed.

| | Statement | Chance |
|---|---|---|
| V1 | prose, first cell: within 0.003 of the fixed smoothing on each of six books | 0.50 |
| V2 | prose, first cell: at least 0.005 better on four of six books | 0.30 |
| V3 | sums, closed stream, 40,000 lines, first cell: the median of five seeds within 2 points of the fixed smoothing's (82.2) | 0.35 |
| V4 | sums, open stream, 400,000 lines, second cell: 85 or more at the end on both seeds | 0.40 |
| V5 | on prose the learned room is larger for cells that have counted once than for cells that have counted 64 times or more, for every kind of run of two bytes or more | 0.85 |
| V6 | in the closed stream of sums at 40,000 lines the learned room of the long-counted cells is under one in a hundred: the system measures that its stream has stopped bringing new things | 0.70 |

On sums in the closed stream I expect a loss (V3 at 0.35): there the rule will believe the stream, as it should.

## 4. Runs, cost, what is not run

Prose: the two lives of `PLAN_15.md` (runs of 0 to 6 bytes), `rule=2 alpha=0.03 room=1`, against the same without `room` (run for `PLAN_22.md`). Sums without classes: five reported seeds at 40,000 lines in the closed stream; seeds 73129 and 93683 through 400,000 lines of the open stream, with and without `room`. Thirty CPU-seconds a short run, about five minutes a long one. Not run: classes; the larger prose systems; streams of other kinds (with `nano/suite`).

## 5. Outcome (entered 20 November after all runs; `res/gate_*room1*.json`, `res/open_alpha0.03_*.json`, `res/room_look_*.json`, the printed lines in `out23/`)

The compiled loop with the learned room equals numpy to 4e-14 (`check23_output.txt`; a single-precision slip in my numpy reference, not in the compiled loop, was found by that check and corrected before any run).

**Prose** (runs of 0 to 6 bytes; the rule of `PLAN_22.md`, addition 1, with 0.03; each book's last tenth right after its stream; the fixed smoothing 0.02, then the learned room).

| | bryant | burgess | carroll | chesterton-ball | austen-emma | melville | the test text's tenth after its stream |
|---|---|---|---|---|---|---|---|
| first cell | 1.5023, 1.5051 | 1.3617, 1.3641 | 1.4447, 1.4488 | 1.6330, 1.6351 | 1.2402, 1.2419 | 1.4683, 1.4732 | 1.5256, 1.5310 |
| second cell | 1.4726, 1.4752 | 1.3191, 1.3227 | 1.3979, 1.4006 | 1.6106, 1.6126 | 1.2082, 1.2094 | 1.4351, 1.4388 | 1.4983, 1.5021 |

**Sums without classes**, never-shown additions after a read line; the fixed smoothing 2, then the learned room.

| | First cell | Second cell |
|---|---|---|
| closed stream, 40,000 lines, five reported seeds, last reading | 83.2, 82.6, 82.2, 80.0, 81.0 (median 82.2) then 77.0, 78.9, 77.9, 74.2, 76.2 (77.0) | 88.6, 85.6, 86.2, 83.8, 87.2 (86.2) then 86.5, 84.6, 84.2, 80.6, 85.8 (84.6) |
| open stream, 400,000 lines, seeds 73129 and 93683: at 40,000 lines, then the mean of the last three readings | 83.2 to 74.2 and 82.6 to 74.8; then 77.0 to 57.7 and 78.9 to 58.5 | 88.6 to **92.2** and 85.6 to **94.4**; then 86.5 to **90.7** and 84.6 to **93.5** |

**What the cells measured** (`room_look.py`). On prose, at the end of the test text's stream, the share of times a speaking cell was followed by a byte it had never counted: for runs of 2 to 6 bytes 0.44 to 0.47 for a cell that had counted once, 0.23 to 0.27 at 2 to 7 counts, 0.07 to 0.11 at 8 to 63, and 0.004 to 0.021 at 64 or more. On sums in the closed stream at 40,000 lines: 0.46 to 0.75 for cells that had counted once, and 0.0001 to 0.0006 for cells that had counted 64 times or more.

**The hypothesis holds in its second half and misses its first by a little.** The second cell ends the open stream at 90.7 and 93.5 (85 asked; under 75 would refute). On prose the first cell is 0.002 to 0.005 worse than with the fixed smoothing: within 0.003 on four books of six, where all six were asked (more than 0.010 worse on any would refute).

| | Statement | Outcome |
|---|---|---|
| V1 | prose, first cell within 0.003 of the fixed smoothing on each of six books | failed (0.0041 and 0.0049 on two books; 0.0017 to 0.0028 on four) |
| V2 | at least 0.005 better on four of six | failed (it is nowhere better) |
| V3 | sums, closed stream, 40,000 lines, first cell within 2 points | failed (5.2 lower), as I expected at 0.35 |
| V4 | sums, open stream, second cell at 85 or more at the end on both seeds | held (90.7 and 93.5) |
| V5 | on prose the learned room is larger for cells that counted once than for cells that counted 64 times or more, for every run of two bytes or more | held (0.44 to 0.47 against 0.004 to 0.021) |
| V6 | in the closed stream of sums the long-counted cells' learned room is under one in a hundred | held (under one in a thousand) |

Readings (mine).
1. **The number can go, at a small price.** With no number added to any count, each kind and age of cell leaving for the unmet what cells like it have needed, prose is 0.002 to 0.005 nats worse than with the value that was tuned on prose, and sums in the open stream are 1 to 1.5 points under the value tuned on sums. With `PLAN_22.md` that makes an answering path with no number set for a text: one number for the step (0.03 here) and nothing else.
2. **The value I had tuned on prose was an estimate of this measurement.** With 0.02 added to each of 65 counts a cell that has counted once leaves 56% for the unmet, one that has counted four times about 24%, thirty times 4%; the cells measured 45%, 25%, 7 to 11%. That is why the tuned value could hardly be beaten, and why it is right for prose and wrong for a text with fourteen byte values.
3. **In the closed stream the rule believes the stream, as it should**: the long-counted cells measure that nothing new comes any more (less than one time in a thousand) and say so, and the score on never-shown sums at 40,000 lines falls by 5 points for the first cell. In the open stream the same rule ends at 91 and 94.
4. **Unplanned, and the larger finding of these runs: with the rule of `PLAN_22.md` the second cell in the open stream rises with reading** (88.6 to 92.2 and 85.6 to 94.4 with the fixed smoothing; 88.2 on both seeds with the hand-set schedule in `PLAN_18.md`). The first cell alone still falls there (to 74, and to 58 with the learned room). So over a long reading the second answering cell is not an improvement to the first; it is what makes the system stand.
5. Not run: the learned room with classes; a spread over the unmet followers that is not even; the larger prose systems; other kinds of stream (with `nano/suite`).
