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
