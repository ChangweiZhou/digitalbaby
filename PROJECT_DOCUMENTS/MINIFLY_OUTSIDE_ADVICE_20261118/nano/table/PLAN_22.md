# One rule for the step of every trust, every trust born at nothing: the same on sums and on prose (roadmap second revision, section 15; the numbers set per text) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。" (my translation: "Given the bitter lesson, holding on to the task of addition may not be the cleverest thing to do; go on trying general ideas."). **Source.** `claude/friendly-darwin-8mmhfm` at c6cef5d55; `gate.py` with one more argument to its kernel (the rule of the step), checked against numpy (`check22_output.txt`: nine cases); nothing else was run. Written before any run. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

1. The programme has carried two settings of the answering cell all day: on sums a step of 0.1 that shrinks as the root of a trust's uses, trusts born at 0.02; on prose a constant step of 0.00005, trusts born at 0.25, the step chosen twice at the edge of its grid (`PLAN_3.md`). The roadmap's table of what transfers lists them as "numbers set per text", and section 15 names them as a debt.
2. Two observations of other lines point the same way: the line `nano/merge` saw its plain system at 80.3 against 76.8 on one seed "when a trust's step shrinks with the mismatch it has met and not with its uses"; the line `nano/layers` saw its widest cell lose on the system with 164 kinds and guessed at steps that overshoot when many parts move at once.
3. A system that is to read any stream cannot have its step chosen for the stream.

## 2. The one change

**Each trust, and each part of the second answering cell, keeps the sum of the squares of the mismatches it has met; its step is one number α divided by the root of that sum; every trust is born at nothing.** A trust that has met large or many mismatches moves little, one that has met few moves much, whatever the size of the messages of its text. The rule is each trust's own (it reads nothing but its own history). α is one number for all texts.

The smoothing of a cell's counts stays as each text has it (2 on sums, 0.02 on prose): that is the one number per text left by this plan; one run with 0.02 on sums is made as a side reading.

*Choosing α, on development material only:* sums seed 1 (40,000 lines, without classes, never-shown additions after a read line, first cell) and the last tenth of the test text right after its stream (first cell; that tenth is worn and is used here as development material, not as a claim). α in 0.01, 0.03, 0.1, 0.3. The rule of choice, fixed now: rank the four on each of the two measures, take the smallest sum of ranks, on a tie the smaller α.

*Claims, with α frozen:* sums on the five reported seeds (without classes; with classes as a second reading, since the search for classes ranks kinds by a weight that this rule changes in meaning); prose in the two lives of `PLAN_15.md` (three worn books, three books opened there), each last tenth right after its stream. Both cells. Against the settings per text (`PLAN_15.md`'s runs: sums 80.1 and 85.6 at the median; the books' figures there).

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* One rule with one α does what the two settings did: the first cell is within 1.5 points of 80.1 on sums (median of five seeds) and within 0.005 nats of the per-text setting on each of the six books. *Refuted if* it is more than 4 points lower on sums or more than 0.015 worse on any book.

| | Statement | Chance |
|---|---|---|
| U1 | sums without classes, first cell, median of five seeds at least 78.6 | 0.50 |
| U2 | at least 82.1 (two points better than the per-text setting) | 0.25 |
| U3 | prose, first cell: within 0.005 nats on each of six books | 0.40 |
| U4 | better than the per-text setting on at least four of six books | 0.30 |
| U5 | the second cell keeps its gain over the first: at least 3 points on sums and at least 0.015 nats on each book | 0.55 |
| U6 | the α that is best on sums and the α that is best on prose are the same or neighbours in the grid | 0.60 |
| U7 | side reading: sums with the smoothing of prose (0.02) under the rule lose more than 5 points | 0.60 |

## 4. Cost, what is not run

Thirty CPU-seconds a run on sums without classes, eight to eighteen a life. Not run: the rule on this line's larger prose systems (another kernel); streams of other kinds (the line `nano/suite` has this as its last step and is given the result); the rule for the cells' counts.
