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

## 5. Outcome (entered 20 November after all runs; `res/gate_*rule1*.json`, the printed lines in `out22/`)

The compiled rule equals numpy to 5e-14 on two cases; the seven earlier cases are unchanged (`check22_output.txt`).

**Development** (sums seed 1 without classes, never-shown additions after a read line, last reading and mean of the last three; the test text's last tenth right after its stream). The life used for the second measure also printed the three worn books for every α; I chose by the rule fixed in section 2, which reads the two development measures only.

| α | sums, first cell | sums, second cell | prose, first cell | prose, second cell |
|---|---|---|---|---|
| the settings per text | 79.4, 79.3 | 84.7, 85.4 | 1.5226 | 1.5058 |
| 0.01 | 78.4, 77.7 | 82.9, 84.2 | 1.5234 | 1.5080 |
| 0.03 | 80.4, 80.7 | 84.9, 86.5 | 1.5254 | 1.5262 |
| 0.1 | 81.7, 82.6 | 85.4, 86.2 | 1.5261 | **1.5928** |
| 0.3 | 81.8, 83.0 | 84.1, 85.6 | 1.5265 | **1.8152** |

The ranks on the two measures are opposite and their sums are equal; by the rule the smaller α is taken: **0.01**.

**Claims, α = 0.01.** Sums, never-shown additions after a read line, last reading, seeds 73129, 93683, 116089, 406364, 545353; the settings per text, then the one rule.

| | First cell | Second cell |
|---|---|---|
| without classes | 82.0, 80.9, 80.1, 78.6, 77.8 (median 80.1) then 81.0, 81.3, 78.9, 78.3, 78.7 (**78.9**) | 88.1, 84.8, 86.3, 84.6, 85.6 (85.6) then 86.1, 83.8, 85.0, 83.6, 85.7 (85.0) |
| with classes (a second reading) | 93.2, 95.2, 92.4, 89.6, 92.9 (92.9) then 88.8, 85.6, 88.1, 88.3, 84.8 (88.1) | 97.4, 98.2, 96.4, 96.8, 97.4 (97.4) then 94.1, 85.7, 94.0, 94.8, 94.1 (94.1) |

Prose, nats a byte on each book's last tenth right after its stream; the settings per text, then the one rule.

| | bryant | burgess | carroll | chesterton-ball | austen-emma | melville |
|---|---|---|---|---|---|---|
| first cell | 1.5023, 1.5024 | 1.3645, 1.3655 | 1.4456, 1.4474 | 1.6316, 1.6301 | 1.2416, 1.2417 | 1.4687, 1.4685 |
| second cell | 1.4812, **1.4766** | 1.3408, **1.3322** | 1.4157, **1.4073** | 1.6116, 1.6114 | 1.2203, **1.2150** | 1.4453, **1.4401** |

The test text's last tenth at the end of the two lives (what is kept): first cell 1.5306 and 1.5370 with the settings per text, 1.5323 and 1.5388 with the rule; second cell 1.5301 and 1.5351, then 1.5388 and 1.5417. Side reading, sums seed 1 with the smoothing of prose (0.02) under the rule: first cell 70.4 where the smoothing of sums gives 78.4; second cell 75.6 against 82.9.

**The hypothesis holds**: with one rule, one number and every trust born at nothing, the first cell is 1.2 points under the per-text setting on sums at the median (1.5 allowed; 4 would refute) and within 0.002 nats on each of six books (0.005 allowed).

| | Statement | Outcome |
|---|---|---|
| U1 | sums without classes, first cell, median at least 78.6 | held (78.9) |
| U2 | at least 82.1 | failed |
| U3 | prose, first cell within 0.005 on each of six books | held (the largest difference is 0.0018) |
| U4 | better than the per-text setting on at least four of six books | failed for the first cell (two of six); the second cell is better on six of six |
| U5 | the second cell keeps its gain: at least 3 points on sums, at least 0.015 nats on each book | held (6.1 points; 0.019 to 0.040) |
| U6 | the best α on sums and on prose the same or neighbours | **failed**: 0.3 on sums, 0.01 on prose, for the first cell; though across the whole grid the first cell moves by 3.4 points and 0.003 nats |
| U7 | sums with the smoothing of prose lose more than 5 points | held (8 and 7 points) |

Readings (mine).
1. **The two settings of the answering cell were not needed.** One rule, in which each trust's step is set by the mismatches that trust has itself met, with every trust born at nothing and one number that was not chosen for either text, gives the first cell what the two hand-set schedules gave it, to 1.2 points and 0.002 nats. For the first cell the number hardly matters over a thirtyfold range.
2. **The second cell is better with the rule on every book** (by up to 0.009 nats), and it is the part that is sensitive: with a larger α its loss on prose rises fast, because a part that has been met a few times takes its first steps at full size. That is a fault of the rule as I wrote it and is not repaired here (a sum that starts at one and not at nothing would be the first thing to try: NOT RUN).
3. **With classes the rule loses 3 to 5 points on sums.** Not looked into; the search for classes weighs kinds by a number whose meaning this rule changes, so this is an unfinished joining and not a verdict on either.
4. **One number per text is left: the smoothing of a cell's counts** (2 on sums, 0.02 on prose). With prose's value sums lose 8 points. That is the next debt of this kind.
5. What is kept of an old text is a little worse with the rule for the second cell (0.007 to 0.009), which is the price of parts that move faster.
6. Given to the line `nano/suite` for its last step, where the rule meets streams of other kinds.

## 6. Addition 1 (20 November; written after section 5 and before the runs it describes): the rule made well-conditioned

*What I found while checking the next plan's program.* The rule as run divides a mismatch by the root of the sum of squared mismatches, a sum that starts at nothing. Where a trust's first mismatch is of the size of rounding (a sender whose message is flat), that quotient is plus or minus one, and the trust takes a whole step of size α in a direction that rounding decides. Two implementations of the same arithmetic then drift apart (the compiled loop and numpy agree to 5e-14 on the cases of `check22_output.txt`, and no longer on a harder case), and, which matters more, a part that has been met a few times moves at full size: the sensitivity of the second cell to α that section 5 reports. The figures of section 5 stand as what that program gave; they are not exactly reproducible by another implementation.

*The change:* **the sum starts at one** (one squared nat, the unit of a message), so the step is α over the root of one plus the sum. A first mismatch then moves a trust by α times the mismatch, not by α. Nothing else changes. `gate.py rule=2`; checked against numpy with and without the next plan's option (`check23_output.txt`, thirteen cases).

*Runs:* the development grid of section 2 again (α in 0.01, 0.03, 0.1, 0.3; the same rule of choice), then the claims of section 2 with the chosen α: sums on the five reported seeds without classes, the two lives on prose.

*Predictions (mine).* W1: the second cell on prose no longer breaks with a larger α: at α = 0.1 it is within 0.010 nats of its figure at α = 0.01 on the development measure (chance 0.60). W2: the chosen α is larger than 0.01 (0.65). W3: with the chosen α the first cell is within 1.5 points of 80.1 on sums and within 0.005 nats on each of six books, as in section 5 (0.65). W4: the second cell is at least as good as the per-text setting on at least five of six books (0.50).
