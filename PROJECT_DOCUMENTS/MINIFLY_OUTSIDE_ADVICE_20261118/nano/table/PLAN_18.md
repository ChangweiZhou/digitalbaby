# Is the fall with long reading what a closed stream teaches? (roadmap second revision, tracks C and K; a question about the test as much as about the system) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "继续大胆尝试，小心求证。" (my translation: "Go on trying boldly, and verify with care."). **Source.** `claude/friendly-darwin-8mmhfm` at 234d7eb61 and the files of `PLAN_17.md` as they stand; `gate.py`, `votes.py`, `nano/merge/merge.py`. Written before `open.py` or the changed `votes.py` were run. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

1. `PLAN_17.md` (its outcome is entered beside this plan): blind spots do not stop the fall. With one cell in eight or one in four never coming to be, the system without classes falls from 78 to 80 at 40,000 lines to 56 to 60 at 400,000, as it does with none (82 and 81 to 63 and 62). My account there, that the trusts are taught on an open book, is refuted in that form.
2. A look made for that plan (`votes.py` at 40,000 and at 400,000 lines, seed 73129, no classes; `out17/votes_none_*.txt`; nothing changed in the system). At the tens digit of never-shown sums, 91 are wrong at 40,000 lines and 183 at 400,000. The kinds that vote most for the wrong byte have changed. At 40,000 lines they are kinds that hold three digits and are tied to the place by a fourth byte, with two followers (the carry they cannot see). At 400,000 lines the largest votes by far come from kinds that hold **both units digits and one of the two tens digits**: (3, 4, 6, 8) alone 191 of 453, then (3, 6, 7, 8), (2, 3, 6, 7), (2, 3, 4, 6); their cells have counted some 365 times and met 7 followers.
3. What such a cell is. With both units digits and one tens digit fixed, the tens digit of the answer runs through all ten values as the other tens digit does. Of those ten pairs about eight are shown and two are not, for ever: the stream is a closed set of 8,000 pairs drawn again and again. So the cell meets eight followers, each some forty-five times in 400,000 lines, and **never the answer of a pair that is held out**, which is exactly the pair it is asked about at the score. The more it has counted, the more its silence about that answer weighs: with 36 counts the unmet follower stands at 2 against 6.5 for a met one, with 365 counts at 2 against 47.

*My account, to be tested:* the fall with long reading is not something the system loses. It is the system learning, correctly, that certain sums do not occur in its stream, and being examined on exactly those. A cap on what a long-counted cell may say (`nano/sparse`) repairs it because it stops any cell from being that sure; reading subtraction brings addition back (`nano/ahead2`) because the same cells then meet other followers and the gap is filled; blind spots do not help because the pairs whose own cell is blind are still shown pairs. The relay system does not show it because it cannot hold two thousand exceptions.

## 2. What is done: three things, none of which changes the system

(a) *The look, made exact.* `votes.py` also says, at the wrong moments, how much of the vote for the wrong byte comes from cells that have never counted the true byte, at 40,000 and at 400,000 lines.

(b) *A diagnostic at the score.* With nothing learned and nothing kept: every cell that has met three or more different followers is made to say of the followers it has not met what it says on average of those it has met; the never-shown sums are scored; everything is put back. Cells of one or two followers are left as they are (a cell that has met one follower two hundred times is the rule itself). This is not offered as a mechanism: it uses no knowledge of sums, but it is a hand's correction for one kind of test.

(c) *An open stream.* The stream of every sums experiment so far is closed. From line 40,000 on, each line is with chance q a pair that is neither shown nor scored (there are 1,500 such pairs among the 2,000 never shown; the 500 scored pairs stay never shown). q = 1 in 100, and 1 in 500. **The system is not changed at all**: what changes is that its stream, like a life, goes on containing things it has not met.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* In an open stream the unchanged system holds its level through long reading: with one line in a hundred a pair not met before, the first cell's score on the 500 never-shown pairs at 400,000 lines (mean of the last three readings, after a read line) is within 5 points of its score at 40,000 lines, on both seeds, without classes. *Refuted if* it is more than 10 points lower on either seed. If it is refuted the fall is not, or not mainly, what section 1 says.

| | Statement | Chance |
|---|---|---|
| O1 | the look: at 400,000 lines more than two thirds of the vote for the wrong byte at wrong tens digits comes from cells that have never counted the true byte | 0.70 |
| O2 | the diagnostic, closed stream, without classes: at 400,000 lines the first cell is within 5 points of its plain score at 40,000 lines, both seeds | 0.55 |
| O3 | the diagnostic changes the score at 40,000 lines by less than 3 points either way | 0.50 |
| O4 | open stream, q = 1 in 100, without classes: the first cell at 400,000 lines within 5 points of its level at 40,000, both seeds | 0.45 |
| O5 | the same: at least 10 points above the closed stream's 65.7 and 64.6, both seeds | 0.60 |
| O6 | q = 1 in 500 gives at least half of what q = 1 in 100 gives (over the closed stream), both seeds | 0.40 |
| O7 | with classes and the second cell, open stream, q = 1 in 100: the second cell's mean of the last three readings is at least 90 on both seeds (76.5 and 91.9 closed) | 0.35 |

## 4. Runs, cost, what is not run

`votes.py` at 40,000 and 400,000 lines, seed 73129, without classes. `open.py`, 400,000 lines, seeds 73129 and 93683: without classes, q = 0 (the check: the figures of `PLAN_17.md`'s runs without blind spots must come again to the digit), 1 in 100, 1 in 500; with classes, q = 1 in 100. Readings at the usual marks, both cells, both scores; the diagnostic at every reading. About five CPU-minutes a run without classes and nine with.

Not run: the other three seeds; the open stream for systems with look-ahead and for the relay system; an open stream from the first line; other shares; the same question on three-digit sums, where nearly every line is new and no fall is expected (the lines `nano/merge` and `nano/ahead2` have that text only to 80,000 lines).

*What it would mean* (mine). If the hypothesis holds, the entry "every counting system loses with long reading" is a statement about a stream that never contains anything new after its first tenth, and the loss is the price of believing that stream. It would not be a defect to repair in the cells, and the cap would be a patch for a test. If it fails, the fall is in the system and is still owed an account.

## 5. Outcome (entered 20 November after all runs; `res/open_*.json`, the printed lines in `out18/`, every reading in `report18.txt` made by `report18.py`)

With no novel lines `open.py` gives the long runs of `PLAN_17.md` again at every reading, to the digit.

**(a) The look** (seed 73129, without classes; wrong tens digits of never-shown sums, after a read line). Votes for the wrong byte against the true one, summed over the wrong moments:

| Lines read | wrong of 500 | from cells that have **never counted the true byte** (of which cells with 64 counts or more) | from cells that have counted it |
|---|---|---|---|
| 40,000 | 91 | +126.5 (+11.6) | -29.8 |
| 400,000 | 183 | **+698.1** (+662.5) | -244.7 |

At both lengths the whole of the net vote for the wrong byte comes from cells that have never met the true one; the cells that have met it vote for it on balance. With reading it is the long-counted ones that grow.

**(b) The diagnostic at the score** and **(c) the open stream.** Never-shown additions, %, after a read line; the score at 40,000 lines, then the mean of the last three readings to 400,000 lines; first cell, second cell.

| System | Seed | Stream | Plain: at 40,000; at the end | With the diagnostic: at 40,000; at the end |
|---|---|---|---|---|
| without classes | 73129 | closed | 82.0, 88.1; 65.7, 71.4 | 82.1, 85.1; **86.1, 87.5** |
| | | open, 1 line in 500 new | 67.0, 79.4 | 86.4, 88.3 |
| | | open, 1 line in 100 new | 70.7, **88.2** | 87.1, 89.2 |
| without classes | 93683 | closed | 80.9, 84.8; 64.6, 72.9 | 80.6, 83.6; **86.3, 87.9** |
| | | open, 1 in 500 | 66.2, 80.1 | 86.6, 88.2 |
| | | open, 1 in 100 | 70.9, **88.2** | 86.4, 88.1 |
| with classes | 73129 | closed (`PLAN_15.md`) | 93.2, 97.4; 78.8, 76.5 | not run |
| | | open, 1 in 100 | 82.2, 89.3 | 93.3, 93.2 |
| with classes | 93683 | closed | 95.2, 98.2; 82.9, 91.9 | not run |
| | | open, 1 in 100 | 85.5, 96.4 | 95.0, 97.9 |

In the open stream with classes the second cell stands at 96.8, 94.7, 97.4, 97.9, 98.1 from 50,000 to 250,000 lines on the first seed and then steps down to 87.4 at the first reading after the classes are found anew, returning to 90.6 by the end; on the second seed it steps down to 89.0 after the finding at 64,000 lines and climbs back to 96.6. About 3,600 novel lines in 360,000 at 1 in 100, about 700 at 1 in 500. 246 to 267 CPU-seconds a run without classes, 365 with.

**The hypothesis as I wrote it is refuted; the account behind it is supported; and what holds is something I did not predict.** I wrote the hypothesis for the first cell: in the open stream it still falls by 11.3 and 10.0 points (10 would refute). The second cell, with the same stream and nothing changed, does not fall: 88.1 and 88.2, 84.8 and 88.2.

| | Statement | Outcome |
|---|---|---|
| O1 | more than two thirds of the wrong vote at 400,000 lines from cells that have never counted the true byte | held (all of it, and more: the rest votes for the true byte) |
| O2 | the diagnostic, closed stream: the first cell at the end within 5 points of its plain score at 40,000 lines | held, and above it (86.1 against 82.0; 86.3 against 80.9) |
| O3 | the diagnostic changes the score at 40,000 lines by under 3 points | held for the first cell (+0.1, -0.3); the second cell loses 3.0 and 1.2 |
| O4 | open stream, 1 in 100, the first cell within 5 points of its 40,000-line level | **failed** (11.3 and 10.0 lower) |
| O5 | the same at least 10 points above the closed stream | failed (+5.0, +6.3) |
| O6 | 1 in 500 gives at least half of what 1 in 100 gives | failed (first cell a quarter; second cell 48% and 47%) |
| O7 | with classes, open stream: the second cell at least 90 at the end on both seeds | failed narrowly (89.3 and 96.4; closed 76.5 and 91.9) |

Readings (mine).
1. **The slow fall with long reading is the vote of cells against followers they have never met.** Take that vote away at the score, changing nothing else and learning nothing, and the system in the closed stream does not fall with reading, it rises (82 to 86, 81 to 86). The stream is a closed set of pairs; a cell that holds three of four digits meets eight of ten possible followers for ever and never the one a held-out pair would give; and the score asks for exactly that one. The system is right about its stream and is marked wrong for it.
2. **In an open stream the system with the second answering cell holds its level, with nothing changed.** One line in a hundred that is new is enough (88.2 on both seeds over the last three readings, where the closed stream leaves 71.4 and 72.9); one in five hundred gives half of that. The first cell alone does not manage it: it has one trust for a kind and a level, and cannot hold a kind down at the one place where its silence misleads without losing it elsewhere. The second cell can, and does, from some 3,600 lines.
3. **With classes the two losses come apart.** In the open stream the second cell stays at 97 to 98 for as long as the classes stand, and what is left are the steps at the findings of classes, from which it now recovers while it reads (89.0 to 96.6; 87.4 to 90.6 and rising). The step is the renewal of classes, which belongs to the line `nano/merge`; the slow fall was the closed stream.
4. **What this changes in the ledger.** "Every counting system loses with long reading" is a statement about a stream that holds nothing new after its first tenth. The cap of `nano/sparse` repairs a test, not a learner: it stops any cell being as sure as its stream entitles it to be. A comparison between the relay system and the counting systems at 400,000 lines should be made in an open stream as well; the relay system has not been run in one (NOT RUN), nor have systems with look-ahead.
5. **My account of 20 November, section 9, has to be narrowed.** There I read the wrong votes at 40,000 lines as coming from kinds that are tied to the place and inexact. The exact look shows that at 40,000 lines too the net wrong vote comes from cells that have never met the true byte (+126.5) and the others vote for it; many of the place-tied cells are of that sort (a cell that holds one units digit and both tens digits has met "carry" nine times in ten and, if the tenth pair is the one held out, never the other). The two descriptions overlap; this one is the measured one.
6. *Limits.* Two seeds, one program, one author; the line `nano/gate2` has been asked to repeat the open stream and the diagnostic in its own program on the other seeds. The diagnostic is a hand's correction for one kind of test and is not proposed as a mechanism. In the open stream the pairs that arrive as new come from 1,500 pairs set aside for it; the 500 scored pairs are never shown.
