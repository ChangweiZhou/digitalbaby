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
