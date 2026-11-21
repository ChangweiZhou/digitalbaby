# Blind spots: cells that never come to be, so that the reading goes on being tested on what it has not been told (roadmap second revision, tracks C and K; long reading) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "继续大胆尝试。" (my translation: "Go on trying boldly."), after "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" **Source.** `claude/friendly-darwin-8mmhfm` at 642914438; `gate.py` and `nano/merge/merge.py` as they stand. Written before the change to the program and before any run. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

1. Every system of cells that count loses on never-shown sums as it reads on: 77.6 to about 62 at 400,000 lines without classes, 92 to 79 and 81 with classes (old score), 91 to 63 with look-ahead (`nano/ahead2`), and the second answering cell of `PLAN_15.md` ends at 76.5 and 91.9. The relay system gains with reading.
2. It is the reading and not what is held: a cap on what a long-counted cell may say repairs most of it (`nano/sparse`); and `nano/ahead2` found, on four seeds of four, that when a system that has fallen to 62% begins to read **subtraction**, its addition returns to 87 to 90% within 50,000 lines with no addition shown.
3. `sure.py` (roadmap, section 9): in the second half of a reading, lines met before carry twenty times the weight of first meetings in what changes the trusts. The 8,000 shown pairs are all met within some 30,000 lines; after that the stream holds nothing new, the cell of a whole pair answers its own line, and the trusts are taught only at moments when that cell speaks. At a never-shown pair it does not. *My account of the fall:* the answering cell is taught on an open book and examined on a closed one, and the longer it reads the more that is so. A second task brings first meetings back, and with them the lessons.

## 2. The one change

**A fixed share of all possible cells never come to be.** A cell whose address falls in one class of B by a fixed scramble of the address (the finishing steps of a common 64-bit mixing function; no knowledge of the kind, the bytes or the text; my first scramble, a single multiplication and shift, left blind addresses in near pairs and was replaced before any run) never counts and never speaks, for every kind alike. B = 8. For those contexts the system has to answer from its other cells for as long as it lives, with the true byte arriving each time: an examination that does not run out, set by chance and not by the program's knowledge of the task. Nothing else changes: the answering cells, their step rules and the classes are as in `PLAN_15.md`; no cap; no gate that knows which cells are blind.

It costs something by construction: the lines whose own cell is blind are answered as never-shown lines are.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* With blind spots the system without classes holds its level through long reading: the first cell's never-shown score at 400,000 lines (mean of the last three readings, after a read line) is within 5 points of its score at 40,000 lines, on both seeds. *Refuted if* it is more than 10 points lower on either seed. (Without blind spots I expect about 15 lower; that run is made beside it.)

| | Statement | Chance |
|---|---|---|
| B1 | without classes, first cell: within 5 points of its 40,000-line level at 400,000 lines, both seeds | 0.40 |
| B2 | without classes, first cell at 400,000 lines: at least 5 points above the same system without blind spots, both seeds | 0.50 |
| B3 | the cost at 40,000 lines is under 3 points (first cell, both seeds, without classes) | 0.60 |
| B4 | with classes, the second cell's mean of the last three readings at 400,000 lines is at least 90 on both seeds (76.5 and 91.9 without blind spots) | 0.30 |
| B5 | prose, runs of 0 to 6 bytes, the life with the three worn books: no book more than 0.005 nats worse with blind spots, first cell | 0.50 |
| B6 | B = 4 holds the level at least as well as B = 8 and stands lower at 40,000 lines | 0.55 |

If B1 and B2 fail, my account in section 1 is wrong or the examination is too small a share of the lessons, and the cap stays the only repair known.

## 4. Runs, size, cost, what is not run

Sums: `gate.py sums`, 400,000 lines, seeds 73129 and 93683, readings at that program's marks, both cells, both scores: without classes with B = 0 (none), 8, 4; with classes with B = 8 (B = 0 is `PLAN_15.md`, addition 1). Prose: `gate.py life books=worn mode=none` with B = 8, both cells (B = 0 is `PLAN_15.md`). About four CPU-minutes a run without classes and eight with. The compiled loop is unchanged; the change is which addresses the program hands to it, and the check is that with B = 0 the figures of `PLAN_15.md` come again to the digit.

Not run: other shares; blind spots for some kinds only; an answering cell that knows which cells are blind; the other three seeds; the life of two tasks; look-ahead systems; the relay system (it has no cells of this kind).

## 5. Outcome (entered 20 November after all runs; `res/gate_sums_blind*.json`, `res/gate_life_blind8_*.json`, the printed lines in `out17/`, every reading in `report17.txt` made by `report17.py`)

With no blind spots the long runs give `PLAN_15.md`'s figures again at 40,000 lines to the digit (82.0 and 88.1; 80.9 and 84.8).

Never-shown two-digit additions, %, after a read line; the score at 40,000 lines and the mean of the last three readings to 400,000 lines; first cell, then second cell.

| System | Seed | No blind spots: at 40,000; at the end | One cell in 8 blind | One cell in 4 blind |
|---|---|---|---|---|
| without classes | 73129 | 82.0, 88.1; **65.7**, 71.4 | 80.0, 86.2; **62.3**, 65.0 | 78.0, 82.9; 58.7, 57.5 |
| without classes | 93683 | 80.9, 84.8; **64.6**, 72.9 | 80.4, 84.5; **63.4**, 67.8 | 79.0, 82.4; 59.3, 63.4 |
| with classes | 73129 | 93.2, 97.4; 78.8, 76.5 (`PLAN_15.md`) | 93.2, 97.2; 79.1, 88.0 | not run |
| with classes | 93683 | 95.2, 98.2; 82.9, 91.9 | 93.5, 97.8; 78.3, 74.4 | not run |

Prose (runs of 0 to 6 bytes, the life with the three worn books; first cell): with one cell in 8 blind 1.5393, 1.4001, 1.4897 on the three books and 1.5567 on the test text's last tenth, against 1.5023, 1.3645, 1.4456 and 1.5226 with none: 0.034 to 0.044 **worse**. (The second cell's gain over the first is 0.019 to 0.035 there as elsewhere.)

**The hypothesis is refuted.** With blind spots the system without classes falls by 17.7 and 17.0 points between 40,000 and 400,000 lines, as it does without them (16.3 and 16.3); 10 would have refuted.

| | Statement | Outcome |
|---|---|---|
| B1 | within 5 points of its 40,000-line level at 400,000 lines | failed (17.7 and 17.0 lower) |
| B2 | at least 5 points above the system without blind spots at the end | failed (3.4 and 1.2 **below** it) |
| B3 | the cost at 40,000 lines under 3 points | held (2.0 and 0.5) |
| B4 | with classes, the second cell at least 90 on both seeds at the end | failed (88.0 and 74.4; without blind spots 76.5 and 91.9: the two seeds changed places) |
| B5 | prose: no book more than 0.005 worse | failed (0.036, 0.036, 0.044 worse) |
| B6 | one in 4 holds the level at least as well as one in 8, and stands lower at 40,000 lines | failed in its first half (it falls by 19 to 20), held in its second |

Readings (mine).
1. **My account of the fall was wrong in the form I gave it.** A permanent share of lines that must be answered without their own cell, with the true byte arriving each time, does not keep the reading right for never-shown sums: the system falls as far with it as without. Blind spots are withdrawn; they also cost 0.04 nats on prose, where almost every moment is already a test.
2. **What the look for this plan showed instead** (`votes.py` at 40,000 and 400,000 lines, in `out17/`): the votes for the wrong byte at 400,000 lines come from kinds that hold both units digits and one tens digit, whose cells have met eight of the ten possible tens digits and never the one that a held-out pair would give. That is not a matter of who teaches the trusts. It is taken up in `PLAN_18.md`, written before this outcome was entered but after these runs were read.
3. With classes the two seeds changed places between the plain run and the run with blind spots (76.5 and 91.9; 88.0 and 74.4), and in each the second cell's falls are steps. That agrees with what the line `nano/gate2` reports of five seeds: whether the second cell ends near 75 or near 90 turns on what happens at a finding of classes, not on the seed's own difficulty.
