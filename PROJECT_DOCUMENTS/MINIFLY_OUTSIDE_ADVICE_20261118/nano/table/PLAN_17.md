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
