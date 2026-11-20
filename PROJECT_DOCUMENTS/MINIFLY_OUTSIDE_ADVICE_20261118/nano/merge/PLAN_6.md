# Line `nano/merge`, sixth plan: the class used as a line; cells keyed on two classes (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Development level; no gate, seal or review is claimed.

**A slip, stated.** While testing the program I ran it once on development seed 1 for 6,000 lines of two-digit sums (P = 32) to see why its check found no kinds, and looked at the result: 92.0% at 6,000 lines, and among the kinds made, the one keyed on the class of the two tens digits and the class of the two units digits. That was before this file was committed. Nothing was chosen on it; the numbers below were written into the program before it. The first run on three-digit sums with classes (seed 73129, `PLAN_5.md`: 94.9 against 92.5) had also been seen.

## 1. The evidence this follows

1. The system finds, inside the kind of the whole pair, the classes of the two units digits (the carry) and the classes of the two tens digits (their sum modulo ten). Each is used with the *bytes* of the other pair: a cell keyed on (carry class, the two tens digits) or on (class modulo ten, the two units digits). The regularity itself is smaller: the tens digit is fixed by the two classes together, thirty cells or so.
2. `PLAN_5.md`, my expectation: on three-digit sums the hundreds digit needs the class of the tens digits and the class of the units digits together with the two hundreds digits: a cell keyed on two classes, which the system as built cannot make.
3. The hypothesis this line was given says "other cells should be keyed on the class instead of on the cell". So far only one kind was: the kind the class was found in.

## 2. The one addition

The class of an X-cell is a line: at every byte it has a value (the class of the X-cell that is active, or none). **For two kept pairs p and q whose X's share no delay, a candidate kind is keyed on (the class of p's X-cell, the class of q's X-cell, the delays of p's Z that are not in q's X).** Its cells count the next byte like any cell; trusts born at nothing; cells start empty at each finding (the classes are made anew there).

*Which candidates are made* is decided on the system's own last W = 50,000 bytes, which it keeps: a candidate's worth is the sum over those bytes of the log of what the candidate's cell says of the byte with that byte left out of the cell, less the same for the better of the two class kinds it is made from, on the same bytes. The candidates are the pairs (p, q) with p among the 16 class kinds of largest trust and q any kept pair. The P2 = 32 of largest positive worth are made. A kind that is made again at the next finding keeps its trusts.

*Laid down, and the cost in generality:* two classes and no more; p's Z as the rest; 16 by trust; the 50,000 bytes kept; 32 kinds. These numbers were set once, here, and are not chosen on anything. `merge6.py` (a subclass of `merge.py`'s system; `check6_output.txt`).

## 3. Hypothesis, what refutes it, predictions (mine)

*Hypothesis.* With the same numbers on every task, cells keyed on two classes (a) lift three-digit never-shown sums by at least 2 points over the frozen setting, (b) do not lower two-digit sums by more than 1 point, (c) do not raise the loss on any book by more than 0.002 nats. *Refuted if* (a) gains under 1 point, or (b) or (c) fails.

| | Statement | Chance |
|---|---|---|
| V1 | three-digit sums, five seeds, median: at least 2 points above the frozen setting | 0.40 |
| V2 | three-digit: the hundreds digit at least 98% at the last reading (median) | 0.35 |
| V3 | two-digit sums, five seeds, median: at least the frozen setting's 91.8 | 0.60 |
| V4 | two-digit: at least 94 | 0.30 |
| V5 | two-digit: the kinds keyed on two classes alone answer at least 85% whole with under 3,000 cells (median) | 0.35 |
| V6 | prose, each of the three books of the first life: not more than 0.002 above the frozen setting | 0.65 |
| V7 | prose: below the frozen setting by at least 0.002 on at least two books | 0.25 |
| V8 | the whole hypothesis | 0.25 |

## 4. Runs, size, cost

First, development material (sums seed 1 at 40,000 lines; prose's last twentieth of the stream; three-digit seed 1), to see that it runs and what it costs; nothing is chosen on it, and if it harms prose there by more than 0.002 nats the runs on the books are still made, as the hypothesis asks. Then the five reported seeds on two-digit and on three-digit sums, and the first life (three books). Kinds: at most 32 more; their cells are counted and reported beside the others. Cost by estimate: a third more than the frozen setting. Not run: three classes; a class of cells that are themselves keyed on a class; the second life (run only if the first life shows a difference of 0.002 either way).
