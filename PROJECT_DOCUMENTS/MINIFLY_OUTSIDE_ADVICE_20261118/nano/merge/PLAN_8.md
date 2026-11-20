# Line `nano/merge`, eighth plan: one more move for the search; a class is split around its worst-foretold member (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Written before any run with `dissolve=2` other than the check (`check_output.txt`: the compiled search against brute force in plain Python, three cases). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

1. On every reported seed the pairs of units digits that lie against their class are pairs that sum to nine; on three seeds of five all ten of them sit with the carrying pairs (`PLAN_2.md`, `PLAN_4.md` note on R1). Their best place by the measure J is a class of their own, which moves of one cell cannot found, and which dissolving a class and seating its members again did not found either.
2. The two long runs with cells keyed on two classes (`PLAN_6.md`): on seed 93683 the system stands at 98.6 to 98.8% of never-shown sums from 300,000 to 400,000 lines; on seed 73129 it ends at 79.4. Looked at after the outcome was entered (`res/s400_C2_seed*.json`): on the first, at the last finding the units-digits classes are 56 and 44 with one stray, and the kinds of largest trust are keyed on (the class of the two tens digits, the class of the two units digits, the byte two back, which is the '=' where the tens digit is named); on the second the classes are 55 and 45 with ten strays, and the same kinds are there. **My reading: where the search seats the pairs that sum to nine rightly, the system holds 98% at long reading; where it does not, it falls. What stands between is the search.**

## 2. The one change (to the search; everything else is setting C2 of `PLAN_6.md`)

After the sweeps, for each class of three or more: its **worst-foretold member** (the member whose leaving costs J least for each of its counts) founds a new class; every other member of the old class then chooses between the old class and the new one by J; the split is kept only if J of the whole grouping rises; this is repeated on the same class up to four times, stopping at the first split that is not kept; then one more sweep over all cells and all classes. `dissolve=2` in `merge.py`. "Up to four times" and "three or more" are laid down; nothing says what a class should be.

## 3. Hypothesis, what refutes it, predictions (mine, before any run)

*Hypothesis.* With the split move, on two-digit sums the units-digits classes are pure for the carry (at most two strays of a hundred) on at least four reported seeds of five, and never-shown sums at 400,000 lines are at least 95% on both seeds of the long reading. *Refuted if* the classes are pure on fewer than three seeds, or either long run ends under 90.

| | Statement | Chance |
|---|---|---|
| X1 | development seed, 40,000 lines: units-digits classes with at most two strays, and the pairs that sum to nine in a class of their own or with the pairs that do not carry | 0.55 |
| X2 | development seed: sums at least 95 (C2 without the split: 95.5) | 0.50 |
| X3 | prose, last twentieth of the stream: not more than 0.002 above C2's 1.5482 | 0.70 |
| X4 | five seeds, 40,000 lines: pure on at least four | 0.45 |
| X5 | five seeds, 40,000 lines, median at least 95 (C2: 93.1) | 0.35 |
| X6 | 400,000 lines: at least 95 on both seeds | 0.30 |
| X7 | three-digit sums, five seeds: within 1 point of C2's 93.9 | 0.60 |

## 4. Runs, size, cost

Development material first (two-digit seed 1; prose's last twentieth): to see the cost of the search on prose (my estimate: three to five times the search of the frozen setting) and that it does no harm. Then the five reported seeds at 40,000 lines, the two long runs, five seeds on three-digit sums; the first life only if the search's cost on prose allows it within this session, else NOT RUN and said so. Not run: moves of more than one founder; a search that merges classes.
