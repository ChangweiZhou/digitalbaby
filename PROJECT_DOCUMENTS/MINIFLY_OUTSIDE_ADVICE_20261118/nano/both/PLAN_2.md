# Why the joined system falls with reading, and one change to how a trust is learned (step 2) (20 November 2026)

**Authorisation and source** as `PLAN_1.md` (Mr. Xu's words quoted there; branch `claude/nano-both-8mmhfm`, this plan written on top of commit 7c43415). Written while the runs of `PLAN_1.md` were under way and after their first readings were seen (two seeds to 150,000 lines: the joined second cell 85.7 and 86.3 at 100,000 lines, where the relay system alone has 92.1 and 89.0 and the joined cell with the relay system silent has 84.2 and 86.0). Written before the program of this plan exists and before any run of it. Development level; no gate, seal or review is claimed.

## 1. What the first readings show, and my account of it (to be tested, not yet shown)

The joined answer falls with reading as the counting system's does, and silencing the relay system at the score changes it by a point or two: the relay system is hardly heard although it has become the better part.

*Account.* A trust changes by (the mismatch of the whole answer) times (the sender's message). On a line that has been shown several times the cells that hold the line exactly (kinds of four delays that cover the four digits of the operands) make the whole answer right, the mismatch is nothing, and no trust changes. Two things then follow from one cause:

1. the relay system reaches its high levels of sureness only after the shown lines are held exactly by counting cells; its trusts at those levels are born at nothing and meet no mismatch to grow on;
2. the same holds inside the counting system. A trust is indexed by the level of the cell's count. As reading goes on, the cells that generalise (kinds of two and three delays, class cells) pass to higher levels of count; the trusts of those new levels are learned only in the company of the exact cells, so they stay where they were born (0.02, and nothing for class kinds). On a never-shown line the exact cells are silent and the cells that generalise speak at levels whose trusts were never taught. This would be the fall with reading of every counting system in the programme, and it would explain the other line's finding that reading a new task brings addition back with no addition shown (new lines give mismatch again).

So the trusts are learned in one company (with the cells that hold the present line) and used in another (without them).

## 2. The one change

**A sender's trust is learned from the answer of the senders that have counted at least as much as it has.** At each byte the speaking senders are put in groups by the doubling of their counts (group b: counts from 2^b to 2^(b+1) - 1; not capped). The answer of group b and above is the competition of the branches summed over the senders of groups b and higher. The trust (and, in the second cell, the parts) of a sender in group b changes by the mismatch of that answer, with the step and the rule as before. The answer the system gives is unchanged in form: all senders (the answer of the lowest group and above).

So a cell that has counted much learns what it is worth without the cells that have counted little, which are the ones that will be silent on a line never met; a cell that has counted little learns what it adds to those that counted more. Nothing names a task, a kind, a place or a line. The relay system takes its place by the same rule: its count is the number of bytes it has been taught.

Arms on the development seed 1, 400,000 lines of addition, readings as `PLAN_1.md`: the counting system alone and the joined system, each with the change (`casc.py`, `cascade=1`); the same program with `cascade=0` must give `both.py`'s figures to rounding. At each reading the trusts of the first cell by level of count (mean weighted by uses) are kept, with and without the change, as a look at the account of section 1.

*What is run next, decided here:* if on seed 1 the joined system with the change is, at 400,000 lines, at least 3 points above the joined system without it, the two arms with the change are run on the five reported seeds and the hypothesis of `PLAN_1.md` is judged on them (stated as a second design, after the first was judged on its own runs). If not, the outcome is reported as a failure of the account or of the change, whichever the looks show.

## 3. Hypothesis, what refutes it, predictions (mine, before any run of this plan)

*Hypothesis.* With the change the counting system no longer falls with reading and the joined system is at least as good as the better part. *Refuted if* on seed 1 at 400,000 lines the counting system alone with the change is under 90 (second cell), or the joined system with the change is more than 3 points under the relay system alone.

| | Statement (seed 1, second cell, after a read line) | Chance |
|---|---|---|
| C1 | counting alone with the change at 400,000 lines at least 93 | 0.50 |
| C2 | counting alone with the change at 40,000 lines within 2 points of the system without it | 0.60 |
| C3 | joined with the change at 400,000 lines at least 95 | 0.40 |
| C4 | joined with the change not more than 3 points under the better part at every reading from 40,000 lines on | 0.35 |
| C5 | without the change, at 400,000 lines, the mean trust of the first cell at the top level of count is under a quarter of that at the level four doublings below it (the account's mark) | 0.50 |

## 4. Cost, what is not run

The kernel makes up to about twenty competitions a byte where it made two; I expect a run of 400,000 lines to stay near ten CPU-minutes. Two runs on seed 1, then ten on the reported seeds if the rule above says so. Not run: other groupings (by exact order of counts; by the capped level), a relay system placed in another group, a change of the step, any search over these; the second task (step 3).
