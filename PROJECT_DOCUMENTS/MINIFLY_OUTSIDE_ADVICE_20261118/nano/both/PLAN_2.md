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

## 5. Outcome of sections 2 to 4 (entered 20 November; `res/s2_*_seed1.json`, the printed lines in `log_s2_*`)

Seed 1 (development), never-shown additions, whole answer, own bytes, after a read line, %. One thread; 427 to 515 CPU-seconds a run. `casc.py check`: the compiled loop with the groups equals numpy to 2e-15 and, with one group, `both.py`'s kernel to 4e-16.

| Lines | Relay alone | Joined, second cell, as `PLAN_1.md` | Joined with the change (relay silent at the score) | Counting alone with the change, second cell |
|---|---|---|---|---|
| 5,000 | 3.2 | 92.9 | 92.8 (92.4) | 89.4 |
| 20,000 | 71.8 | 97.0 | 85.7 (84.1) | 83.3 |
| 40,000 | 75.3 | 96.6 | 96.1 (96.5) | 80.8 |
| 100,000 | 90.5 | 96.6 | 81.0 (66.4) | 76.9 |
| 150,000 | 92.2 | 78.9 | 94.2 (87.2) | 60.2 |
| 250,000 | 91.0 | 77.9 | 95.0 (64.6) | 57.4 |
| 300,000 | 94.1 | 87.2 | 73.2 (33.4) | 64.6 |
| 400,000 | 96.7 | 87.2 | 74.1 (32.5) | 66.6 |

**The hypothesis is refuted, and the change is withdrawn.** The counting system alone with the change is at 66.6 at 400,000 lines (under 90 was the refuting mark) and already 16 points under the unchanged system at 40,000; the joined system with the change ends 22.6 points under the relay system alone. By the rule of section 2 the change is not run on the reported seeds (it ends 13 points under the joined system without it).

Predictions: C1 failed (66.6). C2 failed (80.8 against 96.6). C3 failed (74.1). C4 failed. C5 failed: without the change the mean trust at the top level of count is 0.011 against 0.031 four doublings below for the plain kinds (a third, not under a quarter) and 0.043 against 0.056 for the class kinds; the trusts of the high levels are not left at their birth values. **So item 2 of my account in section 1 is not supported.**

What the change did do: the relay system was heard. Its trust in the first cell stood at 2.6 to 5.2 at its upper levels (0.04 to 1.0 without the change), and between 150,000 and 250,000 lines the joined answer (94.2, 94.3, 95.0) was above the relay system alone (92.2, 93.2, 91.0) and far above the joined system without the change (78 to 79). It then fell by 22 points at the reading after the classes were found anew (2.56 million bytes), where the counting cells, taught as corrections to the relay system, out-vote it on never-shown lines.

**A look that locates the fall** (not a registered measure; `res/s2_joined_c0_seed1.json`, the joined system of `PLAN_1.md` unchanged, two of the five pasts): the second cell's answer when, at the score only, the cells that have counted fewer than a given number of times are silent.

| Lines | All cells | Only counts of 8 and more | 64 and more | 512 and more | Class kinds alone |
|---|---|---|---|---|---|
| 40,000 | 96.6 | 96.7 | 97.1 | 64.5 | 96.5 |
| 100,000 | 96.6 | 96.8 | 97.2 | 96.3 | 95.8 |
| 150,000 | 78.9 | 79.3 | 79.6 | 95.3 | 61.0 |
| 250,000 | 77.9 | 78.4 | 77.9 | 96.0 | 64.8 |
| 300,000 | 87.2 | 87.2 | 88.3 | 98.0 | 94.3 |
| 400,000 | 87.2 | 87.2 | 87.5 | **97.7** | 94.9 |

Reading (mine). **At 400,000 lines the counting cells hold an answer to 97.7% of never-shown sums, above the relay system's 96.7; what loses ten points is the vote of cells that have counted between 64 and 512 times.** My account of which cells these are (from the programs, not measured cell by cell): a kind of four delays that holds three of the four operand digits and one answer digit has, on a shown line, counted that very line among the eight or so lines that share its context, and votes for the true byte among others; on a never-shown line the true byte was never counted in it, and it votes for eight wrong bytes and against the true one, more strongly the longer the reading (the vote grows as the log of its counts). Its trust is learned on shown lines, where it helps. The classes found at 1.28 million bytes on this seed are poor (class kinds alone 61%) and the well-counted cells answer at 95% in spite of them.

So what is wrong is not that some trusts are untaught; it is that **all trusts are taught on lines the cells have already counted, and used on lines they have not**. After the first 20,000 lines the stream holds no line that is new, so nothing in it can teach the difference.

## 6. Addition 1 (20 November; written after section 5 and before the program and runs it describes): lessons on new lines, set by the other kind of cell

*The idea.* The system has two kinds of cell that generalise in different ways. Let each make lines the system has not read, and let the other learn from them. Here the first half: **the relay system composes lines; the lines that are new to the system are read by the answering cells, which change their trusts and parts on them and nothing else** (no cell counts a composed line; the relay system is silent as a sender on them, since it wrote them). On such a line the cells that hold shown lines exactly are silent and the cells that have counted neighbouring lines vote as they will on a never-shown line: the trusts are taught in the company in which they are used. The teacher is right about as often as the relay system is on never-shown lines (75% at 40,000 lines, 97% at 400,000 on this seed), so the lessons are noisy early; a cell that votes against the composed byte more often than not loses trust all the same.

*Composing* (nothing names the task): after each real line the relay system writes one line byte by byte from what it has just read: each byte is drawn in proportion to its branches cut off below at nothing (by chance alone if none is above nothing), until it draws a line end or has written sixteen bytes. *New* means: not among the lines the system has read (the program keeps the set of lines read; a store of runs as in `nano/replay` would say the same). Lines without a line end are dropped. The lessons use the step and rule of the trusts unchanged.

*Arms on seed 1, 400,000 lines* (`dream.py`): (N) joined with lessons on the composed lines that are new; (R, the control) joined with lessons on the composed lines that are not new (lines the system has read); expected to change nothing, since there the answer is already right. The readings and looks of sections 2 and 5.

*Hypothesis.* With lessons on new lines the joined system does not fall with reading. *Refuted if* arm N at 400,000 lines is under 93 or, at any reading from 40,000 lines on, more than 3 points under the better of the relay system alone and the unchanged joined system.

| | Statement (seed 1, joined second cell) | Chance |
|---|---|---|
| D1 | arm N at 400,000 lines at least 95 | 0.40 |
| D2 | arm N at 40,000 lines within 2 points of the unchanged joined system (96.6) | 0.50 |
| D3 | arm N at least 10 points above the unchanged joined system at 150,000 to 250,000 lines (where that falls to 78) | 0.50 |
| D4 | arm R (the control) within 3 points of the unchanged joined system at 400,000 lines | 0.60 |
| D5 | in arm N, silencing the cells that counted fewer than 512 times at the score changes the answer at 400,000 lines by less than 3 points (the lessons have done what the silencing did) | 0.40 |

*What is run next, decided here:* if D1 holds, arm N and the unchanged arms are run on the five reported seeds and the coordinator's hypothesis of `PLAN_1.md` is judged on them as a second design. *Cost:* about 12 CPU-minutes an arm on seed 1. *Not run here:* the other half (the counting system's composed lines taught to the relay system: `PLAN_4.md`), lessons without the novelty test, other ways of composing.
