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

## 7. Addition 2 (20 November; written while arm N of addition 1 was still running, after its readings to 50,000 lines were seen, and before the program `dream2.py` was run beyond its check)

*What arm N shows so far.* It fails from the start: 5.7, 13.4, 17.0, 23.5 and 51.3 at 5,000 to 50,000 lines, where the unchanged joined system has 92.9 to 96.7. The printed counts say why: of the 20,000 lines the relay system composed between 20,000 and 40,000 lines, 15,535 were new, 8,623 were well formed and **274 were right sums**. Two faults of the design, both mine: (i) `cells2.py` never teaches the relay system the first byte of a line, so its composed lines start with a byte of no meaning and half are not well formed; (ii) drawing the answer bytes by chance from branches that are only half sure gives wrong answers, and the lessons then teach the trusts wrong answers on exactly the lines where the counting system was right 97 times in 100. The control arm R is NOT RUN: it was to show that novelty is what matters in a lesson that works, and the lesson does not work. Section 8 will score D1 to D5 when the run ends.

*The changed form (one design, two parts changed, both forced by the faults above).*
1. **The counting system composes, the relay system answers.** After a chunk of 2,000 real lines the counting system writes one line after each of them, each byte drawn from its second answering cell's shares given what stands before (it composes well-formed lines: 99 of 100 in the check). The composed lines the system has not read are the lessons. At each byte of a lesson the relay system gives its answer (its largest branch) for what stands before, and the answering cells change their trusts and parts towards that answer; no cell counts; the relay system is silent as a sender there, and gives no answer at a line's first byte.
2. **A lesson byte is taken with a chance equal to the teacher's edge at its level of sureness**: twice the share of stream bytes at that level on which the relay system's answer was the byte that came, less one; nothing if below. The record is the relay system's own, kept from birth, each answer formed before the byte was learned. So a young or unsure teacher teaches nothing, with no number chosen.

*Arms on seed 1, 400,000 lines* (`dream2.py`): teacher=relay; teacher=self (the control: the target is the counting system's own answer, taken with the same chances; it shows whether the relay system is needed or any sharpening on new lines would do).

*Hypothesis.* With these lessons the joined system does not fall with reading. *Refuted if* the relay-taught arm at 400,000 lines is under 93, or at 40,000 lines more than 3 points under the unchanged joined system (96.6).

| | Statement (seed 1, joined second cell) | Chance |
|---|---|---|
| E1 | relay-taught arm at 40,000 lines within 3 points of the unchanged joined system | 0.55 |
| E2 | relay-taught arm at 400,000 lines at least 95 | 0.30 |
| E3 | relay-taught arm at least 10 points above the unchanged joined system at each of 150,000, 200,000 and 250,000 lines (78.9, 78.3, 77.9) | 0.40 |
| E4 | the self-taught control is within 3 points of the relay-taught arm at 400,000 lines (the relay system is not needed for it) | 0.40 |
| E5 | the relay system's answers on the lesson bytes that are answer digits of right composed sums are the composed bytes at least 90 times in 100 over the life | 0.50 |

*What is run next, decided here:* if E2 holds, the relay-taught arm is run on the five reported seeds beside the unchanged arms of `PLAN_1.md`, and the coordinator's hypothesis is judged on them as a second design. *Cost:* composing costs sixteen answers of the counting system for each chunk; I expect 15 to 20 CPU-minutes an arm. *Not run:* other chances than the edge; lessons from both teachers at once; the relay system taught in return (`PLAN_4.md`).

## 8. Outcome of addition 1 (entered 20 November; `res/s2_dream_new_seed1.json`; 757 CPU-seconds)

Seed 1, joined second cell (first cell), never-shown additions after a read line: 5.7 (24.7), 13.4, 17.0, 23.5 (82.4) at 5,000 to 40,000 lines; 51.3, 89.6, 90.6, 93.0, 92.9, 94.8, 93.6 at 50,000 to 350,000; **93.4 (96.8) at 400,000**. Of 400,000 composed lines 287,478 were new, 155,594 of those well formed, 12,153 right sums.

**Refuted**: far more than 3 points under the unchanged system until 100,000 lines. D1 failed (93.4). D2 failed (23.5 against 96.6). D3 held in the letter (90.6, 93.0, 92.9 against 78.9, 78.3, 77.9) for a reason that is not the hypothesis's: the lessons of a teacher that is wrong wrecked the counting cells' trusts, the real lines then gave the relay system trust as a sender (1.0 to 3.2 at its levels 3 to 8, against 0.03 to 1.0 unchanged), and from 100,000 lines on the joined answer is the relay system's, within three points (with the relay system silent at the score it is 40 to 60). D4 NOT RUN (the control; see section 7). D5 failed (93.4 with all cells, 89.8 with the little-counted silent).

Reading (mine): one thing here is worth keeping. When the counting cells do not answer new lines well, the mismatch rule does give the relay system its trust and the joined answer follows it. The joined system of `PLAN_1.md` fails to follow the relay system because the counting cells leave no mismatch on the lines the trusts are taught on, not because the rule cannot weigh the relay system.

## 9. Outcome of addition 2 (entered 20 November; `res/s2_dream2_relay_seed1.json`; 679 CPU-seconds)

Seed 1, joined second cell, never-shown additions after a read line, beside the unchanged joined system and the relay system alone:

| Lines | Relay alone | Unchanged | With lessons (relay-taught) | New composed lines since the last reading, of them right sums |
|---|---|---|---|---|
| 5,000 | 3.2 | 92.9 | 92.8 | 3,519, 2,864 |
| 20,000 | 71.8 | 97.0 | 95.0 | 2,388, 2,009 |
| 40,000 | 75.3 | 96.6 | **90.2** | 2,185, 1,781 |
| 100,000 | 90.5 | 96.6 | 92.9 | 2,333, 1,851 |
| 150,000 | 92.2 | 78.9 | 81.4 | 1,551, 911 |
| 200,000 | 93.2 | 78.3 | 89.0 | 994, 528 |
| 250,000 | 91.0 | 77.9 | 86.6 | 722, 409 |
| 300,000 | 94.1 | 87.2 | 93.0 | 440, 300 |
| 400,000 | 96.7 | 87.2 | **93.6** | 326, 244 |

**Refuted by its own mark at 40,000 lines** (6.4 under the unchanged system; 3 allowed), though it ends at 93.6, above the mark of 93 and 6.4 above the unchanged system. E1 failed. E2 failed (93.6). E3 failed (2.5, 10.7 and 8.7 above; at least 10 at each was asked). E4 NOT RUN (the self-taught control: not run, because E2 failed and the machine was given to the next form). E5 cannot be scored as written: the program counted the taken answer bytes of all new lines (33,241) and not those of right sums; on 24,173 of them (73%) the relay system's answer was the composed byte of a right sum. The counter is corrected for later runs.

Readings (mine). (1) The lessons do part of what was hoped late in the life and cost six points at 40,000 lines, where the teacher (75% on never-shown sums) is worse than the pupil (97%): the teacher's record on the stream gives an edge of 1.0 at its sure levels, because on lines it has read it is right when sure, and that record flatters it on new lines. (2) Late in the life there are almost no lessons: the counting system composes what it remembers (326 new lines among 50,000 composed at the end, against 3,519 among the first 5,000). (3) A look made beside it (`res/s2_look_half_seed1.json`): hearing only the speakers that have counted at least the middle count among the speakers is far worse than hearing all (47 to 84%); the cut that works in section 5's look lies lower and moves with the reading, and I have no rule for it that names no number. That route is closed here.

## 10. Addition 3 (20 November; written after sections 8 and 9, before the runs it describes; `dream2.py` with `composer=relay`): the relay system composes, and a lesson is taken where the two kinds of cell agree

Two changes to addition 2, each answering one of its two faults.
1. **The relay system composes** (each byte drawn in proportion to its branches cut off below at nothing; the first byte of the line, which it is never taught, is drawn from the counting system's second cell). Its operand digits are near chance at every age, so about a fifth of its lines stay new however long the reading. Its composed answer digits are often wrong while it is young; they are only the past of the bytes after them, and no lesson's target is a composed byte.
2. **Arm A (agreement): a lesson byte is taken where the relay system's answer is the answering cell's own answer** (second cell, all counting cells, the relay system silent), and that byte is the target. A teacher that is wrong where the pupil is right then teaches nothing; what is taught is: on a new line, do not let the cells that vote against the answer on which both kinds agree weaken it. No number is chosen and no record is needed. What it gives up: the bytes where the pupil is wrong and the teacher right.
   **Arm B: the target is the relay system's answer whether or not they agree, taken with the chance of addition 2** (the teacher's edge by its stream record). It keeps those bytes and the harm of a young teacher.

*Hypothesis.* With lessons by agreement the joined system keeps its level at 40,000 lines and does not fall with reading. *Refuted if* arm A at 40,000 lines is more than 3 points under the unchanged system (96.6), or at 400,000 lines under 93.

| | Statement (seed 1, joined second cell) | Chance |
|---|---|---|
| F1 | arm A at 40,000 lines within 3 points of the unchanged system | 0.70 |
| F2 | arm A at 400,000 lines at least 95 | 0.25 |
| F3 | arm A at least 10 points above the unchanged system at each of 150,000, 200,000 and 250,000 lines | 0.30 |
| F4 | arm B at 400,000 lines at least as high as arm A | 0.50 |
| F5 | arm B at 40,000 lines more than 3 points under the unchanged system | 0.70 |

*What is run next, decided here:* the arm that is refuted by neither mark of the hypothesis and is higher at 400,000 lines is run on the five reported seeds beside the arms of `PLAN_1.md` (a second design, judged on the coordinator's marks); if both are refuted, none is, and step 2 ends with what the looks have located. *Cost:* about 12 to 15 CPU-minutes an arm.
