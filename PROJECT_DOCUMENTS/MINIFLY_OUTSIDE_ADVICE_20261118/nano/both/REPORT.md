# nano/both: one system with both kinds of cell on two-digit sums (20 November 2026)

Branch `claude/nano-both-8mmhfm`, made from `claude/friendly-darwin-8mmhfm` at c3f0418cd. Set by the coordinating session after Mr. Xu's words (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and "继续大胆尝试。" (the coordinator's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). Everything here is development level. No gate, seal or review is claimed. Readings are mine and are marked. Each plan was committed before its runs and carries its outcome with every prediction scored; this report gathers them and adds nothing that is not in a plan.

## 1. The question

Two systems learn two-digit addition from a stream of lines `aa+bb=ccc`. The relay system (`loop_small_case/cells2.py`: 256 cells in between, taught by a mismatch relayed over return strengths; 43,278 numbers) learns the rule slowly and goes on gaining with reading. Cells that count, with classes found by search and two answering cells that learn whom to trust (`nano/merge`, `nano/table/gate.py`; 1.2 million cells after 400,000 lines), learn it fast and lose it as they read on. **The coordinator's hypothesis: one system with both kinds of cell is at least as good as the better of the two at every length of reading**; held if, at the median of five seeds, it has at least 96 at 40,000 lines and at least 95 at 400,000.

Every figure below, unless said: never-shown two-digit additions (500 pairs the program never shows), whole answer in the system's own bytes, each line read after a line the system has read (mean of five draws), %; the five reported seeds 73129, 93683, 116089, 406364, 545353, median (range); one thread.

## 2. What was built, and the checks

| File | What it is |
|---|---|
| `both.py` | the joined system: `cells2.py`'s relay system (copied unchanged, its fast kernel) beside `merge.py`'s cells (copied unchanged) with `gate.py`'s two answering cells, the relay system one more sender: its message is the log of its branches cut off at nothing, 0.01 added, made to sum to one; a trust for each of ten bins of its largest share; parts in the second cell like every sender; all born at nothing; its messages for a block of 50 lines come from its strengths at the start of the block |
| `casc.py` | a changed rule for teaching trusts (refuted, `PLAN_2.md`), and the looks |
| `dream.py`, `dream2.py` | lessons on composed new lines (`PLAN_2.md` additions 1 to 5; the second design of `PLAN_5.md`) |
| `life.py` | the life of two tasks with replay to the relay system from `nano/replay/replay.py`'s store (copied unchanged), with or without the lessons |
| `teach.py` | the counting system composes lines for unmet pairs and the relay system learns them (`PLAN_4.md`) |
| `report*.py`, `report_s*.txt` | the tables, every seed and reading |

*Checks.* Every compiled loop has a numpy reference: `both.py check` (1e-15; with no given sender the first cell equals `nano/merge`'s kernel exactly; the relay system read in blocks equals `cells2.py`'s numpy loop to 1e-14), `casc.py check` (2e-15), `dream.py check` and `dream2.py check` (composed bytes are drawn from the stated shares; a lesson changes trusts and parts and no count). **Each part alone repeats its own line's figures to the digit**: the relay system against `loop_small_case/res/past_seed*_{40000,400000}.json` on five seeds; the counting system against `nano/table/res/gate_sums_gate0_modeclasses_seed*.json` on five seeds at 40,000 lines and the two seeds at 400,000; the relay system with replay against `nano/replay/res/m_z_sub_R_<seed>.json` on five seeds at both phase ends.

## 3. Step 1: the relay system as one more sender (`PLAN_1.md`)

| Lines | Relay alone | Counting alone, second cell | Joined, second cell |
|---|---|---|---|
| 40,000 | 82.9 (81.2 to 86.5) | 97.4 (96.4 to 98.2) | **97.1** (96.4 to 98.1) |
| 100,000 | 91.9 | 87.0 | 86.5 |
| 400,000 | 95.5 (95.0 to 97.0) | 75.5 (73.1 to 91.6) | **79.0** (75.1 to 92.2) |

**Refuted at the long reading.** The joined system is the counting system with its fall and one to four points more. The relay system's trust at the levels where it is sure and right ends at 0.03 to 0.11: a trust moves by the mismatch of the whole answer, and on lines read many times the cells that hold the line leave none.

## 4. Step 2: why, and what was tried (`PLAN_2.md`; development seed 1; five forms, each written before its runs)

*Where the fall is* (a look, seed 1, the joined system unchanged): if at the score only the cells that have counted fewer than 512 times are silent, the same system answers **97.7%** at 400,000 lines (87.2 with all cells; the relay system alone 96.7). The counting cells hold the answer; the votes of little-counted cells lose it. My account: a cell keyed on three operand digits and an answer digit has counted, on a shown line, that very line, and votes for the true byte; on a never-shown line the true byte was never counted there and it votes against it, more strongly the longer the reading. **Trusts are taught on lines the cells have counted and used on lines they have not, and after the first 20,000 lines the stream holds no line that is new.**

| Form | What it was | Seed 1: 40,000 / 400,000 lines | Verdict |
|---|---|---|---|
| unchanged | | 96.6 / 87.2 | |
| groups by count | a trust is taught by the answer of the senders that counted at least as much | 96.1 / 74.1 | refuted, withdrawn; my account behind it (untaught high-level trusts) not supported |
| lessons, the relay system composes by chance, every new line taught | | 23.5 / 93.4 | refuted: a young relay system composes wrong sums and its lessons wreck the trusts |
| lessons, the counting system composes, the relay system's answer taught with its own record as the chance | | 90.2 / 93.6 | refuted at 40,000 lines: the teacher is worse than the pupil there; and late the counting system composes only what it remembers |
| lessons by agreement, the relay system composes by chance | a byte is taught where both kinds of cell give the same answer | 89.4 / 95.8 | refuted at 40,000 lines; no fall after 100,000 |
| **lessons by agreement, the composer writes what it is sure of** | | **97.1 / 98.1** | not refuted; taken to the reported seeds |
| the control: the same lessons towards the answering cell's own answer | | 87.3 / 77.4 | collapses after 250,000 lines: the second kind's agreement is needed |

A look that failed and is closed: hearing only the better-counted half of the speaking cells (47 to 84%).

## 5. Step 3: two tasks, the joined system of step 1 unchanged (`PLAN_3.md`)

Addition 400,000 lines then subtraction 200,000; the relay system kept by replay from a store of counting cells as `nano/replay` does.

| | Relay alone, with replay | Joined, second cell | Counting alone, second cell (two seeds) |
|---|---|---|---|
| addition, end of the first phase | 97.5 | 80.4 | 75.5, 90.4 |
| addition, end of the second phase | 93.8 | 58.4 (38.9 to 70.0) | 12.3, 64.0 |
| subtraction (clean), end | 88.1 | 75.6 | 73.8, 74.5 |

In the short life (40,000 then 20,000) the order is the other way round: joined 97.2, 88.6 and 88.9 against the relay system's 89.9, 83.7 and 64.0. **The coordinator's hypothesis (at least 90% of addition kept, subtraction at least 88) is refuted in the long life (72.6%, 75.6) and holds in the short one (91.3%, 88.9).** New, on two seeds: the second answering cell alone loses addition through a long second task (12.3 and 64.0 left) while the first cell keeps 70.8 and 67.0; the counts are all there.

## 6. Step 4: the counting system composes lines for the relay system (`PLAN_4.md`)

At 40,000 real lines the relay system has 82.9 alone, 87.5 with one more shown line after each real line (the control), **89.7** with a composed line for an unmet pair (scored pairs kept out), 91.2 if scored pairs may be composed. 5.8% of composed lines are wrong; the relay system copies about one in six of them. **It does not reach 95, and it mostly repeats what more reading does** (2.2 points over the control, three seeds of five above).

## 7. The second design on the reported seeds: lessons by agreement (`PLAN_5.md`)

*The design.* The joined system of step 1 and one thing more. After each chunk of 2,000 real lines the relay system composes a line after each of them: a byte it is sure of (largest share above one half) is written as its answer, any other is drawn by chance from its shares; the first byte of a line is drawn from the counting system's second cell. Composed lines the system has not read are lessons. **At each byte of a lesson where the relay system's answer is the answering cell's own answer, the two answering cells change their trusts and parts towards it.** No cell counts a lesson, and nothing in a lesson comes from the program.

| Lines | Relay alone | Counting alone | Joined unchanged | **Joined with lessons, scored pairs kept out of the lessons** |
|---|---|---|---|---|
| 5,000 | 3.9 | 94.0 | 94.2 | 93.4 |
| 20,000 | 66.0 | 96.0 | 96.6 | 96.7 |
| 40,000 | 82.9 | **97.4** | 97.1 | 90.8 (88.9 to 97.5) |
| 100,000 | 91.9 | 87.0 | 86.5 | **96.3** (95.2 to 97.5) |
| 200,000 | 94.4 | 90.4 | 89.9 | **97.9** (97.4 to 98.6) |
| 400,000 | 95.5 | 75.5 | 79.0 | **98.8** (95.8 to 99.2) |

(With scored pairs allowed among the lessons: 92.6 at 40,000 lines, 98.7 at 400,000. 863 to 934 CPU-seconds a run, twice the unchanged system's; 1.24 million cells.)

Through two tasks (addition 400,000 lines, then subtraction 200,000; replay to the relay system as in step 3):

| | Relay alone, with replay | Joined unchanged | **Joined with lessons** |
|---|---|---|---|
| addition, end of the first phase | 97.5 | 80.4 | **98.9** (98.6 to 99.4) |
| addition, end of the second phase | 93.8 | 58.4 | **93.4** (91.8 to 94.7) |
| subtraction (clean), end | **88.1** | 75.6 | 81.3 (78.9 to 83.6) |
| share of addition kept | 96.3% | 72.6% | **94.7%** |

In the short life (40,000 then 20,000) it has 93.8, 86.4 and 88.2, under the unchanged joined system's 97.2, 88.6 and 88.9.

## 8. What stands, in one place (mine; development level; sums only)

1. **The hypothesis as first built is refuted**: the relay system as one more sender, with trusts taught by the mismatch, gives the counting system with its fall (79.0 at 400,000 lines against the relay system's 95.5).
2. **Why**: the fall with reading of the counting systems is not in what the cells hold (97.7% at 400,000 lines with the little-counted cells silent) but in trusts that are taught on lines the cells have counted and used on lines they have not. A stream that repeats a closed set of lines stops teaching the difference after some 20,000 lines.
3. **With lessons on new lines that the system composes for itself, taught only where its two kinds of cell agree, one system is above both of its parts from 100,000 lines on** (96 to 99; 98.8 at 400,000 lines on lines met in no form) **and keeps 94.7% of its addition through a second task.** The coordinator's mark at 400,000 lines holds on five seeds; so does the keeping mark of step 3.
4. **It is not at least as good as the better part at every length**: at 40,000 to 50,000 lines it is 5 to 7 points under the counting system alone (the coordinator's first mark fails: 90.8), and it learns the second task less well than the relay system kept by replay (81.3 against 88.1). The first fault is located (the trust of class kinds found anew) and not repaired.
5. **The two kinds of cell work together as teacher and pupil, not as two voices.** The relay system's voice is worth nothing in the joined answer; its composing and its agreement are what the lessons need, and a control on one seed shows that without its agreement the lessons end by destroying what generalises.
6. **The reverse help is small**: the counting system's composed lines bring the relay system to 89.7 at 40,000 lines, 2.2 points more than further shown lines.
7. The best single figures of this line at each length: 97.4 at 40,000 lines (the counting system alone, unchanged); 98.8 at 400,000 lines (joined with lessons); through two long tasks the relay system with replay on the second task (88.1) and the joined system with lessons on the first (98.9, then 93.4, level with the relay system's 93.8).

## 9. Predictions, counted

Of 60 predictions written before their runs, 26 held, 31 failed, 2 were not run and 1 could not be scored as written (a counter of mine was wrong). The plans give each. The ones I got most wrong: that changing the company a trust is taught in by groups of counts would cure the fall (all five failed); that the joined system would keep its addition through a long second task because its level was low (it lost a quarter more); that the lessons would hold 96 at 40,000 lines on the reported seeds.

## 10. Limits and what is not run

One text and two tasks; nothing on prose, where lessons of this kind would be the next test of whether the idea carries. The second design was reached on one development seed through five forms, four of them refuted; the reported seeds were used for no choice. The self-taught control is one seed. In the lives the scored pairs were not kept out of the lessons (on sums keeping them out changed the median by 0.1 at 400,000 lines and 1.8 at 40,000). Task knowledge that remains: `cells2.py` does not teach the relay system a line's first byte, so the relay system is silent there; the program keeps the set of lines read; in step 4 the program draws unmet pairs. "One half" in the composing rule was not varied. Not run: a repair of the fall at 40,000 lines; the life with the scored pairs kept out; replay drawn from the joined system's own cells instead of a third store; three-digit sums; more seeds.

## 11. Added after a late steer: the same question on prose, and what of this line is shown on anything but sums

The coordinator relayed further words of Mr. Xu at 21:42 (Chinese, exactly): "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。" (the coordinator's translation: "Given the bitter lesson, holding on to the task of addition may not be the cleverest thing to do; go on trying general ideas."), and asked that step 4 not be started. **I read that message only at 23:25, after the sections above were written**: step 4 and `PLAN_5.md` had been run by then, against the steer. That is a fault of order of mine (I did not read my notifications while runs were going).

`PLAN_6.md` then put the line's one general question to prose (one run, `nano/table/relay.py`'s system unchanged: T2's cells that count, one answering cell, the relay system of `nano/loop` as a given sender; relay seed 1; 69 CPU-seconds). On the test text's last tenth the loss is 1.4858 as learned; with the relay system's trust in its surest bin (0.165 as learned) set to one it is 1.4957, and every other raising tried is worse still; with its voice silent 1.5237. In the surest bin the whole answer is right 96.6 times in 100 and the relay system 96.4. **So on prose, too, the trust is smallest where the relay system is surest, and there it is what the loss asks for. The fault found on sums belongs to the closed stream of 8,000 repeated lines, not to the rule by which a trust is taught.**

| Result of this line | Shown on anything but sums? |
|---|---|
| the relay system as one more sender does not give the better of the two at a long reading | no; on prose it adds 0.0115 nats (the other line's figure, repeated here to the digit as 1.4858) |
| the fall with reading is in trusts taught on counted lines and used on uncounted ones | sums only; on prose the stream is open and `PLAN_6.md` finds the taught trust of the one sender examined to be the one the unread text asks for (one run) |
| a trust moves only by the whole answer's mismatch, so a sender that agrees with a right answer is not credited | **measured on prose** (`PLAN_6.md`, one text, one seed, one answering cell): true there as well, and not a fault there |
| lessons by agreement on self-composed new lines: above both parts from 100,000 lines on, 98.8 at 400,000 | sums only |
| the same lessons keep addition through a second task (94.7%) | sums only |
| the self-taught control collapses; the second kind's agreement is needed | sums only, one seed |
| the second answering cell alone loses the first task through a long second one | sums only, two seeds (on prose the other line reports that its gain on an old text is lost after other books) |
| composed lines for the relay system mostly repeat more reading | sums only |

What I take from the steer for this line's results (mine): the lessons were built to repair a fault that an open stream does not have. Whether they are worth anything beyond sums turns on one test that is NOT RUN: a system that must keep what it learned on one text while it reads another (where the trusts follow the recent text, as the other line found on its books), with the relay system composing lines of the old text.
