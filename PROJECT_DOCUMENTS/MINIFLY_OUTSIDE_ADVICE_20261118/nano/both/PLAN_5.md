# The second design on the reported seeds: the joined system with lessons by agreement, on sums and through two tasks (20 November 2026)

**Authorisation and source** as `PLAN_1.md` (Mr. Xu's words quoted there; branch `claude/nano-both-8mmhfm`, written on top of commit 6f098ad and the outcome of `PLAN_2.md` addition 4). Written before any run on the reported seeds of the design it describes. Development level; no gate, seal or review is claimed.

## 1. The design (fixed on the development seed 1 in `PLAN_2.md`, sections 6 to 13; nothing is chosen on the reported seeds)

The joined system of `PLAN_1.md` (the relay system of `cells2.py` as one more sender to the two answering cells over `merge.py`'s cells with classes), and one thing more. After each chunk of 2,000 real lines the relay system composes one line after each of them: at each byte its shares are its message's; if its largest share is above one half the byte is its answer, otherwise it is drawn by chance from its shares; the first byte of a line, which `cells2.py` never teaches it, is drawn from the counting system's second cell; a line ends at a line end or after sixteen bytes. The composed lines that the system has not read are lessons. At each byte of a lesson after the first, **where the relay system's answer is the second answering cell's own answer** (all counting cells, the relay system silent), the two answering cells change their trusts and parts towards that byte, with the step and rule of every trust. No cell counts a lesson; the relay system learns nothing from it.

What is task knowledge in it, stated: none in the lesson rule. The relay system's silence at a line's first byte is `cells2.py`'s. "One half" is the only number, and it was not varied. The set of lines read is kept by the program.

## 2. What is run

*Sums* (`dream2.py seed=S teacher=agree composer=relay sure=1`): 400,000 lines of addition on the five reported seeds, readings as `PLAN_1.md`, beside that plan's arms from the same stream (relay alone, counting alone, joined unchanged).

*Two tasks* (`life.py seed=S arm=joined lessons=agree`): the lives of `PLAN_3.md` (400,000 then 200,000; 40,000 then 20,000), replay to the relay system as there, the lessons going on through both phases; beside that plan's arms.

## 3. Hypotheses, what refutes them, predictions (mine, before the runs)

*The coordinator's hypothesis of step 1, on the second design.* At the median of five seeds the joined system is at least 96 at 40,000 lines and at least 95 at 400,000. *Refuted if* it is more than 3 points under the counting system alone at 40,000 lines (97.4) or more than 3 under the relay system alone at 400,000 (95.5).

*The coordinator's hypothesis of step 3, on the second design.* It keeps at least 90% of its addition level through the second task and reaches at least 88 on subtraction (long life, medians).

| | Statement (joined second cell with lessons; median of five seeds) | Chance |
|---|---|---|
| P1 | sums: at least 96 at 40,000 lines | 0.60 |
| P2 | sums: at least 95 at 400,000 lines | 0.65 |
| P3 | sums: at least as high as the relay system alone at 400,000 lines (95.5) | 0.55 |
| P4 | sums: every seed at least 93 at 400,000 lines | 0.45 |
| P5 | sums: at no reading more than 3 points under the better of the relay system alone and the unchanged joined system (medians) | 0.20 |
| Q1 | long life: addition at the end of the first phase at least 95 | 0.55 |
| Q2 | long life: at least 90% of that level kept through the second phase | 0.40 |
| Q3 | long life: subtraction (clean) at least 88 | 0.30 |
| Q4 | long life: not more than 3 points under the relay system with replay on all three measures (97.5, 93.8, 88.1) | 0.20 |
| Q5 | short life: at least 90% kept and subtraction at least 88, as the unchanged system (91.3%, 88.9) | 0.60 |

## 4. Cost, what is not run

Five runs of sums at about 15 CPU-minutes, five long lives at about 25, five short ones at about 3; four at a time, about an hour. Not run: the self-taught control; the counting system with lessons and no relay sender on real lines; other chunk sizes; any second look at the design.

## 5. Addition 1 (20 November; written when four of the five sums runs of section 2 were in and before the runs it describes): the scored pairs kept out of the lessons

*A doubt about my own score, which the readings so far make pressing* (four seeds at 400,000 lines: 99.4, 99.2, 95.5, 98.6). The relay system composes lines with operands drawn by chance, so about one new line in twenty begins as one of the 500 scored lines does. The system is never given the true answer of such a line: a lesson's target is the byte on which its own two kinds of cell agree, and no cell counts a lesson. But the trusts and the parts of the second cell are changed on those lines, so the scored lines are not, strictly, lines the system has met in no form. `PLAN_4.md` kept the scored pairs out of composing for the same reason.

*The run.* The sums runs of section 2 again on the five reported seeds with one difference: a composed line that begins as a scored line does (its six bytes up to the equals sign) is not a lesson (`dream2.py ... excl=1`; this uses the program's knowledge of which lines are scored, for the score's sake only). **The figures of this arm are the ones to be read as never-shown in the strict sense; where the two arms differ, this one stands.**

| | Statement (median of five seeds, joined second cell) | Chance |
|---|---|---|
| X1 | with the scored pairs kept out: at least 95 at 400,000 lines | 0.80 |
| X2 | within 1.5 points of the arm of section 2 at 400,000 lines | 0.70 |
| X3 | within 2 points of it at 40,000 lines | 0.70 |

Cost: five runs of about 15 CPU-minutes beside the lives. The lives of section 2 are not run again in this form (stated as a limit of their figures).

## 6. Outcome (entered 20 November after all twenty runs; `res/s5_*.json`, `res/s5x_*.json`; every seed and reading in `report_s5.txt`, made by `report5.py`)

Never-shown lines, whole answer, own bytes, after a read line, %; five reported seeds, median (range). One thread: 863 to 934 CPU-seconds a sums run with lessons (412 to 445 without), 1,129 to 1,317 a long life (667 to 696 without), 172 to 192 a short one. 1.24 million cells at 400,000 lines of sums, 1.39 million at the end of a long life. Of 400,000 composed lines about 232,000 are new, 166,000 of those well formed and 39,500 right sums; 1.31 of 2.21 million lesson bytes are taken (seed 73129).

**Sums** (the same streams as `PLAN_1.md`):

| Lines | Relay alone | Counting alone, second cell | Joined unchanged | **Joined with lessons** | **The same, scored pairs kept out of the lessons** |
|---|---|---|---|---|---|
| 5,000 | 3.9 | 94.0 | 94.2 | 93.6 | 93.4 |
| 10,000 | 19.6 | 95.0 | 95.3 | 94.5 | 94.4 |
| 20,000 | 66.0 | 96.0 | 96.6 | 96.3 | 96.7 |
| 40,000 | 82.9 | **97.4** | 97.1 | 92.6 (89.4 to 98.1) | 90.8 (88.9 to 97.5) |
| 50,000 | 85.1 | 96.4 | 96.6 | 90.8 | 90.1 |
| 100,000 | 91.9 | 87.0 | 86.5 | **96.7** (95.5 to 98.0) | **96.3** (95.2 to 97.5) |
| 150,000 | 91.5 | 92.6 | 91.3 | **97.8** | **97.4** |
| 200,000 | 94.4 | 90.4 | 89.9 | **98.0** (97.7 to 98.8) | **97.9** (97.4 to 98.6) |
| 250,000 | 95.2 | 90.5 | 91.0 | **98.1** | **97.9** |
| 300,000 | 95.0 | 77.8 | 81.8 | **98.5** | **98.3** |
| 350,000 | 95.0 | 76.4 | 80.4 | **98.6** | **98.7** |
| 400,000 | 95.5 (95.0 to 97.0) | 75.5 | 79.0 | **98.7** (95.5, 98.6, 98.7, 99.2, 99.4) | **98.8** (95.8, 98.4, 98.8, 99.2, 99.2) |
| mean of the last three | 95.4 | 76.5 | 80.4 | 98.6 | 98.5 |

With the relay system silent at the score the answer with lessons is 98.6 at 400,000 lines; the first cell has 95.0.

**The coordinator's hypothesis of step 1, on this design: the mark at 400,000 lines holds; the mark at 40,000 lines fails.** At 400,000 lines the joined system with lessons has 98.8 on lines it has met in no form (98.7 if scored pairs may be among the lessons): 3.3 above the relay system alone, 19.8 above the unchanged joined system, every seed at least 95.8. At 40,000 lines it has 90.8 (92.6), 6.6 (4.8) under the counting system alone: refuted there by the mark set.

P1 failed (92.6; I gave it 0.60). P2 held (98.7). P3 held. P4 held (every seed at least 95.5). P5 failed (at 40,000 and 50,000 lines 4.5 and 5.8 under the unchanged system). X1 held (98.8). X2 held (0.1 apart). X3 held (1.8 apart).

*The dip at 40,000 to 50,000 lines* is on four seeds of five (not on 406364). Looks: at 40,000 lines the class kinds alone answer 71 to 78% on three of the four seeds that dip and 88% on the fourth, against 92 to 95% at 100,000 lines; and with the cells that counted fewer than 64 times silent at the score the same systems answer 93 to 97%. The classes are found anew at 32,000 lines, and a class kind that is new starts with no trust and no parts (`merge.py`'s rule; `nano/table/PLAN_15.md` names the same fall). My account, not tested: the lessons have by then moved the answer on new lines onto the class kinds, so the system loses more than the unchanged one when they are replaced, and gets it back by the next reading but one. On the development seed the same fall came at 100,000 lines, after the finding at 64,000. **It is located and not repaired.**

**Two tasks** (the lives of `PLAN_3.md`; replay to the relay system as there; the lessons through both phases; scored pairs were not kept out of the lessons in these runs):

| Long life: addition 400,000, then subtraction 200,000 | Relay alone, with replay | Joined unchanged (`PLAN_3.md`) | **Joined with lessons** |
|---|---|---|---|
| addition, end of the first phase | 97.5 (96.6 to 97.9) | 80.4 | **98.9** (98.6 to 99.4) |
| addition, end of the second phase, after a subtraction line | 93.8 (91.8 to 95.0) | 58.4 (38.9 to 70.0) | **93.4** (91.8 to 94.7) |
| subtraction (clean), end of the second phase | **88.1** (87.0 to 92.3) | 75.6 | 81.3 (78.9 to 83.6) |
| share of the addition level kept | 96.3% | 72.6% | **94.7%** (92.8 to 95.3) |

| Short life: 40,000, then 20,000 | Relay alone, with replay | Joined unchanged | Joined with lessons |
|---|---|---|---|
| addition, end of the first phase | 89.9 | **97.2** | 93.8 (89.7 to 97.4) |
| addition, end of the second phase | 83.7 | **88.6** | 86.4 (84.3 to 93.2) |
| subtraction (clean), end | 64.0 | **88.9** | 88.2 (87.1 to 90.7) |
| share kept | 94.6% | 91.3% | 93.9% |

**The coordinator's hypothesis of step 3, on this design: in the long life the keeping holds (94.7% of the addition level; 90 asked) and the second task does not reach its mark (81.3; 88 asked); in the short life both hold (93.9%, 88.2).** Q1 held (98.9). Q2 held (94.7%). Q3 failed (81.3). Q4 failed: level with the relay system on addition after the second task (93.4 against 93.8) and above it after the first (98.9 against 97.5), but 6.8 under it on subtraction. Q5 held.

Readings (mine).
1. **From 100,000 lines of reading on, one system with both kinds of cell is above both of its parts at every reading (medians), and at the last reading on four seeds of five (on seed 116089 it has 95.8 and that seed's relay system 97.0)**: 96 to 99 where the relay system alone has 92 to 95.5 and the counting system alone 75 to 93. It does not fall with reading; it rises. This is the statement of the coordinator's hypothesis for the long reading, and it holds on lines kept out of everything.
2. **It does not hold at 40,000 to 50,000 lines**, where the unchanged counting system is 5 to 7 points better. To 20,000 lines the three are level. So no system of this line is at least as good as the better part at every length; the nearest is this one, with one located fault.
3. **The same lessons keep addition through a second task.** The unchanged joined system keeps 73% and its second answering cell alone as little as 16%; with lessons 94.7% is kept and addition after 200,000 lines of subtraction stands at 93.4, level with the relay system kept by replay. The keeping passes down a chain: the store's replay keeps the relay system, and the relay system's composed lines (kept by replay it still composes additions in the second phase; over the whole life 60 to 62% of its new lines are well-formed additions, and their share by phase was not counted) keep the answering cells' trusts for addition.
4. **The second task itself is learned less well than by the relay system** (81.3 against 88.1 in the long life), though better than by the unchanged joined system (75.6).
5. **How the two kinds of cell are joined here is not how the hypothesis pictured it.** The relay system's voice is worth nothing in the answer (silent at the score: the same to 0.1). It teaches: it composes lines that stay new, and its agreement says which answers of the pupil may be taught. The control of `PLAN_2.md` section 15 (one seed) shows the agreement is needed.
6. The teacher need not be good, and lessons need no right answers from outside: under a quarter of the composed new lines are right sums, the relay system answers 83% of never-shown sums at 40,000 lines, and nothing in a lesson comes from the program.

Limits. Sums only; one text, two tasks. The design was fixed on one development seed through five forms, four of which failed; the reported seeds were not used for any choice. The lives were not run with the scored pairs kept out of the lessons (on sums that made no difference). The control was run on the development seed only. The set of lines read is kept by the program. The relay system's silence at a line's first byte is `cells2.py`'s knowledge of where a line starts. The cost is twice the unchanged system's CPU. Not run: prose; other chunk sizes; lessons with the counting system as the only composer under this rule; a repair of the trust of class kinds found anew.
