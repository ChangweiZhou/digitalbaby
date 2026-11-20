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
