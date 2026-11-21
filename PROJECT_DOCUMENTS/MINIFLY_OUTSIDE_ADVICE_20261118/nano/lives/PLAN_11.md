# Addition, then English, then addition again: three learners, and where a rule that was learned is lost (21 November 2026)

**Authorisation and steer.** Mr. Xu's words of this night, in Chinese: "从第一原理来说，学了语文所用的部分和学了数学所用的网络部分可能有重叠，但理论上说并不多。为何会反复出现遗忘现象？反过来说，如果我们可以做到泛化，网络应该学到了加法的规律（而不只是加法的知识或如何去拼凑下个字节程度的规律），遗忘也就应该不再是一个问题。现在问题出在哪儿？" (my translation: "From first principles, the part of the network used after learning language and the part used after learning mathematics may overlap, but in theory not much. Why does forgetting keep appearing? Conversely, if we can generalise, the network should have learned the rule of addition (and not only knowledge of additions, or a rule at the level of piecing together the next byte), and forgetting should then no longer be a problem. Where is the problem now?"); "我们反复强调是local learning" ("we have stressed again and again that it is local learning"); and "这些你自己决定，不需要问我。一切从第一原理和事实出发。" ("Decide these yourself, you need not ask me. Everything from first principles and from facts."). The choice and design of this step are mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Written before any run of its programs. Development level; no checker gate, seal or review is claimed.

## 1. What is known, and the question

* In both learners of this programme the loss of an earlier task has been located in the same place: the answering cell's reading of cells that the tasks share. For the relay network: after a second task "the cells in between still carry most of the first (67% with the answering cell alone taught again)" (roadmap, row K1; `loop_small_case/PLAN_13.md`). For the counting system: three quarters of the loss on a stream that returns is in the trusts (`PLAN_1.md`).
* **A trust, as I built it, is not one connection's strength.** It is one number for every cell of a kind at a level of counts (`nano/table/PLAN_3.md`, roadmap row L-C; entered there as "added from the evidence"). Read as connections from cells to the answering cell, that is one strength shared by many connections; read as a path, it is many cells passing through one connection. Either way every kind of text moves the same numbers. I brought this in without setting it before Mr. Xu as a departure from local learning; a helper's reading of 20 November had named it ("no synapse is shared") and I took that as a matter of wording only. The row L-C itself records that at the time it gained nothing over a reading with no learned weight (1.5226 against 1.5225).
* A learner that had the rule still lost it: the relay network answers 95.5% of never-shown additions and keeps 2% of them through subtraction (roadmap, sections 9 and 12). But subtraction is the hardest case, since the same bytes ask for other answers.

Mr. Xu's first principle gives a sharper test. **English text has almost none of the bytes of sums** (tiny Shakespeare: 27 digits, no "+", no "=" in 1.1 million bytes). So what a learner uses for English and what it uses for addition overlap as little as two kinds of text can. If a learner that has read sums, then reads English, has lost its never-shown additions, the loss cannot be in what it stored about sums; it can only be in what the two share by construction.

## 2. The life and the three learners

**The life.** Phase A: lines of two-digit addition ("ab+cd=efg", 8,000 shown pairs drawn with replacement; the split, the text and the score of `nano/suite/suite.py` and `nano/table/gate.py`). Phase B: the first 400,000 bytes of tiny Shakespeare, once (as many bytes of English as phase A has of sums for the counting learners; with 256 kinds of cell a longer text would also make tens of millions of cells). Phase A again: 5,000 more lines of the same shown pairs. **Measured**, in forked copies or on copies of the state: never-shown additions (500 pairs never in a shown line; the whole answer named in the learner's own bytes, each after a read line; five draws of the line before) at the end of phase A, at the end of phase B with nothing changing, and after 500, 1,000, 2,000 and 5,000 lines of the return.

**Learner 1: cells that count, read through shared trusts** (the programme's system as it stands). Kinds: every set of at most four of the last nine places (256 kinds), as the programme's sums systems have; smoothing 2, trusts born at 0.02, the rule of the step with its number 0.03 (`nano/suite`'s arm in which only the step is replaced by the rule: 81.3 and 91.1 for the two answering cells at 40,000 lines, as that line reports). Two forms: with the second answering cell (S2) and with the first alone (S1). Phase A: 40,000 lines.

**Learner 2: the same cells, every connection with a strength of its own** (`own.py`). No number is shared. Each cell holds its own trust, born at 0.02, and the sum of its own squared drivers; when the cell speaks, its trust moves by the answering cell's mismatch on its own message, with step 0.03 x driver / root of (1 + its own sum). The sum starts at one because a cell that has spoken once must not move by a full step on one driver (the well-conditioned form of `nano/table/PLAN_22.md`, addition 1). No second answering cell (its parts are shared numbers). Form P. A control, form F: every trust stays at 0.02 for ever (no reading is learned at all). Phase A: 40,000 lines. The step's number for P is chosen on the development seed among 0.03, 0.1 and 0.3 by the never-shown additions at the end of phase A, then frozen.

**Learner 3: the relay network** (the learner that generalises; `nano/loop/loop.py`'s system, which is `loop_small_case/cells2.py`'s with the alphabet as an argument; 256 cells, ten places, its numbers as there: 1/4, 1/16, fading 1e-5; least-squares answering cell as on sums). One alphabet for both texts (the byte values of either). Phase A: 400,000 lines (the length at which it answers 95%). Two forms: as it is (R); and with the fading of its answering strengths switched off from phase B on (R0), since a fading of 1e-5 a byte alone takes those strengths to a fiftieth over 400,000 bytes in which sums never come. A diagnostic on R0 after phase B: the return read with the cells in between held still (only the answering cell learns), beside the return read as usual.

## 3. Hypotheses, what refutes them, predictions (mine)

**H1 (the shared reading is where a non-overlapping task is lost).** Learner 1 loses more than half of its never-shown additions through English (both forms). **Refuted if** it keeps more than three quarters: then shared trusts do little harm between texts that share no bytes, and tonight's loss between streams has another cause.

**H2 (a strength for every connection keeps).** Learner 2 (P) keeps its never-shown additions through English to within 2 points. **Refuted if** it loses more than 5 points. Its price is its level at the end of phase A against learner 1's.

**H3 (having the rule is not enough).** The relay network loses more than half of its never-shown additions through English even with the fading off (R0). **Refuted if** R0 keeps more than three quarters: then Mr. Xu's statement holds for this learner, a rule that was learned is kept through a text that does not overlap with it, and what was lost through subtraction was lost to the conflict of answers.

| | Statement (five reported seeds; median) | Chance |
|---|---|---|
| X1 | H1 holds for S1 and for S2 | 0.75 |
| X2 | Learner 1 is back within 5 points of its level after 5,000 lines of the return | 0.45 |
| X3 | H2 holds | 0.80 |
| X4 | P ends phase A within 10 points of S1 | 0.40 |
| X5 | F (no learned reading) ends phase A at least 10 points under P, and loses nothing through English (within 1 point) | 0.50 |
| X6 | R (fading on) keeps under 10% of never-shown additions through English | 0.85 |
| X7 | H3 holds (R0 loses more than half) | 0.65 |
| X8 | R0's return with the cells in between held still comes back as fast as its return read as usual (within 5 points after 5,000 lines): the cells in between kept the sums, the reading lost them | 0.55 |

## 4. Runs, size, cost, what is not run

Development seed 1 (the choice of P's number; a look that the relay network in the wider alphabet learns sums as `cells2.py`'s does: about 83% at 40,000 lines and 95% at 400,000, else the learner is not the one that generalised and the plan stops there for it). Reported: the seeds 73129, 93683, 116089, 406364, 545353, all forms. Size: learner 1 and 2 about 1.2 million cells after phase A (as `nano/suite` reports for sums) and more after English; the relay network 256 cells with about 250,000 strengths. Cost: about 5 CPU-minutes a form a seed for the counting learners, 3 for the relay network. **Not run:** other texts as phase B; subtraction as phase B for learner 2; learner 2 on the six streams (the next plan if H2 holds); remembered trusts on learner 1; a relay network with sparse cells.

## 4a. The look at the relay network, and one correction (development seed, before any run of the life that is scored)

Never-shown sums after 40,000 lines, whole answer after a read line: with the step 1/4 that section 2 names, 64.8% in the alphabet of the sums alone and 59.0% in the alphabet of both texts; with the step 1, 85.0% and 83.5%. The recorded level of `cells2.py`'s plain system is 82.9% (median of five seeds; `loop_small_case/ERRATA_20261120.md`). So the number in section 2 was my mistake about which value the recorded runs used: **the step is 1** (the value in `nano/loop/loop.py`), and with it the network in the wider alphabet is the learner that was recorded. Nothing else is changed. The wider alphabet costs it 1.5 points.

## 4b. The development runs, the choice of P's step, and additions made before the reported runs

Development seed 1; not a result. Never-shown sums, whole answer, per cent (`out/aba_dev_*.txt`, `out/relay_dev_*.txt`):

| Form | end of the sums | end of the English | return: 500 lines | 1,000 | 2,000 | 5,000 | English, nats a byte, last tenth |
|---|---|---|---|---|---|---|---|
| S1 (shared trusts, first cell) | 81.2 | 77.3 | 78.1 | 78.4 | 79.1 | 78.9 | 1.509 |
| S2, second cell (its first cell is S1) | 89.4 | 88.0 | 90.0 | 90.0 | 91.0 | 91.3 | 1.504 |
| P, step 0.03 | 73.0 | 73.2 | 74.1 | 73.6 | 73.8 | 73.0 | 1.559 |
| P, step 0.1 | 78.4 | 77.8 | 78.0 | 78.0 | 78.3 | 77.3 | 1.741 |
| P, step 0.3 | 79.5 | 78.7 | 79.4 | 79.6 | 79.4 | 78.6 | 2.774 |
| F (no trust learned) | 34.7 | 34.7 | 33.9 | 34.4 | 34.3 | 32.7 | 2.355 |
| R (relay network) | 96.0 | 0.0 | 45.9 | 82.9 | 94.2 | 94.2 | |
| R0 (fading off from the English on) | 96.0 | 0.0 | 39.4 | 78.8 | 92.0 | 94.3 | |
| R0, return with the cells in between held still | | | 26.5 | 68.5 | 91.0 | 93.3 | |

The relay network at 40,000 lines: 83.5 (the look of section 4a holds).

**The choice of P's step.** By the rule of section 2 it is **0.3** (79.5 against 78.4 and 73.0). The best is at the edge of the set of three; no larger step was tried. The rule did not look at the English, and there the same learner with 0.3 reads far worse: 2.774 nats a byte in the last tenth, which is worse than in its own first tenth (2.629), against 1.741 with 0.1, 1.559 with 0.03, and 1.509 for the shared trusts. So, on this seed, no one number serves both texts when each cell learns its trust from its own events alone. **Addition 0:** the reported runs carry P at all three steps. The plan's P is the one at 0.3; the other two are set beside it for this reason and are additions.

**Additions.** They were written after the development runs in the table. One slip of mine: I tried their programs once on the development seed before writing this section, so what I expected of them was formed with seed 1 in view; I state no chances for them and count none as a prediction. The figures of seed 1 are given so that the reader has what I had.

1. **S1, S2: the trusts put back.** At the end of the English, in a forked copy, the trusts (and the second cell's parts) of the end of the sums are put back; the counts stay as the English left them. It says how much of the loss is in the shared numbers. (Seed 1, S1: 81.2, which is the level before the English.)
2. **R, R0: parts put back.** At the end of the English, in copies: (a) the answering strengths of the end of the sums put back; (b) the cells in between of the end of the sums put back (their strengths from the lines and their thresholds), the answering strengths as the English left them; (c) the answering strengths and the cells' thresholds put back, so that only the cells' strengths from the lines are as the English left them. (Seed 1: 80.2, 0.0, 95.7.)
3. **R, R0: a yardstick for the return.** Marks after 500, 1,000, 2,000 and 5,000 lines of the first reading; and the same lines read by a copy whose cells in between stay as they were born, with only the answering cell learning. (Seed 1, after 5,000 lines: 5.4 and 0.6; the return after English reaches 94.3, and 93.3 with the cells in between held still.)
4. **R, R0: how many cells are active.** In the 7,500 states in which the never-shown answers are named and in 5,000 states of the English: the mean activity, the number of cells above one half, the cells whose mean activity through the English is under 0.05 and the share of the sums' activity that sits on them, each before and after the English; and by how much the answering strengths of the ten digit branches changed. (Seed 1: 108 of 256 cells above one half in a state of the sums and 90 in a state of the English before the English is read; the digit branches' strengths changed by 0.88 of their size.)

Not changed: the hypotheses, X1 to X8, the forms, the seeds, the scoring. `python3 aba.py check` was run again after the change to the program (`check11_output.txt`; 7 passed; the kernel is the same text).

## 5. Outcome

(to be entered after the runs)
