# A strength that has been changed much changes little: the answering strengths stiffen, each by its own record (21 November 2026)

**Authorisation and steer.** As `PLAN_11.md`, where Mr. Xu's words are quoted in full; the one that governs this step: "这些你自己决定，不需要问我。一切从第一原理和事实出发。" (my translation: "Decide these yourself, you need not ask me. Everything from first principles and from facts."). The choice and design of this step are mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Written with `PLAN_12.md`'s first reported seeds and `PLAN_13.md`'s development seed in view (its reported seeds are running), before any run of `relay14.py` other than its check (`check14_output.txt`, passed). Development level; no checker gate, seal or review is claimed.

## 1. The facts this follows

From `PLAN_12.md` and `PLAN_13.md` (development seed 1 unless said; the network with the four changes):

* **The cells in between can be shared without harm.** The sums use 254 of the 256 cells; the English uses about 110, and every one of them is a cell of the sums. Yet with the answering strengths of before each English put back the network names 90.8, 95.9 and 97.7% of the never-shown sums after the three pieces of English (98.9, 98.5 and 98.9% before them). A cell's strengths from the lines of one text are not touched by the other text, and its threshold is held.
* **The answering strengths cannot be shared in the same way.** There is one strength for each pair of a cell and a branch, and it cannot tell why its cell is active. Through the shared cells the English teaches the digit branches down wherever one of them speaks in a state of the English, and the branches of the English speak in the states of the sums where they were never taught. After the first English the right digit's branch gives 0.07 (0.95 before) and the loudest other branch 0.43; whole answers stand at 0.7%, with the reading among the digits at 88%.
* **Returns help, and do not cure.** Whole answers straight after the three pieces of English: 0.7%, 40.2%, 43.3%; after 100 lines of each return 88%, 93%, 96%. The right digit's branch gives 0.07, 0.43, 0.56 after the three pieces. Each return gives the digit branches back what the English took, through all the cells that are active, the shared ones among them; each piece of English takes it again.

**What a nervous system has for this (my reading of textbook knowledge; I cite no paper).** A synapse that has been built up and used for long resists change, and new learning falls on synapses that are still free. In this network: the strengths that hold the reading of the sums have been driven four million times when the English begins; the strengths from the same cells to the branches of the English have never been driven. If a strength's step falls with its own record of change, the English is written into the free strengths and cannot take much from the old ones; and when the sums return, what silences the branches of the English in the states of the sums falls on the strengths from cells the English never drove, where the English cannot undo it.

This is the rule the counting systems of this programme already use for their trusts (a step divided by the root of the sum of the number's own squared drivers; `nano/table/PLAN_22.md`, `nano/suite/PLAN_4.md`), applied to each answering strength of the relay network. It is local: each strength keeps one more number, its own.

## 2. The change, exactly (`relay14.py`)

**Letter s.** Every answering strength, and every branch's constant, keeps G, the sum of its own squared drivers (the branch's mismatch times the sending cell's activity; for a constant, the mismatch). Its step is the step it has now, divided by the root of (1 + G / kappa). At birth nothing is changed; a strength whose G is a hundred times kappa moves a tenth as far. The return strength of the same connection takes the same change, as before. Nothing else is changed: the four changes of `PLAN_12.md` stay (theta 4), and so do the steps 1 and 1/16. With s off the kernel does `relay12.py`'s arithmetic bit for bit (checked).

**kappa** is chosen on the development seed among 1, 10, 100 and 1,000 by learning alone: the smallest kappa (the strongest stiffening) with which the network (a) ends the sums within 1 point of the form without s and (b) names right on the held-out English, at the end of the first English, within 2 points of it. If none does, the change fails in development, that is recorded, and no reported run is made.

**The life and the marks** are `PLAN_13.md`'s (three returns). The form without s is `PLAN_13.md`'s Rgtqr; one development run here must repeat that plan's development figures exactly, and its reported seeds are taken from that plan and not run again. Added: how far the strengths have stiffened (the divisor of the step, at the median over the digit branches' strengths and over the other branches' strengths that have been used; the share never used).

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** With stiffening answering strengths the network keeps its sums through the English from the second return on: straight after the second English, with nothing changing, it names three quarters or more of the never-shown sums it named before it (median of five seeds). At the first return the branches of the English have never been taught in the states of the sums, so less is asked there: a quarter or more.

**It counts only if** the form learned the sums and the English as the form without s did (conditions (a) and (b) of section 2, on the reported seeds).

**Refuted if**, having learned both, it keeps under a quarter at the second return.

| | Statement (five reported seeds; median; "kept" = whole answers straight after an English over those straight before it) | Chance |
|---|---|---|
| U1 | It learns as the form without s does: (a) and (b) hold on the reported seeds | 0.65 |
| U2 | First return: a quarter or more kept | 0.60 |
| U3 | First return: half or more kept | 0.35 |
| U4 | Second return: three quarters or more kept (the hypothesis) | 0.50 |
| U5 | Third return: three quarters or more kept | 0.55 |
| U6 | After the first English the right digit's branch still gives 0.5 or more in the states of the sums (0.01 to 0.08 without s) | 0.65 |
| U7 | After the second and the third English a branch of another byte is the loudest in under a tenth of the states of the sums (0.26 and 0.22 without s, development seed) | 0.45 |
| U8 | It learns the later English no worse: bytes named right while English 2 and English 3 are read are within 1 point of the form without s | 0.55 |

**The known price, not measured here.** A strength that resists change also resists being corrected when what it holds has become false. Nothing in this plan measures that (property (c), revising); a plan that does is owed before stiffening could be recommended.

## 4. Runs, size, cost, what is not run

Development seed 1: Rgtqr once (must repeat `PLAN_13.md`'s development figures); Rgtqrs at kappa 1, 10, 100, 1,000. Reported: the seeds 73129, 93683, 116089, 406364, 545353, Rgtqrs at the chosen kappa. Size: 256 cells; about 230,000 strengths and 19,275 sums of squared drivers. Cost: about 3 CPU-minutes a run; ten runs. **Not run:** stiffening of the cells' strengths from the lines (they are not where the loss is); the step taken wholly from the record, with no kappa; subtraction after addition; any test of revising; a life with more kinds of text.

## 4a. The development runs and the choice of kappa (development seed 1; before any reported run; `out/relay14_dev_*.txt`)

Never-shown sums, whole answer, per cent.

| Form | End of the sums | Straight after English 1, 2, 3 | The same with only the digit branches answering | After 100 lines of return 1, 2, 3 | Held-out English at the end of English 1 | Right digit's branch after English 1, 2, 3 | Loudest other branch |
|---|---|---|---|---|---|---|---|
| without s (`PLAN_13.md`'s run, repeated here exactly) | 98.9 | 0.7, 40.2, 43.3 | 88.0, 95.0, 96.4 | 88.1, 93.4, 96.3 | 46.3 | 0.07, 0.43, 0.56 | 0.43, 0.24, 0.27 |
| s, kappa 1 | 89.4 | 0.0, 0.2, 0.7 | 80.7, 86.0, 86.6 | 0.0, 3.2, 6.8 | 43.3 | 0.11, 0.16, 0.16 | 0.30, 0.21, 0.19 |
| s, kappa 10 | 93.7 | 0.0, 14.4, 23.3 | 90.3, 91.4, 92.0 | 2.4, 39.4, 48.2 | 47.0 | 0.06, 0.28, 0.33 | 0.42, 0.24, 0.23 |
| s, kappa 100 | 96.8 | 0.0, 15.7, 61.1 | 90.6, 92.8, 94.6 | 3.8, 57.8, 86.4 | 47.4 | -0.07, 0.38, 0.51 | 1.07, 0.38, 0.18 |
| s, kappa 1,000 | 98.7 | 0.0, 27.2, 67.2 | 94.2, 94.5, 95.5 | 43.4, 87.4, 93.8 | 47.2 | -0.12, 0.36, 0.51 | 0.90, 0.27, 0.11 |

**The choice.** Condition (a), the sums within 1 point of the form without s, is met by kappa 1,000 alone; condition (b) by 10, 100 and 1,000. By the rule of section 2, **kappa = 1,000**, the weakest stiffening of the set and at its edge.

**What the development seed already says (mine; one seed; not a result).** The reasoning of section 1 was wrong in its main point. Stiffening does not keep the digit branches from sinking through the first English: the right digit's branch ends at about nought at every kappa, as it does without s. The English asks the digit branches to be silent in every state of the English, and it goes on asking for 400,000 bytes; a smaller step only makes that take longer, and it also makes the return slower (43% after 100 lines of the first return against 88%). What stiffening does do is the second thing section 1 expected: from return to return the branches of the English grow quieter in the states of the sums (0.90, 0.27, 0.11), and at the third return two thirds of the whole answers are there straight away (67.2 against 43.3). The stronger the stiffening, the worse the sums are learned in the first place. The reported seeds are run at kappa 1,000 as the plan binds me to.

## 5. Outcome

(to be entered after the runs)
