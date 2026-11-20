# Line `nano/merge`, seventh plan: the answering cell; a class kind is trusted by whether the member's own cell speaks, and practises the case that it does not (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Written before any run of `merge7.py` other than its check (`check7_output.txt`). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

1. On two-digit sums the class kinds alone answer never-shown sums better than all kinds together (94.6 against 92.2, five seeds; tens digit 98 to 99% against 92%). The class cells are right; the answering cell does not trust them enough where it matters.
2. Why (my reading, `PLAN_1.md` addition 1): a class cell matters exactly where the cell it stands for (the R-cell of the same place) has never been met. On a line that has been shown, that R-cell answers, the mismatch is nothing and no trust moves. The trust of a class kind for the case "the member's own cell is absent" is taught only at first meetings, which on this text are few and early.
3. Three ways of making the shown lines teach it were tried and each harmed prose: silencing the members' cells for good, silencing those the class foretells well, silencing at half the bytes. In all three the system that answers is not the system that learned, or has lost cells that say something.

## 2. The one change (to the answering cell; the classes are the frozen setting's)

**A class kind has two rows of trust**, one used where the R-cell of the same place has counted and speaks, the other where it does not. Each row is changed by the mismatch where it is used, as before. In addition, **practice**: while reading, at every byte and for every kind R that speaks and has class kinds made from it speaking beside it, the answer is formed once more with that one R-cell left out and its class kinds on their second row; the mismatch of that second answer changes those second rows and nothing else. No cell is silenced in the answer the system gives or learns its other trusts from; nothing is added to what a cell holds. No new number. `merge7.py` (a subclass; the compiled loop against numpy in its check).

The second row is the system's answer to "what would I say here had I not met this combination", asked at every byte and taught by the byte that arrives.

## 3. Hypothesis, what refutes it, predictions (mine, before any run)

*Hypothesis.* With the frozen setting's classes and this answering cell, never-shown two-digit sums reach at least 95% at the median of five seeds at 40,000 lines, and no book is more than 0.002 nats above the frozen setting. *Refuted if* sums are under 93, or a book is more than 0.002 above.

| | Statement | Chance |
|---|---|---|
| Y1 | two-digit sums, development seed: at least 95 | 0.45 |
| Y2 | two-digit sums, five seeds, median: at least 95 | 0.35 |
| Y3 | the same under 93 (refutation) | 0.35 |
| Y4 | prose, last twentieth of the stream: not more than 0.002 above the frozen setting's 1.5488 | 0.65 |
| Y5 | prose: at least 0.002 below it | 0.20 |
| Y6 | three-digit sums, five seeds: within 1 point of the frozen setting's 94.1 either way | 0.60 |
| Y7 | 400,000 lines, two seeds: at least 90 on both | 0.20 |

## 4. Runs, size, cost

Development material first (two-digit seed 1; prose's last twentieth). If on development material it is not below the frozen setting by more than 1 point on sums nor above it by more than 0.002 nats on prose: the five reported seeds on two-digit sums, the first life, five seeds on three-digit sums, and 400,000 lines on two seeds. Otherwise it stops there and that is the result. The second rows add 32 x 40 trusts. Cost by estimate: as the frozen setting plus a few percent. Not run: the same for the kinds keyed on two classes; a second row for plain kinds.

## 5. Addition 1 (20 November; written after the two development runs were read, before the runs it describes)

*What was run* (`res/dev7_sums.json`, `res/dev7_prose.json`): two-digit sums, development seed, **90.8** (the frozen setting: 90.8; every reading within a point of it); prose's last twentieth **1.5467** (the frozen setting 1.5488). Y1 failed; Y4 held; Y5 held narrowly (0.0021 below).

*What the result file shows.* The second rows are used (5.3 million uses in all) and end only a little above the first rows (0.31 to 0.32 against 0.24 to 0.25 for the two class kinds that fix the tens digit). My reading: on sums the base's step for a trust is 0.1 over the square root of one plus its *uses*, and a class kind is used, and practised, at every byte of every line, nearly always with no mismatch; after 40,000 lines its step is a few ten-thousandths, and the one byte in a hundred where the practice answer is wrong can no longer move it. The rule that shrinks the step was chosen in `nano/sparse` for kinds that are all born together; it starves a row that is used often and wrong seldom.

*One change, to test that reading*: `step=2`: the step of a trust is 0.1 over the square root of one plus **the sum of the squares of the mismatch that trust has met**, in place of its uses. A trust that has seldom been wrong keeps its step for when it is. Applied to every trust of the sums base (prose has a constant step and is not touched). The 0.1 is the base's and is not chosen again. Because it changes the base, the base is run with it too.

Arms, development seed 1, two-digit sums, 40,000 lines: (a) no classes, `step=2`; (b) the frozen setting's classes with one row, `step=2`; (c) two rows and practice, `step=2`.

*Predictions (mine).* Z1: (a) within 3 points of the base's 76.8 (0.55). Z2: (c) at least 95 (0.40). Z3: (c) at least 2 points above (b) (0.50). *If Z2 or Z3 holds* and (a) is not more than 3 points below 76.8, arm (c) and its base (a) are run on the five reported seeds, with the first life unchanged from the run of section 4 (the step on prose is not touched); otherwise it stops here.

## 6. Outcome (entered 20 November; `res/dev7_*.json`, `out_dev7_*.txt`, `python3 look7.py seed=1 [step=2] [rows=1]`). The plan stops on development material.

Development seed 1, two-digit sums, 40,000 lines, mean of the last three readings (last); prose's last twentieth.

| | sums | prose |
|---|---|---|
| no classes | 76.8 (76.2) | 1.5566 |
| the frozen setting | 90.8 (90.6) | 1.5488 |
| two rows and practice (this plan) | **90.8** (90.6) | **1.5467** |
| (a) no classes, step by the mismatch met | 80.3 (79.2) | not touched |
| (b) classes, one row, step by the mismatch met | 91.1 (90.8; 94.0 at 20,000 lines) | not touched |
| (c) two rows and practice, step by the mismatch met | 91.1 (90.8) | not touched |

**The hypothesis is refuted on development material** (sums under 93 and no gain at all); by section 4 and addition 1 nothing is run on the reported seeds or the books. Y1 failed, Y2 and Y3 are not scored on five seeds (NOT RUN), Y4 held, Y5 held narrowly (0.0021), Y6 and Y7 NOT RUN. Z1 failed in the direction I did not expect (the base is 3.5 points *higher* with the new step), Z2 failed, Z3 failed (arms (b) and (c) are equal).

**Why, and a correction of my own earlier reading** (`look7.py`, development seed; the tens digit with the bytes before it given, 500 never-shown sums and 500 shown sums):

| | the cell of delays 3, 4, 6, 7 at that place has counted | tens digit right | the same with every cell of that kind left out |
|---|---|---|---|
| never-shown sums | on 98.6% | 92.8% | 93.4% |
| shown sums | on 100% | 98.4% | 98.2% |

1. **On a never-shown sum the cell of "the whole pair" has nearly always counted.** A kind is a set of delays with no place in the line; the same four delays hold four digits at two other places of every line (where the second number's first digit is named they are the first number's two digits and two digits of the line before). So that cell exists and speaks, with counts of a digit that follows it by chance. "The member's own cell does not speak" happens on 1.4% of never-shown sums; the second row is all but never used where it was meant for. This is why the plan could not work on this text, whatever the step.
2. **A shown sum is not answered by one cell that remembers it.** With every cell of the kind of delays 3, 4, 6, 7 left out, shown sums are still 98.2% right and never-shown ones 93.4%. The trace of a shown line is in every cell that holds a part of it (a cell of three of its digits has counted that line's own tens digit five times). **My reading in `PLAN_1.md` (addition 1), repeated in `PLAN_4.md` and `PLAN_7.md` section 1, that the plain cell of the whole pair answers shown lines, was wrong in its mechanism.** What stands of it: shown lines are answered right with little help from the class kinds, so the mismatch that would teach the class kinds is rare (1.6% of shown tens digits against 7% of never-shown ones).
3. What practice would have to be, to ask "what would I say had I not met this combination": not one cell left out, but that cell's counts taken out of every cell it refines (they are contained in them, since it refines them). That is the measure J of the search, applied to the answer. NOT BUILT; and it needs a way to know at answering time that a combination is new, which on this text the counts of one kind do not give.
4. On prose the second row does a little (0.0021 nats on development material): there a run of six bytes that has not been met is a real case, and the class kinds get a trust of their own for it. Not run on the books.
5. A side observation, one seed, not followed: with the step of a trust falling with the mismatch it has met rather than with its uses, the plain base on two-digit sums has 80.3 against 76.8. That is a number of the base (`nano/sparse`'s), not of this line.
