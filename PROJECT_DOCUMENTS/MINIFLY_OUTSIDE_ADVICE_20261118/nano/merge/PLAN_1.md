# Line `nano/merge`, first plan: cells that say the same are merged into a class, and other cells are keyed on the class (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" The coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on making bold hypotheses, and try hard on the basis of facts."

**Source.** `claude/friendly-darwin-8mmhfm` at 71172671f; branch `claude/nano-merge-8mmhfm`; folder `nano/merge` only. The hypothesis is the coordinating session's; the form given to it below, the program and the predictions are this session's. Written before any program of this line was run. Development level; one development seed; no gate, seal or review is claimed.

**Copied or re-used, with the commit above.** The text of sums, the split and the form of the score: `loop_small_case/cells2.py` (re-written in `merge.py`, same random streams for the split). The rule of the cells that count and of the trust, with its 40 levels of experience: `nano/table/trust.py` (re-written with the number of byte values as a parameter). The measure on prose: `nano/table/score.py` (copied unchanged). The life on prose: `nano/table/life.py` and `nano/aheadprose/ahead.py` (`life`: each book's last tenth scored right after its stream). The three books are read from `minifly_v148_pretrain_probe_dev_20261025/data/` at the same commit (`git show 71172671f:<path>`; put in `data/`, not committed again).

## 1. The facts this starts from (as the coordinating session gave them; figures are the other lines')

Cells that count, read with a learned trust, keep what they read and learn in one reading. Their store grows as the bytes read to the power 0.80 and three quarters of the cells have been met once (`nano/table/growth.py`). On two-digit sums, cells keyed on at most four bytes answer 77 to 81% of never-shown sums; the gap is the tens digit, which needs the carry; look-ahead guesses lift it to 91% and do nothing on prose. Nothing in the system can say that two cells are alike.

## 2. The question and the one change

**If cells of one kind that are refined alike are merged into a class, and a new kind of cell is keyed on the class together with the refining lines, does the system answer combinations it was never shown, with nothing about the task laid down?**

*The base* is, on each task, the comparator as it stands, in one program (`merge.py`): a chain of delays; a kind of cell is a set of delays; a cell is made when its combination of byte values first occurs and counts the bytes that follow; it says log((c(v) + e) / (n + S e)); the answering cell holds a trust for each kind and each of 40 levels of experience and changes it by the mismatch. Sums: every set of at most four of nine delays (255 kinds), step 0.1 shrinking with use, e = 2, trusts born 0.02 (`nano/sparse`, its shrinking-step arm, the base of its look-ahead). Prose: runs of 0 to 6 bytes (7 kinds), step 0.00005, e = 0.02, trusts born 0.25 (`nano/table`, T0). **These base kinds and numbers differ between the two tasks and are inherited, not chosen here; that is a limit of this plan and is said once, here.** Everything below is one setting for both tasks.

*The change.*
1. **A pair** is a kind R of at least two delays, split into X (one or two of its delays) and Z (the rest). The cells of kind R are the refinements of the X-cells by the values of Z; nothing new is counted to find classes.
2. **Alike.** For a grouping of the X-cells into classes, each count of an R-cell (x, z) is foretold from the pooled counts of the *other* members of x's class at the same z: J = sum over R-cells and bytes v of c(v) log((C(v) - c(v) + e_J) / (N - n + S e_J)), where C, N are the class's pooled counts at z. A cell alone is foretold at 1/S. J is what the class would have said of each member's refinements had that member's own counts been left out; it is the measure of "wherever both have been refined by the same values of Z, their refinements say the same".
3. **Search.** X-cells are taken in the order of their counts; each joins the class that raises J most, or founds a class if none raises it; then two sweeps in which each X-cell is taken out and put where J is highest. The number of classes is whatever results. (A greedy search; more sweeps or restarts are where more computation could be spent.)
4. **Which pairs.** At each finding the candidates are the pairs whose R is among the 16 kinds with the largest trust (mean over uses) and has at least two delays. Each candidate is searched; its worth is J of the classes found less J of all X-cells in one class (what Z alone would foretell, measured the same way). The P pairs of largest positive worth are kept.
5. **The new kind.** For each kept pair, a cell for each (class of the X-cell, values of Z) that has occurred, holding the pooled counts of its members' R-cells and counting on from there. It is silent where the X-cell is in no class of two or more. It speaks like any cell; its kind's trusts are born at nothing.
6. **When.** Classes are found anew, from the counts as they then stand, when the bytes read have doubled since the last finding (first at 10,000 bytes). A pair that is kept again keeps its trusts; a new pair starts at nothing.

*Laid down, and what it costs in generality.* X has at most two delays (cost of search: an X of three delays has too many cells for the present search); pairs are taken inside one existing kind R (so two kinds are "one refines the other" only when their union is itself a kind); 16 kinds by trust; the schedule of findings; two sweeps. Nothing says which delays, which bytes, how many classes, or where in a line.

*Numbers chosen once on development material* (sums seed 1 at 40,000 lines; prose: cells taught on the stream of the test text before its last twentieth, scored on that twentieth): e_J in (0.02, 0.5) and P in (8, 32). Rule: the largest sums score among the settings that do not raise the prose loss by more than 0.002; ties to the lower prose loss. Then frozen.

## 3. Hypothesis (the coordinating session's), what refutes it, predictions (mine, before any run)

*Hypothesis.* With one frozen setting, classes found by the system and cells keyed on them (i) lift never-shown two-digit sums to at least 90% at 40,000 lines with no look-ahead, (ii) on each of the three books are at least 0.005 nats below the same system without classes, and (iii) leave fewer cells than the system without classes at equal or better loss when the R-cells of merged classes are withdrawn. *Refuted if* (i) stays under 83, or if no single setting does (i) and does no harm on prose.

| | Statement | Chance |
|---|---|---|
| M1 | sums, 40,000 lines, median of five seeds (mean of the last three readings): at least 90 | 0.50 |
| M2 | the same is under 83 (refutation) | 0.25 |
| M3 | the pair of the two units digits against the two tens digits is among the kept pairs at the end on at least four seeds of five, and each of its classes of two or more is pure for the carry (all members carry, or none does) | 0.50 |
| M4 | classes drawn by chance with the same sizes: within 3 points of the system without classes | 0.85 |
| M5 | classes found at half the stream and class cells counting only the second half: at least 85 | 0.45 |
| M6 | sums, 400,000 lines: at least 90 (two seeds if the cost does not allow five) | 0.30 |
| M7 | prose: each of the three books at least 0.005 nats below the base | 0.20 |
| M8 | prose: no book more than 0.002 above the base | 0.60 |
| M9 | with the R-cells of merged classes withdrawn: fewer cells and the score not lower on sums | 0.50 |
| M10 | the same on prose (loss not higher than the base on any book) | 0.15 |
| M11 | the whole hypothesis, (i) to (iii) | 0.12 |

Where I expect it to fail, if it fails: at the first findings two X-cells share few refinements (after 1,000 lines two pairs of units digits share about one pair of tens digits), so early classes will be impure; and on prose the classes may be real and the cells keyed on them earn little trust, because a shorter run already says what the class says.

## 4. Runs, size, cost, what is not run

Step 1 (this plan): `merge.py check` (compiled against numpy and plain Python); sums seed 1 at 40,000 lines with and without classes, with a look at what the classes are; the choice of numbers on both tasks. Step 2 (a second plan, written after step 1 is read): five reported seeds (73129, 93683, 116089, 406364, 545353) at 40,000 lines and at 400,000 as the cost allows; the two controls; the life on prose with and without classes and with classes by chance; the numbers of cells and counts and how they grow.

Size: sums base 255 kinds, about 1.2 million cells (as `nano/sparse` reports); prose base 7 kinds, about 0.6 million cells. The class kinds add at most P kinds and their cells; all are counted and reported. Cost unknown before the first run; my estimate is one CPU-minute for sums at 40,000 lines and ten for the life on prose. Four cores.

Not run: X of more than two delays; classes of classes; pairs across two kinds whose union is not a kind; subtraction and keeping; three-digit sums; word cells (T2) as the base on prose; look-ahead together with classes; the cap on a cell's weight.

## 5. Addition 1 (20 November; written after the first development run on sums was read and before the runs it describes)

*What was run* (development seed 1, 40,000 lines; `res/dev_sums_none.json`, `res/dev_sums_P32_eJ0.02.json`; `python3 look.py seed=1`): without classes 76.8 (mean of the last three readings; 17.8 CPU-seconds, 1,213,119 cells); with classes beside the plain cells, P = 32, e_J = 0.02: **90.8** (50.3 CPU-seconds of which 29.5 in the search, 1,231,004 cells). The check passes (`check_output.txt`).

*What the look shows.* At the last finding the pair (R = delays 3, 4, 6, 7; X = delays 3, 6: where the tens digit is named these are the two units digits against the two tens digits) is kept, with two classes of 53 and 47 pairs of digits; 98 of the 100 pairs of digits lie with their class by whether they sum to ten or more. The system also found a second grouping that I had not thought of: X = delays 4, 7 (the two tens digits) falls into ten classes of ten, the pairs with the same sum modulo ten; a cell keyed on that class and the two units digits also fixes the tens digit. On the 500 never-shown sums, with the hundreds digit given, the class kinds alone name the tens digit on 98.6%, the plain kinds alone on 75.8%, and all together on 92.8%: where the two units digits sum to exactly ten the plain kinds outvote the class kinds (49% right). So the classes are found; what is short is the trust. My reading of why: on a line that has been shown before, the plain cell of the whole pair answers, the mismatch is nothing and no trust moves; the answering cell is taught almost only on lines met for the first time, most of which come before the classes are pure.

*The one change.* I had built "a class cell beside its members' cells". The hypothesis says "merged", and its part (iii) says "withdrawn". **Replacement:** where the X-cell of an R-cell is in a class of a kept pair, the R-cell is silent (it still counts, because the search for classes at the next finding reads its counts; the number of cells is reported both with these cells and without them). The class cell stands in its place. A shown line is then answered through the class, as a never-shown line is, and the mismatch on shown lines teaches the trust of the class kinds. `rep=1` in `merge.py`. Nothing else changes; the same on both tasks.

*Predictions (mine).* A1: sums, development seed, replacement: at least 96 (chance 0.50); under 90.8, that is worse than classes beside the cells (0.15). A2: prose, last twentieth of the stream: replacement is more than 0.002 nats worse than the base's 1.5566 (0.55), because a run of six bytes is then answered by a class cell that pools other runs. A3: prose with classes beside the cells is within 0.002 of the base (0.6).

*The choice of numbers* is widened by this one switch: rep in (0, 1) beside e_J in (0.02, 0.5) and P in (8, 32), by the rule of section 2.

## 6. Outcome of step 1 (entered 20 November after the runs; `res/dev_*.json`, table in `report_dev.txt`, check in `check_output.txt`)

Development seed 1 on sums (never-shown additions, own answer bytes, whole answer, %, mean of the last three readings of 40,000 lines); prose on the last twentieth of the stream (nats a byte). Cells: those that have counted; "without" leaves out the R-cells whose X-cell is in a class of a kept pair.

| Setting | sums | cells (without) | CPU s | prose | cells (without) | CPU s |
|---|---|---|---|---|---|---|
| no classes | 76.8 | 1,213,119 | 18 | 1.5566 | 445,161 | 3 |
| beside, P = 8, e_J = 0.02 | 82.2 | 1,220,018 (1,167,763) | 58 | 1.5542 | 666,152 (478,254) | 69 |
| **beside, P = 32, e_J = 0.02** | **90.8** | 1,231,004 (1,188,731) | 50 | **1.5488** | 2,292,678 (1,847,612) | 94 |
| beside, P = 8, e_J = 0.5 | 78.5 | 1,217,831 (1,159,825) | 56 | 1.5539 | 498,228 (310,772) | 62 |
| beside, P = 32, e_J = 0.5 | 89.6 | 1,228,296 (1,175,694) | 48 | 1.5512 | 1,807,364 (1,362,269) | 83 |
| replacement, P = 8, e_J = 0.02 | 81.1 | 1,217,832 (1,174,198) | 57 | 1.5637 | 666,152 (478,254) | 68 |
| replacement, P = 32, e_J = 0.02 | 93.8 | 1,230,459 (1,160,902) | 55 | 1.5612 | 2,292,678 (1,847,612) | 91 |
| replacement, P = 8, e_J = 0.5 | 77.2 | 1,216,791 (1,162,605) | 55 | 1.5950 | 498,228 (310,772) | 60 |
| replacement, P = 32, e_J = 0.5 | 84.7 | 1,227,416 (1,136,785) | 55 | 1.6051 | 1,807,364 (1,362,269) | 83 |

**Chosen by the rule of section 2 and frozen: classes beside the cells (rep = 0), P = 32, e_J = 0.02.** Every setting with replacement raises the prose loss by more than 0.002 and is excluded by the rule; among the rest this one is highest on sums and also lowest on prose.

Addition 1's predictions: A1 failed (replacement 93.8 where at least 96 was predicted; it is above 90.8, so the second part held). A2 held (replacement is 0.0046 worse than the base on prose). A3 failed in the direction I did not expect: classes beside the cells are 0.0078 nats *better* than the base on prose, not within 0.002.

Readings (mine; one seed, development level).
1. **The classes come out as the carry, and as something more.** Inside the kind of the whole pair the system finds the two units digits grouped by whether they sum to ten or more (98 of 100 pairs with their class on this seed), and the two tens digits grouped by their sum modulo ten (ten classes of ten). Either gives the tens digit of a never-shown pair; the cells keyed on classes alone name it on 98.6% (99.8% with replacement). Nothing about sums was laid down.
2. **The answering cell does not yet use them fully**: 92.8% (95.6% with replacement) of tens digits with all kinds voting. With replacement the shown lines too are answered through the class and the trust learns more, which helps sums and harms prose.
3. **On prose, replacement harms and classes beside the cells help.** A run of six bytes says something its class does not.
4. **More pairs are better on both tasks** (P = 8 to 32: 82 to 91 on sums, 0.002 to 0.008 nats on prose); the grid stopped at 32.
5. **The store is not smaller.** On prose the class cells are four times the base's cells; leaving out the R-cells that a class stands for still leaves four times the base (1.85 million against 0.45 million), and scoring with those cells silent, without having learned so, gives 2.58 nats. Part (iii) of the hypothesis will be scored in step 2 but on this evidence it fails on prose.
