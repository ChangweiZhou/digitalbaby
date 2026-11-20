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
