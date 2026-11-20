# Line `nano/merge`: cells that are refined alike are merged into a class, and other cells are keyed on the class. Report of 20 November 2026

By a separate agent session started by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc); the frame is `../../ROADMAP_SECOND_REVISION_20261120.md`. **Authorisation:** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on making bold hypotheses, and try hard on the basis of facts.") **Status:** development level. One development seed for choosing numbers; five reported seeds (73129, 93683, 116089, 406364, 545353) for what is said of sums, with a repeat on five more; six books for what is said of prose; nothing repeated by anyone else; no gate, seal or review is claimed. The hypothesis is the coordinating session's; the form given to it, the programs and the readings marked "mine" are this session's. Source: `claude/friendly-darwin-8mmhfm` at 71172671f; branch `claude/nano-merge-8mmhfm`.

Every step has a plan file committed before its runs (`PLAN_1.md` to `PLAN_11.md`), with the outcome and every prediction scored below it. Two slips, each stated where it happened: a test run of 6,000 lines on the development seed was looked at before `PLAN_6.md` was committed; one arm of `PLAN_10.md`'s addition 3 did not test what its plan said (a fault of the program, corrected and the arm run again).

## 1. In brief

1. **The system finds classes of cells by itself, and on sums they are the arithmetic.** With nothing about sums laid down, the pairs of units digits fall into two classes by whether they carry (90 to 98 of 100 pairs with their class; the strays are always pairs that sum to nine), and the pairs of tens digits fall into ten classes by their sum modulo ten, which I had not thought of. Classes drawn by chance with the same sizes do nothing.
2. **Cells keyed on a class answer what was never shown.** Never-shown two-digit sums, 40,000 lines, no look-ahead, median of five seeds: **91.8%** against 77.5% without classes (90.4 against 78.1 on five further seeds; look-ahead, as its lines report: 91.4). Part (i) of the hypothesis holds.
3. **The same program with the same numbers gains on prose**: 0.010 to 0.016 nats a byte on six books (three of them opened only after the setting was frozen), against 0.002 to 0.003 with classes by chance. Part (ii) holds. **And on a third task with nothing chosen on it**, three-digit sums: 94.1 against 92.3.
4. **The store does not shrink.** Part (iii) fails: class cells are added (on prose the store is five times the base's), and the cells a class stands for cannot be withdrawn on prose without loss, by any of three rules tried. Yet on two-digit sums the class cells alone, 1.5% of the store, answer never-shown sums better than the whole store (94.6 against 92.2): the regularity is held in few cells, and the system does not yet run on them.
5. **A class used as a line** (cells keyed on two classes at once; `PLAN_6.md`) lifts two-digit sums further (93.1 at the median, and 94 to 95 within 10,000 to 15,000 lines), does nothing on three-digit sums (93.9 against 94.1), and on prose its gain is recency bought at the cost of the old text. Its hypothesis is refuted as written.
6. **Kinds that the system makes for itself** (`PLAN_10.md`: a kind of largest trust refined by one more delay, chosen on the system's own recent bytes) lift never-shown two-digit sums to **98.0%** at 40,000 lines (five seeds; 98.0 again on five further seeds) and to **99.3 and 98.7% at 400,000 lines** (two seeds), with no look-ahead: the first system of this line that does not fall with reading. Three-digit sums: 95.9 (plain 92.3). The made kinds hold the whole pair and a byte that fixes the place in the line; classes found in them are no longer mixed. On prose it changes little on the books and, like item 5, harms the old text (1.5685 at the end of the life against 1.5306 with no classes).
7. **That harm has one cause and one rule removes it** (`PLAN_11.md`): a kind born late holds only what was read after its birth. If a made kind is born with the pooled counts of a kind that refines it, wherever the system has one, the test text at the end of the life is 1.5272 (the frozen setting 1.5264), sums stay at **97.9%** (five seeds; 98.1 on five further seeds; 99.3 and 98.4 at 400,000 lines; 95.8 on three-digit sums), and the six books stand where the frozen setting had them (0.010 to 0.015 nats below no classes): which also shows that the further gain of items 5 and 6 on the books was recency. **This setting, F, is the line's best single setting**: parts (i) and (ii) of the hypothesis together, with the old text kept. The store is still larger than the base's.
8. **Where it is short** (each general, none cured here): the answering cell (on a text of repeated lines the class kinds are seldom needed for the lines read, so the mismatch that would teach their trust is rare; a second row of trust taught by practice did not cure it, `PLAN_7.md`, and showed that my first account of the mechanism was wrong: section 7); the kinds themselves (a kind is a set of delays with no place in the line, so a cell mixes what follows it at several places; the measure of alikeness is taken over that mixture, which is why the pairs that sum to nine are seated wrongly; I first put this down to the search and was wrong, `PLAN_8.md`); the choice of where to search (by trust: a kind is forgotten when its classes succeed, repaired in `PLAN_6.md`); and the fall of the plain cells with long reading, which classes do not cure (78.9 and 81.3 at 400,000 lines on two seeds, against 62.1 and 59.1 without).

## 2. The system

`merge.py`, one program for both tasks; numpy and plain-Python references for every compiled part (`check_output.txt`, `check3_output.txt`, `check6_output.txt`, `check7_output.txt`, `check10_output.txt`).

*The base* is each task's comparator as it stands: a chain of delays; a kind of cell is a set of delays; a cell is made when its combination of byte values first occurs and counts the bytes that follow; it says log((c(v) + e) / (n + S e)); the answering cell holds a trust for each kind and each of 40 levels of experience and changes it by the mismatch (`nano/table/trust.py`). Sums: every set of at most four of nine delays (255 kinds), step 0.1 shrinking with use, e = 2, trusts born 0.02 (`nano/sparse`). Prose: runs of 0 to 6 bytes (7 kinds), step 0.00005, e = 0.02, trusts born 0.25 (`nano/table`, T0). **The base's kinds and numbers differ between the tasks and are inherited; that is a limit of this line.** The base in this program reproduces the other lines' figures (sums 77.5 against 77.3 to 77.4; three-digit sums seed by seed; the three books to the fourth place).

*The class mechanism*, one setting on every task:
1. A **pair** is a kind R of at least two delays split into X (one or two of its delays) and Z (the rest). R's cells are the refinements of the X-cells by the values of Z. Nothing new is counted to find classes.
2. **Alike.** For a grouping of the X-cells, each count of an R-cell (x, z) is foretold from the pooled counts of the *other* members of x's class at the same z: J = sum of c(v) log((C(v) - c(v) + e_J) / (N - n + S e_J)). A cell alone is foretold at 1/S.
3. **Search.** X-cells in the order of their counts; each joins the class that raises J most or founds one; two sweeps of single moves. The number of classes is whatever results.
4. **Where.** Candidates are the pairs inside the 16 kinds of largest trust. The 32 pairs of largest positive worth (J of the classes less J of one class) are kept.
5. **The new kind.** For each kept pair, a cell for each (class, values of Z) that has occurred, holding the pooled counts of its members' R-cells and counting on; silent where the X-cell is in no class of two or more; trusts born at nothing.
6. **When.** Classes are found anew when the bytes read have doubled (first at 10,000).

*Chosen on development material and frozen* (`PLAN_1.md`): class cells beside their members' cells (not in place of them), 32 pairs, e_J = 0.02. *Laid down* (the cost in generality is said in `PLAN_1.md`): X of at most two delays; pairs inside one existing kind; 16 kinds by trust; the schedule; two sweeps. Nothing says which delays, which bytes, how many classes, or where in a line.

## 3. Two-digit sums

Never-shown additions (`cells2.py`'s text and split), the system writing its own answer bytes, whole answer, %, 40,000 lines, mean of the last three readings; seeds in the order above.

| Arm | each seed | median | last reading, median | hundreds, tens, units | cells | CPU s |
|---|---|---|---|---|---|---|
| no classes | 79.0, 77.9, 77.5, 76.1, 77.2 | 77.5 | 77.6 | 98.4, 77.6, 100 | 1,213,003 | 19 |
| **classes, the frozen setting** | 91.8, 93.3, 91.7, 88.0, 92.1 | **91.8** | 92.2 | 98.0, 92.4, 100 | 1,231,175 | 52 (30 in the search) |
| control: classes by chance, the same sizes | 80.9, 78.3, 78.1, 77.3, 77.9 | 78.1 | 78.8 | 98.4, 78.8, 100 | 1,249,586 | 64 |
| control: classes found once at 20,000 lines, their cells counting only the second half | 87.6, 89.5, 89.3, 87.5, 88.1 | 88.1 | 89.6 | 98.2, 89.6, 100 | 1,238,417 | 32 |
| replacement (the class cell in place of its members' cells; excluded for prose) | 93.1, 94.9, 92.5, 91.5, 93.1 | 93.1 | 93.0 | 98.2, 93.0, 100 | 1,230,553 | 55 |
| a kind's trust counts its children's (C, `PLAN_6.md`) | 90.9, 92.9, 91.5, 87.4, 91.5 | 91.5 | 90.8 | 98.0, 90.8, 100 | 1,236,252 | 64 |
| with cells keyed on two classes (C2, `PLAN_6.md`) | 92.8, 96.7, 95.1, 89.7, 93.1 | **93.1** | 92.8 | 98.2, 92.8, 100 | 1,238,148 | 67 |
| **with kinds made while reading (M, `PLAN_10.md`)** | 98.5, 98.0, 96.8, 97.9, 98.5 | **98.0** | 98.2 | 98.2, 99.4, 100 | 1,440,880 | 99 |
| control: M with its made kinds drawn by chance | 92.5, 98.1, 95.7, 90.3, 93.9 | 93.9 | 94.0 | 98.2, 94.2, 100 | 1,440,021 | 96 |
| **M with made kinds born with the past (F, `PLAN_11.md`)** | 97.9, 97.9, 96.7, 96.7, 98.3 | **97.9** | 98.0 | 98.0, 99.0, 100 | 1,434,990 | 98 |
| as other lines report: look-ahead; relay | | 91.4; 66 | | | 3.3 million; 256 cells in between | |

*A repeat on seeds 2001 to 2005* (`PLAN_9.md`; used by no line for choosing; the same session and program): no classes 78.1; the frozen setting 88.4, 90.4, 93.6, 93.5, 85.2, median **90.4**; C2 93.7, 91.1, 96.5, 94.5, 92.9, median **93.7**; M 98.4, 97.3, 98.0, 99.1, 97.2, median **98.0**; F 98.1, 97.1, 98.1, 99.0, 96.1, median **98.1**.

By lines read (median): without classes 64.8, 72.4, 77.8, 77.6 at 5,000, 10,000, 20,000, 40,000; the frozen setting 86.8, 88.4, 90.0, 92.2; C2 92.6, 94.0, 91.2, 92.8 (95.0 at 15,000); M 92.6, 95.8, 97.2, 98.2.

*What the classes are* (last finding; members in the result files). The pair inside the kind of delays 3, 4, 6, 7 with X = delays 3, 6 (where the tens digit is named: the two units digits against the two tens digits) is kept on all five seeds. Its classes: 53 and 47 pairs of digits on two seeds (98 of 100 with their class by "sum to ten or more"), 45 and 55 on three (90 of 100). The pairs against their class are in every case pairs that sum to nine. I first took this for a limit of the search (single moves) and tried two further moves, dissolving and reseating a class (`PLAN_4.md`) and splitting a class around its worst-foretold member (`PLAN_8.md`); neither changes the classes, and a direct computation shows why: **the measure J scores the grouping that was found higher than the purer ones** (development seed at 32,000 lines: -454,931 nats as found; -460,471 with the two strays moved to the pairs that do not carry; -458,621 with the pairs that sum to nine in a class of their own). My present account: the cells the measure reads mix several places of the line. The cell of delays 3, 4, 6, 7 holds the whole pair and its tens digit where the tens digit is named, and four other digits followed by a digit that is there by chance where the second number's first digit is named (about three such counts beside four of the first sort); and the same X-cell is the two tens digits where the hundreds digit is named. The grouping that is best for the mixture is not exactly the carry. A look confirms it (`look8.py`, a diagnostic with one kind laid down by hand, not a system): on the system's kind the measure prefers "nine or more against eight or less" by 2,177 nats; on a kind with the place in it (the same four delays and the delay that holds '=') it prefers the carry; and counting only the place of the tens digit the carry is exact to 54 nats in 32,000 counts where the other grouping costs 8,785. With X = delays 4, 7 the classes are the ten sets of pairs with the same sum modulo ten, on every seed.

*Class kinds alone* (`alone.py`, the frozen setting, last reading): with only the class kinds speaking, 94.2, 96.8, 94.6, 93.4, 95.8 (median **94.6**) with 16,288 to 21,182 cells; all kinds together 92.2 (median); the plain kinds alone 76.0 with 1.21 million cells. The eight class kinds of largest trust, 6,700 to 8,400 cells: 93.8 to 96.0. On the development seed the class kinds alone name the tens digit on 98.6% and all kinds together on 92.8%: where the two units digits sum to exactly ten the plain cells outvote the class cells.

*Long reading*, 400,000 lines, two seeds (73129, 93683): a reading, not a claim. Mean of the last three readings:

| | 73129 | 93683 | at 40,000 lines |
|---|---|---|---|
| no classes | 62.1 | 59.1 | 79.2, 79.0 |
| the frozen setting | 78.9 | 81.3 | 92.2, 94.2 |
| C | 80.1 | 81.3 | 90.8, 94.0 |
| C2 | 81.3 | **98.7** | 92.8, 97.0 |
| M | **99.3** | **98.7** | 98.8, 98.2 |
| F | **99.3** | **98.4** | 98.0, 98.0 |

The plain cells, the frozen setting and C fall with reading, as `nano/sparse` found for the plain cells (the weight of long-counted cells; its cap was not used here). **M and F do not**: with M no reading after 20,000 lines is under 98 on either seed, with F none is under 97 (about 510 CPU-seconds a run, 1.58 million cells). With C2 one seed of the two does not end low (at its last finding the pairs that sum to nine happen to be seated with the pairs that do not carry, 99 of 100 pairs with their class, and the kinds of largest trust are keyed on the class of the tens digits, the class of the units digits and the '=' sign; on the other seed all ten sit with the carrying pairs): seed 93683 reads 97.0, 96.4, 81.8 at 40,000, 100,000 and 200,000 lines and then 98.8, 98.6, 98.6 from 300,000 on, after the finding at 2,560,000 bytes; seed 73129 falls like the others (79.4 at the end). One seed of two is an observation and nothing more; what the kinds keyed on two classes were at that finding is in `res/s400_C2_seed93683.json`. About 240 to 320 CPU-seconds a run with classes, 175 without.

## 4. Prose

The life of `nano/table/life.py`, scored as `nano/aheadprose` does: the test text's stream read once, then three books; each book's last tenth right after its stream, nothing changing during a scoring; the test text's last tenth once, at the end of the life. Nats a byte. The second life has three books this line opened only after the setting was frozen and the first life read.

| Arm | bryant-stories | burgess-busterbrown | carroll-alice | test text, at the end | cells at the end | CPU s |
|---|---|---|---|---|---|---|
| no classes (equal to T0) | 1.5023 | 1.3645 | 1.4456 | 1.5306 | 579,255 | 6 |
| **classes, the frozen setting** | **1.4920** | **1.3487** | **1.4307** | 1.5264 | 2,961,530 | 200 (134 in the search) |
| control: classes by chance | 1.5003 | 1.3619 | 1.4428 | not scored | 3,398,416 | 202 |
| replacement | 1.5031 | 1.3554 | 1.4511 | not scored | 2,961,530 | 194 |
| with cells keyed on two classes (C2) | 1.4885 | 1.3349 | 1.4019 | **1.5550** | 3,032,556 | 215 |
| with kinds made while reading (M) | 1.4883 | 1.3371 | 1.3921 | **1.5685** | 2,779,094 | 235 |
| M without the kinds keyed on two classes | 1.4922 | 1.3391 | 1.3983 | 1.5573 | 2,769,853 | 200 |
| **M with made kinds born with the past (F)** | 1.4919 | 1.3518 | 1.4305 | **1.5272** | 2,336,138 | 249 |
| as `nano/table` reports: T2 with word cells | 1.4802 | 1.3162 | 1.4109 | | | |

| Arm, second life | chesterton-thursday | chesterton-brown | austen-persuasion | cells at the end | CPU s |
|---|---|---|---|---|---|
| no classes | 1.4411 | 1.3996 | 1.2700 | 691,801 | 8 |
| **classes, the frozen setting** | **1.4263** | **1.3856** | **1.2594** | 3,208,291 | 219 |
| with cells keyed on two classes (C2) | 1.4242 | 1.3663 | 1.2446 | 3,280,465 | 258 |
| with kinds made while reading (M) | 1.4253 | 1.3673 | 1.2459 | 2,994,942 | 245 |
| **M with made kinds born with the past (F)** | 1.4273 | 1.3853 | 1.2593 | 2,979,800 | 265 |

The frozen setting is below the base by 0.0103, 0.0157, 0.0149 on the first three books and by 0.0148, 0.0140, 0.0107 on the second three; classes by chance by 0.0020, 0.0025, 0.0028. It is about a third of what the word cells gain, with nothing about words laid down. On the test text's own last tenth at the end of the life it is 0.0042 below the base.

*What the classes are on prose* (from `res/life_cls.json`): of the 54 pairs inside runs of two to six bytes, 49 have positive worth and 32 are kept; most have the nearest byte in X (classes of the last byte, or of the last byte with one further back, 50 to 110 classes of 1,000 to 2,900 X-cells). I did not study their members.

*The controls on "found on one part, used on another"* (`PLAN_4.md`, sections 8 to 10). Classes found once at 500,000 bytes do as well as classes found at every doubling when their cells start pooled (0.009 to 0.015 on six books). When the class cells start empty the books gain more (0.016 to 0.031 found once; from 0.086 better to 0.015 *worse* found at every doubling). My reading: an empty start makes the class cells a store of recent text, and the gain depends on where a finding falls; it is not claimed for classes. The comparison that would settle it (plain cells that start empty at the same moments) is NOT RUN.

*The store* (`report_life.txt`). Cells that have counted against bytes read, fitted from 125,000 to 1,000,000 bytes of the test text: without classes the power 0.60 (runs only; `growth.py`'s 0.80 includes the word cells); with classes 0.48, with 5.1 times the cells at the end. Leaving out the R-cells whose X-cell is in a class still leaves 4.1 times the base, and scoring with them silent gives 2.5 nats. Three ways of withdrawing were tried: every member's cell (`PLAN_1.md`, addition 1: sums 93.1, prose worse than the base on two books of three); only the cells the class foretells at least as well as they foretell themselves (`PLAN_3.md`: refuted on development material on both tasks); silent at half the bytes while reading (`PLAN_4.md`, arm Pr: worse on both). My reading: a cell met once, read at the small trust learned for such cells, still adds something its class cell does not replace.

## 5. Three-digit sums: a task on which nothing was chosen (`PLAN_5.md`)

`nano/ahead2`'s text, split and length (80,000 lines; a fifth of the million pairs never shown, 500 scored; about 95% of the lines read are first readings). Mean of the last three readings.

| | each seed | median | thousands, hundreds, tens, units (last, median) | cells | CPU s |
|---|---|---|---|---|---|
| no classes | 92.5, 89.9, 92.7, 90.8, 92.3 | 92.3 | 99.6, 95.2, 98.0, 100 | 1,994,138 | 50 |
| the frozen setting | 94.9, 93.0, 94.0, 94.1, 95.0 | **94.1** | 99.6, 95.6, 99.4, 100 | 2,029,513 | 145 |
| with cells keyed on two classes (C2) | 93.9, 92.9, 94.1, 93.2, 95.1 | 93.9 | 99.6, 96.4, 98.4, 100 | 2,035,158 | 167 |
| with kinds made while reading (M) | 96.3, 95.3, 95.7, 95.9, 96.3 | **95.9** | 99.6, 96.0, 100, 100 | 2,880,206 | 397 |
| M with made kinds born with the past (F) | 96.2, 95.8, 95.6, 95.5, 96.1 | **95.8** | 99.6, 95.8, 100, 100 | 2,867,696 | 415 |
| as `nano/ahead2` reports: look-ahead | | 99.7 | all 100 | 4.5 million | 205 |

The frozen setting is above the base on every seed (1.3 to 3.3 points) and far above it early (87.2 against 62.6 at 5,000 lines). The gain is the tens digit; the hundreds digit, which needs the class of the tens digits and the class of the units digits together, does not move. Look-ahead does far better on this task.

## 6. A class used as a line: cells keyed on two classes (`PLAN_6.md`)

For two kept pairs whose X's share no delay, a kind keyed on (the class of one X-cell, the class of the other, the delays of the first pair's Z not in the second's X). Which are made is decided on the system's own last 50,000 bytes (each byte foretold by its cell with that byte left out, less the better of the two class kinds it is made from); 32 at most; cells start empty at each finding. Numbers set once and not chosen. Finding the cause of a collapse on the development seed led to one repair of the first mechanism: the trust of a kind, for choosing where to search, counts the trust of the kinds made from it (without it a kind is dropped from the search as soon as its classes take over its trust, and the classes vanish; this also happens to the frozen setting late in the 400,000-line run of seed 73129).

Results are in the tables above (rows C and C2). Two-digit sums: 93.1 at the median against 91.8, and 92.6 and 94.0 at 5,000 and 10,000 lines against 86.8 and 88.4; the kinds keyed on two classes hold 3,500 to 5,100 cells and alone name the tens digit on 89 to 98%. Three-digit sums: no gain (93.9 against 94.1); the kind that would be needed is not made, because no kept pair has the two units digits as its X where the hundreds digit is named (my reading of `res/dev6_s3_K2.json`). Prose: the six books are lower than the frozen setting by 0.004, 0.014, 0.029 and 0.002, 0.019, 0.015, and the test text at the end of the first life is 0.029 *higher* (1.5550; above the system with no classes): the cells keyed on two classes start empty at each finding and hold only recent text. **The hypothesis of `PLAN_6.md` is refuted** (its first part: at least 2 points on three-digit sums).

## 6b. Kinds that the system makes while reading (`PLAN_10.md`)

*Why.* `PLAN_7.md` and `PLAN_8.md` (the correction in section 7 below and the account of the strays in section 3) came to one cause: a kind is a set of delays with no place in the line, its cells mix what follows at several places, and both the answer and the measure of alikeness are taken on that mixture.

*The addition* (to C2; one setting on every task; numbers set once and not chosen). At each finding, the candidates are every (R, d) with R one of the 16 kinds of largest trust and d a delay R does not have. The worth of a candidate is measured on the system's own last 50,000 bytes: how well the cells of (R with d) foretell those bytes, each left out of its own cell, less the same for R. The 16 of largest positive worth are made, or kept with their cells and trusts if already there. A made kind counts and speaks like any other, can be refined again, and is a candidate for the search for classes. Nothing says which delay or that a place matters. `merge10.py`; its check counts the cells of the made kinds again by hand.

*What it made.* On two-digit sums: delays (2, 3, 4, 6, 7), (3, 4, 6, 7, 8), (1, 3, 4, 6, 7) on four or five seeds of five. Where the tens digit is named these are the whole pair with, in turn, the '=' sign, the end of the line before, and the hundreds digit: a byte that says where in the line the cell is. On prose: runs with a gap, such as delays (1, 3) and (1, 2, 4).

*What it did* (rows M in the tables of sections 3 to 5). Two-digit never-shown sums 98.0% at the median of five seeds at 40,000 lines (lowest 96.8), 95.8% at 10,000 lines, 98.0% on five further seeds, and 99.3 and 98.7% at 400,000 lines. With only the class kinds speaking, the tens digit of never-shown sums is named on 100, 100, 100, 100 and 99.8%. The route is not the one I predicted (a pure carry class in a made kind: that pair is not among the kept ones). It is: the classes of the two tens digits by their sum modulo ten, found in a made kind, with the place and the two units digits as the rest of the key. Three-digit sums: 95.9 (the tens digit 100%; the hundreds digit 96.0, still short for the reason of section 5). Prose: the development material and the three books within 0.002 of C2 or better; **the test text at the end of the life 1.5685, worse than C2 (1.5550), the frozen setting (1.5264) and no classes (1.5306).**

*The control.* With the same number of made kinds drawn by chance from the same candidates, the median is 93.9 (C2, with no made kinds: 93.1); the one seed that reaches 98 by chance is the one whose draw included the whole pair with the '=' sign. The worth measured on the system's own recent bytes is what finds the kinds that matter.

*Its hypothesis* (`PLAN_10.md`): at least 96% on two-digit sums held; "units-digits classes pure in a made kind" failed; "no book more than 0.002 above C2" failed by 0.0002 nats on one book. Refuted as written, on that margin; what stands is said above.

*The old text, and the rule that keeps it* (`PLAN_10.md`, addition 3; `PLAN_11.md`). M without the kinds keyed on two classes still ends the life at 1.5573, so the made kinds harm the old text as much as those do (I had blamed only the kinds keyed on two classes). What the two have in common: they are born empty, some of them inside the books, and hold nothing of the text before their birth. The rule: **a kind that is made is born as the pooled counts of a kind that holds all its delays, where the system has one** (the one with the fewest delays), and counts on. On prose such a kind always exists (a run with a gap lies inside a run); on sums it exists for most kinds keyed on two classes and for no made kind. No new number. With it (setting F): the test text at the end of the life 1.5272; the six books within 0.003 of the frozen setting (two of them above it, by 0.0031 and 0.0010); two-digit sums 97.9, 97.9, 96.7, 96.7, 98.3. `PLAN_11.md`'s hypothesis holds in the two parts its refutation was stated on and fails the clause "no book above the frozen setting" by 0.003 nats. F on the fresh seeds, at 400,000 lines and on three-digit sums (`PLAN_11.md`, addition 1) equals M to a few tenths: 98.1; 99.3 and 98.4; 95.8.

*Cost.* 99 CPU-seconds for 40,000 lines of two-digit sums against 19 for the plain cells; 1.44 million cells against 1.21 million. A made kind of five delays holds up to 180,000 cells. This buys accuracy with a larger store, not a smaller one.

## 7. The hypothesis of this line, scored on the frozen setting

*On the latest setting, F (not the one the hypothesis was frozen on; reached on the development seed after the earlier settings' results had been seen; each figure a single run of it on that material):* (i) 97.9; (ii) 0.010, 0.013, 0.015, 0.014, 0.014, 0.011 nats on the six books; (iii) fails (more cells). *On the frozen setting, as planned:*

(i) never-shown two-digit sums at least 90% at 40,000 lines with no look-ahead: **holds** (91.8; lowest seed 88.0; under 83 would have refuted). (ii) each of the three books at least 0.005 nats below the same system without classes: **holds** (0.010, 0.016, 0.015; and 0.015, 0.014, 0.011 on three further books). (iii) fewer cells at equal or better loss when cells of a merged class are withdrawn: **fails** on both tasks as the system runs. One setting does (i) and does no harm on prose. My chance for the whole had been 0.12; for (ii) alone 0.20, which was too low.

Controls: classes by chance with the same sizes do nothing on sums (78.1 against 77.5) and a fifth of the gain on prose; classes found on one half of the stream serve on the other on both tasks; what the classes are on sums is the carry and the sum modulo ten.

*If it fails, where* (the coordinating session's question). Alike cells **are found** at this amount of reading, on both tasks. Cells keyed on classes **earn trust on prose and too little on two-digit sums**. The store **is not made smaller**, because nothing found here lets a class cell take the place of its members on prose.

*The answering cell on sums, and a correction* (`PLAN_7.md`, section 6). I first read the shortfall as: a shown line is answered by the plain cell of the whole pair, so the class kinds are taught only at first meetings. A direct look (development seed, the tens digit with the bytes before it given) shows the mechanism is otherwise. (a) The cell of the four delays that hold the whole pair has counted on 98.6% of *never-shown* sums: a kind has no place in the line, and the same four delays hold four digits at two other places of every line. (b) With every cell of that kind left out, shown sums are still 98.2% right and never-shown ones 93.4%: the trace of a shown line is in every cell that holds part of it. What stands: on the lines read the class kinds are seldom needed (the tens digit is wrong on 1.6% of shown sums, on 7% of never-shown ones), so little teaches their trust. A second row of trust for "the member's own cell does not speak", taught by practice with that cell left out, changes nothing on sums (90.8 on the development seed, with either step rule) because that case all but never occurs there; on prose's development material it gains 0.002 nats. It was stopped on development material as its plan said. The statement in the plans of other lines that a never-shown pair is a never-seen combination for the cells is, on this evidence, not true of cells keyed on delays alone.

## 8. Predictions

Scored in the plan files, failures included. Of note: I gave part (ii) one chance in five and it held on six books; I predicted replacement would reach 96 (93.8); that classes beside the cells would be within 0.002 of the base on prose (0.008 better); that the units-digits classes would be pure for the carry (pure on no seed: the pairs that sum to nine); that 400,000 lines would hold 90 (79 and 81); that "found once, cells empty" would beat the frozen setting on development prose as it had on the books (worse); that practice would help sums (worse on both); that 64 pairs would beat 32 (equal); that cells keyed on two classes would lift three-digit sums by two points (nothing); that a second row of trust taught by practice would lift two-digit sums to 95 (no change); that a further move of the search would make the units-digits classes pure (the measure prefers them as they are); that the made kinds would give a pure carry class (the gain came through the classes of the tens digits instead).

## 9. What is laid down, what was not run, what I would do next

Laid down: section 2's list; for the two-class kinds, two classes and no more, the 50,000 bytes kept, 16 by trust, 32 kinds; for the made kinds, one delay at a time and 16 kinds. Settings C2 and M were arrived at on the development seed after the frozen setting's results on the reported seeds had been seen, and were then run on the reported seeds once; M was also run on five seeds it had never met. NOT RUN: subtraction and keeping across tasks; the word cells as the base on prose; X of more than two delays; classes of cells that are themselves keyed on a class; the cap on a long-counted cell's weight together with classes; three of five seeds at 400,000 lines; a second repeat of anything by another session.

What I would do next (mine): (0) setting F repeated by another session, and run with subtraction after addition (keeping across tasks was not run in this line at all); then the store: now that made kinds and class kinds hold what matters on sums, which plain kinds can be let go; (1) practice done as the search's own measure does it: the answer formed with a combination's counts taken out of every cell that it refines, and the trusts taught on that answer; with it, kinds that carry a place in the line, so that "this combination is new here" can be known (`PLAN_7.md` shows that one cell left out is not that question); (2) a worth for a class together with another (the hundreds digit of three-digit sums needs two classes of which one is of no use alone), and pairs whose union is not an existing kind, judged on the system's own recent bytes as in `PLAN_6.md`; (3) class cells that are kept and refined from one finding to the next instead of made anew, which is also what a smaller store needs: the plain cells could then be let go once their class has stood for long enough.

## 10. Files

`PLAN_1.md` to `PLAN_11.md` (plans, additions, outcomes); `merge.py` (the system; `check`), `merge3.py` (`PLAN_3.md`), `merge6.py` (`PLAN_6.md`), `merge7.py` (`PLAN_7.md`), `merge10.py` (`PLAN_10.md`, `PLAN_11.md`), `alone.py`, `look.py`, `look7.py`, `look8.py`, `report.py`; `score.py` copied unchanged from `nano/table`; `res/*.json` (every run), `out_*.txt` (one line a run), `report_*.txt` (tables), `jobs_*.txt` (the commands). The books are read from `data/` (not committed; `git show 71172671f:minifly_v148_pretrain_probe_dev_20261025/data/<book>.txt` and the two folders named in `PLAN_4.md`).
