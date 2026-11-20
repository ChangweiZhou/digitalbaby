# Line `nano/merge`, fourth plan: where the frozen setting loses, taken one change at a time on development material (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Written after step 2's runs at 40,000 lines and its life on prose were read (the 400,000-line runs with classes still running) and before any run this plan describes. Development level; no gate, seal or review is claimed. **What I had seen when writing this:** the five reported seeds at 40,000 lines and the three books under the frozen setting and its controls. Anything chosen below is therefore chosen with those in sight; section 4 says how that is met.

## 1. The evidence this follows (step 2; figures in `report_s40.txt`, `report_life.txt`; scored in `PLAN_2.md` when the long runs end)

1. Sums, five seeds, 40,000 lines: 77.5 without classes, **91.8** with (88.0 to 93.3), 78.1 with classes by chance, 88.1 with classes found once at half, 93.1 with replacement.
2. On three seeds of five the pairs of units digits fall into two classes that are *not* the carry: one holds the 45 pairs that sum to ten or more and the ten that sum to nine, the other the 45 that sum to eight or less (on the other two seeds: 47 and 53, two strays). My account, from the measure J: the same X-cell also stands where the hundreds digit is named, as the two tens digits, and there the pairs that sum to nine behave like neither group. Their best place is a class of their own (I computed J for it by hand: far higher), but the search moves one cell at a time and a single cell leaving for an empty class loses. **A limit of the search, not of the measure.**
3. With only the class kinds speaking (development seed; `alone.py`): 93.8% of never-shown sums with 17,885 cells, against 90.6% with all 1.23 million cells speaking; four class kinds with 3,628 cells give 94.8%. **The plain cells outvote the class cells where they are wrong.**
4. Prose, the three books: classes lower the loss by 0.010, 0.016, 0.015 nats; classes by chance by 0.002 to 0.003; replacement raises it on two books. **Classes found once at 500,000 bytes, their cells starting empty, lower it by 0.016, 0.031, 0.021**: more than classes found anew at every doubling with pooled counts. I did not expect this. Of the 32 kept pairs 30 or 31 persist from one finding to the next, so it is not the trusts being born again; what differs is that the classes themselves are made anew eight times, and that the cells start pooled.

## 2. The arms: each is one change from the frozen setting, on development material only

Sums: development seed 1, 40,000 lines. Prose: taught on the stream before its last twentieth, scored on that twentieth (frozen: 90.8 and 1.5488; no classes: 76.8 and 1.5566).

| Arm | The change | What it tests |
|---|---|---|
| F | none (the frozen setting run again with the program as it now is) | that the program still gives 90.8 and 1.5488 |
| D | `dissolve=1`: after the sweeps each class of two or more is dissolved once, its members seated again one by one (free to found classes), the result kept only if J rises; then one more sweep | evidence 2: more search |
| Oe | classes found once, at half the reading, cells starting empty (`mode=half`) | evidence 4 on development material |
| Op | the same with the cells starting pooled (`mode=half pool=1`) | whether "once" or "empty" is what helps |
| E | found at every doubling, cells starting empty (`empty=1`) | the same from the other side |
| Pr | `practice=0.5`: while reading, at half of the bytes (drawn by chance) the R-cells that a class stands for are silent as in replacement; when answering, nothing is silent | evidence 3: the trust of class kinds is taught only where the member's own cell does not already answer |
| P64 | at most 64 kept pairs | more search |

*A second setting.* A change is "good" if on development material it is not below the frozen setting by more than 1 point on sums nor above it by more than 0.002 nats on prose, and gains at least 1.5 points on sums or 0.002 nats on prose. The good changes that can be combined are combined (Oe, Op and E exclude one another: the best of them on prose that is good), the combination is run once on development material, and it is adopted as the second setting if it is itself good and not below every one of its parts on both tasks; otherwise the single best good change is the second setting (best: the largest sum of its gain in points on sums and a thousand times its gain in nats on prose). If none is good, there is no second setting.

## 3. Predictions (mine, before any run)

| | Statement | Chance |
|---|---|---|
| R1 | D: on the development seed the pair of units digits has every class of two or more pure for the carry | 0.55 |
| R2 | D: sums at least 92.3 | 0.35 |
| R3 | Oe: prose below the frozen setting by at least 0.002 | 0.60 |
| R4 | Oe: sums below the frozen setting by more than 1 point | 0.75 |
| R5 | Op: prose not below Oe | 0.55 |
| R6 | E: prose below the frozen setting by at least 0.002 | 0.45 |
| R7 | Pr: sums at least 92.5 | 0.50 |
| R8 | Pr: prose not above the frozen setting by more than 0.002 | 0.50 |
| R9 | P64: better than the frozen setting on both tasks | 0.45 |
| R10 | a second setting is adopted | 0.55 |

## 4. If a second setting is adopted

It is run (a) on the five reported seeds at 40,000 lines; (b) on the life with the three books, named as a setting chosen after those books had been seen under the frozen setting; (c) on a **second life with three books this line has not opened**: `chesterton-thursday`, `chesterton-brown`, `austen-persuasion` (from `minifly_v153_bits_dev_v1_20261025/data/` and `minifly_v149_input_ceiling_dev_20261025/data/` at the base commit; the three smallest of the texts not yet used, with at most two by one author), read after the test text's stream in that order, each last tenth scored right after its stream; with no classes, with the frozen setting and with the second setting. The frozen setting on the second life is a repeat of part (ii) of the hypothesis on new material. *Prediction (mine):* on each of the three new books the frozen setting is at least 0.005 nats below the base (chance 0.75); the second setting is below the frozen setting on at least two of the three (0.5).

## 5. Size, cost, what is not run

As step 2: a development run about one CPU-minute on sums and one and a half on prose; fourteen runs. The second life about seven CPU-minutes a run. Not run: a trust that depends on which other cells speak; classes kept from one finding to the next and only refined; X of more than two delays; classes of classes; subtraction and keeping.

## 6. Outcome of the arms (entered 20 November after the fourteen development runs; `res/dev4_*.json`, `out_dev4_*.txt`)

| Arm | sums, seed 1, last three readings | prose, last twentieth | prose cells | CPU s, sums; prose |
|---|---|---|---|---|
| no classes | 76.8 | 1.5566 | 445,161 | 18; 3 |
| F, the frozen setting run again | 90.8 | 1.5488 | 2,292,678 | 50; 95 |
| D, dissolving | 91.3 | 1.5489 | 2,281,008 | 69; 178 |
| Oe, found once at half, cells empty | 86.7 | 1.5550 | 1,681,591 | 34; 35 |
| Op, found once at half, cells pooled | 89.5 | 1.5507 | 2,138,631 | 31; 40 |
| E, every doubling, cells empty | 87.2 | 1.5555 | 1,518,731 | 55; 78 |
| Pr, practice at half the bytes | 88.1 | 1.5746 | 2,292,678 | 55; 95 |
| P64, 64 pairs | 90.6 | 1.5489 | 3,955,149 | 62; 123 |

**No change is good by the rule of section 2; there is no second setting.** The frozen setting is the best of the eight on both tasks, within half a point and a ten-thousandth of a nat.

Predictions. R1: see the note below. R2 failed (91.3; 92.3 predicted). R3 failed: found once with empty cells is 0.006 nats *worse* than the frozen setting on development prose. R4 held (86.7). R5 failed (pooled cells are better than empty ones, 1.5507 against 1.5550). R6 failed (0.0067 worse). R7 failed (88.1). R8 failed (0.026 worse). R9 failed (64 pairs equal 32 on both). R10 failed.

Readings (mine; one seed and one piece of prose).
1. **What I took from the books was not there on development material.** On the three books, classes found once with empty cells beat the frozen setting by 0.006 to 0.015 nats; on the last twentieth of the test text's own stream they are 0.006 worse, and each of its two parts (once; empty) is worse taken alone. The difference between the two measurements: the books are new text read after the classes were made. My reading is that a class cell pooled from the old text carries the old text's counts into the new, where a cell that starts empty does not; that is a statement about change of text, and it is tested in addition 1.
2. **Practice harms both tasks.** A trust taught half the time without the member cells and used with them is right for neither.
3. **More pairs than 32 add nothing**, on either task: on prose the kept pairs beyond 32 are nearly all there are (49 of 54 have positive worth), on sums the further pairs are poor ones.
4. **Dissolving** costs twice the search on prose and changes nothing there; on sums see the note.

*Note on R1* (from `res/dev4_sums_D.json` against `res/dev4_sums_F.json`, and the members of the classes in `res/s40_cls_seed*.json`). **R1 failed.** With and without dissolving the pairs of units digits fall into the same two classes on the development seed: 53 that do not carry, and 47 of which 45 carry and two do not, (6, 3) and (1, 8). On the reported seeds the members that lie against their class are in every case pairs that sum to nine (all ten of them on seed 73129, two on seed 93683), which is what my account in section 1 said; but dissolving does not cure it. Seated again one by one, the first pair that sums to nine still joins the carrying class, where it is foretold a little, rather than found a class where it is foretold by nobody yet; a class of the pairs that sum to nine can only be founded by two of them at once. The search would need a move of two cells together. Not built.

## 7. Addition 1 (20 November; written after section 6 and before the runs it describes)

No second setting was adopted, so section 4's condition is not met. The second life is run all the same, because two of its questions do not need a second setting: (a) does part (ii) of the hypothesis repeat on books this line has not opened, with the frozen setting; (b) does "found once, cells empty" beat the frozen setting again on new books, as it did on the first three and did not on development material. Runs: the second life (`chesterton-thursday`, `chesterton-brown`, `austen-persuasion`, in that order after the test text's stream) with no classes, with the frozen setting, and with classes found once at 500,000 bytes with empty cells. Also, as a measurement on the frozen setting (no new mechanism): `alone.py` on the five reported seeds (the score with only the class kinds speaking and the number of cells that takes).

*Predictions (mine).* S1: on each new book the frozen setting is at least 0.005 nats below the base (0.75, as written in section 4). S2: found once with empty cells is below the frozen setting on at least two of the three new books (0.60). S3: class kinds alone, median of five seeds, at least 90% whole (0.60), with under 25,000 cells (0.85).

## 8. Outcome of addition 1 (entered 20 November; `res/life2_*.json`, `report_life2.txt`; `res/alone_seed*.json`, `out_alone_seed*.txt`)

*The second life* (the test text's stream, then three books this line had not opened; each last tenth scored right after its stream; nats a byte):

| | chesterton-thursday | chesterton-brown | austen-persuasion | cells at the end | CPU s |
|---|---|---|---|---|---|
| no classes | 1.4411 | 1.3996 | 1.2700 | 691,801 | 8 |
| frozen setting | 1.4263 | 1.3856 | 1.2594 | 3,208,291 | 219 |
| classes found once at 500,000 bytes, cells empty | 1.4173 | 1.3807 | 1.2529 | 3,045,848 | 95 |

The frozen setting is below the base by 0.0148, 0.0140, 0.0107: **S1 held; part (ii) of the hypothesis repeats on three unopened books.** Found once with empty cells is below the frozen setting on all three (by 0.009, 0.005, 0.007): **S2 held**; with the first life that is six books of six, against the opposite on the test text's own last twentieth.

*Class kinds alone, five reported seeds, 40,000 lines* (whole answer, last reading; all kinds; class kinds alone with the number of class cells; plain kinds alone):

| seed | all kinds | class kinds alone | class cells | plain kinds alone | plain cells |
|---|---|---|---|---|---|
| 73129 | 92.2 | 94.2 | 18,715 | 76.0 | 1,212,927 |
| 93683 | 94.2 | 96.8 | 18,596 | 76.0 | 1,212,579 |
| 116089 | 93.6 | 94.6 | 17,855 | 77.4 | 1,213,218 |
| 406364 | 87.6 | 93.4 | 21,182 | 72.8 | 1,213,070 |
| 545353 | 92.2 | 95.8 | 16,288 | 74.6 | 1,213,003 |
| median | 92.2 | **94.6** | 18,596 | 76.0 | 1,213,003 |

**S3 held** (94.6, at least 90 predicted; under 25,000 cells). With only class kinds speaking the tens digit is named on 98.2 to 99.8% and the hundreds digit on 95 to 97%; the eight class kinds of largest trust, 6,700 to 8,400 cells, give 93.8 to 96.0. On every seed the class kinds alone are above all kinds together.

Reading (mine). On sums the class cells are a store of one and a half hundredths of the size that answers never-shown sums better than the whole. That is part (iii) of the hypothesis in the sense "the regularity is held in few cells"; it is not part (iii) as the system runs, because the plain cells are still there, still needed to find the classes, and still outvote the class cells.

## 9. Addition 2 (20 November; written after section 8 and before its runs): which half of "found once, cells empty" helps on a new text

Runs: both lives with (a) classes found once at 500,000 bytes and cells starting *pooled* (`mode=half pool=1`), (b) classes found at every doubling and cells starting *empty* (`empty=1`). Beside the frozen setting (every doubling, pooled) and "once, empty" this is the two-by-two table on six books.

*Predictions (mine).* My reading of section 6 is that an empty cell does not carry the old text's counts into the new one. If so: U1, on at least five books of six "once, pooled" is nearer to the frozen setting than to "once, empty" (0.55); U2, on at least four of six "every doubling, empty" is below the frozen setting (0.55).

## 10. Outcome of addition 2 (entered 20 November; `res/life_Op.json`, `res/life_E.json`, `res/life2_Op.json`, `res/life2_E.json`; `report_life.txt`, `report_life2.txt`)

Loss on each book less the loss with no classes, nats a byte (negative is better):

| Classes found | cells start | bryant-stories | burgess-busterbrown | carroll-alice | chesterton-thursday | chesterton-brown | austen-persuasion |
|---|---|---|---|---|---|---|---|
| at every doubling | pooled (the frozen setting) | -0.0103 | -0.0157 | -0.0149 | -0.0148 | -0.0140 | -0.0107 |
| once, at 500,000 bytes | pooled | -0.0092 | -0.0152 | -0.0134 | -0.0141 | -0.0119 | -0.0100 |
| once, at 500,000 bytes | empty | -0.0158 | -0.0308 | -0.0212 | -0.0238 | -0.0189 | -0.0171 |
| at every doubling | empty | -0.0224 | -0.0477 | -0.0855 | **+0.0153** | -0.0458 | -0.0393 |

U1 held (found once with pooled cells is nearer to the frozen setting than to "once, empty" on six books of six). U2 held (empty cells at every doubling are below the frozen setting on five books of six).

Readings (mine).
1. **How often the classes are made does not matter on these books; whether their cells start empty does.** Once or at every doubling, pooled cells give the same 0.010 to 0.015.
2. **What an empty start adds is recency, not better classes.** A class cell that starts empty holds only what has been read since the last finding. In the first life the last finding falls at 1,280,000 bytes, inside the second book, so where the third book is scored the class cells hold little but that book and the end of the one before: 0.086 nats. In the second life the same finding falls just before the first book's last tenth, the class cells are nearly empty there, and the loss is 0.015 nats *above* the system with no classes. A gain that depends on where a finding happens to fall is not a result about classes. The line `nano/table` has cells whose counts fade for this purpose; the comparison that would settle it, plain cells that start empty at the same moments, is NOT RUN.
3. For the hypothesis of this line the figure that stands is the frozen setting's: 0.010 to 0.016 nats on six books, the cells pooled. The larger figures above are reported and are not claimed for classes.
